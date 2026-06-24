"""Shared base for funasr-based speaker embedding engines (ERes2NetV2, CamPlus).

Uses a model-instance pool (queue.Queue) so concurrent callers each get their
own model copy — no global inference lock, no serialisation.
"""
from __future__ import annotations

import logging
import os
import queue
import threading

import numpy as np

from .base import SpeakerEmbedding

logger = logging.getLogger(__name__)

# Default pool size: half the CPU cores, clamped to [2, 8].
# Override with SPEAKER_MODEL_POOL_SIZE env var.
_DEFAULT_POOL_SIZE = max(2, min((os.cpu_count() or 4) // 2, 8))


class FunasrSpeakerEngine:
    """Base class for funasr AutoModel speaker engines.

    Subclasses set _MODEL_ID_DEFAULT and _ENV_VAR at class level.

    Instead of a single model guarded by a threading.Lock, we keep a pool of
    *pool_size* independent model instances.  ``_extract_sync`` leases one from
    the pool via ``Queue.get()`` (blocking when all are busy — natural
    backpressure) and returns it via ``Queue.put()`` when done.
    """

    _MODEL_ID_DEFAULT: str = ""
    _ENV_VAR: str = ""
    _EMBEDDING_DIM: int = 192

    def __init__(self) -> None:
        self._model_id: str = os.environ.get(self._ENV_VAR) or self._MODEL_ID_DEFAULT
        pool_size_raw = os.environ.get("SPEAKER_MODEL_POOL_SIZE", "")
        try:
            self._pool_size: int = int(pool_size_raw) if pool_size_raw else _DEFAULT_POOL_SIZE
        except ValueError:
            self._pool_size = _DEFAULT_POOL_SIZE
        self._pool: queue.Queue = queue.Queue(maxsize=self._pool_size)
        self._load_lock = threading.Lock()
        self._loaded: bool = False
        self._load_failed: bool = False

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------

    def _build_model(self):
        """Create a *fresh* funasr AutoModel instance."""
        from funasr import AutoModel

        kwargs: dict = dict(
            disable_update=True,
            disable_progress_bar=True,
            disable_log=True,
        )
        if os.path.exists(self._model_id):
            kwargs["model"] = self._MODEL_ID_DEFAULT
            kwargs["model_path"] = self._model_id
        else:
            kwargs["model"] = self._model_id
        return AutoModel(**kwargs)

    def _fill_pool(self) -> None:
        """Load *pool_size* model instances and push them into the queue.

        Called exactly once (lazily, on first extract).  Thread-safe via
        ``_load_lock``.
        """
        if self._loaded:
            return
        if self._load_failed:
            raise RuntimeError(
                f"{self.__class__.__name__} previously failed to load "
                f"— check model path: {self._model_id}"
            )
        with self._load_lock:
            if self._loaded:
                return
            if self._load_failed:
                raise RuntimeError(
                    f"{self.__class__.__name__} previously failed to load "
                    f"— check model path: {self._model_id}"
                )
            try:
                for i in range(self._pool_size):
                    model = self._build_model()
                    self._pool.put(model)
                self._loaded = True
                logger.info(
                    "%s pool ready: %d × %s",
                    self.__class__.__name__,
                    self._pool_size,
                    self._model_id,
                )
            except Exception:
                self._load_failed = True
                logger.error(
                    "%s failed to fill model pool: %s",
                    self.__class__.__name__,
                    self._model_id,
                    exc_info=True,
                )
                raise

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def _extract_sync(self, pcm_bytes: bytes) -> np.ndarray | None:
        self._fill_pool()
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Lease a model instance from the pool.  Blocks when every instance
        # is busy — that's deliberate backpressure instead of unbounded queuing.
        model = self._pool.get()
        try:
            result = model.generate(input=audio)
        except Exception as exc:
            logger.warning("%s extract error: %s", self.__class__.__name__, exc)
            return None
        finally:
            self._pool.put(model)  # always return the instance

        if not result:
            return None
        emb = result[0].get("spk_embedding")
        if emb is None:
            return None
        vec = (
            emb.squeeze().numpy()
            if hasattr(emb, "numpy")
            else np.array(emb, dtype=np.float32).flatten()
        )
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else vec

    async def extract(self, pcm_bytes: bytes) -> SpeakerEmbedding | None:
        import asyncio

        result = await asyncio.to_thread(self._extract_sync, pcm_bytes)
        if result is None:
            return None
        return SpeakerEmbedding(vector=result, dim=len(result))

    async def aclose(self) -> None:
        pass
