"""Shared base for funasr-based speaker embedding engines (ERes2NetV2, CamPlus)."""
from __future__ import annotations

import logging
import threading

import numpy as np

from .base import SpeakerEmbedding

logger = logging.getLogger(__name__)


class FunasrSpeakerEngine:
    """Base class for funasr AutoModel speaker engines.

    Subclasses set _MODEL_ID_DEFAULT and _ENV_VAR at class level.
    """

    _MODEL_ID_DEFAULT: str = ""
    _ENV_VAR: str = ""
    _EMBEDDING_DIM: int = 192

    def __init__(self) -> None:
        import os
        self._model_id: str = os.environ.get(self._ENV_VAR) or self._MODEL_ID_DEFAULT
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_failed = False

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_failed:
            raise RuntimeError(
                f"{self.__class__.__name__} previously failed to load — check model path: {self._model_id}"
            )
        with self._load_lock:
            if self._model is not None:
                return
            if self._load_failed:
                raise RuntimeError(
                    f"{self.__class__.__name__} previously failed to load — check model path: {self._model_id}"
                )
            try:
                import os
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
                self._model = AutoModel(**kwargs)
                logger.info("%s loaded: %s", self.__class__.__name__, self._model_id)
            except Exception:
                self._load_failed = True
                logger.error(
                    "%s failed to load model: %s", self.__class__.__name__, self._model_id, exc_info=True
                )
                raise

    def _extract_sync(self, pcm_bytes: bytes) -> np.ndarray | None:
        self._load()
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            with self._infer_lock:
                result = self._model.generate(input=audio)
        except Exception as exc:
            logger.warning("%s extract error: %s", self.__class__.__name__, exc)
            return None
        if not result:
            return None
        emb = result[0].get("spk_embedding")
        if emb is None:
            return None
        vec = emb.squeeze().numpy() if hasattr(emb, "numpy") else np.array(emb, dtype=np.float32).flatten()
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
