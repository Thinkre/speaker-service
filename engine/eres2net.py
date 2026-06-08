"""Local ERes2NetV2 speaker embedding engine via funasr.AutoModel."""

import asyncio
import logging
import threading

import numpy as np

from .base import SpeakerEmbedding

logger = logging.getLogger(__name__)

_MODEL_ID_DEFAULT = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
_EMBEDDING_DIM = 192


def _resolve_model_id() -> str:
    import os
    return os.environ.get("ERES2NET_MODEL_PATH") or _MODEL_ID_DEFAULT


class ERes2NetEngine:
    """Local ERes2NetV2 speaker embedding using funasr.AutoModel (lazy-loaded)."""

    def __init__(self) -> None:
        self._model_id = _resolve_model_id()
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def embedding_dim(self) -> int:
        return _EMBEDDING_DIM

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import os
            from funasr import AutoModel
            model_id = self._model_id
            kwargs: dict = dict(
                disable_update=True,
                disable_progress_bar=True,
                disable_log=True,
            )
            # 로컬 경로인 경우 model_path로 분리 (funasr는 model=로컬경로를 ID로 인식 못함)
            if os.path.exists(model_id):
                kwargs["model"] = _MODEL_ID_DEFAULT
                kwargs["model_path"] = model_id
            else:
                kwargs["model"] = model_id
            self._model = AutoModel(**kwargs)
            logger.info("ERes2NetV2 loaded: %s", model_id)

    def _extract_sync(self, pcm_bytes: bytes) -> np.ndarray | None:
        self._load()
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            result = self._model.generate(input=audio)
        except Exception as exc:
            logger.warning("ERes2Net extract error: %s", exc)
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
        result = await asyncio.to_thread(self._extract_sync, pcm_bytes)
        if result is None:
            return None
        return SpeakerEmbedding(vector=result, dim=len(result))

    async def aclose(self) -> None:
        pass
