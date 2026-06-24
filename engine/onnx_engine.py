"""ONNX-based speaker embedding engine — onnxruntime inference.

Replaces the funasr-based engine with identical output (same fbank + CMN).
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as Kaldi

from .base import SpeakerEmbedding

logger = logging.getLogger(__name__)

_MODEL_PATH = os.environ.get("ERES2NET_MODEL_PATH", "models/onnx/speaker_embedding_v2.onnx")


class ONNXSpeakerEngine:
    """Speaker embedding via onnxruntime — no funasr / PyTorch model dependency."""

    def __init__(self) -> None:
        self._model_path = _MODEL_PATH
        self._lock = threading.Lock()
        self._session: ort.InferenceSession | None = None
        self._loaded = False
        self._load_failed = False

    @property
    def embedding_dim(self) -> int:
        return 192

    def _load(self) -> None:
        if self._loaded:
            return
        if self._load_failed:
            raise RuntimeError(f"ONNX model previously failed to load: {self._model_path}")
        with self._lock:
            if self._loaded:
                return
            if self._load_failed:
                raise RuntimeError(f"ONNX model previously failed to load: {self._model_path}")
            try:
                self._session = ort.InferenceSession(self._model_path)
                self._input_name = self._session.get_inputs()[0].name
                self._loaded = True
                logger.info("ONNX engine loaded: %s", self._model_path)
            except Exception:
                self._load_failed = True
                logger.error("ONNX engine failed to load: %s", self._model_path, exc_info=True)
                raise

    def _extract_fbank(self, pcm_bytes: bytes) -> np.ndarray:
        audio = torch.from_numpy(
            np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        ).unsqueeze(0)
        feat = Kaldi.fbank(audio, num_mel_bins=80, sample_frequency=16000, dither=0)
        feat = feat - feat.mean(0, keepdim=True)  # CMN
        return feat.numpy()

    def _extract_sync(self, pcm_bytes: bytes) -> np.ndarray | None:
        self._load()
        feat = self._extract_fbank(pcm_bytes)
        try:
            out = self._session.run(None, {self._input_name: feat.astype(np.float32)})
        except Exception as exc:
            logger.warning("ONNX inference error: %s", exc)
            return None
        if not out:
            return None
        vec = np.array(out[0]).flatten()
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
