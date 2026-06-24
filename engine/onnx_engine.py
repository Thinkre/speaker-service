"""ONNX-based speaker embedding engine — onnxruntime + kaldi-native-fbank.

No torch / torchaudio dependency.
"""
from __future__ import annotations

import logging
import os
import threading

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

from .base import SpeakerEmbedding

logger = logging.getLogger(__name__)

_SR = 16000
_MODEL_PATH = os.environ.get("ERES2NET_MODEL_PATH", "models/onnx/speaker_embedding_v2.onnx")


def _make_fbank() -> knf.OnlineFbank:
    """Create a configured Kaldi fbank extractor (matches 3D-Speaker params)."""
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.samp_freq = _SR
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = 0  # Nyquist (8000Hz at 16kHz)
    return knf.OnlineFbank(opts)


class ONNXSpeakerEngine:
    """Speaker embedding via onnxruntime — no funasr, no torch."""

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
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        fb = _make_fbank()
        fb.accept_waveform(_SR, audio.tolist())
        n = fb.num_frames_ready
        frames = [np.array(fb.get_frame(i)) for i in range(n)]
        feat = np.stack(frames) if frames else np.zeros((0, 80), dtype=np.float32)
        # CMN (cepstral mean normalization)
        feat = feat - feat.mean(0, keepdims=True)
        return feat  # (frames, 80)

    def _extract_sync(self, pcm_bytes: bytes) -> np.ndarray | None:
        self._load()
        feat = self._extract_fbank(pcm_bytes)
        try:
            out = self._session.run(None, {self._input_name: feat.astype(np.float32)[None, :, :]})
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
