"""Speaker Embedding HTTP client SDK — wraps POST /v1/audio/speaker/embedding."""

from __future__ import annotations

import base64
import io
import logging
import os
import wave

import httpx
import numpy as np

logger = logging.getLogger(__name__)


def _pcm_to_wav_b64(pcm: bytes, sr: int = 16000) -> str:
    """Encode int16 mono PCM as base64 WAV."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


class SpeakerClient:
    """Thin wrapper around the Speaker Embedding HTTP API."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        timeout: float = 30.0,
        token: str | None = None,
    ) -> None:
        self._url = f"http://{host}:{port}/v1/audio/speaker/embedding"
        self._token = token or os.environ.get("API_TOKEN", "")
        self._client = httpx.Client(timeout=timeout)

    def extract_embedding(
        self,
        pcm: bytes,
        engine: str = "eres2netv2",
        user: str = "speaker-client",
    ) -> np.ndarray | None:
        """Extract L2-normalized speaker embedding from int16 mono 16kHz PCM.

        Returns None on failure.
        """
        b64 = _pcm_to_wav_b64(pcm)
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            resp = self._client.post(
                self._url,
                json={"base64": b64, "model": engine, "user": user},
                headers=headers,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("SpeakerClient HTTP error: %s", exc)
            return None

        data = resp.json()
        if data.get("error") or not data.get("embeddings"):
            if data.get("error"):
                logger.warning("SpeakerClient API error: %s", data["error"])
            return None
        return np.array(data["embeddings"], dtype=np.float32)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SpeakerClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()
