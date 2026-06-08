"""Speaker Embedding gRPC client SDK."""
from __future__ import annotations

import os
import sys

import grpc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generated import speaker_pb2, speaker_pb2_grpc


class SpeakerClient:
    """Thin wrapper around the SpeakerService gRPC stub."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50052,
        timeout: float = 15.0,
    ) -> None:
        self._channel = grpc.insecure_channel(f"{host}:{port}")
        self._stub = speaker_pb2_grpc.SpeakerServiceStub(self._channel)
        self._timeout = timeout

    def extract_embedding(
        self,
        pcm: bytes,
        engine: str = "eresnetv2",
    ) -> np.ndarray | None:
        """Extract L2-normalized speaker embedding. Returns None on failure."""
        req = speaker_pb2.ExtractRequest(pcm=pcm, engine=engine)
        resp = self._stub.ExtractEmbedding(req, timeout=self._timeout)
        if not resp.success or not resp.embedding:
            return None
        return np.array(resp.embedding, dtype=np.float32)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "SpeakerClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()
