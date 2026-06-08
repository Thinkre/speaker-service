"""Speaker Embedding gRPC client SDK."""
from __future__ import annotations

import logging
import os
import sys

import grpc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generated import speaker_pb2, speaker_pb2_grpc

logger = logging.getLogger(__name__)


class SpeakerClient:
    """Thin wrapper around the SpeakerService gRPC stub."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50052,
        timeout: float = 15.0,
        tls: bool = False,
        wait_for_ready: bool = False,
        connect_timeout: float = 5.0,
    ) -> None:
        addr = f"{host}:{port}"
        if tls:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(addr, credentials)
        else:
            self._channel = grpc.insecure_channel(addr)

        if wait_for_ready:
            try:
                grpc.channel_ready_future(self._channel).result(timeout=connect_timeout)
            except grpc.FutureTimeoutError:
                self._channel.close()
                raise ConnectionError(f"Could not connect to gRPC server at {addr} within {connect_timeout}s")

        self._stub = speaker_pb2_grpc.SpeakerServiceStub(self._channel)
        self._timeout = timeout

    def extract_embedding(
        self,
        pcm: bytes,
        engine: str = "eresnetv2",
    ) -> np.ndarray | None:
        """Extract L2-normalized speaker embedding. Returns None on failure."""
        req = speaker_pb2.ExtractRequest(pcm=pcm, engine=engine)
        try:
            resp = self._stub.ExtractEmbedding(req, timeout=self._timeout)
        except grpc.RpcError as exc:
            logger.warning("ExtractEmbedding RPC error: %s %s", exc.code(), exc.details())
            return None
        if not resp.success or not resp.embedding:
            if resp.error_message:
                logger.warning("ExtractEmbedding failed: %s", resp.error_message)
            return None
        return np.array(resp.embedding, dtype=np.float32)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "SpeakerClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()
