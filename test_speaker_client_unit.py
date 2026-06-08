"""Unit tests for SpeakerClient — mock gRPC channel."""
import numpy as np
from unittest.mock import MagicMock, patch
from client.speaker_client import SpeakerClient
from generated import speaker_pb2


def test_extract_embedding_success():
    with patch("client.speaker_client.grpc.insecure_channel"), \
         patch("client.speaker_client.speaker_pb2_grpc.SpeakerServiceStub") as StubCls:
        stub = MagicMock()
        StubCls.return_value = stub
        vec = [0.1] * 192
        stub.ExtractEmbedding.return_value = speaker_pb2.ExtractResponse(
            embedding=vec, dim=192, success=True
        )

        client = SpeakerClient()
        result = client.extract_embedding(b"\x00" * 16000)

        assert result is not None
        assert result.shape == (192,)
        assert abs(result[0] - 0.1) < 1e-5


def test_extract_embedding_failure_returns_none():
    with patch("client.speaker_client.grpc.insecure_channel"), \
         patch("client.speaker_client.speaker_pb2_grpc.SpeakerServiceStub") as StubCls:
        stub = MagicMock()
        StubCls.return_value = stub
        stub.ExtractEmbedding.return_value = speaker_pb2.ExtractResponse(
            embedding=[], dim=0, success=False
        )

        client = SpeakerClient()
        result = client.extract_embedding(b"\x00" * 100)

        assert result is None
