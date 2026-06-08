"""Speaker Embedding gRPC server — ERes2NetV2 / CamPlus."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from concurrent import futures

import grpc
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from generated import speaker_pb2, speaker_pb2_grpc

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _make_engine(engine_name: str):
    name = (engine_name or "eresnetv2").lower()
    if name in ("eresnetv2", "eres2net"):
        from engine.eres2net import ERes2NetEngine
        return ERes2NetEngine()
    if name == "campplus":
        from engine.campplus import CamPlusEngine
        return CamPlusEngine()
    raise ValueError(f"Unknown speaker engine: {name!r}. Choose eresnetv2 | campplus")


class SpeakerServicer(speaker_pb2_grpc.SpeakerServiceServicer):
    """Stateless speaker embedding servicer — one engine per type, lazy-loaded."""

    def __init__(self) -> None:
        self._engines: dict[str, object] = {}

    def _get_engine(self, name: str):
        key = (name or "eresnetv2").lower()
        if key not in self._engines:
            self._engines[key] = _make_engine(key)
            logger.info("Loaded speaker engine: %s", key)
        return self._engines[key]

    def ExtractEmbedding(self, request, context):
        engine = self._get_engine(request.engine)
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(engine.extract(request.pcm))
            loop.close()
        except Exception as exc:
            logger.warning("ExtractEmbedding error: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return speaker_pb2.ExtractResponse(success=False)

        if result is None:
            return speaker_pb2.ExtractResponse(success=False, dim=0)

        vec: np.ndarray = result.vector
        return speaker_pb2.ExtractResponse(
            embedding=vec.tolist(),
            dim=result.dim,
            success=True,
        )


def serve() -> None:
    port = os.environ.get("PORT", "50052")
    host = os.environ.get("HOST", "0.0.0.0")
    addr = f"{host}:{port}"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    speaker_pb2_grpc.add_SpeakerServiceServicer_to_server(SpeakerServicer(), server)
    server.add_insecure_port(addr)
    server.start()
    logger.info("Speaker gRPC server listening on %s", addr)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
