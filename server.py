"""Speaker Embedding gRPC server — ERes2NetV2 / CamPlus."""
from __future__ import annotations

import logging
import os
import signal
import threading
from concurrent import futures

import grpc
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(__file__))

from generated import speaker_pb2, speaker_pb2_grpc

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

_engines: dict[str, object] = {}
_engines_lock = threading.Lock()


def _make_engine(name: str):
    if name in ("eresnetv2", "eres2net"):
        from engine.eres2net import ERes2NetEngine
        return ERes2NetEngine()
    if name == "campplus":
        from engine.campplus import CamPlusEngine
        return CamPlusEngine()
    raise ValueError(f"Unknown engine: {name!r}")


def _get_engine(name: str):
    key = (name or "eresnetv2").lower()
    if key not in _engines:
        with _engines_lock:
            if key not in _engines:
                _engines[key] = _make_engine(key)
                logger.info("Loaded speaker engine: %s", key)
    return _engines[key]


class SpeakerServicer(speaker_pb2_grpc.SpeakerServiceServicer):

    def ExtractEmbedding(self, request, context):
        key = (request.engine or "eresnetv2").lower()
        try:
            engine = _get_engine(key)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return speaker_pb2.ExtractResponse(success=False, error_message=str(exc))

        try:
            result = engine._extract_sync(request.pcm)
        except Exception as exc:
            logger.warning("ExtractEmbedding error engine=%s: %s", key, exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return speaker_pb2.ExtractResponse(success=False, error_message=str(exc))

        if result is None:
            return speaker_pb2.ExtractResponse(success=False, dim=0)

        vec: np.ndarray = result
        return speaker_pb2.ExtractResponse(
            embedding=vec.tolist(),
            dim=len(vec),
            success=True,
        )


def serve() -> None:
    port = os.environ.get("PORT", "50052")
    host = os.environ.get("HOST", "0.0.0.0")
    max_workers = int(os.environ.get("GRPC_WORKERS", os.cpu_count() or 4))
    addr = f"{host}:{port}"

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    speaker_pb2_grpc.add_SpeakerServiceServicer_to_server(SpeakerServicer(), server)
    server.add_insecure_port(addr)
    server.start()
    logger.info("Speaker gRPC server listening on %s (workers=%d)", addr, max_workers)

    signal.signal(signal.SIGTERM, lambda *_: server.stop(grace=5))
    signal.signal(signal.SIGINT,  lambda *_: server.stop(grace=5))
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
