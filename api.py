"""HTTP API for speaker embedding — synchronous, one segment in, one embedding out.

POST /v1/audio/speaker/embedding

Request (JSON):
    url         string   optional  WAV audio URL (16kHz mono)
    base64      string   optional  base64-encoded WAV bytes
    model       string   required  "eresnetv2" | "campplus"
    normalize   bool     optional  L2 normalisation hint (always applied by engine)
    user        string   required  caller identifier (logged per request)
    sample_rate number   optional  sample rate hint; read from WAV header if omitted

Response (JSON):
    task        string   task type ("speaker_embedding")
    task_id     string   request-scoped UUID
    duration    number   audio duration in seconds
    embeddings  list     flat float32 embedding vector (L2-normalised)
    dimensions  number   vector dimension (192 or 512)
    error       string   non-empty on failure
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
import uuid
import wave
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, field_validator
from scipy.signal import resample_poly

load_dotenv()

logger = logging.getLogger(__name__)

# ── auth ───────────────────────────────────────────────────────────────────────
API_TOKEN = os.environ.get("API_TOKEN", "")


def verify_auth(authorization: Annotated[str, Header(alias="Authorization")] = "") -> None:
    """Validate Bearer token when API_TOKEN is configured."""
    if not API_TOKEN:
        logger.warning("API_TOKEN not set — accepting unauthenticated requests")
        return
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization[7:]
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

# ── constants ─────────────────────────────────────────────────────────────────
MAX_PAYLOAD_BYTES = 50 * 1024 * 1024   # 50 MB hard limit
INFERENCE_TIMEOUT = 30.0               # seconds before 504

# ── engine cache (thread-safe) ────────────────────────────────────────────────
_engines: dict[str, Any] = {}
_engines_lock = threading.Lock()


def _get_engine(model: str):
    key = model.lower()
    if key not in _engines:
        with _engines_lock:
            if key not in _engines:
                if key in ("eresnetv2", "eres2net"):
                    from engine.eres2net import ERes2NetEngine
                    _engines[key] = ERes2NetEngine()
                elif key == "campplus":
                    from engine.campplus import CamPlusEngine
                    _engines[key] = CamPlusEngine()
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown model: {model!r}.")
    return _engines[key]


# ── shared HTTP client (reused across requests) ───────────────────────────────
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=30, write=10, pool=5))
    yield
    await _http_client.aclose()


app = FastAPI(title="Speaker Embedding Service", version="1.0.0", lifespan=lifespan)


# ── models ────────────────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    url: str | None = None
    base64: str | None = None
    model: str
    normalize: bool = True   # L2-normalisation hint; always applied by engine
    user: str
    sample_rate: int | None = None  # optional hint; read from WAV header if omitted

    @field_validator("model")
    @classmethod
    def _check_model(cls, v: str) -> str:
        if v.lower() not in ("eresnetv2", "eres2net", "campplus"):
            raise ValueError(f"model must be eresnetv2 or campplus, got {v!r}")
        return v


class EmbeddingResponse(BaseModel):
    task: str = "speaker_embedding"
    task_id: str = ""
    duration: float = 0.0
    embeddings: list[float] = []   # flat L2-normalised float32 vector
    dimensions: int = 0            # 192 or 512
    error: str = ""


# ── helpers ───────────────────────────────────────────────────────────────────

def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int, float]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono WAV, got {wf.getnchannels()} channels.")
        sr = wf.getframerate()
        n = wf.getnframes()
        return wf.readframes(n), sr, n / sr


def _resample(pcm: bytes, src_sr: int, dst_sr: int = 16000) -> bytes:
    logger.warning("Resampling audio from %dHz to %dHz — use 16kHz input for best quality", src_sr, dst_sr)
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    resampled = resample_poly(samples, dst_sr, src_sr).astype(np.int16)
    return resampled.tobytes()


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    try:
        _get_engine("eresnetv2")
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ── route ─────────────────────────────────────────────────────────────────────

@app.post("/v1/audio/speaker/embedding", response_model=EmbeddingResponse)
async def speaker_embedding(
    req: EmbeddingRequest,
    http_req: Request,
    _auth: None = Depends(verify_auth),
) -> EmbeddingResponse:
    task_id = uuid.uuid4().hex
    logger.info("embedding request user=%s model=%s task_id=%s", req.user, req.model, task_id)

    # ── load audio bytes ──────────────────────────────────────────────────────
    if req.base64:
        if len(req.base64) > MAX_PAYLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio payload too large (max 50MB).")
        try:
            wav_bytes = base64.b64decode(req.base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding.")

    elif req.url:
        try:
            r = await _http_client.get(req.url)
            r.raise_for_status()
            if len(r.content) > MAX_PAYLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Audio URL content too large (max 50MB).")
            wav_bytes = r.content
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}")
    else:
        raise HTTPException(status_code=400, detail="Provide either url or base64.")

    # ── decode WAV → PCM ──────────────────────────────────────────────────────
    try:
        pcm, sr, duration = _wav_to_pcm(wav_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV: {exc}")

    if sr != 16000:
        pcm = _resample(pcm, sr)

    # ── extract embedding (with timeout) ──────────────────────────────────────
    try:
        engine = _get_engine(req.model)
        result = await asyncio.wait_for(engine.extract(pcm), timeout=INFERENCE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    if result is None:
        raise HTTPException(status_code=422, detail="Extraction failed (audio too short or model error).")

    vec: np.ndarray = result.vector
    return EmbeddingResponse(
        task="speaker_embedding",
        task_id=task_id,
        duration=round(duration, 3),
        embeddings=vec.tolist(),
        dimensions=len(vec),
    )


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import uvicorn
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("HTTP_PORT", "8080")),
    )
