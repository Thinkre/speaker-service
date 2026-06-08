"""HTTP API for speaker embedding — synchronous, one segment in, one embedding out.

POST /v1/audio/speaker/embedding

Request (JSON):
    url         string   optional  WAV audio URL (16kHz mono)
    base64      string   optional  base64-encoded WAV bytes
    model       string   required  "eresnetv2" | "campplus"
    normalize   bool     optional  default true
    user        string   required  caller identifier

Response (JSON):
    embeddings  list     each item: {id, start, end, confidence, embedding, dimensions}
    error       string   non-empty on failure
"""
from __future__ import annotations

import base64
import io
import wave
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from engine.eres2net import ERes2NetEngine
from engine.campplus import CamPlusEngine

app = FastAPI(title="Speaker Embedding Service", version="1.0.0")

_engines: dict[str, Any] = {}


def _get_engine(model: str):
    key = model.lower()
    if key not in _engines:
        if key in ("eresnetv2", "eres2net"):
            _engines[key] = ERes2NetEngine()
        elif key == "campplus":
            _engines[key] = CamPlusEngine()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model!r}. Use eresnetv2 or campplus.")
    return _engines[key]


# ── models ────────────────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    url: str | None = None
    base64: str | None = None
    model: str
    normalize: bool = True
    user: str

    @field_validator("model")
    @classmethod
    def _check_model(cls, v: str) -> str:
        if v.lower() not in ("eresnetv2", "eres2net", "campplus"):
            raise ValueError(f"model must be eresnetv2 or campplus, got {v!r}")
        return v


class EmbeddingSegment(BaseModel):
    id: int
    start: float
    end: float
    confidence: float
    embedding: list[float]
    dimensions: int


class EmbeddingResponse(BaseModel):
    embeddings: list[EmbeddingSegment]
    error: str = ""


# ── helpers ───────────────────────────────────────────────────────────────────

def _wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int, float]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        return wf.readframes(n), sr, n / sr


def _resample(pcm: bytes, src_sr: int, dst_sr: int = 16000) -> bytes:
    if src_sr == dst_sr:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    new_len = int(len(samples) * dst_sr / src_sr)
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, new_len),
        np.arange(len(samples)),
        samples,
    ).astype(np.int16)
    return resampled.tobytes()


# ── route ─────────────────────────────────────────────────────────────────────

@app.post("/v1/audio/speaker/embedding", response_model=EmbeddingResponse)
async def speaker_embedding(req: EmbeddingRequest) -> EmbeddingResponse:
    # load audio bytes
    if req.base64:
        try:
            wav_bytes = base64.b64decode(req.base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding.")
    elif req.url:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(req.url)
                r.raise_for_status()
                wav_bytes = r.content
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}")
    else:
        raise HTTPException(status_code=400, detail="Provide either url or base64.")

    # decode WAV → PCM
    try:
        pcm, sr, duration = _wav_to_pcm(wav_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV: {exc}")

    if sr != 16000:
        pcm = _resample(pcm, sr)

    # extract embedding
    try:
        engine = _get_engine(req.model)
        result = await engine.extract(pcm)
    except HTTPException:
        raise
    except Exception as exc:
        return EmbeddingResponse(embeddings=[], error=str(exc))

    if result is None:
        return EmbeddingResponse(
            embeddings=[],
            error="Extraction failed (audio too short or model error).",
        )

    vec: np.ndarray = result.vector
    return EmbeddingResponse(
        embeddings=[EmbeddingSegment(
            id=0,
            start=0.0,
            end=round(duration, 3),
            confidence=1.0,
            embedding=vec.tolist(),
            dimensions=len(vec),
        )],
    )


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("HTTP_PORT", "8080")),
    )
