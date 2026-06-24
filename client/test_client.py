"""Smoke test — HTTP Speaker Embedding API, using a real WAV from testset.

Usage:
    uv run python client/test_client.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
import wave
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client.speaker_client import SpeakerClient

HTTP_URL  = "http://localhost:8080/v1/audio/speaker/embedding"
API_TOKEN = os.environ.get("API_TOKEN", "")
WAV_PATH  = Path("data/audio/far/R8002_M8002_MS802.wav")
SEGMENT   = (9.35, 10.35)   # 1s slice, speaker N_SPK8007


def _slice_wav(wav_path: Path, start_s: float, end_s: float) -> tuple[bytes, bytes]:
    """Return (pcm_bytes, wav_bytes) for the given time range."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        wf.setpos(int(start_s * sr))
        n = int((end_s - start_s) * sr)
        pcm = wf.readframes(n)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return pcm, buf.getvalue()


def test_http_sdk(pcm: bytes) -> np.ndarray | None:
    print("\n── HTTP SDK ────────────────────────────────────")
    with SpeakerClient(host="localhost", port=8080) as client:
        emb = client.extract_embedding(pcm, engine="eresnetv2")
    if emb is None:
        print("  FAIL: returned None")
        return None
    print(f"  OK  shape={emb.shape}  norm={np.linalg.norm(emb):.4f}  emb[:3]={emb[:3].tolist()}")
    return emb


async def test_http_raw(wav_bytes: bytes) -> np.ndarray | None:
    print("\n── HTTP Raw ────────────────────────────────────")
    b64 = base64.b64encode(wav_bytes).decode()
    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            HTTP_URL,
            json={"base64": b64, "model": "eresnetv2", "user": "test"},
            headers=headers,
        )
    data = resp.json()
    if data.get("error"):
        print(f"  FAIL: {data['error']}")
        return None
    vec = np.array(data["embeddings"], dtype=np.float32)
    print(f"  OK  dimensions={data['dimensions']}  duration={data['duration']}s  task_id={data.get('task_id', 'N/A')}")
    print(f"      norm={np.linalg.norm(vec):.4f}  emb[:3]={vec[:3].tolist()}")
    return vec


def compare(sdk_emb: np.ndarray, raw_emb: np.ndarray) -> None:
    print("\n── Consistency ──────────────────────────────────")
    cosine = float(np.dot(sdk_emb, raw_emb))
    print(f"  cosine(SDK, Raw) = {cosine:.6f}  (expect ≈ 1.0)")


async def main() -> None:
    if not WAV_PATH.exists():
        print(f"WAV not found: {WAV_PATH}")
        sys.exit(1)

    pcm, wav_bytes = _slice_wav(WAV_PATH, *SEGMENT)
    print(f"Audio: {WAV_PATH.name}  [{SEGMENT[0]}s – {SEGMENT[1]}s]  pcm={len(pcm)} bytes")

    sdk_emb = test_http_sdk(pcm)
    raw_emb = await test_http_raw(wav_bytes)

    if sdk_emb is not None and raw_emb is not None:
        compare(sdk_emb, raw_emb)


if __name__ == "__main__":
    asyncio.run(main())
