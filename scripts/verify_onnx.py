"""Verify ONNX model matches PyTorch (funasr) — same fbank + CMN → compare cosine.

Usage:
    uv run python scripts/verify_onnx.py
"""
from __future__ import annotations

import sys, wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as Kaldi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.eres2net import ERes2NetEngine

SR = 16000
WAV = Path("data/audio/far/R8002_M8002_MS802.wav")
SEGMENT = (9.35, 10.35)  # 1s slice
ONNX_MODEL = "models/onnx/speaker_embedding_v2.onnx"


def load_pcm(wav_path: Path, start_s: float, end_s: float) -> bytes:
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getframerate() == SR
        wf.setpos(int(start_s * SR))
        return wf.readframes(int((end_s - start_s) * SR))


def fbank(pcm: bytes) -> np.ndarray:
    """torchaudio Kaldi fbank + CMN, matches funasr & 3D-Speaker."""
    audio = torch.from_numpy(
        np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    ).unsqueeze(0)
    feat = Kaldi.fbank(audio, num_mel_bins=80, sample_frequency=SR, dither=0)
    feat = feat - feat.mean(0, keepdim=True)  # CMN
    return feat.numpy()


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def main() -> None:
    if not WAV.exists():
        print(f"WAV not found: {WAV}")
        sys.exit(1)
    if not Path(ONNX_MODEL).exists():
        print(f"ONNX model not found: {ONNX_MODEL}")
        sys.exit(1)

    pcm = load_pcm(WAV, *SEGMENT)
    print(f"Audio: {WAV.name}  [{SEGMENT[0]}s – {SEGMENT[1]}s]  pcm={len(pcm)} bytes")

    feat = fbank(pcm)
    print(f"Fbank: {feat.shape}  mean={feat.mean():.4f}  std={feat.std():.4f}")

    # ── ONNX ────────────────────────────────────────────────────────
    sess = ort.InferenceSession(ONNX_MODEL)
    onnx_inp = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {onnx_inp: feat.astype(np.float32)[None, :, :]})
    onnx_emb = normalize(np.array(onnx_out[0]).flatten())

    # ── PyTorch (funasr) ────────────────────────────────────────────
    engine = ERes2NetEngine()
    pt_raw = engine._extract_sync(pcm)
    pt_emb = normalize(pt_raw)

    # ── Compare ─────────────────────────────────────────────────────
    cosine = float(np.dot(onnx_emb, pt_emb))
    match = cosine > 0.999
    print(f"\nONNX  emb[:6]: {onnx_emb[:6].tolist()}")
    print(f"PT    emb[:6]: {pt_emb[:6].tolist()}")
    print(f"cosine = {cosine:.6f}  {'✅ MATCH' if match else '❌ MISMATCH'}")

    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
