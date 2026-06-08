"""Speaker verification evaluation — RTTM-guided segment pairs → EER/minDCF.

Flow:
  1. Parse RTTM to get (start_s, dur_s, speaker) segments
  2. Slice raw WAV PCM for each segment (min 0.5 s)
  3. Send each segment to the gRPC service → L2-normalised embedding
  4. Build positive pairs (same speaker) and negative pairs (different speaker)
  5. Compute cosine similarity for every pair
  6. Report EER and minDCF

Usage:
    python client/eval_speaker.py --data data
    python client/eval_speaker.py --data data --engine campplus
    python client/eval_speaker.py --data data --min-dur 0.5 --max-pairs 2000
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from client.speaker_client import SpeakerClient

_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # int16


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class Segment:
    clip: str
    speaker: str
    start_s: float
    end_s: float
    pcm: bytes


# ── RTTM / audio helpers ─────────────────────────────────────────────────────

def _parse_rttm(path: Path) -> list[tuple[float, float, str]]:
    """Return list of (start_s, end_s, speaker)."""
    rows = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        dur = float(parts[4])
        spk = parts[7]
        rows.append((start, start + dur, spk))
    return rows


def _read_pcm(wav_path: Path) -> tuple[bytes, int]:
    """Return (raw_pcm_bytes, sample_rate)."""
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getsampwidth() == 2, "need int16 WAV"
        return wf.readframes(wf.getnframes()), wf.getframerate()


def _slice_pcm(pcm: bytes, start_s: float, end_s: float, sr: int) -> bytes:
    bps = sr * 2
    b0 = int(start_s * bps) & ~1  # align to 2-byte boundary
    b1 = int(end_s * bps) & ~1
    b0 = max(0, min(b0, len(pcm)))
    b1 = max(b0, min(b1, len(pcm)))
    return pcm[b0:b1]


# ── load segments from testset ───────────────────────────────────────────────

def _load_segments(data_dir: Path, min_dur: float) -> list[Segment]:
    segments: list[Segment] = []
    for subdir in ["far", "near"]:
        audio_dir = data_dir / "audio" / subdir
        rttm_dir = data_dir / "rttm" / subdir
        if not audio_dir.exists():
            continue
        for wav_path in sorted(audio_dir.glob("*.wav")):
            # try exact match first, then stem prefix
            rttm_path = rttm_dir / (wav_path.stem + ".rttm")
            if not rttm_path.exists():
                matches = list(rttm_dir.glob(wav_path.stem.split("_")[0] + "*.rttm"))
                if not matches:
                    continue
                rttm_path = matches[0]

            pcm, sr = _read_pcm(wav_path)
            for start_s, end_s, spk in _parse_rttm(rttm_path):
                dur = end_s - start_s
                if dur < min_dur:
                    continue
                seg_pcm = _slice_pcm(pcm, start_s, end_s, sr)
                if len(seg_pcm) < int(min_dur * sr * 2):
                    continue
                segments.append(Segment(
                    clip=wav_path.stem,
                    speaker=spk,
                    start_s=start_s,
                    end_s=end_s,
                    pcm=seg_pcm,
                ))
    return segments


# ── extract embeddings ────────────────────────────────────────────────────────

def _extract_embeddings(
    segments: list[Segment],
    client: SpeakerClient,
    engine: str,
) -> list[np.ndarray | None]:
    embs: list[np.ndarray | None] = []
    for i, seg in enumerate(segments):
        emb = client.extract_embedding(seg.pcm, engine=engine)
        embs.append(emb)
        if (i + 1) % 10 == 0 or (i + 1) == len(segments):
            print(f"  extracted {i+1}/{len(segments)}", end="\r", flush=True)
    print()
    return embs


# ── build pairs ───────────────────────────────────────────────────────────────

def _build_pairs(
    segments: list[Segment],
    embs: list[np.ndarray | None],
    max_pairs: int,
) -> tuple[list[float], list[int]]:
    """Return (scores, labels) where label=1 means same speaker."""
    # group by speaker
    by_spk: dict[str, list[int]] = {}
    for i, (seg, emb) in enumerate(zip(segments, embs)):
        if emb is None:
            continue
        by_spk.setdefault(seg.speaker, []).append(i)

    speakers = list(by_spk.keys())

    pos_pairs: list[tuple[int, int]] = []
    for spk, idxs in by_spk.items():
        for a, b in itertools.combinations(idxs, 2):
            pos_pairs.append((a, b))

    # Lazily sample negative pairs without enumerating the full Cartesian product.
    rng = random.Random(42)
    speaker_pairs = list(itertools.combinations(speakers, 2))
    n_neg_needed = min(len(pos_pairs), max_pairs // 2)
    neg_sample: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    max_attempts = n_neg_needed * 10
    attempts = 0
    while len(neg_sample) < n_neg_needed and attempts < max_attempts:
        s1, s2 = rng.choice(speaker_pairs)
        a = rng.choice(by_spk[s1])
        b = rng.choice(by_spk[s2])
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            neg_sample.append((a, b))
        attempts += 1

    n = min(len(pos_pairs), len(neg_sample), max_pairs // 2)
    if n == 0:
        return [], []

    random.seed(42)
    pos_sample = random.sample(pos_pairs, n)
    neg_sample = neg_sample[:n]

    scores: list[float] = []
    labels: list[int] = []
    for a, b in pos_sample:
        scores.append(float(np.dot(embs[a], embs[b])))  # L2-normalised → cosine
        labels.append(1)
    for a, b in neg_sample:
        scores.append(float(np.dot(embs[a], embs[b])))
        labels.append(0)

    return scores, labels


# ── metrics ───────────────────────────────────────────────────────────────────

def _eer(scores: list[float], labels: list[int]) -> float:
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    thresholds = np.sort(np.unique(scores_arr))

    fars, frrs = [], []
    for t in thresholds:
        preds = (scores_arr >= t).astype(int)
        tp = np.sum((preds == 1) & (labels_arr == 1))
        fp = np.sum((preds == 1) & (labels_arr == 0))
        fn = np.sum((preds == 0) & (labels_arr == 1))
        tn = np.sum((preds == 0) & (labels_arr == 0))
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)
    # find crossing
    diffs = fars - frrs
    idx = np.argmin(np.abs(diffs))
    eer = (fars[idx] + frrs[idx]) / 2
    return float(eer * 100)


def _min_dcf(scores: list[float], labels: list[int], p_target: float = 0.01) -> float:
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    thresholds = np.sort(np.unique(scores_arr))

    c_miss, c_fa = 1.0, 1.0
    best = float("inf")
    for t in thresholds:
        preds = (scores_arr >= t).astype(int)
        tp = np.sum((preds == 1) & (labels_arr == 1))
        fp = np.sum((preds == 1) & (labels_arr == 0))
        fn = np.sum((preds == 0) & (labels_arr == 1))
        tn = np.sum((preds == 0) & (labels_arr == 0))
        pmiss = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        pfa = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        dcf = c_miss * pmiss * p_target + c_fa * pfa * (1 - p_target)
        if dcf < best:
            best = dcf
    return float(best)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker verification eval using RTTM ground truth")
    parser.add_argument("--data", default="data", help="testset root (contains audio/ rttm/)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--engine", default="eresnetv2", choices=["eresnetv2", "campplus"])
    parser.add_argument("--min-dur", type=float, default=0.5, help="min segment duration in seconds")
    parser.add_argument("--max-pairs", type=int, default=5000, help="max pairs per pos/neg class")
    args = parser.parse_args()

    data_dir = Path(args.data)
    print(f"Loading segments from {data_dir} (min_dur={args.min_dur}s) ...")
    segments = _load_segments(data_dir, args.min_dur)
    if not segments:
        print("No segments found.")
        sys.exit(1)

    speakers = {s.speaker for s in segments}
    print(f"  {len(segments)} segments, {len(speakers)} speakers: {sorted(speakers)}")

    print(f"Extracting embeddings via {args.host}:{args.port} (engine={args.engine}) ...")
    with SpeakerClient(host=args.host, port=args.port) as client:
        embs = _extract_embeddings(segments, client, args.engine)

    ok = sum(1 for e in embs if e is not None)
    print(f"  {ok}/{len(segments)} succeeded")

    print("Building pairs ...")
    scores, labels = _build_pairs(segments, embs, args.max_pairs)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  {n_pos} positive pairs, {n_neg} negative pairs")

    if not scores:
        print("Not enough pairs to evaluate.")
        sys.exit(1)

    eer = _eer(scores, labels)
    dcf = _min_dcf(scores, labels)

    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    print()
    print("=" * 50)
    print(f"  EER:          {eer:.2f}%")
    print(f"  minDCF:       {dcf:.4f}")
    print(f"  pos cosine:   mean={np.mean(pos_scores):.3f}  std={np.std(pos_scores):.3f}")
    print(f"  neg cosine:   mean={np.mean(neg_scores):.3f}  std={np.std(neg_scores):.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
