"""Speaker service evaluation script.

Evaluates DER/JER by:
1. Loading local FireRed VAD (direct import, not via gRPC) to segment audio
2. Calling speaker gRPC service for embeddings
3. Running online (streaming SpectralClusterer) and/or offline (full-batch
   OfflineSpectralClusterer) clustering locally
4. Comparing with reference RTTM using pyannote.metrics

Usage:
    python client/eval_speaker.py --testset ../data/testsets/alimeeting_mini
    python client/eval_speaker.py --testset ../data/testsets/alimeeting_mini --mode offline
    python client/eval_speaker.py --testset ../data/testsets/alimeeting_mini --host 192.168.1.10
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client.speaker_client import SpeakerClient
from clustering.spectral import SpectralClusterer, SpectralConfig
from clustering.offline_spectral import OfflineSpectralClusterer, OfflineSpectralConfig

_CHUNK_BYTES = 3072   # 96ms @ 16kHz int16
_SAMPLE_RATE = 16000
_BYTES_PER_MS = _SAMPLE_RATE * 2 // 1000
_WINDOW_MS = 1000
_STEP_MS = 500

# WeSpeaker baseline from CLAUDE.md (FireRed + step=0.5, alimeeting_mini)
_BASELINE_DER = {
    "R8002_M8002_MS802": 16.26,
    "R8008_M8014_MS807": 14.39,
    "R8009_M8021_MS810": 16.26,
    "R8002_M8002_N_SPK8005": 31.95,
    "avg": 17.1,
}


@dataclass
class ClipResult:
    clip: str
    mode: str         # "online" | "offline"
    der: float
    jer: float
    miss: float
    false_alarm: float
    confusion: float
    n_ref_speakers: int
    n_hyp_speakers: int
    baseline_der: float


def _load_vad():
    """Load FireRed VAD locally (no gRPC)."""
    parent = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, parent)
    from engine.firered_vad import FireRedVADEngine
    return FireRedVADEngine(speech_threshold=0.10, min_silence_ms=100)


def _parse_rttm(path: Path):
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            ann[Segment(start, start + dur)] = spk
    return ann


def _write_rttm_from_windows(
    window_spans: list[tuple[int, int, int]],  # (start_ms, end_ms, speaker_id)
    clip_id: str,
    path: Path,
) -> None:
    """Write RTTM from overlapping windows.

    Windows are 1s wide stepped every 0.5s.  We resolve overlap by assigning
    each 0.5s slot to the majority speaker, then merge consecutive same-speaker
    slots.  Windows with speaker_id < 0 (UNKNOWN) are skipped.
    """
    slots: dict[int, list[int]] = {}
    for start_ms, end_ms, spk_id in window_spans:
        if spk_id < 0:
            continue
        for t in range(start_ms, end_ms, _STEP_MS):
            slots.setdefault(t, []).append(spk_id)

    resolved = {
        t: collections.Counter(ids).most_common(1)[0][0]
        for t, ids in slots.items()
    }

    rows: list[tuple[int, int, int]] = []
    for t in sorted(resolved):
        sid = resolved[t]
        if rows and rows[-1][2] == sid and rows[-1][1] == t:
            rows[-1] = (rows[-1][0], t + _STEP_MS, sid)
        else:
            rows.append((t, t + _STEP_MS, sid))

    with open(path, "w") as f:
        for start_ms, end_ms, sid in rows:
            f.write(
                f"SPEAKER {clip_id} 1 {start_ms/1000:.3f} {(end_ms-start_ms)/1000:.3f} "
                f"<NA> <NA> SPEAKER_{sid} <NA> <NA>\n"
            )


def _compute_der_jer(
    ref_rttm: Path,
    hyp_rttm_path: str,
) -> tuple[float, float, float, float, float]:
    """Return (der, jer, miss, fa, conf) as fractions (multiply by 100 for %)."""
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    ref = _parse_rttm(ref_rttm)
    hyp = _parse_rttm(Path(hyp_rttm_path))
    der_metric = DiarizationErrorRate(collar=0.25)
    jer_metric = JaccardErrorRate(collar=0.25)
    der_val = float(der_metric(ref, hyp))
    jer_val = float(jer_metric(ref, hyp))
    detail = der_metric(ref, hyp, detailed=True)
    return (
        der_val,
        jer_val,
        float(detail.get("missed detection", 0.0)),
        float(detail.get("false alarm", 0.0)),
        float(detail.get("confusion", 0.0)),
    )


def _count_ref_speakers(ref_rttm: Path) -> int:
    speakers: set[str] = set()
    with open(ref_rttm) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                speakers.add(parts[7])
    return len(speakers)


def _extract_windows(
    segments,
    client: SpeakerClient,
    engine: str,
) -> list[tuple[int, int, np.ndarray]]:
    """Return list of (start_ms, end_ms, embedding) for all valid windows."""
    result = []
    for seg in segments:
        seg_len_ms = seg.end_ms - seg.start_ms
        if seg_len_ms < 200:
            continue
        pos = 0
        while pos < seg_len_ms:
            win_start_ms = seg.start_ms + pos
            win_end_ms = min(seg.end_ms, win_start_ms + _WINDOW_MS)
            b0 = pos * _BYTES_PER_MS
            b1 = (pos + (win_end_ms - win_start_ms)) * _BYTES_PER_MS
            win_pcm = seg.pcm[b0:b1]
            if len(win_pcm) < 200 * _BYTES_PER_MS:
                break
            emb = client.extract_embedding(win_pcm, engine=engine)
            if emb is not None:
                result.append((win_start_ms, win_end_ms, emb))
            pos += _STEP_MS
    return result


def _eval_clip(
    wav_path: Path,
    rttm_path: Path,
    client: SpeakerClient,
    vad,
    engine: str,
    clip_id: str,
    mode: str,
) -> list[ClipResult]:
    """Evaluate one clip. Returns 1 or 2 ClipResult depending on mode."""
    import wave

    n_ref_spk = _count_ref_speakers(rttm_path)
    baseline = _BASELINE_DER.get(clip_id, _BASELINE_DER["avg"])

    def _empty(m: str) -> ClipResult:
        return ClipResult(
            clip=clip_id, mode=m, der=100.0, jer=100.0, miss=100.0,
            false_alarm=0.0, confusion=0.0,
            n_ref_speakers=n_ref_spk, n_hyp_speakers=0,
            baseline_der=baseline,
        )

    with wave.open(str(wav_path), "rb") as wf:
        pcm_data = wf.readframes(wf.getnframes())

    vad.reset()
    segments = []
    for i in range(0, len(pcm_data), _CHUNK_BYTES):
        segments.extend(vad.process_chunk(pcm_data[i: i + _CHUNK_BYTES]))
    segments.extend(vad.flush())

    if not segments:
        modes = ["online", "offline"] if mode == "both" else [mode]
        return [_empty(m) for m in modes]

    windows = _extract_windows(segments, client, engine)
    if not windows:
        modes = ["online", "offline"] if mode == "both" else [mode]
        return [_empty(m) for m in modes]

    results = []

    # ── Online mode ──────────────────────────────────────────────────────────
    if mode in ("online", "both"):
        cfg = SpectralConfig(min_speakers=1, max_speakers=max(4, n_ref_spk + 1))
        clusterer = SpectralClusterer(cfg)
        spans = []
        for start_ms, end_ms, emb in windows:
            w = clusterer.assign(emb, start_ms, end_ms)
            spans.append((w.start_ms, w.end_ms, w.speaker_id))

        with tempfile.NamedTemporaryFile(suffix=".rttm", delete=False) as tmp:
            hyp_path = tmp.name
        _write_rttm_from_windows(spans, clip_id, Path(hyp_path))
        der, jer, miss, fa, conf = _compute_der_jer(rttm_path, hyp_path)
        os.unlink(hyp_path)

        n_hyp = len(set(s for _, _, s in spans if s >= 0))
        results.append(ClipResult(
            clip=clip_id, mode="online",
            der=round(der * 100, 2), jer=round(jer * 100, 2),
            miss=round(miss * 100, 2), false_alarm=round(fa * 100, 2),
            confusion=round(conf * 100, 2),
            n_ref_speakers=n_ref_spk, n_hyp_speakers=n_hyp,
            baseline_der=baseline,
        ))

    # ── Offline mode ─────────────────────────────────────────────────────────
    if mode in ("offline", "both"):
        embs = [emb for _, _, emb in windows]
        cfg = OfflineSpectralConfig(min_speakers=1, max_speakers=max(4, n_ref_spk + 1))
        labels = OfflineSpectralClusterer(cfg).cluster(embs)
        spans = [
            (windows[i][0], windows[i][1], labels[i])
            for i in range(len(windows))
        ]

        with tempfile.NamedTemporaryFile(suffix=".rttm", delete=False) as tmp:
            hyp_path = tmp.name
        _write_rttm_from_windows(spans, clip_id, Path(hyp_path))
        der, jer, miss, fa, conf = _compute_der_jer(rttm_path, hyp_path)
        os.unlink(hyp_path)

        n_hyp = len(set(labels))
        results.append(ClipResult(
            clip=clip_id, mode="offline",
            der=round(der * 100, 2), jer=round(jer * 100, 2),
            miss=round(miss * 100, 2), false_alarm=round(fa * 100, 2),
            confusion=round(conf * 100, 2),
            n_ref_speakers=n_ref_spk, n_hyp_speakers=n_hyp,
            baseline_der=baseline,
        ))

    return results


def _collect_clips(testset: Path) -> list[tuple[Path, Path, str]]:
    triples = []
    for subdir in ["far", "near"]:
        audio_dir = testset / "audio" / subdir
        rttm_dir = testset / "rttm" / subdir
        if not audio_dir.exists():
            continue
        for wav in sorted(audio_dir.glob("*.wav")):
            rttm = rttm_dir / (wav.stem + ".rttm")
            if not rttm.exists():
                session = "_".join(wav.stem.split("_")[:2])
                rttm = rttm_dir / (session + ".rttm")
            if rttm.exists():
                triples.append((wav, rttm, wav.stem))
    return triples


def _avg(results: list[ClipResult], field: str) -> float:
    vals = [getattr(r, field) for r in results]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _print_table(results: list[ClipResult]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        modes = sorted(set(r.mode for r in results))
        console = Console()

        for m in modes:
            rs = [r for r in results if r.mode == m]
            table = Table(title=f"Speaker Evaluation — {m.upper()} clustering", show_footer=True)
            table.add_column("Clip", style="cyan")
            table.add_column("DER%", justify="right", style="green")
            table.add_column("JER%", justify="right")
            table.add_column("Miss%", justify="right")
            table.add_column("FA%", justify="right")
            table.add_column("Conf%", justify="right")
            table.add_column("Baseline%", justify="right", style="yellow")
            for r in rs:
                table.add_row(r.clip, f"{r.der:.2f}", f"{r.jer:.2f}",
                              f"{r.miss:.2f}", f"{r.false_alarm:.2f}",
                              f"{r.confusion:.2f}", f"{r.baseline_der:.2f}")
            table.add_row(
                "[bold]Average[/bold]",
                f"[bold]{_avg(rs,'der'):.2f}[/bold]",
                f"{_avg(rs,'jer'):.2f}", "", "", "",
                f"{_BASELINE_DER['avg']:.2f}",
            )
            console.print(table)

        if len(modes) == 2:
            cmp = Table(title="Online vs Offline comparison", show_footer=True)
            cmp.add_column("Clip", style="cyan")
            cmp.add_column("Online DER%", justify="right")
            cmp.add_column("Offline DER%", justify="right", style="green")
            cmp.add_column("Baseline DER%", justify="right", style="yellow")
            clips = sorted(set(r.clip for r in results))
            for clip in clips:
                on = next((r for r in results if r.clip == clip and r.mode == "online"), None)
                off = next((r for r in results if r.clip == clip and r.mode == "offline"), None)
                cmp.add_row(
                    clip,
                    f"{on.der:.2f}" if on else "—",
                    f"{off.der:.2f}" if off else "—",
                    f"{on.baseline_der:.2f}" if on else "—",
                )
            on_rs = [r for r in results if r.mode == "online"]
            off_rs = [r for r in results if r.mode == "offline"]
            cmp.add_row(
                "[bold]Average[/bold]",
                f"{_avg(on_rs,'der'):.2f}",
                f"[bold]{_avg(off_rs,'der'):.2f}[/bold]",
                f"{_BASELINE_DER['avg']:.2f}",
            )
            console.print(cmp)

    except ImportError:
        modes = sorted(set(r.mode for r in results))
        for m in modes:
            rs = [r for r in results if r.mode == m]
            print(f"\n=== {m.upper()} Results ===")
            print(f"{'Clip':<30} {'DER%':>7} {'JER%':>7} {'Base%':>7}")
            print("-" * 52)
            for r in rs:
                print(f"{r.clip:<30} {r.der:>7.2f} {r.jer:>7.2f} {r.baseline_der:>7.2f}")
            print(f"{'Average':<30} {_avg(rs,'der'):>7.2f}")


def _write_json(results: list[ClipResult], out_dir: Path, ts: str) -> Path:
    modes = sorted(set(r.mode for r in results))
    summary: dict = {"baseline_avg_der": _BASELINE_DER["avg"]}
    for m in modes:
        rs = [r for r in results if r.mode == m]
        summary[f"{m}_avg_der"] = _avg(rs, "der")
        summary[f"{m}_avg_jer"] = _avg(rs, "jer")
    data = {
        "timestamp": ts,
        "clips": [asdict(r) for r in results],
        "summary": summary,
    }
    path = out_dir / f"eval_speaker_{ts}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def _write_markdown(results: list[ClipResult], out_dir: Path, ts: str) -> Path:
    modes = sorted(set(r.mode for r in results))
    lines = [
        "# Speaker Service Evaluation Report",
        "",
        f"**Date:** {ts}",
        "",
        "## Summary",
        "",
    ]

    if len(modes) == 2:
        on_avg = _avg([r for r in results if r.mode == "online"], "der")
        off_avg = _avg([r for r in results if r.mode == "offline"], "der")
        lines += [
            "| Metric | Online | Offline | WeSpeaker Baseline |",
            "|--------|-------:|--------:|--------------------|",
            f"| Avg DER% | {on_avg:.2f} | **{off_avg:.2f}** | {_BASELINE_DER['avg']:.2f} |",
            "",
        ]
    else:
        m = modes[0]
        rs = [r for r in results if r.mode == m]
        avg = _avg(rs, "der")
        lines += [
            f"| Metric | {m.capitalize()} | WeSpeaker Baseline |",
            "|--------|-------:|--------------------|",
            f"| Avg DER% | {avg:.2f} | {_BASELINE_DER['avg']:.2f} |",
            "",
        ]

    for m in modes:
        rs = [r for r in results if r.mode == m]
        lines += [
            f"## {m.capitalize()} Clustering",
            "",
            "| Clip | DER% | JER% | Miss% | FA% | Conf% | Ref Spk | Hyp Spk | Baseline% |",
            "|------|-----:|-----:|------:|----:|------:|--------:|--------:|----------:|",
        ]
        for r in rs:
            lines.append(
                f"| {r.clip} | {r.der:.2f} | {r.jer:.2f} | {r.miss:.2f} | "
                f"{r.false_alarm:.2f} | {r.confusion:.2f} | {r.n_ref_speakers} | "
                f"{r.n_hyp_speakers} | {r.baseline_der:.2f} |"
            )
        avg_der = _avg(rs, "der")
        avg_jer = _avg(rs, "jer")
        lines += [
            f"| **Average** | **{avg_der:.2f}** | **{avg_jer:.2f}** | | | | | | **{_BASELINE_DER['avg']:.2f}** |",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- Baseline: WeSpeaker ERes2NetV2 + FireRed VAD (step=0.5), alimeeting_mini, collar=0.25s",
        "- Online: streaming SpectralClusterer (incremental assign per window)",
        "- Offline: full-batch OfflineSpectralClusterer (sees all embeddings before clustering)",
        "- Both use: gRPC speaker embedding service + local FireRed VAD + 1s/0.5s sliding window",
        "- collar = 0.25s (industry standard)",
    ]

    path = out_dir / f"eval_speaker_{ts}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Speaker gRPC service on alimeeting_mini")
    parser.add_argument("--testset", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--engine", default="eresnetv2", choices=["eresnetv2", "campplus"])
    parser.add_argument("--mode", default="both", choices=["online", "offline", "both"])
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    testset = Path(args.testset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = _collect_clips(testset)
    if not triples:
        print(f"No clips found in {testset}")
        sys.exit(1)

    print("Loading local FireRed VAD ...")
    vad = _load_vad()

    print(f"Found {len(triples)} clips. Connecting to {args.host}:{args.port} (mode={args.mode}) ...")

    all_results: list[ClipResult] = []
    with SpeakerClient(host=args.host, port=args.port) as client:
        for i, (wav, rttm, clip_id) in enumerate(triples):
            print(f"  [{i+1}/{len(triples)}] {wav.name} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            clip_results = _eval_clip(wav, rttm, client, vad, args.engine, clip_id, args.mode)
            elapsed = time.perf_counter() - t0
            summary = "  ".join(f"{r.mode}={r.der:.1f}%" for r in clip_results)
            print(f"{summary}  ({elapsed:.1f}s)")
            all_results.extend(clip_results)

    _print_table(all_results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = _write_json(all_results, out_dir, ts)
    md_path = _write_markdown(all_results, out_dir, ts)
    print(f"\nReports saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
