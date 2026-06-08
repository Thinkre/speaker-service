"""FireRed VAD engine — DFSMN streaming VAD with per-frame decisions."""

from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np

from .vad_base import SpeechSegment, VadEvent

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_BYTES_PER_MS = _SAMPLE_RATE * 2 // 1000   # int16 mono
_FRAME_MS = 10                               # FireRed VAD outputs 10ms frames

# Default model path — resolved relative to CWD at runtime.
# Override via FIRERED_VAD_PATH env var.
_DEFAULT_RELATIVE = "./models/FireRedVAD/Stream-VAD"


class FireRedVADEngine:
    """FireRedVAD streaming VAD (DFSMN-based, F1 97.57%).

    Requires:
        pip install git+https://github.com/FireRedTeam/FireRedVAD.git
    Model: models/FireRedVAD/Stream-VAD  (or set FIRERED_VAD_PATH)
    """

    def __init__(
        self,
        model_path: str | None = None,
        speech_threshold: float = 0.5,
        min_silence_ms: int = 200,
        speech_pad_ms: int = 30,
        ring_buffer_ms: int = 60_000,
    ) -> None:
        self._model_path = os.path.abspath(
            model_path
            or os.environ.get("FIRERED_VAD_PATH")
            or _DEFAULT_RELATIVE
        )
        self._speech_threshold = speech_threshold
        self._min_silence_ms = min_silence_ms
        self._speech_pad_ms = speech_pad_ms
        self._ring_capacity = ring_buffer_ms * _BYTES_PER_MS
        self._model = None
        self._reset_state()

    def _load(self) -> None:
        if self._model is not None:
            return
        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
        cfg = FireRedStreamVadConfig(
            use_gpu=False,
            smooth_window_size=5,
            speech_threshold=self._speech_threshold,
            pad_start_frame=5,
            min_speech_frame=8,
            max_speech_frame=2000,
            min_silence_frame=max(1, self._min_silence_ms // _FRAME_MS),
        )
        self._model = FireRedStreamVad.from_pretrained(self._model_path, cfg)
        logger.info("FireRedVAD loaded from %s", self._model_path)

    def _reset_state(self) -> None:
        self._elapsed_ms: int = 0
        self._speech_buf: list[bytes] = []
        self._speech_start_ms: int = 0
        self._silence_ms: int = 0
        self._in_speech: bool = False
        # Ring buffer for PCM slice extraction
        self._ring: deque[tuple[int, bytes]] = deque()
        self._ring_bytes: int = 0

    def _push_ring(self, start_ms: int, pcm: bytes) -> None:
        self._ring.append((start_ms, pcm))
        self._ring_bytes += len(pcm)
        while self._ring_bytes > self._ring_capacity and self._ring:
            _, old = self._ring.popleft()
            self._ring_bytes -= len(old)

    def _slice_pcm(self, start_ms: int, end_ms: int) -> bytes:
        result = bytearray()
        for chunk_start, chunk_pcm in self._ring:
            chunk_ms = len(chunk_pcm) // _BYTES_PER_MS
            chunk_end = chunk_start + chunk_ms
            if chunk_end <= start_ms or chunk_start >= end_ms:
                continue
            b0 = (max(start_ms, chunk_start) - chunk_start) * _BYTES_PER_MS
            b1 = (min(end_ms, chunk_end) - chunk_start) * _BYTES_PER_MS
            result.extend(chunk_pcm[b0:b1])
        return bytes(result) if result else bytes((end_ms - start_ms) * _BYTES_PER_MS)

    def process_chunk(self, pcm_bytes: bytes) -> list[SpeechSegment]:
        self._load()
        chunk_start_ms = self._elapsed_ms
        chunk_ms = len(pcm_bytes) // _BYTES_PER_MS
        self._push_ring(chunk_start_ms, pcm_bytes)

        # detect_chunk expects int16 array (AudioFeat.extract handles normalization internally)
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        try:
            frame_results = self._model.detect_chunk(samples)
        except Exception as exc:
            logger.warning("FireRedVAD chunk error: %s", exc)
            self._elapsed_ms += chunk_ms
            return []

        # Aggregate frame decisions into the same state machine as SileroVAD
        is_speech = any(f.is_speech for f in frame_results) if frame_results else False
        segments = self._update_state(pcm_bytes, chunk_ms, is_speech)
        return segments

    def _update_state(self, pcm_bytes: bytes, chunk_ms: int, is_speech: bool) -> list[SpeechSegment]:
        segments: list[SpeechSegment] = []
        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._speech_start_ms = max(0, self._elapsed_ms - self._speech_pad_ms)
                self._silence_ms = 0
            self._speech_buf.append(pcm_bytes)
            self._silence_ms = 0
        else:
            if self._in_speech:
                self._silence_ms += chunk_ms
                self._speech_buf.append(pcm_bytes)
                if self._silence_ms >= self._min_silence_ms:
                    end_ms = self._elapsed_ms - self._silence_ms + self._speech_pad_ms
                    segments.append(SpeechSegment(
                        start_ms=self._speech_start_ms,
                        end_ms=end_ms,
                        pcm=self._slice_pcm(self._speech_start_ms, end_ms),
                    ))
                    self._speech_buf = []
                    self._in_speech = False
                    self._silence_ms = 0
        self._elapsed_ms += chunk_ms
        return segments

    def flush(self) -> list[SpeechSegment]:
        if not self._in_speech or not self._speech_buf:
            return []
        end_ms = self._elapsed_ms
        seg = SpeechSegment(
            start_ms=self._speech_start_ms,
            end_ms=end_ms,
            pcm=self._slice_pcm(self._speech_start_ms, end_ms),
        )
        self._speech_buf = []
        self._in_speech = False
        return [seg]

    def reset(self) -> None:
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass
        self._reset_state()

    def process_chunk_events(self, pcm_bytes: bytes) -> list[VadEvent]:
        was_in_speech = self._in_speech
        segments = self.process_chunk(pcm_bytes)

        events: list[VadEvent] = []

        if not was_in_speech and self._in_speech:
            events.append(VadEvent(kind="speech_start"))

        if self._in_speech:
            events.append(VadEvent(kind="audio_chunk", pcm=pcm_bytes))

        for seg in segments:
            events.append(VadEvent(kind="speech_end", segment=seg))

        return events
