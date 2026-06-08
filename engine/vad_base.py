from typing import Protocol, runtime_checkable, Literal
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpeechSegment:
    start_ms: int
    end_ms: int
    pcm: bytes


@dataclass
class VadEvent:
    kind: Literal["speech_start", "audio_chunk", "speech_end"]
    pcm: bytes = field(default=b"")
    segment: SpeechSegment | None = field(default=None)


@runtime_checkable
class AbstractVADEngine(Protocol):
    def process_chunk(self, pcm_bytes: bytes) -> list[SpeechSegment]: ...
    def flush(self) -> list[SpeechSegment]: ...
    def reset(self) -> None: ...
