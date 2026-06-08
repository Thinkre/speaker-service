from typing import Protocol, runtime_checkable
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerEmbedding:
    vector: np.ndarray   # L2-normalized, dtype=float32
    dim: int


@runtime_checkable
class AbstractSpeakerEngine(Protocol):
    async def extract(self, pcm_bytes: bytes) -> "SpeakerEmbedding | None": ...
    async def aclose(self) -> None: ...

    @property
    def embedding_dim(self) -> int: ...
