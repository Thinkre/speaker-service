"""Base types and protocols for speaker clustering backends."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class SpeakerWindow:
    speaker_id: int
    start_ms: int
    end_ms: int
    is_provisional: bool = False


ONLINE_BACKENDS = frozenset({"spectral", "centroid_bank"})
OFFLINE_BACKENDS = frozenset({"hac", "hdbscan", "diart", "offline_spectral"})


class OnlineClustererBackend(Protocol):
    """增量式：每次 assign() 立即返回当前窗口的 speaker_id。"""
    def assign(self, embedding: np.ndarray, start_ms: int, end_ms: int) -> SpeakerWindow: ...
    def reset(self) -> None: ...


class OfflineClustererBackend(Protocol):
    """批量式：接收全量 embeddings，一次性返回所有 speaker_id。"""
    def cluster(self, embeddings: list[np.ndarray]) -> list[int]: ...
    def reset(self) -> None: ...


# Backwards-compatible alias
ClustererBackend = OnlineClustererBackend


@dataclass
class BackendConfig:
    backend: Literal["centroid_bank", "spectral", "hdbscan", "hac", "diart"] = "centroid_bank"
    asr: Literal["bella"] = "bella"
    vad_params: dict[str, Any] = field(default_factory=dict)
    cluster_params: dict[str, Any] = field(default_factory=dict)
    asr_params: dict[str, Any] = field(default_factory=dict)
