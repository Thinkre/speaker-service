"""Offline spectral clustering backend — full-batch diarization.

Implements the OfflineClustererBackend Protocol: accepts all embeddings at once
and returns a speaker label per embedding.  Uses the same WeSpeaker affinity
formula (0.5*(1+cos) + top-K hard prune) and eigengap heuristic as the online
SpectralClusterer so offline and online results are consistent.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import SpectralClustering

from .base import _cosine_sim
from .spectral import _prune_affinity, _eigengap_n_components, _maybe_prune, _PRUNE_MIN_N


@dataclass
class OfflineSpectralConfig:
    min_speakers: int = 1
    max_speakers: int = 4
    n_components: int | None = None  # None = auto via eigengap


class OfflineSpectralClusterer:
    """Full-batch spectral clustering for file diarization.

    Accepts all embeddings at once (via ``cluster()``) and returns a list of
    integer speaker labels aligned 1-to-1 with the input list.
    """

    def __init__(self, config: OfflineSpectralConfig | None = None) -> None:
        self._cfg = config or OfflineSpectralConfig()

    def cluster(self, embeddings: list[np.ndarray]) -> list[int]:
        if not embeddings:
            return []
        if len(embeddings) == 1:
            return [0]

        embs = [np.array(e, dtype=np.float32) for e in embeddings]
        # L2-normalise
        for i, e in enumerate(embs):
            norm = np.linalg.norm(e)
            if norm > 1e-8:
                embs[i] = e / norm

        n = len(embs)
        # Build WeSpeaker-style affinity: 0.5*(1+cos), then top-K prune.
        # Vectorised: stack into matrix and use BLAS matmul (embeddings already L2-normalised).
        E = np.stack(embs)  # (n, d)
        cosine = 0.5 * (1.0 + E @ E.T)
        np.fill_diagonal(cosine, 1.0)
        cosine = cosine.astype(np.float32)

        # Apply prune only when n >= _PRUNE_MIN_N to avoid eigengap collapse on small matrices.
        affinity = _maybe_prune(cosine)

        if self._cfg.n_components is not None:
            k = min(self._cfg.n_components, n)
        else:
            k = _eigengap_n_components(
                affinity,
                min_k=self._cfg.min_speakers,
                max_k=min(self._cfg.max_speakers, n),
            )

        k = max(1, min(k, n))

        if k == 1:
            return [0] * n

        try:
            labels = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=42,
                n_init=10,
            ).fit_predict(affinity)
            return [int(x) for x in labels]
        except Exception:
            return [0] * n

    def reset(self) -> None:
        pass  # stateless; nothing to clear
