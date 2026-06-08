"""SpectralClusterer — windowed spectral clustering for online speaker diarization."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import SpectralClustering

from .base import SpeakerWindow, _cosine_sim

# Returned when no reliable speaker can be assigned (cold-start or low similarity).
# Maps to UNKNOWN_SPEAKER=-1 in the pipeline, which marks the result as provisional.
_UNKNOWN = -1

# Minimum cosine similarity required to accept a nearest-confirmed match.
# Below this the embedding is too dissimilar to any known speaker → return _UNKNOWN.
_MIN_PROVISIONAL_SIM = 0.3

_EMA_ALPHA = 0.85  # weight of existing mean when updating with a new embedding

def _ema_mean(existing: np.ndarray, new: np.ndarray) -> np.ndarray:
    updated = _EMA_ALPHA * existing + (1 - _EMA_ALPHA) * new
    norm = np.linalg.norm(updated)
    return updated / norm if norm > 1e-8 else updated


def _prune_affinity(M: np.ndarray) -> np.ndarray:
    """WeSpeaker-style hard prune: top-K neighbours → 1.0, rest → 0.0.

    Only effective when n >= 16 (keep=10 must represent ≤62.5% of edges to
    create meaningful block structure).  For smaller matrices, use the raw
    cosine affinity via _maybe_prune() instead of calling this directly.
    """
    n = M.shape[0]
    keep = min(10, n - 2)   # at least 2 rows zeroed; valid for n >= 4
    result = np.copy(M)
    for i in range(n):
        idx = np.argsort(result[i])
        result[i, idx[:-keep]] = 0.0
        result[i, idx[-keep:]] = 1.0
    return 0.5 * (result + result.T)


# Minimum matrix size for _prune_affinity to produce useful block-diagonal structure.
# Below this, keep=10 covers > 62.5% of edges → near-uniform graph → eigengap collapses.
_PRUNE_MIN_N = 16


def _maybe_prune(M: np.ndarray) -> np.ndarray:
    """Apply _prune_affinity only when n >= _PRUNE_MIN_N; return raw matrix otherwise."""
    return _prune_affinity(M) if M.shape[0] >= _PRUNE_MIN_N else M


def _eigengap_n_components(affinity: np.ndarray, min_k: int, max_k: int) -> int:
    """Estimate number of clusters via eigengap heuristic on the affinity matrix."""
    n = affinity.shape[0]
    degree = np.maximum(affinity.sum(axis=1), 1e-8)
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    norm_affinity = d_inv_sqrt[:, None] * affinity * d_inv_sqrt[None, :]
    eigvals = np.sort(np.linalg.eigvalsh(norm_affinity))[::-1]
    cap = max(min_k, min(max_k, n - 1))
    gaps = np.diff(eigvals[:cap + 1])
    best_k = int(np.argmax(-gaps)) + 1
    return max(min_k, min(best_k, max_k))


@dataclass
class SpectralConfig:
    window_size: int = 20
    spectral_step: int = 5
    n_components: int | None = None  # None = auto via eigengap
    min_speakers: int = 1
    max_speakers: int = 4
    reassign_threshold: float = 0.4
    # Minimum buffer size before first spectral run. Default 0 = auto
    # (max(min_speakers*2, 12) to satisfy _prune_affinity keep=10 constraint).
    # Set to 6–8 to reduce cold-start lag at the cost of less stable early clustering.
    min_buf: int = 0


class SpectralClusterer:
    """Windowed spectral clustering — no CentroidBank dependency.

    Every call:
    1. Normalise and buffer the embedding.
    2. Provisional assignment: nearest global mean (cosine), or new ID if buffer
       is too small for spectral / similarity is below threshold.
    3. Every ``spectral_step`` calls (buffer >= min_buf):
       - Build cosine affinity matrix.
       - Run SpectralClustering.
       - Map cluster labels → persistent global IDs (greedy cosine match).
       - Return final (non-provisional) assignment.
    """

    def __init__(self, config: SpectralConfig | None = None) -> None:
        self._cfg = config or SpectralConfig()
        self._buffer: deque[tuple[np.ndarray, int, int]] = deque(maxlen=self._cfg.window_size)
        self._call_count = 0
        self._next_global_id = 0
        self._label_to_global: dict[int, int] = {}
        # Persistent map: global_id -> representative mean embedding.
        # Never shrinks; speakers absent from a window remain matchable.
        self._global_means: dict[int, np.ndarray] = {}
        # Backfill queue: after each spectral run, stores (start_ms, end_ms, global_id)
        # for all buffer frames so the pipeline can correct prior provisional assignments.
        self._pending_backfills: list[tuple[int, int, int]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(self, embedding: np.ndarray, start_ms: int, end_ms: int) -> SpeakerWindow:
        emb = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        self._buffer.append((emb, start_ms, end_ms))
        self._call_count += 1

        # Run spectral every spectral_step calls once buffer is large enough.
        # Default auto: max(min_speakers*2, 12) ensures _prune_affinity keep=10 is valid (n >= 12).
        min_buf = (self._cfg.min_buf if self._cfg.min_buf > 0
                   else max(self._cfg.min_speakers * 2, 12))
        if self._call_count % self._cfg.spectral_step == 0 and len(self._buffer) >= min_buf:
            return self._spectral_reassign(emb, start_ms, end_ms)

        # Provisional: only pick from spectral-confirmed speakers, never create new IDs.
        # Returns _UNKNOWN (-1) during cold-start or when similarity is too low.
        provisional_id = self._nearest_confirmed(emb)
        return SpeakerWindow(speaker_id=provisional_id, start_ms=start_ms, end_ms=end_ms, is_provisional=True)

    def pop_backfills(self) -> list[tuple[int, int, int]]:
        """Return and clear pending backfill corrections since last spectral run.

        Each entry is (start_ms, end_ms, global_speaker_id) covering a buffer
        frame that now has a confirmed speaker after a spectral reassignment.
        Call this after assign() to retrieve any corrections to provisional frames.
        """
        result = self._pending_backfills
        self._pending_backfills = []
        return result

    def reset(self) -> None:
        self._buffer.clear()
        self._call_count = 0
        self._next_global_id = 0
        self._label_to_global.clear()
        self._global_means.clear()
        self._pending_backfills.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_confirmed(self, emb: np.ndarray) -> int:
        """Return the nearest spectral-confirmed global speaker (never creates IDs).

        Returns _UNKNOWN (-1) when:
        - No spectral run has completed yet (cold-start, _global_means is empty).
        - The best cosine similarity is below _MIN_PROVISIONAL_SIM (embedding is
          too dissimilar to every known speaker — likely a different, unconfirmed
          speaker still accumulating windows).

        The caller marks _UNKNOWN results as provisional so downstream code can
        withhold or later correct the assignment.
        """
        if not self._global_means:
            return _UNKNOWN
        best_gid, best_sim = _UNKNOWN, _MIN_PROVISIONAL_SIM
        for gid, g_mean in self._global_means.items():
            sim = _cosine_sim(emb, g_mean)
            if sim > best_sim:
                best_sim, best_gid = sim, gid
        return best_gid

    def _build_affinity(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Build affinity matrix following WeSpeaker's approach.

        Cosine similarity mapped to [0,1] via 0.5*(1+cos).  Pruning is applied
        only when n >= _PRUNE_MIN_N — below that threshold the prune creates a
        near-uniform graph that destroys the eigengap signal.
        """
        n = len(embeddings)
        cosine = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                # 0.5*(1+cos) maps cosine ∈ [-1,1] → [0,1], same as WeSpeaker
                sim = float(0.5 * (1.0 + _cosine_sim(embeddings[i], embeddings[j])))
                cosine[i, j] = cosine[j, i] = sim
        return _maybe_prune(cosine)

    def _detect_n_components(self, affinity: np.ndarray) -> int:
        if self._cfg.n_components is not None:
            return min(self._cfg.n_components, affinity.shape[0])
        return _eigengap_n_components(
            affinity,
            min_k=self._cfg.min_speakers,
            max_k=min(self._cfg.max_speakers, affinity.shape[0]),
        )

    def _spectral_reassign(self, emb: np.ndarray, start_ms: int, end_ms: int) -> SpeakerWindow:
        buf_embs = [e for e, _, _ in self._buffer]
        affinity = self._build_affinity(buf_embs)
        n_clusters = max(1, min(self._detect_n_components(affinity), len(buf_embs)))

        if n_clusters == 1:
            labels = np.zeros(len(buf_embs), dtype=int)
        else:
            try:
                labels = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="precomputed",
                    random_state=42,
                    n_init=5,
                ).fit_predict(affinity)
            except Exception:
                labels = np.zeros(len(buf_embs), dtype=int)

        # Only update global means when multi-speaker is detected.
        # A single-speaker window cannot represent absent speakers and would
        # corrupt their stored mean, causing ID gaps on the next multi-speaker run.
        if n_clusters > 1 or not self._global_means:
            self._update_label_map(buf_embs, labels, n_clusters)

        # Emit backfill corrections for all buffer frames so the pipeline can
        # retroactively correct provisional utterances from the cold-start period.
        for (_, f_start, f_end), lbl in zip(self._buffer, labels.tolist()):
            gid = self._label_to_global.get(int(lbl), 0)
            self._pending_backfills.append((f_start, f_end, gid))

        current_label = int(labels[-1])
        global_id = self._label_to_global.get(current_label, 0)
        return SpeakerWindow(speaker_id=global_id, start_ms=start_ms, end_ms=end_ms, is_provisional=False)

    def _update_label_map(
        self,
        embeddings: list[np.ndarray],
        labels: np.ndarray,
        n_clusters: int,
    ) -> None:
        """Map spectral cluster labels → persistent global speaker IDs.

        Labels are matched to existing globals by cosine similarity.  Labels with
        no close match get new IDs.  Globals not seen this window are kept in
        ``_global_means`` so they remain matchable in future runs (no ID gaps).
        """
        # Compute mean embedding per cluster label.
        cluster_means: dict[int, np.ndarray] = {}
        for label in range(n_clusters):
            idxs = np.where(labels == label)[0]
            if len(idxs) == 0:
                continue
            mean_emb = np.mean([embeddings[i] for i in idxs], axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-8:
                mean_emb = mean_emb / norm
            cluster_means[label] = mean_emb

        # Sort labels by first appearance time so speaker 0 = first to speak.
        label_first_ms: dict[int, int] = {}
        for (_, s_ms, _), lbl in zip(self._buffer, labels.tolist()):
            lbl = int(lbl)
            if lbl not in label_first_ms or s_ms < label_first_ms[lbl]:
                label_first_ms[lbl] = s_ms

        sorted_labels = sorted(
            [lbl for lbl in cluster_means if lbl in label_first_ms],
            key=lambda lbl: label_first_ms[lbl],
        )

        new_label_to_global: dict[int, int] = {}
        used_globals: set[int] = set()

        for label in sorted_labels:
            mean_emb = cluster_means[label]

            # Match to the nearest unmatched global above the similarity threshold.
            # If no match, allocate a new global ID (= genuinely new speaker).
            best_gid, best_sim = -1, self._cfg.reassign_threshold - 1e-6
            for gid, g_mean in self._global_means.items():
                if gid in used_globals:
                    continue
                sim = _cosine_sim(mean_emb, g_mean)
                if sim > best_sim:
                    best_sim, best_gid = sim, gid

            if best_gid == -1:
                best_gid = self._next_global_id
                self._next_global_id += 1

            new_label_to_global[label] = best_gid
            # Update global mean with EMA to track slow drift without discarding
            # long-term identity signal.  Direct replacement causes the mean to
            # wander across speaker turns, which breaks similarity matching.
            if best_gid in self._global_means:
                self._global_means[best_gid] = _ema_mean(self._global_means[best_gid], mean_emb)
            else:
                self._global_means[best_gid] = mean_emb
            used_globals.add(best_gid)

        self._label_to_global = new_label_to_global
