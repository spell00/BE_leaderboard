"""Label-free representation diagnostics and checkpoint selection."""

from __future__ import annotations

import math

import numpy as np
from sklearn.neighbors import NearestNeighbors


def effective_rank(values: np.ndarray) -> float:
    """Entropy effective rank of a sample-by-feature matrix."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2:
        return 0.0
    x = x - np.mean(x, axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False, full_matrices=False)
    energy = np.square(singular)
    total = float(np.sum(energy))
    if not np.isfinite(total) or total <= 0:
        return 0.0
    probabilities = energy[energy > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def neighborhood_preservation(raw: np.ndarray, latent: np.ndarray, k: int = 15) -> float:
    """Mean k-neighbour Jaccard overlap between raw and latent spaces."""
    raw = np.asarray(raw, dtype=float)
    latent = np.asarray(latent, dtype=float)
    if raw.shape[0] != latent.shape[0] or raw.shape[0] < 3:
        return 0.0
    k = max(1, min(int(k), raw.shape[0] - 1))
    raw_neighbors = NearestNeighbors(n_neighbors=k + 1).fit(raw).kneighbors(return_distance=False)[:, 1:]
    latent_neighbors = NearestNeighbors(n_neighbors=k + 1).fit(latent).kneighbors(return_distance=False)[:, 1:]
    overlaps = []
    for left, right in zip(raw_neighbors, latent_neighbors):
        a, b = set(left.tolist()), set(right.tolist())
        overlaps.append(len(a & b) / max(len(a | b), 1))
    return float(np.mean(overlaps))


def batch_mixing_entropy(latent: np.ndarray, batches: np.ndarray, k: int = 15) -> float:
    """Local batch entropy normalized by the entropy of the global batch mix."""
    latent = np.asarray(latent, dtype=float)
    batches = np.asarray(batches).astype(str)
    if latent.shape[0] != len(batches) or latent.shape[0] < 3:
        return 0.0
    _, global_counts = np.unique(batches, return_counts=True)
    if len(global_counts) < 2:
        return 1.0
    global_prob = global_counts / global_counts.sum()
    global_entropy = float(-np.sum(global_prob * np.log(global_prob)))
    k = max(1, min(int(k), latent.shape[0] - 1))
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(latent).kneighbors(return_distance=False)[:, 1:]
    local = []
    for indices in neighbors:
        _, counts = np.unique(batches[indices], return_counts=True)
        probabilities = counts / counts.sum()
        local.append(float(-np.sum(probabilities * np.log(probabilities))) / global_entropy)
    return float(np.clip(np.mean(local), 0.0, 1.0))


def checkpoint_metrics(
    raw: np.ndarray,
    latent: np.ndarray,
    batches: np.ndarray,
    masked_reconstruction_mse: float,
    *,
    k: int = 15,
) -> dict[str, float]:
    """Compute the complete label-free checkpoint score."""
    raw = np.asarray(raw, dtype=float)
    latent = np.asarray(latent, dtype=float)
    variance = float(np.var(raw))
    reconstruction_quality = 1.0 / (1.0 + max(float(masked_reconstruction_mse), 0.0) / max(variance, 1e-12))
    raw_rank = effective_rank(raw)
    latent_rank = effective_rank(latent)
    rank_retention = float(np.clip(latent_rank / max(raw_rank, 1.0), 0.0, 1.0))
    neighborhood = neighborhood_preservation(raw, latent, k=k)
    mixing = batch_mixing_entropy(latent, batches, k=k)
    collapsed = bool(rank_retention < 0.05 or neighborhood < 0.02 or not np.isfinite(latent).all())
    score = (
        0.30 * reconstruction_quality
        + 0.30 * neighborhood
        + 0.20 * rank_retention
        + 0.20 * mixing
    )
    if collapsed:
        score = -1.0
    return {
        "unsupervised_score": float(score),
        "masked_reconstruction_quality": float(reconstruction_quality),
        "neighborhood_preservation": float(neighborhood),
        "rank_retention": float(rank_retention),
        "batch_mixing_entropy": float(mixing),
        "collapsed": float(collapsed),
    }


class UnsupervisedMonitor:
    """Reusable monitor that caches raw-space quantities across epochs."""

    def __init__(self, raw: np.ndarray, batches: np.ndarray, *, k: int = 15):
        self.raw = np.asarray(raw, dtype=float)
        self.batches = np.asarray(batches).astype(str)
        self.k = max(1, min(int(k), len(self.raw) - 1))
        self.raw_rank = effective_rank(self.raw)
        self.variance = float(np.var(self.raw))
        self.raw_neighbors = NearestNeighbors(n_neighbors=self.k + 1).fit(self.raw).kneighbors(return_distance=False)[:, 1:]
        _, counts = np.unique(self.batches, return_counts=True)
        probabilities = counts / counts.sum()
        self.global_batch_entropy = float(-np.sum(probabilities * np.log(probabilities))) if len(counts) > 1 else 0.0

    def score(self, latent: np.ndarray, masked_reconstruction_mse: float) -> dict[str, float]:
        latent = np.asarray(latent, dtype=float)
        reconstruction_quality = 1.0 / (
            1.0 + max(float(masked_reconstruction_mse), 0.0) / max(self.variance, 1e-12)
        )
        latent_rank = effective_rank(latent)
        rank_retention = float(np.clip(latent_rank / max(self.raw_rank, 1.0), 0.0, 1.0))
        latent_neighbors = NearestNeighbors(n_neighbors=self.k + 1).fit(latent).kneighbors(return_distance=False)[:, 1:]
        overlaps = []
        for left, right in zip(self.raw_neighbors, latent_neighbors):
            a, b = set(left.tolist()), set(right.tolist())
            overlaps.append(len(a & b) / max(len(a | b), 1))
        neighborhood = float(np.mean(overlaps))
        if self.global_batch_entropy <= 0:
            mixing = 1.0
        else:
            entropies = []
            for indices in latent_neighbors:
                _, counts = np.unique(self.batches[indices], return_counts=True)
                probabilities = counts / counts.sum()
                entropies.append(float(-np.sum(probabilities * np.log(probabilities))) / self.global_batch_entropy)
            mixing = float(np.clip(np.mean(entropies), 0.0, 1.0))
        collapsed = bool(rank_retention < 0.05 or neighborhood < 0.02 or not np.isfinite(latent).all())
        score = 0.30 * reconstruction_quality + 0.30 * neighborhood + 0.20 * rank_retention + 0.20 * mixing
        if collapsed:
            score = -1.0
        return {
            "unsupervised_score": float(score),
            "masked_reconstruction_quality": float(reconstruction_quality),
            "neighborhood_preservation": float(neighborhood),
            "rank_retention": float(rank_retention),
            "batch_mixing_entropy": float(mixing),
            "collapsed": float(collapsed),
        }


def select_checkpoint(rows: list[dict[str, float]]) -> dict[str, float]:
    """Select solely from label-free fields, ignoring any oracle diagnostics."""
    if not rows:
        raise ValueError("At least one checkpoint row is required")
    return max(rows, key=lambda row: (float(row["unsupervised_score"]), -int(row.get("epoch", 0))))
