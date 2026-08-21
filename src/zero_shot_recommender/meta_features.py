"""Fixed-length dataset descriptors for the zero-shot recommender."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


META_FEATURE_NAMES = [
    # Original 19: names and ordering are a compatibility contract.
    "log_n_samples", "log_n_features", "log_feature_sample_ratio",
    "n_classes", "n_batches", "class_imbalance", "batch_imbalance",
    "missing_class_batch_fraction", "zero_fraction", "missing_fraction",
    "feature_mean_median", "feature_mean_iqr", "feature_std_median",
    "feature_std_iqr", "near_constant_fraction", "pc1_variance_ratio",
    "pc5_variance_ratio", "effective_rank_ratio", "batch_centroid_ratio",
    # Feature distributions (20).
    "feature_mean_mean", "feature_mean_std", "feature_mean_skew",
    "feature_mean_kurtosis", "feature_mean_p05", "feature_mean_p95",
    "feature_std_mean", "feature_std_std", "feature_std_skew",
    "feature_std_kurtosis", "feature_std_p05", "feature_std_p95",
    "feature_skew_median", "feature_skew_iqr", "feature_skew_p90",
    "feature_kurtosis_median", "feature_kurtosis_iqr", "feature_kurtosis_p90",
    "feature_zero_fraction_std", "feature_missing_fraction_std",
    # Robust shape/outliers (8).
    "feature_mad_median", "feature_mad_iqr", "feature_bowley_skew_median",
    "feature_tail_asymmetry_median", "feature_tail_weight_median",
    "feature_tail_weight_iqr", "feature_outlier_fraction_3mad_median",
    "feature_outlier_fraction_3mad_p95",
    # Sample heterogeneity (6).
    "sample_mean_std", "sample_mean_iqr", "sample_std_mean", "sample_std_std",
    "sample_zero_fraction_std", "sample_l2_norm_cv",
    # Correlation/redundancy (5).
    "absolute_correlation_median", "absolute_correlation_iqr",
    "absolute_correlation_p95", "fraction_abs_correlation_gt_0_5",
    "fraction_abs_correlation_gt_0_9",
    # Spectral geometry (5).
    "pc10_variance_ratio", "pcs_for_50pct_variance_ratio",
    "pcs_for_90pct_variance_ratio", "spectral_entropy", "stable_rank_ratio",
    # BERNN structure (3).
    "class_batch_normalized_mutual_information", "batch_knn_purity",
    "batch_covariance_heterogeneity",
]


def _imbalance(values: pd.Series) -> float:
    counts = values.astype(str).value_counts().to_numpy(dtype=float)
    return float(counts.max() / max(counts.min(), 1.0)) if len(counts) else 1.0


def _iqr(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def _safe_skew_kurtosis(values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    centered = values - np.mean(values, axis=axis, keepdims=True)
    variance = np.mean(np.square(centered), axis=axis)
    scale = np.sqrt(variance)
    skew = np.mean(np.power(centered, 3), axis=axis) / np.maximum(scale**3, 1e-12)
    kurtosis = np.mean(np.power(centered, 4), axis=axis) / np.maximum(scale**4, 1e-12) - 3.0
    constant = variance < 1e-12
    return np.where(constant, 0.0, skew), np.where(constant, 0.0, kurtosis)


def _batch_covariance_heterogeneity(values: np.ndarray, batches: pd.Series, seed: int) -> float:
    """Relative batch-covariance dispersion in a deterministic 32-D projection."""
    unique = sorted(batches.unique())
    if len(unique) < 2:
        return 0.0
    width = min(32, values.shape[1])
    if values.shape[1] > width:
        rng = np.random.default_rng(seed + 17)
        projection = rng.normal(size=(values.shape[1], width)) / math.sqrt(width)
        embedded = values @ projection
    else:
        embedded = values
    covariances = []
    for batch in unique:
        subset = embedded[(batches == batch).to_numpy()]
        covariance = np.cov(subset, rowvar=False) if len(subset) >= 2 else np.zeros((width, width))
        covariances.append(np.atleast_2d(covariance))
    covariances = np.stack(covariances)
    center = covariances.mean(axis=0)
    dispersion = np.mean(np.linalg.norm(covariances - center, axis=(1, 2)))
    return float(dispersion / max(np.linalg.norm(center), 1e-12))


def extract_meta_features(
    X: pd.DataFrame | np.ndarray,
    class_labels,
    batch_labels,
    *,
    max_features: int = 2048,
    max_samples: int = 1024,
    seed: int = 42,
) -> dict[str, float]:
    """Return the deterministic 66-descriptor dataset representation.

    Expensive geometry descriptors use a deterministic row/feature sample so
    large single-cell datasets do not require a full dense SVD or exact k-NN
    over every cell. Dataset size, class/batch structure, and missingness still
    use the complete dataset.
    """
    raw = np.asarray(X, dtype=float)
    y = pd.Series(class_labels).astype(str).reset_index(drop=True)
    batches = pd.Series(batch_labels).astype(str).reset_index(drop=True)
    if raw.ndim != 2 or len(raw) != len(y) or len(raw) != len(batches):
        raise ValueError("X, class_labels, and batch_labels must have matching rows")
    missing_mask = ~np.isfinite(raw)
    missing_fraction = float(missing_mask.mean())
    values = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(seed)
    if values.shape[1] > max_features:
        columns = np.sort(rng.choice(values.shape[1], size=max_features, replace=False))
        feature_sampled, feature_sampled_missing = values[:, columns], missing_mask[:, columns]
    else:
        feature_sampled, feature_sampled_missing = values, missing_mask
    if len(feature_sampled) > max_samples:
        rows = np.sort(rng.choice(len(feature_sampled), size=max_samples, replace=False))
        sampled = feature_sampled[rows]
        sampled_missing = feature_sampled_missing[rows]
        geometry_batches = batches.iloc[rows].reset_index(drop=True)
    else:
        sampled, sampled_missing = feature_sampled, feature_sampled_missing
        geometry_batches = batches

    means, stds = sampled.mean(axis=0), sampled.std(axis=0)
    combinations = pd.MultiIndex.from_product([y.unique(), batches.unique()])
    observed = pd.MultiIndex.from_frame(pd.DataFrame({"class": y, "batch": batches}).drop_duplicates())
    missing_combos = 1.0 - len(observed) / max(len(combinations), 1)

    centered = sampled - sampled.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False) if centered.size else np.array([])
    energy = singular**2
    ratios = energy / energy.sum() if energy.size and energy.sum() > 0 else np.zeros_like(energy)
    if ratios.size:
        effective_rank = np.exp(-np.sum(ratios * np.log(np.maximum(ratios, 1e-12))))
        rank_ratio = float(effective_rank / max(min(centered.shape), 1))
    else:
        rank_ratio = 0.0

    global_center = sampled.mean(axis=0)
    total_dispersion = float(np.mean(np.linalg.norm(sampled - global_center, axis=1)))
    centroids = np.vstack([
        sampled[(geometry_batches == batch).to_numpy()].mean(axis=0)
        for batch in sorted(geometry_batches.unique())
    ])
    centroid_dispersion = float(np.mean(np.linalg.norm(centroids - centroids.mean(axis=0), axis=1)))

    feature_skew, feature_kurtosis = _safe_skew_kurtosis(sampled, axis=0)
    mean_skew, mean_kurtosis = _safe_skew_kurtosis(means[None, :], axis=1)
    std_skew, std_kurtosis = _safe_skew_kurtosis(stds[None, :], axis=1)
    q05, q25, q50, q75, q95 = np.quantile(sampled, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    mad = np.median(np.abs(sampled - q50), axis=0)
    middle_width, full_width = np.maximum(q75 - q25, 1e-12), np.maximum(q95 - q05, 1e-12)
    bowley = (q75 + q25 - 2.0 * q50) / middle_width
    tail_asymmetry = ((q95 - q50) - (q50 - q05)) / full_width
    tail_weight = full_width / middle_width
    outlier_fraction = np.mean(np.abs(sampled - q50) > 3.0 * np.maximum(mad, 1e-12), axis=0)
    feature_zero_fractions = np.mean(sampled == 0, axis=0)
    feature_missing_fractions = sampled_missing.mean(axis=0)

    sample_means, sample_stds = sampled.mean(axis=1), sampled.std(axis=1)
    sample_zero_fractions = np.mean(sampled == 0, axis=1)
    sample_norms = np.linalg.norm(sampled, axis=1)

    # Correlations use at most 512 deterministically selected features.
    correlation_columns = min(512, sampled.shape[1])
    correlation_data = sampled[:, :correlation_columns]
    correlation_data = correlation_data[:, correlation_data.std(axis=0) > 1e-12]
    if correlation_data.shape[1] >= 2:
        correlation = np.abs(np.corrcoef(correlation_data, rowvar=False))
        absolute_correlations = correlation[np.triu_indices_from(correlation, k=1)]
    else:
        absolute_correlations = np.zeros(1)

    cumulative = np.cumsum(ratios)
    spectral_entropy = (
        float(-np.sum(ratios * np.log(np.maximum(ratios, 1e-12))) / np.log(len(ratios)))
        if len(ratios) > 1 else 0.0
    )
    stable_rank = float(energy.sum() / max(energy[0], 1e-12)) if energy.size else 0.0
    k = min(15, len(sampled) - 1)
    if k >= 1:
        indices = NearestNeighbors(n_neighbors=k + 1).fit(sampled).kneighbors(return_distance=False)[:, 1:]
        batch_array = geometry_batches.to_numpy()
        batch_knn_purity = float(np.mean(batch_array[indices] == batch_array[:, None]))
    else:
        batch_knn_purity = 0.0

    result = {
        "log_n_samples": math.log1p(values.shape[0]), "log_n_features": math.log1p(values.shape[1]),
        "log_feature_sample_ratio": math.log1p(values.shape[1] / max(values.shape[0], 1)),
        "n_classes": float(y.nunique()), "n_batches": float(batches.nunique()),
        "class_imbalance": _imbalance(y), "batch_imbalance": _imbalance(batches),
        "missing_class_batch_fraction": float(missing_combos), "zero_fraction": float(np.mean(sampled == 0)),
        "missing_fraction": missing_fraction, "feature_mean_median": float(np.median(means)),
        "feature_mean_iqr": _iqr(means), "feature_std_median": float(np.median(stds)),
        "feature_std_iqr": _iqr(stds), "near_constant_fraction": float(np.mean(stds < 1e-8)),
        "pc1_variance_ratio": float(ratios[0]) if len(ratios) else 0.0,
        "pc5_variance_ratio": float(ratios[:5].sum()), "effective_rank_ratio": rank_ratio,
        "batch_centroid_ratio": centroid_dispersion / max(total_dispersion, 1e-12),
        "feature_mean_mean": float(means.mean()), "feature_mean_std": float(means.std()),
        "feature_mean_skew": float(mean_skew[0]), "feature_mean_kurtosis": float(mean_kurtosis[0]),
        "feature_mean_p05": float(np.quantile(means, .05)), "feature_mean_p95": float(np.quantile(means, .95)),
        "feature_std_mean": float(stds.mean()), "feature_std_std": float(stds.std()),
        "feature_std_skew": float(std_skew[0]), "feature_std_kurtosis": float(std_kurtosis[0]),
        "feature_std_p05": float(np.quantile(stds, .05)), "feature_std_p95": float(np.quantile(stds, .95)),
        "feature_skew_median": float(np.median(feature_skew)), "feature_skew_iqr": _iqr(feature_skew),
        "feature_skew_p90": float(np.quantile(feature_skew, .9)),
        "feature_kurtosis_median": float(np.median(feature_kurtosis)), "feature_kurtosis_iqr": _iqr(feature_kurtosis),
        "feature_kurtosis_p90": float(np.quantile(feature_kurtosis, .9)),
        "feature_zero_fraction_std": float(feature_zero_fractions.std()),
        "feature_missing_fraction_std": float(feature_missing_fractions.std()),
        "feature_mad_median": float(np.median(mad)), "feature_mad_iqr": _iqr(mad),
        "feature_bowley_skew_median": float(np.median(bowley)),
        "feature_tail_asymmetry_median": float(np.median(tail_asymmetry)),
        "feature_tail_weight_median": float(np.median(tail_weight)), "feature_tail_weight_iqr": _iqr(tail_weight),
        "feature_outlier_fraction_3mad_median": float(np.median(outlier_fraction)),
        "feature_outlier_fraction_3mad_p95": float(np.quantile(outlier_fraction, .95)),
        "sample_mean_std": float(sample_means.std()), "sample_mean_iqr": _iqr(sample_means),
        "sample_std_mean": float(sample_stds.mean()), "sample_std_std": float(sample_stds.std()),
        "sample_zero_fraction_std": float(sample_zero_fractions.std()),
        "sample_l2_norm_cv": float(sample_norms.std() / max(abs(sample_norms.mean()), 1e-12)),
        "absolute_correlation_median": float(np.median(absolute_correlations)),
        "absolute_correlation_iqr": _iqr(absolute_correlations),
        "absolute_correlation_p95": float(np.quantile(absolute_correlations, .95)),
        "fraction_abs_correlation_gt_0_5": float(np.mean(absolute_correlations > .5)),
        "fraction_abs_correlation_gt_0_9": float(np.mean(absolute_correlations > .9)),
        "pc10_variance_ratio": float(ratios[:10].sum()),
        "pcs_for_50pct_variance_ratio": float((np.searchsorted(cumulative, .5) + 1) / max(len(ratios), 1)),
        "pcs_for_90pct_variance_ratio": float((np.searchsorted(cumulative, .9) + 1) / max(len(ratios), 1)),
        "spectral_entropy": spectral_entropy,
        "stable_rank_ratio": stable_rank / max(min(centered.shape), 1),
        "class_batch_normalized_mutual_information": float(normalized_mutual_info_score(y, batches)),
        "batch_knn_purity": batch_knn_purity,
        "batch_covariance_heterogeneity": _batch_covariance_heterogeneity(sampled, geometry_batches, seed),
    }
    return {name: float(np.nan_to_num(result[name], nan=0.0, posinf=0.0, neginf=0.0)) for name in META_FEATURE_NAMES}
