#!/usr/bin/env python3
"""Audit GEO batch-effect benchmark CSVs with PCA, UMAP, and batch metrics.

This script is meant to run after ``prepare_geo_batch_datasets.R`` has produced
BERNN-ready CSVs with ``label,batch,sample_id,<features...>`` columns.

It generates:

- a dataset-overview table with counts and batch/label-separation metrics,
- per-dataset PCA and UMAP projections colored by batch and label,
- CSV / Markdown / JSON summaries under ``--output-dir``,
- optional Weights & Biases logs for the same metrics and figures.

The key audit question is simple: do the reconstructed datasets still show a
substantial batch signal while preserving biological labels? The script answers
that using complementary descriptive and predictive metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, silhouette_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
import warnings

try:
    import plotly.express as px
except Exception as exc:  # pragma: no cover - hard fail in normal use
    raise RuntimeError(
        "plotly is required for the GEO audit plots; install plotly and retry"
    ) from exc

try:
    import umap
except Exception:
    umap = None

try:  # optional, but useful for the baseline sweep
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

ROOT = Path(__file__).resolve().parent.parent
META_COLUMNS = {
    "label", "labels", "group",
    "batch", "batches",
    "sample_id", "sample", "samples",
    "name", "names",
}


@dataclass
class DatasetAudit:
    dataset: str
    source_csv: str
    n_samples: int
    n_features: int
    n_labels: int
    n_batches: int
    missing_feature_fraction: float
    label_entropy_norm: float
    batch_entropy_norm: float
    label_imbalance_ratio: float
    batch_imbalance_ratio: float
    label_bacc_cv: float
    label_mcc_cv: float
    batch_bacc_cv: float
    batch_mcc_cv: float
    label_silhouette: float
    batch_silhouette: float
    label_nn_purity: float
    batch_nn_purity: float
    batch_samples_mean: float
    batch_samples_std: float
    batch_samples_min: float
    batch_samples_max: float
    pca_var_2d: float
    umap_available: bool

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "source_csv": self.source_csv,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_labels": self.n_labels,
            "n_batches": self.n_batches,
            "missing_feature_fraction": self.missing_feature_fraction,
            "label_entropy_norm": self.label_entropy_norm,
            "batch_entropy_norm": self.batch_entropy_norm,
            "label_imbalance_ratio": self.label_imbalance_ratio,
            "batch_imbalance_ratio": self.batch_imbalance_ratio,
            "label_bacc_cv": self.label_bacc_cv,
            "label_mcc_cv": self.label_mcc_cv,
            "batch_bacc_cv": self.batch_bacc_cv,
            "batch_mcc_cv": self.batch_mcc_cv,
            "label_silhouette": self.label_silhouette,
            "batch_silhouette": self.batch_silhouette,
            "label_nn_purity": self.label_nn_purity,
            "batch_nn_purity": self.batch_nn_purity,
            "batch_samples_mean": self.batch_samples_mean,
            "batch_samples_std": self.batch_samples_std,
            "batch_samples_min": self.batch_samples_min,
            "batch_samples_max": self.batch_samples_max,
            "pca_var_2d": self.pca_var_2d,
            "umap_available": self.umap_available,
        }


@dataclass
class BaselineResult:
    dataset: str
    task: str
    preprocessor: str
    model: str
    cv_strategy: str
    n_splits: int
    bacc_cv: float
    mcc_cv: float
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "preprocessor": self.preprocessor,
            "model": self.model,
            "cv_strategy": self.cv_strategy,
            "n_splits": self.n_splits,
            "bacc_cv": self.bacc_cv,
            "mcc_cv": self.mcc_cv,
            "note": self.note,
        }


def _bool_arg(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "t"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "geo_batch",
        help="Directory containing prepared GEO CSVs (combined BERNN-ready files).",
    )
    parser.add_argument(
        "--csv",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit combined GEO CSVs to audit. Overrides --input-dir discovery when provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "geo_batch_audit",
        help="Directory where tables, embeddings, and HTML plots will be written.",
    )
    parser.add_argument(
        "--wandb-project",
        default="BE_leaderboard_geo_audit",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA/UMAP and classifier splits.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=15,
        help="Number of neighbors used for local batch / label purity.",
    )
    parser.add_argument(
        "--max-embedding-samples",
        type=int,
        default=None,
        help="Optional cap on rows used for UMAP. PCA always uses the full dataset.",
    )
    parser.add_argument(
        "--copy-html",
        action="store_true",
        help="Also write a small HTML index page linking all per-dataset plots.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip the classifier/scaler baseline sweep.",
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run the expensive baseline sweep in this script (normally use benchmark_geo_baselines.py).",
    )
    return parser.parse_args(argv)


def _coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def _entropy_norm(values: Iterable[str]) -> float:
    counts = pd.Series(list(values)).value_counts(dropna=False)
    total = float(counts.sum())
    if total == 0 or len(counts) <= 1:
        return 0.0
    probs = counts.to_numpy(dtype=float) / total
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(entropy / math.log(len(counts)))


def _imbalance_ratio(values: Iterable[str]) -> float:
    counts = pd.Series(list(values)).value_counts(dropna=False)
    if counts.empty:
        return float("nan")
    minimum = float(counts.min())
    if minimum <= 0:
        return float("inf")
    return float(counts.max() / minimum)


def _choose_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None


def _fit_batch_scaler(features: np.ndarray, batches: np.ndarray, scaler_kind: str) -> np.ndarray:
    transformed = np.empty_like(features, dtype=float)
    for batch in pd.unique(pd.Series(batches)):
        idx = batches == batch
        subset = features[idx]
        if subset.shape[0] == 0:
            continue
        if scaler_kind == "standard":
            scaler = StandardScaler()
        elif scaler_kind == "robust":
            scaler = RobustScaler(with_centering=True, with_scaling=True)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown batch scaler kind: {scaler_kind}")
        try:
            transformed[idx] = scaler.fit_transform(subset)
        except Exception:
            # For degenerate singleton batches, fall back to a safe zero-centered block.
            transformed[idx] = subset - np.nanmean(subset, axis=0, keepdims=True)
    return transformed


def _apply_preprocessor(features: np.ndarray, batches: np.ndarray, preprocessor: str) -> np.ndarray:
    if preprocessor == "raw":
        return features
    if preprocessor == "standard":
        return StandardScaler().fit_transform(features)
    if preprocessor == "robust":
        return RobustScaler(with_centering=True, with_scaling=True).fit_transform(features)
    if preprocessor == "robust_per_batch":
        return _fit_batch_scaler(features, batches, "robust")
    if preprocessor == "standard_per_batch":
        return _fit_batch_scaler(features, batches, "standard")
    raise ValueError(f"Unknown preprocessor: {preprocessor}")


def _build_cv(encoded: np.ndarray, groups: np.ndarray, seed: int, max_splits: int = 5):
    """Build leakage-safe CV splits, keeping every group in one fold."""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return None, "insufficient_groups", 0
    n_splits = min(max_splits, n_groups)
    if n_splits < 2:
        return None, "insufficient_groups", 0
    # Stratification is useful for biological labels, while groups enforce
    # the benchmark rule that one GEO study cannot cross a split boundary.
    try:
        return (
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            "stratified_group_kfold",
            n_splits,
        )
    except Exception:
        return GroupKFold(n_splits=n_splits), "group_kfold", n_splits


def _group_cv_for_task(encoded: np.ndarray, groups: np.ndarray, seed: int, task: str):
    """Build grouped CV for the biological prediction task only.

    ``label`` is the supervised target. GEO study/batch is used exclusively
    as the grouping variable, keeping every study in one fold. Batch is not a
    prediction target because predicting held-out study IDs is invalid here.
    """
    if task != "label":
        return None, "not_a_prediction_target", 0
    return _build_cv(encoded, groups, seed)


def _legacy_build_cv(encoded: np.ndarray, seed: int, max_splits: int = 5):
    """Retained only for compatibility with callers outside this script."""
    counts = np.bincount(encoded)
    if len(counts) < 2:
        return None, "insufficient_classes", 0
    min_count = int(counts.min())
    if min_count >= 2:
        n_splits = min(max_splits, min_count)
        if n_splits >= 2:
            return (
                StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
                "stratified_kfold",
                n_splits,
            )
    n_splits = min(max_splits, len(encoded))
    if n_splits >= 2:
        return (
            KFold(n_splits=n_splits, shuffle=True, random_state=seed),
            "kfold",
            n_splits,
        )
    return None, "insufficient_samples", 0


def _make_classifier(model_name: str, n_classes: int, seed: int):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, solver="lbfgs")
    if model_name == "linear_svc":
        return LinearSVC(max_iter=5000)
    if model_name == "svc_rbf":
        return SVC(kernel="rbf", gamma="scale")
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        params = {
            "objective": objective,
            "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "random_state": seed,
            "n_jobs": 1,
        }
        if n_classes > 2:
            params["num_class"] = n_classes
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model: {model_name}")


def load_combined_csv(path: Path) -> dict:
    frame = pd.read_csv(path)
    label_col = _choose_column(frame, ["label", "labels", "group"])
    batch_col = _choose_column(frame, ["batch", "batches"])
    sample_col = _choose_column(frame, ["sample_id", "sample", "samples", "name", "names"])
    if label_col is None:
        raise ValueError(f"{path} does not contain a label column (label/labels/group)")
    if batch_col is None:
        raise ValueError(f"{path} does not contain a batch column (batch/batches)")

    feature_cols = [column for column in frame.columns if column not in META_COLUMNS]
    features = _coerce_numeric_frame(frame[feature_cols]).fillna(0.0)
    if features.shape[1] == 0:
        raise ValueError(f"{path} does not contain any numeric feature columns")

    sample_ids = (
        frame[sample_col].astype(str).to_numpy()
        if sample_col is not None
        else np.array([f"sample_{i}" for i in range(len(frame))], dtype=str)
    )
    return {
        "dataset": path.stem,
        "source_csv": str(path),
        "frame": frame,
        "features": features,
        "label": frame[label_col].astype(str).to_numpy(),
        "batch": frame[batch_col].astype(str).to_numpy(),
        "sample_id": sample_ids,
    }


def _safe_pca(features: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    n_samples, n_features = features.shape
    max_components = min(n_components, n_samples - 1, n_features)
    if max_components < 2:
        raise ValueError("Need at least two samples and two features for PCA")
    model = PCA(n_components=max_components, random_state=seed)
    transformed = model.fit_transform(features)
    return transformed, float(model.explained_variance_ratio_[:2].sum())


def _safe_umap(features: np.ndarray, seed: int, max_samples: int | None = None) -> np.ndarray | None:
    if umap is None:
        return None
    subset = features
    if max_samples is not None and len(features) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(features), size=max_samples, replace=False))
        subset = features[indices]
    if len(subset) < 3:
        return None
    n_neighbors = min(15, max(2, int(round(np.sqrt(len(subset))))))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.25,
        metric="euclidean",
        random_state=seed,
        init="spectral",
    )
    try:
        return reducer.fit_transform(subset)
    except (TypeError, ValueError, RuntimeError) as exc:
        # UMAP is optional visualization; dependency incompatibilities
        # must not prevent the audit metrics and W&B export from completing.
        warnings.warn(f"UMAP unavailable; continuing without UMAP: {exc}")
        return None


def _classifier_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    task: str,
) -> tuple[float, float, str, int]:
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    cv, cv_strategy, n_splits = _group_cv_for_task(encoded, groups, seed, task)
    if cv is None:
        return float("nan"), float("nan"), cv_strategy, n_splits
    classifier = LogisticRegression(max_iter=2000, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        predicted = cross_val_predict(classifier, embedding, encoded, cv=cv, groups=groups)
    return (
        float(balanced_accuracy_score(encoded, predicted)),
        float(matthews_corrcoef(encoded, predicted)),
        cv_strategy,
        n_splits,
    )


def _neighbor_purity(embedding: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(embedding) < 3:
        return float("nan")
    n_neighbors = min(len(embedding), k + 1)
    if n_neighbors < 2:
        return float("nan")
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    neighbors.fit(embedding)
    indices = neighbors.kneighbors(return_distance=False)
    same_fraction = []
    for row_index, row in enumerate(indices):
        row = row[row != row_index]
        if len(row) == 0:
            continue
        same_fraction.append(float(np.mean(labels[row] == labels[row_index])))
    return float(np.mean(same_fraction)) if same_fraction else float("nan")


def _silhouette(embedding: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return float("nan")
    try:
        return float(silhouette_score(embedding, labels, metric="euclidean"))
    except Exception:
        return float("nan")


def _plot_embedding(embedding: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray, title: str):
    plot_frame = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "label": labels,
            "sample_id": sample_ids,
        }
    )
    return px.scatter(
        plot_frame,
        x="x",
        y="y",
        color="label",
        hover_data={"sample_id": True, "x": False, "y": False, "label": True},
        title=title,
        render_mode="webgl",
        opacity=0.8,
    )


def _batch_counts_table(batches: np.ndarray) -> pd.DataFrame:
    counts = (
        pd.Series(batches, dtype=str)
        .value_counts(dropna=False)
        .rename_axis("batch")
        .reset_index(name="n_samples")
        .sort_values(["n_samples", "batch"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total = float(counts["n_samples"].sum())
    counts["fraction"] = counts["n_samples"] / total if total > 0 else 0.0
    return counts


def audit_dataset(dataset: dict, seed: int, knn_neighbors: int, max_embedding_samples: int | None, output_dir: Path):
    features = dataset["features"].to_numpy(dtype=float, copy=True)
    scaled = StandardScaler().fit_transform(features)
    pca_full, pca_var_2d = _safe_pca(scaled, n_components=20, seed=seed)
    pca2d = pca_full[:, :2]

    if max_embedding_samples is not None and len(scaled) > max_embedding_samples:
        rng = np.random.default_rng(seed)
        selection = np.sort(rng.choice(len(scaled), size=max_embedding_samples, replace=False))
        umap_input = pca_full[selection] if pca_full.shape[1] >= 5 else scaled[selection]
        umap_sample_ids = dataset["sample_id"][selection]
        umap_labels = {
            "batch": dataset["batch"][selection],
            "label": dataset["label"][selection],
        }
    else:
        umap_input = pca_full if pca_full.shape[1] >= 5 else scaled
        umap_sample_ids = dataset["sample_id"]
        umap_labels = {
            "batch": dataset["batch"],
            "label": dataset["label"],
        }

    umap2d = _safe_umap(umap_input, seed=seed, max_samples=None)

    label_bacc, label_mcc, label_cv_strategy, label_n_splits = _classifier_metrics(
        pca_full, dataset["label"], dataset["batch"], seed=seed, task="label"
    )
    # Batch/study is a grouping variable, not a supervised target. Its effect
    # is assessed by PCA, silhouettes, and nearest-neighbor purity below.
    batch_bacc = float("nan")
    batch_mcc = float("nan")
    batch_cv_strategy = "not_a_prediction_target"
    batch_n_splits = 0

    label_silhouette = _silhouette(pca2d, dataset["label"])
    batch_silhouette = _silhouette(pca2d, dataset["batch"])
    label_nn_purity = _neighbor_purity(pca_full, dataset["label"], k=knn_neighbors)
    batch_nn_purity = _neighbor_purity(pca_full, dataset["batch"], k=knn_neighbors)
    batch_counts = _batch_counts_table(dataset["batch"])
    batch_samples_mean = float(batch_counts["n_samples"].mean()) if len(batch_counts) else float("nan")
    batch_samples_std = float(batch_counts["n_samples"].std(ddof=0)) if len(batch_counts) else float("nan")
    batch_samples_min = float(batch_counts["n_samples"].min()) if len(batch_counts) else float("nan")
    batch_samples_max = float(batch_counts["n_samples"].max()) if len(batch_counts) else float("nan")

    audit = DatasetAudit(
        dataset=dataset["dataset"],
        source_csv=dataset["source_csv"],
        n_samples=int(len(dataset["label"])),
        n_features=int(features.shape[1]),
        n_labels=int(pd.Series(dataset["label"]).nunique()),
        n_batches=int(pd.Series(dataset["batch"]).nunique()),
        missing_feature_fraction=float(np.isnan(features).mean()),
        label_entropy_norm=_entropy_norm(dataset["label"]),
        batch_entropy_norm=_entropy_norm(dataset["batch"]),
        label_imbalance_ratio=_imbalance_ratio(dataset["label"]),
        batch_imbalance_ratio=_imbalance_ratio(dataset["batch"]),
        label_bacc_cv=label_bacc,
        label_mcc_cv=label_mcc,
        batch_bacc_cv=batch_bacc,
        batch_mcc_cv=batch_mcc,
        label_silhouette=label_silhouette,
        batch_silhouette=batch_silhouette,
        label_nn_purity=label_nn_purity,
        batch_nn_purity=batch_nn_purity,
        batch_samples_mean=batch_samples_mean,
        batch_samples_std=batch_samples_std,
        batch_samples_min=batch_samples_min,
        batch_samples_max=batch_samples_max,
        pca_var_2d=pca_var_2d,
        umap_available=umap2d is not None,
    )

    dataset_dir = output_dir / dataset["dataset"]
    dataset_dir.mkdir(parents=True, exist_ok=True)

    embeddings = pd.DataFrame(
        {
            "sample_id": dataset["sample_id"],
            "label": dataset["label"],
            "batch": dataset["batch"],
            "pca_1": pca2d[:, 0],
            "pca_2": pca2d[:, 1],
        }
    )
    embeddings.to_csv(dataset_dir / "pca_embeddings.csv", index=False)

    figures = {
        "pca_batch": _plot_embedding(pca2d, dataset["batch"], dataset["sample_id"], f"{dataset['dataset']} - PCA colored by batch"),
        "pca_label": _plot_embedding(pca2d, dataset["label"], dataset["sample_id"], f"{dataset['dataset']} - PCA colored by label"),
    }
    if umap2d is not None:
        umap_embeddings = pd.DataFrame(
            {
                "sample_id": umap_sample_ids,
                "label": umap_labels["label"],
                "batch": umap_labels["batch"],
                "umap_1": umap2d[:, 0],
                "umap_2": umap2d[:, 1],
            }
        )
        umap_embeddings.to_csv(dataset_dir / "umap_embeddings.csv", index=False)
        figures["umap_batch"] = _plot_embedding(umap2d, umap_labels["batch"], umap_sample_ids, f"{dataset['dataset']} - UMAP colored by batch")
        figures["umap_label"] = _plot_embedding(umap2d, umap_labels["label"], umap_sample_ids, f"{dataset['dataset']} - UMAP colored by label")
    else:
        umap_embeddings = None

    for name, figure in figures.items():
        figure.write_html(dataset_dir / f"{name}.html", include_plotlyjs="cdn")

    (dataset_dir / "metrics.json").write_text(json.dumps(audit.to_dict(), indent=2) + "\n")
    write_batch_counts(dataset_dir, batch_counts)
    audit_meta = {
        "label_cv_strategy": label_cv_strategy,
        "label_n_splits": label_n_splits,
        "batch_cv_strategy": batch_cv_strategy,
        "batch_n_splits": batch_n_splits,
    }
    return audit, figures, audit_meta, batch_counts


def evaluate_baselines(dataset: dict, seed: int) -> list[BaselineResult]:
    features = dataset["features"].to_numpy(dtype=float, copy=True)
    batches = dataset["batch"]
    baseline_preprocessors = [
        "raw",
        "standard",
        "robust",
        "standard_per_batch",
        "robust_per_batch",
    ]
    model_names = ["logistic_regression", "linear_svc", "svc_rbf"]
    if XGBClassifier is not None:
        model_names.append("xgboost")

    results: list[BaselineResult] = []
    for preprocessor in baseline_preprocessors:
        print(f"[baseline] {dataset['dataset']}: preprocessing={preprocessor}", flush=True)
        transformed = _apply_preprocessor(features, batches, preprocessor)
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            representation, _ = _safe_pca(transformed, n_components=50, seed=seed)
        except Exception as exc:
            results.append(
                BaselineResult(
                    dataset=dataset["dataset"],
                    task="all",
                    preprocessor=preprocessor,
                    model="pca",
                    cv_strategy="failed",
                    n_splits=0,
                    bacc_cv=float("nan"),
                    mcc_cv=float("nan"),
                    note=f"PCA failed: {type(exc).__name__}: {exc}",
                )
            )
            continue

        # Predict biological label only; batches are supplied as CV groups.
        for task_name, task_labels in (("label", dataset["label"]),):
            print(f"[baseline] {dataset['dataset']}: task={task_name}, preprocessor={preprocessor}", flush=True)
            encoded = LabelEncoder().fit_transform(task_labels)
            cv, cv_strategy, n_splits = _group_cv_for_task(
                encoded, batches, seed, task_name
            )
            if cv is None:
                results.append(
                    BaselineResult(
                        dataset=dataset["dataset"],
                        task=task_name,
                        preprocessor=preprocessor,
                        model="all",
                        cv_strategy=cv_strategy,
                        n_splits=n_splits,
                        bacc_cv=float("nan"),
                        mcc_cv=float("nan"),
                        note="insufficient samples/classes",
                    )
                )
                continue
            n_classes = len(np.unique(encoded))
            for model_name in model_names:
                print(f"[baseline] {dataset['dataset']}: {task_name}/{preprocessor}/{model_name}", flush=True)
                try:
                    classifier = _make_classifier(model_name, n_classes=n_classes, seed=seed)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        predicted = cross_val_predict(
                            classifier, representation, encoded, cv=cv, groups=batches
                        )
                    results.append(
                        BaselineResult(
                            dataset=dataset["dataset"],
                            task=task_name,
                            preprocessor=preprocessor,
                            model=model_name,
                            cv_strategy=cv_strategy,
                            n_splits=n_splits,
                            bacc_cv=float(balanced_accuracy_score(encoded, predicted)),
                            mcc_cv=float(matthews_corrcoef(encoded, predicted)),
                        )
                    )
                except Exception as exc:
                    results.append(
                        BaselineResult(
                            dataset=dataset["dataset"],
                            task=task_name,
                            preprocessor=preprocessor,
                            model=model_name,
                            cv_strategy=cv_strategy,
                            n_splits=n_splits,
                            bacc_cv=float("nan"),
                            mcc_cv=float("nan"),
                            note=f"{type(exc).__name__}: {exc}",
                        )
                    )
    return results


def write_batch_counts(dataset_dir: Path, batch_counts: pd.DataFrame) -> None:
    batch_counts.to_csv(dataset_dir / "batch_counts.csv", index=False)
    (dataset_dir / "batch_counts.json").write_text(batch_counts.to_json(orient="records", indent=2) + "\n")
    batch_counts.to_markdown(dataset_dir / "batch_counts.md", index=False)


def discover_csvs(args) -> list[Path]:
    if args.csv:
        csvs = [path.resolve() for path in args.csv]
    elif args.input_dir is not None:
        csvs = []
        for path in sorted(args.input_dir.glob("*.csv")):
            if path.name.endswith("_expression.csv") or path.name.endswith("_metadata.csv"):
                continue
            csvs.append(path.resolve())
    else:
        raise ValueError("Provide either --csv or --input-dir")

    seen = set()
    resolved = []
    for path in csvs:
        if path in seen:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        resolved.append(path)
        seen.add(path)
    if not resolved:
        raise ValueError("No dataset CSVs found")
    return resolved


def _as_wandb_scalar(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        return value
    return value


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csvs = discover_csvs(args)
    audits: list[DatasetAudit] = []
    figures_by_dataset: dict[str, dict[str, object]] = {}
    baseline_rows: list[BaselineResult] = []
    batch_counts_by_dataset: dict[str, pd.DataFrame] = {}
    for path in csvs:
        dataset = load_combined_csv(path)
        audit, figures, audit_meta, batch_counts = audit_dataset(
            dataset=dataset,
            seed=args.seed,
            knn_neighbors=args.knn_neighbors,
            max_embedding_samples=args.max_embedding_samples,
            output_dir=output_dir,
        )
        audits.append(audit)
        figures_by_dataset[audit.dataset] = figures
        batch_counts_by_dataset[audit.dataset] = batch_counts
        if args.run_baselines and not args.skip_baselines:
            baseline_rows.extend(evaluate_baselines(dataset, seed=args.seed))

    overview = pd.DataFrame([audit.to_dict() for audit in audits]).sort_values("dataset")
    overview.to_csv(output_dir / "dataset_overview.csv", index=False)
    overview.to_json(output_dir / "dataset_overview.json", orient="records", indent=2)
    overview.to_markdown(output_dir / "dataset_overview.md", index=False)

    if baseline_rows:
        baseline_overview = pd.DataFrame([row.to_dict() for row in baseline_rows]).sort_values(
            ["dataset", "task", "preprocessor", "model"]
        )
        baseline_overview.to_csv(output_dir / "baseline_overview.csv", index=False)
        baseline_overview.to_json(output_dir / "baseline_overview.json", orient="records", indent=2)
        baseline_overview.to_markdown(output_dir / "baseline_overview.md", index=False)
    else:
        baseline_overview = pd.DataFrame()

    if args.copy_html:
        index_lines = [
            "<html><head><meta charset='utf-8'><title>GEO batch audit</title></head><body>",
            "<h1>GEO batch audit</h1>",
            "<p>Per-dataset PCA and UMAP projections plus batch/label metrics.</p>",
            "<ul>",
        ]
        for audit in audits:
            dataset_dir = output_dir / audit.dataset
            index_lines.append(f"<li><strong>{audit.dataset}</strong><ul>")
            for name in ("pca_batch", "pca_label", "umap_batch", "umap_label"):
                html_path = dataset_dir / f"{name}.html"
                if html_path.exists():
                    rel = html_path.relative_to(output_dir).as_posix()
                    index_lines.append(f"<li><a href='{rel}'>{name}</a></li>")
            index_lines.append("</ul></li>")
        index_lines.extend(["</ul></body></html>"])
        (output_dir / "index.html").write_text("\n".join(index_lines) + "\n")

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            run_config = {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            }
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=run_config,
                reinit=True,
            )
            wandb.define_metric("dataset_index")
            wandb.define_metric("datasets/*", step_metric="dataset_index")
            wandb.define_metric("overview/*")
            wandb.define_metric("baselines/*")

            table = wandb.Table(dataframe=overview)
            wandb_run.log({"overview/table": table}, commit=False)
            for index, audit in enumerate(audits):
                prefix = f"datasets/{audit.dataset}"
                payload = {
                    f"{prefix}/dataset_index": index,
                    f"{prefix}/n_samples": audit.n_samples,
                    f"{prefix}/n_features": audit.n_features,
                    f"{prefix}/n_labels": audit.n_labels,
                    f"{prefix}/n_batches": audit.n_batches,
                    f"{prefix}/missing_feature_fraction": audit.missing_feature_fraction,
                    f"{prefix}/label_entropy_norm": audit.label_entropy_norm,
                    f"{prefix}/batch_entropy_norm": audit.batch_entropy_norm,
                    f"{prefix}/label_imbalance_ratio": audit.label_imbalance_ratio,
                    f"{prefix}/batch_imbalance_ratio": audit.batch_imbalance_ratio,
                    f"{prefix}/label_bacc_cv": audit.label_bacc_cv,
                    f"{prefix}/label_mcc_cv": audit.label_mcc_cv,
                    f"{prefix}/batch_bacc_cv": audit.batch_bacc_cv,
                    f"{prefix}/batch_mcc_cv": audit.batch_mcc_cv,
                    f"{prefix}/label_silhouette": audit.label_silhouette,
                    f"{prefix}/batch_silhouette": audit.batch_silhouette,
                    f"{prefix}/label_nn_purity": audit.label_nn_purity,
                    f"{prefix}/batch_nn_purity": audit.batch_nn_purity,
                    f"{prefix}/batch_samples_mean": audit.batch_samples_mean,
                    f"{prefix}/batch_samples_std": audit.batch_samples_std,
                    f"{prefix}/batch_samples_min": audit.batch_samples_min,
                    f"{prefix}/batch_samples_max": audit.batch_samples_max,
                    f"{prefix}/pca_var_2d": audit.pca_var_2d,
                    f"{prefix}/umap_available": float(audit.umap_available),
                }
                wandb_run.log(payload, commit=False)

                figures = figures_by_dataset[audit.dataset]
                wandb_payload = {}
                for figure_name, figure in figures.items():
                    wandb_payload[f"{prefix}/{figure_name}"] = wandb.Plotly(figure)
                wandb_run.log(wandb_payload, commit=False)

                batch_counts = batch_counts_by_dataset[audit.dataset]
                wandb_run.log(
                    {
                    f"{prefix}/batch_counts_table": wandb.Table(dataframe=batch_counts),
                    f"{prefix}/min_batch_samples": float(batch_counts["n_samples"].min()),
                    f"{prefix}/max_batch_samples": float(batch_counts["n_samples"].max()),
                    f"{prefix}/median_batch_samples": float(batch_counts["n_samples"].median()),
                    f"{prefix}/mean_batch_samples": float(batch_counts["n_samples"].mean()),
                    f"{prefix}/std_batch_samples": float(batch_counts["n_samples"].std(ddof=0)),
                },
                    commit=False,
                )

            if not baseline_overview.empty:
                wandb_run.log({"baselines/table": wandb.Table(dataframe=baseline_overview)}, commit=False)
                for index, row in enumerate(baseline_rows):
                    prefix = f"baselines/{row.dataset}/{row.task}/{row.preprocessor}/{row.model}"
                    payload = {
                        f"{prefix}/dataset_index": index,
                        f"{prefix}/bacc_cv": row.bacc_cv,
                        f"{prefix}/mcc_cv": row.mcc_cv,
                        f"{prefix}/cv_strategy": row.cv_strategy,
                        f"{prefix}/n_splits": row.n_splits,
                    }
                    if row.note:
                        payload[f"{prefix}/note"] = row.note
                    wandb_run.log(payload, commit=False)

            wandb_run.summary["overview/num_datasets"] = len(audits)
            wandb_run.summary["overview/num_baselines"] = int(len(baseline_rows))
            for column in overview.columns:
                if column in {"dataset", "source_csv"}:
                    continue
                if pd.api.types.is_numeric_dtype(overview[column]):
                    wandb_run.summary[f"overview/mean_{column}"] = float(overview[column].mean())
            if not baseline_overview.empty:
                numeric_cols = [col for col in baseline_overview.columns if pd.api.types.is_numeric_dtype(baseline_overview[col])]
                for column in numeric_cols:
                    wandb_run.summary[f"baselines/mean_{column}"] = float(baseline_overview[column].mean())
        except Exception as exc:
            print(f"[wandb] disabled ({type(exc).__name__}: {exc}); continuing without W&B")

    print(overview.to_string(index=False))
    for dataset_name, batch_counts in batch_counts_by_dataset.items():
        print(f"\nBatch counts for {dataset_name}:")
        print(batch_counts.to_string(index=False))
    if not baseline_overview.empty:
        print("\nBaseline sweep:")
        print(baseline_overview.to_string(index=False))
    print(f"\nWrote audit outputs to {output_dir}")
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
