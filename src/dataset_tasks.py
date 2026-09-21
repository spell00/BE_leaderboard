"""Dataset-specific task definitions shared by training and online evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

ALZHEIMER_DATASET = "massbench_alzheimer"
ALZHEIMER_SUPERVISED_LABELS = frozenset({"CU", "DEM-AD"})
POOL_LABEL = "pool"
UNSUPERVISED_LABEL = "-1"


def normalized_labels(labels) -> pd.Series:
    """Return stripped nullable-string labels without changing row order."""
    return pd.Series(labels).astype("string").str.strip()


def alzheimer_supervised_mask(labels) -> pd.Series:
    """Rows belonging to the declared Alzheimer supervised task (CU vs DEM-AD)."""
    return normalized_labels(labels).isin(ALZHEIMER_SUPERVISED_LABELS)


def prepare_alzheimer_development_labels(labels) -> tuple[pd.Series, pd.Series]:
    """Map non-CU/DEM-AD diagnoses to the pooled unsupervised class."""
    normalized = normalized_labels(labels)
    supervised = normalized.isin(ALZHEIMER_SUPERVISED_LABELS)
    mapped = normalized.where(supervised, POOL_LABEL)
    return mapped, supervised


def prepare_builtin_training_frame(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the task definition used by offline HPO to a public train frame."""
    if "label" not in frame.columns:
        raise ValueError("Training frame must contain a label column")

    out = frame.copy()
    if dataset == ALZHEIMER_DATASET:
        mapped, _ = prepare_alzheimer_development_labels(out["label"])
        out["label"] = mapped
        return out.reset_index(drop=True)

    labels = normalized_labels(out["label"])
    labelled = labels.notna() & labels.ne("")
    out = out.loc[labelled].copy()
    out["label"] = labels.loc[labelled]
    return out.reset_index(drop=True)


def model_labels_for_alzheimer(labels) -> pd.Series:
    """Convert pooled rows to BERNN's -1 unsupervised sentinel."""
    normalized = normalized_labels(labels)
    return normalized.where(
        normalized.isin(ALZHEIMER_SUPERVISED_LABELS),
        UNSUPERVISED_LABEL,
    )


def task_feature_columns(frame: pd.DataFrame) -> list[str]:
    meta_columns = {"name", "names", "batch", "batches", "label", "labels", "group"}
    return [column for column in frame.columns if column not in meta_columns]


def clean_task_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Match hp_search feature preprocessing: float conversion and non-finite -> 0."""
    values = frame[feature_columns].astype(float).reset_index(drop=True)
    return values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
