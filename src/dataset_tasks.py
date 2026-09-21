"""Dataset-specific task definitions shared by training and online evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

ALZHEIMER_DATASET = "massbench_alzheimer"
ALZHEIMER_SUPERVISED_LABELS = frozenset({"CU", "DEM-AD"})
POOL_LABEL = "pool"
UNSUPERVISED_LABEL = "-1"

# Synchronized meta-HPO uses three grouped folds by default. Datasets with fewer
# technical batches automatically reduce the fold count (e.g. adenocarcinoma: 2).
META_HPO_N_REPEATS = 3
META_HPO_CV_RANDOM_STATE = 0


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


def _natural_batch_key(value: object):
    """Deterministic human/numeric ordering for batch identifiers."""
    import re

    text = str(value)
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
        if part != ""
    )


def cyclic_train_valid_test_splits(batches, eligible_mask=None):
    """Rotate eligible batch roles so each is validation once and test once.

    Batches with no eligible/scorable rows remain training-only in every round.
    With ordered eligible batches [1, 2, 3], this yields:
      round 1: train=[1], valid=[2], test=[3]
      round 2: train=[2], valid=[3], test=[1]
      round 3: train=[3], valid=[1], test=[2]

    With >3 eligible batches, all batches other than the current validation/test
    batches are used for training. At least three evaluable batches are required.
    """
    values = np.asarray(pd.Series(batches).astype(str))
    all_batches = sorted(set(values.tolist()), key=_natural_batch_key)

    if eligible_mask is None:
        eligible_values = values
    else:
        eligible_mask = np.asarray(eligible_mask, dtype=bool)
        if eligible_mask.shape != values.shape:
            raise ValueError("eligible_mask must have one boolean per sample")
        eligible_values = values[eligible_mask]

    ordered = sorted(set(eligible_values.tolist()), key=_natural_batch_key)
    if len(ordered) < 3:
        raise ValueError(
            "Cyclic train/valid/test batch CV requires at least 3 evaluable batches; "
            f"found {len(ordered)}: {ordered}"
        )

    splits = []
    for round_index in range(len(ordered)):
        valid_batch = ordered[(round_index + 1) % len(ordered)]
        test_batch = ordered[(round_index + 2) % len(ordered)]
        train_batches = [
            batch for batch in all_batches
            if batch not in {valid_batch, test_batch}
        ]

        train_idx = np.flatnonzero(np.isin(values, train_batches))
        valid_idx = np.flatnonzero(values == valid_batch)
        test_idx = np.flatnonzero(values == test_batch)
        if not len(train_idx) or not len(valid_idx) or not len(test_idx):
            raise ValueError(
                "Cyclic batch CV produced an empty split: "
                f"train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}"
            )

        splits.append({
            "round": round_index + 1,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
            "train_batches": train_batches,
            "valid_batch": valid_batch,
            "test_batch": test_batch,
        })

    valid_roles = [row["valid_batch"] for row in splits]
    test_roles = [row["test_batch"] for row in splits]
    if sorted(valid_roles, key=_natural_batch_key) != ordered:
        raise AssertionError("Each evaluable batch must appear exactly once as validation")
    if sorted(test_roles, key=_natural_batch_key) != ordered:
        raise AssertionError("Each evaluable batch must appear exactly once as test")

    return splits
