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


def prepare_research_source_frame(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare one user-selected source CSV for supervised research CV.

    Unlike the legacy fixed-test path, rows with missing/blank labels are always
    ignored. This is important for *_all.csv files built by concatenating a
    labeled train split with an unlabeled public test/inference split. For
    Alzheimer, labeled non-CU/DEM-AD diagnoses are still retained as pooled
    unsupervised rows after unlabeled rows have been removed.
    """
    if "label" not in frame.columns:
        raise ValueError("Selected research source file must contain a label column")

    labels = normalized_labels(frame["label"])
    out = frame.copy()

    if dataset == ALZHEIMER_DATASET:
        # CU / DEM-AD stay supervised.
        # Other known Alzheimer diagnoses remain pooled/unsupervised.
        # Missing labels are also retained as unsupervised samples.
        mapped, _ = prepare_alzheimer_development_labels(labels)
        mapped = mapped.fillna(POOL_LABEL)
        mapped = mapped.where(mapped.ne(""), POOL_LABEL)
        out["label"] = mapped
    else:
        # Keep unlabeled rows for unsupervised/batch-learning use.
        # They must not contribute to supervised classification loss.
        out["label"] = labels.fillna(UNSUPERVISED_LABEL)
        out["label"] = out["label"].where(
            out["label"].ne(""),
            UNSUPERVISED_LABEL,
        )

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


def cyclic_train_valid_splits(batches, eligible_mask=None, n_splits: int = -1):
    """Rotate grouped batch roles for train/validation-only research CV.

    With n_splits=-1, each evaluable batch is validation exactly once and every
    other batch trains. A positive n_splits groups batches deterministically.
    No test or inference matrix participates in this protocol.
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
    n_batches = len(ordered)
    if n_batches < 2:
        raise ValueError(
            "Train/validation batch CV requires at least 2 evaluable batches; "
            f"found {n_batches}: {ordered}"
        )

    try:
        requested = int(n_splits)
    except (TypeError, ValueError):
        raise ValueError("Number of batch CV folds must be -1 or an integer >= 2")

    if requested == -1:
        resolved = n_batches
    elif requested < 2:
        raise ValueError(
            "Number of batch CV folds must be -1 (leave-one-batch-out) or at least 2"
        )
    elif requested > n_batches:
        raise ValueError(
            f"Requested {requested} batch CV folds, but only {n_batches} evaluable "
            "batches are available"
        )
    else:
        resolved = requested

    grouped = [
        [str(value) for value in group.tolist()]
        for group in np.array_split(np.asarray(ordered, dtype=object), resolved)
    ]
    if any(not group for group in grouped):
        raise AssertionError("Batch CV grouping produced an empty group")

    splits = []
    for round_index in range(resolved):
        valid_batches = grouped[round_index]
        valid_set = set(valid_batches)
        train_batches = [batch for batch in all_batches if batch not in valid_set]

        train_idx = np.flatnonzero(np.isin(values, train_batches))
        valid_idx = np.flatnonzero(np.isin(values, valid_batches))
        if not len(train_idx) or not len(valid_idx):
            raise ValueError(
                "Train/validation batch CV produced an empty split: "
                f"train={len(train_idx)} valid={len(valid_idx)}"
            )

        splits.append({
            "round": round_index + 1,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "train_batches": train_batches,
            "valid_batches": valid_batches,
            "valid_batch": valid_batches[0] if len(valid_batches) == 1 else valid_batches,
            "n_splits": resolved,
            "requested_n_splits": requested,
        })

    valid_roles = sorted(
        [batch for row in splits for batch in row["valid_batches"]],
        key=_natural_batch_key,
    )
    if valid_roles != ordered:
        raise AssertionError("Each evaluable batch must appear exactly once as validation")

    return splits


def cyclic_train_valid_test_splits(
    batches, eligible_mask=None, n_splits: int = -1, batch_order=None
):
    """Rotate grouped batch roles for symmetric train/validation/test CV.

    n_splits=-1 is leave-one-batch-out over every evaluable batch. For a
    positive n_splits, evaluable batches are deterministically partitioned
    into that many groups; every batch still appears exactly once in validation
    and exactly once in test, while the remaining groups train.

    With ordered eligible batches [1, 2, 3] and n_splits=-1:
      round 1: train=[1], valid=[2], test=[3]
      round 2: train=[2], valid=[3], test=[1]
      round 3: train=[3], valid=[1], test=[2]

    At least three CV groups are required because each round needs separate
    train, validation, and test groups.
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

    eligible_set = set(eligible_values.tolist())
    if batch_order is None:
        ordered = sorted(eligible_set, key=_natural_batch_key)
    else:
        ordered = [str(value) for value in batch_order]
        if len(ordered) != len(set(ordered)):
            raise ValueError("batch_order must not contain duplicates")
        if set(ordered) != eligible_set:
            raise ValueError(
                "batch_order must contain every evaluable batch exactly once; "
                f"expected={sorted(eligible_set, key=_natural_batch_key)} got={ordered}"
            )
    n_batches = len(ordered)
    if n_batches < 3:
        raise ValueError(
            "Cyclic train/valid/test batch CV requires at least 3 evaluable batches; "
            f"found {n_batches}: {ordered}"
        )

    try:
        requested = int(n_splits)
    except (TypeError, ValueError):
        raise ValueError("Number of batch CV folds must be -1 or an integer >= 3")

    if requested == -1:
        resolved = n_batches
    elif requested < 3:
        raise ValueError(
            "Number of batch CV folds must be -1 (leave-one-batch-out) or at least 3"
        )
    elif requested > n_batches:
        raise ValueError(
            f"Requested {requested} batch CV folds, but only {n_batches} evaluable "
            "batches are available"
        )
    else:
        resolved = requested

    # Split the naturally ordered evaluable batches as evenly as possible. This
    # keeps every batch represented even when, for example, 20 batches are
    # evaluated with 5 CV folds.
    grouped = [
        [str(value) for value in group.tolist()]
        for group in np.array_split(np.asarray(ordered, dtype=object), resolved)
    ]
    if any(not group for group in grouped):
        raise AssertionError("Batch CV grouping produced an empty group")

    splits = []
    for round_index in range(resolved):
        valid_batches = grouped[(round_index + 1) % resolved]
        test_batches = grouped[(round_index + 2) % resolved]
        held_out = set(valid_batches) | set(test_batches)
        train_batches = [batch for batch in all_batches if batch not in held_out]

        train_idx = np.flatnonzero(np.isin(values, train_batches))
        valid_idx = np.flatnonzero(np.isin(values, valid_batches))
        test_idx = np.flatnonzero(np.isin(values, test_batches))
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
            "valid_batches": valid_batches,
            "test_batches": test_batches,
            "valid_batch": valid_batches[0] if len(valid_batches) == 1 else valid_batches,
            "test_batch": test_batches[0] if len(test_batches) == 1 else test_batches,
            "n_splits": resolved,
            "requested_n_splits": requested,
        })

    valid_roles = sorted(
        [batch for row in splits for batch in row["valid_batches"]],
        key=_natural_batch_key,
    )
    test_roles = sorted(
        [batch for row in splits for batch in row["test_batches"]],
        key=_natural_batch_key,
    )
    expected_roles = sorted(ordered, key=_natural_batch_key)
    if valid_roles != expected_roles:
        raise AssertionError("Each evaluable batch must appear exactly once as validation")
    if test_roles != expected_roles:
        raise AssertionError("Each evaluable batch must appear exactly once as test")

    return splits
