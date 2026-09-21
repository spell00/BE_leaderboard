"""Regression tests for the Alzheimer semi-supervised task definition."""

import numpy as np
import pandas as pd

from src.code_challenge import _submission_cv_splits
from src.dataset_tasks import (
    ALZHEIMER_DATASET,
    POOL_LABEL,
    UNSUPERVISED_LABEL,
    model_labels_for_alzheimer,
    prepare_alzheimer_development_labels,
)


def test_alzheimer_labels_match_meta_training_task():
    raw = pd.Series(["CU", "DEM-AD", "MCI-AD", "NPH", "", None])
    mapped, supervised = prepare_alzheimer_development_labels(raw)

    assert supervised.tolist() == [True, True, False, False, False, False]
    assert mapped.astype(str).tolist() == [
        "CU",
        "DEM-AD",
        POOL_LABEL,
        POOL_LABEL,
        POOL_LABEL,
        POOL_LABEL,
    ]

    model_labels = model_labels_for_alzheimer(mapped)
    assert model_labels.astype(str).tolist() == [
        "CU",
        "DEM-AD",
        UNSUPERVISED_LABEL,
        UNSUPERVISED_LABEL,
        UNSUPERVISED_LABEL,
        UNSUPERVISED_LABEL,
    ]


def test_alzheimer_cv_splits_on_supervised_rows_then_expands_whole_batches():
    rows = []
    for batch_index in range(10):
        batch = f"B{batch_index:02d}"
        rows.extend([
            (batch, "CU"),
            (batch, "DEM-AD"),
            (batch, POOL_LABEL),
            (batch, POOL_LABEL),
        ])

    batches = pd.Series([row[0] for row in rows])
    labels = pd.Series([row[1] for row in rows])
    X = pd.DataFrame({"feature": np.arange(len(rows), dtype=float)})

    protocol, splits = _submission_cv_splits(
        ALZHEIMER_DATASET,
        X,
        labels,
        batches,
    )

    assert "Alzheimer semi-supervised" in protocol
    assert len(splits) == 3

    all_indices = set(range(len(rows)))
    for train_idx, valid_idx in splits:
        train_set = set(map(int, train_idx))
        valid_set = set(map(int, valid_idx))
        assert train_set.isdisjoint(valid_set)
        assert train_set | valid_set == all_indices

        train_batches = set(batches.iloc[train_idx])
        valid_batches = set(batches.iloc[valid_idx])
        assert train_batches.isdisjoint(valid_batches)

        # Whole batches, including pooled rows, must stay on one side.
        for batch in set(batches):
            batch_indices = set(np.flatnonzero(batches.to_numpy() == batch))
            assert batch_indices <= train_set or batch_indices <= valid_set


def test_adenocarcinoma_uses_two_batch_holdout_folds():
    batches = pd.Series(["1"] * 6 + ["2"] * 6)
    labels = pd.Series(["1", "1", "QC", "0", "1", "1"] * 2)
    X = pd.DataFrame({"feature": np.arange(len(labels), dtype=float)})

    protocol, splits = _submission_cv_splits(
        "massbench_adenocarcinoma",
        X,
        labels,
        batches,
    )

    assert "StratifiedGroupKFold(n_splits=2" in protocol
    assert len(splits) == 2

    for train_idx, valid_idx in splits:
        train_batches = set(batches.iloc[train_idx].astype(str))
        valid_batches = set(batches.iloc[valid_idx].astype(str))
        assert train_batches.isdisjoint(valid_batches)
        assert len(train_batches) == 1
        assert len(valid_batches) == 1
