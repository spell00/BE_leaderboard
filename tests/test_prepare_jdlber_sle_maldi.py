import numpy as np
import pandas as pd
import pytest

from scripts.prepare_jdlber_sle_maldi import (
    aggregate_subjects,
    canonical_mz_name,
    validate_feature_alignment,
)


def test_canonical_mz_name_normalizes_equivalent_headers():
    assert canonical_mz_name("103.0") == "103"
    assert canonical_mz_name("103") == "103"
    assert canonical_mz_name("172.810") == "172.81"


def test_aggregate_subjects_collapses_replicates_without_leakage():
    source = {
        "labels": np.array([0, 0, 1]),
        "boards": np.array([1, 1, 1]),
        "features": np.array(
            [
                [1.0, 4.0],
                [3.0, 6.0],
                [10.0, 20.0],
            ]
        ),
        "feature_names": ["100.9", "101.9"],
        "replicate_counts": [2, 1],
    }

    frame, stats = aggregate_subjects(source, "median")

    assert frame[["name", "batch", "label"]].to_dict("records") == [
        {
            "name": "JDLBER_board1_subject001",
            "batch": "1",
            "label": "HC",
        },
        {
            "name": "JDLBER_board1_subject002",
            "batch": "1",
            "label": "SLE",
        },
    ]
    assert frame[["100.9", "101.9"]].to_numpy().tolist() == [
        [2.0, 5.0],
        [10.0, 20.0],
    ]
    assert stats == {
        "subjects": 2,
        "spectra": 3,
        "replicate_count_distribution": {"2": 1, "1": 1},
    }


def test_aggregate_subjects_rejects_mixed_subject_labels():
    source = {
        "labels": np.array([0, 1]),
        "boards": np.array([1, 1]),
        "features": np.array([[1.0], [2.0]]),
        "feature_names": ["100.9"],
        "replicate_counts": [2],
    }

    with pytest.raises(ValueError, match="inconsistent replicate labels"):
        aggregate_subjects(source, "median")


def test_feature_alignment_requires_same_canonical_grid():
    batches = [
        {"feature_names": ["100.9", "103"]},
        {"feature_names": ["100.9", "103"]},
    ]
    assert validate_feature_alignment(batches) == ["100.9", "103"]

    with pytest.raises(ValueError, match="does not align"):
        validate_feature_alignment(
            [
                {"feature_names": ["100.9", "103"]},
                {"feature_names": ["100.9", "104"]},
            ]
        )
