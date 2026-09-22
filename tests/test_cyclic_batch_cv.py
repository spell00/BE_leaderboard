import numpy as np
import pandas as pd
import pytest

from src.dataset_tasks import cyclic_train_valid_test_splits
from src import code_challenge


def _role_batches(splits, key):
    return sorted(batch for split in splits for batch in split[key])


def test_cyclic_minus_one_is_true_batch_lbo():
    batches = np.repeat([str(i) for i in range(1, 21)], 2)
    splits = cyclic_train_valid_test_splits(batches, n_splits=-1)

    assert len(splits) == 20
    assert all(len(split["valid_batches"]) == 1 for split in splits)
    assert all(len(split["test_batches"]) == 1 for split in splits)
    assert _role_batches(splits, "valid_batches") == sorted(set(batches))
    assert _role_batches(splits, "test_batches") == sorted(set(batches))


def test_cyclic_five_groups_all_batches_without_dropping_any():
    batches = np.repeat([str(i) for i in range(1, 21)], 2)
    splits = cyclic_train_valid_test_splits(batches, n_splits=5)

    assert len(splits) == 5
    assert all(len(split["valid_batches"]) == 4 for split in splits)
    assert all(len(split["test_batches"]) == 4 for split in splits)
    assert _role_batches(splits, "valid_batches") == sorted(set(batches))
    assert _role_batches(splits, "test_batches") == sorted(set(batches))


def test_requested_folds_cannot_exceed_evaluable_batches():
    batches = np.repeat(["1", "2", "3", "4"], 2)
    with pytest.raises(ValueError, match="only 4 evaluable batches"):
        cyclic_train_valid_test_splits(batches, n_splits=5)


def test_at_least_three_groups_are_required():
    batches = np.repeat(["1", "2", "3", "4", "5"], 2)
    with pytest.raises(ValueError, match="at least 3"):
        cyclic_train_valid_test_splits(batches, n_splits=2)


def test_geo_cyclic_loader_does_not_require_private_inference(tmp_path, monkeypatch):
    dataset = "normal_tissue_878"
    base = tmp_path / "data" / "datasets" / dataset
    base.mkdir(parents=True)
    pd.DataFrame(
        {
            "name": [f"s{i}" for i in range(6)],
            "batch": ["1", "1", "2", "2", "3", "3"],
            "label": ["blood", "colon", "blood", "colon", "blood", "colon"],
            "f1": [1.0, 2.0, 1.5, 2.5, 0.5, 3.0],
            "f2": [0.0, 1.0, 0.2, 1.2, 0.4, 1.4],
        }
    ).to_csv(base / f"{dataset}_train.csv", index=False)

    monkeypatch.setattr(code_challenge, "ROOT", tmp_path)

    def _private_should_not_be_called(*args, **kwargs):
        raise AssertionError("GEO cyclic loading must not request private inference")

    monkeypatch.setattr(
        code_challenge,
        "load_private_inference",
        _private_should_not_be_called,
    )
    monkeypatch.setattr(
        code_challenge,
        "load_private_labels",
        _private_should_not_be_called,
    )

    X, y, batches, names = code_challenge._load_cyclic_research_dataset(dataset)

    assert len(X) == 6
    assert len(y) == 6
    assert len(names) == 6
    assert sorted(batches.unique().tolist()) == ["1", "2", "3"]
