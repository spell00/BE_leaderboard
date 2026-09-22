import numpy as np
import pytest

from src.dataset_tasks import cyclic_train_valid_test_splits


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
