import numpy as np


def test_bacteria_mz10_uses_five_group_rotation():
    from scripts.run_bacteria_2024_mz10_optuna import (
        N_SPLITS,
        choose_five_batch_groups,
        make_splitter,
    )

    batches = np.repeat([f"b{i}" for i in range(1, 16)], 6)
    labels = np.tile(np.array(["a", "b", "c", "a", "b", "c"]), 15)

    groups, group_sizes, _ = choose_five_batch_groups(labels, batches, seed=42)
    assert N_SPLITS == 5
    assert len(groups) == 5
    assert len(group_sizes) == 5
    flattened = [batch for group in groups for batch in group]
    assert sorted(flattened, key=lambda value: int(value[1:])) == [
        f"b{i}" for i in range(1, 16)
    ]

    splits = make_splitter(labels, batches, groups)(batches)
    assert len(splits) == 5

    valid_roles = []
    test_roles = []
    for split in splits:
        train = set(split["train_batches"])
        valid = set(split["valid_batches"])
        test = set(split["test_batches"])
        assert train.isdisjoint(valid)
        assert train.isdisjoint(test)
        assert valid.isdisjoint(test)
        assert train | valid | test == set(flattened)
        valid_roles.extend(split["valid_batches"])
        test_roles.extend(split["test_batches"])

    assert sorted(valid_roles, key=lambda value: int(value[1:])) == [
        f"b{i}" for i in range(1, 16)
    ]
    assert sorted(test_roles, key=lambda value: int(value[1:])) == [
        f"b{i}" for i in range(1, 16)
    ]
