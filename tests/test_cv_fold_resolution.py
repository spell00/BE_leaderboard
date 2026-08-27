import numpy as np

from scripts.hp_search import cached_cv_splits, resolve_n_repeats


def test_requested_cv3_resolves_to_cv2_for_two_development_batches(tmp_path):
    y = np.asarray([0, 1, 0, 1])
    batches = np.asarray(["1", "1", "2", "2"])
    assert resolve_n_repeats(3, batches) == 2
    splits = cached_cv_splits(y, batches, 3, tmp_path / "splits.npz")
    assert len(splits) == 2
    assert all(len(train) == 2 and len(valid) == 2 for train, valid in splits)


def test_empty_legacy_cache_is_rejected_and_rebuilt(tmp_path):
    y = np.asarray([0, 1, 0, 1]).astype(str)
    batches = np.asarray(["1", "1", "2", "2"])
    path = tmp_path / "splits.npz"
    cached_cv_splits(y, batches, 3, path)
    with np.load(path, allow_pickle=False) as payload:
        legacy = {key: payload[key] for key in payload.files if key != "resolved_n_repeats"}
    legacy.update(
        count=np.asarray(3, dtype=np.int64),
        train_2=np.arange(len(y), dtype=np.int64),
        valid_2=np.asarray([], dtype=np.int64),
    )
    np.savez_compressed(path, **legacy)

    rebuilt = cached_cv_splits(y, batches, 3, path)
    assert len(rebuilt) == 2
    assert all(len(train) and len(valid) for train, valid in rebuilt)
    with np.load(path, allow_pickle=False) as payload:
        assert int(payload["resolved_n_repeats"].item()) == 2
        assert int(payload["count"].item()) == 2
