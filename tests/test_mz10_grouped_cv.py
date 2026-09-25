from argparse import Namespace

import numpy as np
import pandas as pd

from scripts import hp_search
from scripts.hp_search import (
    assert_bernn_label_mapping_contract,
    cached_fold_feature_indices,
    percentile_prune_cutoff,
    sample_mz10_highrange_nonvariational_config,
    sample_mz10_highrange_plain_config,
    sample_mz10_nonvariational_config,
    sample_mz10_plain_config,
)


def test_mz10_uses_hybrid_fold_safe_feature_selection():
    from scripts.run_independent_optuna_all_datasets import (
        GROUPED_FEATURE_METHODS,
    )

    assert GROUPED_FEATURE_METHODS["bacteria_2024_mz10"] == "hybrid_xgboost_f"


class _Trial:
    number = 0


class _UpperBoundTrial:
    number = 1

    def suggest_float(self, _name, _low, high, **_kwargs):
        return high

    def suggest_int(self, _name, _low, high, **_kwargs):
        return high

    def suggest_categorical(self, _name, choices):
        return choices[0]


class _LowerBoundTrial(_UpperBoundTrial):
    def suggest_float(self, _name, low, _high, **_kwargs):
        return low

    def suggest_int(self, _name, low, _high, **_kwargs):
        return low


def test_mz10_sanity_config_is_plain_mlp():
    cfg = sample_mz10_plain_config(
        _Trial(), Namespace(max_warmup=50, force_sanity_config=True)
    )
    assert cfg["dloss"] == "no"
    assert cfg["variational"] is False
    assert cfg["kan"] is False
    assert cfg["class_triplet"] is False
    assert cfg["n_layers"] == 2
    assert cfg["layer1"] == 512


def test_mz10_extended_trial_zero_uses_requested_nonvariational_family():
    cfg = sample_mz10_nonvariational_config(
        _Trial(), Namespace(max_warmup=50)
    )
    assert cfg["dloss"] == "inverseTriplet"
    assert cfg["kan"] is True
    assert cfg["class_triplet"] is True
    assert cfg["variational"] is False
    assert cfg["beta"] == 0.0
    assert cfg["nu"] <= 0.5


def test_mz10_highrange_plain_starts_at_old_upper_bounds():
    cfg = sample_mz10_highrange_plain_config(_Trial(), Namespace(max_warmup=150))
    assert cfg["warmup"] == 50
    assert cfg["dropout"] == 0.2
    assert cfg["layer1"] == 512
    assert cfg["nu"] == 0.0


def test_mz10_highrange_extended_starts_at_old_upper_bounds():
    cfg = sample_mz10_highrange_nonvariational_config(
        _Trial(), Namespace(max_warmup=150)
    )
    assert cfg["warmup"] == 50
    assert cfg["dropout"] == 0.2
    assert cfg["layer1"] == 512
    assert cfg["variational"] is False


def test_mz10_highrange_plain_expands_width_to_2048():
    cfg = sample_mz10_highrange_plain_config(
        _UpperBoundTrial(), Namespace(max_warmup=150)
    )
    assert cfg["layer1"] == 2048


def test_mz10_highrange_extended_expands_width_to_2048():
    cfg = sample_mz10_highrange_nonvariational_config(
        _UpperBoundTrial(), Namespace(max_warmup=150)
    )
    assert cfg["layer1"] == 2048


def test_mz10_highrange_new_lower_bounds_are_warmup_one_and_zero_dropout():
    plain = sample_mz10_highrange_plain_config(
        _LowerBoundTrial(), Namespace(max_warmup=150)
    )
    extended = sample_mz10_highrange_nonvariational_config(
        _LowerBoundTrial(), Namespace(max_warmup=150)
    )
    assert plain["warmup"] == extended["warmup"] == 1
    assert plain["dropout"] == extended["dropout"] == 0.0
    assert plain["layer1"] == extended["layer1"] == 512


def test_percentile_pruning_waits_for_five_reference_trials():
    assert percentile_prune_cutoff([0.1, 0.2, 0.3, 0.4]) is None
    assert percentile_prune_cutoff([0.1, 0.2, 0.3, 0.4, 0.5]) == 0.2


def test_bernn_preserves_encoded_class_id_order():
    labels = np.asarray(["blank"] + [f"species_{i:02d}" for i in range(28)])
    assert_bernn_label_mapping_contract(labels)


def test_fold_feature_selection_uses_training_rows_and_cache(tmp_path):
    X = pd.DataFrame(
        {
            "train_signal": [0, 0, 10, 10, 0, 0],
            "validation_only": [0, 0, 0, 0, 100, 200],
            "constant": [1, 1, 1, 1, 1, 1],
            "noise": [0, 1, 0, 1, 0, 1],
        },
        dtype=np.float32,
    )
    y = np.asarray(["a", "a", "b", "b", "a", "b"])
    train_idx = np.asarray([0, 1, 2, 3])
    first = cached_fold_feature_indices(
        X, y, train_idx, fold_idx=0, k=1, cache_dir=tmp_path
    )
    second = cached_fold_feature_indices(
        X, y, train_idx, fold_idx=0, k=1, cache_dir=tmp_path
    )
    assert first.tolist() == [0]
    assert second.tolist() == first.tolist()
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_hybrid_selector_combines_xgboost_gain_and_training_signal(
    tmp_path, monkeypatch
):
    X = pd.DataFrame(
        {
            "xgb_interaction": [0, 1, 0, 1, 9, 9],
            "univariate": [0, 0, 10, 10, 9, 9],
            "validation_only": [0, 0, 0, 0, 100, 200],
            "noise": [0, 1, 0, 1, 0, 1],
        },
        dtype=np.float32,
    )
    y = np.asarray(["a", "a", "b", "b", "a", "b"])
    train_idx = np.asarray([0, 1, 2, 3])
    monkeypatch.setattr(
        hp_search,
        "_xgboost_gain_scores",
        lambda values, labels: np.asarray([100.0, 0.0, 0.0, 0.0]),
    )
    selected = cached_fold_feature_indices(
        X,
        y,
        train_idx,
        fold_idx=0,
        k=2,
        cache_dir=tmp_path,
        method="hybrid_xgboost_f",
    )
    assert selected.tolist() == [0, 1]
    assert 2 not in selected
