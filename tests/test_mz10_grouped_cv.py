from argparse import Namespace

import numpy as np
import pandas as pd

from scripts import hp_search
from scripts.hp_search import (
    assert_bernn_label_mapping_contract,
    cached_fold_feature_indices,
    percentile_prune_cutoff,
    predict_final_mcc_from_partial,
    report_repeat_progress,
    load_fold_checkpoint,
    save_fold_checkpoint,
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


def test_repeat1_prunes_against_historical_intermediate_values():
    import optuna
    import pytest

    study = optuna.create_study(direction="maximize")
    for value in [0.40, 0.50, 0.60, 0.70, 0.80]:
        study.add_trial(
            optuna.trial.create_trial(
                value=value,
                intermediate_values={hp_search.FOLD_PRUNE_STEP_BASE + 1: value},
            )
        )
    trial = study.ask()
    args = Namespace(
        enable_optuna_pruning=True,
        resolved_n_repeats=5,
        prune_min_reference_trials=5,
        repeat1_prune_percentile=25.0,
        repeat1_hard_floor=-1.0,
    )
    with pytest.raises(optuna.TrialPruned):
        report_repeat_progress(trial, [0.30], args, 0)
    assert trial.user_attrs["pruning_stage"] == "repeat_1_percentile"
    assert trial.user_attrs["resource_repeats_completed"] == 1


def test_repeat2_is_final_when_protocol_has_only_two_repeats():
    import optuna

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    args = Namespace(
        enable_optuna_pruning=True,
        resolved_n_repeats=2,
        repeat1_hard_floor=-1.0,
        repeat2_hard_floor=0.99,
    )
    report_repeat_progress(trial, [0.5, 0.4], args, 1)
    assert trial.user_attrs["resource_repeats_completed"] == 2


def test_fold_checkpoint_roundtrip_and_config_guard(tmp_path):
    args = Namespace(fold_checkpoint_dir=str(tmp_path))
    config = {"lr": 1e-3, "n_layers": 2}
    save_fold_checkpoint(
        args,
        config,
        0,
        {"valid_mcc": 0.61, "metrics": {"valid_mcc": 0.61}},
    )
    restored = load_fold_checkpoint(args, config, 0)
    assert restored["valid_mcc"] == 0.61
    assert load_fold_checkpoint(args, {"lr": 2e-3, "n_layers": 2}, 0) is None


def test_partial_repeat_predictor_learns_historical_final_mapping():
    import optuna

    study = optuna.create_study(direction="maximize")
    step = hp_search.FOLD_PRUNE_STEP_BASE + 1
    for partial in np.linspace(0.2, 0.9, 8):
        final = 0.1 + 0.8 * float(partial)
        study.add_trial(
            optuna.trial.create_trial(
                value=final,
                intermediate_values={step: float(partial)},
            )
        )
    trial = study.ask()
    predicted, residual_std, count = predict_final_mcc_from_partial(
        trial, step, 0.5, min_reference_trials=8
    )
    assert count == 8
    assert abs(predicted - 0.5) < 1e-8
    assert residual_std < 1e-8


def test_run_trial_reuses_completed_repeat_checkpoints(tmp_path, monkeypatch):
    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 2.0, 2.0]})
    y = np.asarray(["a", "a", "b", "b"])
    batches = np.asarray(["b1", "b1", "b2", "b2"])
    splits = [
        (np.asarray([0, 1]), np.asarray([2, 3])),
        (np.asarray([2, 3]), np.asarray([0, 1])),
    ]
    monkeypatch.setattr(hp_search, "cached_cv_splits", lambda *a, **k: splits)
    monkeypatch.setattr(
        hp_search,
        "cached_fold_feature_indices",
        lambda *a, **k: np.asarray([0, 1]),
    )
    calls = []

    def fake_fit(cfg, args, data, exp_id, seed, keep_models):
        calls.append(exp_id)
        return object(), 0.4 if exp_id.endswith("fold0") else 0.6

    monkeypatch.setattr(hp_search, "_fit_one", fake_fit)
    monkeypatch.setattr(hp_search, "extract_mlflow_metrics", lambda *a, **k: {})
    monkeypatch.setattr(hp_search, "extract_mlflow_epoch_traces", lambda *a, **k: [])

    args = Namespace(
        dataset="demo",
        n_repeats=2,
        resolved_n_repeats=2,
        cv_split_cache=None,
        feature_select_k=0,
        feature_select_cache_dir=None,
        feature_select_method="f_classif",
        seed=42,
        enable_optuna_pruning=False,
        optuna_trial=None,
        fold_checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    cfg = {"marker": "same-config"}
    first, _ = hp_search.run_trial(cfg, args, (X, y, batches), "resume-test")
    assert first == 0.5
    assert len(calls) == 2

    monkeypatch.setattr(
        hp_search,
        "_fit_one",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fit should not rerun")),
    )
    second, metrics = hp_search.run_trial(cfg, args, (X, y, batches), "resume-test")
    assert second == 0.5
    assert metrics["valid_mcc_folds"] == [0.4, 0.6]
