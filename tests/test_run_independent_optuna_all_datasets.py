from pathlib import Path

from scripts.run_independent_optuna_all_datasets import (
    CYCLIC_CV_FOLDS,
    DATASETS,
    GROUPED_CV_FOLDS,
    GROUPED_FEATURE_COUNTS,
    import_seed_trials,
    parse_args,
    mz10_sanity_gate,
    selected_datasets,
    save_wandb_files,
    wandb_metrics,
)


class _FakeRun:
    def __init__(self):
        self.saved = []

    def save(self, path, **kwargs):
        self.saved.append((Path(path), kwargs))


def test_default_dataset_order_puts_new_benchmarks_first():
    assert [dataset for dataset, _ in DATASETS[:3]] == [
        "bacteria_2024_mz10",
        "jdlber_sle_maldi",
        "seqc_maqc",
    ]
    assert len(DATASETS) == 8
    assert len({dataset for dataset, _ in DATASETS}) == 8


def test_mz10_uses_grouped_five_fold_cv_and_10k_features():
    assert dict(DATASETS)["bacteria_2024_mz10"] == "grouped_cv"
    assert GROUPED_CV_FOLDS["bacteria_2024_mz10"] == 5
    assert GROUPED_FEATURE_COUNTS["bacteria_2024_mz10"] == 10_000
    assert CYCLIC_CV_FOLDS["scib_pancreas"] == 3
    assert "scib_pancreas" not in dict(DATASETS)


def test_feature_selector_can_be_overridden_for_ablation():
    args = parse_args(["--feature-select-method", "xgboost_gain"])
    assert args.feature_select_method == "xgboost_gain"


def test_seed_trial_paths_are_repeatable():
    args = parse_args([
        "--seed-trials-from", "old-a/trials.json",
        "--seed-trials-from", "old-b/trials.json",
    ])
    assert args.seed_trials_from == [Path("old-a/trials.json"), Path("old-b/trials.json")]


def test_imports_compatible_completed_trial_as_tpe_evidence(tmp_path):
    import json
    import optuna

    config = {
        "dloss": "no", "variational": False, "kan": False,
        "class_triplet": False, "lr": 1e-3, "wd": 1e-5,
        "smoothing": 0.01, "dropout": 0.2, "warmup": 47,
        "n_layers": 1, "layer1": 512, "scaler": "standard",
    }
    path = tmp_path / "trials.json"
    path.write_text(json.dumps([{
        "protocol": "grouped_cv", "valid_mcc": 0.8, "config": config,
        "metrics": {"valid_mcc_folds": [0.7, 0.8, 0.75, 0.85, 0.9]},
        "fit_seconds": 123,
    }]))
    study = optuna.create_study(direction="maximize")

    assert import_seed_trials(
        study, [path], "bacteria_2024_mz10", "grouped_cv", "highrange_plain"
    ) == 1
    seeded = study.trials[0]
    assert seeded.user_attrs["transfer_seed"] is True
    assert seeded.value == 0.8
    assert seeded.intermediate_values[200_002] == 0.75


def test_mz10_extended_search_disables_variational_family():
    args = parse_args(["--mz10-search-space", "extended_nonvariational"])
    assert args.mz10_search_space == "extended_nonvariational"


def test_mz10_sanity_gate_checks_training_and_grouped_validation():
    passed, details = mz10_sanity_gate(
        {"mcc_train_all_concentrations": 0.95}, 0.40
    )
    assert passed
    assert details["passed"]
    assert not mz10_sanity_gate(
        {"mcc_train_all_concentrations": 0.80}, 0.40
    )[0]
    assert not mz10_sanity_gate(
        {"mcc_train_all_concentrations": 0.95}, 0.20
    )[0]


def test_explicit_dataset_subset_keeps_requested_order():
    assert selected_datasets("massbench_alzheimer,seqc_maqc") == [
        ("massbench_alzheimer", "fixed_external"),
        ("seqc_maqc", "cyclic"),
    ]


def test_wandb_payload_contains_all_valid_and_test_fold_values():
    payload = wandb_metrics(
        {
            "valid_mcc": 0.7,
            "test_mcc": 0.6,
            "valid_mcc_folds": [0.6, 0.8],
            "test_mcc_folds": [0.5, 0.7],
            "test_mcc_fold_mean": 0.6,
        }
    )
    assert payload["metrics/valid_mcc"] == 0.7
    assert payload["metrics/test_mcc"] == 0.6
    assert payload["metrics/test_mcc_fold_mean"] == 0.6
    assert payload["folds/valid_mcc/fold_0"] == 0.6
    assert payload["folds/valid_mcc/fold_1"] == 0.8
    assert payload["folds/test_mcc/fold_0"] == 0.5
    assert payload["folds/test_mcc/fold_1"] == 0.7


def test_save_wandb_files_uploads_nested_files_without_globbing(tmp_path):
    source = tmp_path / "dataset" / "provenance"
    (source / "repository").mkdir(parents=True)
    (source / "manifest.json").write_text("{}\n")
    (source / "repository" / "train.py").write_text("WIDTH = 2048\n")
    run = _FakeRun()

    count = save_wandb_files(run, [source], base_path=tmp_path)

    assert count == 2
    assert {path.name for path, _ in run.saved} == {"manifest.json", "train.py"}
    assert all(kwargs["policy"] == "now" for _, kwargs in run.saved)
    assert all(kwargs["glob"] is False for _, kwargs in run.saved)
    assert all(kwargs["base_path"] == str(tmp_path.resolve()) for _, kwargs in run.saved)


def test_pruning_and_historical_bootstrap_are_enabled_by_default():
    args = parse_args([])
    assert args.pruning is True
    assert args.auto_seed_history is True
    assert args.seed_from_wandb is True
    assert args.repeat1_prune_percentile == 25.0
    assert args.repeat2_prune_percentile == 50.0


def test_generic_completed_trial_without_fold_array_still_seeds_tpe(tmp_path):
    import json
    import optuna

    config = {
        "dloss": "DANN", "variational": True, "kan": False,
        "class_triplet": True, "class_triplet_w": 0.3,
        "lr": 1e-3, "wd": 1e-5, "nu": 0.1,
        "smoothing": 0.01, "margin": 1.0, "dropout": 0.2,
        "thres": 0.01, "warmup": 20, "n_layers": 2,
        "layer1": 700, "scaler": "robust", "gamma": 0.1, "beta": 0.1,
    }
    path = tmp_path / "trials.json"
    path.write_text(json.dumps([{
        "protocol": "fixed_external",
        "valid_mcc": 0.72,
        "config": config,
        "metrics": {},
    }]))
    study = optuna.create_study(direction="maximize")
    assert import_seed_trials(
        study,
        [path],
        "normal_tissue_878",
        "fixed_external",
        "plain",
        expected_repeats=3,
        max_warmup=50,
    ) == 1
    seeded = study.trials[0]
    assert seeded.value == 0.72
    assert seeded.intermediate_values == {}
    assert seeded.user_attrs["transfer_seed"] is True
