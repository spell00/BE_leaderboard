from scripts.run_independent_optuna_all_datasets import (
    DATASETS,
    selected_datasets,
    wandb_metrics,
)


def test_default_dataset_order_puts_new_benchmarks_first():
    assert [dataset for dataset, _ in DATASETS[:4]] == [
        "jdlber_sle_maldi",
        "seqc_maqc",
        "bacteria_2024_mz10",
        "scib_pancreas",
    ]
    assert len(DATASETS) == 9
    assert len({dataset for dataset, _ in DATASETS}) == 9


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
