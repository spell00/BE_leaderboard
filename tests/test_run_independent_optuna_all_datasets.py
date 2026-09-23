from scripts.run_independent_optuna_all_datasets import (
    DATASETS,
    attempted,
    failed_trials,
    persist_multifidelity_trials,
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



def test_failed_trials_do_not_consume_requested_search_budget(tmp_path):
    import json
    import optuna

    study = optuna.create_study(direction="maximize")

    complete = study.ask()
    complete.set_user_attr("config", {"lr": 1e-3})
    complete.set_user_attr("budget_step", 100)
    study.tell(complete, 0.7)

    pruned = study.ask()
    pruned.set_user_attr("config", {"lr": 2e-3})
    pruned.set_user_attr("partial_total", 0.4)
    pruned.set_user_attr("budget_step", 20)
    study.tell(pruned, state=optuna.trial.TrialState.PRUNED)

    failed = study.ask()
    failed.set_user_attr("config", {"lr": 3e-3})
    failed.set_user_attr("error", "RuntimeError: boom")
    failed.set_user_attr("budget_step", 10)
    failed.set_user_attr("partial_total", 0.2)
    study.tell(failed, state=optuna.trial.TrialState.FAIL)

    assert len(attempted(study)) == 2
    assert len(failed_trials(study)) == 1

    persist_multifidelity_trials(tmp_path, study, max_resource=100)
    rows = json.loads((tmp_path / "multifidelity_trials.json").read_text())
    by_state = {row["state"]: row for row in rows}
    assert by_state["COMPLETE"]["total_source"] == "observed"
    assert by_state["PRUNED"]["total_source"] == "partial_proxy"
    assert by_state["FAIL"]["total_source"] == "failed"
    assert by_state["FAIL"]["total"] is None
