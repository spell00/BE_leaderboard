#!/usr/bin/env python3
"""Build the reusable BERNN meta-HPO trial bank.

This is the expensive Stage-0 control.  Five independent Optuna studies run in
round-robin order:

  * four source/meta-training datasets;
  * massbench_alzheimer as a target-specific Optuna baseline only.

The Alzheimer study is NEVER used to train a meta-model or to choose categorical
freezes.  Once this bank exists, meta-model/surrogate/RL experiments can restart
without rerunning the source Optuna studies.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts import hp_search
from src.dataset_splits import load_dataset_partitions
from src.meta_hpo_bank import (
    ALZHEIMER_BASELINE_ROLE,
    SOURCE_ROLE,
    BankTrial,
    canonical_config,
    write_bank,
)

# Preserve the CV settings from the user's current run_optuna_comparison.py.
DATASET_CV_FOLDS = {
    "normal_tissue_878": 3,
    "colon_3041": 3,
    "massbench_adenocarcinoma": 2,
    "massbench_benchmark": 3,
    "massbench_alzheimer": 3,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest", type=Path,
        default=ROOT / "config" / "evolution_development_datasets.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "meta_hpo_trial_bank")
    parser.add_argument("--n-trials", type=int, default=20, help="Independent Optuna trials per dataset.")
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=3, help="Fallback CV folds for datasets without an override.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-alzheimer-baseline", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="BE_leaderboard_meta_evolution")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args(argv)


def recommended_batch_size(batch_labels, cap: int = 32) -> int:
    labels = np.asarray(batch_labels).astype(str)
    _, counts = np.unique(labels, return_counts=True)
    smallest_training_fold = int(labels.size - counts.max()) if len(counts) > 1 else int(labels.size)
    return max(1, min(int(cap), max(1, smallest_training_fold // 2)))


def _dataset_cv_settings(dataset_id: str, default_folds: int, batches) -> tuple[int, int]:
    requested = int(DATASET_CV_FOLDS.get(dataset_id, default_folds))
    resolved = int(hp_search.resolve_n_repeats(requested, batches))
    return requested, resolved


def _completed(study):
    import optuna

    return [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def _mark_interrupted_failed(study) -> None:
    import optuna

    running = [
        trial for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.RUNNING
    ]
    for trial in running:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        print(f"[trial-bank] marked interrupted {study.study_name} trial {trial.number} FAIL", flush=True)


def _make_run_args(args, dataset_id: str, dataset_index: int, batches):
    run_args = hp_search.parse_args([])
    run_args.dataset = dataset_id
    run_args.n_epochs = int(args.n_epochs)
    run_args.n_repeats, run_args.resolved_n_repeats = _dataset_cv_settings(
        dataset_id, args.n_repeats, batches
    )
    run_args.num_workers = int(args.num_workers)
    run_args.device = args.device
    run_args.seed = int(args.seed) + int(dataset_index)
    run_args.no_wandb = True
    run_args.combine_test = False
    run_args.max_warmup = max(1, min(50, int(args.n_epochs)))
    run_args.log1p = True
    run_args.bs = recommended_batch_size(batches, cap=args.batch_size)
    run_args.results_dir = str(args.output_dir / "bernn" / dataset_id)
    run_args.cv_split_cache = str(args.output_dir / "cv_splits" / f"{dataset_id}.npz")
    return run_args


def _export_bank(studies, roles, output_dir: Path) -> list[BankTrial]:
    trials: list[BankTrial] = []
    for dataset_id, study in studies.items():
        complete = sorted(_completed(study), key=lambda trial: int(trial.number))
        for trial_index, trial in enumerate(complete):
            attrs = dict(trial.user_attrs)
            config = attrs.get("config")
            if not config:
                continue
            trials.append(BankTrial(
                dataset_id=dataset_id,
                role=roles[dataset_id],
                trial_index=trial_index,
                optuna_trial_number=int(trial.number),
                valid_mcc=float(trial.value),
                test_mcc=float(attrs.get("test_mcc", np.nan)),
                fit_seconds=float(attrs.get("fit_seconds", np.nan)),
                config=canonical_config(config),
                valid_mcc_folds=tuple(float(v) for v in attrs.get("valid_mcc_folds", ())),
                test_mcc_folds=tuple(float(v) for v in attrs.get("test_mcc_folds", ())),
                error=attrs.get("error"),
            ))
    write_bank(trials, output_dir)
    return trials


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    partitions = load_dataset_partitions(args.split_manifest)
    source_ids = tuple(partitions.train)
    if len(source_ids) != 4:
        print(f"[trial-bank] warning: expected 4 source datasets, manifest has {len(source_ids)}", flush=True)
    validation_ids = tuple(partitions.validation)
    if len(validation_ids) != 1:
        raise ValueError(f"Expected one validation dataset (Alzheimer), got {validation_ids}")
    alzheimer_id = validation_ids[0]

    roles = {name: SOURCE_ROLE for name in source_ids}
    dataset_ids = list(source_ids)
    if not args.no_alzheimer_baseline:
        roles[alzheimer_id] = ALZHEIMER_BASELINE_ROLE
        dataset_ids.append(alzheimer_id)

    metadata_path = args.output_dir / "run_metadata.json"
    if metadata_path.exists() and not args.resume:
        raise FileExistsError(f"{args.output_dir} already has a trial bank; use --resume")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {
            "created_at_unix": time.time(),
            "run_id": uuid.uuid4().hex[:10],
            "hostname": platform.node(),
        }
    metadata.update({
        "schema_version": 2,
        "arm": "independent_optuna_trial_bank",
        "source_datasets": list(source_ids),
        "alzheimer_baseline_dataset": None if args.no_alzheimer_baseline else alzheimer_id,
        "n_trials_per_dataset": int(args.n_trials),
        "n_epochs": int(args.n_epochs),
        "cv_overrides": DATASET_CV_FOLDS,
        "log1p": True,
        "alzheimer_role": "baseline_only_never_meta_training",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    datasets = {name: hp_search.load_dataset(name) for name in dataset_ids}
    fixed_tests = {name: hp_search.load_fixed_test_dataset(name) for name in dataset_ids}

    import mlflow
    import optuna

    tracking_dir = args.output_dir / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{(args.output_dir / 'optuna.sqlite3').resolve()}")
    studies = {
        name: optuna.create_study(
            study_name=f"stage0_{name}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
            storage=storage,
            load_if_exists=True,
        )
        for name in dataset_ids
    }
    if args.resume:
        for study in studies.values():
            _mark_interrupted_failed(study)

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=metadata.get("wandb_run_id") or metadata["run_id"],
            resume="allow",
            config={
                **vars(args),
                "source_datasets": list(source_ids),
                "alzheimer_baseline_dataset": None if args.no_alzheimer_baseline else alzheimer_id,
                "cv_overrides": DATASET_CV_FOLDS,
            },
        )
        metadata["wandb_run_id"] = wandb_run.id
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        wandb.define_metric("bank_step")
        wandb.define_metric("bank/*", step_metric="bank_step")

    try:
        # Round-robin: after bank_step N, every study has N+1 completed trials.
        for bank_step in range(int(args.n_trials)):
            step_payload = {"bank_step": bank_step}
            for dataset_index, dataset_id in enumerate(dataset_ids):
                study = studies[dataset_id]
                complete = _completed(study)
                if len(complete) <= bank_step:
                    trial = study.ask()
                    X, y, batches = datasets[dataset_id]
                    run_args = _make_run_args(args, dataset_id, dataset_index, batches)
                    config = hp_search.sample_config(trial, run_args)
                    # Trial-bank protocol fixes preprocessing; do not let this study
                    # spend samples rediscovering log1p.
                    config["log1p"] = True
                    started = time.monotonic()
                    metrics = {}
                    error = None
                    try:
                        score, metrics = hp_search.run_trial(
                            config,
                            run_args,
                            (X, y, batches),
                            f"stage0_{metadata['run_id']}_{dataset_id}_t{trial.number}",
                            fixed_test_data=fixed_tests[dataset_id],
                        )
                        score = float(score)
                    except Exception as exc:  # invalid configs remain part of HPO cost
                        score = -1.0
                        error = f"{type(exc).__name__}: {exc}"
                        print(f"[trial-bank] {dataset_id} trial {trial.number} failed: {error}", flush=True)
                    fit_seconds = time.monotonic() - started
                    trial.set_user_attr("config", canonical_config(config))
                    trial.set_user_attr("test_mcc", float(metrics.get("test_mcc", np.nan)))
                    trial.set_user_attr("valid_mcc_folds", [float(v) for v in metrics.get("valid_mcc_folds", [])])
                    trial.set_user_attr("test_mcc_folds", [float(v) for v in metrics.get("test_mcc_folds", [])])
                    trial.set_user_attr("fit_seconds", float(fit_seconds))
                    if error:
                        trial.set_user_attr("error", error)
                    study.tell(trial, score)
                    complete = _completed(study)

                current = sorted(complete, key=lambda t: int(t.number))[bank_step]
                best = max(complete[: bank_step + 1], key=lambda t: float(t.value))
                prefix = "source" if roles[dataset_id] == SOURCE_ROLE else "alzheimer_baseline"
                step_payload[f"bank/{prefix}/{dataset_id}/current_valid_mcc"] = float(current.value)
                step_payload[f"bank/{prefix}/{dataset_id}/best_valid_mcc"] = float(best.value)
                print(
                    f"[trial-bank] step={bank_step + 1}/{args.n_trials} {dataset_id}: "
                    f"current={float(current.value):.4f} best={float(best.value):.4f}",
                    flush=True,
                )
                _export_bank(studies, roles, args.output_dir)

            if wandb_run is not None:
                wandb_run.log(step_payload)

        trials = _export_bank(studies, roles, args.output_dir)
        summary = {
            "completed": {
                name: len([trial for trial in trials if trial.dataset_id == name])
                for name in dataset_ids
            },
            "best_valid_mcc": {
                name: max(
                    trial.valid_mcc for trial in trials
                    if trial.dataset_id == name
                )
                for name in dataset_ids
            },
        }
        (args.output_dir / "bank_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if wandb_run is not None:
            wandb_run.summary.update(summary["best_valid_mcc"])
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        try:
            storage.engine.dispose()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
