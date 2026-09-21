#!/usr/bin/env python3
"""Synchronized four-dataset HPO with benchmark-selected direct meta-learning."""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import optuna
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import hp_search
from src.direct_meta_checkpoint import save_selected_meta_model
from src.meta_hpo_bank import BankTrial, config_feature_vector
from src.meta_hpo_models import (
    _decode_direct,
    normalize_source_meta,
    train_direct_meta_model,
)
from src.meta_hpo_utils import sample_bernn_config
from src.meta_leaderboard import update_best
from src.zero_shot_recommender.meta_features import (
    META_FEATURE_NAMES,
    extract_meta_features,
)


DATASETS = (
    "normal_tissue_878",
    "colon_3041",
    "massbench_adenocarcinoma",
    "massbench_benchmark",
)
META_TRAIN_DATASETS = DATASETS[:3]
META_VALIDATION_DATASET = DATASETS[3]
TARGET_DATASET = "massbench_alzheimer"

META_MODEL_CANDIDATES = (
    (32, 1e-3),
    (64, 3e-3),
    (128, 1e-2),
)
META_MODEL_EPOCHS = 200


class WandbHeartbeat:
    """Keep long BERNN fits visibly alive in W&B between round-level logs."""

    def __init__(self, run, interval_seconds):
        self.run = run
        self.interval_seconds = max(15.0, float(interval_seconds))
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.state = {
            "round": 0,
            "dataset": "initializing",
            "phase": "startup",
        }
        self.thread = threading.Thread(
            target=self._loop,
            name="wandb-heartbeat",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def update(self, *, round_number, dataset, phase):
        with self.state_lock:
            self.state = {
                "round": int(round_number),
                "dataset": str(dataset),
                "phase": str(phase),
            }

    def _loop(self):
        while not self.stop_event.wait(self.interval_seconds):
            with self.state_lock:
                state = dict(self.state)

            try:
                self.run.log(
                    {
                        "runtime/heartbeat_unix": time.time(),
                        "runtime/active_round": state["round"],
                        "runtime/active_dataset": state["dataset"],
                        "runtime/phase": state["phase"],
                    }
                )
            except Exception as exc:
                print(
                    f"[wandb-heartbeat] warning: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=min(5.0, self.interval_seconds))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")

    parser.add_argument(
        "--wandb-project",
        default="BE_leaderboard_meta_evolution",
    )
    parser.add_argument(
        "--wandb-run-name",
        default="synchronized-meta-hpo-benchmark-selected",
    )
    parser.add_argument(
        "--wandb-heartbeat-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument("--wandb-id", default=None)

    return parser.parse_args()


def trial_payload(trial):
    """Convert an Optuna trial into the compact structure used by this script."""
    attrs = dict(trial.user_attrs)

    return {
        "trial_number": int(trial.number),
        "valid_mcc": float(trial.value),
        "test_mcc": float(attrs.get("test_mcc", np.nan)),
        "fit_seconds": float(attrs.get("fit_seconds", np.nan)),
        "config": attrs["config"],
        "valid_mcc_folds": tuple(attrs.get("valid_mcc_folds", ())),
        "test_mcc_folds": tuple(attrs.get("test_mcc_folds", ())),
        "error": attrs.get("error"),
    }


def completed_trials(study):
    """Return completed Optuna trials that have an objective value."""
    return [
        trial
        for trial in study.trials
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
        )
    ]


def best_trial_payload(study):
    """Return the completed trial with the highest validation MCC."""
    return max(
        (trial_payload(trial) for trial in completed_trials(study)),
        key=lambda row: row["valid_mcc"],
    )


def all_dataset_telemetry(current, best):
    """Flatten current/best dataset metrics and hyperparameters for W&B."""
    telemetry = {}

    for dataset_id, current_trial in current.items():
        rows = (
            ("current", current_trial),
            ("best", best[dataset_id]),
        )

        for label, row in rows:
            prefix = f"dataset/{dataset_id}/{label}"

            for metric in ("valid_mcc", "test_mcc", "fit_seconds"):
                value = row.get(metric)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    telemetry[f"{prefix}/{metric}"] = float(value)

            for name, value in row.get("config", {}).items():
                key = f"{prefix}/hparams/{name}"

                if isinstance(value, (bool, int, float, np.integer, np.floating)):
                    telemetry[key] = float(value)
                elif isinstance(value, str):
                    telemetry[key] = value

    return telemetry


def load_datasets_and_meta_features():
    """Load all source/validation/target datasets and compute meta-features."""
    dataset_ids = DATASETS + (TARGET_DATASET,)
    data = {
        dataset_id: hp_search.load_dataset(dataset_id)
        for dataset_id in dataset_ids
    }

    meta = {}
    for dataset_id, (X, y, batches) in data.items():
        features = extract_meta_features(X, y, batches)
        meta[dataset_id] = np.asarray(
            [features[name] for name in META_FEATURE_NAMES],
            dtype=np.float32,
        )

    return data, meta


def create_studies(output_dir, seed):
    """Create or reopen one synchronized Optuna study per source dataset."""
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{(output_dir / 'optuna.sqlite3').resolve()}"
    )

    return {
        dataset_id: optuna.create_study(
            study_name=f"sync_{dataset_id}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            storage=storage,
            load_if_exists=True,
        )
        for dataset_id in DATASETS
    }


def make_run_args(base_hp_args, args, dataset_id, seed, batch_count):
    """Build the hp_search argument namespace for one dataset run."""
    run = argparse.Namespace(**vars(base_hp_args))
    run.dataset = dataset_id
    run.seed = seed
    run.bs = min(args.batch_size, max(1, batch_count))
    run.results_dir = str(args.output_dir / dataset_id)
    run.cv_split_cache = str(
        args.output_dir / "cv_splits" / f"{dataset_id}.npz"
    )
    run.resolved_n_repeats = hp_search.resolve_n_repeats(
        run.n_repeats,
        batch_count,
    )
    return run


def execute_dataset_trial(
    *,
    study,
    dataset_id,
    step,
    dataset_index,
    data,
    base_hp_args,
    args,
):
    """Run one HPO trial for one dataset and record its Optuna metadata."""
    trial = study.ask()
    X, y, batches = data[dataset_id]

    run = make_run_args(
        base_hp_args=base_hp_args,
        args=args,
        dataset_id=dataset_id,
        seed=args.seed + step * 10 + dataset_index,
        batch_count=batches,
    )

    config = sample_bernn_config(trial, run, {})
    started = time.monotonic()
    error = None
    metrics = {}

    try:
        score, metrics = hp_search.run_trial(
            config,
            run,
            (X, y, batches),
            f"sync_meta_{step}_{dataset_id}",
            fixed_test_data=hp_search.load_fixed_test_dataset(dataset_id),
        )
    except Exception as exc:
        score = -1.0
        error = f"{type(exc).__name__}: {exc}"

    trial.set_user_attr("config", config)
    trial.set_user_attr(
        "test_mcc",
        float(metrics.get("test_mcc", np.nan)),
    )
    trial.set_user_attr(
        "fit_seconds",
        time.monotonic() - started,
    )
    trial.set_user_attr("error", error)

    study.tell(trial, float(score))


def select_meta_model(
    *,
    step,
    args,
    best,
    meta,
):
    """Train candidate direct meta-models and select using benchmark error."""
    source_meta, mean, scale = normalize_source_meta(
        meta,
        META_TRAIN_DATASETS,
    )

    normalized_meta = {
        dataset_id: (meta[dataset_id] - mean) / scale
        for dataset_id in (META_VALIDATION_DATASET, TARGET_DATASET)
    }

    max_warmup = max(1, min(50, args.n_epochs))

    source_bank = {
        dataset_id: BankTrial(
            dataset_id=dataset_id,
            role="source",
            trial_index=step,
            optuna_trial_number=best[dataset_id]["trial_number"],
            valid_mcc=best[dataset_id]["valid_mcc"],
            test_mcc=best[dataset_id]["test_mcc"],
            fit_seconds=best[dataset_id]["fit_seconds"],
            config=best[dataset_id]["config"],
        )
        for dataset_id in META_TRAIN_DATASETS
    }

    candidates = []

    for hidden_size, learning_rate in META_MODEL_CANDIDATES:
        model, history, diagnostics = train_direct_meta_model(
            source_bank,
            source_meta,
            meta[TARGET_DATASET],
            max_warmup=max_warmup,
            hidden_size=hidden_size,
            epochs=META_MODEL_EPOCHS,
            lr=learning_rate,
            seed=args.seed + step,
        )

        benchmark_tensor = torch.tensor(
            normalized_meta[META_VALIDATION_DATASET][None, :],
            dtype=torch.float32,
        )

        with torch.no_grad():
            predicted_config = _decode_direct(
                model.forward(benchmark_tensor),
                0,
                max_warmup,
                {},
            )

        predicted_vector = config_feature_vector(
            predicted_config,
            max_warmup=max_warmup,
        )
        reference_vector = config_feature_vector(
            best[META_VALIDATION_DATASET]["config"],
            max_warmup=max_warmup,
        )

        prediction_error = float(
            np.linalg.norm(predicted_vector - reference_vector)
        )

        candidates.append(
            (
                prediction_error,
                hidden_size,
                learning_rate,
                model,
                diagnostics,
            )
        )

    (
        validation_error,
        hidden_size,
        learning_rate,
        model,
        diagnostics,
    ) = min(candidates, key=lambda item: item[0])

    model.eval()

    target_tensor = torch.tensor(
        normalized_meta[TARGET_DATASET][None, :],
        dtype=torch.float32,
    )

    with torch.no_grad():
        target_config = _decode_direct(
            model.forward(target_tensor),
            0,
            max_warmup,
            {},
        )

    return {
        "model": model,
        "diagnostics": diagnostics,
        "validation_error": validation_error,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "target_config": target_config,
        "meta_mean": mean,
        "meta_scale": scale,
        "max_warmup": max_warmup,
    }


def evaluate_target(
    *,
    step,
    args,
    base_hp_args,
    data,
    target_config,
):
    """Evaluate the selected meta-predicted configuration on Alzheimer data."""
    X, y, batches = data[TARGET_DATASET]

    run = make_run_args(
        base_hp_args=base_hp_args,
        args=args,
        dataset_id=TARGET_DATASET,
        seed=args.seed + 10_000 + step,
        batch_count=batches,
    )
    run.results_dir = str(args.output_dir / "alzheimer")
    run.cv_split_cache = str(
        args.output_dir / "cv_splits" / "massbench_alzheimer.npz"
    )

    try:
        score, metrics = hp_search.run_trial(
            target_config,
            run,
            (X, y, batches),
            f"sync_meta_alzheimer_{step}",
            fixed_test_data=hp_search.load_fixed_test_dataset(TARGET_DATASET),
        )
    except Exception as exc:
        score = -1.0
        metrics = {"error": f"{type(exc).__name__}: {exc}"}

    return float(score), metrics


def append_jsonl(path, record):
    """Append one durable JSON line to the round ledger."""
    with path.open("a") as file:
        file.write(json.dumps(record, default=str) + "\n")
        file.flush()
        os.fsync(file.fileno())


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, meta = load_datasets_and_meta_features()
    studies = create_studies(args.output_dir, args.seed)

    ledger_path = args.output_dir / "rounds.jsonl"
    old_records = (
        [json.loads(line) for line in ledger_path.open()]
        if args.resume and ledger_path.exists()
        else []
    )
    completed_rounds = {
        int(record["round"])
        for record in old_records
    }

    base_hp_args = hp_search.parse_args([])
    base_hp_args.n_epochs = args.n_epochs
    base_hp_args.n_repeats = args.n_repeats
    base_hp_args.num_workers = args.num_workers
    base_hp_args.device = args.device

    import wandb

    wandb_run = (
        None
        if args.no_wandb
        else wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_id,
            resume="allow" if args.wandb_id else None,
            config={
                "protocol": (
                    "synchronized_four_dataset_hpo_"
                    "benchmark_meta_selection"
                ),
                "train_datasets": DATASETS,
                "meta_train_datasets": META_TRAIN_DATASETS,
                "meta_validation_dataset": META_VALIDATION_DATASET,
                "target_dataset": TARGET_DATASET,
                "n_trials": args.n_trials,
                "wandb_heartbeat_seconds": args.wandb_heartbeat_seconds,
            },
        )
    )

    heartbeat = (
        None
        if wandb_run is None
        else WandbHeartbeat(wandb_run, args.wandb_heartbeat_seconds)
    )

    if heartbeat:
        heartbeat.start()

    try:
        for step in range(args.n_trials):
            current = {}

            # --------------------------------------------------------------
            # 1. Run one synchronized HPO trial on each source/benchmark set
            # --------------------------------------------------------------
            for dataset_index, dataset_id in enumerate(DATASETS):
                if heartbeat:
                    heartbeat.update(
                        round_number=step + 1,
                        dataset=dataset_id,
                        phase="dataset_hpo",
                    )

                complete = completed_trials(studies[dataset_id])

                if len(complete) <= step:
                    execute_dataset_trial(
                        study=studies[dataset_id],
                        dataset_id=dataset_id,
                        step=step,
                        dataset_index=dataset_index,
                        data=data,
                        base_hp_args=base_hp_args,
                        args=args,
                    )
                    complete = completed_trials(studies[dataset_id])

                current[dataset_id] = trial_payload(complete[step])

            if step in completed_rounds:
                continue

            # --------------------------------------------------------------
            # 2. Determine the best HPO configuration seen per dataset
            # --------------------------------------------------------------
            best = {
                dataset_id: best_trial_payload(studies[dataset_id])
                for dataset_id in DATASETS
            }

            if heartbeat:
                heartbeat.update(
                    round_number=step + 1,
                    dataset=META_VALIDATION_DATASET,
                    phase="meta_model_selection",
                )

            dataset_telemetry = all_dataset_telemetry(current, best)

            # --------------------------------------------------------------
            # 3. Select a direct meta-model using benchmark prediction error
            # --------------------------------------------------------------
            meta_selection = select_meta_model(
                step=step,
                args=args,
                best=best,
                meta=meta,
            )

            model = meta_selection["model"]
            diagnostics = meta_selection["diagnostics"]
            validation_error = meta_selection["validation_error"]
            hidden_size = meta_selection["hidden_size"]
            learning_rate = meta_selection["learning_rate"]
            target_config = meta_selection["target_config"]

            checkpoint_metadata = {
                "round": step,
                "run_name": args.wandb_run_name,
                "training_seed": args.seed + step,
                "meta_hidden_size": hidden_size,
                "meta_lr": learning_rate,
                "meta_epochs": META_MODEL_EPOCHS,
                "benchmark_prediction_error": float(validation_error),
                "best_source": best,
                "alzheimer_config": target_config,
                "training_diagnostics": diagnostics,
                "raw_meta_features": {
                    dataset_id: values.tolist()
                    for dataset_id, values in meta.items()
                },
                "run_args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "reconstructed": False,
            }

            checkpoint_kwargs = {
                "meta_mean": meta_selection["meta_mean"],
                "meta_scale": meta_selection["meta_scale"],
                "max_warmup": meta_selection["max_warmup"],
                "metadata": checkpoint_metadata,
            }

            checkpoint_path = save_selected_meta_model(
                args.output_dir,
                model,
                **checkpoint_kwargs,
            )

            if wandb_run:
                wandb_run.save(
                    str(checkpoint_path.resolve()),
                    base_path=str(args.output_dir.resolve()),
                    policy="now",
                )

            # --------------------------------------------------------------
            # 4. Evaluate selected configuration on the Alzheimer target
            # --------------------------------------------------------------
            if heartbeat:
                heartbeat.update(
                    round_number=step + 1,
                    dataset=TARGET_DATASET,
                    phase="target_evaluation",
                )

            target_score, target_metrics = evaluate_target(
                step=step,
                args=args,
                base_hp_args=base_hp_args,
                data=data,
                target_config=target_config,
            )

            checkpoint_metadata.update(
                alzheimer_valid_mcc=target_score,
                alzheimer_metrics=target_metrics,
            )

            save_selected_meta_model(
                args.output_dir,
                model,
                **checkpoint_kwargs,
            )

            if wandb_run:
                checkpoint_dir = args.output_dir / "meta_checkpoints"

                for saved_path in checkpoint_dir.glob("*.pt"):
                    if (
                        saved_path == checkpoint_path
                        or saved_path.name.startswith("best_")
                    ):
                        wandb_run.save(
                            str(saved_path.resolve()),
                            base_path=str(args.output_dir.resolve()),
                            policy="now",
                        )

            # --------------------------------------------------------------
            # 5. Update global target leaderboard and round ledger
            # --------------------------------------------------------------
            universal_best, universal_is_best = update_best(
                args.output_dir.parent / "best_alzheimer.json",
                score=target_score,
                strategy="synchronized_meta_hpo",
                trial=step + 1,
                config=target_config,
                run_name=args.wandb_run_name,
                output_dir=str(args.output_dir),
                extra={
                    "benchmark_hparam_error": float(validation_error),
                },
            )

            previous_best = max(
                (
                    float(record.get("alzheimer_valid_mcc", -1))
                    for record in old_records
                ),
                default=-1,
            )
            best_alzheimer_score = max(previous_best, target_score)

            record = {
                "round": step,
                "source_trials": [
                    current[dataset_id]
                    for dataset_id in DATASETS
                ],
                "best_source": best,
                "meta_hidden_size": hidden_size,
                "meta_lr": learning_rate,
                "benchmark_prediction_error": validation_error,
                "benchmark_reference_config": (
                    best[META_VALIDATION_DATASET]["config"]
                ),
                "alzheimer_config": target_config,
                "alzheimer_valid_mcc": target_score,
                "alzheimer_metrics": target_metrics,
                "best_alzheimer_valid_mcc": best_alzheimer_score,
                "leaderboard/best_alzheimer_valid_mcc": float(
                    universal_best["score"]
                ),
                "leaderboard/is_current_best": int(universal_is_best),
                "universal_best_alzheimer": universal_best,
                "meta_checkpoint": str(
                    checkpoint_path.relative_to(args.output_dir)
                ),
            }

            append_jsonl(ledger_path, record)

            if wandb_run:
                numeric_target_config = {
                    f"alzheimer/{key}": value
                    for key, value in target_config.items()
                    if isinstance(value, (int, float, bool))
                }

                wandb_run.log(
                    {
                        "round": step,
                        "benchmark_hparam_error": validation_error,
                        "alzheimer_valid_mcc": target_score,
                        "best_alzheimer_valid_mcc": best_alzheimer_score,
                        "leaderboard/best_alzheimer_valid_mcc": float(
                            universal_best["score"]
                        ),
                        "leaderboard/is_current_best": int(
                            universal_is_best
                        ),
                        "meta_hidden_size": hidden_size,
                        "meta_lr": learning_rate,
                        **dataset_telemetry,
                        **numeric_target_config,
                    }
                )

            print(
                f"[sync-meta] round={step + 1}/{args.n_trials} "
                f"benchmark_error={validation_error:.4f} "
                f"Alzheimer={target_score:.4f}",
                flush=True,
            )

            old_records.append(record)

    finally:
        if heartbeat:
            heartbeat.stop()

        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
