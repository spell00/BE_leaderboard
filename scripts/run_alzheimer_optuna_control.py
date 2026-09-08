#!/usr/bin/env python3
"""Classical Alzheimer-only Optuna control with a sealed fixed test."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hp_search
from scripts.run_optuna_comparison import _dataset_cv_settings, _fold_scores_payload
from src.evolutionary_meta import recommended_batch_size


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="BE_leaderboard_meta_evolution")
    parser.add_argument("--wandb-run-name", default="alzheimer-classical-optuna-control")
    return parser.parse_args(argv)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.n_repeats != 3:
        raise ValueError("Alzheimer control requires grouped CV=3")
    if args.n_trials < 1:
        raise ValueError("n_trials must be positive")
    if importlib.metadata.version("bernn") != "1.0.6":
        raise RuntimeError("This experiment requires bernn==1.0.6")

    dataset_id = "massbench_alzheimer"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "run_metadata.json"
    if metadata_path.exists() and not args.resume:
        raise FileExistsError("pass --resume")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({
        "arm": "alzheimer_classical_optuna_control",
        "candidate_budget": int(args.n_trials),
        "dataset": dataset_id,
        "selection_protocol": "development_cv_only",
        "cross_test": 1,
        "fixed_test_access_during_search": "monitoring_only_excluded_from_selection",
        "wandb_run_id": metadata.get("wandb_run_id") or uuid.uuid4().hex[:8],
    })
    atomic_json(metadata_path, metadata)

    import optuna

    storage = f"sqlite:///{(args.output_dir / 'optuna.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name="alzheimer_classical_control",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )
    if args.resume:
        for trial in study.get_trials(deepcopy=False):
            if trial.state == optuna.trial.TrialState.RUNNING:
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                print(f"[alzheimer-control] marked interrupted trial {trial.number} FAIL", flush=True)

    X, y, batches = hp_search.load_dataset(dataset_id)
    fixed_test = hp_search.load_fixed_test_dataset(dataset_id)
    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=metadata["wandb_run_id"],
            resume="allow",
            config=metadata,
        )

    complete = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    for candidate_index in range(len(complete), args.n_trials):
        trial = study.ask()
        run_args = hp_search.parse_args([])
        run_args.dataset = dataset_id
        run_args.n_epochs = args.n_epochs
        run_args.n_repeats, run_args.resolved_n_repeats = _dataset_cv_settings(
            dataset_id, args.n_repeats, batches
        )
        run_args.num_workers = args.num_workers
        run_args.device = args.device
        run_args.seed = args.seed
        run_args.no_wandb = True
        run_args.combine_test = False
        run_args.log1p = True
        run_args.max_warmup = max(1, min(50, args.n_epochs))
        run_args.bs = recommended_batch_size(batches, cap=args.batch_size)
        run_args.cv_split_cache = str(args.output_dir / "cv_splits" / f"{dataset_id}.npz")
        config = hp_search.sample_config(trial, run_args)
        config.update({
            "batch_size": int(run_args.bs),
            "cv_folds": int(run_args.resolved_n_repeats),
            "num_workers": int(args.num_workers),
            "lisi_enabled": False,
        })
        metrics = {}
        error = None
        try:
            score, metrics = hp_search.run_trial(
                config,
                run_args,
                (X, y, batches),
                f"alzheimer_classical_{metadata['wandb_run_id']}_t{trial.number}",
                fixed_test_data=fixed_test,
            )
        except Exception as exc:
            score = -1.0
            error = f"{type(exc).__name__}: {exc}"
        trial.set_user_attr("config", config)
        trial.set_user_attr("test_mcc", float(metrics.get("test_mcc", np.nan)))
        trial.set_user_attr("fold_scores", _fold_scores_payload(metrics))
        if error:
            trial.set_user_attr("error", error)
        study.tell(trial, float(score))
        record = {
            "candidate_index": candidate_index,
            "trial_number": trial.number,
            "valid_mcc": float(score),
            "test_mcc": float(metrics.get("test_mcc", np.nan)),
            "config": config,
            "fold_scores": _fold_scores_payload(metrics),
            "fixed_test_observed": True,
            "test_mcc_role": "monitoring_only_excluded_from_selection",
            "error": error,
        }
        with (args.output_dir / "trials.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        if wandb_run is not None:
            wandb_run.log({
                "candidate_index": candidate_index,
                "control/valid_mcc": float(score),
                "control/test_mcc": float(metrics.get("test_mcc", np.nan)),
                "control/best_valid_mcc": float(study.best_value),
            })

    recommendation = {
        "selection_protocol": "development_cv_only",
        "source_candidates": int(args.n_trials),
        "fixed_test_observed": True,
        "test_mcc_role": "monitoring_only_excluded_from_selection",
        "source_best_trial_number": int(study.best_trial.number),
        "source_best_valid_mcc": float(study.best_value),
        "target_configs": {dataset_id: study.best_trial.user_attrs["config"]},
    }
    atomic_json(args.output_dir / "final_recommendations.json", recommendation)
    if wandb_run is not None:
        wandb_run.summary["selection_protocol"] = "development_cv_only"
        wandb_run.summary["fixed_test_observed"] = True
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
