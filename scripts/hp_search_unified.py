#!/usr/bin/env python3
"""One conditional Optuna study spanning every BERNN architecture/head family."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hp_search as neural
from scripts import hp_search_head_sweep as frozen
from scripts.wandb_cv_logging import log_compact_epoch_charts, log_compact_fold_chart, set_numeric_summary
from src.zero_shot_recommender.data import append_trial
from src.zero_shot_recommender.schema import TrialRecord


REPRESENTATION_HEADS = {
    "knn": "knn",
    "prototype": "prototype_mean",
}
DLOSSES = ["no", "inverseTriplet", "revTriplet", "DANN", "normae"]


class PrefixedTrial:
    """Give each conditional branch an independent Optuna parameter namespace."""

    def __init__(self, trial, prefix: str):
        self._trial = trial
        self._prefix = prefix

    def __getattr__(self, name):
        return getattr(self._trial, name)

    @property
    def params(self):
        prefix = self._prefix
        return {
            key[len(prefix):]: value
            for key, value in self._trial.params.items() if key.startswith(prefix)
        }

    def suggest_float(self, name, *args, **kwargs):
        return self._trial.suggest_float(self._prefix + name, *args, **kwargs)

    def suggest_int(self, name, *args, **kwargs):
        return self._trial.suggest_int(self._prefix + name, *args, **kwargs)

    def suggest_categorical(self, name, choices):
        return self._trial.suggest_categorical(self._prefix + name, choices)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="massbench_adenocarcinoma")
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=30)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "representation_head_hpo")
    parser.add_argument(
        "--trainer-types", nargs="+",
        choices=["representation", "joint", "two_stage"],
        default=["representation", "joint"],
        help=(
            "Trainer families to search. two_stage is opt-in because the current "
            "BERNN holdout path frequently fails with sentinel/unlabelled test labels."
        ),
    )
    parser.add_argument("--dlosses", nargs="+", choices=DLOSSES, default=DLOSSES)
    return parser.parse_args(argv)


def _base_neural_args(args, batches):
    value = neural.parse_args([])
    value.dataset = args.dataset
    value.n_trials = args.n_trials
    value.n_epochs = args.n_epochs
    value.n_repeats = -1
    value.resolved_n_repeats = neural.resolve_n_repeats(-1, batches)
    value.device = args.device
    value.seed = args.seed
    value.no_wandb = True
    value.results_dir = str(args.output_dir)
    value.max_warmup = max(1, args.early_stop * 2)
    return value


def _write_live_tables(study, output: Path, dataset_id: str) -> None:
    """Materialize comparison tables without dataset-specific continuous HPs."""
    rows = []
    for item in study.trials:
        try:
            config = json.loads(item.user_attrs.get("resolved_config", "{}"))
        except (TypeError, ValueError):
            config = {}
        row = {
            "dataset_id": dataset_id,
            "trial": item.number,
            "state": item.state.name,
            "model_family": config.get("model_family"),
            "trainer_type": config.get("trainer_type", item.params.get("trainer_type")),
            "head_type": config.get("head_type", item.params.get("representation_head")),
            "kan": config.get("kan", item.params.get("kan", False)),
            "class_triplet": config.get("class_triplet", item.params.get("class_triplet")),
            "variational": config.get("variational", item.params.get("variational")),
            "dloss": config.get("dloss", item.params.get("dloss")),
            "valid_mcc": item.value,
        }
        for key, value in item.user_attrs.items():
            if not key.startswith("metric_"):
                continue
            try:
                row[key.removeprefix("metric_")] = float(value)
            except (TypeError, ValueError):
                pass
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "trial_metrics.csv", index=False)
    complete = table[(table["state"] == "COMPLETE") & table["valid_mcc"].notna()]
    if complete.empty:
        return
    dimensions = [
        "dataset_id", "model_family", "trainer_type", "head_type", "kan",
        "class_triplet", "variational", "dloss",
    ]
    numeric = [
        name for name in complete.select_dtypes(include=[np.number]).columns
        if name != "trial"
    ]
    summary = complete.groupby(dimensions, dropna=False)[numeric].agg(["count", "mean", "std", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "configuration_stats.csv", index=False)
    best_row = complete.loc[complete["valid_mcc"].idxmax()].to_dict()
    (output / "best_so_far.json").write_text(json.dumps(best_row, indent=2, default=str))


def main(argv=None):
    import mlflow
    import optuna
    from bernn.dl.train.train_ae_head_sweep import AEHeadSweepTrainer

    args = parse_args(argv)
    output = args.output_dir / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    frozen.HEAD_SWEEP_DIR = output / "representation"
    frozen.HEAD_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    neural.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(neural.MLRUNS_DIR.as_uri())

    train = frozen._load_train_fixed_test_dataset(args.dataset, include_test_labels=True)
    X_train, y_train, batches_train, names_train, X_test, y_test, batches_test, names_test = train
    neural_data = (X_train, y_train.to_numpy(), batches_train.to_numpy())
    neural_args = _base_neural_args(args, batches_train.to_numpy())
    ledger = output / "unified_trials.jsonl"

    sampler = optuna.samplers.TPESampler(
        seed=args.seed, multivariate=True, group=True,
        n_startup_trials=max(50, args.n_trials // 10),
    )
    storage = f"sqlite:///{(output / 'study.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name=f"unified_{args.dataset}", direction="maximize",
        sampler=sampler, storage=storage, load_if_exists=True,
    )

    def objective(trial):
        trainer_type = trial.suggest_categorical("trainer_type", list(args.trainer_types))
        dloss = trial.suggest_categorical("dloss", list(args.dlosses))
        variational = trial.suggest_categorical("variational", [False, True])
        class_triplet = trial.suggest_categorical("class_triplet", [False, True])
        class_triplet_w = (
            trial.suggest_float("class_triplet_w", 1e-4, 10.0, log=True)
            if class_triplet else 0.0
        )
        config = {
            "trainer_type": trainer_type, "dloss": dloss,
            "variational": variational, "class_triplet": class_triplet,
            "class_triplet_w": class_triplet_w, "transductive": trainer_type == "representation",
        }
        # Save branch identity before training so failed/interrupted trials remain
        # attributable in local and W&B tables.
        trial.set_user_attr("resolved_config", json.dumps(config, default=str))
        try:
            if trainer_type == "representation":
                public_head = trial.suggest_categorical("representation_head", list(REPRESENTATION_HEADS))
                head = REPRESENTATION_HEADS[public_head]
                branch_args = frozen.parse_args([])
                branch_args.dataset = args.dataset
                branch_args.n_epochs = args.n_epochs
                branch_args.early_stop = args.early_stop
                branch_args.device = args.device
                branch_args.seed = args.seed
                branch_args.n_cv = 5
                branch_args.head_types = [head]
                branch_args.class_triplet_w = class_triplet_w
                sweep_args = frozen._make_sweep_args(
                    args.dataset, dloss, variational, class_triplet, branch_args,
                    n_features=int(X_train.shape[1]),
                )
                trainer = frozen.BatchCVHeadSweepTrainer(
                    AEHeadSweepTrainer, sweep_args, str(ROOT / "data"),
                    X_train, y_train, batches_train, names_train,
                    X_test, y_test, batches_test, names_test, n_cv=5,
                )
                branch_trial = PrefixedTrial(trial, "representation__")
                config.update({"model_family": f"representation_{public_head}", "head_type": head, "kan": False})
                trial.set_user_attr("resolved_config", json.dumps(config, default=str))
                score = float(trainer.objective(branch_trial))
                config.update(branch_trial.params)
                trial.set_user_attr("resolved_config", json.dumps(config, default=str))
            else:
                kan = False
                branch_args = argparse.Namespace(**vars(neural_args))
                branch_args.model_type = trainer_type
                branch_args.dloss = dloss
                branch_args.variational = variational
                branch_args.kan = kan
                branch_args.class_triplet = class_triplet
                branch_args.class_triplet_w = class_triplet_w
                branch_trial = PrefixedTrial(trial, "neural_")
                sampled = neural.sample_config(branch_trial, branch_args)
                exp_id = f"unified_{args.dataset}_t{trial.number}"
                family = f"{trainer_type}_nn"
                config.update(sampled)
                config.update({"model_family": family, "head_type": "nn"})
                trial.set_user_attr("resolved_config", json.dumps(config, default=str))
                fixed_test_data = None
                if y_test is not None:
                    fixed_test_data = (X_test, y_test.to_numpy(), batches_test.to_numpy())
                score, metrics = neural.run_trial(
                    sampled, branch_args, neural_data, exp_id,
                    fixed_test_data=fixed_test_data,
                )
                for key, value in metrics.items():
                    trial.set_user_attr(f"metric_{key}", value)
            trial.set_user_attr("resolved_config", json.dumps(config, default=str))
            return score
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise

    def callback(current_study, trial):
        try:
            config = json.loads(trial.user_attrs.get("resolved_config", "{}"))
        except ValueError:
            config = {}
        complete = trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
        family = config.get("model_family", "failed")
        append_trial(ledger, TrialRecord(
            dataset_id=args.dataset, model_family=family,
            score=float(trial.value) if complete else -1.0,
            config=config or dict(trial.params),
            status="complete" if complete else "failed", seed=args.seed,
            metrics={key: value for key, value in trial.user_attrs.items() if key.startswith("metric_")},
            source="single_conditional_study",
        ))
        if os.getenv("WANDB_DISABLED", "").lower() not in {"1", "true", "yes"}:
            try:
                import wandb
                run = wandb.init(
                    project=os.getenv("WANDB_SUPERVISED_PROJECT", "BE_leaderboard_representation_then_head"),
                    entity=os.getenv("WANDB_ENTITY", "adlab"),
                    group=f"unified_{args.dataset}", name=f"trial_{trial.number}",
                    job_type="representation_then_head",
                    # Include all sampled hyperparameters and the dataset/family.
                    # Full trial record in W&B for search analysis.
                    config={**config, "dataset_id": args.dataset},
                    reinit="finish_previous",
                )
                payload = {"valid_mcc": trial.value} if complete else {"failed": 1}
                # Add all sampled hyperparameters to payload
                for key, value in config.items():
                    if key not in {"dataset_id"}:  # dataset_id goes in config, not payload
                        try:
                            payload[f"hp_{key}"] = float(value) if isinstance(value, (int, float)) else str(value)
                        except (TypeError, ValueError):
                            payload[f"hp_{key}"] = str(value)
                for key, value in trial.user_attrs.items():
                    if key.startswith("metric_"):
                        metric_key = key.removeprefix("metric_")
                        if metric_key.endswith("_folds"):
                            payload[metric_key] = value
                            continue
                        try:
                            payload[metric_key] = float(value)
                        except (TypeError, ValueError):
                            pass
                    elif key.endswith("_folds"):
                        # Representation/head-sweep branch attrs are not prefixed
                        # with metric_; keep fold vectors so cv_fold/{metric}
                        # charts are created.
                        payload[key] = value
                    elif key not in {"epoch_metrics_json", "resolved_config", "valid_fold_details", "test_predictions_json", "head_params", "all_head_params_json"}:
                        # Numeric representation/head-sweep summaries such as
                        # train_mcc, test_mcc, valid_mcc_std, batch_nbe, ...
                        try:
                            payload[key] = float(value)
                        except (TypeError, ValueError):
                            pass
                # Scalars belong in the run table/summary, not separate charts.
                set_numeric_summary(run, payload)
                log_compact_fold_chart(run, payload)

                # All averaged epoch metrics are bundled into at most four custom
                # charts. Raw fold/rep detail remains in MLflow and Optuna attrs.
                traces = trial.user_attrs.get("metric__epoch_traces_by_fold", {})
                log_compact_epoch_charts(run, traces)

                # Representation families expose AE epoch rows as one flat list;
                # regroup them by zero-based fold before using the same loggers.
                try:
                    representation_rows = json.loads(trial.user_attrs.get("epoch_metrics_json", "[]"))
                except (TypeError, ValueError):
                    representation_rows = []
                representation_by_fold = {}
                for row in representation_rows:
                    try:
                        fold = int(row.get("fold", 1)) - 1
                    except Exception:
                        fold = 0
                    representation_by_fold.setdefault(str(fold), []).append(row)
                log_compact_epoch_charts(run, representation_by_fold)
                run.summary["dataset_id"] = args.dataset
                run.summary["model_family"] = family
                if trial.user_attrs.get("error"):
                    run.summary["error"] = trial.user_attrs["error"]
                run.finish()
            except Exception as exc:
                print(f"[wandb] trial {trial.number} logging failed: {exc}", flush=True)
        study.trials_dataframe().to_csv(output / "trials_with_hparams.csv", index=False)
        _write_live_tables(current_study, output, args.dataset)
        print(
            f"[unified] trial={trial.number} state={trial.state.name} "
            f"family={family} score={trial.value} best={current_study.best_value if complete else 'unchanged'}",
            flush=True,
        )

    print(
        f"[unified] dataset={args.dataset} trials={args.n_trials} storage={storage}\n"
        f"[unified] conditional space: {' | '.join(args.trainer_types)} -> representation final {{prototype,knn}} sweep",
        flush=True,
    )
    study.optimize(
        objective, n_trials=args.n_trials, timeout=args.timeout,
        callbacks=[callback], gc_after_trial=True, catch=(Exception,),
    )
    (output / "best.json").write_text(json.dumps({
        "value": study.best_value, "params": study.best_trial.params,
        "attrs": study.best_trial.user_attrs,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
