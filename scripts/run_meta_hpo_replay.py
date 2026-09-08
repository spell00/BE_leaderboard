#!/usr/bin/env python3
"""Replay a frozen BERNN trial bank into direct-meta, surrogate, and RL scenarios.

The four source Optuna studies are never rerun.  Alzheimer Optuna trials in the
bank are baseline-only and are never passed into a model fit.

Real Alzheimer BERNN evaluation is intentionally sparse:
  * direct meta-model: selected meta-training epochs;
  * surrogate: once after convergence/search per source prefix;
  * RL: selected policy-training epochs.

Every cheap prediction can still be logged every epoch, so overfitting dynamics
can be inspected without paying for a BERNN fit at every meta epoch.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts import hp_search
from src.dataset_splits import load_dataset_partitions
from src.meta_hpo_bank import (
    SOURCE_ROLE,
    BankTrial,
    alzheimer_baseline_curve,
    apply_fixed_categories,
    best_by_dataset,
    categorical_consensus,
    load_bank,
    load_fixed_categories,
    protocol_digest,
    source_dataset_ids,
    trials_at_prefix,
)
from src.meta_hpo_models import fit_extra_trees_surrogate, fit_mlp_surrogate, leave_one_dataset_out_surrogate_rmse, optimize_surrogate_tpe, optimize_surrogate_evolution, train_direct_meta_model, train_reinforce_policy
from src.meta_leaderboard import update_best
from src.zero_shot_recommender.meta_features import META_FEATURE_NAMES, extract_meta_features

DATASET_CV_FOLDS = {
    "normal_tissue_878": 3,
    "colon_3041": 3,
    "massbench_adenocarcinoma": 2,
    "massbench_benchmark": 3,
    "massbench_alzheimer": 3,
}

SCENARIOS = {
    "direct",
    "surrogate_extra_trees",
    "surrogate_mlp",
    "surrogate_evolution",
    "meta_rl",
    "target_rl",
}


def _int_list(text: str) -> list[int]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return sorted(set(values))


def _str_list(text: str) -> list[str]:
    return [token.strip() for token in str(text).split(",") if token.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-bank", type=Path, required=True)
    parser.add_argument(
        "--split-manifest", type=Path,
        default=ROOT / "config" / "evolution_development_datasets.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "meta_hpo_replay")
    parser.add_argument("--scenarios", default="direct,surrogate_extra_trees,surrogate_mlp,surrogate_evolution,meta_rl,target_rl")
    parser.add_argument("--source-prefixes", default="1,2,3,5,10,15,20")

    parser.add_argument("--categorical-consensus", type=Path, default=None)
    parser.add_argument("--freeze-policy", choices=("none", "strict", "robust"), default="none")
    parser.add_argument(
        "--consensus-scope", choices=("full", "prefix"), default="full",
        help="full uses the completed Stage-0 bank; prefix recomputes consensus using only the first N trials/dataset.",
    )
    parser.add_argument("--consensus-top-k", type=int, default=5)
    parser.add_argument("--consensus-min-support", type=float, default=0.80)

    parser.add_argument("--n-epochs", type=int, default=1000, help="BERNN epochs for real Alzheimer checks.")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--meta-hidden-size", type=int, default=64)
    parser.add_argument("--meta-epochs", type=int, default=1000)
    parser.add_argument("--meta-lr", type=float, default=1e-2)
    parser.add_argument("--direct-eval-epochs", default="10,50,200,500,1000")

    parser.add_argument("--surrogate-search-trials", type=int, default=3000)
    parser.add_argument("--surrogate-risk-penalty", type=float, default=0.25)
    parser.add_argument("--surrogate-mlp-epochs", type=int, default=1000)
    parser.add_argument("--surrogate-ensemble-size", type=int, default=5)
    parser.add_argument("--skip-source-lodo", action="store_true")
    parser.add_argument("--evolution-population-size", type=int, default=64)
    parser.add_argument("--evolution-generations", type=int, default=100)
    parser.add_argument("--evolution-eval-generations", default="25,50,100")

    parser.add_argument("--rl-surrogate", choices=("extra_trees", "mlp"), default="extra_trees")
    parser.add_argument("--rl-epochs", type=int, default=1000)
    parser.add_argument("--rl-eval-epochs", default="200,1000")
    parser.add_argument("--rl-batch-size", type=int, default=32)
    parser.add_argument("--rl-hidden-size", type=int, default=96)
    parser.add_argument("--rl-lr", type=float, default=3e-3)
    parser.add_argument("--rl-entropy", type=float, default=0.01)

    parser.add_argument("--dry-run-validation", action="store_true",
                        help="Train cheap meta models but do not launch real Alzheimer BERNN checks.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="BE_leaderboard_meta_evolution")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args(argv)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _meta_vector(dataset) -> np.ndarray:
    X, y, batches = dataset
    values = extract_meta_features(X, y, batches)
    return np.asarray([values[name] for name in META_FEATURE_NAMES], dtype=np.float32)


def _recommended_batch_size(batch_labels, cap: int = 32) -> int:
    labels = np.asarray(batch_labels).astype(str)
    _, counts = np.unique(labels, return_counts=True)
    smallest_training_fold = int(labels.size - counts.max()) if len(counts) > 1 else int(labels.size)
    return max(1, min(int(cap), max(1, smallest_training_fold // 2)))


def _validation_cache(path: Path) -> dict[str, dict[str, Any]]:
    return {row["cache_key"]: row for row in _load_jsonl(path) if "cache_key" in row}


def _evaluate_alzheimer(
    config: dict[str, Any],
    *,
    scenario: str,
    prefix: int,
    checkpoint_kind: str,
    checkpoint: int,
    args,
    alzheimer_id: str,
    alzheimer_data,
    alzheimer_fixed_test,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    requested_folds = int(DATASET_CV_FOLDS.get(alzheimer_id, args.n_repeats))
    _, _, batches = alzheimer_data
    resolved_folds = int(hp_search.resolve_n_repeats(requested_folds, batches))
    protocol = {
        "dataset": alzheimer_id,
        "n_epochs": int(args.n_epochs),
        "n_repeats": requested_folds,
        "resolved_n_repeats": resolved_folds,
        "batch_size_cap": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed) + 10000,
        "log1p": True,
    }
    key = protocol_digest(config, protocol)
    if key in cache:
        cached = dict(cache[key])
        cached["cache_reused"] = True
        return cached

    if args.dry_run_validation:
        return {
            "cache_key": key,
            "cache_reused": False,
            "actual_valid_mcc": float("nan"),
            "actual_test_mcc": float("nan"),
            "actual_valid_mcc_folds": [],
            "actual_test_mcc_folds": [],
            "config": config,
            "protocol": protocol,
            "dry_run": True,
        }

    run_args = hp_search.parse_args([])
    run_args.dataset = alzheimer_id
    run_args.n_epochs = int(args.n_epochs)
    run_args.n_repeats = requested_folds
    run_args.resolved_n_repeats = resolved_folds
    run_args.num_workers = int(args.num_workers)
    run_args.device = args.device
    run_args.seed = int(args.seed) + 10000
    run_args.no_wandb = True
    run_args.combine_test = False
    run_args.max_warmup = max(1, min(50, int(args.n_epochs)))
    run_args.log1p = True
    run_args.bs = _recommended_batch_size(batches, cap=args.batch_size)
    run_args.results_dir = str(args.output_dir / "alzheimer_bernn" / scenario)
    run_args.cv_split_cache = str(args.output_dir / "cv_splits" / f"{alzheimer_id}.npz")
    started = time.monotonic()
    metrics = {}
    error = None
    try:
        valid_mcc, metrics = hp_search.run_trial(
            config,
            run_args,
            alzheimer_data,
            f"meta_replay_{scenario}_p{prefix}_{checkpoint_kind}{checkpoint}_{key[:10]}",
            fixed_test_data=alzheimer_fixed_test,
        )
        valid_mcc = float(valid_mcc)
    except Exception as exc:
        valid_mcc = -1.0
        error = f"{type(exc).__name__}: {exc}"
        print(f"[meta-replay] Alzheimer evaluation failed: {error}", flush=True)
    row = {
        "cache_key": key,
        "cache_reused": False,
        "actual_valid_mcc": valid_mcc,
        "actual_test_mcc": float(metrics.get("test_mcc", np.nan)),
        "actual_valid_mcc_folds": list(metrics.get("valid_mcc_folds", [])),
        "actual_test_mcc_folds": list(metrics.get("test_mcc_folds", [])),
        "fit_seconds": time.monotonic() - started,
        "config": config,
        "protocol": protocol,
        "error": error,
        "dry_run": False,
    }
    _append_jsonl(args.output_dir / "alzheimer_validation_cache.jsonl", row)
    cache[key] = row
    return row


def _scenario_result_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("scenario"), int(row.get("source_prefix", -1)),
        row.get("checkpoint_kind"), int(row.get("checkpoint", -1)),
    )


def _fixed_for_prefix(args, trials, source_ids, prefix: int, full_fixed: dict[str, Any]) -> dict[str, Any]:
    if args.freeze_policy == "none":
        return {}
    if args.consensus_scope == "full":
        return dict(full_fixed)
    payload = categorical_consensus(
        trials,
        source_ids,
        prefix=prefix,
        top_k=args.consensus_top_k,
        min_support=args.consensus_min_support,
    )
    return dict(payload[f"{args.freeze_policy}_fixed"])


def _adjust_best(best: dict[str, BankTrial], fixed: dict[str, Any]) -> dict[str, BankTrial]:
    if not fixed:
        return best
    return {
        name: replace(trial, config=apply_fixed_categories(trial.config, fixed))
        for name, trial in best.items()
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    scenarios = _str_list(args.scenarios)
    unknown = sorted(set(scenarios) - SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}; valid={sorted(SCENARIOS)}")
    direct_eval_epochs = _int_list(args.direct_eval_epochs)
    rl_eval_epochs = _int_list(args.rl_eval_epochs)
    evolution_eval_generations = _int_list(args.evolution_eval_generations)

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"{args.output_dir} is not empty; use --resume or a fresh directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trials = load_bank(args.trial_bank)
    source_ids = source_dataset_ids(trials)
    if len(source_ids) != 4:
        raise ValueError(f"This experiment expects four source datasets, got {source_ids}")
    max_prefix = min(
        len([trial for trial in trials if trial.role == SOURCE_ROLE and trial.dataset_id == name])
        for name in source_ids
    )
    prefixes = [value for value in _int_list(args.source_prefixes) if 1 <= value <= max_prefix]
    if not prefixes:
        raise ValueError(f"No valid source prefixes; bank supports 1..{max_prefix}")

    partitions = load_dataset_partitions(args.split_manifest)
    validation_ids = tuple(partitions.validation)
    if len(validation_ids) != 1:
        raise ValueError(f"Expected one validation dataset, got {validation_ids}")
    alzheimer_id = validation_ids[0]

    source_datasets = {name: hp_search.load_dataset(name) for name in source_ids}
    source_meta = {name: _meta_vector(source_datasets[name]) for name in source_ids}
    alzheimer_data = hp_search.load_dataset(alzheimer_id)
    alzheimer_meta = _meta_vector(alzheimer_data)
    alzheimer_fixed_test = hp_search.load_fixed_test_dataset(alzheimer_id)
    max_warmup = max(1, min(50, int(args.n_epochs)))
    hp_args = SimpleNamespace(n_epochs=int(args.n_epochs), max_warmup=max_warmup)

    full_fixed = {}
    if args.freeze_policy != "none":
        if args.categorical_consensus is not None and args.consensus_scope == "full":
            full_fixed = load_fixed_categories(args.categorical_consensus, policy=args.freeze_policy)
        elif args.consensus_scope == "full":
            payload = categorical_consensus(
                trials, source_ids, prefix=None,
                top_k=args.consensus_top_k,
                min_support=args.consensus_min_support,
            )
            full_fixed = dict(payload[f"{args.freeze_policy}_fixed"])

    metadata = {
        "schema_version": 2,
        "trial_bank": str(args.trial_bank),
        "source_datasets": list(source_ids),
        "validation_dataset": alzheimer_id,
        "source_prefixes": prefixes,
        "scenarios": scenarios,
        "freeze_policy": args.freeze_policy,
        "consensus_scope": args.consensus_scope,
        "full_fixed_categories": full_fixed,
        "information_budget_note": (
            "full consensus consumes the completed Stage-0 source bank before replay"
            if args.freeze_policy != "none" and args.consensus_scope == "full"
            else "each source prefix uses no later source-trial categorical information"
        ),
        "alzheimer_optuna_baseline_curve": alzheimer_baseline_curve(trials),
        "alzheimer_scores_used_for_training": False,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    results_path = args.output_dir / "scenario_results.jsonl"
    prediction_path = args.output_dir / "cheap_predictions.jsonl"
    existing_results = _load_jsonl(results_path)
    completed_keys = {_scenario_result_key(row) for row in existing_results}
    validation_cache = _validation_cache(args.output_dir / "alzheimer_validation_cache.jsonl")

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), **metadata},
        )

    validation_counter = 0

    def persist_actual(
        *, scenario: str, prefix: int, checkpoint_kind: str, checkpoint: int,
        config: dict[str, Any], predicted_mcc=None, predicted_std=None,
        extra: dict[str, Any] | None = None,
    ):
        nonlocal validation_counter
        result_key = (scenario, int(prefix), checkpoint_kind, int(checkpoint))
        if result_key in completed_keys:
            return
        actual = _evaluate_alzheimer(
            config,
            scenario=scenario,
            prefix=prefix,
            checkpoint_kind=checkpoint_kind,
            checkpoint=checkpoint,
            args=args,
            alzheimer_id=alzheimer_id,
            alzheimer_data=alzheimer_data,
            alzheimer_fixed_test=alzheimer_fixed_test,
            cache=validation_cache,
        )
        row = {
            "scenario": scenario,
            "source_prefix": int(prefix),
            "checkpoint_kind": checkpoint_kind,
            "checkpoint": int(checkpoint),
            "fixed_categories": current_fixed,
            "predicted_mcc": None if predicted_mcc is None else float(predicted_mcc),
            "predicted_std": None if predicted_std is None else float(predicted_std),
            "actual_valid_mcc": float(actual["actual_valid_mcc"]),
            "actual_test_mcc": float(actual["actual_test_mcc"]),
            "surrogate_signed_error": (
                None if predicted_mcc is None or not np.isfinite(actual["actual_valid_mcc"])
                else float(predicted_mcc) - float(actual["actual_valid_mcc"])
            ),
            "config": config,
            "cache_key": actual["cache_key"],
            "cache_reused": bool(actual.get("cache_reused", False)),
            "extra": extra or {},
        }
        universal_best, universal_is_best = update_best(args.output_dir.parent / "best_alzheimer.json", score=float(row["actual_valid_mcc"]), strategy=f"meta_replay:{scenario}", trial=validation_counter + 1, config=config, run_name=args.wandb_run_name, output_dir=str(args.output_dir), extra={"source_prefix": int(prefix), "checkpoint_kind": checkpoint_kind, "checkpoint": int(checkpoint), "actual_test_mcc": float(row["actual_test_mcc"])})
        _append_jsonl(results_path, row)
        completed_keys.add(result_key)
        validation_counter += 1
        print(
            f"[meta-replay] {scenario} prefix={prefix} {checkpoint_kind}={checkpoint}: "
            f"Alzheimer={row['actual_valid_mcc']:.4f}"
            + ("" if predicted_mcc is None else f" predicted={float(predicted_mcc):.4f}"),
            flush=True,
        )

        if wandb_run is not None:
            payload = {
                "validation_index": validation_counter,
                "source_prefix": int(prefix),
                f"{scenario}/actual_alzheimer_valid_mcc": row["actual_valid_mcc"],
                f"{scenario}/checkpoint": int(checkpoint),
            }
            payload["leaderboard/best_alzheimer_valid_mcc"] = float(universal_best["score"]); payload["leaderboard/is_current_best"] = int(universal_is_best)
            if predicted_mcc is not None:
                payload[f"{scenario}/predicted_alzheimer_mcc"] = float(predicted_mcc)
                payload[f"{scenario}/prediction_error"] = row["surrogate_signed_error"]
            wandb_run.log(payload)

    try:
        for prefix in prefixes:
            prefix_trials = trials_at_prefix(trials, prefix, role=SOURCE_ROLE)
            best = best_by_dataset(prefix_trials, source_ids)
            current_fixed = _fixed_for_prefix(args, trials, source_ids, prefix, full_fixed)
            adjusted_best = _adjust_best(best, current_fixed)
            print(
                f"\n[meta-replay] source prefix {prefix}/{max_prefix}; "
                f"source rows={len(prefix_trials)} fixed={json.dumps(current_fixed, sort_keys=True)}",
                flush=True,
            )

            # 1) Direct metadata -> best-known hparams. Reinitialized from scratch per prefix.
            if "direct" in scenarios:
                _, history, diagnostics = train_direct_meta_model(
                    adjusted_best,
                    source_meta,
                    alzheimer_meta,
                    max_warmup=max_warmup,
                    fixed_categories=current_fixed,
                    hidden_size=args.meta_hidden_size,
                    epochs=args.meta_epochs,
                    lr=args.meta_lr,
                    seed=args.seed + prefix,
                )
                for record in history:
                    _append_jsonl(prediction_path, {
                        "scenario": "direct",
                        "source_prefix": prefix,
                        "meta_epoch": record.epoch,
                        "train_loss": record.train_loss,
                        "config": record.target_config,
                    })
                by_epoch = {record.epoch: record for record in history}
                selected = sorted(set([epoch for epoch in direct_eval_epochs if epoch in by_epoch] + [history[-1].epoch]))
                for epoch in selected:
                    record = by_epoch[epoch]
                    persist_actual(
                        scenario="direct", prefix=prefix,
                        checkpoint_kind="meta_epoch", checkpoint=epoch,
                        config=record.target_config,
                        extra={"train_loss": record.train_loss, "meta_diagnostics": diagnostics},
                    )

            # Fit shared surrogates lazily. All source trials are valid score targets.
            extra_surrogate = mlp_surrogate = None
            lodo_extra = lodo_mlp = None
            needs_extra = ("surrogate_extra_trees" in scenarios) or ("surrogate_evolution" in scenarios) or (args.rl_surrogate == "extra_trees" and any(name in scenarios for name in ("meta_rl", "target_rl")))
            needs_mlp = ("surrogate_mlp" in scenarios) or (args.rl_surrogate == "mlp" and any(name in scenarios for name in ("meta_rl", "target_rl")))

            if needs_extra:
                extra_surrogate = fit_extra_trees_surrogate(
                    prefix_trials, source_meta, max_warmup=max_warmup, seed=args.seed + prefix
                )
                if not args.skip_source_lodo:
                    lodo_extra = leave_one_dataset_out_surrogate_rmse(
                        prefix_trials, source_meta, max_warmup=max_warmup,
                        model_type="extra_trees", seed=args.seed + prefix,
                    )
            if needs_mlp:
                mlp_surrogate = fit_mlp_surrogate(
                    prefix_trials, source_meta, max_warmup=max_warmup,
                    epochs=args.surrogate_mlp_epochs,
                    ensemble_size=args.surrogate_ensemble_size,
                    seed=args.seed + prefix,
                )
                if not args.skip_source_lodo:
                    lodo_mlp = leave_one_dataset_out_surrogate_rmse(
                        prefix_trials, source_meta, max_warmup=max_warmup,
                        model_type="mlp", seed=args.seed + prefix,
                    )

            if "surrogate_extra_trees" in scenarios:
                proposal = optimize_surrogate_tpe(
                    extra_surrogate, alzheimer_meta, hp_args,
                    fixed_categories=current_fixed,
                    n_trials=args.surrogate_search_trials,
                    risk_penalty=args.surrogate_risk_penalty,
                    seed=args.seed + 100 * prefix,
                )
                persist_actual(
                    scenario="surrogate_extra_trees", prefix=prefix,
                    checkpoint_kind="surrogate_final", checkpoint=1,
                    config=proposal.config,
                    predicted_mcc=proposal.predicted_mcc,
                    predicted_std=proposal.predicted_std,
                    extra={"acquisition": proposal.acquisition, "source_lodo_rmse": lodo_extra},
                )

            if "surrogate_mlp" in scenarios:
                proposal = optimize_surrogate_tpe(
                    mlp_surrogate, alzheimer_meta, hp_args,
                    fixed_categories=current_fixed,
                    n_trials=args.surrogate_search_trials,
                    risk_penalty=args.surrogate_risk_penalty,
                    seed=args.seed + 200 * prefix,
                )
                persist_actual(
                    scenario="surrogate_mlp", prefix=prefix,
                    checkpoint_kind="surrogate_final", checkpoint=1,
                    config=proposal.config,
                    predicted_mcc=proposal.predicted_mcc,
                    predicted_std=proposal.predicted_std,
                    extra={"acquisition": proposal.acquisition, "source_lodo_rmse": lodo_mlp},
                )


            if "surrogate_evolution" in scenarios:
                evolution_surrogate = extra_surrogate if extra_surrogate is not None else mlp_surrogate
                proposal, evolution_history = optimize_surrogate_evolution(
                    evolution_surrogate, alzheimer_meta,
                    max_warmup=max_warmup,
                    fixed_categories=current_fixed,
                    population_size=args.evolution_population_size,
                    generations=args.evolution_generations,
                    elite_count=max(1, min(8, args.evolution_population_size - 1)),
                    tournament_size=max(2, min(3, args.evolution_population_size)),
                    risk_penalty=args.surrogate_risk_penalty,
                    seed=args.seed + 250 * prefix,
                )
                for record in evolution_history:
                    _append_jsonl(prediction_path, {
                        "scenario": "surrogate_evolution",
                        "source_prefix": prefix,
                        "generation": record.generation,
                        "best_acquisition": record.best_acquisition,
                        "predicted_mcc": record.predicted_mcc,
                        "predicted_std": record.predicted_std,
                        "config": record.config,
                    })
                by_generation = {record.generation: record for record in evolution_history}
                selected = sorted(set([g for g in evolution_eval_generations if g in by_generation] + [evolution_history[-1].generation]))
                for generation in selected:
                    record = by_generation[generation]
                    persist_actual(
                        scenario="surrogate_evolution", prefix=prefix,
                        checkpoint_kind="evolution_generation", checkpoint=generation,
                        config=record.config,
                        predicted_mcc=record.predicted_mcc,
                        predicted_std=record.predicted_std,
                        extra={
                            "acquisition": record.best_acquisition,
                            "source_lodo_rmse": lodo_extra,
                            "population_size": args.evolution_population_size,
                        },
                    )

            rl_surrogate = extra_surrogate if args.rl_surrogate == "extra_trees" else mlp_surrogate
            if any(name in scenarios for name in ("meta_rl", "target_rl")) and rl_surrogate is None:
                raise RuntimeError("RL requested but its surrogate was not fitted")

            for scenario, mode in (("meta_rl", "meta"), ("target_rl", "target")):
                if scenario not in scenarios:
                    continue
                _, history = train_reinforce_policy(
                    rl_surrogate,
                    source_meta,
                    alzheimer_meta,
                    max_warmup=max_warmup,
                    fixed_categories=current_fixed,
                    mode=mode,
                    epochs=args.rl_epochs,
                    batch_size=args.rl_batch_size,
                    hidden_size=args.rl_hidden_size,
                    lr=args.rl_lr,
                    risk_penalty=args.surrogate_risk_penalty,
                    entropy_coef=args.rl_entropy,
                    seed=args.seed + (3000 if mode == "meta" else 6000) + prefix,
                )
                for record in history:
                    _append_jsonl(prediction_path, {
                        "scenario": scenario,
                        "source_prefix": prefix,
                        "rl_epoch": record.epoch,
                        "mean_reward": record.mean_reward,
                        "best_reward": record.best_reward,
                        "predicted_mcc": record.predicted_mcc,
                        "predicted_std": record.predicted_std,
                        "config": record.target_config,
                    })
                by_epoch = {record.epoch: record for record in history}
                selected = sorted(set([epoch for epoch in rl_eval_epochs if epoch in by_epoch] + [history[-1].epoch]))
                for epoch in selected:
                    record = by_epoch[epoch]
                    persist_actual(
                        scenario=scenario, prefix=prefix,
                        checkpoint_kind="rl_epoch", checkpoint=epoch,
                        config=record.target_config,
                        predicted_mcc=record.predicted_mcc,
                        predicted_std=record.predicted_std,
                        extra={
                            "mean_surrogate_reward": record.mean_reward,
                            "best_surrogate_reward": record.best_reward,
                            "rl_surrogate": args.rl_surrogate,
                            "source_lodo_rmse": lodo_extra if args.rl_surrogate == "extra_trees" else lodo_mlp,
                            "real_alzheimer_reward_used_for_rl": False,
                        },
                    )

        return 0
    finally:

        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
