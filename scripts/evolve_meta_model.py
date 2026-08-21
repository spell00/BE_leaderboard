#!/usr/bin/env python3
"""Evolve a dataset-conditioned BERNN policy across whole biological datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from src.dataset_splits import load_dataset_partitions
from src.evolutionary_meta import (
    EvolutionConfig,
    PolicyShape,
    aggregate_dataset_scores,
    breed_next_population,
    config_dict,
    decode_config,
    genome_digest,
    initialize_population,
    normalize_meta_features,
    recommended_batch_size,
)
from src.zero_shot_recommender.meta_features import (
    META_FEATURE_NAMES,
    extract_meta_features,
)

SCORE_THRESHOLDS = {
    "massbench_adenocarcinoma": {"reference": 0.24, "acceptable": 0.19},
    "massbench_alzheimer": {"reference": 0.44, "acceptable": 0.39},
    "massbench_benchmark": {"reference": 0.75, "acceptable": 0.70},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "config" / "evolution_development_datasets.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "meta_evolution")
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--mutation-scale", type=float, default=0.1)
    parser.add_argument("--worst-dataset-weight", type=float, default=0.25)
    parser.add_argument("--validation-patience", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--n-repeats", type=int, default=-1)
    parser.add_argument(
        "--validation-epoch-interval", type=int, default=10,
        help="Log held-out validation epoch telemetry at epoch 0 and this interval.",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-evaluator", action="store_true", help="Exercise evolution without training BERNN")
    parser.add_argument("--no-wandb", action="store_true", help="Disable live Weights & Biases telemetry")
    parser.add_argument("--wandb-project", default="BE_leaderboard_meta_evolution")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def _load_datasets(dataset_ids):
    datasets = {}
    meta = {}
    for dataset_id in dataset_ids:
        X, y, batches = hp_search.load_dataset(dataset_id)
        datasets[dataset_id] = (X, y, batches)
        extracted = extract_meta_features(X, y, batches)
        meta[dataset_id] = np.asarray([extracted[name] for name in META_FEATURE_NAMES], dtype=np.float32)
    return datasets, meta



def predict_config_for_dataset(
    X,
    y,
    batches,
    *,
    model_path=None,
    meta_model_path=None,
    checkpoint_path=None,
    max_warmup=50,
    **kwargs,
):
    """Predict exactly one BERNN config for one unseen dataset.

    The frozen evolutionary meta-policy is loaded from ``best_policy.npz``.
    Meta-features for the unseen dataset are normalized with the mean/scale
    learned from meta-training datasets and stored in that artifact.
    """
    policy_path = model_path or meta_model_path or checkpoint_path
    if policy_path is None:
        policy_path = ROOT / "results" / "meta_evolution" / "best_policy.npz"

    policy_path = Path(policy_path)
    if policy_path.is_dir():
        policy_path = policy_path / "best_policy.npz"
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Meta-policy not found: {policy_path}. "
            "Pass --meta-model-path /path/to/best_policy.npz."
        )

    saved = np.load(policy_path)
    required = {"genome", "meta_mean", "meta_scale", "n_inputs", "hidden_size"}
    missing = required.difference(saved.files)
    if missing:
        raise ValueError(
            f"Meta-policy {policy_path} is missing required arrays: {sorted(missing)}"
        )

    genome = saved["genome"]
    meta_mean = np.asarray(saved["meta_mean"], dtype=np.float32)
    meta_scale = np.asarray(saved["meta_scale"], dtype=np.float32)
    n_inputs = int(np.asarray(saved["n_inputs"]).reshape(-1)[0])
    hidden_size = int(np.asarray(saved["hidden_size"]).reshape(-1)[0])

    extracted = extract_meta_features(X, y, batches)
    raw_meta = np.asarray(
        [extracted[name] for name in META_FEATURE_NAMES],
        dtype=np.float32,
    )

    if raw_meta.shape[0] != n_inputs:
        raise ValueError(
            f"Meta-feature count mismatch: policy expects {n_inputs}, "
            f"current extractor produced {raw_meta.shape[0]}."
        )
    if meta_mean.shape != raw_meta.shape or meta_scale.shape != raw_meta.shape:
        raise ValueError(
            "Saved meta normalization shape does not match current meta-features: "
            f"mean={meta_mean.shape}, scale={meta_scale.shape}, current={raw_meta.shape}."
        )

    safe_scale = np.where(np.abs(meta_scale) < 1e-12, 1.0, meta_scale)
    normalized_meta = (raw_meta - meta_mean) / safe_scale

    shape = PolicyShape(n_inputs=n_inputs, hidden_size=hidden_size)
    return decode_config(
        genome,
        normalized_meta,
        shape,
        max_warmup=int(max_warmup),
    )


def _append_solution_record(output_dir: Path, record: dict, train_datasets: list[str]) -> None:
    jsonl_path = output_dir / "solutions.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")

    csv_path = output_dir / "solutions.csv"
    columns = [
        "solution_step",
        "generation",
        "population_index",
        "genome_digest",
        "train_fitness",
        "mean_mcc",
        "worst_mcc",
        *[f"mcc_{name}" for name in train_datasets],
        *[f"test_mcc_{name}" for name in train_datasets],
    ]
    row = {name: record.get(name) for name in columns}
    for name in train_datasets:
        row[f"mcc_{name}"] = record["train_scores"][name]
        row[f"test_mcc_{name}"] = record["test_scores"].get(name)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _mean_epoch_metric(metrics: dict, metric_name: str) -> dict[int, float]:
    """Average one BERNN epoch metric over grouped outer-CV folds/repeats."""
    values: dict[int, list[float]] = {}
    for rows in metrics.get("_epoch_traces_by_fold", {}).values():
        for row in rows:
            epoch = int(row["epoch"])
            matching = [
                float(value) for key, value in row.items()
                if key == metric_name or key.endswith(f"/{metric_name}")
            ]
            if matching:
                values.setdefault(epoch, []).extend(matching)
    return {epoch: float(np.mean(epoch_values)) for epoch, epoch_values in values.items()}


def _classifier_widths(config: dict) -> list[int]:
    widths = [max(16, int(config["layer1"]))]
    for _ in range(1, max(1, int(config["n_layers"]))):
        widths.append(max(widths[-1] // 2, 16))
    return widths


def _numeric_config_telemetry(dataset_id: str, config: dict) -> dict[str, float]:
    telemetry = {}
    for name, value in config.items():
        if isinstance(value, (bool, np.bool_)):
            telemetry[f"solutions/config/{dataset_id}/{name}"] = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            telemetry[f"solutions/config/{dataset_id}/{name}"] = float(value)
    widths = _classifier_widths(config)
    telemetry[f"solutions/config/{dataset_id}/n_neurons"] = float(sum(widths))
    for layer_index, width in enumerate(widths, 1):
        telemetry[f"solutions/config/{dataset_id}/layer{layer_index}_width"] = float(width)
    return telemetry


def _score_figure(dataset_id: str, rows: list[tuple[int, float, float]]):
    """Plot validation/test MCC plus BERNN-best and acceptable-goal traces."""
    import plotly.graph_objects as go

    x = [row[0] for row in rows]
    valid = [row[1] for row in rows]
    test = [row[2] for row in rows]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=valid, mode="lines+markers", name="valid_mcc"))
    figure.add_trace(go.Scatter(x=x, y=test, mode="lines+markers", name="test_mcc"))
    thresholds = SCORE_THRESHOLDS[dataset_id]
    figure.add_trace(go.Scatter(
        x=x, y=[thresholds["reference"]] * len(x), mode="lines",
        line={"dash": "solid", "width": 2}, name="Best valid MCC",
    ))
    figure.add_trace(go.Scatter(
        x=x, y=[thresholds["acceptable"]] * len(x), mode="lines",
        line={"dash": "dash", "width": 2}, name="goal",
    ))
    figure.update_layout(
        title=f"{dataset_id}: MCC versus BERNN thresholds",
        xaxis_title="solution step", yaxis_title="grouped-CV MCC",
        yaxis_range=[-1.0, 1.0],
    )
    return figure


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.generations < 1 or args.validation_patience < 1:
        raise ValueError("generations and validation_patience must be positive")
    partitions = load_dataset_partitions(args.split_manifest)

    # Only train + validation datasets are loaded during meta-model learning.
    # Held-out meta-test datasets are not touched until evolution, validation
    # selection, and early stopping are completely finished.
    development_ids = partitions.train + partitions.validation
    heldout_ids = tuple(partitions.test)
    datasets, raw_meta = _load_datasets(development_ids)

    # Fixed within-dataset test labels are monitoring-only. They never enter
    # fitness, breeding, champion selection, early stopping, preprocessing,
    # or BERNN fitting.
    fixed_tests = {
        dataset_id: hp_search.load_fixed_test_dataset(dataset_id)
        for dataset_id in partitions.train
    }
    train_matrix = np.stack([raw_meta[name] for name in partitions.train])
    validation_matrix = np.stack([raw_meta[name] for name in partitions.validation])
    normalized_train, mean, scale = normalize_meta_features(train_matrix)
    normalized_validation, _, _ = normalize_meta_features(train_matrix, validation_matrix)
    normalized_meta = {
        **{name: row for name, row in zip(partitions.train, normalized_train)},
        **{name: row for name, row in zip(partitions.validation, normalized_validation)},
    }

    evolution = EvolutionConfig(
        population_size=args.population_size,
        elite_count=args.elite_count,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        worst_dataset_weight=args.worst_dataset_weight,
    )
    evolution.validate()
    shape = PolicyShape(len(META_FEATURE_NAMES), args.hidden_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = _sha256(args.split_manifest)
    checkpoint_path = args.output_dir / "checkpoint.npz"
    state_path = args.output_dir / "state.json"
    rng = np.random.default_rng(args.seed)
    generation_start = 0
    best_validation = -np.inf
    stale_generations = 0
    solution_step = 0

    if args.resume:
        if not checkpoint_path.exists() or not state_path.exists():
            raise FileNotFoundError("--resume requires checkpoint.npz and state.json")
        state = json.loads(state_path.read_text())
        if state["development_manifest_sha256"] != manifest_hash:
            raise ValueError("Development dataset manifest changed since checkpoint creation")
        expected_shape = {"n_inputs": shape.n_inputs, "hidden_size": shape.hidden_size}
        if state.get("policy_shape") != expected_shape:
            raise ValueError("Policy shape differs from the saved checkpoint")
        if state.get("evolution_config") != config_dict(evolution):
            raise ValueError("Evolution configuration differs from the saved checkpoint")
        saved = np.load(checkpoint_path)
        population = saved["population"]
        generation_start = int(state["next_generation"])
        best_validation = float(state["best_validation"])
        stale_generations = int(state["stale_generations"])
        solution_step = int(state.get("solution_step", generation_start * args.population_size))
        rng.bit_generator.state = state["rng_state"]
    else:
        population = initialize_population(shape, evolution, rng)

    hp_args = hp_search.parse_args([])
    hp_args.n_epochs = args.n_epochs
    hp_args.n_repeats = args.n_repeats
    hp_args.device = args.device
    hp_args.no_wandb = True
    hp_args.combine_test = False
    hp_args.max_warmup = max(1, min(50, args.n_epochs))
    score_cache = {}

    wandb_run = None
    if not args.no_wandb:
        import wandb

        run_metadata_path = args.output_dir / "run_metadata.json"
        if run_metadata_path.exists():
            run_metadata = json.loads(run_metadata_path.read_text())
        else:
            run_metadata = {"wandb_run_id": uuid.uuid4().hex[:8]}
            _atomic_json(run_metadata_path, run_metadata)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=run_metadata["wandb_run_id"],
            resume="allow",
            config={
                **vars(args),
                "split_manifest": str(args.split_manifest),
                "output_dir": str(args.output_dir),
                "train_datasets": list(partitions.train),
                "validation_datasets": list(partitions.validation),
                "heldout_datasets": list(heldout_ids),
            },
        )
        wandb.define_metric("solution_step")
        wandb.define_metric("solutions/*", step_metric="solution_step")
        wandb.define_metric("generation")
        wandb.define_metric("champion/*", step_metric="generation")
        wandb.define_metric("validation_epoch_step")
        wandb.define_metric("validation_epoch/*", step_metric="validation_epoch_step")
        for dataset_id, thresholds in SCORE_THRESHOLDS.items():
            wandb_run.summary[f"reference_mcc/{dataset_id}"] = thresholds["reference"]
            wandb_run.summary[f"acceptable_mcc/{dataset_id}"] = thresholds["acceptable"]

    def evaluate(genome, dataset_id):
        key = f"{genome_digest(genome)}:{dataset_id}"
        if key in score_cache:
            return score_cache[key]
        metrics = {}
        try:
            if args.smoke_evaluator:
                target = raw_meta[dataset_id]
                score = float(np.tanh(np.mean(genome[: min(len(genome), len(target))]) + np.mean(target) * 1e-4))
            else:
                X, y, batches = datasets[dataset_id]
                config = decode_config(genome, normalized_meta[dataset_id], shape, max_warmup=hp_args.max_warmup)
                run_args = argparse.Namespace(**vars(hp_args))
                run_args.dataset = dataset_id
                run_args.seed = args.seed
                run_args.bs = recommended_batch_size(batches, cap=hp_args.bs)
                run_args.results_dir = str(args.output_dir / dataset_id)
                run_args.resolved_n_repeats = hp_search.resolve_n_repeats(run_args.n_repeats, batches)
                exp_id = f"meta_evolution_{dataset_id}_{genome_digest(genome)[:12]}"
                score, metrics = hp_search.run_trial(
                    config, run_args, (X, y, batches), exp_id,
                    fixed_test_data=fixed_tests.get(dataset_id),
                )
        except Exception as exc:  # noqa: BLE001 - invalid candidates receive sentinel fitness
            score = -1.0
            print(
                f"[evolution] failed genome={genome_digest(genome)[:12]} "
                f"dataset={dataset_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )
        score_cache[key] = (float(score), metrics)
        return score_cache[key]

    ledger = args.output_dir / "generations.jsonl"
    score_history = {dataset_id: [] for dataset_id in partitions.train}
    for generation in range(generation_start, args.generations):
        train_scores = np.empty((len(population), len(partitions.train)), dtype=np.float64)
        train_fitness = np.empty(len(population), dtype=np.float64)
        for population_index, genome in enumerate(population):
            scores = np.asarray([evaluate(genome, name)[0] for name in partitions.fitness_datasets()])
            fitness = aggregate_dataset_scores(scores, evolution.worst_dataset_weight)
            train_scores[population_index] = scores
            train_fitness[population_index] = fitness
            decoded = {
                name: decode_config(genome, normalized_meta[name], shape, max_warmup=hp_args.max_warmup)
                for name in partitions.train
            }
            solution_record = {
                "solution_step": solution_step,
                "generation": generation,
                "population_index": population_index,
                "genome_digest": genome_digest(genome),
                "train_fitness": float(fitness),
                "mean_mcc": float(np.mean(scores)),
                "worst_mcc": float(np.min(scores)),
                "train_scores": dict(zip(partitions.train, scores.tolist())),
                "test_scores": {
                    name: float(evaluate(genome, name)[1].get("test_mcc", np.nan))
                    for name in partitions.train
                },
                "config_by_dataset": decoded,
            }
            solution_record["total_valid_mcc"] = float(aggregate_dataset_scores(scores, evolution.worst_dataset_weight))
            solution_record["total_test_mcc"] = float(aggregate_dataset_scores(
                solution_record["test_scores"].values(),
                evolution.worst_dataset_weight,
            ))
            for dataset_id, score in zip(partitions.train, scores):
                score_history[dataset_id].append((
                    solution_step,
                    float(score),
                    solution_record["test_scores"][dataset_id],
                ))
            _append_solution_record(args.output_dir, solution_record, list(partitions.train))
            if wandb_run is not None:
                config_telemetry = {}
                for dataset_id, dataset_config in decoded.items():
                    config_telemetry.update(_numeric_config_telemetry(dataset_id, dataset_config))
                wandb_run.log({
                    "solution_step": solution_step,
                    "solutions/train_fitness": float(fitness),
                    "solutions/mean_mcc": float(np.mean(scores)),
                    "solutions/worst_mcc": float(np.min(scores)),
                    "solutions/total_valid_mcc": solution_record["total_valid_mcc"],
                    "solutions/total_test_mcc": solution_record["total_test_mcc"],
                    **{
                        f"solutions/metrics/{name}/valid_mcc": float(score)
                        for name, score in zip(partitions.train, scores)
                    },
                    **{
                        f"solutions/metrics/{name}/test_mcc": solution_record["test_scores"][name]
                        for name in partitions.train
                    },
                    **config_telemetry,
                    **{
                        f"reference_mcc/{name}": float(thresholds["reference"])
                        for name, thresholds in SCORE_THRESHOLDS.items()
                    },
                    **{
                        f"acceptable_mcc/{name}": float(thresholds["acceptable"])
                        for name, thresholds in SCORE_THRESHOLDS.items()
                    },
                })
            print(f"[evolution] solution {solution_step} complete: {json.dumps(solution_record)}", flush=True)
            solution_step += 1
        champion_index = int(np.argmax(train_fitness))
        champion = population[champion_index].copy()
        validation_results = [evaluate(champion, name) for name in partitions.selection_datasets()]
        validation_scores = [result[0] for result in validation_results]
        validation_score = aggregate_dataset_scores(validation_scores, evolution.worst_dataset_weight)
        record = {
            "generation": generation,
            "train_champion_fitness": float(train_fitness[champion_index]),
            "train_scores": dict(zip(partitions.train, train_scores[champion_index].tolist())),
            "validation_score": validation_score,
            "validation_scores": dict(zip(partitions.validation, validation_scores)),
            "champion_digest": genome_digest(champion),
            "champion_config_by_dataset": {
                name: decode_config(champion, normalized_meta[name], shape, max_warmup=hp_args.max_warmup)
                for name in development_ids
            },
        }
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        print(json.dumps(record, indent=2), flush=True)
        if wandb_run is not None:
            champion_payload = {
                "generation": generation,
                "champion/train_fitness": float(train_fitness[champion_index]),
                "champion/validation_score": float(validation_score),
                **{
                    f"champion/validation_mcc/{name}": float(score)
                    for name, score in zip(partitions.validation, validation_scores)
                },
            }
            for dataset_id, rows in score_history.items():
                champion_payload[f"solutions/mcc/{dataset_id}"] = _score_figure(dataset_id, rows)
            wandb_run.log(champion_payload)
            interval = max(1, int(args.validation_epoch_interval))
            for dataset_index, (dataset_id, (_, validation_metrics)) in enumerate(
                zip(partitions.validation, validation_results)
            ):
                epoch_mcc = _mean_epoch_metric(validation_metrics, "valid_mcc")
                if not epoch_mcc:
                    continue
                final_epoch = max(epoch_mcc)
                for epoch, mcc in sorted(epoch_mcc.items()):
                    if epoch != 0 and epoch % interval != 0 and epoch != final_epoch:
                        continue
                    epoch_step = (
                        generation * (args.n_epochs + 1) * len(partitions.validation)
                        + dataset_index * (args.n_epochs + 1)
                        + epoch
                    )
                    wandb_run.log({
                        "validation_epoch_step": epoch_step,
                        "validation_epoch/generation": generation,
                        "validation_epoch/epoch": epoch,
                        f"validation_epoch/valid_mcc/{dataset_id}": mcc,
                    })

        if validation_score > best_validation:
            best_validation = validation_score
            stale_generations = 0
            np.savez_compressed(
                args.output_dir / "best_policy.npz",
                genome=champion,
                meta_mean=mean,
                meta_scale=scale,
                n_inputs=np.asarray([shape.n_inputs]),
                hidden_size=np.asarray([shape.hidden_size]),
            )
            _atomic_json(args.output_dir / "best_policy.json", record)
        else:
            stale_generations += 1

        next_population = breed_next_population(population, train_fitness, evolution, rng)
        np.savez_compressed(checkpoint_path, population=next_population, meta_mean=mean, meta_scale=scale)
        _atomic_json(state_path, {
            "next_generation": generation + 1,
            "best_validation": best_validation,
            "stale_generations": stale_generations,
            "solution_step": solution_step,
            "rng_state": rng.bit_generator.state,
            "development_manifest_sha256": manifest_hash,
            "policy_shape": {"n_inputs": shape.n_inputs, "hidden_size": shape.hidden_size},
            "evolution_config": config_dict(evolution),
            "train_datasets": list(partitions.train),
            "validation_datasets": list(partitions.validation),
            "heldout_datasets": list(heldout_ids),
        })
        population = next_population
        if stale_generations >= args.validation_patience:
            print(f"[evolution] validation early stop after generation {generation}", flush=True)
            break

    # -------------------------------------------------------------------------
    # FINAL HELD-OUT META-TEST EVALUATION
    #
    # This happens only after the evolutionary search and validation-based
    # policy selection are over. Held-out results are reporting-only: they do
    # not affect fitness, breeding, champion selection, normalization
    # statistics, early stopping, or the saved best policy.
    # -------------------------------------------------------------------------
    if heldout_ids:
        print(
            f"[evolution] evaluating final policy on held-out datasets: {list(heldout_ids)}",
            flush=True,
        )

        heldout_datasets, heldout_raw_meta = _load_datasets(heldout_ids)
        heldout_matrix = np.stack([heldout_raw_meta[name] for name in heldout_ids])

        # Apply the normalization learned from meta-training datasets only.
        normalized_heldout, _, _ = normalize_meta_features(
            train_matrix,
            heldout_matrix,
        )
        heldout_meta = {
            name: row
            for name, row in zip(heldout_ids, normalized_heldout)
        }

        heldout_fixed_tests = {
            dataset_id: hp_search.load_fixed_test_dataset(dataset_id)
            for dataset_id in heldout_ids
        }

        best_policy_path = args.output_dir / "best_policy.npz"
        if not best_policy_path.exists():
            raise FileNotFoundError(
                "Cannot evaluate held-out datasets because best_policy.npz was not created"
            )

        best_policy = np.load(best_policy_path)
        best_genome = best_policy["genome"]
        best_digest = genome_digest(best_genome)

        heldout_results = {}
        heldout_configs = {}

        for dataset_index, dataset_id in enumerate(heldout_ids):
            X, y, batches = heldout_datasets[dataset_id]

            # Zero-shot BERNN hyperparameters predicted by the frozen meta-policy.
            config = decode_config(
                best_genome,
                heldout_meta[dataset_id],
                shape,
                max_warmup=hp_args.max_warmup,
            )
            heldout_configs[dataset_id] = config

            run_args = argparse.Namespace(**vars(hp_args))
            run_args.dataset = dataset_id
            run_args.seed = args.seed + 10_000 + dataset_index
            run_args.bs = recommended_batch_size(batches, cap=hp_args.bs)
            run_args.results_dir = str(args.output_dir / "heldout" / dataset_id)
            run_args.resolved_n_repeats = hp_search.resolve_n_repeats(
                run_args.n_repeats,
                batches,
            )

            metrics = {}
            try:
                if args.smoke_evaluator:
                    target = heldout_raw_meta[dataset_id]
                    valid_mcc = float(
                        np.tanh(
                            np.mean(best_genome[: min(len(best_genome), len(target))])
                            + np.mean(target) * 1e-4
                        )
                    )
                else:
                    exp_id = (
                        f"meta_evolution_HELDOUT_{dataset_id}_{best_digest[:12]}"
                    )
                    valid_mcc, metrics = hp_search.run_trial(
                        config,
                        run_args,
                        (X, y, batches),
                        exp_id,
                        fixed_test_data=heldout_fixed_tests[dataset_id],
                    )

                result = {
                    "valid_mcc": float(valid_mcc),
                    "test_mcc": float(metrics.get("test_mcc", np.nan)),
                }
            except Exception as exc:  # held-out failures are reported, never optimized
                result = {
                    "valid_mcc": -1.0,
                    "test_mcc": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(
                    f"[evolution] held-out evaluation failed dataset={dataset_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            heldout_results[dataset_id] = result
            print(
                f"[evolution] HELDOUT {dataset_id}: "
                f"valid_mcc={result['valid_mcc']:.4f}, "
                f"test_mcc={result['test_mcc']:.4f}",
                flush=True,
            )

        finite_valid = [
            result["valid_mcc"]
            for result in heldout_results.values()
            if np.isfinite(result["valid_mcc"])
        ]
        finite_test = [
            result["test_mcc"]
            for result in heldout_results.values()
            if np.isfinite(result["test_mcc"])
        ]

        heldout_summary = {
            "mean_valid_mcc": float(np.mean(finite_valid)) if finite_valid else np.nan,
            "worst_valid_mcc": float(np.min(finite_valid)) if finite_valid else np.nan,
            "total_valid_mcc": float(aggregate_dataset_scores(finite_valid, evolution.worst_dataset_weight)) if finite_valid else np.nan,
            "mean_test_mcc": float(np.mean(finite_test)) if finite_test else np.nan,
            "worst_test_mcc": float(np.min(finite_test)) if finite_test else np.nan,
            "total_test_mcc": float(aggregate_dataset_scores(finite_test, evolution.worst_dataset_weight)) if finite_test else np.nan,
        }

        _atomic_json(
            args.output_dir / "heldout_results.json",
            {
                "policy_digest": best_digest,
                "datasets": heldout_results,
                "config_by_dataset": heldout_configs,
                "summary": heldout_summary,
            },
        )

        if wandb_run is not None:
            heldout_payload = {}

            for dataset_id, result in heldout_results.items():
                heldout_payload[
                    f"heldout/metrics/{dataset_id}/valid_mcc"
                ] = result["valid_mcc"]
                heldout_payload[
                    f"heldout/metrics/{dataset_id}/test_mcc"
                ] = result["test_mcc"]

                config_telemetry = _numeric_config_telemetry(
                    dataset_id,
                    heldout_configs[dataset_id],
                )
                for key, value in config_telemetry.items():
                    heldout_payload[
                        key.replace("solutions/config/", "heldout/config/")
                    ] = value

            for metric_name, value in heldout_summary.items():
                heldout_payload[f"heldout/{metric_name}"] = value

            wandb_run.log(heldout_payload)

            # Headline final results remain easy to find on the W&B run page.
            for dataset_id, result in heldout_results.items():
                wandb_run.summary[
                    f"heldout/{dataset_id}/valid_mcc"
                ] = result["valid_mcc"]
                wandb_run.summary[
                    f"heldout/{dataset_id}/test_mcc"
                ] = result["test_mcc"]

            for metric_name, value in heldout_summary.items():
                wandb_run.summary[f"heldout/{metric_name}"] = value

    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())