#!/usr/bin/env python3
"""Classic per-dataset Optuna baseline for the evolutionary BERNN experiment.

One solution step asks each independent dataset study for one TPE trial. This
matches one evolutionary solution's three development-dataset BERNN evaluations,
while keeping test labels monitoring-only.
"""

from __future__ import annotations

import uuid
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hp_search
from scripts.evolve_meta_model import (
    SCORE_THRESHOLDS,
    _numeric_config_telemetry,
    _score_figure,
)
from src.dataset_splits import load_dataset_partitions
from src.evolutionary_meta import aggregate_dataset_scores, recommended_batch_size
from src.zero_shot_recommender.meta_features import (
    META_FEATURE_NAMES,
    extract_meta_features,
)


_ACTIVE_STORAGE = None
_ACTIVE_WANDB_RUN = None


def _cleanup_runtime_resources() -> None:
    """Close run-level clients and the shared Optuna SQLAlchemy engine."""
    global _ACTIVE_STORAGE, _ACTIVE_WANDB_RUN
    if _ACTIVE_WANDB_RUN is not None:
        try:
            _ACTIVE_WANDB_RUN.finish()
        except Exception:
            pass
        finally:
            _ACTIVE_WANDB_RUN = None
    if _ACTIVE_STORAGE is not None:
        try:
            _ACTIVE_STORAGE.engine.dispose()
        except Exception:
            pass
        finally:
            _ACTIVE_STORAGE = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest", type=Path,
        default=ROOT / "config" / "evolution_development_datasets.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "optuna_comparison")
    parser.add_argument("--n-trials", type=int, default=1000, help="Trials per dataset.")
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=-1)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="BE_leaderboard_meta_evolution")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--meta-hidden-size",
        type=int,
        default=64,
        help="Hidden width of the online shallow meta-network.",
    )
    parser.add_argument(
        "--meta-epochs",
        type=int,
        default=1000,
        help="Optimization epochs used to refit the tiny online meta-network.",
    )
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def _gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0].strip()
    except Exception:
        return "unavailable"


def _completed_trials(study):
    import optuna

    return [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def _trial_payload(trial) -> dict:
    attrs = dict(trial.user_attrs)
    return {
        "trial_number": int(trial.number),
        "valid_mcc": float(trial.value),
        "test_mcc": float(attrs.get("test_mcc", np.nan)),
        "fit_seconds": float(attrs.get("fit_seconds", np.nan)),
        "config": attrs.get("config", {}),
        "error": attrs.get("error"),
    }


def _best_payload(study) -> dict:
    return _trial_payload(max(_completed_trials(study), key=lambda trial: float(trial.value)))


def _aggregate_solution_metric(solution: dict, dataset_ids: tuple[str, ...], metric_name: str) -> float:
    values = [solution[dataset_id][metric_name] for dataset_id in dataset_ids]
    return float(aggregate_dataset_scores(values))

def _meta_vector(dataset) -> np.ndarray:
    """Dataset descriptor vector used as input to the shallow meta-network."""
    X, y, batches = dataset
    extracted = extract_meta_features(X, y, batches)
    return np.asarray(
        [extracted[name] for name in META_FEATURE_NAMES],
        dtype=np.float32,
    )


def _scale01(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def _unscale01(value: float, low: float, high: float) -> float:
    return float(low + np.clip(float(value), 0.0, 1.0) * (high - low))


def _log_scale01(value: float, low: float, high: float) -> float:
    value = float(np.clip(float(value), low, high))
    return _scale01(np.log10(value), np.log10(low), np.log10(high))


def _log_unscale01(value: float, low: float, high: float) -> float:
    return float(
        10 ** _unscale01(value, np.log10(low), np.log10(high))
    )


_DLOSS = tuple(hp_search.DLOSS_CHOICES)
_SCALERS = tuple(hp_search.SCALER_CHOICES)
_N_LAYERS = (1, 2, 3, 4, 5)


def _one_hot(value, choices) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _encode_config(config: dict, max_warmup: int) -> np.ndarray:
    """Encode a BERNN config into bounded meta-network targets."""
    encoded = [
        float(bool(config.get("variational", False))),
        float(bool(config.get("kan", False))),
        float(bool(config.get("class_triplet", False))),
        float(bool(config.get("log1p", True))),
        _scale01(config.get("class_triplet_w", 0.0), 0.0, 1.0),
        _log_scale01(config["lr"], 1e-4, 1e-2),
        _log_scale01(config["wd"], 1e-6, 1e-3),
        _scale01(config["nu"], 1e-4, 1e2),
        _scale01(config["smoothing"], 0.0, 0.2),
        _scale01(config["margin"], 0.0, 10.0),
        _scale01(config["dropout"], 0.0, 0.5),
        _scale01(config["thres"], 0.0, 0.1),
        _scale01(config["warmup"], 1.0, float(max_warmup)),
        _scale01(config["layer1"], 512.0, 1024.0),
        _log_scale01(
            config.get("gamma", 1e-2) if float(config.get("gamma", 0.0)) > 0 else 1e-2,
            1e-2,
            1e2,
        ),
        _log_scale01(
            config.get("beta", 1e-2) if float(config.get("beta", 0.0)) > 0 else 1e-2,
            1e-2,
            1e2,
        ),
        *_one_hot(config["dloss"], _DLOSS),
        *_one_hot(config["scaler"], _SCALERS),
        *_one_hot(int(config["n_layers"]), _N_LAYERS),
    ]
    return np.asarray(encoded, dtype=np.float32)


def _decode_config(encoded: np.ndarray, max_warmup: int) -> dict:
    """Decode one bounded output vector back to a valid BERNN config."""
    z = np.clip(np.asarray(encoded, dtype=float), 0.0, 1.0)
    i = 0

    variational = bool(z[i] >= 0.5); i += 1
    kan = bool(z[i] >= 0.5); i += 1
    class_triplet = bool(z[i] >= 0.5); i += 1
    log1p = bool(z[i] >= 0.5); i += 1
    class_triplet_w = _unscale01(z[i], 0.0, 1.0); i += 1
    lr = _log_unscale01(z[i], 1e-4, 1e-2); i += 1
    wd = _log_unscale01(z[i], 1e-6, 1e-3); i += 1
    nu = _unscale01(z[i], 1e-4, 1e2); i += 1
    smoothing = _unscale01(z[i], 0.0, 0.2); i += 1
    margin = _unscale01(z[i], 0.0, 10.0); i += 1
    dropout = _unscale01(z[i], 0.0, 0.5); i += 1
    thres = _unscale01(z[i], 0.0, 0.1); i += 1
    warmup = int(round(_unscale01(z[i], 1.0, float(max_warmup)))); i += 1
    layer1 = int(round(_unscale01(z[i], 512.0, 1024.0))); i += 1
    gamma_candidate = _log_unscale01(z[i], 1e-2, 1e2); i += 1
    beta_candidate = _log_unscale01(z[i], 1e-2, 1e2); i += 1

    dloss_slice = z[i:i + len(_DLOSS)]
    dloss = _DLOSS[int(np.argmax(dloss_slice))]
    i += len(_DLOSS)

    scaler_slice = z[i:i + len(_SCALERS)]
    scaler = _SCALERS[int(np.argmax(scaler_slice))]
    i += len(_SCALERS)

    layer_slice = z[i:i + len(_N_LAYERS)]
    n_layers = _N_LAYERS[int(np.argmax(layer_slice))]

    gamma = gamma_candidate if dloss in hp_search.ADVERSARIAL_DLOSS else 0.0
    beta = beta_candidate if variational else 0.0

    return {
        "model_type": "joint",
        "dloss": dloss,
        "variational": variational,
        "kan": kan,
        "class_triplet": class_triplet,
        "log1p": bool(log1p),
        "class_triplet_w": float(class_triplet_w),
        "lr": float(lr),
        "wd": float(wd),
        "nu": float(nu),
        "smoothing": float(smoothing),
        "margin": float(margin),
        "dropout": float(dropout),
        "thres": float(thres),
        "warmup": int(np.clip(warmup, 1, max_warmup)),
        "n_layers": int(n_layers),
        "layer1": int(np.clip(layer1, 512, 1024)),
        "scaler": scaler,
        "gamma": float(gamma),
        "beta": float(beta),
    }


def _fit_online_meta_network(
    *,
    studies,
    datasets,
    validation_dataset,
    args,
) -> tuple[dict, dict]:
    """Fit meta-features -> current best Optuna configs, then predict validation.

    Only partitions.train contribute targets. Validation descriptors are used
    strictly as an inference input; validation MCC never enters this fit.
    """
    import torch
    from torch import nn

    source_ids = [
        dataset_id
        for dataset_id, study in studies.items()
        if _completed_trials(study)
    ]
    if not source_ids:
        raise RuntimeError("Online meta-network requires at least one completed train study")

    train_meta = np.stack([_meta_vector(datasets[name]) for name in source_ids])
    validation_meta = _meta_vector(validation_dataset)[None, :]

    # Descriptor normalization is learned from meta-training datasets only.
    mean = train_meta.mean(axis=0, keepdims=True)
    scale = train_meta.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    train_meta = (train_meta - mean) / scale
    validation_meta = (validation_meta - mean) / scale

    max_warmup = max(1, min(50, int(args.n_epochs)))
    target_configs = {
        name: _best_payload(studies[name])["config"]
        for name in source_ids
    }
    target_scores = {
        name: float(_best_payload(studies[name])["valid_mcc"])
        for name in source_ids
    }
    targets = np.stack([
        _encode_config(target_configs[name], max_warmup)
        for name in source_ids
    ])

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    x = torch.tensor(train_meta, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    xv = torch.tensor(validation_meta, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(x.shape[1], int(args.meta_hidden_size)),
        nn.ReLU(),
        nn.Linear(int(args.meta_hidden_size), y.shape[1]),
        nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    model.train()
    final_loss = np.nan
    for _ in range(int(args.meta_epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    model.eval()
    with torch.no_grad():
        predicted = model(xv).cpu().numpy()[0]

    config = _decode_config(predicted, max_warmup)
    diagnostics = {
        "meta_train_loss": float(final_loss),
        "source_best_scores": target_scores,
        "source_best_configs": target_configs,
    }
    return config, diagnostics


def _add_config_to_payload(payload: dict, prefix: str, config: dict) -> None:
    """Log scalar predicted hyperparameters without inventing transfer metrics."""
    for key, value in config.items():
        metric = f"{prefix}/{key}"
        if isinstance(value, (bool, int, float, np.number)):
            payload[metric] = float(value) if not isinstance(value, bool) else int(value)
        elif isinstance(value, str):
            payload[metric] = value


def _append_solution(output_dir: Path, record: dict, dataset_ids: tuple[str, ...]) -> None:
    with (output_dir / "solutions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")
    csv_path = output_dir / "solutions.csv"
    columns = ["solution_step", "wall_clock_seconds"]
    for dataset_id in dataset_ids:
        columns.extend([
            f"valid_mcc_{dataset_id}", f"test_mcc_{dataset_id}",
            f"best_valid_mcc_{dataset_id}", f"best_test_mcc_{dataset_id}",
            f"fit_seconds_{dataset_id}",
        ])
    row = {"solution_step": record["solution_step"], "wall_clock_seconds": record["wall_clock_seconds"]}
    for dataset_id in dataset_ids:
        current = record["current"][dataset_id]
        best = record["best"][dataset_id]
        row.update({
            f"valid_mcc_{dataset_id}": current["valid_mcc"],
            f"test_mcc_{dataset_id}": current["test_mcc"],
            f"best_valid_mcc_{dataset_id}": best["valid_mcc"],
            f"best_test_mcc_{dataset_id}": best["test_mcc"],
            f"fit_seconds_{dataset_id}": current["fit_seconds"],
        })
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _main(argv=None) -> int:
    global _ACTIVE_STORAGE, _ACTIVE_WANDB_RUN
    args = parse_args(argv)
    if args.n_trials < 1:
        raise ValueError("n_trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if not args.resume:
            raise FileExistsError(f"{args.output_dir} already contains a run; pass --resume")
    else:
        metadata = {
            "created_at_unix": time.time(),
            "hostname": platform.node(),
            "gpu": _gpu_name(),
            "wandb_run_id": None,
        }

    partitions = load_dataset_partitions(args.split_manifest)

    # Optuna/TPE learns ONLY from meta-training datasets.
    dataset_ids = tuple(partitions.train)
    validation_ids = tuple(partitions.validation)

    datasets = {
        dataset_id: hp_search.load_dataset(dataset_id)
        for dataset_id in dataset_ids
    }
    fixed_tests = {
        dataset_id: hp_search.load_fixed_test_dataset(dataset_id)
        for dataset_id in dataset_ids
    }

    # Meta-validation datasets are loaded for monitoring/selection telemetry only.
    # They NEVER receive an Optuna study and their scores are NEVER passed to
    # study.tell(), so TPE cannot learn from them.
    validation_datasets = {
        dataset_id: hp_search.load_dataset(dataset_id)
        for dataset_id in validation_ids
    }
    validation_fixed_tests = {
        dataset_id: hp_search.load_fixed_test_dataset(dataset_id)
        for dataset_id in validation_ids
    }

    import mlflow
    import optuna

    # Keep MLflow fully run-local. Reusing the repository-wide store would make
    # a smoke run and a real run with the same trial number average together.
    tracking_dir = args.output_dir / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    storage_url = f"sqlite:///{(args.output_dir / 'optuna.sqlite3').resolve()}"
    storage = optuna.storages.RDBStorage(url=storage_url)
    _ACTIVE_STORAGE = storage
    studies = {
        dataset_id: optuna.create_study(
            study_name=f"classic_{dataset_id}", direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            storage=storage, load_if_exists=True,
        )
        for dataset_id in dataset_ids
    }

    wandb_run = None
    if not args.no_wandb:
        import wandb

        if metadata["wandb_run_id"] is None:
            metadata["wandb_run_id"] = uuid.uuid4().hex[:8]
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=metadata["wandb_run_id"], resume="allow",
            config={
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "split_manifest": str(args.split_manifest),
                "output_dir": str(args.output_dir),
                "train_datasets": list(dataset_ids),
                "validation_datasets": list(validation_ids),
                "method": "classic_optuna_tpe_online_meta_network_validation",
                "host": metadata["hostname"],
                "gpu": metadata["gpu"],
            },
        )
        _ACTIVE_WANDB_RUN = wandb_run
        wandb.define_metric("solution_step")
        wandb.define_metric("solutions/*", step_metric="solution_step")
        wandb.define_metric("validation/*", step_metric="solution_step")
        for dataset_id, thresholds in SCORE_THRESHOLDS.items():
            wandb_run.summary[f"reference_mcc/{dataset_id}"] = thresholds["reference"]
            wandb_run.summary[f"acceptable_mcc/{dataset_id}"] = thresholds["acceptable"]
    _atomic_json(metadata_path, metadata)

    print(
        "[optuna-comparison] ONLINE META NETWORK: current best Optuna configs from "
        "partitions.train are the only meta-training targets. At the START of each "
        "step after step 0, dataset meta-features are passed through the refitted "
        "shallow network to predict ONE config for partitions.validation. Validation "
        "scores never enter TPE or meta-network fitting.",
        flush=True,
    )
    history = {dataset_id: [] for dataset_id in dataset_ids}

    # One meta-network prediction/evaluation per unseen validation dataset and step.
    validation_history = {validation_id: [] for validation_id in validation_ids}

    existing_path = args.output_dir / "solutions.jsonl"
    if existing_path.exists():
        for line in existing_path.read_text().splitlines():
            if not line.strip():
                continue
            old = json.loads(line)
            for dataset_id in dataset_ids:
                current = old["current"][dataset_id]
                history[dataset_id].append((old["solution_step"], current["valid_mcc"], current["test_mcc"]))

    for solution_step in range(args.n_trials):
        # --------------------------------------------------------------
        # VALIDATION FIRST
        #
        # Step 0 has no Optuna-derived meta-training targets yet. From
        # step 1 onward, refit the shallow meta-network from CURRENT best
        # train-study configs and predict validation hparams BEFORE asking
        # Optuna for any new trial.
        # --------------------------------------------------------------
        if solution_step > 0 and validation_ids:
            print(
                f"[optuna-comparison] solution {solution_step}: "
                "validation-first online meta-network fit/prediction",
                flush=True,
            )
            validation_payload = {"solution_step": solution_step}

            for validation_index, validation_id in enumerate(validation_ids):
                Xv, yv, batches_v = validation_datasets[validation_id]

                predicted_config, meta_diagnostics = _fit_online_meta_network(
                    studies=studies,
                    datasets=datasets,
                    validation_dataset=(Xv, yv, batches_v),
                    args=args,
                )

                print(
                    f"[optuna-comparison] META PREDICTED {validation_id} "
                    f"step={solution_step}: {json.dumps(predicted_config, sort_keys=True)}",
                    flush=True,
                )

                validation_args = hp_search.parse_args([])
                validation_args.dataset = validation_id
                validation_args.n_epochs = args.n_epochs
                validation_args.n_repeats = args.n_repeats
                validation_args.device = args.device
                validation_args.seed = args.seed + 10000 + validation_index
                validation_args.no_wandb = True
                validation_args.combine_test = False
                validation_args.max_warmup = max(1, min(50, args.n_epochs))
                validation_args.bs = recommended_batch_size(
                    batches_v,
                    cap=validation_args.bs,
                )
                validation_args.resolved_n_repeats = hp_search.resolve_n_repeats(
                    args.n_repeats,
                    batches_v,
                )
                validation_args.results_dir = str(
                    args.output_dir / "validation" / validation_id / "online_meta_network"
                )

                validation_metrics = {}
                try:
                    valid_mcc, validation_metrics = hp_search.run_trial(
                        predicted_config,
                        validation_args,
                        (Xv, yv, batches_v),
                        f"online_meta_validation_{validation_id}_s{solution_step}",
                        fixed_test_data=validation_fixed_tests[validation_id],
                    )
                    valid_mcc = float(valid_mcc)
                    test_mcc = float(validation_metrics.get("test_mcc", np.nan))
                except Exception as exc:
                    valid_mcc = -1.0
                    test_mcc = np.nan
                    print(
                        f"[optuna-comparison] meta validation failed "
                        f"validation={validation_id}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )

                validation_history[validation_id].append(
                    (solution_step, valid_mcc, test_mcc)
                )
                validation_payload[
                    f"validation/{validation_id}/valid_mcc"
                ] = valid_mcc
                validation_payload[
                    f"validation/{validation_id}/test_mcc"
                ] = test_mcc
                validation_payload[
                    f"validation/{validation_id}/meta_train_loss"
                ] = meta_diagnostics["meta_train_loss"]

                for source_id, source_score in meta_diagnostics[
                    "source_best_scores"
                ].items():
                    validation_payload[
                        f"validation/{validation_id}/source_best_valid_mcc/{source_id}"
                    ] = source_score

                _add_config_to_payload(
                    validation_payload,
                    f"validation/{validation_id}/predicted_hparams",
                    predicted_config,
                )

                print(
                    f"[optuna-comparison] META VALIDATION {validation_id}: "
                    f"valid_mcc={valid_mcc:.4f}, test_mcc={test_mcc:.4f}",
                    flush=True,
                )

            if wandb_run is not None and len(validation_payload) > 1:
                wandb_run.log(validation_payload)

        # --------------------------------------------------------------
        # NEW TPE TRAINING TRIALS -- train datasets only
        # --------------------------------------------------------------
        current = {}
        for dataset_index, dataset_id in enumerate(dataset_ids):
            completed = _completed_trials(studies[dataset_id])
            if len(completed) <= solution_step:
                trial = studies[dataset_id].ask()
                hp_args = hp_search.parse_args([])
                hp_args.dataset = dataset_id
                hp_args.n_epochs = args.n_epochs
                hp_args.n_repeats = args.n_repeats
                hp_args.device = args.device
                hp_args.seed = args.seed + dataset_index
                hp_args.no_wandb = True
                hp_args.combine_test = False
                hp_args.max_warmup = max(1, min(50, args.n_epochs))
                _, _, batches = datasets[dataset_id]
                hp_args.bs = recommended_batch_size(batches, cap=hp_args.bs)
                hp_args.resolved_n_repeats = hp_search.resolve_n_repeats(
                    args.n_repeats,
                    batches,
                )
                config = hp_search.sample_config(trial, hp_args)
                started = time.monotonic()
                error = None
                metrics = {}
                try:
                    experiment_namespace = metadata["wandb_run_id"] or args.output_dir.name
                    score, metrics = hp_search.run_trial(
                        config,
                        hp_args,
                        datasets[dataset_id],
                        f"classic_optuna_{experiment_namespace}_{dataset_id}_t{trial.number}",
                        fixed_test_data=fixed_tests[dataset_id],
                    )
                except Exception as exc:
                    score = -1.0
                    error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[optuna-comparison] {dataset_id} trial "
                        f"{trial.number} failed: {error}",
                        flush=True,
                    )
                trial.set_user_attr("config", config)
                trial.set_user_attr(
                    "test_mcc",
                    float(metrics.get("test_mcc", np.nan)),
                )
                trial.set_user_attr(
                    "fit_seconds",
                    time.monotonic() - started,
                )
                if error:
                    trial.set_user_attr("error", error)
                studies[dataset_id].tell(trial, float(score))
                completed = _completed_trials(studies[dataset_id])

            current[dataset_id] = _trial_payload(completed[solution_step])

        if len(history[dataset_ids[0]]) > solution_step:
            continue

        best = {dataset_id: _best_payload(studies[dataset_id]) for dataset_id in dataset_ids}
        record = {
            "solution_step": solution_step,
            "wall_clock_seconds": time.time() - float(metadata["created_at_unix"]),
            "current": current,
            "best": best,
            "total_valid_mcc": _aggregate_solution_metric(current, dataset_ids, "valid_mcc"),
            "total_test_mcc": _aggregate_solution_metric(current, dataset_ids, "test_mcc"),
            "best_total_valid_mcc": _aggregate_solution_metric(best, dataset_ids, "valid_mcc"),
            "best_total_test_mcc": _aggregate_solution_metric(best, dataset_ids, "test_mcc"),
        }
        _append_solution(args.output_dir, record, dataset_ids)
        for dataset_id in dataset_ids:
            row = current[dataset_id]
            history[dataset_id].append((solution_step, row["valid_mcc"], row["test_mcc"]))
        if wandb_run is not None:
            telemetry = {}
            for dataset_id in dataset_ids:
                telemetry.update(_numeric_config_telemetry(dataset_id, current[dataset_id]["config"]))
            payload = {
                "solution_step": solution_step,
                "comparison/wall_clock_seconds": record["wall_clock_seconds"],
                "comparison/model_fits_completed": float((solution_step + 1) * len(dataset_ids)),
                **telemetry,
            }
            for dataset_id in dataset_ids:
                payload.update({
                    f"solutions/metrics/{dataset_id}/valid_mcc": current[dataset_id]["valid_mcc"],
                    f"solutions/metrics/{dataset_id}/test_mcc": current[dataset_id]["test_mcc"],
                    f"solutions/best/{dataset_id}/valid_mcc": best[dataset_id]["valid_mcc"],
                    f"solutions/best/{dataset_id}/test_mcc": best[dataset_id]["test_mcc"],
                    f"comparison/fit_seconds/{dataset_id}": current[dataset_id]["fit_seconds"],
                    f"solutions/mcc/{dataset_id}": _score_figure(dataset_id, history[dataset_id]),
                })
                payload.update({
                    "solutions/total_valid_mcc": record["total_valid_mcc"],
                    "solutions/total_test_mcc": record["total_test_mcc"],
                    "solutions/best_total_valid_mcc": record["best_total_valid_mcc"],
                    "solutions/best_total_test_mcc": record["best_total_test_mcc"],
                })
            wandb_run.log(payload)
        print(f"[optuna-comparison] solution {solution_step} complete: {json.dumps(record)}", flush=True)

    return 0


def main(argv=None) -> int:
    try:
        return _main(argv)
    finally:
        _cleanup_runtime_resources()


if __name__ == "__main__":
    raise SystemExit(main())