#!/usr/bin/env python3
"""Run 20 independent BERNN Optuna studies across all leaderboard datasets.

Each dataset has its own Optuna study, SQLite file, result directory, and W&B
run. The launcher assigns one dataset process per GPU and queues the newly added
datasets first.

Existing datasets keep the meta-hpo-bank fixed-external protocol: Optuna sees
only grouped-CV validation MCC, while the labeled *_inference.csv cross-test is
monitoring-only. Most new whole-dataset benchmarks use cyclic batch
train/valid/test. The high-concentration mz10 task uses grouped train/validation
CV with training-fold-only feature selection and no per-trial test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hp_search
from src.evolutionary_meta import recommended_batch_size
from src.run_provenance import capture_run_provenance

DATASETS = (
    # New datasets first.
    ("bacteria_2024_mz10", "grouped_cv"),
    ("jdlber_sle_maldi", "cyclic"),
    ("seqc_maqc", "cyclic"),
    # Existing meta-hpo-bank datasets.
    ("normal_tissue_878", "fixed_external"),
    ("colon_3041", "fixed_external"),
    ("massbench_adenocarcinoma", "fixed_external"),
    ("massbench_benchmark", "fixed_external"),
    ("massbench_alzheimer", "fixed_external"),
)
CYCLIC_CV_FOLDS = {
    # Large whole-dataset benchmarks use grouped three-fold cyclic CV instead
    # of one round per acquisition batch.
    "scib_pancreas": 3,
}
GROUPED_CV_FOLDS = {"bacteria_2024_mz10": 5}
GROUPED_FEATURE_COUNTS = {"bacteria_2024_mz10": 10_000}
GROUPED_FEATURE_METHODS = {"bacteria_2024_mz10": "hybrid_xgboost_f"}
MZ10_SANITY_MIN_VALID_MCC = 0.35
MZ10_SANITY_MIN_TRAIN_MCC = 0.90
CV_FOLDS = {
    "normal_tissue_878": 3,
    "colon_3041": 3,
    "massbench_adenocarcinoma": 2,
    "massbench_benchmark": 3,
    "massbench_alzheimer": 3,
}
PREPARE = {
    "jdlber_sle_maldi": [sys.executable, str(ROOT / "scripts" / "prepare_jdlber_sle_maldi.py")],
    "seqc_maqc": [sys.executable, str(ROOT / "scripts" / "download_prepare_seqc_maqc.py")],
    "scib_pancreas": [sys.executable, str(ROOT / "scripts" / "download_prepare_scib_pancreas.py")],
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "independent_optuna_all_20")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=1000)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--datasets", default=None, help="Optional comma-separated subset.")
    p.add_argument("--prepare-missing", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--seed-trials-from", type=Path, action="append", default=[],
        help=("Import compatible completed trials.json records into a new study "
              "as TPE evidence without retraining them; repeatable."),
    )
    p.add_argument(
        "--auto-seed-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically reuse compatible local historical trials (default: enabled).",
    )
    p.add_argument(
        "--seed-from-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Best-effort import of compatible trials.json files from prior W&B runs.",
    )
    p.add_argument(
        "--pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable epoch pruning plus 1 -> 2 -> all-repeat promotion (default: enabled).",
    )
    p.add_argument("--repeat1-prune-percentile", type=float, default=25.0)
    p.add_argument("--repeat2-prune-percentile", type=float, default=50.0)
    p.add_argument("--prune-min-reference-trials", type=int, default=5)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="BE_leaderboard_meta_evolution")
    p.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", "adlab"))
    p.add_argument("--wandb-group", default="independent-optuna-all-datasets-20")
    p.add_argument(
        "--feature-select-method",
        choices=("f_classif", "xgboost_gain", "hybrid_xgboost_f"),
        default=None,
        help="Override the grouped-CV training-fold feature selector.",
    )
    p.add_argument(
        "--mz10-search-space",
        choices=(
            "plain",
            "extended_nonvariational",
            "highrange_plain",
            "highrange_extended_nonvariational",
        ),
        default="plain",
        help="Choose the mz10 BERNN family search; extended mode keeps VAE disabled.",
    )
    p.add_argument("--worker-dataset", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--worker-protocol",
        choices=("fixed_external", "cyclic", "grouped_cv"),
        help=argparse.SUPPRESS,
    )
    return p.parse_args(argv)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def save_wandb_files(run, paths, *, base_path: Path) -> int:
    """Upload explicit files or directory trees into the run's Files tab."""
    files = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
    unique_files = sorted({path.resolve() for path in files})
    for path in unique_files:
        run.save(
            str(path),
            base_path=str(base_path.resolve()),
            policy="now",
            glob=False,
        )
    return len(unique_files)


def selected_datasets(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return list(DATASETS)
    protocol = dict(DATASETS)
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in protocol]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}; known={list(protocol)}")
    return [(name, protocol[name]) for name in names]


def required_path(dataset: str, protocol: str) -> Path:
    base = ROOT / "data" / "datasets" / dataset
    if protocol in {"cyclic", "grouped_cv"}:
        sparse_matrix = base / f"{dataset}_features_csr.npz"
        sparse_metadata = base / f"{dataset}_metadata.csv"
        sparse_names = base / f"{dataset}_feature_names.npy"
        if all(path.exists() for path in (sparse_matrix, sparse_metadata, sparse_names)):
            return sparse_matrix
    suffix = "_all.csv" if protocol in {"cyclic", "grouped_cv"} else "_train.csv"
    return base / f"{dataset}{suffix}"


def prepare_missing(dataset: str, protocol: str) -> None:
    target = required_path(dataset, protocol)
    if target.exists():
        return
    command = PREPARE.get(dataset)
    if not command:
        raise FileNotFoundError(f"Missing {target}; no automatic preparer is registered")
    print(f"[prepare] {dataset}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not target.exists():
        raise FileNotFoundError(f"Preparation did not create {target}")


def completed(study):
    import optuna
    return [
        t for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


def mz10_sanity_gate(metrics: dict, valid_mcc: float) -> tuple[bool, dict]:
    """Require basic learning before spending the remaining mz10 HPO budget."""
    train_mcc = metrics.get(
        "mcc_train_all_concentrations", metrics.get("train_mcc")
    )
    valid_ok = float(valid_mcc) >= MZ10_SANITY_MIN_VALID_MCC
    train_ok = train_mcc is None or float(train_mcc) >= MZ10_SANITY_MIN_TRAIN_MCC
    details = {
        "passed": bool(valid_ok and train_ok),
        "valid_mcc": float(valid_mcc),
        "train_mcc": None if train_mcc is None else float(train_mcc),
        "min_valid_mcc": MZ10_SANITY_MIN_VALID_MCC,
        "min_train_mcc": MZ10_SANITY_MIN_TRAIN_MCC,
    }
    return details["passed"], details


def interrupted_trials(study):
    """Return RUNNING trials so --resume can continue their completed repeats."""
    import optuna
    return [
        trial for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.RUNNING
    ]


def mark_interrupted_failed(study) -> None:
    """Legacy escape hatch; normal --resume no longer calls this."""
    import optuna
    for trial in interrupted_trials(study):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        print(f"[resume] marked trial {trial.number} FAIL", flush=True)


def discover_local_seed_trials(dataset: str, current_output: Path) -> list[Path]:
    """Find prior persisted trial banks for this dataset without touching current output."""
    results_root = ROOT / "results"
    if not results_root.exists():
        return []
    current = (current_output / "trials.json").resolve()
    found = []
    for path in results_root.rglob("trials.json"):
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved == current:
            continue
        if dataset not in str(path.parent):
            continue
        found.append(path)
    return sorted(set(found))


def download_wandb_seed_trials(args, dataset: str, protocol: str, output_dir: Path) -> list[Path]:
    """Best-effort recovery of historical trials.json files already uploaded to W&B."""
    if not bool(getattr(args, "seed_from_wandb", True)):
        return []
    try:
        import wandb

        api = wandb.Api(timeout=20)
        project = f"{args.wandb_entity}/{args.wandb_project}"
        runs = api.runs(project, filters={"config.dataset": dataset}, per_page=50)
        root = output_dir / "seed_history" / "wandb"
        paths = []
        for run in runs:
            # Never seed a new study from a currently active W&B run. Finished
            # Optuna trials inside older finished/crashed runs are safe to reuse.
            if str(getattr(run, "state", "")).lower() == "running":
                continue
            config = dict(getattr(run, "config", {}) or {})
            run_protocol = config.get("protocol")
            if run_protocol and str(run_protocol) != str(protocol):
                continue
            run_root = root / str(run.id)
            run_root.mkdir(parents=True, exist_ok=True)
            for remote in run.files():
                if not str(remote.name).endswith("trials.json"):
                    continue
                downloaded = remote.download(root=str(run_root), replace=True)
                candidate = Path(downloaded.name)
                if candidate.exists():
                    paths.append(candidate)

            # Older runs did not always persist valid_mcc_folds into trials.json,
            # but the launcher logged those values and the sampled config to W&B
            # history. Reconstruct a lightweight seed bank from that history.
            config_keys = (
                "dloss", "variational", "kan", "class_triplet", "class_triplet_w",
                "lr", "wd", "nu", "smoothing", "margin", "dropout", "thres",
                "warmup", "n_layers", "layer1", "scaler", "gamma", "beta",
            )
            history_keys = [
                "trial_number", "metrics/valid_mcc", "metrics/test_mcc",
                "runtime/fit_seconds",
                *[f"folds/valid_mcc/fold_{i}" for i in range(5)],
                *[f"config/{key}" for key in config_keys],
            ]
            history_records = []
            try:
                for item in run.scan_history(keys=history_keys, page_size=200):
                    valid = item.get("metrics/valid_mcc")
                    if valid is None or not math.isfinite(float(valid)):
                        continue
                    sampled = {
                        key: item.get(f"config/{key}")
                        for key in config_keys
                        if item.get(f"config/{key}") is not None
                    }
                    folds = []
                    for i in range(5):
                        value = item.get(f"folds/valid_mcc/fold_{i}")
                        if value is None or not math.isfinite(float(value)):
                            break
                        folds.append(float(value))
                    history_records.append({
                        "protocol": protocol,
                        "valid_mcc": float(valid),
                        "test_mcc": item.get("metrics/test_mcc"),
                        "fit_seconds": item.get("runtime/fit_seconds"),
                        "config": sampled,
                        "metrics": {"valid_mcc_folds": folds} if folds else {},
                    })
            except Exception as exc:
                print(
                    f"[transfer seed] W&B history scan skipped for {run.id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if history_records:
                history_path = run_root / "wandb_history_trials.json"
                atomic_json(history_path, history_records)
                paths.append(history_path)
        if paths:
            print(f"[transfer seed] recovered {len(paths)} W&B trial file(s)", flush=True)
        return paths
    except Exception as exc:
        print(f"[transfer seed] W&B history unavailable: {type(exc).__name__}: {exc}", flush=True)
        return []


def _seed_trial_path(path: Path, dataset: str) -> Path:
    if path.is_file():
        return path
    direct = path / "trials.json"
    nested = path / dataset / "trials.json"
    if direct.exists():
        return direct
    return nested


def _highrange_seed_params(config: dict, search_space: str):
    """Return current mz10-space params/distributions, or None if incompatible."""
    import optuna

    if search_space in {"plain", "extended_nonvariational"}:
        if bool(config.get("variational", False)):
            return None
        try:
            dloss = str(config.get("dloss", "no"))
            class_triplet = bool(config.get("class_triplet", False))
            params = {
                "lr": float(config["lr"]),
                "wd": float(config["wd"]),
                "smoothing": float(config["smoothing"]),
                "dropout": float(config["dropout"]),
                "warmup": int(config["warmup"]),
                "n_layers": int(config["n_layers"]),
                "layer1": int(config["layer1"]),
                "scaler": str(config["scaler"]),
            }
        except (KeyError, TypeError, ValueError):
            return None
        distributions = {
            "lr": optuna.distributions.FloatDistribution(3e-4, 3e-3, log=True),
            "wd": optuna.distributions.FloatDistribution(1e-6, 1e-4, log=True),
            "smoothing": optuna.distributions.FloatDistribution(0.0, 0.05),
            "dropout": optuna.distributions.FloatDistribution(
                0.0, 0.2 if search_space == "plain" else 0.2
            ),
            "warmup": optuna.distributions.IntDistribution(10, 50),
            "n_layers": optuna.distributions.CategoricalDistribution((1, 2)),
            "layer1": optuna.distributions.IntDistribution(256, 512, step=64),
            "scaler": optuna.distributions.CategoricalDistribution(("standard", "robust")),
        }
        if search_space == "plain":
            if dloss != "no" or bool(config.get("kan", False)) or class_triplet:
                return None
            try:
                params["nu"] = float(config["nu"])
            except (KeyError, TypeError, ValueError):
                return None
            distributions["nu"] = optuna.distributions.FloatDistribution(0.25, 4.0, log=True)
        else:
            allowed = ("no", "inverseTriplet", "DANN", "normae", "revDANN", "revTriplet")
            if dloss not in allowed:
                return None
            params.update({
                "dloss": dloss,
                "kan": bool(config.get("kan", False)),
                "class_triplet": class_triplet,
                "margin": float(config.get("margin", 1.0)),
            })
            distributions.update({
                "dloss": optuna.distributions.CategoricalDistribution(allowed),
                "kan": optuna.distributions.CategoricalDistribution((False, True)),
                "class_triplet": optuna.distributions.CategoricalDistribution((False, True)),
                "margin": optuna.distributions.FloatDistribution(0.5, 5.0),
            })
            if class_triplet:
                params["class_triplet_w"] = float(config.get("class_triplet_w", 0.0))
                distributions["class_triplet_w"] = optuna.distributions.FloatDistribution(
                    0.05, 1.0, log=True
                )
            if dloss != "no":
                params["nu"] = float(config.get("nu", 0.0))
                distributions["nu"] = optuna.distributions.FloatDistribution(
                    1e-3, 0.5, log=True
                )
            if dloss in hp_search.ADVERSARIAL_DLOSS:
                params["gamma"] = float(config.get("gamma", 0.0))
                distributions["gamma"] = optuna.distributions.FloatDistribution(
                    1e-3, 0.5, log=True
                )
        try:
            for name, value in params.items():
                dist = distributions[name]
                if not dist._contains(dist.to_internal_repr(value)):
                    return None
        except (TypeError, ValueError):
            return None
        return params, distributions

    layer1 = int(config.get("layer1", -1))
    warmup = int(config.get("warmup", -1))
    dropout = float(config.get("dropout", float("nan")))
    if not (512 <= layer1 <= 2048 and (layer1 - 512) % 128 == 0):
        return None
    if not (1 <= warmup <= 150 and 0.0 <= dropout <= 0.5):
        return None
    if bool(config.get("variational", False)):
        return None

    params = {
        "lr": float(config["lr"]), "wd": float(config["wd"]),
        "smoothing": float(config["smoothing"]), "dropout": dropout,
        "warmup": warmup, "n_layers": int(config["n_layers"]),
        "layer1": layer1, "scaler": str(config["scaler"]),
    }
    distributions = {
        "lr": optuna.distributions.FloatDistribution(3e-4, 3e-3, log=True),
        "wd": optuna.distributions.FloatDistribution(1e-6, 1e-4, log=True),
        "smoothing": optuna.distributions.FloatDistribution(0.0, 0.05),
        "dropout": optuna.distributions.FloatDistribution(0.0, 0.5),
        "warmup": optuna.distributions.IntDistribution(1, 150),
        "n_layers": optuna.distributions.CategoricalDistribution((1, 2)),
        "layer1": optuna.distributions.IntDistribution(512, 2048, step=128),
        "scaler": optuna.distributions.CategoricalDistribution(("standard", "robust")),
    }
    if search_space == "highrange_plain":
        if any((config.get("dloss", "no") != "no", bool(config.get("kan", False)),
                bool(config.get("class_triplet", False)))):
            return None
        return params, distributions
    if search_space != "highrange_extended_nonvariational":
        return None

    dloss = str(config.get("dloss", "no"))
    allowed = ("no", "inverseTriplet", "DANN", "normae", "revDANN", "revTriplet")
    if dloss not in allowed:
        return None
    params.update({
        "dloss": dloss,
        "kan": bool(config.get("kan", False)),
        "class_triplet": bool(config.get("class_triplet", False)),
        "margin": float(config.get("margin", 1.0)),
    })
    distributions.update({
        "dloss": optuna.distributions.CategoricalDistribution(allowed),
        "kan": optuna.distributions.CategoricalDistribution((False, True)),
        "class_triplet": optuna.distributions.CategoricalDistribution((False, True)),
        "margin": optuna.distributions.FloatDistribution(0.5, 5.0),
    })
    if params["class_triplet"]:
        params["class_triplet_w"] = float(config.get("class_triplet_w", 0.0))
        distributions["class_triplet_w"] = optuna.distributions.FloatDistribution(0.05, 1.0, log=True)
    if dloss != "no":
        params["nu"] = float(config.get("nu", 0.0))
        distributions["nu"] = optuna.distributions.FloatDistribution(1e-3, 0.5, log=True)
    if dloss in hp_search.ADVERSARIAL_DLOSS:
        params["gamma"] = float(config.get("gamma", 0.0))
        distributions["gamma"] = optuna.distributions.FloatDistribution(1e-3, 0.5, log=True)
    try:
        for name, value in params.items():
            if not distributions[name]._contains(distributions[name].to_internal_repr(value)):
                return None
    except (TypeError, ValueError):
        return None
    return params, distributions


def _generic_seed_params(config: dict, max_warmup: int):
    """Map a historical generic BERNN config into the current Optuna space."""
    import optuna

    try:
        dloss = str(config["dloss"])
        variational = bool(config["variational"])
        params = {
            "dloss": dloss,
            "variational": variational,
            "kan": bool(config["kan"]),
            "class_triplet": bool(config["class_triplet"]),
            "class_triplet_w": float(config.get("class_triplet_w", 0.0)),
            "lr": float(config["lr"]),
            "wd": float(config["wd"]),
            "nu": float(config["nu"]),
            "smoothing": float(config["smoothing"]),
            "margin": float(config["margin"]),
            "dropout": float(config["dropout"]),
            "thres": float(config["thres"]),
            "warmup": int(config["warmup"]),
            "n_layers": int(config["n_layers"]),
            "layer1": int(config["layer1"]),
            "scaler": str(config["scaler"]),
        }
    except (KeyError, TypeError, ValueError):
        return None

    distributions = {
        "dloss": optuna.distributions.CategoricalDistribution(tuple(hp_search.DLOSS_CHOICES)),
        "variational": optuna.distributions.CategoricalDistribution((False, True)),
        "kan": optuna.distributions.CategoricalDistribution((False, True)),
        "class_triplet": optuna.distributions.CategoricalDistribution((False, True)),
        "class_triplet_w": optuna.distributions.FloatDistribution(0.0, 1.0),
        "lr": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
        "wd": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
        "nu": optuna.distributions.FloatDistribution(1e-4, 1e2),
        "smoothing": optuna.distributions.FloatDistribution(0.0, 0.2),
        "margin": optuna.distributions.FloatDistribution(0.0, 10.0),
        "dropout": optuna.distributions.FloatDistribution(0.0, 0.5),
        "thres": optuna.distributions.FloatDistribution(0.0, 0.1),
        "warmup": optuna.distributions.IntDistribution(1, int(max_warmup)),
        "n_layers": optuna.distributions.CategoricalDistribution((1, 2, 3, 4, 5)),
        "layer1": optuna.distributions.IntDistribution(512, 1024),
        "scaler": optuna.distributions.CategoricalDistribution(tuple(hp_search.SCALER_CHOICES)),
    }
    if dloss in hp_search.ADVERSARIAL_DLOSS:
        try:
            params["gamma"] = float(config["gamma"])
        except (KeyError, TypeError, ValueError):
            return None
        distributions["gamma"] = optuna.distributions.FloatDistribution(1e-2, 1e2, log=True)
    if variational:
        try:
            params["beta"] = float(config["beta"])
        except (KeyError, TypeError, ValueError):
            return None
        distributions["beta"] = optuna.distributions.FloatDistribution(1e-2, 1e2, log=True)

    try:
        for name, value in params.items():
            dist = distributions[name]
            if not dist._contains(dist.to_internal_repr(value)):
                return None
    except (TypeError, ValueError):
        return None
    return params, distributions


def _compatible_seed_params(config: dict, dataset: str, search_space: str, max_warmup: int):
    if dataset == "bacteria_2024_mz10":
        return _highrange_seed_params(config, search_space)
    return _generic_seed_params(config, max_warmup)


def import_seed_trials(
    study,
    paths,
    dataset: str,
    protocol: str,
    search_space: str,
    expected_repeats: int = 5,
    max_warmup: int = 50,
) -> int:
    """Import compatible completed results as Optuna observations, never reruns."""
    import optuna

    existing = {
        json.dumps(t.params, sort_keys=True, default=str)
        for t in study.get_trials(deepcopy=False)
    }
    imported = 0
    for raw in paths:
        path = _seed_trial_path(Path(raw), dataset)
        if not path.exists():
            raise FileNotFoundError(f"Seed trial file not found: {path}")
        for row in json.loads(path.read_text()):
            if row.get("protocol") != protocol or row.get("valid_mcc") is None:
                continue
            config = dict(row.get("config", {}))
            compatible = _compatible_seed_params(
                config, dataset, search_space, max_warmup
            )
            folds = row.get("metrics", {}).get("valid_mcc_folds", [])
            if compatible is None:
                continue
            if folds and len(folds) != int(expected_repeats):
                continue
            params, distributions = compatible
            signature = json.dumps(params, sort_keys=True, default=str)
            if signature in existing:
                continue
            intermediate = {
                hp_search.FOLD_PRUNE_STEP_BASE + index: float(np.mean(folds[:index]))
                for index in range(1, len(folds) + 1)
            }
            frozen = optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=float(row["valid_mcc"]),
                intermediate_values=intermediate,
                user_attrs={
                    "transfer_seed": True,
                    "transfer_seed_source": str(path),
                    "config": config,
                    "metrics": row.get("metrics", {}),
                    "fit_seconds": row.get("fit_seconds"),
                },
            )
            study.add_trial(frozen)
            existing.add(signature)
            imported += 1
    return imported


def json_metric(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            converted = json_metric(item)
            if converted is not None:
                out.append(converted)
        return out
    if isinstance(value, str):
        return value
    return None


def compact_metrics(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if str(key).startswith("_epoch_traces"):
            continue
        converted = json_metric(value)
        if converted is not None:
            out[str(key)] = converted
    return out


def wandb_metrics(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        converted = json_metric(value)
        if isinstance(converted, (int, float)):
            out[f"metrics/{key}"] = converted
    for split in ("valid", "test"):
        values = metrics.get(f"{split}_mcc_folds", [])
        if isinstance(values, (list, tuple)):
            for fold, value in enumerate(values):
                converted = json_metric(value)
                if converted is not None:
                    out[f"folds/{split}_mcc/fold_{fold}"] = converted
    return out


def config_metrics(config: dict) -> dict:
    out = {}
    for key, value in config.items():
        if isinstance(value, (bool, np.bool_)):
            out[f"config/{key}"] = int(value)
        elif isinstance(value, (int, float, np.integer, np.floating, str)):
            out[f"config/{key}"] = value
    return out


def records_for(study, protocol: str) -> list[dict]:
    rows = []
    for index, trial in enumerate(sorted(completed(study), key=lambda x: x.number)):
        attrs = dict(trial.user_attrs)
        metrics = dict(attrs.get("metrics", {}))
        rows.append({
            "trial_index": index,
            "trial_number": int(trial.number),
            "valid_mcc": float(trial.value),
            "test_mcc": metrics.get("test_mcc"),
            "fit_seconds": attrs.get("fit_seconds"),
            "protocol": protocol,
            "config": attrs.get("config", {}),
            "metrics": metrics,
            "error": attrs.get("error"),
            "selection_metric": "valid_mcc",
            "test_role": "monitoring_only_excluded_from_optuna",
        })
    return rows


def persist_trials(output_dir: Path, rows: list[dict]) -> None:
    atomic_json(output_dir / "trials.json", rows)
    fields = (
        "trial_index", "trial_number", "valid_mcc", "test_mcc",
        "valid_mcc_folds", "test_mcc_folds",
        "valid_mcc_std", "test_mcc_std", "test_mcc_fold_mean",
        "test_mcc_fold_std", "test_mcc_global_oof", "fit_seconds",
        "protocol", "error",
    )
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "trial_index": row["trial_index"],
                "trial_number": row["trial_number"],
                "valid_mcc": row["valid_mcc"],
                "test_mcc": row["test_mcc"],
                "valid_mcc_folds": json.dumps(metrics.get("valid_mcc_folds", [])),
                "test_mcc_folds": json.dumps(metrics.get("test_mcc_folds", [])),
                "valid_mcc_std": metrics.get("valid_mcc_std"),
                "test_mcc_std": metrics.get("test_mcc_std"),
                "test_mcc_fold_mean": metrics.get("test_mcc_fold_mean"),
                "test_mcc_fold_std": metrics.get("test_mcc_fold_std"),
                "test_mcc_global_oof": metrics.get("test_mcc_global_oof"),
                "fit_seconds": row["fit_seconds"],
                "protocol": row["protocol"],
                "error": row["error"],
            })


def run_worker(args) -> int:
    dataset, protocol = args.worker_dataset, args.worker_protocol
    feature_select_method = (
        args.feature_select_method or GROUPED_FEATURE_METHODS.get(dataset)
    )
    out = args.output_dir / dataset
    out.mkdir(parents=True, exist_ok=True)
    if args.prepare_missing:
        prepare_missing(dataset, protocol)
    target = required_path(dataset, protocol)
    if not target.exists():
        raise FileNotFoundError(f"Missing {target}; use --prepare-missing or prepare it first")

    dataset_files = [target]
    if protocol in {"cyclic", "grouped_cv"}:
        target = required_path(dataset, protocol)
        if target.suffix == ".npz":
            source_file = target.name
            data = hp_search.load_cyclic_dataset(dataset)
        else:
            source_file = f"{dataset}_all.csv"
            data = hp_search.load_cyclic_dataset(dataset, source_file=source_file)
        fixed_test = None
    else:
        source_file = f"{dataset}_train.csv"
        inference = ROOT / "data" / "datasets" / dataset / f"{dataset}_inference.csv"
        if not inference.exists():
            raise FileNotFoundError(f"Missing labeled fixed cross-test file {inference}")
        dataset_files.append(inference)
        data = hp_search.load_dataset(dataset)
        fixed_test = hp_search.load_fixed_test_dataset(dataset)
    for seed_path in args.seed_trials_from:
        resolved_seed = _seed_trial_path(Path(seed_path), dataset)
        if resolved_seed.exists():
            dataset_files.append(resolved_seed)

    provenance_snapshot, provenance = capture_run_provenance(
        repo_root=ROOT,
        output_dir=out,
        dataset_files=dataset_files,
        argv=[sys.executable, *sys.argv],
    )

    X, _, batches = data
    batch_values = sorted(np.unique(np.asarray(batches).astype(str)).tolist())
    meta_path = out / "run_metadata.json"
    exists = meta_path.exists()
    if exists and not args.resume:
        raise FileExistsError(f"{out} already exists; pass --resume")
    meta = json.loads(meta_path.read_text()) if exists else {}
    meta.update({
        "schema_version": 1,
        "run_id": meta.get("run_id") or uuid.uuid4().hex[:10],
        "dataset": dataset,
        "protocol": protocol,
        "source_file": source_file,
        "n_trials": args.n_trials,
        "n_epochs": args.n_epochs,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_batches": len(batch_values),
        "batches": batch_values,
        "selection_metric": "valid_mcc",
        "pruning": {
            "enabled": bool(args.pruning),
            "warmup_pruning": False,
            "classifier_patience": 30,
            "epoch_checks": "classifier epochs 10,15,20,... on promotion repeats 1-2 only",
            "minimum_reference_trials": int(args.prune_min_reference_trials),
            "epoch_reference_percentile": hp_search.PRUNE_PERCENTILE,
            "repeat1_reference_percentile": float(args.repeat1_prune_percentile),
            "repeat2_reference_percentile": float(args.repeat2_prune_percentile),
            "promotion_schedule": "repeat 1 -> repeat 2 -> all remaining repeats",
            "checkpoint_resume": True,
        },
        "test_role": (
            "not_used_grouped_cv" if protocol == "grouped_cv"
            else "monitoring_only_excluded_from_optuna"
        ),
        "feature_selection": (
            {
                "method": feature_select_method,
                "k": GROUPED_FEATURE_COUNTS.get(dataset),
                "fit_scope": "training_fold_only",
                "xgboost_source": "MSML3 mz10 parameters; refit per fold",
            }
            if protocol == "grouped_cv" else None
        ),
        "mz10_search_space": (
            args.mz10_search_space if dataset == "bacteria_2024_mz10" else None
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wandb_run_id": meta.get("wandb_run_id") or uuid.uuid4().hex[:8],
        "provenance": {
            "launch_id": provenance["launch_id"],
            "manifest": str(provenance_snapshot / "manifest.json"),
            "repository_file_count": len(provenance["repository_files"]),
            "training_code_file_count": len(provenance["training_code_files"]),
            "training_code_log": provenance["training_code_log"],
            "bernn_file_count": len(provenance["bernn_files"]),
            "dataset_files": provenance["dataset_files"],
        },
    })
    atomic_json(meta_path, meta)

    import mlflow
    import optuna

    mlruns = out / "mlruns"
    mlruns.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlruns.resolve().as_uri())
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{(out / 'optuna.sqlite3').resolve()}")
    study = optuna.create_study(
        study_name=f"independent_{dataset}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=5),
        storage=storage,
        load_if_exists=True,
    )

    if protocol == "fixed_external":
        seed_requested = CV_FOLDS.get(dataset, args.n_repeats)
    elif protocol == "grouped_cv":
        seed_requested = GROUPED_CV_FOLDS[dataset]
    else:
        seed_requested = CYCLIC_CV_FOLDS.get(dataset, len(batch_values))
    expected_repeats = hp_search.resolve_n_repeats(seed_requested, batches)
    warmup_cap = 150 if args.mz10_search_space.startswith("highrange_") else 50
    max_warmup = max(1, min(warmup_cap, args.n_epochs))

    seed_paths = list(args.seed_trials_from)
    if args.auto_seed_history:
        seed_paths.extend(discover_local_seed_trials(dataset, out))
    seed_paths.extend(download_wandb_seed_trials(args, dataset, protocol, out))
    # Preserve order but avoid importing the same file twice.
    seed_paths = list(dict.fromkeys(Path(path) for path in seed_paths))
    imported_seed_trials = import_seed_trials(
        study,
        seed_paths,
        dataset,
        protocol,
        args.mz10_search_space,
        expected_repeats=expected_repeats,
        max_warmup=max_warmup,
    )
    if imported_seed_trials:
        print(
            f"[transfer seed] imported {imported_seed_trials} compatible completed trials "
            f"before new sampling",
            flush=True,
        )
    meta["hpo_bootstrap"] = {
        "compatible_seed_trials": int(imported_seed_trials),
        "expected_repeats": int(expected_repeats),
        "seed_sources": [str(path) for path in seed_paths],
        "auto_local_history": bool(args.auto_seed_history),
        "wandb_history": bool(args.seed_from_wandb),
    }
    atomic_json(meta_path, meta)

    wb = None
    if not args.no_wandb:
        import wandb
        wb = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"optuna-{dataset}-{args.n_trials}",
            id=meta["wandb_run_id"],
            resume="allow",
            config={**meta, "batch_size_cap": args.batch_size, "num_workers": args.num_workers},
            tags=["independent_optuna", dataset, protocol],
        )
        wandb.define_metric("trial_index")
        wandb.define_metric("metrics/*", step_metric="trial_index")
        wandb.define_metric("folds/*", step_metric="trial_index")
        wandb.define_metric("best/*", step_metric="trial_index")
        artifact_name = f"{dataset}-run-provenance-{meta['wandb_run_id']}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="run-provenance",
            description="Exact source, BERNN, environment, git, and data-hash snapshot used by this launch.",
            metadata={
                "launch_id": provenance["launch_id"],
                "dataset": dataset,
                "protocol": protocol,
                "git_head": provenance["git"]["head"].get("output", "").strip(),
                "dataset_files": provenance["dataset_files"],
            },
        )
        artifact.add_dir(str(provenance_snapshot))
        wb.log_artifact(artifact, aliases=["latest", provenance["launch_id"]])
        files_tab_count = save_wandb_files(
            wb,
            [provenance_snapshot, Path(provenance["training_code_log"]), meta_path],
            base_path=args.output_dir,
        )
        wb.summary.update({
            "provenance/artifact": artifact_name,
            "provenance/launch_id": provenance["launch_id"],
            "provenance/git_head": provenance["git"]["head"].get("output", "").strip(),
            "provenance/repository_files": len(provenance["repository_files"]),
            "provenance/training_code_files": len(provenance["training_code_files"]),
            "provenance/bernn_files": len(provenance["bernn_files"]),
            "provenance/files_tab_files": files_tab_count,
            "provenance/dataset_sha256": ",".join(
                row["sha256"] for row in provenance["dataset_files"]
            ),
            "hpo/historical_seed_trials": int(imported_seed_trials),
            "hpo/expected_repeats": int(expected_repeats),
        })

    try:
        completed_before = sorted(completed(study), key=lambda item: item.number)
        resume_queue = sorted(interrupted_trials(study), key=lambda item: item.number) if args.resume else []
        terminal_states = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }
        attempts_before = sum(
            trial.state in terminal_states and not trial.user_attrs.get("transfer_seed", False)
            for trial in study.get_trials(deepcopy=False)
        )
        sanity_trials = [
            item for item in completed_before
            if item.user_attrs.get("config", {}).get("search_stage") == "sanity"
        ]
        sanity_blocked = False
        if (
            dataset == "bacteria_2024_mz10"
            and args.mz10_search_space == "plain"
            and sanity_trials
        ):
            sanity_trial = sanity_trials[0]
            sanity_metrics = dict(sanity_trial.user_attrs.get("metrics", {}))
            sanity_passed, sanity_details = mz10_sanity_gate(
                sanity_metrics, float(sanity_trial.value)
            )
            atomic_json(out / "sanity_gate.json", sanity_details)
            sanity_blocked = not sanity_passed
        print(
            f"[worker] {dataset}: protocol={protocol} samples={len(X)} "
            f"features={X.shape[1]} batches={len(batch_values)} "
            f"attempted={attempts_before}/{args.n_trials} transfer_seeds={imported_seed_trials}",
            flush=True,
        )
        if sanity_blocked:
            print(
                f"[sanity gate] mz10 remains below the learning gate: {sanity_details}; "
                "constrained HPO will not consume the remaining budget",
                flush=True,
            )
        trial_indices = range(attempts_before, args.n_trials) if not sanity_blocked else ()
        for index in trial_indices:
            if resume_queue:
                frozen = resume_queue.pop(0)
                trial = optuna.trial.Trial(study, frozen._trial_id)
                print(
                    f"[resume] continuing trial {trial.number} from completed-repeat checkpoints",
                    flush=True,
                )
            else:
                trial = study.ask()
            run_args = hp_search.parse_args([])
            run_args.dataset = dataset
            run_args.n_epochs = args.n_epochs
            run_args.num_workers = args.num_workers
            run_args.device = "cuda"
            run_args.seed = args.seed
            run_args.no_wandb = True
            run_args.combine_test = False
            warmup_cap = 150 if args.mz10_search_space.startswith("highrange_") else 50
            run_args.max_warmup = max(1, min(warmup_cap, args.n_epochs))
            run_args.log1p = True
            run_args.bs = recommended_batch_size(batches, cap=args.batch_size)
            run_args.results_dir = str(out / "bernn")
            run_args.classifier_patience = 30
            run_args.early_warmup_stop = 0
            run_args.enable_optuna_pruning = bool(args.pruning)
            run_args.repeat1_prune_percentile = float(args.repeat1_prune_percentile)
            run_args.repeat2_prune_percentile = float(args.repeat2_prune_percentile)
            run_args.prune_min_reference_trials = int(args.prune_min_reference_trials)
            run_args.epoch_prune_percentile = hp_search.PRUNE_PERCENTILE
            run_args.epoch_prune_start = 10
            run_args.epoch_prune_interval = 5
            run_args.repeat1_hard_floor = 0.35 if dataset == "bacteria_2024_mz10" else 0.0
            run_args.repeat2_hard_floor = 0.55 if dataset == "bacteria_2024_mz10" else None
            run_args.optuna_trial = trial
            run_args.fold_checkpoint_dir = str(
                out / "trial_checkpoints" / f"trial_{trial.number:05d}"
            )

            if protocol == "fixed_external":
                requested = CV_FOLDS.get(dataset, args.n_repeats)
                run_args.n_repeats = requested
                run_args.resolved_n_repeats = hp_search.resolve_n_repeats(requested, batches)
                run_args.cv_split_cache = str(out / "cv_splits.npz")
            elif protocol == "grouped_cv":
                requested = GROUPED_CV_FOLDS[dataset]
                run_args.n_repeats = requested
                run_args.resolved_n_repeats = hp_search.resolve_n_repeats(requested, batches)
                run_args.trainer_n_repeats = 1
                run_args.cv_split_cache = str(out / "cv_splits.npz")
                run_args.feature_select_k = GROUPED_FEATURE_COUNTS[dataset]
                run_args.feature_select_method = feature_select_method
                run_args.feature_select_cache_dir = str(out / "feature_selection")
            else:
                requested = CYCLIC_CV_FOLDS.get(dataset, len(batch_values))
                run_args.n_repeats = requested
                run_args.resolved_n_repeats = hp_search.resolve_n_repeats(
                    requested, batches
                )
                run_args.trainer_n_repeats = 1

            is_plain_mz10 = (
                dataset == "bacteria_2024_mz10"
                and args.mz10_search_space == "plain"
            )
            is_mz10_sanity = is_plain_mz10 and not sanity_trials
            if dataset == "bacteria_2024_mz10":
                if args.mz10_search_space == "highrange_extended_nonvariational":
                    config = hp_search.sample_mz10_highrange_nonvariational_config(
                        trial, run_args
                    )
                    config["search_stage"] = "highrange_extended_nonvariational"
                elif args.mz10_search_space == "extended_nonvariational":
                    config = hp_search.sample_mz10_nonvariational_config(trial, run_args)
                    config["search_stage"] = "extended_nonvariational"
                elif args.mz10_search_space == "highrange_plain":
                    config = hp_search.sample_mz10_highrange_plain_config(trial, run_args)
                    config["search_stage"] = "highrange_plain"
                else:
                    run_args.force_sanity_config = is_mz10_sanity
                    config = hp_search.sample_mz10_plain_config(trial, run_args)
                    config["search_stage"] = "sanity" if is_mz10_sanity else "constrained_hpo"
            else:
                config = hp_search.sample_config(trial, run_args)
            config["log1p"] = True
            config.update({
                "batch_size": int(run_args.bs),
                "cv_folds": int(run_args.resolved_n_repeats),
                "num_workers": int(run_args.num_workers),
                "lisi_enabled": False,
            })

            metrics, error = {}, None
            started = time.monotonic()
            try:
                exp_id = f"independent_{meta['run_id']}_{dataset}_t{trial.number}"
                if protocol == "cyclic":
                    score, metrics = hp_search.run_cyclic_batch_trial(config, run_args, data, exp_id)
                else:
                    score, metrics = hp_search.run_trial(
                        config, run_args, data, exp_id, fixed_test_data=fixed_test
                    )
                score = float(score)
            except optuna.TrialPruned as exc:
                fit_seconds = time.monotonic() - started
                trial.set_user_attr("config", config)
                trial.set_user_attr("fit_seconds", fit_seconds)
                trial.set_user_attr("pruned_reason", str(exc))
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                print(f"[worker] {dataset} trial {trial.number} pruned: {exc}", flush=True)
                rows = records_for(study, protocol)
                persist_trials(out, rows)
                if wb:
                    wb.log({
                        "trial_index": index,
                        "trial_number": int(trial.number),
                        "status/pruned": 1,
                        "runtime/fit_seconds": fit_seconds,
                        "pruning/partial_valid_mcc": trial.user_attrs.get(
                            "partial_valid_mcc_mean",
                            trial.user_attrs.get("pruning_best_valid_mcc"),
                        ),
                        "pruning/classifier_epoch": trial.user_attrs.get(
                            "pruning_classifier_epoch"
                        ),
                        "pruning/stage": trial.user_attrs.get("pruning_stage"),
                        "pruning/repeat": trial.user_attrs.get("pruning_repeat"),
                        "pruning/reference_count": trial.user_attrs.get(
                            "pruning_reference_count"
                        ),
                        "pruning/cutoff": trial.user_attrs.get("pruning_cutoff"),
                        "pruning/resource_repeats_completed": trial.user_attrs.get(
                            "resource_repeats_completed"
                        ),
                        "pruning/predicted_final_mcc": trial.user_attrs.get(
                            "predicted_final_mcc"
                        ),
                        "pruning/predicted_final_mcc_std": trial.user_attrs.get(
                            "predicted_final_mcc_std"
                        ),
                        "pruning/predicted_final_cutoff": trial.user_attrs.get(
                            "predicted_final_cutoff"
                        ),
                        **config_metrics(config),
                    })
                    save_wandb_files(
                        wb,
                        [out / "trials.json", out / "trials.csv", out / "optuna.sqlite3"],
                        base_path=args.output_dir,
                    )
                continue
            except Exception as exc:
                score = -1.0
                error = f"{type(exc).__name__}: {exc}"
                print(f"[worker] {dataset} trial {trial.number} failed: {error}", flush=True)

            clean = compact_metrics(metrics)
            clean.setdefault("valid_mcc", score)
            fit_seconds = time.monotonic() - started
            trial.set_user_attr("config", config)
            trial.set_user_attr("metrics", clean)
            trial.set_user_attr("fit_seconds", fit_seconds)
            if error:
                trial.set_user_attr("error", error)
            study.tell(trial, score)

            rows = records_for(study, protocol)
            persist_trials(out, rows)
            if wb:
                save_wandb_files(
                    wb,
                    [
                        out / "trials.json",
                        out / "trials.csv",
                        out / "cv_splits.npz",
                        out / "feature_selection",
                        out / "sanity_gate.json",
                    ],
                    base_path=args.output_dir,
                )
            done = sorted(completed(study), key=lambda x: x.number)
            current = done[-1]
            best = max(done, key=lambda x: float(x.value))
            current_metrics = dict(current.user_attrs.get("metrics", {}))
            best_metrics = dict(best.user_attrs.get("metrics", {}))
            if wb:
                wb.log({
                    "trial_index": index,
                    "trial_number": int(current.number),
                    "metrics/valid_mcc": float(current.value),
                    "metrics/test_mcc": current_metrics.get("test_mcc"),
                    "runtime/fit_seconds": current.user_attrs.get("fit_seconds"),
                    "best/valid_mcc": float(best.value),
                    "best/test_mcc": best_metrics.get("test_mcc"),
                    "best/trial_number": int(best.number),
                    "pruning/resource_repeats_completed": current.user_attrs.get(
                        "resource_repeats_completed"
                    ),
                    "pruning/predicted_final_mcc": current.user_attrs.get(
                        "predicted_final_mcc"
                    ),
                    "pruning/predicted_final_mcc_std": current.user_attrs.get(
                        "predicted_final_mcc_std"
                    ),
                    **wandb_metrics(current_metrics),
                    **config_metrics(dict(current.user_attrs.get("config", {}))),
                })
            print(
                f"[worker] {dataset} {index + 1}/{args.n_trials}: "
                f"valid={float(current.value):.4f} test={current_metrics.get('test_mcc')} "
                f"best_valid={float(best.value):.4f} best_test={best_metrics.get('test_mcc')}",
                flush=True,
            )
            if is_mz10_sanity:
                sanity_passed, sanity_details = mz10_sanity_gate(
                    current_metrics, float(current.value)
                )
                atomic_json(out / "sanity_gate.json", sanity_details)
                print(f"[sanity gate] mz10 {sanity_details}", flush=True)
                sanity_trials.append(current)
                if not sanity_passed:
                    print(
                        "[sanity gate] stopping mz10 before constrained HPO; "
                        "the plain BERNN MLP did not clear the learning gate",
                        flush=True,
                    )
                    break

        finished = completed(study)
        if not finished:
            print(f"[worker] {dataset}: no completed trials", flush=True)
            return 1
        best = max(finished, key=lambda x: float(x.value))
        best_metrics = dict(best.user_attrs.get("metrics", {}))
        summary = {
            "dataset": dataset,
            "protocol": protocol,
            "completed_trials": len(completed(study)),
            "best_trial_number": int(best.number),
            "best_valid_mcc": float(best.value),
            "paired_test_mcc_at_best_valid": best_metrics.get("test_mcc"),
            "best_config": dict(best.user_attrs.get("config", {})),
            "selection_metric": "valid_mcc",
            "test_role": "monitoring_only_excluded_from_optuna",
        }
        atomic_json(out / "summary.json", summary)
        if wb:
            wb.summary.update(summary)
        return 0
    finally:
        if wb:
            save_wandb_files(
                wb,
                [
                    meta_path,
                    out / "trials.json",
                    out / "trials.csv",
                    out / "summary.json",
                    out / "sanity_gate.json",
                    out / "cv_splits.npz",
                    out / "feature_selection",
                    out / "optuna.sqlite3",
                    args.output_dir / "logs" / f"{dataset}.log",
                ],
                base_path=args.output_dir,
            )
            wb.finish()
        try:
            storage.engine.dispose()
        except Exception:
            pass


def run_launcher(args) -> int:
    jobs = selected_datasets(args.datasets)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "datasets": [{"dataset": d, "protocol": p} for d, p in jobs],
        "new_datasets_first": True,
        "gpus": gpus,
        "n_trials_per_dataset": args.n_trials,
        "n_epochs": args.n_epochs,
        "selection_metric": "valid_mcc",
        "test_role": "monitoring_only_excluded_from_optuna",
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
    }
    atomic_json(args.output_dir / "experiment_manifest.json", manifest)

    queue, active, failures = list(jobs), {}, []
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            dataset, protocol = queue.pop(0)
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker-dataset", dataset, "--worker-protocol", protocol,
                "--output-dir", str(args.output_dir),
                "--n-trials", str(args.n_trials), "--n-epochs", str(args.n_epochs),
                "--n-repeats", str(args.n_repeats), "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers), "--seed", str(args.seed),
                "--wandb-project", args.wandb_project,
                "--wandb-entity", args.wandb_entity,
                "--wandb-group", args.wandb_group,
                "--repeat1-prune-percentile", str(args.repeat1_prune_percentile),
                "--repeat2-prune-percentile", str(args.repeat2_prune_percentile),
                "--prune-min-reference-trials", str(args.prune_min_reference_trials),
            ]
            cmd.append("--pruning" if args.pruning else "--no-pruning")
            cmd.append(
                "--auto-seed-history" if args.auto_seed_history
                else "--no-auto-seed-history"
            )
            cmd.append(
                "--seed-from-wandb" if args.seed_from_wandb
                else "--no-seed-from-wandb"
            )
            if args.resume:
                cmd.append("--resume")
            if args.no_wandb:
                cmd.append("--no-wandb")
            if args.prepare_missing:
                cmd.append("--prepare-missing")
            if args.feature_select_method:
                cmd.extend(["--feature-select-method", args.feature_select_method])
            if args.mz10_search_space != "plain":
                cmd.extend(["--mz10-search-space", args.mz10_search_space])
            for seed_path in args.seed_trials_from:
                cmd.extend(["--seed-trials-from", str(seed_path)])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            logs = args.output_dir / "logs"
            logs.mkdir(exist_ok=True)
            log_path = logs / f"{dataset}.log"
            stream = log_path.open("a" if args.resume else "w", encoding="utf-8")
            process = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            active[gpu] = (dataset, protocol, process, stream, log_path)
            print(f"[launcher] GPU {gpu} <- {dataset} ({protocol}) log={log_path}", flush=True)

        time.sleep(2)
        for gpu, job in list(active.items()):
            dataset, protocol, process, stream, log_path = job
            code = process.poll()
            if code is None:
                continue
            stream.close()
            print(f"[launcher] GPU {gpu} finished {dataset} exit={code}", flush=True)
            if code:
                failures.append({
                    "dataset": dataset, "protocol": protocol,
                    "exit_code": code, "log": str(log_path),
                })
            del active[gpu]

    summaries = []
    for dataset, _ in jobs:
        path = args.output_dir / dataset / "summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text()))
    atomic_json(args.output_dir / "all_dataset_summary.json", summaries)
    manifest["failures"] = failures
    manifest["completed_datasets"] = [row["dataset"] for row in summaries]
    manifest["completed_at_unix"] = time.time()
    atomic_json(args.output_dir / "experiment_manifest.json", manifest)
    if failures:
        print(json.dumps({"failures": failures}, indent=2), flush=True)
        return 1
    print("[launcher] all dataset studies completed successfully", flush=True)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker_dataset:
        if not args.worker_protocol:
            raise ValueError("--worker-protocol is required in worker mode")
        return run_worker(args)
    return run_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
