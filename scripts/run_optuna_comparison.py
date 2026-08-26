#!/usr/bin/env python3
"""Classic per-dataset Optuna baseline for the evolutionary BERNN experiment.

One solution step asks each independent dataset study for one TPE trial. This
matches one evolutionary solution's three development-dataset BERNN evaluations,
while keeping test labels monitoring-only.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import uuid
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_PATH = Path(__file__).resolve()

TRACKED_DOC_FILES = (
    "README.md",
    "Application notes/README.md",
    "Application notes/manuscript.md",
    "Application notes/datasets-and-splits.md",
    "Application notes/release-manifest.md",
    "Application notes/validation-report.md",
    "Application notes/submission-checklist.md",
    "paper/main_text.md",
)

TRACKED_CODE_SEEDS = (
    "scripts/run_optuna_comparison.py",
    "scripts/hp_search.py",
    "scripts/evolve_meta_model.py",
    "src/dataset_splits.py",
    "src/evolutionary_meta.py",
)

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
_THRESHOLD_STATE: dict[str, dict[str, float]] = {}
DATASET_CV_FOLDS = {
    "normal_tissue_878": 3,
    "colon_3041": 3,
    "massbench_adenocarcinoma": 3,
    "massbench_benchmark": 3,
    "massbench_alzheimer": 3,
}


def _dataset_cv_settings(dataset_id: str, default_n_repeats: int, batches) -> tuple[int, int]:
    """Return the requested and resolved fold counts for one dataset."""
    requested = int(DATASET_CV_FOLDS.get(dataset_id, default_n_repeats))
    resolved = hp_search.resolve_n_repeats(requested, batches)
    return requested, resolved


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


def _candidate_repo_roots() -> list[Path]:
    candidates: list[Path] = []
    env_roots = [
        os.environ.get("BE_LEADERBOARD_ROOT"),
        os.environ.get("BE_LEADERBOARD_REPO"),
        os.environ.get("CODE_ROOT"),
    ]
    for raw in env_roots:
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend([
        ROOT,
        SCRIPT_PATH.parent.parent,
        Path("/home/sp/BE_leaderboard"),
        Path("/home/simon/BE_leaderboard"),
        Path("/home/simonp/BE_leaderboard"),
    ])

    seen = set()
    ordered = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _discover_repo_root() -> Path:
    for candidate in _candidate_repo_roots():
        if (candidate / "README.md").exists() and (candidate / "Application notes" / "README.md").exists():
            return candidate
    return ROOT.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_to_path(repo_root: Path, module_name: str) -> Path | None:
    if not module_name:
        return None
    module_parts = module_name.split(".")
    candidates = [
        repo_root.joinpath(*module_parts).with_suffix(".py"),
        repo_root.joinpath(*module_parts, "__init__.py"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _relative_module_to_path(repo_root: Path, source_path: Path, level: int, module_name: str | None) -> Path | None:
    if level <= 0:
        return _module_to_path(repo_root, module_name or "")
    anchor = source_path.parent
    for _ in range(level - 1):
        anchor = anchor.parent
    if module_name:
        candidate = anchor.joinpath(*module_name.split("."))
        for resolved in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if resolved.exists():
                return resolved
    for alias_candidate in (anchor / "__init__.py", anchor.with_suffix(".py")):
        if alias_candidate.exists():
            return alias_candidate
    return None


def _expand_python_dependencies(repo_root: Path, seed_relpaths: tuple[str, ...]) -> list[Path]:
    queue = deque()
    for relpath in seed_relpaths:
        candidate = repo_root / relpath
        if candidate.exists():
            queue.append(candidate)

    seen: set[Path] = set()
    ordered: list[Path] = []

    while queue:
        path = queue.popleft().resolve()
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
        if path.suffix != ".py":
            continue

        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    candidate = _module_to_path(repo_root, alias.name)
                    if candidate is not None and candidate not in seen:
                        queue.append(candidate)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    candidate = _relative_module_to_path(repo_root, path, node.level, node.module)
                    if candidate is not None and candidate not in seen:
                        queue.append(candidate)
                else:
                    if node.module:
                        candidate = _module_to_path(repo_root, node.module)
                        if candidate is not None and candidate not in seen:
                            queue.append(candidate)
                        for alias in node.names:
                            candidate = _module_to_path(repo_root, f"{node.module}.{alias.name}")
                            if candidate is not None and candidate not in seen:
                                queue.append(candidate)

    return ordered


def _file_record(repo_root: Path, path: Path, group: str) -> dict:
    record = {
        "group": group,
        "path": path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        record.update({
            "sha256": _sha256_file(path),
            "size_bytes": stat.st_size,
            "mtime_unix": stat.st_mtime,
        })
    else:
        record.update({
            "sha256": None,
            "size_bytes": None,
            "mtime_unix": None,
        })
    return record


def _build_code_manifest(repo_root: Path) -> dict:
    doc_records = [_file_record(repo_root, repo_root / relpath, "doc") for relpath in TRACKED_DOC_FILES]
    code_seed_records = [_file_record(repo_root, repo_root / relpath, "code_seed") for relpath in TRACKED_CODE_SEEDS]
    expanded_code_paths = _expand_python_dependencies(repo_root, TRACKED_CODE_SEEDS)
    expanded_code_relpaths = {record["path"] for record in code_seed_records if record["exists"]}
    code_records = []
    for path in expanded_code_paths:
        relpath = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else str(path)
        if relpath in expanded_code_relpaths:
            continue
        code_records.append(_file_record(repo_root, path, "code_dependency"))
        expanded_code_relpaths.add(relpath)

    all_records = doc_records + code_seed_records + code_records
    digest = hashlib.sha256(
        json.dumps(
            [
                {
                    "group": record["group"],
                    "path": record["path"],
                    "exists": record["exists"],
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                }
                for record in all_records
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "repo_root": str(repo_root),
        "script_path": str(SCRIPT_PATH),
        "files": all_records,
        "summary": {
            "doc_count": len(doc_records),
            "code_seed_count": len(code_seed_records),
            "code_dependency_count": len(code_records),
            "existing_file_count": sum(1 for record in all_records if record["exists"]),
            "missing_file_count": sum(1 for record in all_records if not record["exists"]),
            "manifest_sha256": digest,
        },
    }


def _git_info(repo_root: Path) -> dict:
    info = {"head": None, "dirty": None}
    try:
        info["head"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short"],
            text=True,
        ).strip()
        info["dirty"] = bool(status)
    except Exception:
        pass
    return info


def ensure_score_thresholds(dataset_ids) -> None:
    for dataset_id in dataset_ids:
        if dataset_id not in _THRESHOLD_STATE:
            _THRESHOLD_STATE[dataset_id] = dict(SCORE_THRESHOLDS.get(dataset_id, {}))


def update_score_threshold(dataset_id: str, score: float) -> None:
    state = _THRESHOLD_STATE.setdefault(dataset_id, dict(SCORE_THRESHOLDS.get(dataset_id, {})))
    state["best_valid_mcc"] = max(float(score), float(state.get("best_valid_mcc", float("-inf"))))


def score_threshold_payload(dataset_ids) -> dict[str, float]:
    payload: dict[str, float] = {}
    for dataset_id in dataset_ids:
        state = _THRESHOLD_STATE.setdefault(dataset_id, dict(SCORE_THRESHOLDS.get(dataset_id, {})))
        for key in ("reference", "acceptable", "best_valid_mcc"):
            if key in state:
                payload[f"score_thresholds/{dataset_id}/{key}"] = float(state[key])
    return payload


def _write_code_manifest(output_dir: Path, manifest: dict) -> Path:
    path = output_dir / "code_manifest.json"
    _atomic_json(path, manifest)
    return path


def _resolve_log1p_mode(mode: str) -> bool | None:
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode == "optimize":
        return None
    raise ValueError(f"Unknown log1p mode: {mode}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest", type=Path,
        default=ROOT / "config" / "evolution_development_datasets.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "optuna_comparison")
    parser.add_argument("--n-trials", type=int, default=1000, help="Trials per dataset.")
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Maximum BERNN batch size; may be lowered for very small dataset folds.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="PyTorch DataLoader subprocesses per BERNN loader.",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="BE_leaderboard_meta_evolution")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--meta-hidden-size", type=int, default=64)
    parser.add_argument("--meta-epochs", type=int, default=1000)
    parser.add_argument(
        "--log1p-mode",
        choices=("on", "off", "optimize"),
        default="on",
        help="Keep log1p on, force it off, or let Optuna search it.",
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
        "fold_scores": attrs.get("fold_scores", {}),
        "config": attrs.get("config", {}),
        "error": attrs.get("error"),
    }


def _fold_scores_payload(metrics: dict) -> dict[str, list[float]]:
    """Return JSON-safe, per-fold MCC values without replacing their means."""
    payload = {}
    for split in ("valid", "test"):
        values = metrics.get(f"{split}_mcc_folds", [])
        if isinstance(values, (list, tuple, np.ndarray)):
            payload[split] = [float(value) for value in values]
    return payload


def _fold_score_rows(
    phase: str,
    solution_step: int,
    dataset_id: str,
    fold_scores: dict[str, list[float]],
    trial_number: int | None = None,
) -> list[dict]:
    rows = []
    for split, values in fold_scores.items():
        for fold_index, score in enumerate(values):
            rows.append({
                "phase": phase,
                "solution_step": int(solution_step),
                "trial_number": "" if trial_number is None else int(trial_number),
                "dataset": dataset_id,
                "split": split,
                "fold": int(fold_index),
                "mcc": float(score),
            })
    return rows


def _append_fold_scores(output_dir: Path, rows: list[dict]) -> None:
    """Persist every CV point in a tidy CSV suitable for arbitrary replots."""
    if not rows:
        return
    path = output_dir / "cv_fold_scores.csv"
    fieldnames = ("phase", "solution_step", "trial_number", "dataset", "split", "fold", "mcc")
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _wandb_fold_table(wandb, rows: list[dict]):
    columns = ["phase", "solution_step", "trial_number", "dataset", "split", "fold", "mcc"]
    return wandb.Table(columns=columns, data=[[row[column] for column in columns] for row in rows])


def _fold_boxplot(rows: list[dict], title: str):
    """Make Plotly boxes with the individual fold MCC values overlaid."""
    import plotly.express as px

    return px.box(
        rows,
        x="dataset",
        y="mcc",
        color="split",
        points="all",
        hover_data=["solution_step", "trial_number", "fold"],
        title=title,
    )


def _best_payload(study) -> dict:
    return _trial_payload(max(_completed_trials(study), key=lambda trial: float(trial.value)))


def _aggregate_metric(solution: dict, dataset_ids: tuple[str, ...], metric: str) -> float:
    return float(aggregate_dataset_scores([solution[name][metric] for name in dataset_ids]))


def _meta_vector(dataset) -> np.ndarray:
    X, y, batches = dataset
    features = extract_meta_features(X, y, batches)
    return np.asarray([features[name] for name in META_FEATURE_NAMES], dtype=np.float32)


def _scale01(value, low, high):
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def _unscale01(value, low, high):
    return float(low + np.clip(float(value), 0.0, 1.0) * (high - low))


def _log_scale01(value, low, high):
    return _scale01(np.log10(np.clip(float(value), low, high)), np.log10(low), np.log10(high))


def _log_unscale01(value, low, high):
    return float(10 ** _unscale01(value, np.log10(low), np.log10(high)))


_DLOSS = tuple(hp_search.DLOSS_CHOICES)
_SCALERS = tuple(hp_search.SCALER_CHOICES)
_N_LAYERS = (1, 2, 3, 4, 5)


def _one_hot(value, choices):
    return [1.0 if value == choice else 0.0 for choice in choices]


def _encode_config(config: dict, max_warmup: int) -> np.ndarray:
    """Encode source-study hparams as bounded joint-meta-model targets."""
    return np.asarray([
        float(bool(config.get("variational", False))),
        float(bool(config.get("kan", False))),
        float(bool(config.get("class_triplet", False))),
        1.0,  # log1p is always on, but remains an explicit logged target.
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
        _log_scale01(config.get("gamma") or 1e-2, 1e-2, 1e2),
        _log_scale01(config.get("beta") or 1e-2, 1e-2, 1e2),
        *_one_hot(config["dloss"], _DLOSS),
        *_one_hot(config["scaler"], _SCALERS),
        *_one_hot(int(config["n_layers"]), _N_LAYERS),
    ], dtype=np.float32)


def _decode_config(encoded: np.ndarray, max_warmup: int) -> dict:
    z = np.clip(np.asarray(encoded, dtype=float), 0.0, 1.0)
    i = 0
    variational = bool(z[i] >= 0.5); i += 1
    kan = bool(z[i] >= 0.5); i += 1
    class_triplet = bool(z[i] >= 0.5); i += 1
    i += 1  # retained log1p dimension; preprocessing is invariantly enabled.
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
    dloss = _DLOSS[int(np.argmax(z[i:i + len(_DLOSS)]))]; i += len(_DLOSS)
    scaler = _SCALERS[int(np.argmax(z[i:i + len(_SCALERS)]))]; i += len(_SCALERS)
    n_layers = _N_LAYERS[int(np.argmax(z[i:i + len(_N_LAYERS)]))]
    return {
        "model_type": "joint", "dloss": dloss,
        "variational": variational, "kan": kan,
        "class_triplet": class_triplet,
        "class_triplet_w": class_triplet_w,
        "lr": lr, "wd": wd, "nu": nu, "smoothing": smoothing,
        "margin": margin, "dropout": dropout, "thres": thres,
        "warmup": int(np.clip(warmup, 1, max_warmup)),
        "n_layers": n_layers, "layer1": int(np.clip(layer1, 512, 1024)),
        "log1p": True, "scaler": scaler,
        "gamma": gamma_candidate if dloss in hp_search.ADVERSARIAL_DLOSS else 0.0,
        "beta": beta_candidate if variational else 0.0,
    }


def _fit_joint_meta_model(studies, datasets, validation_datasets, args):
    """Fit one model on all train datasets; validation only receives predictions."""
    import torch
    from torch import nn

    source_ids = [name for name in studies if _completed_trials(studies[name])]
    if len(source_ids) != len(studies):
        raise RuntimeError("Every meta-training dataset needs a completed Optuna trial")
    validation_ids = tuple(validation_datasets)
    train_meta = np.stack([_meta_vector(datasets[name]) for name in source_ids])
    valid_meta = np.stack([_meta_vector(validation_datasets[name]) for name in validation_ids])
    mean = train_meta.mean(axis=0, keepdims=True)
    scale = train_meta.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    train_meta = (train_meta - mean) / scale
    valid_meta = (valid_meta - mean) / scale
    max_warmup = max(1, min(50, int(args.n_epochs)))
    source_best = {name: _best_payload(studies[name]) for name in source_ids}
    targets = np.stack([_encode_config(source_best[name]["config"], max_warmup) for name in source_ids])
    torch.manual_seed(int(args.seed))
    model = nn.Sequential(
        nn.Linear(train_meta.shape[1], int(args.meta_hidden_size)), nn.ReLU(),
        nn.Linear(int(args.meta_hidden_size), targets.shape[1]), nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.tensor(train_meta, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    model.train()
    final_loss = np.nan
    for _ in range(int(args.meta_epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(valid_meta, dtype=torch.float32)).cpu().numpy()
    configs = {name: _decode_config(row, max_warmup) for name, row in zip(validation_ids, predictions)}
    diagnostics = {
        "meta_train_loss": final_loss,
        "source_best_scores": {name: source_best[name]["valid_mcc"] for name in source_ids},
        "source_best_configs": {name: source_best[name]["config"] for name in source_ids},
    }
    checkpoint = {
        "state_dict": model.state_dict(), "meta_mean": mean.squeeze(0),
        "meta_scale": scale.squeeze(0), "source_ids": source_ids,
        "validation_ids": list(validation_ids), "diagnostics": diagnostics,
        "meta_hidden_size": int(args.meta_hidden_size), "max_warmup": max_warmup,
    }
    return configs, diagnostics, checkpoint


def _log_hparams(payload: dict, prefix: str, config: dict) -> None:
    for name, value in config.items():
        if isinstance(value, (bool, np.bool_)):
            payload[f"{prefix}/{name}"] = int(value)
        elif isinstance(value, (int, float, np.integer, np.floating, str)):
            payload[f"{prefix}/{name}"] = value


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
    if args.n_repeats != 3:
        raise ValueError("Dataset-conditional meta-learning requires grouped CV=3")
    if args.n_trials < 1:
        raise ValueError("n_trials must be positive")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if args.log1p_mode != "on":
        raise ValueError("This comparison requires --log1p-mode on")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bernn_version = importlib.metadata.version("bernn")
    metadata_path = args.output_dir / "run_metadata.json"
    had_persisted_wandb_id = False
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        had_persisted_wandb_id = bool(metadata.get("wandb_run_id"))
        if not args.resume:
            raise FileExistsError(f"{args.output_dir} already contains a run; pass --resume")
    else:
        metadata = {
            "created_at_unix": time.time(),
            "hostname": platform.node(),
            "gpu": _gpu_name(),
            "wandb_run_id": None,
        }

    repo_root = _discover_repo_root()
    git_info = _git_info(repo_root)
    code_manifest = _build_code_manifest(repo_root)
    code_manifest_path = _write_code_manifest(args.output_dir, code_manifest)
    metadata.update({
        "arm": "dataset_conditional_meta_learning",
        "candidate_budget": int(args.n_trials),
        "budget_unit": "solution_step_one_trial_per_train_dataset",
        "per_dataset_trial_budget": int(args.n_trials),
        "total_train_dataset_fits": int(args.n_trials * len(load_dataset_partitions(args.split_manifest).train)),
        "repo_root": str(repo_root),
        "git_head": git_info["head"],
        "git_dirty": git_info["dirty"],
        "code_manifest_path": str(code_manifest_path),
        "code_manifest_sha256": code_manifest["summary"]["manifest_sha256"],
        "code_file_count": code_manifest["summary"]["existing_file_count"],
        "code_missing_file_count": code_manifest["summary"]["missing_file_count"],
        "log1p_mode": args.log1p_mode,
        "bernn_version": bernn_version,
    })
    if not args.no_wandb and not metadata.get("wandb_run_id"):
        metadata["wandb_run_id"] = uuid.uuid4().hex[:8]
    # Persist the W&B identity before any expensive setup. If the process is
    # interrupted later, --resume can still reconnect to exactly this run.
    _atomic_json(metadata_path, metadata)
    print(
        "[optuna-comparison] code tracking: "
        f"{code_manifest['summary']['existing_file_count']} files present, "
        f"{code_manifest['summary']['missing_file_count']} missing, "
        f"manifest={code_manifest['summary']['manifest_sha256'][:12]}",
        flush=True,
    )
    print(
        f"[optuna-comparison] BERNN {bernn_version}; "
        f"num_workers={args.num_workers}; CV overrides={DATASET_CV_FOLDS}",
        flush=True,
    )

    partitions = load_dataset_partitions(args.split_manifest)

    # Optuna/TPE learns ONLY from meta-training datasets.
    dataset_ids = tuple(partitions.train)
    validation_ids = tuple(partitions.validation)
    heldout_ids = tuple(getattr(partitions, "test", ()))
    threshold_dataset_ids = tuple(dict.fromkeys((*dataset_ids, *validation_ids, *heldout_ids)))
    ensure_score_thresholds(threshold_dataset_ids)

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

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=metadata["wandb_run_id"],
            resume="must" if args.resume and had_persisted_wandb_id else "never",
            config={
                **vars(args), "split_manifest": str(args.split_manifest),
                "output_dir": str(args.output_dir),
                "repo_root": str(repo_root),
                "git_head": git_info["head"],
                "git_dirty": git_info["dirty"],
                "code_manifest_path": str(code_manifest_path),
                "code_manifest_sha256": code_manifest["summary"]["manifest_sha256"],
                "code_file_count": code_manifest["summary"]["existing_file_count"],
                "code_missing_file_count": code_manifest["summary"]["missing_file_count"],
                "log1p_mode": args.log1p_mode,
                "train_datasets": list(dataset_ids),
                "validation_datasets": list(validation_ids),
                "method": "joint_meta_hpo_from_scratch_validation",
                "validation_initialization": "from_scratch",
                "preprocessing/log1p": True,
                "bernn_version": bernn_version,
                "host": metadata["hostname"],
                "gpu": metadata["gpu"],
            },
        )
        _ACTIVE_WANDB_RUN = wandb_run
        wandb.define_metric("solution_step")
        wandb.define_metric("solutions/*", step_metric="solution_step")
        wandb.define_metric("validation/*", step_metric="solution_step")
        wandb_run.summary.update(score_threshold_payload(threshold_dataset_ids))
        wandb_run.summary.update({
            "code/file_count": code_manifest["summary"]["existing_file_count"],
            "code/missing_file_count": code_manifest["summary"]["missing_file_count"],
            "code/manifest_sha256": code_manifest["summary"]["manifest_sha256"],
        })
        code_artifact = wandb.Artifact(
            name=f"{metadata['wandb_run_id']}-code",
            type="code",
            description="Tracked source files and README context used by the Optuna comparison run.",
        )
        code_artifact.add_file(str(code_manifest_path), name="code_manifest.json")
        for record in code_manifest["files"]:
            if not record["exists"]:
                continue
            source_path = repo_root / record["path"]
            artifact_name = record["path"].replace("/", "__")
            code_artifact.add_file(str(source_path), name=artifact_name)
        wandb_run.log_artifact(code_artifact)
    _atomic_json(metadata_path, metadata)

    print(
        "[optuna-comparison] Independent train-only TPE studies provide targets "
        "for ONE joint meta-model across all partitions.train datasets. The joint "
        "model predicts hparams for partitions.validation, where BERNN is initialized "
        "and trained from scratch. Validation scores never train TPE or the meta-model.",
        flush=True,
    )
    history = {dataset_id: [] for dataset_id in dataset_ids}

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
                update_score_threshold(dataset_id, float(current["valid_mcc"]))

    for solution_step in range(args.n_trials):
        # ------------------------------------------------------------------
        # VALIDATION FIRST (joint meta-HPO, then a fresh validation fit)
        #
        # Starting at step 1, fit one meta-model using BOTH train datasets' current
        # best configurations as targets. It predicts one config for Alzheimer,
        # whose BERNN weights are initialized and trained from scratch.
        # ------------------------------------------------------------------
        if solution_step > 0 and validation_ids:
            print(
                f"[optuna-comparison] solution {solution_step}: "
                f"validation-first evaluation before new TPE trials",
                flush=True,
            )

            validation_payload = {"solution_step": solution_step}
            validation_fold_rows = []

            predicted_configs, meta_diagnostics, meta_checkpoint = _fit_joint_meta_model(
                studies, datasets, validation_datasets, args
            )
            validation_results = {}

            for validation_index, validation_id in enumerate(validation_ids):
                Xv, yv, batches_v = validation_datasets[validation_id]
                predicted_config = predicted_configs[validation_id]
                validation_args = hp_search.parse_args([])
                validation_args.dataset = validation_id
                validation_args.n_epochs = args.n_epochs
                validation_args.n_repeats, validation_args.resolved_n_repeats = _dataset_cv_settings(
                    validation_id, args.n_repeats, batches_v
                )
                validation_args.num_workers = args.num_workers
                validation_args.device = args.device
                validation_args.seed = args.seed + 10000 + validation_index
                validation_args.no_wandb = True
                validation_args.combine_test = False
                validation_args.max_warmup = max(1, min(50, args.n_epochs))
                validation_args.log1p = True
                validation_args.bs = recommended_batch_size(batches_v, cap=args.batch_size)
                predicted_config.update({
                    "batch_size": int(validation_args.bs),
                    "cv_folds": int(validation_args.resolved_n_repeats),
                    "num_workers": int(validation_args.num_workers),
                    "lisi_enabled": False,
                })
                validation_args.results_dir = str(
                    args.output_dir / "validation" / validation_id / "joint_meta_model"
                )
                validation_args.cv_split_cache = str(
                    args.output_dir / "cv_splits" / f"{validation_id}.npz"
                )
                metrics = {}
                try:
                    valid_mcc, metrics = hp_search.run_trial(
                        predicted_config, validation_args, (Xv, yv, batches_v),
                        f"joint_meta_validation_{validation_id}_s{solution_step}",
                        fixed_test_data=validation_fixed_tests[validation_id],
                    )
                    valid_mcc = float(valid_mcc)
                    test_mcc = float(metrics.get("test_mcc", np.nan))
                except Exception as exc:
                    valid_mcc, test_mcc = -1.0, np.nan
                    print(f"[optuna-comparison] validation failed {validation_id}: {type(exc).__name__}: {exc}", flush=True)
                fold_scores = _fold_scores_payload(metrics)
                validation_history[validation_id].append((solution_step, valid_mcc, test_mcc))
                validation_results[validation_id] = {
                    "valid_mcc": valid_mcc, "test_mcc": test_mcc,
                    "fold_scores": fold_scores,
                    "config": predicted_config,
                }
                validation_fold_rows.extend(_fold_score_rows(
                    "validation", solution_step, validation_id, fold_scores
                ))
                update_score_threshold(validation_id, valid_mcc)
                validation_payload[f"validation/{validation_id}/valid_mcc"] = valid_mcc
                validation_payload[f"validation/{validation_id}/test_mcc"] = test_mcc
                validation_payload[f"validation/{validation_id}/meta_train_loss"] = meta_diagnostics["meta_train_loss"]
                for split, values in fold_scores.items():
                    for fold_index, value in enumerate(values):
                        validation_payload[
                            f"validation/folds/{validation_id}/{split}_mcc/fold_{fold_index}"
                        ] = value
                _log_hparams(validation_payload, f"validation/{validation_id}/hparams", predicted_config)

            total_validation_mcc = _aggregate_metric(validation_results, validation_ids, "valid_mcc")
            validation_payload["validation/total_valid_mcc"] = total_validation_mcc
            validation_payload["validation/total_mcc_valid"] = total_validation_mcc
            validation_record = {
                "solution_step": solution_step,
                "total_valid_mcc": total_validation_mcc,
                "datasets": validation_results,
                "meta_train_loss": meta_diagnostics["meta_train_loss"],
                "source_best_scores": meta_diagnostics["source_best_scores"],
                "source_best_configs": meta_diagnostics["source_best_configs"],
            }
            with (args.output_dir / "validation_solutions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(validation_record, default=str) + "\n")
            _append_fold_scores(args.output_dir, validation_fold_rows)
            import torch
            torch.save(meta_checkpoint, args.output_dir / "latest_joint_meta_model.pt")

            if wandb_run is not None and len(validation_payload) > 1:
                if validation_fold_rows:
                    validation_payload["validation/cv_fold_scores"] = _wandb_fold_table(
                        wandb, validation_fold_rows
                    )
                    validation_payload["validation/cv_boxplot"] = _fold_boxplot(
                        validation_fold_rows,
                        f"Validation CV fold MCCs - solution {solution_step}",
                    )
                wandb_run.summary.update(score_threshold_payload(threshold_dataset_ids))
                wandb_run.log(validation_payload)

        current = {}
        for dataset_index, dataset_id in enumerate(dataset_ids):
            completed = _completed_trials(studies[dataset_id])
            if len(completed) <= solution_step:
                trial = studies[dataset_id].ask()
                hp_args = hp_search.parse_args([])
                hp_args.dataset = dataset_id
                hp_args.n_epochs = args.n_epochs
                _, _, batches = datasets[dataset_id]
                hp_args.n_repeats, hp_args.resolved_n_repeats = _dataset_cv_settings(
                    dataset_id, args.n_repeats, batches
                )
                hp_args.num_workers = args.num_workers
                hp_args.device = args.device
                hp_args.seed = args.seed + dataset_index
                hp_args.no_wandb = True
                hp_args.combine_test = False
                hp_args.max_warmup = max(1, min(50, args.n_epochs))
                hp_args.log1p = True
                hp_args.bs = recommended_batch_size(batches, cap=args.batch_size)
                hp_args.cv_split_cache = str(
                    args.output_dir / "cv_splits" / f"{dataset_id}.npz"
                )
                config = hp_search.sample_config(trial, hp_args)
                config.update({
                    "batch_size": int(hp_args.bs),
                    "cv_folds": int(hp_args.resolved_n_repeats),
                    "num_workers": int(hp_args.num_workers),
                    "lisi_enabled": False,
                })
                started = time.monotonic()
                error = None
                metrics = {}
                try:
                    experiment_namespace = metadata["wandb_run_id"] or args.output_dir.name
                    score, metrics = hp_search.run_trial(
                        config, hp_args, datasets[dataset_id],
                        f"classic_optuna_{experiment_namespace}_{dataset_id}_t{trial.number}",
                        fixed_test_data=fixed_tests[dataset_id],
                    )
                except Exception as exc:  # invalid trials remain part of search cost
                    score = -1.0
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"[optuna-comparison] {dataset_id} trial {trial.number} failed: {error}", flush=True)
                trial.set_user_attr("config", config)
                trial.set_user_attr("test_mcc", float(metrics.get("test_mcc", np.nan)))
                trial.set_user_attr("fold_scores", _fold_scores_payload(metrics))
                trial.set_user_attr("fit_seconds", time.monotonic() - started)
                if error:
                    trial.set_user_attr("error", error)
                studies[dataset_id].tell(trial, float(score))
                update_score_threshold(dataset_id, float(score))
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
            "total_valid_mcc": _aggregate_metric(current, dataset_ids, "valid_mcc"),
            "best_total_valid_mcc": _aggregate_metric(best, dataset_ids, "valid_mcc"),
        }
        _append_solution(args.output_dir, record, dataset_ids)
        solution_fold_rows = []
        for dataset_id in dataset_ids:
            row = current[dataset_id]
            history[dataset_id].append((solution_step, row["valid_mcc"], row["test_mcc"]))
            solution_fold_rows.extend(_fold_score_rows(
                "solutions",
                solution_step,
                dataset_id,
                row.get("fold_scores", {}),
                trial_number=row.get("trial_number"),
            ))
        _append_fold_scores(args.output_dir, solution_fold_rows)
        if wandb_run is not None:
            telemetry = {}
            for dataset_id in dataset_ids:
                telemetry.update(
                    _numeric_config_telemetry(dataset_id, current[dataset_id]["config"])
                )
                _log_hparams(
                    telemetry,
                    f"solutions/config/{dataset_id}",
                    current[dataset_id]["config"],
                )
            payload = {
                "solution_step": solution_step,
                "comparison/wall_clock_seconds": record["wall_clock_seconds"],
                "comparison/model_fits_completed": float((solution_step + 1) * len(dataset_ids)),
                "solutions/total_valid_mcc": record["total_valid_mcc"],
                "solutions/total_mcc_valid": record["total_valid_mcc"],
                "solutions/best_total_valid_mcc": record["best_total_valid_mcc"],
                **telemetry,
            }
            for dataset_id in dataset_ids:
                payload.update({
                    f"solutions/metrics/{dataset_id}/valid_mcc": current[dataset_id]["valid_mcc"],
                    f"solutions/metrics/{dataset_id}/test_mcc": current[dataset_id]["test_mcc"],
                    f"solutions/best/{dataset_id}/valid_mcc": best[dataset_id]["valid_mcc"],
                    f"solutions/best/{dataset_id}/test_mcc": best[dataset_id]["test_mcc"],
                    f"comparison/fit_seconds/{dataset_id}": current[dataset_id]["fit_seconds"],
                    f"solutions/mcc/{dataset_id}": _score_figure(
                        dataset_id,
                        history[dataset_id],
                        thresholds=_THRESHOLD_STATE.get(dataset_id),
                    ),
                })
                for split, values in current[dataset_id].get("fold_scores", {}).items():
                    for fold_index, value in enumerate(values):
                        payload[
                            f"solutions/folds/{dataset_id}/{split}_mcc/fold_{fold_index}"
                        ] = value
                _log_hparams(
                    payload,
                    f"solutions/best_config/{dataset_id}",
                    best[dataset_id]["config"],
                )
            if solution_fold_rows:
                payload["solutions/cv_fold_scores"] = _wandb_fold_table(wandb, solution_fold_rows)
                payload["solutions/cv_boxplot"] = _fold_boxplot(
                    solution_fold_rows,
                    f"Training CV fold MCCs - solution {solution_step}",
                )
            wandb_run.log(payload)
            wandb_run.summary.update(score_threshold_payload(threshold_dataset_ids))
        print(f"[optuna-comparison] solution {solution_step} complete: {json.dumps(record)}", flush=True)

    return 0


def main(argv=None) -> int:
    try:
        return _main(argv)
    finally:
        _cleanup_runtime_resources()


if __name__ == "__main__":
    raise SystemExit(main())
