"""Shared utilities for BERNN meta-HPO experiments.

Designed for BE_leaderboard.  This module deliberately keeps the real BERNN
objective in scripts.hp_search and only handles reusable experiment mechanics:

* categorical consensus from independent per-dataset Optuna controls;
* a sampler that can freeze categorical/discrete parameters while leaving all
  continuous parameters searchable;
* config encoding/decoding for direct meta-HPO and surrogate models;
* cumulative source-trial replay from run_optuna_comparison.py solutions.jsonl.

Alzheimer scores must never enter these utilities as training targets for a
source-only meta-learning experiment.  They are monitoring/evaluation only.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts import hp_search


# Categorical/discrete knobs that we allow the Stage-0 Optuna controls to freeze.
# We intentionally do NOT freeze any genuinely continuous hyperparameters.
CATEGORICAL_FIELDS = (
    "dloss",
    "variational",
    "kan",
    "class_triplet",
    "scaler",
    "n_layers",
)

DLOSS_CHOICES = tuple(hp_search.DLOSS_CHOICES)
SCALER_CHOICES = tuple(hp_search.SCALER_CHOICES)
N_LAYER_CHOICES = (1, 2, 3, 4, 5)

# Direct/surrogate encoding. log1p is omitted because the comparison protocol
# fixes it to True. model_type is also fixed to joint.
CONFIG_VECTOR_NAMES = (
    "variational",
    "kan",
    "class_triplet",
    "class_triplet_w",
    "lr",
    "wd",
    "nu",
    "smoothing",
    "margin",
    "dropout",
    "thres",
    "warmup",
    "layer1",
    "gamma",
    "beta",
    *[f"dloss={value}" for value in DLOSS_CHOICES],
    *[f"scaler={value}" for value in SCALER_CHOICES],
    *[f"n_layers={value}" for value in N_LAYER_CHOICES],
)


@dataclass(frozen=True)
class TrialPoint:
    solution_step: int
    dataset_id: str
    trial_number: int
    score: float
    config: dict[str, Any]


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the BERNN knobs relevant to these experiments in stable form."""
    keys = (
        "model_type", "dloss", "variational", "kan", "class_triplet",
        "class_triplet_w", "lr", "wd", "nu", "smoothing", "margin",
        "dropout", "thres", "warmup", "n_layers", "layer1", "log1p",
        "scaler", "gamma", "beta",
    )
    out = {key: _json_value(config[key]) for key in keys if key in config}
    out.setdefault("model_type", "joint")
    out.setdefault("log1p", True)
    return out


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_fixed_categories(config: dict[str, Any], fixed: dict[str, Any] | None) -> dict[str, Any]:
    """Override only declared categorical/discrete knobs and repair conditionals."""
    out = dict(config)
    for name, value in (fixed or {}).items():
        if name not in CATEGORICAL_FIELDS:
            continue
        if name in {"variational", "kan", "class_triplet"}:
            value = bool(value)
        elif name == "n_layers":
            value = int(value)
        else:
            value = str(value)
        out[name] = value

    # Conditional values must be semantically zero when their parent switch is off.
    if out.get("dloss", "no") not in hp_search.ADVERSARIAL_DLOSS:
        out["gamma"] = 0.0
    if not bool(out.get("variational", False)):
        out["beta"] = 0.0
    if not bool(out.get("class_triplet", False)):
        out["class_triplet_w"] = 0.0
    out["model_type"] = "joint"
    out["log1p"] = True
    return canonical_config(out)


def sample_bernn_config(trial, hp_args, fixed_categories: dict[str, Any] | None = None) -> dict[str, Any]:
    """Mirror hp_search.sample_config while allowing all categorical knobs to freeze.

    Continuous knobs remain sampled even if all Stage-0 source champions happen
    to have similar values.  This implements the requested categorical-only
    search-space reduction.
    """
    fixed = fixed_categories or {}

    def cat(name: str, choices):
        if name in fixed:
            value = fixed[name]
            if name == "n_layers":
                value = int(value)
            if value not in choices:
                raise ValueError(f"Fixed categorical {name}={value!r} is not in {tuple(choices)!r}")
            return value
        return trial.suggest_categorical(name, list(choices))

    dloss = str(cat("dloss", DLOSS_CHOICES))
    variational = bool(cat("variational", (False, True)))
    kan = bool(cat("kan", (False, True)))
    class_triplet = bool(cat("class_triplet", (False, True)))
    scaler = str(cat("scaler", SCALER_CHOICES))
    n_layers = int(cat("n_layers", N_LAYER_CHOICES))

    # log1p is invariant in the comparison protocol.
    class_triplet_w = (
        trial.suggest_float("class_triplet_w", 0.0, 1.0)
        if class_triplet else 0.0
    )
    max_warmup = int(getattr(hp_args, "max_warmup", max(1, min(50, int(hp_args.n_epochs)))))

    config = {
        "model_type": "joint",
        "dloss": dloss,
        "variational": variational,
        "kan": kan,
        "class_triplet": class_triplet,
        "class_triplet_w": float(class_triplet_w),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "nu": trial.suggest_float("nu", 1e-4, 1e2),
        "smoothing": trial.suggest_float("smoothing", 0.0, 0.2),
        "margin": trial.suggest_float("margin", 0.0, 10.0),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "thres": trial.suggest_float("thres", 0.0, 0.1),
        "warmup": trial.suggest_int("warmup", 1, max_warmup),
        "n_layers": n_layers,
        "layer1": trial.suggest_int("layer1", 512, 1024),
        "log1p": True,
        "scaler": scaler,
        "gamma": trial.suggest_float("gamma", 1e-2, 1e2, log=True)
                 if dloss in hp_search.ADVERSARIAL_DLOSS else 0.0,
        "beta": trial.suggest_float("beta", 1e-2, 1e2, log=True)
                if variational else 0.0,
    }
    return apply_fixed_categories(config, fixed)


def load_solution_rows(path_or_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "solutions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No source solution ledger at {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["solution_step"]))
    return rows


def trial_points(rows: Iterable[dict[str, Any]], *, upto_step: int | None = None) -> list[TrialPoint]:
    out: list[TrialPoint] = []
    for row in rows:
        step = int(row["solution_step"])
        if upto_step is not None and step > int(upto_step):
            break
        for dataset_id, payload in row["current"].items():
            out.append(TrialPoint(
                solution_step=step,
                dataset_id=str(dataset_id),
                trial_number=int(payload.get("trial_number", step)),
                score=float(payload["valid_mcc"]),
                config=canonical_config(payload["config"]),
            ))
    return out


def best_points_by_dataset(points: Iterable[TrialPoint], dataset_ids: Iterable[str]) -> dict[str, TrialPoint]:
    dataset_ids = tuple(dataset_ids)
    best: dict[str, TrialPoint] = {}
    for point in points:
        if point.dataset_id not in dataset_ids:
            continue
        if point.dataset_id not in best or point.score > best[point.dataset_id].score:
            best[point.dataset_id] = point
    missing = [name for name in dataset_ids if name not in best]
    if missing:
        raise ValueError(f"No completed source trials for: {missing}")
    return best


def _serializable_counter(counter: Counter) -> dict[str, int]:
    return {json.dumps(_json_value(key), sort_keys=True): int(value) for key, value in counter.items()}


def categorical_consensus(
    points: Iterable[TrialPoint],
    dataset_ids: Iterable[str],
    *,
    top_k: int = 10,
    min_support: float = 0.90,
) -> dict[str, Any]:
    """Compute strict-champion and robust top-K categorical consensus.

    `strict_fixed`: all source champions have exactly the same value.
    `robust_fixed`: additionally requires each source dataset's top-K trials to
    support that value at least `min_support`.
    """
    dataset_ids = tuple(dataset_ids)
    points = [p for p in points if p.dataset_id in dataset_ids and np.isfinite(p.score)]
    grouped = {name: [] for name in dataset_ids}
    for point in points:
        grouped[point.dataset_id].append(point)
    for name in dataset_ids:
        grouped[name].sort(key=lambda p: p.score, reverse=True)
        if not grouped[name]:
            raise ValueError(f"No source trials for {name}")

    strict_fixed: dict[str, Any] = {}
    robust_fixed: dict[str, Any] = {}
    fields: dict[str, Any] = {}
    for field in CATEGORICAL_FIELDS:
        champions = {name: _json_value(grouped[name][0].config.get(field)) for name in dataset_ids}
        champion_values = list(champions.values())
        strict_value = champion_values[0] if all(value == champion_values[0] for value in champion_values) else None
        if strict_value is not None:
            strict_fixed[field] = strict_value

        by_dataset = {}
        robust_ok = strict_value is not None
        for name in dataset_ids:
            selected = grouped[name][: max(1, min(int(top_k), len(grouped[name])))]
            values = [_json_value(point.config.get(field)) for point in selected]
            counts = Counter(values)
            mode_value, mode_count = counts.most_common(1)[0]
            support = float(mode_count / len(values))
            by_dataset[name] = {
                "champion": champions[name],
                "top_k_n": len(values),
                "mode": _json_value(mode_value),
                "mode_support": support,
                "counts": _serializable_counter(counts),
            }
            robust_ok = robust_ok and mode_value == strict_value and support >= float(min_support)
        if robust_ok:
            robust_fixed[field] = strict_value
        fields[field] = {
            "champions": champions,
            "strict_consensus": strict_value,
            "robust_consensus": strict_value if robust_ok else None,
            "datasets": by_dataset,
        }

    return {
        "schema_version": 1,
        "dataset_ids": list(dataset_ids),
        "categorical_fields": list(CATEGORICAL_FIELDS),
        "top_k": int(top_k),
        "min_support": float(min_support),
        "strict_fixed": strict_fixed,
        "robust_fixed": robust_fixed,
        "fields": fields,
    }


def load_fixed_categories(path: str | Path | None, *, policy: str = "robust") -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    if policy not in {"robust", "strict"}:
        raise ValueError("policy must be 'robust' or 'strict'")
    key = f"{policy}_fixed"
    if key in payload:
        fixed = payload[key]
    elif "fixed_categories" in payload:
        fixed = payload["fixed_categories"]
    else:
        fixed = payload
    return {name: value for name, value in fixed.items() if name in CATEGORICAL_FIELDS}


def _scale01(value: float, low: float, high: float) -> float:
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def _unscale01(value: float, low: float, high: float) -> float:
    return float(low + np.clip(float(value), 0.0, 1.0) * (high - low))


def _log_scale01(value: float, low: float, high: float) -> float:
    value = float(np.clip(float(value), low, high))
    return _scale01(math.log10(value), math.log10(low), math.log10(high))


def _log_unscale01(value: float, low: float, high: float) -> float:
    return float(10 ** _unscale01(value, math.log10(low), math.log10(high)))


def _one_hot(value: Any, choices: tuple[Any, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def encode_config(config: dict[str, Any], max_warmup: int) -> np.ndarray:
    """Encode one BERNN config into a bounded vector for meta/surrogate models."""
    config = canonical_config(config)
    return np.asarray([
        float(bool(config.get("variational", False))),
        float(bool(config.get("kan", False))),
        float(bool(config.get("class_triplet", False))),
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
        *_one_hot(config["dloss"], DLOSS_CHOICES),
        *_one_hot(config["scaler"], SCALER_CHOICES),
        *_one_hot(int(config["n_layers"]), N_LAYER_CHOICES),
    ], dtype=np.float32)


def decode_config_vector(encoded: np.ndarray, max_warmup: int, fixed_categories: dict[str, Any] | None = None) -> dict[str, Any]:
    z = np.clip(np.asarray(encoded, dtype=float), 0.0, 1.0)
    i = 0
    variational = bool(z[i] >= 0.5); i += 1
    kan = bool(z[i] >= 0.5); i += 1
    class_triplet = bool(z[i] >= 0.5); i += 1
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
    dloss = DLOSS_CHOICES[int(np.argmax(z[i:i + len(DLOSS_CHOICES)]))]; i += len(DLOSS_CHOICES)
    scaler = SCALER_CHOICES[int(np.argmax(z[i:i + len(SCALER_CHOICES)]))]; i += len(SCALER_CHOICES)
    n_layers = N_LAYER_CHOICES[int(np.argmax(z[i:i + len(N_LAYER_CHOICES)]))]
    config = {
        "model_type": "joint",
        "dloss": dloss,
        "variational": variational,
        "kan": kan,
        "class_triplet": class_triplet,
        "class_triplet_w": class_triplet_w,
        "lr": lr,
        "wd": wd,
        "nu": nu,
        "smoothing": smoothing,
        "margin": margin,
        "dropout": dropout,
        "thres": thres,
        "warmup": int(np.clip(warmup, 1, max_warmup)),
        "n_layers": n_layers,
        "layer1": int(np.clip(layer1, 512, 1024)),
        "log1p": True,
        "scaler": scaler,
        "gamma": gamma_candidate if dloss in hp_search.ADVERSARIAL_DLOSS else 0.0,
        "beta": beta_candidate if variational else 0.0,
    }
    return apply_fixed_categories(config, fixed_categories)


def normalize_meta(train_meta: np.ndarray, target_meta: np.ndarray | None = None):
    train = np.asarray(train_meta, dtype=np.float32)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    target = train if target_meta is None else np.asarray(target_meta, dtype=np.float32)
    return (target - mean) / scale, mean, scale
