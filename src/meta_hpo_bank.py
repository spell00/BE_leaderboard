"""Utilities for reusable BERNN meta-HPO trial banks.

The trial bank is the durable expensive artifact.  It contains independent
Optuna trials for the four meta-training datasets plus an Alzheimer-only Optuna
baseline.  Source trials may train meta-models; Alzheimer baseline trials may
never do so.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts import hp_search

SOURCE_ROLE = "source"
ALZHEIMER_BASELINE_ROLE = "alzheimer_baseline"

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

# log1p and model_type are protocol constants in these experiments.
CONFIG_KEYS = (
    "model_type", "dloss", "variational", "kan", "class_triplet",
    "class_triplet_w", "lr", "wd", "nu", "smoothing", "margin",
    "dropout", "thres", "warmup", "n_layers", "layer1", "log1p",
    "scaler", "gamma", "beta",
)


@dataclass(frozen=True)
class BankTrial:
    dataset_id: str
    role: str
    trial_index: int
    optuna_trial_number: int
    valid_mcc: float
    test_mcc: float
    fit_seconds: float
    config: dict[str, Any]
    valid_mcc_folds: tuple[float, ...] = ()
    test_mcc_folds: tuple[float, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "role": self.role,
            "trial_index": int(self.trial_index),
            "optuna_trial_number": int(self.optuna_trial_number),
            "valid_mcc": float(self.valid_mcc),
            "test_mcc": float(self.test_mcc),
            "fit_seconds": float(self.fit_seconds),
            "config": canonical_config(self.config),
            "valid_mcc_folds": [float(v) for v in self.valid_mcc_folds],
            "test_mcc_folds": [float(v) for v in self.test_mcc_folds],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BankTrial":
        return cls(
            dataset_id=str(value["dataset_id"]),
            role=str(value["role"]),
            trial_index=int(value["trial_index"]),
            optuna_trial_number=int(value.get("optuna_trial_number", value["trial_index"])),
            valid_mcc=float(value["valid_mcc"]),
            test_mcc=float(value.get("test_mcc", np.nan)),
            fit_seconds=float(value.get("fit_seconds", np.nan)),
            config=canonical_config(value["config"]),
            valid_mcc_folds=tuple(float(v) for v in value.get("valid_mcc_folds", ())),
            test_mcc_folds=tuple(float(v) for v in value.get("test_mcc_folds", ())),
            error=value.get("error"),
        )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    out = {key: _json_scalar(config[key]) for key in CONFIG_KEYS if key in config}
    out.setdefault("model_type", "joint")
    out.setdefault("log1p", True)
    if out.get("dloss", "no") not in hp_search.ADVERSARIAL_DLOSS:
        out["gamma"] = 0.0
    if not bool(out.get("variational", False)):
        out["beta"] = 0.0
    if not bool(out.get("class_triplet", False)):
        out["class_triplet_w"] = 0.0
    return out


def config_digest(config: dict[str, Any], *, precision: int = 10) -> str:
    normalized = {}
    for key, value in canonical_config(config).items():
        if isinstance(value, float) and math.isfinite(value):
            value = round(value, int(precision))
        normalized[key] = value
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_digest(config: dict[str, Any], protocol: dict[str, Any]) -> str:
    payload = {
        "config": canonical_config(config),
        "protocol": {key: _json_scalar(value) for key, value in protocol.items()},
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_bank(trials: Iterable[BankTrial], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (trial.to_dict() for trial in trials),
        key=lambda row: (row["role"], row["dataset_id"], row["trial_index"]),
    )

    jsonl = output / "trial_bank.jsonl"
    tmp = jsonl.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=str) + "\n")
    tmp.replace(jsonl)

    csv_path = output / "trial_bank.csv"
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    fields = (
        "dataset_id", "role", "trial_index", "optuna_trial_number",
        "valid_mcc", "test_mcc", "fit_seconds", "error", "config_json",
    )
    with tmp_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "dataset_id": row["dataset_id"],
                "role": row["role"],
                "trial_index": row["trial_index"],
                "optuna_trial_number": row["optuna_trial_number"],
                "valid_mcc": row["valid_mcc"],
                "test_mcc": row["test_mcc"],
                "fit_seconds": row["fit_seconds"],
                "error": row["error"],
                "config_json": json.dumps(row["config"], sort_keys=True),
            })
    tmp_csv.replace(csv_path)


def load_bank(path_or_dir: str | Path) -> list[BankTrial]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "trial_bank.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Trial bank not found: {path}")
    trials = [
        BankTrial.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    trials.sort(key=lambda t: (t.role, t.dataset_id, t.trial_index))
    return trials


def source_trials(trials: Iterable[BankTrial]) -> list[BankTrial]:
    return [trial for trial in trials if trial.role == SOURCE_ROLE]


def alzheimer_baseline_trials(trials: Iterable[BankTrial]) -> list[BankTrial]:
    return [trial for trial in trials if trial.role == ALZHEIMER_BASELINE_ROLE]


def source_dataset_ids(trials: Iterable[BankTrial]) -> tuple[str, ...]:
    return tuple(sorted({trial.dataset_id for trial in trials if trial.role == SOURCE_ROLE}))


def trials_at_prefix(
    trials: Iterable[BankTrial],
    prefix: int,
    *,
    role: str = SOURCE_ROLE,
) -> list[BankTrial]:
    limit = int(prefix)
    return [
        trial for trial in trials
        if trial.role == role and int(trial.trial_index) < limit
    ]


def best_by_dataset(
    trials: Iterable[BankTrial],
    dataset_ids: Iterable[str],
) -> dict[str, BankTrial]:
    dataset_ids = tuple(dataset_ids)
    grouped: dict[str, list[BankTrial]] = {name: [] for name in dataset_ids}
    for trial in trials:
        if trial.dataset_id in grouped and np.isfinite(trial.valid_mcc):
            grouped[trial.dataset_id].append(trial)
    missing = [name for name, rows in grouped.items() if not rows]
    if missing:
        raise ValueError(f"Missing completed trials for: {missing}")
    return {
        name: max(rows, key=lambda row: float(row.valid_mcc))
        for name, rows in grouped.items()
    }


def alzheimer_baseline_curve(trials: Iterable[BankTrial]) -> list[dict[str, float]]:
    rows = sorted(alzheimer_baseline_trials(trials), key=lambda t: t.trial_index)
    best = -np.inf
    curve = []
    for trial in rows:
        best = max(best, float(trial.valid_mcc))
        curve.append({
            "trial_index": int(trial.trial_index),
            "current_valid_mcc": float(trial.valid_mcc),
            "best_valid_mcc": float(best),
        })
    return curve


def categorical_consensus(
    trials: Iterable[BankTrial],
    dataset_ids: Iterable[str],
    *,
    prefix: int | None = None,
    top_k: int = 5,
    min_support: float = 0.80,
) -> dict[str, Any]:
    """Return strict champion consensus and a top-K-supported variant.

    Strict consensus matches the requested rule literally: if all four
    independently optimized source champions have the same categorical value,
    that field is eligible to be frozen.  Robust consensus additionally requires
    the same value to dominate the top-K trials of every source dataset.
    """
    dataset_ids = tuple(dataset_ids)
    rows = [trial for trial in trials if trial.role == SOURCE_ROLE]
    if prefix is not None:
        rows = [trial for trial in rows if trial.trial_index < int(prefix)]
    grouped = {name: [] for name in dataset_ids}
    for trial in rows:
        if trial.dataset_id in grouped and np.isfinite(trial.valid_mcc):
            grouped[trial.dataset_id].append(trial)
    for name in dataset_ids:
        grouped[name].sort(key=lambda t: t.valid_mcc, reverse=True)
        if not grouped[name]:
            raise ValueError(f"No source trials for {name}")

    strict_fixed: dict[str, Any] = {}
    robust_fixed: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for field in CATEGORICAL_FIELDS:
        champion_values = {
            name: _json_scalar(grouped[name][0].config.get(field))
            for name in dataset_ids
        }
        values = list(champion_values.values())
        strict_value = values[0] if all(value == values[0] for value in values) else None
        if strict_value is not None:
            strict_fixed[field] = strict_value

        robust = strict_value is not None
        per_dataset = {}
        for name in dataset_ids:
            selected = grouped[name][: max(1, min(int(top_k), len(grouped[name])))]
            selected_values = [_json_scalar(row.config.get(field)) for row in selected]
            counts = Counter(selected_values)
            mode, count = counts.most_common(1)[0]
            support = float(count / len(selected_values))
            per_dataset[name] = {
                "champion": champion_values[name],
                "top_k_n": len(selected_values),
                "mode": _json_scalar(mode),
                "mode_support": support,
                "counts": {json.dumps(_json_scalar(k)): int(v) for k, v in counts.items()},
            }
            robust = bool(robust and mode == strict_value and support >= float(min_support))
        if robust:
            robust_fixed[field] = strict_value
        diagnostics[field] = {
            "champions": champion_values,
            "strict_consensus": strict_value,
            "robust_consensus": strict_value if robust else None,
            "datasets": per_dataset,
        }

    return {
        "schema_version": 2,
        "source_dataset_ids": list(dataset_ids),
        "prefix": None if prefix is None else int(prefix),
        "top_k": int(top_k),
        "min_support": float(min_support),
        "categorical_fields": list(CATEGORICAL_FIELDS),
        "strict_fixed": strict_fixed,
        "robust_fixed": robust_fixed,
        "diagnostics": diagnostics,
    }


def load_fixed_categories(
    path: str | Path | None,
    *,
    policy: str = "strict",
) -> dict[str, Any]:
    if path is None or str(path).lower() in {"", "none"}:
        return {}
    payload = json.loads(Path(path).read_text())
    if policy == "none":
        return {}
    if policy not in {"strict", "robust"}:
        raise ValueError("freeze policy must be one of: none, strict, robust")
    fixed = payload.get(f"{policy}_fixed", {})
    return {key: value for key, value in fixed.items() if key in CATEGORICAL_FIELDS}


def apply_fixed_categories(config: dict[str, Any], fixed: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(config)
    for key, value in (fixed or {}).items():
        if key not in CATEGORICAL_FIELDS:
            continue
        if key in {"variational", "kan", "class_triplet"}:
            out[key] = bool(value)
        elif key == "n_layers":
            out[key] = int(value)
        else:
            out[key] = str(value)
    return canonical_config(out)


def _scale01(value: float, low: float, high: float) -> float:
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def _unscale01(value: float, low: float, high: float) -> float:
    return float(low + np.clip(float(value), 0.0, 1.0) * (high - low))


def _log_scale01(value: float, low: float, high: float) -> float:
    value = float(np.clip(float(value), low, high))
    return _scale01(math.log(value), math.log(low), math.log(high))


def _log_unscale01(value: float, low: float, high: float) -> float:
    return float(math.exp(math.log(low) + np.clip(float(value), 0.0, 1.0) * (math.log(high) - math.log(low))))


CONTINUOUS_FIELDS = (
    "class_triplet_w", "lr", "wd", "nu", "smoothing", "margin",
    "dropout", "thres", "warmup", "layer1", "gamma", "beta",
)


def encode_continuous(config: dict[str, Any], max_warmup: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = canonical_config(config)
    values = np.asarray([
        _scale01(cfg.get("class_triplet_w", 0.0), 0.0, 1.0),
        _log_scale01(cfg["lr"], 1e-4, 1e-2),
        _log_scale01(cfg["wd"], 1e-6, 1e-3),
        _scale01(cfg["nu"], 1e-4, 1e2),
        _scale01(cfg["smoothing"], 0.0, 0.2),
        _scale01(cfg["margin"], 0.0, 10.0),
        _scale01(cfg["dropout"], 0.0, 0.5),
        _scale01(cfg["thres"], 0.0, 0.1),
        _scale01(cfg["warmup"], 1.0, float(max_warmup)),
        _scale01(cfg["layer1"], 512.0, 1024.0),
        _log_scale01(cfg.get("gamma") or 1e-2, 1e-2, 1e2),
        _log_scale01(cfg.get("beta") or 1e-2, 1e-2, 1e2),
    ], dtype=np.float32)
    mask = np.asarray([
        float(bool(cfg.get("class_triplet", False))),
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        float(cfg.get("dloss", "no") in hp_search.ADVERSARIAL_DLOSS),
        float(bool(cfg.get("variational", False))),
    ], dtype=np.float32)
    return values, mask


def decode_continuous(values: np.ndarray, config: dict[str, Any], max_warmup: int) -> dict[str, Any]:
    z = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    if z.shape != (len(CONTINUOUS_FIELDS),):
        raise ValueError(f"Expected {len(CONTINUOUS_FIELDS)} continuous values, got {z.shape}")
    out = dict(config)
    out.update({
        "class_triplet_w": _unscale01(z[0], 0.0, 1.0),
        "lr": _log_unscale01(z[1], 1e-4, 1e-2),
        "wd": _log_unscale01(z[2], 1e-6, 1e-3),
        "nu": _unscale01(z[3], 1e-4, 1e2),
        "smoothing": _unscale01(z[4], 0.0, 0.2),
        "margin": _unscale01(z[5], 0.0, 10.0),
        "dropout": _unscale01(z[6], 0.0, 0.5),
        "thres": _unscale01(z[7], 0.0, 0.1),
        "warmup": int(round(_unscale01(z[8], 1.0, float(max_warmup)))),
        "layer1": int(round(_unscale01(z[9], 512.0, 1024.0))),
        "gamma": _log_unscale01(z[10], 1e-2, 1e2),
        "beta": _log_unscale01(z[11], 1e-2, 1e2),
    })
    return canonical_config(out)


def config_feature_vector(config: dict[str, Any], max_warmup: int) -> np.ndarray:
    """Fixed-width vector used by score surrogates."""
    cfg = canonical_config(config)
    continuous, _ = encode_continuous(cfg, max_warmup)
    return np.asarray([
        float(bool(cfg.get("variational", False))),
        float(bool(cfg.get("kan", False))),
        float(bool(cfg.get("class_triplet", False))),
        *continuous.tolist(),
        *[float(cfg.get("dloss") == value) for value in DLOSS_CHOICES],
        *[float(cfg.get("scaler") == value) for value in SCALER_CHOICES],
        *[float(int(cfg.get("n_layers", 1)) == value) for value in N_LAYER_CHOICES],
    ], dtype=np.float32)


def sample_config(trial, hp_args, fixed_categories: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sample BERNN hparams while freezing categorical fields only when requested."""
    fixed = fixed_categories or {}

    def cat(name: str, choices):
        if name in fixed:
            value = fixed[name]
            if name == "n_layers":
                value = int(value)
            if value not in choices:
                raise ValueError(f"Fixed {name}={value!r} not in {tuple(choices)!r}")
            return value
        return trial.suggest_categorical(name, list(choices))

    dloss = str(cat("dloss", DLOSS_CHOICES))
    variational = bool(cat("variational", (False, True)))
    kan = bool(cat("kan", (False, True)))
    class_triplet = bool(cat("class_triplet", (False, True)))
    scaler = str(cat("scaler", SCALER_CHOICES))
    n_layers = int(cat("n_layers", N_LAYER_CHOICES))
    max_warmup = int(getattr(hp_args, "max_warmup", max(1, min(50, int(hp_args.n_epochs)))))

    config = {
        "model_type": "joint",
        "dloss": dloss,
        "variational": variational,
        "kan": kan,
        "class_triplet": class_triplet,
        "class_triplet_w": trial.suggest_float("class_triplet_w", 0.0, 1.0) if class_triplet else 0.0,
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
