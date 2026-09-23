"""Single-target learning-curve extrapolation for multi-fidelity HPO.

This module predicts one quantity only: the eventual validation total
(currently final CV validation MCC). Ranking and uncertainty are diagnostics
derived from predictions, never additional training objectives.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.meta_hpo_bank import config_feature_vector


CURVE_FEATURE_NAMES = (
    "budget_fraction",
    "last_total",
    "best_total",
    "mean_total",
    "std_total",
    "recent_slope",
    "current_fold_valid_mcc",
    "best_current_fold_valid_mcc",
)


def _clean_history(history: Iterable[dict]) -> list[dict]:
    rows = []
    seen_steps = set()
    for raw in history:
        try:
            step = int(raw["step"])
            score = float(raw["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if step <= 0 or not np.isfinite(score) or step in seen_steps:
            continue
        row = dict(raw)
        row["step"] = step
        row["score"] = score
        rows.append(row)
        seen_steps.add(step)
    rows.sort(key=lambda row: row["step"])
    return rows


def curve_summary(history: Iterable[dict], max_resource: int) -> np.ndarray:
    rows = _clean_history(history)
    if not rows:
        raise ValueError("At least one finite learning-curve observation is required")
    steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    recent_n = min(5, len(rows))
    recent_x = steps[-recent_n:]
    recent_y = scores[-recent_n:]
    if recent_n >= 2 and np.ptp(recent_x) > 0:
        slope = float(np.polyfit(recent_x, recent_y, 1)[0] * max(1, int(max_resource)))
    else:
        slope = 0.0
    last = rows[-1]
    current_fold = float(last.get("current_fold_valid_mcc", scores[-1]))
    best_current_fold = float(last.get("best_current_fold_valid_mcc", current_fold))
    return np.asarray([
        float(np.clip(steps[-1] / max(1, int(max_resource)), 0.0, 1.0)),
        float(scores[-1]),
        float(np.max(scores)),
        float(np.mean(scores)),
        float(np.std(scores)),
        slope,
        current_fold,
        best_current_fold,
    ], dtype=np.float32)


def prefix_histories(
    history: Iterable[dict],
    *,
    max_prefixes: int = 6,
    min_points: int = 2,
) -> list[list[dict]]:
    rows = _clean_history(history)
    if len(rows) < max(1, int(min_points)):
        return []
    start = max(1, int(min_points)) - 1
    candidates = np.arange(start, len(rows), dtype=int)
    if len(candidates) > int(max_prefixes):
        chosen = np.unique(
            np.linspace(start, len(rows) - 1, int(max_prefixes)).round().astype(int)
        )
    else:
        chosen = candidates
    return [rows[: int(index) + 1] for index in chosen]


@dataclass(frozen=True)
class CompletedCurveTrial:
    config: dict[str, Any]
    final_total: float
    history: tuple[dict, ...]
    max_resource: int
    dataset_meta: tuple[float, ...] = ()


class PartialCurveTotalRegressor:
    """ExtraTrees extrapolator with one scalar target: final observed total."""

    def __init__(self, model, max_warmup: int, meta_mean, meta_scale):
        self.model = model
        self.max_warmup = int(max_warmup)
        self.meta_mean = np.asarray(meta_mean, dtype=np.float32)
        self.meta_scale = np.asarray(meta_scale, dtype=np.float32)

    def _features(
        self,
        config: dict[str, Any],
        history: Iterable[dict],
        max_resource: int,
        dataset_meta: Iterable[float] = (),
    ) -> np.ndarray:
        meta = np.asarray(tuple(dataset_meta), dtype=np.float32)
        if self.meta_mean.size:
            if meta.shape != self.meta_mean.shape:
                raise ValueError(
                    f"Expected {self.meta_mean.size} dataset meta-features, got {meta.size}"
                )
            meta = (meta - self.meta_mean) / self.meta_scale
        elif meta.size:
            raise ValueError("This regressor was trained without dataset meta-features")
        return np.concatenate([
            meta,
            config_feature_vector(config, self.max_warmup),
            curve_summary(history, max_resource),
        ]).astype(np.float32)

    def predict(
        self,
        config: dict[str, Any],
        history: Iterable[dict],
        max_resource: int,
        dataset_meta: Iterable[float] = (),
    ) -> tuple[float, float]:
        X = self._features(config, history, max_resource, dataset_meta)[None, :]
        per_tree = np.asarray(
            [float(tree.predict(X)[0]) for tree in self.model.estimators_],
            dtype=float,
        )
        return float(np.mean(per_tree)), float(np.std(per_tree))


def fit_partial_curve_total_regressor(
    trials: Iterable[CompletedCurveTrial],
    *,
    max_warmup: int,
    n_estimators: int = 300,
    min_samples_leaf: int = 2,
    max_prefixes_per_trial: int = 6,
    seed: int = 42,
) -> PartialCurveTotalRegressor:
    from sklearn.ensemble import ExtraTreesRegressor

    trials = [
        trial for trial in trials
        if np.isfinite(float(trial.final_total)) and _clean_history(trial.history)
    ]
    if len(trials) < 3:
        raise ValueError("At least 3 completed curve trials are required")

    meta_sizes = {len(trial.dataset_meta) for trial in trials}
    if len(meta_sizes) != 1:
        raise ValueError("All completed trials must have the same dataset meta-feature width")
    meta_size = next(iter(meta_sizes))
    if meta_size:
        meta_matrix = np.asarray([trial.dataset_meta for trial in trials], dtype=np.float32)
        meta_mean = meta_matrix.mean(axis=0)
        meta_scale = meta_matrix.std(axis=0)
        meta_scale[meta_scale < 1e-8] = 1.0
    else:
        meta_mean = np.empty(0, dtype=np.float32)
        meta_scale = np.empty(0, dtype=np.float32)

    X_rows, y_rows = [], []
    for trial in trials:
        prefixes = prefix_histories(
            trial.history,
            max_prefixes=max_prefixes_per_trial,
            min_points=2,
        )
        for prefix in prefixes:
            meta = np.asarray(trial.dataset_meta, dtype=np.float32)
            if meta_size:
                meta = (meta - meta_mean) / meta_scale
            X_rows.append(np.concatenate([
                meta,
                config_feature_vector(trial.config, int(max_warmup)),
                curve_summary(prefix, trial.max_resource),
            ]))
            y_rows.append(float(trial.final_total))

    if len(X_rows) < 3:
        raise ValueError("Not enough curve prefixes to fit the final-total regressor")
    model = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=max(1, int(min_samples_leaf)),
        max_features=0.8,
        bootstrap=True,
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32))
    return PartialCurveTotalRegressor(model, max_warmup, meta_mean, meta_scale)


def _read_curve(run_dir: Path, row: dict) -> tuple[dict, ...]:
    rel = row.get("curve_path")
    if not rel:
        return ()
    path = run_dir / str(rel)
    if not path.exists():
        return ()
    return tuple(json.loads(path.read_text()))


def backfill_multifidelity_totals(
    run_dir: str | Path,
    *,
    max_warmup: int,
    seed: int = 42,
    min_completed: int = 3,
) -> dict[str, int]:
    """Predict final totals for pruned trials without changing Optuna labels."""
    run_dir = Path(run_dir)
    path = run_dir / "multifidelity_trials.json"
    if not path.exists():
        return {"completed": 0, "predicted": 0}
    rows = json.loads(path.read_text())
    completed = []
    for row in rows:
        if row.get("state") != "COMPLETE" or row.get("observed_total") is None:
            continue
        history = _read_curve(run_dir, row)
        if not history:
            continue
        completed.append(CompletedCurveTrial(
            config=dict(row.get("config", {})),
            final_total=float(row["observed_total"]),
            history=history,
            max_resource=int(row.get("max_resource") or 1),
        ))
    if len(completed) < int(min_completed):
        return {"completed": len(completed), "predicted": 0}

    predictor = fit_partial_curve_total_regressor(
        completed,
        max_warmup=int(max_warmup),
        seed=int(seed),
    )
    predicted = 0
    for row in rows:
        if row.get("state") != "PRUNED":
            continue
        history = _read_curve(run_dir, row)
        if not history:
            continue
        mean, std = predictor.predict(
            dict(row.get("config", {})),
            history,
            int(row.get("max_resource") or 1),
        )
        row["predicted_total"] = float(mean)
        row["predicted_total_std"] = float(std)
        row["total"] = float(mean)
        row["total_source"] = "predicted"
        predicted += 1

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    tmp.replace(path)
    return {"completed": len(completed), "predicted": predicted}
