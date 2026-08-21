from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, matthews_corrcoef


DATASETS = [
    "massbench_adenocarcinoma",
    "massbench_alzheimer",
    "massbench_benchmark",
]


def _safe_mcc(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute MCC while silencing sklearn's single-label confusion-matrix warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"A single label was found in 'y_true' and 'y_pred'.*",
            category=UserWarning,
            module=r"sklearn\.metrics\._classification",
        )
        return float(matthews_corrcoef(y_true, y_pred))


def evaluate_predictions(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    predicted_proba: object | None = None,
    groups: pd.DataFrame | pd.Series | None = None,
) -> dict[str, float | int]:
    """Score predictions against private reference labels.

    Args:
        predictions: DataFrame with columns [name, prediction] from user.
        reference:   DataFrame with columns [name, prediction] — ground truth.
    Returns:
        Dict with test_mcc (primary), accuracy, macro_f1, n_samples,
        optional calibration metrics, and optional group_scores.
    """
    merged = reference.merge(
        predictions.rename(columns={"prediction": "pred_user"}),
        on="name",
        how="left",
    )

    missing = int(merged["pred_user"].isna().sum())
    if missing:
        raise ValueError(
            f"Submission is missing predictions for {missing} sample(s)."
        )

    extra = len(predictions) - len(reference)
    if extra > 0:
        raise ValueError(
            f"Submission contains {extra} extra sample name(s) not in the reference."
        )

    y_true = merged["prediction"].astype(str).reset_index(drop=True)
    y_pred = merged["pred_user"].astype(str).reset_index(drop=True)

    result: dict[str, float | int | dict] = {
        "test_mcc": _safe_mcc(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "n_samples": int(len(merged)),
        # Fallback values when model probabilities are unavailable.
        "log_loss": -1.0,
        "brier_score": -1.0,
        "ece": -1.0,
    }

    labels = sorted(y_true.unique())
    y_true_idx = pd.Categorical(y_true, categories=labels).codes

    # Optional probability-based calibration metrics.
    if predicted_proba is not None:
        try:
            proba = np.asarray(predicted_proba, dtype=float)
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])
            if proba.ndim == 2 and len(proba) == len(y_true) and proba.shape[1] >= 2:
                row_sum = np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
                proba = proba / row_sum

                result["log_loss"] = float(log_loss(y_true, proba, labels=labels))

                if len(labels) == 2:
                    # For binary Brier score, use positive class probability.
                    pos_idx = 1
                    y_binary = (y_true_idx == pos_idx).astype(float)
                    result["brier_score"] = float(brier_score_loss(y_binary, proba[:, pos_idx]))
                else:
                    one_hot = np.eye(len(labels))[y_true_idx]
                    result["brier_score"] = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

                # Expected Calibration Error (ECE, 10 bins).
                conf = np.max(proba, axis=1)
                pred_idx = np.argmax(proba, axis=1)
                correct = (pred_idx == y_true_idx).astype(float)
                bins = np.linspace(0.0, 1.0, 11)
                ece = 0.0
                for i in range(10):
                    if i < 9:
                        mask = (conf >= bins[i]) & (conf < bins[i + 1])
                    else:
                        mask = (conf >= bins[i]) & (conf <= bins[i + 1])
                    if np.any(mask):
                        acc_bin = float(np.mean(correct[mask]))
                        conf_bin = float(np.mean(conf[mask]))
                        ece += (float(np.sum(mask)) / float(len(conf))) * abs(acc_bin - conf_bin)
                result["ece"] = float(ece)
        except Exception:
            # Keep evaluation robust when proba shape is invalid.
            pass

    # Optional per-group scoring (e.g. per batch).
    if groups is not None:
        try:
            if isinstance(groups, pd.Series):
                gdf = pd.DataFrame({"group": groups.astype(str).reset_index(drop=True)})
                gdf["name"] = reference["name"].astype(str).reset_index(drop=True)
            else:
                gdf = groups.copy()
                if "name" not in gdf.columns:
                    raise ValueError("groups must include a 'name' column")
                if "group" not in gdf.columns:
                    if "batch" in gdf.columns:
                        gdf = gdf.rename(columns={"batch": "group"})
                    else:
                        raise ValueError("groups must include a 'group' or 'batch' column")
                gdf = gdf[["name", "group"]]
                gdf["name"] = gdf["name"].astype(str)
                gdf["group"] = gdf["group"].astype(str)

            group_merged = merged.merge(gdf, on="name", how="left")
            by_group: dict[str, dict[str, float | int]] = {}
            for grp, part in group_merged.groupby("group", dropna=False):
                if part.empty:
                    continue
                y_t = part["prediction"].astype(str)
                y_p = part["pred_user"].astype(str)
                key = "__missing__" if pd.isna(grp) else str(grp)
                by_group[key] = {
                    "test_mcc": _safe_mcc(y_t, y_p),
                    "accuracy": float(accuracy_score(y_t, y_p)),
                    "macro_f1": float(f1_score(y_t, y_p, average="macro")),
                    "n_samples": int(len(part)),
                }

            result["group_scores"] = by_group
        except Exception:
            pass

    return result


def sorted_board(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    # Sort by test_mcc primarily
    if "test_mcc" in df.columns:
        return df.sort_values(["test_mcc", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    elif "mcc" in df.columns:
        return df.sort_values(["mcc", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    else:
        return df.sort_values("accuracy", ascending=False).reset_index(drop=True)


def append_result(
    board: pd.DataFrame,
    team: str,
    model: str,
    dataset: str,
    metrics: dict[str, float | int],
) -> pd.DataFrame:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "team": team,
        "model": model,
        "dataset": dataset,
        "test_mcc": round(metrics.get("test_mcc", metrics.get("mcc", 0.0)), 4),
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "n_samples": metrics["n_samples"],
    }
    return sorted_board(pd.concat([board, pd.DataFrame([row])], ignore_index=True))
