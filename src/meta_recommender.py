"""Deployment wrapper for the pretrained BERNN direct meta-network."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.direct_meta_checkpoint import (
    load_direct_meta_checkpoint,
    predict_direct_meta_config,
)
from src.zero_shot_recommender.meta_features import (
    META_FEATURE_NAMES,
    extract_meta_features,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "models" / "meta_bernn" / "best_meta_model.pt"
CHECKPOINT_ENV_VAR = "BERNN_META_CHECKPOINT"
_REQUIRED_COLUMNS = ("name", "batch", "label")


def resolve_checkpoint_path(path: str | Path | None = None) -> Path:
    """Resolve the deployment checkpoint path."""
    if path is not None:
        resolved = Path(path).expanduser()
    else:
        configured = os.getenv(CHECKPOINT_ENV_VAR)
        resolved = Path(configured).expanduser() if configured else DEFAULT_CHECKPOINT

    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.resolve()


@lru_cache(maxsize=4)
def _load_cached_checkpoint(path_text: str):
    """Load and cache one checkpoint for the lifetime of the app process."""
    return load_direct_meta_checkpoint(Path(path_text))


def load_recommender(path: str | Path | None = None):
    """Return the cached pretrained meta-model and checkpoint payload."""
    checkpoint_path = resolve_checkpoint_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Pretrained BERNN meta-network checkpoint was not found at "
            f"{checkpoint_path}. Add the selected .pt file there or set "
            f"{CHECKPOINT_ENV_VAR}."
        )
    return _load_cached_checkpoint(str(checkpoint_path))


def recommender_evaluation_protocol(
    checkpoint_path: str | Path | None = None,
) -> str:
    """Return the data/evaluation protocol recorded in the selected checkpoint."""
    _, checkpoint = load_recommender(checkpoint_path)
    raw = str(
        checkpoint.get("metadata", {}).get(
            "evaluation_protocol",
            "fixed_external_test_v1",
        )
    )
    if raw in {"cyclic_batches", "cyclic_train_valid_test_by_batch_v1"}:
        return "cyclic_batches"
    return "fixed_external"


def validate_recommender_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a dataframe for zero-shot BERNN recommendation."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Dataset is empty.")

    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Dataset must contain the columns name, batch, and label. "
            f"Missing: {', '.join(missing)}"
        )

    feature_columns = [
        column for column in df.columns if column not in _REQUIRED_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("Dataset must contain at least one numeric feature column.")

    normalized = df.copy()
    normalized["name"] = normalized["name"].astype(str)
    normalized["batch"] = normalized["batch"].astype(str)
    normalized["label"] = normalized["label"].astype(str)

    if normalized["name"].duplicated().any():
        raise ValueError("Sample names must be unique.")
    if normalized["batch"].nunique() < 2:
        raise ValueError("At least two batches are required.")
    if normalized["label"].nunique() < 2:
        raise ValueError("At least two classes are required.")

    original = normalized[feature_columns]
    numeric = original.apply(pd.to_numeric, errors="coerce")

    # Missing/non-finite numeric values are intentional inputs to
    # extract_meta_features(): that function measures missingness and replaces
    # non-finite values only for geometry calculations. Reject only cells that
    # contained a non-empty, non-numeric value.
    original_nonempty = original.notna() & original.astype(str).apply(
        lambda column: column.str.strip().ne("")
    )
    failed_parse = original_nonempty & numeric.isna()
    if failed_parse.to_numpy().any():
        bad_count = int(failed_parse.to_numpy().sum())
        bad_columns = list(failed_parse.columns[failed_parse.any(axis=0)][:5])
        raise ValueError(
            f"Feature matrix contains {bad_count} non-numeric values "
            f"(example columns: {bad_columns}). Missing numeric values are allowed."
        )

    normalized.loc[:, feature_columns] = numeric
    return normalized


def extract_dataframe_meta_features(df: pd.DataFrame) -> dict[str, float]:
    """Calculate the deterministic dataset descriptors used by the model."""
    normalized = validate_recommender_dataframe(df)
    feature_columns = [
        column for column in normalized.columns if column not in _REQUIRED_COLUMNS
    ]
    return extract_meta_features(
        normalized[feature_columns],
        normalized["label"],
        normalized["batch"],
    )


def recommend_bernn_config(
    df: pd.DataFrame,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recommend BERNN hyperparameters for a dataframe."""
    meta_features = extract_dataframe_meta_features(df)
    model, checkpoint = load_recommender(checkpoint_path)
    config = predict_direct_meta_config(model, checkpoint, meta_features)

    return {
        "config": config,
        "meta_features": {
            name: float(meta_features[name]) for name in META_FEATURE_NAMES
        },
        "checkpoint_path": str(resolve_checkpoint_path(checkpoint_path)),
        "checkpoint_metadata": dict(checkpoint.get("metadata", {})),
        "format_version": checkpoint.get("format_version"),
        "torch_version": checkpoint.get("torch_version"),
    }


def recommendation_tables(
    result: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a recommendation result into compact UI tables."""
    config = result.get("config", {})
    config_table = pd.DataFrame(
        [{"hyperparameter": key, "value": value} for key, value in config.items()]
    )
    meta = result.get("meta_features", {})
    meta_table = pd.DataFrame(
        [{"meta_feature": key, "value": float(value)} for key, value in meta.items()]
    )
    return config_table, meta_table
