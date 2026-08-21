"""Durable Hugging Face Dataset storage for Real leaderboard rows."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from src.database import real_leaderboard_score

log = logging.getLogger(__name__)

REAL_RESULTS_DATASET = os.getenv(
    "HF_REAL_RESULTS_DATASET", "spell0/batch-effects-leaderboard-results"
)
REAL_RESULTS_FILE = os.getenv("HF_REAL_RESULTS_FILE", "real_leaderboard.json")
REAL_RESULTS_SYNC_ENABLED = os.getenv("HF_REAL_RESULTS_SYNC", "").lower() in {
    "1",
    "true",
    "yes",
} or bool(
    os.getenv("SPACE_ID")
    or os.getenv("SPACE_AUTHOR_NAME")
    or os.getenv("SPACE_REPO_NAME")
    or os.getenv("SPACE_HOST")
)


def _token() -> str | None:
    return os.getenv("HF_TOKEN")


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        numeric = float(value)
        if not math.isfinite(numeric):
            return default
        return numeric
    except Exception:
        return default


def _finite_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        numeric = float(value)
        if not math.isfinite(numeric):
            return default
        return int(numeric)
    except Exception:
        return default


def _folds(value: Any) -> list[float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split("|")]
        return [float(part) for part in parts if part]
    if isinstance(value, list | tuple):
        return [
            float(item) for item in value
            if _finite_float(item) is not None
        ]
    return []


def _created_at(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value:
        return str(value)
    return datetime.now(timezone.utc).isoformat()


def normalize_real_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a public, JSON-safe Real leaderboard row."""
    is_public = bool(row.get("is_public", False))
    valid_mcc = _finite_float(row.get("valid_mcc"), 0.0)
    test_mcc = _finite_float(row.get("test_mcc"), 0.0)
    normalized = {
        "username": str(row.get("username") or "anonymous"),
        "dataset": str(row.get("dataset") or "massbench_benchmark"),
        "submission_name": str(row.get("submission_name") or "submission"),
        "score": real_leaderboard_score(valid_mcc, test_mcc),
        "test_mcc": test_mcc,
        "valid_mcc": valid_mcc,
        "valid_mcc_folds": _folds(row.get("valid_mcc_folds", [])),
        "train_mcc": _finite_float(row.get("train_mcc"), -1.0),
        "accuracy": _finite_float(row.get("accuracy"), 0.0),
        "macro_f1": _finite_float(row.get("macro_f1"), 0.0),
        "n_samples": _finite_int(row.get("n_samples"), 0),
        "log_loss": _finite_float(row.get("log_loss")) if "log_loss" in row else None,
        "brier_score": _finite_float(row.get("brier_score")) if "brier_score" in row else None,
        "ece": _finite_float(row.get("ece")) if "ece" in row else None,
        "batch_silhouette": _finite_float(row.get("batch_silhouette")) if "batch_silhouette" in row else None,
        "batch_centroid_dispersion": _finite_float(row.get("batch_centroid_dispersion")) if "batch_centroid_dispersion" in row else None,
        "batch_nbe": _finite_float(row.get("batch_nbe")) if "batch_nbe" in row else None,
        "batch_nmi": _finite_float(row.get("batch_nmi")) if "batch_nmi" in row else None,
        "batch_nri": _finite_float(row.get("batch_nri")) if "batch_nri" in row else None,
        "created_at": _created_at(row.get("created_at")),
        "version_created": str(row.get("version_created") or ""),
        "version_evaluated": str(row.get("version_evaluated") or ""),
        "is_public": is_public,
        "correction_code": str(row.get("correction_code") or "") if is_public else "",
        "model_code": str(row.get("model_code") or "") if is_public else "",
    }
    normalized["result_id"] = real_result_id(normalized)
    return normalized


def real_result_id(row: dict[str, Any]) -> str:
    """Stable identity for deduplicating rows across Space restarts."""
    identity = {
        "username": row.get("username"),
        "dataset": row.get("dataset"),
        "submission_name": row.get("submission_name"),
        "test_mcc": _finite_float(row.get("test_mcc"), 0.0),
        "valid_mcc": _finite_float(row.get("valid_mcc"), 0.0),
        "accuracy": _finite_float(row.get("accuracy"), 0.0),
        "macro_f1": _finite_float(row.get("macro_f1"), 0.0),
        "n_samples": _finite_int(row.get("n_samples"), 0),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def merge_real_result_rows(*row_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rows in row_sets:
        for row in rows:
            normalized = normalize_real_result_row(row)
            merged[normalized["result_id"]] = normalized
    return sorted(
        merged.values(),
        key=lambda row: (
            float(row.get("score") or 0.0),
            float(row.get("accuracy") or 0.0),
            str(row.get("created_at") or ""),
        ),
        reverse=True,
    )


def load_real_result_rows() -> list[dict[str, Any]]:
    if not REAL_RESULTS_DATASET:
        return []
    try:
        path = hf_hub_download(
            repo_id=REAL_RESULTS_DATASET,
            filename=REAL_RESULTS_FILE,
            repo_type="dataset",
            token=_token(),
        )
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        return merge_real_result_rows(rows)
    except Exception as exc:
        log.info("Could not load Real leaderboard results from Hugging Face: %s", exc)
        return []


def upload_real_result_rows(rows: list[dict[str, Any]]) -> int:
    if not REAL_RESULTS_DATASET:
        log.info("HF_REAL_RESULTS_DATASET not set; Real leaderboard saved locally only.")
        return 0
    if not REAL_RESULTS_SYNC_ENABLED:
        log.info(
            "Real leaderboard sync disabled outside Spaces; set HF_REAL_RESULTS_SYNC=1 to enable."
        )
        return 0
    token = _token()
    if not token:
        log.warning("HF_TOKEN not set; skipping Real leaderboard push to Hugging Face.")
        return 0

    merged = merge_real_result_rows(load_real_result_rows(), rows)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Public aggregate Real leaderboard results for the Batch Effects Leaderboard Space.",
        "rows": merged,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        temp_path = fh.name

    try:
        api = HfApi(token=token)
        api.create_repo(
            repo_id=REAL_RESULTS_DATASET,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo=REAL_RESULTS_FILE,
            repo_id=REAL_RESULTS_DATASET,
            repo_type="dataset",
            commit_message="Update Real leaderboard results",
        )
        log.info("Real leaderboard pushed to Hugging Face: %s/%s", REAL_RESULTS_DATASET, REAL_RESULTS_FILE)
        return len(merged)
    except Exception as exc:
        log.warning("Hugging Face Real leaderboard push failed: %s", exc)
        return 0
    finally:
        try:
            Path(temp_path).unlink()
        except OSError:
            pass
