"""Validation and private staging for community benchmark dataset proposals."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
STAGING_ROOT = ROOT / "data" / "dataset_submissions"
MAX_DATASET_BYTES = 50 * 1024 * 1024
REQUIRED_COLUMNS = ("name", "batch", "label")


class DatasetSubmissionError(ValueError):
    """Raised when a proposed dataset does not satisfy the contribution contract."""


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return clean[:64] or "dataset"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_proposal(
    csv_path: str | Path,
    *,
    title: str,
    version: str,
    description: str,
    modality: str,
    task: str,
    provenance: str,
    license_name: str,
    redistribution_confirmed: bool,
) -> tuple[pd.DataFrame, dict]:
    """Validate one matrix-ready CSV and return its frame and metadata."""
    path = Path(csv_path)
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise DatasetSubmissionError("Upload one matrix-ready CSV file.")
    size = path.stat().st_size
    if size > MAX_DATASET_BYTES:
        raise DatasetSubmissionError(
            f"Dataset is {size / (1024 * 1024):.1f} MiB; the hosted limit is 50 MiB."
        )
    required_text = {
        "title": title,
        "version": version,
        "description": description,
        "modality": modality,
        "task": task,
        "provenance": provenance,
        "license": license_name,
    }
    missing_text = [key for key, value in required_text.items() if not str(value).strip()]
    if missing_text:
        raise DatasetSubmissionError(
            "Missing required metadata: " + ", ".join(missing_text) + "."
        )
    if not redistribution_confirmed:
        raise DatasetSubmissionError(
            "Confirm redistribution rights and that the upload contains no restricted personal data."
        )

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DatasetSubmissionError(f"Could not read CSV: {exc}") from exc
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise DatasetSubmissionError(
            "Missing required columns: " + ", ".join(missing_columns) + "."
        )
    if frame.empty:
        raise DatasetSubmissionError("Dataset contains no samples.")
    if frame["name"].isna().any() or frame["name"].astype(str).str.strip().eq("").any():
        raise DatasetSubmissionError("Every sample requires a non-empty name.")
    if frame["name"].astype(str).duplicated().any():
        raise DatasetSubmissionError("Sample names must be unique.")
    if frame["batch"].isna().any() or frame["batch"].astype(str).str.strip().eq("").any():
        raise DatasetSubmissionError("Every sample requires a non-empty batch value.")
    if frame["batch"].astype(str).nunique() < 2:
        raise DatasetSubmissionError("At least two batches are required.")
    if frame["label"].isna().any() or frame["label"].astype(str).str.strip().eq("").any():
        raise DatasetSubmissionError("Every sample requires a label or an unlabeled sentinel.")
    labels = frame["label"].dropna().astype(str)
    labels = labels[~labels.isin({"", "-1", "pool", "unlabelled", "unlabeled"})]
    if labels.nunique() < 2:
        raise DatasetSubmissionError("At least two supervised biological classes are required.")
    feature_columns = [column for column in frame.columns if column not in REQUIRED_COLUMNS]
    if not feature_columns:
        raise DatasetSubmissionError("At least one numeric feature column is required.")
    non_numeric = [
        column for column in feature_columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        preview = ", ".join(map(str, non_numeric[:8]))
        raise DatasetSubmissionError(f"Feature columns must be numeric: {preview}.")
    feature_values = frame[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise DatasetSubmissionError("Feature columns cannot contain missing or infinite values.")

    metadata = {
        **{key: str(value).strip() for key, value in required_text.items()},
        "n_samples": int(len(frame)),
        "n_features": int(len(feature_columns)),
        "n_batches": int(frame["batch"].dropna().astype(str).nunique()),
        "n_classes": int(labels.nunique()),
        "size_bytes": int(size),
        "sha256": _sha256(path),
        "required_columns": list(REQUIRED_COLUMNS),
        "status": "pending_review",
    }
    return frame, metadata


def stage_dataset_proposal(csv_path: str | Path, submitted_by: str, **metadata) -> dict:
    """Validate and stage a proposal locally and, when configured, on the Hub."""
    _, record = validate_dataset_proposal(csv_path, **metadata)
    submission_id = f"{_slug(record['title'])}-{uuid.uuid4().hex[:12]}"
    destination = STAGING_ROOT / submission_id
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(csv_path, destination / "dataset.csv")
    record.update(
        {
            "submission_id": submission_id,
            "submitted_by": str(submitted_by),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    (destination / "metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    repo_id = os.getenv("HF_DATASET_SUBMISSIONS_REPO", "").strip()
    token = (
        os.getenv("HF_DATASET_SUBMISSIONS_TOKEN", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
    )
    if repo_id and token:
        HfApi(token=token).upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(destination),
            path_in_repo=f"pending/{submission_id}",
            commit_message=f"Stage dataset proposal {submission_id}",
        )
        record["durable_staging"] = "huggingface-private-dataset"
    else:
        record["durable_staging"] = "local-only"
    return record
