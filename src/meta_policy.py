"""Load and apply the durable BERNN evolutionary meta-policy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from src.evolutionary_meta import PolicyShape, decode_config
from src.zero_shot_recommender.meta_features import META_FEATURE_NAMES, extract_meta_features


META_POLICY_REPO = os.getenv("HF_META_POLICY_REPO", "spell0/BE-leaderboard-meta-model")
POLICY_FILE = "best_policy.npz"
METADATA_FILE = "best_policy.json"


def load_policy_metadata(repo_id: str = META_POLICY_REPO, *, token: str | None = None) -> dict[str, Any] | None:
    """Return persistent champion metadata, or ``None`` before first publication."""
    try:
        path = hf_hub_download(repo_id, METADATA_FILE, repo_type="model", token=token)
    except Exception:
        return None
    return json.loads(Path(path).read_text())


def predict_meta_bernn_config(X, y, batches, *, repo_id: str = META_POLICY_REPO) -> dict[str, Any]:
    """Predict BERNN hyperparameters for one dataset using the Hub champion."""
    path = hf_hub_download(repo_id, POLICY_FILE, repo_type="model")
    with np.load(path) as saved:
        required = {"genome", "meta_mean", "meta_scale", "n_inputs", "hidden_size"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Published meta-policy is missing arrays: {sorted(missing)}")
        genome = saved["genome"]
        mean = np.asarray(saved["meta_mean"], dtype=np.float32)
        scale = np.asarray(saved["meta_scale"], dtype=np.float32)
        shape = PolicyShape(
            int(np.asarray(saved["n_inputs"]).reshape(-1)[0]),
            int(np.asarray(saved["hidden_size"]).reshape(-1)[0]),
        )
    extracted = extract_meta_features(X, y, batches)
    raw = np.asarray([extracted[name] for name in META_FEATURE_NAMES], dtype=np.float32)
    if raw.shape != mean.shape or raw.shape != scale.shape:
        raise ValueError("Dataset meta-features do not match the published policy schema")
    normalized = (raw - mean) / np.where(scale < 1e-8, 1.0, scale)
    return decode_config(genome, normalized, shape)


def publish_policy_if_improved(
    policy_path: str | Path,
    metadata: dict[str, Any],
    *,
    repo_id: str = META_POLICY_REPO,
    token: str | None = None,
) -> tuple[bool, float | None]:
    """Atomically publish only when validation beats the persistent Hub score."""
    candidate = float(metadata["validation_score"])
    previous_metadata = load_policy_metadata(repo_id, token=token)
    previous = (
        float(previous_metadata["validation_score"])
        if previous_metadata is not None and "validation_score" in previous_metadata
        else None
    )
    if previous is not None and candidate <= previous:
        return False, previous

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(
                path_in_repo=POLICY_FILE, path_or_fileobj=str(policy_path)
            ),
            CommitOperationAdd(
                path_in_repo=METADATA_FILE,
                path_or_fileobj=json.dumps(metadata, indent=2, default=str).encode(),
            ),
        ],
        commit_message=f"Promote meta-policy at validation MCC {candidate:.6f}",
    )
    return True, previous
