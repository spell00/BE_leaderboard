"""Restore leaderboard datasets from the dedicated Hugging Face storage Space."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_STORAGE_REPO = "spell0/Batch-effects-leaderboard-storage"


def ensure_hf_datasets(root: Path) -> None:
    """Download data/datasets from the storage Space when running on HF Spaces."""

    # Local development already has the datasets checked out.
    if not os.getenv("SPACE_ID"):
        return

    if os.getenv("BE_SKIP_DATASET_SYNC", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print("[hf-storage] Dataset synchronization disabled.", flush=True)
        return

    repo_id = os.getenv(
        "BE_DATA_STORAGE_REPO",
        DEFAULT_STORAGE_REPO,
    ).strip()

    print(
        f"[hf-storage] Restoring data/datasets from {repo_id}...",
        flush=True,
    )

    snapshot_download(
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=["data/datasets/**"],
        local_dir=str(root),
        token=os.getenv("HF_TOKEN") or None,
    )

    dataset_dir = root / "data" / "datasets"

    if not dataset_dir.exists():
        raise RuntimeError(
            f"Dataset synchronization completed but {dataset_dir} does not exist."
        )

    csv_count = sum(1 for _ in dataset_dir.rglob("*.csv"))

    print(
        f"[hf-storage] Dataset restore complete: {csv_count} CSV files available.",
        flush=True,
    )
