from __future__ import annotations

import os
import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

log = logging.getLogger(__name__)

# Local data root (for fallback when HF_TOKEN not set)
_LOCAL_ROOT = Path(__file__).resolve().parent.parent / "data" / "datasets"

PRIVATE_LABELS_DATASET = os.getenv(
    "HF_PRIVATE_LABELS_DATASET", "spell0/massbench-private-labels"
)
RESULTS_DATASET = os.getenv("HF_RESULTS_DATASET", "")
RESULTS_FILE = os.getenv("HF_RESULTS_FILE", "leaderboard.csv")


def _token() -> str | None:
    return os.getenv("HF_TOKEN")


def download_private_inference_file(dataset_name: str, filename: str | None = None) -> str:
    """Download a private inference CSV and return its local cached path.

    Falls back to local data/datasets/{dataset}/{dataset}_inference.csv when
    HF_TOKEN is not set (development mode).
    """
    token = _token()
    if not token:
        # Local fallback (dev mode — inference labels are never exposed in UI)
        local = _LOCAL_ROOT / dataset_name / f"{dataset_name}_inference.csv"
        if local.exists():
            log.warning(
                "HF_TOKEN not set — using local inference file: %s "
                "(dev mode; do not expose this path in the UI)", local
            )
            return str(local)
        raise EnvironmentError(
            f"HF_TOKEN is not set and no local inference file found at {local}. "
            "Set HF_TOKEN to use the private HuggingFace dataset, or provide a "
            f"local file at data/datasets/{dataset_name}/{dataset_name}_inference.csv "
            "for development."
        )

    candidates: list[str] = []
    if filename:
        candidates.append(filename)
        if filename.endswith("_train.csv") or filename.endswith("_test.csv"):
            candidates.append(filename.rsplit("_", 1)[0] + "_inference.csv")

    candidates.append(f"{dataset_name}_inference.csv")

    deduped: list[str] = []
    for c in candidates:
        if c not in deduped:
            deduped.append(c)
    candidates = deduped

    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            return hf_hub_download(
                repo_id=PRIVATE_LABELS_DATASET,
                filename=candidate,
                repo_type="dataset",
                token=token,
            )
        except Exception as exc:
            last_exc = exc

    raise FileNotFoundError(
        f"Could not download private inference file for {dataset_name}. "
        f"Tried: {', '.join(candidates)}. Last error: {last_exc}"
    )


def load_private_labels(dataset_name: str) -> pd.DataFrame:
    """Download the private ground truth labels for *dataset_name*.

    Falls back to local data/datasets/{dataset}/{dataset}_predictions.csv when
    HF_TOKEN is not set (development mode).
    Returns DataFrame with columns [name, prediction].
    """
    token = _token()
    if not token:
        local = _LOCAL_ROOT / dataset_name / f"{dataset_name}_predictions.csv"
        if local.exists():
            log.warning(
                "HF_TOKEN not set — using local predictions file: %s (dev mode)", local
            )
            df = pd.read_csv(local)
            df["name"] = df["name"].astype(str)
            df["prediction"] = df["prediction"].astype(str)
            return df
        raise EnvironmentError(
            f"HF_TOKEN is not set and no local predictions file found at {local}. "
            "Set HF_TOKEN to use the private HuggingFace dataset, or provide a "
            f"local file at data/datasets/{dataset_name}/{dataset_name}_predictions.csv."
        )

    filename = f"{dataset_name}_predictions.csv"
    path = hf_hub_download(
        repo_id=PRIVATE_LABELS_DATASET,
        filename=filename,
        repo_type="dataset",
        token=token,
    )
    df = pd.read_csv(path)
    df["name"] = df["name"].astype(str)
    df["prediction"] = df["prediction"].astype(str)
    return df


def load_private_inference(dataset_name: str) -> pd.DataFrame:
    """Download the private inference matrix for *dataset_name*.

    Tries {dataset_name}_inference.csv first, then local fallback when
    HF_TOKEN is not set.
    """
    path = download_private_inference_file(dataset_name)
    df = pd.read_csv(path)
    if "name" in df.columns:
        df["name"] = df["name"].astype(str)
    if "batch" in df.columns:
        df["batch"] = df["batch"].astype(str)
    return df


def load_leaderboard(local_fallback: str | Path) -> pd.DataFrame:
    """Return leaderboard DataFrame, pulling from Hub dataset when configured."""
    fallback = Path(local_fallback)
    _ensure_leaderboard_file(fallback)

    if not RESULTS_DATASET:
        return pd.read_csv(fallback)

    token = _token()
    try:
        path = hf_hub_download(
            repo_id=RESULTS_DATASET,
            filename=RESULTS_FILE,
            repo_type="dataset",
            token=token,
        )
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(fallback)


def save_leaderboard(df: pd.DataFrame, local_path: str | Path) -> None:
    """Persist leaderboard locally and push to HuggingFace Hub (best-effort).

    The local write always happens.  HF push is attempted when RESULTS_DATASET
    and HF_TOKEN are set; any error is logged as a warning so a transient HF
    outage never blocks a submission.
    """
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    if not RESULTS_DATASET:
        log.info("HF_RESULTS_DATASET not set — leaderboard saved locally only.")
        return
    token = _token()
    if not token:
        log.warning("HF_TOKEN not set — skipping leaderboard push to HuggingFace.")
        return
    try:
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=RESULTS_FILE,
            repo_id=RESULTS_DATASET,
            repo_type="dataset",
            commit_message="Update leaderboard results",
        )
        log.info("Leaderboard pushed to HuggingFace: %s / %s", RESULTS_DATASET, RESULTS_FILE)
    except Exception as exc:
        log.warning(
            "HuggingFace leaderboard push failed (results saved locally): %s", exc
        )


def _ensure_leaderboard_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _empty_leaderboard().to_csv(path, index=False)


def _empty_leaderboard() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["timestamp", "team", "model", "dataset", "accuracy", "macro_f1", "n_samples"]
    )
