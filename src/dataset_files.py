"""Dataset-file helpers for research CV and inference workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MATRIX_METADATA_COLUMNS = {
    "name",
    "names",
    "batch",
    "batches",
    "label",
    "labels",
    "group",
}


def dataset_directory(root: str | Path, dataset: str) -> Path:
    return Path(root) / "data" / "datasets" / str(dataset)


def resolve_dataset_matrix_file(
    root: str | Path,
    dataset: str,
    filename: str,
) -> Path:
    """Resolve one CSV inside a dataset directory without allowing traversal."""
    base = dataset_directory(root, dataset).resolve()
    name = Path(str(filename or "")).name
    if not name or name != str(filename):
        raise ValueError("Dataset file must be a filename from the selected dataset directory")
    if not name.lower().endswith(".csv"):
        raise ValueError("Dataset file must be a CSV")
    path = (base / name).resolve()
    if path.parent != base:
        raise ValueError("Dataset file must stay inside the selected dataset directory")
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def ensure_all_dataset_file(root: str | Path, dataset: str) -> Path:
    """Create <dataset>_all.csv by concatenating the public train and test CSVs.

    The merge is intentionally literal: labels are preserved when present and
    remain missing when the test file is unlabeled. Research CV later ignores
    rows without labels; the inference workflow can still predict those rows.
    """
    base = dataset_directory(root, dataset)
    train_path = base / f"{dataset}_train.csv"
    test_path = base / f"{dataset}_test.csv"
    all_path = base / f"{dataset}_all.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")

    train = pd.read_csv(train_path)
    frames = [train]
    if test_path.exists():
        test = pd.read_csv(test_path)
        # Keep a stable union of columns, with train ordering first.
        columns = list(train.columns) + [
            column for column in test.columns if column not in train.columns
        ]
        for frame in (train, test):
            for column in columns:
                if column not in frame.columns:
                    frame[column] = pd.NA
        frames = [train[columns], test[columns]]

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "name" in merged.columns:
        names = merged["name"].astype(str)
        if names.duplicated().any():
            duplicates = names[names.duplicated()].head(5).tolist()
            raise ValueError(
                f"Cannot create {all_path.name}: duplicate sample names {duplicates}"
            )

    base.mkdir(parents=True, exist_ok=True)
    merged.to_csv(all_path, index=False)
    return all_path


def ensure_all_dataset_files(
    root: str | Path,
    datasets: list[str] | tuple[str, ...] | set[str],
) -> dict[str, str]:
    """Best-effort creation of _all.csv files for a collection of datasets."""
    results: dict[str, str] = {}
    for dataset in datasets:
        try:
            results[str(dataset)] = str(ensure_all_dataset_file(root, str(dataset)))
        except Exception as exc:
            results[str(dataset)] = f"ERROR: {type(exc).__name__}: {exc}"
    return results


def research_source_filenames(root: str | Path, dataset: str) -> list[str]:
    """Return preferred single-file supervised sources (_all first, then _train)."""
    try:
        ensure_all_dataset_file(root, dataset)
    except Exception:
        pass

    base = dataset_directory(root, dataset)
    preferred = [
        f"{dataset}_all.csv",
        f"{dataset}_train.csv",
    ]
    return [name for name in preferred if (base / name).exists()]


def inference_filenames(root: str | Path, dataset: str) -> list[str]:
    """Return matrix-like CSVs that can be selected as inference targets."""
    try:
        ensure_all_dataset_file(root, dataset)
    except Exception:
        pass

    base = dataset_directory(root, dataset)
    if not base.exists():
        return []

    files = []
    for path in sorted(base.glob("*.csv")):
        name = path.name
        lower = name.lower()
        if lower.endswith("_predictions.csv") or lower == "dataset_template.csv":
            continue
        files.append(name)

    priority = {
        f"{dataset}_test.csv": 0,
        f"{dataset}_inference.csv": 1,
        f"{dataset}_all.csv": 2,
        f"{dataset}_train.csv": 3,
    }
    return sorted(files, key=lambda name: (priority.get(name, 10), name.lower()))
