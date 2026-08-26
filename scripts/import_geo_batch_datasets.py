#!/usr/bin/env python3
"""Convert GEO batch combined CSVs into the leaderboard dataset layout.

The input is the combined GEO export produced by scripts/prepare_geo_batch_datasets.R:
  label,batch,sample_id,<numeric features...>

The output matches the existing prepared datasets under data/datasets/<dataset>/:
  <dataset>_train.csv
  <dataset>_test.csv
  <dataset>_inference.csv
  <dataset>_predictions.csv
  provenance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
META_COLUMNS = {
    "name", "names",
    "label", "labels", "group",
    "batch", "batches",
    "sample_id", "sample", "samples",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "data" / "geo_batch",
                        help="Directory containing the combined GEO CSVs.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "datasets",
                        help="Prepared-dataset root, typically data/datasets.")
    parser.add_argument("datasets", nargs="*", default=None,
                        help="Optional dataset basenames to export. Defaults to every combined CSV in --source-root.")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction of samples to hold out as test, split by batch group.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-trials", type=int, default=256,
                        help="Number of randomized group splits to evaluate for balance.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_dataset_name(path: Path) -> str:
    stem = path.stem
    for suffix in ("_inference", "_train", "_test", "_predictions", "_metadata", "_expression"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def discover_source_files(source_root: Path, datasets: list[str] | None) -> list[Path]:
    if datasets:
        paths = [source_root / f"{dataset}.csv" for dataset in datasets]
    else:
        paths = []
        for path in sorted(source_root.glob("*.csv")):
            if path.name.endswith(("_expression.csv", "_metadata.csv", "_predictions.csv", "_test.csv")):
                continue
            paths.append(path)
    if not paths:
        raise ValueError(f"No source CSVs found in {source_root}")
    return paths


def load_combined_csv(path: Path) -> dict:
    frame = pd.read_csv(path)
    lower_map = {column.lower(): column for column in frame.columns}
    label_col = next((lower_map[name] for name in ("label", "labels", "group") if name in lower_map), None)
    batch_col = next((lower_map[name] for name in ("batch", "batches") if name in lower_map), None)
    sample_col = next((lower_map[name] for name in ("sample_id", "sample", "samples", "name", "names") if name in lower_map), None)
    if label_col is None:
        raise ValueError(f"{path} lacks a label column")
    if batch_col is None:
        raise ValueError(f"{path} lacks a batch column")
    if sample_col is None:
        raise ValueError(f"{path} lacks a sample/name column")

    feature_cols = [column for column in frame.columns if column not in META_COLUMNS]
    if not feature_cols:
        raise ValueError(f"{path} has no feature columns")
    features = frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {
        "dataset": infer_dataset_name(path),
        "source_csv": path,
        "frame": frame,
        "label_col": label_col,
        "batch_col": batch_col,
        "sample_col": sample_col,
        "feature_cols": feature_cols,
        "features": features,
    }


def _distribution(series: pd.Series, labels: Iterable[str]) -> pd.Series:
    return series.astype(str).value_counts(normalize=True).reindex(list(labels), fill_value=0.0)


def choose_split(frame: pd.DataFrame, label_col: str, batch_col: str, test_fraction: float, seed: int, trials: int):
    labels = sorted(frame[label_col].astype(str).unique())
    overall = _distribution(frame[label_col], labels)
    target_test = max(1, int(round(len(frame) * test_fraction)))
    batches = frame[batch_col].astype(str).to_numpy()
    y = frame[label_col].astype(str).to_numpy()
    X = np.zeros((len(frame), 1), dtype=np.float32)

    best = None
    best_score = float("inf")
    for trial in range(max(1, trials)):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed + trial)
        train_idx, test_idx = next(splitter.split(X, y, batches))
        train_labels = pd.Series(y[train_idx])
        test_labels = pd.Series(y[test_idx])
        if train_labels.nunique() < 2 or test_labels.nunique() < 2:
            score = 10.0
        else:
            train_dist = _distribution(train_labels, labels)
            test_dist = _distribution(test_labels, labels)
            size_penalty = abs(len(test_idx) - target_test) / max(1, target_test)
            label_penalty = float(np.abs(train_dist - overall).sum() + np.abs(test_dist - overall).sum())
            batch_penalty = 0.0
            if train_labels.nunique() < len(labels):
                batch_penalty += 0.5
            if test_labels.nunique() < len(labels):
                batch_penalty += 0.5
            score = 2.0 * size_penalty + label_penalty + batch_penalty
        if score < best_score:
            best_score = score
            best = (train_idx, test_idx)
    if best is None:
        raise RuntimeError("Could not derive a valid train/test split")
    return best[0], best[1], best_score


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_dataset(spec: dict, output_root: Path, test_fraction: float, seed: int, trials: int, overwrite: bool) -> dict:
    dataset_id = spec["dataset"]
    source_csv = spec["source_csv"]
    frame = spec["frame"]
    label_col = spec["label_col"]
    batch_col = spec["batch_col"]
    sample_col = spec["sample_col"]
    feature_cols = spec["feature_cols"]
    features = spec["features"].reset_index(drop=True)

    train_idx, test_idx, split_score = choose_split(frame, label_col, batch_col, test_fraction, seed, trials)
    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    test_frame = frame.iloc[test_idx].reset_index(drop=True)
    train_features = features.iloc[train_idx].reset_index(drop=True)
    test_features = features.iloc[test_idx].reset_index(drop=True)

    names_train = train_frame[sample_col].astype(str).to_numpy()
    names_test = test_frame[sample_col].astype(str).to_numpy()
    batches_train = train_frame[batch_col].astype(str).to_numpy()
    batches_test = test_frame[batch_col].astype(str).to_numpy()
    labels_train = train_frame[label_col].astype(str).to_numpy()
    labels_test = test_frame[label_col].astype(str).to_numpy()

    train_out = pd.concat([
        pd.DataFrame({"name": names_train, "batch": batches_train, "label": labels_train}),
        train_features,
    ], axis=1)
    test_out = pd.concat([
        pd.DataFrame({"name": names_test, "batch": batches_test}),
        test_features,
    ], axis=1)
    inference_out = pd.concat([
        pd.DataFrame({"name": names_test, "batch": batches_test, "label": labels_test}),
        test_features,
    ], axis=1)
    predictions_out = pd.DataFrame({"name": names_test, "prediction": labels_test})

    target_dir = output_root / dataset_id
    if target_dir.exists() and not overwrite:
        raise FileExistsError(f"{target_dir} already exists; rerun with --overwrite to replace it")
    target_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(train_out, target_dir / f"{dataset_id}_train.csv")
    _write_csv(test_out, target_dir / f"{dataset_id}_test.csv")
    _write_csv(inference_out, target_dir / f"{dataset_id}_inference.csv")
    _write_csv(predictions_out, target_dir / f"{dataset_id}_predictions.csv")

    provenance = {
        "dataset_id": dataset_id,
        "source_csv": str(source_csv),
        "source_sha256": sha256_file(source_csv),
        "source_rows": int(len(frame)),
        "source_features": len(feature_cols),
        "output_train_rows": int(len(train_out)),
        "output_test_rows": int(len(test_out)),
        "output_features": int(train_features.shape[1]),
        "n_batches": int(frame[batch_col].astype(str).nunique()),
        "n_labels": int(frame[label_col].astype(str).nunique()),
        "label_counts": frame[label_col].astype(str).value_counts().to_dict(),
        "batch_counts": frame[batch_col].astype(str).value_counts().to_dict(),
        "test_fraction": test_fraction,
        "split_seed": seed,
        "split_trials": trials,
        "split_score": split_score,
        "preprocessing": "batch-preserving group split; no additional transformation",
        "role": "prepared_geo_batch",
        "feature_columns_sha256": sha256_bytes("\n".join(feature_cols).encode()),
    }
    (target_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {dataset_id}: train={len(train_out)} test={len(test_out)} features={train_features.shape[1]} -> {target_dir}")
    return provenance


def main(argv=None) -> int:
    args = parse_args(argv)
    source_paths = discover_source_files(args.source_root, args.datasets)
    provenances = []
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        provenances.append(export_dataset(load_combined_csv(path), args.output_root, args.test_fraction, args.seed, args.split_trials, args.overwrite))
    summary_path = args.output_root / "geo_batch_dataset_manifest.json"
    summary_path.write_text(json.dumps(provenances, indent=2) + "\n")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
