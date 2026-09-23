#!/usr/bin/env python3
"""Prepare the JDLBER SLE MALDI-MS cohort for BE_leaderboard.

The upstream JDLBER repository publishes three preprocessed MALDI-MS plate files
(1.csv, 2.csv, 3.csv) plus one "sumple-num-*.csv" file per plate describing how
many technical replicate spectra belong to each subject.

This script:
  1. downloads/caches the six source files from a pinned upstream commit;
  2. reconstructs subject boundaries from the replicate-count files;
  3. verifies that every subject has one consistent diagnosis and batch;
  4. median-aggregates replicate spectra to one vector per subject;
  5. writes a single BERNN research file:
       data/datasets/jdlber_sle_maldi/jdlber_sle_maldi_all.csv
     with columns:
       name,batch,label,<814 aligned m/z features...>
  6. writes provenance.json with source hashes and validation statistics.

Source labels are decoded from the published plate composition:
  0 -> HC
  1 -> SLE

The third published CSV is named 3.csv but its internal "board" value is 4.
The original board identifiers are preserved as batch labels (1, 2, 4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATASET_ID = "jdlber_sle_maldi"

UPSTREAM_REPOSITORY = "https://github.com/n778509775/JDLBER"
UPSTREAM_COMMIT = "a0173b0a9dcbff502fc8485c92d95844d7cab69c"
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/n778509775/JDLBER/"
    f"{UPSTREAM_COMMIT}/data"
)

SOURCE_BATCHES = (
    {
        "source_batch": "1",
        "matrix": "1.csv",
        "counts": "sumple-num-1.csv",
        "git_blob_sha": "db7a4973246f3e018a1adc8345e46d3e57327efd",
        "counts_git_blob_sha": "f44a0190ad84f34dc61de8f9d4db690210b8e4ec",
    },
    {
        "source_batch": "2",
        "matrix": "2.csv",
        "counts": "sumple-num-2.csv",
        "git_blob_sha": "0a3a1177f5fd9a1fc6d51306b11eaa57b7972b9e",
        "counts_git_blob_sha": "0742cdb515c7662d1cde028bb2685e1dbb003d90",
    },
    {
        "source_batch": "3",
        "matrix": "3.csv",
        "counts": "sumple-num-3.csv",
        "git_blob_sha": "9a6cc67a9ab9b8db6b0890efa7c2f487b9e2ba44",
        "counts_git_blob_sha": "5f4d40698f449cc507385b4ed2baa0d907a488e9",
    },
)

LABEL_MAP = {0: "HC", 1: "SLE"}

EXPECTED = {
    "subjects": 598,
    "spectra": 2982,
    "features": 814,
    "label_counts": {"HC": 292, "SLE": 306},
    "batch_counts": {"1": 201, "2": 212, "4": 185},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "data" / "source" / DATASET_ID,
        help="Cache directory for the six upstream JDLBER CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "datasets",
        help="BE_leaderboard prepared-dataset root.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("median", "mean"),
        default="median",
        help="How to collapse technical replicate spectra for each subject.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload upstream files even when cached copies already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prepared _all.csv and provenance.json.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Skip checks against the published 598-subject cohort totals.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    if temp.exists():
        temp.unlink()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BE_leaderboard-jdlber-preparer/1.0"},
    )
    try:
        with urllib.request.urlopen(request) as response, temp.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temp.replace(destination)
    finally:
        if temp.exists():
            temp.unlink()

    return destination


def canonical_mz_name(value: object) -> str:
    """Normalize textual m/z headers such as 103 and 103.0 to one name."""
    text = str(value).strip()
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Non-numeric m/z feature name: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"Non-finite m/z feature name: {value!r}")

    canonical = format(number.normalize(), "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    return canonical


def read_replicate_counts(path: Path) -> list[int]:
    frame = pd.read_csv(path)
    if frame.shape[1] != 1:
        raise ValueError(f"{path} should contain exactly one replicate-count column")

    values = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite replicate counts")
    if not np.all(values == np.floor(values)):
        raise ValueError(f"{path} contains non-integer replicate counts")

    counts = values.astype(int).tolist()
    if not counts or any(value <= 0 for value in counts):
        raise ValueError(f"{path} contains invalid replicate counts")
    return counts


def load_source_batch(matrix_path: Path, counts_path: Path) -> dict:
    frame = pd.read_csv(matrix_path)
    if frame.shape[1] < 3:
        raise ValueError(f"{matrix_path} must contain labels, board, and feature columns")

    first_two = [str(column).strip().lower() for column in frame.columns[:2]]
    if first_two != ["labels", "board"]:
        raise ValueError(
            f"{matrix_path} expected first columns ['labels', 'board']; found {frame.columns[:2].tolist()}"
        )

    raw_feature_names = list(frame.columns[2:])
    feature_names = [canonical_mz_name(name) for name in raw_feature_names]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError(f"{matrix_path} has duplicate m/z features after canonicalization")

    labels = pd.to_numeric(frame.iloc[:, 0], errors="raise")
    boards = pd.to_numeric(frame.iloc[:, 1], errors="raise")
    features = (
        frame.iloc[:, 2:]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float64, copy=True)
    )

    if not np.all(np.isfinite(labels.to_numpy(dtype=float))):
        raise ValueError(f"{matrix_path} contains non-finite labels")
    if not np.all(np.isfinite(boards.to_numpy(dtype=float))):
        raise ValueError(f"{matrix_path} contains non-finite board identifiers")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{matrix_path} contains non-finite feature values")

    label_values = labels.to_numpy(dtype=int)
    if not np.all(labels.to_numpy(dtype=float) == label_values):
        raise ValueError(f"{matrix_path} contains non-integer labels")
    unknown_labels = sorted(set(label_values.tolist()) - set(LABEL_MAP))
    if unknown_labels:
        raise ValueError(f"{matrix_path} contains unknown labels: {unknown_labels}")

    board_values = boards.to_numpy(dtype=int)
    if not np.all(boards.to_numpy(dtype=float) == board_values):
        raise ValueError(f"{matrix_path} contains non-integer board identifiers")

    counts = read_replicate_counts(counts_path)
    if sum(counts) != len(frame):
        raise ValueError(
            f"{matrix_path}: replicate counts sum to {sum(counts)}, but matrix has {len(frame)} rows"
        )

    return {
        "frame": frame,
        "labels": label_values,
        "boards": board_values,
        "features": features,
        "feature_names": feature_names,
        "replicate_counts": counts,
    }


def aggregate_subjects(source: dict, aggregation: str) -> tuple[pd.DataFrame, dict]:
    labels = source["labels"]
    boards = source["boards"]
    features = source["features"]
    counts = source["replicate_counts"]
    feature_names = source["feature_names"]

    rows = []
    offset = 0
    replicate_distribution: dict[str, int] = {}

    for subject_index, n_replicates in enumerate(counts, start=1):
        stop = offset + n_replicates
        subject_labels = labels[offset:stop]
        subject_boards = boards[offset:stop]
        subject_features = features[offset:stop]

        unique_labels = np.unique(subject_labels)
        unique_boards = np.unique(subject_boards)
        if len(unique_labels) != 1:
            raise ValueError(
                f"Subject {subject_index} has inconsistent replicate labels: {unique_labels.tolist()}"
            )
        if len(unique_boards) != 1:
            raise ValueError(
                f"Subject {subject_index} spans multiple boards: {unique_boards.tolist()}"
            )

        raw_label = int(unique_labels[0])
        board = str(int(unique_boards[0]))
        if aggregation == "median":
            values = np.median(subject_features, axis=0)
        elif aggregation == "mean":
            values = np.mean(subject_features, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        row = {
            "name": f"JDLBER_board{board}_subject{subject_index:03d}",
            "batch": board,
            "label": LABEL_MAP[raw_label],
        }
        row.update(dict(zip(feature_names, values.tolist())))
        rows.append(row)

        key = str(n_replicates)
        replicate_distribution[key] = replicate_distribution.get(key, 0) + 1
        offset = stop

    if offset != len(labels):
        raise AssertionError("Subject reconstruction did not consume every spectrum")

    return pd.DataFrame(rows), {
        "subjects": len(rows),
        "spectra": int(len(labels)),
        "replicate_count_distribution": replicate_distribution,
    }


def validate_feature_alignment(loaded_batches: list[dict]) -> list[str]:
    reference = loaded_batches[0]["feature_names"]
    for index, source in enumerate(loaded_batches[1:], start=2):
        current = source["feature_names"]
        if current != reference:
            mismatch = next(
                (
                    (i, left, right)
                    for i, (left, right) in enumerate(zip(reference, current))
                    if left != right
                ),
                None,
            )
            if len(current) != len(reference):
                detail = f"feature counts differ: {len(reference)} vs {len(current)}"
            else:
                detail = f"first mismatch: {mismatch}"
            raise ValueError(f"Batch {index} m/z grid does not align with batch 1 ({detail})")
    return reference


def validate_published_totals(frame: pd.DataFrame, spectra: int, n_features: int) -> None:
    observed = {
        "subjects": int(len(frame)),
        "spectra": int(spectra),
        "features": int(n_features),
        "label_counts": {
            str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()
        },
        "batch_counts": {
            str(k): int(v) for k, v in frame["batch"].astype(str).value_counts().sort_index().items()
        },
    }
    if observed != EXPECTED:
        raise ValueError(
            "Prepared cohort does not match published JDLBER totals.\n"
            f"Expected: {json.dumps(EXPECTED, sort_keys=True)}\n"
            f"Observed: {json.dumps(observed, sort_keys=True)}"
        )


def prepare_dataset(args) -> tuple[Path, Path]:
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    target_dir = output_root / DATASET_ID
    all_path = target_dir / f"{DATASET_ID}_all.csv"
    provenance_path = target_dir / "provenance.json"

    if (all_path.exists() or provenance_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"{target_dir} already contains prepared output; rerun with --overwrite to replace it"
        )

    loaded_batches = []
    source_records = []
    for spec in SOURCE_BATCHES:
        matrix_url = f"{RAW_BASE_URL}/{spec['matrix']}"
        counts_url = f"{RAW_BASE_URL}/{spec['counts']}"
        matrix_path = download(
            matrix_url,
            source_root / spec["matrix"],
            force=args.force_download,
        )
        counts_path = download(
            counts_url,
            source_root / spec["counts"],
            force=args.force_download,
        )
        source = load_source_batch(matrix_path, counts_path)
        loaded_batches.append(source)
        source_records.append(
            {
                "source_batch_filename": spec["source_batch"],
                "matrix_file": spec["matrix"],
                "matrix_url": matrix_url,
                "matrix_sha256": sha256_file(matrix_path),
                "matrix_git_blob_sha": spec["git_blob_sha"],
                "replicate_counts_file": spec["counts"],
                "replicate_counts_url": counts_url,
                "replicate_counts_sha256": sha256_file(counts_path),
                "replicate_counts_git_blob_sha": spec["counts_git_blob_sha"],
                "raw_spectra": int(len(source["labels"])),
                "subjects": int(len(source["replicate_counts"])),
                "board_values": sorted(set(map(int, source["boards"].tolist()))),
            }
        )

    feature_names = validate_feature_alignment(loaded_batches)

    subject_frames = []
    aggregation_records = []
    for spec, source in zip(SOURCE_BATCHES, loaded_batches):
        subjects, stats = aggregate_subjects(source, args.aggregation)
        subject_frames.append(subjects)
        aggregation_records.append(
            {
                "source_file": spec["matrix"],
                "board_values": sorted(subjects["batch"].astype(str).unique().tolist()),
                **stats,
                "label_counts": {
                    str(k): int(v)
                    for k, v in subjects["label"].value_counts().sort_index().items()
                },
            }
        )

    output = pd.concat(subject_frames, ignore_index=True)
    if output["name"].duplicated().any():
        duplicates = output.loc[output["name"].duplicated(), "name"].head(10).tolist()
        raise ValueError(f"Duplicate generated subject names: {duplicates}")

    spectra = sum(int(record["spectra"]) for record in aggregation_records)
    if not args.no_strict:
        validate_published_totals(output, spectra=spectra, n_features=len(feature_names))

    target_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(all_path, index=False)

    provenance = {
        "dataset_id": DATASET_ID,
        "source_repository": UPSTREAM_REPOSITORY,
        "source_commit": UPSTREAM_COMMIT,
        "source_files": source_records,
        "output_file": all_path.name,
        "output_sha256": sha256_file(all_path),
        "aggregation": args.aggregation,
        "aggregation_scope": "technical replicate spectra within subject",
        "subject_boundary_source": "sumple-num-*.csv in the upstream JDLBER repository",
        "label_mapping": {"0": "HC", "1": "SLE"},
        "batch_policy": (
            "Preserve the upstream board column. The three prepared source files "
            "therefore use batch identifiers 1, 2, and 4."
        ),
        "samples": int(len(output)),
        "raw_spectra": int(spectra),
        "features": int(len(feature_names)),
        "batches": int(output["batch"].astype(str).nunique()),
        "labels": int(output["label"].nunique()),
        "label_counts": {
            str(k): int(v) for k, v in output["label"].value_counts().sort_index().items()
        },
        "batch_counts": {
            str(k): int(v)
            for k, v in output["batch"].astype(str).value_counts().sort_index().items()
        },
        "feature_columns_sha256": hashlib.sha256(
            "\n".join(feature_names).encode("utf-8")
        ).hexdigest(),
        "source_batch_summaries": aggregation_records,
        "preprocessing": (
            "Use upstream preprocessed/aligned MALDI-MS intensities; canonicalize "
            "equivalent textual m/z headers; median-aggregate technical replicates "
            "to one row per subject; no additional normalization or batch correction."
        ),
        "strict_published_totals_checked": not args.no_strict,
        "role": "research_whole_dataset",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(
        f"wrote {all_path} shape={output.shape} "
        f"subjects={len(output)} features={len(feature_names)} "
        f"batches={provenance['batch_counts']} labels={provenance['label_counts']}"
    )
    print(f"wrote {provenance_path}")
    return all_path, provenance_path


def main(argv=None) -> int:
    args = parse_args(argv)
    prepare_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
