#!/usr/bin/env python3
"""Create <dataset>_all.csv files by concatenating public train + test CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset_files import ensure_all_dataset_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    "normal_tissue_878",
    "colon_3041",
    "massbench_adenocarcinoma",
    "massbench_alzheimer",
    "massbench_benchmark",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset key to build. Repeat for multiple datasets; default builds all five.",
    )
    args = parser.parse_args()

    datasets = tuple(args.datasets or DEFAULT_DATASETS)
    failed = False
    for dataset in datasets:
        try:
            path = ensure_all_dataset_file(ROOT, dataset)
            print(f"{dataset}: {path}")
        except Exception as exc:
            failed = True
            print(f"{dataset}: ERROR {type(exc).__name__}: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
