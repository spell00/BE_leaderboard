#!/usr/bin/env python3
"""Analyze a completed Stage-0 trial bank and derive source-only categorical consensus."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.meta_hpo_bank import (
    alzheimer_baseline_curve,
    categorical_consensus,
    load_bank,
    source_dataset_ids,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", type=int, default=None,
                        help="Use only the first N source trials/dataset. Default: full bank.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=float, default=0.80)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    trials = load_bank(args.trial_bank)
    source_ids = source_dataset_ids(trials)
    if len(source_ids) != 4:
        print(f"[bank-analysis] warning: expected 4 source datasets, found {source_ids}")

    output_dir = args.output_dir or (args.trial_bank if args.trial_bank.is_dir() else args.trial_bank.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    consensus = categorical_consensus(
        trials,
        source_ids,
        prefix=args.prefix,
        top_k=args.top_k,
        min_support=args.min_support,
    )
    consensus_path = output_dir / "categorical_consensus.json"
    consensus_path.write_text(json.dumps(consensus, indent=2) + "\n")

    baseline = alzheimer_baseline_curve(trials)
    baseline_path = output_dir / "alzheimer_optuna_baseline_curve.csv"
    with baseline_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("trial_index", "current_valid_mcc", "best_valid_mcc"))
        writer.writeheader()
        writer.writerows(baseline)

    print(f"[bank-analysis] source datasets: {', '.join(source_ids)}")
    print(f"[bank-analysis] strict categorical freezes: {json.dumps(consensus['strict_fixed'], sort_keys=True)}")
    print(f"[bank-analysis] robust categorical freezes: {json.dumps(consensus['robust_fixed'], sort_keys=True)}")
    print(f"[bank-analysis] wrote {consensus_path}")
    if baseline:
        print(
            f"[bank-analysis] Alzheimer target-specific Optuna baseline: "
            f"best={baseline[-1]['best_valid_mcc']:.4f} after {len(baseline)} trials"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
