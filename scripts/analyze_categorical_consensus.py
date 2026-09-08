#!/usr/bin/env python3
"""Find categorical BERNN knobs that Stage-0 independent Optuna controls agree on.

Run this after (or during) run_optuna_comparison.py.  It reads solutions.jsonl,
never touches Alzheimer scores, and writes an artifact consumable by the pruned
meta-HPO/evolution/replay runners.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_splits import load_dataset_partitions
from src.meta_hpo_utils import categorical_consensus, load_solution_rows, trial_points


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run-dir", type=Path, required=True,
        help="Output directory of run_optuna_comparison.py (contains solutions.jsonl).",
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=ROOT / "config" / "evolution_development_datasets.json",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-support", type=float, default=0.90)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    partitions = load_dataset_partitions(args.split_manifest)
    rows = load_solution_rows(args.source_run_dir)
    points = trial_points(rows)
    report = categorical_consensus(
        points,
        partitions.train,
        top_k=args.top_k,
        min_support=args.min_support,
    )
    report["source_run_dir"] = str(args.source_run_dir)
    report["source_solution_steps"] = len(rows)
    output = args.output or (args.source_run_dir / "categorical_consensus.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {output}")
    print("Strict champion consensus:")
    print(json.dumps(report["strict_fixed"], indent=2))
    print("Robust top-K consensus (recommended for freezing):")
    print(json.dumps(report["robust_fixed"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
