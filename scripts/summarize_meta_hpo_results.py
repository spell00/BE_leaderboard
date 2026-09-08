#!/usr/bin/env python3
"""Create compact CSV summaries from a trial bank and replay result directory."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.meta_hpo_bank import alzheimer_baseline_curve, load_bank


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-bank", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bank = load_bank(args.trial_bank)
    baseline = alzheimer_baseline_curve(bank)
    baseline_path = args.replay_dir / "summary_alzheimer_optuna_baseline.csv"
    with baseline_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("trial_index", "current_valid_mcc", "best_valid_mcc"))
        writer.writeheader(); writer.writerows(baseline)

    results_path = args.replay_dir / "scenario_results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()] if results_path.exists() else []
    summary_path = args.replay_dir / "summary_scenario_results.csv"
    fields = (
        "scenario", "source_prefix", "checkpoint_kind", "checkpoint",
        "predicted_mcc", "predicted_std", "actual_valid_mcc", "actual_test_mcc",
        "surrogate_signed_error", "cache_reused", "config_json",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row.get(key) for key in fields if key != "config_json"},
                "config_json": json.dumps(row.get("config", {}), sort_keys=True),
            })
    print(f"wrote {baseline_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
