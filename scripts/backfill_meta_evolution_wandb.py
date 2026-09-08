#!/usr/bin/env python3
"""Restore omitted meta-evolution W&B telemetry from its durable solution ledger.

W&B history events are append-only. This appends correction events using each
original ``solution_step`` as the metric's x-axis, without changing a trained
candidate, its score, or the evolutionary checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evolve_meta_model import _score_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project", default="BE_leaderboard_meta_evolution")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads((args.output_dir / "run_metadata.json").read_text())
    records = [
        json.loads(line)
        for line in (args.output_dir / "solutions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("No durable solution records available for backfill")
    train_datasets = metadata["train_datasets"]
    histories = {dataset_id: [] for dataset_id in train_datasets}
    run = wandb.init(
        project=args.project,
        id=metadata["wandb_run_id"],
        resume="must",
        job_type="telemetry_backfill",
    )
    wandb.define_metric("solution_step")
    wandb.define_metric("solutions/*", step_metric="solution_step")
    for record in sorted(records, key=lambda row: int(row["solution_step"])):
        step = int(record["solution_step"])
        for dataset_id in train_datasets:
            histories[dataset_id].append((
                step,
                float(record["train_scores"][dataset_id]),
                float(record.get("test_scores", {}).get(dataset_id, float("nan"))),
            ))
        run.log({
            "solution_step": step,
            "solutions/total_valid_mcc": float(record["total_valid_mcc"]),
            "solutions/total_test_mcc": float(record["total_test_mcc"]),
            **{
                f"solutions/mcc/{dataset_id}": _score_figure(
                    dataset_id, histories[dataset_id]
                )
                for dataset_id in train_datasets
            },
        })
    run.summary["telemetry_backfill_source"] = "solutions.jsonl"
    run.summary["telemetry_backfill_records"] = len(records)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
