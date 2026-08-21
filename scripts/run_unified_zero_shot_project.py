#!/usr/bin/env python3
"""Collect unified HPO data for every recommender model family."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".conda" / "bin" / "python3.12"
DATASETS = ("massbench_adenocarcinoma", "massbench_alzheimer", "massbench_benchmark")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "representation_head_project")
    parser.add_argument(
        "--trainer-types", nargs="+",
        choices=["representation", "joint", "two_stage"],
        default=["representation", "joint"],
        help="Trainer families to search; two_stage is opt-in until its holdout path is fixed.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "wandb_project": "BE_leaderboard_representation_then_head",
        "datasets": args.datasets,
        "families": args.trainer_types,
        "runs": [],
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for dataset in args.datasets:
        dataset_root = args.output_root / dataset
        command = [
            str(PYTHON), str(ROOT / "scripts" / "hp_search_unified.py"),
            "--dataset", dataset, "--n-trials", str(args.n_trials),
            "--n-epochs", str(args.n_epochs), "--early-stop", str(args.early_stop),
            "--device", args.device, "--seed", str(args.seed),
            "--output-dir", str(args.output_root),
            "--trainer-types", *args.trainer_types,
        ]
        code = subprocess.run(command, cwd=ROOT, check=False).returncode
        manifest["runs"].append({"dataset_id": dataset, "search": "single_conditional_study", "returncode": code})
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if code:
            return code
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
