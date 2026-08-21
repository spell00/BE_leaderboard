#!/usr/bin/env python3
"""Run independent BERNN searches that will become zero-shot meta-training data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROJECT_PYTHON = ROOT / ".conda" / "bin" / "python3.12"
DEFAULT_DATASETS = (
    "massbench_adenocarcinoma",
    "massbench_alzheimer",
    "massbench_benchmark",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=30)
    parser.add_argument("--n-cv", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "zero_shot_hpo")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--wait-for-pids", nargs="*", type=int, default=[],
                        help="Wait for existing experiment PIDs before starting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = PROJECT_PYTHON if PROJECT_PYTHON.exists() else Path(sys.executable)
    if python != Path(sys.executable):
        print(f"[zero-shot-hpo] using project interpreter {python}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "datasets": args.datasets,
        "n_trials": args.n_trials,
        "n_epochs": args.n_epochs,
        "n_cv": args.n_cv,
        "device": args.device,
        "seed": args.seed,
        "runs": [],
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    while True:
        active = []
        for pid in args.wait_for_pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                active.append(pid)
            else:
                active.append(pid)
        if not active:
            break
        print(f"[zero-shot-hpo] waiting for existing experiment PIDs {active}", flush=True)
        time.sleep(60)

    for dataset in args.datasets:
        output_dir = args.output_root / dataset
        command = [
            str(python),
            str(ROOT / "scripts" / "hp_search_head_sweep.py"),
            "--dataset", dataset,
            "--n-trials", str(args.n_trials),
            "--n-epochs", str(args.n_epochs),
            "--early-stop", str(args.early_stop),
            "--n-cv", str(args.n_cv),
            "--device", args.device,
            "--seed", str(args.seed),
            "--output-dir", str(output_dir),
            "--no-register-defaults",
        ]
        print(f"[zero-shot-hpo] starting {dataset}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        run = {"dataset_id": dataset, "returncode": completed.returncode, "output_dir": str(output_dir)}
        manifest["runs"].append(run)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if completed.returncode and not args.continue_on_error:
            return completed.returncode

    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return 0 if all(run["returncode"] == 0 for run in manifest["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
