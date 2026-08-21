#!/usr/bin/env python3
"""Run the inductive/transductive label-free stopping comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".conda" / "bin" / "python3.12"
DATASETS = ("massbench_adenocarcinoma", "massbench_alzheimer", "massbench_benchmark")
MODES = ("inductive", "transductive")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait-for-pids", nargs="*", type=int, default=[])
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "unsupervised_stopping")
    return parser.parse_args()


def _active_pids(pids: list[int]) -> list[int]:
    active = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            active.append(pid)
        else:
            active.append(pid)
    return active


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "datasets": args.datasets,
        "modes": args.modes,
        "n_trials": args.n_trials,
        "n_epochs": args.n_epochs,
        "selection_uses_class_labels": False,
        "wandb_project": "BE_leaderboard_unsupervised_stopping",
        "runs": [],
    }
    manifest_path = args.output_root / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    while active := _active_pids(args.wait_for_pids):
        print(f"[unsupervised-experiment] waiting for PIDs {active}", flush=True)
        time.sleep(60)

    for dataset in args.datasets:
        for mode in args.modes:
            command = [
                str(PYTHON), str(ROOT / "scripts" / "hp_search_unsupervised_stopping.py"),
                "--dataset", dataset,
                "--mode", mode,
                "--n-trials", str(args.n_trials),
                "--n-epochs", str(args.n_epochs),
                "--early-stop", str(args.early_stop),
                "--device", args.device,
                "--seed", str(args.seed),
                "--output-dir", str(args.output_root),
            ]
            print(f"[unsupervised-experiment] starting dataset={dataset} mode={mode}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            manifest["runs"].append({"dataset_id": dataset, "mode": mode, "returncode": completed.returncode})
            manifest_path.write_text(json.dumps(manifest, indent=2))
            if completed.returncode:
                return completed.returncode
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
