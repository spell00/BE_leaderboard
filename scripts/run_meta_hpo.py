#!/usr/bin/env python3
"""Optuna-integrated meta-learning HPO wrapper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["massbench_adenocarcinoma", "massbench_alzheimer", "massbench_benchmark"])
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "meta_hpo")
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop-ae", type=int, default=30)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--meta-mode", type=str, default="A", choices=["A", "B"],
                        help="A: fast differentiable surrogate (default), B: expensive black-box trials")
    parser.add_argument("--surrogate-folds", type=int, default=2,
                        help="[Mode A only] Number of CV folds for fast surrogate evaluation")
    parser.add_argument("--surrogate-epochs", type=int, default=20,
                        help="[Mode A only] Max epochs for fast surrogate training")
    return parser.parse_args(argv)


def objective(trial: optuna.Trial, args) -> float:
    """Optuna objective: run meta-learning with suggested hyperparameters."""
    
    # Suggest meta-network architecture
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    learning_rate = trial.suggest_float("meta_lr", 1e-4, 1e-1, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32) if n_layers > 1 else 128
    
    # Create trial-specific output directory
    trial_output = args.output_root / f"trial_{trial.number}"
    trial_output.mkdir(parents=True, exist_ok=True)
    
    print(
        f"\n[optuna] trial={trial.number} n_layers={n_layers} "
        f"meta_lr={learning_rate:.2e} hidden_dim={hidden_dim}",
        flush=True
    )
    
    # Build command
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "hp_search_meta.py"),
        "--datasets", *args.datasets,
        "--n-layers", str(n_layers),
        "--hidden-dim", str(hidden_dim),
        "--learning-rate", str(learning_rate),
        "--device", args.device,
        "--seed", str(args.seed + trial.number),
        "--output-dir", str(trial_output),
        "--n-epochs", str(args.n_epochs),
        "--early-stop-ae", str(args.early_stop_ae),
        "--meta-mode", args.meta_mode,
        "--surrogate-folds", str(args.surrogate_folds),
        "--surrogate-epochs", str(args.surrogate_epochs),
    ]
    
    if args.no_wandb:
        cmd.append("--no-wandb")
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Run meta-learning trial
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"[optuna] trial {trial.number} failed", flush=True)
        return float("-inf")
    
    # For Mode A (surrogate), the output doesn't write a ledger; just return a placeholder
    # For Mode B (black-box), read from ledger if available
    if args.meta_mode == "A":
        print(f"[optuna] trial {trial.number} completed (mode A)", flush=True)
        return 0.5  # Placeholder; real performance logged to W&B
    
    ledger_path = trial_output / "meta_trials.jsonl"
    if not ledger_path.exists():
        print(f"[optuna] trial {trial.number} ledger not found", flush=True)
        return float("-inf")
    
    import json
    meta_valid_mccs = []
    with open(ledger_path) as f:
        for line in f:
            record = json.loads(line)
            meta_valid_mccs.append(record.get("valid_mcc", 0.0))
    
    if not meta_valid_mccs:
        return float("-inf")
    
    # Average over all epochs (last 20 or so should be best due to early stopping)
    final_meta_valid_mcc = float(__import__("numpy").mean(meta_valid_mccs[-20:]))
    
    print(
        f"[optuna] trial={trial.number} final_meta_valid_mcc={final_meta_valid_mcc:.4f}",
        flush=True
    )
    
    return final_meta_valid_mcc


def main():
    args = parse_args()
    
    print(f"[optuna] meta_mode={args.meta_mode} (A=surrogate, B=black-box)", flush=True)
    print(f"[optuna] datasets={args.datasets} n_trials={args.n_trials}", flush=True)
    
    # Create study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    print(f"\n[optuna] Best trial: {study.best_trial.number}", flush=True)
    print(f"[optuna] Best value: {study.best_trial.value}", flush=True)
    print(f"[optuna] Best params: {study.best_trial.params}", flush=True)


if __name__ == "__main__":
    main()
