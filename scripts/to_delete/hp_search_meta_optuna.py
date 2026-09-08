#!/usr/bin/env python3
"""Optuna-integrated meta-learning HPO wrapper."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hp_search_meta import main as run_meta_learning


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["massbench_adenocarcinoma", "massbench_alzheimer", "massbench_benchmark"])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "meta_hpo")
    return parser.parse_args(argv)


def objective(trial: optuna.Trial, datasets: list[str], device: str, seed: int, output_dir: Path) -> float:
    """Optuna objective: meta-learning run with 1, 2, or 3 layers."""
    
    # Suggest network architecture
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32) if n_layers > 1 else 128
    
    # Run meta-learning with these hyperparameters
    meta_valid_mcc = run_meta_learning([
        "--datasets", *datasets,
        "--n-layers", str(n_layers),
        "--hidden-dim", str(hidden_dim),
        "--learning-rate", str(learning_rate),
        "--device", device,
        "--seed", str(seed + trial.number),
        "--output-dir", str(output_dir / f"trial_{trial.number}"),
    ])
    
    return float(meta_valid_mcc)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name="meta_hpo", direction="maximize", sampler=sampler,
        storage=f"sqlite:///{(args.output_dir / 'study.sqlite3').resolve()}",
        load_if_exists=True,
    )
    
    print(
        f"[meta-optuna] dataset={args.datasets} n_trials={args.n_trials} seed={args.seed}",
        flush=True
    )
    
    study.optimize(
        lambda trial: objective(trial, args.datasets, args.device, args.seed, args.output_dir),
        n_trials=args.n_trials,
    )
    
    print(
        f"[meta-optuna] best_value={study.best_value:.4f} best_trial={study.best_trial.number}",
        flush=True
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
