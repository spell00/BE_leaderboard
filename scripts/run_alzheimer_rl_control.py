#!/usr/bin/env python3
"""Target-specific RL control: optimize Alzheimer using real BERNN MCC reward.

This arm is intentionally NOT a held-out meta-validation method.  Alzheimer MCC
updates the RL policy exactly as Alzheimer MCC updates its target-specific Optuna
baseline.  Use it to compare target optimization efficiency (best MCC vs number
of Alzheimer BERNN trials), not to claim zero-shot generalization.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts import hp_search
from src.dataset_splits import load_dataset_partitions
from src.meta_hpo_bank import (
    alzheimer_baseline_curve,
    categorical_consensus,
    load_bank,
    load_fixed_categories,
    protocol_digest,
    source_dataset_ids,
)
from src.meta_hpo_models import ReinforcePolicy
from src.zero_shot_recommender.meta_features import META_FEATURE_NAMES, extract_meta_features


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-bank", type=Path, required=True)
    parser.add_argument(
        "--split-manifest", type=Path,
        default=ROOT / "config" / "evolution_development_datasets.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "alzheimer_rl_control")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-hidden-size", type=int, default=96)
    parser.add_argument("--policy-lr", type=float, default=3e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--baseline-decay", type=float, default=0.90)
    parser.add_argument("--categorical-consensus", type=Path, default=None)
    parser.add_argument("--freeze-policy", choices=("none", "strict", "robust"), default="none")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _meta_vector(dataset):
    X, y, batches = dataset
    values = extract_meta_features(X, y, batches)
    return np.asarray([values[name] for name in META_FEATURE_NAMES], dtype=np.float32)


def _recommended_batch_size(batch_labels, cap: int):
    labels = np.asarray(batch_labels).astype(str)
    _, counts = np.unique(labels, return_counts=True)
    smallest_training_fold = int(labels.size - counts.max()) if len(counts) > 1 else int(labels.size)
    return max(1, min(int(cap), max(1, smallest_training_fold // 2)))


def _append(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str) + "\n")


def _load(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv=None) -> int:
    import torch

    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"{args.output_dir} is not empty; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bank = load_bank(args.trial_bank)
    source_ids = source_dataset_ids(bank)
    partitions = load_dataset_partitions(args.split_manifest)
    alzheimer_id = tuple(partitions.validation)[0]
    source_data = {name: hp_search.load_dataset(name) for name in source_ids}
    source_meta = np.stack([_meta_vector(source_data[name]) for name in source_ids])
    alzheimer_data = hp_search.load_dataset(alzheimer_id)
    alzheimer_meta = _meta_vector(alzheimer_data)
    alzheimer_fixed = hp_search.load_fixed_test_dataset(alzheimer_id)
    meta_mean = source_meta.mean(axis=0)
    meta_scale = source_meta.std(axis=0)
    meta_scale[meta_scale < 1e-8] = 1.0
    target_x = torch.tensor(((alzheimer_meta - meta_mean) / meta_scale)[None, :], dtype=torch.float32)

    if args.freeze_policy == "none":
        fixed_categories = {}
    elif args.categorical_consensus is not None:
        fixed_categories = load_fixed_categories(args.categorical_consensus, policy=args.freeze_policy)
    else:
        consensus = categorical_consensus(bank, source_ids)
        fixed_categories = dict(consensus[f"{args.freeze_policy}_fixed"])

    max_warmup = max(1, min(50, int(args.n_epochs)))
    torch.manual_seed(int(args.seed))
    policy = ReinforcePolicy(len(META_FEATURE_NAMES), args.policy_hidden_size, fixed_categories)
    policy._max_warmup = max_warmup
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.policy_lr))

    records_path = args.output_dir / "rl_trials.jsonl"
    existing = _load(records_path)
    # Exact training resume would require serializing optimizer/RNG after every
    # trial.  To keep the contract honest, resume is allowed only when no prior
    # policy trials exist; the expensive BERNN cache itself is still reusable.
    if existing:
        raise RuntimeError(
            "Policy-state resume is intentionally not approximated. Start a fresh RL output dir; "
            "the Stage-0 bank remains reusable."
        )

    cache_path = args.output_dir / "bernn_cache.jsonl"
    cache = {row["cache_key"]: row for row in _load(cache_path)}
    baseline = None
    ema = None
    best_mcc = -np.inf

    for trial_index in range(int(args.n_trials)):
        configs, log_prob, entropy = policy.sample(target_x)
        config = configs[0]
        requested_folds = int(args.n_repeats)
        _, _, batches = alzheimer_data
        resolved = int(hp_search.resolve_n_repeats(requested_folds, batches))
        protocol = {
            "dataset": alzheimer_id,
            "n_epochs": int(args.n_epochs),
            "n_repeats": requested_folds,
            "resolved_n_repeats": resolved,
            "seed": int(args.seed) + 10000,
            "batch_size_cap": int(args.batch_size),
        }
        key = protocol_digest(config, protocol)
        if key in cache:
            evaluation = cache[key]
            cache_reused = True
        else:
            run_args = hp_search.parse_args([])
            run_args.dataset = alzheimer_id
            run_args.n_epochs = int(args.n_epochs)
            run_args.n_repeats = requested_folds
            run_args.resolved_n_repeats = resolved
            run_args.num_workers = int(args.num_workers)
            run_args.device = args.device
            run_args.seed = int(args.seed) + 10000
            run_args.no_wandb = True
            run_args.combine_test = False
            run_args.max_warmup = max_warmup
            run_args.log1p = True
            run_args.bs = _recommended_batch_size(batches, args.batch_size)
            run_args.results_dir = str(args.output_dir / "bernn")
            run_args.cv_split_cache = str(args.output_dir / "cv_splits" / f"{alzheimer_id}.npz")
            started = time.monotonic()
            metrics = {}
            error = None
            try:
                score, metrics = hp_search.run_trial(
                    config, run_args, alzheimer_data,
                    f"alzheimer_real_rl_t{trial_index}_{key[:10]}",
                    fixed_test_data=alzheimer_fixed,
                )
                score = float(score)
            except Exception as exc:
                score = -1.0
                error = f"{type(exc).__name__}: {exc}"
            evaluation = {
                "cache_key": key,
                "valid_mcc": score,
                "test_mcc": float(metrics.get("test_mcc", np.nan)),
                "fit_seconds": time.monotonic() - started,
                "config": config,
                "protocol": protocol,
                "error": error,
            }
            _append(cache_path, evaluation)
            cache[key] = evaluation
            cache_reused = False

        reward_value = float(evaluation["valid_mcc"])
        baseline_before = 0.0 if ema is None else float(ema)
        advantage = reward_value - baseline_before
        loss = -(torch.tensor(advantage, dtype=torch.float32) * log_prob.mean()) - float(args.entropy_coef) * entropy.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema = reward_value if ema is None else float(args.baseline_decay) * ema + (1.0 - float(args.baseline_decay)) * reward_value
        best_mcc = max(best_mcc, reward_value)
        row = {
            "trial_index": trial_index,
            "reward_valid_mcc": reward_value,
            "best_valid_mcc": float(best_mcc),
            "baseline_before": baseline_before,
            "advantage": advantage,
            "test_mcc": float(evaluation.get("test_mcc", np.nan)),
            "cache_reused": cache_reused,
            "config": config,
            "fixed_categories": fixed_categories,
        }
        _append(records_path, row)
        print(
            f"[alzheimer-real-rl] trial={trial_index + 1}/{args.n_trials} "
            f"MCC={reward_value:.4f} best={best_mcc:.4f}",
            flush=True,
        )

    metadata = {
        "role": "target_specific_control_not_meta_validation",
        "alzheimer_mcc_used_as_reward": True,
        "source_trial_bank_used_for_policy_training": False,
        "source_trial_bank_used_for_categorical_consensus": bool(fixed_categories),
        "fixed_categories": fixed_categories,
        "alzheimer_optuna_baseline_curve": alzheimer_baseline_curve(bank),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
