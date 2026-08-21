#!/usr/bin/env python3
"""Run a frozen evolved policy once on the sealed dataset-level test manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hp_search
from src.evolutionary_meta import PolicyShape, decode_config, recommended_batch_size
from src.final_test_manifest import load_final_test_dataset_ids
from src.zero_shot_recommender.meta_features import (
    META_FEATURE_NAMES,
    extract_meta_features,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=ROOT / "config" / "final_test_datasets.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "meta_evolution_final_test")
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=-1)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required acknowledgement that the policy and all selection decisions are frozen.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.confirm_final_test:
        raise PermissionError("Refusing to open the sealed test manifest without --confirm-final-test")
    result_path = args.output_dir / "final_test_results.json"
    if result_path.exists():
        raise FileExistsError(
            f"Final test was already recorded at {result_path}; repeated test-driven iteration is forbidden"
        )
    saved = np.load(args.policy)
    required = {"genome", "meta_mean", "meta_scale", "n_inputs", "hidden_size"}
    missing = required - set(saved.files)
    if missing:
        raise ValueError(f"Frozen policy is missing fields: {sorted(missing)}")
    shape = PolicyShape(int(saved["n_inputs"][0]), int(saved["hidden_size"][0]))
    genome = saved["genome"]
    mean, scale = saved["meta_mean"], saved["meta_scale"]
    if (
        shape.n_inputs != len(META_FEATURE_NAMES)
        or mean.shape != (shape.n_inputs,)
        or scale.shape != (shape.n_inputs,)
    ):
        raise ValueError("Frozen policy meta-feature schema is incompatible")

    dataset_ids = load_final_test_dataset_ids(args.test_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for dataset_id in dataset_ids:
        X, y, batches = hp_search.load_dataset(dataset_id)
        extracted = extract_meta_features(X, y, batches)
        raw_meta = np.asarray([extracted[name] for name in META_FEATURE_NAMES], dtype=np.float32)
        normalized_meta = ((raw_meta - mean) / scale).astype(np.float32)
        config = decode_config(genome, normalized_meta, shape, max_warmup=max(1, min(50, args.n_epochs)))
        run_args = hp_search.parse_args([])
        run_args.dataset = dataset_id
        run_args.n_epochs = args.n_epochs
        run_args.n_repeats = args.n_repeats
        run_args.resolved_n_repeats = hp_search.resolve_n_repeats(args.n_repeats, batches)
        run_args.device = args.device
        run_args.seed = args.seed
        run_args.bs = recommended_batch_size(batches, cap=run_args.bs)
        run_args.no_wandb = True
        run_args.combine_test = False
        run_args.results_dir = str(args.output_dir / dataset_id)
        score, metrics = hp_search.run_trial(
            config,
            run_args,
            (X, y, batches),
            f"meta_evolution_final_{dataset_id}",
            fixed_test_data=None,
        )
        results[dataset_id] = {"score": float(score), "config": config, "metrics": metrics}

    payload = {
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "test_manifest_sha256": hashlib.sha256(args.test_manifest.read_bytes()).hexdigest(),
        "datasets": results,
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, result_path)
    print(f"Final test results written once to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
