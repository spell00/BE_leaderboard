#!/usr/bin/env python3
"""Evaluate frozen Alzheimer recommendations together, exactly once."""

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
from scripts.run_optuna_comparison import _dataset_cv_settings, _fold_scores_payload
from src.evolutionary_meta import recommended_batch_size


METHOD_DIRS = {
    "meta_evolution": "meta_evolution",
    "dataset_conditional": "dataset_conditional",
    "global_shared": "global_shared",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-arm-dir", type=Path, required=True)
    parser.add_argument("--alzheimer-control-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-budget", type=int, required=True)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.confirm_final_test:
        raise PermissionError("Refusing to open Alzheimer fixed test without --confirm-final-test")
    if args.n_repeats != 3:
        raise ValueError("Final comparison requires grouped CV=3")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "final_test_results.json"
    opened_path = args.output_dir / "FINAL_TEST_OPENED.json"
    if result_path.exists() or opened_path.exists():
        raise FileExistsError(
            f"Final Alzheimer test was already opened under {args.output_dir}; refusing repetition"
        )

    recommendation_paths = {
        method: args.three_arm_dir / dirname / "final_recommendations.json"
        for method, dirname in METHOD_DIRS.items()
    }
    recommendation_paths["alzheimer_classical_optuna"] = (
        args.alzheimer_control_dir / "final_recommendations.json"
    )
    recommendations = {}
    for method, path in recommendation_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Frozen recommendation missing for {method}: {path}")
        payload = json.loads(path.read_text())
        observed = payload.get("target_scores_observed", payload.get("fixed_test_observed", False))
        if observed:
            raise ValueError(f"{method} recommendation reports target/test feedback")
        candidates = int(payload.get("source_candidates", payload.get("source_solution_steps", -1)))
        if candidates != args.expected_budget:
            raise ValueError(
                f"{method} has {candidates} candidates; expected {args.expected_budget}"
            )
        recommendations[method] = payload

    opened = {
        "protocol": "single_irreversible_alzheimer_fixed_test_opening",
        "expected_budget": int(args.expected_budget),
        "recommendation_sha256": {
            method: sha256(path) for method, path in recommendation_paths.items()
        },
    }
    temporary_opened = opened_path.with_suffix(".json.tmp")
    temporary_opened.write_text(json.dumps(opened, indent=2) + "\n")
    os.replace(temporary_opened, opened_path)

    dataset_id = "massbench_alzheimer"
    dataset = hp_search.load_dataset(dataset_id)
    fixed_test = hp_search.load_fixed_test_dataset(dataset_id)
    X, y, batches = dataset
    results = {}
    for method, payload in recommendations.items():
        config = dict(payload["target_configs"][dataset_id])
        run_args = hp_search.parse_args([])
        run_args.dataset = dataset_id
        run_args.n_epochs = args.n_epochs
        run_args.n_repeats, run_args.resolved_n_repeats = _dataset_cv_settings(
            dataset_id, args.n_repeats, batches
        )
        run_args.num_workers = args.num_workers
        run_args.device = args.device
        run_args.seed = args.seed
        run_args.no_wandb = True
        run_args.combine_test = False
        run_args.log1p = True
        run_args.max_warmup = max(1, min(50, args.n_epochs))
        run_args.bs = recommended_batch_size(batches, cap=args.batch_size)
        run_args.cv_split_cache = str(args.output_dir / "cv_splits" / f"{dataset_id}.npz")
        run_args.results_dir = str(args.output_dir / method)
        config.update({
            "batch_size": int(run_args.bs),
            "cv_folds": int(run_args.resolved_n_repeats),
            "num_workers": int(args.num_workers),
            "lisi_enabled": False,
        })
        try:
            valid_mcc, metrics = hp_search.run_trial(
                config,
                run_args,
                dataset,
                f"alzheimer_FINAL_ONCE_{method}",
                fixed_test_data=fixed_test,
            )
            results[method] = {
                "valid_mcc": float(valid_mcc),
                "test_mcc": float(metrics.get("test_mcc", np.nan)),
                "fold_scores": _fold_scores_payload(metrics),
                "config": config,
            }
        except Exception as exc:
            results[method] = {
                "valid_mcc": -1.0,
                "test_mcc": np.nan,
                "config": config,
                "error": f"{type(exc).__name__}: {exc}",
            }

    output = {
        "protocol": "frozen_recommendations_single_alzheimer_fixed_test_opening",
        "expected_budget": int(args.expected_budget),
        "recommendation_sha256": {
            method: sha256(path) for method, path in recommendation_paths.items()
        },
        "results": results,
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, default=str) + "\n")
    os.replace(temporary, result_path)
    print(f"Final Alzheimer comparison written once to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
