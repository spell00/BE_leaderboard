#!/usr/bin/env python3
"""Rebuild historical selected meta nets without rerunning expensive BERNN fits.

Only publish a reconstructed checkpoint when its decoded Alzheimer configuration
and benchmark prediction error match the historical record within tolerance.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import hp_search
from src.direct_meta_checkpoint import (
    load_direct_meta_checkpoint, predict_direct_meta_config, save_selected_meta_model,
)
from src.meta_hpo_bank import BankTrial, config_feature_vector
from src.meta_hpo_models import normalize_source_meta, train_direct_meta_model
from src.zero_shot_recommender.meta_features import META_FEATURE_NAMES, extract_meta_features

TRAIN_IDS = ("normal_tissue_878", "colon_3041", "massbench_adenocarcinoma")
VALID_ID = "massbench_benchmark"
TARGET_ID = "massbench_alzheimer"


def config_matches(actual, expected):
    if actual.keys() != expected.keys():
        return False
    return all(
        np.isclose(actual[k], v, rtol=1e-5, atol=1e-8)
        if isinstance(v, float) else actual[k] == v
        for k, v in expected.items()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--round", type=int, help="Zero-based round from rounds.jsonl")
    selection.add_argument("--all", action="store_true", help="Recover every recorded round")
    parser.add_argument("--seed", type=int, default=42, help="Original runner base seed")
    parser.add_argument("--n-epochs", type=int, default=1000, help="Original BERNN epoch limit")
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.output_dir / "rounds.jsonl").read_text().splitlines() if line.strip()]
    if args.round is not None:
        records = [r for r in records if r["round"] == args.round]
    elif not args.all:
        records = [max(records, key=lambda r: r["alzheimer_valid_mcc"])]
    if not records:
        parser.error("No matching completed round")

    meta = {}
    for dataset in TRAIN_IDS + (VALID_ID, TARGET_ID):
        print(f"[recovery] Extracting meta-features: {dataset}", flush=True)
        features = extract_meta_features(*hp_search.load_dataset(dataset))
        meta[dataset] = np.asarray([features[n] for n in META_FEATURE_NAMES], dtype=np.float32)
    source_meta, mean, scale = normalize_source_meta(meta, TRAIN_IDS)
    max_warmup = max(1, min(50, args.n_epochs))
    failures = []
    for record in records:
        step = int(record["round"])
        # Preserve the original runner's training and inference normalization.
        trials = {
            d: BankTrial(dataset_id=d, role="source", trial_index=step,
                         optuna_trial_number=record["best_source"][d]["trial_number"],
                         valid_mcc=record["best_source"][d]["valid_mcc"],
                         test_mcc=record["best_source"][d]["test_mcc"],
                         fit_seconds=record["best_source"][d]["fit_seconds"],
                         config=record["best_source"][d]["config"])
            for d in TRAIN_IDS
        }
        model, _, diagnostics = train_direct_meta_model(
            trials, source_meta, meta[TARGET_ID], max_warmup=max_warmup,
            hidden_size=record["meta_hidden_size"], epochs=200,
            lr=record["meta_lr"], seed=args.seed + step,
        )
        inference = dict(n_meta=len(META_FEATURE_NAMES), meta_feature_names=list(META_FEATURE_NAMES),
                         meta_mean=mean.tolist(), meta_scale=scale.tolist(),
                         max_warmup=max_warmup, fixed_categories={})
        predicted = predict_direct_meta_config(model, inference, meta[TARGET_ID])
        benchmark = predict_direct_meta_config(model, inference, meta[VALID_ID])
        error = float(np.linalg.norm(
            config_feature_vector(benchmark, max_warmup=max_warmup)
            - config_feature_vector(record["benchmark_reference_config"], max_warmup=max_warmup)))
        if (not config_matches(predicted, record["alzheimer_config"])
                or not np.isclose(error, record["benchmark_prediction_error"], rtol=1e-5, atol=1e-8)):
            failures.append(step)
            print(f"[recovery] round={step}: MISMATCH; no checkpoint saved. "
                  f"benchmark error={error}, expected={record['benchmark_prediction_error']}; "
                  f"predicted Alzheimer config={json.dumps(predicted)}", flush=True)
            continue
        metadata = {**record, "reconstructed": True, "training_seed": args.seed + step,
                    "meta_epochs": 200, "training_diagnostics": diagnostics,
                    "raw_meta_features": {d: v.tolist() for d, v in meta.items()},
                    "recovery_verification": {"alzheimer_config_matches": True,
                                              "benchmark_error_matches": True,
                                              "rtol": 1e-5, "atol": 1e-8}}
        # Keep recovered models separate from checkpoints captured during a live fit.
        path = save_selected_meta_model(
            args.output_dir / "recovered", model, meta_mean=mean, meta_scale=scale,
            max_warmup=max_warmup, metadata=metadata)
        restored, saved = load_direct_meta_checkpoint(path)
        if predict_direct_meta_config(restored, saved, meta[TARGET_ID]) != predicted:
            raise RuntimeError(f"Checkpoint reload changed predictions: {path}")
        print(f"[recovery] round={step}: verified and saved {path}", flush=True)
    if failures:
        raise SystemExit(f"Could not reproduce rounds {failures}; inspect original code, data and settings.")


if __name__ == "__main__":
    main()
