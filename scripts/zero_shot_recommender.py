#!/usr/bin/env python3
"""CLI for preparing, training, and querying the zero-shot recommender."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hp_search_head_sweep import _load_train_fixed_test_dataset
from src.zero_shot_recommender.data import best_record_groups, load_trials
from src.zero_shot_recommender.meta_features import extract_meta_features
from src.zero_shot_recommender.training import (
    load_recommender,
    recommend,
    save_recommender,
    train_recommender,
)


DEFAULT_DATASETS = ("massbench_adenocarcinoma", "massbench_alzheimer", "massbench_benchmark")


def extract_all(dataset_names) -> dict:
    output = {}
    for dataset in dataset_names:
        X, y, batches, *_ = _load_train_fixed_test_dataset(dataset)
        output[dataset] = extract_meta_features(X, y, batches)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    meta = sub.add_parser("extract-meta")
    meta.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    meta.add_argument("--output", type=Path, default=ROOT / "results" / "zero_shot_recommender" / "meta_features.json")
    train = sub.add_parser("train")
    train.add_argument("--trials", nargs="+", type=Path, default=[ROOT / "results" / "zero_shot_hpo" / "unified_trials.jsonl"])
    train.add_argument("--meta", type=Path, default=ROOT / "results" / "zero_shot_recommender" / "meta_features.json")
    train.add_argument("--output-dir", type=Path, default=ROOT / "results" / "zero_shot_recommender" / "model")
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--epochs", type=int, default=500)
    query = sub.add_parser("recommend")
    query.add_argument("--dataset", required=True)
    query.add_argument("--model-dir", type=Path, default=ROOT / "results" / "zero_shot_recommender" / "model")
    query.add_argument("--target", default="all", help="Scoped target name, 'global', or 'all'")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "extract-meta":
        values = extract_all(args.datasets)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(values, indent=2))
        print(f"Wrote {len(values)} dataset descriptors to {args.output}")
        return
    if args.command == "train":
        groups = best_record_groups(load_trials(args.trials))
        meta = json.loads(args.meta.read_text())
        manifest = {"targets": {}}
        for target, records in groups.items():
            model, metadata = train_recommender(records, meta, hidden_size=args.hidden_size, epochs=args.epochs)
            save_recommender(model, metadata, args.output_dir / target)
            manifest["targets"][target] = {
                "training_rows": len(records),
                "model_family": records[0].model_family if target != "global" else None,
                "head_type": records[0].config.get("head_type") if target != "global" else None,
            }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Trained {len(groups)} scoped recommenders in {args.output_dir}")
        return
    meta = extract_all([args.dataset])[args.dataset]
    manifest_path = args.model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    targets = list(manifest["targets"]) if args.target == "all" else [args.target]
    unknown = sorted(set(targets) - set(manifest["targets"]))
    if unknown:
        raise ValueError(f"Unknown target(s) {unknown}; available={list(manifest['targets'])}")
    output = {}
    for target in targets:
        model, metadata = load_recommender(args.model_dir / target)
        output[target] = recommend(model, metadata, meta)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
