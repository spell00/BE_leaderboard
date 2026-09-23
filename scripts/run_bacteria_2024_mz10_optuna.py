#!/usr/bin/env python3
"""Independent BERNN Optuna sweep for the paper-aligned MSML bacteria mz10 matrix."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts import hp_search
from scripts.run_independent_optuna_all_datasets import (
    atomic_json,
    compact_metrics,
    completed,
    config_metrics,
    mark_interrupted_failed,
    persist_trials,
    records_for,
    wandb_metrics,
)
from src.evolutionary_meta import recommended_batch_size

DATASET = "bacteria_2024_mz10"
N_SPLITS = 5
DATA_DIR = ROOT / "data" / "datasets" / DATASET
MATRIX_PATH = DATA_DIR / f"{DATASET}_features_csr.npz"
META_PATH = DATA_DIR / f"{DATASET}_metadata.csv"
FEATURE_PATH = DATA_DIR / f"{DATASET}_feature_names.npy"
PROVENANCE_PATH = DATA_DIR / "provenance.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "independent_optuna_all_20")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="BE_leaderboard_meta_evolution")
    p.add_argument("--wandb-group", default="independent-optuna-all-datasets-20")
    return p.parse_args(argv)


def load_bundle():
    missing = [p for p in (MATRIX_PATH, META_PATH, FEATURE_PATH, PROVENANCE_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing bacteria mz10 bundle file(s): {missing}")

    meta = pd.read_csv(META_PATH)
    features = np.load(FEATURE_PATH, allow_pickle=True).astype(str)
    matrix = sparse.load_npz(MATRIX_PATH).tocsr().astype(np.float32)

    if matrix.shape != (len(meta), len(features)):
        raise ValueError(
            f"Bundle shape mismatch: matrix={matrix.shape}, metadata={len(meta)}, features={len(features)}"
        )
    required = {"name", "label", "batch"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Metadata must contain {sorted(required)}; got {list(meta.columns)}")

    expected_batches = {f"b{i}" for i in range(1, 16)}
    batches = meta["batch"].astype(str).to_numpy()
    labels = meta["label"].astype(str).to_numpy()
    if set(batches) != expected_batches:
        raise ValueError(f"Expected B1-B15 only; got {sorted(set(batches))}")
    if len(set(labels)) != 29:
        raise ValueError(f"Expected 29 classes; got {len(set(labels))}")
    if len(meta) != 1858 or meta["name"].nunique() != 1857:
        raise ValueError(
            f"Paper-aligned cohort changed: rows={len(meta)}, unique_names={meta['name'].nunique()}"
        )

    # Dense only in RAM; compact on disk. FP16 halves host-memory pressure and
    # matches the requested CUDA training precision.
    dense = matrix.toarray().astype(np.float16, copy=False)
    del matrix
    X = pd.DataFrame(dense, columns=features, copy=False)
    print(
        f"[bacteria data] samples={len(X)} features={X.shape[1]} "
        f"classes={len(set(labels))} batches={len(set(batches))} dtype={X.dtypes.iloc[0]}",
        flush=True,
    )
    return X, labels, batches, meta


def natural_batch_key(value: str):
    value = str(value)
    if value.startswith("b") and value[1:].isdigit():
        return (0, int(value[1:]))
    return (1, value)


def choose_five_batch_groups(labels, batches, seed):
    labels = np.asarray(labels).astype(str)
    batches = np.asarray(batches).astype(str)
    all_batches = set(batches.tolist())
    best = None

    # StratifiedGroupKFold assigns whole technical batches while balancing the
    # 29-class distribution. Try deterministic seeds and prefer a partition for
    # which every held-out class remains represented in the 60% training set.
    for candidate_seed in range(int(seed), int(seed) + 500):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=candidate_seed)
        groups = []
        for _, held_idx in splitter.split(np.zeros(len(labels)), labels, groups=batches):
            group = sorted(set(batches[held_idx].tolist()), key=natural_batch_key)
            groups.append(group)

        flattened = [b for group in groups for b in group]
        if len(flattened) != len(set(flattened)) or set(flattened) != all_batches:
            continue

        missing_total = 0
        missing_by_round = []
        group_sizes = []
        for group in groups:
            group_sizes.append(int(np.isin(batches, group).sum()))
        for round_index in range(N_SPLITS):
            valid_batches = groups[round_index]
            test_batches = groups[(round_index + 1) % N_SPLITS]
            held = set(valid_batches) | set(test_batches)
            train_mask = ~np.isin(batches, list(held))
            held_mask = ~train_mask
            train_classes = set(labels[train_mask].tolist())
            held_classes = set(labels[held_mask].tolist())
            missing = sorted(held_classes - train_classes)
            missing_total += len(missing)
            missing_by_round.append(missing)

        imbalance = max(group_sizes) - min(group_sizes)
        score = (missing_total, imbalance, candidate_seed)
        if best is None or score < best[0]:
            best = (score, groups, group_sizes, missing_by_round)
        if missing_total == 0 and imbalance <= max(30, int(len(labels) * 0.05)):
            break

    if best is None:
        raise RuntimeError("Could not construct five grouped batch folds")

    score, groups, group_sizes, missing_by_round = best
    if score[0] != 0:
        raise ValueError(
            "No five-fold partition found with complete training class coverage; "
            f"best_missing={missing_by_round}"
        )
    return groups, group_sizes, score[2]


def make_splitter(labels, expected_batches, groups):
    labels = np.asarray(labels).astype(str)
    expected_batches = np.asarray(expected_batches).astype(str)

    def split5(batches, eligible_mask=None):
        values = np.asarray(pd.Series(batches).astype(str))
        if values.shape != expected_batches.shape or not np.array_equal(values, expected_batches):
            raise ValueError("bacteria_2024_mz10 split called with unexpected batch ordering")
        if eligible_mask is not None and not np.all(np.asarray(eligible_mask, dtype=bool)):
            raise ValueError("bacteria_2024_mz10 has no unsupervised rows")

        splits = []
        for round_index in range(N_SPLITS):
            valid_batches = groups[round_index]
            test_batches = groups[(round_index + 1) % N_SPLITS]
            held = set(valid_batches) | set(test_batches)
            train_batches = sorted(
                set(values.tolist()) - held,
                key=natural_batch_key,
            )
            train_idx = np.flatnonzero(np.isin(values, train_batches))
            valid_idx = np.flatnonzero(np.isin(values, valid_batches))
            test_idx = np.flatnonzero(np.isin(values, test_batches))
            train_classes = set(labels[train_idx].tolist())
            missing = (set(labels[valid_idx].tolist()) | set(labels[test_idx].tolist())) - train_classes
            if missing:
                raise ValueError(f"Fold {round_index + 1} training set misses held-out classes: {sorted(missing)}")
            splits.append({
                "round": round_index + 1,
                "train_idx": train_idx,
                "valid_idx": valid_idx,
                "test_idx": test_idx,
                "train_batches": train_batches,
                "valid_batches": list(valid_batches),
                "test_batches": list(test_batches),
                "valid_batch": list(valid_batches),
                "test_batch": list(test_batches),
                "n_splits": N_SPLITS,
                "requested_n_splits": N_SPLITS,
            })
        return splits

    return split5


def install_fp16_training():
    original = hp_search.build_trainer_config

    def build_fp16(cfg, args, exp_id):
        training_config = original(cfg, args, exp_id)
        training_config.precision = "fp16"
        training_config.tf32 = False
        return training_config

    hp_search.build_trainer_config = build_fp16


def main(argv=None):
    args = parse_args(argv)
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive")

    X, y, batches, metadata = load_bundle()
    groups, group_sizes, split_seed = choose_five_batch_groups(y, batches, args.seed)
    split_payload = {
        "dataset": DATASET,
        "protocol": "five_group_cyclic_train_valid_test",
        "n_splits": N_SPLITS,
        "split_seed": int(split_seed),
        "groups": groups,
        "group_sample_counts": group_sizes,
        "role_rotation": "fold i: valid=group i, test=group i+1, train=remaining 3 groups",
        "selection_metric": "mean validation MCC",
        "test_role": "monitoring_only_excluded_from_optuna",
    }
    print("[bacteria splits] " + json.dumps(split_payload, sort_keys=True), flush=True)

    # Reuse the canonical BERNN cyclic trial implementation with our fixed
    # five-group split plan and FP16 TrainingConfig.
    hp_search.cyclic_train_valid_test_splits = make_splitter(y, batches, groups)
    install_fp16_training()

    out = args.output_dir / DATASET
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "split_groups.json", split_payload)

    run_meta_path = out / "run_metadata.json"
    exists = run_meta_path.exists()
    if exists and not args.resume:
        raise FileExistsError(f"{out} already exists; pass --resume")
    run_meta = json.loads(run_meta_path.read_text()) if exists else {}
    run_meta.update({
        "schema_version": 1,
        "run_id": run_meta.get("run_id") or uuid.uuid4().hex[:10],
        "dataset": DATASET,
        "protocol": "cyclic5",
        "source_file": str(MATRIX_PATH.relative_to(ROOT)),
        "metadata_file": str(META_PATH.relative_to(ROOT)),
        "n_trials": args.n_trials,
        "n_epochs": args.n_epochs,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(set(y.tolist())),
        "n_batches": len(set(batches.tolist())),
        "cv_folds": N_SPLITS,
        "precision": "fp16",
        "source_transform": "logaddinloop",
        "additional_log1p": False,
        "selection_metric": "valid_mcc",
        "test_role": "monitoring_only_excluded_from_optuna",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wandb_run_id": run_meta.get("wandb_run_id") or uuid.uuid4().hex[:8],
    })
    atomic_json(run_meta_path, run_meta)

    import mlflow
    import optuna

    mlruns = out / "mlruns"
    mlruns.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlruns.resolve().as_uri())
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{(out / 'optuna.sqlite3').resolve()}")
    study = optuna.create_study(
        study_name=f"independent_{DATASET}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )
    if args.resume:
        mark_interrupted_failed(study)

    wb = None
    if not args.no_wandb:
        import wandb
        wb = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"optuna-{DATASET}-{args.n_trials}",
            id=run_meta["wandb_run_id"],
            resume="allow",
            config={**run_meta, "batch_size_cap": args.batch_size, "num_workers": args.num_workers},
            tags=["independent_optuna", DATASET, "cyclic5", "paper_aligned", "fp16"],
        )
        wandb.define_metric("trial_index")
        wandb.define_metric("metrics/*", step_metric="trial_index")
        wandb.define_metric("folds/*", step_metric="trial_index")
        wandb.define_metric("best/*", step_metric="trial_index")

    data = (X, y, batches)
    try:
        start = len(completed(study))
        print(
            f"[worker] {DATASET}: samples={len(X)} features={X.shape[1]} "
            f"batches=15 folds=5 precision=fp16 completed={start}/{args.n_trials}",
            flush=True,
        )
        for index in range(start, args.n_trials):
            trial = study.ask()
            run_args = hp_search.parse_args([])
            run_args.dataset = DATASET
            run_args.n_epochs = args.n_epochs
            run_args.num_workers = args.num_workers
            run_args.device = args.device
            run_args.seed = args.seed
            run_args.no_wandb = True
            run_args.combine_test = False
            run_args.max_warmup = max(1, min(50, args.n_epochs))
            run_args.log1p = False
            run_args.bs = recommended_batch_size(batches, cap=args.batch_size)
            run_args.results_dir = str(out / "bernn")
            run_args.n_repeats = N_SPLITS
            run_args.resolved_n_repeats = N_SPLITS
            run_args.trainer_n_repeats = 1

            config = hp_search.sample_config(trial, run_args)
            config["log1p"] = False
            config.update({
                "batch_size": int(run_args.bs),
                "cv_folds": N_SPLITS,
                "num_workers": int(run_args.num_workers),
                "lisi_enabled": False,
                "precision": "fp16",
            })

            metrics, error = {}, None
            started = time.monotonic()
            try:
                exp_id = f"independent_{run_meta['run_id']}_{DATASET}_t{trial.number}"
                score, metrics = hp_search.run_cyclic_batch_trial(config, run_args, data, exp_id)
                score = float(score)
            except Exception as exc:
                score = -1.0
                error = f"{type(exc).__name__}: {exc}"
                print(f"[worker] {DATASET} trial {trial.number} failed: {error}", flush=True)

            clean = compact_metrics(metrics)
            clean.setdefault("valid_mcc", score)
            fit_seconds = time.monotonic() - started
            trial.set_user_attr("config", config)
            trial.set_user_attr("metrics", clean)
            trial.set_user_attr("fit_seconds", fit_seconds)
            if error:
                trial.set_user_attr("error", error)
            study.tell(trial, score)

            rows = records_for(study, "cyclic5")
            persist_trials(out, rows)
            done = sorted(completed(study), key=lambda item: item.number)
            current = done[-1]
            best = max(done, key=lambda item: float(item.value))
            current_metrics = dict(current.user_attrs.get("metrics", {}))
            best_metrics = dict(best.user_attrs.get("metrics", {}))
            if wb:
                wb.log({
                    "trial_index": index,
                    "trial_number": int(current.number),
                    "metrics/valid_mcc": float(current.value),
                    "metrics/test_mcc": current_metrics.get("test_mcc"),
                    "runtime/fit_seconds": current.user_attrs.get("fit_seconds"),
                    "best/valid_mcc": float(best.value),
                    "best/test_mcc": best_metrics.get("test_mcc"),
                    "best/trial_number": int(best.number),
                    **wandb_metrics(current_metrics),
                    **config_metrics(dict(current.user_attrs.get("config", {}))),
                })
            print(
                f"[worker] {DATASET} {index + 1}/{args.n_trials}: "
                f"valid={float(current.value):.4f} test={current_metrics.get('test_mcc')} "
                f"best_valid={float(best.value):.4f} best_test={best_metrics.get('test_mcc')}",
                flush=True,
            )

        best = max(completed(study), key=lambda item: float(item.value))
        best_metrics = dict(best.user_attrs.get("metrics", {}))
        summary = {
            "dataset": DATASET,
            "protocol": "cyclic5",
            "completed_trials": len(completed(study)),
            "best_trial_number": int(best.number),
            "best_valid_mcc": float(best.value),
            "paired_test_mcc_at_best_valid": best_metrics.get("test_mcc"),
            "best_config": dict(best.user_attrs.get("config", {})),
            "selection_metric": "valid_mcc",
            "test_role": "monitoring_only_excluded_from_optuna",
            "precision": "fp16",
            "cv_folds": N_SPLITS,
        }
        atomic_json(out / "summary.json", summary)
        if wb:
            wb.summary.update(summary)
        return 0
    finally:
        if wb:
            wb.finish()
        try:
            storage.engine.dispose()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
