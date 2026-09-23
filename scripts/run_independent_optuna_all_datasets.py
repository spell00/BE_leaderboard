#!/usr/bin/env python3
"""Run 20 independent BERNN Optuna studies across all leaderboard datasets.

Each dataset has its own Optuna study, SQLite file, result directory, and W&B
run. The launcher assigns one dataset process per GPU and queues the newly added
datasets first.

Existing datasets keep the meta-hpo-bank fixed-external protocol: Optuna sees
only grouped-CV validation MCC, while the labeled *_inference.csv cross-test is
monitoring-only. New whole-dataset benchmarks use cyclic batch train/valid/test
on *_all.csv; test MCC is likewise excluded from selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hp_search
from src.evolutionary_meta import recommended_batch_size

DATASETS = (
    # New datasets first.
    ("jdlber_sle_maldi", "cyclic"),
    ("seqc_maqc", "cyclic"),
    ("scib_pancreas", "cyclic"),
    # Existing meta-hpo-bank datasets.
    ("normal_tissue_878", "fixed_external"),
    ("colon_3041", "fixed_external"),
    ("massbench_adenocarcinoma", "fixed_external"),
    ("massbench_benchmark", "fixed_external"),
    ("massbench_alzheimer", "fixed_external"),
)
CV_FOLDS = {
    "normal_tissue_878": 3,
    "colon_3041": 3,
    "massbench_adenocarcinoma": 2,
    "massbench_benchmark": 3,
    "massbench_alzheimer": 3,
}
PREPARE = {
    "jdlber_sle_maldi": [sys.executable, str(ROOT / "scripts" / "prepare_jdlber_sle_maldi.py")],
    "seqc_maqc": [sys.executable, str(ROOT / "scripts" / "download_prepare_seqc_maqc.py")],
    "scib_pancreas": [sys.executable, str(ROOT / "scripts" / "download_prepare_scib_pancreas.py")],
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "independent_optuna_all_20")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=1000)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--datasets", default=None, help="Optional comma-separated subset.")
    p.add_argument("--prepare-missing", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="BE_leaderboard_meta_evolution")
    p.add_argument("--wandb-group", default="independent-optuna-all-datasets-20")
    p.add_argument("--worker-dataset", default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-protocol", choices=("fixed_external", "cyclic"), help=argparse.SUPPRESS)
    return p.parse_args(argv)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def selected_datasets(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return list(DATASETS)
    protocol = dict(DATASETS)
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in protocol]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}; known={list(protocol)}")
    return [(name, protocol[name]) for name in names]


def required_path(dataset: str, protocol: str) -> Path:
    base = ROOT / "data" / "datasets" / dataset
    suffix = "_all.csv" if protocol == "cyclic" else "_train.csv"
    return base / f"{dataset}{suffix}"


def prepare_missing(dataset: str, protocol: str) -> None:
    target = required_path(dataset, protocol)
    if target.exists():
        return
    command = PREPARE.get(dataset)
    if not command:
        raise FileNotFoundError(f"Missing {target}; no automatic preparer is registered")
    print(f"[prepare] {dataset}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not target.exists():
        raise FileNotFoundError(f"Preparation did not create {target}")


def completed(study):
    import optuna
    return [
        t for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


def mark_interrupted_failed(study) -> None:
    import optuna
    for trial in study.get_trials(deepcopy=False):
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            print(f"[resume] marked trial {trial.number} FAIL", flush=True)


def json_metric(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            converted = json_metric(item)
            if converted is not None:
                out.append(converted)
        return out
    if isinstance(value, str):
        return value
    return None


def compact_metrics(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if str(key).startswith("_epoch_traces"):
            continue
        converted = json_metric(value)
        if converted is not None:
            out[str(key)] = converted
    return out


def wandb_metrics(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        converted = json_metric(value)
        if isinstance(converted, (int, float)):
            out[f"metrics/{key}"] = converted
    for split in ("valid", "test"):
        values = metrics.get(f"{split}_mcc_folds", [])
        if isinstance(values, (list, tuple)):
            for fold, value in enumerate(values):
                converted = json_metric(value)
                if converted is not None:
                    out[f"folds/{split}_mcc/fold_{fold}"] = converted
    return out


def config_metrics(config: dict) -> dict:
    out = {}
    for key, value in config.items():
        if isinstance(value, (bool, np.bool_)):
            out[f"config/{key}"] = int(value)
        elif isinstance(value, (int, float, np.integer, np.floating, str)):
            out[f"config/{key}"] = value
    return out


def records_for(study, protocol: str) -> list[dict]:
    rows = []
    for index, trial in enumerate(sorted(completed(study), key=lambda x: x.number)):
        attrs = dict(trial.user_attrs)
        metrics = dict(attrs.get("metrics", {}))
        rows.append({
            "trial_index": index,
            "trial_number": int(trial.number),
            "valid_mcc": float(trial.value),
            "test_mcc": metrics.get("test_mcc"),
            "fit_seconds": attrs.get("fit_seconds"),
            "protocol": protocol,
            "config": attrs.get("config", {}),
            "metrics": metrics,
            "error": attrs.get("error"),
            "selection_metric": "valid_mcc",
            "test_role": "monitoring_only_excluded_from_optuna",
        })
    return rows


def persist_trials(output_dir: Path, rows: list[dict]) -> None:
    atomic_json(output_dir / "trials.json", rows)
    fields = (
        "trial_index", "trial_number", "valid_mcc", "test_mcc",
        "valid_mcc_folds", "test_mcc_folds",
        "valid_mcc_std", "test_mcc_std", "test_mcc_fold_mean",
        "test_mcc_fold_std", "test_mcc_global_oof", "fit_seconds",
        "protocol", "error",
    )
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "trial_index": row["trial_index"],
                "trial_number": row["trial_number"],
                "valid_mcc": row["valid_mcc"],
                "test_mcc": row["test_mcc"],
                "valid_mcc_folds": json.dumps(metrics.get("valid_mcc_folds", [])),
                "test_mcc_folds": json.dumps(metrics.get("test_mcc_folds", [])),
                "valid_mcc_std": metrics.get("valid_mcc_std"),
                "test_mcc_std": metrics.get("test_mcc_std"),
                "test_mcc_fold_mean": metrics.get("test_mcc_fold_mean"),
                "test_mcc_fold_std": metrics.get("test_mcc_fold_std"),
                "test_mcc_global_oof": metrics.get("test_mcc_global_oof"),
                "fit_seconds": row["fit_seconds"],
                "protocol": row["protocol"],
                "error": row["error"],
            })


def run_worker(args) -> int:
    dataset, protocol = args.worker_dataset, args.worker_protocol
    out = args.output_dir / dataset
    out.mkdir(parents=True, exist_ok=True)
    if args.prepare_missing:
        prepare_missing(dataset, protocol)
    target = required_path(dataset, protocol)
    if not target.exists():
        raise FileNotFoundError(f"Missing {target}; use --prepare-missing or prepare it first")

    if protocol == "cyclic":
        source_file = f"{dataset}_all.csv"
        data = hp_search.load_cyclic_dataset(dataset, source_file=source_file)
        fixed_test = None
    else:
        source_file = f"{dataset}_train.csv"
        inference = ROOT / "data" / "datasets" / dataset / f"{dataset}_inference.csv"
        if not inference.exists():
            raise FileNotFoundError(f"Missing labeled fixed cross-test file {inference}")
        data = hp_search.load_dataset(dataset)
        fixed_test = hp_search.load_fixed_test_dataset(dataset)

    X, _, batches = data
    batch_values = sorted(np.unique(np.asarray(batches).astype(str)).tolist())
    meta_path = out / "run_metadata.json"
    exists = meta_path.exists()
    if exists and not args.resume:
        raise FileExistsError(f"{out} already exists; pass --resume")
    meta = json.loads(meta_path.read_text()) if exists else {}
    meta.update({
        "schema_version": 1,
        "run_id": meta.get("run_id") or uuid.uuid4().hex[:10],
        "dataset": dataset,
        "protocol": protocol,
        "source_file": source_file,
        "n_trials": args.n_trials,
        "n_epochs": args.n_epochs,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_batches": len(batch_values),
        "batches": batch_values,
        "selection_metric": "valid_mcc",
        "test_role": "monitoring_only_excluded_from_optuna",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wandb_run_id": meta.get("wandb_run_id") or uuid.uuid4().hex[:8],
    })
    atomic_json(meta_path, meta)

    import mlflow
    import optuna

    mlruns = out / "mlruns"
    mlruns.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlruns.resolve().as_uri())
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{(out / 'optuna.sqlite3').resolve()}")
    study = optuna.create_study(
        study_name=f"independent_{dataset}",
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
            name=f"optuna-{dataset}-{args.n_trials}",
            id=meta["wandb_run_id"],
            resume="allow",
            config={**meta, "batch_size_cap": args.batch_size, "num_workers": args.num_workers},
            tags=["independent_optuna", dataset, protocol],
        )
        wandb.define_metric("trial_index")
        wandb.define_metric("metrics/*", step_metric="trial_index")
        wandb.define_metric("folds/*", step_metric="trial_index")
        wandb.define_metric("best/*", step_metric="trial_index")

    try:
        start = len(completed(study))
        print(
            f"[worker] {dataset}: protocol={protocol} samples={len(X)} "
            f"features={X.shape[1]} batches={len(batch_values)} completed={start}/{args.n_trials}",
            flush=True,
        )
        for index in range(start, args.n_trials):
            trial = study.ask()
            run_args = hp_search.parse_args([])
            run_args.dataset = dataset
            run_args.n_epochs = args.n_epochs
            run_args.num_workers = args.num_workers
            run_args.device = "cuda"
            run_args.seed = args.seed
            run_args.no_wandb = True
            run_args.combine_test = False
            run_args.max_warmup = max(1, min(50, args.n_epochs))
            run_args.log1p = True
            run_args.bs = recommended_batch_size(batches, cap=args.batch_size)
            run_args.results_dir = str(out / "bernn")

            if protocol == "fixed_external":
                requested = CV_FOLDS.get(dataset, args.n_repeats)
                run_args.n_repeats = requested
                run_args.resolved_n_repeats = hp_search.resolve_n_repeats(requested, batches)
                run_args.cv_split_cache = str(out / "cv_splits.npz")
            else:
                run_args.n_repeats = len(batch_values)
                run_args.resolved_n_repeats = len(batch_values)
                run_args.trainer_n_repeats = 1

            config = hp_search.sample_config(trial, run_args)
            config["log1p"] = True
            config.update({
                "batch_size": int(run_args.bs),
                "cv_folds": int(run_args.resolved_n_repeats),
                "num_workers": int(run_args.num_workers),
                "lisi_enabled": False,
            })

            metrics, error = {}, None
            started = time.monotonic()
            try:
                exp_id = f"independent_{meta['run_id']}_{dataset}_t{trial.number}"
                if protocol == "cyclic":
                    score, metrics = hp_search.run_cyclic_batch_trial(config, run_args, data, exp_id)
                else:
                    score, metrics = hp_search.run_trial(
                        config, run_args, data, exp_id, fixed_test_data=fixed_test
                    )
                score = float(score)
            except Exception as exc:
                score = -1.0
                error = f"{type(exc).__name__}: {exc}"
                print(f"[worker] {dataset} trial {trial.number} failed: {error}", flush=True)

            clean = compact_metrics(metrics)
            clean.setdefault("valid_mcc", score)
            fit_seconds = time.monotonic() - started
            trial.set_user_attr("config", config)
            trial.set_user_attr("metrics", clean)
            trial.set_user_attr("fit_seconds", fit_seconds)
            if error:
                trial.set_user_attr("error", error)
            study.tell(trial, score)

            rows = records_for(study, protocol)
            persist_trials(out, rows)
            done = sorted(completed(study), key=lambda x: x.number)
            current = done[-1]
            best = max(done, key=lambda x: float(x.value))
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
                f"[worker] {dataset} {index + 1}/{args.n_trials}: "
                f"valid={float(current.value):.4f} test={current_metrics.get('test_mcc')} "
                f"best_valid={float(best.value):.4f} best_test={best_metrics.get('test_mcc')}",
                flush=True,
            )

        best = max(completed(study), key=lambda x: float(x.value))
        best_metrics = dict(best.user_attrs.get("metrics", {}))
        summary = {
            "dataset": dataset,
            "protocol": protocol,
            "completed_trials": len(completed(study)),
            "best_trial_number": int(best.number),
            "best_valid_mcc": float(best.value),
            "paired_test_mcc_at_best_valid": best_metrics.get("test_mcc"),
            "best_config": dict(best.user_attrs.get("config", {})),
            "selection_metric": "valid_mcc",
            "test_role": "monitoring_only_excluded_from_optuna",
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


def run_launcher(args) -> int:
    jobs = selected_datasets(args.datasets)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_missing:
        for dataset, protocol in jobs:
            prepare_missing(dataset, protocol)

    manifest = {
        "schema_version": 1,
        "datasets": [{"dataset": d, "protocol": p} for d, p in jobs],
        "new_datasets_first": True,
        "gpus": gpus,
        "n_trials_per_dataset": args.n_trials,
        "n_epochs": args.n_epochs,
        "selection_metric": "valid_mcc",
        "test_role": "monitoring_only_excluded_from_optuna",
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
    }
    atomic_json(args.output_dir / "experiment_manifest.json", manifest)

    queue, active, failures = list(jobs), {}, []
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            dataset, protocol = queue.pop(0)
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker-dataset", dataset, "--worker-protocol", protocol,
                "--output-dir", str(args.output_dir),
                "--n-trials", str(args.n_trials), "--n-epochs", str(args.n_epochs),
                "--n-repeats", str(args.n_repeats), "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers), "--seed", str(args.seed),
                "--wandb-project", args.wandb_project, "--wandb-group", args.wandb_group,
            ]
            if args.resume:
                cmd.append("--resume")
            if args.no_wandb:
                cmd.append("--no-wandb")
            if args.prepare_missing:
                cmd.append("--prepare-missing")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            logs = args.output_dir / "logs"
            logs.mkdir(exist_ok=True)
            log_path = logs / f"{dataset}.log"
            stream = log_path.open("a" if args.resume else "w", encoding="utf-8")
            process = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            active[gpu] = (dataset, protocol, process, stream, log_path)
            print(f"[launcher] GPU {gpu} <- {dataset} ({protocol}) log={log_path}", flush=True)

        time.sleep(2)
        for gpu, job in list(active.items()):
            dataset, protocol, process, stream, log_path = job
            code = process.poll()
            if code is None:
                continue
            stream.close()
            print(f"[launcher] GPU {gpu} finished {dataset} exit={code}", flush=True)
            if code:
                failures.append({
                    "dataset": dataset, "protocol": protocol,
                    "exit_code": code, "log": str(log_path),
                })
            del active[gpu]

    summaries = []
    for dataset, _ in jobs:
        path = args.output_dir / dataset / "summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text()))
    atomic_json(args.output_dir / "all_dataset_summary.json", summaries)
    manifest["failures"] = failures
    manifest["completed_datasets"] = [row["dataset"] for row in summaries]
    manifest["completed_at_unix"] = time.time()
    atomic_json(args.output_dir / "experiment_manifest.json", manifest)
    if failures:
        print(json.dumps({"failures": failures}, indent=2), flush=True)
        return 1
    print("[launcher] all dataset studies completed successfully", flush=True)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker_dataset:
        if not args.worker_protocol:
            raise ValueError("--worker-protocol is required in worker mode")
        return run_worker(args)
    return run_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
