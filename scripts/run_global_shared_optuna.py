#!/usr/bin/env python3
"""Optimize one shared BERNN configuration across all development datasets."""
from __future__ import annotations

import argparse, importlib.metadata, json, os, sys, time, uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import hp_search
from scripts.run_optuna_comparison import _dataset_cv_settings, _fold_scores_payload
from src.dataset_splits import load_dataset_partitions
from src.evolutionary_meta import aggregate_dataset_scores, recommended_batch_size


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-manifest", type=Path, default=ROOT / "config/evolution_development_datasets.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--n-epochs", type=int, default=1000)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--worst-dataset-weight", type=float, default=.25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="BE_leaderboard_meta_evolution")
    p.add_argument("--wandb-run-name")
    return p.parse_args(argv)


def _atomic(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)
    if args.n_repeats != 3:
        raise ValueError("Global shared HPO requires grouped CV=3")
    if args.n_trials < 1: raise ValueError("n_trials must be positive")
    if importlib.metadata.version("bernn") != "1.0.5":
        raise RuntimeError("This experiment requires bernn==1.0.5")
    parts = load_dataset_partitions(args.split_manifest)
    train_ids, validation_ids = tuple(parts.train), tuple(parts.validation)
    if validation_ids != ("massbench_alzheimer",):
        raise ValueError("Validation must be massbench_alzheimer only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "run_metadata.json"
    if meta_path.exists() and not args.resume: raise FileExistsError("pass --resume")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"arm":"global_shared_hparams", "candidate_budget":args.n_trials,
                 "budget_unit":"shared_config_evaluated_on_all_train_datasets",
                 "train_datasets":train_ids, "validation_datasets":validation_ids,
                 "aggregate":"aggregate_dataset_scores", "worst_dataset_weight":args.worst_dataset_weight,
                 "bernn_version":"1.0.5", "log1p":True, "seed":args.seed,
                 "wandb_run_id":meta.get("wandb_run_id") or uuid.uuid4().hex[:8]})
    _atomic(meta_path, meta)
    import optuna
    storage = f"sqlite:///{(args.output_dir / 'optuna.sqlite3').resolve()}"
    study = optuna.create_study(study_name="global_shared", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed), storage=storage, load_if_exists=True)
    datasets = {} if args.smoke else {d: hp_search.load_dataset(d) for d in train_ids}
    fixed = {} if args.smoke else {d: hp_search.load_fixed_test_dataset(d) for d in train_ids}
    validation = {} if args.smoke else {d: hp_search.load_dataset(d) for d in validation_ids}
    validation_fixed = {} if args.smoke else {d: hp_search.load_fixed_test_dataset(d) for d in validation_ids}
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
            id=meta["wandb_run_id"], resume="allow", config=meta)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for step in range(len(complete), args.n_trials):
        trial = study.ask()
        template = hp_search.parse_args([])
        template.n_epochs, template.max_warmup = args.n_epochs, max(1, min(50,args.n_epochs))
        template.log1p, template.no_wandb, template.combine_test = True, True, False
        config = hp_search.sample_config(trial, template)
        scores, rows = [], {}
        for i, dataset_id in enumerate(train_ids):
            if args.smoke:
                score, metrics = float(np.tanh((trial.number + i + 1)/20)), {}
            else:
                X,y,batches = datasets[dataset_id]
                run_args = hp_search.parse_args([])
                run_args.dataset, run_args.n_epochs = dataset_id, args.n_epochs
                run_args.n_repeats, run_args.resolved_n_repeats = _dataset_cv_settings(dataset_id,args.n_repeats,batches)
                run_args.num_workers, run_args.device, run_args.seed = args.num_workers,args.device,args.seed+i
                run_args.no_wandb, run_args.combine_test, run_args.log1p = True,False,True
                run_args.max_warmup = max(1,min(50,args.n_epochs))
                run_args.bs = recommended_batch_size(batches,cap=args.batch_size)
                run_args.cv_split_cache = str(args.output_dir/"cv_splits"/f"{dataset_id}.npz")
                cfg = dict(config, batch_size=run_args.bs, cv_folds=run_args.resolved_n_repeats,
                           num_workers=args.num_workers, lisi_enabled=False)
                try:
                    score,metrics=hp_search.run_trial(cfg,run_args,datasets[dataset_id],
                        f"global_shared_{meta['wandb_run_id']}_{trial.number}_{dataset_id}",fixed_test_data=fixed[dataset_id])
                except Exception as exc:
                    score,metrics=-1.0,{"error":f"{type(exc).__name__}: {exc}"}
            score=float(score); scores.append(score)
            rows[dataset_id]={"valid_mcc":score,"test_mcc":float(metrics.get("test_mcc",np.nan)),
                              "fold_scores":_fold_scores_payload(metrics)}
        fitness=aggregate_dataset_scores(scores,args.worst_dataset_weight)
        trial.set_user_attr("config",config); trial.set_user_attr("datasets",rows)
        study.tell(trial,fitness)
        best_config = dict(study.best_trial.user_attrs["config"])
        validation_rows = {}
        for i, dataset_id in enumerate(validation_ids):
            if args.smoke:
                valid_score, valid_metrics = 0.0, {}
            else:
                X,y,batches = validation[dataset_id]
                run_args = hp_search.parse_args([])
                run_args.dataset, run_args.n_epochs = dataset_id, args.n_epochs
                run_args.n_repeats, run_args.resolved_n_repeats = _dataset_cv_settings(dataset_id,args.n_repeats,batches)
                run_args.num_workers, run_args.device, run_args.seed = args.num_workers,args.device,args.seed+10000+i
                run_args.no_wandb, run_args.combine_test, run_args.log1p = True,False,True
                run_args.max_warmup=max(1,min(50,args.n_epochs)); run_args.bs=recommended_batch_size(batches,cap=args.batch_size)
                run_args.cv_split_cache = str(args.output_dir/"cv_splits"/f"{dataset_id}.npz")
                cfg=dict(best_config,batch_size=run_args.bs,cv_folds=run_args.resolved_n_repeats,
                         num_workers=args.num_workers,lisi_enabled=False)
                try:
                    valid_score,valid_metrics=hp_search.run_trial(cfg,run_args,validation[dataset_id],
                        f"global_shared_validation_{meta['wandb_run_id']}_{trial.number}_{dataset_id}",
                        fixed_test_data=validation_fixed[dataset_id])
                except Exception as exc:
                    valid_score,valid_metrics=-1.0,{"error":f"{type(exc).__name__}: {exc}"}
            validation_rows[dataset_id]={"valid_mcc":float(valid_score),
                "test_mcc":float(valid_metrics.get("test_mcc",np.nan)),"fold_scores":_fold_scores_payload(valid_metrics)}
        record={"candidate_index":step,"trial_number":trial.number,"aggregate_valid_mcc":fitness,
                "config":config,"datasets":rows,"validation":validation_rows,
                "validation_uses_current_global_best":True,"monitoring_only_fixed_test":True}
        with (args.output_dir/"trials.jsonl").open("a") as f: f.write(json.dumps(record,default=str)+"\n")
        if wandb_run: wandb_run.log({"candidate_index":step,"global/aggregate_valid_mcc":fitness})
    if wandb_run: wandb_run.finish()
    return 0

if __name__ == "__main__": raise SystemExit(main())
