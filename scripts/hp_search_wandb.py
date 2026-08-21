"""W&B wrapper for hp_search.py — a drop-in replacement for make_objective.

Logs every trial (params + all captured bernn metrics) to a Weights & Biases
run. MLflow is still used underneath: run_trial() trains each CV fold with
log_mlflow=True, so per-fold metrics land in mlruns/ regardless of W&B. W&B here
just adds a per-trial dashboard view on top.

W&B calls are best-effort: a login/init/log failure never kills the sweep (a
37h unattended run shouldn't die on a transient network blip) — it falls back to
MLflow-only logging for that trial.
"""

import os

import wandb

from wandb_cv_logging import log_compact_epoch_charts, log_compact_fold_chart, set_numeric_summary

# Override where runs land via env before `dvc repro` (both inherited by the
# DVC subprocess): WANDB_PROJECT, WANDB_ENTITY. Entity defaults to your ~/.netrc
# default (adlab). Set WANDB_MODE=offline to log locally only (no cloud sync).
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "BE_leaderboard_zero_shot_hparams")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "adlab")


def make_objective_wandb(args, data):
    """Same objective as hp_search.make_objective, plus per-trial W&B logging.

    IMPORTANT ordering: bernn's per-fold MLflow runs (with background logging
    threads) run FIRST, with no active W&B run. An *online* W&B run held open
    during training races bernn's MLflow run lifecycle and triggers
    'MlflowException: Run ... not found'. So we only open the W&B run afterward,
    purely to mirror the trial's aggregated metrics to the dashboard.
    """
    from hp_search import sample_config, run_trial

    def objective(trial):
        cfg = sample_config(trial, args)
        for k, v in cfg.items():
            trial.set_user_attr(k, v)
        exp_id = f"{args.out_prefix}_{args.dataset}_t{trial.number}"
        trial.set_user_attr("mlflow_exp", exp_id)

        # --- MLflow-backed training first (no W&B run active) ---
        try:
            score, metrics = run_trial(cfg, args, data, exp_id)
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            print(f"[trial {trial.number}] FAILED: {type(exc).__name__}: {exc}")
            _log_wandb_trial(args, trial, cfg, {"valid_mcc": -1.0, "error": str(exc)}, exit_code=1)
            return -1.0

        for mk, mv in metrics.items():
            trial.set_user_attr(f"metric_{mk}", mv)
        # --- then mirror the aggregate to W&B ---
        _log_wandb_trial(args, trial, cfg, {"valid_mcc": score, **metrics})
        print(f"[trial {trial.number}] valid MCC = {score:.4f}  "
              f"(dloss={cfg['dloss']} vae={cfg['variational']} kan={cfg['kan']}) "
              f"[{len(metrics)} metrics captured]")
        return score

    return objective


def _log_wandb_trial(args, trial, cfg, payload, exit_code=0):
    """Open a short W&B run, log the trial payload, finish. Best-effort."""
    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"trial_{trial.number}",
            group=f"{args.out_prefix}_{args.dataset}",
            config={**cfg, "dataset": args.dataset, "n_epochs": args.n_epochs,
                    "n_repeats": args.n_repeats,
                    "rec_loss": getattr(args, "rec_loss", "l1"),
                    "class_triplet_w": getattr(args, "class_triplet_w", 1.0),
                    "trial": trial.number},
            reinit="finish_previous",
        )
        metric_payload = dict(payload)
        epoch_traces = metric_payload.pop("_epoch_traces_by_fold", {})
        primary_valid = metric_payload.get("valid_mcc")
        if primary_valid is not None:
            metric_payload["best_valid_mcc"] = primary_valid
            metric_payload["objective_mcc"] = primary_valid
            metric_payload["objective_valid_mcc"] = primary_valid

        # Friendly aliases for BERNN's sanitized MLflow metric names. Keep the
        # raw names too, but expose the obvious columns in W&B summaries/charts.
        alias_candidates = {
            "train_mcc": ["mcc_train_all_concentrations", "mcc_train_all", "train_mcc"],
            "valid_mcc_bernn": ["mcc_valid_all_concentrations", "mcc_valid_all"],
            "test_mcc": ["mcc_test_all_concentrations", "mcc_test_all", "test_mcc"],
            "train_accuracy": ["acc_train_all_concentrations", "acc_train_all", "train_accuracy"],
            "valid_accuracy": ["acc_valid_all_concentrations", "acc_valid_all", "valid_accuracy"],
            "test_accuracy": ["acc_test_all_concentrations", "acc_test_all", "test_accuracy"],
            "batch_entropy": ["batch_entropy_valid_enc_domains", "batch_entropy"],
            "batch_silhouette": ["silhouette_valid_enc_domains", "batch_silhouette"],
            "batch_lisi": ["lisi_valid_enc_domains", "batch_lisi", "batch_ilisi"],
            "batch_kbet": ["kbet_valid_enc_domains", "batch_kbet"],
            "batch_ari": ["adjusted_rand_score_valid_enc_domains", "batch_ari"],
            "batch_ami": ["adjusted_mutual_info_score_valid_enc_domains", "batch_ami"],
        }
        for alias, candidates in alias_candidates.items():
            if alias in metric_payload:
                continue
            for candidate in candidates:
                if candidate in metric_payload:
                    metric_payload[alias] = metric_payload[candidate]
                    break

        for key in ("valid_mcc", "test_mcc", "train_mcc", "best_valid_mcc"):
            try:
                run.define_metric(key, summary="max")
            except Exception:
                pass
        set_numeric_summary(run, metric_payload)
        log_compact_fold_chart(run, metric_payload)
        log_compact_epoch_charts(run, epoch_traces)
        run.finish(exit_code=exit_code)
    except Exception as exc:
        print(f"[wandb] logging failed for trial {trial.number}: {exc}")
