"""Per-family BERNN hyperparameter sweep + full metric registration.

Runs ``hp_search`` once for each of the 6 BERNN model families
(``hp_search.PRESETS``), fixing (dloss, variational) and tuning the continuous
knobs within each. For every family it:

  1. runs an Optuna study (objective = bernn validation MCC), writing
     ``results/<preset>_trials.csv`` with ALL captured metrics (acc/mcc/top3,
     losses, and the batch-effect suite: batch_entropy/silhouette/lisi/kbet/ARI
     /AMI for enc & rec) and ``results/<preset>_best.json``;
  2. re-trains the winning config once with ``keep_models=True`` and computes the
     detailed classification report (accuracy, balanced accuracy, and macro
     precision / recall==sensitivity / specificity / f1, plus per-class) from the
     retained valid/test prediction CSVs — the numbers bernn does not log itself;
  3. records the family's tuned continuous hparams.

Finally it writes:

  * ``results/sweep_summary.csv``   — one row per family, ranked by valid MCC
  * ``results/bernn_tuned_defaults.json`` — per-family config overrides, ready
    for ``register_bernn_defaults.py`` to fold into src/baselines.py.

Usage
-----
    python hp_search_sweep.py --n-trials 100 --n-epochs 200 --device cuda
    python hp_search_sweep.py --presets ae_dann vae_dann --n-trials 40   # subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hp_search as hp

ROOT = hp.ROOT
RESULTS_DIR = ROOT / "results"

# Metrics we surface (from MLflow) in the ranked summary, if present. Keys are
# the sanitized bernn/MLflow names (see results/*_trials.csv for the full 160+).
# The batch-effect suite is reported on the encoded space vs domains (batches) —
# lower silhouette / ARI on domains == better batch mixing (BERNN paper signal).
_SUMMARY_METRIC_KEYS = [
    "mcc_valid_all_concentrations", "mcc_test_all_concentrations",
    "acc_valid_all_concentrations", "acc_test_all_concentrations",
    "top3_valid_all_concentrations",
    "batch_entropy_valid_enc_domains", "silhouette_valid_enc_domains",
    "adjusted_rand_score_valid_enc_domains", "lisi_valid_enc_domains",
    "kbet_valid_enc_domains",
]


def read_predictions(csv_path: Path):
    """Return (y_true, y_pred) from a bernn *_predictions.csv.

    Layout (verified): col '0' = true label, cols '1'..'N' = class probabilities,
    col '<N+1>' = predicted label (argmax), last col = sample id.
    """
    df = pd.read_csv(csv_path, index_col=0)
    cols = list(df.columns)
    y_true = df[cols[0]].to_numpy().astype(int)
    # predicted label = the integer column just before the string id column
    pred_col = cols[-2]
    y_pred = df[pred_col].to_numpy().astype(int)
    return y_true, y_pred


def classification_report(csv_path: Path, split: str) -> dict:
    """acc / balanced_acc / macro precision-recall-specificity-f1 for one split."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, precision_recall_fscore_support)
    if not csv_path.exists():
        return {}
    y_true, y_pred = read_predictions(csv_path)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    # macro specificity: per class, TN / (TN + FP)
    specs = []
    total = cm.sum()
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    return {
        f"{split}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{split}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{split}_precision_macro": float(prec),
        f"{split}_sensitivity_macro": float(rec),      # recall == sensitivity
        f"{split}_specificity_macro": float(np.nanmean(specs)),
        f"{split}_f1_macro": float(f1),
    }


def make_family_args(base, preset: str, model_type: str, kan: bool = False) -> argparse.Namespace:
    """Clone the CLI args for one family/trainer type and set output prefix."""
    fam = hp.PRESETS[preset]
    a = argparse.Namespace(**vars(base))
    a.preset = preset
    a.model_type = model_type
    a.dloss = fam["dloss"]
    a.variational = fam["variational"]
    a.kan = bool(kan)
    a.class_triplet = int(fam.get("class_triplet", False))  # new in 0.6.3
    a.out_prefix = f"bernn_hpsearch_{'kan_' if kan else ''}{model_type}_{preset}"
    a.results_dir = str(getattr(base, "results_dir", RESULTS_DIR))
    return a


def run_family(preset: str, model_type: str, kan: bool, base_args, data) -> dict:
    """Search + retrain-winner + detailed report for one family/trainer type."""
    import mlflow

    args = make_family_args(base_args, preset, model_type, kan=kan)
    family_name = f"{'kan_' if kan else ''}{model_type}_{preset}"
    print(
        "\n" + "#" * 72
        + f"\n# FAMILY: {family_name}  (trainer={model_type} dloss={args.dloss} vae={args.variational})\n"
        + "#" * 72
    )

    study = hp.run_study(args, data)
    best = hp.persist_study(study, args)
    cfg = hp.cfg_from_best(best)

    # Re-train the winner so its predictions survive, then compute the metrics
    # bernn never logs (sensitivity/specificity/precision).
    detail = {}
    try:
        retrain_exp = f"{args.out_prefix}_{args.dataset}_BESTretrain"
        _, retrain_metrics, bm_dir = hp.retrain_best(cfg, args, data, retrain_exp)
        for split in ("valid", "test", "train"):
            detail.update(classification_report(bm_dir / f"{split}_predictions.csv", split))
    except Exception as exc:
        print(f"[{family_name}] winner retrain / report failed: {type(exc).__name__}: {exc}")

    row = {
        "preset": family_name,
        "base_preset": preset,
        "model_type": model_type,
        "dloss": args.dloss,
        "variational": args.variational,
        # Metric contract shared with head_sweep: valid_mcc is the objective.
        "valid_mcc": best["_valid_mcc"],
        "valid_mcc_objective": best["_valid_mcc"],
        "trial_valid_mcc": best.get("metric_valid_mcc", best["_valid_mcc"]),
        "best_trial": best.get("_trial_number"),
        "n_trials": args.n_trials,
    }
    # MLflow-captured metrics (mean over CV repeats) for the winning trial.
    # Keep every numeric metric using its sanitized name, not just the summary list.
    for k, v in best.items():
        if not k.startswith("metric_"):
            continue
        name = k[len("metric_"):]
        row[name] = v
    for k in _SUMMARY_METRIC_KEYS:
        row[k] = best.get(f"metric_{k}", row.get(k, np.nan))
    row["train_mcc"] = row.get("mcc_train_all_concentrations", row.get("mcc_train_all", np.nan))
    row["test_mcc"] = row.get("mcc_test_all_concentrations", row.get("mcc_test_all", np.nan))
    row["train_accuracy"] = row.get("acc_train_all_concentrations", row.get("acc_train_all", np.nan))
    row["test_accuracy"] = row.get("acc_test_all_concentrations", row.get("acc_test_all", np.nan))
    row["batch_entropy"] = row.get("batch_entropy_valid_enc_domains", row.get("batch_entropy", np.nan))
    row["batch_silhouette"] = row.get("silhouette_valid_enc_domains", row.get("batch_silhouette", np.nan))
    row["batch_lisi"] = row.get("lisi_valid_enc_domains", row.get("batch_lisi", np.nan))
    row["batch_kbet"] = row.get("kbet_valid_enc_domains", row.get("batch_kbet", np.nan))
    row["batch_ari"] = row.get("adjusted_rand_score_valid_enc_domains", row.get("batch_ari", np.nan))
    row["batch_ami"] = row.get("adjusted_mutual_info_score_valid_enc_domains", row.get("batch_ami", np.nan))
    row.update(detail)
    # keep the tuned config for registration
    row["_config"] = cfg
    return row


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-family BERNN sweep + metric capture.")
    p.add_argument("--dataset", default="massbench_benchmark")
    p.add_argument("--path", default=None,
                   help="Directory containing --csv-file for exact BERNN custom CSV sweeps.")
    p.add_argument("--csv-file", dest="csv_file", default=None,
                   help="Custom BERNN CSV such as intensities.csv.")
    p.add_argument("--combine-test", action="store_true")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--n-repeats", type=int, default=-1,
                   help="CV folds for the BERNN wrapper. Use -1 for leave-one-batch-out.")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--max-warmup", type=int, default=50)
    p.add_argument("--device", default=hp._default_device(), choices=["cpu", "cuda"],
                   help="Compute device (default: cuda if a GPU is available, else cpu).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rec-loss", dest="rec_loss", default="l1", choices=["l1", "mse"],
                   help="AE reconstruction loss used by BERNN training (default: l1).")
    p.add_argument("--class-triplet-w", dest="class_triplet_w", type=float, default=1.0,
                   help="Weight for supervised class-triplet loss when enabled.")
    p.add_argument("--presets", nargs="*", default=list(hp.PRESETS),
                   choices=list(hp.PRESETS),
                   help="Families to sweep (default: all, including class_triplet variants).")
    p.add_argument("--trainer-types", default="joint,two_stage",
                   help="Comma-separated BERNN trainer types to run: joint,two_stage.")
    p.add_argument("--kan-values", nargs="+", choices=["false", "true"], default=["false", "true"],
                   help="Architecture variants to run (default: both MLP and KAN).")
    p.add_argument("--no-class-triplet", action="store_true",
                   help="Skip class_triplet=True preset variants.")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable Weights & Biases logging (MLflow stays on).")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                   help="Dataset-specific output directory.")
    args = p.parse_args(argv)
    return args


def main(argv=None):
    import mlflow

    global RESULTS_DIR
    args = parse_args(argv)
    RESULTS_DIR = Path(args.results_dir).resolve()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hp.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(hp.MLRUNS_DIR.as_uri())

    print(f"Loading dataset '{args.dataset}' ...")
    X, y, batches = hp.load_dataset(
        args.dataset,
        combine_test=args.combine_test,
        path=args.path,
        csv_file=args.csv_file,
    )
    print(f"  X={X.shape}  classes={sorted(set(y))}  batches={sorted(set(batches))}")
    args.resolved_n_repeats = hp.resolve_n_repeats(args.n_repeats, batches)
    if int(args.n_repeats) == -1:
        print(f"  n_repeats=-1 resolved to leave-one-batch-out CV: {args.resolved_n_repeats} folds")
    else:
        print(f"  n_repeats={args.n_repeats} resolved to {args.resolved_n_repeats} CV folds")
    data = (X, y, batches)

    # Filter presets: when class_triplet is disabled, skip CT variants
    active_presets = args.presets
    if hasattr(args, 'no_class_triplet') and args.no_class_triplet:
        active_presets = [p for p in args.presets if not hp.PRESETS[p].get('class_triplet', False)]
        print(f'class_triplet disabled: running {len(active_presets)} presets (no CT variants)')
    else:
        print(f'class_triplet enabled: running all {len(active_presets)} presets')
    trainer_types = [x.strip() for x in str(args.trainer_types).replace(";", ",").split(",") if x.strip()]
    bad = [x for x in trainer_types if x not in {"joint", "two_stage"}]
    if bad:
        raise ValueError(f"Unsupported trainer type(s): {bad}. Use joint,two_stage.")
    kan_values = [value == "true" for value in args.kan_values]
    rows = [
        run_family(preset, trainer, kan, args, data)
        for trainer in trainer_types
        for kan in kan_values
        for preset in active_presets
    ]

    # --- ranked summary --------------------------------------------------
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "_config"} for r in rows])
    summary = summary.sort_values("valid_mcc_objective", ascending=False).reset_index(drop=True)
    summary_csv = RESULTS_DIR / "sweep_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # --- per-family tuned defaults (for register_bernn_defaults.py) ------
    tuned = {r["preset"]: {**r["_config"], "_valid_mcc": r["valid_mcc"]} for r in rows}
    tuned_json = RESULTS_DIR / "bernn_tuned_defaults.json"
    tuned_json.write_text(json.dumps(tuned, indent=2))

    print("\n" + "=" * 72)
    print("SWEEP COMPLETE — families ranked by validation MCC:")
    cols = ["preset", "model_type", "valid_mcc", "test_mcc", "train_mcc", "valid_sensitivity_macro",
            "valid_specificity_macro", "valid_precision_macro", "silhouette_valid_enc_domains"]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    print(f"\nSummary CSV   : {summary_csv}")
    print(f"Tuned defaults: {tuned_json}")
    print("Register with : python register_bernn_defaults.py")
    print("=" * 72)
    return summary


if __name__ == "__main__":
    main()
