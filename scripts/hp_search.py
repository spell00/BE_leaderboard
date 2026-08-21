"""Hyperparameter search for BERNN on the batch-effects leaderboard datasets.

Searches the full BERNN hyperparameter space (model-selection choices + the
continuous/int knobs from the paper) with Optuna, using an explicit outer-fold holdout MCC as the objective. The best configuration it finds can be dropped
straight into the leaderboard's parameterized BERNN baseline
(``src.baselines.build_bernn_code``).

BERNN paper : https://www.nature.com/articles/s41467-024-48177-5
BERNN repo  : https://github.com/spell00/BERNN_MSMS
This project: BE_leaderboard (real-track model selection, src/baselines.py)

Why Optuna (not bernn's built-in Ax): bernn optimizes with Ax, which requires
sqlalchemy < 2.0, but this project pins sqlalchemy >= 2.0 for its database. So we
run our own TPE search here and mirror the paper's search ranges exactly.

Metric capture (the whole point of the rewrite)
-----------------------------------------------
The Optuna *objective* is the held-out MCC from trainer.predict(X_valid), but that single
scalar is no longer all we keep. Each trial trains with ``log_mlflow=True`` into
a per-trial MLflow experiment, and after ``fit``/``predict`` we read every metric
bernn logged back out of MLflow (averaged over the CV repeats) and attach it to
the trial. So the trials CSV now carries, for train/valid/test:

  * classification : acc, mcc, top3, closs  (bernn's ``add_to_mlflow`` keys)
  * batch effect   : batch_entropy, silhouette, lisi, kbet, adjusted_rand_score,
                     adjusted_mutual_info_score  -- for the encoded (enc) and
                     reconstructed (rec) representations, i.e. the "normalized
                     batch effect" family from the BERNN paper
  * domain         : dom_acc, dom_loss, rec_loss

Sensitivity / specificity / precision are NOT logged by bernn; they are computed
per-family for the *winning* config from the retained best-model prediction CSVs
by ``hp_search_sweep.py``.

The search space (ranges taken verbatim from
bernn/dl/train/train_ae_classifier_holdout.py):

  model-selection (categorical)
    dloss       : no | DANN | revDANN | inverseTriplet | normae  (revTriplet re-enabled: fixed in 0.6.3)
    variational : {False, True}   (AE vs VAE)
    kan         : {False, True}    (MLP vs KAN)
  continuous / int
    lr          : [1e-4, 1e-2]  (log)
    wd          : [1e-6, 1e-3]  (log)
    nu          : [1e-4, 1e2]
    smoothing   : [0.0, 0.2]
    margin      : [0.0, 10.0]
    dropout     : [0.0, 0.5]
    thres       : [0.0, 0.1]
    warmup      : [1, max_warmup]
    n_layers    : {1, 2, 3, 4, 5}
    layer1      : [512, 1024]
    scaler      : standard | robust | standard_per_batch | robust_per_batch
    log1p       : {False, True}
  conditional
    gamma       : [1e-2, 1e2] (log)  -- only when dloss is adversarial
    beta        : [1e-2, 1e2] (log)  -- only when variational (VAE)

NOTE: only the joint trainer (TrainAEClassifierHoldout) is searched.
TrainAEThenClassifierHoldout (two_stage) was broken in bernn 0.5.8; fixed in 0.6.3
(undefined 'h' at train_ae_then_classifier_holdout.py:904).

This is a heavy, offline script (each trial trains a full model). Run it on a
machine with a GPU when possible (``--device cuda``); it is NOT meant to run
inside the Hugging Face Space.

Usage
-----
    python hp_search.py --dataset massbench_benchmark --n-trials 100 \
        --n-epochs 200 --device cuda

    # restrict the model family (e.g. only VAE + inverseTriplet)
    python hp_search.py --dloss inverseTriplet --variational true

    # run one named preset (fixes dloss + variational) into results/<preset>/
    python hp_search.py --preset ae_dann --n-trials 100

Outputs (written under --results-dir, prefixed with --out-prefix)
    <prefix>_trials.csv   every trial + sampled params + ALL captured metrics
    <prefix>_best.json    the best config found + its captured metrics
    prints a ready-to-use CONFIG and the generated fit_predict code.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Quiet the heavy TF/CUDA/Ax import noise before bernn is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
logging.getLogger("ax").setLevel(logging.ERROR)

DATASETS_DIR = ROOT / "data" / "datasets"
# MLflow tracking store (DVC-tracked). Every trial's metrics land here.
MLRUNS_DIR = ROOT / "mlruns"


def _default_device():
    """'cuda' when a GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# Non-searched columns in the dataset CSVs (name,batch,label,<features...>).
_META_COLS = ("name", "names", "batch", "batches", "label", "labels", "group")

ALZHEIMER_DATASET = "massbench_alzheimer"
ALZHEIMER_SUPERVISED_LABELS = frozenset({"CU", "DEM-AD"})
POOL_LABEL = "pool"

# revTriplet re-enabled: was broken in bernn 0.5.8 but fixed in 0.6.3.
ADVERSARIAL_DLOSS = {"DANN", "revDANN", "inverseTriplet", "normae", "revTriplet"}
DLOSS_CHOICES = ["no", "DANN", "revDANN", "inverseTriplet", "normae", "revTriplet"]
SCALER_CHOICES = ["standard", "robust", "standard_per_batch", "robust_per_batch"]

# The 6 BERNN model families the leaderboard exposes (mirror src.baselines.BERNN_PRESETS).
# Each fixes (dloss, variational); the search tunes the continuous knobs within it.
# Classic 6 families + class_triplet variants (new in 0.6.3).
# class_triplet adds a supervised triplet margin loss on class-labelled
# embeddings ON TOP OF dloss during AE encoder training.
PRESETS = {
    # ── class_triplet OFF (classic behaviour) ──
    "ae_inversetriplet":          {"dloss": "inverseTriplet", "variational": False, "class_triplet": False},
    "vae_inversetriplet":         {"dloss": "inverseTriplet", "variational": True,  "class_triplet": False},
    "ae_dann":                    {"dloss": "DANN",           "variational": False, "class_triplet": False},
    "vae_dann":                   {"dloss": "DANN",           "variational": True,  "class_triplet": False},
    "ae_normae":                  {"dloss": "normae",         "variational": False, "class_triplet": False},
    "ae_no_correction":           {"dloss": "no",             "variational": False, "class_triplet": False},
    # ── class_triplet ON (new in 0.6.3) ──
    "ae_inversetriplet_ct":       {"dloss": "inverseTriplet", "variational": False, "class_triplet": True},
    "vae_inversetriplet_ct":      {"dloss": "inverseTriplet", "variational": True,  "class_triplet": True},
    "ae_dann_ct":                 {"dloss": "DANN",           "variational": False, "class_triplet": True},
    "vae_dann_ct":                {"dloss": "DANN",           "variational": True,  "class_triplet": True},
    "ae_revtriplet":              {"dloss": "revTriplet",     "variational": False, "class_triplet": False},
    "ae_revtriplet_ct":           {"dloss": "revTriplet",     "variational": False, "class_triplet": True},
}


def load_dataset(name: str, combine_test: bool = False, path: str | None = None, csv_file: str | None = None):
    """Load features / labels / batches from ``data/datasets/<name>/``.

    Returns (X: DataFrame[float], y: np.ndarray[str], batches: np.ndarray[str]).
    Uses the train split by default; bernn splits it internally into
    train/valid/test for cross-validated model selection.
    """
    if csv_file:
        csv_path = Path(path or ".") / csv_file
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"No custom BERNN CSV at {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        label_col = next((c for c in ("labels", "label", "group") if c in df.columns), None)
        if label_col is None:
            raise ValueError(f"Custom BERNN CSV must contain labels/label/group; found {list(df.columns[:8])}")
        batch_col = next((c for c in ("batches", "batch") if c in df.columns), None)
        labelled = df[label_col].notna() & df[label_col].astype("string").str.strip().ne("")
        ignored = int((~labelled).sum())
        if ignored:
            print(f"[data] Ignoring {ignored} unlabeled row(s) from {csv_path}")
            df = df.loc[labelled].copy()
        feature_cols = [c for c in df.columns if c not in _META_COLS]
        X = df[feature_cols].astype(float).reset_index(drop=True)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = df[label_col].astype(str).to_numpy()
        batches = (
            df[batch_col].astype(str).to_numpy()
            if batch_col is not None
            else np.array(["batch0"] * len(df), dtype=str)
        )
        return X, y, batches

    base = DATASETS_DIR / name
    train_csv = base / f"{name}_train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"No train CSV for dataset '{name}' at {train_csv}")
    df = pd.read_csv(train_csv)
    if combine_test:
        test_csv = base / f"{name}_test.csv"
        if test_csv.exists():
            df = pd.concat([df, pd.read_csv(test_csv)], ignore_index=True)

    if name == ALZHEIMER_DATASET:
        # Alzheimer is a binary supervised task (DEM-AD vs CU). Every other
        # diagnosis and missing label remains available to BERNN's
        # reconstruction/domain losses as an unlabeled pooled sample.
        labels = df["label"].astype("string").str.strip()
        supervised = labels.isin(ALZHEIMER_SUPERVISED_LABELS)
        df = df.copy()
        df["label"] = labels.where(supervised, POOL_LABEL)
        print(
            f"[data] Alzheimer semi-supervised task: {int(supervised.sum())} "
            f"DEM-AD/CU rows + {int((~supervised).sum())} pooled rows"
        )
    else:
        labelled = df["label"].notna() & df["label"].astype("string").str.strip().ne("")
        ignored = int((~labelled).sum())
        if ignored:
            print(f"[data] Ignoring {ignored} unlabeled row(s) from dataset '{name}'")
            df = df.loc[labelled].copy()

    feature_cols = [c for c in df.columns if c not in _META_COLS]
    X = df[feature_cols].astype(float).reset_index(drop=True)
    # Match the real pipeline (code_challenge.py): missing / non-finite -> 0.
    # MassBench intensities use NaN for "not detected"; leaving them in makes bernn NaN out.
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["label"].astype(str).to_numpy()
    batches = df["batch"].astype(str).to_numpy()
    return X, y, batches


def load_fixed_test_dataset(name: str):
    """Load labeled fixed cross-test samples for monitoring, never selection.

    Public ``*_test.csv`` files omit labels. Matching local
    ``*_inference.csv`` files contain the same samples plus labels. Their spectra
    and batch IDs participate transductively in AE/domain training; labels are
    used only for test monitoring and external scoring. Alzheimer monitoring is
    restricted to the declared binary task (DEM-AD versus CU).
    """
    csv_path = DATASETS_DIR / name / f"{name}_inference.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No labeled fixed-test CSV for '{name}' at {csv_path}")
    df = pd.read_csv(csv_path)
    labelled = df["label"].notna() & df["label"].astype("string").str.strip().ne("")
    if name == ALZHEIMER_DATASET:
        labelled &= df["label"].astype("string").str.strip().isin(ALZHEIMER_SUPERVISED_LABELS)
    df = df.loc[labelled].copy()
    feature_cols = [column for column in df.columns if column not in _META_COLS]
    X = df[feature_cols].astype(float).reset_index(drop=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, df["label"].astype(str).to_numpy(), df["batch"].astype(str).to_numpy()


def build_trainer_config(cfg: dict, args, exp_id: str):
    """Construct a TrainingConfig for one trial and inject all tunable knobs."""
    from bernn.config.training_config import TrainingConfig

    tc = TrainingConfig(
        optimize_hyperparams=False,   # we drive the search ourselves
        dloss=cfg["dloss"],
        variational=cfg["variational"],
        kan=cfg["kan"],
        n_layers=int(cfg["n_layers"]),
        layer1=int(cfg["layer1"]),
        rec_loss=getattr(args, "rec_loss", "l1"),
        use_l1=(getattr(args, "rec_loss", "l1") == "l1"),
        prune_network=False,          # keep pruning off during search (stability/speed)
        tied_weights=False,
        use_mapping=True,
        update_grid=bool(cfg["kan"]),  # KAN-only; enabling with MLP crashes bernn
        groupkfold=True,
        n_epochs=int(args.n_epochs),
        # BERNN itself uses n_repeats as split count/retry state; hp_search adds
        # an outer CV wrapper and passes the resolved fold count through.
        n_repeats=int(getattr(args, "resolved_n_repeats", args.n_repeats)),
        bs=int(args.bs),
        device=args.device,
        exp_id=exp_id,                 # MLflow experiment bernn logs this trial into
    )
    # bernn reads self.args.dataset for its log dir; TrainingConfig lacks the field.
    tc.dataset = args.dataset
    # class_triplet: supervised triplet loss on class-labelled embeddings (0.6.3+)
    tc.class_triplet = bool(cfg.get("class_triplet", False))
    tc.class_triplet_w = float(cfg.get("class_triplet_w", getattr(args, "class_triplet_w", 1.0)))
    # Inject the searched hyperparameters as attributes (read via getattr in _train).
    tc.lr = float(cfg["lr"])
    tc.wd = float(cfg["wd"])
    tc.nu = float(cfg["nu"])
    tc.smoothing = float(cfg["smoothing"])
    tc.margin = float(cfg["margin"])
    tc.dropout = float(cfg["dropout"])
    tc.thres = float(cfg["thres"])
    tc.warmup = int(cfg["warmup"])
    tc.log1p = bool(cfg["log1p"])
    tc.scaler = cfg["scaler"]
    tc.gamma = float(cfg["gamma"])   # forced to 0 internally when dloss is non-adversarial
    tc.beta = float(cfg["beta"])     # forced to 0 internally when not variational
    return tc


def sample_config(trial, args) -> dict:
    """Sample one full BERNN configuration from the paper's search space."""
    dloss = args.dloss or trial.suggest_categorical("dloss", DLOSS_CHOICES)
    if args.variational is None:
        variational = trial.suggest_categorical("variational", [False, True])
    else:
        variational = args.variational
    kan = trial.suggest_categorical("kan", [False, True]) if args.kan is None else args.kan
    class_triplet = trial.suggest_categorical("class_triplet", [False, True]) if args.class_triplet is None else args.class_triplet
    log1p = trial.suggest_categorical("log1p", [False, True]) if args.log1p is None else args.log1p
    if args.class_triplet is None:
        class_triplet_w = trial.suggest_float("class_triplet_w", 0.0, 1.0, log=False)

    cfg = {
        "model_type": getattr(args, "model_type", "joint"),
        "dloss": dloss,
        "variational": bool(variational),
        "kan": bool(kan),
        "class_triplet": bool(class_triplet),
        "class_triplet_w": float(class_triplet_w),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "nu": trial.suggest_float("nu", 1e-4, 1e2),
        "smoothing": trial.suggest_float("smoothing", 0.0, 0.2),
        "margin": trial.suggest_float("margin", 0.0, 10.0),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "thres": trial.suggest_float("thres", 0.0, 0.1),
        "warmup": trial.suggest_int("warmup", 1, args.max_warmup),
        "n_layers": trial.suggest_categorical("n_layers", [1, 2, 3, 4, 5]),
        "layer1": trial.suggest_int("layer1", 512, 1024),
        "log1p": bool(log1p),
        "scaler": trial.suggest_categorical("scaler", SCALER_CHOICES),
        # conditional — only meaningful in their regime (bernn zeroes them otherwise)
        "gamma": trial.suggest_float("gamma", 1e-2, 1e2, log=True) if dloss in ADVERSARIAL_DLOSS else 0.0,
        "beta": trial.suggest_float("beta", 1e-2, 1e2, log=True) if variational else 0.0,
    }
    return cfg


def _sanitize(key: str) -> str:
    """MLflow metric key -> CSV/user_attr-friendly slug (mcc/valid/all -> mcc_valid_all)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", key.strip()).strip("_")


def extract_mlflow_metrics(exp_id: str) -> dict:
    """Average every metric bernn logged for this trial across its CV-repeat runs.

    bernn calls ``mlflow.start_run()`` once per repeat inside ``exp_id`` and logs
    the final value of each metric to that run. We take each run's *final* value
    (``run.data.metrics``) and average over runs, so the returned dict is the
    per-trial mean of acc/mcc/top3/losses/dom_acc plus the batch-effect suite
    (batch_entropy/silhouette/lisi/kbet/ARI/AMI for enc & rec).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_id)
    if exp is None:
        return {}
    runs = client.search_runs([exp.experiment_id], max_results=1000)
    if not runs:
        return {}
    acc: dict[str, list] = {}
    for run in runs:
        for k, v in run.data.metrics.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            acc.setdefault(k, []).append(float(v))
    return {_sanitize(k): float(np.mean(vs)) for k, vs in acc.items() if vs}


def extract_mlflow_epoch_traces(exp_id: str) -> list[dict]:
    """Read BERNN's per-epoch accuracy/MCC histories for one CV fold."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(exp_id)
    if experiment is None:
        return []
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    aliases = {
        "train/mcc": "train_mcc", "valid/mcc": "valid_mcc",
        "test/mcc": "test_mcc", "train/acc": "train_accuracy",
        "valid/acc": "valid_accuracy", "test/acc": "test_accuracy",
        "acc/train/all_concentrations": "train_accuracy",
        "acc/valid/all_concentrations": "valid_accuracy",
        "acc/test/all_concentrations": "test_accuracy",
        "mcc/train/all_concentrations": "train_mcc",
        "mcc/valid/all_concentrations": "valid_mcc",
        "mcc/test/all_concentrations": "test_mcc",
        "train/balanced_accuracy": "train_balanced_accuracy",
        "valid/balanced_accuracy": "valid_balanced_accuracy",
        "test/balanced_accuracy": "test_balanced_accuracy",
        "train/f1_macro": "train_f1_macro",
        "valid/f1_macro": "valid_f1_macro",
        "test/f1_macro": "test_f1_macro",
        "train/closs": "train_classification_loss",
        "valid/closs": "valid_classification_loss",
    }
    by_epoch: dict[int, dict] = {}
    for run_index, run in enumerate(runs):
        for raw_name, alias in aliases.items():
            for metric in client.get_metric_history(run.info.run_id, raw_name):
                epoch = int(metric.step)
                row = by_epoch.setdefault(epoch, {"epoch": epoch})
                key = alias if len(runs) == 1 else f"rep{run_index}/{alias}"
                row[key] = float(metric.value)
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def bernn_params_from_cfg(cfg: dict) -> dict:
    """BERNN-style params dict, matching the raw module/Ax call shape."""
    return {
        "n_layers": int(cfg["n_layers"]),
        "nu": float(cfg["nu"]),
        "lr": float(cfg["lr"]),
        "wd": float(cfg["wd"]),
        "smoothing": float(cfg["smoothing"]),
        "margin": float(cfg["margin"]),
        "dropout": float(cfg["dropout"]),
        "scaler": cfg["scaler"],
        "layer1": int(cfg["layer1"]),
        "gamma": float(cfg.get("gamma", 0.0)),
        "l1": float(cfg.get("l1", 0.0)),
        "prune_threshold": float(cfg.get("prune_threshold", 0.0)),
        "warmup": int(cfg["warmup"]),
        "beta": float(cfg.get("beta", 0.0)),
        "thres": float(cfg.get("thres", 0.0)),
        "reg_entropy": float(cfg.get("reg_entropy", 0.0)),
    }


def _apply_log1p_preprocessing(frame):
    """Apply the requested log1p preprocessing directly to feature matrices."""
    if frame is None:
        return None
    values = np.asarray(frame, dtype=float)
    values = np.clip(values, 0.0, None)
    transformed = np.log1p(values)
    if isinstance(frame, pd.DataFrame):
        return pd.DataFrame(transformed, index=frame.index, columns=frame.columns)
    return transformed


def _close_fit_resources(trainer) -> None:
    """Release resources owned by one BERNN fold, including failed fits."""
    close = getattr(trainer, "close_resources", None)
    if callable(close):
        close()

    try:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass

    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _fit_one(cfg, args, data, exp_id: str, seed: int, keep_models: bool):
    """One stable BERNN fit. Returns (trainer, heldout_mcc_or_monitor_mcc).

    BERNN fit is sklearn-style: train on X/y only. If data contains an external
    validation tuple, score it with trainer.predict(X_valid) here.
    """
    from sklearn.metrics import matthews_corrcoef
    from bernn import TrainAEClassifierHoldout, TrainAEThenClassifierHoldout
    from src.baselines import set_bernn_seed

    X_valid = y_valid = batches_valid = None
    X_test = y_test = batches_test = None
    if len(data) == 3:
        X, y, batches = data
    elif len(data) == 6:
        X, y, batches, X_valid, y_valid, batches_valid = data
    elif len(data) == 9:
        X, y, batches, X_valid, y_valid, batches_valid, X_test, y_test, batches_test = data
    else:
        raise ValueError(f"Expected 3, 6, or 9 data items, got {len(data)}")
    tc = build_trainer_config(cfg, args, exp_id)
    set_bernn_seed(seed)       # reproducibility: same config/seed -> same result
    trainer_cls = TrainAEThenClassifierHoldout if cfg.get("model_type") == "two_stage" else TrainAEClassifierHoldout
    trainer = trainer_cls(
        config=tc, log_metrics=True, keep_models=keep_models, log_mlflow=True,
    )
    trainer.seed = int(seed)   # vary stochastic BERNN initialization across folds
    import inspect

    fit_sig = inspect.signature(trainer.fit)
    legacy_holdout_args = {"X_test", "y_test", "batches_test"}
    positional_holdout_args = {
        name for name, param in fit_sig.parameters.items()
        if name in legacy_holdout_args
        and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    }
    if positional_holdout_args:
        raise RuntimeError(
            "Installed BERNN still uses legacy positional fit(..., X_test=...). "
            "Install the sklearn-style BERNN patch/release before running hparam search."
        )
    fit_kwargs = {
        "groups_train": batches.copy(),
        "params": bernn_params_from_cfg(cfg),
        "cross_validation": False,
        "cross_test": False,
    }
    # The outer CV fold is the real supervised validation set. Passing it into
    # BERNN prevents its sklearn-style fallback from cloning training rows into
    # the valid/test monitor loaders. This makes epoch logs and early stopping
    # genuinely held-out while still allowing unsupervised/transductive use of
    # validation features through BERNN's combined `all` loader.
    if X_valid is not None and y_valid is not None:
        fit_kwargs.update({
            "X_valid": X_valid.copy(),
            "y_valid": np.asarray(y_valid).copy(),
            "groups_valid": np.asarray(batches_valid).copy(),
        })
        if X_test is not None:
            fit_kwargs.update({
                "X_test": X_test.copy(),
                "groups_test": np.asarray(batches_test).copy(),
            })
            if y_test is not None:
                # Local fixed-test HPO experiments have labels; pass them so BERNN
                # can log test MCC/accuracy each epoch. The submission runner still
                # keeps hidden benchmark labels out of user fit() calls.
                fit_kwargs["y_test"] = np.asarray(y_test).copy()
        else:
            # BERNN currently requires a non-empty test loader. In pure CV mode,
            # reuse the validation features without labels, so test metrics remain
            # unavailable but training still has a held-out validation monitor.
            fit_kwargs.update({
                "X_test": X_valid.copy(),
                "groups_test": np.asarray(batches_valid).copy(),
            })
    try:
        trainer.fit(X_train=X.copy(), y_train=y.copy(), **fit_kwargs)
        if X_valid is not None and y_valid is not None:
            try:
                preds = trainer.predict(X_valid.copy(), groups_test=np.asarray(batches_valid).copy())
            except TypeError:
                preds = trainer.predict(X_valid.copy())
            y_valid_array = np.asarray(y_valid).astype(str)
            preds_array = np.asarray(preds).astype(str)
            labeled = y_valid_array != "-1"
            if not np.any(labeled):
                raise ValueError("Validation fold contains no labeled samples")
            supervised_mcc = float(matthews_corrcoef(
                y_valid_array[labeled], preds_array[labeled],
            ))
            if np.any(~labeled):
                print(
                    "[hp-search] restored-prediction SUPERVISED valid MCC "
                    f"= {supervised_mcc:.4f} "
                    f"({int(labeled.sum())}/{len(labeled)} labeled rows; "
                    "pooled '-1' rows excluded from the objective)",
                    flush=True,
                )
            return trainer, supervised_mcc
        return trainer, float(getattr(trainer, "best_mcc", -1.0))
    finally:
        _close_fit_resources(trainer)


def resolve_n_repeats(n_repeats: int, batches) -> int:
    """Resolve -1 to grouped CV capped at five folds."""
    requested = int(n_repeats)
    if requested == -1:
        unique_batches = np.unique(np.asarray(batches, dtype=str))
        if len(unique_batches) < 2:
            raise ValueError(
                "n_repeats=-1 requires at least 2 unique batch IDs; "
                f"found {len(unique_batches)}."
            )
        return int(min(5, len(unique_batches)))
    if requested < 2:
        raise ValueError("--n-repeats must be -1 or >= 2 for BERNN CV split handling.")
    return requested


def cv_splits(y, batches, n_repeats: int):
    """Yield grouped train/test indices for the requested BERNN CV wrapper."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = np.asarray(y)
    batches = np.asarray(batches)
    if int(n_repeats) == -1:
        resolved_n_repeats = resolve_n_repeats(n_repeats, batches)
        splitter = StratifiedGroupKFold(n_splits=resolved_n_repeats, shuffle=True, random_state=0)
        yield from splitter.split(np.zeros(len(y)), y, batches)
        return
    if len(np.unique(batches)) > 1:
        splitter = StratifiedGroupKFold(n_splits=n_repeats, shuffle=True, random_state=0)
        yield from splitter.split(np.zeros(len(y)), y, batches)
    else:
        splitter = StratifiedKFold(n_splits=n_repeats, shuffle=True, random_state=0)
        yield from splitter.split(np.zeros(len(y)), y)


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Mean metric dictionaries and retain per-fold vectors for dashboards.

    Scalar means/stds stay as summary metrics. ``*_folds`` lists are consumed by
    the W&B compact fold chart logger, so CV detail stays in one chart instead
    of becoming separate scalar panels.
    """
    keys = {k for m in fold_metrics for k in m if not k.startswith("_")}
    out: dict[str, float] = {}
    for k in keys:
        if k == "fold":
            continue
        vals = []
        for m in fold_metrics:
            if k not in m:
                continue
            try:
                val = float(m[k])
            except (TypeError, ValueError):
                continue
            if not math.isnan(val):
                vals.append(val)
        if not vals:
            continue
        out[k] = float(np.mean(vals))
        if len(vals) > 1:
            out[f"{k}_std"] = float(np.std(vals))
            out[f"{k}_folds"] = vals
    return out


def run_trial(cfg: dict, args, data, exp_id: str, fixed_test_data=None):
    """Run one BERNN train/eval trial.

    One sampled config is evaluated over the resolved CV folds. ``n_repeats=-1``
    uses stratified grouped CV with at most five folds, so technical batches
    never cross train/validation boundaries.
    """
    X, y, batches = data
    fold_scores = []
    fold_metrics = []
    epoch_traces_by_fold = {}
    fixed_test_predictions = []
    resolved_n_repeats = int(getattr(args, "resolved_n_repeats", resolve_n_repeats(args.n_repeats, batches)))
    is_alzheimer = getattr(args, "dataset", "") == ALZHEIMER_DATASET
    X_fixed = y_fixed = batches_fixed = None
    if bool(cfg.get("log1p", False)):
        X = _apply_log1p_preprocessing(X)

    if fixed_test_data is not None:
        X_fixed, y_fixed, batches_fixed = fixed_test_data
        if not (len(X_fixed) == len(y_fixed) == len(batches_fixed)):
            raise ValueError("Fixed-test features, labels, and batches must have equal length")
        if bool(cfg.get("log1p", False)):
            X_fixed = _apply_log1p_preprocessing(X_fixed)
    if is_alzheimer:
        supervised_indices = np.flatnonzero(np.isin(y.astype(str), list(ALZHEIMER_SUPERVISED_LABELS)))
        pool_indices = np.flatnonzero(~np.isin(y.astype(str), list(ALZHEIMER_SUPERVISED_LABELS)))
        if not len(pool_indices):
            raise ValueError("Alzheimer semi-supervised task requires pooled samples")
        split_iter = (
            (supervised_indices[train_sub], supervised_indices[valid_sub])
            for train_sub, valid_sub in cv_splits(
                y[supervised_indices], batches[supervised_indices], int(args.n_repeats)
            )
        )
    else:
        pool_indices = np.array([], dtype=int)
        split_iter = cv_splits(y, batches, int(args.n_repeats))

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        model_y = np.asarray(y).astype(object).copy()
        if is_alzheimer:
            # Split whole technical batches, not only their labeled rows. Pool
            # diagnoses remain in their batch's train/valid partition with a -1
            # sentinel, so they contribute only to reconstruction/domain losses.
            train_batch_values = set(np.asarray(batches)[train_idx].astype(str))
            valid_batch_values = set(np.asarray(batches)[test_idx].astype(str))
            train_idx = np.flatnonzero(np.isin(np.asarray(batches).astype(str), list(train_batch_values)))
            test_idx = np.flatnonzero(np.isin(np.asarray(batches).astype(str), list(valid_batch_values)))
            model_y[pool_indices] = "-1"

        train_batch_values = set(np.asarray(batches)[train_idx].astype(str))
        valid_batch_values = set(np.asarray(batches)[test_idx].astype(str))
        fixed_batch_values = set() if batches_fixed is None else set(np.asarray(batches_fixed).astype(str))
        if train_batch_values & valid_batch_values:
            raise AssertionError("A batch appears in both train and validation")
        if (train_batch_values | valid_batch_values) & fixed_batch_values:
            raise AssertionError("A development batch also appears in fixed cross-test")
        if set(train_idx) & set(test_idx) or len(train_idx) + len(test_idx) != len(X):
            raise AssertionError("Every development sample must appear exactly once in train or validation")

        fold_exp_id = f"{exp_id}_fold{fold_idx}"
        fold_data = (
            X.iloc[train_idx].reset_index(drop=True),
            model_y[train_idx],
            batches[train_idx],
        )
        fold_args = argparse.Namespace(**vars(args))
        fold_args.n_repeats = resolved_n_repeats
        fold_args.resolved_n_repeats = resolved_n_repeats
        fit_data = (
            fold_data[0],
            fold_data[1],
            fold_data[2],
            X.iloc[test_idx].reset_index(drop=True),
            model_y[test_idx],
            batches[test_idx],
        )
        if fixed_test_data is not None:
            # Fully transductive external cross-test: features, batch IDs, and
            # labels are supplied once. BERNN uses all spectra for AE/domain
            # learning, but classifier gradients remain train-only; test labels
            # are monitoring-only and never drive early stopping/model selection.
            fit_data = fit_data + (
                X_fixed.copy(),
                np.asarray(y_fixed).copy(),
                np.asarray(batches_fixed).copy(),
            )
        print(
            f"[trial split] fold {fold_idx + 1}/{resolved_n_repeats} "
            f"train={len(train_idx)} valid={len(test_idx)} "
            f"cross_test={0 if X_fixed is None else len(X_fixed)} "
            f"train_batches={sorted(train_batch_values)} "
            f"valid_batches={sorted(valid_batch_values)} "
            f"test_batches={sorted(fixed_batch_values)}",
            flush=True,
        )
        trainer, mcc = _fit_one(
            cfg,
            fold_args,
            fit_data,
            fold_exp_id,
            seed=int(getattr(args, "seed", 42)) + fold_idx,
            keep_models=False,
        )
        metrics = extract_mlflow_metrics(fold_exp_id)
        epoch_traces = extract_mlflow_epoch_traces(fold_exp_id)
        if epoch_traces:
            metrics["_epoch_traces"] = epoch_traces
            epoch_traces_by_fold[str(fold_idx)] = epoch_traces
        metrics["valid_mcc"] = float(mcc)
        if fixed_test_data is not None:
            from sklearn.metrics import matthews_corrcoef

            try:
                predictions = trainer.predict(
                    X_fixed.copy(), groups_test=np.asarray(batches_fixed).copy()
                )
            except TypeError:
                predictions = trainer.predict(X_fixed.copy())
            predictions = np.asarray(predictions).astype(str).reshape(-1)
            if len(predictions) != len(y_fixed):
                raise ValueError(
                    f"Fixed-test prediction length mismatch: {len(predictions)} != {len(y_fixed)}"
                )
            fixed_test_predictions.append(predictions)
            metrics["test_mcc"] = float(matthews_corrcoef(
                np.asarray(y_fixed).astype(str), predictions
            ))
        metrics["fold"] = float(fold_idx)
        fold_scores.append(float(mcc))
        fold_metrics.append(metrics)
        held_batches = sorted(set(np.asarray(batches)[test_idx].astype(str)))
        print(
            f"[trial cv] fold {fold_idx + 1}/{resolved_n_repeats} "
            f"valid MCC = {mcc:.4f} held_batches={held_batches}"
        )
    metrics = _aggregate_fold_metrics(fold_metrics)
    if epoch_traces_by_fold:
        metrics["_epoch_traces_by_fold"] = epoch_traces_by_fold
    metrics["valid_mcc"] = float(np.mean(fold_scores))
    metrics["valid_mcc_std"] = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
    if fixed_test_predictions:
        from sklearn.metrics import matthews_corrcoef

        prediction_matrix = np.stack(fixed_test_predictions)
        ensemble = np.asarray([
            max(sorted(set(column)), key=list(column).count)
            for column in prediction_matrix.T
        ])
        y_fixed = np.asarray(fixed_test_data[1]).astype(str)
        metrics["test_mcc_fold_mean"] = float(metrics.get("test_mcc", np.nan))
        metrics["test_mcc"] = float(matthews_corrcoef(y_fixed, ensemble))
    metrics["resolved_n_repeats"] = float(resolved_n_repeats)
    if is_alzheimer:
        metrics["supervised_samples"] = float(len(supervised_indices))
        metrics["pooled_unsupervised_samples"] = float(len(pool_indices))
    return float(metrics["valid_mcc"]), metrics


# Keys that make up a BERNN config (the ones sample_config emits / that a retrain needs).
CONFIG_KEYS = (
    "model_type", "dloss", "variational", "kan", "class_triplet", "class_triplet_w", "lr", "wd", "nu", "smoothing",
    "margin", "dropout", "thres", "warmup", "n_layers", "layer1", "log1p", "scaler", "gamma", "beta",
)


def cfg_from_best(best: dict) -> dict:
    """Pull just the model-config keys out of a persisted _best.json dict."""
    return {k: best[k] for k in CONFIG_KEYS if k in best}


def best_model_dir(cfg: dict, args) -> Path:
    """Where bernn copies the best model + prediction CSVs for this family."""
    return (ROOT / "logs" / "best_models" / "ae_classifier_holdout" / args.dataset /
            f"{cfg['dloss']}_vae{cfg['variational']}")


def retrain_best(cfg: dict, args, data, exp_id: str):
    """Re-train the winning config once (keep_models=True) so its prediction CSVs
    survive for the detailed sensitivity/specificity report.

    Returns (best_valid_mcc, metrics_dict, best_model_dir). A single stable
    single-split fit (not the CV loop): the CV estimate is already captured during
    the search; here we just need one representative model's predictions. During a
    search each trial resets self.best_mcc=-1 (fresh trainer) so bernn's shared
    best_models/<dloss>_vae<var> dir would otherwise hold the *last* trial, not the
    winner — hence this deliberate re-fit.
    """
    trainer, score = _fit_one(cfg, args, data, exp_id, seed=0, keep_models=True)
    return score, extract_mlflow_metrics(exp_id), best_model_dir(cfg, args)


def make_objective(args, data):
    def objective(trial):
        cfg = sample_config(trial, args)
        # store the resolved config on the trial for CSV export
        for k, v in cfg.items():
            trial.set_user_attr(k, v)
        exp_id = f"{args.out_prefix}_{args.dataset}_t{trial.number}"
        trial.set_user_attr("mlflow_exp", exp_id)
        try:
            score, metrics = run_trial(cfg, args, data, exp_id)
        except Exception as exc:  # a bad config shouldn't kill the whole study
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            print(f"[trial {trial.number}] FAILED: {type(exc).__name__}: {exc}")
            return -1.0
        # attach every captured metric so it lands in trials_dataframe()
        for mk, mv in metrics.items():
            trial.set_user_attr(f"metric_{mk}", mv)
        print(f"[trial {trial.number}] valid MCC = {score:.4f}  "
              f"(dloss={cfg['dloss']} vae={cfg['variational']} kan={cfg['kan']}) "
              f"[{len(metrics)} metrics captured]")
        return score
    return objective


def run_study(args, data):
    """Run one Optuna study and return it (does not persist).

    Uses W&B logging when available (falls back to plain make_objective otherwise).
    MLflow is always active underneath via run_trial.
    """
    import optuna

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"{args.out_prefix}_{args.dataset}")

    objective = make_objective(args, data)   # MLflow-only default
    if not getattr(args, "no_wandb", False):
        try:
            import wandb
            from hp_search_wandb import make_objective_wandb
            wandb.login()   # uses ~/.netrc / WANDB_API_KEY; raises if unavailable
            objective = make_objective_wandb(args, data)
            print("[wandb] per-trial logging enabled (MLflow still active per trial)")
        except Exception as exc:
            print(f"[wandb] disabled ({type(exc).__name__}: {exc}); using MLflow only")

    study.optimize(objective, n_trials=args.n_trials,
                   timeout=args.timeout, gc_after_trial=True)
    return study


def persist_study(study, args) -> dict:
    """Write <prefix>_trials.csv + <prefix>_best.json under results-dir. Return best dict."""
    from src.zero_shot_recommender.data import append_trial
    from src.zero_shot_recommender.schema import TrialRecord

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    trials_csv = results_dir / f"{args.out_prefix}_trials.csv"
    study.trials_dataframe().to_csv(trials_csv, index=False)

    ledger = results_dir / "unified_trials.jsonl"
    for trial in study.trials:
        attrs = dict(trial.user_attrs)
        kan = bool(attrs.get("kan", False))
        model_type = str(attrs.get("model_type", getattr(args, "model_type", "joint")))
        model_family = (
            "kan_two_stage" if kan and model_type == "two_stage" else
            "kan_joint" if kan else
            "two_stage_nn" if model_type == "two_stage" else
            "joint_nn"
        )
        config = {
            **{k: attrs[k] for k in CONFIG_KEYS if k in attrs},
            "model_family": model_family,
            "head_type": "nn",
            "optimizer": "adam",
            "transductive": bool(getattr(args, "combine_test", False)),
            "batch_size": int(getattr(args, "bs", 32)),
            "n_epochs": int(getattr(args, "n_epochs", 1000)),
        }
        complete = trial.value is not None and np.isfinite(float(trial.value)) and "error" not in attrs
        append_trial(
            ledger,
            TrialRecord(
                dataset_id=str(args.dataset),
                model_family=model_family,
                score=float(trial.value) if complete else -1.0,
                config=config,
                status="complete" if complete else "failed",
                seed=int(getattr(args, "seed", 42)),
                metrics={k.removeprefix("metric_"): v for k, v in attrs.items() if k.startswith("metric_")},
                source="classic_sweep",
            ),
        )

    best = dict(study.best_trial.user_attrs)
    best.pop("error", None)
    best["_valid_mcc"] = study.best_value
    best["_trial_number"] = study.best_trial.number
    best_json = results_dir / f"{args.out_prefix}_best.json"
    best_json.write_text(json.dumps(best, indent=2))

    print("\n" + "=" * 70)
    print(f"BEST valid MCC = {study.best_value:.4f}  (trial #{study.best_trial.number})")
    print(f"Trials CSV : {trials_csv}")
    print(f"Best JSON  : {best_json}")
    print("=" * 70)
    return best


def _bool_arg(v):
    if v is None:
        return None
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BERNN hyperparameter search (Optuna).")
    p.add_argument("--dataset", default="massbench_benchmark",
                   help="Dataset folder under data/datasets/ (default: massbench_benchmark)")
    p.add_argument("--path", default=None,
                   help="Directory containing --csv-file for exact BERNN custom CSV runs.")
    p.add_argument("--csv-file", dest="csv_file", default=None,
                   help="Custom BERNN CSV such as intensities.csv. Uses labels/label/group and batches/batch columns.")
    p.add_argument("--combine-test", action="store_true",
                   help="Also use the test split's features for the search (labels included).")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--timeout", type=int, default=None, help="Optional wall-clock limit (seconds).")
    p.add_argument("--n-epochs", type=int, default=200,
                   help="Epochs per trial (higher = better estimate, slower).")
    p.add_argument("--n-repeats", type=int, default=-1,
                   help="CV folds for the BERNN wrapper. Use -1 for grouped CV capped at 5 folds.")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--max-warmup", type=int, default=50, help="Upper bound of the warmup search range.")
    p.add_argument("--device", default=_default_device(), choices=["cpu", "cuda"],
                   help="Compute device (default: cuda if a GPU is available, else cpu).")
    p.add_argument("--seed", type=int, default=42)
    # Optional fixes to shrink the space to one model family:
    p.add_argument("--dloss", default=None, choices=DLOSS_CHOICES,
                   help="Fix the domain loss instead of searching it.")
    p.add_argument("--variational", type=_bool_arg, default=None,
                   help="Fix AE(false)/VAE(true) instead of searching it.")
    p.add_argument("--kan", type=_bool_arg, default=None,
                   help="Fix MLP(false)/KAN(true) instead of searching it.")
    p.add_argument("--model-type", default="joint", choices=["joint", "two_stage"],
                   help="BERNN trainer type to use for each fit.")
    p.add_argument("--class-triplet", type=_bool_arg, default=None,
                   help="Enable supervised class-triplet loss.")
    p.add_argument("--class-triplet-w", dest="class_triplet_w", type=float, default=None,
                   help="Weight for supervised class-triplet loss when enabled.")
    p.add_argument("--log1p", type=_bool_arg, default=None,
                   help="Fix log1p preprocessing instead of searching it.")
    p.add_argument("--rec-loss", dest="rec_loss", default="l1", choices=["l1", "mse"],
                   help="AE reconstruction loss used by BERNN training.")
    p.add_argument("--preset", default=None, choices=list(PRESETS),
                   help="Run one named family (fixes dloss+variational; sets out-prefix).")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable Weights & Biases logging (MLflow stays on).")
    p.add_argument("--out-prefix", default="bernn_hpsearch",
                   help="Prefix for the output CSV/JSON files.")
    p.add_argument("--results-dir", default=str(ROOT / "results"),
                   help="Directory for the trials CSV / best JSON (default: results/).")
    args = p.parse_args(argv)
    if args.preset:
        fam = PRESETS[args.preset]
        args.dloss = fam["dloss"]
        args.variational = fam["variational"]
        args.class_triplet = fam.get("class_triplet", args.class_triplet)
        if args.out_prefix == "bernn_hpsearch":
            args.out_prefix = f"bernn_hpsearch_{args.preset}"
    return args


def main(argv=None):
    import mlflow

    args = parse_args(argv) if not isinstance(argv, argparse.Namespace) else argv
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())

    print(f"Loading dataset '{args.dataset}' ...")
    X, y, batches = load_dataset(
        args.dataset,
        combine_test=args.combine_test,
        path=args.path,
        csv_file=args.csv_file,
    )
    print(f"  X={X.shape}  classes={sorted(set(y))}  batches={sorted(set(batches))}")
    args.resolved_n_repeats = resolve_n_repeats(args.n_repeats, batches)
    if int(args.n_repeats) == -1:
        print(f"  n_repeats=-1 resolved to grouped CV capped at 5: {args.resolved_n_repeats} folds")
    else:
        print(f"  n_repeats={args.n_repeats} resolved to {args.resolved_n_repeats} CV folds")
    data = (X, y, batches)

    study = run_study(args, data)
    best = persist_study(study, args)

    # --- emit a ready-to-use leaderboard config --------------------------
    try:
        from src.baselines import bernn_config, build_bernn_code
        overrides = {k: v for k, v in best.items()
                     if not k.startswith("_") and not k.startswith("metric_")
                     and k not in ("mlflow_exp",) and k in bernn_config()}
        full = bernn_config(**overrides)
        full["n_epochs"] = args.n_epochs
        print("\nPaste this into the leaderboard's BERNN baseline CONFIG:")
        print(json.dumps(full, indent=2))
        print("\n----- generated fit_predict -----")
        print(build_bernn_code(full))
    except Exception as exc:
        print(f"(could not render leaderboard code: {exc})")

    return study


if __name__ == "__main__":
    main()