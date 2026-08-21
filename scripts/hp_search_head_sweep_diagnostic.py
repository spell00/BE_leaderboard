"""AE-encoder + interchangeable classification head sweep (bernn-msms 0.6.3).

Uses ``bernn.dl.train.train_ae_head_sweep.AEHeadSweepTrainer`` directly.
Builds the bernn data dict from the leaderboard CSV (name, batch, label, features).

DVC: dvc repro head_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import hp_search as hp
from wandb_cv_logging import log_compact_epoch_charts, log_compact_fold_chart, set_numeric_summary


# ---------------------------------------------------------------------------
# bernn version requirement. AEHeadSweepTrainer got several fixes this pipeline
# depends on (>=0.6.7): samples_weights synthesised instead of None, positive
# TripletMarginLoss margin for torch>=2, ncols clamped to avoid out-of-bounds
# CUDA asserts, and rec["mean"] list handling. Fail fast against a stale install.
# ---------------------------------------------------------------------------
def _require_bernn(min_version=(0, 6, 7)):
    import bernn
    try:
        got = tuple(int(x) for x in bernn.__version__.split(".")[:3])
    except Exception:
        got = (0, 0, 0)
    if got < min_version:
        raise RuntimeError(
            f"bernn>={'.'.join(map(str, min_version))} required, found "
            f"{bernn.__version__}. Rebuild/reinstall the bernn package."
        )


HEAD_SWEEP_DIR = ROOT / "results" / "head_sweep"
RESULTS_DIR    = ROOT / "results"

DLOSS_CHOICES = ["inverseTriplet", "DANN", "revTriplet", "normae", "no"]
DEFAULT_HEAD_TYPES = [
    "xgboost", "random_forest", "linear_svc", "svc_rbf",
    "logistic_regression", "knn", "gradient_boosting",
    "prototype_mean", "prototype_kmeans",
]
LEADERBOARD_HEAD_SWEEP_PRESETS = {
    (False, "inverseTriplet", False): "ae_head_sweep_triplet",
    (False, "DANN", False): "ae_head_sweep_dann",
    (False, "no", False): "ae_head_sweep_no",
}

_META_COLS = ("name", "batch", "label")

# ---------------------------------------------------------------------------
# W&B logging. The head sweep logs to its OWN project, separate from the classic
# hp_search runs in 'BE_leaderboard' (kept intact). One W&B run per trial,
# grouped by preset/family, so head types can be compared in the dashboard.
# Override project/entity via env before launch (inherited by `dvc repro`).
# Set WANDB_MODE=offline for local-only, or WANDB_DISABLED=true to skip entirely.
# ---------------------------------------------------------------------------
WANDB_PROJECT = os.getenv("WANDB_HEAD_SWEEP_PROJECT", "BE_leaderboard_zero_shot_hparams")
WANDB_ENTITY  = os.getenv("WANDB_ENTITY", "adlab")
WANDB_MODE    = os.getenv("WANDB_MODE") or None
WANDB_ENABLED = os.getenv("WANDB_DISABLED", "").lower() not in ("1", "true", "yes")
os.environ.setdefault("WANDB_SILENT", "true")  # avoid per-trial URL spam


def _assert_materialized_csv(path: Path) -> None:
    """Reject Git LFS pointer text before pandas produces a misleading error."""
    with path.open("rb") as stream:
        first_line = stream.readline(200).strip()
    if first_line == b"version https://git-lfs.github.com/spec/v1":
        raise RuntimeError(
            f"Dataset file is an unresolved Git LFS pointer: {path}. "
            "Materialize it with `git lfs pull` or download it from the "
            "project's Hugging Face Space before running the search."
        )



def _load_train_fixed_test_dataset(name: str, *, include_test_labels: bool = False):
    """Load the leaderboard train split plus the fixed private/app test split.

    Fixed-test labels are deliberately not read during HPO.  The opt-in flag is
    reserved for a separate, post-selection evaluation command.
    """
    base = hp.DATASETS_DIR / name
    train_csv = base / f"{name}_train.csv"
    test_csv = base / f"{name}_test.csv"
    labels_csv = base / f"{name}_predictions.csv"
    inference_csv = base / f"{name}_inference.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"No train CSV for dataset '{name}' at {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"No fixed test CSV for dataset '{name}' at {test_csv}")
    _assert_materialized_csv(train_csv)
    _assert_materialized_csv(test_csv)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    feature_cols = [c for c in train_df.columns if c not in hp._META_COLS]
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Fixed test CSV is missing {len(missing)} train feature columns; first={missing[:5]}")

    X_train = train_df[feature_cols].astype(float).reset_index(drop=True)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df["label"].astype(str).reset_index(drop=True)
    batches_train = train_df["batch"].astype(str).reset_index(drop=True)
    train_names = train_df.get("name", pd.Series([f"train_{i}" for i in range(len(train_df))])).astype(str).reset_index(drop=True)

    X_test = test_df[feature_cols].astype(float).reset_index(drop=True)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    batches_test = test_df["batch"].astype(str).reset_index(drop=True)
    test_names = test_df.get("name", pd.Series([f"test_{i}" for i in range(len(test_df))])).astype(str).reset_index(drop=True)

    y_test = None
    if include_test_labels and labels_csv.exists():
        labels_df = pd.read_csv(labels_csv)
        if {"name", "prediction"}.issubset(labels_df.columns):
            labels_df = labels_df[["name", "prediction"]].copy()
            labels_df["name"] = labels_df["name"].astype(str)
            label_map = labels_df.set_index("name")["prediction"].astype(str)
            y_test = test_names.map(label_map).astype(str).reset_index(drop=True)
    elif include_test_labels and inference_csv.exists():
        inf_df = pd.read_csv(inference_csv)
        if "label" in inf_df.columns and len(inf_df) == len(test_df):
            y_test = inf_df["label"].astype(str).reset_index(drop=True)

    return X_train, y_train, batches_train, train_names, X_test, y_test, batches_test, test_names


def _batch_cv_splits(X_train: pd.DataFrame, y_train: pd.Series, batches_train: pd.Series, n_cv: int, seed: int):
    """Return the project-standard CV splits.

    n_cv=1 explicitly requests one stratified 80/20 sample holdout. n_cv>=2
    always keeps batches disjoint. When fewer batches than requested folds are
    available, leave-one-batch-out is used instead of a leaking sample fallback.
    """
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, train_test_split

    labels = y_train.astype(str).reset_index(drop=True)
    groups = batches_train.astype(str).reset_index(drop=True)
    if int(n_cv) == 1:
        idx = np.arange(len(labels))
        train_idx, valid_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        protocol = f"StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state={seed})"
        return protocol, [(np.asarray(train_idx), np.asarray(valid_idx))]
    if int(n_cv) < 1:
        raise ValueError(f"n_cv must be >=1, got {n_cv}")
    n_batches = int(groups.nunique())
    if n_batches < 2:
        raise ValueError(
            f"Held-out-batch CV requires at least 2 labeled training batches; found {n_batches}. "
            "Use --n-cv 1 only if a sample-stratified holdout is explicitly intended."
        )
    if n_batches >= n_cv:
        splitter = StratifiedGroupKFold(n_splits=n_cv, shuffle=True, random_state=seed)
        protocol = f"StratifiedGroupKFold(n_splits={n_cv}, shuffle=True, random_state={seed})"
        return protocol, list(splitter.split(X_train, labels, groups))
    splitter = LeaveOneGroupOut()
    protocol = f"LeaveOneGroupOut(n_splits={n_batches}; requested_n_cv={n_cv})"
    return protocol, list(splitter.split(X_train, labels, groups))


def _build_bernn_data_for_fold(
    X_train_all: pd.DataFrame,
    y_train_all: pd.Series,
    batches_train_all: pd.Series,
    names_train_all: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | None,
    batches_test: pd.Series,
    names_test: pd.Series,
):
    """Build BERNN data for one outer fold.

    Fixed-test labels are included when available solely for diagnostic epoch
    metrics. They never affect optimization, early stopping, or model selection.
    """
    from sklearn.preprocessing import LabelEncoder

    y_train_all = y_train_all.astype(str).reset_index(drop=True)
    batches_train_all = batches_train_all.astype(str).reset_index(drop=True)
    names_train_all = names_train_all.astype(str).reset_index(drop=True)
    batches_test = batches_test.astype(str).reset_index(drop=True)
    names_test = names_test.astype(str).reset_index(drop=True)

    le_label = LabelEncoder().fit(y_train_all)
    dummy_test_label = le_label.classes_[0]
    if y_test is None:
        y_test_for_loader = pd.Series([dummy_test_label] * len(X_test))
    else:
        y_test_for_loader = y_test.astype(str).reset_index(drop=True)
        unknown = sorted(set(y_test_for_loader.unique()) - set(le_label.classes_))
        if unknown:
            raise ValueError(f"Fixed test contains labels unseen in training: {unknown}")
    all_batches_raw = pd.concat([batches_train_all, batches_test], ignore_index=True)
    le_batch = LabelEncoder().fit(all_batches_raw)

    split_defs = {
        "train": (X_train_all.iloc[train_idx].reset_index(drop=True), y_train_all.iloc[train_idx].reset_index(drop=True), batches_train_all.iloc[train_idx].reset_index(drop=True), names_train_all.iloc[train_idx].reset_index(drop=True)),
        "valid": (X_train_all.iloc[valid_idx].reset_index(drop=True), y_train_all.iloc[valid_idx].reset_index(drop=True), batches_train_all.iloc[valid_idx].reset_index(drop=True), names_train_all.iloc[valid_idx].reset_index(drop=True)),
        "test":  (X_test.reset_index(drop=True), y_test_for_loader, batches_test.reset_index(drop=True), names_test.reset_index(drop=True)),
    }

    data = {key: {} for key in ["inputs", "batches", "cats", "labels", "names", "orders", "sets"]}
    for group, (Xg, yg, bg, ng) in split_defs.items():
        data["inputs"][group] = pd.DataFrame(Xg.to_numpy(), columns=X_train_all.columns)
        data["batches"][group] = le_batch.transform(bg)
        data["cats"][group] = le_label.transform(yg.astype(str))
        data["labels"][group] = yg.astype(str).to_numpy()
        data["names"][group] = pd.Series(ng.astype(str).to_numpy())
        data["orders"][group] = np.zeros(len(Xg), dtype=int)
        data["sets"][group] = np.array([group for _ in range(len(Xg))])

    data["inputs"]["all"] = pd.concat([data["inputs"]["train"], data["inputs"]["valid"], data["inputs"]["test"]], ignore_index=True)
    data["batches"]["all"] = np.concatenate([data["batches"]["train"], data["batches"]["valid"], data["batches"]["test"]])
    data["cats"]["all"] = np.concatenate([data["cats"]["train"], data["cats"]["valid"], data["cats"]["test"]])
    data["labels"]["all"] = np.concatenate([data["labels"]["train"], data["labels"]["valid"], data["labels"]["test"]])
    data["names"]["all"] = pd.concat([data["names"]["train"], data["names"]["valid"], data["names"]["test"]], ignore_index=True)
    data["orders"]["all"] = np.zeros(len(data["inputs"]["all"]), dtype=int)
    data["sets"]["all"] = np.array(["all" for _ in range(len(data["inputs"]["all"]))])

    return data, le_label.classes_, le_batch.classes_, le_label


class BatchCVHeadSweepTrainer:
    """Optuna objective matching app.py: held-out-batch folds + fixed-test ensemble."""

    def __init__(self, base_trainer_cls, args, path, X_train, y_train, batches_train, names_train,
                 X_test, y_test, batches_test, names_test, n_cv=5):
        self.base_trainer_cls = base_trainer_cls
        self.args = args
        self.path = path
        self.X_train = X_train
        self.y_train = y_train.astype(str).reset_index(drop=True)
        self.batches_train = batches_train.astype(str).reset_index(drop=True)
        self.names_train = names_train.astype(str).reset_index(drop=True)
        self.X_test = X_test
        self.y_test = None if y_test is None else y_test.astype(str).reset_index(drop=True)
        self.batches_test = batches_test.astype(str).reset_index(drop=True)
        self.names_test = names_test.astype(str).reset_index(drop=True)
        self.n_cv = int(n_cv)
        self.protocol, self.splits = _batch_cv_splits(X_train, self.y_train, self.batches_train, self.n_cv, args.seed)
        self.best_valid_mcc = float("-inf")
        self.best_head_type = None
        self.best_head_params = None

    def objective(self, trial):
        import copy as _copy
        import gc
        import torch
        from collections import defaultdict
        from sklearn.metrics import matthews_corrcoef
        from bernn.utils.utils import scale_data
        from bernn.dl.models.pytorch.utils.dataset import get_loaders, get_loaders_no_pool
        from bernn.dl.train.head_classifier import fit_and_score_head
        import bernn.dl.train.train_ae_head_sweep as hs

        args = self.args
        ae_params = hs._suggest_ae_params(trial, args)
        available_heads = list(getattr(args, "head_types", None) or DEFAULT_HEAD_TYPES)
        available_heads = [h for h in available_heads if h != "xgboost" or getattr(hs, "_HAS_XGB", False)]
        if not available_heads:
            trial.set_user_attr("head_error", "No classifier heads are enabled")
            return float("-inf")
        head_params_by_type = {h: hs._suggest_head_params(trial, h) for h in available_heads}

        head_valid_mccs = defaultdict(list)
        head_train_mccs = defaultdict(list)
        head_test_preds = defaultdict(list)
        fold_details = []
        ae_mccs = []
        batch_metric_rows = []
        all_epoch_metrics = []
        head_errors = defaultdict(list)
        label_encoder = None

        for fold, (fold_train_idx, fold_valid_idx) in enumerate(self.splits, start=1):
            data, unique_labels, unique_batches, label_encoder = _build_bernn_data_for_fold(
                self.X_train, self.y_train, self.batches_train, self.names_train,
                fold_train_idx, fold_valid_idx,
                self.X_test, self.y_test, self.batches_test, self.names_test,
            )
            fold_args = _copy.copy(args)
            fold_args.scaler = ae_params.get("scaler", "standard")
            fold_args.ncols = ae_params.get("ncols", -1)
            fold_args.exp_id = f"{getattr(args, 'exp_id', 'be_leaderboard_head_sweep')}_fold{fold}"
            trainer = self.base_trainer_cls(
                args=fold_args,
                path=self.path,
                unique_labels=list(unique_labels),
                unique_batches=list(unique_batches),
                data=data,
                n_cv=1,
            )
            ae = trainer._build_ae(ae_params["layer1"], ae_params["layer2"])

            scaled_data = _copy.deepcopy(data)
            scaled_data, _ = scale_data(fold_args.scaler, scaled_data, args.device)
            for g in list(scaled_data["inputs"].keys()):
                scaled_data["inputs"][g] = scaled_data["inputs"][g].round(4)

            samples_weights = {}
            for g in ("train", "valid", "test"):
                cats_g = np.asarray(scaled_data["cats"][g])
                if g == "train" and len(cats_g):
                    cls, cnt = np.unique(cats_g, return_counts=True)
                    weights = {int(c): 1.0 / max(int(n), 1) for c, n in zip(cls, cnt)}
                    samples_weights[g] = [weights[int(c)] for c in cats_g]
                else:
                    samples_weights[g] = [1.0] * len(cats_g)

            dloss = getattr(args, "dloss", "inverseTriplet")
            bs = getattr(args, "bs", 32)
            try:
                loaders = get_loaders(scaled_data, False, samples_weights, dloss, None, None, bs, args.device)
            except Exception:
                loaders = get_loaders_no_pool(scaled_data, False, samples_weights, dloss, None, None, bs, args.device)

            try:
                ae_mcc, epoch_metrics = trainer._train_ae(ae, ae_params, loaders, trial_num=f"{trial.number}/fold{fold}")
                ae_mccs.append(float(ae_mcc))
                for row in epoch_metrics:
                    row = dict(row)
                    row["fold"] = fold
                    all_epoch_metrics.append(row)
            except Exception as exc:
                trial.set_user_attr("ae_error", f"fold {fold}: {exc}")
                return float("-inf")

            ae.eval()
            for param in ae.enc.parameters():
                param.requires_grad = False
            for param in ae.dec.parameters():
                param.requires_grad = False

            try:
                X_tr, y_tr, b_tr = hs.extract_embeddings_labels_batches(ae, loaders["train"], args.device)
                X_vl, y_vl, b_vl = hs.extract_embeddings_labels_batches(ae, loaders["valid"], args.device)
                X_te, _dummy_y_te, b_te = hs.extract_embeddings_labels_batches(ae, loaders["test"], args.device)
            except Exception as exc:
                trial.set_user_attr("embed_error", f"fold {fold}: {exc}")
                return float("-inf")

            if len(X_tr) == 0 or len(X_vl) == 0:
                trial.set_user_attr("embed_error", f"fold {fold}: empty train/valid embeddings")
                return float("-inf")

            for head_type in available_heads:
                head_params = head_params_by_type.get(head_type, {})
                try:
                    fitted_head, tr_mcc, vl_mcc = fit_and_score_head(X_tr, y_tr, X_vl, y_vl, head_type, head_params)
                    test_preds = hs._predict_with_optional_label_decoder(fitted_head, X_te)
                    head_valid_mccs[head_type].append(float(vl_mcc))
                    head_train_mccs[head_type].append(float(tr_mcc))
                    head_test_preds[head_type].append(pd.Series(test_preds).astype(str).reset_index(drop=True))
                except Exception as exc:
                    head_errors[head_type].append(f"fold {fold}: {exc}")

            try:
                batch_metric_rows.append(hs._embedding_batch_effect_metrics((X_tr, b_tr), (X_vl, b_vl), (X_te, b_te)))
            except Exception:
                pass

            fold_details.append({
                "fold": fold,
                "n_train": int(len(fold_train_idx)),
                "n_valid": int(len(fold_valid_idx)),
                "n_test": int(len(self.X_test)),
                "train_batches": sorted(self.batches_train.iloc[fold_train_idx].unique().tolist()),
                "valid_batches": sorted(self.batches_train.iloc[fold_valid_idx].unique().tolist()),
                "test_batches": sorted(self.batches_test.unique().tolist()),
            })

            del ae, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        complete_heads = {h: vals for h, vals in head_valid_mccs.items() if len(vals) == len(self.splits)}
        if not complete_heads:
            trial.set_user_attr("head_error", json.dumps({k: v[:3] for k, v in head_errors.items()}))
            return float("-inf")

        head_type, vals = max(complete_heads.items(), key=lambda kv: float(np.mean(kv[1])))
        valid_mcc = float(np.mean(vals))
        valid_std = float(np.std(vals))
        train_mcc = float(np.mean(head_train_mccs[head_type])) if head_train_mccs[head_type] else float("nan")
        test_mcc = float("nan")
        if self.y_test is not None and head_test_preds[head_type]:
            votes = pd.concat(head_test_preds[head_type], axis=1)
            consensus = votes.mode(axis=1).iloc[:, 0].astype(str).reset_index(drop=True)
            y_ref = self.y_test.astype(str).reset_index(drop=True)
            if label_encoder is not None and set(consensus.unique()).issubset(set(map(str, range(len(label_encoder.classes_))))):
                consensus = pd.Series(label_encoder.inverse_transform(consensus.astype(int))).astype(str)
            test_mcc = float(matthews_corrcoef(y_ref, consensus))
            trial.set_user_attr("test_predictions_json", json.dumps(consensus.tolist()))

        batch_metrics = {}
        if batch_metric_rows:
            keys = {k for row in batch_metric_rows for k in row}
            for key in keys:
                nums = []
                for row in batch_metric_rows:
                    if key in row and row[key] is not None:
                        try:
                            val = float(row[key])
                        except Exception:
                            continue
                        if np.isfinite(val):
                            nums.append(val)
                if nums:
                    batch_metrics[key] = float(np.mean(nums))

        trial.set_user_attr("valid_mcc", valid_mcc)
        trial.set_user_attr("valid_mcc_std", valid_std)
        trial.set_user_attr("valid_mcc_folds", json.dumps([float(x) for x in vals]))
        trial.set_user_attr("valid_fold_details", json.dumps(fold_details))
        trial.set_user_attr("cv_protocol", self.protocol)
        trial.set_user_attr("train_mcc", train_mcc)
        trial.set_user_attr("head_cv_train_mcc", train_mcc)
        trial.set_user_attr("test_mcc", test_mcc)
        trial.set_user_attr("head_type", head_type)
        trial.set_user_attr("head_params", json.dumps(head_params_by_type.get(head_type, {})))
        trial.set_user_attr("all_head_params_json", json.dumps(head_params_by_type))
        for evaluated_head, evaluated_values in complete_heads.items():
            trial.set_user_attr(f"head_valid_mcc__{evaluated_head}", float(np.mean(evaluated_values)))
        trial.set_user_attr("ae_classifier_mcc", float(np.mean(ae_mccs)) if ae_mccs else float("nan"))
        try:
            trial.set_user_attr("epoch_metrics_json", json.dumps(all_epoch_metrics))
        except Exception:
            pass
        for k, v in batch_metrics.items():
            trial.set_user_attr(k, v)

        if valid_mcc > self.best_valid_mcc:
            self.best_valid_mcc = valid_mcc
            self.best_head_type = head_type
            self.best_head_params = head_params_by_type.get(head_type, {})
        return valid_mcc

def _make_sweep_args(dataset, dloss, variational, class_triplet, args, n_features=-1):
    return types.SimpleNamespace(
        device=args.device,
        dloss=dloss,
        variational=variational,
        class_triplet=class_triplet,
        class_triplet_w=float(getattr(args, "class_triplet_w", 1.0)),
        rec_loss=getattr(args, "rec_loss", "l1"),
        use_l1=(getattr(args, "rec_loss", "l1") == "l1"),
        tied_weights=False,
        use_mapping=True,
        # Critical for the ML-head sweep: BERNN evaluates all head_types listed here
        # on frozen AE embeddings. The installed trainer reads args.head_types, not
        # the module-level HEAD_TYPES variable.
        head_types=list(getattr(args, "head_types", None) or DEFAULT_HEAD_TYPES),
        kan=False,
        zinb=0,
        n_epochs=args.n_epochs,
        early_stop=args.early_stop,
        bs=args.bs,
        groupkfold=True,
        random_recs=0,
        predict_tests=0,
        threshold=0.0,
        bdisc=1,
        n_repeats=3,
        n_meta=0,
        embeddings_meta=0,
        remove_zeros=0,
        log1p=1,
        log_metrics=0,
        log_plots=0,
        log_tb=0,
        log_mlflow=0,
        log_tracking=0,
        keep_models=False,
        pool=0,
        bad_batches="",
        exp_id="be_leaderboard_head_sweep",
        dataset=dataset,
        path=str(ROOT / "data"),
        csv_file=f"{dataset}_train.csv",
        seed=args.seed,
        scaler="standard",
        ncols=-1,
        n_features=n_features,
    )


def _log_wandb_head_trial(preset_name, dataset, trial):
    """Best-effort: mirror one head-sweep trial to W&B as its own run. A logging
    failure never kills the sweep — a long unattended run shouldn't die on a
    transient network blip."""
    if not WANDB_ENABLED:
        return
    import wandb
    try:
        val = trial.value
        cv  = float(val) if (val is not None and val != float("-inf")) else None
        a   = trial.user_attrs
        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, mode=WANDB_MODE,
            name=f"{preset_name}_t{trial.number}",
            group=preset_name, job_type="head_sweep",
            config={**trial.params, "preset": preset_name,
                    "dataset": dataset,
                    "trial": trial.number},
            reinit="finish_previous",
        )
        payload = {"valid_mcc": cv} if cv is not None else {"failed": 1}
        for k, v in a.items():
            if k.endswith("_json"):
                continue
            if k.endswith("_folds"):
                payload[k] = v
                continue
            try:
                payload[k] = float(v)
            except (TypeError, ValueError):
                pass
        if cv is None:
            err = a.get("ae_error") or a.get("embed_error") or a.get("head_error")
            if err:
                run.summary["error"] = str(err)[:300]

        set_numeric_summary(run, payload)
        log_compact_fold_chart(run, payload)

        # Replay per-epoch AE training curves into the same W&B trial run, but
        # concatenate folds onto one x-axis instead of creating fold-specific
        # metric names/charts.
        try:
            epoch_rows = json.loads(a.get("epoch_metrics_json", "[]"))
        except Exception:
            epoch_rows = []
        epoch_rows_by_fold = {}
        for row in epoch_rows:
            try:
                fold = int(row.get("fold", 1)) - 1
            except Exception:
                fold = 0
            epoch_rows_by_fold.setdefault(str(fold), []).append(row)
        log_compact_epoch_charts(run, epoch_rows_by_fold)
        run.finish()
    except Exception as exc:
        print(f"[wandb] head trial {getattr(trial, 'number', '?')} log failed: {exc}")


def _run_sweep_wandb(trainer, preset_name, dataset, n_trials):
    """Same study setup as bernn's AEHeadSweepTrainer.run_sweep, plus a per-trial
    W&B callback (run_sweep does not accept callbacks)."""
    import optuna
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=max(10, n_trials // 10))
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        study_name=f"be_{preset_name}", direction="maximize",
        sampler=sampler, pruner=pruner, load_if_exists=True,
    )

    def _cb(study, trial):
        from src.zero_shot_recommender.data import append_trial
        from src.zero_shot_recommender.schema import TrialRecord

        _log_wandb_head_trial(preset_name, dataset, trial)
        # Verbose per-trial line to stdout (captured in head_sweep.log / tmux).
        val = trial.value
        a   = trial.user_attrs
        failed = val is None or val == float("-inf")
        mcc = "  FAIL" if failed else f"{val:7.4f}"
        try:
            best = study.best_value
        except Exception:
            best = float("nan")
        head = a.get("head_type") or trial.params.get("head_type", "?")
        test_m = a.get("test_mcc")
        ae_m = a.get("ae_classifier_mcc")
        nbe = a.get("batch_nbe")
        extra = ""
        if test_m is not None and not (isinstance(test_m, float) and test_m != test_m):
            extra += f"  test={test_m:.3f}"
        if nbe is not None and not (isinstance(nbe, float) and nbe != nbe):
            extra += f"  nBE={nbe:.3f}"
        if ae_m is not None:
            extra += f"  ae_cls={ae_m:.3f}"
        if failed:
            err = a.get("ae_error") or a.get("embed_error")
            if err:
                extra += f"  err={str(err)[:90]}"
        base_config = {
            **trial.params,
            "model_family": "frozen_ae_classical",
            "dloss": str(getattr(trainer.args, "dloss", "no")),
            "variational": bool(getattr(trainer.args, "variational", False)),
            "kan": bool(getattr(trainer.args, "kan", False)),
            "class_triplet": bool(getattr(trainer.args, "class_triplet", False)),
            "class_triplet_w": float(getattr(trainer.args, "class_triplet_w", 0.0)),
            "transductive": True,
            "batch_size": int(getattr(trainer.args, "bs", 32)),
            "n_epochs": int(getattr(trainer.args, "n_epochs", 1000)),
            "optimizer": "adam",
        }
        head_scores = {
            key.split("__", 1)[1]: float(value)
            for key, value in a.items() if str(key).startswith("head_valid_mcc__")
        }
        try:
            all_head_params = json.loads(a.get("all_head_params_json", "{}"))
        except (TypeError, ValueError):
            all_head_params = {}
        ledger = HEAD_SWEEP_DIR.parent / "unified_trials.jsonl"
        if failed or not head_scores:
            append_trial(
                ledger,
                TrialRecord(
                    dataset_id=dataset, model_family="frozen_ae_classical", score=-1.0,
                    config={**base_config, "head_type": "none"}, status="failed",
                    seed=int(getattr(trainer.args, "seed", 42)),
                    metrics={k: v for k, v in a.items() if not str(k).endswith("_json")},
                    source="head_sweep",
                ),
            )
        else:
            for evaluated_head, evaluated_score in head_scores.items():
                head_params = all_head_params.get(evaluated_head, {})
                canonical = {}
                if evaluated_head == "knn":
                    canonical["knn_k"] = head_params.get("n_neighbors", head_params.get("knn_k", 5))
                if evaluated_head in {"logistic_regression", "linear_svc", "svc_rbf"}:
                    canonical["head_C"] = head_params.get("C", 1.0)
                if evaluated_head in {"random_forest", "xgboost", "gradient_boosting"}:
                    canonical["n_estimators"] = head_params.get("n_estimators", 100)
                    canonical["max_depth"] = head_params.get("max_depth", 6)
                append_trial(
                    ledger,
                    TrialRecord(
                        dataset_id=dataset,
                        model_family="frozen_ae_classical",
                        score=evaluated_score,
                        config={**base_config, **head_params, **canonical, "head_type": evaluated_head},
                        status="complete",
                        seed=int(getattr(trainer.args, "seed", 42)),
                        metrics={"valid_mcc": evaluated_score},
                        source="head_sweep",
                    ),
                )
        print(f"[{preset_name}] trial {trial.number:>3}/{n_trials}  "
              f"valid_mcc={mcc}  head={head:<16}{extra}  best={best:.4f}", flush=True)

    study.optimize(trainer.objective, n_trials=n_trials, gc_after_trial=True,
                   catch=(Exception,), callbacks=[_cb])
    return study


def _run_family(
    dloss, variational, class_triplet, args,
    X_train, y_train, batches_train, names_train,
    X_test, y_test, batches_test, names_test,
):
    from bernn.dl.train.train_ae_head_sweep import AEHeadSweepTrainer
    import bernn.dl.train.train_ae_head_sweep as head_sweep_module
    _require_bernn()

    preset_name = (
        f"{'vae' if variational else 'ae'}"
        f"_{dloss.lower()}"
        f"{'_ct' if class_triplet else ''}"
    )
    print(f"\n{'='*60}")
    print(f"[head_sweep] {preset_name}  "
          f"(dloss={dloss} vae={variational} class_triplet={class_triplet})")
    print(f"{'='*60}")

    out_dir = HEAD_SWEEP_DIR / preset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_args = _make_sweep_args(args.dataset, dloss, variational, class_triplet, args,
                                    n_features=int(X_train.shape[1]))
    if getattr(args, "head_types", None):
        selected_heads = [h.strip() for h in args.head_types if str(h).strip()]
        if selected_heads:
            head_sweep_module.HEAD_TYPES = selected_heads
            head_sweep_module.HEAD_TYPES_NO_XGB = [h for h in selected_heads if h != "xgboost"]
            print(f"[head_sweep] limiting head types: {selected_heads}")

    try:
        trainer = BatchCVHeadSweepTrainer(
            AEHeadSweepTrainer,
            args=sweep_args,
            path=str(ROOT / "data"),
            X_train=X_train,
            y_train=y_train,
            batches_train=batches_train,
            names_train=names_train,
            X_test=X_test,
            y_test=y_test,
            batches_test=batches_test,
            names_test=names_test,
            n_cv=args.n_cv,
        )
        print(f"[head_sweep] CV protocol: {trainer.protocol}")
        for _fold, (_tr, _vl) in enumerate(trainer.splits, start=1):
            print(
                f"[head_sweep][fold {_fold}/{len(trainer.splits)}] "
                f"train_batches={sorted(batches_train.iloc[_tr].unique().tolist())} "
                f"valid_batches={sorted(batches_train.iloc[_vl].unique().tolist())} "
                f"fixed_test_batches={sorted(batches_test.unique().tolist())}",
                flush=True,
            )

        study    = _run_sweep_wandb(trainer, preset_name, args.dataset, args.n_trials)
        best_t   = study.best_trial
        best_mcc = float(best_t.value) if best_t else np.nan
        best_attrs = dict(best_t.user_attrs) if best_t else {}
        best_head = str(best_attrs.get("head_type") or best_t.params.get("head_type", "unknown")) if best_t else "unknown"

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  [WARN] {preset_name} failed: {exc}")
        best_mcc  = np.nan
        best_head = "error"
        best_attrs = {}

    row = {
        "preset":         preset_name,
        "leaderboard_preset": LEADERBOARD_HEAD_SWEEP_PRESETS.get((bool(variational), dloss, bool(class_triplet))),
        "dloss":          dloss,
        "variational":    variational,
        "class_triplet":  class_triplet,
        "best_head_type": best_head,
        "valid_mcc":      best_mcc,
        "test_mcc":       best_attrs.get("test_mcc", np.nan),
        "train_mcc":      best_attrs.get("train_mcc", np.nan),
        "head_cv_train_mcc": best_attrs.get("head_cv_train_mcc", np.nan),
        "ae_classifier_mcc": best_attrs.get("ae_classifier_mcc", np.nan),
        "batch_silhouette": best_attrs.get("batch_silhouette", np.nan),
        "batch_centroid_dispersion": best_attrs.get("batch_centroid_dispersion", np.nan),
        "batch_nbe": best_attrs.get("batch_nbe", np.nan),
        "batch_nmi": best_attrs.get("batch_nmi", np.nan),
        "batch_nri": best_attrs.get("batch_nri", np.nan),
        "batch_metric_samples": best_attrs.get("batch_metric_samples", np.nan),
    }
    for _k, _v in best_attrs.items():
        if _k.endswith("_json") or _k in row:
            continue
        try:
            row[_k] = float(_v)
        except (TypeError, ValueError):
            pass
    best_params = dict(getattr(best_t, "params", {}) or {}) if "best_t" in locals() and best_t else {}
    row["best_params"] = best_params
    tuned = {
        "model_type": "head_sweep",
        "dloss": dloss,
        "variational": bool(variational),
        "class_triplet": bool(class_triplet),
        "class_triplet_w": float(getattr(args, "class_triplet_w", 1.0)),
        "triplet_dloss": dloss in {"inverseTriplet", "revTriplet"},
        "rec_loss": getattr(args, "rec_loss", "l1"),
        "n_epochs": int(args.n_epochs),
        "warmup": int(best_params.get("warmup", getattr(args, "early_stop", 30))),
        "bs": int(args.bs),
        "device": args.device,
        "scaler": best_params.get("scaler", "standard"),
        "n_layers": 1,
        "layer1": int(best_params.get("layer1", 256)),
        "lr": float(best_params.get("lr", 1e-3)),
        "wd": float(best_params.get("wd", 1e-5)),
        "nu": float(best_params.get("nu", 1.0)),
        "margin": float(best_params.get("margin", 1.0)),
        "smoothing": float(best_params.get("smoothing", 0.1)),
        "dropout": float(best_params.get("dropout", 0.1)),
        "thres": 0.0,
        "gamma": float(best_params.get("gamma", 0.0 if dloss == "no" else 0.1)),
        "beta": float(best_params.get("beta", 0.0 if not variational else 0.1)),
        "_valid_mcc": best_mcc,
        "_test_mcc": best_attrs.get("test_mcc", np.nan),
        "_best_head_type": best_head,
    }
    row["leaderboard_config"] = tuned
    (out_dir / "best.json").write_text(json.dumps(row, indent=2, default=str))
    return row


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default="massbench_benchmark")
    p.add_argument("--n-trials",      type=int,  default=100)
    p.add_argument("--n-epochs",      type=int,  default=300)
    p.add_argument("--early-stop",    type=int,  default=30)
    p.add_argument("--n-cv",          type=int,  default=5)
    p.add_argument("--bs",            type=int,  default=32)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--rec-loss", dest="rec_loss", default="l1", choices=["l1", "mse"],
                   help="AE reconstruction loss used by BERNN training (default: l1).")
    p.add_argument("--class-triplet-w", dest="class_triplet_w", type=float, default=1.0,
                   help="Weight for supervised class-triplet loss when enabled.")
    p.add_argument("--class-triplet",
                   type=lambda x: x.lower() not in ("false", "0", "no"),
                   default=True)
    p.add_argument("--class-triplet-only", action="store_true",
                   help="When --class-triplet is true, run only class_triplet=True families instead of both True and False.")
    p.add_argument("--dloss-choices", nargs="*", default=DLOSS_CHOICES)
    p.add_argument("--head-types", nargs="*", default=None,
                   help="Optional subset of BERNN head types to sample during Optuna.")
    p.add_argument("--ae-only", action="store_true",
                   help="Run only deterministic AE head-sweep families, skipping VAE.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Dataset-specific output directory (default: results/head_sweep).")
    p.add_argument("--no-register-defaults", action="store_true",
                   help="Do not merge this run into the repository-wide tuned defaults.")
    return p.parse_args(argv)


def main(argv=None):
    global HEAD_SWEEP_DIR, RESULTS_DIR
    args = parse_args(argv)
    if args.output_dir is not None:
        HEAD_SWEEP_DIR = args.output_dir.resolve()
        RESULTS_DIR = HEAD_SWEEP_DIR
    HEAD_SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{args.dataset}' ...")
    X_train, y_train, batches_train, names_train, X_test, y_test, batches_test, names_test = _load_train_fixed_test_dataset(args.dataset)
    print(
        f"  train_X={X_train.shape} train_classes={sorted(set(y_train))} "
        f"train_batches={sorted(set(batches_train))}"
    )
    print(
        f"  fixed_test_X={X_test.shape} fixed_test_batches={sorted(set(batches_test))} "
        f"fixed_test_labels_available={y_test is not None}"
    )
    print(f"  head_types={args.head_types or 'BERNN-default/listed ML heads'}")
    print("  protocol: 5-fold held-out-batch CV on labeled train batches; fixed app-style test batches are predicted by every fold and ensembled")
    print("  transductive unsupervised preprocessing may see fixed-test features/batches, never fixed-test class labels")

    ct_values = [True] if (args.class_triplet and args.class_triplet_only) else ([True, False] if args.class_triplet else [False])

    rows = []
    variational_values = [False] if args.ae_only else [False, True]
    for dloss in args.dloss_choices:
        for variational in variational_values:
            for ct in ct_values:
                row = _run_family(
                    dloss, variational, ct, args,
                    X_train, y_train, batches_train, names_train,
                    X_test, y_test, batches_test, names_test,
                )
                rows.append(row)

    summary = (
        pd.DataFrame(rows)
        .sort_values("valid_mcc", ascending=False)
        .reset_index(drop=True)
    )
    summary.insert(0, "dataset_id", args.dataset)
    summary.to_csv(HEAD_SWEEP_DIR / "summary.csv", index=False)
    (HEAD_SWEEP_DIR / "best.json").write_text(
        json.dumps(summary.iloc[0].to_dict() if not summary.empty else {}, indent=2, default=str)
    )

    head_tuned = {}
    for row in rows:
        preset = row.get("leaderboard_preset")
        cfg = row.get("leaderboard_config")
        if preset and isinstance(cfg, dict):
            head_tuned[preset] = cfg
    (HEAD_SWEEP_DIR / "bernn_tuned_defaults.json").write_text(
        json.dumps(head_tuned, indent=2, default=str)
    )
    if not args.no_register_defaults:
        merged_tuned_path = RESULTS_DIR / "bernn_tuned_defaults.json"
        try:
            merged_tuned = json.loads(merged_tuned_path.read_text())
        except (OSError, ValueError):
            merged_tuned = {}
        merged_tuned.update(head_tuned)
        merged_tuned_path.write_text(json.dumps(merged_tuned, indent=2, default=str))

    metrics = {
        "head_sweep_best_mcc":
            float(summary["valid_mcc"].max()) if not summary.empty else None,
        "head_sweep_best_test_mcc":
            float(summary.iloc[0]["test_mcc"]) if not summary.empty and pd.notna(summary.iloc[0]["test_mcc"]) else None,
        "head_sweep_best_batch_nbe":
            float(summary.iloc[0]["batch_nbe"]) if not summary.empty and pd.notna(summary.iloc[0]["batch_nbe"]) else None,
        "head_sweep_best_batch_silhouette":
            float(summary.iloc[0]["batch_silhouette"]) if not summary.empty and pd.notna(summary.iloc[0]["batch_silhouette"]) else None,
        "head_sweep_best_batch_centroid_dispersion":
            float(summary.iloc[0]["batch_centroid_dispersion"]) if not summary.empty and pd.notna(summary.iloc[0]["batch_centroid_dispersion"]) else None,
        "head_sweep_best_batch_nmi":
            float(summary.iloc[0]["batch_nmi"]) if not summary.empty and pd.notna(summary.iloc[0]["batch_nmi"]) else None,
        "head_sweep_best_batch_nri":
            float(summary.iloc[0]["batch_nri"]) if not summary.empty and pd.notna(summary.iloc[0]["batch_nri"]) else None,
        "head_sweep_best_preset":
            str(summary.iloc[0]["preset"]) if not summary.empty else None,
        "head_sweep_best_head_type":
            str(summary.iloc[0]["best_head_type"]) if not summary.empty else None,
    }
    metrics["dataset_id"] = args.dataset
    (HEAD_SWEEP_DIR / "head_sweep_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n" + "="*72)
    print("HEAD SWEEP COMPLETE — ranked by validation MCC:")
    print(summary[["preset", "best_head_type", "valid_mcc", "test_mcc", "batch_nbe", "batch_nmi", "batch_nri"]].to_string(index=False))
    print("="*72)
    return summary


if __name__ == "__main__":
    main()
