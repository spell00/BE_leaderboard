#!/usr/bin/env python3
"""Representation-first BERNN sweep.

Broadly optimizes AE/VAE, KAN, dloss, class-triplet and AE hparams. During AE
training it evaluates frozen representations with KNN k=1..20 at every epoch,
keeps the best epoch representation, and uses KNN validation MCC as the trial
objective. Heavier ML heads are intentionally deferred until after this finds the
best representation.
"""
from __future__ import annotations

import argparse, copy, json, os, sys, types, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import hp_search as hp
import hp_search_head_sweep as base

import torch
import optuna
from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import bernn.dl.train.train_ae_head_sweep as bm
from bernn.dl.models.pytorch import KANAutoEncoder2
from bernn.dl.models.pytorch.utils.utils import get_optimizer
from bernn.utils.utils import scale_data
from bernn.dl.train.train_ae_head_sweep import (
    AEHeadSweepTrainer,
    get_loaders, get_loaders_no_pool,
    extract_embeddings_labels_batches,
    _classification_metrics,
    _embedding_batch_effect_metrics,
    _suggest_ae_params as _orig_suggest_ae_params,
)
from bernn.dl.train.train_ae import compute_class_triplet
to_categorical = bm.to_categorical
try:
    from bernn.dl.models.pytorch.utils.utils import ReverseLayerF
except Exception:
    from bernn.dl.train.train_ae_head_sweep import ReverseLayerF

OUT_ROOT = ROOT / "results" / "head_sweep" / "representation_knn"
DLOSS_CHOICES = ["inverseTriplet", "DANN", "revTriplet", "normae", "no"]


def suggest_ae_params_rep(trial, args):
    p = _orig_suggest_ae_params(trial, args)
    # Broaden the continuous regularization. BERNN default range was tiny for gamma.
    if args.dloss in ["revTriplet", "revDANN", "DANN", "inverseTriplet", "normae"]:
        p["gamma"] = trial.suggest_float("gamma", 1e-8, 1e1, log=True)
    if getattr(args, "variational", False):
        p["beta"] = trial.suggest_float("beta", 1e-4, 1e2, log=True)
    if getattr(args, "class_triplet", False):
        p["class_triplet_w"] = trial.suggest_float("class_triplet_w", 1e-4, 10.0, log=True)
    else:
        p["class_triplet_w"] = 0.0
    p["dropout"] = trial.suggest_float("dropout", 0.0, 0.6)
    p["smoothing"] = trial.suggest_float("smoothing", 0.0, 0.3)
    return p

bm._suggest_ae_params = suggest_ae_params_rep


def patched_build_ae(self, layer1:int, layer2:int):
    args = self.args
    n_features = self.data["inputs"]["train"].shape[1]
    cls = KANAutoEncoder2 if getattr(args, "kan", False) else bm.AutoEncoder
    ae = cls(
        n_features,
        n_batches=len(self.unique_batches),
        nb_classes=len(self.unique_labels),
        mapper=getattr(args, "use_mapping", False),
        variational=getattr(args, "variational", False),
        layer1=layer1,
        layer2=layer2,
        dropout=0.0,
        n_layers=2,
        prune_threshold=0.0,
        conditional=False,
        add_noise=0,
        tied_weights=getattr(args, "tied_weights", False),
        update_grid=bool(getattr(args, "kan", False)),
        device=args.device,
    ).to(args.device)
    return ae


def best_knn_metrics(X_train, y_train, X_valid, y_valid, k_max=20):
    best = {"knn_valid_mcc": float("-inf"), "knn_k": None, "knn_metric": None,
            "knn_weights": None, "knn_train_mcc": float("nan")}
    if len(X_train) == 0 or len(X_valid) == 0 or len(np.unique(y_train)) < 2:
        return best
    for metric in ("euclidean", "manhattan", "cosine"):
        for weights in ("uniform", "distance"):
            for k in range(1, min(k_max, len(X_train)) + 1):
                try:
                    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
                    clf.fit(X_train, y_train)
                    pv = clf.predict(X_valid)
                    mcc = float(matthews_corrcoef(y_valid, pv))
                    if mcc > best["knn_valid_mcc"]:
                        pt = clf.predict(X_train)
                        best = {
                            "knn_valid_mcc": mcc,
                            "knn_k": int(k),
                            "knn_metric": metric,
                            "knn_weights": weights,
                            "knn_train_mcc": float(matthews_corrcoef(y_train, pt)),
                            "knn_valid_accuracy": float(accuracy_score(y_valid, pv)),
                            "knn_valid_balanced_accuracy": float(balanced_accuracy_score(y_valid, pv)),
                        }
                except Exception:
                    continue
    return best


def patched_train_ae(self, ae, params, loaders, trial_num=None, wandb_run=None):
    args = self.args
    nu, lr, wd = params["nu"], params["lr"], params["wd"]
    optimizer_ae = get_optimizer(ae, lr, wd, "adam")
    optimizer_c  = get_optimizer(ae.classifier, nu * lr, wd, "adam")
    sceloss, celoss, mseloss, triplet_loss = self._get_losses(ae, params)
    dloss      = getattr(args, "dloss", "inverseTriplet")
    n_epochs   = getattr(args, "n_epochs", 200)
    early_stop = getattr(args, "early_stop", 30)
    k_max      = int(getattr(args, "knn_k_max", 20))
    best_valid_mcc = float("-inf")
    best_state = None
    best_row = {}
    early_stop_counter = 0
    epoch_metrics = []
    tag = f"[trial {trial_num}]" if trial_num is not None else "[AE]"

    for epoch in range(n_epochs):
        ae.train()
        epoch_rec = epoch_d = epoch_c = 0.0
        n_batches = 0
        for batch in loaders.get("train", []):
            raw = batch[:11] if len(batch) >= 11 else (*batch, *([None] * max(0, 11 - len(batch))))
            inputs, _names, labels, domain, to_rec, _not_rec, pos_to_rec, neg_to_rec, pos_batch, neg_batch, _ = raw
            if inputs is None:
                break
            inputs = inputs.to(args.device).float()
            to_rec = to_rec.to(args.device).float() if to_rec is not None else inputs
            optimizer_ae.zero_grad(); optimizer_c.zero_grad()
            enc, rec, _zinb, _kld = ae(inputs, to_rec, domain, sampling=True)
            rec_mean = rec["mean"] if isinstance(rec, dict) else rec
            rec_val = rec_mean[-1] if isinstance(rec_mean, (list, tuple)) else rec_mean
            if enc.abs().sum() == 0:
                continue
            rec_target = to_rec.clamp(0.0, 1.0) if params.get("scaler") == "binarize" else to_rec
            rec_loss_val = mseloss(rec_val, rec_target)
            gamma = params.get("gamma", 0.0)
            d_loss = torch.tensor(0.0, device=args.device)
            if gamma > 0 and dloss != "no" and dloss in ["revTriplet", "inverseTriplet"]:
                if pos_batch is not None and neg_batch is not None:
                    pb = pos_batch.to(args.device).float(); nb = neg_batch.to(args.device).float()
                    pos_enc, _, _, _ = ae(pb, pb, domain, sampling=True)
                    neg_enc, _, _, _ = ae(nb, nb, domain, sampling=True)
                    if dloss == "revTriplet":
                        d_loss = triplet_loss(ReverseLayerF.apply(enc, 1), ReverseLayerF.apply(pos_enc, 1), ReverseLayerF.apply(neg_enc, 1))
                    else:
                        d_loss = triplet_loss(enc, pos_enc, neg_enc)
            cats = to_categorical(labels, len(self.unique_labels)).to(args.device)
            c_loss = sceloss(ae.classifier(enc), cats.argmax(1))
            loss = rec_loss_val + gamma * d_loss + c_loss
            if getattr(args, "class_triplet", False) and pos_to_rec is not None:
                loss = loss + float(params.get("class_triplet_w", getattr(args, "class_triplet_w", 1.0))) * compute_class_triplet(
                    ae, enc, pos_to_rec, neg_to_rec, domain, args.device,
                    margin=max(float(params.get("margin", 1.0)), 1e-6), mapping=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer_ae.step(); optimizer_c.step()
            epoch_rec += float(rec_loss_val.item())
            epoch_d += float(d_loss.item()) if hasattr(d_loss, "item") else float(d_loss)
            epoch_c += float(c_loss.item())
            n_batches += 1
        if n_batches:
            epoch_rec /= n_batches; epoch_d /= n_batches; epoch_c /= n_batches

        ae.eval()
        with torch.no_grad():
            X_train, y_train, _ = extract_embeddings_labels_batches(ae, loaders["train"], args.device)
            X_valid, y_valid, _ = extract_embeddings_labels_batches(ae, loaders["valid"], args.device)
        knn = best_knn_metrics(X_train, y_train, X_valid, y_valid, k_max=k_max)
        valid_mcc = float(knn.get("knn_valid_mcc", float("-inf")))
        row = {"epoch": epoch + 1, "rec": epoch_rec, "d": epoch_d, "c": epoch_c, **knn}
        epoch_metrics.append(row)
        print(f"{tag} epoch={epoch+1:4d}/{n_epochs} rec={epoch_rec:.4f} d={epoch_d:.4f} c={epoch_c:.4f} knn_mcc={valid_mcc:+.4f} k={knn.get('knn_k')} metric={knn.get('knn_metric')} best={best_valid_mcc:+.4f} es={early_stop_counter}/{early_stop}", flush=True)
        if wandb_run is not None:
            try:
                wandb_run.log({f"knn_epoch/{k}": v for k, v in row.items() if isinstance(v, (int, float))}, step=epoch+1)
            except Exception:
                pass
        if valid_mcc > best_valid_mcc:
            best_valid_mcc = valid_mcc
            best_state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
            best_row = row
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop:
                print(f"{tag} early stop at epoch {epoch+1} (patience={early_stop})", flush=True)
                break
    if best_state is not None:
        ae.load_state_dict(best_state)
    self.best_representation_epoch = int(best_row.get("epoch", -1)) if best_row else -1
    self.best_representation_knn = best_row
    return best_valid_mcc, epoch_metrics

AEHeadSweepTrainer._build_ae = patched_build_ae
AEHeadSweepTrainer._train_ae = patched_train_ae


def make_args(dataset, dloss, variational, kan, class_triplet, cli, n_features):
    a = base._make_sweep_args(dataset, dloss, variational, class_triplet, cli, n_features=n_features)
    a.kan = bool(kan)
    a.head_types = ["knn"]
    a.knn_k_max = cli.knn_k_max
    a.exp_id = f"rep_knn_{"kan" if kan else "mlp"}_{"vae" if variational else "ae"}_{dloss.lower()}_{"ct" if class_triplet else "noct"}"
    return a


def run_trial_family(dloss, variational, kan, class_triplet, cli, X, y, batches):
    preset = f"{"kan" if kan else "mlp"}_{"vae" if variational else "ae"}_{dloss.lower()}_{"ct" if class_triplet else "noct"}"
    print("\n" + "="*72)
    print(f"[rep_knn] {preset}")
    print("="*72)
    data, unique_labels, unique_batches = base._build_bernn_data(X, y, batches, seed=cli.seed)
    args = make_args(cli.dataset, dloss, variational, kan, class_triplet, cli, int(X.shape[1]))
    trainer = AEHeadSweepTrainer(args=args, path=str(ROOT / "data"), unique_labels=list(unique_labels), unique_batches=list(unique_batches), data=data, n_cv=cli.n_cv)

    sampler = optuna.samplers.TPESampler(seed=cli.seed, multivariate=True, group=True, n_startup_trials=max(10, cli.n_trials//10))
    # Use Gaussian-process sampler if available/requested; fall back to robust TPE on this Optuna.
    if cli.sampler == "gp" and hasattr(optuna.samplers, "GPSampler"):
        sampler = optuna.samplers.GPSampler(seed=cli.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(study_name=f"be_{preset}", direction="maximize", sampler=sampler, pruner=pruner)

    def cb(study, trial):
        a = trial.user_attrs
        val = trial.value if trial.value is not None else float("nan")
        try:
            best_val = study.best_value
        except Exception:
            best_val = float("nan")
        err = a.get("ae_error") or a.get("embed_error") or a.get("head_error") or ""
        if err:
            err = " err=" + str(err)[:120]
        print("[%s] trial %3d/%d rep_knn_mcc=%.4f best=%.4f best_epoch=%s k=%s%s" % (
            preset, trial.number, cli.n_trials, val, best_val,
            a.get("best_representation_epoch"), a.get("best_knn_k"), err), flush=True)


    # Wrap objective so we can copy trainer attrs to trial attrs after each call.
    orig_objective = trainer.objective
    def objective(trial):
        val = orig_objective(trial)
        rep = getattr(trainer, "best_representation_knn", {}) or {}
        trial.set_user_attr("best_representation_epoch", getattr(trainer, "best_representation_epoch", -1))
        for k, v in rep.items():
            key = f"best_{k}" if not str(k).startswith("knn_") else f"best_{k}"
            try: trial.set_user_attr(key, float(v) if isinstance(v, np.floating) else v)
            except Exception: trial.set_user_attr(key, str(v))
        return val

    study.optimize(objective, n_trials=cli.n_trials, gc_after_trial=True, catch=(Exception,), callbacks=[cb])
    best = study.best_trial
    row = {
        "preset": preset, "dloss": dloss, "variational": bool(variational), "kan": bool(kan), "class_triplet": bool(class_triplet),
        "valid_mcc": float(best.value), "best_trial": int(best.number), "best_epoch": best.user_attrs.get("best_representation_epoch"),
        "best_knn_k": best.user_attrs.get("best_knn_k"), "best_knn_metric": best.user_attrs.get("best_knn_metric"), "best_knn_weights": best.user_attrs.get("best_knn_weights"),
        "params": best.params, "attrs": best.user_attrs,
    }
    out = OUT_ROOT / preset
    out.mkdir(parents=True, exist_ok=True)
    (out / "best.json").write_text(json.dumps(row, indent=2, default=str))
    return row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="massbench_benchmark")
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--n-epochs", type=int, default=1000)
    p.add_argument("--early-stop", type=int, default=30)
    p.add_argument("--n-cv", type=int, default=3)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rec-loss", dest="rec_loss", default="l1", choices=["l1", "mse"])
    p.add_argument("--class-triplet-w", dest="class_triplet_w", type=float, default=1.0)
    p.add_argument("--knn-k-max", type=int, default=20)
    p.add_argument("--dloss-choices", nargs="*", default=DLOSS_CHOICES)
    p.add_argument("--sampler", choices=["tpe", "gp"], default="gp")
    p.add_argument("--family-limit", type=int, default=None, help="Optional debug cap on number of representation families to run.")
    return p.parse_args()


def main():
    cli = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Loading dataset {cli.dataset!r} ...")
    X, y, batches = hp.load_dataset(cli.dataset)
    print(f"  X={X.shape} classes={sorted(set(y))} batches={sorted(set(batches))}")
    print("  representation objective: best per-epoch KNN validation MCC, k=1..20, metrics euclidean/manhattan/cosine, weights uniform/distance")
    print("  scaling/batch correction: data[all] includes train+valid+test, so per-batch maps include every split")
    print("  families: dloss x AE/VAE x MLP/KAN x class_triplet on/off; class_triplet_w optimized when on")
    rows = []
    family_count = 0
    for dloss in cli.dloss_choices:
        for variational in [False, True]:
            for kan in [False, True]:
                for ct in [False, True]:
                    if cli.family_limit is not None and family_count >= cli.family_limit:
                        break
                    rows.append(run_trial_family(dloss, variational, kan, ct, cli, X, y, batches))
                    family_count += 1
                    pd.DataFrame(rows).sort_values("valid_mcc", ascending=False).to_csv(OUT_ROOT / "summary.partial.csv", index=False)
                if cli.family_limit is not None and family_count >= cli.family_limit:
                    break
            if cli.family_limit is not None and family_count >= cli.family_limit:
                break
        if cli.family_limit is not None and family_count >= cli.family_limit:
            break
    summary = pd.DataFrame(rows).sort_values("valid_mcc", ascending=False).reset_index(drop=True)
    summary.to_csv(OUT_ROOT / "summary.csv", index=False)
    (OUT_ROOT / "best.json").write_text(json.dumps(summary.iloc[0].to_dict(), indent=2, default=str))
    print("\nREPRESENTATION KNN SWEEP COMPLETE")
    print(summary[["preset", "valid_mcc", "best_epoch", "best_knn_k", "best_knn_metric", "best_knn_weights"]].to_string(index=False))

if __name__ == "__main__":
    main()
