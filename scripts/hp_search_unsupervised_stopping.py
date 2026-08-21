#!/usr/bin/env python3
"""BERNN hyperparameter search with label-free checkpoint selection.

Class labels are used only by the post-hoc oracle diagnostic. They never enter
the representation loss, checkpoint score, early stopping, or Optuna objective.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder


SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import hp_search_head_sweep as head_sweep
from src.unsupervised_monitor import UnsupervisedMonitor, select_checkpoint

import bernn.dl.train.train_ae_head_sweep as bernn_head
from bernn.dl.models.pytorch.utils.utils import get_optimizer
from bernn.dl.models.pytorch import KANAutoEncoder2
from bernn.dl.train.train_ae_head_sweep import AEHeadSweepTrainer
from bernn.utils.utils import scale_data


SUPPORTED_DLOSSES = ("no", "inverseTriplet", "revTriplet")


def _empty_frame(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=columns, dtype=float)


def build_unlabeled_data(X: pd.DataFrame, batches: pd.Series) -> tuple[dict, list[str]]:
    """Build BERNN data whose training labels are a single inert dummy class."""
    X = X.astype(float).reset_index(drop=True)
    batches = batches.astype(str).reset_index(drop=True)
    batch_encoder = LabelEncoder().fit(batches)
    encoded_batches = batch_encoder.transform(batches)
    n = len(X)
    # BERNN's loader requires non-empty valid/test datasets. These are interface
    # sentinels only and are never used for loss or checkpoint selection. The
    # first row remains in train, so no sample is sacrificed.
    sentinel_X = X.iloc[:1].copy()
    sentinel_batch = encoded_batches[:1].copy()
    sentinel_int = np.zeros(1, dtype=int)
    sentinel_label = np.repeat("unlabeled", 1)
    sentinel_name = pd.Series(["sentinel_duplicate"])
    data = {key: {} for key in ("inputs", "batches", "cats", "labels", "names", "orders", "sets")}
    data["inputs"].update({"train": X.copy(), "valid": sentinel_X.copy(), "test": sentinel_X.copy(), "all": X.copy()})
    data["batches"].update({"train": encoded_batches, "valid": sentinel_batch.copy(), "test": sentinel_batch.copy(), "all": encoded_batches.copy()})
    data["cats"].update({"train": np.zeros(n, dtype=int), "valid": sentinel_int.copy(), "test": sentinel_int.copy(), "all": np.zeros(n, dtype=int)})
    data["labels"].update({"train": np.repeat("unlabeled", n), "valid": sentinel_label.copy(), "test": sentinel_label.copy(), "all": np.repeat("unlabeled", n)})
    data["names"].update({"train": pd.Series([f"sample_{i}" for i in range(n)]), "valid": sentinel_name.copy(), "test": sentinel_name.copy(), "all": pd.Series([f"sample_{i}" for i in range(n)])})
    data["orders"].update({"train": np.zeros(n, dtype=int), "valid": sentinel_int.copy(), "test": sentinel_int.copy(), "all": np.zeros(n, dtype=int)})
    data["sets"].update({"train": np.repeat("train", n), "valid": np.repeat("valid", 1), "test": np.repeat("test", 1), "all": np.repeat("all", n)})
    return data, batch_encoder.classes_.astype(str).tolist()


def _latent_and_masked_mse(ae, raw: np.ndarray, batch_ids: np.ndarray, device: str, mask: np.ndarray, bs: int):
    latent_parts = []
    squared_error = 0.0
    masked_count = 0
    ae.eval()
    with torch.no_grad():
        for start in range(0, len(raw), bs):
            stop = min(start + bs, len(raw))
            original = torch.as_tensor(raw[start:stop], dtype=torch.float32, device=device)
            domain = torch.as_tensor(batch_ids[start:stop], dtype=torch.long, device=device)
            chunk_mask = torch.as_tensor(mask[start:stop], dtype=torch.bool, device=device)
            masked = original.clone()
            masked[chunk_mask] = 0.0
            enc, rec, _zinb, _kld = ae(masked, masked, domain, sampling=False)
            rec_mean = rec["mean"] if isinstance(rec, dict) else rec
            reconstructed = rec_mean[-1] if isinstance(rec_mean, (list, tuple)) else rec_mean
            latent_parts.append(enc.detach().cpu().numpy())
            if bool(chunk_mask.any()):
                squared_error += float(torch.square(reconstructed[chunk_mask] - original[chunk_mask]).sum().item())
                masked_count += int(chunk_mask.sum().item())
    return np.concatenate(latent_parts, axis=0), squared_error / max(masked_count, 1)


def _oracle_mcc(latent: np.ndarray, labels: pd.Series, batches: pd.Series, public_count: int, seed: int) -> float:
    """Post-hoc batch-held-out probe; never used for checkpoint selection."""
    y = labels.iloc[:public_count].astype(str).reset_index(drop=True)
    groups = batches.iloc[:public_count].astype(str).reset_index(drop=True)
    scores = []
    for train_idx, valid_idx in LeaveOneGroupOut().split(latent[:public_count], y, groups):
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        model.fit(latent[train_idx], y.iloc[train_idx])
        scores.append(matthews_corrcoef(y.iloc[valid_idx], model.predict(latent[valid_idx])))
    return float(np.mean(scores))


def train_trial(trial, cli, X_learning, batches_learning, public_labels, public_batches, public_count):
    args = head_sweep._make_sweep_args(
        cli.dataset, cli.dloss, cli.variational, False, cli, n_features=X_learning.shape[1]
    )
    args.scaler = cli.scaler
    args.kan = cli.kan
    args.class_triplet = False
    args.head_types = []
    data, unique_batches = build_unlabeled_data(X_learning, batches_learning)
    trainer = AEHeadSweepTrainer(
        args=args,
        path=str(ROOT / "data"),
        unique_labels=["unlabeled"],
        unique_batches=unique_batches,
        data=data,
        n_cv=1,
    )
    params = bernn_head._suggest_ae_params(trial, args)
    params["scaler"] = cli.scaler
    params["class_triplet_w"] = 0.0
    if cli.kan:
        ae = KANAutoEncoder2(
            X_learning.shape[1],
            n_batches=len(unique_batches),
            nb_classes=1,
            mapper=True,
            variational=cli.variational,
            layer1=params["layer1"],
            layer2=params["layer2"],
            dropout=0.0,
            n_layers=2,
            prune_threshold=0.0,
            conditional=False,
            add_noise=0,
            tied_weights=False,
            update_grid=True,
            device=cli.device,
        ).to(cli.device)
    else:
        ae = trainer._build_ae(params["layer1"], params["layer2"])

    scaled_data, _ = scale_data(cli.scaler, copy.deepcopy(data), cli.device)
    for group in scaled_data["inputs"]:
        scaled_data["inputs"][group] = scaled_data["inputs"][group].round(4)
    weights = {"train": [1.0] * len(X_learning), "valid": [1.0], "test": [1.0]}
    try:
        loaders = bernn_head.get_loaders(scaled_data, False, weights, cli.dloss, None, None, cli.bs, cli.device)
    except Exception:
        loaders = bernn_head.get_loaders_no_pool(scaled_data, False, weights, cli.dloss, None, None, cli.bs, cli.device)

    raw = np.asarray(scaled_data["inputs"]["train"], dtype=float)
    batch_ids = np.asarray(scaled_data["batches"]["train"], dtype=int)
    rng = np.random.default_rng(cli.seed)
    mask = rng.random(raw.shape) < cli.mask_fraction
    mask[:, 0] = True
    monitor = UnsupervisedMonitor(raw, batch_ids, k=cli.monitor_k)
    optimizer = get_optimizer(ae, params["lr"], params["wd"], "adam")
    _sceloss, _celoss, reconstruction_loss, triplet_loss = trainer._get_losses(ae, params)
    rows = []
    states = []
    patience = 0

    for epoch in range(1, cli.n_epochs + 1):
        ae.train()
        rec_total = domain_total = 0.0
        n_batches = 0
        for batch in loaders["train"]:
            raw_batch = batch[:11] if len(batch) >= 11 else (*batch, *([None] * max(0, 11 - len(batch))))
            inputs, _names, _labels, domain, to_rec, _not_rec, _pos_class, _neg_class, pos_batch, neg_batch, _ = raw_batch
            if inputs is None:
                continue
            inputs = inputs.to(cli.device).float()
            target = to_rec.to(cli.device).float() if to_rec is not None else inputs
            optimizer.zero_grad()
            enc, rec, _zinb, kld = ae(inputs, target, domain, sampling=True)
            rec_mean = rec["mean"] if isinstance(rec, dict) else rec
            reconstructed = rec_mean[-1] if isinstance(rec_mean, (list, tuple)) else rec_mean
            rec_target = target.clamp(0.0, 1.0) if cli.scaler == "binarize" else target
            rec_value = reconstruction_loss(reconstructed, rec_target)
            domain_value = torch.tensor(0.0, device=cli.device)
            if cli.dloss in ("inverseTriplet", "revTriplet") and pos_batch is not None and neg_batch is not None:
                positive = pos_batch.to(cli.device).float()
                negative = neg_batch.to(cli.device).float()
                positive_enc, _, _, _ = ae(positive, positive, domain, sampling=True)
                negative_enc, _, _, _ = ae(negative, negative, domain, sampling=True)
                if cli.dloss == "revTriplet":
                    try:
                        from bernn.dl.models.pytorch.utils.utils import ReverseLayerF
                    except Exception:
                        ReverseLayerF = bernn_head.ReverseLayerF
                    domain_value = triplet_loss(ReverseLayerF.apply(enc, 1), ReverseLayerF.apply(positive_enc, 1), ReverseLayerF.apply(negative_enc, 1))
                else:
                    domain_value = triplet_loss(enc, positive_enc, negative_enc)
            loss = rec_value + float(params.get("gamma", 0.0)) * domain_value
            if cli.variational and kld is not None:
                kld_value = kld if torch.is_tensor(kld) else torch.as_tensor(kld, device=cli.device)
                loss = loss + float(params.get("beta", 0.0)) * kld_value.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer.step()
            rec_total += float(rec_value.item())
            domain_total += float(domain_value.item())
            n_batches += 1

        latent, masked_mse = _latent_and_masked_mse(ae, raw, batch_ids, cli.device, mask, cli.bs)
        label_free_metrics = monitor.score(latent, masked_mse)
        row = {
            "epoch": epoch,
            "train_reconstruction_loss": rec_total / max(n_batches, 1),
            "train_domain_loss": domain_total / max(n_batches, 1),
            **label_free_metrics,
            "oracle_mcc": _oracle_mcc(latent, public_labels, public_batches, public_count, cli.seed),
        }
        rows.append(row)
        states.append({key: value.detach().cpu().clone() for key, value in ae.state_dict().items()})
        selected = select_checkpoint(rows)
        patience = 0 if selected["epoch"] == epoch else patience + 1
        print(
            f"[trial {trial.number}] epoch={epoch}/{cli.n_epochs} unsup={row['unsupervised_score']:.4f} "
            f"recon={row['masked_reconstruction_quality']:.4f} neigh={row['neighborhood_preservation']:.4f} "
            f"rank={row['rank_retention']:.4f} mix={row['batch_mixing_entropy']:.4f} "
            f"oracle(posthoc)={row['oracle_mcc']:.4f}", flush=True,
        )
        if patience >= cli.early_stop:
            break

    selected = select_checkpoint(rows)
    oracle = max(rows, key=lambda row: row["oracle_mcc"])
    ae.load_state_dict(states[int(selected["epoch"]) - 1])
    trial.set_user_attr("selected_epoch", int(selected["epoch"]))
    trial.set_user_attr("selected_oracle_mcc", float(selected["oracle_mcc"]))
    trial.set_user_attr("oracle_best_epoch", int(oracle["epoch"]))
    trial.set_user_attr("oracle_best_mcc", float(oracle["oracle_mcc"]))
    trial.set_user_attr("oracle_gap", float(oracle["oracle_mcc"] - selected["oracle_mcc"]))
    trial.set_user_attr("epoch_metrics_json", json.dumps(rows))
    return float(selected["unsupervised_score"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="massbench_benchmark")
    parser.add_argument("--mode", choices=("inductive", "transductive"), default="transductive")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--early-stop", type=int, default=30)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dloss", choices=SUPPORTED_DLOSSES, default="inverseTriplet")
    parser.add_argument("--scaler", default="standard")
    parser.add_argument("--variational", action="store_true")
    parser.add_argument("--kan", action="store_true")
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--monitor-k", type=int, default=15)
    parser.add_argument("--rec-loss", dest="rec_loss", default="l1", choices=("l1", "mse"))
    parser.add_argument("--class-triplet-w", dest="class_triplet_w", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "unsupervised_stopping")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    X_train, y_train, batches_train, _names_train, X_test, _y_test, batches_test, _names_test = head_sweep._load_train_fixed_test_dataset(cli.dataset)
    if cli.mode == "transductive":
        X_learning = pd.concat([X_train, X_test], ignore_index=True)
        batches_learning = pd.concat([batches_train, batches_test], ignore_index=True)
    else:
        X_learning, batches_learning = X_train.copy(), batches_train.copy()
    print(
        f"[unsupervised-stopping] mode={cli.mode} learning_samples={len(X_learning)} "
        f"public_labeled_samples={len(X_train)} class_labels_in_loss=False", flush=True,
    )
    sampler = optuna.samplers.TPESampler(seed=cli.seed)
    suffix = f"{cli.dataset}_{cli.mode}"
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(cli.output_dir / f'{suffix}_study.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name=f"unsupervised_stopping_{suffix}", direction="maximize",
        sampler=sampler, storage=storage, load_if_exists=True,
    )

    def callback(current_study, trial):
        current_study.trials_dataframe().to_csv(cli.output_dir / f"{suffix}_trials.csv", index=False)
        if os.getenv("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
            return
        try:
            import wandb
            run = wandb.init(
                project=os.getenv("WANDB_UNSUPERVISED_PROJECT", "BE_leaderboard_unsupervised_stopping"),
                entity=os.getenv("WANDB_ENTITY", "adlab"),
                group=f"{cli.dataset}_{cli.mode}", name=f"trial_{trial.number}",
                job_type="label_free_hpo",
                config={
                    "dataset_id": cli.dataset, "mode": cli.mode,
                    "selection_uses_class_labels": False, **trial.params,
                }, reinit="finish_previous",
            )
            payload = {
                "unsupervised_score": trial.value,
                "selected_probe_mcc": trial.user_attrs.get("selected_oracle_mcc"),
                "oracle_best_probe_mcc": trial.user_attrs.get("oracle_best_mcc"),
                "oracle_gap": trial.user_attrs.get("oracle_gap"),
                "selected_epoch": trial.user_attrs.get("selected_epoch"),
            }
            run.log({key: value for key, value in payload.items() if value is not None})
            run.summary["dataset_id"] = cli.dataset
            run.summary["mode"] = cli.mode
            run.finish()
        except Exception as exc:
            print(f"[wandb] unsupervised trial {trial.number} logging failed: {exc}", flush=True)

    study.optimize(
        lambda trial: train_trial(
            trial, cli, X_learning, batches_learning,
            y_train, batches_train, len(X_train),
        ),
        n_trials=cli.n_trials,
        gc_after_trial=True,
        callbacks=[callback],
    )
    study.trials_dataframe().to_csv(cli.output_dir / f"{suffix}_trials.csv", index=False)
    result = {
        "dataset_id": cli.dataset,
        "mode": cli.mode,
        "selection_uses_class_labels": False,
        "best_trial": study.best_trial.number,
        "unsupervised_score": study.best_value,
        "params": study.best_trial.params,
        "attrs": study.best_trial.user_attrs,
    }
    (cli.output_dir / f"{suffix}_best.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
