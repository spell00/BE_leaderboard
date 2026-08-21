"""Training and serialization helpers for the shallow recommender."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .meta_features import META_FEATURE_NAMES
from .model import MixedOutputRecommender, decode_prediction, encode_config, mixed_output_loss
from .schema import CONFIG_SCHEMA, TrialRecord


def train_recommender(
    records: list[TrialRecord],
    meta_features: dict[str, dict[str, float]],
    *,
    hidden_size: int = 64,
    epochs: int = 500,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[MixedOutputRecommender, dict[str, Any]]:
    if not records:
        raise ValueError("No trial records were supplied")
    missing = sorted({record.dataset_id for record in records} - set(meta_features))
    if missing:
        raise ValueError(f"Missing meta-features for datasets: {missing}")
    torch.manual_seed(seed)
    X_raw = np.asarray([[meta_features[r.dataset_id][name] for name in META_FEATURE_NAMES] for r in records], dtype=np.float32)
    mean = X_raw.mean(axis=0)
    scale = X_raw.std(axis=0)
    scale[scale < 1e-8] = 1.0
    X = torch.tensor((X_raw - mean) / scale, dtype=torch.float32)
    encoded = [encode_config(record.config) for record in records]
    targets = {name: torch.stack([row[name] for row in encoded]) for name in encoded[0]}
    model = MixedOutputRecommender(len(META_FEATURE_NAMES), hidden_size=hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = mixed_output_loss(model(X), targets)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    metadata = {
        "meta_feature_names": META_FEATURE_NAMES,
        "meta_mean": mean.tolist(),
        "meta_scale": scale.tolist(),
        "schema": CONFIG_SCHEMA,
        "hidden_size": hidden_size,
        "training_rows": len(records),
        "dataset_ids": sorted({record.dataset_id for record in records}),
        "loss_history": history,
    }
    return model, metadata


def save_recommender(model: MixedOutputRecommender, metadata: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_recommender(output_dir: str | Path) -> tuple[MixedOutputRecommender, dict[str, Any]]:
    output = Path(output_dir)
    metadata = json.loads((output / "metadata.json").read_text())
    model = MixedOutputRecommender(len(metadata["meta_feature_names"]), hidden_size=int(metadata["hidden_size"]))
    model.load_state_dict(torch.load(output / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, metadata


def recommend(model, metadata, dataset_meta: dict[str, float]) -> dict[str, Any]:
    raw = np.asarray([[dataset_meta[name] for name in metadata["meta_feature_names"]]], dtype=np.float32)
    mean = np.asarray(metadata["meta_mean"], dtype=np.float32)
    scale = np.asarray(metadata["meta_scale"], dtype=np.float32)
    inputs = torch.tensor((raw - mean) / scale, dtype=torch.float32)
    with torch.no_grad():
        return decode_prediction(model(inputs))
