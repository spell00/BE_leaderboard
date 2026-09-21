"""Portable inference checkpoints for the synchronized direct meta network."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

from src.meta_hpo_models import (
    BOOLEAN_FIELDS, CATEGORICAL_CHOICES, CONTINUOUS_FIELDS,
    DirectBERNNMetaModel, _decode_direct,
)
from src.zero_shot_recommender.meta_features import META_FEATURE_NAMES, extract_meta_features


def _schema():
    return {
        "categorical_choices": {k: list(v) for k, v in CATEGORICAL_CHOICES.items()},
        "boolean_fields": list(BOOLEAN_FIELDS),
        "continuous_fields": list(CONTINUOUS_FIELDS),
    }


def save_direct_meta_checkpoint(path, model, *, meta_mean, meta_scale,
                                max_warmup, metadata):
    """Atomically save weights, inference normalization, decoding schema and provenance."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},
        "n_meta": model.model.shared[0].in_features,
        "hidden_size": model.model.shared[0].out_features,
        "fixed_categories": model.fixed,
        "meta_feature_names": list(META_FEATURE_NAMES),
        "meta_mean": np.asarray(meta_mean, dtype=np.float32).tolist(),
        "meta_scale": np.asarray(meta_scale, dtype=np.float32).tolist(),
        "max_warmup": int(max_warmup),
        "schema": _schema(),
        "torch_version": str(torch.__version__),
        "metadata": metadata,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            torch.save(checkpoint, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_direct_meta_checkpoint(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint["schema"] != _schema():
        raise ValueError("Unsupported direct meta checkpoint format or decoding schema")
    if checkpoint["meta_feature_names"] != list(META_FEATURE_NAMES):
        raise ValueError("Checkpoint meta-feature names/order differ from this code version")
    n_meta = checkpoint["n_meta"]
    mean = np.asarray(checkpoint["meta_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["meta_scale"], dtype=np.float32)
    if (n_meta != len(META_FEATURE_NAMES) or mean.shape != (n_meta,)
            or scale.shape != (n_meta,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Invalid checkpoint normalization")
    model = DirectBERNNMetaModel(n_meta, checkpoint["hidden_size"], checkpoint["fixed_categories"])
    model.model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def predict_direct_meta_config(model, checkpoint, meta_features):
    """Predict from raw meta-features using the checkpoint's inference transform."""
    import torch

    if isinstance(meta_features, dict):
        meta_features = [meta_features[name] for name in checkpoint["meta_feature_names"]]
    values = np.asarray(meta_features, dtype=np.float32)
    if values.shape != (checkpoint["n_meta"],) or not np.isfinite(values).all():
        raise ValueError("Expected one finite vector of raw dataset meta-features")
    mean = np.asarray(checkpoint["meta_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["meta_scale"], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        outputs = model.forward(torch.tensor(((values - mean) / scale)[None, :]))
    return _decode_direct(outputs, 0, checkpoint["max_warmup"], checkpoint["fixed_categories"])


def predict_from_checkpoint(path, X, y, batches):
    """Recommend BERNN hyperparameters for a dataset using a saved meta network."""
    model, checkpoint = load_direct_meta_checkpoint(path)
    return predict_direct_meta_config(model, checkpoint, extract_meta_features(X, y, batches))


def save_selected_meta_model(output_dir, model, *, meta_mean, meta_scale,
                             max_warmup, metadata):
    """Keep every selected round and separately named best-by-metric checkpoints."""
    import torch

    directory = Path(output_dir) / "meta_checkpoints"
    kwargs = dict(meta_mean=meta_mean, meta_scale=meta_scale,
                  max_warmup=max_warmup, metadata=metadata)
    path = save_direct_meta_checkpoint(
        directory / f"round_{int(metadata['round']):04d}.pt", model, **kwargs)
    for filename, metric, maximize in (
        ("best_benchmark.pt", "benchmark_prediction_error", False),
        ("best_alzheimer.pt", "alzheimer_valid_mcc", True),
    ):
        score = metadata.get(metric)
        if score is None or not np.isfinite(score):
            continue
        best_path = directory / filename
        if best_path.exists():
            prior = torch.load(best_path, map_location="cpu", weights_only=True)["metadata"]
            previous = prior[metric]
            same_round = metadata["round"] == prior["round"] and score == previous
            if not same_round and ((maximize and score <= previous) or (not maximize and score >= previous)):
                continue
        save_direct_meta_checkpoint(best_path, model, **kwargs)
    return path
