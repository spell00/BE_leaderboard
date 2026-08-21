"""One-hidden-layer mixed-output hyperparameter recommender."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .schema import CONFIG_SCHEMA, conditional_continuous_mask


class MixedOutputRecommender(nn.Module):
    """Shared shallow representation with categorical, boolean and regression heads."""

    def __init__(self, n_meta_features: int, hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_meta_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.categorical_heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, len(options))
            for name, options in CONFIG_SCHEMA["categorical"].items()
        })
        self.boolean_head = nn.Linear(hidden_size, len(CONFIG_SCHEMA["boolean"]))
        self.continuous_head = nn.Linear(hidden_size, len(CONFIG_SCHEMA["continuous"]))

    def forward(self, meta_features: torch.Tensor) -> dict[str, Any]:
        hidden = self.shared(meta_features)
        return {
            "categorical": {name: head(hidden) for name, head in self.categorical_heads.items()},
            "boolean": self.boolean_head(hidden),
            "continuous": self.continuous_head(hidden),
        }


def _continuous_to_unit(name: str, value: float) -> float:
    spec = CONFIG_SCHEMA["continuous"][name]
    low, high = float(spec["minimum"]), float(spec["maximum"])
    value = float(np.clip(value, low, high))
    if spec["transform"] == "log":
        return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    return (value - low) / (high - low)


def _continuous_from_unit(name: str, value: float) -> float:
    spec = CONFIG_SCHEMA["continuous"][name]
    low, high = float(spec["minimum"]), float(spec["maximum"])
    value = float(np.clip(value, 0.0, 1.0))
    if spec["transform"] == "log":
        return float(math.exp(math.log(low) + value * (math.log(high) - math.log(low))))
    return float(low + value * (high - low))


def encode_config(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    categorical = []
    for name, options in CONFIG_SCHEMA["categorical"].items():
        value = str(config.get(name, options[0]))
        categorical.append(options.index(value) if value in options else 0)
    booleans = [float(bool(config.get(name, False))) for name in CONFIG_SCHEMA["boolean"]]
    masks = conditional_continuous_mask(config)
    continuous, continuous_mask = [], []
    for name, spec in CONFIG_SCHEMA["continuous"].items():
        fallback = spec["minimum"]
        continuous.append(_continuous_to_unit(name, float(config.get(name, fallback))))
        continuous_mask.append(float(masks[name] and name in config))
    return {
        "categorical": torch.tensor(categorical, dtype=torch.long),
        "boolean": torch.tensor(booleans, dtype=torch.float32),
        "continuous": torch.tensor(continuous, dtype=torch.float32),
        "continuous_mask": torch.tensor(continuous_mask, dtype=torch.float32),
    }


def mixed_output_loss(outputs: dict[str, Any], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    categorical_losses = []
    for index, name in enumerate(CONFIG_SCHEMA["categorical"]):
        categorical_losses.append(nn.functional.cross_entropy(outputs["categorical"][name], targets["categorical"][:, index]))
    boolean_loss = nn.functional.binary_cross_entropy_with_logits(outputs["boolean"], targets["boolean"])
    predicted_continuous = torch.sigmoid(outputs["continuous"])
    raw_continuous = nn.functional.smooth_l1_loss(predicted_continuous, targets["continuous"], reduction="none")
    mask = targets["continuous_mask"]
    continuous_loss = (raw_continuous * mask).sum() / mask.sum().clamp_min(1.0)
    return torch.stack(categorical_losses).mean() + boolean_loss + continuous_loss


def decode_prediction(outputs: dict[str, Any], row: int = 0) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name, options in CONFIG_SCHEMA["categorical"].items():
        index = int(outputs["categorical"][name][row].argmax().item())
        config[name] = options[index]
    for index, name in enumerate(CONFIG_SCHEMA["boolean"]):
        config[name] = bool(torch.sigmoid(outputs["boolean"][row, index]).item() >= 0.5)
    masks = conditional_continuous_mask(config)
    units = torch.sigmoid(outputs["continuous"][row]).detach().cpu().numpy()
    for index, name in enumerate(CONFIG_SCHEMA["continuous"]):
        if masks[name]:
            value = _continuous_from_unit(name, float(units[index]))
            if name in {"layer1", "layer2", "batch_size", "n_epochs", "knn_k", "n_estimators", "max_depth"}:
                value = int(round(value))
            config[name] = value
    return config
