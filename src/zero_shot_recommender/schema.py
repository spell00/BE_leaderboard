"""Canonical model-family and conditional hyperparameter schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CONFIG_SCHEMA = {
    "categorical": {
        "model_family": [
            "representation_knn", "representation_prototype",
            "frozen_ae_classical", "two_stage_nn", "joint_nn",
            "kan_two_stage", "kan_joint",
        ],
        "head_type": [
            "none", "nn", "knn", "logistic_regression", "linear_svc",
            "svc_rbf", "random_forest", "xgboost", "gradient_boosting",
            "prototype_mean", "prototype_kmeans",
        ],
        "dloss": ["no", "inverseTriplet", "revTriplet", "DANN", "normae"],
        "scaler": ["standard", "robust", "minmax", "none"],
        "optimizer": ["adam", "adamw", "sgd"],
    },
    "boolean": ["variational", "kan", "class_triplet", "transductive"],
    "continuous": {
        "layer1": {"transform": "log", "minimum": 4.0, "maximum": 2048.0},
        "layer2": {"transform": "log", "minimum": 2.0, "maximum": 1024.0},
        "lr": {"transform": "log", "minimum": 1e-6, "maximum": 1e-1},
        "wd": {"transform": "log", "minimum": 1e-10, "maximum": 1e-1},
        "dropout": {"transform": "linear", "minimum": 0.0, "maximum": 0.8},
        "gamma": {"transform": "log", "minimum": 1e-10, "maximum": 10.0},
        "beta": {"transform": "log", "minimum": 1e-6, "maximum": 100.0},
        "margin": {"transform": "log", "minimum": 1e-4, "maximum": 20.0},
        "class_triplet_w": {"transform": "log", "minimum": 1e-6, "maximum": 20.0},
        "batch_size": {"transform": "log", "minimum": 4.0, "maximum": 1024.0},
        "n_epochs": {"transform": "log", "minimum": 10.0, "maximum": 10000.0},
        "knn_k": {"transform": "log", "minimum": 1.0, "maximum": 101.0},
        "head_C": {"transform": "log", "minimum": 1e-6, "maximum": 1e6},
        "n_estimators": {"transform": "log", "minimum": 10.0, "maximum": 5000.0},
        "max_depth": {"transform": "linear", "minimum": 1.0, "maximum": 64.0},
    },
}


FAMILY_DEFAULT_HEAD = {
    "representation_knn": "knn",
    "representation_prototype": "prototype_mean",
    "frozen_ae_classical": "knn",
    "two_stage_nn": "nn",
    "joint_nn": "nn",
    "kan_two_stage": "nn",
    "kan_joint": "nn",
}


def conditional_continuous_mask(config: dict[str, Any]) -> dict[str, bool]:
    """Return which regression targets are meaningful for this configuration."""
    head = str(config.get("head_type", "none"))
    dloss = str(config.get("dloss", "no"))
    return {
        "layer1": True,
        "layer2": True,
        "lr": True,
        "wd": True,
        "dropout": True,
        "gamma": dloss != "no",
        "beta": bool(config.get("variational", False)),
        "margin": dloss in {"inverseTriplet", "revTriplet"},
        "class_triplet_w": bool(config.get("class_triplet", False)),
        "batch_size": True,
        "n_epochs": True,
        "knn_k": head == "knn",
        "head_C": head in {"logistic_regression", "linear_svc", "svc_rbf"},
        "n_estimators": head in {"random_forest", "xgboost", "gradient_boosting"},
        "max_depth": head in {"random_forest", "xgboost", "gradient_boosting"},
    }


@dataclass
class TrialRecord:
    dataset_id: str
    model_family: str
    score: float
    config: dict[str, Any]
    status: str = "complete"
    seed: int = 42
    metrics: dict[str, Any] = field(default_factory=dict)
    source: str = "unified_search"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrialRecord":
        return cls(**value)
