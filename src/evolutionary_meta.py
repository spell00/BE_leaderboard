"""Deterministic genetic evolution for a dataset-conditioned BERNN policy."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np

DLOSSES = ("no", "DANN", "revDANN", "inverseTriplet", "normae", "revTriplet")
SCALERS = ("standard", "robust", "standard_per_batch", "robust_per_batch")
DEPTHS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class PolicyShape:
    n_inputs: int
    hidden_size: int = 16

    @property
    def n_outputs(self) -> int:
        # categorical logits + three booleans + twelve continuous controls
        return len(DLOSSES) + len(SCALERS) + len(DEPTHS) + 3 + 12

    @property
    def genome_size(self) -> int:
        return (
            self.n_inputs * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.n_outputs
            + self.n_outputs
        )


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 12
    elite_count: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.05
    mutation_scale: float = 0.1
    worst_dataset_weight: float = 0.25

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be in [1, population_size)")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be in [2, population_size]")
        for name in ("crossover_rate", "mutation_rate", "worst_dataset_weight"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.mutation_scale <= 0:
            raise ValueError("mutation_scale must be positive")


def genome_digest(genome: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(genome, dtype=np.float32).tobytes()).hexdigest()


def initialize_population(shape: PolicyShape, config: EvolutionConfig, rng: np.random.Generator) -> np.ndarray:
    config.validate()
    scale = math.sqrt(2.0 / max(shape.n_inputs + shape.hidden_size, 1))
    return rng.normal(0.0, scale, size=(config.population_size, shape.genome_size)).astype(np.float32)


def _unpack(genome: np.ndarray, shape: PolicyShape):
    flat = np.asarray(genome, dtype=np.float32)
    if flat.shape != (shape.genome_size,):
        raise ValueError(f"Expected genome shape {(shape.genome_size,)}, got {flat.shape}")
    cursor = 0
    count = shape.n_inputs * shape.hidden_size
    w1 = flat[cursor:cursor + count].reshape(shape.n_inputs, shape.hidden_size)
    cursor += count
    b1 = flat[cursor:cursor + shape.hidden_size]
    cursor += shape.hidden_size
    count = shape.hidden_size * shape.n_outputs
    w2 = flat[cursor:cursor + count].reshape(shape.hidden_size, shape.n_outputs)
    cursor += count
    b2 = flat[cursor:cursor + shape.n_outputs]
    return w1, b1, w2, b2


def policy_outputs(genome: np.ndarray, meta_features: np.ndarray, shape: PolicyShape) -> np.ndarray:
    values = np.asarray(meta_features, dtype=np.float32)
    if values.shape != (shape.n_inputs,):
        raise ValueError(f"Expected {shape.n_inputs} meta-features, got {values.shape}")
    w1, b1, w2, b2 = _unpack(genome, shape)
    hidden = np.tanh(values @ w1 + b1)
    return hidden @ w2 + b2


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


def _linear(value: float, low: float, high: float) -> float:
    return low + _sigmoid(value) * (high - low)


def _log(value: float, low: float, high: float) -> float:
    return float(math.exp(math.log(low) + _sigmoid(value) * (math.log(high) - math.log(low))))


def decode_config(genome: np.ndarray, meta_features: np.ndarray, shape: PolicyShape, *, max_warmup: int = 50) -> dict:
    """Decode one policy genome into a valid stable joint-BERNN configuration."""
    output = policy_outputs(genome, meta_features, shape)
    cursor = 0
    dloss = DLOSSES[int(np.argmax(output[cursor:cursor + len(DLOSSES)]))]
    cursor += len(DLOSSES)
    scaler = SCALERS[int(np.argmax(output[cursor:cursor + len(SCALERS)]))]
    cursor += len(SCALERS)
    n_layers = DEPTHS[int(np.argmax(output[cursor:cursor + len(DEPTHS)]))]
    cursor += len(DEPTHS)
    variational = bool(output[cursor] >= 0.0)
    class_triplet = bool(output[cursor + 1] >= 0.0)
    log1p = bool(output[cursor + 2] >= 0.0)
    cursor += 3
    controls = output[cursor:cursor + 12]
    config = {
        "model_type": "joint",
        "dloss": dloss,
        "variational": variational,
        "kan": False,
        "class_triplet": class_triplet,
        "log1p": log1p,
        "lr": _log(controls[0], 1e-4, 1e-2),
        "wd": _log(controls[1], 1e-6, 1e-3),
        "nu": _log(controls[2], 1e-4, 1e2),
        "smoothing": _linear(controls[3], 0.0, 0.2),
        "margin": _linear(controls[4], 0.0, 10.0),
        "dropout": _linear(controls[5], 0.0, 0.5),
        "thres": _linear(controls[6], 0.0, 0.1),
        "warmup": round(_linear(controls[7], 1.0, float(max_warmup))),
        "n_layers": n_layers,
        "layer1": round(_linear(controls[8], 512.0, 1024.0)),
        "scaler": scaler,
        "gamma": _log(controls[9], 1e-2, 1e2) if dloss != "no" else 0.0,
        "beta": _log(controls[10], 1e-2, 1e2) if variational else 0.0,
        "class_triplet_w": _log(controls[11], 1e-4, 10.0) if class_triplet else 0.0,
    }
    return config


def normalize_meta_features(train_rows: np.ndarray, rows: np.ndarray | None = None):
    """Fit normalization on training datasets only and transform requested rows."""
    train = np.asarray(train_rows, dtype=np.float64)
    if train.ndim != 2 or not np.isfinite(train).all():
        raise ValueError("Training meta-features must be a finite 2D matrix")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    target = train if rows is None else np.asarray(rows, dtype=np.float64)
    transformed = (target - mean) / scale
    return transformed.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def aggregate_dataset_scores(scores: Iterable[float], worst_dataset_weight: float = 0.25) -> float:
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        raise ValueError("At least one dataset score is required")
    values = np.where(np.isfinite(values), values, -1.0)
    weight = float(worst_dataset_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("worst_dataset_weight must be between 0 and 1")
    return float((1.0 - weight) * values.mean() + weight * values.min())


def recommended_batch_size(batch_labels, cap: int = 128) -> int:
    """Choose a non-empty loader size for leave-one-batch-out training folds."""
    labels = np.asarray(batch_labels).astype(str)
    if labels.size < 2:
        raise ValueError("At least two samples are required")
    _, counts = np.unique(labels, return_counts=True)
    smallest_training_fold = int(labels.size - counts.max()) if len(counts) > 1 else int(labels.size)
    return max(1, min(int(cap), max(1, smallest_training_fold // 2)))


def score_population(
    population: np.ndarray,
    dataset_ids: Iterable[str],
    evaluator: Callable[[np.ndarray, str], float],
    worst_dataset_weight: float,
):
    dataset_ids = tuple(dataset_ids)
    if not dataset_ids:
        raise ValueError("Evolution requires at least one training dataset")
    per_dataset = np.empty((len(population), len(dataset_ids)), dtype=np.float64)
    for genome_index, genome in enumerate(population):
        for dataset_index, dataset_id in enumerate(dataset_ids):
            try:
                score = float(evaluator(genome, dataset_id))
            except Exception:  # noqa: BLE001 - one invalid genome must not kill evolution
                score = -1.0
            per_dataset[genome_index, dataset_index] = score if np.isfinite(score) else -1.0
    fitness = np.asarray([
        aggregate_dataset_scores(row, worst_dataset_weight) for row in per_dataset
    ])
    return fitness, per_dataset


def _tournament(fitness: np.ndarray, size: int, rng: np.random.Generator) -> int:
    contenders = rng.choice(len(fitness), size=size, replace=False)
    return int(contenders[np.argmax(fitness[contenders])])


def breed_next_population(
    population: np.ndarray,
    fitness: np.ndarray,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Breed using train fitness only; validation scores are intentionally absent."""
    config.validate()
    population = np.asarray(population, dtype=np.float32)
    fitness = np.asarray(fitness, dtype=float)
    if population.shape[0] != config.population_size or fitness.shape != (config.population_size,):
        raise ValueError("Population and fitness do not match EvolutionConfig")
    order = np.argsort(fitness)[::-1]
    children = [population[index].copy() for index in order[:config.elite_count]]
    while len(children) < config.population_size:
        left = population[_tournament(fitness, config.tournament_size, rng)]
        right = population[_tournament(fitness, config.tournament_size, rng)]
        if rng.random() < config.crossover_rate:
            mask = rng.random(left.shape) < 0.5
            child = np.where(mask, left, right).astype(np.float32)
        else:
            child = left.copy()
        mutation_mask = rng.random(child.shape) < config.mutation_rate
        child[mutation_mask] += rng.normal(0.0, config.mutation_scale, mutation_mask.sum()).astype(np.float32)
        children.append(child)
    return np.stack(children)


def config_dict(config: EvolutionConfig) -> dict:
    return asdict(config)
