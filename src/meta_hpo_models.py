"""Cheap meta-HPO models trained from a frozen BERNN trial bank.

No function in this module accepts Alzheimer MCC as a training target.  The
caller may pass Alzheimer *meta-features* for zero-shot recommendation, but real
Alzheimer scores remain external validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from scripts import hp_search
from src.meta_hpo_bank import (
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    DLOSS_CHOICES,
    N_LAYER_CHOICES,
    SCALER_CHOICES,
    BankTrial,
    apply_fixed_categories,
    canonical_config,
    config_feature_vector,
    decode_continuous,
    encode_continuous,
    sample_config,
)

BOOLEAN_FIELDS = ("variational", "kan", "class_triplet")
CATEGORICAL_CHOICES = {
    "dloss": DLOSS_CHOICES,
    "scaler": SCALER_CHOICES,
    "n_layers": N_LAYER_CHOICES,
}


@dataclass
class DirectEpochRecord:
    epoch: int
    train_loss: float
    target_config: dict[str, Any]


@dataclass
class SurrogateProposal:
    config: dict[str, Any]
    predicted_mcc: float
    predicted_std: float
    acquisition: float


@dataclass
class RLEpochRecord:
    epoch: int
    mean_reward: float
    best_reward: float
    target_config: dict[str, Any]
    predicted_mcc: float
    predicted_std: float


def normalize_source_meta(source_meta: dict[str, np.ndarray], dataset_ids: Iterable[str]):
    dataset_ids = tuple(dataset_ids)
    matrix = np.stack([np.asarray(source_meta[name], dtype=np.float32) for name in dataset_ids])
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = {name: (np.asarray(source_meta[name], dtype=np.float32) - mean) / scale for name in dataset_ids}
    return normalized, mean.astype(np.float32), scale.astype(np.float32)


class DirectBERNNMetaModel:
    """Mixed-output metadata -> best-known BERNN configuration network."""

    def __init__(self, n_meta: int, hidden_size: int, fixed_categories: dict[str, Any] | None = None):
        import torch
        from torch import nn

        self.fixed = dict(fixed_categories or {})
        self.model = nn.Module()
        self.model.shared = nn.Sequential(
            nn.Linear(int(n_meta), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.ReLU(),
        )
        self.model.cat_heads = nn.ModuleDict({
            name: nn.Linear(int(hidden_size), len(CATEGORICAL_CHOICES[name]))
            for name in CATEGORICAL_CHOICES
            if name not in self.fixed
        })
        self.model.bool_heads = nn.ModuleDict({
            name: nn.Linear(int(hidden_size), 1)
            for name in BOOLEAN_FIELDS
            if name not in self.fixed
        })
        self.model.cont_head = nn.Linear(int(hidden_size), len(CONTINUOUS_FIELDS))

        def forward(x):
            hidden = self.model.shared(x)
            return {
                "cat": {name: head(hidden) for name, head in self.model.cat_heads.items()},
                "bool": {name: head(hidden).squeeze(-1) for name, head in self.model.bool_heads.items()},
                "cont": self.model.cont_head(hidden),
            }

        self.forward = forward

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


def _direct_targets(best_trials: dict[str, BankTrial], dataset_ids: tuple[str, ...], max_warmup: int):
    import torch

    categorical = {}
    for name, choices in CATEGORICAL_CHOICES.items():
        categorical[name] = torch.tensor([
            choices.index(best_trials[dataset_id].config[name])
            for dataset_id in dataset_ids
        ], dtype=torch.long)
    booleans = {
        name: torch.tensor([
            float(bool(best_trials[dataset_id].config[name]))
            for dataset_id in dataset_ids
        ], dtype=torch.float32)
        for name in BOOLEAN_FIELDS
    }
    cont_values, cont_masks = [], []
    for dataset_id in dataset_ids:
        values, mask = encode_continuous(best_trials[dataset_id].config, max_warmup)
        cont_values.append(values)
        cont_masks.append(mask)
    return {
        "cat": categorical,
        "bool": booleans,
        "cont": torch.tensor(np.stack(cont_values), dtype=torch.float32),
        "cont_mask": torch.tensor(np.stack(cont_masks), dtype=torch.float32),
    }


def _direct_loss(outputs, targets, fixed_categories):
    import torch
    from torch import nn

    parts = []
    for name, logits in outputs["cat"].items():
        if name in fixed_categories:
            continue
        parts.append(nn.functional.cross_entropy(logits, targets["cat"][name]))
    for name, logits in outputs["bool"].items():
        if name in fixed_categories:
            continue
        parts.append(nn.functional.binary_cross_entropy_with_logits(logits, targets["bool"][name]))
    predicted_cont = torch.sigmoid(outputs["cont"])
    raw = nn.functional.smooth_l1_loss(predicted_cont, targets["cont"], reduction="none")
    masked = (raw * targets["cont_mask"]).sum() / targets["cont_mask"].sum().clamp_min(1.0)
    parts.append(masked)
    return torch.stack(parts).mean()


def _decode_direct(outputs, row: int, max_warmup: int, fixed_categories: dict[str, Any] | None):
    import torch

    fixed = dict(fixed_categories or {})
    config: dict[str, Any] = {
        "model_type": "joint",
        "log1p": True,
    }
    for name, choices in CATEGORICAL_CHOICES.items():
        if name in fixed:
            config[name] = fixed[name]
        else:
            config[name] = choices[int(outputs["cat"][name][row].argmax().item())]
    for name in BOOLEAN_FIELDS:
        if name in fixed:
            config[name] = bool(fixed[name])
        else:
            config[name] = bool(torch.sigmoid(outputs["bool"][name][row]).item() >= 0.5)
    cont = torch.sigmoid(outputs["cont"][row]).detach().cpu().numpy()
    config = decode_continuous(cont, config, max_warmup)
    return apply_fixed_categories(config, fixed)


def train_direct_meta_model(
    best_trials: dict[str, BankTrial],
    source_meta: dict[str, np.ndarray],
    target_meta: np.ndarray,
    *,
    max_warmup: int,
    fixed_categories: dict[str, Any] | None = None,
    hidden_size: int = 64,
    epochs: int = 1000,
    lr: float = 1e-2,
    seed: int = 42,
    record_every_epoch: bool = True,
    callback: Callable[[DirectEpochRecord], None] | None = None,
) -> tuple[DirectBERNNMetaModel, list[DirectEpochRecord], dict[str, Any]]:
    """Reinitialize and fit metadata -> current best source configs from scratch."""
    import torch

    dataset_ids = tuple(best_trials)
    normalized, mean, scale = normalize_source_meta(source_meta, dataset_ids)
    X = torch.tensor(np.stack([normalized[name] for name in dataset_ids]), dtype=torch.float32)
    target_x = torch.tensor(((np.asarray(target_meta, dtype=np.float32) - mean) / scale)[None, :], dtype=torch.float32)
    targets = _direct_targets(best_trials, dataset_ids, max_warmup)

    torch.manual_seed(int(seed))
    model = DirectBERNNMetaModel(X.shape[1], hidden_size, fixed_categories)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-5)
    history: list[DirectEpochRecord] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward(X)
        loss = _direct_loss(outputs, targets, model.fixed)
        loss.backward()
        optimizer.step()

        if record_every_epoch or epoch == int(epochs):
            model.eval()
            with torch.no_grad():
                target_outputs = model.forward(target_x)
                config = _decode_direct(target_outputs, 0, max_warmup, model.fixed)
            record = DirectEpochRecord(epoch=epoch, train_loss=float(loss.detach().cpu()), target_config=config)
            history.append(record)
            if callback is not None:
                callback(record)

    metadata = {
        "dataset_ids": list(dataset_ids),
        "meta_mean": mean.tolist(),
        "meta_scale": scale.tolist(),
        "hidden_size": int(hidden_size),
        "epochs": int(epochs),
        "fixed_categories": dict(fixed_categories or {}),
        "final_train_loss": history[-1].train_loss,
    }
    return model, history, metadata


class MLPSurrogateEnsemble:
    def __init__(self, models, meta_mean, meta_scale, score_mean, score_scale, max_warmup):
        self.models = list(models)
        self.meta_mean = np.asarray(meta_mean, dtype=np.float32)
        self.meta_scale = np.asarray(meta_scale, dtype=np.float32)
        self.score_mean = float(score_mean)
        self.score_scale = float(score_scale)
        self.max_warmup = int(max_warmup)

    def _X(self, meta_rows: np.ndarray, configs: list[dict[str, Any]]) -> np.ndarray:
        meta_rows = np.asarray(meta_rows, dtype=np.float32)
        if meta_rows.ndim == 1:
            meta_rows = np.repeat(meta_rows[None, :], len(configs), axis=0)
        scaled = (meta_rows - self.meta_mean) / self.meta_scale
        cfg = np.stack([config_feature_vector(config, self.max_warmup) for config in configs])
        return np.concatenate([scaled, cfg], axis=1).astype(np.float32)

    def predict(self, meta_rows: np.ndarray, configs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        X = torch.tensor(self._X(meta_rows, configs), dtype=torch.float32)
        members = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                values = model(X).squeeze(-1).cpu().numpy() * self.score_scale + self.score_mean
                members.append(values)
        matrix = np.stack(members)
        return matrix.mean(axis=0), matrix.std(axis=0)


class ExtraTreesSurrogate:
    def __init__(self, model, meta_mean, meta_scale, max_warmup):
        self.model = model
        self.meta_mean = np.asarray(meta_mean, dtype=np.float32)
        self.meta_scale = np.asarray(meta_scale, dtype=np.float32)
        self.max_warmup = int(max_warmup)

    def _X(self, meta_rows: np.ndarray, configs: list[dict[str, Any]]) -> np.ndarray:
        meta_rows = np.asarray(meta_rows, dtype=np.float32)
        if meta_rows.ndim == 1:
            meta_rows = np.repeat(meta_rows[None, :], len(configs), axis=0)
        scaled = (meta_rows - self.meta_mean) / self.meta_scale
        cfg = np.stack([config_feature_vector(config, self.max_warmup) for config in configs])
        return np.concatenate([scaled, cfg], axis=1).astype(np.float32)

    def predict(self, meta_rows: np.ndarray, configs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        X = self._X(meta_rows, configs)
        per_tree = np.stack([tree.predict(X) for tree in self.model.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)


def _surrogate_matrix(trials: list[BankTrial], source_meta: dict[str, np.ndarray], max_warmup: int):
    dataset_ids = sorted({trial.dataset_id for trial in trials})
    unique_meta = np.stack([np.asarray(source_meta[name], dtype=np.float32) for name in dataset_ids])
    meta_mean = unique_meta.mean(axis=0)
    meta_scale = unique_meta.std(axis=0)
    meta_scale[meta_scale < 1e-8] = 1.0
    X = np.stack([
        np.concatenate([
            (np.asarray(source_meta[trial.dataset_id], dtype=np.float32) - meta_mean) / meta_scale,
            config_feature_vector(trial.config, max_warmup),
        ])
        for trial in trials
    ]).astype(np.float32)
    y = np.asarray([trial.valid_mcc for trial in trials], dtype=np.float32)
    return X, y, meta_mean, meta_scale


def fit_mlp_surrogate(
    trials: Iterable[BankTrial],
    source_meta: dict[str, np.ndarray],
    *,
    max_warmup: int,
    hidden_size: int = 128,
    epochs: int = 1000,
    lr: float = 3e-3,
    ensemble_size: int = 5,
    seed: int = 42,
) -> MLPSurrogateEnsemble:
    import torch
    from torch import nn

    trials = list(trials)
    X, y, meta_mean, meta_scale = _surrogate_matrix(trials, source_meta, max_warmup)
    score_mean = float(y.mean())
    score_scale = float(y.std())
    if score_scale < 1e-6:
        score_scale = 1.0
    y_scaled = (y - score_mean) / score_scale
    grouped = {
        name: np.asarray([i for i, trial in enumerate(trials) if trial.dataset_id == name], dtype=int)
        for name in sorted({trial.dataset_id for trial in trials})
    }
    models = []
    for member in range(int(ensemble_size)):
        member_seed = int(seed) + 1009 * member
        rng = np.random.default_rng(member_seed)
        sample = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in grouped.values()
        ])
        rng.shuffle(sample)
        X_t = torch.tensor(X[sample], dtype=torch.float32)
        y_t = torch.tensor(y_scaled[sample, None], dtype=torch.float32)
        torch.manual_seed(member_seed)
        model = nn.Sequential(
            nn.Linear(X.shape[1], int(hidden_size)), nn.ReLU(),
            nn.Linear(int(hidden_size), int(hidden_size)), nn.ReLU(),
            nn.Linear(int(hidden_size), 1),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
        for _ in range(int(epochs)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(X_t)
            loss = nn.functional.mse_loss(pred, y_t)
            loss.backward()
            optimizer.step()
        models.append(model.eval())
    return MLPSurrogateEnsemble(models, meta_mean, meta_scale, score_mean, score_scale, max_warmup)


def fit_extra_trees_surrogate(
    trials: Iterable[BankTrial],
    source_meta: dict[str, np.ndarray],
    *,
    max_warmup: int,
    n_estimators: int = 500,
    min_samples_leaf: int = 2,
    seed: int = 42,
) -> ExtraTreesSurrogate:
    from sklearn.ensemble import ExtraTreesRegressor

    trials = list(trials)
    X, y, meta_mean, meta_scale = _surrogate_matrix(trials, source_meta, max_warmup)
    model = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        max_features=0.8,
        bootstrap=True,
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(X, y)
    return ExtraTreesSurrogate(model, meta_mean, meta_scale, max_warmup)


def optimize_surrogate_tpe(
    surrogate,
    target_meta: np.ndarray,
    hp_args,
    *,
    fixed_categories: dict[str, Any] | None = None,
    n_trials: int = 3000,
    risk_penalty: float = 0.25,
    seed: int = 42,
) -> SurrogateProposal:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=int(seed),
            n_startup_trials=min(50, max(10, int(n_trials) // 20)),
        ),
    )

    def objective(trial):
        config = sample_config(trial, hp_args, fixed_categories)
        mean, std = surrogate.predict(np.asarray(target_meta), [config])
        acquisition = float(mean[0] - float(risk_penalty) * std[0])
        trial.set_user_attr("config", config)
        trial.set_user_attr("predicted_mcc", float(mean[0]))
        trial.set_user_attr("predicted_std", float(std[0]))
        return acquisition

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = study.best_trial
    return SurrogateProposal(
        config=canonical_config(best.user_attrs["config"]),
        predicted_mcc=float(best.user_attrs["predicted_mcc"]),
        predicted_std=float(best.user_attrs["predicted_std"]),
        acquisition=float(best.value),
    )


def leave_one_dataset_out_surrogate_rmse(
    trials: Iterable[BankTrial],
    source_meta: dict[str, np.ndarray],
    *,
    max_warmup: int,
    model_type: str = "extra_trees",
    seed: int = 42,
) -> float:
    """Cross-dataset score-prediction diagnostic using source datasets only."""
    trials = list(trials)
    dataset_ids = sorted({trial.dataset_id for trial in trials})
    squared = []
    for holdout in dataset_ids:
        train = [trial for trial in trials if trial.dataset_id != holdout]
        test = [trial for trial in trials if trial.dataset_id == holdout]
        if not train or not test:
            continue
        if model_type == "mlp":
            surrogate = fit_mlp_surrogate(train, source_meta, max_warmup=max_warmup, epochs=500, ensemble_size=3, seed=seed)
        else:
            surrogate = fit_extra_trees_surrogate(train, source_meta, max_warmup=max_warmup, n_estimators=250, seed=seed)
        mean, _ = surrogate.predict(
            np.asarray(source_meta[holdout], dtype=np.float32),
            [trial.config for trial in test],
        )
        actual = np.asarray([trial.valid_mcc for trial in test], dtype=float)
        squared.extend(((mean - actual) ** 2).tolist())
    return float(np.sqrt(np.mean(squared))) if squared else float("nan")


class ReinforcePolicy:
    """Small stochastic mixed-action policy for surrogate-reward RL."""

    def __init__(self, n_meta: int, hidden_size: int, fixed_categories: dict[str, Any] | None = None):
        import torch
        from torch import nn

        self.fixed = dict(fixed_categories or {})
        self.net = nn.Module()
        self.net.shared = nn.Sequential(
            nn.Linear(int(n_meta), int(hidden_size)), nn.Tanh(),
            nn.Linear(int(hidden_size), int(hidden_size)), nn.Tanh(),
        )
        self.net.cat_heads = nn.ModuleDict({
            name: nn.Linear(int(hidden_size), len(choices))
            for name, choices in CATEGORICAL_CHOICES.items()
            if name not in self.fixed
        })
        self.net.bool_heads = nn.ModuleDict({
            name: nn.Linear(int(hidden_size), 1)
            for name in BOOLEAN_FIELDS
            if name not in self.fixed
        })
        self.net.cont_mean = nn.Linear(int(hidden_size), len(CONTINUOUS_FIELDS))
        self.net.cont_log_std = nn.Parameter(torch.full((len(CONTINUOUS_FIELDS),), -0.7))

    def parameters(self):
        return self.net.parameters()

    def _hidden(self, x):
        return self.net.shared(x)

    def sample(self, x):
        import torch

        h = self._hidden(x)
        batch = h.shape[0]
        configs = [{"model_type": "joint", "log1p": True} for _ in range(batch)]
        log_prob = torch.zeros(batch, device=h.device)
        entropy = torch.zeros(batch, device=h.device)

        for name, choices in CATEGORICAL_CHOICES.items():
            if name in self.fixed:
                for config in configs:
                    config[name] = self.fixed[name]
                continue
            dist = torch.distributions.Categorical(logits=self.net.cat_heads[name](h))
            action = dist.sample()
            log_prob = log_prob + dist.log_prob(action)
            entropy = entropy + dist.entropy()
            for i, index in enumerate(action.detach().cpu().tolist()):
                configs[i][name] = choices[int(index)]

        for name in BOOLEAN_FIELDS:
            if name in self.fixed:
                for config in configs:
                    config[name] = bool(self.fixed[name])
                continue
            logits = self.net.bool_heads[name](h).squeeze(-1)
            dist = torch.distributions.Bernoulli(logits=logits)
            action = dist.sample()
            log_prob = log_prob + dist.log_prob(action)
            entropy = entropy + dist.entropy()
            for i, value in enumerate(action.detach().cpu().tolist()):
                configs[i][name] = bool(value >= 0.5)

        mean = self.net.cont_mean(h)
        std = torch.exp(self.net.cont_log_std).clamp(0.03, 2.0)
        dist = torch.distributions.Normal(mean, std)
        latent = dist.sample()
        log_prob = log_prob + dist.log_prob(latent).sum(dim=1)
        entropy = entropy + dist.entropy().sum(dim=1)
        unit = torch.sigmoid(latent).detach().cpu().numpy()
        for i in range(batch):
            configs[i] = decode_continuous(unit[i], configs[i], max_warmup=self._max_warmup)
            configs[i] = apply_fixed_categories(configs[i], self.fixed)
        return configs, log_prob, entropy

    def deterministic(self, x, *, max_warmup: int):
        import torch

        self._max_warmup = int(max_warmup)
        h = self._hidden(x)
        configs = [{"model_type": "joint", "log1p": True} for _ in range(h.shape[0])]
        for name, choices in CATEGORICAL_CHOICES.items():
            if name in self.fixed:
                for config in configs:
                    config[name] = self.fixed[name]
            else:
                indices = self.net.cat_heads[name](h).argmax(dim=1).detach().cpu().tolist()
                for i, index in enumerate(indices):
                    configs[i][name] = choices[int(index)]
        for name in BOOLEAN_FIELDS:
            if name in self.fixed:
                for config in configs:
                    config[name] = bool(self.fixed[name])
            else:
                values = (torch.sigmoid(self.net.bool_heads[name](h).squeeze(-1)) >= 0.5).detach().cpu().tolist()
                for i, value in enumerate(values):
                    configs[i][name] = bool(value)
        unit = torch.sigmoid(self.net.cont_mean(h)).detach().cpu().numpy()
        for i in range(h.shape[0]):
            configs[i] = decode_continuous(unit[i], configs[i], max_warmup)
            configs[i] = apply_fixed_categories(configs[i], self.fixed)
        return configs


def train_reinforce_policy(
    surrogate,
    source_meta: dict[str, np.ndarray],
    target_meta: np.ndarray,
    *,
    max_warmup: int,
    fixed_categories: dict[str, Any] | None = None,
    mode: str = "meta",
    epochs: int = 1000,
    batch_size: int = 32,
    hidden_size: int = 96,
    lr: float = 3e-3,
    risk_penalty: float = 0.25,
    entropy_coef: float = 0.01,
    seed: int = 42,
    callback: Callable[[RLEpochRecord], None] | None = None,
) -> tuple[ReinforcePolicy, list[RLEpochRecord]]:
    """Train contextual source meta-RL or target-directed surrogate RL.

    mode='meta': contexts are sampled from the source datasets.  The policy is
    frozen and queried on Alzheimer metadata after each epoch.

    mode='target': every context is Alzheimer metadata, but reward still comes
    only from the source-trained surrogate.  This deliberately tests surrogate
    exploitation; real Alzheimer MCC must remain an external evaluation.
    """
    import torch

    if mode not in {"meta", "target"}:
        raise ValueError("mode must be 'meta' or 'target'")
    source_ids = tuple(source_meta)
    source_matrix = np.stack([np.asarray(source_meta[name], dtype=np.float32) for name in source_ids])
    mean = source_matrix.mean(axis=0)
    scale = source_matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    source_norm = (source_matrix - mean) / scale
    target_norm = (np.asarray(target_meta, dtype=np.float32) - mean) / scale

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    policy = ReinforcePolicy(source_matrix.shape[1], hidden_size, fixed_categories)
    policy._max_warmup = int(max_warmup)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(lr))
    target_tensor = torch.tensor(target_norm[None, :], dtype=torch.float32)
    history = []
    best_reward_seen = -np.inf

    for epoch in range(1, int(epochs) + 1):
        if mode == "meta":
            indices = rng.integers(0, len(source_ids), size=int(batch_size))
            contexts_raw = source_matrix[indices]
            contexts_norm = source_norm[indices]
        else:
            contexts_raw = np.repeat(np.asarray(target_meta, dtype=np.float32)[None, :], int(batch_size), axis=0)
            contexts_norm = np.repeat(target_norm[None, :], int(batch_size), axis=0)

        x = torch.tensor(contexts_norm, dtype=torch.float32)
        configs, log_prob, entropy = policy.sample(x)
        pred_mean, pred_std = surrogate.predict(contexts_raw, configs)
        reward_np = pred_mean - float(risk_penalty) * pred_std
        reward = torch.tensor(reward_np, dtype=torch.float32)
        advantage = (reward - reward.mean()) / reward.std().clamp_min(1e-6)
        loss = -(advantage.detach() * log_prob).mean() - float(entropy_coef) * entropy.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        target_config = policy.deterministic(target_tensor, max_warmup=max_warmup)[0]
        target_mean, target_std = surrogate.predict(np.asarray(target_meta), [target_config])
        best_reward_seen = max(best_reward_seen, float(np.max(reward_np)))
        record = RLEpochRecord(
            epoch=epoch,
            mean_reward=float(np.mean(reward_np)),
            best_reward=float(best_reward_seen),
            target_config=target_config,
            predicted_mcc=float(target_mean[0]),
            predicted_std=float(target_std[0]),
        )
        history.append(record)
        if callback is not None:
            callback(record)
    return policy, history

@dataclass
class EvolutionGenerationRecord:
    generation: int
    best_acquisition: float
    predicted_mcc: float
    predicted_std: float
    config: dict[str, Any]


def _genome_to_config(genome: np.ndarray, max_warmup: int, fixed_categories: dict[str, Any] | None = None) -> dict[str, Any]:
    z = np.asarray(genome, dtype=float)
    cursor = 0
    config = {
        "model_type": "joint",
        "log1p": True,
        "variational": bool(z[cursor] >= 0.0),
        "kan": bool(z[cursor + 1] >= 0.0),
        "class_triplet": bool(z[cursor + 2] >= 0.0),
    }
    cursor += 3
    # Sigmoid maps unconstrained evolutionary controls into the bounded encoding.
    unit = 1.0 / (1.0 + np.exp(-np.clip(z[cursor:cursor + len(CONTINUOUS_FIELDS)], -30.0, 30.0)))
    cursor += len(CONTINUOUS_FIELDS)
    dloss_logits = z[cursor:cursor + len(DLOSS_CHOICES)]
    cursor += len(DLOSS_CHOICES)
    scaler_logits = z[cursor:cursor + len(SCALER_CHOICES)]
    cursor += len(SCALER_CHOICES)
    depth_logits = z[cursor:cursor + len(N_LAYER_CHOICES)]
    config["dloss"] = DLOSS_CHOICES[int(np.argmax(dloss_logits))]
    config["scaler"] = SCALER_CHOICES[int(np.argmax(scaler_logits))]
    config["n_layers"] = N_LAYER_CHOICES[int(np.argmax(depth_logits))]
    config = decode_continuous(unit, config, max_warmup)
    return apply_fixed_categories(config, fixed_categories)


def optimize_surrogate_evolution(
    surrogate,
    target_meta: np.ndarray,
    *,
    max_warmup: int,
    fixed_categories: dict[str, Any] | None = None,
    population_size: int = 64,
    generations: int = 100,
    elite_count: int = 8,
    tournament_size: int = 3,
    mutation_rate: float = 0.10,
    mutation_scale: float = 0.35,
    risk_penalty: float = 0.25,
    seed: int = 42,
) -> tuple[SurrogateProposal, list[EvolutionGenerationRecord]]:
    """Cheap evolutionary optimization of the target surrogate."""
    if not 1 <= elite_count < population_size:
        raise ValueError("elite_count must be in [1, population_size)")
    if not 2 <= tournament_size <= population_size:
        raise ValueError("invalid tournament_size")
    genome_size = (
        3 + len(CONTINUOUS_FIELDS)
        + len(DLOSS_CHOICES) + len(SCALER_CHOICES) + len(N_LAYER_CHOICES)
    )
    rng = np.random.default_rng(int(seed))
    population = rng.normal(0.0, 1.0, size=(int(population_size), genome_size)).astype(np.float32)
    history: list[EvolutionGenerationRecord] = []
    global_best = None

    def tournament(fitness):
        indices = rng.choice(len(fitness), size=int(tournament_size), replace=False)
        return int(indices[np.argmax(fitness[indices])])

    for generation in range(1, int(generations) + 1):
        configs = [_genome_to_config(row, max_warmup, fixed_categories) for row in population]
        mean, std = surrogate.predict(np.asarray(target_meta), configs)
        acquisition = mean - float(risk_penalty) * std
        best_index = int(np.argmax(acquisition))
        record = EvolutionGenerationRecord(
            generation=generation,
            best_acquisition=float(acquisition[best_index]),
            predicted_mcc=float(mean[best_index]),
            predicted_std=float(std[best_index]),
            config=configs[best_index],
        )
        history.append(record)
        if global_best is None or record.best_acquisition > global_best.best_acquisition:
            global_best = record

        order = np.argsort(acquisition)[::-1]
        children = [population[index].copy() for index in order[:int(elite_count)]]
        while len(children) < int(population_size):
            left = population[tournament(acquisition)]
            right = population[tournament(acquisition)]
            mask = rng.random(genome_size) < 0.5
            child = np.where(mask, left, right).astype(np.float32)
            mutation = rng.random(genome_size) < float(mutation_rate)
            if np.any(mutation):
                child[mutation] += rng.normal(0.0, float(mutation_scale), int(mutation.sum())).astype(np.float32)
            children.append(child)
        population = np.stack(children)

    assert global_best is not None
    return SurrogateProposal(
        config=global_best.config,
        predicted_mcc=global_best.predicted_mcc,
        predicted_std=global_best.predicted_std,
        acquisition=global_best.best_acquisition,
    ), history
