"""Offline/replay meta-HPO models built from cumulative source Optuna trials.

All model fitting here is cheap relative to BERNN.  The caller decides when to
perform the real Alzheimer BERNN evaluation.  No Alzheimer score is accepted as
training input by any class in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from scripts import hp_search
from src.meta_hpo_utils import (
    DLOSS_CHOICES,
    N_LAYER_CHOICES,
    SCALER_CHOICES,
    TrialPoint,
    apply_fixed_categories,
    decode_config_vector,
    encode_config,
    sample_bernn_config,
)


@dataclass
class DirectResult:
    config: dict[str, Any]
    train_loss: float


@dataclass
class SurrogateResult:
    config: dict[str, Any]
    predicted_mcc: float
    predicted_std: float
    acquisition: float


class SurrogateEnsemble:
    def __init__(self, models, meta_mean, meta_scale, score_mean, score_scale, max_warmup):
        self.models = list(models)
        self.meta_mean = np.asarray(meta_mean, dtype=np.float32)
        self.meta_scale = np.asarray(meta_scale, dtype=np.float32)
        self.score_mean = float(score_mean)
        self.score_scale = float(score_scale)
        self.max_warmup = int(max_warmup)

    def _features(self, meta_rows: np.ndarray, configs: list[dict[str, Any]]) -> np.ndarray:
        meta_rows = np.asarray(meta_rows, dtype=np.float32)
        if meta_rows.ndim == 1:
            meta_rows = np.repeat(meta_rows[None, :], len(configs), axis=0)
        if len(meta_rows) != len(configs):
            raise ValueError("meta_rows and configs must have the same length")
        meta_scaled = (meta_rows - self.meta_mean) / self.meta_scale
        config_rows = np.stack([encode_config(config, self.max_warmup) for config in configs])
        return np.concatenate([meta_scaled, config_rows], axis=1).astype(np.float32)

    def predict(self, meta_rows: np.ndarray, configs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        X = torch.tensor(self._features(meta_rows, configs), dtype=torch.float32)
        members = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(X).squeeze(-1).cpu().numpy()
                pred = pred * self.score_scale + self.score_mean
                members.append(pred)
        matrix = np.stack(members)
        return matrix.mean(axis=0), matrix.std(axis=0)


def _build_mlp(n_inputs: int, hidden_size: int, dropout: float = 0.05):
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Linear(n_inputs, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
    )


def fit_direct_best_config_model(
    best_points: dict[str, TrialPoint],
    source_meta: dict[str, np.ndarray],
    target_meta: np.ndarray,
    *,
    max_warmup: int,
    fixed_categories: dict[str, Any] | None = None,
    hidden_size: int = 64,
    epochs: int = 2000,
    lr: float = 1e-2,
    seed: int = 42,
) -> DirectResult:
    """Fit metadata -> current best-known source config from scratch."""
    import torch
    from torch import nn

    dataset_ids = tuple(best_points)
    train_meta = np.stack([source_meta[name] for name in dataset_ids]).astype(np.float32)
    mean = train_meta.mean(axis=0)
    scale = train_meta.std(axis=0)
    scale[scale < 1e-8] = 1.0
    X = torch.tensor((train_meta - mean) / scale, dtype=torch.float32)
    Y = torch.tensor(np.stack([
        encode_config(best_points[name].config, max_warmup) for name in dataset_ids
    ]), dtype=torch.float32)

    torch.manual_seed(int(seed))
    model = nn.Sequential(
        nn.Linear(X.shape[1], int(hidden_size)),
        nn.ReLU(),
        nn.Linear(int(hidden_size), Y.shape[1]),
        nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    final_loss = float("nan")
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    model.eval()
    target = torch.tensor(((np.asarray(target_meta, dtype=np.float32) - mean) / scale)[None, :])
    with torch.no_grad():
        encoded = model(target).squeeze(0).cpu().numpy()
    config = decode_config_vector(encoded, max_warmup, fixed_categories)
    return DirectResult(config=config, train_loss=final_loss)


def fit_surrogate_ensemble(
    points: Iterable[TrialPoint],
    source_meta: dict[str, np.ndarray],
    *,
    max_warmup: int,
    hidden_size: int = 128,
    epochs: int = 1500,
    lr: float = 3e-3,
    ensemble_size: int = 5,
    seed: int = 42,
) -> SurrogateEnsemble:
    """Fit (dataset metadata, hparams) -> MCC on every accumulated source trial."""
    import torch
    from torch import nn

    points = list(points)
    if not points:
        raise ValueError("No source trials available for surrogate training")
    unique_ids = sorted({point.dataset_id for point in points})
    train_meta_unique = np.stack([source_meta[name] for name in unique_ids]).astype(np.float32)
    meta_mean = train_meta_unique.mean(axis=0)
    meta_scale = train_meta_unique.std(axis=0)
    meta_scale[meta_scale < 1e-8] = 1.0

    X = np.stack([
        np.concatenate([
            (np.asarray(source_meta[p.dataset_id], dtype=np.float32) - meta_mean) / meta_scale,
            encode_config(p.config, max_warmup),
        ])
        for p in points
    ]).astype(np.float32)
    y = np.asarray([p.score for p in points], dtype=np.float32)
    score_mean = float(y.mean())
    score_scale = float(y.std())
    if score_scale < 1e-6:
        score_scale = 1.0
    y_scaled = (y - score_mean) / score_scale

    # Dataset-stratified bootstrap gives ensemble diversity without allowing a
    # member to entirely forget one of the four meta-training datasets.
    grouped_indices = {
        name: np.asarray([i for i, p in enumerate(points) if p.dataset_id == name], dtype=int)
        for name in unique_ids
    }
    models = []
    for member in range(int(ensemble_size)):
        member_seed = int(seed) + 1009 * member
        rng = np.random.default_rng(member_seed)
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in grouped_indices.values()
        ])
        rng.shuffle(sampled)
        X_member = torch.tensor(X[sampled], dtype=torch.float32)
        y_member = torch.tensor(y_scaled[sampled, None], dtype=torch.float32)

        torch.manual_seed(member_seed)
        model = _build_mlp(X.shape[1], int(hidden_size))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
        for _ in range(int(epochs)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(X_member)
            loss = nn.functional.mse_loss(pred, y_member)
            loss.backward()
            optimizer.step()
        model.eval()
        models.append(model)

    return SurrogateEnsemble(
        models=models,
        meta_mean=meta_mean,
        meta_scale=meta_scale,
        score_mean=score_mean,
        score_scale=score_scale,
        max_warmup=max_warmup,
    )


def surrogate_tpe_search(
    surrogate: SurrogateEnsemble,
    target_meta: np.ndarray,
    *,
    fixed_categories: dict[str, Any] | None = None,
    n_candidates: int = 3000,
    risk_penalty: float = 0.25,
    seed: int = 42,
) -> SurrogateResult:
    """Use cheap TPE against the transferable surrogate, then return its optimum."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hp_args = SimpleNamespace(n_epochs=max(1, surrogate.max_warmup), max_warmup=surrogate.max_warmup)
    sampler = optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=min(50, max(10, n_candidates // 20)))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        config = sample_bernn_config(trial, hp_args, fixed_categories)
        mean, std = surrogate.predict(np.asarray(target_meta), [config])
        acquisition = float(mean[0] - float(risk_penalty) * std[0])
        trial.set_user_attr("config", config)
        trial.set_user_attr("predicted_mcc", float(mean[0]))
        trial.set_user_attr("predicted_std", float(std[0]))
        return acquisition

    study.optimize(objective, n_trials=int(n_candidates), show_progress_bar=False)
    trial = study.best_trial
    return SurrogateResult(
        config=trial.user_attrs["config"],
        predicted_mcc=float(trial.user_attrs["predicted_mcc"]),
        predicted_std=float(trial.user_attrs["predicted_std"]),
        acquisition=float(trial.value),
    )


class ContextualPolicy:
    """Small mixed-action policy used with REINFORCE and surrogate rewards."""

    def __init__(self, n_meta: int, hidden_size: int = 96):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Linear(n_meta, hidden_size), nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                )
                self.dloss = nn.Linear(hidden_size, len(DLOSS_CHOICES))
                self.scaler = nn.Linear(hidden_size, len(SCALER_CHOICES))
                self.n_layers = nn.Linear(hidden_size, len(N_LAYER_CHOICES))
                self.booleans = nn.Linear(hidden_size, 3)
                # 12 Beta distributions: alpha logits + beta logits.
                self.continuous = nn.Linear(hidden_size, 24)

            def forward(self, x):
                h = self.body(x)
                return {
                    "dloss": self.dloss(h),
                    "scaler": self.scaler(h),
                    "n_layers": self.n_layers(h),
                    "booleans": self.booleans(h),
                    "continuous": self.continuous(h),
                }

        self.model = Net()


def _policy_sample(policy: ContextualPolicy, meta_batch, max_warmup: int, fixed_categories, deterministic=False):
    import torch
    from torch.distributions import Bernoulli, Beta, Categorical

    outputs = policy.model(meta_batch)
    batch_size = meta_batch.shape[0]
    fixed = fixed_categories or {}
    log_prob = torch.zeros(batch_size, dtype=torch.float32)
    entropy = torch.zeros(batch_size, dtype=torch.float32)

    def sample_cat(name, logits, choices):
        nonlocal log_prob, entropy
        if name in fixed:
            value = fixed[name]
            if name == "n_layers":
                value = int(value)
            index = choices.index(value)
            return torch.full((batch_size,), index, dtype=torch.long)
        dist = Categorical(logits=logits)
        index = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = log_prob + dist.log_prob(index)
        entropy = entropy + dist.entropy()
        return index

    dloss_idx = sample_cat("dloss", outputs["dloss"], DLOSS_CHOICES)
    scaler_idx = sample_cat("scaler", outputs["scaler"], SCALER_CHOICES)
    layers_idx = sample_cat("n_layers", outputs["n_layers"], N_LAYER_CHOICES)

    bool_values = []
    bool_names = ("variational", "kan", "class_triplet")
    for index, name in enumerate(bool_names):
        if name in fixed:
            bool_values.append(torch.full((batch_size,), float(bool(fixed[name]))))
            continue
        dist = Bernoulli(logits=outputs["booleans"][:, index])
        sample = (torch.sigmoid(outputs["booleans"][:, index]) >= 0.5).float() if deterministic else dist.sample()
        log_prob = log_prob + dist.log_prob(sample)
        entropy = entropy + dist.entropy()
        bool_values.append(sample)

    raw = outputs["continuous"].reshape(batch_size, 12, 2)
    alpha = torch.nn.functional.softplus(raw[:, :, 0]) + 1.0
    beta = torch.nn.functional.softplus(raw[:, :, 1]) + 1.0
    dist = Beta(alpha, beta)
    units = alpha / (alpha + beta) if deterministic else dist.sample()
    log_prob = log_prob + dist.log_prob(units).sum(dim=1)
    entropy = entropy + dist.entropy().sum(dim=1)

    configs = []
    for row in range(batch_size):
        # Assemble the same bounded vector used by decode_config_vector.
        variational = float(bool_values[0][row].item())
        kan = float(bool_values[1][row].item())
        class_triplet = float(bool_values[2][row].item())
        cont = units[row].detach().cpu().numpy()
        vector = [
            variational, kan, class_triplet,
            cont[0],  # class_triplet_w
            cont[1],  # lr
            cont[2],  # wd
            cont[3],  # nu
            cont[4],  # smoothing
            cont[5],  # margin
            cont[6],  # dropout
            cont[7],  # thres
            cont[8],  # warmup
            cont[9],  # layer1
            cont[10], # gamma
            cont[11], # beta
        ]
        vector += [1.0 if i == int(dloss_idx[row]) else 0.0 for i in range(len(DLOSS_CHOICES))]
        vector += [1.0 if i == int(scaler_idx[row]) else 0.0 for i in range(len(SCALER_CHOICES))]
        vector += [1.0 if i == int(layers_idx[row]) else 0.0 for i in range(len(N_LAYER_CHOICES))]
        configs.append(decode_config_vector(np.asarray(vector), max_warmup, fixed))
    return configs, log_prob, entropy


def train_surrogate_rl_policy(
    surrogate: SurrogateEnsemble,
    source_meta: dict[str, np.ndarray],
    target_meta: np.ndarray,
    *,
    fixed_categories: dict[str, Any] | None = None,
    mode: str = "meta",
    epochs: int = 1500,
    batch_size: int = 64,
    hidden_size: int = 96,
    lr: float = 2e-3,
    entropy_coef: float = 2e-3,
    risk_penalty: float = 0.25,
    final_candidates: int = 3000,
    seed: int = 42,
) -> SurrogateResult:
    """Train a REINFORCE policy using ONLY surrogate-predicted reward.

    mode='meta': contexts are sampled from the four source datasets; the frozen
    policy is then applied to Alzheimer metadata.

    mode='target': the policy is optimized against the surrogate at Alzheimer
    metadata directly.  This still never sees a real Alzheimer MCC as reward;
    the real BERNN run remains an external validity check.
    """
    import torch

    if mode not in {"meta", "target"}:
        raise ValueError("mode must be 'meta' or 'target'")
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    # Match the surrogate's source-only metadata normalization.
    source_ids = tuple(source_meta)
    source_matrix = np.stack([source_meta[name] for name in source_ids]).astype(np.float32)
    source_scaled = (source_matrix - surrogate.meta_mean) / surrogate.meta_scale
    target_scaled = (np.asarray(target_meta, dtype=np.float32) - surrogate.meta_mean) / surrogate.meta_scale

    policy = ContextualPolicy(source_matrix.shape[1], hidden_size=int(hidden_size))
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=float(lr))
    moving_baseline = 0.0

    for epoch in range(int(epochs)):
        if mode == "meta":
            indices = rng.integers(0, len(source_ids), size=int(batch_size))
            meta_raw = np.stack([source_meta[source_ids[i]] for i in indices]).astype(np.float32)
            meta_scaled = source_scaled[indices]
        else:
            meta_raw = np.repeat(np.asarray(target_meta, dtype=np.float32)[None, :], int(batch_size), axis=0)
            meta_scaled = np.repeat(target_scaled[None, :], int(batch_size), axis=0)

        meta_tensor = torch.tensor(meta_scaled, dtype=torch.float32)
        configs, log_prob, entropy = _policy_sample(
            policy, meta_tensor, surrogate.max_warmup, fixed_categories, deterministic=False
        )
        means, stds = surrogate.predict(meta_raw, configs)
        rewards = means - float(risk_penalty) * stds
        reward_mean = float(np.mean(rewards))
        moving_baseline = 0.95 * moving_baseline + 0.05 * reward_mean if epoch else reward_mean
        advantage = torch.tensor(rewards - moving_baseline, dtype=torch.float32)
        loss = -(advantage.detach() * log_prob).mean() - float(entropy_coef) * entropy.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 5.0)
        optimizer.step()

    # Sample many target actions and let the surrogate rank them.  This retains
    # RL's learned proposal distribution while avoiding a noisy single draw.
    policy.model.eval()
    all_configs = []
    chunk = 256
    with torch.no_grad():
        remaining = int(final_candidates)
        while remaining > 0:
            n = min(chunk, remaining)
            meta = torch.tensor(np.repeat(target_scaled[None, :], n, axis=0), dtype=torch.float32)
            configs, _, _ = _policy_sample(
                policy, meta, surrogate.max_warmup, fixed_categories, deterministic=False
            )
            all_configs.extend(configs)
            remaining -= n
    means, stds = surrogate.predict(np.asarray(target_meta), all_configs)
    acquisition = means - float(risk_penalty) * stds
    index = int(np.argmax(acquisition))
    return SurrogateResult(
        config=all_configs[index],
        predicted_mcc=float(means[index]),
        predicted_std=float(stds[index]),
        acquisition=float(acquisition[index]),
    )
