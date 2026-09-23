"""Safe multi-fidelity helpers for BERNN Optuna studies.

The scheduler always observes the same scalar used for final model selection:
validation MCC averaged over the folds observed so far. Cross-test/test metrics
never enter pruning decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def make_pruner(
    name: str,
    *,
    min_resource: int,
    max_resource: int,
    reduction_factor: int = 3,
    bootstrap_count: int = 0,
):
    import optuna

    name = str(name).lower()
    if name == "none":
        return optuna.pruners.NopPruner()
    min_resource = max(1, min(int(min_resource), int(max_resource)))
    reduction_factor = max(2, int(reduction_factor))
    if name in {"sha", "asha", "successive_halving"}:
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min_resource,
            reduction_factor=reduction_factor,
            bootstrap_count=max(0, int(bootstrap_count)),
        )
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max(min_resource, int(max_resource)),
            reduction_factor=reduction_factor,
            bootstrap_count=max(0, int(bootstrap_count)),
        )
    raise ValueError(f"Unknown pruner {name!r}; use none, sha, or hyperband")


def deterministic_force_full(
    trial_number: int,
    *,
    seed: int,
    fraction: float,
    first_n: int = 0,
) -> bool:
    if int(trial_number) < max(0, int(first_n)):
        return True
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= 0:
        return False
    # Stable per-trial draw: independent of process order / GPU scheduling.
    rng = np.random.default_rng(int(seed) + 104729 * (int(trial_number) + 1))
    return bool(rng.random() < fraction)


@dataclass
class OptunaProgressReporter:
    trial: object
    force_full: bool = False
    report_every: int = 10
    max_resource: int | None = None

    def __post_init__(self):
        self.report_every = max(1, int(self.report_every))
        self.history: list[dict] = []
        self.last_reported_step = -1
        self.would_prune_at_step: int | None = None

    def __call__(self, event: dict) -> None:
        import optuna

        step = int(event["step"])
        score = float(event["score"])
        if step <= self.last_reported_step or not math.isfinite(score):
            return
        # Always keep the first observation. Thereafter downsample callback
        # traffic without changing BERNN's own epoch loop.
        if self.last_reported_step >= 0 and step % self.report_every != 0:
            return

        row = dict(event)
        row["step"] = step
        row["score"] = score
        if self.max_resource:
            row["budget_fraction"] = float(
                np.clip(step / float(self.max_resource), 0.0, 1.0)
            )
        self.history.append(row)
        self.last_reported_step = step

        self.trial.report(score, step)
        if not self.trial.should_prune():
            return
        if self.would_prune_at_step is None:
            self.would_prune_at_step = step
            self.trial.set_user_attr("would_prune_at_step", int(step))
        if self.force_full:
            self.trial.set_user_attr("prune_decision_ignored_for_audit", True)
            return
        self.trial.set_user_attr("pruned_at_step", int(step))
        raise optuna.TrialPruned(
            f"multi-fidelity pruner stopped trial at resource step {step}"
        )

    @property
    def last_score(self) -> float | None:
        return float(self.history[-1]["score"]) if self.history else None

    @property
    def last_step(self) -> int:
        return int(self.history[-1]["step"]) if self.history else 0
