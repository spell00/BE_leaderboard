import pytest

from src.multifidelity_hpo import (
    OptunaProgressReporter,
    deterministic_force_full,
    make_pruner,
)


class FakeTrial:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.reported = []
        self.attrs = {}

    def report(self, value, step):
        self.reported.append((float(value), int(step)))

    def should_prune(self):
        return self.decisions.pop(0) if self.decisions else False

    def set_user_attr(self, key, value):
        self.attrs[key] = value


def test_force_full_is_deterministic_and_respects_first_n():
    assert deterministic_force_full(0, seed=42, fraction=0.0, first_n=2)
    assert deterministic_force_full(1, seed=42, fraction=0.0, first_n=2)
    assert not deterministic_force_full(2, seed=42, fraction=0.0, first_n=2)
    a = deterministic_force_full(17, seed=123, fraction=0.4)
    b = deterministic_force_full(17, seed=123, fraction=0.4)
    assert a == b


def test_reporter_prunes_when_not_forced():
    import optuna

    trial = FakeTrial([True])
    reporter = OptunaProgressReporter(
        trial, force_full=False, report_every=1, max_resource=100
    )
    with pytest.raises(optuna.TrialPruned):
        reporter({"step": 10, "score": 0.41, "granularity": "epoch"})
    assert trial.reported == [(0.41, 10)]
    assert trial.attrs["pruned_at_step"] == 10
    assert reporter.history[0]["budget_fraction"] == pytest.approx(0.1)


def test_reporter_keeps_audit_trial_running_after_prune_signal():
    trial = FakeTrial([True, False])
    reporter = OptunaProgressReporter(
        trial, force_full=True, report_every=1, max_resource=100
    )
    reporter({"step": 10, "score": 0.41, "granularity": "epoch"})
    reporter({"step": 20, "score": 0.50, "granularity": "epoch"})
    assert reporter.would_prune_at_step == 10
    assert trial.attrs["prune_decision_ignored_for_audit"] is True
    assert reporter.last_score == pytest.approx(0.50)


def test_pruner_factory_supports_safe_modes():
    import optuna

    assert isinstance(
        make_pruner("none", min_resource=10, max_resource=100),
        optuna.pruners.NopPruner,
    )
    assert isinstance(
        make_pruner("sha", min_resource=10, max_resource=100),
        optuna.pruners.SuccessiveHalvingPruner,
    )
    assert isinstance(
        make_pruner("hyperband", min_resource=10, max_resource=100),
        optuna.pruners.HyperbandPruner,
    )
