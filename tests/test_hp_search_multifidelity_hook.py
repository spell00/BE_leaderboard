import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import hp_search


def _config():
    return {"model_type": "joint"}


def _data():
    return (
        pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]}),
        np.asarray(["a", "a", "b", "b"]),
        np.asarray(["1", "1", "2", "2"]),
    )


def _install_fake_dependencies(monkeypatch, trainer_cls):
    bernn = types.ModuleType("bernn")
    bernn.TrainAEClassifierHoldout = trainer_cls
    bernn.TrainAEThenClassifierHoldout = trainer_cls
    monkeypatch.setitem(sys.modules, "bernn", bernn)

    baselines = types.ModuleType("src.baselines")
    baselines.set_bernn_seed = lambda _seed: None
    monkeypatch.setitem(sys.modules, "src.baselines", baselines)

    monkeypatch.setattr(hp_search, "build_trainer_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(hp_search, "bernn_params_from_cfg", lambda _cfg: {})
    monkeypatch.setattr(hp_search, "_close_fit_resources", lambda _trainer: None)


class CallbackTrainer:
    def __init__(
        self, config=None, log_metrics=False, keep_models=False,
        log_mlflow=False, epoch_callback=None,
    ):
        self.epoch_callback = epoch_callback
        self.best_mcc = 0.67

    def fit(self, X_train, y_train, **kwargs):
        if self.epoch_callback:
            self.epoch_callback({
                "epoch": 0,
                "valid_mcc": 0.4,
                "best_valid_mcc": 0.4,
            })


class LegacyTrainer:
    def __init__(self, config=None, log_metrics=False, keep_models=False, log_mlflow=False, **kwargs):
        assert "epoch_callback" not in kwargs
        self.best_mcc = 0.55

    def fit(self, X_train, y_train, **kwargs):
        return None


def test_fit_one_passes_explicit_epoch_callback(monkeypatch):
    _install_fake_dependencies(monkeypatch, CallbackTrainer)
    seen = []
    trainer, score = hp_search._fit_one(
        _config(), SimpleNamespace(), _data(), "test_exp", 42, False,
        epoch_callback=seen.append,
    )
    assert score == pytest.approx(0.67)
    assert trainer._external_epoch_callback_supported is True
    assert seen == [{"epoch": 0, "valid_mcc": 0.4, "best_valid_mcc": 0.4}]


def test_fit_one_does_not_trust_legacy_kwargs_as_callback_support(monkeypatch):
    _install_fake_dependencies(monkeypatch, LegacyTrainer)
    trainer, score = hp_search._fit_one(
        _config(), SimpleNamespace(), _data(), "test_exp", 42, False,
        epoch_callback=lambda _payload: None,
    )
    assert score == pytest.approx(0.55)
    assert trainer._external_epoch_callback_supported is False


def test_pruning_exception_from_callback_propagates(monkeypatch):
    import optuna

    _install_fake_dependencies(monkeypatch, CallbackTrainer)

    def prune(_payload):
        raise optuna.TrialPruned("stop")

    with pytest.raises(optuna.TrialPruned, match="stop"):
        hp_search._fit_one(
            _config(), SimpleNamespace(), _data(), "test_exp", 42, False,
            epoch_callback=prune,
        )
