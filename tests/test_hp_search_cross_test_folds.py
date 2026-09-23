import argparse

import numpy as np
import pandas as pd

from scripts import hp_search


class _Predictor:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, X, groups_test=None):
        return self.predictions.copy()


def test_run_trial_persists_each_fixed_cross_test_mcc(monkeypatch):
    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0]})
    y = np.asarray(["a", "b", "a", "b"])
    batches = np.asarray(["b1", "b1", "b2", "b2"])

    X_fixed = pd.DataFrame({"f1": [10.0, 11.0]})
    y_fixed = np.asarray(["a", "b"])
    batches_fixed = np.asarray(["b3", "b3"])

    splits = [
        (np.asarray([0, 1]), np.asarray([2, 3])),
        (np.asarray([2, 3]), np.asarray([0, 1])),
    ]
    monkeypatch.setattr(hp_search, "cached_cv_splits", lambda *args, **kwargs: splits)
    monkeypatch.setattr(hp_search, "extract_mlflow_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(hp_search, "extract_mlflow_epoch_traces", lambda *args, **kwargs: {})

    def fake_fit_one(cfg, args, data, exp_id, seed, keep_models):
        if exp_id.endswith("_fold0"):
            return _Predictor(["a", "b"]), 0.25
        return _Predictor(["b", "a"]), 0.75

    monkeypatch.setattr(hp_search, "_fit_one", fake_fit_one)

    args = argparse.Namespace(
        dataset="demo",
        n_repeats=2,
        resolved_n_repeats=2,
        seed=42,
    )
    score, metrics = hp_search.run_trial(
        {},
        args,
        (X, y, batches),
        "test_cross_test_folds",
        fixed_test_data=(X_fixed, y_fixed, batches_fixed),
    )

    assert score == 0.5
    assert metrics["valid_mcc_folds"] == [0.25, 0.75]
    assert metrics["test_mcc_folds"] == [1.0, -1.0]
    assert metrics["test_mcc_fold_mean"] == 0.0
    assert metrics["test_mcc_fold_std"] == 1.0
    assert metrics["cross_test"] == 1.0
