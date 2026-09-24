import json

import numpy as np
import pytest

from src.learning_curve_surrogate import (
    CompletedCurveTrial,
    backfill_multifidelity_totals,
    curve_summary,
    fit_partial_curve_total_regressor,
)


def config(lr, dropout):
    return {
        "model_type": "joint",
        "dloss": "no",
        "variational": False,
        "kan": False,
        "class_triplet": False,
        "class_triplet_w": 0.0,
        "lr": lr,
        "wd": 1e-5,
        "nu": 1.0,
        "smoothing": 0.05,
        "margin": 1.0,
        "dropout": dropout,
        "thres": 0.01,
        "warmup": 10,
        "n_layers": 2,
        "layer1": 768,
        "log1p": True,
        "scaler": "standard",
        "gamma": 0.0,
        "beta": 0.0,
    }


def history(offset):
    return tuple(
        {
            "step": step,
            "score": offset + gain,
            "current_fold_valid_mcc": offset + gain,
            "best_current_fold_valid_mcc": offset + gain,
        }
        for step, gain in [(10, 0.02), (20, 0.08), (40, 0.13), (80, 0.16)]
    )


def test_curve_summary_is_fixed_width_and_uses_budget_fraction():
    values = curve_summary(history(0.4), max_resource=100)
    assert values.shape == (8,)
    assert values[0] == pytest.approx(0.8)
    assert values[1] == pytest.approx(0.56)


def test_single_target_curve_regressor_predicts_finite_total():
    rows = [
        CompletedCurveTrial(config(1e-3, 0.1), 0.61, history(0.40), 100),
        CompletedCurveTrial(config(2e-3, 0.2), 0.72, history(0.50), 100),
        CompletedCurveTrial(config(3e-3, 0.3), 0.83, history(0.60), 100),
    ]
    model = fit_partial_curve_total_regressor(
        rows, max_warmup=50, n_estimators=25, min_samples_leaf=1
    )
    mean, std = model.predict(config(2e-3, 0.2), history(0.50)[:3], 100)
    assert np.isfinite(mean)
    assert np.isfinite(std)
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0


def test_backfill_marks_predictions_without_overwriting_observed_totals(tmp_path):
    rows = []
    for trial_number, final in enumerate((0.61, 0.72, 0.83)):
        curve = history(0.40 + 0.10 * trial_number)
        curve_path = tmp_path / "curves" / f"trial_{trial_number:05d}.json"
        curve_path.parent.mkdir(exist_ok=True)
        curve_path.write_text(json.dumps(curve))
        rows.append({
            "trial_number": trial_number,
            "state": "COMPLETE",
            "observed_total": final,
            "total": final,
            "total_source": "observed",
            "config": config(1e-3 * (trial_number + 1), 0.1 * (trial_number + 1)),
            "curve_path": str(curve_path.relative_to(tmp_path)),
            "max_resource": 100,
        })

    pruned_curve = history(0.47)[:2]
    pruned_path = tmp_path / "curves" / "trial_00003.json"
    pruned_path.write_text(json.dumps(pruned_curve))
    rows.append({
        "trial_number": 3,
        "state": "PRUNED",
        "observed_total": None,
        "partial_total": pruned_curve[-1]["score"],
        "total": pruned_curve[-1]["score"],
        "total_source": "partial_proxy",
        "config": config(1.5e-3, 0.15),
        "curve_path": str(pruned_path.relative_to(tmp_path)),
        "max_resource": 100,
    })
    (tmp_path / "multifidelity_trials.json").write_text(json.dumps(rows))

    result = backfill_multifidelity_totals(
        tmp_path, max_warmup=50, seed=7, min_completed=3
    )
    assert result == {"completed": 3, "predicted": 1}

    updated = json.loads((tmp_path / "multifidelity_trials.json").read_text())
    assert all(row["total_source"] == "observed" for row in updated[:3])
    assert updated[3]["total_source"] == "predicted"
    assert updated[3]["observed_total"] is None
    assert np.isfinite(updated[3]["predicted_total"])
    assert np.isfinite(updated[3]["predicted_total_std"])
