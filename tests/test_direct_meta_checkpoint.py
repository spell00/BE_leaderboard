import numpy as np
import pytest
import torch

from src.direct_meta_checkpoint import (
    load_direct_meta_checkpoint, predict_direct_meta_config,
    save_direct_meta_checkpoint, save_selected_meta_model,
)
from src.meta_hpo_models import DirectBERNNMetaModel, _decode_direct
from src.zero_shot_recommender.meta_features import META_FEATURE_NAMES


def test_reload_preserves_predictions_and_normalization(tmp_path):
    n = len(META_FEATURE_NAMES)
    model = DirectBERNNMetaModel(n, 8, {"dloss": "no", "kan": False})
    mean = np.arange(n, dtype=np.float32)
    scale = np.linspace(1, 3, n, dtype=np.float32)
    raw = np.linspace(-2, 4, n, dtype=np.float32)
    with torch.no_grad():
        expected = _decode_direct(model.forward(torch.tensor(((raw - mean) / scale)[None])),
                                  0, 17, model.fixed)
    path = save_direct_meta_checkpoint(tmp_path / "model.pt", model, meta_mean=mean,
                                      meta_scale=scale, max_warmup=17, metadata={"round": 28})
    restored, checkpoint = load_direct_meta_checkpoint(path)
    assert predict_direct_meta_config(restored, checkpoint, raw) == expected
    assert predict_direct_meta_config(restored, checkpoint, dict(zip(META_FEATURE_NAMES, raw))) == expected
    assert checkpoint["metadata"]["round"] == 28
    for name, tensor in model.model.state_dict().items():
        assert torch.equal(tensor, restored.model.state_dict()[name])


def test_each_round_survives_and_best_metrics_are_independent(tmp_path):
    n = len(META_FEATURE_NAMES)
    model = DirectBERNNMetaModel(n, 8)
    kwargs = dict(meta_mean=np.zeros(n), meta_scale=np.ones(n), max_warmup=50)
    for step, error, mcc in [(0, 0.2, 0.1), (1, 0.5, 0.7), (2, 0.8, 0.4)]:
        # Save before the expensive target evaluation too.
        metadata = {"round": step, "benchmark_prediction_error": error}
        path = save_selected_meta_model(tmp_path, model, metadata=metadata, **kwargs)
        assert path.exists()
        metadata["alzheimer_valid_mcc"] = mcc
        save_selected_meta_model(tmp_path, model, metadata=metadata, **kwargs)
    directory = tmp_path / "meta_checkpoints"
    assert len(list(directory.glob("round_*.pt"))) == 3
    assert load_direct_meta_checkpoint(directory / "best_benchmark.pt")[1]["metadata"]["round"] == 0
    assert load_direct_meta_checkpoint(directory / "best_alzheimer.pt")[1]["metadata"]["round"] == 1


def test_incompatible_feature_order_is_rejected(tmp_path):
    n = len(META_FEATURE_NAMES)
    path = save_direct_meta_checkpoint(tmp_path / "model.pt", DirectBERNNMetaModel(n, 8),
                                      meta_mean=np.zeros(n), meta_scale=np.ones(n),
                                      max_warmup=50, metadata={})
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["meta_feature_names"].reverse()
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="names/order"):
        load_direct_meta_checkpoint(path)


def test_failed_save_leaves_previous_checkpoint_intact(tmp_path, monkeypatch):
    n = len(META_FEATURE_NAMES)
    model = DirectBERNNMetaModel(n, 8)
    kwargs = dict(meta_mean=np.zeros(n), meta_scale=np.ones(n), max_warmup=50, metadata={})
    path = save_direct_meta_checkpoint(tmp_path / "model.pt", model, **kwargs)
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_direct_meta_checkpoint(path, model, **kwargs)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_runner_checkpoints_before_target_fit_and_links_ledger(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import run_synchronized_meta_hpo as runner

    n = len(META_FEATURE_NAMES)
    model = DirectBERNNMetaModel(n, 32)
    with torch.no_grad():
        config = _decode_direct(model.forward(torch.zeros(1, n)), 0, 50, {})
    monkeypatch.setattr(sys, "argv", ["run_synchronized_meta_hpo.py", "--output-dir", str(tmp_path),
                                     "--n-trials", "1", "--no-wandb"])
    monkeypatch.setattr(runner.hp_search, "load_dataset",
                        lambda name: (np.ones((4, 2)), np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])))
    monkeypatch.setattr(runner.hp_search, "load_fixed_test_dataset", lambda name: None)
    monkeypatch.setattr(runner, "extract_meta_features", lambda *args: dict.fromkeys(META_FEATURE_NAMES, 1.0))
    monkeypatch.setattr(runner, "sample_bernn_config", lambda *args: config.copy())
    monkeypatch.setattr(runner, "train_direct_meta_model", lambda *args, **kwargs: (model, [], {}))

    def run_trial(cfg, run, *args, **kwargs):
        if run.dataset == "massbench_alzheimer":
            _, saved = load_direct_meta_checkpoint(tmp_path / "meta_checkpoints" / "round_0000.pt")
            assert saved["metadata"]["alzheimer_config"] == cfg
            assert "alzheimer_valid_mcc" not in saved["metadata"]
            raise RuntimeError("target evaluation failed")
        return 0.5, {"test_mcc": 0.4}

    monkeypatch.setattr(runner.hp_search, "run_trial", run_trial)
    runner.main()
    record = json.loads((tmp_path / "rounds.jsonl").read_text())
    _, checkpoint = load_direct_meta_checkpoint(tmp_path / record["meta_checkpoint"])
    assert checkpoint["metadata"]["alzheimer_valid_mcc"] == -1.0
    assert "target evaluation failed" in checkpoint["metadata"]["alzheimer_metrics"]["error"]
