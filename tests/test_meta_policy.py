import json
from pathlib import Path

import numpy as np

from src import meta_policy
from src.baselines import MODEL_EXAMPLES
from src.code_challenge import _validate_code
from src.evolutionary_meta import PolicyShape


def test_meta_model_is_an_additional_valid_bernn_example():
    code = MODEL_EXAMPLES["bernn_meta_predicted"]["code"]
    assert "predict_meta_bernn_config(X_train, y_train, batches_train)" in code
    assert "BERNN — Meta-model predicted" == MODEL_EXAMPLES["bernn_meta_predicted"]["name"]
    _validate_code(code, "meta BERNN")


def test_predict_uses_published_policy(monkeypatch, tmp_path):
    shape = PolicyShape(len(meta_policy.META_FEATURE_NAMES), 2)
    policy = tmp_path / "best_policy.npz"
    np.savez_compressed(
        policy,
        genome=np.zeros(shape.genome_size, dtype=np.float32),
        meta_mean=np.zeros(shape.n_inputs, dtype=np.float32),
        meta_scale=np.ones(shape.n_inputs, dtype=np.float32),
        n_inputs=np.asarray([shape.n_inputs]),
        hidden_size=np.asarray([shape.hidden_size]),
    )
    monkeypatch.setattr(meta_policy, "hf_hub_download", lambda *args, **kwargs: str(policy))
    monkeypatch.setattr(
        meta_policy,
        "extract_meta_features",
        lambda X, y, batches: {name: 0.0 for name in meta_policy.META_FEATURE_NAMES},
    )
    config = meta_policy.predict_meta_bernn_config([[0]], [0], ["batch"])
    assert config["model_type"] == "joint"
    assert config["dloss"] == "no"


def test_publish_never_replaces_a_better_persistent_score(monkeypatch, tmp_path):
    policy = tmp_path / "best_policy.npz"
    policy.write_bytes(b"policy")
    monkeypatch.setattr(meta_policy, "load_policy_metadata", lambda *args, **kwargs: {"validation_score": 0.7})
    published, previous = meta_policy.publish_policy_if_improved(
        policy, {"validation_score": 0.6}, repo_id="owner/model"
    )
    assert published is False
    assert previous == 0.7


def test_missing_persistent_score_allows_first_publication(monkeypatch, tmp_path):
    policy = tmp_path / "best_policy.npz"
    policy.write_bytes(b"policy")
    monkeypatch.setattr(meta_policy, "load_policy_metadata", lambda *args, **kwargs: None)

    class FakeApi:
        def __init__(self, token=None):
            self.operations = None
        def create_repo(self, *args, **kwargs):
            return None
        def create_commit(self, **kwargs):
            self.operations = kwargs["operations"]

    monkeypatch.setattr(meta_policy, "HfApi", FakeApi)
    published, previous = meta_policy.publish_policy_if_improved(
        policy, {"validation_score": 0.6}, repo_id="owner/model"
    )
    assert published is True
    assert previous is None
