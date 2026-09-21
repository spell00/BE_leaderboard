import json

import pandas as pd

from src.dataset_submission import stage_dataset_proposal
from src.zero_shot_recommender.meta_features import META_FEATURE_NAMES


def test_staging_new_dataset_extracts_and_persists_meta_features(tmp_path, monkeypatch):
    from src import dataset_submission

    monkeypatch.setattr(dataset_submission, "STAGING_ROOT", tmp_path / "staging")
    monkeypatch.delenv("HF_DATASET_SUBMISSIONS_REPO", raising=False)
    monkeypatch.delenv("HF_DATASET_SUBMISSIONS_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    csv_path = tmp_path / "new_dataset.csv"
    pd.DataFrame(
        {
            "name": [f"sample_{i}" for i in range(8)],
            "batch": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "label": ["x", "y", "x", "y", "x", "y", "x", "y"],
            "f1": [0.1, 0.2, 0.4, 0.3, 1.0, 1.2, 0.9, 1.1],
            "f2": [2.0, 2.2, 1.9, 2.1, 3.0, 3.1, 2.9, 3.2],
            "f3": [5.0, 4.8, 5.2, 5.1, 6.0, 6.2, 5.9, 6.1],
        }
    ).to_csv(csv_path, index=False)

    record = stage_dataset_proposal(
        csv_path,
        submitted_by="tester",
        title="New dataset",
        version="1",
        description="test",
        modality="proteomics",
        task="classification",
        provenance="synthetic test data",
        license_name="CC0",
        redistribution_confirmed=True,
    )

    assert record["meta_feature_names"] == list(META_FEATURE_NAMES)
    assert list(record["meta_features"]) == list(META_FEATURE_NAMES)
    assert all(isinstance(record["meta_features"][name], float) for name in META_FEATURE_NAMES)

    metadata_path = (
        dataset_submission.STAGING_ROOT
        / record["submission_id"]
        / "metadata.json"
    )
    saved = json.loads(metadata_path.read_text())
    assert saved["meta_feature_names"] == list(META_FEATURE_NAMES)
    assert saved["meta_features"] == record["meta_features"]
