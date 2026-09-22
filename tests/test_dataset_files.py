from pathlib import Path

import pandas as pd

from src.dataset_files import (
    ensure_all_dataset_file,
    inference_filenames,
    research_source_filenames,
)
from src.dataset_tasks import prepare_research_source_frame


def test_all_file_is_literal_train_plus_test_and_keeps_unlabeled_rows(tmp_path):
    dataset = "demo"
    base = tmp_path / "data" / "datasets" / dataset
    base.mkdir(parents=True)

    train = pd.DataFrame(
        {
            "name": ["a", "b"],
            "batch": ["1", "2"],
            "label": ["case", "control"],
            "f1": [1.0, 2.0],
        }
    )
    test = pd.DataFrame(
        {
            "name": ["c"],
            "batch": ["3"],
            "f1": [3.0],
        }
    )
    train.to_csv(base / f"{dataset}_train.csv", index=False)
    test.to_csv(base / f"{dataset}_test.csv", index=False)

    all_path = ensure_all_dataset_file(tmp_path, dataset)
    merged = pd.read_csv(all_path)

    assert merged["name"].tolist() == ["a", "b", "c"]
    assert pd.isna(merged.loc[2, "label"])

    supervised = prepare_research_source_frame(dataset, merged)
    assert supervised["name"].tolist() == ["a", "b"]


def test_research_source_choices_prefer_all_then_train(tmp_path):
    dataset = "demo"
    base = tmp_path / "data" / "datasets" / dataset
    base.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "name": ["a", "b"],
            "batch": ["1", "2"],
            "label": ["x", "y"],
            "f1": [1.0, 2.0],
        }
    )
    frame.to_csv(base / f"{dataset}_train.csv", index=False)
    frame.iloc[:1].to_csv(base / f"{dataset}_test.csv", index=False)

    names = research_source_filenames(tmp_path, dataset)
    assert names == [f"{dataset}_all.csv", f"{dataset}_train.csv"]


def test_inference_files_exclude_predictions_csv(tmp_path):
    dataset = "demo"
    base = tmp_path / "data" / "datasets" / dataset
    base.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "name": ["a", "b"],
            "batch": ["1", "2"],
            "label": ["x", "y"],
            "f1": [1.0, 2.0],
        }
    )
    frame.to_csv(base / f"{dataset}_train.csv", index=False)
    frame.iloc[:1].to_csv(base / f"{dataset}_test.csv", index=False)
    pd.DataFrame({"name": ["a"], "prediction": ["x"]}).to_csv(
        base / f"{dataset}_predictions.csv",
        index=False,
    )

    names = inference_filenames(tmp_path, dataset)
    assert f"{dataset}_predictions.csv" not in names
    assert names[0] == f"{dataset}_test.csv"
