"""Loader used only by the one-time final evaluation command."""

from __future__ import annotations

import json
from pathlib import Path


def load_final_test_dataset_ids(path: str | Path) -> tuple[str, ...]:
    source = Path(path)
    raw = json.loads(source.read_text())
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported final-test manifest schema in {source}")
    datasets = tuple(str(value) for value in raw.get("datasets", ()))
    if not datasets:
        raise ValueError("Final-test manifest must contain at least one dataset")
    if len(datasets) != len(set(datasets)):
        raise ValueError("Final-test manifest contains duplicate datasets")
    return datasets
