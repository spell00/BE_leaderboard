"""Dataset-level partitions for evolutionary model development."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SPLIT_FILE = Path(__file__).resolve().parent.parent / "config" / "evolution_development_datasets.json"


@dataclass(frozen=True)
class DatasetPartitions:
    train: tuple[str, ...]
    validation: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = {
            "train": self.train,
            "validation": self.validation,
        }
        for name, values in groups.items():
            if not values:
                raise ValueError(f"Dataset partition '{name}' must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"Dataset partition '{name}' contains duplicates")

        for left_name, left in groups.items():
            for right_name, right in groups.items():
                if left_name >= right_name:
                    continue
                overlap = sorted(set(left) & set(right))
                if overlap:
                    raise ValueError(
                        f"Dataset partitions '{left_name}' and '{right_name}' overlap: {overlap}"
                    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Sequence[str]]) -> DatasetPartitions:
        return cls(
            train=tuple(value.get("train", ())),
            validation=tuple(value.get("validation", ())),
        )

    def fitness_datasets(self) -> tuple[str, ...]:
        """Datasets allowed to influence evolutionary fitness."""
        return self.train

    def selection_datasets(self) -> tuple[str, ...]:
        """Held-out datasets allowed for model selection and early stopping."""
        return self.validation


def load_dataset_partitions(path: str | Path = DEFAULT_SPLIT_FILE) -> DatasetPartitions:
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported dataset split schema in {source}")
    return DatasetPartitions.from_mapping(raw.get("partitions", {}))
