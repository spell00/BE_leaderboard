"""Trial ingestion and meta-learning table construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import TrialRecord


def append_trial(path: str | Path, record: TrialRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record.to_dict(), default=str) + "\n")


def load_trials(paths: Iterable[str | Path]) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(TrialRecord.from_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"Invalid trial record at {path}:{line_number}: {exc}") from exc
    return records


def best_records(records: Iterable[TrialRecord], *, per_family: bool = False) -> list[TrialRecord]:
    """Select best complete result per dataset, optionally per model family."""
    winners: dict[tuple[str, ...], TrialRecord] = {}
    for record in records:
        if record.status != "complete":
            continue
        key = (record.dataset_id, record.model_family) if per_family else (record.dataset_id,)
        if key not in winners or float(record.score) > float(winners[key].score):
            winners[key] = record
    return list(winners.values())


def best_record_groups(records: Iterable[TrialRecord]) -> dict[str, list[TrialRecord]]:
    """Return dataset winners globally and for every recommendable target.

    Classical heads get an individual model. Neural configurations are grouped
    by model family, since their head is part of the end-to-end architecture.
    """
    complete = [record for record in records if record.status == "complete"]
    groups: dict[str, list[TrialRecord]] = {"global": best_records(complete)}
    targets = sorted({
        (record.model_family, str(record.config.get("head_type", "neural")))
        for record in complete
    })
    for family, head in targets:
        selected = [
            record for record in complete
            if record.model_family == family
            and str(record.config.get("head_type", "neural")) == head
        ]
        key = f"{family}__{head}"
        groups[key] = best_records(selected)
    return {key: value for key, value in groups.items() if value}
