from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["name", "prediction"]


@dataclass
class SubmissionValidationResult:
    valid: bool
    message: str
    frame: pd.DataFrame | None = None


def load_and_validate_submission(file_path: str | Path) -> SubmissionValidationResult:
    """Load a CSV submission and ensure required schema is present.

    Expected columns: name, prediction
    """
    path = Path(file_path)
    if not path.exists():
        return SubmissionValidationResult(False, f"File not found: {path}")

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return SubmissionValidationResult(False, f"Could not read CSV: {exc}")

    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        return SubmissionValidationResult(
            False,
            "Missing required columns: " + ", ".join(missing),
        )

    if frame.empty:
        return SubmissionValidationResult(False, "Submission is empty.")

    if frame["name"].duplicated().any():
        return SubmissionValidationResult(False, "Duplicate name values detected.")

    cleaned = frame[REQUIRED_COLUMNS].copy()
    cleaned["name"] = cleaned["name"].astype(str)
    cleaned["prediction"] = cleaned["prediction"].astype(str)

    return SubmissionValidationResult(True, "Submission validated.", cleaned)
