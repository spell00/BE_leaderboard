"""Isolated evaluator process for Real leaderboard submissions.

This module intentionally has no dependency on app.py, so starting an evaluator does
not rebuild the Gradio UI or repeat application-level database/Hub startup work.
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback
from pathlib import Path

from src.code_challenge import CodeValidationError, run_code_submission


def evaluate(payload: dict) -> dict:
    try:
        _, metrics, _, _ = run_code_submission(**payload)
        return {"ok": True, "metrics": metrics}
    except CodeValidationError as exc:
        return {"ok": False, "kind": "validation", "message": str(exc)}
    except BaseException as exc:
        return {
            "ok": False,
            "kind": "error",
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }


def main(input_path: str, output_path: str) -> int:
    source = Path(input_path)
    destination = Path(output_path)
    with source.open("rb") as fh:
        payload = pickle.load(fh)
    result = evaluate(payload)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m src.submission_worker INPUT.pkl OUTPUT.pkl")
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
