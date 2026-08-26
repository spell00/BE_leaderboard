from __future__ import annotations
print("Starting MassBench Batch Effects Leaderboard app...")

import os
import sys
import traceback
import argparse
import pickle
import subprocess
import tempfile
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from datetime import datetime
from typing import TextIO

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import pandas as pd
import json
import math
print(f"Using Python {sys.version} at {sys.executable}")
# try:
#    import gradio_client.utils as _gradio_client_utils
#
#    _orig_json_schema_to_python_type = _gradio_client_utils._json_schema_to_python_type
#
#    def _json_schema_to_python_type_bool_safe(schema, defs):
#        if isinstance(schema, bool):
#            return "Any" if schema else "None"
#        return _orig_json_schema_to_python_type(schema, defs)
#
#    _gradio_client_utils._json_schema_to_python_type = _json_schema_to_python_type_bool_safe
#except Exception:
#    pass

from src.baselines import (
    BATCH_CORRECTION_EXAMPLES,
    MODEL_EXAMPLES,
    get_baseline_text,
    BERNN_KNOBS,
    bernn_config,
    build_bernn_code,
    maybe_register_tuned,
)
from src.code_challenge import CodeValidationError
from src.database import DatabaseManager, PROJECT_VERSION, real_leaderboard_score
from src.dataset_info import get_dataset_info_markdown
from src.dataset_submission import DatasetSubmissionError, stage_dataset_proposal
from src.real_results_store import (
    load_real_result_rows,
    merge_real_result_rows,
    normalize_real_result_row,
    upload_real_result_rows,
)

print(f"Gradio version: {gr.__version__}, Pandas version: {pd.__version__}")
# print(f"Using SQLite version: {DatabaseManager.get_sqlite_version()}")
SEED_REAL_RESULTS = ROOT / "data" / "seed_real_leaderboard.json"
RUN_LOG_DIR = ROOT / "logs" / "ui_runs"
LATEST_RUN_LOG = RUN_LOG_DIR / "latest.log"
UI_LOG_MAX_CHARS = 30_000
UI_LOG_TRIM_AT_CHARS = UI_LOG_MAX_CHARS * 2
_ACTIVE_REAL_RUNS: dict[str, dict] = {}
_ACTIVE_REAL_RUNS_LOCK = threading.Lock()


def _launch_port() -> int | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=None)
    args, _ = parser.parse_known_args()
    if args.port is not None:
        return args.port

    env_port = os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT")
    if not env_port:
        return None
    try:
        return int(env_port)
    except ValueError:
        print(f"Ignoring invalid port value: {env_port!r}")
        return None

db = DatabaseManager(ROOT / "data" / "leaderboard.db")

DATASET_LABELS = {
    "massbench_adenocarcinoma": "MassBench Adenocarcinoma",
    "massbench_alzheimer": "MassBench Alzheimer",
    "massbench_benchmark": "MassBench Benchmark",
}

HF_TOKEN_SET = bool(os.getenv("HF_TOKEN"))

DEFAULT_CORRECTION_CODE = BATCH_CORRECTION_EXAMPLES["none"]["code"]
DEFAULT_MODEL_CODE = MODEL_EXAMPLES["gaussian_nb"]["code"]


def get_dataset_download_files(dataset: str) -> tuple[str | None, str | None]:
    base = ROOT / "data" / "datasets" / dataset
    train_path = base / f"{dataset}_train.csv"
    test_path = base / f"{dataset}_test.csv"
    return (
        str(train_path) if train_path.exists() else None,
        str(test_path) if test_path.exists() else None,
    )


def _format_exec_error(exc: Exception) -> str:
    tb = traceback.format_exc()
    tb_lines = [line for line in tb.strip().splitlines() if line.strip()]
    tail = "\n".join(tb_lines[-14:])
    return (
        f"Execution failed: {type(exc).__name__}: {exc}\n\n"
        "Traceback (last lines):\n"
        f"{tail}"
    )


def _finite_float(value, default: float | None = None) -> float | None:
    """Return a JSON/SQLite-safe float, or default for missing/non-finite values."""
    try:
        if value is None:
            return default
        numeric = float(value)
        if not math.isfinite(numeric):
            return default
        return numeric
    except Exception:
        return default


def _finite_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        numeric = float(value)
        if not math.isfinite(numeric):
            return default
        return int(numeric)
    except Exception:
        return default


def _parse_datetime(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _existing_real_result_ids() -> set[str]:
    return {
        normalize_real_result_row(row)["result_id"]
        for row in db.get_leaderboard()
    }


def _insert_real_result_rows(rows: list[dict], source: str) -> int:
    existing_ids = _existing_real_result_ids()
    inserted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = normalize_real_result_row(row)
        if normalized["result_id"] in existing_ids:
            continue
        try:
            submission = db.create_submission(
                username=normalized["username"],
                dataset=normalized["dataset"],
                submission_name=normalized["submission_name"],
                correction_code=normalized["correction_code"],
                model_code=normalized["model_code"],
                is_public=bool(normalized["is_public"]),
                created_at=_parse_datetime(normalized.get("created_at")),
                version_created=normalized.get("version_created") or None,
            )
            db.create_score(
                submission_id=submission.id,
                accuracy=_finite_float(normalized.get("accuracy"), 0.0),
                macro_f1=_finite_float(normalized.get("macro_f1"), 0.0),
                n_samples=_finite_int(normalized.get("n_samples"), 0),
                test_mcc=_finite_float(normalized.get("test_mcc"), 0.0),
                valid_mcc=_finite_float(normalized.get("valid_mcc"), 0.0),
                valid_mcc_folds=normalized.get("valid_mcc_folds", []),
                train_mcc=_finite_float(normalized.get("train_mcc"), -1.0),
                log_loss=_finite_float(normalized.get("log_loss")) if normalized.get("log_loss") is not None else None,
                brier_score=_finite_float(normalized.get("brier_score")) if normalized.get("brier_score") is not None else None,
                ece=_finite_float(normalized.get("ece")) if normalized.get("ece") is not None else None,
                batch_silhouette=_finite_float(normalized.get("batch_silhouette")) if normalized.get("batch_silhouette") is not None else None,
                batch_centroid_dispersion=_finite_float(normalized.get("batch_centroid_dispersion")) if normalized.get("batch_centroid_dispersion") is not None else None,
                batch_nbe=_finite_float(normalized.get("batch_nbe")) if normalized.get("batch_nbe") is not None else None,
                batch_nmi=_finite_float(normalized.get("batch_nmi")) if normalized.get("batch_nmi") is not None else None,
                batch_nri=_finite_float(normalized.get("batch_nri")) if normalized.get("batch_nri") is not None else None,
                version=normalized.get("version_evaluated") or None,
                created_at=_parse_datetime(normalized.get("created_at")),
            )
            inserted += 1
            existing_ids.add(normalized["result_id"])
        except Exception as exc:
            print(f"[{source}] Skipped Real leaderboard row: {type(exc).__name__}: {exc}")
    if inserted:
        print(f"[{source}] Inserted {inserted} Real leaderboard rows")
    return inserted


def seed_real_leaderboard_missing_rows() -> None:
    """Load committed aggregate baseline rows that are missing from the DB."""
    if not SEED_REAL_RESULTS.exists():
        return
    try:
        payload = json.loads(SEED_REAL_RESULTS.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[seed] Could not read {SEED_REAL_RESULTS}: {type(exc).__name__}: {exc}")
        return
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    rows = [{**row, "is_public": True} for row in rows if isinstance(row, dict)]
    _insert_real_result_rows(rows, "seed")


def sync_real_leaderboard_from_hub() -> None:
    rows = load_real_result_rows()
    if rows:
        _insert_real_result_rows(rows, "hf-real-results")


def sync_real_leaderboard_to_hub() -> None:
    rows = merge_real_result_rows(db.get_leaderboard())
    if rows:
        count = upload_real_result_rows(rows)
        if count:
            print(f"[hf-real-results] Synced {count} Real leaderboard rows")


def _json_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid Gradio JSON serialization errors from NaN/inf dataframe cells."""
    if df.empty:
        return df
    clean = df.replace([float("inf"), float("-inf")], pd.NA)
    return clean.astype(object).where(pd.notna(clean), None)


def _captured_logs(buffer: StringIO) -> str:
    text = buffer.getvalue().strip()
    return text if text else "No logs captured for this run."


def _trim_log_buffer(buffer: StringIO) -> None:
    """Bound the in-memory UI log while preserving complete on-disk logs."""
    if buffer.tell() <= UI_LOG_TRIM_AT_CHARS:
        return
    tail = buffer.getvalue()[-UI_LOG_MAX_CHARS:]
    buffer.seek(0)
    buffer.truncate(0)
    buffer.write("[showing latest log tail]\n")
    buffer.write(tail)


class _RunLogCapture:
    """Capture stdout/stderr in memory and tee it to a refreshable log file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._console: TextIO = sys.stdout
        self._buffer = StringIO()
        self._file: TextIO = path.open("w", encoding="utf-8", buffering=1)
        self._latest: TextIO = LATEST_RUN_LOG.open("w", encoding="utf-8", buffering=1)

    def write(self, text: str) -> int:
        self._buffer.write(text)
        _trim_log_buffer(self._buffer)
        written = self._file.write(text)
        self._latest.write(text)
        self._console.write(text)
        self._file.flush()
        self._latest.flush()
        self._console.flush()
        return written

    def flush(self) -> None:
        self._file.flush()
        self._latest.flush()
        self._console.flush()

    def close(self) -> None:
        self._file.close()
        self._latest.close()

    def getvalue(self) -> str:
        return self._buffer.getvalue()


def _slug(value: str) -> str:
    safe = []
    for ch in str(value).strip().lower():
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        elif ch in {" ", "/", "\\", "."}:
            safe.append("-")
    return "".join(safe).strip("-") or "unknown"


def _new_run_log_path(team: str, model_name: str, dataset: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUN_LOG_DIR / f"{ts}_{_slug(dataset)}_{_slug(team)}_{_slug(model_name)}.log"


def _hf_username(
    profile: gr.OAuthProfile | None = None,
    request: gr.Request | None = None,
) -> str:
    if profile is None:
        username = ""
    else:
        username = getattr(profile, "username", None) or getattr(profile, "name", None)
    if not username and request is not None:
        username = getattr(request, "username", None)
    return str(username or "").strip()


def read_latest_run_logs() -> str:
    if not LATEST_RUN_LOG.exists():
        return "No run logs have been written yet."

    max_bytes = UI_LOG_MAX_CHARS * 4
    with LATEST_RUN_LOG.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        text = fh.read().decode("utf-8", errors="replace").strip()

    if not text:
        return "Latest run log is currently empty."
    if size > max_bytes or len(text) > UI_LOG_MAX_CHARS:
        return "[showing latest log tail]\n" + text[-UI_LOG_MAX_CHARS:]
    return text


sync_real_leaderboard_from_hub()


class SubmissionCancelled(Exception):
    """Raised when a user stops an active Real benchmark run."""


def _active_real_run_key(team: str) -> str:
    return str(team or "").strip() or "__anonymous__"


def _terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _forward_worker_output(process: subprocess.Popen, stream: TextIO) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        stream.write(line)
        stream.flush()


def _run_code_submission_cancellable(
    *,
    run_key: str,
    team: str,
    model_name: str,
    dataset: str,
    correction_code: str,
    model_code: str,
) -> dict:
    with _ACTIVE_REAL_RUNS_LOCK:
        existing = _ACTIVE_REAL_RUNS.get(run_key)
        if existing and not existing.get("stop_event", threading.Event()).is_set():
            raise RuntimeError("A Real benchmark submission is already running for this Hugging Face user.")
        stop_event = threading.Event()
        _ACTIVE_REAL_RUNS[run_key] = {
            "stop_event": stop_event,
            "process": None,
            "team": team,
            "model_name": model_name,
            "dataset": dataset,
            "started_at": time.time(),
        }

    try:
        with tempfile.TemporaryDirectory(prefix="massbench-submission-") as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "input.pkl"
            output_path = temp_root / "output.pkl"
            with input_path.open("wb") as fh:
                pickle.dump(
                    {
                        "team": team,
                        "model_name": model_name,
                        "dataset": dataset,
                        "correction_code": correction_code,
                        "model_code": model_code,
                    },
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            process = subprocess.Popen(
                [sys.executable, "-m", "src.submission_worker", str(input_path), str(output_path)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            log_stream = sys.stdout
            output_thread = threading.Thread(
                target=_forward_worker_output,
                args=(process, log_stream),
                name=f"submission-log-{process.pid}",
                daemon=True,
            )
            output_thread.start()

            with _ACTIVE_REAL_RUNS_LOCK:
                if run_key in _ACTIVE_REAL_RUNS:
                    _ACTIVE_REAL_RUNS[run_key]["process"] = process

            while process.poll() is None:
                if stop_event.wait(timeout=0.5):
                    print(f"[submission] Stop requested for {team} / {model_name} on {dataset}", flush=True)
                    _terminate_process(process)
                    output_thread.join(timeout=2)
                    raise SubmissionCancelled("Submission stopped by user before completion.")

            output_thread.join(timeout=5)
            if stop_event.is_set():
                raise SubmissionCancelled("Submission stopped by user before completion.")
            if not output_path.exists():
                raise RuntimeError(
                    f"Submission worker exited unexpectedly with code {process.returncode}."
                )
            with output_path.open("rb") as fh:
                payload = pickle.load(fh)

        if payload.get("ok"):
            return payload["metrics"]
        if payload.get("kind") == "validation":
            raise CodeValidationError(payload.get("message", "Submission rejected."))
        message = payload.get("message") or "Submission worker failed."
        tb = payload.get("traceback")
        if tb:
            message = f"{payload.get('type', 'Error')}: {message}\n\n{tb}"
        raise RuntimeError(message)
    finally:
        with _ACTIVE_REAL_RUNS_LOCK:
            current = _ACTIVE_REAL_RUNS.get(run_key)
            if current and current.get("stop_event") is stop_event:
                _ACTIVE_REAL_RUNS.pop(run_key, None)


def stop_real(
    profile: gr.OAuthProfile | None = None,
    request: gr.Request | None = None,
) -> tuple[str, str]:
    team = _hf_username(profile, request)
    if not team:
        return "Please sign in with Hugging Face before stopping a submission.", read_latest_run_logs()
    run_key = _active_real_run_key(team)
    with _ACTIVE_REAL_RUNS_LOCK:
        active = _ACTIVE_REAL_RUNS.get(run_key)
        if not active:
            return "No active Real benchmark submission found for your Hugging Face user.", read_latest_run_logs()
        active["stop_event"].set()
        process = active.get("process")
    _terminate_process(process)
    return "Stop requested. The active Real benchmark submission is being terminated.", read_latest_run_logs()
seed_real_leaderboard_missing_rows()
sync_real_leaderboard_to_hub()


def get_real_board(dataset: str | None = None) -> pd.DataFrame:
    leaderboard = db.get_leaderboard(dataset)
    if not leaderboard:
        return pd.DataFrame(columns=["username", "dataset", "submission_name", "score", "valid_mcc", "test_mcc", "created_at"])

    core_cols = [
        "username",
        "dataset",
        "submission_name",
        "score",
        "test_mcc",
        "valid_mcc",
        "valid_mcc_folds",
        "accuracy",
        "macro_f1",
        "n_samples",
        "created_at",
        "batch_nbe",  # Always include NBE
    ]
    # Remove log_loss from optional columns, always include batch_nbe
    optional_cols = [
        "brier_score",
        "ece",
        "batch_silhouette",
        "batch_centroid_dispersion",
        "batch_nmi",
        "batch_nri",
    ]

    # Show optional metric columns only when at least one row has a value.
    present_optional_cols = []
    for col in optional_cols:
        if any(row.get(col) is not None for row in leaderboard):
            present_optional_cols.append(col)

    display_cols = core_cols + present_optional_cols
    filtered = []
    for row in leaderboard:
        rounded_row = {}
        for k in display_cols:
            v = row.get(k)
            if isinstance(v, float):
                rounded_row[k] = round(v, 4)
            else:
                rounded_row[k] = v
        filtered.append(rounded_row)

    return _json_safe_dataframe(pd.DataFrame(filtered))




def get_dataset_info(dataset: str) -> str:
    """Get formatted dataset information."""
    return get_dataset_info_markdown(dataset)


# Dataset submission order metadata
DATASET_SUBMISSION_ORDER = {
    "massbench_benchmark": 1,
    "massbench_adenocarcinoma": 2,
    "massbench_alzheimer": 3,
}

def get_dataset_dropdown_choices():
    """Get dataset dropdown choices sorted by submission order."""
    sorted_datasets = sorted(
        DATASET_LABELS.items(),
        key=lambda x: DATASET_SUBMISSION_ORDER.get(x[0], 999)
    )
    return [(label, key) for key, label in sorted_datasets]

def apply_dataset_selection(selected_dataset):
    """Apply the selected dataset and update main view."""
    info = get_dataset_info(selected_dataset)
    board = get_real_board(selected_dataset)
    train_path, test_path = get_dataset_download_files(selected_dataset)
    return (
        selected_dataset,
        board,
        info,
        str(train_path) if train_path else "",
        str(test_path) if test_path else "",
    )


def submit_dataset_proposal(
    title: str,
    version: str,
    description: str,
    modality: str,
    task: str,
    provenance: str,
    license_name: str,
    redistribution_confirmed: bool,
    csv_file,
    profile: gr.OAuthProfile | None = None,
    request: gr.Request | None = None,
) -> str:
    """Validate and privately stage a proposed benchmark dataset."""
    username = _hf_username(profile, request)
    if not username:
        return "Please sign in with Hugging Face before adding a dataset."
    if csv_file is None:
        return "Choose one matrix-ready CSV file."

    csv_path = getattr(csv_file, "name", None) or str(csv_file)
    try:
        record = stage_dataset_proposal(
            csv_path,
            submitted_by=username,
            title=title,
            version=version,
            description=description,
            modality=modality,
            task=task,
            provenance=provenance,
            license_name=license_name,
            redistribution_confirmed=bool(redistribution_confirmed),
        )
    except DatasetSubmissionError as exc:
        return f"Dataset rejected: {exc}"
    except Exception as exc:
        return f"Dataset could not be staged: {type(exc).__name__}: {exc}"

    if record.get("durable_staging") != "huggingface-private-dataset":
        return (
            "The file passed validation, but durable private staging is not "
            "configured. Please contact the maintainer before uploading again."
        )
    return (
        f"Dataset {record['submission_id']} passed automatic validation and was "
        f"submitted for private curator review: {record['n_samples']} samples, "
        f"{record['n_features']} features, {record['n_batches']} batches, and "
        f"{record['n_classes']} labeled classes."
    )

def submit_real(
    model_name: str,
    dataset: str,
    correction_code: str,
    model_code: str,
    custom_pip: str = "",
    profile: gr.OAuthProfile | None = None,
    request: gr.Request | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Run a real benchmark submission and return updated leaderboard, status, and logs."""
    team = _hf_username(profile, request)
    model_name = str(model_name or "")
    dataset = str(dataset or "")
    correction_code = str(correction_code or "")
    model_code = str(model_code or "")
    custom_pip = str(custom_pip or "")
    print(f"[submission] Received submission from {team.strip() or 'anonymous'} / {model_name.strip() or 'unnamed'} on {dataset}", flush=True)
    if not dataset.strip():
        return get_real_board(dataset), "Dataset is required.", ""
    if not team:
        return get_real_board(dataset), "Please sign in with Hugging Face before submitting.", ""
    if not model_name.strip():
        return get_real_board(dataset), "Submission name is required.", ""

    logs_buffer = _RunLogCapture(_new_run_log_path(team, model_name, dataset))
    print(f"[submission] Logs will be captured to {logs_buffer.path}", flush=True)
    stdout_redirect = redirect_stdout(logs_buffer)
    stderr_redirect = redirect_stderr(logs_buffer)
    stdout_redirect.__enter__()
    stderr_redirect.__enter__()

    def _finish(board: pd.DataFrame, message: str) -> tuple[pd.DataFrame, str, str]:
        captured = _captured_logs(logs_buffer)
        try:
            stderr_redirect.__exit__(None, None, None)
            stdout_redirect.__exit__(None, None, None)
        except Exception:
            pass
        try:
            logs_buffer.close()
        except Exception:
            pass
        return board, message, captured

    # with redirect_stdout(logs_buffer), redirect_stderr(logs_buffer):
    print(f"[submission] Starting {team.strip() or 'anonymous'} / {model_name.strip() or 'unnamed'} on {dataset}", flush=True)

    if custom_pip and custom_pip.strip():
        pkgs = [p.strip() for p in custom_pip.replace(",", " ").split() if p.strip()]
        if pkgs:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + pkgs,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                install_msg = result.stdout + "\n" + result.stderr
            except Exception as exc:
                install_msg = f"Install failed: {exc}"
            print("[pip-install]")
            print(install_msg)

    print(f"[submission] Running code submission for {team.strip() or 'anonymous'} / {model_name.strip() or 'unnamed'} on {dataset}", flush=True)

    print(f"[submission] boarded dataset: {dataset}", flush=True)

    if not HF_TOKEN_SET:
        print(f"[submission] HF_TOKEN is not configured for submission on {dataset}", flush=True)
        return _finish(get_real_board(dataset), "HF_TOKEN is not configured on this Space. The evaluator cannot access private data — contact the organiser.")

    print(f"[submission] Running code submission for {team.strip()} / {model_name.strip()} on {dataset}", flush=True)
    try:
        try:
            metrics = _run_code_submission_cancellable(
                run_key=_active_real_run_key(team.strip()),
                team=team.strip(),
                model_name=model_name.strip(),
                dataset=dataset,
                correction_code=correction_code,
                model_code=model_code,
            )
        except CodeValidationError as exc:
            return _finish(get_real_board(dataset), f"Submission rejected: {exc}")
        except SubmissionCancelled as exc:
            return _finish(get_real_board(dataset), str(exc))

        print(f"[submission] Code submission completed for {team.strip()} / {model_name.strip()} on {dataset}", flush=True)

        submission = db.create_submission(
            username=team.strip(),
            dataset=dataset,
            submission_name=model_name.strip(),
            correction_code=correction_code,
            model_code=model_code,
            is_public=False,
        )

        print(f"[submission] Created submission record {submission.id} for {team.strip()} / {model_name.strip()} on {dataset}", flush=True)
        db.create_score(
            submission_id=submission.id,
            accuracy=_finite_float(metrics.get("accuracy"), 0.0),
            macro_f1=_finite_float(metrics.get("macro_f1"), 0.0),
            n_samples=_finite_int(metrics.get("n_samples"), 0),
            test_mcc=_finite_float(metrics.get("test_mcc", metrics.get("mcc")), 0.0),
            # Use the model-reported CV metrics (e.g. BERNN's mean validation MCC),
            # not a hardcoded -1 — run_code_submission surfaces these via extra_metrics.
            valid_mcc=_finite_float(metrics.get("valid_mcc"), -1.0),
            valid_mcc_folds=[
                float(value) for value in metrics.get("valid_mcc_folds", [])
            ],
            train_mcc=_finite_float(metrics.get("train_mcc"), -1.0),
            log_loss=_finite_float(metrics.get("log_loss")) if "log_loss" in metrics else None,
            brier_score=_finite_float(metrics.get("brier_score")) if "brier_score" in metrics else None,
            ece=_finite_float(metrics.get("ece")) if "ece" in metrics else None,
            batch_silhouette=_finite_float(metrics.get("batch_silhouette")) if "batch_silhouette" in metrics else None,
            batch_centroid_dispersion=_finite_float(metrics.get("batch_centroid_dispersion")) if "batch_centroid_dispersion" in metrics else None,
            batch_nbe=_finite_float(metrics.get("batch_nbe")) if "batch_nbe" in metrics else None,
            batch_nmi=_finite_float(metrics.get("batch_nmi")) if "batch_nmi" in metrics else None,
            batch_nri=_finite_float(metrics.get("batch_nri")) if "batch_nri" in metrics else None,
        )

        print(f"[submission] Recorded score for submission {submission.id} for {team.strip()} / {model_name.strip()} on {dataset}", flush=True)
        sync_real_leaderboard_to_hub()

        # Live per-family default update: if this is a BERNN submission whose CV
        # validation MCC beats the family's registered default, promote its config.
        promoted = None

        print(f"[submission] Checking for BERNN default update for {team.strip()} / {model_name.strip()} on {dataset}", flush=True)

        try:
            promoted = maybe_register_tuned(metrics.get("bernn_config"),
                                            metrics.get("valid_mcc", -1.0))
        except Exception as exc:  # never let a default-update failure break a submission
            print(f"[bernn-default] update skipped: {type(exc).__name__}: {exc}")
    except Exception as exc:
        print(f"[submission] ERROR during submission for {team.strip()} / {model_name.strip()} on {dataset}: {type(exc).__name__}: {exc}", flush=True)
        return _finish(get_real_board(dataset), _format_exec_error(exc))
        # return _finish(get_real_board(dataset), _format_exec_error(exc), _captured_logs(logs_buffer))

    print(f"[submission] Submission completed for {team.strip()} / {model_name.strip()} on {dataset} 1", flush=True)

    test_mcc = float(metrics.get("test_mcc", metrics.get("mcc", 0.0)))
    valid_mcc = float(metrics.get("valid_mcc", -1.0))
    official_score = real_leaderboard_score(valid_mcc, test_mcc)
    metrics["score"] = official_score

    msg = (
        f"Real benchmark score on {DATASET_LABELS[dataset]}. "
        f"Score={official_score:.4f} (lower of Valid MCC and Test MCC), "
        f"Valid MCC={valid_mcc:.4f}, "
        f"Test MCC={test_mcc:.4f}, "
        f"N={metrics.get('n_samples', 0)}"
    )
    if metrics.get("model_kind"):
        msg += f"\nExecuted model: {metrics['model_kind']}"
    if promoted:
        msg += (f" — new best for BERNN family '{promoted}' "
                f"(valid MCC {float(metrics.get('valid_mcc', -1.0)):.4f}); default updated")
    if "log_loss" in metrics:
        msg += f", LogLoss={float(metrics.get('log_loss', 0.0)):.4f}"
    if "brier_score" in metrics:
        msg += f", Brier={float(metrics.get('brier_score', 0.0)):.4f}"
    if "ece" in metrics:
        msg += f", ECE={float(metrics.get('ece', 0.0)):.4f}"
    if "batch_silhouette" in metrics:
        msg += f", BatchSil={float(metrics.get('batch_silhouette', -1.0)):.4f}"
    if "batch_centroid_dispersion" in metrics:
        msg += f", BatchDisp={float(metrics.get('batch_centroid_dispersion', -1.0)):.4f}"
    if "batch_nbe" in metrics:
        msg += f", NBE={float(metrics.get('batch_nbe', -1.0)):.4f}"
    if "batch_nmi" in metrics:
        msg += f", BatchNMI={float(metrics.get('batch_nmi', -1.0)):.4f}"
    if "batch_nri" in metrics:
        msg += f", BatchNRI={float(metrics.get('batch_nri', -1.0)):.4f}"

    fold_scores = metrics.get("valid_mcc_folds", [])
    if isinstance(fold_scores, list) and fold_scores:
        msg += "\n\nCross-validation scores:"
        for fold, score in enumerate(fold_scores, start=1):
            msg += f"\n- Fold {fold}: MCC={float(score):.4f}"
        if "valid_mcc_std" in metrics:
            msg += f"\n- Mean ± SD: {float(metrics['valid_mcc']):.4f} ± {float(metrics['valid_mcc_std']):.4f}"
        if metrics.get("cv_protocol"):
            msg += f"\n- Protocol: {metrics['cv_protocol']}"

    group_scores = metrics.get("group_scores") if isinstance(metrics, dict) else None
    if isinstance(group_scores, dict) and group_scores:
        lines = ["", "Per-group scores:"]
        for grp in sorted(group_scores.keys()):
            row = group_scores.get(grp, {})
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {grp}: MCC={float(row.get('test_mcc', 0.0)):.4f}, "
                f"Acc={float(row.get('accuracy', 0.0)):.4f}, "
                f"F1={float(row.get('macro_f1', 0.0)):.4f}, "
                f"N={int(row.get('n_samples', 0))}"
            )
        msg += "\n" + "\n".join(lines)

    print(f"[submission] Submission completed for {team.strip()} / {model_name.strip()} on {dataset} 2", flush=True)

    # return _finish(get_real_board(dataset), msg)
    return _finish(get_real_board(dataset), msg)

def on_board_click(
    evt: gr.SelectData,
    dataset: str,
    profile: gr.OAuthProfile | None = None,
    request: gr.Request | None = None,
) -> tuple:
    """
    Load correction/model code from the selected leaderboard row.

    Users may view:
    - their own submissions
    - public submissions

    Prevent crashes from:
    - invalid row indexes
    - filtered leaderboard mismatches
    - missing database fields
    """

    try:
        if evt is None or evt.index is None:
            return gr.update(), gr.update()

        row_index = evt.index[0]

        # IMPORTANT:
        # Load the same dataset displayed in the UI
        board = db.get_leaderboard(dataset)

        if not board:
            return gr.update(), gr.update()

        if row_index < 0 or row_index >= len(board):
            return gr.update(), gr.update()

        row = board[row_index]

        row_team = str(row.get("username", "")).strip()
        is_public = bool(row.get("is_public", False))

        current_team = _hf_username(profile, request)
        if row_team != current_team and not is_public:
            print(
                f"[load-code] Access denied. "
                f"user={current_team}, owner={row_team}"
            )

            return (
                gr.update(),
                gr.update(),
            )

        correction_code = row.get("correction_code", "")
        model_code = row.get("model_code", "")

        print(
            f"[load-code] Loaded submission from "
            f"{row_team}: {row.get('submission_name', '')}"
        )

        return (
            gr.update(value=correction_code),
            gr.update(value=model_code),
        )

    except Exception:
        print("[load-code] ERROR")
        print(traceback.format_exc())

        return (
            gr.update(),
            gr.update(),
        )


def load_baseline(choice: str, is_correction: bool) -> str:
    """Load a baseline code example."""
    if is_correction:
        if choice in BATCH_CORRECTION_EXAMPLES:
            return BATCH_CORRECTION_EXAMPLES[choice]["code"]
    else:
        if choice in MODEL_EXAMPLES:
            return MODEL_EXAMPLES[choice]["code"]
    return ""


_BERNN_ORDER = [k["key"] for k in BERNN_KNOBS]


def generate_bernn_code(*values) -> str:
    """Build a BERNN fit function from the UI control values (knob order)."""
    cfg = dict(zip(_BERNN_ORDER, values))
    for knob in BERNN_KNOBS:
        val = cfg.get(knob["key"])
        if val is None:
            continue
        if knob["kind"] == "int":
            cfg[knob["key"]] = int(val)
        elif knob["kind"] == "float":
            cfg[knob["key"]] = float(val)
        elif knob["kind"] == "bool":
            cfg[knob["key"]] = bool(val)
    return build_bernn_code(bernn_config(**cfg))


def apply_bernn_preset(preset: str) -> list:
    """Return control values (knob order) for the chosen preset."""
    cfg = bernn_config(preset)
    return [cfg[key] for key in _BERNN_ORDER]


def download_code(correction_code: str, model_code: str) -> str:
    """Create a downloadable Python file with both functions."""
    content = f'''"""
Batch Correction and Model Code
Generated: {datetime.now().isoformat()}

Available preloaded libraries:
- Fundamentals: numpy (np), scipy, pandas (pd)
- ML: scikit-learn (StandardScaler, LogisticRegression, RandomForestClassifier, etc.)
- Batch Correction: scanpy (sc), harmonypy
- Deep Learning: torch, jax/jaxlib
- bernn: TrainAEClassifierHoldout, TrainAEThenClassifierHoldout, TrainAE
- Optimization: optuna, ax-platform
- Utilities: imbalanced-learn, shap, statsmodels, networkx

"""

{correction_code}

{model_code}
'''
    return content


with gr.Blocks(title="MassBench Batch Effects Leaderboard") as demo:
    gr.Markdown(f"""
# MassBench Batch Effects Classification Leaderboard

**Project Version: {PROJECT_VERSION}**

Run a reproducible server-side benchmark or propose a matrix-ready dataset.

{get_baseline_text()}
""")
    gr.LoginButton()

    with gr.Tabs():
        with gr.TabItem("Real Leaderboard (Code Run)"):
            gr.Markdown("""
### Real Benchmark Submission
Submit batch correction and model code. Evaluation runs server-side.
- Click a leaderboard row to auto-fill code if you own it or it is public
- Supported batch correction: ComBat-like, Harmony (harmonypy), scanpy, bernn (TrainAEClassifierHoldout/TrainAEThenClassifierHoldout) methods
- Official score is the lower of validation MCC and hidden test MCC. For example, Valid MCC=0.60 and Test MCC=0.80 scores 0.60; Valid MCC=0.80 and Test MCC=0.60 also scores 0.60. This discourages lucky or overfit test runs.
""")

            r_model_in = gr.Textbox(label="Submission Name", value="my_submission")

            r_dataset_in = gr.Dropdown(
                choices=[(label, key) for key, label in DATASET_LABELS.items()],
                value="massbench_benchmark",
                label="Dataset",
            )
            r_dataset_info = gr.Markdown(
                value=get_dataset_info("massbench_benchmark"),
                label="Dataset Information"
            )
            r_board_out = gr.Dataframe(
                label="Real Leaderboard",
                value=get_real_board("massbench_benchmark"),
                wrap=True,
                interactive=False,
            )
            r_dataset_in.change(
                fn=get_dataset_info,
                inputs=[r_dataset_in],
                outputs=[r_dataset_info],
            )
            # Update Real Leaderboard table when dataset changes
            r_dataset_in.change(
                fn=get_real_board,
                inputs=[r_dataset_in],
                outputs=[r_board_out],
            )
            # Interactive Dataset Selector Modal
            with gr.Group(visible=False) as dataset_selector_modal:
                gr.Markdown("""
## Select a Dataset

Browse available datasets and view detailed information.
Datasets are ordered by submission date.
""")
                with gr.Row():
                    ds_selector_dropdown = gr.Dropdown(
                        choices=get_dataset_dropdown_choices(),
                        value="massbench_benchmark",
                        label="Available Datasets"
                    )
                    ds_selector_close = gr.Button("Close", scale=0)
                ds_selector_info = gr.Markdown(value=get_dataset_info("massbench_benchmark"))
                ds_selector_dropdown.change(
                    fn=get_dataset_info,
                    inputs=[ds_selector_dropdown],
                    outputs=[ds_selector_info]
                )
                with gr.Row():
                    ds_selector_cancel = gr.Button("Cancel")
                    ds_selector_confirm = gr.Button("Select Dataset", variant="primary")
            
            r_open_selector = gr.Button("📋 Select Dataset")

            with gr.Row():
                r_train_download = gr.File(
                    label="Training split",
                    value=str(ROOT / "data" / "datasets" / "massbench_benchmark" / "massbench_benchmark_train.csv"),
                    interactive=False,
                )
                r_test_download = gr.File(
                    label="Public test split",
                    value=str(ROOT / "data" / "datasets" / "massbench_benchmark" / "massbench_benchmark_test.csv"),
                    interactive=False,
                )
            r_dataset_in.change(
                fn=get_dataset_download_files,
                inputs=[r_dataset_in],
                outputs=[r_train_download, r_test_download],
            )

            gr.Markdown("#### Custom Package Install (optional)")
            with gr.Row():
                r_pip_pkg = gr.Textbox(
                    label="Install custom pip package(s)",
                    placeholder="e.g. xgboost==2.1.4 or lightgbm catboost",
                    scale=3,
                )
                r_pip_btn = gr.Button("Install Package(s)", scale=1)
            r_pip_out = gr.Textbox(label="Install Output", interactive=False)

            gr.Markdown("#### Batch Correction")
            with gr.Row():
                r_corr_baseline = gr.Dropdown(
                    choices=[(v["name"], k) for k, v in BATCH_CORRECTION_EXAMPLES.items()],
                    value="none",
                    label="Baseline (selection replaces code)",
                    scale=1,
                )
                r_corr_load_btn = gr.Button("Load", scale=1)

            r_correction_code = gr.Code(
                label="Batch Correction Code (define batch_correct)",
                language="python",
                value=DEFAULT_CORRECTION_CODE,
                lines=12,
            )
            r_corr_load_btn.click(
                fn=lambda x: load_baseline(x, True),
                inputs=[r_corr_baseline],
                outputs=[r_correction_code],
            )
            r_corr_baseline.change(
                fn=lambda x: load_baseline(x, True),
                inputs=[r_corr_baseline],
                outputs=[r_correction_code],
            )

            gr.Markdown("#### Model")
            with gr.Row():
                r_model_baseline = gr.Dropdown(
                    choices=[(v["name"], k) for k, v in MODEL_EXAMPLES.items()],
                    value="gaussian_nb",
                    label="Baseline (selection replaces code)",
                    scale=1,
                )
                r_model_load_btn = gr.Button("Load", scale=1)

            r_model_code = gr.Code(
                label="Model Code (define fit or build_model)",
                language="python",
                value=DEFAULT_MODEL_CODE,
                lines=12,
            )
            r_model_load_btn.click(
                fn=lambda x: load_baseline(x, False),
                inputs=[r_model_baseline],
                outputs=[r_model_code],
            )
            r_model_baseline.change(
                fn=lambda x: load_baseline(x, False),
                inputs=[r_model_baseline],
                outputs=[r_model_code],
            )

            with gr.Accordion("Run logs", open=False):
                r_logs_refresh = gr.Button("Refresh logs", variant="secondary")
                r_logs_out = gr.Textbox(
                    label="Printed output",
                    value="Logs from the next real benchmark submission will appear here. Click Refresh logs while a run is active.",
                    lines=18,
                    max_lines=30,
                    interactive=False,
                )
                r_logs_refresh.click(
                    fn=read_latest_run_logs,
                    inputs=[],
                    outputs=[r_logs_out],
                    queue=False,
                )

            with gr.Row():
                r_download_btn = gr.Button("Download Code", scale=1)
                r_submit_btn = gr.Button("Submit Real Benchmark", variant="primary", scale=2)
                r_stop_btn = gr.Button("Stop", variant="stop", scale=1)

            r_status_out = gr.Textbox(label="Status", interactive=False)

            r_download_code = gr.Textbox(
                label="Downloaded Code",
                value="",
                interactive=False,
                visible=False,
            )

            r_board_out.select(
                fn=on_board_click,
                inputs=[
                    r_dataset_in,
                ],
                outputs=[
                    r_correction_code,
                    r_model_code,
                ],
            )

            r_download_btn.click(
                fn=download_code,
                inputs=[r_correction_code, r_model_code],
                outputs=[r_download_code],
            )

            r_submit_btn.click(
                fn=submit_real,
                inputs=[r_model_in, r_dataset_in, r_correction_code, r_model_code, r_pip_pkg],
                outputs=[r_board_out, r_status_out, r_logs_out],
                api_name="submit_real",
            )
            r_stop_btn.click(
                fn=stop_real,
                inputs=[],
                outputs=[r_status_out, r_logs_out],
                queue=False,
            )

            def install_custom_package(
                package_str: str,
                profile: gr.OAuthProfile | None = None,
                request: gr.Request | None = None,
            ) -> str:
                if not _hf_username(profile, request):
                    return "Please sign in with Hugging Face before installing packages."
                if not package_str.strip():
                    return "No package specified."
                pkgs = [p.strip() for p in package_str.replace(",", " ").split() if p.strip()]
                if not pkgs:
                    return "No valid package name(s)."
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install"] + pkgs,
                        capture_output=True,
                        text=True,
                        timeout=180,
                        check=False,
                    )
                    return result.stdout + "\n" + result.stderr
                except Exception as exc:
                    return f"Install failed: {exc}"

            r_pip_btn.click(install_custom_package, inputs=[r_pip_pkg], outputs=[r_pip_out])


            # Dataset Selector Event Handlers
            def open_dataset_selector(current_dataset):
                return {
                    dataset_selector_modal: gr.update(visible=True),
                    ds_selector_dropdown: gr.update(value=current_dataset),
                    ds_selector_info: gr.update(value=get_dataset_info(current_dataset))
                }
            
            def close_dataset_selector():
                return dataset_selector_modal.update(visible=False)
            
            def confirm_dataset_selection(selected_dataset):
                new_dataset, new_board, new_info, train_file, test_file = apply_dataset_selection(selected_dataset)
                return {
                    dataset_selector_modal: gr.update(visible=False),
                    r_dataset_in: gr.update(value=new_dataset),
                    r_board_out: gr.update(value=new_board),
                    r_dataset_info: gr.update(value=new_info),
                    r_train_download: gr.update(value=train_file),
                    r_test_download: gr.update(value=test_file)
                }
            
            r_open_selector.click(
                fn=open_dataset_selector,
                inputs=[r_dataset_in],
                outputs=[dataset_selector_modal, ds_selector_dropdown, ds_selector_info]
            )
            
            ds_selector_close.click(
                fn=close_dataset_selector,
                outputs=[dataset_selector_modal]
            )
            
            ds_selector_cancel.click(
                fn=close_dataset_selector,
                outputs=[dataset_selector_modal]
            )
            
            ds_selector_confirm.click(
                fn=confirm_dataset_selection,
                inputs=[ds_selector_dropdown],
                outputs=[dataset_selector_modal, r_dataset_in, r_board_out, r_dataset_info, r_train_download, r_test_download]
            )

        with gr.TabItem("Add a Dataset"):
            gr.Markdown("""
## Add a benchmark dataset

Upload one comma-separated, sample-by-feature CSV for private curator review.
The first three columns have fixed meanings:

| Column | Required content |
|---|---|
| `name` | A non-empty, unique sample identifier. Do not include personal identifiers. |
| `batch` | The acquisition batch, study, site, instrument run, or other technical domain. At least two distinct values are required. |
| `label` | The biological class to predict. At least two labeled classes are required. Use `-1`, `pool`, `unlabelled`, or `unlabeled` for rows that should contribute only to unsupervised/batch learning. |
| Remaining columns | Numeric features only, one feature per column, with no missing or infinite values. |

Formatting rules:

- One sample per row; the first row is the header.
- UTF-8 comma-separated CSV, maximum **50 MiB**.
- Do not add a saved dataframe index or unnamed first column.
- Sample names must be unique; every row needs a batch and label value.
- Provide provenance and a redistribution license. Restricted or identifying
  data must not be uploaded.
- Passing automatic checks does not publish the dataset. A curator reviews
  provenance, licensing, leakage, class/batch structure, and the final split.

Example header and rows:

```csv
name,batch,label,feature_1,feature_2
sample_001,batch_1,case,12.4,0.82
sample_002,batch_1,control,10.1,0.77
sample_003,batch_2,case,13.0,0.91
sample_004,batch_2,control,9.8,0.73
```
""")
            dataset_template = gr.File(
                label="Download CSV template",
                value=str(ROOT / "data" / "datasets" / "dataset_template.csv"),
                interactive=False,
            )
            with gr.Row():
                dataset_title = gr.Textbox(label="Dataset title")
                dataset_version = gr.Textbox(label="Proposed version", value="1.0.0")
            dataset_description = gr.Textbox(
                label="What the samples and prediction task represent", lines=3
            )
            with gr.Row():
                dataset_modality = gr.Textbox(
                    label="Modality / domain", placeholder="For example: LC-MS metabolomics"
                )
                dataset_task = gr.Textbox(
                    label="Prediction task", placeholder="For example: case versus control"
                )
            dataset_provenance = gr.Textbox(
                label="Source accession, DOI, or stable provenance URL"
            )
            dataset_license = gr.Textbox(
                label="Redistribution license", placeholder="For example: CC-BY-4.0"
            )
            dataset_rights = gr.Checkbox(
                label=(
                    "I confirm redistribution rights and that this upload contains "
                    "no restricted personal or identifying data."
                )
            )
            dataset_file = gr.File(label="Matrix-ready CSV", file_types=[".csv"])
            dataset_submit = gr.Button("Submit for curator review", variant="primary")
            dataset_status = gr.Textbox(label="Submission status", interactive=False)
            dataset_submit.click(
                fn=submit_dataset_proposal,
                inputs=[
                    dataset_title,
                    dataset_version,
                    dataset_description,
                    dataset_modality,
                    dataset_task,
                    dataset_provenance,
                    dataset_license,
                    dataset_rights,
                    dataset_file,
                ],
                outputs=[dataset_status],
                api_name="submit_dataset_proposal",
            )


# _orig_get_api_info = demo.get_api_info

# def _safe_get_api_info(*args, **kwargs):
#     try:
#         return _orig_get_api_info(*args, **kwargs)
#     except TypeError as exc:
#         if "argument of type 'bool' is not iterable" not in str(exc):
#             raise
#         return {"named_endpoints": {}, "unnamed_endpoints": {}}

# demo.get_api_info = _safe_get_api_info


if __name__ == "__main__":
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "allowed_paths": [str(ROOT / "data" / "datasets")],
        "show_error": True,
        "ssr_mode": False,
    }
    port = _launch_port()
    if port is not None:
        launch_kwargs["server_port"] = port
        print(f"Launching Gradio on port {port}")

    demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
