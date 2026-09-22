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
import numpy as np
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
    family_for_config,
    maybe_register_tuned,
)
from src.code_challenge import (
    CodeValidationError,
    cyclic_evaluable_batch_count,
    run_file_inference,
)
from src.database import DatabaseManager, PROJECT_VERSION, real_leaderboard_score
from src.dataset_info import get_dataset_info_markdown
from src.dataset_files import (
    ensure_all_dataset_files,
    inference_filenames,
    research_source_filenames,
)
from src.dataset_submission import DatasetSubmissionError, stage_dataset_proposal
from src.dataset_tasks import (
    clean_task_features,
    prepare_builtin_training_frame,
    task_feature_columns,
)
from src.real_results_store import (
    load_real_result_rows,
    merge_real_result_rows,
    normalize_real_result_row,
    upload_real_result_rows,
)
from src.meta_recommender import (
    recommend_bernn_config,
    recommendation_tables,
    resolve_checkpoint_path,
    recommender_evaluation_protocol,
)
from src.zero_shot_recommender.meta_features import META_FEATURE_NAMES

print(f"Gradio version: {gr.__version__}, Pandas version: {pd.__version__}")
# print(f"Using SQLite version: {DatabaseManager.get_sqlite_version()}")
SEED_REAL_RESULTS = ROOT / "data" / "seed_real_leaderboard.json"
RUN_LOG_DIR = ROOT / "logs" / "ui_runs"
LATEST_RUN_LOG = RUN_LOG_DIR / "latest.log"
UI_LOG_MAX_CHARS = 30_000
UI_LOG_TRIM_AT_CHARS = UI_LOG_MAX_CHARS * 2
LEADERBOARD_UI_LIMIT = 40
_ACTIVE_REAL_RUNS: dict[str, dict] = {}
_ACTIVE_REAL_RUNS_LOCK = threading.Lock()


def _launch_options() -> tuple[int | None, str | None]:
    """Parse app launch overrides while ignoring unrelated Gradio/HF arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--meta-checkpoint", default=None)
    args, _ = parser.parse_known_args()

    port = args.port
    if port is None:
        env_port = os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT")
        if env_port:
            try:
                port = int(env_port)
            except ValueError:
                print(f"Ignoring invalid port value: {env_port!r}")

    checkpoint = args.meta_checkpoint or os.environ.get("BERNN_META_CHECKPOINT")
    if checkpoint:
        checkpoint = str(Path(checkpoint).expanduser().resolve())
        os.environ["BERNN_META_CHECKPOINT"] = checkpoint

    return port, checkpoint


APP_PORT, APP_META_CHECKPOINT = _launch_options()

db = DatabaseManager(ROOT / "data" / "leaderboard.db")

DATASET_LABELS = {
    "normal_tissue_878": "Normal Tissue 878",
    "colon_3041": "Colon 3041",
    "massbench_adenocarcinoma": "MassBench Adenocarcinoma",
    "massbench_alzheimer": "MassBench Alzheimer",
    "massbench_benchmark": "MassBench Benchmark",
}

ALL_DATASET_BUILD_STATUS = ensure_all_dataset_files(ROOT, set(DATASET_LABELS))
for _dataset_key, _status in ALL_DATASET_BUILD_STATUS.items():
    print(f"[dataset-all] {_dataset_key}: {_status}", flush=True)


def dataset_source_choices(dataset: str):
    """Return (_all first, _train second) choices for supervised research."""
    names = research_source_filenames(ROOT, dataset)
    choices = []
    for name in names:
        label = (
            f"Whole dataset — {name}"
            if name.endswith("_all.csv")
            else f"Training split — {name}"
        )
        choices.append((label, name))
    return choices


def default_dataset_source(dataset: str) -> str | None:
    names = research_source_filenames(ROOT, dataset)
    return names[0] if names else None


def inference_file_choices(dataset: str):
    return [(name, name) for name in inference_filenames(ROOT, dataset)]


HF_TOKEN_SET = bool(os.getenv("HF_TOKEN"))
RESEARCH_CYCLIC_ENABLED = (
    os.getenv(
        "ENABLE_RESEARCH_CYCLIC",
        "0" if os.getenv("SPACE_ID") else "1",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)

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
                evaluation_protocol=normalized.get("evaluation_protocol") or "fixed_external",
                cv_folds=_finite_int(normalized.get("cv_folds"), 0),
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
    evaluation_protocol: str = "fixed_external",
    cyclic_cv_folds: int = -1,
    dataset_file: str | None = None,
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
                        "evaluation_protocol": evaluation_protocol,
                        "cyclic_cv_folds": int(cyclic_cv_folds),
                        "dataset_file": dataset_file,
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


def _normalize_cyclic_cv_folds(value, min_groups: int = 3) -> int:
    """Parse rotating batch-CV count for test or validation-only modes."""
    if value is None or str(value).strip() == "":
        return -1
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Number of batch CV folds must be -1 or an integer >= {min_groups}"
        )
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"Number of batch CV folds must be -1 or an integer >= {min_groups}"
        )
    folds = int(numeric)
    if folds != -1 and folds < min_groups:
        raise ValueError(
            f"Number of batch CV folds must be -1 (leave-one-batch-out) "
            f"or at least {min_groups}"
        )
    return folds


def cyclic_cv_status(
    dataset: str,
    evaluation_protocol: str,
    cyclic_cv_folds=-1,
    dataset_file: str | None = None,
) -> str:
    if evaluation_protocol not in {"cyclic_batches", "validation_only"}:
        return ""
    min_groups = 2 if evaluation_protocol == "validation_only" else 3
    try:
        folds = _normalize_cyclic_cv_folds(cyclic_cv_folds, min_groups=min_groups)
        n_batches = cyclic_evaluable_batch_count(dataset, dataset_file)
    except Exception as exc:
        return f"**Batch CV unavailable:** {exc}"

    source_name = dataset_file or default_dataset_source(dataset) or "selected file"
    if folds > n_batches:
        return (
            f"**Not enough evaluable batches.** Requested {folds} folds but "
            f"{source_name} has {n_batches} evaluable batches. Use -1 for LBO."
        )

    if evaluation_protocol == "validation_only":
        resolved = n_batches if folds == -1 else folds
        return (
            f"**Validation-only research mode.** Source: {source_name}. "
            f"{resolved} rotating train/valid rounds; no test or inference matrix "
            "is supplied to the model. This mode is not a leaderboard."
        )

    if folds == -1:
        return (
            f"**Leaderboard eligible — LBO (-1).** Source: {source_name}. "
            f"{n_batches} evaluable batches → {n_batches} rotating rounds; each "
            "batch is validation once and test once."
        )
    if folds == 5:
        return (
            f"**Leaderboard eligible — 5-fold batch CV.** Source: {source_name}. "
            f"{n_batches} evaluable batches are partitioned across 5 rotating groups."
        )
    return (
        f"**Research-only configuration.** {folds}-fold train/valid/test batch CV "
        "will run, but only -1 (LBO) and 5 count for a leaderboard."
    )


def _leaderboard_slice(
    evaluation_protocol: str,
    cyclic_cv_folds=-1,
) -> tuple[str, int] | None:
    if evaluation_protocol == "fixed_external":
        return "fixed_external", 0
    if evaluation_protocol != "cyclic_batches":
        return None
    try:
        folds = _normalize_cyclic_cv_folds(cyclic_cv_folds)
    except ValueError:
        return None
    if folds not in {-1, 5}:
        return None
    return "cyclic_batches", folds


def get_real_board(
    dataset: str | None = None,
    evaluation_protocol: str = "fixed_external",
    cyclic_cv_folds=-1,
) -> pd.DataFrame:
    board_slice = _leaderboard_slice(evaluation_protocol, cyclic_cv_folds)
    if board_slice is None:
        leaderboard = []
    else:
        protocol, folds = board_slice
        if protocol == "cyclic_batches" and dataset:
            try:
                n_batches = cyclic_evaluable_batch_count(dataset)
                if folds == 5 and n_batches < 5:
                    leaderboard = []
                else:
                    leaderboard = db.get_leaderboard(dataset, protocol, folds)
            except Exception:
                leaderboard = []
        else:
            leaderboard = db.get_leaderboard(dataset, protocol, folds)

    if not leaderboard:
        return pd.DataFrame(columns=[
            "username", "dataset", "submission_name", "score",
            "valid_mcc", "test_mcc", "created_at",
        ])

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
        "batch_nbe",
    ]
    optional_cols = [
        "brier_score",
        "ece",
        "batch_silhouette",
        "batch_centroid_dispersion",
        "batch_nmi",
        "batch_nri",
    ]

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

    frame = _json_safe_dataframe(pd.DataFrame(filtered))
    if dataset is not None and len(frame) > LEADERBOARD_UI_LIMIT:
        frame = frame.head(LEADERBOARD_UI_LIMIT)
    return frame



def get_dataset_info(dataset: str) -> str:
    """Get formatted dataset information."""
    return get_dataset_info_markdown(dataset)


# Dataset submission order metadata
DATASET_SUBMISSION_ORDER = {
    "normal_tissue_878": 1,
    "colon_3041": 2,
    "massbench_adenocarcinoma": 3,
    "massbench_benchmark": 4,
    "massbench_alzheimer": 5,
}

def get_dataset_dropdown_choices():
    """Get dataset dropdown choices sorted by submission order."""
    sorted_datasets = sorted(
        DATASET_LABELS.items(),
        key=lambda x: DATASET_SUBMISSION_ORDER.get(x[0], 999)
    )
    return [(label, key) for key, label in sorted_datasets]

def apply_dataset_selection(
    selected_dataset,
    evaluation_protocol="fixed_external",
    cyclic_cv_folds=-1,
):
    """Apply the selected dataset and update the matching leaderboard view."""
    info = get_dataset_info(selected_dataset)
    board = get_real_board(selected_dataset, evaluation_protocol, cyclic_cv_folds)
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
    evaluation_protocol: str = "fixed_external",
    cyclic_cv_folds=-1,
    dataset_file: str | None = None,
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
    evaluation_protocol = str(evaluation_protocol or "fixed_external")
    dataset_file = str(dataset_file or default_dataset_source(dataset) or "")
    try:
        cyclic_cv_folds = _normalize_cyclic_cv_folds(
            cyclic_cv_folds,
            min_groups=2 if evaluation_protocol == "validation_only" else 3,
        )
    except ValueError as exc:
        return get_real_board(dataset, evaluation_protocol, -1), str(exc), ""

    print(
        f"[submission] Received submission from {team.strip() or 'anonymous'} / "
        f"{model_name.strip() or 'unnamed'} on {dataset}; "
        f"evaluation_protocol={evaluation_protocol}; cyclic_cv_folds={cyclic_cv_folds}; "
        f"dataset_file={dataset_file!r}",
        flush=True,
    )
    if not dataset.strip():
        return get_real_board(dataset), "Dataset is required.", ""
    if not team:
        return get_real_board(dataset), "Please sign in with Hugging Face before submitting.", ""
    if not model_name.strip():
        return get_real_board(dataset, evaluation_protocol, cyclic_cv_folds), "Submission name is required.", ""

    cyclic_n_batches = None
    cyclic_leaderboard_eligible = False
    if evaluation_protocol in {"cyclic_batches", "validation_only"}:
        try:
            cyclic_n_batches = cyclic_evaluable_batch_count(dataset, dataset_file)
        except Exception as exc:
            return (
                get_real_board(dataset, evaluation_protocol, cyclic_cv_folds),
                f"Could not determine evaluable batches for rotating CV: {exc}",
                "",
            )
        if cyclic_cv_folds > cyclic_n_batches:
            message = (
                f"Not enough evaluable batches: requested {cyclic_cv_folds} folds, "
                f"but {dataset} has {cyclic_n_batches}. Use -1 for leave-one-batch-out."
            )
            if cyclic_n_batches < 5:
                message += " The 5-fold leaderboard is unavailable for this dataset."
            return get_real_board(dataset, evaluation_protocol, cyclic_cv_folds), message, ""
        cyclic_leaderboard_eligible = (
            evaluation_protocol == "cyclic_batches"
            and cyclic_cv_folds in {-1, 5}
        )

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

    if evaluation_protocol in {"cyclic_batches", "validation_only"} and not RESEARCH_CYCLIC_ENABLED:
        return _finish(
            get_real_board(dataset, evaluation_protocol, cyclic_cv_folds),
            "Cyclic batch rotation is disabled on this deployment. "
            "Enable it for local research with ENABLE_RESEARCH_CYCLIC=1.",
        )

    if evaluation_protocol == "fixed_external" and not HF_TOKEN_SET:
        print(f"[submission] HF_TOKEN is not configured for submission on {dataset}", flush=True)
        return _finish(get_real_board(dataset), "HF_TOKEN is not configured on this Space. The evaluator cannot access private data — contact the organiser.")

    print(
        f"[submission] Running code submission for {team.strip()} / "
        f"{model_name.strip()} on {dataset}; protocol={evaluation_protocol}",
        flush=True,
    )
    try:
        try:
            metrics = _run_code_submission_cancellable(
                run_key=_active_real_run_key(team.strip()),
                team=team.strip(),
                model_name=model_name.strip(),
                dataset=dataset,
                correction_code=correction_code,
                model_code=model_code,
                evaluation_protocol=evaluation_protocol,
                cyclic_cv_folds=cyclic_cv_folds,
            )
        except CodeValidationError as exc:
            return _finish(
                get_real_board(dataset, evaluation_protocol, cyclic_cv_folds),
                f"Submission rejected: {exc}",
            )
        except SubmissionCancelled as exc:
            return _finish(
                get_real_board(dataset, evaluation_protocol, cyclic_cv_folds),
                str(exc),
            )

        print(f"[submission] Code submission completed for {team.strip()} / {model_name.strip()} on {dataset}", flush=True)

        if evaluation_protocol == "cyclic_batches":
            valid_mcc = float(metrics.get("valid_mcc", -1.0))
            test_mcc = float(metrics.get("test_mcc", metrics.get("mcc", -1.0)))
            fold_valid = metrics.get("valid_mcc_folds", [])
            fold_test = metrics.get("test_mcc_folds", [])
            resolved_folds = int(metrics.get("cyclic_cv_folds_resolved", len(fold_valid)))
            mode_label = (
                f"LBO (-1; {resolved_folds} rounds)"
                if cyclic_cv_folds == -1
                else f"{cyclic_cv_folds}-fold batch CV"
            )
            msg = (
                f"Rotating batch result on {DATASET_LABELS.get(dataset, dataset)} "
                f"[{mode_label}]. Mean Valid MCC={valid_mcc:.4f}, "
                f"Mean Test MCC={test_mcc:.4f}, N={metrics.get('n_samples', 0)}."
            )
            if fold_valid:
                msg += "\n\nRotating batch rounds:"
                details = metrics.get("valid_fold_details", [])
                for idx, valid_score in enumerate(fold_valid):
                    test_score = fold_test[idx] if idx < len(fold_test) else float("nan")
                    detail = details[idx] if idx < len(details) else {}
                    msg += (
                        f"\n- R{idx + 1}: train={detail.get('train_batches', [])}, "
                        f"valid={detail.get('valid_batches', [])} MCC={float(valid_score):.4f}, "
                        f"test={detail.get('test_batches', [])} MCC={float(test_score):.4f}"
                    )
                if "test_mcc_std" in metrics or "test_mcc_fold_std" in metrics:
                    msg += (
                        f"\n- Mean test MCC={test_mcc:.4f} "
                        f"± {float(metrics.get('test_mcc_std', metrics.get('test_mcc_fold_std', 0.0))):.4f}"
                    )

            if cyclic_leaderboard_eligible:
                submission = db.create_submission(
                    username=team.strip(),
                    dataset=dataset,
                    submission_name=model_name.strip(),
                    correction_code=correction_code,
                    model_code=model_code,
                    is_public=False,
                )
                db.create_score(
                    submission_id=submission.id,
                    accuracy=_finite_float(metrics.get("accuracy"), 0.0),
                    macro_f1=_finite_float(metrics.get("macro_f1"), 0.0),
                    n_samples=_finite_int(metrics.get("n_samples"), 0),
                    test_mcc=test_mcc,
                    valid_mcc=valid_mcc,
                    valid_mcc_folds=[float(value) for value in fold_valid],
                    evaluation_protocol="cyclic_batches",
                    cv_folds=cyclic_cv_folds,
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
                sync_real_leaderboard_to_hub()
                msg += (
                    "\n\nThis result was saved to the "
                    + ("LBO (-1)" if cyclic_cv_folds == -1 else "5-fold")
                    + " leaderboard."
                )
            else:
                msg += (
                    f"\n\n{cyclic_cv_folds}-fold rotating CV is research-only and "
                    "does not count for a leaderboard. Only -1 (LBO) and 5 are leaderboard modes."
                )
            return _finish(
                get_real_board(dataset, evaluation_protocol, cyclic_cv_folds),
                msg,
            )

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
            evaluation_protocol="fixed_external",
            cv_folds=0,
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
        return _finish(
            get_real_board(dataset, evaluation_protocol, cyclic_cv_folds),
            _format_exec_error(exc),
        )
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
    evaluation_protocol: str = "fixed_external",
    cyclic_cv_folds=-1,
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

        # Load the exact leaderboard slice displayed in the UI.
        board_slice = _leaderboard_slice(evaluation_protocol, cyclic_cv_folds)
        if board_slice is None:
            return gr.update(), gr.update()
        protocol, folds = board_slice
        board = db.get_leaderboard(dataset, protocol, folds)

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



def _normalize_recommender_protocol(protocol: str | None) -> str:
    """Normalize UI/checkpoint protocol names to the two app evaluation modes."""
    if str(protocol or "").strip() in {
        "cyclic_batches",
        "cyclic_train_valid_test_by_batch_v1",
    }:
        return "cyclic_batches"
    return "fixed_external"


def _load_recommender_input(
    dataset: str,
    uploaded_file,
    evaluation_protocol: str,
) -> tuple[pd.DataFrame, str]:
    """Load the exact dataset universe selected for the upcoming evaluation."""
    protocol = _normalize_recommender_protocol(evaluation_protocol)
    if uploaded_file:
        path = getattr(uploaded_file, "name", uploaded_file)
        return pd.read_csv(path), f"uploaded file: {Path(path).name}"

    if protocol == "cyclic_batches":
        from scripts.hp_search import load_cyclic_dataset

        X, y, batches = load_cyclic_dataset(dataset)
        normalized = pd.DataFrame({
            "name": [f"{dataset}_{index}" for index in range(len(X))],
            "batch": pd.Series(batches).astype(str),
            "label": pd.Series(y).astype(str),
        })
        normalized = pd.concat(
            [normalized.reset_index(drop=True), X.reset_index(drop=True)],
            axis=1,
        )
        return (
            normalized,
            f"{DATASET_LABELS.get(dataset, dataset)} (cyclic batch universe)",
        )

    path = ROOT / "data" / "datasets" / dataset / f"{dataset}_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training split not found: {path}")

    frame = prepare_builtin_training_frame(dataset, pd.read_csv(path))
    feature_columns = task_feature_columns(frame)
    cleaned = clean_task_features(frame, feature_columns)

    normalized = frame[["name", "batch", "label"]].reset_index(drop=True).copy()
    for column in feature_columns:
        normalized[column] = cleaned[column]

    return (
        normalized,
        f"{DATASET_LABELS.get(dataset, dataset)} (fixed-external development universe)",
    )


def run_meta_recommendation(
    dataset: str,
    uploaded_file,
    evaluation_protocol: str = "fixed_external",
):
    """Run zero-shot BERNN recommendation on the selected evaluation universe."""
    selected_protocol = _normalize_recommender_protocol(evaluation_protocol)
    print(
        f"[meta-recommender] click received dataset={dataset!r} "
        f"uploaded={bool(uploaded_file)} protocol={selected_protocol!r}",
        flush=True,
    )
    try:
        frame, source_name = _load_recommender_input(
            dataset,
            uploaded_file,
            selected_protocol,
        )
        result = recommend_bernn_config(frame)
        config_table, meta_table = recommendation_tables(result)

        decoded = result["config"]
        full_config = bernn_config(**decoded)
        generated_code = build_bernn_code(full_config)
        family = family_for_config(full_config)
        model_choice = (
            f"bernn_{family}"
            if family and f"bernn_{family}" in MODEL_EXAMPLES
            else "bernn"
        )

        metadata = result.get("checkpoint_metadata", {})
        checkpoint_protocol_raw = metadata.get(
            "evaluation_protocol",
            "fixed_external_test_v1",
        )
        checkpoint_protocol = _normalize_recommender_protocol(
            checkpoint_protocol_raw
        )
        protocol_matches_checkpoint = selected_protocol == checkpoint_protocol
        parity_verified = False

        if not uploaded_file and protocol_matches_checkpoint:
            reference_meta = metadata.get("raw_meta_features", {}).get(dataset)
            if reference_meta is not None:
                current_meta = np.asarray(
                    [result["meta_features"][name] for name in META_FEATURE_NAMES],
                    dtype=np.float32,
                )
                reference_meta = np.asarray(reference_meta, dtype=np.float32)
                if current_meta.shape != reference_meta.shape or not np.allclose(
                    current_meta,
                    reference_meta,
                    rtol=1e-6,
                    atol=1e-8,
                    equal_nan=True,
                ):
                    max_abs = (
                        float(np.nanmax(np.abs(current_meta - reference_meta)))
                        if current_meta.shape == reference_meta.shape
                        else float("inf")
                    )
                    raise RuntimeError(
                        f"Built-in dataset '{dataset}' does not reproduce the "
                        "meta-features stored in this checkpoint for protocol "
                        f"'{selected_protocol}' "
                        f"(max absolute difference={max_abs:.6g})."
                    )
                parity_verified = True

        checkpoint_name = Path(result["checkpoint_path"]).name
        round_number = metadata.get("round")
        benchmark_error = metadata.get("benchmark_prediction_error")

        details = [
            f"Recommendation generated for {source_name}.",
            f"Selected recommendation/evaluation protocol: {selected_protocol}",
            f"Checkpoint: {checkpoint_name}",
            f"Checkpoint training protocol: {checkpoint_protocol_raw}",
            f"Input: {len(frame)} samples, {max(len(frame.columns) - 3, 0)} features",
            f"Meta-features: {len(result.get('meta_features', {}))}",
        ]
        if parity_verified:
            details.append("Training-data meta-feature parity: verified")
        elif not uploaded_file and not protocol_matches_checkpoint:
            details.append(
                "Checkpoint/input protocol differs: stored checkpoint meta-features "
                "are not expected to match and parity was not enforced. The new "
                "recommendation uses the selected evaluation universe."
            )
        if round_number is not None:
            details.append(f"Checkpoint round: {round_number}")
        if benchmark_error is not None:
            details.append(
                f"Benchmark hyperparameter prediction error: {float(benchmark_error):.4f}"
            )

        source_reference = metadata.get("best_source", {}).get(dataset)
        if isinstance(source_reference, dict):
            if source_reference.get("valid_mcc") is not None:
                details.append(
                    f"Original HPO source valid MCC: "
                    f"{float(source_reference['valid_mcc']):.4f}"
                )
            if source_reference.get("test_mcc") is not None:
                details.append(
                    f"Original HPO source fixed-test MCC: "
                    f"{float(source_reference['test_mcc']):.4f}"
                )
            reference_config = source_reference.get("config")
            if isinstance(reference_config, dict):
                shared_keys = sorted(set(decoded) & set(reference_config))
                differing = [
                    key for key in shared_keys
                    if (
                        not np.isclose(decoded[key], reference_config[key], rtol=1e-6, atol=1e-9)
                        if isinstance(decoded[key], (int, float, np.integer, np.floating))
                           and isinstance(reference_config[key], (int, float, np.integer, np.floating))
                        else decoded[key] != reference_config[key]
                    )
                ]
                details.append(
                    f"Predicted vs original HPO config: "
                    f"{len(shared_keys) - len(differing)}/{len(shared_keys)} shared fields match"
                )
                if differing:
                    details.append(
                        "Different fields: " + ", ".join(differing)
                    )

        config_text = "\n".join(
            f"{row.hyperparameter}: {row.value}"
            for row in config_table.itertuples(index=False)
        )
        meta_text = "\n".join(
            f"{row.meta_feature}: {row.value:.6g}"
            for row in meta_table.itertuples(index=False)
        )
        print(
            f"[meta-recommender] completed source={source_name!r} "
            f"samples={len(frame)} features={max(len(frame.columns) - 3, 0)}",
            flush=True,
        )
        return (
            config_text,
            meta_text,
            "\n".join(details),
            generated_code,
            gr.update(interactive=True),
            gr.update(value=model_choice),
            generated_code,
            gr.update(value=dataset) if not uploaded_file else gr.update(),
            gr.update(value=selected_protocol),
        )
    except Exception as exc:
        print(
            f"[meta-recommender] failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return (
            "",
            "",
            _format_exec_error(exc),
            "",
            gr.update(interactive=False),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def recommender_checkpoint_status() -> str:
    path = resolve_checkpoint_path()
    if path.exists():
        return f"Checkpoint ready: {path}"
    return (
        f"Checkpoint not found: {path}\n"
        "Set BERNN_META_CHECKPOINT to an existing .pt file or copy the selected "
        "checkpoint to models/meta_bernn/best_meta_model.pt."
    )


with gr.Blocks(title="MassBench Batch Effects Leaderboard") as demo:
    gr.Markdown(f"""
# MassBench Batch Effects Classification Leaderboard

**Project Version: {PROJECT_VERSION}**

Run a reproducible server-side benchmark or propose a matrix-ready dataset.

{get_baseline_text()}
""")
    gr.LoginButton()

    with gr.Accordion("BERNN Recommender", open=True):
        gr.Markdown(
            "## Zero-shot BERNN hyperparameter recommender\n\n"
            "Use the pretrained meta-network to predict a BERNN configuration directly from "
            "dataset-level statistics. No Optuna search is run here."
        )

        meta_checkpoint_status = gr.Textbox(
            label="Checkpoint",
            value=recommender_checkpoint_status(),
            interactive=False,
        )

        with gr.Row():
            meta_dataset = gr.Dropdown(
                choices=[(label, key) for key, label in DATASET_LABELS.items()],
                value="massbench_benchmark",
                label="Existing dataset",
            )
            meta_upload = gr.File(
                label="Or upload CSV",
                file_types=[".csv"],
            )

        meta_eval_protocol = gr.Radio(
            choices=[
                (
                    "Never-seen external test — recommend from development data only",
                    "fixed_external",
                ),
                (
                    "Rotating batch CV — recommend from the complete cyclic batch universe",
                    "cyclic_batches",
                ),
            ],
            value="fixed_external",
            label="Recommendation / evaluation protocol",
            info=(
                "This controls which samples are used to compute the meta-features. "
                "For adenocarcinoma: fixed uses 434 samples (batches 1+2); cyclic "
                "uses all 642 samples (batches 1+2+3). It is synchronized with "
                "the benchmark protocol below."
            ),
        )

        meta_run = gr.Button(
            "Recommend BERNN configuration",
            variant="primary",
        )

        meta_status = gr.Textbox(
            label="Status",
            lines=6,
            interactive=False,
        )
        meta_config = gr.Textbox(
            label="Recommended hyperparameters",
            lines=16,
            max_lines=24,
            interactive=False,
        )

        with gr.Accordion("Dataset meta-features", open=False):
            meta_features_table = gr.Textbox(
                label="Computed descriptors",
                lines=18,
                max_lines=30,
                interactive=False,
            )

        with gr.Accordion("Predicted BERNN model code", open=True):
            meta_code = gr.Textbox(
                label="Runnable fit(...) code using the predicted hyperparameters",
                lines=28,
                max_lines=40,
                interactive=False,
            )

        meta_apply = gr.Button(
            "Use recommended BERNN configuration",
            variant="secondary",
            interactive=False,
        )




    with gr.Tabs():
        with gr.TabItem("Real Leaderboard (Code Run)"):
            gr.Markdown("""
### Real Benchmark Submission
Submit batch correction and model code. Evaluation runs server-side.
- Click a leaderboard row to auto-fill code if you own it or it is public
- Supported batch correction: ComBat-like, Harmony (harmonypy), scanpy, bernn (TrainAEClassifierHoldout/TrainAEThenClassifierHoldout) methods
- Leaderboards are separated by evaluation protocol. Fixed-external uses the hidden test split; rotating -1 and rotating 5 use mean held-out batch Test MCC. The score is the lower of Valid MCC and Test MCC.
""")

            r_model_in = gr.Textbox(label="Submission Name", value="my_submission")

            r_dataset_in = gr.Dropdown(
                choices=[(label, key) for key, label in DATASET_LABELS.items()],
                value="massbench_benchmark",
                label="Dataset",
            )

            gr.Markdown("#### Test / validation protocol")
            r_eval_protocol = gr.Radio(
                choices=[
                    (
                        "Never-seen external test — keep the designated test batch(es) "
                        "out of train/validation",
                        "fixed_external",
                    ),
                    (
                        "Rotating batch CV — -1 uses leave-one-batch-out",
                        "cyclic_batches",
                    ),
                ],
                value="fixed_external",
                label="Choose protocol before running",
                info=(
                    "Adenocarcinoma example: never-seen mode uses batches 1/2 for "
                    "train-validation and keeps batch 3 as external test (2 folds). "
                    "Rotating -1 runs true leave-one-batch-out. A value of 5 groups "
                    "all evaluable batches into five rotating CV folds."
                ),
            )
            r_cyclic_cv_folds = gr.Number(
                value=-1,
                precision=0,
                label="Number of rotating batch CV folds",
                info=(
                    "-1 = leave-one-batch-out (one round per evaluable batch). "
                    "5 = five grouped batch folds. Other values are research-only."
                ),
                visible=False,
            )
            r_cv_status = gr.Markdown("")
            r_protocol_summary = gr.Markdown(
                "**Selected:** Never-seen external test. "
                "For adenocarcinoma this means 2 train/validation folds and batch 3 is never "
                "used for training or validation."
            )
            r_dataset_info = gr.Markdown(
                value=get_dataset_info("massbench_benchmark"),
                label="Dataset Information"
            )
            def protocol_summary(protocol: str, dataset: str) -> str:
                if protocol == "cyclic_batches":
                    if dataset == "massbench_adenocarcinoma":
                        return (
                            "**Selected: Rotating batch CV.** With the default -1 setting, "
                            "adenocarcinoma has 3 LBO rounds: R1 train=1, valid=2, test=3; "
                            "R2 train=2, valid=3, test=1; R3 train=3, valid=1, test=2."
                        )
                    return (
                        "**Selected: Rotating batch CV.** -1 performs true batch LBO. "
                        "A positive fold count partitions all evaluable batches into that many "
                        "groups so every batch is still validation once and test once."
                    )
                if dataset == "massbench_adenocarcinoma":
                    return (
                        "**Selected: Never-seen external test.** Adenocarcinoma uses 2 "
                        "train/validation folds across batches 1 and 2; batch 3 remains entirely "
                        "outside train/validation and is used only as the external test."
                    )
                return (
                    "**Selected: Never-seen external test.** The designated external test "
                    "batch(es) are never used for training or validation."
                )

            # Keep the recommender and evaluator protocol selectors synchronized.
            # Programmatic updates do not retrain the recommender; when changing
            # protocol, click Recommend again to regenerate protocol-matched code.
            r_eval_protocol.input(
                fn=lambda protocol: protocol,
                inputs=[r_eval_protocol],
                outputs=[meta_eval_protocol],
                queue=False,
            )
            meta_eval_protocol.input(
                fn=lambda protocol: protocol,
                inputs=[meta_eval_protocol],
                outputs=[r_eval_protocol],
                queue=False,
            )

            r_eval_protocol.change(
                fn=protocol_summary,
                inputs=[r_eval_protocol, r_dataset_in],
                outputs=[r_protocol_summary],
                queue=False,
            )
            r_dataset_in.change(
                fn=protocol_summary,
                inputs=[r_eval_protocol, r_dataset_in],
                outputs=[r_protocol_summary],
                queue=False,
            )

            def cv_control_state(protocol: str, dataset: str, folds):
                return (
                    gr.update(visible=(protocol == "cyclic_batches")),
                    cyclic_cv_status(dataset, protocol, folds),
                )

            r_eval_protocol.change(
                fn=cv_control_state,
                inputs=[r_eval_protocol, r_dataset_in, r_cyclic_cv_folds],
                outputs=[r_cyclic_cv_folds, r_cv_status],
                queue=False,
            )
            r_dataset_in.change(
                fn=cv_control_state,
                inputs=[r_eval_protocol, r_dataset_in, r_cyclic_cv_folds],
                outputs=[r_cyclic_cv_folds, r_cv_status],
                queue=False,
            )
            r_cyclic_cv_folds.change(
                fn=lambda dataset, protocol, folds: cyclic_cv_status(dataset, protocol, folds),
                inputs=[r_dataset_in, r_eval_protocol, r_cyclic_cv_folds],
                outputs=[r_cv_status],
                queue=False,
            )

            r_board_out = gr.Dataframe(
                label=f"Real Leaderboard (top {LEADERBOARD_UI_LIMIT} rows)",
                value=get_real_board("massbench_benchmark", "fixed_external", -1),
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
                inputs=[r_dataset_in, r_eval_protocol, r_cyclic_cv_folds],
                outputs=[r_board_out],
            )
            r_eval_protocol.change(
                fn=get_real_board,
                inputs=[r_dataset_in, r_eval_protocol, r_cyclic_cv_folds],
                outputs=[r_board_out],
                queue=False,
            )
            r_cyclic_cv_folds.change(
                fn=get_real_board,
                inputs=[r_dataset_in, r_eval_protocol, r_cyclic_cv_folds],
                outputs=[r_board_out],
                queue=False,
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
            r_model_baseline.input(
                fn=lambda x: load_baseline(x, False),
                inputs=[r_model_baseline],
                outputs=[r_model_code],
            )

            meta_run.click(
                fn=run_meta_recommendation,
                inputs=[meta_dataset, meta_upload, meta_eval_protocol],
                outputs=[
                    meta_config,
                    meta_features_table,
                    meta_status,
                    meta_code,
                    meta_apply,
                    r_model_baseline,
                    r_model_code,
                    r_dataset_in,
                    r_eval_protocol,
                ],
                api_name="recommend_bernn",
                queue=False,
                show_progress="full",
            )

            meta_apply.click(
                fn=lambda code: code,
                inputs=[meta_code],
                outputs=[r_model_code],
                queue=False,
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
                    r_eval_protocol,
                    r_cyclic_cv_folds,
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
                inputs=[
                    r_model_in,
                    r_dataset_in,
                    r_correction_code,
                    r_model_code,
                    r_pip_pkg,
                    r_eval_protocol,
                    r_cyclic_cv_folds,
                ],
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
            
            def confirm_dataset_selection(selected_dataset, evaluation_protocol, cyclic_cv_folds):
                new_dataset, new_board, new_info, train_file, test_file = apply_dataset_selection(
                    selected_dataset,
                    evaluation_protocol,
                    cyclic_cv_folds,
                )
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
                inputs=[ds_selector_dropdown, r_eval_protocol, r_cyclic_cv_folds],
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
    if APP_PORT is not None:
        launch_kwargs["server_port"] = APP_PORT
        print(f"Launching Gradio on port {APP_PORT}")
    if APP_META_CHECKPOINT:
        print(f"Using BERNN meta-network checkpoint: {APP_META_CHECKPOINT}")

    demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
