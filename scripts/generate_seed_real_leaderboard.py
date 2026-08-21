from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import BATCH_CORRECTION_EXAMPLES, MODEL_EXAMPLES
from src.code_challenge import CodeValidationError, run_code_submission
from src.database import real_leaderboard_score


FAST_MODEL_KEYS = [
    key for key in MODEL_EXAMPLES
    if not key.startswith("bernn")
]


def _finite(value, default=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="massbench_benchmark")
    parser.add_argument("--username", default="spell0")
    parser.add_argument("--output", default="data/seed_real_leaderboard.json")
    parser.add_argument("--include-bernn", action="store_true")
    args = parser.parse_args()

    model_keys = list(MODEL_EXAMPLES) if args.include_bernn else FAST_MODEL_KEYS
    rows = []
    errors = []
    total = len(BATCH_CORRECTION_EXAMPLES) * len(model_keys)
    done = 0

    for corr_key, corr in BATCH_CORRECTION_EXAMPLES.items():
        for model_key in model_keys:
            model = MODEL_EXAMPLES[model_key]
            done += 1
            submission_name = f"{corr['name']} x {model['name']}"
            print(f"[{done}/{total}] {submission_name}", flush=True)
            try:
                _, metrics, _, _ = run_code_submission(
                    team=args.username,
                    model_name=submission_name,
                    dataset=args.dataset,
                    correction_code=corr["code"],
                    model_code=model["code"],
                )
            except (CodeValidationError, Exception) as exc:
                errors.append({
                    "submission_name": submission_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  skipped: {type(exc).__name__}: {exc}", flush=True)
                continue

            valid_mcc = _finite(metrics.get("valid_mcc"), 0.0)
            test_mcc = _finite(metrics.get("test_mcc", metrics.get("mcc")), 0.0)
            rows.append({
                "username": args.username,
                "dataset": args.dataset,
                "submission_name": submission_name,
                "correction_key": corr_key,
                "model_key": model_key,
                "correction_code": corr["code"],
                "model_code": model["code"],
                "score": real_leaderboard_score(valid_mcc, test_mcc),
                "test_mcc": test_mcc,
                "valid_mcc": valid_mcc,
                "valid_mcc_folds": [
                    _finite(value) for value in metrics.get("valid_mcc_folds", [])
                    if _finite(value) is not None
                ],
                "train_mcc": _finite(metrics.get("train_mcc"), -1.0),
                "accuracy": _finite(metrics.get("accuracy"), 0.0),
                "macro_f1": _finite(metrics.get("macro_f1"), 0.0),
                "n_samples": int(metrics.get("n_samples", 0) or 0),
                "log_loss": _finite(metrics.get("log_loss")) if "log_loss" in metrics else None,
                "brier_score": _finite(metrics.get("brier_score")) if "brier_score" in metrics else None,
                "ece": _finite(metrics.get("ece")) if "ece" in metrics else None,
                "batch_silhouette": _finite(metrics.get("batch_silhouette")) if "batch_silhouette" in metrics else None,
                "batch_centroid_dispersion": _finite(metrics.get("batch_centroid_dispersion")) if "batch_centroid_dispersion" in metrics else None,
                "batch_nbe": _finite(metrics.get("batch_nbe")) if "batch_nbe" in metrics else None,
                "batch_nmi": _finite(metrics.get("batch_nmi")) if "batch_nmi" in metrics else None,
                "batch_nri": _finite(metrics.get("batch_nri")) if "batch_nri" in metrics else None,
            })

    rows.sort(key=lambda row: (row["score"], row["accuracy"]), reverse=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "username": args.username,
        "include_bernn": bool(args.include_bernn),
        "rows": rows,
        "errors": errors,
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {out_path}")
    if errors:
        print(f"Skipped {len(errors)} combinations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
