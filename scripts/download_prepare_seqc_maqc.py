#!/usr/bin/env python3
"""Prepare the compact SEQC/MAQC-III benchmark and optionally upload it to HF.

The R helper installs and loads the Bioconductor seqc experiment package,
aggregates Illumina lane/flowcell counts to 108 prepared A-D replicate libraries
across six sites, retains every non-ERCC RefSeq gene, and writes log1p(CPM).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

DATASET_ID = "seqc_maqc"
DEFAULT_HF_REPO = "spell0/Batch-effects-leaderboard-data"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Leaderboard repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prepared seqc_maqc dataset.",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload prepared CSV/provenance files to the HF Dataset repo.",
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face Dataset repo (default: {DEFAULT_HF_REPO}).",
    )
    return parser.parse_args()


def upload_to_hf(dataset_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for path in sorted(dataset_dir.iterdir()):
        if path.suffix.lower() not in {".csv", ".json"}:
            continue
        path_in_repo = f"data/datasets/{DATASET_ID}/{path.name}"
        print(f"[hf] Uploading {path.name} -> {repo_id}/{path_in_repo}", flush=True)
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            commit_message=f"Add {DATASET_ID} {path.name}",
        )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError(
            "Rscript was not found. Install R first; the preparation helper "
            "installs the required R/Bioconductor packages automatically."
        )

    helper = repo_root / "scripts" / "prepare_seqc_maqc.R"
    if not helper.exists():
        raise FileNotFoundError(helper)

    command = [rscript, str(helper), "--repo-root", str(repo_root)]
    if args.overwrite:
        command.append("--overwrite")

    print("[prepare] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)

    dataset_dir = repo_root / "data" / "datasets" / DATASET_ID
    all_path = dataset_dir / f"{DATASET_ID}_all.csv"
    if not all_path.exists():
        raise RuntimeError(f"Expected output was not created: {all_path}")

    print(f"[done] Preferred source: {all_path}", flush=True)

    if args.upload_hf:
        upload_to_hf(dataset_dir, args.hf_repo)
        print(
            f"[done] Uploaded to https://huggingface.co/datasets/{args.hf_repo}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
