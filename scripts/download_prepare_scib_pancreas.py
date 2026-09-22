#!/usr/bin/env python3
"""Download and prepare the public scIB pancreas dataset for the leaderboard.

Examples
--------
Prepare locally using the default Figshare article::

    python scripts/download_prepare_scib_pancreas.py

Prepare and upload the resulting dataset files to Hugging Face::

    python scripts/download_prepare_scib_pancreas.py --upload-hf

Use an already-downloaded H5AD instead of downloading::

    python scripts/download_prepare_scib_pancreas.py --source /path/to/pancreas.h5ad
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen


FIGSHARE_ARTICLE_ID = 25953868
FIGSHARE_API = "https://api.figshare.com/v2"
DEFAULT_HF_REPO = "spell0/Batch-effects-leaderboard-data"
DATASET_ID = "scib_pancreas"
CHUNK_SIZE = 8 * 1024 * 1024


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
        "--source",
        type=Path,
        default=None,
        help="Existing H5AD file. If supplied, Figshare download is skipped.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory for downloaded source data (default: data/raw/scib_pancreas).",
    )
    parser.add_argument(
        "--figshare-article-id",
        type=int,
        default=FIGSHARE_ARTICLE_ID,
        help="Figshare article ID to query.",
    )
    parser.add_argument(
        "--figshare-file",
        default=None,
        help="Exact Figshare filename to use when the article contains multiple H5AD files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the H5AD even when the local file already exists.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=2048,
        help="Number of highest-variance genes retained by the existing importer.",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload prepared CSV/provenance files to the Hugging Face Dataset repo.",
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face Dataset repo (default: {DEFAULT_HF_REPO}).",
    )
    return parser.parse_args()


def fetch_json(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": "BE-leaderboard-scib-import/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def figshare_files(article_id: int) -> list[dict]:
    metadata = fetch_json(f"{FIGSHARE_API}/articles/{article_id}")
    files = metadata.get("files", [])
    if not files:
        raise RuntimeError(f"Figshare article {article_id} contains no downloadable files")
    return files


def choose_h5ad(files: list[dict], exact_name: str | None) -> dict:
    if exact_name:
        matches = [row for row in files if row.get("name") == exact_name]
        if not matches:
            available = ", ".join(str(row.get("name")) for row in files)
            raise RuntimeError(
                f"Figshare file {exact_name!r} not found. Available files: {available}"
            )
        return matches[0]

    candidates = [
        row
        for row in files
        if str(row.get("name", "")).lower().endswith(".h5ad")
    ]
    if not candidates:
        available = ", ".join(str(row.get("name")) for row in files)
        raise RuntimeError(f"No .h5ad file found. Available files: {available}")

    def rank(row: dict):
        name = str(row.get("name", "")).lower()
        preferred = (
            int("unintegrated" in name),
            int("raw" in name),
            int("pancreas" in name),
        )
        return preferred + (int(row.get("size") or 0),)

    return max(candidates, key=rank)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def download_figshare_file(file_record: dict, destination: Path, force: bool) -> Path:
    expected_size = int(file_record.get("size") or 0)
    expected_md5 = (
        file_record.get("supplied_md5")
        or file_record.get("computed_md5")
        or file_record.get("md5")
    )

    if destination.exists() and not force:
        size_ok = not expected_size or destination.stat().st_size == expected_size
        md5_ok = not expected_md5 or md5sum(destination) == expected_md5
        if size_ok and md5_ok:
            print(f"[download] Reusing verified source: {destination}", flush=True)
            return destination

    download_url = (
        file_record.get("download_url")
        or f"{FIGSHARE_API}/file/download/{file_record['id']}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[download] {file_record.get('name')} "
        f"({expected_size / 1024**2:.1f} MiB) from {download_url}",
        flush=True,
    )

    request = Request(
        download_url,
        headers={"User-Agent": "BE-leaderboard-scib-import/1.0"},
    )

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".part",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            downloaded = 0
            with urlopen(request, timeout=120) as response:
                while True:
                    block = response.read(CHUNK_SIZE)
                    if not block:
                        break
                    output.write(block)
                    downloaded += len(block)
                    print(
                        f"\r[download] {downloaded / 1024**2:.1f} MiB",
                        end="",
                        flush=True,
                    )
            output.flush()
            os.fsync(output.fileno())
        print(flush=True)

        if expected_size and temporary.stat().st_size != expected_size:
            raise RuntimeError(
                f"Downloaded size mismatch: expected {expected_size}, "
                f"got {temporary.stat().st_size}"
            )
        if expected_md5:
            actual_md5 = md5sum(temporary)
            if actual_md5 != expected_md5:
                raise RuntimeError(
                    f"Downloaded MD5 mismatch: expected {expected_md5}, got {actual_md5}"
                )

        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_existing_importer(
    source: Path,
    repo_root: Path,
    max_features: int,
) -> Path:
    importer = repo_root / "scripts" / "import_scib_pancreas.py"
    if not importer.exists():
        raise FileNotFoundError(f"Existing importer not found: {importer}")

    command = [
        sys.executable,
        str(importer),
        str(source),
        str(repo_root),
        "--max-features",
        str(max_features),
    ]
    print("[prepare] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)

    dataset_dir = repo_root / "data" / "datasets" / DATASET_ID
    train_path = dataset_dir / f"{DATASET_ID}_train.csv"
    if not train_path.exists():
        raise RuntimeError(f"Importer did not create expected file: {train_path}")

    all_path = dataset_dir / f"{DATASET_ID}_all.csv"
    print(f"[prepare] Creating preferred source {all_path.name}", flush=True)
    shutil.copy2(train_path, all_path)
    return dataset_dir


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
    raw_dir = (
        args.raw_dir.expanduser().resolve()
        if args.raw_dir is not None
        else repo_root / "data" / "raw" / DATASET_ID
    )

    if args.source is not None:
        source = args.source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        print(f"[download] Using existing source: {source}", flush=True)
    else:
        files = figshare_files(args.figshare_article_id)
        selected = choose_h5ad(files, args.figshare_file)
        source = raw_dir / str(selected["name"])
        print(
            f"[download] Selected Figshare file: {selected['name']} "
            f"(id={selected.get('id')})",
            flush=True,
        )
        source = download_figshare_file(selected, source, args.force_download)

    dataset_dir = run_existing_importer(source, repo_root, args.max_features)

    print(f"[done] Prepared dataset: {dataset_dir}", flush=True)
    print(f"[done] Preferred source: {dataset_dir / (DATASET_ID + '_all.csv')}", flush=True)

    if args.upload_hf:
        upload_to_hf(dataset_dir, args.hf_repo)
        print(f"[done] Uploaded to https://huggingface.co/datasets/{args.hf_repo}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
