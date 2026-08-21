#!/usr/bin/env python3
"""Convert the unintegrated scIB pancreas counts to leaderboard CSV format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("--max-features", type=int, default=2048)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    dataset_id = "scib_pancreas"
    adata = ad.read_h5ad(args.source)
    required = {"tech", "celltype"}
    if not required.issubset(adata.obs):
        raise ValueError(f"Missing scIB metadata columns: {sorted(required - set(adata.obs))}")

    matrix = adata.layers.get("counts", adata.X)
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    totals[totals <= 0] = 1.0
    matrix = matrix.multiply((10_000.0 / totals)[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)

    means = np.asarray(matrix.mean(axis=0)).ravel()
    squared_means = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variances = squared_means - means * means
    keep = np.argsort(variances)[::-1][: min(args.max_features, matrix.shape[1])]
    feature_names = np.asarray(adata.var_names.astype(str))[keep]
    selected = matrix[:, keep].toarray().astype(np.float32, copy=False)

    features = pd.DataFrame(selected, columns=feature_names)
    output = pd.concat(
        [
            pd.DataFrame(
                {
                    "name": adata.obs_names.astype(str),
                    "batch": adata.obs["tech"].astype(str).to_numpy(),
                    "label": adata.obs["celltype"].astype(str).to_numpy(),
                }
            ),
            features,
        ],
        axis=1,
    )
    target_dir = args.repo_root / "data" / "datasets" / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{dataset_id}_train.csv"
    output.to_csv(target, index=False)
    digest = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    provenance = {
        "dataset_id": dataset_id,
        "source": "https://figshare.com/articles/dataset/scIB_pancreas_dataset/25953868",
        "source_file": args.source.name,
        "source_sha256": sha256_file(args.source),
        "samples": int(output.shape[0]),
        "features": len(feature_names),
        "batches": int(output["batch"].nunique()),
        "labels": int(output["label"].nunique()),
        "preprocessing": "library-size normalize to 10000, log1p, top variance genes",
        "selected_feature_names_sha256": digest,
        "role": "sealed_test",
    }
    (target_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {target} shape={output.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
