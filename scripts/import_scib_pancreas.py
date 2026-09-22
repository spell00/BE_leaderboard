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
    parser.add_argument("--max-features", type=int, default=0, help="Maximum number of variance-ranked genes to retain; 0 keeps all genes.")
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

    if args.max_features and args.max_features > 0 and args.max_features < matrix.shape[1]:
        means = np.asarray(matrix.mean(axis=0)).ravel()
        squared_means = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        variances = squared_means - means * means
        keep = np.argsort(variances)[::-1][: args.max_features]
        matrix = matrix[:, keep].tocsr()
        feature_names = np.asarray(adata.var_names.astype(str))[keep]
        feature_selection = f"top {len(feature_names)} variance genes"
    else:
        feature_names = np.asarray(adata.var_names.astype(str))
        feature_selection = "all genes"

    metadata = pd.DataFrame(
        {
            "name": adata.obs_names.astype(str),
            "batch": adata.obs["tech"].astype(str).to_numpy(),
            "label": adata.obs["celltype"].astype(str).to_numpy(),
        }
    )

    target_dir = args.repo_root / "data" / "datasets" / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{dataset_id}_train.csv"

    # Keep the normalized matrix sparse and only densify small row chunks.
    # This allows the default all-gene export without materializing the entire
    # cells-by-genes matrix as one large dense array.
    chunk_rows = 256
    first = True
    for start in range(0, matrix.shape[0], chunk_rows):
        stop = min(start + chunk_rows, matrix.shape[0])
        dense = matrix[start:stop].toarray().astype(np.float32, copy=False)
        features = pd.DataFrame(dense, columns=feature_names)
        output_chunk = pd.concat(
            [
                metadata.iloc[start:stop].reset_index(drop=True),
                features,
            ],
            axis=1,
        )
        output_chunk.to_csv(
            target,
            index=False,
            mode="w" if first else "a",
            header=first,
        )
        first = False
        print(
            f"wrote rows {start}:{stop} / {matrix.shape[0]} "
            f"with {len(feature_names)} features",
            flush=True,
        )

    digest = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    provenance = {
        "dataset_id": dataset_id,
        "source": "https://figshare.com/articles/dataset/scIB_pancreas_dataset/25953868",
        "source_file": args.source.name,
        "source_sha256": sha256_file(args.source),
        "samples": int(matrix.shape[0]),
        "features": len(feature_names),
        "batches": int(metadata["batch"].nunique()),
        "labels": int(metadata["label"].nunique()),
        "preprocessing": f"library-size normalize to 10000, log1p, {feature_selection}",
        "selected_feature_names_sha256": digest,
        "role": "sealed_test",
    }
    (target_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {target} shape=({matrix.shape[0]}, {len(feature_names) + 3})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
