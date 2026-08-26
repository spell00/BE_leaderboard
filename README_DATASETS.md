# Dataset Reproduction

> **Dataset creation note:** do not treat the generated CSVs as canonical source
> files. Recreate them with the commands below so their provenance, sample
> selection, and batch-disjoint train/test split remain auditable.

This workflow reconstructs the two GEO cohorts used by the comparison:

- `normal_tissue_878`: 878 samples, 41 GEO series, labels `blood`, `colon`, and
  `lung`.
- `colon_3041`: 3,041 samples, 60 GEO series, labels `normal` and `cancer`.

The source is Figshare article `12105336` plus the GEO Series referenced by its
supplementary tables 9 and 10. The reconstruction deliberately leaves batch
effects uncorrected.

## Requirements

- R with network access. The preparation script installs `jsonlite`, `readxl`,
  `data.table`, `GEOquery`, and `Biobase` when missing.
- The Python environment from the main [README.md](README.md).
- Enough disk space for the GEO cache and generated expression matrices.

## 1. Reconstruct the combined CSVs

```bash
Rscript scripts/prepare_geo_batch_datasets.R \
  --dataset=both \
  --outdir=data/geo_batch
```

This produces BERNN-ready combined files with the contract
`label,batch,sample_id,<numeric features...>`, alongside expression-only,
metadata, cache, and JSON summary files.

To rebuild just one cohort, use `--dataset=normal` or `--dataset=colon`.

## 2. Audit batch and label structure

```bash
python scripts/verify_geo_batch_effects.py \
  --input-dir data/geo_batch \
  --output-dir results/geo_batch_audit \
  --copy-html \
  --no-wandb
```

The audit discovers every combined CSV in `data/geo_batch` and excludes the
separate `_expression.csv` and `_metadata.csv` exports. Its outputs are easy to
browse from `results/geo_batch_audit/index.html`; raw embeddings and metrics are
stored beside each plot.

## 3. Create the leaderboard train/test layout

```bash
python scripts/import_geo_batch_datasets.py \
  --source-root data/geo_batch \
  --output-root data/datasets \
  --test-fraction 0.2 \
  --seed 42 \
  --split-trials 256 \
  --overwrite
```

The importer searches randomized group splits for a reasonably balanced test
set while keeping GEO batches disjoint between development and fixed test data.
It writes, for each dataset:

- `<dataset>_train.csv`
- `<dataset>_test.csv`
- `<dataset>_inference.csv`
- `<dataset>_predictions.csv`
- `provenance.json`

With seed 42, the expected development/test sample counts are 698/180 for
`normal_tissue_878` and 2,435/602 for `colon_3041`.

## 4. Verify the generated artifacts

```bash
python - <<'PY'
import json
from pathlib import Path

for dataset in ("normal_tissue_878", "colon_3041"):
    path = Path("data/datasets") / dataset / "provenance.json"
    provenance = json.loads(path.read_text())
    print(dataset, json.dumps(provenance, indent=2))
PY
```

Before training, confirm the audit index exists and inspect the PCA/UMAP plots
for both batch separation and biological-label structure.
