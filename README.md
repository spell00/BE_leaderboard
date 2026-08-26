# BE Leaderboard Meta-HPO Reproduction

This repository reconstructs the two GEO batch-effect datasets, audits their
batch structure, and runs the joint Optuna/BERNN meta-learning comparison.

Dataset creation is intentionally documented separately in
[README_DATASETS.md](README_DATASETS.md). The generated expression matrices are
large derived artifacts and should be rebuilt from their published sources.

## Start-to-finish commands

Run all commands from the repository root.

### 1. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

BERNN is pinned to version 1.0.4, which exposes `num_workers` through its public
training API.

### 2. Reconstruct and import the datasets

The R script installs its missing CRAN/Bioconductor packages, downloads the
published Figshare/GEO inputs, and creates the combined CSVs.

```bash
Rscript scripts/prepare_geo_batch_datasets.R \
  --dataset=both \
  --outdir=data/geo_batch

python scripts/import_geo_batch_datasets.py \
  --source-root data/geo_batch \
  --output-root data/datasets \
  --seed 42 \
  --overwrite
```

See [README_DATASETS.md](README_DATASETS.md) for expected sizes, provenance,
split guarantees, and verification commands.

### 3. Generate the PCA/UMAP batch audit

```bash
python scripts/verify_geo_batch_effects.py \
  --input-dir data/geo_batch \
  --output-dir results/geo_batch_audit \
  --copy-html \
  --no-wandb
```

Open `results/geo_batch_audit/index.html`. Each dataset directory contains PCA
and UMAP embeddings and interactive plots colored by batch and biological
label. Summary metrics are in `dataset_overview.csv`, `.json`, and `.md`.

To also log the audit to W&B, omit `--no-wandb` and optionally provide
`--wandb-project` and `--wandb-run-name`.

### 4. Run the joint Optuna comparison

```bash
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="optuna-tpe-bernn104-cv3-nw4-transductive-${RUN_STAMP}"
RUN_DIR="results/optuna_tpe_bernn104_cv3_nw4_transductive_${RUN_STAMP}"
mkdir -p "$RUN_DIR"

PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 python -u \
  scripts/run_optuna_comparison.py \
  --n-trials 1000 \
  --n-epochs 1000 \
  --n-repeats -1 \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --wandb-project BE_leaderboard_meta_evolution \
  --wandb-run-name "$RUN_NAME" \
  --output-dir "$RUN_DIR" \
  --log1p-mode on \
  --meta-hidden-size 64 \
  --meta-epochs 1000 \
  2>&1 | tee "$RUN_DIR/launch.log"
```

The two reconstructed GEO datasets, `normal_tissue_878` and `colon_3041`, use
three grouped CV folds. Hyperparameters are sampled once per joint solution and
then evaluated across every participating dataset and fold. Fixed test sets are
used only for monitoring, never for model selection.

Four DataLoader workers are the measured default for this workload: they were
faster than two, while eight gave no further improvement on the reference VM.

### 5. Resume an interrupted run

Reuse the exact same `RUN_NAME` and `RUN_DIR`, then repeat the command above with
`--resume`. The output directory contains the persisted W&B identity and Optuna
database required for recovery.

```bash
python -u scripts/run_optuna_comparison.py \
  --resume \
  --n-trials 1000 --n-epochs 1000 --n-repeats -1 \
  --num-workers 4 --device cuda --seed 42 \
  --wandb-project BE_leaderboard_meta_evolution \
  --wandb-run-name "$RUN_NAME" --output-dir "$RUN_DIR" \
  --log1p-mode on --meta-hidden-size 64 --meta-epochs 1000
```

## Results

The comparison writes its durable state under `RUN_DIR`, including:

- `optuna.sqlite3` and `run_metadata.json`
- `solutions.jsonl` and `solutions.csv` as solutions complete
- `validation_solutions.jsonl`
- `latest_joint_meta_model.pt`
- `launch.log`

W&B logs scores, figures, sampled configurations, and best configurations. The
domain-loss choice is categorical and appears under keys such as
`solutions/config/<dataset>/dloss` after a complete joint solution.
