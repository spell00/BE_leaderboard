# BERNN meta-HPO v2: frozen Optuna bank + replay experiments

This bundle implements the revised protocol:

1. **Build the expensive Optuna bank once.**
   - 4 source datasets: `normal_tissue_878`, `colon_3041`, `massbench_adenocarcinoma`, `massbench_benchmark`.
   - `massbench_alzheimer`: an independent target-specific Optuna baseline only.
2. **Never recompute that bank when restarting a meta-model experiment.**
3. **Train/retrain meta-models from scratch from prefixes of the frozen source bank.**
4. **Predict Alzheimer every cheap meta/RL epoch, but run real Alzheimer BERNN only at selected checkpoints.**
5. **Alzheimer Optuna trials and real Alzheimer validation MCC never enter zero-shot/meta/surrogate training.**
6. **Categorical-only pruning is source-only.** Continuous hparams are never frozen from consensus.
7. A separate `run_alzheimer_rl_control.py` is provided for RL that really uses Alzheimer MCC as reward. That arm is a target-specific control, not held-out validation.

## Files to copy into `BE_leaderboard`

```text
src/meta_hpo_bank.py
src/meta_hpo_models.py
scripts/build_meta_hpo_trial_bank.py
scripts/analyze_meta_hpo_bank.py
scripts/run_meta_hpo_replay.py
scripts/run_alzheimer_rl_control.py
scripts/summarize_meta_hpo_results.py
```

The scripts reuse the existing project contracts from:

- `scripts/hp_search.py`
- `src/dataset_splits.py`
- `src/zero_shot_recommender/meta_features.py`

The bundle preserves the CV override in the current pasted comparison script:

```text
normal_tissue_878          3 folds
colon_3041                 3 folds
massbench_adenocarcinoma   2 folds
massbench_benchmark        3 folds
massbench_alzheimer        3 folds
```

If you want adenocarcinoma back at 3 folds, change `DATASET_CV_FOLDS` in the two scripts before starting Stage 0. Do not change it after the bank is partly built.

---

# 1. Build the trial bank once

Recommended first real run:

```bash
python scripts/build_meta_hpo_trial_bank.py \
  --n-trials 20 \
  --n-epochs 1000 \
  --device cuda \
  --seed 42 \
  --output-dir results/meta_hpo_bank_20 \
  --wandb-run-name meta-hpo-bank-20
```

This runs the five independent Optuna studies in round-robin order. After bank step 20:

- 80 source BERNN configurations exist: `4 × 20`;
- 20 Alzheimer Optuna configurations exist as a **baseline/oracle HPO curve**;
- no meta-model has been trained yet.

Resume after interruption:

```bash
python scripts/build_meta_hpo_trial_bank.py \
  --n-trials 20 \
  --n-epochs 1000 \
  --device cuda \
  --seed 42 \
  --output-dir results/meta_hpo_bank_20 \
  --resume
```

Primary durable outputs:

```text
results/meta_hpo_bank_20/optuna.sqlite3
results/meta_hpo_bank_20/trial_bank.jsonl
results/meta_hpo_bank_20/trial_bank.csv
results/meta_hpo_bank_20/bank_summary.json
```

`trial_bank.jsonl` is the artifact reused by the later scenarios.

---

# 2. Analyze source categorical consensus

```bash
python scripts/analyze_meta_hpo_bank.py \
  --trial-bank results/meta_hpo_bank_20 \
  --top-k 5 \
  --min-support 0.80
```

Outputs:

```text
results/meta_hpo_bank_20/categorical_consensus.json
results/meta_hpo_bank_20/alzheimer_optuna_baseline_curve.csv
```

The eligible categorical/discrete fields are:

```text
dloss
variational
kan
class_triplet
scaler
n_layers
```

`strict_fixed` implements the literal requested rule:

> freeze a categorical field if the best independently optimized configuration for all four source datasets has exactly the same value.

`robust_fixed` additionally requires the same category to dominate each source dataset's top-K trials.

Continuous variables such as `lr`, `wd`, `dropout`, `nu`, `warmup`, and `layer1` are **never** frozen by cross-dataset consensus.

---

# 3. Replay the cheap meta-learning scenarios

## First run the unpruned experiment

This gives the honest curve as a function of the number of source Optuna trials observed.

```bash
python scripts/run_meta_hpo_replay.py \
  --trial-bank results/meta_hpo_bank_20 \
  --source-prefixes 1,2,3,5,10,15,20 \
  --freeze-policy none \
  --meta-epochs 1000 \
  --direct-eval-epochs 10,50,200,500,1000 \
  --surrogate-search-trials 3000 \
  --rl-epochs 1000 \
  --rl-eval-epochs 200,1000 \
  --evolution-generations 100 \
  --evolution-eval-generations 25,50,100 \
  --n-epochs 1000 \
  --device cuda \
  --output-dir results/meta_hpo_replay_unpruned
```

Default scenarios:

### `direct`

For source prefix `N`:

```text
metadata(source 1) -> best hparams found in first N Optuna trials
metadata(source 2) -> best hparams found in first N Optuna trials
metadata(source 3) -> best hparams found in first N Optuna trials
metadata(source 4) -> best hparams found in first N Optuna trials
```

A fresh mixed-output network is initialized and trained from scratch.

- categorical heads: cross-entropy;
- boolean heads: BCE;
- continuous heads: masked Smooth-L1;
- Alzheimer hparams are predicted every meta epoch;
- real Alzheimer BERNN is run only at `--direct-eval-epochs` plus the final epoch.

This directly tests the idea that the meta-network should fully fit the source hparam targets before being judged on Alzheimer.

### `surrogate_extra_trees`

Uses **all `4 × N` source trials**:

```text
(dataset meta-features, hparams) -> observed valid MCC
```

An ExtraTrees surrogate estimates both mean MCC and tree disagreement. TPE then performs thousands of cheap target-space searches against:

```text
predicted_MCC - risk_penalty * uncertainty
```

Only the final surrogate-selected Alzheimer config is run with real BERNN for each source prefix.

### `surrogate_mlp`

Same training examples as above but uses a bootstrapped neural ensemble as the transferable score surrogate.

### `surrogate_evolution`

Runs evolutionary search **against the source-trained score surrogate**, not against four new BERNN fits. This is the cheap counterpart to the existing real-BERNN evolutionary policy.

Real Alzheimer BERNN is checked only at selected surrogate-evolution generations.

### `meta_rl`

Contextual REINFORCE policy:

```text
source dataset metadata -> stochastic hparams
                         -> source-trained surrogate reward
```

The policy is trained across the four source contexts, then queried on Alzheimer metadata. Real Alzheimer MCC never becomes RL reward.

### `target_rl`

The policy is optimized at Alzheimer **metadata**, but the reward is still only the source-trained surrogate prediction. This is deliberately a stress test for surrogate exploitation.

The output logs both:

```text
surrogate-predicted Alzheimer MCC
real Alzheimer MCC
prediction error
```

If predicted MCC rises while real MCC stays flat or falls, the surrogate is being exploited rather than transferring correctly.

---

# 4. Categorical-pruned replay

After Stage 0, you can run a second replay using the final source consensus:

```bash
python scripts/run_meta_hpo_replay.py \
  --trial-bank results/meta_hpo_bank_20 \
  --categorical-consensus results/meta_hpo_bank_20/categorical_consensus.json \
  --freeze-policy strict \
  --consensus-scope full \
  --source-prefixes 1,2,3,5,10,15,20 \
  --n-epochs 1000 \
  --device cuda \
  --output-dir results/meta_hpo_replay_strict_pruned
```

Important interpretation:

`--consensus-scope full` means the categorical freeze consumed all 20 Stage-0 trials per source before the replay started. Therefore the pruned `N=1`, `N=2`, ... curves **must not be reported as if the algorithm only had N source trials total**. The run metadata records this explicitly.

For a prefix-pure learning curve instead:

```bash
--freeze-policy strict --consensus-scope prefix
```

Then categorical consensus at prefix N only uses source trials `1..N`.

---

# 5. Real Alzheimer RL control

This is intentionally separate because Alzheimer MCC is now the reward:

```bash
python scripts/run_alzheimer_rl_control.py \
  --trial-bank results/meta_hpo_bank_20 \
  --n-trials 20 \
  --n-epochs 1000 \
  --device cuda \
  --output-dir results/alzheimer_real_rl_control
```

Optional source-derived categorical freeze:

```bash
python scripts/run_alzheimer_rl_control.py \
  --trial-bank results/meta_hpo_bank_20 \
  --categorical-consensus results/meta_hpo_bank_20/categorical_consensus.json \
  --freeze-policy strict \
  --n-trials 20 \
  --n-epochs 1000 \
  --device cuda \
  --output-dir results/alzheimer_real_rl_strict_pruned
```

Compare this arm against `alzheimer_optuna_baseline_curve.csv` as:

```text
best Alzheimer MCC vs number of real Alzheimer BERNN evaluations
```

Do **not** treat this RL arm as held-out validation because Alzheimer MCC directly updates the policy.

---

# Real-compute strategy

## What should run first?

Run only this first:

```text
Stage 0 independent Optuna bank
```

Do not start the week-long real-source evolution in parallel just because multiple scenarios exist. The frozen bank is more valuable first because it unlocks all of the cheap replay scenarios.

## Can replay scenarios run at the same time?

The cheap fitting/search parts can. The expensive part is still the selected real Alzheimer BERNN checks.

With one GPU, I recommend either:

1. run one replay process containing all cheap scenarios so shared surrogates are reused and Alzheimer evaluations are serialized; or
2. run separate replay scenarios sequentially.

With multiple GPUs, one scenario/output directory per GPU is reasonable.

## Approximate validation counts with the defaults

For 7 source prefixes:

```text
direct:               5 selected meta epochs × 7 = 35 Alzheimer configs
surrogate_extra_trees:                         7
surrogate_mlp:                                 7
surrogate_evolution:  3 generations × 7 =     21
meta_rl:              2 RL epochs × 7 =        14
target_rl:            2 RL epochs × 7 =        14
```

That is still substantial. You do **not** need to run all of them immediately.

A sensible first replay is:

```bash
--scenarios direct,surrogate_extra_trees,target_rl \
--source-prefixes 5,10,20 \
--direct-eval-epochs 50,200,1000 \
--rl-eval-epochs 200,1000
```

Then expand only the scenarios that look promising.

---

# Output files

`run_meta_hpo_replay.py` writes:

```text
run_metadata.json
cheap_predictions.jsonl
scenario_results.jsonl
alzheimer_validation_cache.jsonl
```

`cheap_predictions.jsonl` contains every cheap meta/RL/evolution prediction.

`scenario_results.jsonl` contains only checkpoints for which a real Alzheimer validation was requested, including:

```text
scenario
source_prefix
checkpoint kind / number
predicted MCC (surrogate arms)
predicted uncertainty
actual Alzheimer valid MCC
actual Alzheimer test MCC
surrogate signed error
recommended hparams
```

`alzheimer_validation_cache.jsonl` is keyed by exact BERNN config + evaluation protocol. If two scenarios genuinely propose the same resolved config under the same protocol, BERNN does not need to be rerun.

---

# Recommended first sequence

```bash
# 1. expensive bank once
python scripts/build_meta_hpo_trial_bank.py \
  --n-trials 20 --n-epochs 1000 --device cuda \
  --output-dir results/meta_hpo_bank_20

# 2. inspect source-only categorical agreement
python scripts/analyze_meta_hpo_bank.py \
  --trial-bank results/meta_hpo_bank_20

# 3. small unpruned replay first
python scripts/run_meta_hpo_replay.py \
  --trial-bank results/meta_hpo_bank_20 \
  --scenarios direct,surrogate_extra_trees,target_rl \
  --source-prefixes 5,10,20 \
  --direct-eval-epochs 50,200,1000 \
  --rl-eval-epochs 200,1000 \
  --freeze-policy none \
  --n-epochs 1000 --device cuda \
  --output-dir results/meta_hpo_replay_first

# 4. only then expand the experiment matrix
```

This keeps the first expensive investment reusable and prevents each new meta-learning idea from costing another week of four-source BERNN optimization.


# 6. Optional compact CSV summary

```bash
python scripts/summarize_meta_hpo_results.py \
  --trial-bank results/meta_hpo_bank_20 \
  --replay-dir results/meta_hpo_replay_first
```

This writes `summary_alzheimer_optuna_baseline.csv` and `summary_scenario_results.csv`.

# 7. Synchronized meta-network checkpoints and recovery

`scripts/run_synchronized_meta_hpo.py` now saves the selected meta network before
each round's Alzheimer BERNN evaluation. Under its `--output-dir`:

- `meta_checkpoints/round_0028.pt` is the selected network for zero-based round 28
  (round 29 in the console).
- `meta_checkpoints/best_benchmark.pt` has the lowest benchmark hyperparameter
  prediction error among saved rounds.
- `meta_checkpoints/best_alzheimer.pt` has the highest measured Alzheimer
  validation MCC among saved rounds.
- `rounds.jsonl` links each new result to its `meta_checkpoint`.

Every checkpoint includes CPU model weights, architecture, feature names/order,
inference normalization, hyperparameter decoding schema, source configurations,
raw dataset meta-features, seed, and training settings. Scores are added after
evaluation. Writes are atomic; with W&B enabled, checkpoints are also queued for
upload to the run's Files tab. Local saving works with `--no-wandb`.
An already running Python process must be restarted with the existing resume
arguments to use the updated saving code.

Earlier runs saved trial records but no meta-network weights. Recover all recorded
rounds of this run without repeating BERNN HPO or Alzheimer training:

```bash
python scripts/recover_synchronized_meta_model.py \
  --output-dir results/synchronized_meta_hpo_vm2_v1 --all
```

Use `--round 28` for a specific zero-based round, or omit both selectors to recover
the highest Alzheimer validation MCC round. `--seed` (default 42) and `--n-epochs`
(default 1000) must match the original run. Recovery uses each round's recorded
source winners, selected hidden size and learning rate, and 200 meta-training
epochs. It preserves the historical runner's training/inference transforms.
It verifies the decoded Alzheimer configuration and benchmark error against the
ledger (`rtol=1e-5`, `atol=1e-8`), then verifies save/reload prediction equality.
Mismatches are reported without saving that round.

Recovered checkpoints are explicitly marked `reconstructed` and stored separately
under `recovered/meta_checkpoints/`, with the same filenames and best-by-metric
aliases. Matching predictions do not establish that every weight is identical to
the unavailable original checkpoint. The Alzheimer MCC remains the historical
measurement; recovery does not repeat its BERNN evaluation.

Load a checkpoint to recommend BERNN hyperparameters for a new dataset:

```python
from src.direct_meta_checkpoint import predict_from_checkpoint

config = predict_from_checkpoint(
    "results/synchronized_meta_hpo_vm2_v1/recovered/meta_checkpoints/round_0028.pt",
    X_train, y_train, batches_train,
)
```

This loads the meta network and returns a BERNN configuration. Training BERNN on
that configuration is a separate step.


---

# Independent Optuna refresh across every current dataset

Use the dedicated two-GPU launcher when each dataset must have its own Optuna
study rather than sharing one meta/evolutionary search.

```bash
python scripts/run_independent_optuna_all_datasets.py \
  --n-trials 20 \
  --n-epochs 1000 \
  --gpus 0,1 \
  --prepare-missing \
  --output-dir results/independent_optuna_all_20 \
  --wandb-group independent-optuna-all-datasets-20
```

The queue is intentionally new-first:

```text
jdlber_sle_maldi
seqc_maqc
scib_pancreas
normal_tissue_878
colon_3041
massbench_adenocarcinoma
massbench_benchmark
massbench_alzheimer
```

Every dataset owns a separate Optuna SQLite database and a separate W&B run
inside the common group. Two dataset workers run concurrently, one per GPU.
When one finishes, the next queued dataset takes that GPU.

The five existing meta-HPO datasets retain the fixed-external protocol used by
the previous trial bank: grouped-CV validation MCC is the Optuna objective and
the labeled `*_inference.csv` cross-test is monitoring-only. The three new
whole-dataset benchmarks use cyclic batch train/valid/test on `*_all.csv`.
In both protocols, test scores are excluded from Optuna selection.

Every trial now persists and logs:

- mean validation MCC;
- every validation-fold MCC;
- paired test/cross-test MCC;
- every test-fold MCC;
- test fold mean and standard deviation when applicable;
- cyclic global out-of-fold test MCC when applicable;
- all other scalar validation/test metrics returned by BERNN;
- the sampled hyperparameters and fit time.

Resume an interrupted experiment without mixing studies:

```bash
python scripts/run_independent_optuna_all_datasets.py \
  --n-trials 20 \
  --n-epochs 1000 \
  --gpus 0,1 \
  --output-dir results/independent_optuna_all_20 \
  --wandb-group independent-optuna-all-datasets-20 \
  --resume
```

DRIAMS is not included in this list until a compact prepared dataset is actually
available in the repository; it should be added as another independent study
rather than silently falling back to the 145 GB upstream download.
