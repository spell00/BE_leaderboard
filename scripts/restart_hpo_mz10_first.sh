#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="results/independent_optuna_all_20"
LOG="$OUT/launcher_tmux.log"
SESSION="optuna_all"
mkdir -p "$OUT/logs"

echo "[restart] stopping previous independent-HPO launcher and its workers"

declare -a launchers=()
while read -r pid; do
  [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] || continue
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  if [[ "$cmd" == *"run_independent_optuna_all_datasets.py"* ]] \
     && [[ "$cmd" != *"--worker-dataset"* ]]; then
    launchers+=("$pid")
  fi
done < <(pgrep -f "run_independent_optuna_all_datasets.py" || true)

collect_descendants() {
  local parent="$1"
  local child
  while read -r child; do
    [[ -n "$child" ]] || continue
    collect_descendants "$child"
    victims+=("$child")
  done < <(pgrep -P "$parent" || true)
}

declare -a victims=()
for launcher in "${launchers[@]}"; do
  collect_descendants "$launcher"
  victims+=("$launcher")
done

# Stop an older direct mz10 run or insertion watcher so the new queue cannot
# launch a duplicate copy later.
for pattern in "run_bacteria_2024_mz10_optuna.py" "insert_bacteria_mz10_after_current.sh"; do
  while read -r pid; do
    [[ -n "$pid" && "$pid" != "$$" ]] && victims+=("$pid")
  done < <(pgrep -f "$pattern" || true)
done

mapfile -t victims < <(printf '%s\n' "${victims[@]:-}" | awk 'NF && !seen[$0]++')
if (( ${#victims[@]} )); then
  echo "[restart] TERM -> ${victims[*]}"
  kill -TERM "${victims[@]}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    alive=()
    for pid in "${victims[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    (( ${#alive[@]} == 0 )) && break
    sleep 1
  done
  for pid in "${victims[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "[restart] KILL -> $pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
else
  echo "[restart] no previous matching HPO processes found"
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

for path in \
  data/datasets/bacteria_2024_mz10/bacteria_2024_mz10_features_csr.npz \
  data/datasets/bacteria_2024_mz10/bacteria_2024_mz10_metadata.csv \
  data/datasets/bacteria_2024_mz10/bacteria_2024_mz10_feature_names.npy \
  data/datasets/bacteria_2024_mz10/provenance.json
do
  [[ -f "$path" ]] || { echo "[restart] missing required mz10 file: $path" >&2; exit 1; }
done

echo "[restart] launching mz10-first queue on GPUs 0,1"
tmux new-session -d -s "$SESSION" "cd '$ROOT' && \
  /home/simonp/anaconda3/bin/python scripts/run_independent_optuna_all_datasets.py \
    --n-trials 20 \
    --n-epochs 1000 \
    --n-repeats 3 \
    --gpus 0,1 \
    --prepare-missing \
    --resume \
    --output-dir '$OUT' \
    --wandb-group independent-optuna-all-datasets-20 \
    2>&1 | tee -a '$LOG'"

sleep 3
echo "[restart] tmux session: $SESSION"
echo "[restart] queue log: $LOG"
tmux capture-pane -p -t "$SESSION" -S -40 || true
