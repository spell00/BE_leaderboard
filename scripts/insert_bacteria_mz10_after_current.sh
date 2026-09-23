#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="results/independent_optuna_all_20/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/bacteria_2024_mz10_insert.log"

find_launcher() {
  local pid cmd
  while read -r pid; do
    [[ -r "/proc/$pid/cmdline" ]] || continue
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    if [[ "$cmd" == *"run_independent_optuna_all_datasets.py"* ]] \
       && [[ "$cmd" == *"--gpus"* ]] \
       && [[ "$cmd" != *"--worker-dataset"* ]]; then
      echo "$pid"
      return 0
    fi
  done < <(pgrep -f "run_independent_optuna_all_datasets.py" || true)
  return 1
}

process_done() {
  local pid="$1"
  [[ -r "/proc/$pid/stat" ]] || return 0
  local state
  state="$(awk '{print $3}' "/proc/$pid/stat")"
  [[ "$state" == "Z" || "$state" == "X" ]]
}

launcher_pid="$(find_launcher)" || {
  echo "Could not find the active independent Optuna launcher" >&2
  exit 1
}

mapfile -t worker_pids < <(pgrep -P "$launcher_pid" || true)
workers=()
gpus=()
for pid in "${worker_pids[@]}"; do
  [[ -r "/proc/$pid/cmdline" ]] || continue
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmd" == *"--worker-dataset"* ]] || continue
  gpu="$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)"
  [[ -n "$gpu" ]] || {
    echo "Could not resolve CUDA_VISIBLE_DEVICES for worker $pid" >&2
    exit 1
  }
  workers+=("$pid")
  gpus+=("$gpu")
  echo "[insert] watching worker pid=$pid gpu=$gpu cmd=$cmd" | tee -a "$LOG"
done

if (( ${#workers[@]} == 0 )); then
  echo "No active dataset workers found under launcher $launcher_pid" >&2
  exit 1
fi

echo "[insert] pausing launcher supervisor pid=$launcher_pid; active workers continue" | tee -a "$LOG"
kill -STOP "$launcher_pid"

resume_launcher() {
  if kill -0 "$launcher_pid" 2>/dev/null; then
    kill -CONT "$launcher_pid" 2>/dev/null || true
    echo "[insert] resumed launcher pid=$launcher_pid" | tee -a "$LOG"
  fi
}
trap resume_launcher EXIT

free_gpu=""
finished_pid=""
while [[ -z "$free_gpu" ]]; do
  for i in "${!workers[@]}"; do
    pid="${workers[$i]}"
    if process_done "$pid"; then
      free_gpu="${gpus[$i]}"
      finished_pid="$pid"
      break
    fi
  done
  [[ -n "$free_gpu" ]] || sleep 10
done

echo "[insert] worker pid=$finished_pid finished; inserting bacteria_2024_mz10 on GPU $free_gpu" | tee -a "$LOG"

resume_arg=()
if [[ -f "results/independent_optuna_all_20/bacteria_2024_mz10/run_metadata.json" ]]; then
  resume_arg=(--resume)
fi

CUDA_VISIBLE_DEVICES="$free_gpu" /home/simonp/anaconda3/bin/python \
  scripts/run_bacteria_2024_mz10_optuna.py \
  --n-trials 20 \
  --n-epochs 1000 \
  --batch-size 32 \
  --num-workers 4 \
  --output-dir results/independent_optuna_all_20 \
  --wandb-group independent-optuna-all-datasets-20 \
  "${resume_arg[@]}" \
  2>&1 | tee -a "$LOG"

echo "[insert] bacteria_2024_mz10 finished; returning GPU $free_gpu to the original queue" | tee -a "$LOG"
