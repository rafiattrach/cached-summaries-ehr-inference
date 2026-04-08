#!/bin/bash
# Grid search over recent context window size N and summary size S.
# Runs baseline, cached (FiLM + token), and oracle conditions across
# all N x S combinations, with three seeds per configuration.
#
# Usage:
#   bash scripts/sweep_context_budget.sh [--gpu 0] [--epochs 10] [--out_dir /path/to/outputs]
#
# The full sweep (N in {8,16,32,64,128,256}, S in {128,256,512}) produces
# 252 total runs. Adjust N_VALUES / S_VALUES to run a subset.

set -euo pipefail

GPU=0
EPOCHS=10
OUT_DIR="$HOME/outputs/context_budget_sweep"
SEEDS=(0 1 2)
N_VALUES=(8 16 32 64 128 256)
S_VALUES=(128 256 512)
REPO_DIR="$(pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)      GPU="$2";     shift 2 ;;
    --epochs)   EPOCHS="$2";  shift 2 ;;
    --out_dir)  OUT_DIR="$2"; shift 2 ;;
    --repo_dir) REPO_DIR="$2"; shift 2 ;;
    *) echo "[WARN] unknown arg $1"; shift ;;
  esac
done

LAUNCHER="bash $REPO_DIR/scripts/run_experiment.sh"

for N in "${N_VALUES[@]}"; do
  for S in "${S_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      echo "==> baseline  N=${N} seed=${SEED}"
      $LAUNCHER -m baseline -n "$N" -g "$GPU" -e "$EPOCHS" \
        -o "${OUT_DIR}/seed${SEED}"

      echo "==> oracle  N=${N} S=${S} seed=${SEED}"
      $LAUNCHER -m oracle -n "$N" -s "$S" -g "$GPU" -e "$EPOCHS" \
        -o "${OUT_DIR}/seed${SEED}"

      echo "==> hybrid film recent  N=${N} S=${S} seed=${SEED}"
      $LAUNCHER -m hybrid -a film -v recent -n "$N" -s "$S" -g "$GPU" -e "$EPOCHS" \
        -o "${OUT_DIR}/seed${SEED}"

      echo "==> hybrid token recent  N=${N} S=${S} seed=${SEED}"
      $LAUNCHER -m hybrid -a token -v recent -n "$N" -s "$S" -g "$GPU" -e "$EPOCHS" \
        -o "${OUT_DIR}/seed${SEED}"

    done
  done
done

echo "Sweep complete. Results in ${OUT_DIR}"
