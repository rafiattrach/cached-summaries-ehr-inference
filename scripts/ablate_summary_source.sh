#!/bin/bash
# Ablation: recent vs distant summary source across all N values.
# Holds S=256 and apply=film fixed; varies which part of the history
# is encoded into the summary vector.
#
# Recent  = the S events immediately preceding the inference tail
# Distant = the earliest S events in the patient record
#
# Usage:
#   bash scripts/ablate_summary_source.sh [--gpu 0] [--epochs 10]

set -euo pipefail

GPU=0
EPOCHS=10
SEEDS=(0 1 2)
S=256
APPLY="film"
N_VALUES=(8 16 32 64 128 256)
OUT_DIR="$HOME/outputs/ablation_summary_source"
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
  for SEED in "${SEEDS[@]}"; do
    for VARIANT in recent distant; do
      echo "==> N=${N} variant=${VARIANT} seed=${SEED}"
      $LAUNCHER -m hybrid -a "$APPLY" -v "$VARIANT" \
        -n "$N" -s "$S" -g "$GPU" -e "$EPOCHS" \
        -o "${OUT_DIR}/seed${SEED}"
    done
  done
done

echo "Summary source ablation complete. Results in ${OUT_DIR}"
