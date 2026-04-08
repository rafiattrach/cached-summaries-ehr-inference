#!/bin/bash
# Collect per-seed metrics from a completed sweep directory and compute
# mean +/- std across seeds. Runs the meds-torch evaluation loop on
# held-out test splits for each checkpoint found under OUT_DIR.
#
# Usage:
#   bash scripts/evaluate_multiseed.sh --sweep_dir /path/to/outputs/context_budget_sweep \
#                                      [--gpu 0]

set -euo pipefail

SWEEP_DIR="$HOME/outputs/context_budget_sweep"
GPU=0
REPO_DIR="$(pwd)"
VENV_ACTIVATE="$HOME/venv/bin/activate"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep_dir)  SWEEP_DIR="$2";  shift 2 ;;
    --gpu)        GPU="$2";        shift 2 ;;
    --repo_dir)   REPO_DIR="$2";   shift 2 ;;
    --venv)       VENV_ACTIVATE="$2"; shift 2 ;;
    *) echo "[WARN] unknown arg $1"; shift ;;
  esac
done

if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE" || true
fi

echo "[evaluate] scanning checkpoints under ${SWEEP_DIR}"

find "$SWEEP_DIR" -name "best_model.ckpt" | sort | while read -r CKPT; do
  RUN_DIR=$(dirname "$(dirname "$CKPT")")
  CONFIG="$RUN_DIR/.hydra/config.yaml"
  OUT="$RUN_DIR/test_metrics.json"

  if [ -f "$OUT" ]; then
    echo "[skip] already evaluated: $RUN_DIR"
    continue
  fi

  echo "[eval] $CKPT"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}" \
    meds-torch-eval \
      ckpt_path="$CKPT" \
      paths.output_dir="$RUN_DIR" \
      hydra.searchpath="[pkg://meds_torch.configs,$(pwd)/experiments/configs]" \
    2>&1 | tail -5
done

echo "[evaluate] done"
