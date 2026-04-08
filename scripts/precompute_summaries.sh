#!/bin/bash
# Offline summary precomputation using BioClinical-ModernBERT.
# Run this once before any cached model training. Summaries are written
# to disk and reused across all downstream training runs.
#
# Usage:
#   bash scripts/precompute_summaries.sh \
#       --data_dir /path/to/triplet_tensors \
#       --output_dir /path/to/triplet_tensors/hybrid_summary_cache \
#       [--summary_size 256] [--variant recent|distant] [--gpu 0]
#
# Output: one .pt file per patient under output_dir/cache_bioclinical_modernbert_base/

set -euo pipefail

DATA_DIR="$HOME/triplet_tensors"
OUTPUT_DIR="$HOME/triplet_tensors/hybrid_summary_cache"
CODES_CSV="$(pwd)/mapping/meds_triplet_descriptions.csv"
SUMMARY_SIZE=256
VARIANT="recent"
GPU=0
MODEL_NAME="thomas-sounack/BioClinical-ModernBERT-base"
BATCH_SIZE=128
REPO_DIR="$(pwd)"
VENV_ACTIVATE="$HOME/venv/bin/activate"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)    DATA_DIR="$2";    shift 2 ;;
    --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
    --codes_csv)   CODES_CSV="$2";   shift 2 ;;
    --summary_size) SUMMARY_SIZE="$2"; shift 2 ;;
    --variant)     VARIANT="$2";     shift 2 ;;
    --gpu)         GPU="$2";         shift 2 ;;
    --model)       MODEL_NAME="$2";  shift 2 ;;
    --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
    --repo_dir)    REPO_DIR="$2";    shift 2 ;;
    --venv)        VENV_ACTIVATE="$2"; shift 2 ;;
    *) echo "[WARN] unknown arg $1"; shift ;;
  esac
done

if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE" || true
fi

echo "[precompute] data_dir=${DATA_DIR}"
echo "[precompute] output_dir=${OUTPUT_DIR}"
echo "[precompute] variant=${VARIANT}  summary_size=${SUMMARY_SIZE}"
echo "[precompute] model=${MODEL_NAME}"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}" \
  python src/meds_torch/utils/precompute_long_context_summaries.py \
    --data_dir "$DATA_DIR" \
    --cache_root "$OUTPUT_DIR" \
    --mapping_csv "$CODES_CSV" \
    --S "$SUMMARY_SIZE" \
    --variant "$VARIANT" \
    --model "$MODEL_NAME" \
    --N 64

echo "[precompute] done. Summaries written to ${OUTPUT_DIR}"
