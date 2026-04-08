#!/bin/bash
# Core training launcher for hybrid summary encoder experiments.
#
# Usage:
#   bash scripts/run_experiment.sh -m baseline|hybrid|oracle \
#                                  [-a token|film] [-v recent|distant] \
#                                  -n N -s S -g GPU [-e EPOCHS] [-b BATCH] \
#                                  [-o OUT_PARENT] [-d TASK] [-r REPO_DIR] [-p VENV_ACTIVATE]
#
# Mode:
#   baseline  recent window only (triplet encoder, no summary)
#   hybrid    recent window + cached summary (FiLM or token injection)
#   oracle    full context window as upper bound
#
# Examples:
#   bash scripts/run_experiment.sh -m hybrid -a film -v recent -n 64 -s 256 -g 0
#   bash scripts/run_experiment.sh -m baseline -n 64 -g 0
#   bash scripts/run_experiment.sh -m oracle -n 64 -s 256 -g 0

set -euo pipefail

MODE="hybrid"      # baseline | hybrid | oracle
APPLY="film"       # token | film (hybrid only)
VARIANT="recent"   # recent | distant (hybrid only)
N=8
S=256
GPU=0
EPOCHS=1
BATCH=64
OUT_PARENT="$HOME/smoke_outputs/hybrid_summary_runs"
TASK="mortality/in_icu/first_24h"
REPO_DIR="$HOME/cached-summaries-ehr-inference"
VENV_ACTIVATE="$HOME/venv/bin/activate"

while getopts ":m:a:v:n:s:g:e:b:o:d:r:p:S:" opt; do
  case $opt in
    m) MODE="$OPTARG" ;;
    a) APPLY="$OPTARG" ;;
    v) VARIANT="$OPTARG" ;;
    n) N="$OPTARG" ;;
    s) S="$OPTARG" ;;
    g) GPU="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    b) BATCH="$OPTARG" ;;
    o) OUT_PARENT="$OPTARG" ;;
    d) TASK="$OPTARG" ;;
    r) REPO_DIR="$OPTARG" ;;
    p) VENV_ACTIVATE="$OPTARG" ;;
    S) OUT_PARENT="$OPTARG" ;;
    *) echo "[WARN] unknown flag -$OPTARG" ;;
  esac
done

# Optional environment setup (local only; no git operations)
if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE" || true
fi

# Resolve cache root per S (hybrid only)
if [ "$S" = "256" ]; then
  CACHE_ROOT="$HOME/triplet_tensors/hybrid_summary_cache/cache_bioclinical_modernbert_base"
else
  CACHE_ROOT="$HOME/triplet_tensors/hybrid_summary_cache_S${S}/cache_bioclinical_modernbert_base"
fi

# Build label
if [ "$MODE" = "hybrid" ]; then
  LABEL="${MODE}_${APPLY}_${VARIANT}_N${N}_S${S}"
elif [ "$MODE" = "oracle" ]; then
  LABEL="${MODE}_N${N}_ctx$((N+S-1))"
else
  LABEL="${MODE}_N${N}"
fi
OUT_BASE="$OUT_PARENT/$LABEL"

# Clean base for a fresh timestamped run
rm -rf "$OUT_BASE"; mkdir -p "$OUT_BASE"

# Encoder stats (hybrid only)
# No global HYBRID_SUMMARY_STATS_PATH/WANDB exports; set inline at launch

# Common args
COMMON_ARGS=(
  experiment=triplet_mtr logger=csv
  paths.data_dir="$HOME/triplet_tensors"
  paths.meds_cohort_dir="$HOME/MEDS_cohort"
  data.task_name="$TASK"
  data.task_root_dir="$HOME/MEDS_cohort/tasks"
  paths.output_dir="$OUT_BASE"
  ++data.subsequence_sampling_strategy=to_end
  ++data.do_include_subject_id=true ++data.do_include_subsequence_indices=true ++data.do_include_end_time=true
  ++data.dataloader.batch_size="$BATCH"
  trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs="$EPOCHS" trainer.precision=32 trainer.strategy=auto
  +evaluation.metric_name=auc +evaluation.higher_is_better=true
  hydra.searchpath="[pkg://meds_torch.configs,$(pwd)/experiments/configs]"
)

# Mode-specific args
if [ "$MODE" = "baseline" ]; then
  TRAIN_ARGS=(
    model/input_encoder=triplet_encoder ++model.token_dim=128
    ++model.max_seq_len="$N" ++data.max_seq_len="$N"
  )
elif [ "$MODE" = "oracle" ]; then
  CTX=$((N + S - 1))
  TRAIN_ARGS=(
    model/input_encoder=triplet_encoder ++model.token_dim=128
    ++model.max_seq_len="$CTX" ++data.max_seq_len="$CTX"
  )
else # hybrid
  TRAIN_ARGS=(
    model/input_encoder=hybrid_summary_encoder ++model.token_dim=128
    ++model.max_seq_len="$N" ++data.max_seq_len="$N"
    ++model.input_encoder.summary.enabled=true
    ++model.input_encoder.summary.variant="$VARIANT"
    ++model.input_encoder.summary.S="$S"
    ++model.input_encoder.summary.apply="$APPLY"
    ++model.input_encoder.summary.policy=omit_if_L_lt_N
    ++model.input_encoder.summary.cache_root="$CACHE_ROOT"
    ++model.input_encoder.summary.task_name="$TASK"
    ++model.input_encoder.summary.strict=true
  )
fi

# Run
START=$(date +%s)
# optional GPU sampler (util%, mem MB, power W) to a temp file; no artifacts left behind
TMP_SAMPLER_FILE=""
SAMP_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  TMP_SAMPLER_FILE=$(mktemp -t gpu_sampler_${GPU}_XXXXXX)
  ( while :; do nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -i "$GPU" >> "$TMP_SAMPLER_FILE" 2>/dev/null; sleep 1; done ) & SAMP_PID=$!
fi
if [ "$MODE" = "hybrid" ]; then
  env PYTHONUNBUFFERED=1 PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}" HYBRID_SUMMARY_STATS_PATH="$OUT_BASE/summary_stats.tmp.json" WANDB_DISABLED=true CUDA_VISIBLE_DEVICES="$GPU" \
    meds-torch-train "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}"
else
  env PYTHONUNBUFFERED=1 PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}" WANDB_DISABLED=true CUDA_VISIBLE_DEVICES="$GPU" \
    meds-torch-train "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}"
fi
END=$(date +%s)
# stop GPU sampler and compute peaks; remove temp file
if [ -n "$SAMP_PID" ]; then kill "$SAMP_PID" 2>/dev/null || true; fi
GPU_MEM_PEAK=""; GPU_UTIL_PEAK=""; GPU_POWER_PEAK=""
if [ -n "$TMP_SAMPLER_FILE" ] && [ -f "$TMP_SAMPLER_FILE" ]; then
  GPU_UTIL_PEAK=$(awk -F',' 'u+0>max{max=u+0}{u=$1} END{if (max=="") print ""; else print int(max)}' "$TMP_SAMPLER_FILE" 2>/dev/null || echo "")
  GPU_MEM_PEAK=$(awk -F',' 'm+0>max{max=m+0}{m=$2} END{if (max=="") print ""; else print int(max)}' "$TMP_SAMPLER_FILE" 2>/dev/null || echo "")
  GPU_POWER_PEAK=$(awk -F',' 'p+0>max{max=p+0}{p=$3} END{if (max=="") print ""; else print int(max)}' "$TMP_SAMPLER_FILE" 2>/dev/null || echo "")
  rm -f "$TMP_SAMPLER_FILE" || true
fi

# Helper: find the latest csv/version_* directory robustly (cross-platform)
find_latest_csv_dir() {
  python - "$OUT_BASE" <<'PY'
import os, sys, glob
base = sys.argv[1]
candidates = []
# Prefer timestamped subdirs
patterns = [
  os.path.join(base, '*', 'csv', 'version_*', 'metrics.csv'),
  os.path.join(base, '*', 'csv', 'metrics.csv'),
  os.path.join(base, 'csv', 'version_*', 'metrics.csv'),
  os.path.join(base, 'csv', 'metrics.csv'),
]
for pat in patterns:
    for p in glob.glob(pat):
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        candidates.append((mt, p))
if not candidates:
    print("")
else:
    candidates.sort()
    latest = candidates[-1][1]
    print(os.path.dirname(latest))
PY
}

CSV_DIR=$(find_latest_csv_dir)
if [ -z "${CSV_DIR}" ] || [ ! -d "${CSV_DIR}" ]; then
  echo "[WARN] metrics.csv not found; meta files not written"
  exit 0
fi
CSV="$CSV_DIR/metrics.csv"
RUN_ROOT=$(dirname "$(dirname "$CSV_DIR")")
# RUN_ROOT is <time_output_dir>
LOG_DIR="$RUN_ROOT/logs"

# Move encoder stats next to metrics.csv (hybrid only)
if [ "$MODE" = "hybrid" ] && [ -f "$OUT_BASE/summary_stats.tmp.json" ]; then
  mv "$OUT_BASE/summary_stats.tmp.json" "$CSV_DIR/summary_stats.json" || true
fi

# Resolve checkpoint robustly
CKPT=""
# 1) parse from latest log file (logs/ or root .log)
LOG=""
if [ -d "$LOG_DIR" ]; then
  LOG=$(ls -t "$LOG_DIR"/* 2>/dev/null | head -1 || true)
fi
if [ -z "${LOG}" ]; then
  LOG=$(ls -t "$RUN_ROOT"/*.log 2>/dev/null | head -1 || true)
fi
if [ -n "${LOG:-}" ] && [ -f "$LOG" ]; then
  CKPT=$(grep -m1 -E "Best ckpt path:" "$LOG" 2>/dev/null | sed -E 's/.*Best ckpt path: //' || true)
fi
# 2) newest in time_output_dir/checkpoints
if [ -z "$CKPT" ]; then
  CKPT=$(ls -t "$RUN_ROOT"/checkpoints/*.ckpt 2>/dev/null | head -1 || true)
fi
# 3) copied best model
if [ -z "$CKPT" ] && [ -f "$RUN_ROOT/checkpoints/best_model.ckpt" ]; then
  CKPT="$RUN_ROOT/checkpoints/best_model.ckpt"
fi

# Param count from checkpoint or fallback to logs
PARAMS_NUM=""
if [ -n "${CKPT:-}" ] && [ -f "$CKPT" ]; then
  PARAMS_NUM=$(python - "$CKPT" <<'PY'
import sys, torch
p=sys.argv[1]
try:
  sd=torch.load(p, map_location='cpu', weights_only=False)
  if isinstance(sd, dict) and 'state_dict' in sd:
    sd=sd['state_dict']
  n=sum(int(getattr(v,'numel',lambda:0)()) for v in (sd.values() if isinstance(sd, dict) else []))
  print(n)
except Exception as e:
  print("")
PY
)
fi
RAW_PARAMS=""
if [ -n "$PARAMS_NUM" ]; then
  RAW_PARAMS=$(python - "$PARAMS_NUM" <<'PY'
import sys
n=int(sys.argv[1])
print(f"{n/1_000_000:.1f}M" if n>=1_000_000 else (f"{n/1_000:.0f}K" if n>=1_000 else str(n)))
PY
)
fi

# Last test metrics from CSV
TEST_AUC="null"; TEST_ACC="null"; TEST_APR="null"; TEST_LOSS="null"
if [ -f "$CSV" ]; then
  read -r TEST_AUC TEST_ACC TEST_APR TEST_LOSS < <(python - "$CSV" <<'PY'
import sys,csv
# default nulls
d={"test/auc":"null","test/acc":"null","test_apr":"null","test/loss":"null"}
with open(sys.argv[1]) as f:
  r=csv.DictReader(f)
  for row in r:
    for k in list(d.keys()):
      v=row.get(k)
      if v not in (None, "", "nan"):
        d[k]=v
print(d["test/auc"], d["test/acc"], d["test_apr"], d["test/loss"])
PY
)
fi

# EDT + duration via Python for cross-platform robustness
WALL_S=$((END-START))
read -r START_EDT END_EDT WALL_MIN WALL_HMS < <(python - "$START" "$END" <<'PY'
import sys, datetime
try:
  from zoneinfo import ZoneInfo
  tz = ZoneInfo("America/New_York")
except Exception:
  tz = datetime.timezone(datetime.timedelta(hours=-4))
start=int(sys.argv[1])
end=int(sys.argv[2])
wall_s = max(0, end-start)
wall_min = f"{(wall_s/60):.2f}"
hms = f"{wall_s//3600:02d}:{(wall_s%3600)//60:02d}:{wall_s%60:02d}"
se = datetime.datetime.fromtimestamp(start, tz=tz).strftime("%Y-%m-%dT%H:%M:%S%z")
ee = datetime.datetime.fromtimestamp(end, tz=tz).strftime("%Y-%m-%dT%H:%M:%S%z")
print(se, ee, wall_min, hms)
PY
)

# Write meta next to metrics.csv
cat > "$CSV_DIR/run_info.json" <<EOF
{
  "label": "${LABEL}",
  "mode": "${MODE}",
  "apply": "${APPLY}",
  "variant": "${VARIANT}",
  "N": ${N},
  "S": ${S},
  "start_edt": "${START_EDT}",
  "end_edt": "${END_EDT}",
  "wall_s": ${WALL_S},
  "wall_min": ${WALL_MIN},
  "wall_hms": "${WALL_HMS}",
  "test_auc": ${TEST_AUC},
  "test_acc": ${TEST_ACC},
  "test_apr": ${TEST_APR},
  "test_loss": ${TEST_LOSS},
  "param_count_num": $( [ -n "${PARAMS_NUM}" ] && echo "${PARAMS_NUM}" || echo "null" ),
  "param_count_raw": "${RAW_PARAMS}",
  "gpu_mem_peak_mb": $( [ -n "${GPU_MEM_PEAK}" ] && echo "${GPU_MEM_PEAK}" || echo "null" ),
  "gpu_util_peak_pct": $( [ -n "${GPU_UTIL_PEAK}" ] && echo "${GPU_UTIL_PEAK}" || echo "null" ),
  "gpu_power_peak_w": $( [ -n "${GPU_POWER_PEAK}" ] && echo "${GPU_POWER_PEAK}" || echo "null" )
}
EOF

# Minimal final listing (quiet)
ls -l "$CSV_DIR/metrics.csv" "$CSV_DIR/run_info.json" 2>/dev/null || true
if [ "$MODE" = "hybrid" ]; then
  [ -f "$CSV_DIR/summary_stats.json" ] && echo "summary_stats.json present" || true
fi 