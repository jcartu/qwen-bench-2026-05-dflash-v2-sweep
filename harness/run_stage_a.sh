#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_stage_a.sh — Stage A: buffer/graph sweep (3×3=9 configs)
#
# num_speculative_tokens fixed at Repne's baseline (8) per his suggestion that
# the parameter can be tested independently in Stage B.
#
# Grid:
#   max-num-batched-tokens ∈ {8192, 16384, 32768}
#   max-cudagraph-capture-size ∈ {64, 128, 256}
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/sweep_lib.sh"

STAGE="stage-a"
STAGE_DIR="$STUDY_ROOT/configs/$STAGE"
mkdir -p "$STAGE_DIR"

declare -a BATCHED=(8192 16384 32768)
declare -a CAPTURE=(64 128 256)
NUM_SPEC=8

STAGE_START=$(date +%s)
TOTAL=9
CUR=0

for b in "${BATCHED[@]}"; do
  for c in "${CAPTURE[@]}"; do
    CUR=$((CUR+1))
    LABEL="b${b}_c${c}_n${NUM_SPEC}"
    OUT="$STAGE_DIR/$LABEL"
    mkdir -p "$OUT"

    if [ -f "$OUT/throughput.json" ] && [ -f "$OUT/prefill.json" ]; then
      echo "[$(date '+%H:%M:%S')] $LABEL already complete — skipping"
      continue
    fi

    CFG_START=$(date +%s)
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "[$(date '+%H:%M:%S MSK')] STAGE A — config $CUR/$TOTAL: $LABEL"
    echo "  max-num-batched-tokens=$b max-cudagraph-capture-size=$c num_spec=$NUM_SPEC"
    eta_remaining $((CUR-1)) $TOTAL $(( $(date +%s) - STAGE_START )) "STAGE-A"
    echo "═══════════════════════════════════════════════════════════════"

    launch_config "$LABEL" "$b" "$c" "$NUM_SPEC" "$OUT"
    if ! wait_for_ready "$OUT" 600; then
      echo "  [SKIP] launch failed for $LABEL"
      stop_config
      echo "FAIL: $LABEL" >> "$STAGE_DIR/_failures.log"
      continue
    fi

    echo "  [$(date '+%H:%M:%S')] settling 60s..."
    settle 60

    echo "  [$(date '+%H:%M:%S')] gates..."
    run_gates "$OUT" || echo "  (gates returned non-zero)"

    echo "  [$(date '+%H:%M:%S')] throughput matrix (3 conc × 5 ctx × 60s)..."
    run_throughput "$OUT" || echo "  (throughput non-zero)"

    echo "  [$(date '+%H:%M:%S')] prefill (5 contexts)..."
    run_prefill "$OUT" || echo "  (prefill non-zero)"

    stop_config

    CFG_ELAPSED=$(( $(date +%s) - CFG_START ))
    echo "$LABEL,${CFG_ELAPSED}s" >> "$STAGE_DIR/_elapsed.csv"
    echo "  [$(date '+%H:%M:%S')] done in ${CFG_ELAPSED}s"
  done
done

STAGE_ELAPSED=$(( $(date +%s) - STAGE_START ))
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[STAGE A COMPLETE] $TOTAL configs in $((STAGE_ELAPSED/60))m $((STAGE_ELAPSED%60))s"
echo "═══════════════════════════════════════════════════════════════"
