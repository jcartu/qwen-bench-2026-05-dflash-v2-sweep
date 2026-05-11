#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_stage_b.sh — Stage B: num_speculative_tokens sweep (4 configs)
#
# Uses the buffer/graph values that won Stage A (passed via env or read from
# stage-a/_winner.txt).
#
# Grid:
#   num_speculative_tokens ∈ {4, 8, 15, 16}
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/sweep_lib.sh"

WINNER_FILE="${STUDY_ROOT}/configs/stage-a/_winner.txt"
if [ -z "${WIN_BATCHED:-}" ] || [ -z "${WIN_CAPTURE:-}" ]; then
  if [ ! -f "$WINNER_FILE" ]; then
    echo "ERROR: no WIN_BATCHED/WIN_CAPTURE env vars and $WINNER_FILE missing" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$WINNER_FILE"
fi

echo "[STAGE B] using buffer/graph winners: batched=$WIN_BATCHED capture=$WIN_CAPTURE"

STAGE="stage-b"
STAGE_DIR="$STUDY_ROOT/configs/$STAGE"
mkdir -p "$STAGE_DIR"

declare -a NUM_SPEC_VALUES=(4 8 15 16)

STAGE_START=$(date +%s)
TOTAL=4
CUR=0

for n in "${NUM_SPEC_VALUES[@]}"; do
  CUR=$((CUR+1))
  LABEL="b${WIN_BATCHED}_c${WIN_CAPTURE}_n${n}"
  OUT="$STAGE_DIR/$LABEL"
  mkdir -p "$OUT"

  if [ -f "$OUT/throughput.json" ] && [ -f "$OUT/prefill.json" ]; then
    echo "[$(date '+%H:%M:%S')] $LABEL already complete — skipping"
    continue
  fi

  CFG_START=$(date +%s)
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "[$(date '+%H:%M:%S MSK')] STAGE B — config $CUR/$TOTAL: $LABEL"
  echo "  batched=$WIN_BATCHED capture=$WIN_CAPTURE num_spec=$n"
  eta_remaining $((CUR-1)) $TOTAL $(( $(date +%s) - STAGE_START )) "STAGE-B"
  echo "═══════════════════════════════════════════════════════════════"

  launch_config "$LABEL" "$WIN_BATCHED" "$WIN_CAPTURE" "$n" "$OUT"
  if ! wait_for_ready "$OUT" 600; then
    echo "  [SKIP] launch failed for $LABEL"
    stop_config
    echo "FAIL: $LABEL" >> "$STAGE_DIR/_failures.log"
    continue
  fi

  echo "  [$(date '+%H:%M:%S')] settling 60s..."
  settle 60
  echo "  [$(date '+%H:%M:%S')] gates..."
  run_gates "$OUT" || echo "  (gates non-zero)"
  echo "  [$(date '+%H:%M:%S')] throughput matrix..."
  run_throughput "$OUT" || echo "  (throughput non-zero)"
  echo "  [$(date '+%H:%M:%S')] prefill..."
  run_prefill "$OUT" || echo "  (prefill non-zero)"

  stop_config

  CFG_ELAPSED=$(( $(date +%s) - CFG_START ))
  echo "$LABEL,${CFG_ELAPSED}s" >> "$STAGE_DIR/_elapsed.csv"
  echo "  [$(date '+%H:%M:%S')] done in ${CFG_ELAPSED}s"
done

STAGE_ELAPSED=$(( $(date +%s) - STAGE_START ))
echo ""
echo "[STAGE B COMPLETE] $TOTAL configs in $((STAGE_ELAPSED/60))m $((STAGE_ELAPSED%60))s"
