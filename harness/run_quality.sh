#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_quality.sh — Quality phase: HumanEval + MBPP @ c=8
#
# Runs on:
#   1. Stage B winner (the overall best speed config)
#   2. Repne's baseline (max-num-batched-tokens=32768, capture=256, num_spec=8)
#
# Per user's choice: 2 configs only, ~1h total wall time.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/sweep_lib.sh"

STAGE="quality"
STAGE_DIR="$STUDY_ROOT/configs/$STAGE"
mkdir -p "$STAGE_DIR"

# Read Stage B winner
SB_WINNER_FILE="${STUDY_ROOT}/configs/stage-b/_winner.txt"
if [ ! -f "$SB_WINNER_FILE" ]; then
  echo "ERROR: $SB_WINNER_FILE not found — run pick_stage_b_winner.py first" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$SB_WINNER_FILE"
echo "[QUALITY] Stage B winner: batched=$WIN_BATCHED capture=$WIN_CAPTURE num_spec=$WIN_NUM_SPEC"

# Build config list: (label, batched, capture, num_spec)
declare -a CONFIGS=(
  "winner|${WIN_BATCHED}|${WIN_CAPTURE}|${WIN_NUM_SPEC}"
  "repne_baseline|32768|256|8"
)

STAGE_START=$(date +%s)
TOTAL=${#CONFIGS[@]}
CUR=0

for entry in "${CONFIGS[@]}"; do
  CUR=$((CUR+1))
  IFS='|' read -r name b c n <<< "$entry"
  LABEL="${name}_b${b}_c${c}_n${n}"
  OUT="$STAGE_DIR/$LABEL"
  mkdir -p "$OUT"

  if [ -f "$OUT/humaneval_summary.json" ] && [ -f "$OUT/mbpp_summary.json" ]; then
    echo "[$(date '+%H:%M:%S')] $LABEL already complete — skipping"
    continue
  fi

  CFG_START=$(date +%s)
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "[$(date '+%H:%M:%S MSK')] QUALITY — config $CUR/$TOTAL: $LABEL"
  echo "  batched=$b capture=$c num_spec=$n"
  eta_remaining $((CUR-1)) $TOTAL $(( $(date +%s) - STAGE_START )) "QUALITY"
  echo "═══════════════════════════════════════════════════════════════"

  launch_config "$LABEL" "$b" "$c" "$n" "$OUT"
  if ! wait_for_ready "$OUT" 600; then
    echo "  [SKIP] launch failed"
    stop_config
    continue
  fi
  echo "  [$(date '+%H:%M:%S')] settling 60s..."
  settle 60
  echo "  [$(date '+%H:%M:%S')] gates..."
  run_gates "$OUT" || true
  echo "  [$(date '+%H:%M:%S')] HumanEval + MBPP @ c=8..."
  run_quality "$OUT"

  stop_config

  CFG_ELAPSED=$(( $(date +%s) - CFG_START ))
  echo "$LABEL,${CFG_ELAPSED}s" >> "$STAGE_DIR/_elapsed.csv"
  echo "  [$(date '+%H:%M:%S')] done in ${CFG_ELAPSED}s"
done

STAGE_ELAPSED=$(( $(date +%s) - STAGE_START ))
echo "[QUALITY COMPLETE] $TOTAL configs in $((STAGE_ELAPSED/60))m $((STAGE_ELAPSED%60))s"
