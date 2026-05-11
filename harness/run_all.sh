#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_all.sh — Master orchestrator for the v2 DFlash sweep study.
#
# Sequence:
#   1. Stop the SOTA service to free GPU 0+1
#   2. Run Stage A (9 configs, 3×3 buffer/graph sweep)
#   3. Pick Stage A winner
#   4. Run Stage B (4 configs, num_speculative_tokens sweep at Stage A winner)
#   5. Pick Stage B winner
#   6. Run Quality phase (HumanEval + MBPP on Stage B winner + Repne's baseline)
#   7. Restart the SOTA service
#
# GPU 2 (oss-120b for Hindsight) is NEVER touched.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export STUDY_ROOT

LOG="$STUDY_ROOT/logs/run_all_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p "$STUDY_ROOT/logs"

WALL_START=$(date +%s)

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Qwen3.6-27B BF16+DFlash v2 Parameter Sweep"
echo "  STUDY_ROOT: $STUDY_ROOT"
echo "  Started:    $(date '+%Y-%m-%d %H:%M:%S MSK')"
echo "  Log:        $LOG"
echo "═══════════════════════════════════════════════════════════════════════"

# Step 1: stop SOTA service
echo "[$(date '+%H:%M:%S')] Stopping vllm-qwen36-27b-sota.service..."
systemctl --user stop vllm-qwen36-27b-sota.service || true
sleep 10
# Make sure the SOTA container is fully gone
docker rm -f vllm-qwen36-27b-sota >/dev/null 2>&1 || true
docker rm -f vllm-sweep >/dev/null 2>&1 || true
sleep 5
echo "[$(date '+%H:%M:%S')] SOTA stopped. GPU 2 (oss-120b) still running:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Step 2: Stage A
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  STAGE A — buffer/graph sweep (9 configs)"
echo "═══════════════════════════════════════════════════════════════════════"
bash "$SCRIPT_DIR/run_stage_a.sh" 2>&1 | tee -a "$LOG"

# Step 3: pick Stage A winner
echo ""
echo "[$(date '+%H:%M:%S')] Picking Stage A winner..."
python3 "$SCRIPT_DIR/pick_stage_a_winner.py" 2>&1 | tee -a "$LOG"

# Step 4: Stage B
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  STAGE B — num_speculative_tokens sweep at Stage A winner (4 configs)"
echo "═══════════════════════════════════════════════════════════════════════"
bash "$SCRIPT_DIR/run_stage_b.sh" 2>&1 | tee -a "$LOG"

# Step 5: pick Stage B winner
echo ""
echo "[$(date '+%H:%M:%S')] Picking Stage B winner..."
python3 "$SCRIPT_DIR/pick_stage_b_winner.py" 2>&1 | tee -a "$LOG"

# Step 6: Quality phase
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  QUALITY — HumanEval + MBPP @ c=8 on Stage B winner + Repne baseline"
echo "═══════════════════════════════════════════════════════════════════════"
bash "$SCRIPT_DIR/run_quality.sh" 2>&1 | tee -a "$LOG"

# Step 7: restart SOTA
echo ""
echo "[$(date '+%H:%M:%S')] Restarting vllm-qwen36-27b-sota.service..."
systemctl --user start vllm-qwen36-27b-sota.service

WALL_ELAPSED=$(( $(date +%s) - WALL_START ))
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  SWEEP COMPLETE in $((WALL_ELAPSED/3600))h $(((WALL_ELAPSED%3600)/60))m $((WALL_ELAPSED%60))s"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S MSK')"
echo "═══════════════════════════════════════════════════════════════════════"
