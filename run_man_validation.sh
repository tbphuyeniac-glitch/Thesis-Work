#!/usr/bin/env bash
# ============================================================
# run_man_validation.sh
# Run Thesis C vs Man-BFP-TC vs Man-Joint-TC (Oracle)
# with demand shock, across small/medium/large store sizes.
#
# All three models share the SAME per-scenario demand shock
# (generated deterministically from scenario seed).
#
# Phase 1 cost = routing cost  (planned on forecast demand)
# Phase 2 cost = holding + LT transshipment + shortage
#               (evaluated on realized/shocked demand)
#
# Usage:
#   bash run_man_validation.sh                    # full 30-scenario run
#   bash run_man_validation.sh --debug            # 9-scenario quick test
#   bash run_man_validation.sh --sigma 0.0        # deterministic baseline (no shock)
#   bash run_man_validation.sh --sigma 0.20       # 20% demand shock (default)
#   ALLOW_GNN_FALLBACK=1 bash run_man_validation.sh  # skip GNN checkpoint check
# ============================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────
DATA_CSV="test data.csv"
CHECKPOINT="GNN/trained_models/irplt_teacher_filtered_local200_fixed/bigat/pairwise_rank/best_model.pt"
SIGMA="0.20"         # N(1, σ²) demand shock between Phase 1 and Phase 2
OUTPUT_BASE="Results/man_validation"
DEBUG_FLAG=""
SEED=42

# Man et al. setup — single-period rolling BFP
STORE_LIMITS="5 8 10"        # small=5, medium=8, large=10
SCENARIOS_PER_SIZE=10
WINDOW_LENGTH=8              # multi-period horizon (thesis C advantage)
MAN_TIME_LIMIT=1200
THESIS_C_TIME_LIMIT=1200

# ── Argument parsing ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)    DEBUG_FLAG="--debug"; shift ;;
        --sigma)    SIGMA="$2"; shift 2 ;;
        --output)   OUTPUT_BASE="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --window)   WINDOW_LENGTH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Derived output directory (encodes key params in name) ────
SIGMA_TAG=$(echo "$SIGMA" | sed 's/\./_/g')
if [[ -n "$DEBUG_FLAG" ]]; then
    OUTPUT_DIR="${OUTPUT_BASE}/sigma${SIGMA_TAG}_debug_$(date +%Y%m%d_%H%M)"
else
    OUTPUT_DIR="${OUTPUT_BASE}/sigma${SIGMA_TAG}_$(date +%Y%m%d_%H%M)"
fi

# ── Print header ─────────────────────────────────────────────
echo "======================================================"
echo " Man et al. vs Thesis C vs Oracle Validation"
echo "======================================================"
echo "  data_csv:           $DATA_CSV"
echo "  checkpoint:         $CHECKPOINT"
echo "  output_dir:         $OUTPUT_DIR"
echo "  store_limits:       $STORE_LIMITS"
echo "  scenarios_per_size: $SCENARIOS_PER_SIZE"
echo "  window_length:      $WINDOW_LENGTH periods"
echo "  demand_shock_sigma: $SIGMA"
echo "  MAN_TIME_LIMIT:     ${MAN_TIME_LIMIT}s"
echo "  THESIS_C_TIME_LIMIT:${THESIS_C_TIME_LIMIT}s"
echo "  debug:              ${DEBUG_FLAG:-no}"
echo "======================================================"
echo ""

# ── Run validation ───────────────────────────────────────────
MAN_TIME_LIMIT=$MAN_TIME_LIMIT \
THESIS_C_TIME_LIMIT=$THESIS_C_TIME_LIMIT \
python3 Validate_with_Man_Kaggle.py \
    --data_csv       "$DATA_CSV" \
    --checkpoint     "$CHECKPOINT" \
    --output_dir     "$OUTPUT_DIR" \
    --store_limits   $STORE_LIMITS \
    --scenarios_per_size $SCENARIOS_PER_SIZE \
    --window_length  $WINDOW_LENGTH \
    --demand_shock_sigma "$SIGMA" \
    --joint_tc \
    --seed           $SEED \
    $DEBUG_FLAG

# ── Post-process: full analysis table ───────────────────────
echo ""
echo "======================================================"
echo " Running post-processing analysis..."
echo "======================================================"
python3 analyze_man_results.py --input_dir "$OUTPUT_DIR" --sigma "$SIGMA"

echo ""
echo "======================================================"
echo " Done. Results in: $OUTPUT_DIR"
echo "======================================================"
