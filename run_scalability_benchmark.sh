#!/usr/bin/env bash
# run_scalability_benchmark.sh
#
# Runs Gurobi MIP vs ALNS vs Achamrah GA/SA scalability benchmark across
# 6 thesis scenarios. Each scenario uses the first 60 distinct periods.
#
# Solvers:
#   Gurobi MIP    — exact, no time limit
#   ALNS          — 1000 iterations, no time limit
#   Achamrah GA/SA — Phase 1 RMILP + Phase 2 GA+SA, tier-based time limits:
#                    small=300s  medium=600s  large=1200s
#
# Scenarios:
#   small_4s2p   : stores≤4,  skus≤2
#   small_5s2p   : stores≤5,  skus≤2
#   medium_7s3p  : stores≤7,  skus≤3
#   medium_7s4p  : stores≤7,  skus≤4
#   large_10s5p  : stores≤10, skus≤5
#   large_12s5p  : stores≤12, skus≤5
#
# Output lands in Results/benchmark/scalability/<scenario>/ and
# Results/benchmark/solver_scalability_summary.csv.
#
# Usage:
#   bash run_scalability_benchmark.sh
#   bash run_scalability_benchmark.sh --replications 3    # optional: 3 reps per scenario
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="$SCRIPT_DIR/1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
MAIN_PY="$SCRIPT_DIR/irp_gurobi_converted.py"

# Default: 1 replication per scenario (fastest — set to 3 for error-bar charts)
REPLICATIONS=1

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --replications|-r)
            REPLICATIONS="$2"; shift 2 ;;
        --dataset|-d)
            DATASET="$2"; shift 2 ;;
        *)
            echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: dataset not found: $DATASET"
    exit 1
fi
if [[ ! -f "$MAIN_PY" ]]; then
    echo "ERROR: irp_gurobi_converted.py not found: $MAIN_PY"
    exit 1
fi

echo "============================================================"
echo "  IRP Solver Scalability Benchmark"
echo "  Dataset      : $DATASET"
echo "  Replications : $REPLICATIONS"
echo "  Periods      : first 60 distinct dates"
echo "  Solvers      : Gurobi MIP (exact, no TL)  +  ALNS (1000 iter, no TL)"
echo "============================================================"
echo ""

IRP_RUN_SCALABILITY_BENCHMARK=1 \
IRP_DATASET_PATH="$DATASET" \
IRP_SCALABILITY_PERIOD_COUNT=60 \
IRP_SCALABILITY_REPLICATIONS="$REPLICATIONS" \
IRP_SCALABILITY_SEED_BASE=42 \
IRP_CLEAN_RESULTS=0 \
python3 "$MAIN_PY"

echo ""
echo "============================================================"
echo "  Benchmark complete."
echo "  Summary CSV : Results/benchmark/solver_scalability_summary.csv"
echo "  Summary chart: Results/charts/solver_scalability.png"
echo "  Per-scenario : Results/benchmark/scalability/<scenario>/"
echo "============================================================"
