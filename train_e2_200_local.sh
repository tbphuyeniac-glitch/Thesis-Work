#!/usr/bin/env bash
# Launches 200-epoch E2 BiGAT training in background, survives terminal close + sleep.
# Mirrors train_e1_200_local.sh but consumes the E2 multi-feature filtered dataset
# (irplt_teacher_E2_filtered) and writes the checkpoint to a separate output dir
# so the E1 and E2 runs do NOT clobber each other when launched in parallel.
set -e
cd "/Users/trannguyenhung/Documents/THESIS/Code/Current Code"

OUT_DIR="GNN/trained_models/irplt_teacher_E2_filtered_local200_fixed/bigat/pairwise_rank"
LOG_FILE="/tmp/e2_train_200_fixed.log"

mkdir -p "$OUT_DIR"

nohup caffeinate -i python3 GNN/03_train_bigat.py \
  --data-dir GNN/data/irplt_teacher_E2_filtered \
  --out-dir "$OUT_DIR" \
  --dataset-type teacher \
  --epochs 200 \
  --device cpu \
  --objective pairwise_rank \
  --seed 0 \
  --hidden-dim 16 \
  --lr 1e-3 \
  --dropout 0.1 \
  --patience 1000000 \
  --auto-resume \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started (E2)"
echo "  PID:        $PID"
echo "  Log file:   $LOG_FILE"
echo "  Output dir: $OUT_DIR"
echo ""
echo "Monitor:   tail -f $LOG_FILE"
echo "Kill:      kill $PID"
echo "Status:    ps -p $PID"
