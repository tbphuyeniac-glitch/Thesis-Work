#!/usr/bin/env bash
# Launches 200-epoch E1 BiGAT training in background, survives terminal close + sleep.
set -e
cd "/Users/trannguyenhung/Documents/THESIS/Code/Current Code"

OUT_DIR="GNN/trained_models/irplt_teacher_filtered_local200_fixed/bigat/pairwise_rank"
LOG_FILE="/tmp/e1_train_200_fixed.log"

mkdir -p "$OUT_DIR"

nohup caffeinate -i python3 GNN/03_train_bigat.py \
  --data-dir GNN/data/irplt_teacher_filtered \
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
echo "Training started"
echo "  PID:        $PID"
echo "  Log file:   $LOG_FILE"
echo "  Output dir: $OUT_DIR"
echo ""
echo "Monitor:   tail -f $LOG_FILE"
echo "Kill:      kill $PID"
echo "Status:    ps -p $PID"
