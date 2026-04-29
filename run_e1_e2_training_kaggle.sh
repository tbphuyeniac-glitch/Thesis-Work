#!/bin/bash
# =====================================================================
# Train both GNN variants in PARALLEL on Kaggle T4×2.
# =====================================================================
# Variant 1 (E1, GPU 0):  rc-only feature mask, label_mode=rc_top_in_group
# Variant 2 (E2, GPU 1):  full features, label_mode=selected_in_rmp
#
# Each invocation pins to a specific GPU via CUDA_VISIBLE_DEVICES.
# Both run in background; `wait` blocks until both finish.
# =====================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/kaggle/working/Thesis-Work}"
E1_TEACHER_CSV="${E1_TEACHER_CSV:-${REPO_ROOT}/Results_E1_no_penalty/scenarios/aggregate_teacher_rows.csv}"
E1_MANIFEST="${E1_MANIFEST:-${REPO_ROOT}/Results_E1_no_penalty/scenarios/scenarios_manifest.json}"
E2_TEACHER_CSV="${E2_TEACHER_CSV:-${REPO_ROOT}/Results_E2_with_penalty/scenarios/aggregate_teacher_rows.csv}"
E2_MANIFEST="${E2_MANIFEST:-${REPO_ROOT}/Results_E2_with_penalty/scenarios/scenarios_manifest.json}"

OUT_E1="${REPO_ROOT}/Results_E1_no_penalty/gnn_training"
OUT_E2="${REPO_ROOT}/Results_E2_with_penalty/gnn_training"
MAX_EPOCHS="${MAX_EPOCHS:-300}"

mkdir -p "${OUT_E1}" "${OUT_E2}"

echo "[parallel-train] E1 (rc-only, GPU 0) and E2 (multi-feature, GPU 1) starting"

# NEW v2 EXPERIMENT CONFIG
# Both variants use IDENTICAL training recipe except for input feature mask
# (clean ablation per user requirement #4):
#   --objective pairwise_rank           — softplus over column pairs (graded)
#   --ranking-target teacher_score      — λ·max(0, -rc) under penalty regime
#   --label-mode selected_in_rmp        — same binary labels (legacy default)
# Only differs:
#   E1 (rc-only):    --column-feature-mask rc_only
#   E2 (multi):      --column-feature-mask all
# Override below with E1_RANKING_TARGET / E2_RANKING_TARGET if needed.

OBJECTIVE="${OBJECTIVE:-pairwise_rank}"
RANKING_TARGET="${RANKING_TARGET:-teacher_score}"
RANK_K_VALUES="${RANK_K_VALUES:-1,3,5,10}"
LABEL_MODE="${LABEL_MODE:-selected_in_rmp}"

# ── Variant E1: rc-only on GPU 0 ─────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python -u "${REPO_ROOT}/GNN/train_bipat_from_aggregate.py" \
    --teacher-csv "${E1_TEACHER_CSV}" \
    --manifest    "${E1_MANIFEST}" \
    --out-dir     "${OUT_E1}" \
    --max-epochs  "${MAX_EPOCHS}" \
    --rows-per-epoch 5000 \
    --valid-rows-per-epoch 2000 \
    --max-rows-per-source-instance-per-epoch 200 \
    --batch-size 128 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --early-stopping-patience 30 \
    --full-valid-every 5 \
    --seed 42 \
    --hidden-dim 64 \
    --dropout 0.1 \
    --objective "${OBJECTIVE}" \
    --ranking-target "${RANKING_TARGET}" \
    --rank-k-values "${RANK_K_VALUES}" \
    --label-mode "${LABEL_MODE}" \
    --column-feature-mask rc_only \
    --device cuda \
    > "${OUT_E1}/train_stdout.log" 2>&1 &
PID_E1=$!
echo "[parallel-train] E1 (rc-only) PID=${PID_E1}"

# ── Variant E2: multi-feature on GPU 1 ───────────────────────────────
CUDA_VISIBLE_DEVICES=1 python -u "${REPO_ROOT}/GNN/train_bipat_from_aggregate.py" \
    --teacher-csv "${E2_TEACHER_CSV}" \
    --manifest    "${E2_MANIFEST}" \
    --out-dir     "${OUT_E2}" \
    --max-epochs  "${MAX_EPOCHS}" \
    --rows-per-epoch 5000 \
    --valid-rows-per-epoch 2000 \
    --max-rows-per-source-instance-per-epoch 200 \
    --batch-size 128 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --early-stopping-patience 30 \
    --full-valid-every 5 \
    --seed 42 \
    --hidden-dim 64 \
    --dropout 0.1 \
    --objective "${OBJECTIVE}" \
    --ranking-target "${RANKING_TARGET}" \
    --rank-k-values "${RANK_K_VALUES}" \
    --label-mode "${LABEL_MODE}" \
    --column-feature-mask all \
    --device cuda \
    > "${OUT_E2}/train_stdout.log" 2>&1 &
PID_E2=$!
echo "[parallel-train] E2 (multi-feature) PID=${PID_E2}"

# Wait for both. If one fails, propagate exit code so the Kaggle cell errors.
echo "[parallel-train] waiting for both jobs to finish..."
FAIL=0
wait ${PID_E1} || { echo "[parallel-train] E1 FAILED (PID ${PID_E1})"; FAIL=1; }
wait ${PID_E2} || { echo "[parallel-train] E2 FAILED (PID ${PID_E2})"; FAIL=1; }

if [[ ${FAIL} -ne 0 ]]; then
    echo "[parallel-train] one or both variants failed — see *_stdout.log"
    exit ${FAIL}
fi
echo "[parallel-train] BOTH variants finished. checkpoints:"
ls -la "${OUT_E1}/" | head -10
ls -la "${OUT_E2}/" | head -10
