"""Generate a richer multi-panel training chart from training_history.json.

Usage:
    python3 GNN/plot_training_metrics.py \
        --history GNN/trained_models/irplt_teacher_filtered_local50/bigat/pairwise_rank/training_history.json
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="Path to training_history.json")
    ap.add_argument("--out", default=None, help="Optional output png path (default: same dir / training_metrics_panels.png)")
    args = ap.parse_args()

    hist_path = Path(args.history)
    if not hist_path.exists():
        raise SystemExit(f"history not found: {hist_path}")
    history = json.loads(hist_path.read_text())
    if not history:
        raise SystemExit("empty history")
    out_path = Path(args.out) if args.out else hist_path.with_name("training_metrics_panels.png")

    epochs = [r["epoch"] for r in history]
    train_loss = [r.get("train_loss", float("nan")) for r in history]
    valid_loss = [r.get("valid_loss", float("nan")) for r in history]
    mrr = [r.get("ranking_valid_mrr", float("nan")) for r in history]
    top1 = [r.get("ranking_valid_top1", float("nan")) for r in history]
    top3 = [r.get("ranking_valid_top3", float("nan")) for r in history]
    top5 = [r.get("ranking_valid_top5", float("nan")) for r in history]
    pos_score = [r.get("pos_score", float("nan")) for r in history]
    neg_score = [r.get("neg_score", float("nan")) for r in history]
    gap = [r.get("score_gap", float("nan")) for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Panel 1: loss curves
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, marker="o", label="Train loss")
    ax.plot(epochs, valid_loss, marker="s", label="Valid loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Train / Valid Loss")
    ax.grid(True, alpha=0.3); ax.legend()

    # Panel 2: MRR
    ax = axes[0, 1]
    ax.plot(epochs, mrr, marker="o", color="tab:green", label="Valid MRR")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MRR"); ax.set_title("Mean Reciprocal Rank (higher = better)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3); ax.legend()

    # Panel 3: top-k accuracy
    ax = axes[1, 0]
    ax.plot(epochs, top1, marker="o", label="Top-1")
    ax.plot(epochs, top3, marker="s", label="Top-3")
    ax.plot(epochs, top5, marker="^", label="Top-5")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Hit rate"); ax.set_title("Top-k Hit Rate (higher = better)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3); ax.legend()

    # Panel 4: pos/neg score and gap
    ax = axes[1, 1]
    ax.plot(epochs, pos_score, marker="o", label="Pos score")
    ax.plot(epochs, neg_score, marker="s", label="Neg score")
    ax.plot(epochs, gap, marker="^", color="tab:red", label="Score gap (pos-neg)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Discriminative Score Gap")
    ax.grid(True, alpha=0.3); ax.legend()

    fig.suptitle(f"BiGAT training metrics — {hist_path.parent.name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
