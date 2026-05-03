from __future__ import annotations
"""
GNN/eval_dantzig_vs_gnn.py — Within-group ranking comparison: Dantzig vs BiGAT.

For each (p, t) test group with K columns and >=1 positive label:
  - Dantzig MRR : rank columns by reduced_cost ASCENDING (most negative first),
                  MRR = 1 / (rank of best positive).
  - GNN MRR    : rank columns by BiGAT logits DESCENDING.
  - Spearman   : within-group Spearman(-RC, label) — proxy for how strongly
                 the Dantzig signal aligns with the teacher label.

Hard groups are those where Spearman <= --hard-rho-threshold. The GNN
contribution claim is meaningful only if GNN beats Dantzig MRR overall, AND
especially on hard groups where -RC alone is uninformative.

CLI
---
--data-dir, --split, --checkpoint, --hard-rho-threshold, --out-file
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import utilities
from models.attention.model import BiGATColumnScorer


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _mrr_from_order(order: np.ndarray, labels_bin: np.ndarray) -> float:
    """Return 1/rank of the first positive in `order`. NaN if no positive."""
    for rank, idx in enumerate(order, start=1):
        if labels_bin[idx] > 0.5:
            return 1.0 / float(rank)
    return float("nan")


def _load_checkpoint(path: Path, device: str):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    config = dict(ckpt["config"])
    dropout = config.pop("dropout", 0.0)
    model = BiGATColumnScorer(**config, dropout=dropout).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Within-group Dantzig vs GNN MRR comparison.")
    parser.add_argument("--data-dir", default="GNN/data/irplt_teacher_E2_filtered")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument(
        "--checkpoint",
        default="GNN/trained_models/irplt_teacher_E2_filtered_3epochs/bigat/pairwise_rank/best_model.pt",
    )
    parser.add_argument("--hard-rho-threshold", type=float, default=0.3,
                        help="Groups with Spearman(-RC,label) <= this are 'hard'.")
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    raw_paths = utilities.iter_sample_files(args.data_dir, args.split)
    if not raw_paths:
        raise RuntimeError(f"no {args.split} samples in {args.data_dir}")

    # Load tensor samples; apply the checkpoint's training-time normalization to
    # the GNN-bound features. Dantzig metrics are rank-only and use the raw RC
    # values directly from the unnormalized payload.
    raw_samples = [utilities.load_raw_sample(p) for p in raw_paths]
    tensor_samples = [utilities.graph_to_tensors(s) for s in raw_samples]
    model, ckpt = _load_checkpoint(Path(args.checkpoint), args.device)
    norm_samples, _ = utilities.normalize_dataset(tensor_samples, stats=ckpt["normalization"])

    rc_idx = utilities.COLUMN_FEATURE_NAMES.index("reduced_cost")

    rows: List[Dict] = []
    skipped_no_positive = 0
    skipped_singleton = 0
    for raw, normed in zip(raw_samples, norm_samples):
        # Apply legacy 12->10 slice symmetrically to the raw (unnormalized)
        # column matrix used for Dantzig — the loader does it for the tensor
        # path but here we pull from the raw dict directly.
        col = np.asarray(raw["column_features"], dtype=np.float64)
        target_dim = len(utilities.COLUMN_FEATURE_NAMES)
        legacy_dim = target_dim + len(utilities._LEGACY_COLUMN_FEATURE_INDICES_TO_DROP)
        if col.shape[1] == legacy_dim:
            keep = [i for i in range(legacy_dim) if i not in set(utilities._LEGACY_COLUMN_FEATURE_INDICES_TO_DROP)]
            col = col[:, keep]
        labels_bin = (np.asarray(raw["labels"], dtype=np.float64) > 0.5).astype(np.float64)

        if col.shape[0] < 2:
            skipped_singleton += 1
            continue
        if labels_bin.sum() < 1:
            skipped_no_positive += 1
            continue

        rc = col[:, rc_idx]
        # Dantzig: most-negative RC first => ASCENDING sort.
        dantzig_order = np.argsort(rc, kind="stable")
        dantzig_mrr = _mrr_from_order(dantzig_order, labels_bin)

        # GNN: highest logit first => DESCENDING sort.
        with torch.no_grad():
            graph = {k: v.to(args.device) for k, v in normed.items()}
            logits = model(
                graph["column_features"],
                graph["constraint_features"],
                graph["edge_index_col_to_con"],
                graph["edge_attr_col_to_con"],
            ).detach().cpu().numpy()
        gnn_order = np.argsort(-logits, kind="stable")
        gnn_mrr = _mrr_from_order(gnn_order, labels_bin)

        # Spearman(-RC, label): high when low-RC columns are positive.
        rho = _spearman_rho(-rc, labels_bin)

        rows.append({
            "k": int(col.shape[0]),
            "n_positive": int(labels_bin.sum()),
            "rc_min": float(rc.min()),
            "rc_max": float(rc.max()),
            "spearman_neg_rc_label": float(rho) if not np.isnan(rho) else None,
            "dantzig_mrr": float(dantzig_mrr),
            "gnn_mrr": float(gnn_mrr),
        })

    n_eval = len(rows)
    if n_eval == 0:
        raise RuntimeError("no eligible (size>=2, has_positive) groups in split")

    dantzig_mrrs = np.array([r["dantzig_mrr"] for r in rows])
    gnn_mrrs = np.array([r["gnn_mrr"] for r in rows])
    rho_values = np.array([r["spearman_neg_rc_label"] for r in rows], dtype=np.float64)
    rho_nan_mask = np.isnan(rho_values)
    # Treat ill-defined Spearman (constant labels — should not happen post-filter)
    # as 0.0 so a hard-group threshold still classifies them as hard.
    rho_for_thresh = np.where(rho_nan_mask, 0.0, rho_values)
    hard_mask = rho_for_thresh <= args.hard_rho_threshold

    summary = {
        "data_dir": args.data_dir,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "n_groups_in_split": len(raw_samples),
        "n_groups_evaluated": n_eval,
        "skipped_singleton": skipped_singleton,
        "skipped_no_positive": skipped_no_positive,
        "hard_rho_threshold": args.hard_rho_threshold,
        "n_hard_groups": int(hard_mask.sum()),
        "n_easy_groups": int((~hard_mask).sum()),
        "full_test": {
            "dantzig_mrr_mean": float(dantzig_mrrs.mean()),
            "gnn_mrr_mean": float(gnn_mrrs.mean()),
            "gnn_minus_dantzig": float(gnn_mrrs.mean() - dantzig_mrrs.mean()),
            "gnn_beats_dantzig_pct": float((gnn_mrrs > dantzig_mrrs).mean()),
            "tied_pct": float((gnn_mrrs == dantzig_mrrs).mean()),
            "dantzig_beats_gnn_pct": float((gnn_mrrs < dantzig_mrrs).mean()),
        },
    }
    if hard_mask.any():
        summary["hard_groups"] = {
            "dantzig_mrr_mean": float(dantzig_mrrs[hard_mask].mean()),
            "gnn_mrr_mean": float(gnn_mrrs[hard_mask].mean()),
            "gnn_minus_dantzig": float(gnn_mrrs[hard_mask].mean() - dantzig_mrrs[hard_mask].mean()),
            "gnn_beats_dantzig_pct": float((gnn_mrrs[hard_mask] > dantzig_mrrs[hard_mask]).mean()),
        }
    if (~hard_mask).any():
        summary["easy_groups"] = {
            "dantzig_mrr_mean": float(dantzig_mrrs[~hard_mask].mean()),
            "gnn_mrr_mean": float(gnn_mrrs[~hard_mask].mean()),
            "gnn_minus_dantzig": float(gnn_mrrs[~hard_mask].mean() - dantzig_mrrs[~hard_mask].mean()),
            "gnn_beats_dantzig_pct": float((gnn_mrrs[~hard_mask] > dantzig_mrrs[~hard_mask]).mean()),
        }

    print(json.dumps(summary, indent=2))
    if args.out_file:
        out = Path(args.out_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_group": rows}, f, indent=2)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
