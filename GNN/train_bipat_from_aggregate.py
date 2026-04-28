from __future__ import annotations

"""
GNN/train_bipat_from_aggregate.py
=================================
Train BiGAT directly from `aggregate_teacher_rows.csv` with controlled dynamic
sampling and curriculum learning.

Key differences vs `03_train_bigat.py`:
- No pre-built .pkl graph dataset on disk — graphs are reconstructed in-memory
  from the aggregate CSV at startup, grouped by
  (source_instance, branch_node_id, episode, product, period, constraint_state).
- Each epoch samples a controlled, balanced subset of teacher rows
  (default 5,000) with diversity quotas across source_instance, shock_profile,
  store_limit, sku_limit. Loss is computed only on sampled columns even though
  the full graph is forwarded for context.
- Curriculum learning: easy pool (epochs 1–30) → full (31–120) → hard (121+).

CLI
---
python GNN/train_bipat_from_aggregate.py \
  --teacher-csv /kaggle/working/Results/scenarios/aggregate_teacher_rows.csv \
  --manifest    /kaggle/working/Results/scenarios/scenarios_manifest.json \
  --out-dir     /kaggle/working/Results/gnn_training \
  --rows-per-epoch 5000 \
  --valid-rows-per-epoch 2000 \
  --max-epochs 200 \
  --batch-size 128 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --early-stopping-patience 20 \
  --full-valid-every 10 \
  --seed 42
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import utilities  # noqa: E402  — imported for sample builder + feature constants
from models.attention.model import BiGATColumnScorer  # noqa: E402

try:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# =====================================================================
# Constants
# =====================================================================
GroupKey = Tuple[str, str, str, str, str, str]  # matches build_teacher_graph_dataset
EASY_END_EPOCH = 30
HARD_START_EPOCH = 121


# =====================================================================
# I/O helpers
# =====================================================================

def _atomic_torch_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _atomic_json_save(obj: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _raise_csv_field_size_limit() -> None:
    cur = sys.maxsize
    while True:
        try:
            csv.field_size_limit(cur)
            break
        except OverflowError:
            cur //= 10


def _load_aggregate_rows(path: Path) -> List[Dict[str, str]]:
    _raise_csv_field_size_limit()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _group_key(row: Dict[str, Any]) -> GroupKey:
    """Match the GroupKey logic from `build_teacher_graph_dataset.py`."""
    import hashlib

    source_instance = str(row.get("source_instance") or row.get("instance_id") or "default")
    branch_node = str(row.get("branch_node_id") or row.get("node_id") or "root")
    episode = str(row.get("episode") or row.get("episode_id") or "0")
    product = str(row.get("product") or row.get("sku") or "unknown_product")
    period = str(row.get("period") or row.get("time_period") or "unknown_period")
    stored_hash = str(row.get("constraint_state_hash") or "").strip()
    if stored_hash:
        constraint_state = stored_hash
    else:
        constraint_json = str(row.get("constraint_features_json") or "")
        constraint_state = (
            hashlib.sha1(constraint_json.encode("utf-8")).hexdigest()[:12]
            if constraint_json
            else "no_constraints"
        )
    return source_instance, branch_node, episode, product, period, constraint_state


def _propagate_constraint_features(rows: List[Dict[str, Any]]) -> None:
    """Fill empty `constraint_features_json` from first non-empty in same group."""
    first_by_group: Dict[GroupKey, str] = {}
    for row in rows:
        key = _group_key(row)
        value = str(row.get("constraint_features_json") or "").strip()
        if value and key not in first_by_group:
            first_by_group[key] = value
    for row in rows:
        if not str(row.get("constraint_features_json") or "").strip():
            row["constraint_features_json"] = first_by_group.get(_group_key(row), "")


# =====================================================================
# Manifest backfill
# =====================================================================

def _build_manifest_lookup(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """source_instance -> {split, store_limit, sku_limit, shock_profile, ...}."""
    lookup: Dict[str, Dict[str, Any]] = {}
    split_assignment: Dict[str, str] = manifest.get("split_assignment", {}) or {}
    for sc in manifest.get("scenarios", []):
        sid = str(sc.get("source_instance") or sc.get("scenario_id") or "")
        if not sid:
            continue
        lookup[sid] = {
            "split": split_assignment.get(sid, sc.get("split", "train")),
            "store_limit": sc.get("store_limit"),
            "sku_limit": sc.get("sku_limit"),
            "shock_profile": sc.get("shock_profile"),
            "base_dataset_id": sc.get("base_dataset_id"),
            "shock_seed": sc.get("shock_seed"),
        }
    return lookup


def _coerce_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "" or str(v).strip().lower() in {"none", "nan"}:
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _backfill_metadata(
    rows: List[Dict[str, str]],
    manifest_lookup: Dict[str, Dict[str, Any]],
) -> None:
    """In-place fill of store_limit / sku_limit / shock_profile / split."""
    for row in rows:
        sid = str(row.get("source_instance") or "")
        info = manifest_lookup.get(sid, {})
        for key in ("store_limit", "sku_limit", "shock_profile", "split", "base_dataset_id"):
            cur = str(row.get(key) or "").strip()
            if not cur or cur.lower() in {"none", "nan"}:
                row[key] = info.get(key, "") or ""


# =====================================================================
# Graph sample construction
# =====================================================================

def _build_graphs(rows: List[Dict[str, Any]]) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Group rows by GroupKey and build one in-memory graph sample per group.

    Returns:
        graphs: list of {column_features, constraint_features, edges, ..., metadata}
                length = number of groups
        row_index: list of {row_id, graph_id, col_idx_in_graph, label, metadata...}
                   length = number of rows that successfully landed in a graph
    """
    grouped: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    graphs: List[Dict[str, Any]] = []
    row_index: List[Dict[str, Any]] = []
    skipped_groups = 0

    for key, group_rows in grouped.items():
        # Sort within group so column ordering is deterministic
        try:
            group_rows.sort(key=lambda r: (
                int(float(r.get("column_index") or 10**9)),
                str(r.get("pattern_id", "")),
            ))
        except (TypeError, ValueError):
            group_rows.sort(key=lambda r: str(r.get("pattern_id", "")))

        try:
            sample = utilities.build_training_sample_from_exported_teacher_rows(
                group_rows,
                episode_id=key[2],
                source_instance=key[0],
                product=key[3],
                period=key[4],
                branch_node_id=key[1],
                decision_state_id=key[5],
            )
        except Exception:
            skipped_groups += 1
            continue

        graph_id = len(graphs)
        graphs.append(sample)

        # Record row → (graph_id, col_idx). We use the same sort order used above.
        for col_idx, r in enumerate(group_rows):
            try:
                label_val = float(r.get("teacher_label") or 0.0)
            except (TypeError, ValueError):
                label_val = 0.0
            row_index.append({
                "graph_id": graph_id,
                "col_idx": col_idx,
                "label": int(label_val > 0.5),
                "source_instance": str(r.get("source_instance") or ""),
                "shock_profile": str(r.get("shock_profile") or ""),
                "store_limit": _coerce_int(r.get("store_limit")) or 0,
                "sku_limit": _coerce_int(r.get("sku_limit")) or 0,
                "split": str(r.get("split") or ""),
            })

    if skipped_groups:
        print(f"[graphs] skipped {skipped_groups}/{len(grouped)} groups due to "
              f"missing/invalid features (typical: blank constraint_features_json).")

    return graphs, row_index


def _normalize_graphs(
    graphs: List[Dict[str, Any]],
    train_graph_ids: List[int],
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute / apply per-feature standardization using only TRAIN graphs.

    Mutates `graphs` in place; returns the stats dict.
    """
    if stats is None:
        col_stack = np.concatenate(
            [graphs[gid]["column_features"] for gid in train_graph_ids if graphs[gid]["column_features"].size],
            axis=0,
        ) if train_graph_ids else np.zeros((0, 1), dtype=np.float32)
        con_stack = np.concatenate(
            [graphs[gid]["constraint_features"] for gid in train_graph_ids if graphs[gid]["constraint_features"].size],
            axis=0,
        ) if train_graph_ids else np.zeros((0, 1), dtype=np.float32)
        edge_stack = np.concatenate(
            [graphs[gid]["edge_attr_col_to_con"] for gid in train_graph_ids if graphs[gid]["edge_attr_col_to_con"].size],
            axis=0,
        ) if train_graph_ids else np.zeros((0, 1), dtype=np.float32)

        def _stat(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if arr.size == 0:
                return np.zeros(arr.shape[-1:], dtype=np.float32), np.ones(arr.shape[-1:], dtype=np.float32)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            return mean.astype(np.float32), std.astype(np.float32)

        col_mean, col_std = _stat(col_stack)
        con_mean, con_std = _stat(con_stack)
        edge_mean, edge_std = _stat(edge_stack)
        stats = {
            "column_mean": col_mean.tolist(),
            "column_std": col_std.tolist(),
            "constraint_mean": con_mean.tolist(),
            "constraint_std": con_std.tolist(),
            "edge_mean": edge_mean.tolist(),
            "edge_std": edge_std.tolist(),
        }

    col_mean = np.asarray(stats["column_mean"], dtype=np.float32)
    col_std = np.asarray(stats["column_std"], dtype=np.float32)
    con_mean = np.asarray(stats["constraint_mean"], dtype=np.float32)
    con_std = np.asarray(stats["constraint_std"], dtype=np.float32)
    edge_mean = np.asarray(stats["edge_mean"], dtype=np.float32)
    edge_std = np.asarray(stats["edge_std"], dtype=np.float32)

    for g in graphs:
        if g["column_features"].size:
            g["column_features"] = (g["column_features"] - col_mean) / col_std
        if g["constraint_features"].size:
            g["constraint_features"] = (g["constraint_features"] - con_mean) / con_std
        if g["edge_attr_col_to_con"].size:
            g["edge_attr_col_to_con"] = (g["edge_attr_col_to_con"] - edge_mean) / edge_std

    return stats


# =====================================================================
# Controlled dynamic sampler
# =====================================================================

def _sample_controlled(
    pool: pd.DataFrame,
    n_total: int,
    *,
    rng: random.Random,
    max_per_source_instance: int = 200,
    desired_pos_ratio: float = 0.5,
) -> Tuple[List[int], Dict[str, Any]]:
    """Sample up to n_total row-indices from `pool` (a slice of row_index) with
    label-balance + per-source-instance cap + diversity logging.

    Diversity is enforced by drawing per-source caps; diversity ACROSS
    shock_profile / store_limit / sku_limit is preserved implicitly because
    capping per source_instance prevents any one bucket from dominating.

    Returns
    -------
    indices : List[int]   row-indices into `pool` original index
    stats   : Dict        per-bucket counts for logging
    """
    if pool.empty:
        return [], {"pos": 0, "neg": 0, "by_source_instance": {}}

    n_pos_target = int(round(n_total * desired_pos_ratio))
    n_neg_target = n_total - n_pos_target

    pos_pool = pool[pool["label"] == 1]
    neg_pool = pool[pool["label"] == 0]

    def _draw_with_cap(sub: pd.DataFrame, k: int) -> List[int]:
        if sub.empty or k <= 0:
            return []
        # Group by source_instance, then sample with per-instance cap to ensure diversity
        by_inst = list(sub.groupby("source_instance").indices.items())
        rng.shuffle(by_inst)
        chosen: List[int] = []
        # Round 1: at most max_per_source_instance per source
        for inst, idx_arr in by_inst:
            if len(chosen) >= k:
                break
            n_take = min(len(idx_arr), max_per_source_instance, k - len(chosen))
            picks = rng.sample(list(idx_arr), n_take)
            chosen.extend(picks)
        # Round 2: if still short, lift the per-source cap (with replacement avoided)
        if len(chosen) < k:
            remaining = list(set(sub.index.tolist()) - set(chosen))
            rng.shuffle(remaining)
            chosen.extend(remaining[: k - len(chosen)])
        # Round 3: still short → sample with replacement (rare; tiny pools)
        if len(chosen) < k and chosen:
            extra = [rng.choice(chosen) for _ in range(k - len(chosen))]
            chosen.extend(extra)
        return chosen

    pos_idx = _draw_with_cap(pos_pool, n_pos_target)
    neg_idx = _draw_with_cap(neg_pool, n_neg_target)

    # If one side ran out, top-up from the other
    if len(pos_idx) + len(neg_idx) < n_total:
        deficit = n_total - len(pos_idx) - len(neg_idx)
        if not pos_pool.empty and len(neg_idx) < n_neg_target:
            extra = _draw_with_cap(pos_pool, deficit)
            pos_idx.extend(extra)
        elif not neg_pool.empty:
            extra = _draw_with_cap(neg_pool, deficit)
            neg_idx.extend(extra)

    chosen = pos_idx + neg_idx
    rng.shuffle(chosen)

    stats = {
        "pos": int(sum(1 for i in chosen if pool.loc[i, "label"] == 1)),
        "neg": int(sum(1 for i in chosen if pool.loc[i, "label"] == 0)),
        "by_source_instance": dict(Counter(pool.loc[chosen, "source_instance"]).most_common(10)),
        "by_shock_profile": dict(Counter(pool.loc[chosen, "shock_profile"]).most_common()),
        "by_store_limit": {int(k): int(v) for k, v in Counter(pool.loc[chosen, "store_limit"]).items()},
        "by_sku_limit": {int(k): int(v) for k, v in Counter(pool.loc[chosen, "sku_limit"]).items()},
        "n_unique_source_instances": int(pool.loc[chosen, "source_instance"].nunique()),
    }
    return chosen, stats


def _curriculum_pool(
    train_df: pd.DataFrame,
    epoch: int,
) -> Tuple[pd.DataFrame, str]:
    """Return (subset_dataframe, regime_name) for the given epoch."""
    if epoch <= EASY_END_EPOCH:
        easy = train_df[(train_df["store_limit"] <= 5) & (train_df["sku_limit"] <= 3)]
        if len(easy) >= 100:
            return easy, "easy"
        return train_df, "easy_fallback_full"
    if epoch < HARD_START_EPOCH:
        return train_df, "full"
    # epoch >= 121 → 50% hard pool, 50% full
    hard = train_df[(train_df["store_limit"] >= 7) | (train_df["sku_limit"] == 4)]
    if hard.empty:
        return train_df, "hard_fallback_full"
    return train_df, "mixed_hard_full"  # mixing happens at sample-time below


def _sample_with_curriculum(
    train_df: pd.DataFrame,
    n_total: int,
    epoch: int,
    rng: random.Random,
    max_per_source_instance: int,
) -> Tuple[List[int], Dict[str, Any]]:
    pool, regime = _curriculum_pool(train_df, epoch)
    if regime == "mixed_hard_full":
        hard = train_df[(train_df["store_limit"] >= 7) | (train_df["sku_limit"] == 4)]
        n_hard = n_total // 2
        idx_hard, stats_hard = _sample_controlled(
            hard, n_hard, rng=rng, max_per_source_instance=max_per_source_instance,
        )
        idx_full, stats_full = _sample_controlled(
            train_df, n_total - len(idx_hard), rng=rng,
            max_per_source_instance=max_per_source_instance,
        )
        idx = idx_hard + idx_full
        stats = {
            "regime": regime,
            "hard_pool_size": int(len(hard)),
            "full_pool_size": int(len(train_df)),
            "pos": stats_hard["pos"] + stats_full["pos"],
            "neg": stats_hard["neg"] + stats_full["neg"],
            "by_source_instance_top": dict(
                Counter([*stats_hard["by_source_instance"].keys(),
                         *stats_full["by_source_instance"].keys()]).most_common(10)
            ),
            "by_shock_profile": stats_full.get("by_shock_profile", {}),
            "by_store_limit": stats_full.get("by_store_limit", {}),
            "by_sku_limit": stats_full.get("by_sku_limit", {}),
            "n_unique_source_instances": int(train_df.loc[idx, "source_instance"].nunique()) if idx else 0,
        }
        return idx, stats

    idx, stats = _sample_controlled(
        pool, n_total, rng=rng, max_per_source_instance=max_per_source_instance,
    )
    stats["regime"] = regime
    stats["pool_size"] = int(len(pool))
    return idx, stats


# =====================================================================
# Forward pass + masked loss
# =====================================================================

def _to_tensor_graph(graph: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    """Convert a graph sample (numpy arrays) to torch tensors on device."""
    return {
        "column_features": torch.from_numpy(graph["column_features"]).float().to(device),
        "constraint_features": torch.from_numpy(graph["constraint_features"]).float().to(device),
        "edge_index_col_to_con": torch.from_numpy(graph["edge_index_col_to_con"]).long().to(device),
        "edge_attr_col_to_con": torch.from_numpy(graph["edge_attr_col_to_con"]).float().to(device),
        "labels": torch.from_numpy(graph["labels_binary"]).float().to(device),
    }


def _forward_batched(
    model: torch.nn.Module,
    graphs: List[Dict[str, Any]],
    sampled_rows_df: pd.DataFrame,
    device: str,
    *,
    training: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Forward sampled rows grouped by graph_id, returning (loss, probs, labels).

    Loss is BCEWithLogitsLoss computed only on the columns selected by
    `sampled_rows_df` within each graph.
    """
    model.train(training)
    total_loss = 0.0
    total_n = 0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    by_graph = sampled_rows_df.groupby("graph_id", sort=False)
    if optimizer is not None and training:
        optimizer.zero_grad()

    for gid, sub in by_graph:
        graph = graphs[int(gid)]
        cols = sub["col_idx"].astype(int).values
        labels = torch.from_numpy(sub["label"].astype(np.float32).values).to(device)
        n = labels.numel()

        gtens = _to_tensor_graph(graph, device)
        logits_all = model(
            gtens["column_features"],
            gtens["constraint_features"],
            gtens["edge_index_col_to_con"],
            gtens["edge_attr_col_to_con"],
        )
        # Some graphs may be empty post-skip; guard
        if logits_all.numel() == 0:
            continue
        logits = logits_all[torch.from_numpy(cols).long().to(device)]
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

        if training and optimizer is not None:
            (loss * (n / max(1, len(sampled_rows_df)))).backward()

        total_loss += float(loss.detach().cpu()) * n
        total_n += n
        with torch.no_grad():
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    if training and optimizer is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    mean_loss = total_loss / max(1, total_n)
    probs_arr = np.concatenate(all_probs) if all_probs else np.zeros(0, dtype=np.float32)
    labels_arr = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.float32)
    return mean_loss, probs_arr, labels_arr


def _binary_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if probs.size == 0:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan")}
    preds = (probs >= 0.5).astype(np.int32)
    acc = float((preds == labels).mean())
    out = {"accuracy": acc}
    if _SKLEARN_OK:
        try:
            if 0 < labels.sum() < labels.size:
                out["roc_auc"] = float(roc_auc_score(labels, probs))
                out["pr_auc"] = float(average_precision_score(labels, probs))
            else:
                out["roc_auc"] = float("nan")
                out["pr_auc"] = float("nan")
            p, r, f, _ = precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0,
            )
            out["precision"] = float(p)
            out["recall"] = float(r)
            out["f1"] = float(f)
        except Exception:
            out.setdefault("roc_auc", float("nan"))
            out.setdefault("pr_auc", float("nan"))
            out.setdefault("precision", float("nan"))
            out.setdefault("recall", float("nan"))
            out.setdefault("f1", float("nan"))
    else:
        # Manual P/R/F1
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        out["precision"] = prec
        out["recall"] = rec
        out["f1"] = f1
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


# =====================================================================
# Main training driver
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train BiGAT directly from aggregate teacher rows.")
    parser.add_argument("--teacher-csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rows-per-epoch", type=int, default=5000)
    parser.add_argument("--valid-rows-per-epoch", type=int, default=2000)
    parser.add_argument("--max-rows-per-source-instance-per-epoch", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Reserved for future per-graph mini-batching; current "
                             "implementation forwards one graph at a time.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--full-valid-every", type=int, default=10)
    parser.add_argument("--disable-early-stopping", action="store_true", default=False,
                        help="If set, train for all --max-epochs without early stopping.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    _atomic_json_save(config_dict, out_dir / "config.json")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[setup] device={args.device} sklearn_ok={_SKLEARN_OK}")
    print(f"[setup] teacher_csv={args.teacher_csv}")
    print(f"[setup] manifest={args.manifest}")
    print(f"[setup] out_dir={out_dir}")

    # ------------------------------------------------------------------
    # 1) Load CSV + manifest, backfill metadata
    # ------------------------------------------------------------------
    rows_raw = _load_aggregate_rows(Path(args.teacher_csv))
    if not rows_raw:
        raise RuntimeError(f"No rows in teacher CSV: {args.teacher_csv}")
    print(f"[load] teacher rows = {len(rows_raw):,}")

    manifest = _load_manifest(Path(args.manifest))
    manifest_lookup = _build_manifest_lookup(manifest)
    print(f"[load] manifest scenarios = {len(manifest_lookup)}")

    _backfill_metadata(rows_raw, manifest_lookup)
    _propagate_constraint_features(rows_raw)

    # ------------------------------------------------------------------
    # 2) Build graphs and the per-row index
    # ------------------------------------------------------------------
    t0 = time.time()
    graphs, row_index_records = _build_graphs(rows_raw)
    print(f"[graphs] built {len(graphs)} graphs from {len(row_index_records)} rows "
          f"in {time.time() - t0:.1f}s")
    if not graphs or not row_index_records:
        raise RuntimeError("No usable graphs produced from teacher CSV.")

    row_index = pd.DataFrame(row_index_records)
    row_index["split"] = row_index["source_instance"].map(
        lambda sid: manifest_lookup.get(sid, {}).get("split", row_index_records[0].get("split", "train"))
    )

    df_train = row_index[row_index["split"] == "train"].reset_index().rename(columns={"index": "row_id"})
    df_valid = row_index[row_index["split"] == "valid"].reset_index().rename(columns={"index": "row_id"})
    df_test = row_index[row_index["split"] == "test"].reset_index().rename(columns={"index": "row_id"})

    if df_train.empty:
        raise RuntimeError("Train split is empty after manifest mapping.")

    # Reset positional index so .loc[i, ...] in sampler works as expected
    df_train = df_train.drop(columns=["row_id"]).reset_index(drop=True)
    df_valid = df_valid.drop(columns=["row_id"]).reset_index(drop=True)
    df_test = df_test.drop(columns=["row_id"]).reset_index(drop=True)

    split_summary_df = pd.DataFrame([
        {"split": "train", "rows": len(df_train),
         "n_unique_source_instances": int(df_train["source_instance"].nunique()),
         "pos_rows": int((df_train["label"] == 1).sum()),
         "neg_rows": int((df_train["label"] == 0).sum())},
        {"split": "valid", "rows": len(df_valid),
         "n_unique_source_instances": int(df_valid["source_instance"].nunique()),
         "pos_rows": int((df_valid["label"] == 1).sum()),
         "neg_rows": int((df_valid["label"] == 0).sum())},
        {"split": "test", "rows": len(df_test),
         "n_unique_source_instances": int(df_test["source_instance"].nunique()),
         "pos_rows": int((df_test["label"] == 1).sum()),
         "neg_rows": int((df_test["label"] == 0).sum())},
    ])
    split_summary_df.to_csv(out_dir / "split_summary.csv", index=False)
    print("[split]")
    print(split_summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 3) Normalize features using train graphs only
    # ------------------------------------------------------------------
    train_graph_ids = sorted(df_train["graph_id"].unique().tolist())
    norm_stats = _normalize_graphs(graphs, train_graph_ids)
    print(f"[normalize] applied per-feature standardization (train-only stats)")

    # ------------------------------------------------------------------
    # 4) Build model
    # ------------------------------------------------------------------
    sample0 = graphs[0]
    model = BiGATColumnScorer(
        column_dim=int(sample0["column_features"].shape[1]),
        constraint_dim=int(sample0["constraint_features"].shape[1]),
        edge_dim=int(sample0["edge_attr_col_to_con"].shape[1])
            if sample0["edge_attr_col_to_con"].size else len(utilities.EDGE_FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[model] BiGAT hidden_dim={args.hidden_dim} dropout={args.dropout}")

    # ------------------------------------------------------------------
    # 5) Training loop with curriculum + controlled sampling
    # ------------------------------------------------------------------
    history: List[Dict[str, Any]] = []
    sampling_log: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_prauc = -float("inf")
    bad_epochs = 0
    has_prauc = False  # set True after first valid epoch produces a finite PR-AUC
    rng = random.Random(args.seed)

    best_loss_path = out_dir / "best_valid_loss.pt"
    best_prauc_path = out_dir / "best_valid_prauc.pt"
    last_path = out_dir / "last.pt"
    history_csv = out_dir / "training_log.csv"
    sampling_csv = out_dir / "sampling_log.csv"

    for epoch in range(1, args.max_epochs + 1):
        ep_rng = random.Random(args.seed + epoch)
        t_epoch = time.time()

        # --- Sample train rows ---
        train_idx, train_sampling_stats = _sample_with_curriculum(
            df_train,
            n_total=args.rows_per_epoch,
            epoch=epoch,
            rng=ep_rng,
            max_per_source_instance=args.max_rows_per_source_instance_per_epoch,
        )
        train_batch = df_train.loc[train_idx].reset_index(drop=True)

        # --- Train one pass over the sampled rows ---
        train_loss, train_probs, train_labels = _forward_batched(
            model, graphs, train_batch, args.device,
            training=True, optimizer=optimizer,
        )
        train_metrics = _binary_metrics(train_probs, train_labels)

        # --- Validation: fast every epoch, full every K epochs ---
        if not df_valid.empty:
            valid_rng = random.Random(args.seed * 13 + epoch)
            valid_idx, valid_sampling_stats = _sample_controlled(
                df_valid, args.valid_rows_per_epoch, rng=valid_rng,
                max_per_source_instance=max(args.max_rows_per_source_instance_per_epoch, 50),
            )
            valid_batch = df_valid.loc[valid_idx].reset_index(drop=True) if valid_idx else df_valid.iloc[:0]
            with torch.no_grad():
                v_loss, v_probs, v_labels = _forward_batched(
                    model, graphs, valid_batch, args.device,
                    training=False, optimizer=None,
                )
            v_metrics = _binary_metrics(v_probs, v_labels)
            v_metrics["loss"] = v_loss

            full_valid = (epoch % max(1, args.full_valid_every) == 0)
            if full_valid:
                with torch.no_grad():
                    fv_loss, fv_probs, fv_labels = _forward_batched(
                        model, graphs, df_valid, args.device,
                        training=False, optimizer=None,
                    )
                fv_metrics = _binary_metrics(fv_probs, fv_labels)
                fv_metrics["loss"] = fv_loss
            else:
                fv_metrics = None
        else:
            v_loss = float("nan")
            v_metrics = {"loss": float("nan"), "accuracy": float("nan"), "precision": float("nan"),
                         "recall": float("nan"), "f1": float("nan"), "roc_auc": float("nan"),
                         "pr_auc": float("nan")}
            fv_metrics = None
            valid_sampling_stats = {"pos": 0, "neg": 0, "by_source_instance": {}}

        if math.isfinite(v_metrics.get("pr_auc", float("nan"))):
            has_prauc = True

        # --- Logging row ---
        row = {
            "epoch": epoch,
            "regime": train_sampling_stats.get("regime", ""),
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "valid_loss": v_metrics["loss"],
            "valid_acc": v_metrics["accuracy"],
            "valid_precision": v_metrics["precision"],
            "valid_recall": v_metrics["recall"],
            "valid_f1": v_metrics["f1"],
            "valid_roc_auc": v_metrics["roc_auc"],
            "valid_pr_auc": v_metrics["pr_auc"],
            "full_valid_loss": fv_metrics["loss"] if fv_metrics else "",
            "full_valid_pr_auc": fv_metrics["pr_auc"] if fv_metrics else "",
            "full_valid_f1": fv_metrics["f1"] if fv_metrics else "",
            "full_valid_roc_auc": fv_metrics["roc_auc"] if fv_metrics else "",
            "epoch_seconds": time.time() - t_epoch,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_csv, index=False)

        # --- Sampling log row ---
        sampling_log.append({
            "epoch": epoch,
            "regime": train_sampling_stats.get("regime", ""),
            "pool_size": train_sampling_stats.get("pool_size",
                          train_sampling_stats.get("full_pool_size", len(df_train))),
            "n_sampled": len(train_idx),
            "pos_count": train_sampling_stats["pos"],
            "neg_count": train_sampling_stats["neg"],
            "n_unique_source_instances": train_sampling_stats.get("n_unique_source_instances", 0),
            "by_shock_profile": json.dumps(train_sampling_stats.get("by_shock_profile", {}), default=str),
            "by_store_limit": json.dumps(train_sampling_stats.get("by_store_limit", {}), default=str),
            "by_sku_limit": json.dumps(train_sampling_stats.get("by_sku_limit", {}), default=str),
            "top_source_instances": json.dumps(
                train_sampling_stats.get("by_source_instance",
                                         train_sampling_stats.get("by_source_instance_top", {})),
                default=str,
            ),
        })
        pd.DataFrame(sampling_log).to_csv(sampling_csv, index=False)

        primary_str = (f"PR-AUC={v_metrics['pr_auc']:.4f}"
                       if has_prauc else f"loss={v_metrics['loss']:.4f}")
        full_str = ""
        if fv_metrics is not None:
            full_str = (f" [full] loss={fv_metrics['loss']:.4f} "
                        f"PR-AUC={fv_metrics['pr_auc']:.4f} F1={fv_metrics['f1']:.4f}")
        print(
            f"[epoch {epoch:03d}/{args.max_epochs}] regime={row['regime']:<22s} "
            f"train_loss={train_loss:.4f} train_F1={train_metrics['f1']:.3f} "
            f"valid_loss={v_metrics['loss']:.4f} valid_F1={v_metrics['f1']:.3f} {primary_str}{full_str} "
            f"({row['epoch_seconds']:.1f}s)"
        )

        # --- Checkpointing ---
        ckpt_payload = {
            "state_dict": model.state_dict(),
            "config": {
                "column_dim": model.column_dim,
                "constraint_dim": model.constraint_dim,
                "edge_dim": model.edge_dim,
                "hidden_dim": model.hidden_dim,
                "dropout": args.dropout,
            },
            "normalization": norm_stats,
            "feature_names": {
                "column": utilities.COLUMN_FEATURE_NAMES,
                "constraint": utilities.CONSTRAINT_FEATURE_NAMES,
                "edge": utilities.EDGE_FEATURE_NAMES,
            },
            "valid_metrics": v_metrics,
            "full_valid_metrics": fv_metrics,
            "optimizer_state": optimizer.state_dict(),
            "best_valid_loss": best_val_loss,
            "best_valid_prauc": best_val_prauc,
            "last_epoch": epoch,
            "objective": "binary",
            "dataset_type": "aggregate_teacher_rows",
            "args": config_dict,
        }
        _atomic_torch_save(ckpt_payload, last_path)

        improved = False
        if v_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = v_metrics["loss"]
            ckpt_payload["best_valid_loss"] = best_val_loss
            _atomic_torch_save(ckpt_payload, best_loss_path)
            if not has_prauc:
                improved = True
        cur_pr = v_metrics.get("pr_auc", float("nan"))
        if math.isfinite(cur_pr) and cur_pr > best_val_prauc + 1e-6:
            best_val_prauc = cur_pr
            ckpt_payload["best_valid_prauc"] = best_val_prauc
            _atomic_torch_save(ckpt_payload, best_prauc_path)
            improved = True

        if improved:
            bad_epochs = 0
        else:
            bad_epochs += 1

        # --- Early stopping ---
        if not math.isfinite(train_loss) or math.isnan(train_loss):
            print("[early-stop] train loss is NaN/inf — aborting.")
            break
        if args.disable_early_stopping:
            continue  # skip all early stopping checks
        if epoch < EASY_END_EPOCH:
            continue  # don't early-stop during the easy curriculum
        if bad_epochs >= args.early_stopping_patience:
            print(f"[early-stop] {bad_epochs} epochs without improvement on "
                  f"{'PR-AUC' if has_prauc else 'valid_loss'} — stopping.")
            break

    # ------------------------------------------------------------------
    # 6) Final full validation + test using best checkpoint
    # ------------------------------------------------------------------
    primary_ckpt = best_prauc_path if (has_prauc and best_prauc_path.exists()) else best_loss_path
    if not primary_ckpt.exists():
        primary_ckpt = last_path

    print(f"\n[final] loading best checkpoint: {primary_ckpt.name}")
    state = torch.load(primary_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(state["state_dict"])

    final_results: Dict[str, Any] = {
        "best_checkpoint": str(primary_ckpt),
        "best_valid_loss": best_val_loss,
        "best_valid_prauc": best_val_prauc if math.isfinite(best_val_prauc) else None,
    }

    if not df_valid.empty:
        with torch.no_grad():
            fv_loss, fv_probs, fv_labels = _forward_batched(
                model, graphs, df_valid, args.device,
                training=False, optimizer=None,
            )
        fv = _binary_metrics(fv_probs, fv_labels)
        fv["loss"] = fv_loss
        final_results["full_valid"] = fv
        print(f"[final-valid] loss={fv_loss:.4f} F1={fv['f1']:.4f} "
              f"PR-AUC={fv['pr_auc']:.4f} ROC-AUC={fv['roc_auc']:.4f} "
              f"P={fv['precision']:.4f} R={fv['recall']:.4f}")

    if not df_test.empty:
        with torch.no_grad():
            t_loss, t_probs, t_labels = _forward_batched(
                model, graphs, df_test, args.device,
                training=False, optimizer=None,
            )
        tm = _binary_metrics(t_probs, t_labels)
        tm["loss"] = t_loss
        final_results["test"] = tm
        print(f"[final-test ] loss={t_loss:.4f} F1={tm['f1']:.4f} "
              f"PR-AUC={tm['pr_auc']:.4f} ROC-AUC={tm['roc_auc']:.4f} "
              f"P={tm['precision']:.4f} R={tm['recall']:.4f}")

    _atomic_json_save(final_results, out_dir / "final_test_metrics.json")
    print(f"\n[done] artifacts written to {out_dir}")
    print(f"  - {history_csv.name}")
    print(f"  - {sampling_csv.name}")
    print(f"  - split_summary.csv  config.json  final_test_metrics.json")
    print(f"  - {best_loss_path.name}  ({'exists' if best_loss_path.exists() else 'missing'})")
    print(f"  - {best_prauc_path.name} ({'exists' if best_prauc_path.exists() else 'missing'})")
    print(f"  - {last_path.name}")


if __name__ == "__main__":
    main()
