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

# Module-level switches set by main() before _build_graphs_from_rows runs.
# Defaults preserve the LEGACY behaviour bit-identically:
#   _LABEL_MODE      = "selected_in_rmp"  → label = teacher_label > 0.5
#   _RC_TOP_K        = 1                  → unused under "selected_in_rmp"
#   _OBJECTIVE       = "binary"           → BCE-with-logits (legacy)
#   _RANKING_TARGET  = "teacher_label"    → graded-target falls back to binary
_LABEL_MODE: str = "selected_in_rmp"
_RC_TOP_K: int = 1
_OBJECTIVE: str = "binary"
_RANKING_TARGET: str = "teacher_label"
_RANK_K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)


# Indices into utilities.COLUMN_FEATURE_NAMES that correspond to "rc-only"
# information. The full list is:
#   0  reduced_cost
#   1  total_flow
#   2  n_pairs
#   3  total_need_covered
#   4  total_surplus_consumed
#   5  avg_shortage_ratio       ← service-level
#   6  avg_surplus_ratio        ← service-level
#   7  avg_time_urgency         ← service-level
#   8  avg_negative_reduced_cost
#   9  acceptance_score
#  10  compensation_mean
#  11  column_cost
# rc_only keeps indices 0 and 8 (the two RC-derived features) and zeroes the rest.
_RC_ONLY_ACTIVE_INDICES = (0, 8)


def _resolve_column_feature_mask(spec: str, column_dim: int) -> Optional[List[float]]:
    """Translate --column-feature-mask CLI value into a length-`column_dim`
    list[0/1]. Returns None for "all" (the default → BiGAT uses all-ones,
    bit-identical to legacy)."""
    spec = (spec or "").strip().lower()
    if spec in {"", "all", "none", "full"}:
        return None
    if spec in {"rc", "rc_only", "rc-only"}:
        mask = [0.0] * column_dim
        for idx in _RC_ONLY_ACTIVE_INDICES:
            if 0 <= idx < column_dim:
                mask[idx] = 1.0
        return mask
    parts = [chunk.strip() for chunk in spec.split(",") if chunk.strip()]
    if len(parts) != column_dim:
        raise ValueError(
            f"--column-feature-mask='{spec}' has {len(parts)} entries; expected {column_dim}"
        )
    out: List[float] = []
    for chunk in parts:
        try:
            v = float(chunk)
        except ValueError:
            raise ValueError(f"--column-feature-mask entry '{chunk}' is not numeric")
        out.append(1.0 if v >= 0.5 else 0.0)
    return out


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


def _simple_graph_id(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """Graph ID = (source_instance, branch_node_id, episode) only.

    This groups ALL columns for a single CG episode together, regardless of
    constraint_state_hash. Constraint features are then extracted once per
    graph and shared across all columns in that episode.
    """
    return (
        str(row.get("source_instance") or row.get("instance_id") or "default"),
        str(row.get("branch_node_id") or row.get("node_id") or "root"),
        str(row.get("episode") or row.get("episode_id") or "0"),
    )


def _propagate_constraint_features(rows: List[Dict[str, Any]]) -> None:
    """Fill empty `constraint_features_json` from first non-empty in same (source_instance, branch_node_id, episode) group."""
    first_by_graph: Dict[Tuple[str, str, str], str] = {}
    for row in rows:
        gid = _simple_graph_id(row)
        value = str(row.get("constraint_features_json") or "").strip()
        if value and gid not in first_by_graph:
            first_by_graph[gid] = value
    for row in rows:
        if not str(row.get("constraint_features_json") or "").strip():
            row["constraint_features_json"] = first_by_graph.get(_simple_graph_id(row), "")


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
    Dict[str, Any],
]:
    """Group rows by simple graph_id (source_instance, branch_node_id, episode) and
    build one in-memory graph sample per group. Constraint features are extracted
    once per graph and cached. All columns in a graph use the same constraint features.

    Returns:
        graphs: list of {column_features, constraint_features, edges, ..., metadata}
                length = number of graph groups
        row_index: list of {row_id, graph_id, col_idx_in_graph, label, metadata...}
                   length = number of rows that successfully landed in a graph
        diagnostics: Dict with graph_count, graphs_with_constraints, constraint_feature_dim,
                     avg_constraint_nodes, repeated_constraint_hashes, etc.
    """
    import hashlib

    # Group by simple graph_id only
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_simple_graph_id(row)].append(row)

    graphs: List[Dict[str, Any]] = []
    row_index: List[Dict[str, Any]] = []
    skipped_groups = 0
    groups_with_constraints = 0
    groups_missing_constraints = 0
    constraint_hashes: List[str] = []
    constraint_node_counts: List[int] = []
    constraint_feature_dims: set = set()

    for simple_gid, group_rows in grouped.items():
        source_instance, branch_node_id, episode = simple_gid

        # Sort within group so column ordering is deterministic
        try:
            group_rows.sort(key=lambda r: (
                int(float(r.get("column_index") or 10**9)),
                str(r.get("pattern_id", "")),
            ))
        except (TypeError, ValueError):
            group_rows.sort(key=lambda r: str(r.get("pattern_id", "")))

        try:
            # Find the first non-null constraint_features_json in this graph
            constraint_json = None
            for r in group_rows:
                val = str(r.get("constraint_features_json") or "").strip()
                if val:
                    constraint_json = val
                    break

            if constraint_json:
                groups_with_constraints += 1
                # Hash the constraint features for repetition detection
                constraint_hash = hashlib.sha256(constraint_json.encode()).hexdigest()[:12]
                constraint_hashes.append(constraint_hash)
                # Count nodes in constraint features
                try:
                    con_feat = json.loads(constraint_json)
                    if isinstance(con_feat, list):
                        constraint_node_counts.append(len(con_feat))
                        if con_feat and isinstance(con_feat[0], (list, tuple)):
                            constraint_feature_dims.add(len(con_feat[0]))
                except (json.JSONDecodeError, TypeError):
                    pass
            else:
                groups_missing_constraints += 1

            sample = utilities.build_training_sample_from_exported_teacher_rows(
                group_rows,
                episode_id=episode,
                source_instance=source_instance,
                product=str(group_rows[0].get("product") or "unknown_product"),
                period=str(group_rows[0].get("period") or "unknown_period"),
                branch_node_id=branch_node_id,
                decision_state_id=constraint_json and hashlib.sha1(constraint_json.encode()).hexdigest()[:12] or "no_constraints",
            )
        except Exception as exc:
            skipped_groups += 1
            continue

        graph_id = len(graphs)
        graphs.append(sample)

        # Record row → (graph_id, col_idx). We use the same sort order used above.
        # When label_mode == 'rc_top_in_group', overwrite label so that the K
        # rows with the most-negative reduced_cost in this graph (per
        # (product, period) group) get label=1, and all others get label=0.
        # This isolates "does the GNN learn rc-ranking?" from any label
        # noise in selected_in_rmp. K is controlled by --rc-top-k (default 1).
        if _LABEL_MODE == "rc_top_in_group":
            # Sub-group by (product, period) within this graph for fair ranking.
            from collections import defaultdict as _dd
            sub_groups = _dd(list)
            for col_idx, r in enumerate(group_rows):
                key = (str(r.get("product") or ""), str(r.get("period") or ""))
                try:
                    rc = float(r.get("reduced_cost") or 0.0)
                except (TypeError, ValueError):
                    rc = 0.0
                sub_groups[key].append((col_idx, rc))
            chosen_top: set = set()
            for key, items in sub_groups.items():
                items.sort(key=lambda pair: pair[1])  # ascending: most negative first
                for col_idx, rc in items[: max(1, int(_RC_TOP_K))]:
                    if rc < -1e-9:
                        chosen_top.add(col_idx)
        else:
            chosen_top = None  # signals legacy "selected_in_rmp" mode below

        for col_idx, r in enumerate(group_rows):
            if chosen_top is not None:
                label_int = 1 if col_idx in chosen_top else 0
            else:
                try:
                    label_val = float(r.get("teacher_label") or 0.0)
                except (TypeError, ValueError):
                    label_val = 0.0
                label_int = int(label_val > 0.5)
            # Compute the per-row ranking target. Picked once at row-build time
            # so downstream sampling/forward passes can read it cheaply. The
            # default `teacher_label` keeps target == binary label (legacy);
            # `teacher_score` and `neg_reduced_cost` produce graded targets so
            # pairwise_rank / score_regression have a non-trivial gradient.
            try:
                _rc = float(r.get("reduced_cost") or 0.0)
            except (TypeError, ValueError):
                _rc = 0.0
            try:
                _ts = float(r.get("teacher_score") or 0.0)
            except (TypeError, ValueError):
                _ts = 0.0
            if _RANKING_TARGET == "teacher_score":
                target_score = max(0.0, _ts)
            elif _RANKING_TARGET == "neg_reduced_cost":
                target_score = max(0.0, -_rc)
            else:  # teacher_label (default, backward compat)
                target_score = float(label_int)
            row_index.append({
                "graph_id": graph_id,
                "col_idx": col_idx,
                "label": label_int,
                "target_score": float(target_score),
                "reduced_cost": float(_rc),
                "teacher_score_raw": float(_ts),
                "source_instance": str(r.get("source_instance") or ""),
                "shock_profile": str(r.get("shock_profile") or ""),
                "store_limit": _coerce_int(r.get("store_limit")) or 0,
                "sku_limit": _coerce_int(r.get("sku_limit")) or 0,
                "split": str(r.get("split") or ""),
            })

    # Compute repetition stats
    constraint_hash_counts = Counter(constraint_hashes)
    repeated_hashes = [(h, c) for h, c in constraint_hash_counts.most_common() if c > 1]

    diagnostics = {
        "total_graph_groups": len(grouped),
        "graphs_with_constraints": groups_with_constraints,
        "graphs_missing_constraints": groups_missing_constraints,
        "graphs_skipped": skipped_groups,
        "avg_constraint_nodes": (
            float(sum(constraint_node_counts) / len(constraint_node_counts))
            if constraint_node_counts else 0.0
        ),
        "constraint_feature_dimensions": sorted(list(constraint_feature_dims)),
        "unique_constraint_hashes": len(constraint_hash_counts),
        "repeated_constraint_hashes_count": len(repeated_hashes),
        "repeated_constraint_hashes_top5": [
            {"hash": h, "count": c} for h, c in repeated_hashes[:5]
        ],
    }

    if skipped_groups:
        print(f"[graphs] skipped {skipped_groups}/{len(grouped)} groups due to "
              f"missing/invalid features.")

    print(f"[graphs] diagnostics: {diagnostics}")

    return graphs, row_index, diagnostics


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
        by_inst = list(sub.groupby("source_instance").groups.items())
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
# Loss functions (binary BCE, pairwise ranking, score regression)
# =====================================================================

def _binary_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Legacy BCE-with-logits (per-column independent classification)."""
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")


def _pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Softplus-based pairwise ranking loss.

    When `target_scores` is provided AND has graded values (not all in {0, 1}),
    we form pairs (i, j) such that target_scores[i] > target_scores[j] and
    penalise logits[i] < logits[j]. This produces a graded ranking signal
    suitable for `--ranking-target teacher_score / neg_reduced_cost`.

    Otherwise we fall back to the binary positive-vs-negative formulation
    (matches GNN/03_train_bigat.py:pairwise_ranking_loss). Backward compatible
    when called with `--ranking-target teacher_label`.
    """
    if logits.numel() < 2:
        return logits.sum() * 0.0
    is_graded = (
        target_scores is not None
        and target_scores.numel() == logits.numel()
        and bool((target_scores > 1e-6).any())
        and bool(((target_scores > 1e-6) & (target_scores < 1.0 - 1e-6)).any())
    )
    if is_graded:
        diff_t = target_scores.unsqueeze(0) - target_scores.unsqueeze(1)
        mask = diff_t > 1e-6
        if not bool(mask.any()):
            return logits.sum() * 0.0
        diff_p = logits.unsqueeze(0) - logits.unsqueeze(1)
        return F.softplus(-diff_p[mask]).mean()
    # Binary fallback — same as GNN/03_train_bigat.py
    pos_idx = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    neg_idx = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return logits.sum() * 0.0
    pos_scores = logits[pos_idx][:, None]
    neg_scores = logits[neg_idx][None, :]
    return F.softplus(-(pos_scores - neg_scores)).mean()


def _score_regression_loss(
    logits: torch.Tensor,
    target_scores: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 between sigmoid(logit) and the [0, 1]-normalised target.

    Per-graph normalisation done at call site; this function only consumes the
    already-normalised target.
    """
    return F.smooth_l1_loss(torch.sigmoid(logits), target_scores.float())


def _compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_scores: Optional[torch.Tensor],
) -> torch.Tensor:
    """Dispatch loss based on module-global _OBJECTIVE."""
    obj = _OBJECTIVE
    if obj == "pairwise_rank":
        return _pairwise_ranking_loss(logits, labels, target_scores)
    if obj == "score_regression":
        if target_scores is None or target_scores.numel() == 0:
            return _binary_bce_loss(logits, labels)
        # Per-graph normalise to [0, 1] so smooth_l1 stays well-scaled when
        # different graphs have very different teacher_score magnitudes.
        norm = target_scores.float()
        max_t = norm.max().clamp_min(1e-12)
        return _score_regression_loss(logits, norm / max_t)
    # Default = binary BCE (backward compat with legacy training).
    return _binary_bce_loss(logits, labels)


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
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward sampled rows grouped by graph_id.

    Returns (loss, probs, labels, target_scores, graph_ids). The latter two are
    needed for ranking metrics (MRR, NDCG, P@K, R@K) which are computed
    per-graph then averaged. Loss is dispatched via `_compute_loss` based on
    the module-global _OBJECTIVE.
    """
    model.train(training)
    total_loss = 0.0
    total_n = 0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_graph_ids: List[np.ndarray] = []

    by_graph = sampled_rows_df.groupby("graph_id", sort=False)
    if optimizer is not None and training:
        optimizer.zero_grad()

    _col_dim = getattr(model, "column_dim", None)
    _con_dim = getattr(model, "constraint_dim", None)
    _skipped_dim = 0
    _has_target_score = "target_score" in sampled_rows_df.columns

    for gid, sub in by_graph:
        graph = graphs[int(gid)]
        cols = sub["col_idx"].astype(int).values
        labels = torch.from_numpy(sub["label"].astype(np.float32).values).to(device)
        if _has_target_score:
            target_arr = sub["target_score"].astype(np.float32).values
            target_scores = torch.from_numpy(target_arr).to(device)
        else:
            target_arr = sub["label"].astype(np.float32).values
            target_scores = labels
        n = labels.numel()

        # Skip graphs whose feature dimensions don't match the model to avoid crashes
        if _col_dim is not None and graph["column_features"].shape[1:] and graph["column_features"].shape[1] != _col_dim:
            _skipped_dim += 1
            continue
        if _con_dim is not None and graph["constraint_features"].shape[1:] and graph["constraint_features"].shape[1] != _con_dim:
            _skipped_dim += 1
            continue

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
        loss = _compute_loss(logits, labels, target_scores)

        if training and optimizer is not None:
            (loss * (n / max(1, len(sampled_rows_df)))).backward()

        total_loss += float(loss.detach().cpu()) * n
        total_n += n
        with torch.no_grad():
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_targets.append(target_arr.astype(np.float32))
            all_graph_ids.append(np.full(n, int(gid), dtype=np.int64))

    if training and optimizer is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    if _skipped_dim > 0:
        print(f"[forward] WARNING: skipped {_skipped_dim} graphs with mismatched feature dims "
              f"(expected col={_col_dim} con={_con_dim}). Stale CSV? Run teacher generation again.")

    mean_loss = total_loss / max(1, total_n)
    probs_arr = np.concatenate(all_probs) if all_probs else np.zeros(0, dtype=np.float32)
    labels_arr = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.float32)
    targets_arr = np.concatenate(all_targets) if all_targets else np.zeros(0, dtype=np.float32)
    graph_ids_arr = np.concatenate(all_graph_ids) if all_graph_ids else np.zeros(0, dtype=np.int64)
    return mean_loss, probs_arr, labels_arr, targets_arr, graph_ids_arr


def _ranking_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    target_scores: Optional[np.ndarray] = None,
    graph_ids: Optional[np.ndarray] = None,
    k_values: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Per-graph then macro-averaged ranking metrics.

    Computes:
      mrr, top1_acc, top1_score_match (graded equivalent of top-1),
      ndcg_at_k, prec_at_k, rec_at_k for each k in k_values.

    "Relevant" = label > 0.5 (binary). NDCG uses graded `target_scores` if
    provided, else falls back to binary labels. Graphs with no positive label
    are skipped from MRR/Prec/Rec (NDCG would be 0/0 — undefined). For Top-1
    accuracy a graph with no positive contributes 0.
    """
    nan_out = {
        "mrr": float("nan"),
        "top1_acc": float("nan"),
        "top1_score_match": float("nan"),
        "n_graphs_with_positive": 0,
    }
    for k in k_values:
        nan_out[f"ndcg_at_{k}"] = float("nan")
        nan_out[f"prec_at_{k}"] = float("nan")
        nan_out[f"rec_at_{k}"] = float("nan")
    if probs.size == 0 or graph_ids is None or graph_ids.size == 0:
        return nan_out

    if target_scores is None or target_scores.size != probs.size:
        target_scores = labels.astype(np.float32)

    metrics_list: Dict[str, List[float]] = defaultdict(list)
    n_graphs_with_pos = 0
    unique_gids = np.unique(graph_ids)
    for gid in unique_gids:
        mask = graph_ids == gid
        if not mask.any():
            continue
        scores_g = probs[mask]
        truths_g = labels[mask]
        targets_g = target_scores[mask]
        n_g = scores_g.size
        if n_g == 0:
            continue

        # Sort columns within this graph by predicted score (descending).
        order = np.argsort(-scores_g, kind="mergesort")
        sorted_truths = truths_g[order]
        sorted_targets = targets_g[order]

        # Top-1 metrics — defined for any graph
        metrics_list["top1_acc"].append(float(sorted_truths[0] > 0.5))
        # graded top-1: predicted top vs ground-truth top
        ideal_top = float(np.max(targets_g))
        if ideal_top > 1e-9:
            metrics_list["top1_score_match"].append(float(sorted_targets[0]) / max(ideal_top, 1e-9))
        else:
            metrics_list["top1_score_match"].append(1.0)

        has_positive = bool((truths_g > 0.5).any())
        if has_positive:
            n_graphs_with_pos += 1
            # MRR — reciprocal rank of first relevant
            rel_mask = sorted_truths > 0.5
            first_rel_pos = int(np.argmax(rel_mask)) + 1
            metrics_list["mrr"].append(1.0 / float(first_rel_pos))

        # K-based metrics
        total_positive = float((truths_g > 0.5).sum())
        ideal_targets = np.sort(targets_g)[::-1]
        for k in k_values:
            kk = int(min(max(1, k), n_g))
            top_k_truths = sorted_truths[:kk]
            top_k_targets = sorted_targets[:kk]
            n_rel_in_topk = float((top_k_truths > 0.5).sum())
            metrics_list[f"prec_at_{k}"].append(n_rel_in_topk / kk)
            if total_positive > 0:
                metrics_list[f"rec_at_{k}"].append(n_rel_in_topk / total_positive)
            # NDCG@k uses graded targets for both DCG and IDCG. We use the
            # *linear* gain formulation ( gain(rel) = rel ) instead of the
            # exponential ( 2^rel - 1 ) because teacher_score values can run
            # into the hundreds-thousands (λ·|rc|) and 2^large overflows to
            # +inf. Linear gain keeps NDCG well-defined for unbounded
            # continuous targets and is also the standard choice for
            # learning-to-rank with real-valued relevance.
            log_pos = np.log2(np.arange(2, kk + 2))
            dcg = (top_k_targets / log_pos).sum()
            idcg = (ideal_targets[:kk] / log_pos).sum()
            if idcg > 1e-9:
                metrics_list[f"ndcg_at_{k}"].append(float(dcg / idcg))

    out: Dict[str, float] = {}
    for key, vals in metrics_list.items():
        out[key] = float(np.mean(vals)) if vals else float("nan")
    out["n_graphs_with_positive"] = int(n_graphs_with_pos)
    return out


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
    # Ablation: which label to predict.
    #   selected_in_rmp  — teacher_label from CG (rmp-selected columns), legacy default
    #   rc_top_in_group  — top-K most-negative-RC columns per (graph, product, period)
    parser.add_argument("--label-mode", choices=["selected_in_rmp", "rc_top_in_group"],
                        default="selected_in_rmp",
                        help="Label generation strategy. Default preserves legacy behaviour.")
    parser.add_argument("--rc-top-k", type=int, default=1,
                        help="K for rc_top_in_group label mode (top-K negative-RC = positive).")
    # Ablation: which input features the model can see (rc-only ablation uses
    # a binary mask to disable service-level features without changing the
    # architecture). Default = "all" → all-ones mask → bit-identical to legacy.
    # Aliases supported:
    #   "all"          → all 12 features active
    #   "rc_only"      → only [reduced_cost, avg_negative_reduced_cost] active
    #   "1,0,1,0,..."  → explicit comma-separated 0/1 mask of length 12
    parser.add_argument("--column-feature-mask", default="all",
                        help="Input feature gating: 'all', 'rc_only', or comma-separated 0/1.")
    # GPU pinning helper for parallel runs on T4×2 (one variant per GPU).
    parser.add_argument("--cuda-device-index", type=int, default=None,
                        help="If set and device starts with 'cuda', pins to cuda:<index>.")
    parser.add_argument("--resume-from", default="",
                        help="Path to a checkpoint (last.pt) to warm-start from.")
    # ── Objective / ranking-target / metrics ────────────────────────
    # `binary`  (legacy) → per-column BCE-with-logits on the binary `label`.
    # `pairwise_rank`   → softplus(-(score_i − score_j)) over column pairs
    #                     within a graph; pair ordering uses --ranking-target.
    # `score_regression`→ smooth_l1(sigmoid(score), normalized target).
    parser.add_argument("--objective",
                        choices=["binary", "pairwise_rank", "score_regression"],
                        default="binary",
                        help="Loss function. Default 'binary' preserves legacy training.")
    # `teacher_label`     → use the binary label (0/1) as target; backward compat.
    # `teacher_score`     → λ·max(0, −rc) read from the teacher CSV (recommended
    #                       for pairwise_rank under the SLA-penalty regime).
    # `neg_reduced_cost`  → max(0, −reduced_cost) — pure ranking by RC magnitude.
    parser.add_argument("--ranking-target",
                        choices=["teacher_label", "teacher_score", "neg_reduced_cost"],
                        default="teacher_label",
                        help="Per-row scalar used as the ranking target for "
                             "pairwise_rank / score_regression objectives.")
    parser.add_argument("--rank-k-values", default="1,3,5,10",
                        help="Comma-separated K values for NDCG@K, P@K, R@K.")
    args = parser.parse_args()
    # Push CLI choices into module-level globals BEFORE _build_graphs_from_rows
    # is called so the label transformation picks them up.
    global _LABEL_MODE, _RC_TOP_K, _OBJECTIVE, _RANKING_TARGET, _RANK_K_VALUES
    _LABEL_MODE = str(args.label_mode)
    _RC_TOP_K = max(1, int(args.rc_top_k))
    _OBJECTIVE = str(args.objective)
    _RANKING_TARGET = str(args.ranking_target)
    try:
        _RANK_K_VALUES = tuple(
            int(v) for v in str(args.rank_k_values).split(",") if v.strip()
        ) or (1, 3, 5, 10)
    except ValueError:
        _RANK_K_VALUES = (1, 3, 5, 10)
    print(f"[label-mode] {_LABEL_MODE}  (rc_top_k={_RC_TOP_K})")
    print(f"[objective]  {_OBJECTIVE}  ranking_target={_RANKING_TARGET}  "
          f"rank_k={_RANK_K_VALUES}")
    if _OBJECTIVE in {"pairwise_rank", "score_regression"} \
            and _RANKING_TARGET == "teacher_label":
        print(f"[objective]  WARNING: {_OBJECTIVE} with binary teacher_label target "
              f"falls back to pos/neg pair sampling — for graded ranking pass "
              f"--ranking-target teacher_score (or neg_reduced_cost).")
    if args.cuda_device_index is not None and str(args.device).startswith("cuda"):
        args.device = f"cuda:{int(args.cuda_device_index)}"
        print(f"[device] pinned to {args.device}")

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
    graphs, row_index_records, graph_diagnostics = _build_graphs(rows_raw)
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
    column_dim = int(sample0["column_features"].shape[1])
    feature_mask = _resolve_column_feature_mask(args.column_feature_mask, column_dim)
    if feature_mask is not None:
        active = [i for i, v in enumerate(feature_mask) if v > 0.5]
        print(f"[feature-mask] {args.column_feature_mask}  active_indices={active} "
              f"({sum(feature_mask):.0f}/{column_dim} features)")
    else:
        print(f"[feature-mask] all (default)")
    model = BiGATColumnScorer(
        column_dim=column_dim,
        constraint_dim=int(sample0["constraint_features"].shape[1]),
        edge_dim=int(sample0["edge_attr_col_to_con"].shape[1])
            if sample0["edge_attr_col_to_con"].size else len(utilities.EDGE_FEATURE_NAMES),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        column_feature_mask=feature_mask,
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
    best_val_mrr = -float("inf")  # primary metric for pairwise_rank / score_regression
    bad_epochs = 0
    has_prauc = False  # set True after first valid epoch produces a finite PR-AUC
    rng = random.Random(args.seed)

    best_loss_path = out_dir / "best_valid_loss.pt"
    best_prauc_path = out_dir / "best_valid_prauc.pt"
    best_mrr_path = out_dir / "best_valid_mrr.pt"  # ranking-objective primary
    last_path = out_dir / "last.pt"
    history_csv = out_dir / "training_log.csv"
    sampling_csv = out_dir / "sampling_log.csv"

    # ── Resume support ──────────────────────────────────────────────
    # Reload weights, optimizer, best-tracker, history, and start_epoch.
    # `--resume-from` points at the checkpoint to load; if not set but
    # last.pt exists in --out-dir, auto-resume from there. Either way the
    # next iteration of the for-loop continues from `start_epoch`.
    start_epoch = 1
    resume_path = Path(args.resume_from) if args.resume_from else last_path
    if resume_path.exists() and resume_path.stat().st_size > 0:
        try:
            ckpt = torch.load(resume_path, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
            opt_state = ckpt.get("optimizer_state")
            if opt_state is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception as opt_exc:
                    print(f"[resume] WARNING: optimizer state load failed: {opt_exc}")
            prior_history = ckpt.get("history") or []
            if isinstance(prior_history, list) and prior_history:
                history.extend(prior_history)
            elif history_csv.exists() and history_csv.stat().st_size > 0:
                # Fallback: rebuild history from on-disk training_log.csv if
                # the checkpoint was written before we started persisting it.
                try:
                    csv_history = pd.read_csv(history_csv).to_dict("records")
                    history.extend(csv_history)
                    print(f"[resume] recovered {len(csv_history)} history rows from {history_csv.name}")
                except Exception as csv_exc:
                    print(f"[resume] WARNING: could not parse {history_csv.name}: {csv_exc}")
            best_val_loss = float(ckpt.get("best_valid_loss", best_val_loss) or best_val_loss)
            best_val_prauc = float(ckpt.get("best_valid_prauc", best_val_prauc) or best_val_prauc)
            try:
                best_val_mrr = float(ckpt.get("best_valid_mrr", best_val_mrr) or best_val_mrr)
            except (TypeError, ValueError):
                best_val_mrr = best_val_mrr
            last_epoch_in_ckpt = int(ckpt.get("last_epoch", 0) or 0)
            start_epoch = max(1, last_epoch_in_ckpt + 1)
            print(f"[resume] loaded {resume_path.name}: start_epoch={start_epoch} "
                  f"best_loss={best_val_loss:.4f} best_prauc={best_val_prauc:.4f} "
                  f"history_rows={len(history)}")
        except Exception as exc:
            print(f"[resume] WARNING: failed to load {resume_path}: {exc}")
    else:
        print(f"[resume] no checkpoint at {resume_path} — starting fresh")

    if start_epoch > args.max_epochs:
        print(f"[resume] start_epoch={start_epoch} > max_epochs={args.max_epochs} — nothing to do.")

    for epoch in range(start_epoch, args.max_epochs + 1):
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
        train_loss, train_probs, train_labels, train_targets, train_gids = _forward_batched(
            model, graphs, train_batch, args.device,
            training=True, optimizer=optimizer,
        )
        train_metrics = _binary_metrics(train_probs, train_labels)
        train_rank = _ranking_metrics(
            train_probs, train_labels, train_targets, train_gids, _RANK_K_VALUES,
        )

        # --- Validation: fast every epoch, full every K epochs ---
        if not df_valid.empty:
            valid_rng = random.Random(args.seed * 13 + epoch)
            valid_idx, valid_sampling_stats = _sample_controlled(
                df_valid, args.valid_rows_per_epoch, rng=valid_rng,
                max_per_source_instance=max(args.max_rows_per_source_instance_per_epoch, 50),
            )
            valid_batch = df_valid.loc[valid_idx].reset_index(drop=True) if valid_idx else df_valid.iloc[:0]
            with torch.no_grad():
                v_loss, v_probs, v_labels, v_targets, v_gids = _forward_batched(
                    model, graphs, valid_batch, args.device,
                    training=False, optimizer=None,
                )
            v_metrics = _binary_metrics(v_probs, v_labels)
            v_metrics["loss"] = v_loss
            v_rank = _ranking_metrics(v_probs, v_labels, v_targets, v_gids, _RANK_K_VALUES)
            for k, val in v_rank.items():
                v_metrics[k] = val

            full_valid = (epoch % max(1, args.full_valid_every) == 0)
            if full_valid:
                with torch.no_grad():
                    fv_loss, fv_probs, fv_labels, fv_targets, fv_gids = _forward_batched(
                        model, graphs, df_valid, args.device,
                        training=False, optimizer=None,
                    )
                fv_metrics = _binary_metrics(fv_probs, fv_labels)
                fv_metrics["loss"] = fv_loss
                fv_rank = _ranking_metrics(
                    fv_probs, fv_labels, fv_targets, fv_gids, _RANK_K_VALUES,
                )
                for k, val in fv_rank.items():
                    fv_metrics[k] = val
            else:
                fv_metrics = None
        else:
            v_loss = float("nan")
            v_metrics = {"loss": float("nan"), "accuracy": float("nan"), "precision": float("nan"),
                         "recall": float("nan"), "f1": float("nan"), "roc_auc": float("nan"),
                         "pr_auc": float("nan"), "mrr": float("nan"), "top1_acc": float("nan")}
            fv_metrics = None
            valid_sampling_stats = {"pos": 0, "neg": 0, "by_source_instance": {}}

        if math.isfinite(v_metrics.get("pr_auc", float("nan"))):
            has_prauc = True

        # --- Logging row ---
        row = {
            "epoch": epoch,
            "regime": train_sampling_stats.get("regime", ""),
            "objective": _OBJECTIVE,
            "ranking_target": _RANKING_TARGET,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "train_mrr": train_rank.get("mrr", float("nan")),
            "train_top1_acc": train_rank.get("top1_acc", float("nan")),
            "train_ndcg_at_5": train_rank.get("ndcg_at_5", float("nan")),
            "valid_loss": v_metrics["loss"],
            "valid_acc": v_metrics["accuracy"],
            "valid_precision": v_metrics["precision"],
            "valid_recall": v_metrics["recall"],
            "valid_f1": v_metrics["f1"],
            "valid_roc_auc": v_metrics["roc_auc"],
            "valid_pr_auc": v_metrics["pr_auc"],
            "valid_mrr": v_metrics.get("mrr", float("nan")),
            "valid_top1_acc": v_metrics.get("top1_acc", float("nan")),
            "valid_top1_score_match": v_metrics.get("top1_score_match", float("nan")),
            "valid_ndcg_at_1": v_metrics.get("ndcg_at_1", float("nan")),
            "valid_ndcg_at_3": v_metrics.get("ndcg_at_3", float("nan")),
            "valid_ndcg_at_5": v_metrics.get("ndcg_at_5", float("nan")),
            "valid_ndcg_at_10": v_metrics.get("ndcg_at_10", float("nan")),
            "valid_prec_at_5": v_metrics.get("prec_at_5", float("nan")),
            "valid_rec_at_5": v_metrics.get("rec_at_5", float("nan")),
            "valid_n_graphs_with_positive": v_metrics.get("n_graphs_with_positive", 0),
            "full_valid_loss": fv_metrics["loss"] if fv_metrics else "",
            "full_valid_pr_auc": fv_metrics["pr_auc"] if fv_metrics else "",
            "full_valid_f1": fv_metrics["f1"] if fv_metrics else "",
            "full_valid_roc_auc": fv_metrics["roc_auc"] if fv_metrics else "",
            "full_valid_mrr": fv_metrics.get("mrr", float("nan")) if fv_metrics else "",
            "full_valid_top1_acc": fv_metrics.get("top1_acc", float("nan")) if fv_metrics else "",
            "full_valid_ndcg_at_5": fv_metrics.get("ndcg_at_5", float("nan")) if fv_metrics else "",
            "epoch_seconds": time.time() - t_epoch,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_csv, index=False)

        # --- Sampling log row + per-epoch graph cache diagnostics ---
        sampled_graph_ids = set(train_batch["graph_id"].unique().tolist())
        graphs_with_valid_constraints = sum(
            1 for gid in sampled_graph_ids
            if gid < len(graphs) and graphs[gid]["constraint_features"].size > 0
        )
        sampling_log.append({
            "epoch": epoch,
            "regime": train_sampling_stats.get("regime", ""),
            "pool_size": train_sampling_stats.get("pool_size",
                          train_sampling_stats.get("full_pool_size", len(df_train))),
            "n_sampled": len(train_idx),
            "pos_count": train_sampling_stats["pos"],
            "neg_count": train_sampling_stats["neg"],
            "n_unique_source_instances": train_sampling_stats.get("n_unique_source_instances", 0),
            "n_unique_graphs": len(sampled_graph_ids),
            "graphs_with_valid_constraints": graphs_with_valid_constraints,
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
        # Read SLA penalty config from env so we can persist it into the
        # checkpoint metadata (helps reproducibility — every saved model
        # records exactly which penalty regime its teacher rows came from).
        _sla_penalty_on = os.environ.get("IRP_SLA_PENALTY", "off").lower() in {
            "1", "on", "true", "yes",
        }
        try:
            _sla_mu = float(os.environ.get("IRP_SLA_MU", "0.0") or "0.0")
        except ValueError:
            _sla_mu = 0.0
        try:
            _sla_nu = float(os.environ.get("IRP_SLA_NU", "0.0") or "0.0")
        except ValueError:
            _sla_nu = 0.0
        ckpt_payload = {
            "state_dict": model.state_dict(),
            "config": {
                "column_dim": model.column_dim,
                "constraint_dim": model.constraint_dim,
                "edge_dim": model.edge_dim,
                "hidden_dim": model.hidden_dim,
                "dropout": args.dropout,
                "column_feature_mask": [
                    float(v) for v in model.column_feature_mask.tolist()
                ],
            },
            "normalization": norm_stats,
            "feature_names": {
                "column": utilities.COLUMN_FEATURE_NAMES,
                "constraint": utilities.CONSTRAINT_FEATURE_NAMES,
                "edge": utilities.EDGE_FEATURE_NAMES,
            },
            "graph_diagnostics": graph_diagnostics,
            "valid_metrics": v_metrics,
            "full_valid_metrics": fv_metrics,
            "optimizer_state": optimizer.state_dict(),
            "best_valid_loss": best_val_loss,
            "best_valid_prauc": best_val_prauc,
            "best_valid_mrr": best_val_mrr,
            "last_epoch": epoch,
            # Persist history list inside checkpoint so resume can recover
            # training_log.csv even if the on-disk CSV gets clobbered (e.g.
            # a new run overwriting the file with only post-resume rows).
            "history": list(history),
            # Experiment metadata — every checkpoint records the recipe.
            "objective": _OBJECTIVE,
            "ranking_target": _RANKING_TARGET,
            "label_mode": _LABEL_MODE,
            "feature_mode": str(args.column_feature_mask),
            "rank_k_values": list(_RANK_K_VALUES),
            "sla_penalty": {
                "enabled": bool(_sla_penalty_on),
                "mu": _sla_mu,
                "nu": _sla_nu,
            },
            "dataset_type": "aggregate_teacher_rows",
            "args": config_dict,
        }
        _atomic_torch_save(ckpt_payload, last_path)

        improved = False
        # Best-by-loss is always tracked (even for pairwise_rank loss).
        if v_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = v_metrics["loss"]
            ckpt_payload["best_valid_loss"] = best_val_loss
            _atomic_torch_save(ckpt_payload, best_loss_path)
            if _OBJECTIVE == "binary" and not has_prauc:
                improved = True
            elif _OBJECTIVE != "binary":
                # For ranking objectives, treat loss-improvement as progress
                # until the primary ranking metric (MRR) settles.
                improved = improved or (not math.isfinite(best_val_mrr))
        cur_pr = v_metrics.get("pr_auc", float("nan"))
        if math.isfinite(cur_pr) and cur_pr > best_val_prauc + 1e-6:
            best_val_prauc = cur_pr
            ckpt_payload["best_valid_prauc"] = best_val_prauc
            _atomic_torch_save(ckpt_payload, best_prauc_path)
            if _OBJECTIVE == "binary":
                improved = True
        # Track best MRR — primary ranking metric for pairwise_rank /
        # score_regression. Save under a separate file name so it never
        # collides with the legacy best_valid_prauc.pt path.
        cur_mrr = v_metrics.get("mrr", float("nan"))
        if math.isfinite(cur_mrr) and cur_mrr > best_val_mrr + 1e-6:
            best_val_mrr = cur_mrr
            ckpt_payload["best_valid_mrr"] = best_val_mrr
            _atomic_torch_save(ckpt_payload, best_mrr_path)
            if _OBJECTIVE in {"pairwise_rank", "score_regression"}:
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
    # Pick primary checkpoint based on objective:
    #   ranking objectives (pairwise_rank, score_regression) → best MRR
    #   binary                                               → best PR-AUC
    #   fallback                                             → best loss / last
    if _OBJECTIVE in {"pairwise_rank", "score_regression"} and best_mrr_path.exists():
        primary_ckpt = best_mrr_path
    elif has_prauc and best_prauc_path.exists():
        primary_ckpt = best_prauc_path
    elif best_loss_path.exists():
        primary_ckpt = best_loss_path
    else:
        primary_ckpt = last_path

    print(f"\n[final] loading best checkpoint: {primary_ckpt.name}")
    state = torch.load(primary_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(state["state_dict"])

    final_results: Dict[str, Any] = {
        "best_checkpoint": str(primary_ckpt),
        "best_valid_loss": best_val_loss,
        "best_valid_prauc": best_val_prauc if math.isfinite(best_val_prauc) else None,
        "best_valid_mrr": best_val_mrr if math.isfinite(best_val_mrr) else None,
        "objective": _OBJECTIVE,
        "ranking_target": _RANKING_TARGET,
        "label_mode": _LABEL_MODE,
        "feature_mode": str(args.column_feature_mask),
    }

    if not df_valid.empty:
        with torch.no_grad():
            fv_loss, fv_probs, fv_labels, fv_targets, fv_gids = _forward_batched(
                model, graphs, df_valid, args.device,
                training=False, optimizer=None,
            )
        fv = _binary_metrics(fv_probs, fv_labels)
        fv["loss"] = fv_loss
        fv_rank = _ranking_metrics(fv_probs, fv_labels, fv_targets, fv_gids, _RANK_K_VALUES)
        for k, val in fv_rank.items():
            fv[k] = val
        final_results["full_valid"] = fv
        print(f"[final-valid] loss={fv_loss:.4f} F1={fv['f1']:.4f} "
              f"PR-AUC={fv['pr_auc']:.4f} ROC-AUC={fv['roc_auc']:.4f} "
              f"P={fv['precision']:.4f} R={fv['recall']:.4f}")
        print(f"[final-valid-rank] MRR={fv.get('mrr', float('nan')):.4f} "
              f"top1={fv.get('top1_acc', float('nan')):.4f} "
              f"NDCG@5={fv.get('ndcg_at_5', float('nan')):.4f} "
              f"P@5={fv.get('prec_at_5', float('nan')):.4f} "
              f"R@5={fv.get('rec_at_5', float('nan')):.4f}")

    if not df_test.empty:
        with torch.no_grad():
            t_loss, t_probs, t_labels, t_targets, t_gids = _forward_batched(
                model, graphs, df_test, args.device,
                training=False, optimizer=None,
            )
        tm = _binary_metrics(t_probs, t_labels)
        tm["loss"] = t_loss
        t_rank = _ranking_metrics(t_probs, t_labels, t_targets, t_gids, _RANK_K_VALUES)
        for k, val in t_rank.items():
            tm[k] = val
        final_results["test"] = tm
        print(f"[final-test ] loss={t_loss:.4f} F1={tm['f1']:.4f} "
              f"PR-AUC={tm['pr_auc']:.4f} ROC-AUC={tm['roc_auc']:.4f} "
              f"P={tm['precision']:.4f} R={tm['recall']:.4f}")
        print(f"[final-test-rank ] MRR={tm.get('mrr', float('nan')):.4f} "
              f"top1={tm.get('top1_acc', float('nan')):.4f} "
              f"NDCG@5={tm.get('ndcg_at_5', float('nan')):.4f} "
              f"P@5={tm.get('prec_at_5', float('nan')):.4f} "
              f"R@5={tm.get('rec_at_5', float('nan')):.4f}")

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
