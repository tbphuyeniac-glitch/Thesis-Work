from __future__ import annotations

"""
GNN/build_teacher_graph_dataset.py
==================================
Convert CG teacher CSV rows into train/valid/test graph samples.

Inputs
------
- Results/cg_teacher_dataset.csv (or .pkl.gz for Kaggle) produced during
  CG episodes where collect_teacher_mode=True.

Outputs
-------
- GNN/data/irplt_teacher/{train,valid,test}/sample_*.pkl
- GNN/data/irplt_teacher/dataset_summary.json

Split semantics
---------------
When more than one distinct `source_instance` is present, the split is
performed at the instance level (default --split-by-instance) so that no
single instance contributes groups to more than one split. With only one
instance, the code falls back to group-level random split and prints a
warning that the test split is NOT a strict generalization check.

CLI
---
--teacher-csv, --out-dir, --train-ratio, --valid-ratio, --seed,
--overwrite/--no-overwrite, --split-by-instance/--no-split-by-instance
"""

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import utilities


GroupKey = Tuple[str, str, str, str, str, str]


# ----------------------------------------------------------------------
# Ranking-filter helpers (Fix A: drop signal-less groups + stratified cap)
# ----------------------------------------------------------------------
def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _is_positive_row(row: Dict[str, Any]) -> bool:
    """Mirror utilities.build_training_sample_from_exported_teacher_rows: prefer
    teacher_label, fall back to selected_in_rmp. Anything not clearly truthy
    counts as negative."""
    raw = row.get("teacher_label")
    if raw is not None and str(raw).strip() != "":
        return _safe_float(raw, 0.0) > 0.5
    return utilities._truthy(row.get("selected_in_rmp", False))


def _stratified_cap(
    pos: List[Dict[str, Any]],
    neg: List[Dict[str, Any]],
    max_size: int,
) -> List[Dict[str, Any]]:
    """Subsample a large group to <= max_size while preserving the positive
    ratio. Positives ranked by teacher_score desc (best first); negatives
    ranked by |reduced_cost| asc (hard negatives near the decision boundary
    first). Both buckets are guaranteed non-empty in the output."""
    p_total = len(pos)
    n_total = len(neg)
    g_total = p_total + n_total
    if g_total <= max_size:
        return pos + neg

    ratio = p_total / g_total
    target_pos = max(1, min(p_total, int(round(max_size * ratio))))
    target_neg = max(1, max_size - target_pos)
    # Rebalance if either side runs short.
    if target_neg > n_total:
        target_neg = n_total
        target_pos = max(1, max_size - target_neg)
    if target_pos > p_total:
        target_pos = p_total
        target_neg = max(1, max_size - target_pos)

    pos_sorted = sorted(pos, key=lambda r: -_safe_float(r.get("teacher_score"), 0.0))
    neg_sorted = sorted(neg, key=lambda r: abs(_safe_float(r.get("reduced_cost"), 1e18)))
    return pos_sorted[:target_pos] + neg_sorted[:target_neg]


def _group_pair_count(rows: List[Dict[str, Any]]) -> int:
    """Estimate number of pairwise (pos, neg) comparisons for ranking loss."""
    p = sum(1 for r in rows if _is_positive_row(r))
    return p * (len(rows) - p)


def _group_size_stats(sizes: List[int]) -> Dict[str, float]:
    if not sizes:
        return {"min": 0, "median": 0, "mean": 0.0, "max": 0}
    return {
        "min": int(min(sizes)),
        "median": int(statistics.median(sizes)),
        "mean": float(sum(sizes) / len(sizes)),
        "max": int(max(sizes)),
    }


def filter_and_cap_groups(
    grouped_rows: Dict[GroupKey, List[Dict[str, Any]]],
    *,
    min_size: int,
    max_size: int,
    require_mixed: bool,
) -> Tuple[Dict[GroupKey, List[Dict[str, Any]]], Dict[str, Any]]:
    """Apply ranking-friendly filtering on a per-group basis.

    Rules:
      1. drop groups with size < min_size                       (no ranking signal)
      2. drop groups with all-positive or all-negative labels    (no pairwise signal)
      3. cap groups with size > max_size via stratified subsample (best positives,
         hard negatives), preserving positive ratio.

    Returns (filtered_grouped_rows, report_stats)."""
    out: Dict[GroupKey, List[Dict[str, Any]]] = {}

    sizes_in: List[int] = []
    sizes_out: List[int] = []
    pos_in = 0
    neg_in = 0
    pos_out = 0
    neg_out = 0
    pairs_in = 0
    pairs_out = 0
    n_dropped_small = 0
    n_dropped_zero_pos = 0
    n_dropped_all_pos = 0
    n_capped = 0
    n_kept = 0

    for key, rows in grouped_rows.items():
        n = len(rows)
        sizes_in.append(n)
        pos = [r for r in rows if _is_positive_row(r)]
        neg = [r for r in rows if not _is_positive_row(r)]
        pos_in += len(pos)
        neg_in += len(neg)
        pairs_in += len(pos) * len(neg)

        if n < min_size:
            n_dropped_small += 1
            continue
        if require_mixed:
            if len(pos) == 0:
                n_dropped_zero_pos += 1
                continue
            if len(neg) == 0:
                n_dropped_all_pos += 1
                continue

        if n > max_size:
            kept = _stratified_cap(pos, neg, max_size)
            n_capped += 1
        else:
            kept = rows

        out[key] = kept
        n_kept += 1
        sizes_out.append(len(kept))
        pos_kept = sum(1 for r in kept if _is_positive_row(r))
        neg_kept = len(kept) - pos_kept
        pos_out += pos_kept
        neg_out += neg_kept
        pairs_out += pos_kept * neg_kept

    report = {
        "groups_in": len(grouped_rows),
        "groups_out": n_kept,
        "groups_dropped_too_small": n_dropped_small,
        "groups_dropped_zero_positive": n_dropped_zero_pos,
        "groups_dropped_all_positive": n_dropped_all_pos,
        "groups_capped": n_capped,
        "columns_in": pos_in + neg_in,
        "columns_out": pos_out + neg_out,
        "positives_in": pos_in,
        "positives_out": pos_out,
        "positive_rate_in": (pos_in / max(1, pos_in + neg_in)),
        "positive_rate_out": (pos_out / max(1, pos_out + neg_out)) if (pos_out + neg_out) else 0.0,
        "pairwise_pairs_in": pairs_in,
        "pairwise_pairs_out": pairs_out,
        "pairwise_pairs_reduction": (1.0 - pairs_out / pairs_in) if pairs_in else 0.0,
        "size_stats_in": _group_size_stats(sizes_in),
        "size_stats_out": _group_size_stats(sizes_out),
        "min_size_threshold": min_size,
        "max_size_threshold": max_size,
        "require_mixed_labels": bool(require_mixed),
    }
    return out, report


def print_filter_report(report: Dict[str, Any]) -> None:
    print("\n[Ranking-filter report]")
    print(f"  groups        : {report['groups_in']:>8,} → {report['groups_out']:>8,}  "
          f"(dropped: small={report['groups_dropped_too_small']:,}  "
          f"zero_pos={report['groups_dropped_zero_positive']:,}  "
          f"all_pos={report['groups_dropped_all_positive']:,}  "
          f"capped={report['groups_capped']:,})")
    print(f"  columns       : {report['columns_in']:>8,} → {report['columns_out']:>8,}")
    print(f"  positive rate : {report['positive_rate_in']:.3f} → {report['positive_rate_out']:.3f}")
    print(f"  pairwise pairs: {report['pairwise_pairs_in']:>10,} → {report['pairwise_pairs_out']:>10,}  "
          f"(reduction: {report['pairwise_pairs_reduction']*100:.1f}%)")
    s_in, s_out = report["size_stats_in"], report["size_stats_out"]
    print(f"  group size in : min={s_in['min']:<4} median={s_in['median']:<4} "
          f"mean={s_in['mean']:.1f} max={s_in['max']}")
    print(f"  group size out: min={s_out['min']:<4} median={s_out['median']:<4} "
          f"mean={s_out['mean']:.1f} max={s_out['max']}")


def group_key(row: Dict[str, Any]) -> GroupKey:
    source_instance = str(row.get("source_instance") or row.get("instance_id") or "default")
    branch_node = str(row.get("branch_node_id") or row.get("node_id") or "root")
    episode = str(row.get("episode") or row.get("episode_id") or "0")
    product = str(row.get("product") or row.get("sku") or "unknown_product")
    period = str(row.get("period") or row.get("time_period") or "unknown_period")
    # Prefer the precomputed `constraint_state_hash` column written by recent
    # teacher exports — those leave `constraint_features_json` blank on all
    # rows except the first in each batch to save CSV size. Fall back to
    # hashing the JSON for legacy CSVs that still carry it on every row.
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


def propagate_constraint_features_json(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Within each (group_key) bucket, fill the empty `constraint_features_json`
    entries with the first non-empty value in the group. The teacher exporter
    writes the full JSON only on the first row of each batch to keep the CSV
    small; consumers that previously read the JSON per-row still get a full
    value after this pass."""
    first_by_group: Dict[GroupKey, str] = {}
    for row in rows:
        key = group_key(row)
        value = str(row.get("constraint_features_json") or "").strip()
        if value and key not in first_by_group:
            first_by_group[key] = value
    for row in rows:
        if not str(row.get("constraint_features_json") or "").strip():
            row["constraint_features_json"] = first_by_group.get(group_key(row), "")
    return rows


def _repair_constraint_features_for_group(
    group_rows: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Try to fill in a safe-default `constraint_features_json` for a group
    whose rows have no usable JSON.

    Strategy: infer the number of constraint nodes from the group's edges
    (max constraint_id + 1) and write a zero-padded matrix of shape
    (n_constraints, len(CONSTRAINT_FEATURE_NAMES)).  This lets the graph
    builder still produce a sample — the GNN sees neutral constraint inputs
    rather than the group being silently dropped.

    Returns (repaired, reason).  If repair is impossible (no edges, no column
    features, etc.) returns (False, reason) so the caller can record why.
    """
    n_constraints = 0
    have_column_features = False
    have_edges = False
    for row in group_rows:
        if str(row.get("column_features_json") or "").strip():
            have_column_features = True
        edge_ids_raw = str(row.get("edge_constraint_indices_json") or "").strip()
        if edge_ids_raw:
            try:
                edge_ids = json.loads(edge_ids_raw)
            except (ValueError, TypeError):
                edge_ids = []
            if edge_ids:
                have_edges = True
                n_constraints = max(n_constraints, max(int(c) for c in edge_ids) + 1)

    if not have_column_features:
        return False, "no_column_features"
    if not have_edges:
        return False, "no_edges"

    n_features = len(utilities.CONSTRAINT_FEATURE_NAMES)
    safe_matrix = [[0.0] * n_features for _ in range(n_constraints)]
    safe_json = json.dumps(safe_matrix)
    # Inject on row 0 only — that matches the writer's compaction layout.
    if group_rows:
        group_rows[0]["constraint_features_json"] = safe_json
    return True, "repaired_zero_padded"


def _raise_csv_field_size_limit() -> int:
    """Allow large JSON payloads in legacy teacher CSV fields."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit = int(limit / 10)


def _dataframe_to_records(df: Any) -> List[Dict[str, Any]]:
    import pandas as pd

    df = df.where(pd.notna(df), "")
    return df.to_dict(orient="records")


def read_teacher_rows(path: Path) -> List[Dict[str, Any]]:
    name = path.name.lower()
    if name.endswith((".pkl", ".pickle", ".pkl.gz", ".pickle.gz")):
        import pandas as pd

        return _dataframe_to_records(pd.read_pickle(path))
    if name.endswith(".parquet"):
        import pandas as pd

        return _dataframe_to_records(pd.read_parquet(path))
    if name.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if name.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return payload["rows"]
        raise ValueError(f"Unsupported teacher JSON shape in {path}; expected a row list or {{'rows': [...]}}")

    _raise_csv_field_size_limit()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_groups(
    keys: List[GroupKey],
    train_ratio: float,
    valid_ratio: float,
    rng: random.Random,
    split_by_instance: bool = True,
) -> Dict[str, List[GroupKey]]:
    """Split group keys into train/valid/test.

    When `split_by_instance=True` (default) and more than one distinct
    `source_instance` is present across keys, the split is performed at the
    instance level so that no single source_instance contributes groups to
    more than one split — eliminating cross-split data leakage. When only
    one instance exists, we fall back to group-level random split and warn
    the caller that the test split is NOT a strict generalization check.
    """
    if not keys:
        return {"train": [], "valid": [], "test": []}

    instances = sorted({key[0] for key in keys})
    if split_by_instance and len(instances) > 1:
        shuffled_instances = list(instances)
        rng.shuffle(shuffled_instances)
        n_inst = len(shuffled_instances)
        n_train_i = max(1, int(round(n_inst * train_ratio)))
        n_valid_i = max(1, int(round(n_inst * valid_ratio))) if n_inst >= 3 else 0
        if n_train_i + n_valid_i >= n_inst:
            n_valid_i = max(0, n_inst - n_train_i - 1)
        train_i = set(shuffled_instances[:n_train_i])
        valid_i = set(shuffled_instances[n_train_i:n_train_i + n_valid_i])
        test_i = set(shuffled_instances[n_train_i + n_valid_i:])
        buckets: Dict[str, List[GroupKey]] = {"train": [], "valid": [], "test": []}
        for key in keys:
            inst = key[0]
            if inst in train_i:
                buckets["train"].append(key)
            elif inst in valid_i:
                buckets["valid"].append(key)
            else:
                buckets["test"].append(key)
        return buckets

    # Fallback: only one instance → group-level random split (leakage warning).
    shuffled = list(keys)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    if n and n_train == 0:
        n_train = 1
    if n >= 2 and n_valid == 0:
        n_valid = 1
    if n - n_train - n_valid < 0:
        n_valid = max(0, n - n_train)
    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test": shuffled[n_train + n_valid:],
    }


def write_samples(
    grouped_rows: Dict[GroupKey, List[Dict[str, Any]]],
    split_keys: Iterable[GroupKey],
    split_dir: Path,
) -> Tuple[int, List[str], Dict[str, Any]]:
    split_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped: List[str] = []
    n_columns: List[int] = []
    n_positive: List[int] = []
    adaptive_k_values: List[float] = []
    n_valid = 0
    n_repaired = 0
    skip_reasons: Dict[str, int] = defaultdict(int)
    for key in split_keys:
        source_instance, branch_node, episode, product, period, constraint_state = key
        rows_for_group = grouped_rows[key]

        # Distinguish "JSON already valid" from "JSON missing — try repair".
        any_json = any(
            str(row.get("constraint_features_json") or "").strip()
            for row in rows_for_group
        )
        was_repaired = False
        if not any_json:
            ok, reason = _repair_constraint_features_for_group(rows_for_group)
            if not ok:
                skip_reasons[reason] += 1
                skipped.append(
                    f"{source_instance}/branch={branch_node}/episode={episode}/"
                    f"product={product}/period={period}/state={constraint_state}: "
                    f"unrepairable_no_constraint_features ({reason})"
                )
                continue
            was_repaired = True

        try:
            sample = utilities.build_training_sample_from_exported_teacher_rows(
                rows_for_group,
                episode_id=episode,
                source_instance=source_instance,
                product=product,
                period=period,
                branch_node_id=branch_node,
                decision_state_id=constraint_state,
            )
        except Exception as exc:
            skip_reasons["build_sample_error"] += 1
            skipped.append(
                f"{source_instance}/branch={branch_node}/episode={episode}/"
                f"product={product}/period={period}/state={constraint_state}: {exc}"
            )
            continue
        written += 1
        if was_repaired:
            n_repaired += 1
        else:
            n_valid += 1
        n_columns.append(int(sample["column_features"].shape[0]))
        n_positive.append(int((sample["labels_binary"] > 0.5).sum()))
        adaptive_values = [
            utilities._float_value(row.get("adaptive_k_star"), float("nan"))
            for row in rows_for_group
            if str(row.get("adaptive_k_star", "")).strip() not in {"", "None", "nan"}
        ]
        if adaptive_values:
            adaptive_k_values.append(float(adaptive_values[0]))
        utilities.save_graph_sample(sample, split_dir / f"sample_{written:05d}.pkl")
    diagnostics = {
        "samples_written": written,
        "groups_valid": n_valid,
        "groups_repaired_zero_padded": n_repaired,
        "groups_skipped": sum(skip_reasons.values()),
        "skip_reasons": dict(skip_reasons),
        "total_columns": int(sum(n_columns)),
        "total_positive_columns": int(sum(n_positive)),
        "column_count_min": int(min(n_columns)) if n_columns else 0,
        "column_count_mean": float(sum(n_columns) / len(n_columns)) if n_columns else 0.0,
        "column_count_max": int(max(n_columns)) if n_columns else 0,
        "positive_count_min": int(min(n_positive)) if n_positive else 0,
        "positive_count_mean": float(sum(n_positive) / len(n_positive)) if n_positive else 0.0,
        "positive_count_max": int(max(n_positive)) if n_positive else 0,
        "adaptive_k_star_min": float(min(adaptive_k_values)) if adaptive_k_values else None,
        "adaptive_k_star_mean": float(sum(adaptive_k_values) / len(adaptive_k_values)) if adaptive_k_values else None,
        "adaptive_k_star_max": float(max(adaptive_k_values)) if adaptive_k_values else None,
    }
    return written, skipped, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CG teacher CSV rows into train/valid/test graph samples for teacher-supervised BiGAT training."
    )
    parser.add_argument(
        "--teacher-csv",
        default="Results/cg_teacher_dataset.csv",
        help="Teacher rows file. CSV is supported for legacy runs; .pkl.gz is preferred for large Kaggle exports.",
    )
    parser.add_argument("--out-dir", default="GNN/data/irplt_teacher")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=utilities.valid_seed, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--split-by-instance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When multiple source_instance values exist, split at the instance "
             "level so the test set is strictly unseen. Disable only for debugging.",
    )
    parser.add_argument(
        "--max-skip-ratio",
        type=float,
        default=0.30,
        help="Validation gate: raise if (skipped_groups / total_groups) exceeds this. "
             "Set higher for very small smoke runs; default rejects datasets where >30%% "
             "of teacher groups had to be dropped for missing graph features.",
    )
    # ── Ranking-friendly group filter (Fix A) ───────────────────────────
    parser.add_argument(
        "--ranking-filter-groups",
        action="store_true",
        help="Drop signal-less groups and stratified-cap large groups before "
             "writing samples. Recommended for pairwise_rank training.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="When --ranking-filter-groups is on, drop groups with fewer than this "
             "many columns (no pairwise signal possible).",
    )
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=64,
        help="When --ranking-filter-groups is on, cap larger groups via stratified "
             "subsampling (best positives by teacher_score, hardest negatives by "
             "|reduced_cost|). Curbs O(P×N) compute on giant CG branch nodes.",
    )
    parser.add_argument(
        "--require-mixed-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --ranking-filter-groups is on, drop groups with no positives or "
             "no negatives (they contribute zero gradient to pairwise loss).",
    )
    parser.add_argument(
        "--min-groups-after-filter",
        type=int,
        default=200,
        help="Fail-fast: refuse to build if fewer useful groups remain after filtering.",
    )
    parser.add_argument(
        "--positive-rate-min",
        type=float,
        default=0.10,
        help="Fail-fast: refuse to build if positive rate after filter is below this.",
    )
    parser.add_argument(
        "--positive-rate-max",
        type=float,
        default=0.90,
        help="Fail-fast: refuse to build if positive rate after filter is above this.",
    )
    args = parser.parse_args()

    teacher_csv = Path(args.teacher_csv)
    if not teacher_csv.exists():
        raise FileNotFoundError(f"Teacher CSV not found: {teacher_csv}")

    rows = read_teacher_rows(teacher_csv)
    if not rows:
        raise RuntimeError(
            f"No teacher rows found in {teacher_csv}. Run the CG pipeline first so it exports rich teacher rows."
        )

    # Deduplicated teacher exports blank out `constraint_features_json` on all
    # rows except the first of each batch. Fill them back in before grouping
    # so `build_training_sample_from_exported_teacher_rows` has what it needs.
    rows = propagate_constraint_features_json(rows)

    grouped_rows: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[group_key(row)].append(row)

    # ------------------------------------------------------------------
    # Ranking-friendly group filter (Fix A). Runs BEFORE split so the report
    # reflects the dataset that actually feeds the trainer. Physical cap:
    # capped groups lose column nodes from their bipartite graph, but the
    # constraint subgraph is unchanged. Acceptable simplification for the
    # first iteration; revisit with masked-loss-only capping if needed.
    # ------------------------------------------------------------------
    filter_report: Dict[str, Any] = {}
    if args.ranking_filter_groups:
        grouped_rows, filter_report = filter_and_cap_groups(
            grouped_rows,
            min_size=args.min_group_size,
            max_size=args.max_group_size,
            require_mixed=args.require_mixed_labels,
        )
        print_filter_report(filter_report)

        # Fail-fast checks — refuse to build a dataset that won't train well.
        if filter_report["size_stats_out"]["max"] > args.max_group_size:
            raise RuntimeError(
                f"INTERNAL: capped group exceeds max_size "
                f"({filter_report['size_stats_out']['max']} > {args.max_group_size}). "
                "Filter implementation is buggy."
            )
        if filter_report["groups_out"] < args.min_groups_after_filter:
            raise RuntimeError(
                f"After filter only {filter_report['groups_out']} groups remain "
                f"(< {args.min_groups_after_filter}). Loosen filter thresholds or collect more teacher rows."
            )
        rate_out = filter_report["positive_rate_out"]
        if rate_out < args.positive_rate_min or rate_out > args.positive_rate_max:
            raise RuntimeError(
                f"Positive rate after filter = {rate_out:.3f} is outside acceptable range "
                f"[{args.positive_rate_min:.2f}, {args.positive_rate_max:.2f}]. "
                "Check teacher label generation."
            )
        if filter_report["pairwise_pairs_out"] == 0:
            raise RuntimeError(
                "Filter left zero pairwise pairs — every remaining group is unmixed. "
                "Disable --require-mixed-labels or check teacher labels."
            )

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_keys = list(grouped_rows.keys())
    distinct_instances = sorted({key[0] for key in all_keys})

    # ------------------------------------------------------------------
    # Split selection — three modes, in priority order:
    #   1. row-level `dataset_split` (set by generate_teacher_scenarios.py
    #      after a base-level assignment).  This is the only mode that
    #      provably eliminates base-instance leakage.
    #   2. instance-level random split keyed by source_instance, when there
    #      are >= 2 distinct source_instance values and the user opts in.
    #   3. group-level random split (fallback for single-instance smoke runs).
    # ------------------------------------------------------------------
    has_row_split = any(str(row.get("dataset_split") or "").strip() for row in rows)
    if has_row_split:
        split_mode = "row-level (dataset_split field)"
        split_map: Dict[str, List[GroupKey]] = {"train": [], "valid": [], "test": []}
        unknown_split_groups: List[GroupKey] = []
        for key, group in grouped_rows.items():
            tags = {
                str(row.get("dataset_split") or "").strip()
                for row in group
                if str(row.get("dataset_split") or "").strip()
            }
            if not tags:
                unknown_split_groups.append(key)
                continue
            if len(tags) != 1:
                raise RuntimeError(
                    f"Group {key} has rows tagged with multiple dataset_split values "
                    f"{sorted(tags)}; teacher exporter must assign exactly one split per group."
                )
            tag = next(iter(tags))
            if tag not in split_map:
                raise RuntimeError(
                    f"Group {key} has unknown dataset_split={tag!r} "
                    f"(expected one of train/valid/test)"
                )
            split_map[tag].append(key)
        if unknown_split_groups:
            print(
                f"[Teacher Split] {len(unknown_split_groups)} group(s) have no dataset_split tag. "
                "These will be silently dropped — re-run scenario generation to tag them."
            )
    elif args.split_by_instance and len(distinct_instances) > 1:
        split_mode = "instance-level"
        split_map = split_groups(
            all_keys,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            rng=random.Random(args.seed),
            split_by_instance=True,
        )
    else:
        split_mode = "group-level"
        if args.split_by_instance:
            print(
                f"[Teacher Split] Only {len(distinct_instances)} distinct source_instance found. "
                "Falling back to group-level random split. The test split is NOT a strict "
                "generalization check — consider collecting teacher rows from more CG runs."
            )
        split_map = split_groups(
            all_keys,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            rng=random.Random(args.seed),
            split_by_instance=False,
        )

    # ------------------------------------------------------------------
    # Base-instance leakage assertion: every base_dataset_id must appear in
    # exactly one split.  This catches accidental scenario-level splits.
    # ------------------------------------------------------------------
    base_to_split: Dict[str, str] = {}
    leaked_bases: List[str] = []
    for split_name, keys_list in split_map.items():
        bases_in_split = set()
        for key in keys_list:
            for row in grouped_rows[key]:
                base_id = str(row.get("base_dataset_id") or "").strip()
                if base_id:
                    bases_in_split.add(base_id)
        for base_id in bases_in_split:
            prior = base_to_split.get(base_id)
            if prior is not None and prior != split_name:
                leaked_bases.append(f"{base_id}: {prior}+{split_name}")
            else:
                base_to_split[base_id] = split_name
    if leaked_bases:
        raise RuntimeError(
            "Base-instance leakage detected — the following base_dataset_id values "
            f"appear in more than one split: {leaked_bases[:10]}"
            + (f" (and {len(leaked_bases) - 10} more)" if len(leaked_bases) > 10 else "")
        )

    summary: Dict[str, Any] = {
        "teacher_csv": str(teacher_csv),
        "out_dir": str(out_dir),
        "n_raw_rows": len(rows),
        "n_groups": len(grouped_rows),
        "n_distinct_instances": len(distinct_instances),
        "n_distinct_base_ids": len(base_to_split),
        "split_mode": split_mode,
        "group_key_fields": ["source_instance", "branch_node_id", "episode", "product", "period", "constraint_state_hash"],
        "ranking_filter": {
            "enabled": bool(args.ranking_filter_groups),
            **(filter_report if args.ranking_filter_groups else {}),
        },
        "splits": {},
        "skipped_groups": [],
    }
    summary["instances_per_split"] = {
        split_name: sorted({k[0] for k in keys_list})
        for split_name, keys_list in split_map.items()
    }
    summary["base_ids_per_split"] = {
        split_name: sorted({
            str(row.get("base_dataset_id") or "").strip()
            for k in keys_list
            for row in grouped_rows[k]
            if str(row.get("base_dataset_id") or "").strip()
        })
        for split_name, keys_list in split_map.items()
    }
    for split, keys in split_map.items():
        written, skipped, diagnostics = write_samples(grouped_rows, keys, out_dir / split)
        summary["splits"][split] = {"groups": len(keys), **diagnostics}
        summary["skipped_groups"].extend(skipped)

    # ------------------------------------------------------------------
    # Validation gate: refuse to publish a graph dataset that lost too many
    # groups to missing constraint features.  This is the "loud failure"
    # replacing the silent skips that motivated this fix.
    # ------------------------------------------------------------------
    total_groups_assigned = sum(len(keys) for keys in split_map.values())
    total_skipped = sum(item["groups_skipped"] for item in summary["splits"].values())
    skip_ratio = (total_skipped / total_groups_assigned) if total_groups_assigned > 0 else 0.0
    skip_reason_totals: Dict[str, int] = defaultdict(int)
    for item in summary["splits"].values():
        for reason, count in (item.get("skip_reasons") or {}).items():
            skip_reason_totals[reason] += int(count)
    summary["skip_ratio"] = round(skip_ratio, 6)
    summary["skip_reason_totals"] = dict(skip_reason_totals)
    summary["max_skip_ratio_threshold"] = args.max_skip_ratio
    print(
        f"[Teacher Graph] raw_rows={len(rows)} groups={len(grouped_rows)} "
        f"valid={sum(item.get('groups_valid', 0) for item in summary['splits'].values())} "
        f"repaired={sum(item.get('groups_repaired_zero_padded', 0) for item in summary['splits'].values())} "
        f"skipped={total_skipped} skip_ratio={skip_ratio:.4f} "
        f"reasons={dict(skip_reason_totals)}"
    )
    if skip_ratio > args.max_skip_ratio:
        raise RuntimeError(
            f"Graph builder skip ratio {skip_ratio:.4f} exceeds threshold "
            f"{args.max_skip_ratio:.4f}. Skip reason counts: {dict(skip_reason_totals)}. "
            "Re-run teacher generation — most likely the GNN graph builder failed to "
            "produce constraint features for many batches."
        )

    total_written = sum(item["samples_written"] for item in summary["splits"].values())
    if total_written == 0:
        raise RuntimeError(
            "No teacher graph samples were written. The CSV likely came from an older export without graph JSON fields."
        )

    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Mirror the summary into Results/graphs/ so the thesis reporting tree
    # has a single place to inspect graph-dataset stats without digging into
    # GNN/data/. Only runs when IRP_RESULTS_DIR is set (i.e. inside a
    # pipeline run), so standalone invocations stay self-contained.
    import os as _os
    _results_dir = _os.environ.get("IRP_RESULTS_DIR")
    if _results_dir:
        try:
            mirror = Path(_results_dir) / "graphs"
            mirror.mkdir(parents=True, exist_ok=True)
            with open(mirror / "graph_dataset_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Mirrored graph dataset summary to: {mirror / 'graph_dataset_summary.json'}")
        except OSError as exc:
            print(f"[graphs] mirror skipped ({exc})")

    print(json.dumps(summary, indent=2))
    print(f"Wrote teacher-supervised graph dataset to {out_dir}")


if __name__ == "__main__":
    main()
