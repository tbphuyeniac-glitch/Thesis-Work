from __future__ import annotations

import datetime
import gzip
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


COLUMN_FEATURE_NAMES = [
    "reduced_cost",
    "total_flow",
    "n_pairs",
    "total_need_covered",
    "total_surplus_consumed",
    "avg_shortage_ratio",
    "avg_surplus_ratio",
    "avg_time_urgency",
    "avg_negative_reduced_cost",
    "acceptance_score",
    "compensation_mean",
    "column_cost",
]

CONSTRAINT_FEATURE_NAMES = [
    "dual_value",
    "raw_rhs_amount",
    "covered_amount",
    "residual_slack",
    "urgency_or_scarcity",
    "is_need_constraint",
    "is_surplus_constraint",
]

EDGE_FEATURE_NAMES = [
    "quantity",
    "fraction_covered_or_consumed",
    "dual_times_quantity",
]


def log(message: str, logfile: str | Path | None = None) -> None:
    text = f"[{datetime.datetime.now()}] {message}"
    print(text)
    if logfile is not None:
        with open(logfile, "a", encoding="utf-8") as f:
            print(text, file=f)


def valid_seed(seed: int | str) -> int:
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    return seed


def _positive_normal(rng: np.random.Generator, mean: float, std: float, floor: float = 0.0) -> float:
    return float(max(floor, rng.normal(mean, std)))


def make_synthetic_irplt_graph(
    rng: np.random.Generator,
    n_columns: int = 24,
    n_need_constraints: int = 8,
    n_surplus_constraints: int = 8,
    max_pairs_per_column: int = 3,
) -> Dict[str, Any]:
    """Create one synthetic IRP-LT column/constraint graph sample.

    This is useful for smoke tests and optional warm-up only. Its labels are
    heuristic, not solver-derived teacher labels for the final thesis pipeline.
    """
    n_constraints = int(n_need_constraints + n_surplus_constraints)
    need = rng.uniform(5.0, 40.0, size=n_need_constraints)
    surplus = rng.uniform(5.0, 45.0, size=n_surplus_constraints)
    dual_need = rng.uniform(0.1, 3.0, size=n_need_constraints)
    dual_surplus = rng.uniform(0.0, 1.5, size=n_surplus_constraints)

    total_need = max(float(need.sum()), 1e-9)
    total_surplus = max(float(surplus.sum()), 1e-9)
    need_urgency = rng.uniform(0.1, 1.0, size=n_need_constraints)
    surplus_scarcity = 1.0 / (1.0 + surplus / max(float(np.mean(surplus)), 1e-9))

    constraint_features = []
    for idx in range(n_need_constraints):
        covered = float(rng.uniform(0.0, need[idx]))
        constraint_features.append([
            dual_need[idx],
            need[idx],
            covered,
            max(0.0, need[idx] - covered),
            need_urgency[idx],
            1.0,
            0.0,
        ])
    for idx in range(n_surplus_constraints):
        covered = float(rng.uniform(0.0, surplus[idx]))
        constraint_features.append([
            dual_surplus[idx],
            surplus[idx],
            covered,
            max(0.0, surplus[idx] - covered),
            surplus_scarcity[idx],
            0.0,
            1.0,
        ])

    column_features: List[List[float]] = []
    labels: List[float] = []
    edge_rows: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []
    metadata_columns: List[Dict[str, Any]] = []

    for col_idx in range(n_columns):
        n_pairs = int(rng.integers(1, max_pairs_per_column + 1))
        need_ids = rng.choice(n_need_constraints, size=n_pairs, replace=n_pairs > n_need_constraints)
        surplus_ids = rng.choice(n_surplus_constraints, size=n_pairs, replace=n_pairs > n_surplus_constraints)

        total_flow = 0.0
        column_cost = 0.0
        reduced_cost = 0.0
        shortage_ratios = []
        surplus_ratios = []
        urgencies = []
        negative_pair_rcs = []

        for need_id, surplus_id in zip(need_ids, surplus_ids):
            qty = float(min(need[need_id], surplus[surplus_id], rng.uniform(1.0, 18.0)))
            unit_cost = _positive_normal(rng, mean=0.55, std=0.2, floor=0.05)
            fixed_cost = _positive_normal(rng, mean=1.5, std=0.5, floor=0.0) / n_pairs
            dual_score = float(dual_need[need_id] + dual_surplus[surplus_id])
            pair_rc = fixed_cost + unit_cost * qty - dual_score * qty

            total_flow += qty
            column_cost += fixed_cost + unit_cost * qty
            reduced_cost += pair_rc
            shortage_ratios.append(float(need[need_id] / total_need))
            surplus_ratios.append(float(surplus[surplus_id] / total_surplus))
            urgencies.append(float(need_urgency[need_id]))
            negative_pair_rcs.append(max(0.0, -pair_rc))

            edge_rows.append((col_idx, int(need_id)))
            edge_attrs.append([
                qty,
                qty / max(float(need[need_id]), 1e-9),
                float(dual_need[need_id] * qty),
            ])
            edge_rows.append((col_idx, int(n_need_constraints + surplus_id)))
            edge_attrs.append([
                qty,
                qty / max(float(surplus[surplus_id]), 1e-9),
                float(dual_surplus[surplus_id] * qty),
            ])

        acceptance_score = float(np.clip(rng.beta(3.0, 2.0) if reduced_cost < 0 else rng.beta(1.5, 4.0), 0.0, 1.0))
        compensation_mean = _positive_normal(rng, mean=0.2 * total_flow, std=0.2, floor=0.0)
        label = 1.0 if (reduced_cost < 0 and acceptance_score >= 0.45) else 0.0

        column_features.append([
            reduced_cost,
            total_flow,
            float(n_pairs),
            total_flow,
            total_flow,
            float(np.mean(shortage_ratios)),
            float(np.mean(surplus_ratios)),
            float(np.mean(urgencies)),
            float(np.mean(negative_pair_rcs)),
            acceptance_score,
            compensation_mean,
            column_cost,
        ])
        labels.append(label)
        metadata_columns.append({
            "reduced_cost": reduced_cost,
            "column_cost": column_cost,
            "n_pairs": n_pairs,
        })

    return {
        "column_features": np.asarray(column_features, dtype=np.float32),
        "constraint_features": np.asarray(constraint_features, dtype=np.float32),
        "edge_index_col_to_con": np.asarray(edge_rows, dtype=np.int64).T,
        "edge_attr_col_to_con": np.asarray(edge_attrs, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
        "labels_binary": np.asarray(labels, dtype=np.float32),
        "target_score": np.asarray(labels, dtype=np.float32),
        "column_feature_names": COLUMN_FEATURE_NAMES,
        "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "metadata": {
            "generator": "synthetic_irplt",
            "label_source": "synthetic_heuristic",
            "n_columns": int(n_columns),
            "n_constraints": int(n_constraints),
            "columns": metadata_columns,
        },
    }


def build_bigraph_for_patterns(
    patterns: Sequence[Any],
    data: Any,
    need: Dict[Tuple[str, str, int], float],
    surplus: Dict[Tuple[str, str, int], float],
    dual_need: Dict[Tuple[str, str, int], float],
    dual_surplus: Dict[Tuple[str, str, int], float],
    labels: Sequence[float] | None = None,
    target_scores: Sequence[float] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a BiGAT graph from real LTPattern-like objects.

    This is the compatibility point for `irp_gurobi_converted.py`:
    each pattern becomes a column node, and each active receiver/donor
    constraint touched by a pattern becomes a constraint node.
    """
    constraint_key_to_id: Dict[Tuple[str, str, int, str], int] = {}
    constraint_features: List[List[float]] = []
    graph_metadata = metadata or {}

    batch_covered_by_constraint: Dict[Tuple[str, str, str, int], float] = {}
    for pat in patterns:
        product = pat.product
        period = pat.period
        for (donor, receiver), qty_raw in pat.pattern_flows.items():
            qty = max(0.0, float(qty_raw))
            batch_covered_by_constraint[("need", receiver, product, period)] = (
                batch_covered_by_constraint.get(("need", receiver, product, period), 0.0) + qty
            )
            batch_covered_by_constraint[("surplus", donor, product, period)] = (
                batch_covered_by_constraint.get(("surplus", donor, product, period), 0.0) + qty
            )

    def add_constraint(kind: str, store: str, product: str, period: int) -> int:
        key = (kind, store, product, period)
        if key in constraint_key_to_id:
            return constraint_key_to_id[key]
        if kind == "need":
            amount = float(need.get((store, product, period), 0.0))
            dual = float(dual_need.get((store, product, period), 0.0))
            realized_demand = getattr(data, "realized_demand", {})
            forecast_demand = getattr(data, "demand", {})
            demand_now = float(realized_demand.get((store, product, period), forecast_demand.get((store, product, period), 0.0)))
            urgency = 1.0 if amount > 1e-9 else 1.0 / (1.0 + amount / max(demand_now, 1e-9))
            is_need, is_surplus = 1.0, 0.0
        else:
            amount = float(surplus.get((store, product, period), 0.0))
            dual = float(dual_surplus.get((store, product, period), 0.0))
            urgency = 1.0 / (1.0 + amount)
            is_need, is_surplus = 0.0, 1.0
        covered = min(float(batch_covered_by_constraint.get(key, 0.0)), max(amount, 0.0))
        residual = max(0.0, amount - covered)
        constraint_key_to_id[key] = len(constraint_features)
        constraint_features.append([dual, amount, covered, residual, urgency, is_need, is_surplus])
        return constraint_key_to_id[key]

    total_need_by_pt: Dict[Tuple[str, int], float] = {}
    total_surplus_by_pt: Dict[Tuple[str, int], float] = {}
    for (_, product, period), value in need.items():
        total_need_by_pt[(product, period)] = total_need_by_pt.get((product, period), 0.0) + float(value)
    for (_, product, period), value in surplus.items():
        total_surplus_by_pt[(product, period)] = total_surplus_by_pt.get((product, period), 0.0) + float(value)

    column_features: List[List[float]] = []
    edge_rows: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []

    for col_idx, pat in enumerate(patterns):
        product = pat.product
        period = pat.period
        total_need = max(total_need_by_pt.get((product, period), 0.0), 1e-9)
        total_surplus = max(total_surplus_by_pt.get((product, period), 0.0), 1e-9)

        total_flow = 0.0
        total_need_covered = 0.0
        total_surplus_consumed = 0.0
        shortage_ratios = []
        surplus_ratios = []
        urgencies = []
        neg_pair_rc = []
        reduced_cost = 0.0

        for (donor, receiver), qty_raw in pat.pattern_flows.items():
            qty = float(qty_raw)
            if qty <= 1e-9:
                continue
            unit_cost = float(data.ship_cost_lt.get((donor, receiver, product), data.transship_unit_cost.get((donor, receiver), 0.0)))
            fixed_cost = float(data.fixed_dispatch_lt.get((donor, receiver), 0.0)) / max(len(pat.pattern_flows), 1)
            d_need = float(dual_need.get((receiver, product, period), 0.0))
            d_surplus = float(dual_surplus.get((donor, product, period), 0.0))
            pair_rc = fixed_cost + unit_cost * qty - (d_need + d_surplus) * qty

            receiver_need = float(need.get((receiver, product, period), 0.0))
            donor_surplus = float(surplus.get((donor, product, period), 0.0))
            realized_demand = getattr(data, "realized_demand", {})
            forecast_demand = getattr(data, "demand", {})
            demand_now = float(realized_demand.get((receiver, product, period), forecast_demand.get((receiver, product, period), 0.0)))
            days_until_stockout = max(0.0, demand_now - receiver_need) / max(demand_now, 1e-9) if receiver_need > 1e-9 else math.inf
            time_urgency = 1.0 / (1.0 + days_until_stockout)

            total_flow += qty
            total_need_covered += min(qty, receiver_need)
            total_surplus_consumed += min(qty, donor_surplus)
            shortage_ratios.append(receiver_need / total_need)
            surplus_ratios.append(donor_surplus / total_surplus)
            urgencies.append(time_urgency)
            neg_pair_rc.append(max(0.0, -pair_rc))
            reduced_cost += pair_rc

            need_con = add_constraint("need", receiver, product, period)
            edge_rows.append((col_idx, need_con))
            edge_attrs.append([qty, qty / max(receiver_need, 1e-9), d_need * qty])

            surplus_con = add_constraint("surplus", donor, product, period)
            edge_rows.append((col_idx, surplus_con))
            edge_attrs.append([qty, qty / max(donor_surplus, 1e-9), d_surplus * qty])

        column_cost = float(getattr(pat, "column_cost", 0.0))
        pat_metadata = getattr(pat, "metadata", {}) or {}
        column_features.append([
            reduced_cost,
            total_flow,
            float(len(pat.pattern_flows)),
            total_need_covered,
            total_surplus_consumed,
            float(np.mean(shortage_ratios)) if shortage_ratios else 0.0,
            float(np.mean(surplus_ratios)) if surplus_ratios else 0.0,
            float(np.mean(urgencies)) if urgencies else 0.0,
            float(np.mean(neg_pair_rc)) if neg_pair_rc else 0.0,
            float(pat_metadata.get("mean_acceptance_score", pat_metadata.get("acceptance_score", 0.0))),
            float(pat_metadata.get("mean_compensation", pat_metadata.get("compensation", 0.0))),
            column_cost,
        ])

    if labels is None:
        label_values = [1.0 if row[0] < 0.0 else 0.0 for row in column_features]
    else:
        label_values = [float(value) for value in labels]

    if target_scores is None:
        target_values = label_values
    else:
        target_values = [float(value) for value in target_scores]

    return {
        "column_features": np.asarray(column_features, dtype=np.float32),
        "constraint_features": np.asarray(constraint_features, dtype=np.float32),
        "edge_index_col_to_con": np.asarray(edge_rows, dtype=np.int64).T if edge_rows else np.zeros((2, 0), dtype=np.int64),
        "edge_attr_col_to_con": np.asarray(edge_attrs, dtype=np.float32) if edge_attrs else np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
        "labels": np.asarray(label_values, dtype=np.float32),
        "labels_binary": np.asarray(label_values, dtype=np.float32),
        "target_score": np.asarray(target_values, dtype=np.float32),
        "column_feature_names": COLUMN_FEATURE_NAMES,
        "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "metadata": {"generator": "irplt_patterns", "n_columns": len(patterns), **graph_metadata},
    }


def build_training_sample_from_teacher_rows(
    patterns: Sequence[Any],
    data: Any,
    need: Dict[Tuple[str, str, int], float],
    surplus: Dict[Tuple[str, str, int], float],
    dual_need: Dict[Tuple[str, str, int], float],
    dual_surplus: Dict[Tuple[str, str, int], float],
    teacher_rows: Sequence[Dict[str, Any]],
    episode_id: int | str | None = None,
    source_instance: str | None = None,
) -> Dict[str, Any]:
    """Build a graph sample using solver-derived teacher labels/scores.

    `teacher_rows` should contain rows exported by the CG loop, including
    `pattern_id`, `teacher_label` or `selected_in_rmp`, and optionally
    `teacher_score`.
    """
    row_by_pattern = {str(row.get("pattern_id")): row for row in teacher_rows}
    labels = []
    target_scores = []
    for pat in patterns:
        row = row_by_pattern.get(str(getattr(pat, "pattern_id", "")), {})
        if "teacher_label" in row:
            label = float(row.get("teacher_label") or 0.0)
        else:
            label = 1.0 if _truthy(row.get("selected_in_rmp", False)) else 0.0
        labels.append(label)
        target_scores.append(float(row.get("teacher_score", label) or 0.0))

    selected_count = int(sum(1 for value in labels if value > 0.5))
    adaptive_k_values = [
        row.get("adaptive_k_star")
        for row in teacher_rows
        if row.get("adaptive_k_star") is not None
    ]
    metadata = {
        "generator": "cg_teacher_rows",
        "label_source": "teacher_rmp",
        "episode_id": episode_id,
        "source_instance": source_instance,
        "n_teacher_rows": len(teacher_rows),
        "n_selected_columns": selected_count,
        "adaptive_k_star": adaptive_k_values[0] if adaptive_k_values else None,
    }
    return build_bigraph_for_patterns(
        patterns=patterns,
        data=data,
        need=need,
        surplus=surplus,
        dual_need=dual_need,
        dual_surplus=dual_surplus,
        labels=labels,
        target_scores=target_scores,
        metadata=metadata,
    )


def _json_load_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    return json.loads(text)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _float_value(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_training_sample_from_exported_teacher_rows(
    teacher_rows: Sequence[Dict[str, Any]],
    episode_id: int | str | None = None,
    source_instance: str | None = None,
    product: str | None = None,
    period: int | str | None = None,
    branch_node_id: int | str | None = None,
    decision_state_id: str | None = None,
) -> Dict[str, Any]:
    """Build a graph sample from rich CG teacher rows exported to CSV.

    The solver export stores one row per candidate column plus JSON fields for
    the column feature vector, the full constraint feature matrix, and this
    column's incident edge features. That makes the teacher dataset builder
    independent of live Gurobi/IRP objects.
    """
    rows = list(teacher_rows)
    if not rows:
        raise ValueError("teacher_rows is empty")

    def sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
        return (int(_float_value(row.get("column_index"), 10**9)), str(row.get("pattern_id", "")))

    rows = sorted(rows, key=sort_key)
    constraint_features = None
    for row in rows:
        if row.get("constraint_features_json"):
            constraint_features = _json_load_value(row.get("constraint_features_json"), None)
            break
    if constraint_features is None:
        raise ValueError("teacher rows do not contain constraint_features_json")
    if constraint_features and len(constraint_features[0]) != len(CONSTRAINT_FEATURE_NAMES):
        raise ValueError(
            "teacher rows contain stale constraint feature dimension "
            f"{len(constraint_features[0])}; expected {len(CONSTRAINT_FEATURE_NAMES)}. "
            "Regenerate Results/cg_teacher_dataset.csv with the current CG pipeline."
        )

    column_features: List[List[float]] = []
    labels: List[float] = []
    target_scores: List[float] = []
    edge_rows: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []
    metadata_columns: List[Dict[str, Any]] = []

    for col_idx, row in enumerate(rows):
        if not row.get("column_features_json"):
            raise ValueError(f"teacher row for pattern {row.get('pattern_id')} is missing column_features_json")
        column_features.append([float(value) for value in _json_load_value(row.get("column_features_json"), [])])
        constraint_ids = _json_load_value(row.get("edge_constraint_indices_json"), [])
        attrs = _json_load_value(row.get("edge_attrs_json"), [])
        if len(constraint_ids) != len(attrs):
            raise ValueError(f"edge JSON length mismatch for pattern {row.get('pattern_id')}")
        for constraint_id, edge_attr in zip(constraint_ids, attrs):
            edge_rows.append((col_idx, int(constraint_id)))
            edge_attrs.append([float(value) for value in edge_attr])

        if "teacher_label" in row and str(row.get("teacher_label", "")).strip() != "":
            label = _float_value(row.get("teacher_label"), 0.0)
        else:
            label = 1.0 if _truthy(row.get("selected_in_rmp")) else 0.0
        labels.append(label)
        target_scores.append(_float_value(row.get("teacher_score"), label))
        metadata_columns.append({
            "pattern_id": row.get("pattern_id"),
            "product": row.get("product"),
            "period": row.get("period"),
            "reduced_cost": _float_value(row.get("reduced_cost"), 0.0),
            "gnn_prob": _float_value(row.get("gnn_prob"), 0.0),
            "combined_score": _float_value(row.get("combined_score"), 0.0),
            "selected_by_gnn": _truthy(row.get("selected_by_gnn")),
            "selected_by_classical_fallback": _truthy(row.get("selected_by_classical_fallback")),
            "passed_to_rmp": _truthy(row.get("passed_to_rmp")),
        })

    return {
        "column_features": np.asarray(column_features, dtype=np.float32),
        "constraint_features": np.asarray(constraint_features, dtype=np.float32),
        "edge_index_col_to_con": np.asarray(edge_rows, dtype=np.int64).T if edge_rows else np.zeros((2, 0), dtype=np.int64),
        "edge_attr_col_to_con": np.asarray(edge_attrs, dtype=np.float32) if edge_attrs else np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
        "labels_binary": np.asarray(labels, dtype=np.float32),
        "target_score": np.asarray(target_scores, dtype=np.float32),
        "column_feature_names": COLUMN_FEATURE_NAMES,
        "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "metadata": {
            "generator": "cg_teacher_rows_export",
            "label_source": "teacher_rmp",
            "episode_id": episode_id,
            "source_instance": source_instance,
            "product": product,
            "period": period,
            "branch_node_id": branch_node_id,
            "decision_state_id": decision_state_id,
            "n_teacher_rows": len(rows),
            "n_selected_columns": int(sum(1 for value in labels if value > 0.5)),
            "columns": metadata_columns,
        },
    }


def adaptive_select_indices(
    scores: Sequence[float],
    *,
    tie_breaker: Sequence[float] | None = None,
    selection_mode: str = "cumulative_mass",
    mass_threshold: float = 0.80,
    relative_threshold: float = 0.85,
    min_keep: int = 1,
    max_keep: int | None = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Shared adaptive self-defined-k selector for runtime and offline tests."""
    values = [max(0.0, float(value)) for value in scores]
    if not values:
        return [], {
            "selection_mode": selection_mode,
            "adaptive_k": 0,
            "ordered_indices": [],
            "selected_indices": [],
            "score_mass_total": 0.0,
        }

    tie = [float(value) for value in tie_breaker] if tie_breaker is not None else values
    ordered = sorted(range(len(values)), key=lambda idx: (values[idx], tie[idx]), reverse=True)
    ordered_scores = [values[idx] for idx in ordered]
    if sum(ordered_scores) <= 1e-12:
        ordered_scores = [1.0 for _ in ordered]

    if selection_mode == "relative_threshold":
        best = max(ordered_scores) if ordered_scores else 0.0
        cutoff = best * float(relative_threshold)
        k_star = sum(1 for score in ordered_scores if score >= cutoff)
    elif selection_mode == "cumulative_mass":
        total = max(sum(ordered_scores), 1e-12)
        cumulative = 0.0
        k_star = 0
        for score in ordered_scores:
            cumulative += score / total
            k_star += 1
            if cumulative >= float(mass_threshold):
                break
    else:
        raise ValueError(f"unknown selection_mode={selection_mode!r}")

    k_star = max(int(min_keep), int(k_star))
    if max_keep is not None:
        k_star = min(k_star, int(max_keep))
    k_star = min(k_star, len(ordered))
    selected = ordered[:k_star]
    return selected, {
        "selection_mode": selection_mode,
        "adaptive_k": int(k_star),
        "mass_threshold": float(mass_threshold),
        "relative_threshold": float(relative_threshold),
        "score_mass_total": float(sum(ordered_scores)),
        "ordered_indices": [int(idx) for idx in ordered],
        "selected_indices": [int(idx) for idx in selected],
    }


def save_graph_sample(sample: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wb") as f:
        pickle.dump(sample, f)


def load_graph_sample(path: str | Path) -> Dict[str, torch.Tensor]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        sample = pickle.load(f)
    return graph_to_tensors(sample)


def graph_to_tensors(sample: Dict[str, Any], device: str | torch.device | None = None) -> Dict[str, torch.Tensor]:
    graph = {
        "column_features": torch.as_tensor(sample["column_features"], dtype=torch.float32, device=device),
        "constraint_features": torch.as_tensor(sample["constraint_features"], dtype=torch.float32, device=device),
        "edge_index_col_to_con": torch.as_tensor(sample["edge_index_col_to_con"], dtype=torch.long, device=device),
        "edge_attr_col_to_con": torch.as_tensor(sample["edge_attr_col_to_con"], dtype=torch.float32, device=device),
        "labels": torch.as_tensor(sample["labels"], dtype=torch.float32, device=device),
    }
    if "labels_binary" in sample:
        graph["labels_binary"] = torch.as_tensor(sample["labels_binary"], dtype=torch.float32, device=device)
    if "target_score" in sample:
        graph["target_score"] = torch.as_tensor(sample["target_score"], dtype=torch.float32, device=device)
    return graph


def iter_sample_files(root: str | Path, split: str) -> List[Path]:
    split_dir = Path(root) / split
    return sorted(split_dir.glob("sample_*.pkl")) + sorted(split_dir.glob("sample_*.pkl.gz"))


def load_split(root: str | Path, split: str, device: str | torch.device | None = None) -> List[Dict[str, torch.Tensor]]:
    return [load_graph_sample(path) if device is None else graph_to_tensors(load_raw_sample(path), device) for path in iter_sample_files(root, split)]


def load_raw_sample(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def normalize_dataset(
    samples: Sequence[Dict[str, torch.Tensor]],
    stats: Dict[str, torch.Tensor] | None = None,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    if stats is None:
        col = torch.cat([s["column_features"] for s in samples], dim=0)
        con = torch.cat([s["constraint_features"] for s in samples], dim=0)
        edge = torch.cat([s["edge_attr_col_to_con"] for s in samples], dim=0)
        stats = {
            "column_mean": col.mean(dim=0),
            "column_std": col.std(dim=0, unbiased=False).clamp_min(1e-6),
            "constraint_mean": con.mean(dim=0),
            "constraint_std": con.std(dim=0, unbiased=False).clamp_min(1e-6),
            "edge_mean": edge.mean(dim=0),
            "edge_std": edge.std(dim=0, unbiased=False).clamp_min(1e-6),
        }

    normalized = []
    for sample in samples:
        item = dict(sample)
        item["column_features"] = (item["column_features"] - stats["column_mean"]) / stats["column_std"]
        item["constraint_features"] = (item["constraint_features"] - stats["constraint_mean"]) / stats["constraint_std"]
        item["edge_attr_col_to_con"] = (item["edge_attr_col_to_con"] - stats["edge_mean"]) / stats["edge_std"]
        normalized.append(item)
    return normalized, stats


def topk_accuracy(scores: torch.Tensor, labels: torch.Tensor, ks: Iterable[int] = (1, 3, 5)) -> Dict[str, float]:
    positives = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    if positives.numel() == 0:
        return {f"top{k}_hit": math.nan for k in ks}
    out = {}
    for k in ks:
        k_eff = min(int(k), scores.numel())
        top_idx = torch.topk(scores, k_eff).indices
        hit = torch.isin(top_idx, positives).any().item()
        out[f"top{k}_hit"] = float(hit)
    return out


def binary_metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Secondary threshold metrics for classification-style diagnostics.

    Pairwise ranking runs should primarily use ranking metrics such as MRR,
    mean positive rank, and top-k hits; these binary metrics remain useful as
    auxiliary checks but depend on the arbitrary sigmoid threshold.
    """
    probs = torch.sigmoid(scores)
    preds = probs >= threshold
    gold = labels > 0.5
    tp = torch.logical_and(preds, gold).sum().item()
    fp = torch.logical_and(preds, ~gold).sum().item()
    fn = torch.logical_and(~preds, gold).sum().item()
    tn = torch.logical_and(~preds, ~gold).sum().item()
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
