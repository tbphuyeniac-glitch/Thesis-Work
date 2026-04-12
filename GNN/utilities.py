from __future__ import annotations

import datetime
import gzip
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
    "remaining_amount",
    "slack",
    "urgency_or_scarcity",
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
    """Create one trainable IRP-LT column/constraint graph sample.

    This mirrors the thesis design without calling the optimizer:
    column nodes are LT candidate patterns, constraint nodes are need/surplus
    constraints, and labels mark economically promising columns.
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
        constraint_features.append([
            dual_need[idx],
            need[idx],
            need[idx],
            need_urgency[idx],
        ])
    for idx in range(n_surplus_constraints):
        constraint_features.append([
            dual_surplus[idx],
            surplus[idx],
            surplus[idx],
            surplus_scarcity[idx],
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
        "column_feature_names": COLUMN_FEATURE_NAMES,
        "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "metadata": {
            "generator": "synthetic_irplt",
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
) -> Dict[str, Any]:
    """Build a BiGAT graph from real LTPattern-like objects.

    This is the compatibility point for `irp_gurobi_converted.py`:
    each pattern becomes a column node, and each active receiver/donor
    constraint touched by a pattern becomes a constraint node.
    """
    constraint_key_to_id: Dict[Tuple[str, str, int, str], int] = {}
    constraint_features: List[List[float]] = []

    def add_constraint(kind: str, store: str, product: str, period: int) -> int:
        key = (kind, store, product, period)
        if key in constraint_key_to_id:
            return constraint_key_to_id[key]
        if kind == "need":
            amount = float(need.get((store, product, period), 0.0))
            dual = float(dual_need.get((store, product, period), 0.0))
            demand_now = float(getattr(data, "demand", {}).get((store, product, period), 0.0))
            urgency = 1.0 if amount > 1e-9 else 1.0 / (1.0 + amount / max(demand_now, 1e-9))
        else:
            amount = float(surplus.get((store, product, period), 0.0))
            dual = float(dual_surplus.get((store, product, period), 0.0))
            urgency = 1.0 / (1.0 + amount)
        constraint_key_to_id[key] = len(constraint_features)
        constraint_features.append([dual, amount, amount, urgency])
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
            demand_now = float(getattr(data, "demand", {}).get((receiver, product, period), 0.0))
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
        metadata = getattr(pat, "metadata", {}) or {}
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
            float(metadata.get("mean_acceptance_score", metadata.get("acceptance_score", 0.0))),
            float(metadata.get("mean_compensation", metadata.get("compensation", 0.0))),
            column_cost,
        ])

    if labels is None:
        label_values = [1.0 if row[0] < 0.0 else 0.0 for row in column_features]
    else:
        label_values = [float(value) for value in labels]

    return {
        "column_features": np.asarray(column_features, dtype=np.float32),
        "constraint_features": np.asarray(constraint_features, dtype=np.float32),
        "edge_index_col_to_con": np.asarray(edge_rows, dtype=np.int64).T if edge_rows else np.zeros((2, 0), dtype=np.int64),
        "edge_attr_col_to_con": np.asarray(edge_attrs, dtype=np.float32) if edge_attrs else np.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=np.float32),
        "labels": np.asarray(label_values, dtype=np.float32),
        "column_feature_names": COLUMN_FEATURE_NAMES,
        "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "metadata": {"generator": "irplt_patterns", "n_columns": len(patterns)},
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
            "column_std": col.std(dim=0).clamp_min(1e-6),
            "constraint_mean": con.mean(dim=0),
            "constraint_std": con.std(dim=0).clamp_min(1e-6),
            "edge_mean": edge.mean(dim=0),
            "edge_std": edge.std(dim=0).clamp_min(1e-6),
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
