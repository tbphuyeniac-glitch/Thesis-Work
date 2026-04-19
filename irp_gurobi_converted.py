"""
IRP / IRPT thesis prototype with validation split
=================================================

What this file does
-------------------
- Keeps the user's original dataset mapper structure and validation flow.
- Upgrades the baseline model into an Achamrah-style IRPT model.
- Preserves old fields and adds extra fields needed for route / vehicle / LT modeling.
- Includes:
  - base formulation (2)–(15)
  - valid inequalities (16)–(20)
  - LT column generation with explicit RMP, pricing, dual values, and re-optimization loop
- Does NOT fully implement disjoint path inequalities (21) as true branch-and-cut,
  because this prototype does not implement true callback-based disjoint path cuts.

Notes
-----
- To keep the code practical, product-flow variables are continuous by default.
  Set enforce_integer_flows=True for a smaller test if needed.
- If the model becomes heavy, reduce store_limit / sku_limit / vehicle_count.

Dependencies
------------
pip install pandas openpyxl gurobipy
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import pprint
import subprocess
import sys
import time
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    raise ImportError("Install gurobipy first: pip install gurobipy") from e

GLOBAL_GUROBI_ENV: Optional[gp.Env] = None


def create_gurobi_env() -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("WLSAccessID", os.environ["WLSACCESSID"])
    env.setParam("WLSSecret", os.environ["WLSSECRET"])
    env.setParam("LicenseID", int(os.environ["LICENSEID"]))
    env.start()
    return env


def get_gurobi_env() -> gp.Env:
    global GLOBAL_GUROBI_ENV
    if GLOBAL_GUROBI_ENV is None:
        GLOBAL_GUROBI_ENV = create_gurobi_env()
    return GLOBAL_GUROBI_ENV

Store = str
Product = str
Period = int
Node = str
Vehicle = str


def _grb_status_name(status: int) -> str:
    mapping = {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.UNBOUNDED: "Unbounded",
        GRB.INF_OR_UNBD: "InfOrUnbd",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.INTERRUPTED: "Interrupted",
        GRB.SUBOPTIMAL: "Suboptimal",
        GRB.NUMERIC: "Numeric",
    }
    return mapping.get(status, f"Status_{status}")


def _has_solution(model: gp.Model) -> bool:
    try:
        return model.SolCount > 0
    except Exception:
        return False


def _safe_obj_value(model: gp.Model) -> float:
    return float(model.ObjVal) if _has_solution(model) else math.inf


def _model_efficiency_metrics(model: gp.Model) -> Dict[str, float]:
    node_count = float(getattr(model, "NodeCount", 0.0))
    is_mip = bool(getattr(model, "IsMIP", 0))
    return {
        "gurobi_runtime_seconds": float(getattr(model, "Runtime", 0.0)),
        "lp_iterations": float(getattr(model, "IterCount", 0.0)),
        "barrier_iterations": float(getattr(model, "BarIterCount", 0.0)),
        "nodes_explored": node_count,
        "lp_relaxations_solved_estimate": node_count if is_mip else 1.0,
    }


def _add_efficiency_metrics(target: Dict[str, float], source: Dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = float(target.get(key, 0.0)) + float(value)


def print_efficiency_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n[{title}]")
    if not metrics:
        print("  No efficiency metrics available.")
        return
    for key in [
        "gurobi_runtime_seconds",
        "lp_iterations",
        "barrier_iterations",
        "nodes_explored",
        "lp_relaxations_solved_estimate",
        "rmp_solves",
        "branch_price_nodes_explored",
        "branch_price_nodes_remaining",
    ]:
        if key in metrics:
            print(f"  {key}: {float(metrics[key]):.6f}")


def _safe_var_value(model: gp.Model, var: gp.Var) -> float:
    return float(var.X) if var is not None and _has_solution(model) else 0.0


DEFAULT_GNN_CHECKPOINT = "GNN/trained_models/irplt_teacher/bigat/pairwise_rank/best_model.pt"
RESULTS_DIR = Path(__file__).resolve().parent / "Results"


def _project_path(path: str) -> Path:
    value = Path(str(path).strip()).expanduser()
    if value.is_absolute():
        return value
    return Path(__file__).resolve().parent / value


def load_gnn_training_history(checkpoint_path: str = DEFAULT_GNN_CHECKPOINT) -> List[Dict[str, Any]]:
    history_path = _project_path(checkpoint_path).parent / "training_history.json"
    if not history_path.exists():
        return []
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_gnn_training_history(history: List[Dict[str, Any]], checkpoint_path: str = DEFAULT_GNN_CHECKPOINT) -> None:
    print("\n[GNN Training Loss By Episode]")
    if not history:
        print("  No GNN training history found. Build teacher graphs, then run GNN/03_train_bigat.py for final training.")
        return
    for row in history:
        print(
            f"  epoch={int(row.get('epoch', row.get('episode', 0))):03d} "
            f"| train_loss={float(row.get('train_loss', math.nan)):.6f} "
            f"| valid_loss={float(row.get('valid_loss', math.nan)):.6f} "
            f"| valid_f1={float(row.get('valid_f1', math.nan)):.6f} "
            f"| valid_top1={float(row.get('valid_top1', math.nan)):.6f}"
        )
    chart_path = _project_path(checkpoint_path).parent / "training_loss_curve.png"
    if chart_path.exists():
        print(f"  loss_chart={chart_path}")


def run_teacher_graph_and_gnn_training(
    teacher_csv_path: str,
    build_graphs: bool = True,
    train_gnn: bool = True,
    train_epochs: int = 5,
    resume_checkpoint: bool = False,
    checkpoint_path: str = DEFAULT_GNN_CHECKPOINT,
) -> None:
    teacher_csv = _project_path(teacher_csv_path)
    if not teacher_csv.exists() or teacher_csv.stat().st_size <= 1:
        print(f"[Teacher/GNN] No non-empty teacher CSV found; skip graph/GNN update: {teacher_csv}")
        return

    if build_graphs:
        print("\n[Teacher Graph Dataset Update]")
        subprocess.run(
            [
                sys.executable,
                "GNN/build_teacher_graph_dataset.py",
                "--teacher-csv",
                str(teacher_csv),
                "--out-dir",
                "GNN/data/irplt_teacher",
                "--overwrite",
            ],
            cwd=str(Path(__file__).resolve().parent),
            check=True,
        )

    if train_gnn:
        cmd = [
            sys.executable,
            "GNN/03_train_bigat.py",
            "--data-dir",
            "GNN/data/irplt_teacher",
            "--dataset-type",
            "teacher",
            "--epochs",
            str(int(train_epochs)),
            "--objective",
            "pairwise_rank",
        ]
        checkpoint = _project_path(checkpoint_path)
        if resume_checkpoint and checkpoint.exists():
            cmd.extend(["--resume-checkpoint", str(checkpoint)])
        print("\n[Teacher BiGAT Training Update]")
        subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), check=True)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IRPData:
    periods: List[Period]
    stores: List[Store]
    products: List[Product]
    warehouse: str = "CW"

    demand: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)
    realized_demand: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)
    init_inventory_store: Dict[Tuple[Store, Product], float] = field(default_factory=dict)
    init_inventory_wh: Dict[Product, float] = field(default_factory=dict)

    max_inventory_store: Dict[Tuple[Store, Product], float] = field(default_factory=dict)
    max_inventory_wh: Dict[Product, float] = field(default_factory=dict)

    holding_cost_store: Dict[Tuple[Store, Product], float] = field(default_factory=dict)
    holding_cost_wh: Dict[Product, float] = field(default_factory=dict)
    shortage_cost: Dict[Tuple[Store, Product], float] = field(default_factory=dict)

    ship_cost_cw: Dict[Tuple[Store, Product], float] = field(default_factory=dict)
    ship_cost_lt: Dict[Tuple[Store, Store, Product], float] = field(default_factory=dict)

    fixed_dispatch_cw: Dict[Store, float] = field(default_factory=dict)
    fixed_dispatch_lt: Dict[Tuple[Store, Store], float] = field(default_factory=dict)

    big_m_cw: Dict[Tuple[Store, Product], float] = field(default_factory=dict)
    big_m_lt: Dict[Tuple[Store, Store, Product], float] = field(default_factory=dict)

    # ===== Added for Achamrah-style IRPT =====
    vehicles: List[Vehicle] = field(default_factory=list)
    vehicle_capacity: float = 120.0
    vehicle_fixed_cost: float = 0.0
    max_vehicles_used: int = 3
    alpha: float = 1.0

    # replenishment to warehouse g_{p,t}
    replenishment_wh: Dict[Tuple[Product, Period], float] = field(default_factory=dict)

    # aggregate node capacity C_i in paper
    node_capacity: Dict[Node, float] = field(default_factory=dict)

    # routing distance / cost base d_{i,j}
    distance: Dict[Tuple[Node, Node], float] = field(default_factory=dict)

    # LT unit cost b_{i,j}
    transship_unit_cost: Dict[Tuple[Store, Store], float] = field(default_factory=dict)

    # State after forecast-based DC shipment and hidden realized-demand shock.
    post_shock_inventory: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)
    post_shock_shortage: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)


@dataclass
class BaselineIRPSolution:
    status: str
    objective: float
    ship_cw: Dict[Tuple[Store, Product, Period], float]
    activate_cw: Dict[Tuple[Store, Period], int]
    inv_store: Dict[Tuple[Store, Product, Period], float]
    inv_wh: Dict[Tuple[Product, Period], float]
    shortage: Dict[Tuple[Store, Product, Period], float]

    def summary(self) -> Dict:
        return {
            "status": self.status,
            "objective": self.objective,
            "total_ship_from_cw": sum(self.ship_cw.values()),
            "total_shortage": sum(self.shortage.values()),
        }


@dataclass
class FullIRPTSolution:
    status: str
    objective: float

    direct_ship_q: Dict[Tuple[Store, Product, Period], float]
    inv_store: Dict[Tuple[Store, Product, Period], float]
    inv_wh: Dict[Tuple[Product, Period], float]
    shortage: Dict[Tuple[Store, Product, Period], float]

    x: Dict[Tuple[Node, Node, Vehicle, Period], int]
    u: Dict[Tuple[Vehicle, Period], int]
    z: Dict[Tuple[Node, Vehicle, Period], int]
    q: Dict[Tuple[Product, Node, Node, Vehicle, Period], float]
    y: Dict[Tuple[Store, Store, Product, Vehicle, Period], float]
    deliv: Dict[Tuple[Store, Product, Vehicle, Period], float] = field(default_factory=dict)
    load: Dict[Tuple[Node, Vehicle, Period], float] = field(default_factory=dict)
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict:
        return {
            "status": self.status,
            "objective": self.objective,
            "total_direct_shipments": sum(self.direct_ship_q.values()),
            "total_shortage": sum(self.shortage.values()),
            "total_transshipment": sum(self.y.values()),
            "active_route_arcs": sum(self.x.values()),
            "vehicles_used": sum(self.u.values()),
            "efficiency_metrics": self.efficiency_metrics,
        }


@dataclass
class LTPattern:
    pattern_id: str
    period: Period
    product: Product
    pattern_flows: Dict[Tuple[Store, Store], float]
    column_cost: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class StackelbergParams:
    donor_accept_threshold: float = 0.0
    receiver_accept_threshold: float = 0.0

    donor_risk_weight: float = 1.2
    donor_ship_burden_weight: float = 1.0
    donor_service_loss_weight: float = 1.0

    receiver_shortage_reduction_weight: float = 2.0
    receiver_service_gain_weight: float = 1.0
    receiver_handling_weight: float = 0.5

    min_compensation: float = 0.0
    compensation_cap: float = 999999.0
    acceptance_score_weight: float = 0.6
    economic_score_weight: float = 0.4
    top_k_after_game_per_feature: int = 5
    allow_pricing_fallback_when_no_acceptance: bool = True
    fallback_top_k_after_game_per_feature: int = 5
    fallback_requires_negative_reduced_cost: bool = True


@dataclass
class StackelbergDecision:
    accepted: bool
    compensation: float
    donor_utility: float
    receiver_utility: float
    acceptance_score: float
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class CGSolution:
    status: str
    objective: float
    lambda_values: Dict[str, float]
    selected_patterns: List[str]
    implied_net_lt: Dict[Tuple[Store, Product, Period], float]
    dual_need: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)
    dual_surplus: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)
    active_product_periods: Set[Tuple[Product, Period]] = field(default_factory=set)
    iterations_run: int = 0
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    branch_summary: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict:
        payload = {
            "status": self.status,
            "objective": self.objective,
            "selected_patterns": self.selected_patterns,
            "n_selected_patterns": len(self.selected_patterns),
            "n_active_product_periods": len(self.active_product_periods),
            "iterations_run": self.iterations_run,
            "efficiency_metrics": self.efficiency_metrics,
        }
        if self.branch_summary:
            payload["branch_summary"] = self.branch_summary
        return payload


# ============================================================================
# DATA MAPPER
# ============================================================================

class DatasetToIRPValidationMapper:
    REQUIRED_COLUMNS = [
        "SITE_NAME", "NORMAL_PRICE", "ART_SV_NAME_ENG",
        "SALE_QTY", "END_QTY", "PERIOD"
    ]

    def __init__(
        self,
        excel_path: str,
        sheet_name: Optional[str] = None,
        store_limit: Optional[int] = None,
        sku_limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.excel_path = str(excel_path).strip()
        self.sheet_name = sheet_name
        self.store_limit = store_limit
        self.sku_limit = sku_limit
        self.start_date = start_date
        self.end_date = end_date

    def load_raw(self) -> pd.DataFrame:
        input_path = _project_path(self.excel_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {input_path}")

        suffix = input_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(input_path)
        elif suffix in {".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"}:
            df = pd.read_excel(input_path, sheet_name=self.sheet_name or 0)
        else:
            raise ValueError(
                f"Unsupported dataset file extension '{suffix}'. "
                "Use .csv, .xls, .xlsx, .xlsm, .xlsb, or .ods."
            )

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[self.REQUIRED_COLUMNS].copy()
        df = df.rename(columns={
            "SITE_NAME": "store",
            "NORMAL_PRICE": "price",
            "ART_SV_NAME_ENG": "sku",
            "SALE_QTY": "sale_qty",
            "END_QTY": "end_qty",
            "PERIOD": "period_raw",
        })

        df["store"] = df["store"].astype(str).str.strip()
        df["sku"] = df["sku"].astype(str).str.strip()
        df["sale_qty"] = pd.to_numeric(df["sale_qty"], errors="coerce").fillna(0.0)
        df["end_qty"] = pd.to_numeric(df["end_qty"], errors="coerce").fillna(0.0)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["period_date"] = pd.to_datetime(df["period_raw"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["period_date"]).copy()

        if self.start_date is not None:
            df = df[df["period_date"] >= pd.to_datetime(self.start_date)]
        if self.end_date is not None:
            df = df[df["period_date"] <= pd.to_datetime(self.end_date)]

        return df

    def preprocess(self) -> pd.DataFrame:
        df = self.load_raw()
        grp = (
            df.groupby(["store", "sku", "period_date"], as_index=False)
              .agg(
                  sale_qty=("sale_qty", "sum"),
                  end_qty=("end_qty", "sum"),
                  price=("price", "median"),
              )
        )

        if self.store_limit is not None:
            top_stores = (
                grp.groupby("store")["sale_qty"].sum()
                .sort_values(ascending=False)
                .head(self.store_limit)
                .index.tolist()
            )
            grp = grp[grp["store"].isin(top_stores)].copy()

        if self.sku_limit is not None:
            top_skus = (
                grp.groupby("sku")["sale_qty"].sum()
                .sort_values(ascending=False)
                .head(self.sku_limit)
                .index.tolist()
            )
            grp = grp[grp["sku"].isin(top_skus)].copy()

        grp = grp.sort_values(["store", "sku", "period_date"]).reset_index(drop=True)
        unique_dates = sorted(grp["period_date"].drop_duplicates().tolist())
        date_to_period = {dt: i + 1 for i, dt in enumerate(unique_dates)}
        grp["period"] = grp["period_date"].map(date_to_period)
        return grp

    def _populate_routing_distances(
        self,
        data: IRPData,
        stores: List[Store],
        distance_matrix_path: Optional[str],
    ) -> None:
        all_nodes = [data.warehouse] + stores
        distance_lookup: Dict[Tuple[Node, Node], float] = {}
        matrix_depot = "South Ambient DC"

        if distance_matrix_path:
            matrix_path = _project_path(distance_matrix_path)
            if matrix_path.exists():
                matrix_df = pd.read_csv(matrix_path)
                node_col = matrix_df.columns[0]
                matrix_df[node_col] = matrix_df[node_col].astype(str).str.strip()
                matrix_nodes = [str(c).strip() for c in matrix_df.columns[1:]]
                for _, row in matrix_df.iterrows():
                    src = str(row[node_col]).strip()
                    for dst in matrix_nodes:
                        value = pd.to_numeric(row.get(dst), errors="coerce")
                        if pd.notna(value):
                            distance_lookup[(src, dst)] = float(value)
                depot_candidates = [n for n in matrix_df[node_col].tolist() if "DC" in str(n).upper()]
                if depot_candidates:
                    matrix_depot = str(depot_candidates[0]).strip()

        def matrix_name(node: Node) -> Node:
            return matrix_depot if node == data.warehouse else node

        def fallback_distance(i: Node, j: Node) -> float:
            if i == j:
                return 0.0
            i_idx = 0 if i == data.warehouse else stores.index(i) + 1
            j_idx = 0 if j == data.warehouse else stores.index(j) + 1
            if i == data.warehouse or j == data.warehouse:
                return round(10.0 + 1.7 * max(i_idx, j_idx), 1)
            return round(6.0 + 2.3 * abs(i_idx - j_idx) + 0.4 * ((i_idx + j_idx) % 5), 1)

        for i in all_nodes:
            for j in all_nodes:
                src = matrix_name(i)
                dst = matrix_name(j)
                data.distance[(i, j)] = distance_lookup.get((src, dst), fallback_distance(i, j))

    def build_irp_data(
        self,
        wh_inventory_multiplier: float = 2.5,
        store_capacity_multiplier: float = 1.5,
        shortage_cost_rate: float = 0.25,
        holding_cost_rate: float = 0.01,
        cw_ship_cost_flat: float = 1.0,
        lt_ship_cost_flat: float = 0.6,
        fixed_dispatch_cw: float = 8.0,
        fixed_dispatch_lt: float = 2.0,
        vehicle_count: int = 2,
        vehicle_capacity: float = 120.0,
        vehicle_fixed_cost: float = 50.0,
        alpha: float = 1.0,
        cw_replenishment_factor: float = 0.6,
        cw_capacity_factor: float = 2.0,
        distance_matrix_path: Optional[str] = "Distance data/mm_megamarket_distance_matrix_clean.csv",
        store_initial_inventory_multiplier: float = 1.0,
    ):
        df = self.preprocess()

        stores = sorted(df["store"].unique().tolist())
        products = sorted(df["sku"].unique().tolist())
        periods = sorted(df["period"].unique().tolist())
        data = IRPData(periods=periods, stores=stores, products=products)

        full_index = pd.MultiIndex.from_product(
            [stores, products, periods], names=["store", "sku", "period"]
        )
        base = (
            df.set_index(["store", "sku", "period"])[["sale_qty", "end_qty", "price", "period_date"]]
              .reindex(full_index)
              .reset_index()
        )
        base["sale_qty"] = base["sale_qty"].fillna(0.0)
        base["end_qty"] = base["end_qty"].fillna(0.0)

        sku_price = base.groupby("sku")["price"].median()
        global_price = float(base["price"].median()) if base["price"].notna().any() else 1.0
        base["price"] = base.apply(
            lambda r: sku_price.get(r["sku"], global_price) if pd.isna(r["price"]) else r["price"],
            axis=1
        )
        base["price"] = base["price"].fillna(global_price)

        # Demand from sales for all days
        for _, row in base.iterrows():
            s, p, t = row["store"], row["sku"], int(row["period"])
            data.demand[(s, p, t)] = float(row["sale_qty"])
            data.realized_demand[(s, p, t)] = float(row["sale_qty"])

        # ONLY first-day END_QTY as initial inventory
        first_period = min(periods)
        first_df = base[base["period"] == first_period].copy()
        store_initial_inventory_multiplier = max(0.0, float(store_initial_inventory_multiplier))
        for _, row in first_df.iterrows():
            s, p = row["store"], row["sku"]
            data.init_inventory_store[(s, p)] = max(0.0, store_initial_inventory_multiplier * float(row["end_qty"]))

        for s, p in itertools.product(stores, products):
            data.init_inventory_store.setdefault((s, p), 0.0)

        # Capacity from first-day init and historical maxima only as rough cap proxy
        for s, p in itertools.product(stores, products):
            obs = base[(base["store"] == s) & (base["sku"] == p)]["end_qty"]
            obs_max = float(obs.max()) if not obs.empty else 0.0
            init_inv = data.init_inventory_store[(s, p)]
            data.max_inventory_store[(s, p)] = max(5.0, store_capacity_multiplier * max(obs_max, init_inv, 1.0))

        total_demand_by_sku = base.groupby("sku")["sale_qty"].sum().to_dict()
        for p in products:
            total_dem = float(total_demand_by_sku.get(p, 0.0))
            data.init_inventory_wh[p] = max(0.0, wh_inventory_multiplier * total_dem)
            data.max_inventory_wh[p] = max(data.init_inventory_wh[p], 1.2 * data.init_inventory_wh[p])

        price_by_store_sku = base.groupby(["store", "sku"])["price"].median().to_dict()
        for s, p in itertools.product(stores, products):
            price = float(price_by_store_sku.get((s, p), global_price))
            data.holding_cost_store[(s, p)] = max(0.05, holding_cost_rate * price)
            data.shortage_cost[(s, p)] = max(1.0, shortage_cost_rate * price)
            data.ship_cost_cw[(s, p)] = cw_ship_cost_flat
            data.big_m_cw[(s, p)] = max(
                data.max_inventory_store[(s, p)],
                sum(data.demand[(s, p, t)] for t in periods) + data.max_inventory_store[(s, p)]
            )

        for p in products:
            med_price = float(base.loc[base["sku"] == p, "price"].median()) if (base["sku"] == p).any() else global_price
            data.holding_cost_wh[p] = max(0.02, holding_cost_rate * 0.5 * med_price)

        for i, j, p in itertools.product(stores, stores, products):
            if i == j:
                continue
            data.ship_cost_lt[(i, j, p)] = lt_ship_cost_flat
            data.big_m_lt[(i, j, p)] = max(5.0, 0.5 * sum(data.demand[(j, p, t)] for t in periods))

        for s in stores:
            data.fixed_dispatch_cw[s] = fixed_dispatch_cw
        for i, j in itertools.product(stores, stores):
            if i == j:
                continue
            data.fixed_dispatch_lt[(i, j)] = fixed_dispatch_lt

        # ===== Added Achamrah-style fields =====
        data.vehicles = [f"V{v}" for v in range(1, vehicle_count + 1)]
        data.vehicle_capacity = vehicle_capacity
        data.vehicle_fixed_cost = vehicle_fixed_cost
        data.max_vehicles_used = vehicle_count
        data.alpha = alpha

        # replenishment to warehouse by period/product
        demand_by_sku_period = base.groupby(["sku", "period"])["sale_qty"].sum().to_dict()
        replenishment_cycle = 7   # replenish from DC every 7 days
        for p in products:
            for t in periods:
                if (t - 1) % replenishment_cycle == 0:
                    data.replenishment_wh[(p, t)] = max(
                        0.0,
                        1.2 * float(demand_by_sku_period.get((p, t), 0.0))
                    )
                else:
                    data.replenishment_wh[(p, t)] = 0.0

        # aggregate node capacity for stores
        for s in stores:
            data.node_capacity[s] = sum(data.max_inventory_store[(s, p)] for p in products)

        # aggregate capacity for CW
        data.node_capacity[data.warehouse] = cw_capacity_factor * sum(data.init_inventory_wh[p] for p in products)

        self._populate_routing_distances(data, stores, distance_matrix_path)

        # paper-style LT unit cost b_ij
        for i in stores:
            for j in stores:
                if i == j:
                    continue
                data.transship_unit_cost[(i, j)] = 0.01 * alpha * data.distance[(i, j)]

        # Validation target = actual END_QTY for periods >= 2
        validation_target = base[base["period"] > first_period][["store", "sku", "period", "end_qty"]].copy()
        validation_target = validation_target.rename(columns={"end_qty": "actual_end_qty"})

        metadata = {
            "n_rows_processed": len(base),
            "n_stores": len(stores),
            "n_products": len(products),
            "n_periods": len(periods),
            "validation_rows": len(validation_target),
            "first_period_used_as_initial_inventory": first_period,
            "n_vehicles": len(data.vehicles),
            "vehicle_capacity": data.vehicle_capacity,
            "vehicle_fixed_cost": data.vehicle_fixed_cost,
            "store_initial_inventory_multiplier": store_initial_inventory_multiplier,
        }
        return data, base, validation_target, metadata


# ============================================================================
# ORIGINAL BASELINE MODEL (KEPT)
# ============================================================================

class BaselineIRPModel:
    def __init__(self, data: IRPData):
        self.data = data

    def solve(self, msg: bool = False) -> BaselineIRPSolution:
        d = self.data
        mdl = gp.Model("Baseline_IRP", env=get_gurobi_env())
        mdl.Params.OutputFlag = 1 if msg else 0

        q_cw_keys = [(s, p, t) for s in d.stores for p in d.products for t in d.periods]
        y_cw_keys = [(s, t) for s in d.stores for t in d.periods]
        I_s_keys = [(s, p, t) for s in d.stores for p in d.products for t in d.periods]
        I_w_keys = [(p, t) for p in d.products for t in d.periods]
        B_keys = [(s, p, t) for s in d.stores for p in d.products for t in d.periods]

        q_cw = mdl.addVars(q_cw_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="q_cw")
        y_cw = mdl.addVars(y_cw_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="y_cw")
        I_s = mdl.addVars(I_s_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="I_s")
        I_w = mdl.addVars(I_w_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="I_w")
        B = mdl.addVars(B_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="B")

        mdl.setObjective(
            gp.quicksum(d.ship_cost_cw[(s, p)] * q_cw[(s, p, t)] for s, p, t in q_cw_keys)
            + gp.quicksum(d.fixed_dispatch_cw[s] * y_cw[(s, t)] for s, t in y_cw_keys)
            + gp.quicksum(d.holding_cost_store[(s, p)] * I_s[(s, p, t)] for s, p, t in I_s_keys)
            + gp.quicksum(d.holding_cost_wh[p] * I_w[(p, t)] for p, t in I_w_keys)
            + gp.quicksum(d.shortage_cost[(s, p)] * B[(s, p, t)] for s, p, t in B_keys),
            GRB.MINIMIZE,
        )

        for s, p in itertools.product(d.stores, d.products):
            t0 = d.periods[0]
            mdl.addConstr(
                I_s[(s, p, t0)] == d.init_inventory_store[(s, p)] + q_cw[(s, p, t0)] - d.demand[(s, p, t0)] + B[(s, p, t0)]
            )
            for t_prev, t in zip(d.periods[:-1], d.periods[1:]):
                mdl.addConstr(
                    I_s[(s, p, t)] == I_s[(s, p, t_prev)] + q_cw[(s, p, t)] - d.demand[(s, p, t)] + B[(s, p, t)]
                )

        for p in d.products:
            t0 = d.periods[0]
            mdl.addConstr(I_w[(p, t0)] == d.init_inventory_wh[p] - gp.quicksum(q_cw[(s, p, t0)] for s in d.stores))
            for t_prev, t in zip(d.periods[:-1], d.periods[1:]):
                mdl.addConstr(I_w[(p, t)] == I_w[(p, t_prev)] - gp.quicksum(q_cw[(s, p, t)] for s in d.stores))

        for s, p, t in itertools.product(d.stores, d.products, d.periods):
            mdl.addConstr(I_s[(s, p, t)] <= d.max_inventory_store[(s, p)])
            mdl.addConstr(q_cw[(s, p, t)] <= d.big_m_cw[(s, p)] * y_cw[(s, t)])
        for p, t in itertools.product(d.products, d.periods):
            mdl.addConstr(I_w[(p, t)] <= d.max_inventory_wh[p])

        mdl.optimize()

        return BaselineIRPSolution(
            status=_grb_status_name(mdl.Status),
            objective=_safe_obj_value(mdl),
            ship_cw={(s, p, t): _safe_var_value(mdl, q_cw[(s, p, t)]) for s, p, t in q_cw_keys},
            activate_cw={(s, t): int(round(_safe_var_value(mdl, y_cw[(s, t)]))) for s, t in y_cw_keys},
            inv_store={(s, p, t): _safe_var_value(mdl, I_s[(s, p, t)]) for s, p, t in I_s_keys},
            inv_wh={(p, t): _safe_var_value(mdl, I_w[(p, t)]) for p, t in I_w_keys},
            shortage={(s, p, t): _safe_var_value(mdl, B[(s, p, t)]) for s, p, t in B_keys},
        )


# ============================================================================
# ACHAMRAH-STYLE FULLER IRPT MODEL
# ============================================================================

class AchamrahFullIRPTModel:
    """
    Practical implementation of the paper-style IRPT model.

    Included:
    - Base constraints (2)-(15)
    - Valid inequalities (16)-(20)

    Not fully included:
    - Constraint (21), because the paper separates those cuts dynamically in branch-and-cut.
    """

    def __init__(self, data: IRPData):
        self.data = data

    def solve(
        self,
        msg: bool = False,
        time_limit: Optional[int] = None,
        enforce_integer_flows: bool = False,
        add_valid_16_20: bool = True,
        allow_lateral_transshipment: bool = True,
        min_visit_activity_qty: float = 1.0,
        min_visit_delivery_qty: float = 0.0,
        cw_dispatch_cycle: Optional[int] = 5,
    ) -> FullIRPTSolution:
        d = self.data
        N = d.stores
        P = d.products
        T = d.periods
        V = d.vehicles
        CW = d.warehouse
        N0 = [CW] + N

        mdl = gp.Model("Achamrah_Full_IRPT", env=get_gurobi_env())
        mdl.Params.OutputFlag = 1 if msg else 0
        if time_limit is not None:
            mdl.Params.TimeLimit = time_limit

        flow_vtype = GRB.INTEGER if enforce_integer_flows else GRB.CONTINUOUS

        I_s_keys = [(s, p, t) for s in N for p in P for t in T]
        I_w_keys = [(p, t) for p in P for t in T]
        Qdir_keys = [(s, p, t) for s in N for p in P for t in T]
        q_keys = [(p, i, j, v, t) for p in P for i in N0 for j in N0 if i != j for v in V for t in T]
        y_keys = [(i, j, p, v, t) for i in N for j in N if i != j for p in P for v in V for t in T]
        deliv_keys = [(s, p, v, t) for s in N for p in P for v in V for t in T]
        load_keys = [(i, v, t) for i in N0 for v in V for t in T]
        ordv_keys = [(s, v, t) for s in N for v in V for t in T]
        B_keys = [(s, p, t) for s in N for p in P for t in T]
        x_keys = [(i, j, v, t) for i in N0 for j in N0 if i != j for v in V for t in T]
        u_keys = [(v, t) for v in V for t in T]
        z_keys = [(i, v, t) for i in N0 for v in V for t in T]

        I_s = mdl.addVars(I_s_keys, lb=0.0, vtype=flow_vtype, name="I_s")
        I_w = mdl.addVars(I_w_keys, lb=0.0, vtype=flow_vtype, name="I_w")
        Qdir = mdl.addVars(Qdir_keys, lb=0.0, vtype=flow_vtype, name="Qdir")
        q = mdl.addVars(q_keys, lb=0.0, vtype=flow_vtype, name="q")
        y = mdl.addVars(y_keys, lb=0.0, vtype=flow_vtype, name="y")
        deliv = mdl.addVars(deliv_keys, lb=0.0, vtype=flow_vtype, name="deliv")
        load = mdl.addVars(load_keys, lb=0.0, ub=d.vehicle_capacity, vtype=flow_vtype, name="load")
        ordv = mdl.addVars(ordv_keys, lb=0.0, ub=len(N), vtype=GRB.CONTINUOUS, name="ordv")
        if not allow_lateral_transshipment:
            for key in y_keys:
                y[key].UB = 0.0
        B = mdl.addVars(B_keys, lb=0.0, vtype=flow_vtype, name="B")
        x = mdl.addVars(x_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")
        u = mdl.addVars(u_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="u")
        z = mdl.addVars(z_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

        mdl.setObjective(
            gp.quicksum(d.ship_cost_cw[(s, p)] * Qdir[(s, p, t)] for s, p, t in Qdir_keys)
            + gp.quicksum(d.holding_cost_store[(s, p)] * I_s[(s, p, t)] for s, p, t in I_s_keys)
            + gp.quicksum(d.holding_cost_wh[p] * I_w[(p, t)] for p, t in I_w_keys)
            + gp.quicksum(d.alpha * d.distance[(i, j)] * x[(i, j, v, t)] for i, j, v, t in x_keys)
            + gp.quicksum(d.vehicle_fixed_cost * u[(v, t)] for v, t in u_keys)
            + gp.quicksum(d.transship_unit_cost[(i, j)] * y[(i, j, p, v, t)] for i, j, p, v, t in y_keys)
            + gp.quicksum(d.shortage_cost[(s, p)] * B[(s, p, t)] for s, p, t in B_keys),
            GRB.MINIMIZE,
        )

        first_t = min(T)
        restricted_cw_dispatch_periods: Optional[Set[Period]] = None
        if cw_dispatch_cycle is not None and int(cw_dispatch_cycle) > 1:
            cycle = int(cw_dispatch_cycle)
            restricted_cw_dispatch_periods = {t for t in T if (t - first_t) % cycle == 0}

        for s in N:
            for p in P:
                for t in T:
                    prev = d.init_inventory_store[(s, p)] if t == first_t else I_s[(s, p, t - 1)]
                    mdl.addConstr(
                        I_s[(s, p, t)]
                        == prev
                        + Qdir[(s, p, t)]
                        - d.demand[(s, p, t)]
                        + B[(s, p, t)]
                        + gp.quicksum(y[(j, s, p, v, t)] for j in N if j != s for v in V)
                        - gp.quicksum(y[(s, j, p, v, t)] for j in N if j != s for v in V)
                    )

        for p in P:
            for t in T:
                prev = d.init_inventory_wh[p] if t == first_t else I_w[(p, t - 1)]
                mdl.addConstr(
                    I_w[(p, t)]
                    == prev
                    - gp.quicksum(Qdir[(s, p, t)] for s in N)
                    + d.replenishment_wh[(p, t)]
                )

        for s in N:
            for p in P:
                for t in T:
                    mdl.addConstr(Qdir[(s, p, t)] == gp.quicksum(deliv[(s, p, v, t)] for v in V))
                    if restricted_cw_dispatch_periods is not None and t not in restricted_cw_dispatch_periods:
                        mdl.addConstr(Qdir[(s, p, t)] == 0.0)
                    for v in V:
                        mdl.addConstr(
                            deliv[(s, p, v, t)]
                            + gp.quicksum(y[(i, s, p, v, t)] for i in N if i != s)
                            - gp.quicksum(y[(s, j, p, v, t)] for j in N if j != s)
                            == gp.quicksum(q[(p, i, s, v, t)] for i in N0 if i != s)
                            - gp.quicksum(q[(p, s, j, v, t)] for j in N0 if j != s)
                        )
                        mdl.addConstr(deliv[(s, p, v, t)] <= d.vehicle_capacity * z[(s, v, t)])

        for i in N:
            for v in V:
                for t in T:
                    mdl.addConstr(gp.quicksum(q[(p, i, CW, v, t)] for p in P) == 0)

        for s in N:
            for t in T:
                mdl.addConstr(gp.quicksum(I_s[(s, p, t)] for p in P) <= d.node_capacity[s])
        for t in T:
            mdl.addConstr(gp.quicksum(I_w[(p, t)] for p in P) <= d.node_capacity[CW])

        for i in N0:
            for j in N0:
                if i == j:
                    continue
                for v in V:
                    for t in T:
                        mdl.addConstr(gp.quicksum(q[(p, i, j, v, t)] for p in P) <= d.vehicle_capacity * x[(i, j, v, t)])

        for s in N:
            for p in P:
                for t in T:
                    begin_inv = d.init_inventory_store[(s, p)] if t == first_t else I_s[(s, p, t - 1)]
                    mdl.addConstr(gp.quicksum(y[(s, j, p, v, t)] for j in N if j != s for v in V) <= begin_inv)

        for j in N:
            for v in V:
                for t in T:
                    mdl.addConstr(
                        gp.quicksum(x[(i, j, v, t)] for i in N0 if i != j)
                        == gp.quicksum(x[(j, i, v, t)] for i in N0 if i != j)
                    )

        for j in N:
            for t in T:
                mdl.addConstr(gp.quicksum(x[(i, j, v, t)] for i in N0 if i != j for v in V) <= 1)

        for v in V:
            for t in T:
                mdl.addConstr(gp.quicksum(x[(CW, j, v, t)] for j in N) == u[(v, t)])

        for t in T:
            mdl.addConstr(gp.quicksum(u[(v, t)] for v in V) <= d.max_vehicles_used)

        for i in N0:
            for v in V:
                for t in T:
                    if i == CW:
                        mdl.addConstr(z[(i, v, t)] == u[(v, t)])
                    else:
                        mdl.addConstr(z[(i, v, t)] == gp.quicksum(x[(j, i, v, t)] for j in N0 if j != i))

        min_visit_activity_qty = max(0.0, float(min_visit_activity_qty))
        if min_visit_activity_qty > 0.0:
            for s in N:
                for v in V:
                    for t in T:
                        store_activity = (
                            gp.quicksum(deliv[(s, p, v, t)] for p in P)
                            + gp.quicksum(y[(i, s, p, v, t)] for i in N if i != s for p in P)
                            + gp.quicksum(y[(s, j, p, v, t)] for j in N if j != s for p in P)
                        )
                        mdl.addConstr(store_activity >= min_visit_activity_qty * z[(s, v, t)])

        min_visit_delivery_qty = max(0.0, float(min_visit_delivery_qty))
        if min_visit_delivery_qty > 0.0:
            for s in N:
                for v in V:
                    for t in T:
                        mdl.addConstr(
                            gp.quicksum(deliv[(s, p, v, t)] for p in P)
                            >= min_visit_delivery_qty * z[(s, v, t)]
                        )

        for p in P:
            for i in N0:
                for j in N0:
                    if i == j:
                        continue
                    for v in V:
                        for t in T:
                            mdl.addConstr(q[(p, i, j, v, t)] <= d.vehicle_capacity * x[(i, j, v, t)])

        for v in V:
            for t in T:
                mdl.addConstr(
                    load[(CW, v, t)]
                    == gp.quicksum(deliv[(s, p, v, t)] for s in N for p in P)
                )
                mdl.addConstr(load[(CW, v, t)] <= d.vehicle_capacity * u[(v, t)])

        for s in N:
            for v in V:
                for t in T:
                    mdl.addConstr(
                        load[(s, v, t)]
                        <= d.vehicle_capacity * gp.quicksum(x[(s, j, v, t)] for j in N0 if j != s)
                    )

        for i in N0:
            for j in N:
                if i == j:
                    continue
                for v in V:
                    for t in T:
                        delivered_at_j = gp.quicksum(deliv[(j, p, v, t)] for p in P)
                        mdl.addConstr(
                            load[(j, v, t)]
                            <= load[(i, v, t)] - delivered_at_j + d.vehicle_capacity * (1 - x[(i, j, v, t)])
                        )
                        mdl.addConstr(
                            load[(j, v, t)]
                            >= load[(i, v, t)] - delivered_at_j - d.vehicle_capacity * (1 - x[(i, j, v, t)])
                        )

        for i in N:
            for v in V:
                for t in T:
                    mdl.addConstr(load[(i, v, t)] >= gp.quicksum(q[(p, i, j, v, t)] for p in P for j in N0 if j != i))

        for s in N:
            for v in V:
                for t in T:
                    mdl.addConstr(ordv[(s, v, t)] <= len(N) * z[(s, v, t)])
                    mdl.addConstr(ordv[(s, v, t)] >= z[(s, v, t)])

        for i in N:
            for j in N:
                if i == j:
                    continue
                for v in V:
                    for t in T:
                        mdl.addConstr(
                            ordv[(i, v, t)] - ordv[(j, v, t)] + len(N) * x[(i, j, v, t)]
                            <= len(N) - 1
                        )

        for i in N:
            for j in N:
                if i == j:
                    continue
                for p in P:
                    for v in V:
                        for t in T:
                            mdl.addConstr(y[(i, j, p, v, t)] <= q[(p, i, j, v, t)])

        if add_valid_16_20:
            for i in N:
                for v in V:
                    for t in T:
                        mdl.addConstr(x[(CW, i, v, t)] <= z[(i, v, t)])

            for i in N:
                for j in N:
                    if i == j:
                        continue
                    for v in V:
                        for t in T:
                            mdl.addConstr(x[(i, j, v, t)] <= z[(j, v, t)])

            for i in N:
                for v in V:
                    for t in T:
                        mdl.addConstr(z[(i, v, t)] <= z[(CW, v, t)])

            for idx_v in range(1, len(V)):
                v = V[idx_v]
                v_prev = V[idx_v - 1]
                for t in T:
                    mdl.addConstr(z[(CW, v, t)] <= z[(CW, v_prev, t)])

            for s in N:
                for p in P:
                    for t1 in T:
                        for t2 in T:
                            if t2 < t1:
                                continue
                            total_dem = sum(d.demand[(s, p, tau)] for tau in T if t1 <= tau <= t2)
                            if total_dem <= 1e-9:
                                continue
                            init_term = d.init_inventory_store[(s, p)] if t1 == first_t else I_s[(s, p, t1 - 1)]
                            lhs = (
                                gp.quicksum(z[(s, v, tau)] for v in V for tau in T if t1 <= tau <= t2)
                                + (1.0 / total_dem) * gp.quicksum(
                                    y[(j, s, p, v, tau)]
                                    for j in N if j != s
                                    for v in V
                                    for tau in T if t1 <= tau <= t2
                                )
                            )
                            rhs = (total_dem - init_term) / total_dem
                            mdl.addConstr(lhs >= rhs)

        mdl.optimize()

        return FullIRPTSolution(
            status=_grb_status_name(mdl.Status),
            objective=_safe_obj_value(mdl),
            direct_ship_q={(s, p, t): _safe_var_value(mdl, Qdir[(s, p, t)]) for s in N for p in P for t in T},
            inv_store={(s, p, t): _safe_var_value(mdl, I_s[(s, p, t)]) for s in N for p in P for t in T},
            inv_wh={(p, t): _safe_var_value(mdl, I_w[(p, t)]) for p in P for t in T},
            shortage={(s, p, t): _safe_var_value(mdl, B[(s, p, t)]) for s in N for p in P for t in T},
            x={(i, j, v, t): int(round(_safe_var_value(mdl, x[(i, j, v, t)]))) for i in N0 for j in N0 if i != j for v in V for t in T},
            u={(v, t): int(round(_safe_var_value(mdl, u[(v, t)]))) for v in V for t in T},
            z={(i, v, t): int(round(_safe_var_value(mdl, z[(i, v, t)]))) for i in N0 for v in V for t in T},
            q={(p, i, j, v, t): _safe_var_value(mdl, q[(p, i, j, v, t)]) for p in P for i in N0 for j in N0 if i != j for v in V for t in T},
            y={(i, j, p, v, t): _safe_var_value(mdl, y[(i, j, p, v, t)]) for i in N for j in N if i != j for p in P for v in V for t in T},
            deliv={(s, p, v, t): _safe_var_value(mdl, deliv[(s, p, v, t)]) for s in N for p in P for v in V for t in T},
            load={(i, v, t): _safe_var_value(mdl, load[(i, v, t)]) for i in N0 for v in V for t in T},
            efficiency_metrics=_model_efficiency_metrics(mdl),
        )


# ============================================================================
# COLUMN GENERATION FOR LATERAL TRANSSHIPMENT
# ============================================================================

def generate_random_lt_patterns(
    data: IRPData,
    baseline_solution,
    n_patterns_per_product_period: int = 5,
    max_pairs_in_pattern: int = 4,
    max_qty_per_pair: int = 10,
    lt_activation_threshold: float = 0.0,
    seed: int = 123,
) -> List[LTPattern]:
    """
    Feasible random warm-start patterns. Only generated for active (product, period)
    pairs that pass the minimum LT activation threshold.
    """
    cg = LateralTransshipmentCG(
        data=data,
        baseline_solution=baseline_solution,
        initial_patterns=None,
        lt_activation_threshold=lt_activation_threshold,
    )
    need, surplus = cg._build_need_and_surplus_proxies()
    active_pt = cg._compute_active_product_periods(need, surplus)

    rng = random.Random(seed)
    patterns = []
    for p, t in sorted(active_pt):
        donors = [s for s in data.stores if surplus[(s, p, t)] > 1e-9]
        receivers = [s for s in data.stores if need[(s, p, t)] > 1e-9]
        if not donors or not receivers:
            continue

        candidate_pairs = [(i, j) for i in donors for j in receivers if i != j]
        if not candidate_pairs:
            continue

        for idx in range(1, n_patterns_per_product_period + 1):
            rng.shuffle(candidate_pairs)
            chosen_pairs = candidate_pairs[:rng.randint(1, min(max_pairs_in_pattern, len(candidate_pairs)))]
            donor_left = {i: surplus[(i, p, t)] for i in donors}
            recv_left = {j: need[(j, p, t)] for j in receivers}

            flows: Dict[Tuple[Store, Store], float] = {}
            total_cost = 0.0
            for i, j in chosen_pairs:
                ub = min(donor_left[i], recv_left[j], float(max_qty_per_pair))
                if ub <= 1e-9:
                    continue
                qty = float(rng.uniform(1.0, ub))
                flows[(i, j)] = qty
                donor_left[i] -= qty
                recv_left[j] -= qty
                total_cost += qty * data.ship_cost_lt[(i, j, p)] + data.fixed_dispatch_lt[(i, j)]

            if flows:
                patterns.append(LTPattern(
                    pattern_id=f"LT_{p}_T{t}_{idx}",
                    period=t,
                    product=p,
                    pattern_flows=flows,
                    column_cost=round(total_cost, 6),
                    metadata={"source": "random_warm_start"},
                ))
    return patterns


def format_pattern_detail(pat):
    flow_text = ", ".join(
        [f"{i}->{j}:{qty:.2f}" for (i, j), qty in pat.pattern_flows.items()]
    )
    return (
        f"pattern_id={pat.pattern_id} | "
        f"product={pat.product} | period={pat.period} | "
        f"cost={pat.column_cost:.4f} | flows=[{flow_text}]"
    )


def apply_hidden_local_reallocation_demand_shocks(
    data: IRPData,
    baseline_solution: Optional[FullIRPTSolution] = None,
    shock_probability: float = 0.5,
    max_reallocation_fraction: float = 0.35,
    reallocations_per_product_period: int = 3,
    non_dispatch_shock_multiplier: float = 1.8,
    cw_dispatch_cycle: Optional[int] = 5,
    seed: int = 20260418,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    shock_probability = min(1.0, max(0.0, float(shock_probability)))
    max_reallocation_fraction = min(1.0, max(0.0, float(max_reallocation_fraction)))
    reallocations_per_product_period = max(1, int(reallocations_per_product_period))
    non_dispatch_shock_multiplier = max(1.0, float(non_dispatch_shock_multiplier))
    data.realized_demand = {key: float(value) for key, value in data.demand.items()}

    n_shocked = 0
    shocked_product_periods: Set[Tuple[Product, Period]] = set()
    total_reallocated = 0.0
    first_t = min(data.periods) if data.periods else 0
    cycle = int(cw_dispatch_cycle) if cw_dispatch_cycle is not None else 0
    for p in data.products:
        for t in data.periods:
            is_dispatch_period = True
            if cycle > 1:
                is_dispatch_period = ((t - first_t) % cycle == 0)
            period_multiplier = 1.0 if is_dispatch_period else non_dispatch_shock_multiplier
            effective_probability = min(1.0, shock_probability * period_multiplier)
            if rng.random() > effective_probability:
                continue
            candidates = [
                s for s in data.stores
                if float(data.demand.get((s, p, t), 0.0)) > 1e-9
            ]
            if len(candidates) < 2:
                continue

            for _ in range(reallocations_per_product_period):
                def receiver_score(store: Store) -> float:
                    forecast = float(data.demand.get((store, p, t), 0.0))
                    if baseline_solution is None:
                        return forecast
                    ending_inv = float(baseline_solution.inv_store.get((store, p, t), 0.0))
                    shortage = float(baseline_solution.shortage.get((store, p, t), 0.0))
                    fragility = shortage + max(0.0, forecast - ending_inv)
                    return fragility * period_multiplier + 0.01 * forecast

                receiver = max(candidates, key=receiver_score)
                donor_candidates = [s for s in candidates if s != receiver]
                if not donor_candidates:
                    continue

                def donor_score(store: Store) -> float:
                    realized = float(data.realized_demand.get((store, p, t), 0.0))
                    if baseline_solution is None:
                        return realized
                    ending_inv = float(baseline_solution.inv_store.get((store, p, t), 0.0))
                    shortage = float(baseline_solution.shortage.get((store, p, t), 0.0))
                    return ending_inv - shortage + 0.01 * realized

                donor = max(donor_candidates, key=donor_score)
                receiver_forecast = float(data.demand.get((receiver, p, t), 0.0))
                donor_realized = float(data.realized_demand.get((donor, p, t), 0.0))
                delta = min(
                    period_multiplier * max_reallocation_fraction * max(receiver_forecast, 1.0),
                    max_reallocation_fraction * donor_realized,
                )
                if delta <= 1e-9:
                    continue

                data.realized_demand[(receiver, p, t)] = data.realized_demand.get((receiver, p, t), 0.0) + delta
                data.realized_demand[(donor, p, t)] = max(0.0, donor_realized - delta)
                n_shocked += 1
                shocked_product_periods.add((p, t))
                total_reallocated += delta

    return {
        "shock_model": "hidden_local_reallocation",
        "shock_probability": shock_probability,
        "max_reallocation_fraction": max_reallocation_fraction,
        "reallocations_per_product_period": reallocations_per_product_period,
        "non_dispatch_shock_multiplier": non_dispatch_shock_multiplier,
        "cw_dispatch_cycle": cw_dispatch_cycle,
        "shock_seed": seed,
        "n_store_sku_period_reallocations": n_shocked,
        "n_shocked_product_periods": len(shocked_product_periods),
        "total_reallocated_units": round(total_reallocated, 6),
    }


def build_post_shock_inventory_state(
    data: IRPData,
    baseline_solution: FullIRPTSolution,
) -> Dict[str, Any]:
    data.post_shock_inventory = {}
    data.post_shock_shortage = {}
    total_shortage = 0.0
    total_realized_demand = 0.0
    total_forecast_demand = 0.0

    for s in data.stores:
        for p in data.products:
            prev_inventory = float(data.init_inventory_store.get((s, p), 0.0))
            for t in sorted(data.periods):
                forecast = float(data.demand.get((s, p, t), 0.0))
                realized = float(data.realized_demand.get((s, p, t), forecast))
                shipment = float(baseline_solution.direct_ship_q.get((s, p, t), 0.0))
                available = prev_inventory + shipment
                shortage = max(0.0, realized - available)
                ending_inventory = max(0.0, available - realized)
                data.post_shock_shortage[(s, p, t)] = shortage
                data.post_shock_inventory[(s, p, t)] = ending_inventory
                prev_inventory = ending_inventory
                total_shortage += shortage
                total_realized_demand += realized
                total_forecast_demand += forecast

    return {
        "total_forecast_demand": round(total_forecast_demand, 6),
        "total_realized_demand": round(total_realized_demand, 6),
        "total_post_shock_shortage": round(total_shortage, 6),
        "total_post_shock_inventory": round(sum(data.post_shock_inventory.values()), 6),
    }


def build_post_shock_lt_diagnostics(
    data: IRPData,
    baseline_solution: FullIRPTSolution,
    lt_activation_threshold: float = 0.0,
) -> Dict[str, Any]:
    cg = LateralTransshipmentCG(
        data=data,
        baseline_solution=baseline_solution,
        initial_patterns=None,
        lt_activation_threshold=lt_activation_threshold,
    )
    need, surplus = cg._build_need_and_surplus_proxies()
    active_product_periods = cg._compute_active_product_periods(need, surplus)
    return {
        "n_active_product_periods_after_shock": len(active_product_periods),
        "total_need_after_shock": round(sum(need.values()), 6),
        "total_surplus_after_shock": round(sum(surplus.values()), 6),
        "max_need_after_shock": round(max(need.values()) if need else 0.0, 6),
        "max_surplus_after_shock": round(max(surplus.values()) if surplus else 0.0, 6),
    }


class LateralTransshipmentCG:
    def __init__(
        self,
        data: IRPData,
        baseline_solution,
        initial_patterns: Optional[List[LTPattern]] = None,
        lt_activation_threshold: float = 0.0,
        safety_stock_units: float = 0.0,
        need_lookahead_periods: int = 2,
        surplus_reserve_periods: int = 1,
        rebalance_need_penalty: float = 1.0,
        max_pairs_per_pattern: int = 4,
        max_columns_per_product_period: int = 3,
        top_pairs_per_feature: int = 20,
        top_patterns_per_feature: int = 5,
        feature_ranges: Optional[Dict[str, Dict[str, float]]] = None,
        stackelberg_params: Optional[StackelbergParams] = None,
        use_gnn: bool = False,
        collect_teacher_mode: bool = False,
        runtime_gnn_mode: Optional[bool] = None,
        gnn_checkpoint: Optional[str] = None,
        use_classical_fallback: bool = True,
        gnn_selection_mode: str = "cumulative_mass",
        gnn_mass_threshold: float = 0.55,
        gnn_relative_threshold: float = 0.85,
        gnn_min_keep: int = 1,
        gnn_max_keep: Optional[int] = 150,
        gnn_max_keep_fraction: float = 0.30,
        gnn_root: str = "GNN",
        branch_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.data = data
        self.baseline = baseline_solution
        self.patterns = initial_patterns[:] if initial_patterns else []
        self.lt_activation_threshold = max(0.0, float(lt_activation_threshold))
        self.safety_stock_units = float(safety_stock_units)
        self.need_lookahead_periods = max(1, int(need_lookahead_periods))
        self.surplus_reserve_periods = max(1, int(surplus_reserve_periods))
        self.rebalance_need_penalty = float(rebalance_need_penalty)
        self.max_pairs_per_pattern = int(max_pairs_per_pattern)
        self.max_columns_per_product_period = int(max_columns_per_product_period)
        self.top_pairs_per_feature = int(top_pairs_per_feature)
        self.top_patterns_per_feature = int(top_patterns_per_feature)
        self.feature_ranges = feature_ranges or {
            "shortage_ratio": {"min": 0.00, "max": 1.00},
            "surplus_ratio": {"min": 0.00, "max": 1.00},
            "time_urgency": {"min": 0.00, "max": 1.00},
            "negative_reduced_cost": {"min": 0.00, "max": math.inf},
        }
        self.stackelberg_params = stackelberg_params or StackelbergParams()
        self.collect_teacher_mode = bool(collect_teacher_mode)
        requested_runtime_gnn = bool(use_gnn if runtime_gnn_mode is None else runtime_gnn_mode)
        self.runtime_gnn_mode = False if self.collect_teacher_mode else requested_runtime_gnn
        self.use_gnn = self.runtime_gnn_mode
        self.gnn_checkpoint = gnn_checkpoint or DEFAULT_GNN_CHECKPOINT
        self.gnn_selection_mode = str(gnn_selection_mode)
        self.gnn_mass_threshold = float(gnn_mass_threshold)
        self.gnn_relative_threshold = float(gnn_relative_threshold)
        self.gnn_min_keep = int(gnn_min_keep)
        self.gnn_max_keep = int(gnn_max_keep) if gnn_max_keep is not None else None
        self.gnn_max_keep_fraction = float(gnn_max_keep_fraction)
        self.gnn_prob_weight = 0.60
        self.gnn_rc_gain_weight = 0.30
        self.gnn_acceptance_weight = 0.10
        self.use_classical_fallback = bool(use_classical_fallback)
        self.gnn_root = gnn_root
        self._gnn_loaded = False
        self._gnn_unavailable_reason: Optional[str] = None
        self._gnn_model = None
        self._gnn_checkpoint_payload = None
        self._gnn_build_graph = None
        self._gnn_graph_to_tensors = None
        self._gnn_normalize_dataset = None
        self._gnn_adaptive_select_indices = None
        self.gnn_selection_history: List[Dict[str, Any]] = []
        self.teacher_dataset_rows: List[Dict[str, Any]] = []
        self.cg_history: List[Dict[str, Any]] = []
        self.cg_history_all_nodes: List[Dict[str, Any]] = []
        self.branch_history: List[Dict[str, Any]] = []
        self.column_pool_diagnostics: List[Dict[str, Any]] = []
        self.cg_episode_diagnostics: List[Dict[str, Any]] = []
        self.branch_bounds: Dict[str, Tuple[float, float]] = dict(branch_bounds or {})
        self.current_episode = 0
        self.current_branch_node_id: Optional[int] = None
        self.last_duplicate_rejects = 0
        self.last_signature_rejects = 0
        self.last_empty_rejects = 0
        self.last_added_patterns = 0
        self._last_candidate_pair_count = 0
        self._last_pricing_summary: Dict[str, Any] = {}

    def _compute_combined_column_scores(self, patterns: List[LTPattern], probs: List[float]) -> List[Dict[str, float]]:
        rc_gains = []
        acceptance_scores = []
        for pat in patterns:
            try:
                rc = float(pat.metadata.get("reduced_cost", 0.0))
            except (TypeError, ValueError):
                rc = 0.0
            rc_gains.append(max(0.0, -rc))
            try:
                acceptance = float(pat.metadata.get("mean_acceptance_score", pat.metadata.get("acceptance_score", 0.0)))
            except (TypeError, ValueError):
                acceptance = 0.0
            acceptance_scores.append(min(max(acceptance, 0.0), 1.0))

        max_rc_gain = max(rc_gains) if rc_gains else 0.0
        rows = []
        for idx, pat in enumerate(patterns):
            rc_gain_norm = rc_gains[idx] / max(max_rc_gain, 1e-9) if max_rc_gain > 1e-12 else 0.0
            combined_score = (
                self.gnn_prob_weight * float(probs[idx])
                + self.gnn_rc_gain_weight * rc_gain_norm
                + self.gnn_acceptance_weight * acceptance_scores[idx]
            )
            rows.append({
                "idx": float(idx),
                "gnn_prob": float(probs[idx]),
                "rc_gain": float(rc_gains[idx]),
                "normalized_rc_gain": float(rc_gain_norm),
                "acceptance_score": float(acceptance_scores[idx]),
                "combined_score": float(max(0.0, combined_score)),
                "reduced_cost": float(pat.metadata.get("reduced_cost", 0.0) or 0.0),
            })
        return rows

    @staticmethod
    def _pattern_signature_key(pat: LTPattern) -> Tuple[Product, Period, Tuple[Tuple[Store, Store, float], ...]]:
        return (
            pat.product,
            pat.period,
            tuple(sorted((i, j, round(float(qty), 6)) for (i, j), qty in pat.pattern_flows.items())),
        )

    def _deduplicate_priced_patterns(self, patterns: List[LTPattern], episode: int) -> List[LTPattern]:
        if not patterns:
            self._last_pricing_summary["patterns_built_before_dedup"] = 0
            self._last_pricing_summary["patterns_deduplicated_before_gnn"] = 0
            return patterns

        best_by_signature: Dict[Tuple[Product, Period, Tuple[Tuple[Store, Store, float], ...]], LTPattern] = {}
        existing_signatures = {self._pattern_signature_key(pat) for pat in self.patterns}
        rejected_duplicates: List[LTPattern] = []
        for pat in patterns:
            signature = self._pattern_signature_key(pat)
            if not pat.pattern_flows or signature in existing_signatures:
                rejected_duplicates.append(pat)
                continue
            current = best_by_signature.get(signature)
            if current is None:
                best_by_signature[signature] = pat
                continue

            pat_rc = float(pat.metadata.get("reduced_cost", 0.0) or 0.0)
            current_rc = float(current.metadata.get("reduced_cost", 0.0) or 0.0)
            pat_acceptance = float(pat.metadata.get("mean_acceptance_score", 0.0) or 0.0)
            current_acceptance = float(current.metadata.get("mean_acceptance_score", 0.0) or 0.0)
            pat_is_better = (pat_rc, -pat_acceptance) < (current_rc, -current_acceptance)
            if pat_is_better:
                rejected_duplicates.append(current)
                best_by_signature[signature] = pat
            else:
                rejected_duplicates.append(pat)

        deduped = list(best_by_signature.values())
        for pat in rejected_duplicates:
            self._record_pattern_pool_diagnostic(
                pat=pat,
                stage="pricing_dedup_rejected_before_gnn",
                pattern_reduced_cost=pat.metadata.get("reduced_cost"),
                duplicate_signature_reject=True,
                added_to_pool=False,
                signature=str(self._pattern_signature_key(pat)),
            )

        self._last_pricing_summary["patterns_built_before_dedup"] = len(patterns)
        self._last_pricing_summary["patterns_deduplicated_before_gnn"] = len(patterns) - len(deduped)
        return deduped

    def _adaptive_select_columns(self, score_rows: List[Dict[str, float]]) -> Tuple[List[int], Dict[str, Any]]:
        if not score_rows:
            return [], {"selection_mode": self.gnn_selection_mode, "adaptive_k": 0}

        scores = [row["combined_score"] for row in score_rows]
        if sum(max(0.0, float(score)) for score in scores) <= 1e-12:
            scores = [row["gnn_prob"] for row in score_rows]
        selector = self._gnn_adaptive_select_indices
        if selector is None:
            raise RuntimeError("GNN adaptive selector helper is not loaded")
        max_keep = self.gnn_max_keep
        if self.gnn_max_keep_fraction > 0:
            fraction_cap = max(self.gnn_min_keep, int(math.ceil(len(score_rows) * self.gnn_max_keep_fraction)))
            max_keep = min(max_keep, fraction_cap) if max_keep is not None else fraction_cap
        selected_idx, info = selector(
            scores,
            tie_breaker=[row["gnn_prob"] for row in score_rows],
            selection_mode=self.gnn_selection_mode,
            mass_threshold=self.gnn_mass_threshold,
            relative_threshold=self.gnn_relative_threshold,
            min_keep=self.gnn_min_keep,
            max_keep=max_keep,
        )
        info["max_keep"] = max_keep
        info["max_keep_fraction"] = self.gnn_max_keep_fraction
        return selected_idx, info

    def _apply_classical_fallback(self, patterns: List[LTPattern], selected_idx: List[int]) -> Tuple[List[int], Optional[int]]:
        if not self.use_classical_fallback or not patterns:
            return selected_idx, None

        has_negative_selected = any(
            float(patterns[idx].metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
            for idx in selected_idx
        )
        if selected_idx and has_negative_selected:
            return selected_idx, None

        feasible = [
            (idx, float(pat.metadata.get("reduced_cost", 0.0) or 0.0))
            for idx, pat in enumerate(patterns)
            if float(pat.metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
        ]
        if not feasible:
            return selected_idx, None

        best_idx = min(feasible, key=lambda item: item[1])[0]
        if best_idx not in selected_idx:
            selected_idx = [best_idx] + selected_idx
        return selected_idx, best_idx

    def _record_teacher_rows(
        self,
        *,
        episode: int,
        patterns: List[LTPattern],
        score_rows: List[Dict[str, float]],
        raw_graph: Optional[Dict[str, Any]],
        gnn_selected_idx: List[int],
        final_selected_idx: List[int],
        adaptive_info: Dict[str, Any],
        fallback_idx: Optional[int],
    ) -> None:
        gnn_selected_set = set(gnn_selected_idx)
        final_selected_set = set(final_selected_idx)
        fallback_set = {fallback_idx} if fallback_idx is not None else set()
        score_by_idx = {int(row["idx"]): row for row in score_rows}
        column_features = raw_graph.get("column_features") if raw_graph else None
        constraint_features = raw_graph.get("constraint_features") if raw_graph else None
        edge_index = raw_graph.get("edge_index_col_to_con") if raw_graph else None
        edge_attr = raw_graph.get("edge_attr_col_to_con") if raw_graph else None
        edge_constraint_ids_by_col: Dict[int, List[int]] = {idx: [] for idx in range(len(patterns))}
        edge_attrs_by_col: Dict[int, List[List[float]]] = {idx: [] for idx in range(len(patterns))}
        if edge_index is not None and edge_attr is not None:
            col_indices = edge_index[0].tolist()
            con_indices = edge_index[1].tolist()
            edge_attrs = edge_attr.tolist()
            for edge_pos, col_idx in enumerate(col_indices):
                col_idx = int(col_idx)
                edge_constraint_ids_by_col.setdefault(col_idx, []).append(int(con_indices[edge_pos]))
                edge_attrs_by_col.setdefault(col_idx, []).append([float(value) for value in edge_attrs[edge_pos]])
        constraint_features_json = json.dumps(constraint_features.tolist()) if constraint_features is not None else ""
        full_negative_rc_count = sum(
            1 for pat in patterns
            if float(pat.metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
        )
        for idx, pat in enumerate(patterns):
            row = score_by_idx.get(idx, {})
            self.teacher_dataset_rows.append({
                "source_instance": "irplt_cg",
                "branch_node_id": self.current_branch_node_id,
                "episode": episode,
                "column_index": idx,
                "pattern_id": pat.pattern_id,
                "product": pat.product,
                "period": pat.period,
                "pattern_flows_json": json.dumps([
                    {"donor": donor, "receiver": receiver, "quantity": float(qty)}
                    for (donor, receiver), qty in pat.pattern_flows.items()
                ]),
                "column_features_json": json.dumps(column_features[idx].tolist()) if column_features is not None else "",
                "constraint_features_json": constraint_features_json,
                "edge_constraint_indices_json": json.dumps(edge_constraint_ids_by_col.get(idx, [])),
                "edge_attrs_json": json.dumps(edge_attrs_by_col.get(idx, [])),
                "gnn_prob": row.get("gnn_prob"),
                "combined_score": row.get("combined_score"),
                "normalized_rc_gain": row.get("normalized_rc_gain"),
                "reduced_cost": pat.metadata.get("reduced_cost"),
                "mean_acceptance_score": pat.metadata.get("mean_acceptance_score"),
                "selected_by_gnn": idx in gnn_selected_set,
                "selected_by_classical_fallback": idx in fallback_set,
                "passed_to_rmp": idx in final_selected_set,
                "adaptive_k_star": adaptive_info.get("adaptive_k"),
                "selection_mode": adaptive_info.get("selection_mode"),
                "full_negative_reduced_cost_columns": full_negative_rc_count,
                "classical_fallback_enabled": self.use_classical_fallback,
                "classical_fallback_used": fallback_idx is not None,
                "label_source": "teacher_rmp",
                "selected_in_rmp": False,
                "lambda_value": 0.0,
                "teacher_label": 0,
                "teacher_score": 0.0,
                "cg_objective_after_reopt": None,
                "cg_objective_improvement": None,
            })

    def _annotate_teacher_rows_after_reopt(self, episode: int, sol: CGSolution, improvement: float) -> None:
        selected_by_gnn_in_rmp = 0
        for row in self.teacher_dataset_rows:
            if row.get("episode") != episode:
                continue
            lam = float(sol.lambda_values.get(row["pattern_id"], 0.0))
            reduced_cost = float(row.get("reduced_cost") or 0.0)
            row["lambda_value"] = lam
            row["selected_in_rmp"] = lam > 1e-6
            row["teacher_label"] = 1 if lam > 1e-6 else 0
            if row.get("selected_by_gnn") and row["selected_in_rmp"]:
                selected_by_gnn_in_rmp += 1
            row["teacher_score"] = float(lam * max(0.0, -reduced_cost))
            row["cg_objective_after_reopt"] = float(sol.objective)
            row["cg_objective_improvement"] = float(improvement)
        for row in self.gnn_selection_history:
            if row.get("episode") == episode:
                row["selected_by_gnn_in_rmp"] = selected_by_gnn_in_rmp

    def _mark_teacher_rows_passed_to_rmp(self, episode: int) -> None:
        pattern_ids_in_pool = {pat.pattern_id for pat in self.patterns}
        for row in self.teacher_dataset_rows:
            if row.get("episode") != episode:
                continue
            row["passed_to_rmp"] = row.get("pattern_id") in pattern_ids_in_pool

    def _load_gnn_utilities(self) -> bool:
        gnn_root = _project_path(self.gnn_root)
        if str(gnn_root) not in sys.path:
            sys.path.insert(0, str(gnn_root))
        try:
            utilities_module = importlib.import_module("utilities")
            self._gnn_build_graph = getattr(utilities_module, "build_bigraph_for_patterns")
            self._gnn_graph_to_tensors = getattr(utilities_module, "graph_to_tensors")
            self._gnn_normalize_dataset = getattr(utilities_module, "normalize_dataset")
            self._gnn_adaptive_select_indices = getattr(utilities_module, "adaptive_select_indices")
            return True
        except Exception as exc:
            self._gnn_unavailable_reason = str(exc)
            print(f"[GNN] Could not load graph utilities ({exc})")
            return False

    def _load_gnn_if_needed(self) -> bool:
        if not self.use_gnn:
            return False
        if self._gnn_loaded:
            return self._gnn_model is not None

        self._gnn_loaded = True
        checkpoint_path = _project_path(self.gnn_checkpoint)

        try:
            if not self._load_gnn_utilities():
                return False

            if not checkpoint_path.exists():
                self._gnn_unavailable_reason = f"checkpoint not found: {checkpoint_path}"
                print(f"[GNN] Model disabled, but teacher graph export remains enabled: {self._gnn_unavailable_reason}")
                return False

            torch = importlib.import_module("torch")
            model_module = importlib.import_module("models.attention.model")
            BiGATColumnScorer = getattr(model_module, "BiGATColumnScorer")

            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")

            config = dict(checkpoint.get("config", {}))
            dropout = float(config.pop("dropout", 0.0))
            model = BiGATColumnScorer(**config, dropout=dropout)
            model_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in checkpoint["state_dict"].items()
                if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
            }
            skipped = sorted(set(checkpoint["state_dict"]) - set(compatible_state))
            model.load_state_dict(compatible_state, strict=False)
            if skipped:
                print(f"[GNN] Skipped {len(skipped)} checkpoint tensors with incompatible shapes.")
            model.eval()

            self._gnn_model = model
            self._gnn_checkpoint_payload = checkpoint
            print(f"[GNN] BiGAT checkpoint loaded: {checkpoint_path}")
            return True
        except Exception as exc:
            self._gnn_unavailable_reason = str(exc)
            print(f"[GNN] Disabled: could not load BiGAT checkpoint ({exc})")
            self._gnn_model = None
            return False

    def _select_patterns_with_gnn(
        self,
        patterns: List[LTPattern],
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
    ) -> List[LTPattern]:
        if not patterns:
            return patterns
        model_available = self._load_gnn_if_needed()

        try:
            if self._gnn_build_graph is None:
                return patterns

            raw_graph = self._gnn_build_graph(
                patterns=patterns,
                data=self.data,
                need=need,
                surplus=surplus,
                dual_need=dual_need,
                dual_surplus=dual_surplus,
            )
            if (
                not model_available
                or self._gnn_model is None
                or self._gnn_graph_to_tensors is None
                or self._gnn_normalize_dataset is None
            ):
                probs = [0.0 for _ in patterns]
                score_rows = self._compute_combined_column_scores(patterns, probs)
                gnn_selected_idx = []
                selected_idx = list(range(len(patterns)))
                fallback_idx = None
                adaptive_info = {
                    "selection_mode": "runtime_no_model_pass_through",
                    "adaptive_k": len(patterns),
                    "mass_threshold": float(self.gnn_mass_threshold),
                    "relative_threshold": float(self.gnn_relative_threshold),
                    "score_mass_total": 0.0,
                }
            else:
                torch = importlib.import_module("torch")
                graph = self._gnn_graph_to_tensors(raw_graph)
                stats = (self._gnn_checkpoint_payload or {}).get("normalization")
                if stats:
                    normalized_graphs, _ = self._gnn_normalize_dataset([graph], stats)
                    graph = normalized_graphs[0]

                with torch.no_grad():
                    logits = self._gnn_model(
                        graph["column_features"],
                        graph["constraint_features"],
                        graph["edge_index_col_to_con"],
                        graph["edge_attr_col_to_con"],
                    )
                    probs_tensor = torch.sigmoid(logits).detach().cpu()

                probs = [float(value) for value in probs_tensor.tolist()]
                score_rows = self._compute_combined_column_scores(patterns, probs)
                gnn_selected_idx, adaptive_info = self._adaptive_select_columns(score_rows)
                selected_idx, fallback_idx = self._apply_classical_fallback(patterns, gnn_selected_idx)
            gnn_selected_set = set(gnn_selected_idx)
            selected = [patterns[idx] for idx in selected_idx]
            full_negative_rc_count = sum(
                1 for pat in patterns
                if float(pat.metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
            )

            episode = self.current_episode or (len(self.gnn_selection_history) + 1)
            selected_rows = []
            for idx, pat in enumerate(patterns):
                score = float(score_rows[idx]["gnn_prob"])
                combined_score = float(score_rows[idx]["combined_score"])
                pat.metadata["gnn_score"] = round(score, 6)
                pat.metadata["gnn_combined_score"] = round(combined_score, 6)
                pat.metadata["gnn_selected"] = idx in gnn_selected_set
                pat.metadata["gnn_selected_by_fallback"] = idx == fallback_idx
                pat.metadata["adaptive_k_star"] = adaptive_info.get("adaptive_k")
                self._record_pattern_pool_diagnostic(
                    pat=pat,
                    stage="gnn_scored",
                    pattern_reduced_cost=pat.metadata.get("reduced_cost"),
                    duplicate_id_reject=False,
                    duplicate_signature_reject=False,
                    empty_flow_reject=False,
                    added_to_pool=False,
                    signature=pat.metadata.get("pair_signature", ""),
                )
            for rank, idx in enumerate(selected_idx, start=1):
                pat = patterns[idx]
                score = float(score_rows[idx]["gnn_prob"])
                combined_score = float(score_rows[idx]["combined_score"])
                pat.metadata["gnn_score"] = round(score, 6)
                pat.metadata["gnn_combined_score"] = round(combined_score, 6)
                pat.metadata["gnn_rank"] = rank
                pat.metadata["gnn_selected"] = idx in gnn_selected_set
                selected_rows.append({
                    "rank": rank,
                    "pattern_id": pat.pattern_id,
                    "gnn_prob": score,
                    "combined_score": combined_score,
                    "reduced_cost": pat.metadata.get("reduced_cost"),
                    "column_cost": pat.column_cost,
                    "selected_by_gnn": idx in gnn_selected_set,
                    "selected_by_classical_fallback": idx == fallback_idx,
                })

            self._record_teacher_rows(
                episode=episode,
                patterns=patterns,
                score_rows=score_rows,
                raw_graph=raw_graph,
                gnn_selected_idx=gnn_selected_idx,
                final_selected_idx=selected_idx,
                adaptive_info=adaptive_info,
                fallback_idx=fallback_idx,
            )

            self.gnn_selection_history.append({
                "branch_node_id": self.current_branch_node_id,
                "episode": episode,
                "n_candidates": len(patterns),
                "n_selected": len(selected),
                "selection_mode": adaptive_info.get("selection_mode"),
                "adaptive_k_star": adaptive_info.get("adaptive_k"),
                "negative_reduced_cost_candidates": full_negative_rc_count,
                "classical_fallback_enabled": self.use_classical_fallback,
                "classical_fallback_used": fallback_idx is not None,
                "fallback_idx": fallback_idx,
                "selected": selected_rows,
            })
            print(
                f"[GNN] Episode {episode}: scored {len(patterns)} priced columns, "
                f"adaptive_k={adaptive_info.get('adaptive_k')} kept={len(selected)} "
                f"negative_rc={full_negative_rc_count} fallback_used={fallback_idx is not None}"
            )
            for row in selected_rows[:10]:
                print(
                    f"  rank={row['rank']} | prob={row['gnn_prob']:.6f} "
                    f"| combined={row['combined_score']:.6f} "
                    f"| pattern={row['pattern_id']} | rc={row['reduced_cost']} "
                    f"| column_cost={row['column_cost']:.6f}"
                )
            return selected
        except Exception as exc:
            print(f"[GNN] Scoring skipped for this pricing episode: {exc}")
            return patterns

    def _collect_teacher_batch_without_gnn_prefilter(
        self,
        patterns: List[LTPattern],
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
    ) -> List[LTPattern]:
        if not patterns:
            return patterns
        if not self._load_gnn_utilities() or self._gnn_build_graph is None:
            return patterns
        try:
            raw_graph = self._gnn_build_graph(
                patterns=patterns,
                data=self.data,
                need=need,
                surplus=surplus,
                dual_need=dual_need,
                dual_surplus=dual_surplus,
            )
            probs = [0.0 for _ in patterns]
            score_rows = self._compute_combined_column_scores(patterns, probs)
            selected_idx = list(range(len(patterns)))
            adaptive_info = {
                "selection_mode": "teacher_full_batch_no_gnn_prefilter",
                "adaptive_k": len(patterns),
                "mass_threshold": None,
                "relative_threshold": None,
                "score_mass_total": 0.0,
            }
            for idx, pat in enumerate(patterns):
                pat.metadata["gnn_score"] = None
                pat.metadata["gnn_combined_score"] = None
                pat.metadata["gnn_selected"] = False
                pat.metadata["gnn_selected_by_fallback"] = False
                pat.metadata["adaptive_k_star"] = len(patterns)
                self._record_pattern_pool_diagnostic(
                    pat=pat,
                    stage="teacher_full_batch_recorded_before_rmp",
                    pattern_reduced_cost=pat.metadata.get("reduced_cost"),
                    signature=pat.metadata.get("pair_signature", ""),
                )
            self._record_teacher_rows(
                episode=self.current_episode or (len(self.gnn_selection_history) + 1),
                patterns=patterns,
                score_rows=score_rows,
                raw_graph=raw_graph,
                gnn_selected_idx=[],
                final_selected_idx=selected_idx,
                adaptive_info=adaptive_info,
                fallback_idx=None,
            )
            self.gnn_selection_history.append({
                "branch_node_id": self.current_branch_node_id,
                "episode": self.current_episode or (len(self.gnn_selection_history) + 1),
                "n_candidates": len(patterns),
                "n_selected": len(patterns),
                "selection_mode": adaptive_info["selection_mode"],
                "adaptive_k_star": len(patterns),
                "negative_reduced_cost_candidates": sum(
                    1 for pat in patterns
                    if float(pat.metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
                ),
                "classical_fallback_enabled": False,
                "classical_fallback_used": False,
                "fallback_idx": None,
                "selected": [],
            })
            print(
                f"[Teacher] Episode {self.current_episode}: recorded full priced batch "
                f"without GNN prefilter ({len(patterns)} columns)."
            )
        except Exception as exc:
            print(f"[Teacher] Could not record full priced batch graph: {exc}")
        return patterns

    def add_patterns(self, new_patterns: Iterable[LTPattern]) -> int:
        existing_ids = {p.pattern_id for p in self.patterns}
        existing_signatures = {
            (p.product, p.period, tuple(sorted((i, j, round(q, 6)) for (i, j), q in p.pattern_flows.items())))
            for p in self.patterns
        }
        added = 0
        self.last_duplicate_rejects = 0
        self.last_signature_rejects = 0
        self.last_empty_rejects = 0
        self.last_added_patterns = 0
        for pat in new_patterns:
            signature = (
                pat.product,
                pat.period,
                tuple(sorted((i, j, round(q, 6)) for (i, j), q in pat.pattern_flows.items())),
            )
            duplicate_id = pat.pattern_id in existing_ids
            duplicate_signature = signature in existing_signatures
            empty_flow = not pat.pattern_flows
            added_to_pool = False
            if duplicate_id:
                self.last_duplicate_rejects += 1
            if duplicate_signature:
                self.last_signature_rejects += 1
            if empty_flow:
                self.last_empty_rejects += 1
            if duplicate_id or duplicate_signature or empty_flow:
                self._record_pattern_pool_diagnostic(
                    pat=pat,
                    stage="add_patterns_rejected",
                    pattern_reduced_cost=pat.metadata.get("reduced_cost"),
                    duplicate_id_reject=duplicate_id,
                    duplicate_signature_reject=duplicate_signature,
                    empty_flow_reject=empty_flow,
                    added_to_pool=False,
                    signature=str(signature),
                )
                continue
            self.patterns.append(pat)
            existing_ids.add(pat.pattern_id)
            existing_signatures.add(signature)
            added += 1
            self.last_added_patterns += 1
            added_to_pool = True
            self._record_pattern_pool_diagnostic(
                pat=pat,
                stage="add_patterns_added",
                pattern_reduced_cost=pat.metadata.get("reduced_cost"),
                duplicate_id_reject=False,
                duplicate_signature_reject=False,
                empty_flow_reject=False,
                added_to_pool=added_to_pool,
                signature=str(signature),
            )
        return added

    def _record_pattern_pool_diagnostic(
        self,
        pat: LTPattern,
        stage: str,
        pattern_reduced_cost: Optional[float] = None,
        duplicate_id_reject: bool = False,
        duplicate_signature_reject: bool = False,
        empty_flow_reject: bool = False,
        added_to_pool: bool = False,
        signature: str = "",
    ) -> None:
        if pat.pattern_flows:
            items = list(pat.pattern_flows.items())
        else:
            items = [((None, None), 0.0)]
        for (donor, receiver), qty in items:
            self.column_pool_diagnostics.append({
                "episode": self.current_episode,
                "stage": stage,
                "product": pat.product,
                "period": pat.period,
                "feature_name": pat.metadata.get("feature_name"),
                "donor_store": donor,
                "receiver_store": receiver,
                "qty_cap": qty,
                "reduced_cost_proxy": None,
                "stackelberg_accepted": None,
                "acceptance_score": pat.metadata.get("mean_acceptance_score"),
                "compensation": pat.metadata.get("mean_compensation"),
                "pattern_id": pat.pattern_id,
                "pattern_reduced_cost": pattern_reduced_cost,
                "gnn_score": pat.metadata.get("gnn_score"),
                "gnn_combined_score": pat.metadata.get("gnn_combined_score"),
                "gnn_selected": pat.metadata.get("gnn_selected"),
                "gnn_selected_by_fallback": pat.metadata.get("gnn_selected_by_fallback"),
                "adaptive_k_star": pat.metadata.get("adaptive_k_star"),
                "duplicate_id_reject": duplicate_id_reject,
                "duplicate_signature_reject": duplicate_signature_reject,
                "empty_flow_reject": empty_flow_reject,
                "added_to_pool": added_to_pool,
                "signature": signature,
            })

    def _build_need_and_surplus_proxies(self, master_solution: Optional[CGSolution] = None):
        d = self.data
        need, surplus = {}, {}
        sorted_periods = sorted(d.periods)
        period_pos = {t: idx for idx, t in enumerate(sorted_periods)}
        for s, p, t in itertools.product(d.stores, d.products, d.periods):
            idx = period_pos[t]
            need_window = sorted_periods[idx:idx + self.need_lookahead_periods]
            reserve_window = sorted_periods[idx:idx + self.surplus_reserve_periods]
            demand_cover_target = sum(
                float(d.realized_demand.get((s, p, tau), d.demand.get((s, p, tau), 0.0)))
                for tau in need_window
            )
            reserve_target = max(
                self.safety_stock_units,
                sum(
                    float(d.realized_demand.get((s, p, tau), d.demand.get((s, p, tau), 0.0)))
                    for tau in reserve_window
                ),
            )
            ending_inventory = max(0.0, float(
                d.post_shock_inventory.get((s, p, t), self.baseline.inv_store[(s, p, t)])
            ))
            shortage = max(0.0, float(
                d.post_shock_shortage.get((s, p, t), self.baseline.shortage[(s, p, t)])
            ))
            need[(s, p, t)] = max(shortage, demand_cover_target - ending_inventory)
            surplus[(s, p, t)] = max(0.0, ending_inventory - reserve_target)
        if master_solution is not None:
            for key, net_lt in master_solution.implied_net_lt.items():
                if key not in need:
                    continue
                net_lt = float(net_lt)
                if net_lt > 1e-9:
                    need[key] = max(0.0, need[key] - net_lt)
                elif net_lt < -1e-9:
                    surplus[key] = max(0.0, surplus[key] + net_lt)
        return need, surplus

    def _compute_active_product_periods(self, need, surplus) -> Set[Tuple[Product, Period]]:
        active = set()
        for p in self.data.products:
            for t in self.data.periods:
                total_need = sum(need[(s, p, t)] for s in self.data.stores)
                total_surplus = sum(surplus[(s, p, t)] for s in self.data.stores)
                min_signal = max(self.lt_activation_threshold, 1e-9)
                if total_need > min_signal and total_surplus > min_signal:
                    active.add((p, t))
        return active

    def _feature_value_map(
        self,
        p: Product,
        t: Period,
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
    ) -> Dict[Tuple[Store, Store], Dict[str, float]]:
        total_need = sum(need[(s, p, t)] for s in self.data.stores)
        total_surplus = sum(surplus[(s, p, t)] for s in self.data.stores)

        feature_map: Dict[Tuple[Store, Store], Dict[str, float]] = {}
        for i in self.data.stores:
            for j in self.data.stores:
                if i == j:
                    continue
                donor_surplus = surplus[(i, p, t)]
                recv_need = need[(j, p, t)]
                if donor_surplus <= 1e-9 or recv_need <= 1e-9:
                    continue
                feature_map[(i, j)] = {
                    "shortage_ratio": recv_need / max(total_need, 1e-9),
                    "surplus_ratio": donor_surplus / max(total_surplus, 1e-9),
                    "time_urgency": 1.0 / (1.0 + self._estimate_days_until_stockout(j, p, t, recv_need)),
                }
        return feature_map

    def _estimate_days_until_stockout(self, store: Store, product: Product, period: Period, receiver_need: float) -> float:
        demand_now = max(0.0, float(
            self.data.realized_demand.get(
                (store, product, period),
                self.data.demand.get((store, product, period), 0.0),
            )
        ))
        if receiver_need > 1e-9:
            if demand_now <= 1e-9:
                return 0.0
            serviceable_qty_before_stockout = max(0.0, demand_now - receiver_need)
            return serviceable_qty_before_stockout / demand_now

        ending_inventory = max(0.0, float(
            self.data.post_shock_inventory.get(
                (store, product, period),
                self.baseline.inv_store.get((store, product, period), 0.0),
            )
        ))
        future_demands = [
            max(0.0, float(
                self.data.realized_demand.get(
                    (store, product, tau),
                    self.data.demand.get((store, product, tau), 0.0),
                )
            ))
            for tau in self.data.periods
            if tau >= period
        ]
        positive_future_demands = [value for value in future_demands if value > 1e-9]
        if not positive_future_demands:
            return math.inf
        avg_future_demand = sum(positive_future_demands) / len(positive_future_demands)
        return ending_inventory / max(avg_future_demand, 1e-9)

    def _prune_pairs_by_feature(
        self,
        p: Product,
        t: Period,
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
        episode: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        feature_values = self._feature_value_map(p=p, t=t, need=need, surplus=surplus)
        self._last_candidate_pair_count = len(feature_values)
        pruned: Dict[str, List[Dict[str, Any]]] = {feature: [] for feature in self.feature_ranges.keys()}

        for (i, j), fvals in feature_values.items():
            donor_surplus = surplus[(i, p, t)]
            recv_need = need[(j, p, t)]
            qty_cap = min(donor_surplus, recv_need)
            if qty_cap <= 1e-9:
                continue
            unit_cost = self.data.ship_cost_lt[(i, j, p)]
            fixed_cost = self.data.fixed_dispatch_lt[(i, j)]
            dual_score = dual_need.get((j, p, t), 0.0) + dual_surplus.get((i, p, t), 0.0)
            reduced_cost_proxy = fixed_cost + unit_cost * qty_cap - dual_score * qty_cap
            fvals["negative_reduced_cost"] = max(0.0, -reduced_cost_proxy)
            base_rank_score = dual_score - unit_cost

            payload = {
                "pair": (i, j),
                "qty_cap": qty_cap,
                "fixed_cost": fixed_cost,
                "unit_cost": unit_cost,
                "dual_score": dual_score,
                "reduced_cost_proxy": reduced_cost_proxy,
                "base_rank_score": base_rank_score,
                "feature_values": fvals,
            }

            for feature_name, bounds in self.feature_ranges.items():
                fval = fvals[feature_name]
                if bounds["min"] <= fval <= bounds["max"]:
                    feature_bonus = 0.05 * fval
                    payload_copy = dict(payload)
                    payload_copy["feature_name"] = feature_name
                    payload_copy["feature_score"] = payload["base_rank_score"] + feature_bonus
                    pruned[feature_name].append(payload_copy)
                    self.column_pool_diagnostics.append({
                        "episode": episode,
                        "stage": "candidate_pair_before_stackelberg",
                        "product": p,
                        "period": t,
                        "feature_name": feature_name,
                        "donor_store": i,
                        "receiver_store": j,
                        "qty_cap": qty_cap,
                        "reduced_cost_proxy": reduced_cost_proxy,
                        "stackelberg_accepted": None,
                        "acceptance_score": None,
                        "compensation": None,
                        "pattern_id": "",
                        "pattern_reduced_cost": None,
                        "gnn_score": None,
                        "gnn_combined_score": None,
                        "gnn_selected": None,
                        "gnn_selected_by_fallback": None,
                        "adaptive_k_star": None,
                        "duplicate_id_reject": False,
                        "duplicate_signature_reject": False,
                        "empty_flow_reject": False,
                        "added_to_pool": False,
                        "signature": "",
                    })

        for feature_name, rows in pruned.items():
            rows.sort(key=lambda x: x["feature_score"], reverse=True)
            pruned[feature_name] = rows[:self.top_pairs_per_feature]
        return pruned

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _solve_stackelberg_for_pair(
        self,
        *,
        p: Product,
        t: Period,
        row: Dict[str, Any],
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
    ) -> StackelbergDecision:
        params = self.stackelberg_params

        i, j = row["pair"]
        q = float(max(0.0, row["qty_cap"]))

        if q <= 1e-9:
            return StackelbergDecision(
                accepted=False,
                compensation=0.0,
                donor_utility=-1e9,
                receiver_utility=-1e9,
                acceptance_score=0.0,
                details={"reason": -1.0},
            )

        donor_surplus = max(0.0, surplus[(i, p, t)])
        receiver_need = max(0.0, need[(j, p, t)])

        shortage_risk_increase = q / max(donor_surplus, 1e-9)
        service_level_loss = q / max(donor_surplus + 1.0, 1e-9)
        ship_burden = row["unit_cost"] * q + row["fixed_cost"]

        donor_noncomp_cost = (
            params.donor_risk_weight * shortage_risk_increase
            + params.donor_ship_burden_weight * ship_burden
            + params.donor_service_loss_weight * service_level_loss
        )
        donor_required_comp = params.donor_accept_threshold + donor_noncomp_cost

        shortage_reduction = min(q, receiver_need) / max(receiver_need, 1e-9)
        service_gain = min(q, receiver_need) / max(receiver_need + 1.0, 1e-9)
        handling_cost = 0.25 * row["unit_cost"] * q

        receiver_benefit_before_comp = (
            params.receiver_shortage_reduction_weight * shortage_reduction
            + params.receiver_service_gain_weight * service_gain
            - params.receiver_handling_weight * handling_cost
        )
        receiver_max_comp = receiver_benefit_before_comp - params.receiver_accept_threshold

        compensation = max(params.min_compensation, donor_required_comp)

        accepted = (
            compensation <= receiver_max_comp
            and compensation <= params.compensation_cap
        )

        donor_utility = compensation - donor_noncomp_cost
        receiver_utility = receiver_benefit_before_comp - compensation

        acceptance_score = 0.5 * (
            self._sigmoid(donor_utility) + self._sigmoid(receiver_utility)
        )

        return StackelbergDecision(
            accepted=accepted,
            compensation=float(compensation),
            donor_utility=float(donor_utility),
            receiver_utility=float(receiver_utility),
            acceptance_score=float(acceptance_score if accepted else 0.0),
            details={
                "q": float(q),
                "donor_surplus": float(donor_surplus),
                "receiver_need": float(receiver_need),
                "shortage_risk_increase": float(shortage_risk_increase),
                "service_level_loss": float(service_level_loss),
                "ship_burden": float(ship_burden),
                "shortage_reduction": float(shortage_reduction),
                "service_gain": float(service_gain),
                "handling_cost": float(handling_cost),
                "donor_required_comp": float(donor_required_comp),
                "receiver_max_comp": float(receiver_max_comp),
            },
        )

    def _apply_stackelberg_game_to_pairs(
        self,
        p: Product,
        t: Period,
        pruned_pairs_by_feature: Dict[str, List[Dict[str, Any]]],
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        episode: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        params = self.stackelberg_params
        accepted_by_feature: Dict[str, List[Dict[str, Any]]] = {
            feature_name: [] for feature_name in pruned_pairs_by_feature.keys()
        }

        for feature_name, rows in pruned_pairs_by_feature.items():
            rejected_candidates: List[Dict[str, Any]] = []
            for row in rows:
                decision = self._solve_stackelberg_for_pair(
                    p=p,
                    t=t,
                    row=row,
                    need=need,
                    surplus=surplus,
                )

                row2 = dict(row)
                row2["stackelberg_accepted"] = decision.accepted
                row2["compensation"] = decision.compensation
                row2["donor_utility"] = decision.donor_utility
                row2["receiver_utility"] = decision.receiver_utility
                row2["acceptance_score"] = decision.acceptance_score
                row2["stackelberg_details"] = decision.details
                i, j = row["pair"]
                self.column_pool_diagnostics.append({
                    "episode": episode,
                    "stage": "after_stackelberg",
                    "product": p,
                    "period": t,
                    "feature_name": feature_name,
                    "donor_store": i,
                    "receiver_store": j,
                    "qty_cap": row.get("qty_cap"),
                    "reduced_cost_proxy": row.get("reduced_cost_proxy"),
                    "stackelberg_accepted": decision.accepted,
                    "acceptance_score": decision.acceptance_score,
                    "compensation": decision.compensation,
                    "pattern_id": "",
                    "pattern_reduced_cost": None,
                    "gnn_score": None,
                    "gnn_combined_score": None,
                    "gnn_selected": None,
                    "gnn_selected_by_fallback": None,
                    "adaptive_k_star": None,
                    "duplicate_id_reject": False,
                    "duplicate_signature_reject": False,
                    "empty_flow_reject": False,
                    "added_to_pool": False,
                    "signature": "",
                })

                if not decision.accepted:
                    fallback_economic_score = max(0.0, -float(row.get("reduced_cost_proxy", 0.0)))
                    row2["post_game_score"] = fallback_economic_score
                    row2["stackelberg_fallback_used"] = False
                    rejected_candidates.append(row2)
                    continue

                combined_score = (
                    params.acceptance_score_weight * decision.acceptance_score
                    + params.economic_score_weight * row["base_rank_score"]
                )
                row2["post_game_score"] = combined_score
                row2["stackelberg_fallback_used"] = False
                accepted_by_feature[feature_name].append(row2)

            if not accepted_by_feature[feature_name] and params.allow_pricing_fallback_when_no_acceptance:
                fallback_rows = []
                for row in rejected_candidates:
                    if (
                        params.fallback_requires_negative_reduced_cost
                        and float(row.get("reduced_cost_proxy", 0.0)) >= -1e-9
                    ):
                        continue
                    fallback_row = dict(row)
                    fallback_row["stackelberg_fallback_used"] = True
                    fallback_row["acceptance_score"] = max(
                        float(fallback_row.get("acceptance_score", 0.0)),
                        0.10,
                    )
                    fallback_row["post_game_score"] = (
                        max(0.0, -float(fallback_row.get("reduced_cost_proxy", 0.0)))
                        + 0.01 * float(fallback_row.get("base_rank_score", 0.0))
                    )
                    fallback_rows.append(fallback_row)
                fallback_rows.sort(key=lambda x: x["post_game_score"], reverse=True)
                fallback_rows = fallback_rows[:max(1, int(params.fallback_top_k_after_game_per_feature))]
                accepted_by_feature[feature_name].extend(fallback_rows)
                for row in fallback_rows:
                    i, j = row["pair"]
                    self.column_pool_diagnostics.append({
                        "episode": episode,
                        "stage": "stackelberg_pricing_fallback",
                        "product": p,
                        "period": t,
                        "feature_name": feature_name,
                        "donor_store": i,
                        "receiver_store": j,
                        "qty_cap": row.get("qty_cap"),
                        "reduced_cost_proxy": row.get("reduced_cost_proxy"),
                        "stackelberg_accepted": False,
                        "acceptance_score": row.get("acceptance_score"),
                        "compensation": row.get("compensation"),
                        "pattern_id": "",
                        "pattern_reduced_cost": None,
                        "gnn_score": None,
                        "gnn_combined_score": None,
                        "gnn_selected": None,
                        "gnn_selected_by_fallback": None,
                        "adaptive_k_star": None,
                        "duplicate_id_reject": False,
                        "duplicate_signature_reject": False,
                        "empty_flow_reject": False,
                        "added_to_pool": False,
                        "signature": "",
                    })

            accepted_by_feature[feature_name].sort(
                key=lambda x: x["post_game_score"], reverse=True
            )
            accepted_by_feature[feature_name] = accepted_by_feature[feature_name][
                :params.top_k_after_game_per_feature
            ]

        return accepted_by_feature

    def _build_patterns_from_pruned_pairs(
        self,
        p: Product,
        t: Period,
        pruned_pairs_by_feature: Dict[str, List[Dict[str, Any]]],
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        rc_tol: float,
        episode: int = 0,
    ) -> List[LTPattern]:
        new_patterns: List[LTPattern] = []
        for feature_name, rows in pruned_pairs_by_feature.items():
            if not rows:
                continue
            built_here = 0
            for start_idx in range(min(len(rows), self.top_patterns_per_feature)):
                donor_work = {s: surplus[(s, p, t)] for s in self.data.stores}
                recv_work = {s: need[(s, p, t)] for s in self.data.stores}
                flows: Dict[Tuple[Store, Store], float] = {}
                pattern_cost = 0.0
                reduced_cost = 0.0

                ordered_rows = rows[start_idx:] + rows[:start_idx]
                for row in ordered_rows:
                    if len(flows) >= self.max_pairs_per_pattern:
                        break
                    i, j = row["pair"]
                    qty = min(donor_work.get(i, 0.0), recv_work.get(j, 0.0), row["qty_cap"])
                    if qty <= 1e-9:
                        continue
                    pair_rc = row["fixed_cost"] + row["unit_cost"] * qty - row["dual_score"] * qty
                    if pair_rc >= -1e-9 and flows:
                        continue
                    flows[(i, j)] = qty
                    donor_work[i] -= qty
                    recv_work[j] -= qty
                    pattern_cost += row["fixed_cost"] + row["unit_cost"] * qty
                    reduced_cost += pair_rc

                if flows and reduced_cost < rc_tol:
                    built_here += 1
                    pair_signature = tuple(sorted((i, j, round(qty, 6)) for (i, j), qty in flows.items()))
                    fallback_pair_count = sum(1 for r in rows if r.get("stackelberg_fallback_used"))
                    new_patterns.append(
                        LTPattern(
                            pattern_id=f"PRICED_E{episode}_{feature_name}_{p}_T{t}_{built_here}",
                            period=t,
                            product=p,
                            pattern_flows=flows,
                            column_cost=round(pattern_cost, 6),
                            metadata={
                                "source": "pricing_pruned_feature",
                                "feature_name": feature_name,
                                "reduced_cost": round(reduced_cost, 6),
                                "pruned_pair_count": len(rows),
                                "feature_range": self.feature_ranges[feature_name],
                                "stackelberg_used": True,
                                "stackelberg_fallback_used": fallback_pair_count > 0,
                                "stackelberg_fallback_pair_count": fallback_pair_count,
                                "mean_acceptance_score": round(
                                    sum(r.get("acceptance_score", 0.0) for r in rows) / max(len(rows), 1), 6
                                ),
                                "mean_compensation": round(
                                    sum(r.get("compensation", 0.0) for r in rows) / max(len(rows), 1), 6
                                ),
                                "pair_signature": str(pair_signature),
                            },
                        )
                    )
                elif flows:
                    self.column_pool_diagnostics.append({
                        "episode": episode,
                        "stage": "pattern_rejected_reduced_cost",
                        "product": p,
                        "period": t,
                        "feature_name": feature_name,
                        "donor_store": "",
                        "receiver_store": "",
                        "qty_cap": sum(flows.values()),
                        "reduced_cost_proxy": None,
                        "stackelberg_accepted": None,
                        "acceptance_score": None,
                        "compensation": None,
                        "pattern_id": f"REJECTED_E{episode}_{feature_name}_{p}_T{t}_{start_idx + 1}",
                        "pattern_reduced_cost": round(reduced_cost, 6),
                        "gnn_score": None,
                        "gnn_combined_score": None,
                        "gnn_selected": None,
                        "gnn_selected_by_fallback": None,
                        "adaptive_k_star": None,
                        "duplicate_id_reject": False,
                        "duplicate_signature_reject": False,
                        "empty_flow_reject": False,
                        "added_to_pool": False,
                        "signature": str(tuple(sorted((i, j, round(qty, 6)) for (i, j), qty in flows.items()))),
                    })
        return new_patterns

    def _candidate_patterns_from_duals(
        self,
        need,
        surplus,
        active_product_periods: Set[Tuple[Product, Period]],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
        rc_tol: float = -1e-6,
        episode: int = 0,
    ) -> List[LTPattern]:
        new_patterns: List[LTPattern] = []
        candidate_pairs_before_pruning = 0
        pairs_after_pruning = 0
        pairs_accepted_stackelberg = 0
        pairs_recovered_stackelberg_fallback = 0

        for p, t in sorted(active_product_periods):
            pruned_pairs_by_feature = self._prune_pairs_by_feature(
                p=p,
                t=t,
                need=need,
                surplus=surplus,
                dual_need=dual_need,
                dual_surplus=dual_surplus,
                episode=episode,
            )
            candidate_pairs_before_pruning += self._last_candidate_pair_count
            pairs_after_pruning += sum(len(rows) for rows in pruned_pairs_by_feature.values())
            accepted_pairs_by_feature = self._apply_stackelberg_game_to_pairs(
                p=p,
                t=t,
                pruned_pairs_by_feature=pruned_pairs_by_feature,
                need=need,
                surplus=surplus,
                episode=episode,
            )
            pairs_accepted_stackelberg += sum(
                1
                for rows in accepted_pairs_by_feature.values()
                for row in rows
                if row.get("stackelberg_accepted") and not row.get("stackelberg_fallback_used")
            )
            pairs_recovered_stackelberg_fallback += sum(
                1
                for rows in accepted_pairs_by_feature.values()
                for row in rows
                if row.get("stackelberg_fallback_used")
            )
            feature_patterns = self._build_patterns_from_pruned_pairs(
                p=p,
                t=t,
                pruned_pairs_by_feature=accepted_pairs_by_feature,
                need=need,
                surplus=surplus,
                rc_tol=rc_tol,
                episode=episode,
            )
            new_patterns.extend(feature_patterns)
        patterns_built_before_dedup = len(new_patterns)
        new_patterns = self._deduplicate_priced_patterns(new_patterns, episode=episode)
        patterns_after_dedup = len(new_patterns)
        if self.max_columns_per_product_period > 0:
            capped_patterns = []
            by_product_period: Dict[Tuple[Product, Period], List[LTPattern]] = {}
            for pat in new_patterns:
                by_product_period.setdefault((pat.product, pat.period), []).append(pat)
            for key in sorted(by_product_period):
                rows = by_product_period[key]
                rows.sort(key=lambda pat: float(pat.metadata.get("reduced_cost", 0.0) or 0.0))
                capped_patterns.extend(rows[:self.max_columns_per_product_period])
            new_patterns = capped_patterns
        self._last_pricing_summary = {
            "candidate_pairs_before_pruning": candidate_pairs_before_pruning,
            "pairs_after_pruning": pairs_after_pruning,
            "pairs_accepted_stackelberg": pairs_accepted_stackelberg,
            "pairs_recovered_stackelberg_fallback": pairs_recovered_stackelberg_fallback,
            "patterns_built_before_dedup": patterns_built_before_dedup,
            "patterns_deduplicated_before_gnn": patterns_built_before_dedup - patterns_after_dedup,
            "patterns_removed_by_product_period_cap": patterns_after_dedup - len(new_patterns),
            "patterns_built_before_gnn": len(new_patterns),
            "max_columns_per_product_period": self.max_columns_per_product_period,
        }
        return new_patterns

    def solve_rmp(self, msg: bool = False, return_model: bool = False):
        d = self.data
        need, surplus = self._build_need_and_surplus_proxies()
        active_product_periods = self._compute_active_product_periods(need, surplus)
        mdl = gp.Model("LT_RMP", env=get_gurobi_env())
        mdl.Params.OutputFlag = 1 if msg else 0

        pattern_map = {pat.pattern_id: pat for pat in self.patterns if (pat.product, pat.period) in active_product_periods}
        lam = mdl.addVars(list(pattern_map.keys()), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="lambda")
        infeasible_branch = False
        for pid, (lb, ub) in self.branch_bounds.items():
            lb = max(0.0, float(lb))
            ub = min(1.0, float(ub))
            if pid not in pattern_map:
                if lb > 1e-9:
                    infeasible_branch = True
                continue
            lam[pid].LB = lb
            lam[pid].UB = ub
        residual_need_keys = [(s, p, t) for s in d.stores for p in d.products for t in d.periods]
        residual_need = mdl.addVars(residual_need_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="residual_need")

        baseline_shortage_component = sum(
            d.shortage_cost[(s, p)] * float(
                d.post_shock_shortage.get((s, p, t), self.baseline.shortage[(s, p, t)])
            )
            for s, p, t in itertools.product(d.stores, d.products, d.periods)
        )
        baseline_without_shortage = float(self.baseline.objective) - baseline_shortage_component
        need_penalty = {
            (s, p, t): (
                d.shortage_cost[(s, p)]
                if float(d.post_shock_shortage.get((s, p, t), self.baseline.shortage[(s, p, t)])) > 1e-9
                else self.rebalance_need_penalty
            )
            for s, p, t in residual_need_keys
        }

        mdl.setObjective(
            baseline_without_shortage
            + gp.quicksum(pat.column_cost * lam[pat.pattern_id] for pat in pattern_map.values())
            + gp.quicksum(need_penalty[(s, p, t)] * residual_need[(s, p, t)] for s, p, t in residual_need_keys),
            GRB.MINIMIZE,
        )

        need_constraints = {}
        surplus_constraints = {}

        for s, p, t in residual_need_keys:
            relevant = [pat for pat in pattern_map.values() if pat.product == p and pat.period == t]
            inflow = gp.quicksum(
                qty * lam[pat.pattern_id]
                for pat in relevant
                for (i, j), qty in pat.pattern_flows.items()
                if j == s
            )
            con = mdl.addConstr(residual_need[(s, p, t)] + inflow >= need[(s, p, t)], name=f"need_cover__{len(need_constraints)}")
            need_constraints[(s, p, t)] = con

            if (p, t) not in active_product_periods:
                mdl.addConstr(residual_need[(s, p, t)] == need[(s, p, t)], name=f"inactive_fix__{s}__{t}__{len(need_constraints)}")

        for s, p, t in residual_need_keys:
            if (p, t) not in active_product_periods:
                continue
            relevant = [pat for pat in pattern_map.values() if pat.product == p and pat.period == t]
            outbound = gp.quicksum(
                qty * lam[pat.pattern_id]
                for pat in relevant
                for (i, j), qty in pat.pattern_flows.items()
                if i == s
            )
            con = mdl.addConstr(outbound <= surplus[(s, p, t)], name=f"surplus_cap__{len(surplus_constraints)}")
            surplus_constraints[(s, p, t)] = con

        if infeasible_branch:
            mdl.addConstr(gp.LinExpr(0.0) >= 1.0, name="infeasible_missing_forced_branch_pattern")

        mdl.optimize()

        lambda_values = {pid: _safe_var_value(mdl, lam[pid]) for pid in pattern_map.keys()}
        selected = [k for k, v in lambda_values.items() if v > 1e-6]

        dual_need = {
            key: float(con.Pi) if mdl.Status == GRB.OPTIMAL else 0.0
            for key, con in need_constraints.items()
        }
        dual_surplus = {
            key: float(con.Pi) if mdl.Status == GRB.OPTIMAL else 0.0
            for key, con in surplus_constraints.items()
        }

        implied_net_lt = {}
        for s, p, t in residual_need_keys:
            net = 0.0
            for pat in pattern_map.values():
                if pat.product != p or pat.period != t:
                    continue
                coeff = lambda_values[pat.pattern_id]
                for (i, j), qty in pat.pattern_flows.items():
                    if j == s:
                        net += qty * coeff
                    if i == s:
                        net -= qty * coeff
            implied_net_lt[(s, p, t)] = net

        sol = CGSolution(
            status=_grb_status_name(mdl.Status),
            objective=_safe_obj_value(mdl),
            lambda_values=lambda_values,
            selected_patterns=selected,
            implied_net_lt=implied_net_lt,
            dual_need=dual_need,
            dual_surplus=dual_surplus,
            active_product_periods=active_product_periods,
            efficiency_metrics=_model_efficiency_metrics(mdl),
        )
        if return_model:
            return sol, mdl
        return sol

    def pricing_step(self, master_solution: CGSolution, rc_tol: float = -1e-6) -> List[LTPattern]:
        need, surplus = self._build_need_and_surplus_proxies(master_solution=master_solution)
        active_product_periods = self._compute_active_product_periods(need, surplus)
        new_patterns = self._candidate_patterns_from_duals(
            need=need,
            surplus=surplus,
            active_product_periods=active_product_periods,
            dual_need=master_solution.dual_need,
            dual_surplus=master_solution.dual_surplus,
            rc_tol=rc_tol,
            episode=self.current_episode,
        )
        if self.collect_teacher_mode:
            selected_patterns = self._collect_teacher_batch_without_gnn_prefilter(
                patterns=new_patterns,
                need=need,
                surplus=surplus,
                dual_need=master_solution.dual_need,
                dual_surplus=master_solution.dual_surplus,
            )
        elif self.runtime_gnn_mode:
            selected_patterns = self._select_patterns_with_gnn(
                patterns=new_patterns,
                need=need,
                surplus=surplus,
                dual_need=master_solution.dual_need,
                dual_surplus=master_solution.dual_surplus,
            )
        else:
            selected_patterns = new_patterns
        self._last_pricing_summary["active_product_period_count"] = len(active_product_periods)
        self._last_pricing_summary["patterns_kept_after_gnn"] = len(selected_patterns)
        self._last_pricing_summary["collect_teacher_mode"] = self.collect_teacher_mode
        self._last_pricing_summary["runtime_gnn_mode"] = self.runtime_gnn_mode and not self.collect_teacher_mode
        return selected_patterns

    def run_column_generation(
        self,
        max_iter: int = 10,
        improvement_tol: float = 1e-5,
        rc_tol: float = -1e-6,
        msg: bool = False,
    ) -> CGSolution:
        best_sol = self.solve_rmp(msg=msg)
        best_sol.iterations_run = 0
        rmp_metrics_total = dict(best_sol.efficiency_metrics)
        rmp_metrics_total["rmp_solves"] = 1.0
        best_sol.efficiency_metrics = dict(rmp_metrics_total)
        prev_obj = best_sol.objective
        self.current_episode = 0
        self.cg_history = [{
            "episode": 0,
            "total_cost": float(best_sol.objective),
            "improvement": 0.0,
            "proposed_columns": 0,
            "added_columns": 0,
            "selected_patterns": len(best_sol.selected_patterns),
        }]

        print("\n[CG] Active (product, period) pairs with positive need and surplus:")
        if not best_sol.active_product_periods:
            print("  None. RMP is not activated for any product-period.")
            self.cg_episode_diagnostics.append({
                "episode": 0,
                "active_product_period_count": 0,
                "candidate_pairs_before_pruning": 0,
                "pairs_after_pruning": 0,
                "pairs_accepted_stackelberg": 0,
                "pairs_recovered_stackelberg_fallback": 0,
                "patterns_built_before_dedup": 0,
                "patterns_deduplicated_before_gnn": 0,
                "patterns_built_before_gnn": 0,
                "patterns_kept_after_gnn": 0,
                "duplicate_id_rejects": 0,
                "duplicate_signature_rejects": 0,
                "empty_flow_rejects": 0,
                "patterns_added_to_pool": 0,
                "selected_patterns": len(best_sol.selected_patterns),
                "objective": float(best_sol.objective),
                "improvement": 0.0,
            })
            self._print_cg_episode_history()
            best_sol.efficiency_metrics = dict(rmp_metrics_total)
            return best_sol
        for p, t in sorted(best_sol.active_product_periods):
            print(f"  product={p} | period={t}")

        print("\n[RMP] Initially selected LT patterns:")
        if not best_sol.selected_patterns:
            print("  None")
        else:
            for pat_id in best_sol.selected_patterns:
                pat = next(p for p in self.patterns if p.pattern_id == pat_id)
                print("  " + format_pattern_detail(pat) + f" | lambda={best_sol.lambda_values[pat_id]:.4f}")

        for it in range(1, max_iter + 1):
            self.current_episode = it
            new_patterns = self.pricing_step(best_sol, rc_tol=rc_tol)
            added = self.add_patterns(new_patterns)
            self._mark_teacher_rows_passed_to_rmp(it)
            episode_summary = dict(self._last_pricing_summary)
            episode_summary.update({
                "episode": it,
                "duplicate_id_rejects": self.last_duplicate_rejects,
                "duplicate_signature_rejects": self.last_signature_rejects,
                "empty_flow_rejects": self.last_empty_rejects,
                "patterns_added_to_pool": added,
                "selected_patterns": len(best_sol.selected_patterns),
                "objective": float(best_sol.objective),
                "improvement": 0.0,
            })

            print(f"\n[Pricing] Iter {it}: proposed={len(new_patterns)}, added={added}")
            for pat in new_patterns[:10]:
                print(
                    "  " + format_pattern_detail(pat)
                    + f" | feature={pat.metadata.get('feature_name')}"
                    + f" | rc={pat.metadata.get('reduced_cost')}"
                    + f" | mean_acceptance={pat.metadata.get('mean_acceptance_score')}"
                    + f" | mean_comp={pat.metadata.get('mean_compensation')}"
                    + f" | gnn_score={pat.metadata.get('gnn_score')}"
                )

            if added == 0:
                best_sol.iterations_run = it - 1
                best_sol.efficiency_metrics = dict(rmp_metrics_total)
                self.cg_history.append({
                    "episode": it,
                    "total_cost": float(best_sol.objective),
                    "improvement": 0.0,
                    "proposed_columns": len(new_patterns),
                    "added_columns": added,
                    "selected_patterns": len(best_sol.selected_patterns),
                })
                self.cg_episode_diagnostics.append(episode_summary)
                print("[CG] No negative reduced-cost columns found. Stop.")
                self._print_cg_episode_history()
                return best_sol

            sol = self.solve_rmp(msg=msg)
            sol.iterations_run = it
            _add_efficiency_metrics(rmp_metrics_total, sol.efficiency_metrics)
            rmp_metrics_total["rmp_solves"] = float(rmp_metrics_total.get("rmp_solves", 0.0)) + 1.0
            sol.efficiency_metrics = dict(rmp_metrics_total)
            improvement = prev_obj - sol.objective
            self._annotate_teacher_rows_after_reopt(it, sol, improvement)
            episode_summary["selected_patterns"] = len(sol.selected_patterns)
            episode_summary["objective"] = float(sol.objective)
            episode_summary["improvement"] = float(improvement)
            self.cg_episode_diagnostics.append(episode_summary)
            self.cg_history.append({
                "episode": it,
                "total_cost": float(sol.objective),
                "improvement": float(improvement),
                "proposed_columns": len(new_patterns),
                "added_columns": added,
                "selected_patterns": len(sol.selected_patterns),
            })
            print(f"[CG] Iter {it}: objective = {sol.objective:.6f}, improvement = {improvement:.6f}")

            print("[RMP] Selected LT patterns after re-optimization:")
            if not sol.selected_patterns:
                print("  None")
            else:
                for pat_id in sol.selected_patterns:
                    pat = next(p for p in self.patterns if p.pattern_id == pat_id)
                    print("  " + format_pattern_detail(pat) + f" | lambda={sol.lambda_values[pat_id]:.4f}")

            if improvement <= improvement_tol:
                self._print_cg_episode_history()
                sol.efficiency_metrics = dict(rmp_metrics_total)
                return sol
            prev_obj = sol.objective
            best_sol = sol

        self._print_cg_episode_history()
        best_sol.efficiency_metrics = dict(rmp_metrics_total)
        return best_sol

    @staticmethod
    def _fractional_lambda_values(sol: CGSolution, int_tol: float = 1e-5) -> List[Tuple[str, float]]:
        fractional = []
        for pid, value in sol.lambda_values.items():
            value = float(value)
            if int_tol < value < 1.0 - int_tol:
                fractional.append((pid, value))
        fractional.sort(key=lambda item: abs(item[1] - 0.5))
        return fractional

    def run_branch_and_price(
        self,
        max_iter: int = 10,
        improvement_tol: float = 1e-5,
        rc_tol: float = -1e-6,
        msg: bool = False,
        max_nodes: int = 15,
        max_depth: int = 6,
        int_tol: float = 1e-5,
    ) -> CGSolution:
        original_branch_bounds = dict(self.branch_bounds)
        self.branch_history = []
        self.cg_history_all_nodes = []
        best_integer_sol: Optional[CGSolution] = None
        best_relaxation_sol: Optional[CGSolution] = None
        best_bound = math.inf
        nodes_explored = 0
        nodes_pruned_by_bound = 0
        nodes_pruned_by_integrality = 0
        nodes_pruned_by_depth = 0
        nodes_infeasible = 0
        pending_nodes: List[Dict[str, Any]] = [{
            "node_id": 0,
            "parent_id": None,
            "depth": 0,
            "branch_var": "",
            "branch_sense": "root",
            "branch_value": None,
            "bounds": dict(original_branch_bounds),
        }]
        next_node_id = 1

        print("\n" + "=" * 80)
        print("STEP 3B - Branch-and-price on fractional RMP columns")
        print("=" * 80)

        try:
            while pending_nodes and nodes_explored < max_nodes:
                node = pending_nodes.pop()
                self.branch_bounds = dict(node["bounds"])
                self.current_branch_node_id = int(node["node_id"])
                nodes_explored += 1
                print(
                    f"\n[B&P] Node {node['node_id']} depth={node['depth']} "
                    f"| branch={node['branch_var'] or 'root'} {node['branch_sense']}"
                )
                sol = self.run_column_generation(
                    max_iter=max_iter,
                    improvement_tol=improvement_tol,
                    rc_tol=rc_tol,
                    msg=msg,
                )
                for row in self.cg_history:
                    row_with_node = dict(row)
                    row_with_node["branch_node_id"] = node["node_id"]
                    row_with_node["branch_depth"] = node["depth"]
                    row_with_node["branch_sense_from_parent"] = node["branch_sense"]
                    row_with_node["branch_var_from_parent"] = node["branch_var"]
                    self.cg_history_all_nodes.append(row_with_node)
                fractional = self._fractional_lambda_values(sol, int_tol=int_tol)
                if math.isfinite(sol.objective):
                    best_bound = min(best_bound, float(sol.objective))
                    if best_relaxation_sol is None or sol.objective < best_relaxation_sol.objective:
                        best_relaxation_sol = sol

                branch_pid = fractional[0][0] if fractional else ""
                branch_value = fractional[0][1] if fractional else None
                node_status = "open"

                if not math.isfinite(sol.objective) or sol.status in {"Infeasible", "InfOrUnbd", "Unbounded"}:
                    nodes_infeasible += 1
                    node_status = "infeasible"
                elif best_integer_sol is not None and sol.objective >= best_integer_sol.objective - improvement_tol:
                    nodes_pruned_by_bound += 1
                    node_status = "pruned_by_bound"
                elif not fractional:
                    nodes_pruned_by_integrality += 1
                    node_status = "integer_incumbent"
                    if best_integer_sol is None or sol.objective < best_integer_sol.objective:
                        best_integer_sol = sol
                elif node["depth"] >= max_depth:
                    nodes_pruned_by_depth += 1
                    node_status = "pruned_by_depth"
                else:
                    left_bounds = dict(node["bounds"])
                    left_bounds[branch_pid] = (0.0, 0.0)
                    right_bounds = dict(node["bounds"])
                    right_bounds[branch_pid] = (1.0, 1.0)
                    pending_nodes.append({
                        "node_id": next_node_id,
                        "parent_id": node["node_id"],
                        "depth": node["depth"] + 1,
                        "branch_var": branch_pid,
                        "branch_sense": "<= 0",
                        "branch_value": 0.0,
                        "bounds": left_bounds,
                    })
                    next_node_id += 1
                    pending_nodes.append({
                        "node_id": next_node_id,
                        "parent_id": node["node_id"],
                        "depth": node["depth"] + 1,
                        "branch_var": branch_pid,
                        "branch_sense": ">= 1",
                        "branch_value": 1.0,
                        "bounds": right_bounds,
                    })
                    next_node_id += 1
                    node_status = "branched"

                self.branch_history.append({
                    "node_id": node["node_id"],
                    "parent_id": node["parent_id"],
                    "depth": node["depth"],
                    "status": node_status,
                    "objective": float(sol.objective) if math.isfinite(sol.objective) else math.inf,
                    "fractional_lambda_count": len(fractional),
                    "branch_var": branch_pid,
                    "branch_lambda_value": branch_value,
                    "branch_sense_from_parent": node["branch_sense"],
                    "branch_value_from_parent": node["branch_value"],
                    "incumbent_objective": (
                        float(best_integer_sol.objective)
                        if best_integer_sol is not None and math.isfinite(best_integer_sol.objective)
                        else math.inf
                    ),
                })
                print(
                    f"[B&P] Node {node['node_id']} status={node_status} "
                    f"| obj={sol.objective:.6f} | fractional_lambdas={len(fractional)}"
                )
                if branch_pid and node_status == "branched":
                    print(f"[B&P] Branch on lambda[{branch_pid}] = {branch_value:.6f}")

            chosen_sol = best_integer_sol or best_relaxation_sol
            if chosen_sol is None:
                chosen_sol = CGSolution(
                    status="Infeasible",
                    objective=math.inf,
                    lambda_values={},
                    selected_patterns=[],
                    implied_net_lt={},
                )
            gap = 0.0
            if best_integer_sol is not None and math.isfinite(best_bound):
                gap = max(0.0, (best_integer_sol.objective - best_bound) / max(abs(best_integer_sol.objective), 1e-9))
            chosen_sol.branch_summary = {
                "status": "integer_incumbent" if best_integer_sol is not None else "no_integer_incumbent_returned_best_relaxation",
                "nodes_explored": nodes_explored,
                "nodes_remaining": len(pending_nodes),
                "nodes_pruned_by_bound": nodes_pruned_by_bound,
                "nodes_pruned_by_integrality": nodes_pruned_by_integrality,
                "nodes_pruned_by_depth": nodes_pruned_by_depth,
                "nodes_infeasible": nodes_infeasible,
                "best_bound": best_bound,
                "incumbent_objective": best_integer_sol.objective if best_integer_sol is not None else math.inf,
                "relative_gap": gap,
                "max_nodes": max_nodes,
                "max_depth": max_depth,
                "integer_incumbent_found": best_integer_sol is not None,
            }
            chosen_sol.efficiency_metrics["branch_price_nodes_explored"] = float(nodes_explored)
            chosen_sol.efficiency_metrics["branch_price_nodes_remaining"] = float(len(pending_nodes))
            print("\n[B&P Summary]")
            pprint.pprint(chosen_sol.branch_summary)
            return chosen_sol
        finally:
            self.current_branch_node_id = None
            self.branch_bounds = original_branch_bounds

    def _print_cg_episode_history(self) -> None:
        print("\n[CG Total Cost By Episode]")
        if not self.cg_history:
            print("  No CG episode history recorded.")
            return
        for row in self.cg_history:
            print(
                f"  episode={int(row['episode']):03d} "
                f"| total_cost={float(row['total_cost']):.6f} "
                f"| improvement={float(row['improvement']):.6f} "
                f"| proposed={int(row['proposed_columns'])} "
                f"| added={int(row['added_columns'])} "
                f"| selected_patterns={int(row['selected_patterns'])}"
            )


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def build_predicted_inventory_df(solution) -> pd.DataFrame:
    rows = []
    if hasattr(solution, "inv_store"):
        for (s, p, t), inv in solution.inv_store.items():
            rows.append({"store": s, "sku": p, "period": t, "predicted_end_qty": inv})
    else:
        raise ValueError("Solution object does not contain inv_store")
    return pd.DataFrame(rows)


def build_demand_fulfillment_df(data: IRPData, solution: FullIRPTSolution) -> pd.DataFrame:
    rows = []
    for s in data.stores:
        for t in data.periods:
            total_demand = sum(float(data.demand.get((s, p, t), 0.0)) for p in data.products)
            total_shortage = sum(float(solution.shortage.get((s, p, t), 0.0)) for p in data.products)
            fulfilled_demand = max(0.0, total_demand - total_shortage)
            fulfillment_rate = fulfilled_demand / total_demand if total_demand > 1e-9 else 1.0
            rows.append({
                "store": s,
                "period": t,
                "total_demand": round(total_demand, 6),
                "fulfilled_demand": round(fulfilled_demand, 6),
                "shortage": round(total_shortage, 6),
                "demand_fulfillment_rate": round(fulfillment_rate, 6),
            })
    return pd.DataFrame(rows)


def build_post_shock_fulfillment_df(data: IRPData) -> pd.DataFrame:
    rows = []
    for s in data.stores:
        for t in data.periods:
            total_demand = sum(float(data.realized_demand.get((s, p, t), 0.0)) for p in data.products)
            total_shortage = sum(float(data.post_shock_shortage.get((s, p, t), 0.0)) for p in data.products)
            fulfilled_demand = max(0.0, total_demand - total_shortage)
            fulfillment_rate = fulfilled_demand / total_demand if total_demand > 1e-9 else 1.0
            rows.append({
                "store": s,
                "period": t,
                "total_realized_demand": round(total_demand, 6),
                "fulfilled_demand": round(fulfilled_demand, 6),
                "post_shock_shortage": round(total_shortage, 6),
                "post_shock_fulfillment_rate": round(fulfillment_rate, 6),
            })
    return pd.DataFrame(rows)


def print_demand_fulfillment(fulfillment_df: pd.DataFrame) -> None:
    print("\n[Demand Fulfillment Rate By Store-Period]")
    if fulfillment_df.empty:
        print("  No demand fulfillment rows available.")
        return
    for _, row in fulfillment_df.sort_values(["period", "store"]).iterrows():
        print(
            f"  period={row['period']} | store={row['store']} "
            f"| demand={row['total_demand']:.6f} "
            f"| fulfilled={row['fulfilled_demand']:.6f} "
            f"| shortage={row['shortage']:.6f} "
            f"| fulfillment_rate={row['demand_fulfillment_rate']:.2%}"
        )


def print_post_shock_fulfillment(fulfillment_df: pd.DataFrame) -> None:
    print("\n[Post-Shock Demand Fulfillment Rate By Store-Period]")
    if fulfillment_df.empty:
        print("  No post-shock demand fulfillment rows available.")
        return
    for _, row in fulfillment_df.sort_values(["period", "store"]).iterrows():
        print(
            f"  period={row['period']} | store={row['store']} "
            f"| realized_demand={row['total_realized_demand']:.6f} "
            f"| fulfilled={row['fulfilled_demand']:.6f} "
            f"| post_shock_shortage={row['post_shock_shortage']:.6f} "
            f"| fulfillment_rate={row['post_shock_fulfillment_rate']:.2%}"
        )


def build_full_irpt_cost_breakdown(data: IRPData, solution: FullIRPTSolution) -> Dict[str, float]:
    direct_cw_unit_cost = sum(
        data.ship_cost_cw[(s, p)] * float(solution.direct_ship_q.get((s, p, t), 0.0))
        for s in data.stores
        for p in data.products
        for t in data.periods
    )
    store_holding_cost = sum(
        data.holding_cost_store[(s, p)] * float(solution.inv_store.get((s, p, t), 0.0))
        for s in data.stores
        for p in data.products
        for t in data.periods
    )
    warehouse_holding_cost = sum(
        data.holding_cost_wh[p] * float(solution.inv_wh.get((p, t), 0.0))
        for p in data.products
        for t in data.periods
    )
    route_distance_cost = sum(
        data.alpha * data.distance[(i, j)] * float(solution.x.get((i, j, v, t), 0.0))
        for i in [data.warehouse] + data.stores
        for j in [data.warehouse] + data.stores
        if i != j
        for v in data.vehicles
        for t in data.periods
    )
    vehicle_fixed_cost = sum(
        data.vehicle_fixed_cost * float(solution.u.get((v, t), 0.0))
        for v in data.vehicles
        for t in data.periods
    )
    lateral_transshipment_cost = sum(
        data.transship_unit_cost[(i, j)] * float(solution.y.get((i, j, p, v, t), 0.0))
        for i in data.stores
        for j in data.stores
        if i != j
        for p in data.products
        for v in data.vehicles
        for t in data.periods
    )
    shortage_cost = sum(
        data.shortage_cost[(s, p)] * float(solution.shortage.get((s, p, t), 0.0))
        for s in data.stores
        for p in data.products
        for t in data.periods
    )
    objective_recomputed = (
        direct_cw_unit_cost
        + store_holding_cost
        + warehouse_holding_cost
        + route_distance_cost
        + vehicle_fixed_cost
        + lateral_transshipment_cost
        + shortage_cost
    )
    return {
        "direct_cw_unit_cost": round(direct_cw_unit_cost, 6),
        "store_holding_cost": round(store_holding_cost, 6),
        "warehouse_holding_cost": round(warehouse_holding_cost, 6),
        "route_distance_cost": round(route_distance_cost, 6),
        "vehicle_fixed_cost": round(vehicle_fixed_cost, 6),
        "lateral_transshipment_cost": round(lateral_transshipment_cost, 6),
        "shortage_cost": round(shortage_cost, 6),
        "objective_recomputed": round(objective_recomputed, 6),
        "gurobi_objective": round(float(solution.objective), 6),
    }


def build_realized_operating_cost_breakdown(
    data: IRPData,
    dc_solution: FullIRPTSolution,
    lt_plan_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Evaluate executed DC plan under realized demand and optional LT recourse."""
    direct_cw_unit_cost = sum(
        data.ship_cost_cw[(s, p)] * float(dc_solution.direct_ship_q.get((s, p, t), 0.0))
        for s in data.stores
        for p in data.products
        for t in data.periods
    )
    warehouse_holding_cost = sum(
        data.holding_cost_wh[p] * float(dc_solution.inv_wh.get((p, t), 0.0))
        for p in data.products
        for t in data.periods
    )
    route_distance_cost = sum(
        data.alpha * data.distance[(i, j)] * float(dc_solution.x.get((i, j, v, t), 0.0))
        for i in [data.warehouse] + data.stores
        for j in [data.warehouse] + data.stores
        if i != j
        for v in data.vehicles
        for t in data.periods
    )
    vehicle_fixed_cost = sum(
        data.vehicle_fixed_cost * float(dc_solution.u.get((v, t), 0.0))
        for v in data.vehicles
        for t in data.periods
    )

    lt_net: Dict[Tuple[Store, Product, Period], float] = {
        (s, p, t): 0.0
        for s in data.stores
        for p in data.products
        for t in data.periods
    }
    lateral_transshipment_cost = 0.0
    if lt_plan_df is not None and not lt_plan_df.empty:
        lateral_transshipment_cost = float(pd.to_numeric(lt_plan_df["lt_total_cost"], errors="coerce").fillna(0.0).sum())
        for _, row in lt_plan_df.iterrows():
            p = str(row["sku"])
            t = int(row["period"])
            i = str(row["from_store"])
            j = str(row["to_store"])
            qty = float(row["lt_qty"])
            if (i, p, t) in lt_net:
                lt_net[(i, p, t)] -= qty
            if (j, p, t) in lt_net:
                lt_net[(j, p, t)] += qty

    realized_inventory_after_lt: Dict[Tuple[Store, Product, Period], float] = {}
    realized_shortage_after_lt: Dict[Tuple[Store, Product, Period], float] = {}
    for s in data.stores:
        for p in data.products:
            for t in data.periods:
                post_shock_inv = float(data.post_shock_inventory.get((s, p, t), dc_solution.inv_store.get((s, p, t), 0.0)))
                post_shock_shortage = float(data.post_shock_shortage.get((s, p, t), dc_solution.shortage.get((s, p, t), 0.0)))
                adjusted = post_shock_inv + lt_net.get((s, p, t), 0.0)
                realized_inventory_after_lt[(s, p, t)] = max(0.0, adjusted)
                realized_shortage_after_lt[(s, p, t)] = max(0.0, post_shock_shortage - max(0.0, lt_net.get((s, p, t), 0.0)))
                if adjusted < 0.0:
                    realized_shortage_after_lt[(s, p, t)] += -adjusted

    store_holding_cost = sum(
        data.holding_cost_store[(s, p)] * inv
        for (s, p, _), inv in realized_inventory_after_lt.items()
    )
    shortage_cost = sum(
        data.shortage_cost[(s, p)] * shortage
        for (s, p, _), shortage in realized_shortage_after_lt.items()
    )
    realized_operating_cost = (
        direct_cw_unit_cost
        + store_holding_cost
        + warehouse_holding_cost
        + route_distance_cost
        + vehicle_fixed_cost
        + lateral_transshipment_cost
        + shortage_cost
    )
    return {
        "direct_cw_unit_cost_executed_plan": round(direct_cw_unit_cost, 6),
        "store_holding_cost_realized": round(store_holding_cost, 6),
        "warehouse_holding_cost_executed_plan": round(warehouse_holding_cost, 6),
        "route_distance_cost_executed_plan": round(route_distance_cost, 6),
        "vehicle_fixed_cost_executed_plan": round(vehicle_fixed_cost, 6),
        "lateral_transshipment_cost_realized": round(lateral_transshipment_cost, 6),
        "shortage_cost_realized": round(shortage_cost, 6),
        "total_realized_operating_cost": round(realized_operating_cost, 6),
        "total_realized_shortage_units": round(sum(realized_shortage_after_lt.values()), 6),
        "total_realized_store_inventory_units": round(sum(realized_inventory_after_lt.values()), 6),
    }


def print_cost_breakdown(title: str, cost_breakdown: Dict[str, float]) -> None:
    print(f"\n[{title}]")
    for key, value in cost_breakdown.items():
        print(f"  {key}: {float(value):.6f}")


LT_PLAN_COLUMNS = [
    "source",
    "period",
    "vehicle",
    "from_store",
    "to_store",
    "sku",
    "lt_qty",
    "lt_unit_cost",
    "lt_fixed_cost",
    "lt_total_cost",
    "pattern_id",
    "lambda_value",
]


def build_lt_plan_df_from_solution(
    solution: FullIRPTSolution,
    data: IRPData,
    source: str = "FullIRPT",
) -> pd.DataFrame:
    rows = []
    for (i, j, p, v, t), qty in sorted(solution.y.items(), key=lambda item: (item[0][4], item[0][3], item[0][0], item[0][1], item[0][2])):
        if qty <= 1e-9:
            continue
        unit_cost = float(data.transship_unit_cost.get((i, j), data.ship_cost_lt.get((i, j, p), 0.0)))
        rows.append({
            "source": source,
            "period": t,
            "vehicle": v,
            "from_store": i,
            "to_store": j,
            "sku": p,
            "lt_qty": round(float(qty), 6),
            "lt_unit_cost": round(unit_cost, 6),
            "lt_fixed_cost": 0.0,
            "lt_total_cost": round(unit_cost * float(qty), 6),
            "pattern_id": "",
            "lambda_value": 1.0,
        })
    return pd.DataFrame(rows, columns=LT_PLAN_COLUMNS)


def build_lt_plan_df_from_cg(cg_solution: CGSolution, patterns: List[LTPattern], data: IRPData) -> pd.DataFrame:
    pattern_by_id = {pat.pattern_id: pat for pat in patterns}
    rows = []
    for pat_id in sorted(cg_solution.selected_patterns):
        pat = pattern_by_id.get(pat_id)
        if pat is None:
            continue
        lam = float(cg_solution.lambda_values.get(pat_id, 0.0))
        if lam <= 1e-9:
            continue
        for i, j in sorted(pat.pattern_flows):
            qty = float(pat.pattern_flows[(i, j)]) * lam
            if qty <= 1e-9:
                continue
            unit_cost = float(data.ship_cost_lt.get((i, j, pat.product), data.transship_unit_cost.get((i, j), 0.0)))
            fixed_cost = float(data.fixed_dispatch_lt.get((i, j), 0.0)) * lam
            rows.append({
                "source": "CG_LT",
                "period": pat.period,
                "vehicle": "",
                "from_store": i,
                "to_store": j,
                "sku": pat.product,
                "lt_qty": round(qty, 6),
                "lt_unit_cost": round(unit_cost, 6),
                "lt_fixed_cost": round(fixed_cost, 6),
                "lt_total_cost": round(unit_cost * qty + fixed_cost, 6),
                "pattern_id": pat.pattern_id,
                "lambda_value": round(lam, 6),
            })
    return pd.DataFrame(rows, columns=LT_PLAN_COLUMNS)


def print_lt_plan(lt_plan_df: pd.DataFrame, title: str = "Lateral Transshipment Plan") -> None:
    print(f"\n[{title}]")
    if lt_plan_df.empty:
        print("  No lateral transshipment moves selected.")
        return
    for _, row in lt_plan_df.sort_values(["period", "sku", "from_store", "to_store", "vehicle", "pattern_id"]).iterrows():
        vehicle = row["vehicle"] if str(row["vehicle"]) else "-"
        print(
            f"  source={row['source']} | period={row['period']} | vehicle={vehicle} "
            f"| {row['from_store']} -> {row['to_store']} | sku={row['sku']} "
            f"| qty={row['lt_qty']:.6f} | unit_cost={row['lt_unit_cost']:.6f} "
            f"| fixed_cost={row['lt_fixed_cost']:.6f} | total_cost={row['lt_total_cost']:.6f} "
            f"| pattern={row['pattern_id']} | lambda={row['lambda_value']:.6f}"
        )


def save_cg_cost_curve(cg_history: List[Dict[str, Any]], output_path: str) -> Optional[str]:
    if not cg_history:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = [int(row["episode"]) for row in cg_history]
        total_cost = [float(row["total_cost"]) for row in cg_history]
        plt.figure(figsize=(7, 4))
        plt.plot(episodes, total_cost, marker="o", linewidth=2)
        plt.xlabel("CG episode")
        plt.ylabel("Total cost")
        plt.title("Column Generation Total Cost")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return output_path
    except Exception as exc:
        print(f"[CG] Could not save total-cost chart: {exc}")
        return None


def compute_validation_metrics(comp: pd.DataFrame) -> Dict:
    df = comp.copy()
    df["abs_error"] = (df["predicted_end_qty"] - df["actual_end_qty"]).abs()
    df["sq_error"] = (df["predicted_end_qty"] - df["actual_end_qty"]) ** 2
    df["pct_error"] = df.apply(
        lambda r: abs(r["predicted_end_qty"] - r["actual_end_qty"]) / abs(r["actual_end_qty"])
        if r["actual_end_qty"] not in [0, 0.0] else math.nan,
        axis=1
    )
    mae = float(df["abs_error"].mean()) if len(df) else math.nan
    rmse = float(math.sqrt(df["sq_error"].mean())) if len(df) else math.nan
    bias = float((df["predicted_end_qty"] - df["actual_end_qty"]).mean()) if len(df) else math.nan
    mape = float(df["pct_error"].dropna().mean()) if df["pct_error"].notna().any() else math.nan
    return {"MAE": mae, "RMSE": rmse, "Bias": bias, "MAPE": mape}


# ============================================================================
# ROUTE EXTRACTION HELPERS
# ============================================================================

def extract_routes_from_solution(solution: FullIRPTSolution, warehouse: str = "CW") -> List[Dict[str, Any]]:
    routes: List[Dict[str, Any]] = []
    active_by_vt: Dict[Tuple[Vehicle, Period], List[Tuple[Node, Node]]] = {}
    for (i, j, v, t), val in solution.x.items():
        if val > 0.5:
            active_by_vt.setdefault((v, t), []).append((i, j))

    for (v, t), arcs in sorted(active_by_vt.items(), key=lambda x: (x[0][1], x[0][0])):
        outgoing: Dict[Node, List[Node]] = {}
        incoming: Dict[Node, List[Node]] = {}
        for i, j in arcs:
            outgoing.setdefault(i, []).append(j)
            incoming.setdefault(j, []).append(i)
        degree_warnings = []
        for node in sorted(set(outgoing) | set(incoming)):
            out_deg = len(outgoing.get(node, []))
            in_deg = len(incoming.get(node, []))
            if out_deg > 1 or in_deg > 1:
                degree_warnings.append(f"{node}:in={in_deg},out={out_deg}")
        next_map = {i: js[0] for i, js in outgoing.items() if js}
        if warehouse not in next_map:
            routes.append({
                "period": t,
                "vehicle": v,
                "route": [f"UNRESOLVED_ARCS::{arcs}"],
                "arcs": arcs,
                "total_direct_qty": 0.0,
                "total_lt_qty": 0.0,
                "load_departure": round(float(solution.load.get((warehouse, v, t), 0.0)), 6),
                "load_by_node": {},
                "direct_qty_by_node": {},
                "service_qty_by_node": {},
                "zero_service_nodes": [],
                "degree_warnings": degree_warnings,
                "unvisited_arcs": arcs,
                "product_flow_summary": {},
            })
            continue

        route = [warehouse]
        visited = set()
        cur = warehouse
        while cur in next_map and (cur, next_map[cur]) not in visited:
            nxt = next_map[cur]
            visited.add((cur, nxt))
            route.append(nxt)
            cur = nxt
            if cur == warehouse:
                break
        unvisited_arcs = [arc for arc in arcs if arc not in visited]

        product_flow_summary: Dict[Product, float] = {}
        total_direct_qty = sum(
            float(qty)
            for (s, p, vv, tt), qty in solution.deliv.items()
            if vv == v and tt == t and qty > 1e-9
        )
        total_lt_qty = 0.0
        for i, j in arcs:
            for (p, ii, jj, vv, tt), qty in solution.q.items():
                if ii == i and jj == j and vv == v and tt == t and qty > 1e-9:
                    product_flow_summary[p] = product_flow_summary.get(p, 0.0) + float(qty)
            for (ii, jj, p, vv, tt), qty in solution.y.items():
                if ii == i and jj == j and vv == v and tt == t and qty > 1e-9:
                    total_lt_qty += float(qty)
        load_by_node = {
            node: round(float(solution.load.get((node, v, t), 0.0)), 6)
            for node in route
            if (node, v, t) in solution.load
        }
        direct_qty_by_node = {
            node: round(
                sum(
                    float(qty)
                    for (s, p, vv, tt), qty in solution.deliv.items()
                    if s == node and vv == v and tt == t and qty > 1e-9
                ),
                6,
            )
            for node in route
            if node != warehouse
        }
        service_qty_by_node = {
            node: round(
                direct_qty_by_node.get(node, 0.0)
                + sum(
                    float(qty)
                    for (src, dst, p, vv, tt), qty in solution.y.items()
                    if vv == v and tt == t and qty > 1e-9 and (src == node or dst == node)
                ),
                6,
            )
            for node in route
            if node != warehouse
        }
        zero_service_nodes = [
            node for node, qty in service_qty_by_node.items()
            if qty <= 1e-6
        ]

        routes.append({
            "period": t,
            "vehicle": v,
            "route": route,
            "arcs": arcs,
            "total_direct_qty": round(total_direct_qty, 6),
            "total_lt_qty": round(total_lt_qty, 6),
            "load_departure": round(float(solution.load.get((warehouse, v, t), 0.0)), 6),
            "load_by_node": load_by_node,
            "direct_qty_by_node": direct_qty_by_node,
            "service_qty_by_node": service_qty_by_node,
            "zero_service_nodes": zero_service_nodes,
            "degree_warnings": degree_warnings,
            "unvisited_arcs": unvisited_arcs,
            "product_flow_summary": {k: round(vv, 6) for k, vv in product_flow_summary.items()},
        })
    return routes


def print_routes(routes: List[Dict[str, Any]]) -> None:
    print("\n[Baseline Routing Output]")
    if not routes:
        print("  No active routes found.")
        return
    for row in routes:
        route_str = " -> ".join(row["route"])
        print(
            f"  period={row['period']} | vehicle={row['vehicle']} | route={route_str} "
            f"| direct_qty={row['total_direct_qty']:.2f} | lt_qty={row['total_lt_qty']:.2f} "
            f"| load_departure={row.get('load_departure', 0.0):.2f} "
            f"| product_flow={row['product_flow_summary']}"
        )
        if row.get("degree_warnings") or row.get("unvisited_arcs"):
            print(
                f"    route_warning degree={row.get('degree_warnings', [])} "
                f"unvisited_arcs={row.get('unvisited_arcs', [])}"
            )
        if row.get("zero_service_nodes"):
            print(f"    zero_service_nodes={row['zero_service_nodes']}")



# ============================================================================
# PIPELINE
# ============================================================================

class IRPResearchPipeline:
    def __init__(self, data: IRPData):
        self.data = data

    def run(
        self,
        use_random_initial_patterns: bool = True,
        n_initial_patterns_per_product_period: int = 5,
        cg_iterations: int = 15,
        msg: bool = True,
        time_limit: Optional[int] = None,
        enforce_integer_flows: bool = False,
        cw_dispatch_cycle: Optional[int] = 5,
        use_gnn: bool = False,
        collect_teacher_mode: bool = True,
        runtime_gnn_mode: Optional[bool] = None,
        gnn_checkpoint: str = DEFAULT_GNN_CHECKPOINT,
        use_classical_fallback: bool = True,
        gnn_selection_mode: str = "cumulative_mass",
        gnn_mass_threshold: float = 0.55,
        gnn_relative_threshold: float = 0.85,
        gnn_max_keep: Optional[int] = 150,
        gnn_max_keep_fraction: float = 0.30,
        use_branch_and_price: bool = True,
        bp_max_nodes: int = 15,
        bp_max_depth: int = 6,
        lt_activation_threshold: float = 0.0,
        demand_shock_probability: float = 0.85,
        demand_shock_reallocation_fraction: float = 0.60,
        demand_shock_reallocations_per_product_period: int = 3,
        demand_shock_non_dispatch_multiplier: float = 1.8,
        demand_shock_seed: int = 20260418,
    ) -> Dict:
        pipeline_started_at = time.perf_counter()
        print("=" * 80)
        print("STEP 1 - Solve baseline IRPT")
        print("=" * 80)
        baseline_sol = AchamrahFullIRPTModel(self.data).solve(
            msg=msg,
            time_limit=time_limit,
            enforce_integer_flows=enforce_integer_flows,
            add_valid_16_20=True,
            allow_lateral_transshipment=False,
            cw_dispatch_cycle=cw_dispatch_cycle,
        )
        pprint.pprint(baseline_sol.summary())
        print_efficiency_metrics("Baseline Solver Efficiency", baseline_sol.efficiency_metrics)
        baseline_cost_breakdown = build_full_irpt_cost_breakdown(self.data, baseline_sol)
        print_cost_breakdown("Baseline Full IRPT Cost Breakdown", baseline_cost_breakdown)
        baseline_routes = extract_routes_from_solution(baseline_sol, warehouse=self.data.warehouse)

        print("\n" + "=" * 80)
        print("STEP 1B - Apply hidden realized-demand shock after DC shipment")
        print("=" * 80)
        shock_summary = apply_hidden_local_reallocation_demand_shocks(
            self.data,
            baseline_solution=baseline_sol,
            shock_probability=demand_shock_probability,
            max_reallocation_fraction=demand_shock_reallocation_fraction,
            reallocations_per_product_period=demand_shock_reallocations_per_product_period,
            non_dispatch_shock_multiplier=demand_shock_non_dispatch_multiplier,
            cw_dispatch_cycle=cw_dispatch_cycle,
            seed=demand_shock_seed,
        )
        post_shock_summary = build_post_shock_inventory_state(self.data, baseline_sol)
        post_shock_lt_diagnostics = build_post_shock_lt_diagnostics(
            self.data,
            baseline_sol,
            lt_activation_threshold=lt_activation_threshold,
        )
        demand_fulfillment_df = build_post_shock_fulfillment_df(self.data)
        print("[Hidden Demand Shock Summary]")
        pprint.pprint(shock_summary)
        print("[Post-Shock Inventory State Summary]")
        pprint.pprint(post_shock_summary)
        print("[Post-Shock LT Diagnostics]")
        pprint.pprint(post_shock_lt_diagnostics)
        print_post_shock_fulfillment(demand_fulfillment_df)

        initial_patterns = []
        if use_random_initial_patterns:
            print("\n" + "=" * 80)
            print("STEP 2 - Create demo LT patterns")
            print("=" * 80)
            initial_patterns = generate_random_lt_patterns(
                self.data,
                baseline_solution=baseline_sol,
                n_patterns_per_product_period=n_initial_patterns_per_product_period,
                max_pairs_in_pattern=4,
                lt_activation_threshold=lt_activation_threshold,
                seed=123,
            )
            print(f"Generated {len(initial_patterns)} initial LT patterns")

        print("\n" + "=" * 80)
        print("STEP 3 - Run column generation with RMP + pricing + dual loop")
        print("=" * 80)
        requested_runtime_gnn = bool(use_gnn if runtime_gnn_mode is None else runtime_gnn_mode)
        effective_runtime_gnn = False if collect_teacher_mode else requested_runtime_gnn
        gnn_training_history = load_gnn_training_history(gnn_checkpoint) if effective_runtime_gnn else []
        if effective_runtime_gnn:
            print_gnn_training_history(gnn_training_history, gnn_checkpoint)
        if collect_teacher_mode:
            print("[Teacher] collect_teacher_mode=True: priced batches will bypass GNN filtering before RMP.")
            if requested_runtime_gnn:
                print("[Teacher] runtime_gnn_mode request ignored during teacher collection to avoid self-filtered labels.")

        stackelberg_params = StackelbergParams(
            donor_accept_threshold=0.0,
            receiver_accept_threshold=0.0,
            donor_risk_weight=1.2,
            donor_ship_burden_weight=1.0,
            donor_service_loss_weight=1.0,
            receiver_shortage_reduction_weight=2.0,
            receiver_service_gain_weight=1.0,
            receiver_handling_weight=0.5,
            min_compensation=0.0,
            compensation_cap=50.0,
            acceptance_score_weight=0.6,
            economic_score_weight=0.4,
            top_k_after_game_per_feature=5,
        )

        cg_engine = LateralTransshipmentCG(
            data=self.data,
            baseline_solution=baseline_sol,
            initial_patterns=initial_patterns,
            lt_activation_threshold=lt_activation_threshold,
            max_pairs_per_pattern=4,
            top_pairs_per_feature=20,
            top_patterns_per_feature=5,
            stackelberg_params=stackelberg_params,
            use_gnn=effective_runtime_gnn,
            collect_teacher_mode=collect_teacher_mode,
            runtime_gnn_mode=effective_runtime_gnn,
            gnn_checkpoint=gnn_checkpoint,
            use_classical_fallback=use_classical_fallback,
            gnn_selection_mode=gnn_selection_mode,
            gnn_mass_threshold=gnn_mass_threshold,
            gnn_relative_threshold=gnn_relative_threshold,
            gnn_max_keep=gnn_max_keep,
            gnn_max_keep_fraction=gnn_max_keep_fraction,
        )
        if use_branch_and_price:
            cg_sol = cg_engine.run_branch_and_price(
                max_iter=cg_iterations,
                msg=msg,
                max_nodes=bp_max_nodes,
                max_depth=bp_max_depth,
            )
        else:
            cg_sol = cg_engine.run_column_generation(max_iter=cg_iterations, msg=msg)
        pprint.pprint(cg_sol.summary())
        print_efficiency_metrics("CG RMP Solver Efficiency", cg_sol.efficiency_metrics)
        lt_plan_df = build_lt_plan_df_from_cg(cg_sol, cg_engine.patterns, self.data)
        print_lt_plan(lt_plan_df, title="CG Lateral Transshipment Plan")
        realized_no_lt_cost_breakdown = build_realized_operating_cost_breakdown(
            self.data,
            baseline_sol,
            lt_plan_df=None,
        )
        realized_with_lt_cost_breakdown = build_realized_operating_cost_breakdown(
            self.data,
            baseline_sol,
            lt_plan_df=lt_plan_df,
        )
        print_cost_breakdown("Realized Operating Cost Without LT", realized_no_lt_cost_breakdown)
        print_cost_breakdown("Realized Operating Cost With CG LT", realized_with_lt_cost_breakdown)

        pipeline_runtime_seconds = time.perf_counter() - pipeline_started_at
        comparison = {
            "forecast_dc_plan_objective": baseline_sol.objective,
            "realized_operating_cost_without_lt": realized_no_lt_cost_breakdown["total_realized_operating_cost"],
            "realized_operating_cost_with_cg_lt": realized_with_lt_cost_breakdown["total_realized_operating_cost"],
            "realized_cost_delta_without_minus_with_lt": (
                realized_no_lt_cost_breakdown["total_realized_operating_cost"]
                - realized_with_lt_cost_breakdown["total_realized_operating_cost"]
            ),
            "cg_rmp_surrogate_objective": cg_sol.objective,
            "comparison_note": (
                "DC planning uses forecast demand. Realized operating cost keeps executed DC "
                "shipment/routing/vehicle costs fixed, then evaluates store holding, shortage, "
                "and LT recourse costs after hidden realized-demand shocks."
            ),
            "pipeline_runtime_seconds": pipeline_runtime_seconds,
            "baseline_efficiency_metrics": baseline_sol.efficiency_metrics,
            "cg_rmp_efficiency_metrics": cg_sol.efficiency_metrics,
        }
        print("\n" + "=" * 80)
        print("STEP 4 - Comparison")
        print("=" * 80)
        pprint.pprint(comparison)

        return {
            "baseline_solution": baseline_sol,
            "baseline_routes": baseline_routes,
            "demand_fulfillment": demand_fulfillment_df,
            "baseline_cost_breakdown": baseline_cost_breakdown,
            "realized_no_lt_cost_breakdown": realized_no_lt_cost_breakdown,
            "realized_with_lt_cost_breakdown": realized_with_lt_cost_breakdown,
            "demand_shock_summary": shock_summary,
            "post_shock_summary": post_shock_summary,
            "post_shock_lt_diagnostics": post_shock_lt_diagnostics,
            "cg_solution": cg_sol,
            "lt_plan": lt_plan_df,
            "comparison": comparison,
            "gnn_training_history": gnn_training_history,
            "gnn_selection_history": cg_engine.gnn_selection_history,
            "cg_episode_history": cg_engine.cg_history_all_nodes or cg_engine.cg_history,
            "branch_price_history": cg_engine.branch_history,
            "cg_episode_diagnostics": cg_engine.cg_episode_diagnostics,
            "column_pool_diagnostics": cg_engine.column_pool_diagnostics,
            "teacher_dataset_rows": cg_engine.teacher_dataset_rows,
        }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    EXCEL_PATH = Path(__file__).with_name("1BISCR501V_90100140_20260323-150407111_filtered_sites.csv")

    mapper = DatasetToIRPValidationMapper(
        excel_path=EXCEL_PATH,
        sheet_name="Sheet1",
        store_limit=7,
        sku_limit=3,
        start_date="None",
        end_date="None",
    )

    data, base_df, validation_target, meta = mapper.build_irp_data(
        wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2,
        shortage_cost_rate=0.25,
        holding_cost_rate=0.01,
        cw_ship_cost_flat=1.0,
        lt_ship_cost_flat=0.6,
        fixed_dispatch_cw=8.0,
        fixed_dispatch_lt=2.0,
        vehicle_count=2,
        vehicle_capacity=700.0,
        vehicle_fixed_cost=50.0,
        alpha=1.0,
        cw_replenishment_factor=0.2,
        cw_capacity_factor=2.0,
        store_initial_inventory_multiplier=float(os.environ.get("IRP_STORE_INIT_MULTIPLIER", "0.2")),
    )

    print("Mapped dataset metadata:")
    pprint.pprint(meta)

    validation_target_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_validation_target.csv"
    validation_target.to_csv(validation_target_path, index=False)
    print(f"Saved validation target to: {validation_target_path}")

    collect_teacher_mode = os.environ.get("IRP_COLLECT_TEACHER_MODE", "1").lower() not in {"0", "false", "no"}
    runtime_gnn_mode = os.environ.get("IRP_RUNTIME_GNN_MODE", "0").lower() not in {"0", "false", "no"}
    use_branch_and_price = os.environ.get("IRP_USE_BRANCH_AND_PRICE", "1").lower() not in {"0", "false", "no"}
    build_teacher_graphs = os.environ.get("IRP_BUILD_TEACHER_GRAPHS", "1").lower() not in {"0", "false", "no"}
    train_gnn_after_teacher = os.environ.get("IRP_TRAIN_GNN_AFTER_TEACHER", "1").lower() not in {"0", "false", "no"}
    gnn_train_epochs = int(os.environ.get("IRP_GNN_TRAIN_EPOCHS", "5"))
    lt_activation_threshold = float(os.environ.get("IRP_LT_ACTIVATION_THRESHOLD", "0.0"))
    demand_shock_probability = float(os.environ.get("IRP_DEMAND_SHOCK_PROBABILITY", "0.85"))
    demand_shock_reallocation_fraction = float(os.environ.get("IRP_DEMAND_SHOCK_REALLOCATION_FRACTION", "0.60"))
    demand_shock_reallocations_per_product_period = int(os.environ.get("IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD", "3"))
    demand_shock_non_dispatch_multiplier = float(os.environ.get("IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER", "1.8"))
    demand_shock_seed = int(os.environ.get("IRP_DEMAND_SHOCK_SEED", "20260418"))
    resume_gnn_checkpoint = os.environ.get("IRP_RESUME_GNN_CHECKPOINT", "0").lower() not in {"0", "false", "no"}
    gnn_checkpoint_path = os.environ.get("IRP_GNN_CHECKPOINT", DEFAULT_GNN_CHECKPOINT)

    env_time_limit = os.environ.get("IRP_TIME_LIMIT")

    results = IRPResearchPipeline(data).run(
        use_random_initial_patterns=True,
        n_initial_patterns_per_product_period=5,
        cg_iterations=int(os.environ.get("IRP_CG_ITERATIONS", "15")),
        msg=False,
        time_limit=int(env_time_limit) if env_time_limit else None,
        enforce_integer_flows=False,
        use_gnn=runtime_gnn_mode,
        collect_teacher_mode=collect_teacher_mode,
        runtime_gnn_mode=runtime_gnn_mode,
        gnn_checkpoint=gnn_checkpoint_path,
        use_classical_fallback=False,
        gnn_mass_threshold=0.55,
        gnn_max_keep=150,
        gnn_max_keep_fraction=0.30,
        use_branch_and_price=use_branch_and_price,
        bp_max_nodes=int(os.environ.get("IRP_BP_MAX_NODES", "15")),
        bp_max_depth=int(os.environ.get("IRP_BP_MAX_DEPTH", "6")),
        lt_activation_threshold=lt_activation_threshold,
        demand_shock_probability=demand_shock_probability,
        demand_shock_reallocation_fraction=demand_shock_reallocation_fraction,
        demand_shock_reallocations_per_product_period=demand_shock_reallocations_per_product_period,
        demand_shock_non_dispatch_multiplier=demand_shock_non_dispatch_multiplier,
        demand_shock_seed=demand_shock_seed,
    )

    baseline_routes_df = pd.DataFrame([
        {
            "period": r["period"],
            "vehicle": r["vehicle"],
            "route": " -> ".join(r["route"]),
            "arcs": str(r["arcs"]),
            "total_direct_qty": r["total_direct_qty"],
            "total_lt_qty": r["total_lt_qty"],
            "load_departure": r.get("load_departure", 0.0),
            "load_by_node": str(r.get("load_by_node", {})),
            "direct_qty_by_node": str(r.get("direct_qty_by_node", {})),
            "zero_service_nodes": str(r.get("zero_service_nodes", [])),
            "degree_warnings": str(r.get("degree_warnings", [])),
            "unvisited_arcs": str(r.get("unvisited_arcs", [])),
            "product_flow_summary": str(r["product_flow_summary"]),
        }
        for r in results["baseline_routes"]
    ])
    routes_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_baseline_routes.csv"
    baseline_routes_df.to_csv(routes_path, index=False)
    print(f"Saved baseline routes to: {routes_path}")

    fulfillment_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_demand_fulfillment.csv"
    results["demand_fulfillment"].to_csv(fulfillment_path, index=False)
    print(f"Saved post-shock demand fulfillment rates to: {fulfillment_path}")

    solver_metrics_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_solver_efficiency_metrics.csv"
    pd.DataFrame([
        {"model": "baseline_irpt", **results["baseline_solution"].efficiency_metrics},
        {"model": "cg_rmp_total", **results["cg_solution"].efficiency_metrics},
    ]).to_csv(solver_metrics_path, index=False)
    print(f"Saved solver efficiency metrics to: {solver_metrics_path}")

    cost_breakdown_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_baseline_cost_breakdown.csv"
    pd.DataFrame([results["baseline_cost_breakdown"]]).to_csv(cost_breakdown_path, index=False)
    print(f"Saved baseline cost breakdown to: {cost_breakdown_path}")

    realized_cost_breakdown_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_realized_operating_cost_breakdown.csv"
    pd.DataFrame([
        {"scenario": "without_lt", **results["realized_no_lt_cost_breakdown"]},
        {"scenario": "with_cg_lt", **results["realized_with_lt_cost_breakdown"]},
    ]).to_csv(realized_cost_breakdown_path, index=False)
    print(f"Saved realized operating cost breakdown to: {realized_cost_breakdown_path}")

    demand_shock_summary_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_hidden_demand_shock_summary.csv"
    pd.DataFrame([{
        **results["demand_shock_summary"],
        **results["post_shock_summary"],
        **results["post_shock_lt_diagnostics"],
    }]).to_csv(demand_shock_summary_path, index=False)
    print(f"Saved hidden demand shock summary to: {demand_shock_summary_path}")

    lt_plan_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_lt_plan.csv"
    results["lt_plan"].to_csv(lt_plan_path, index=False)
    print(f"Saved lateral transshipment plan to: {lt_plan_path}")

    gnn_training_history_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_gnn_training_history.csv"
    pd.DataFrame(results["gnn_training_history"]).to_csv(gnn_training_history_path, index=False)
    print(f"Saved GNN training loss history to: {gnn_training_history_path}")

    cg_episode_history_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_gnn_cg_episode_history.csv"
    pd.DataFrame(results["cg_episode_history"]).to_csv(cg_episode_history_path, index=False)
    print(f"Saved CG total-cost episode history to: {cg_episode_history_path}")

    branch_price_history_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_branch_price_history.csv"
    pd.DataFrame(results["branch_price_history"]).to_csv(branch_price_history_path, index=False)
    print(f"Saved branch-and-price history to: {branch_price_history_path}")

    cg_episode_diagnostics_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/cg_episode_diagnostics.csv"
    pd.DataFrame(results["cg_episode_diagnostics"]).to_csv(cg_episode_diagnostics_path, index=False)
    print(f"Saved CG episode diagnostics to: {cg_episode_diagnostics_path}")

    column_pool_diagnostics_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/column_pool_diagnostics.csv"
    pd.DataFrame(results["column_pool_diagnostics"]).to_csv(column_pool_diagnostics_path, index=False)
    print(f"Saved column pool diagnostics to: {column_pool_diagnostics_path}")

    teacher_dataset_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/cg_teacher_dataset.csv"
    teacher_dataset_df = pd.DataFrame(results["teacher_dataset_rows"])
    if teacher_dataset_df.empty:
        print(
            "No CG teacher dataset rows were generated in this run; "
            "skipping teacher CSV overwrite, graph rebuild, and GNN training update."
        )
    else:
        teacher_dataset_df.to_csv(teacher_dataset_path, index=False)
        print(f"Saved CG teacher dataset rows to: {teacher_dataset_path}")
        if collect_teacher_mode:
            run_teacher_graph_and_gnn_training(
                teacher_csv_path=teacher_dataset_path,
                build_graphs=build_teacher_graphs,
                train_gnn=train_gnn_after_teacher,
                train_epochs=gnn_train_epochs,
                resume_checkpoint=resume_gnn_checkpoint,
                checkpoint_path=gnn_checkpoint_path,
            )
            refreshed_history = load_gnn_training_history(gnn_checkpoint_path)
            pd.DataFrame(refreshed_history).to_csv(f"{RESULTS_DIR}/irp_gnn_training_history.csv", index=False)
            print("Refreshed GNN history rows:", len(refreshed_history))

    cg_cost_chart_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_gnn_cg_total_cost_curve.png"
    saved_chart = save_cg_cost_curve(results["cg_episode_history"], cg_cost_chart_path)
    if saved_chart:
        print(f"Saved CG total-cost chart to: {saved_chart}")

    gnn_selection_history_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_gnn_selected_columns.json"
    with open(gnn_selection_history_path, "w", encoding="utf-8") as f:
        json.dump(results["gnn_selection_history"], f, indent=2)
    print(f"Saved GNN selected-column history to: {gnn_selection_history_path}")

    predicted_df = build_predicted_inventory_df(results["baseline_solution"])
    predicted_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_predicted_inventory.csv"
    predicted_df.to_csv(predicted_path, index=False)
    print(f"Saved predicted inventory to: {predicted_path}")

    comparison_df = predicted_df.merge(validation_target, on=["store", "sku", "period"], how="inner")
    comparison_df["error"] = comparison_df["predicted_end_qty"] - comparison_df["actual_end_qty"]
    comparison_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_validation_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved validation comparison to: {comparison_path}")

    metrics = compute_validation_metrics(comparison_df)
    print("\nValidation metrics:")
    pprint.pprint(metrics)

    print("\nDone.")
