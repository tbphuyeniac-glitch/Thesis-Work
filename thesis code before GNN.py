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
import itertools
import math
import random
import pprint
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    raise ImportError("Install gurobipy first: pip install gurobipy") from e

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


def _safe_var_value(model: gp.Model, var: gp.Var) -> float:
    return float(var.X) if var is not None and _has_solution(model) else 0.0


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

    def summary(self) -> Dict:
        return {
            "status": self.status,
            "objective": self.objective,
            "total_direct_shipments": sum(self.direct_ship_q.values()),
            "total_shortage": sum(self.shortage.values()),
            "total_transshipment": sum(self.y.values()),
            "active_route_arcs": sum(self.x.values()),
            "vehicles_used": sum(self.u.values()),
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

    def summary(self) -> Dict:
        return {
            "status": self.status,
            "objective": self.objective,
            "selected_patterns": self.selected_patterns,
            "n_selected_patterns": len(self.selected_patterns),
            "n_active_product_periods": len(self.active_product_periods),
            "iterations_run": self.iterations_run,
        }


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
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.store_limit = store_limit
        self.sku_limit = sku_limit
        self.start_date = start_date
        self.end_date = end_date

    def load_raw(self) -> pd.DataFrame:
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name or 0)
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
        alpha: float = 1.0,
        cw_replenishment_factor: float = 0.6,
        cw_capacity_factor: float = 2.0,
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

        # ONLY first-day END_QTY as initial inventory
        first_period = min(periods)
        first_df = base[base["period"] == first_period].copy()
        for _, row in first_df.iterrows():
            s, p = row["store"], row["sku"]
            data.init_inventory_store[(s, p)] = max(0.0, float(row["end_qty"]))

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
        data.max_vehicles_used = vehicle_count
        data.alpha = alpha

        # replenishment to warehouse by period/product
        demand_by_sku_period = base.groupby(["sku", "period"])["sale_qty"].sum().to_dict()
        for p in products:
            for t in periods:
                data.replenishment_wh[(p, t)] = max(
                    0.0,
                    cw_replenishment_factor * float(demand_by_sku_period.get((p, t), 0.0))
                )

        # aggregate node capacity for stores
        for s in stores:
            data.node_capacity[s] = sum(data.max_inventory_store[(s, p)] for p in products)

        # aggregate capacity for CW
        data.node_capacity[data.warehouse] = cw_capacity_factor * sum(data.init_inventory_wh[p] for p in products)

        # synthetic distances if coordinates are not available
        all_nodes = [data.warehouse] + stores
        for i in all_nodes:
            for j in all_nodes:
                if i == j:
                    data.distance[(i, j)] = 0.0
                elif i == data.warehouse or j == data.warehouse:
                    data.distance[(i, j)] = 10.0
                else:
                    data.distance[(i, j)] = 6.0

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
        mdl = gp.Model("Baseline_IRP")
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
    ) -> FullIRPTSolution:
        d = self.data
        N = d.stores
        P = d.products
        T = d.periods
        V = d.vehicles
        CW = d.warehouse
        N0 = [CW] + N

        mdl = gp.Model("Achamrah_Full_IRPT")
        mdl.Params.OutputFlag = 1 if msg else 0
        if time_limit is not None:
            mdl.Params.TimeLimit = time_limit

        flow_vtype = GRB.INTEGER if enforce_integer_flows else GRB.CONTINUOUS

        I_s_keys = [(s, p, t) for s in N for p in P for t in T]
        I_w_keys = [(p, t) for p in P for t in T]
        Qdir_keys = [(s, p, t) for s in N for p in P for t in T]
        q_keys = [(p, i, j, v, t) for p in P for i in N0 for j in N0 if i != j for v in V for t in T]
        y_keys = [(i, j, p, v, t) for i in N for j in N if i != j for p in P for v in V for t in T]
        B_keys = [(s, p, t) for s in N for p in P for t in T]
        x_keys = [(i, j, v, t) for i in N0 for j in N0 if i != j for v in V for t in T]
        u_keys = [(v, t) for v in V for t in T]
        z_keys = [(i, v, t) for i in N0 for v in V for t in T]

        I_s = mdl.addVars(I_s_keys, lb=0.0, vtype=flow_vtype, name="I_s")
        I_w = mdl.addVars(I_w_keys, lb=0.0, vtype=flow_vtype, name="I_w")
        Qdir = mdl.addVars(Qdir_keys, lb=0.0, vtype=flow_vtype, name="Qdir")
        q = mdl.addVars(q_keys, lb=0.0, vtype=flow_vtype, name="q")
        y = mdl.addVars(y_keys, lb=0.0, vtype=flow_vtype, name="y")
        if not allow_lateral_transshipment:
            for key in y_keys:
                y[key].UB = 0.0
        B = mdl.addVars(B_keys, lb=0.0, vtype=flow_vtype, name="B")
        x = mdl.addVars(x_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")
        u = mdl.addVars(u_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="u")
        z = mdl.addVars(z_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

        mdl.setObjective(
            gp.quicksum(d.holding_cost_store[(s, p)] * I_s[(s, p, t)] for s, p, t in I_s_keys)
            + gp.quicksum(d.holding_cost_wh[p] * I_w[(p, t)] for p, t in I_w_keys)
            + gp.quicksum(d.alpha * d.distance[(i, j)] * x[(i, j, v, t)] for i, j, v, t in x_keys)
            + gp.quicksum(d.transship_unit_cost[(i, j)] * y[(i, j, p, v, t)] for i, j, p, v, t in y_keys)
            + gp.quicksum(d.shortage_cost[(s, p)] * B[(s, p, t)] for s, p, t in B_keys),
            GRB.MINIMIZE,
        )

        first_t = min(T)

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
                    mdl.addConstr(
                        Qdir[(s, p, t)]
                        + gp.quicksum(y[(i, s, p, v, t)] for i in N if i != s for v in V)
                        - gp.quicksum(y[(s, j, p, v, t)] for j in N if j != s for v in V)
                        == gp.quicksum(q[(p, i, s, v, t)] for i in N0 if i != s for v in V)
                        - gp.quicksum(q[(p, s, j, v, t)] for j in N0 if j != s for v in V)
                    )

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
                        mdl.addConstr(gp.quicksum(q[(p, i, j, v, t)] for p in P) <= d.vehicle_capacity * u[(v, t)])

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

        for p in P:
            for i in N0:
                for j in N0:
                    if i == j:
                        continue
                    for v in V:
                        for t in T:
                            mdl.addConstr(q[(p, i, j, v, t)] <= d.vehicle_capacity * x[(i, j, v, t)])

        for s in N:
            for p in P:
                for t in T:
                    mdl.addConstr(Qdir[(s, p, t)] == gp.quicksum(q[(p, CW, s, v, t)] for v in V))

        for i in N:
            for j in N:
                if i == j:
                    continue
                for p in P:
                    for v in V:
                        for t in T:
                            mdl.addConstr(y[(i, j, p, v, t)] <= q[(p, i, j, v, t)])

        if add_valid_16_20:
            for i in N0:
                for v in V:
                    for t in T:
                        if i == CW:
                            mdl.addConstr(z[(i, v, t)] == u[(v, t)])
                        else:
                            mdl.addConstr(z[(i, v, t)] == gp.quicksum(x[(j, i, v, t)] for j in N0 if j != i))

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
        )


# ============================================================================
# COLUMN GENERATION FOR LATERAL TRANSSHIPMENT
# ============================================================================

def generate_random_lt_patterns(
    data: IRPData,
    baseline_solution,
    n_patterns_per_product_period: int = 2,
    max_pairs_in_pattern: int = 2,
    max_qty_per_pair: int = 10,
    lt_activation_threshold: float = 10.0,
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


class LateralTransshipmentCG:
    def __init__(
        self,
        data: IRPData,
        baseline_solution,
        initial_patterns: Optional[List[LTPattern]] = None,
        lt_activation_threshold: float = 10.0,
        safety_stock_units: float = 0.0,
        max_pairs_per_pattern: int = 3,
        max_columns_per_product_period: int = 3,
        top_pairs_per_feature: int = 8,
        top_patterns_per_feature: int = 2,
        feature_ranges: Optional[Dict[str, Dict[str, float]]] = None,
        stackelberg_params: Optional[StackelbergParams] = None,
    ):
        self.data = data
        self.baseline = baseline_solution
        self.patterns = initial_patterns[:] if initial_patterns else []
        self.lt_activation_threshold = float(lt_activation_threshold)
        self.safety_stock_units = float(safety_stock_units)
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

    def add_patterns(self, new_patterns: Iterable[LTPattern]) -> int:
        existing_ids = {p.pattern_id for p in self.patterns}
        existing_signatures = {
            (p.product, p.period, tuple(sorted((i, j, round(q, 6)) for (i, j), q in p.pattern_flows.items())))
            for p in self.patterns
        }
        added = 0
        for pat in new_patterns:
            signature = (
                pat.product,
                pat.period,
                tuple(sorted((i, j, round(q, 6)) for (i, j), q in pat.pattern_flows.items())),
            )
            if pat.pattern_id in existing_ids or signature in existing_signatures or not pat.pattern_flows:
                continue
            self.patterns.append(pat)
            existing_ids.add(pat.pattern_id)
            existing_signatures.add(signature)
            added += 1
        return added

    def _build_need_and_surplus_proxies(self):
        d = self.data
        need, surplus = {}, {}
        for s, p, t in itertools.product(d.stores, d.products, d.periods):
            need[(s, p, t)] = max(0.0, float(self.baseline.shortage[(s, p, t)]))
            surplus[(s, p, t)] = max(0.0, float(self.baseline.inv_store[(s, p, t)]) - self.safety_stock_units)
        return need, surplus

    def _compute_active_product_periods(self, need, surplus) -> Set[Tuple[Product, Period]]:
        active = set()
        for p in self.data.products:
            for t in self.data.periods:
                total_need = sum(need[(s, p, t)] for s in self.data.stores)
                total_surplus = sum(surplus[(s, p, t)] for s in self.data.stores)
                if total_need >= self.lt_activation_threshold and total_surplus >= self.lt_activation_threshold:
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
        demand_now = max(0.0, float(self.data.demand.get((store, product, period), 0.0)))
        if receiver_need > 1e-9:
            if demand_now <= 1e-9:
                return 0.0
            serviceable_qty_before_stockout = max(0.0, demand_now - receiver_need)
            return serviceable_qty_before_stockout / demand_now

        ending_inventory = max(0.0, float(self.baseline.inv_store.get((store, product, period), 0.0)))
        future_demands = [
            max(0.0, float(self.data.demand.get((store, product, tau), 0.0)))
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
    ) -> Dict[str, List[Dict[str, Any]]]:
        feature_values = self._feature_value_map(p=p, t=t, need=need, surplus=surplus)
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
    ) -> Dict[str, List[Dict[str, Any]]]:
        params = self.stackelberg_params
        accepted_by_feature: Dict[str, List[Dict[str, Any]]] = {
            feature_name: [] for feature_name in pruned_pairs_by_feature.keys()
        }

        for feature_name, rows in pruned_pairs_by_feature.items():
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

                if not decision.accepted:
                    continue

                combined_score = (
                    params.acceptance_score_weight * decision.acceptance_score
                    + params.economic_score_weight * row["base_rank_score"]
                )
                row2["post_game_score"] = combined_score
                accepted_by_feature[feature_name].append(row2)

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
                    new_patterns.append(
                        LTPattern(
                            pattern_id=f"PRICED_{feature_name}_{p}_T{t}_{built_here}",
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
                                "mean_acceptance_score": round(
                                    sum(r.get("acceptance_score", 0.0) for r in rows) / max(len(rows), 1), 6
                                ),
                                "mean_compensation": round(
                                    sum(r.get("compensation", 0.0) for r in rows) / max(len(rows), 1), 6
                                ),
                            },
                        )
                    )
        return new_patterns

    def _candidate_patterns_from_duals(
        self,
        need,
        surplus,
        active_product_periods: Set[Tuple[Product, Period]],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
        rc_tol: float = -1e-6,
    ) -> List[LTPattern]:
        new_patterns: List[LTPattern] = []

        for p, t in sorted(active_product_periods):
            pruned_pairs_by_feature = self._prune_pairs_by_feature(
                p=p,
                t=t,
                need=need,
                surplus=surplus,
                dual_need=dual_need,
                dual_surplus=dual_surplus,
            )
            accepted_pairs_by_feature = self._apply_stackelberg_game_to_pairs(
                p=p,
                t=t,
                pruned_pairs_by_feature=pruned_pairs_by_feature,
                need=need,
                surplus=surplus,
            )
            feature_patterns = self._build_patterns_from_pruned_pairs(
                p=p,
                t=t,
                pruned_pairs_by_feature=accepted_pairs_by_feature,
                need=need,
                surplus=surplus,
                rc_tol=rc_tol,
            )
            new_patterns.extend(feature_patterns)
        return new_patterns

    def solve_rmp(self, msg: bool = False, return_model: bool = False):
        d = self.data
        need, surplus = self._build_need_and_surplus_proxies()
        active_product_periods = self._compute_active_product_periods(need, surplus)
        mdl = gp.Model("LT_RMP")
        mdl.Params.OutputFlag = 1 if msg else 0

        pattern_map = {pat.pattern_id: pat for pat in self.patterns if (pat.product, pat.period) in active_product_periods}
        lam = mdl.addVars(list(pattern_map.keys()), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="lambda")
        residual_need_keys = [(s, p, t) for s in d.stores for p in d.products for t in d.periods]
        residual_need = mdl.addVars(residual_need_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="residual_need")

        baseline_shortage_component = sum(
            d.shortage_cost[(s, p)] * float(self.baseline.shortage[(s, p, t)])
            for s, p, t in itertools.product(d.stores, d.products, d.periods)
        )
        baseline_without_shortage = float(self.baseline.objective) - baseline_shortage_component

        mdl.setObjective(
            baseline_without_shortage
            + gp.quicksum(pat.column_cost * lam[pat.pattern_id] for pat in pattern_map.values())
            + gp.quicksum(d.shortage_cost[(s, p)] * residual_need[(s, p, t)] for s, p, t in residual_need_keys),
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
        )
        if return_model:
            return sol, mdl
        return sol

    def pricing_step(self, master_solution: CGSolution, rc_tol: float = -1e-6) -> List[LTPattern]:
        need, surplus = self._build_need_and_surplus_proxies()
        new_patterns = self._candidate_patterns_from_duals(
            need=need,
            surplus=surplus,
            active_product_periods=master_solution.active_product_periods,
            dual_need=master_solution.dual_need,
            dual_surplus=master_solution.dual_surplus,
            rc_tol=rc_tol,
        )
        return new_patterns

    def run_column_generation(
        self,
        max_iter: int = 10,
        improvement_tol: float = 1e-5,
        rc_tol: float = -1e-6,
        msg: bool = False,
    ) -> CGSolution:
        best_sol = self.solve_rmp(msg=msg)
        best_sol.iterations_run = 0
        prev_obj = best_sol.objective

        print("\n[CG] Active (product, period) pairs after 10-unit LT trigger:")
        if not best_sol.active_product_periods:
            print("  None. RMP is not activated for any product-period.")
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
            new_patterns = self.pricing_step(best_sol, rc_tol=rc_tol)
            added = self.add_patterns(new_patterns)

            print(f"\n[Pricing] Iter {it}: proposed={len(new_patterns)}, added={added}")
            for pat in new_patterns[:10]:
                print(
                    "  " + format_pattern_detail(pat)
                    + f" | feature={pat.metadata.get('feature_name')}"
                    + f" | rc={pat.metadata.get('reduced_cost')}"
                    + f" | mean_acceptance={pat.metadata.get('mean_acceptance_score')}"
                    + f" | mean_comp={pat.metadata.get('mean_compensation')}"
                )

            if added == 0:
                best_sol.iterations_run = it - 1
                print("[CG] No negative reduced-cost columns found. Stop.")
                return best_sol

            sol = self.solve_rmp(msg=msg)
            sol.iterations_run = it
            improvement = prev_obj - sol.objective
            print(f"[CG] Iter {it}: objective = {sol.objective:.6f}, improvement = {improvement:.6f}")

            print("[RMP] Selected LT patterns after re-optimization:")
            if not sol.selected_patterns:
                print("  None")
            else:
                for pat_id in sol.selected_patterns:
                    pat = next(p for p in self.patterns if p.pattern_id == pat_id)
                    print("  " + format_pattern_detail(pat) + f" | lambda={sol.lambda_values[pat_id]:.4f}")

            if improvement <= improvement_tol:
                return sol
            prev_obj = sol.objective
            best_sol = sol

        return best_sol


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
        next_map = {i: j for i, j in arcs}
        if warehouse not in next_map:
            routes.append({
                "period": t,
                "vehicle": v,
                "route": [f"UNRESOLVED_ARCS::{arcs}"],
                "arcs": arcs,
                "total_direct_qty": 0.0,
                "total_lt_qty": 0.0,
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

        product_flow_summary: Dict[Product, float] = {}
        total_direct_qty = 0.0
        total_lt_qty = 0.0
        for i, j in arcs:
            for (p, ii, jj, vv, tt), qty in solution.q.items():
                if ii == i and jj == j and vv == v and tt == t and qty > 1e-9:
                    product_flow_summary[p] = product_flow_summary.get(p, 0.0) + float(qty)
                    if i == warehouse and j != warehouse:
                        total_direct_qty += float(qty)
            for (ii, jj, p, vv, tt), qty in solution.y.items():
                if ii == i and jj == j and vv == v and tt == t and qty > 1e-9:
                    total_lt_qty += float(qty)

        routes.append({
            "period": t,
            "vehicle": v,
            "route": route,
            "arcs": arcs,
            "total_direct_qty": round(total_direct_qty, 6),
            "total_lt_qty": round(total_lt_qty, 6),
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
            f"| product_flow={row['product_flow_summary']}"
        )



# ============================================================================
# PIPELINE
# ============================================================================

class IRPResearchPipeline:
    def __init__(self, data: IRPData):
        self.data = data

    def run(
        self,
        use_random_initial_patterns: bool = True,
        n_initial_patterns_per_product_period: int = 2,
        cg_iterations: int = 3,
        msg: bool = True,
        time_limit: int = 300,
        enforce_integer_flows: bool = False,
    ) -> Dict:
        print("=" * 80)
        print("STEP 1 - Solve Achamrah-style full IRPT")
        print("=" * 80)
        baseline_sol = AchamrahFullIRPTModel(self.data).solve(
            msg=msg,
            time_limit=time_limit,
            enforce_integer_flows=enforce_integer_flows,
            add_valid_16_20=True,
            allow_lateral_transshipment=False,
        )
        pprint.pprint(baseline_sol.summary())
        baseline_routes = extract_routes_from_solution(baseline_sol, warehouse=self.data.warehouse)
        print_routes(baseline_routes)

        initial_patterns = []
        if use_random_initial_patterns:
            print("\n" + "=" * 80)
            print("STEP 2 - Create demo LT patterns")
            print("=" * 80)
            initial_patterns = generate_random_lt_patterns(
                self.data,
                baseline_solution=baseline_sol,
                n_patterns_per_product_period=n_initial_patterns_per_product_period,
                lt_activation_threshold=10.0,
                seed=123,
            )
            print(f"Generated {len(initial_patterns)} initial LT patterns")

        print("\n" + "=" * 80)
        print("STEP 3 - Run column generation with RMP + pricing + dual loop")
        print("=" * 80)
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
            stackelberg_params=stackelberg_params,
        )
        cg_sol = cg_engine.run_column_generation(max_iter=cg_iterations, msg=msg)
        pprint.pprint(cg_sol.summary())
        lt_plan_df = build_lt_plan_df_from_cg(cg_sol, cg_engine.patterns, self.data)
        print_lt_plan(lt_plan_df, title="CG Lateral Transshipment Plan")

        comparison = {
            "baseline_objective": baseline_sol.objective,
            "cg_objective": cg_sol.objective,
            "estimated_improvement": baseline_sol.objective - cg_sol.objective,
        }
        print("\n" + "=" * 80)
        print("STEP 4 - Comparison")
        print("=" * 80)
        pprint.pprint(comparison)

        return {
            "baseline_solution": baseline_sol,
            "baseline_routes": baseline_routes,
            "cg_solution": cg_sol,
            "lt_plan": lt_plan_df,
            "comparison": comparison,
        }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    EXCEL_PATH = "/Users/trannguyenhung/Downloads/1BISCR501V_90100140_20260313-185009126.xlsx"

    mapper = DatasetToIRPValidationMapper(
        excel_path=EXCEL_PATH,
        sheet_name="Sheet1",
        store_limit=3,
        sku_limit=2,
        start_date=None,
        end_date=None,
    )

    data, base_df, validation_target, meta = mapper.build_irp_data(
        wh_inventory_multiplier=2.5,
        store_capacity_multiplier=1.5,
        shortage_cost_rate=0.25,
        holding_cost_rate=0.01,
        cw_ship_cost_flat=1.0,
        lt_ship_cost_flat=0.6,
        fixed_dispatch_cw=8.0,
        fixed_dispatch_lt=2.0,
        vehicle_count=1,
        vehicle_capacity=120.0,
        alpha=1.0,
        cw_replenishment_factor=0.6,
        cw_capacity_factor=2.0,
    )

    print("Mapped dataset metadata:")
    pprint.pprint(meta)

    validation_target_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_validation_target.csv"
    validation_target.to_csv(validation_target_path, index=False)
    print(f"Saved validation target to: {validation_target_path}")

    results = IRPResearchPipeline(data).run(
        use_random_initial_patterns=True,
        n_initial_patterns_per_product_period=2,
        cg_iterations=3,
        msg=False,
        time_limit=60,
        enforce_integer_flows=False,
    )

    baseline_routes_df = pd.DataFrame([
        {
            "period": r["period"],
            "vehicle": r["vehicle"],
            "route": " -> ".join(r["route"]),
            "arcs": str(r["arcs"]),
            "total_direct_qty": r["total_direct_qty"],
            "total_lt_qty": r["total_lt_qty"],
            "product_flow_summary": str(r["product_flow_summary"]),
        }
        for r in results["baseline_routes"]
    ])
    routes_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_baseline_routes.csv"
    baseline_routes_df.to_csv(routes_path, index=False)
    print(f"Saved baseline routes to: {routes_path}")

    lt_plan_path = "/Users/trannguyenhung/Documents/THESIS/Code/Current Code/Results/irp_lt_plan.csv"
    results["lt_plan"].to_csv(lt_plan_path, index=False)
    print(f"Saved lateral transshipment plan to: {lt_plan_path}")

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


