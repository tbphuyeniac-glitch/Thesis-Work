"""
tsrfp_runner.py — Man-Joint-TC Oracle runner for the thesis benchmark framework.

Wraps solve_joint_tc (solve_TSRFP_TC.py) in a JointTCRunner class that matches
the ManBFPTCRunner interface so it can be plugged into the same scenario loop.

This is an ORACLE benchmark: both Stage 1 and Stage 2 are optimized jointly,
so Stage 1 routing can indirectly anticipate the realized demand shock.
It is labelled "Man-Joint-TC" to distinguish it from the true TSRFP-TC
(which would require a C&CG robust loop where Stage 1 cannot see dim_real).

Interface matches ManBFPTCRunner:
    runner = JointTCRunner(...)
    result, lt_moves = runner.run_scenario(scenario, df_slice, dist_dict,
                                           demand_shock=shock)
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from solve_TSRFP_TC import solve_joint_tc, solve_tsrfp_tc


@dataclass
class JointTCResult:
    """Result dataclass for Man-Joint-TC, field-aligned with ManBFPTCResult."""

    # Core fields matching ManBFPTCResult
    scenario_id:        str   = ""
    method:             str   = "Man-Joint-TC"
    status:             str   = ""
    success:            bool  = False
    runtime_seconds:    float = float("nan")
    total_cost:         float = float("nan")
    holding_cost:       float = float("nan")
    routing_cost:       float = float("nan")
    transshipment_cost: float = float("nan")
    shortage_cost:      float = float("nan")
    shortage_qty:       float = float("nan")
    service_level:      float = float("nan")
    lt_total_qty:       float = float("nan")
    n_lt_moves:         int   = 0
    n_routes:           int   = 0
    mip_gap:            float = float("nan")
    error_message:      str   = ""
    error_traceback:    str   = ""

    # BFP-TC-aligned extras (same field names as ManBFPTCResult)
    instance_id:    str   = ""
    num_stores:     int   = 0
    num_products:   int   = 0
    num_vehicles:   int   = 0
    num_periods:    int   = 0
    stage1_cost:    float = float("nan")
    lt_cost:        float = float("nan")
    final_stockout: float = float("nan")
    stockout_rate:  float = float("nan")

    # Joint-TC / TSRFP-specific extras
    ccg_iterations:   int   = 0
    final_robust_gap: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JointTCRunner:
    """Run Man-Joint-TC Oracle benchmark via rolling-horizon single-period solves.

    Mirrors ManBFPTCRunner interface:
        run_scenario(scenario, df_slice, dist_dict, demand_shock=None)
        -> (JointTCResult, lt_moves_list)

    All cost parameters, bk, LT cost, and demand handling are identical to
    ManBFPTCRunner so results are directly comparable.
    """

    METHOD_NAME = "Man-Joint-TC"

    def __init__(
        self,
        time_limit: int = 1200,
        mip_gap: float = 0.01,
        threads: int = 4,
        vehicle_count: int = 2,
        vehicle_capacity: float = 1500.0,
        holding_cost_rate: float = 0.1,
        shortage_cost_rate: float = 2.0,
        lt_cost_multiplier: float = 1.2,
        product_value_default: float = 1.0,
        product_volume_default: float = 1.0,
        store_capacity_floor: float = 10000.0,
        store_capacity_init_multiplier: float = 5.0,
        store_capacity_buffer: float = 5000.0,
        dc_capacity: float = 1e12,
        dc_initial_stock_per_product: float = 10000.0,
        distance_fallback: float = 100.0,
    ):
        self.time_limit         = int(time_limit)
        self.mip_gap            = float(mip_gap)
        self.threads            = int(threads)
        self.vehicle_count      = int(vehicle_count)
        self.vehicle_capacity   = float(vehicle_capacity)
        self.holding_cost_rate  = float(holding_cost_rate)
        self.shortage_cost_rate = float(shortage_cost_rate)
        self.lt_cost_multiplier = float(lt_cost_multiplier)
        self.product_value_default   = float(product_value_default)
        self.product_volume_default  = float(product_volume_default)
        self.store_capacity_floor          = float(store_capacity_floor)
        self.store_capacity_init_multiplier = float(store_capacity_init_multiplier)
        self.store_capacity_buffer         = float(store_capacity_buffer)
        self.dc_capacity                   = float(dc_capacity)
        self.dc_initial_stock_per_product  = float(dc_initial_stock_per_product)
        self.distance_fallback             = float(distance_fallback)
        self.source_name = self.METHOD_NAME

    # ------------------------------------------------------------------
    # Build static inputs — identical to ManBFPTCRunner._build_static_inputs
    # ------------------------------------------------------------------

    def _build_static_inputs(
        self,
        stores: List[str],
        skus: List[str],
        dist_dict: Dict[Tuple[str, str], float],
        df_slice: pd.DataFrame,
    ) -> Dict[str, Any]:
        N = len(stores)
        M = len(skus)

        wh_name = "__WAREHOUSE__"
        node_names = [wh_name] + stores
        cij = [[0.0] * (N + 1) for _ in range(N + 1)]
        for i, ni in enumerate(node_names):
            for j, nj in enumerate(node_names):
                if i == j:
                    continue
                d = dist_dict.get((ni, nj)) or dist_dict.get((nj, ni)) or self.distance_fallback
                cij[i][j] = float(d)

        # bk = 0 to match Man-BFP-TC harmonized config (same as ManBFPTCRunner)
        Qk = [self.vehicle_capacity for _ in range(self.vehicle_count)]
        bk = [0.0 for _ in range(self.vehicle_count)]

        # him[0][m] = DC; him[1..N][m] = stores  (same as ManBFPTCRunner)
        him = [[0.0] * M for _ in range(N + 1)]
        for i in range(1, N + 1):
            for m in range(M):
                him[i][m] = self.holding_cost_rate

        # vim[0..N][m] = 1.0 (volume factor, same index convention as BFP-TC)
        vim = [[self.product_volume_default] * M for _ in range(N + 1)]

        pim = [[self.product_value_default] * M for _ in range(N + 1)]

        store_to_idx = {s: i for i, s in enumerate(stores)}
        sku_to_idx   = {s: i for i, s in enumerate(skus)}

        # Initial store inventory from first-period end_qty — identical to ManBFPTCRunner
        import numpy as np
        store_init_inv = np.zeros((N, M), dtype=float)
        for _, row in df_slice.iterrows():
            s = str(row["store"]); p = str(row["sku"])
            if s not in store_to_idx or p not in sku_to_idx:
                continue
            si = store_to_idx[s]; pi = sku_to_idx[p]
            if store_init_inv[si, pi] == 0.0:
                store_init_inv[si, pi] = float(row["end_qty"])

        # Node capacities — match ManBFPTCRunner formula
        Ui = [self.dc_capacity]
        for si in range(N):
            init_total = float(store_init_inv[si].sum())
            cap = max(
                self.store_capacity_floor,
                init_total * self.store_capacity_init_multiplier + self.store_capacity_buffer,
            )
            Ui.append(cap)

        Iim0_dc     = [self.dc_initial_stock_per_product] * M
        Iim0_stores = store_init_inv.tolist()

        return {
            "N": N, "M": M,
            "cij": cij, "Qk": Qk, "bk": bk,
            "him": him, "vim": vim, "pim": pim,
            "Ui": Ui,
            "Iim0_dc": Iim0_dc,
            "Iim0_stores": Iim0_stores,
            "store_to_idx": store_to_idx,
            "sku_to_idx": sku_to_idx,
        }

    def _period_demand(
        self,
        df_slice: pd.DataFrame,
        period_date,
        store_to_idx: Dict[str, int],
        sku_to_idx: Dict[str, int],
        N: int,
        M: int,
    ) -> List[List[float]]:
        rows = df_slice[df_slice["period_date"] == period_date]
        dim = [[0.0] * M for _ in range(N)]
        for _, row in rows.iterrows():
            s, p, q = str(row["store"]), str(row["sku"]), float(row["sale_qty"])
            if s in store_to_idx and p in sku_to_idx:
                dim[store_to_idx[s]][sku_to_idx[p]] += q
        return dim

    # ------------------------------------------------------------------
    # run_scenario — mirrors ManBFPTCRunner.run_scenario exactly
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        scenario,
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
        demand_shock: Optional[Dict[Tuple[str, str, int], float]] = None,
    ) -> Tuple["JointTCResult", List[Dict[str, Any]]]:
        t0 = time.time()
        scenario_id = getattr(scenario, "scenario_id", "")
        lt_moves_out: List[Dict[str, Any]] = []

        try:
            stores = list(getattr(scenario, "selected_stores", []))
            skus   = list(getattr(scenario, "selected_skus",   []))
            if not stores or not skus:
                raise ValueError("Scenario has empty selected_stores or selected_skus")

            static = self._build_static_inputs(stores, skus, dist_dict, df_slice)
            N, M = static["N"], static["M"]
            store_to_idx = static["store_to_idx"]
            sku_to_idx   = static["sku_to_idx"]
            id_to_store  = {i: s for s, i in store_to_idx.items()}
            id_to_sku    = {i: s for s, i in sku_to_idx.items()}

            periods_dt  = sorted(df_slice["period_date"].unique())
            num_periods = len(periods_dt)

            Iim0_dc     = list(static["Iim0_dc"])
            Iim0_stores = [list(r) for r in static["Iim0_stores"]]

            per_period_tl = max(30, self.time_limit // max(num_periods, 1))

            # Accumulators (same fields as ManBFPTCRunner)
            agg_routing = agg_veh_fixed = 0.0
            agg_s1_holding = agg_s1_shortage = agg_s1_shortage_units = 0.0
            agg_s2_lt = agg_s2_holding = agg_s2_shortage = agg_s2_shortage_units = 0.0
            agg_total_demand = agg_lt_qty = 0.0
            max_mip_gap = 0.0
            any_infeasible = False
            num_lt_moves = 0

            for t_idx, period_date in enumerate(periods_dt, start=1):
                dim_actual   = self._period_demand(df_slice, period_date,
                                                   store_to_idx, sku_to_idx, N, M)
                dim_forecast = [list(r) for r in dim_actual]

                if demand_shock:
                    dim_real = [
                        [dim_actual[i][m] * demand_shock.get(
                             (id_to_store[i], id_to_sku[m], t_idx), 1.0)
                         for m in range(M)]
                        for i in range(N)
                    ]
                else:
                    dim_real = [list(r) for r in dim_actual]

                Iim0 = [list(Iim0_dc)] + [list(r) for r in Iim0_stores]

                res = solve_joint_tc(
                    retailer_number=N,
                    product_kind=M,
                    vehicle_number=self.vehicle_count,
                    Ui=static["Ui"],
                    cij=static["cij"],
                    pim=static["pim"],
                    Qk=static["Qk"],
                    bk=static["bk"],
                    vim=static["vim"],
                    him=static["him"],
                    Iim0=Iim0,
                    dim_real=dim_real,
                    dim=dim_forecast,
                    shortage_penalty_stage1=self.shortage_cost_rate,
                    shortage_penalty_stage2=self.shortage_cost_rate,
                    lt_cost_multiplier=self.lt_cost_multiplier,
                    time_limit=per_period_tl,
                    mip_gap=self.mip_gap,
                    threads=self.threads,
                    verbose=False,
                )

                if res.get("infeasible"):
                    any_infeasible = True
                    break

                agg_routing         += float(res.get("stage1_routing_cost",    0.0))
                agg_veh_fixed       += float(res.get("stage1_vehicle_fixed_cost", 0.0))
                agg_s1_holding      += float(res.get("stage1_holding_cost",    0.0))
                agg_s1_shortage     += float(res.get("stage1_forecast_shortage", 0.0))
                agg_s1_shortage_units += float(
                    res.get("stage1_forecast_shortage_units", 0.0))
                agg_s2_lt           += float(res.get("stage2_lt_transport_cost", 0.0))
                agg_s2_holding      += float(res.get("stage2_holding_cost",    0.0))
                agg_s2_shortage     += float(res.get("stage2_remaining_shortage", 0.0))
                agg_s2_shortage_units += float(
                    res.get("stage2_remaining_shortage_units", 0.0))
                max_mip_gap = max(max_mip_gap, float(res.get("mip_gap", 0.0)))

                agg_total_demand += sum(dim_actual[i][m]
                                        for i in range(N) for m in range(M))

                for mv in res.get("lt_moves", []):
                    qty = float(mv.get("qty", 0.0))
                    if qty <= 1e-9:
                        continue
                    fi, ti, pi = int(mv["from_store_idx"]), int(mv["to_store_idx"]), int(mv["product_idx"])
                    lt_moves_out.append({
                        "scenario_id": scenario_id,
                        "method":      self.METHOD_NAME,
                        "period":      t_idx,
                        "from_store":  id_to_store.get(fi, str(fi)),
                        "to_store":    id_to_store.get(ti, str(ti)),
                        "sku":         id_to_sku.get(pi,  str(pi)),
                        "lt_qty":      qty,
                        "lt_unit_cost": float(mv.get("unit_cost", 0.0)),
                        "lt_total_cost": qty * float(mv.get("unit_cost", 0.0)),
                        "vehicle":     "N/A",
                    })
                    agg_lt_qty += qty
                    num_lt_moves += 1

                Iim0_dc     = list(res["post_lt_dc_inventory"])
                Iim0_stores = [list(r) for r in res["post_lt_inventory"]]

            runtime = time.time() - t0

            if any_infeasible:
                return JointTCResult(
                    scenario_id=scenario_id, method=self.METHOD_NAME,
                    status="failed", success=False, runtime_seconds=runtime,
                    instance_id=scenario_id,
                    num_stores=N, num_products=M,
                    num_vehicles=self.vehicle_count, num_periods=num_periods,
                    error_message="At least one period of Man-Joint-TC was infeasible",
                ), lt_moves_out

            routing_cost       = agg_routing + agg_veh_fixed
            transshipment_cost = agg_s2_lt
            holding_cost       = agg_s1_holding + agg_s2_holding
            shortage_cost      = agg_s1_shortage + agg_s2_shortage
            stage1_total       = agg_routing + agg_veh_fixed + agg_s1_holding + agg_s1_shortage
            lt_total           = agg_s2_lt + agg_s2_holding + agg_s2_shortage
            total_cost         = stage1_total + lt_total

            shortage_qty    = agg_s1_shortage_units + agg_s2_shortage_units
            final_stockout  = agg_s2_shortage_units
            stockout_rate   = final_stockout / agg_total_demand if agg_total_demand > 0 else 0.0
            service_level   = max(0.0, 1.0 - shortage_qty / max(agg_total_demand, 1.0))

            print(
                f"  [Man-Joint-TC]  routing={routing_cost:.2f}  "
                f"holding={holding_cost:.2f}  shortage={shortage_cost:.2f}  "
                f"lt={transshipment_cost:.2f}  total={total_cost:.2f}"
            )

            return JointTCResult(
                scenario_id=scenario_id,
                method=self.METHOD_NAME,
                status="success",
                success=True,
                runtime_seconds=runtime,
                total_cost=total_cost,
                holding_cost=holding_cost,
                routing_cost=routing_cost,
                transshipment_cost=transshipment_cost,
                shortage_cost=shortage_cost,
                shortage_qty=shortage_qty,
                service_level=service_level,
                lt_total_qty=agg_lt_qty,
                n_lt_moves=num_lt_moves,
                mip_gap=max_mip_gap,
                instance_id=scenario_id,
                num_stores=N, num_products=M,
                num_vehicles=self.vehicle_count,
                num_periods=num_periods,
                stage1_cost=stage1_total,
                lt_cost=lt_total,
                final_stockout=final_stockout,
                stockout_rate=stockout_rate,
            ), lt_moves_out

        except Exception as exc:
            runtime = time.time() - t0
            tb = traceback.format_exc()
            status = "timeout" if runtime >= self.time_limit * 0.98 else "failed"
            return JointTCResult(
                scenario_id=scenario_id,
                method=self.METHOD_NAME,
                status=status,
                success=False,
                runtime_seconds=runtime,
                error_message=str(exc),
                error_traceback=tb,
            ), lt_moves_out


# ===========================================================================
# TSRFPTCResult + TSRFPTCRunner  (true C&CG robust method)
# ===========================================================================

@dataclass
class TSRFPTCResult:
    """Result dataclass for Man-TSRFP-TC (C&CG robust), field-aligned with JointTCResult."""

    scenario_id:        str   = ""
    method:             str   = "Man-TSRFP-TC"
    status:             str   = ""
    success:            bool  = False
    runtime_seconds:    float = float("nan")
    total_cost:         float = float("nan")
    holding_cost:       float = float("nan")
    routing_cost:       float = float("nan")
    transshipment_cost: float = float("nan")
    shortage_cost:      float = float("nan")
    shortage_qty:       float = float("nan")
    service_level:      float = float("nan")
    lt_total_qty:       float = float("nan")
    n_lt_moves:         int   = 0
    n_routes:           int   = 0
    mip_gap:            float = float("nan")
    error_message:      str   = ""
    error_traceback:    str   = ""

    instance_id:    str   = ""
    num_stores:     int   = 0
    num_products:   int   = 0
    num_vehicles:   int   = 0
    num_periods:    int   = 0
    stage1_cost:    float = float("nan")
    lt_cost:        float = float("nan")
    final_stockout: float = float("nan")
    stockout_rate:  float = float("nan")

    ccg_iterations:   int   = 0
    final_robust_gap: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TSRFPTCRunner:
    """Run Man-TSRFP-TC (true C&CG robust) benchmark via rolling-horizon solves.

    Stage 1 plans with forecast demand only. Stage 2 is recourse under worst-case
    demand from the uncertainty set. Final metrics evaluated on actual dim_real.
    Mirrors JointTCRunner / ManBFPTCRunner interface.
    """

    METHOD_NAME = "Man-TSRFP-TC"

    def __init__(
        self,
        time_limit: int = 1200,
        mip_gap: float = 0.01,
        threads: int = 4,
        vehicle_count: int = 2,
        vehicle_capacity: float = 1500.0,
        holding_cost_rate: float = 0.1,
        shortage_cost_rate: float = 2.0,
        lt_cost_multiplier: float = 1.2,
        product_value_default: float = 1.0,
        product_volume_default: float = 1.0,
        store_capacity_floor: float = 10000.0,
        store_capacity_init_multiplier: float = 5.0,
        store_capacity_buffer: float = 5000.0,
        dc_capacity: float = 1e12,
        dc_initial_stock_per_product: float = 10000.0,
        distance_fallback: float = 100.0,
        b_budget: float = 0.3,
        rho: float = 0.5,
        max_ccg_iter: int = 20,
        ccg_tol: float = 1e-3,
    ):
        self.time_limit         = int(time_limit)
        self.mip_gap            = float(mip_gap)
        self.threads            = int(threads)
        self.vehicle_count      = int(vehicle_count)
        self.vehicle_capacity   = float(vehicle_capacity)
        self.holding_cost_rate  = float(holding_cost_rate)
        self.shortage_cost_rate = float(shortage_cost_rate)
        self.lt_cost_multiplier = float(lt_cost_multiplier)
        self.product_value_default   = float(product_value_default)
        self.product_volume_default  = float(product_volume_default)
        self.store_capacity_floor          = float(store_capacity_floor)
        self.store_capacity_init_multiplier = float(store_capacity_init_multiplier)
        self.store_capacity_buffer         = float(store_capacity_buffer)
        self.dc_capacity                   = float(dc_capacity)
        self.dc_initial_stock_per_product  = float(dc_initial_stock_per_product)
        self.distance_fallback             = float(distance_fallback)
        self.b_budget    = float(b_budget)
        self.rho         = float(rho)
        self.max_ccg_iter = int(max_ccg_iter)
        self.ccg_tol     = float(ccg_tol)
        self.source_name = self.METHOD_NAME

    def _build_static_inputs(
        self,
        stores: List[str],
        skus: List[str],
        dist_dict: Dict[Tuple[str, str], float],
        df_slice: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Identical to JointTCRunner._build_static_inputs."""
        N = len(stores)
        M = len(skus)
        wh_name = "__WAREHOUSE__"
        node_names = [wh_name] + stores
        cij = [[0.0] * (N + 1) for _ in range(N + 1)]
        for i, ni in enumerate(node_names):
            for j, nj in enumerate(node_names):
                if i == j:
                    continue
                d = dist_dict.get((ni, nj)) or dist_dict.get((nj, ni)) or self.distance_fallback
                cij[i][j] = float(d)

        Qk = [self.vehicle_capacity for _ in range(self.vehicle_count)]
        bk = [0.0 for _ in range(self.vehicle_count)]
        him = [[0.0] * M for _ in range(N + 1)]
        for i in range(1, N + 1):
            for m in range(M):
                him[i][m] = self.holding_cost_rate
        vim = [[self.product_volume_default] * M for _ in range(N + 1)]
        pim = [[self.product_value_default] * M for _ in range(N + 1)]

        store_to_idx = {s: i for i, s in enumerate(stores)}
        sku_to_idx   = {s: i for i, s in enumerate(skus)}

        import numpy as np
        store_init_inv = np.zeros((N, M), dtype=float)
        for _, row in df_slice.iterrows():
            s = str(row["store"]); p = str(row["sku"])
            if s not in store_to_idx or p not in sku_to_idx:
                continue
            si = store_to_idx[s]; pi = sku_to_idx[p]
            if store_init_inv[si, pi] == 0.0:
                store_init_inv[si, pi] = float(row["end_qty"])

        Ui = [self.dc_capacity]
        for si in range(N):
            init_total = float(store_init_inv[si].sum())
            cap = max(
                self.store_capacity_floor,
                init_total * self.store_capacity_init_multiplier + self.store_capacity_buffer,
            )
            Ui.append(cap)

        Iim0_dc     = [self.dc_initial_stock_per_product] * M
        Iim0_stores = store_init_inv.tolist()

        return {
            "N": N, "M": M,
            "cij": cij, "Qk": Qk, "bk": bk,
            "him": him, "vim": vim, "pim": pim,
            "Ui": Ui,
            "Iim0_dc": Iim0_dc,
            "Iim0_stores": Iim0_stores,
            "store_to_idx": store_to_idx,
            "sku_to_idx": sku_to_idx,
        }

    def _period_demand(
        self,
        df_slice: pd.DataFrame,
        period_date,
        store_to_idx: Dict[str, int],
        sku_to_idx: Dict[str, int],
        N: int,
        M: int,
    ) -> List[List[float]]:
        rows = df_slice[df_slice["period_date"] == period_date]
        dim = [[0.0] * M for _ in range(N)]
        for _, row in rows.iterrows():
            s, p, q = str(row["store"]), str(row["sku"]), float(row["sale_qty"])
            if s in store_to_idx and p in sku_to_idx:
                dim[store_to_idx[s]][sku_to_idx[p]] += q
        return dim

    def run_scenario(
        self,
        scenario,
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
        demand_shock: Optional[Dict[Tuple[str, str, int], float]] = None,
    ) -> Tuple["TSRFPTCResult", List[Dict[str, Any]]]:
        t0 = time.time()
        scenario_id = getattr(scenario, "scenario_id", "")
        lt_moves_out: List[Dict[str, Any]] = []

        try:
            stores = list(getattr(scenario, "selected_stores", []))
            skus   = list(getattr(scenario, "selected_skus",   []))
            if not stores or not skus:
                raise ValueError("Scenario has empty selected_stores or selected_skus")

            static = self._build_static_inputs(stores, skus, dist_dict, df_slice)
            N, M = static["N"], static["M"]
            store_to_idx = static["store_to_idx"]
            sku_to_idx   = static["sku_to_idx"]
            id_to_store  = {i: s for s, i in store_to_idx.items()}
            id_to_sku    = {i: s for s, i in sku_to_idx.items()}

            periods_dt  = sorted(df_slice["period_date"].unique())
            num_periods = len(periods_dt)

            Iim0_dc     = list(static["Iim0_dc"])
            Iim0_stores = [list(r) for r in static["Iim0_stores"]]

            per_period_tl = max(60, self.time_limit // max(num_periods, 1))

            agg_routing = agg_veh_fixed = 0.0
            agg_s1_holding = agg_s1_shortage = agg_s1_shortage_units = 0.0
            agg_s2_lt = agg_s2_holding = agg_s2_shortage = agg_s2_shortage_units = 0.0
            agg_total_demand = agg_lt_qty = 0.0
            max_mip_gap = 0.0
            total_ccg_iters = 0
            max_robust_gap  = 0.0
            any_infeasible  = False
            num_lt_moves    = 0

            for t_idx, period_date in enumerate(periods_dt, start=1):
                dim_actual   = self._period_demand(df_slice, period_date,
                                                   store_to_idx, sku_to_idx, N, M)
                dim_forecast = [list(r) for r in dim_actual]

                if demand_shock:
                    dim_real = [
                        [dim_actual[i][m] * demand_shock.get(
                             (id_to_store[i], id_to_sku[m], t_idx), 1.0)
                         for m in range(M)]
                        for i in range(N)
                    ]
                else:
                    dim_real = [list(r) for r in dim_actual]

                Iim0 = [list(Iim0_dc)] + [list(r) for r in Iim0_stores]

                res = solve_tsrfp_tc(
                    retailer_number=N,
                    product_kind=M,
                    vehicle_number=self.vehicle_count,
                    Ui=static["Ui"],
                    cij=static["cij"],
                    pim=static["pim"],
                    Qk=static["Qk"],
                    bk=static["bk"],
                    vim=static["vim"],
                    him=static["him"],
                    Iim0=Iim0,
                    dim=dim_forecast,
                    dim_real=dim_real,
                    shortage_penalty_stage1=self.shortage_cost_rate,
                    shortage_penalty_stage2=self.shortage_cost_rate,
                    lt_cost_multiplier=self.lt_cost_multiplier,
                    b_budget=self.b_budget,
                    rho=self.rho,
                    max_ccg_iter=self.max_ccg_iter,
                    ccg_tol=self.ccg_tol,
                    time_limit=per_period_tl,
                    mip_gap=self.mip_gap,
                    threads=self.threads,
                    verbose=False,
                )

                if res.get("infeasible"):
                    any_infeasible = True
                    break

                agg_routing           += float(res.get("stage1_routing_cost",    0.0))
                agg_veh_fixed         += float(res.get("stage1_vehicle_fixed_cost", 0.0))
                agg_s1_holding        += float(res.get("stage1_holding_cost",    0.0))
                agg_s1_shortage       += float(res.get("stage1_forecast_shortage", 0.0))
                agg_s1_shortage_units += float(res.get("stage1_forecast_shortage_units", 0.0))
                agg_s2_lt             += float(res.get("stage2_lt_transport_cost", 0.0))
                agg_s2_holding        += float(res.get("stage2_holding_cost",    0.0))
                agg_s2_shortage       += float(res.get("stage2_remaining_shortage", 0.0))
                agg_s2_shortage_units += float(res.get("stage2_remaining_shortage_units", 0.0))
                max_mip_gap    = max(max_mip_gap, float(res.get("mip_gap", 0.0)))
                total_ccg_iters += int(res.get("ccg_iterations", 0))
                max_robust_gap  = max(max_robust_gap, float(res.get("final_robust_gap", 0.0)))

                agg_total_demand += sum(dim_actual[i][m]
                                        for i in range(N) for m in range(M))

                for mv in res.get("lt_moves", []):
                    qty = float(mv.get("qty", 0.0))
                    if qty <= 1e-9:
                        continue
                    fi, ti, pi = int(mv["from_store_idx"]), int(mv["to_store_idx"]), int(mv["product_idx"])
                    lt_moves_out.append({
                        "scenario_id": scenario_id,
                        "method":      self.METHOD_NAME,
                        "period":      t_idx,
                        "from_store":  id_to_store.get(fi, str(fi)),
                        "to_store":    id_to_store.get(ti, str(ti)),
                        "sku":         id_to_sku.get(pi,  str(pi)),
                        "lt_qty":      qty,
                        "lt_unit_cost": float(mv.get("unit_cost", 0.0)),
                        "lt_total_cost": qty * float(mv.get("unit_cost", 0.0)),
                        "vehicle":     "N/A",
                    })
                    agg_lt_qty   += qty
                    num_lt_moves += 1

                Iim0_dc     = list(res["post_lt_dc_inventory"])
                Iim0_stores = [list(r) for r in res["post_lt_inventory"]]

            runtime = time.time() - t0

            if any_infeasible:
                return TSRFPTCResult(
                    scenario_id=scenario_id, method=self.METHOD_NAME,
                    status="failed", success=False, runtime_seconds=runtime,
                    instance_id=scenario_id,
                    num_stores=N, num_products=M,
                    num_vehicles=self.vehicle_count, num_periods=num_periods,
                    error_message="At least one period of Man-TSRFP-TC was infeasible",
                ), lt_moves_out

            routing_cost       = agg_routing + agg_veh_fixed
            transshipment_cost = agg_s2_lt
            holding_cost       = agg_s1_holding + agg_s2_holding
            shortage_cost      = agg_s1_shortage + agg_s2_shortage
            stage1_total       = agg_routing + agg_veh_fixed + agg_s1_holding + agg_s1_shortage
            lt_total           = agg_s2_lt + agg_s2_holding + agg_s2_shortage
            total_cost         = stage1_total + lt_total

            shortage_qty   = agg_s1_shortage_units + agg_s2_shortage_units
            final_stockout = agg_s2_shortage_units
            stockout_rate  = final_stockout / agg_total_demand if agg_total_demand > 0 else 0.0
            service_level  = max(0.0, 1.0 - shortage_qty / max(agg_total_demand, 1.0))

            print(
                f"  [Man-TSRFP-TC]  routing={routing_cost:.2f}  "
                f"holding={holding_cost:.2f}  shortage={shortage_cost:.2f}  "
                f"lt={transshipment_cost:.2f}  total={total_cost:.2f}  "
                f"ccg_iters={total_ccg_iters}"
            )

            return TSRFPTCResult(
                scenario_id=scenario_id,
                method=self.METHOD_NAME,
                status="success",
                success=True,
                runtime_seconds=runtime,
                total_cost=total_cost,
                holding_cost=holding_cost,
                routing_cost=routing_cost,
                transshipment_cost=transshipment_cost,
                shortage_cost=shortage_cost,
                shortage_qty=shortage_qty,
                service_level=service_level,
                lt_total_qty=agg_lt_qty,
                n_lt_moves=num_lt_moves,
                mip_gap=max_mip_gap,
                instance_id=scenario_id,
                num_stores=N, num_products=M,
                num_vehicles=self.vehicle_count,
                num_periods=num_periods,
                stage1_cost=stage1_total,
                lt_cost=lt_total,
                final_stockout=final_stockout,
                stockout_rate=stockout_rate,
                ccg_iterations=total_ccg_iters,
                final_robust_gap=max_robust_gap,
            ), lt_moves_out

        except Exception as exc:
            runtime = time.time() - t0
            tb = traceback.format_exc()
            status = "timeout" if runtime >= self.time_limit * 0.98 else "failed"
            return TSRFPTCResult(
                scenario_id=scenario_id,
                method=self.METHOD_NAME,
                status=status,
                success=False,
                runtime_seconds=runtime,
                error_message=str(exc),
                error_traceback=tb,
            ), lt_moves_out
