"""
Man-BFP-TC thesis-pipeline wrapper.

This file is the bridge between the single-period two-stage Man-BFP-TC solver
(`GitHub Code - Man et al./LR-IRP-LT/solve_BFP_TC.py`) and the thesis benchmark
runner used in `Validate_with_Achamrah_Kaggle.py`.

It exposes:
    * `ManBFPTCResult`  — a result dataclass whose field names match the
       `ScenarioResult` dataclass in the validator, plus a few BFP-specific
       extras (stage1_cost, lt_cost, final_stockout, stockout_rate, ...).
    * `ManBFPTCRunner`  — a runner with `run_scenario(scenario, df_slice,
       dist_dict)` that mirrors `AchamrahRunner.run_scenario(...)`, so it
       can be plugged into the same experiment loop. It does NOT modify any
       file in `Validate_with_Achamrah_Kaggle.py`; the experiment loop must
       opt in.

Strategy
--------
Man's BFP is single-period. To run it on the multi-period thesis scenarios
we apply a rolling-horizon driver:

    for each period t in scenario.H:
        build single-period inputs (forecast = actual demand for period t,
        I_initial = post-LT inventory carried from t-1)
        call solve_man_bfp_tc(...)
        accumulate costs / shortages / LT moves
        roll inventory: store inventory <- post-LT physical inventory;
                        DC inventory   <- DC inventory after Stage 1.

Because the thesis CSV provides only actual demand (`sale_qty`), the forecast
is set equal to the actual demand. Any forecast/actual divergence study can
later be added by perturbing `dim` before the call.
"""

from __future__ import annotations

import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Make solve_BFP_TC importable from its sibling folder                         #
# --------------------------------------------------------------------------- #

_HERE = Path(__file__).resolve().parent
_SOLVER_DIR = _HERE / "GitHub Code - Man et al." / "LR-IRP-LT"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

from solve_BFP_TC import solve_man_bfp_tc  # noqa: E402


# --------------------------------------------------------------------------- #
# Result dataclass — fields aligned with ScenarioResult in the validator       #
# --------------------------------------------------------------------------- #

@dataclass
class ManBFPTCResult:
    """Aligned with `ScenarioResult` in `Validate_with_Achamrah_Kaggle.py`,
    plus BFP-specific extras the user listed in the task spec."""

    # ScenarioResult-shaped fields
    scenario_id:        str = ""
    method:             str = "Man-BFP-TC"
    status:             str = ""           # "success" | "failed" | "timeout"
    success:            bool = False
    runtime_seconds:    float = 0.0
    total_cost:         float = float("nan")
    holding_cost:       float = float("nan")
    routing_cost:       float = float("nan")
    transshipment_cost: float = float("nan")
    shortage_cost:      float = float("nan")
    shortage_qty:       float = float("nan")
    service_level:      float = float("nan")
    lt_total_qty:       float = float("nan")
    n_lt_moves:         int = 0
    n_routes:           int = 0
    n_columns_generated: int = 0
    n_columns_selected:  int = 0
    gnn_checkpoint_used: str = ""
    achamrah_vehicle_indexed_lt: Optional[bool] = None
    num_vars:            int = 0
    num_constrs:         int = 0
    mip_gap:             float = float("nan")
    error_message:       str = ""
    error_traceback:     str = ""

    # Man-BFP-TC extras (exactly the fields the user listed in the task)
    instance_id:    str = ""
    num_stores:     int = 0
    num_products:   int = 0
    num_vehicles:   int = 0
    num_periods:    int = 0
    stage1_cost:    float = float("nan")
    lt_cost:        float = float("nan")
    final_stockout: float = float("nan")
    stockout_rate:  float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Thesis-pipeline runner                                                       #
# --------------------------------------------------------------------------- #

class ManBFPTCRunner:
    """Run Man-BFP-TC on a thesis scenario via rolling-horizon single-period BFPs.

    Mirrors the public surface of `AchamrahRunner` in
    `Validate_with_Achamrah_Kaggle.py`:
        * `__init__(time_limit, mip_gap, threads, ...)`
        * `run_scenario(scenario, df_slice, dist_dict) -> (result, lt_moves_list)`
    so it can be wired into the same scenario loop without modifying the CG/GNN
    pipeline.
    """

    METHOD_NAME = "Man-BFP-TC"

    def __init__(
        self,
        time_limit: int = 1200,
        mip_gap: float = 0.01,
        threads: int = 4,
        # Fleet — match thesis defaults (VEHICLE_COUNT, VEHICLE_CAPACITY)
        vehicle_count: int = 2,
        vehicle_capacity: float = 1500.0,
        # Costs — match thesis defaults so total_cost is comparable to Achamrah
        holding_cost_rate: float = 0.1,
        shortage_cost_rate: float = 2.0,
        # Man-BFP specifics
        lt_cost_multiplier: float = 1.2,         # Man's cij_trans = 1.2 * cij
        product_value_default: float = 1.0,      # paper's v_im (unused in TC obj
                                                 # but still needed for solve API)
        product_volume_default: float = 1.0,     # paper's O_im
        # Capacity sizing
        store_capacity_floor: float = 10000.0,
        store_capacity_init_multiplier: float = 5.0,
        store_capacity_buffer: float = 5000.0,
        dc_capacity: float = 1e12,
        dc_initial_stock_per_product: float = 10000.0,
        # Distance fallback for warehouse <-> store and any missing pair
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
        self.product_value_default  = float(product_value_default)
        self.product_volume_default = float(product_volume_default)

        self.store_capacity_floor          = float(store_capacity_floor)
        self.store_capacity_init_multiplier = float(store_capacity_init_multiplier)
        self.store_capacity_buffer         = float(store_capacity_buffer)
        self.dc_capacity                   = float(dc_capacity)
        self.dc_initial_stock_per_product  = float(dc_initial_stock_per_product)
        self.distance_fallback             = float(distance_fallback)

        self.source_name = self.METHOD_NAME

    # ------------------------------------------------------------------ #
    # Scenario -> per-period numeric inputs                               #
    # ------------------------------------------------------------------ #

    def _build_static_inputs(
        self,
        stores: List[str],
        skus: List[str],
        dist_dict: Dict[Tuple[str, str], float],
        df_slice: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Build the static parts of the BFP-TC inputs (capacities, distances,
        product params, initial inventory). Returns a dict that
        `_build_period_inputs` consumes."""
        N = len(stores)          # retailer count
        M = len(skus)             # product count

        # Distance matrix in (N+1) x (N+1) form, 0 = DC.
        wh_name = "__WAREHOUSE__"
        node_names = [wh_name] + stores
        cij = [[0.0] * (N + 1) for _ in range(N + 1)]
        for i, ni in enumerate(node_names):
            for j, nj in enumerate(node_names):
                if i == j:
                    cij[i][j] = 0.0
                    continue
                d = dist_dict.get((ni, nj))
                if d is None:
                    d = dist_dict.get((nj, ni))
                if d is None:
                    d = self.distance_fallback
                cij[i][j] = float(d)

        # Vehicle capacity; fixed dispatch cost set to 0 to match Thesis C
        # (data.vehicle_fixed_cost = 0.0 in ThesisCRunner._build_irp_data).
        # Man's paper uses bk = Q/10 but that inflates Man's routing cost by
        # ~300–2400 per period vs Thesis C which has no fixed cost — unfair comparison.
        Qk = [self.vehicle_capacity for _ in range(self.vehicle_count)]
        bk = [0.0 for _ in range(self.vehicle_count)]

        # Product value & volume: defaults; uniform across stores.
        pim = [[self.product_value_default for _ in range(M)] for _ in range(N)]
        vim = [[self.product_volume_default for _ in range(M)] for _ in range(N)]

        # Holding cost: 0 at DC (cost-of-holding at warehouse is treated as
        # 0 to match the thesis pipeline), holding_cost_rate at stores.
        him = [[0.0 for _ in range(M)]]
        for _ in range(N):
            him.append([self.holding_cost_rate for _ in range(M)])

        # Capacities: store capacity ~ thesis pipeline rule; DC effectively
        # unbounded.
        sku_to_idx = {s: i for i, s in enumerate(skus)}
        store_to_idx = {s: i for i, s in enumerate(stores)}

        store_init_inv = np.zeros((N, M), dtype=float)
        for _, row in df_slice.iterrows():
            s = row["store"]; p = row["sku"]
            if s not in store_to_idx or p not in sku_to_idx:
                continue
            si = store_to_idx[s]; pi = sku_to_idx[p]
            # Use the first-period end_qty as initial inventory (matches Achamrah).
            if store_init_inv[si, pi] == 0.0:
                store_init_inv[si, pi] = float(row["end_qty"])

        Ui = [self.dc_capacity]
        for si in range(N):
            init_total_si = float(store_init_inv[si].sum())
            cap = max(
                self.store_capacity_floor,
                init_total_si * self.store_capacity_init_multiplier
                    + self.store_capacity_buffer,
            )
            Ui.append(cap)

        # Initial inventory rolling state.
        Iim0_dc = [self.dc_initial_stock_per_product for _ in range(M)]
        Iim0_stores = store_init_inv.tolist()

        return dict(
            N=N, M=M,
            cij=cij, Qk=Qk, bk=bk,
            pim=pim, vim=vim, him=him,
            Ui=Ui,
            Iim0_dc=Iim0_dc,
            Iim0_stores=Iim0_stores,
            store_to_idx=store_to_idx, sku_to_idx=sku_to_idx,
            stores=stores, skus=skus,
        )

    def _period_demand(
        self,
        df_slice: pd.DataFrame,
        period_date,
        store_to_idx: Dict[str, int],
        sku_to_idx: Dict[str, int],
        N: int,
        M: int,
    ) -> List[List[float]]:
        d = [[0.0] * M for _ in range(N)]
        period_rows = df_slice[df_slice["period_date"] == period_date]
        for _, row in period_rows.iterrows():
            s = row["store"]; p = row["sku"]
            if s not in store_to_idx or p not in sku_to_idx:
                continue
            d[store_to_idx[s]][sku_to_idx[p]] = float(row["sale_qty"])
        return d

    # ------------------------------------------------------------------ #
    # Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def run_scenario(
        self,
        scenario,                                  # ScenarioSpec, duck-typed
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
        demand_shock: Optional[Dict[Tuple[str, str, int], float]] = None,
    ) -> Tuple[ManBFPTCResult, List[Dict[str, Any]]]:
        """Run Man-BFP-TC in rolling-horizon mode.

        demand_shock: optional {(store_name, sku_name, period_int): multiplier}.
            Stage-1 (routing) always uses forecast demand (dim_forecast = sale_qty).
            Stage-2 (LT) uses dim_real = sale_qty * shock, simulating the scenario
            where actual demand deviates from the plan — matching Man et al.'s
            two-stage stochastic design.  If None, dim_real = dim_forecast (deterministic).
        """
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
            cij = static["cij"]  # distance matrix; index 0=DC, 1..N=stores

            periods_dt = sorted(df_slice["period_date"].unique())
            num_periods = len(periods_dt)

            # Rolling state for inventory.
            Iim0_dc     = list(static["Iim0_dc"])
            Iim0_stores = [list(row) for row in static["Iim0_stores"]]

            # Allocate per-period time budget proportional to count.
            per_period_time_limit = max(
                30,
                int(self.time_limit / max(num_periods, 1)),
            )

            # Cost / metric accumulators.
            agg_routing = 0.0
            agg_vehicle_fixed = 0.0
            agg_stage1_holding = 0.0
            agg_stage1_forecast_shortage = 0.0
            agg_stage1_forecast_shortage_units = 0.0
            agg_stage2_lt_transport = 0.0
            agg_stage2_holding = 0.0
            agg_stage2_remaining_shortage = 0.0
            agg_stage2_remaining_shortage_units = 0.0
            agg_total_demand = 0.0
            agg_lt_total_qty = 0.0
            max_mip_gap = 0.0
            any_infeasible = False
            num_lt_moves = 0
            id_to_store = {i: s for s, i in store_to_idx.items()}
            id_to_sku   = {i: s for s, i in sku_to_idx.items()}

            for t_idx, period_date in enumerate(periods_dt, start=1):
                dim_actual = self._period_demand(
                    df_slice, period_date, store_to_idx, sku_to_idx, N, M,
                )
                # Stage-1 always plans with forecast (= historical sale_qty).
                dim_forecast = [list(row) for row in dim_actual]

                # Stage-2 sees realized demand: apply shock if provided, else
                # realized = forecast (deterministic baseline).
                if demand_shock:
                    dim_real = [
                        [
                            dim_actual[i][m] * demand_shock.get(
                                (id_to_store[i], id_to_sku[m], t_idx), 1.0
                            )
                            for m in range(M)
                        ]
                        for i in range(N)
                    ]
                else:
                    dim_real = [list(row) for row in dim_actual]

                # Iim0 in Man's layout: index 0 = DC, 1..N = stores
                Iim0 = [list(Iim0_dc)] + [list(row) for row in Iim0_stores]

                res = solve_man_bfp_tc(
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
                    time_limit=per_period_time_limit,
                    mip_gap=self.mip_gap,
                    threads=self.threads,
                    verbose=False,
                )

                if res.get("infeasible"):
                    any_infeasible = True
                    # Stop rolling — record what we have and bail out.
                    break

                agg_routing += float(res.get("stage1_routing_cost", 0.0))
                agg_vehicle_fixed += float(res.get("stage1_vehicle_fixed_cost", 0.0))
                agg_stage1_holding += float(res.get("stage1_holding_cost", 0.0))
                agg_stage1_forecast_shortage += float(res.get("stage1_forecast_shortage", 0.0))
                agg_stage1_forecast_shortage_units += float(
                    res.get("stage1_forecast_shortage_units", 0.0))
                # Thesis math model: LT cost = 0.01 * dist_{ij} * qty (per-unit, not per-arc)
                # Replace Man's arc-gate cost (cij_trans * zij) with thesis per-unit formula.
                period_thesis_lt = 0.0
                for mv in res.get("lt_moves", []):
                    qty = float(mv.get("qty", 0.0))
                    if qty <= 1e-9:
                        continue
                    fi = int(mv["from_store_idx"])
                    ti = int(mv["to_store_idx"])
                    pi = int(mv["product_idx"])
                    thesis_unit = 0.01 * cij[fi + 1][ti + 1]
                    thesis_total = thesis_unit * qty
                    period_thesis_lt += thesis_total
                    lt_moves_out.append({
                        "scenario_id": scenario_id,
                        "method":      self.METHOD_NAME,
                        "period":      t_idx,
                        "from_store":  id_to_store.get(fi, str(fi)),
                        "to_store":    id_to_store.get(ti, str(ti)),
                        "sku":         id_to_sku.get(pi,   str(pi)),
                        "lt_qty":      qty,
                        "lt_unit_cost": thesis_unit,
                        "lt_total_cost": thesis_total,
                        "vehicle":     "N/A",
                    })
                    agg_lt_total_qty += qty
                    num_lt_moves += 1
                agg_stage2_lt_transport += period_thesis_lt

                agg_stage2_holding += float(res.get("stage2_holding_cost", 0.0))
                agg_stage2_remaining_shortage += float(res.get("stage2_remaining_shortage", 0.0))
                agg_stage2_remaining_shortage_units += float(
                    res.get("stage2_remaining_shortage_units", 0.0))
                max_mip_gap = max(max_mip_gap, float(res.get("mip_gap", 0.0)))

                # Period demand contribution to total demand (for service level).
                agg_total_demand += sum(dim_actual[i][m]
                                        for i in range(N) for m in range(M))

                # Roll inventory state to next period.
                Iim0_dc     = list(res["post_lt_dc_inventory"])
                Iim0_stores = [list(row) for row in res["post_lt_inventory"]]

            runtime = time.time() - t0

            if any_infeasible:
                status = "failed"
                success = False
                result = ManBFPTCResult(
                    scenario_id=scenario_id,
                    method=self.METHOD_NAME,
                    status=status,
                    success=success,
                    runtime_seconds=runtime,
                    instance_id=scenario_id,
                    num_stores=N, num_products=M,
                    num_vehicles=self.vehicle_count,
                    num_periods=num_periods,
                    error_message="At least one period of Man-BFP-TC was infeasible",
                )
                return result, lt_moves_out

            # Combine into ScenarioResult-shaped totals.
            routing_cost = agg_routing + agg_vehicle_fixed
            transshipment_cost = agg_stage2_lt_transport
            holding_cost = agg_stage1_holding + agg_stage2_holding
            shortage_cost = (
                agg_stage1_forecast_shortage + agg_stage2_remaining_shortage
            )
            stage1_total = (
                agg_routing + agg_vehicle_fixed
                + agg_stage1_holding + agg_stage1_forecast_shortage
            )
            lt_total = (
                agg_stage2_lt_transport
                + agg_stage2_holding + agg_stage2_remaining_shortage
            )
            total_cost = stage1_total + lt_total

            shortage_qty = (
                agg_stage1_forecast_shortage_units
                + agg_stage2_remaining_shortage_units
            )
            final_stockout = agg_stage2_remaining_shortage_units
            stockout_rate = (
                final_stockout / agg_total_demand if agg_total_demand > 0 else 0.0
            )
            service_level = max(
                0.0,
                1.0 - shortage_qty / max(agg_total_demand, 1.0),
            )

            print(
                f"  [Man-BFP-TC breakdown] routing={routing_cost:.2f}  "
                f"holding={holding_cost:.2f}  shortage={shortage_cost:.2f}  "
                f"lt={transshipment_cost:.2f}  total={total_cost:.2f}"
            )

            result = ManBFPTCResult(
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
                lt_total_qty=agg_lt_total_qty,
                n_lt_moves=num_lt_moves,
                n_routes=0,
                mip_gap=max_mip_gap,
                # extras
                instance_id=scenario_id,
                num_stores=N, num_products=M,
                num_vehicles=self.vehicle_count,
                num_periods=num_periods,
                stage1_cost=stage1_total,
                lt_cost=lt_total,
                final_stockout=final_stockout,
                stockout_rate=stockout_rate,
            )
            return result, lt_moves_out

        except Exception as exc:
            runtime = time.time() - t0
            tb = traceback.format_exc()
            status = "timeout" if runtime >= self.time_limit * 0.98 else "failed"
            print(f"[Man-BFP-TC] {scenario_id} {status}: {exc}")
            print(f"[Man-BFP-TC TRACEBACK]:\n{tb}")
            result = ManBFPTCResult(
                scenario_id=scenario_id,
                method=self.METHOD_NAME,
                status=status,
                success=False,
                runtime_seconds=runtime,
                instance_id=scenario_id,
                error_message=str(exc),
                error_traceback=tb,
            )
            return result, []


# --------------------------------------------------------------------------- #
# Standalone CLI for ad-hoc smoke tests                                        #
# --------------------------------------------------------------------------- #

def _smoke_self_test() -> int:
    """Build a tiny synthetic scenario, run the BFP-TC, and print the result.

    Useful as a 1-minute sanity check that solve_BFP_TC + the wrapper are
    importable and produce the expected output keys.
    """
    from types import SimpleNamespace

    stores = ["S1", "S2", "S3"]
    skus   = ["P1", "P2"]
    periods = pd.to_datetime(["2025-01-06", "2025-01-13"])

    rows = []
    rng = np.random.default_rng(0)
    for s in stores:
        for p in skus:
            for t in periods:
                rows.append({
                    "store": s, "sku": p, "period_date": t,
                    "sale_qty": float(rng.integers(20, 60)),
                    "end_qty":  float(rng.integers(30, 50)),
                })
    df_slice = pd.DataFrame(rows)

    # Synthetic distance matrix between store names.
    coords = {s: rng.uniform(0, 100, size=2) for s in stores}
    coords["__WAREHOUSE__"] = np.array([0.0, 0.0])
    dist_dict: Dict[Tuple[str, str], float] = {}
    for a in coords:
        for b in coords:
            if a != b:
                dist_dict[(a, b)] = float(np.linalg.norm(coords[a] - coords[b]))

    scenario = SimpleNamespace(
        scenario_id="smoke_test_3x2x2",
        selected_stores=stores,
        selected_skus=skus,
    )

    runner = ManBFPTCRunner(time_limit=120, mip_gap=0.05, threads=4,
                            vehicle_count=2, vehicle_capacity=400.0)
    result, lt_moves = runner.run_scenario(scenario, df_slice, dist_dict)

    print("\n=== Man-BFP-TC smoke result ===")
    for k, v in result.to_dict().items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v) or abs(v) < 1e6):
            print(f"  {k:30s} {v}")
        else:
            print(f"  {k:30s} {v}")
    print(f"  lt_moves: {len(lt_moves)} moves")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(_smoke_self_test())
