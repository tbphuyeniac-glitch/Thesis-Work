"""
Man-BFP-TC: Total-cost variant of the deterministic BFP from solve_DB_FP.py.

Adapted from `LR-IRP-LT/solve_DB_FP.py`. The original two-stage structure is
preserved (Stage 1 = DC->store replenishment + routing using forecast demand;
Stage 2 = store-to-store lateral transshipment after actual demand is revealed).

Differences vs. solve_DB_FP.py:
  * The fractional logistics-ratio objective is replaced with direct total-cost
    minimization.
  * Dinkelbach's algorithm (`alpha` / `beta` updates) is removed.
  * Stage 1 includes a forecast-shortage penalty: pen_s1 * (dim_forecast - sdim).
  * Stage 2 uses explicit non-negative `remaining_shortage` variables and a
    remaining-shortage penalty.
  * Capacity coverage at the DC and at retailers is fixed to use sum_m vim*Iim0
    instead of the original code's vim[<last m>] * sum_m Iim0 multiplier.

Solve_DB_FP's other constraints (routing, capacity, sub-tour elimination,
inventory balance, LT feasibility) are kept unchanged.

Two ways to use this file:
  1. As a script:    `python solve_BFP_TC.py <dat_file>`  (single instance,
     same .dat parser as solve_DB_FP.py).
  2. As a library:   `from solve_BFP_TC import solve_man_bfp_tc` and pass numeric
     inputs directly. The thesis-pipeline wrapper `man_bfp_tc_benchmark.py` calls
     this entry point.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from gurobipy import GRB, LinExpr, Model, quicksum

try:
    # Allow `python solve_BFP_TC.py <dat>` as a drop-in for solve_DB_FP.py.
    from revise_data import parse_data_file  # type: ignore
except Exception:  # pragma: no cover
    parse_data_file = None  # only needed for the script entry point


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _power_set_binary(node_list: Sequence[int]) -> List[List[int]]:
    """All subsets of `node_list` (used for sub-tour elimination, like the original)."""
    n = len(node_list)
    out: List[List[int]] = []
    for i in range(2 ** n):
        sub: List[int] = []
        for j in range(n):
            if (i >> j) % 2:
                sub.append(node_list[j])
        out.append(sub)
    return out


# --------------------------------------------------------------------------- #
# Single-period two-stage BFP-TC solver                                        #
# --------------------------------------------------------------------------- #

def solve_man_bfp_tc(
    retailer_number: int,
    product_kind: int,
    vehicle_number: int,
    Ui: Sequence[float],
    cij: Sequence[Sequence[float]],
    pim: Sequence[Sequence[float]],
    Qk: Sequence[float],
    bk: Sequence[float],
    vim: Sequence[Sequence[float]],
    him: Sequence[Sequence[float]],
    Iim0: Sequence[Sequence[float]],
    dim_real: Sequence[Sequence[float]],
    dim: Sequence[Sequence[float]],
    *,
    shortage_penalty_stage1: float = 0.25,
    shortage_penalty_stage2: float = 0.25,
    lt_cost_multiplier: float = 1.2,
    time_limit: int = 600,
    mip_gap: float = 0.01,
    threads: int = 4,
    verbose: bool = False,
) -> Dict:
    """Solve a single-period two-stage Man-BFP-TC instance.

    Inputs follow the Man et al. naming convention used in `solve_DB_FP.py`:
      * retailer_number, product_kind, vehicle_number  -> sizes |N'|, |M|, |K|.
      * Ui[0]      -> DC capacity;  Ui[i+1]      -> retailer i capacity.
      * Iim0[0][m] -> DC inventory; Iim0[i+1][m] -> retailer i inventory of m.
      * him[0][m]  -> DC holding cost; him[i+1][m] -> retailer i holding cost.
      * cij[i][j]  -> distance/cost between nodes 0..N (0 = DC).
      * Qk[k] / bk[k]  -> vehicle k capacity / fixed cost.
      * pim[i][m]  -> value of product m at retailer i  (paper's v_im).
      * vim[i][m]  -> volume / occupancy of product m at retailer i  (paper's O_im).
      * dim[i][m]      -> forecast demand (used in Stage 1).
      * dim_real[i][m] -> actual demand (used in Stage 2).

    Returns a dict with stage-level costs, shortages, decision variables, and
    solver diagnostics. See bottom of function for keys.
    """

    nodes_number = retailer_number + 1
    cij_trans = [[c * lt_cost_multiplier for c in row] for row in cij]
    eps_eq = 1e-5  # tolerance for inventory balance equalities (matches original)

    pen1 = float(shortage_penalty_stage1)
    pen2 = float(shortage_penalty_stage2)

    t0 = time.time()

    # ----------------------------------------------------------------------- #
    # Stage 1: DC -> retailer replenishment & routing                         #
    # ----------------------------------------------------------------------- #
    s1 = Model("BFP_TC_stage1")
    s1.setParam("OutputFlag", 1 if verbose else 0)
    s1.setParam("TimeLimit", time_limit)
    s1.setParam("MIPGap", mip_gap)
    s1.setParam("Threads", threads)
    s1.setParam("NodefileStart", 0.5)

    # Decision variables (same as solve_DB_FP.py, less the Dinkelbach scaffolding).
    rm: Dict[int, "Var"] = {}
    xijk: Dict[int, Dict[int, Dict[int, "Var"]]] = {}
    yik: Dict[int, Dict[int, "Var"]] = {}
    Iim: Dict[int, Dict[int, "Var"]] = {}
    qimk: Dict[int, Dict[int, Dict[int, "Var"]]] = {}
    sdim: Dict[Tuple[int, int], "Var"] = {}

    for i in range(retailer_number):
        for m in range(product_kind):
            sdim[i, m] = s1.addVar(lb=0.0, ub=float(dim[i][m]),
                                   vtype=GRB.CONTINUOUS, name=f"sdim_{i+1}_{m+1}")

    for m in range(product_kind):
        rm[m] = s1.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"r_{m+1}")

    for i in range(nodes_number):
        xijk[i] = {}
        for j in range(nodes_number):
            if i < j:
                xijk[i][j] = {}
                for k in range(vehicle_number):
                    if i == 0:
                        xijk[i][j][k] = s1.addVar(lb=0, ub=2, vtype=GRB.INTEGER,
                                                  name=f"x_{i}_{j}_{k+1}")
                    else:
                        xijk[i][j][k] = s1.addVar(vtype=GRB.BINARY,
                                                  name=f"x_{i}_{j}_{k+1}")

    for i in range(nodes_number):
        yik[i] = {}
        for k in range(vehicle_number):
            yik[i][k] = s1.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k+1}")

    for i in range(nodes_number):
        Iim[i] = {}
        for m in range(product_kind):
            Iim[i][m] = s1.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"I_{i}_{m+1}")

    for i in range(retailer_number):
        qimk[i] = {}
        for m in range(product_kind):
            qimk[i][m] = {}
            for k in range(vehicle_number):
                qimk[i][m][k] = s1.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                          name=f"q_{i+1}_{m+1}_{k+1}")

    # ----- Stage 1 objective: direct total cost -----
    obj1 = LinExpr()
    # routing cost
    for i in range(nodes_number):
        for j in range(nodes_number):
            if i < j:
                for k in range(vehicle_number):
                    obj1.addTerms(cij[i][j], xijk[i][j][k])
    # vehicle fixed cost (only paid if vehicle leaves the DC)
    for k in range(vehicle_number):
        obj1.addTerms(bk[k], yik[0][k])
    # holding cost (DC + retailers)
    for i in range(nodes_number):
        for m in range(product_kind):
            obj1.addTerms(him[i][m], Iim[i][m])
    # forecast shortage penalty:  pen1 * (dim - sdim)  ==  pen1*dim_const  -  pen1*sdim
    forecast_shortage_const = 0.0
    for i in range(retailer_number):
        for m in range(product_kind):
            forecast_shortage_const += pen1 * float(dim[i][m])
            obj1.addTerms(-pen1, sdim[i, m])
    s1.setObjective(obj1 + forecast_shortage_const, GRB.MINIMIZE)

    # ----- Stage 1 constraints (same as solve_DB_FP.py) -----

    # cons 2 (DC inventory balance, kept as equality with the original tolerance form).
    for m in range(product_kind):
        qsum = LinExpr()
        for k in range(vehicle_number):
            for i in range(retailer_number):
                qsum.addTerms(1.0, qimk[i][m][k])
        s1.addConstr(Iim[0][m] - (Iim0[0][m] + rm[m] - qsum) <= eps_eq,
                     name=f"cons2_m{m+1}_ub")
        s1.addConstr(Iim[0][m] - (Iim0[0][m] + rm[m] - qsum) >= -eps_eq,
                     name=f"cons2_m{m+1}_lb")

    # cons 3 (retailer inventory balance with satisfied demand sdim).
    for i in range(retailer_number):
        for m in range(product_kind):
            qsum_k = LinExpr()
            for k in range(vehicle_number):
                qsum_k.addTerms(1.0, qimk[i][m][k])
            s1.addConstr(Iim[i+1][m] - (Iim0[i+1][m] + qsum_k - sdim[i, m]) <= eps_eq,
                         name=f"cons3_i{i+1}_m{m+1}_ub")
            s1.addConstr(Iim[i+1][m] - (Iim0[i+1][m] + qsum_k - sdim[i, m]) >= -eps_eq,
                         name=f"cons3_i{i+1}_m{m+1}_lb")

    # cons 4 (DC capacity).  We use sum_m vim[0][m]*Iim0[0][m] -- fixes a bug in
    # solve_DB_FP.py where the I0 term used vim[0][<last m>] for every product.
    dc_init_volume = sum(vim[0][m] * Iim0[0][m] for m in range(product_kind))
    s1.addConstr(quicksum(vim[0][m] * rm[m] for m in range(product_kind))
                 + dc_init_volume <= Ui[0],
                 name="cons4_dc_capacity")

    # cons 5 (retailer capacity; same volume-summation fix).
    for i in range(retailer_number):
        i0_volume = sum(vim[i][m] * Iim0[i+1][m] for m in range(product_kind))
        s1.addConstr(quicksum(vim[i][m] * qimk[i][m][k]
                              for m in range(product_kind)
                              for k in range(vehicle_number))
                     + i0_volume <= Ui[i+1],
                     name=f"cons5_i{i+1}")

    # cons 6 (link delivery to vehicle visit).
    for i in range(retailer_number):
        for k in range(vehicle_number):
            s1.addConstr(quicksum(vim[i][m] * qimk[i][m][k]
                                  for m in range(product_kind))
                         <= Ui[i+1] * yik[i+1][k],
                         name=f"cons6_i{i+1}_k{k+1}")

    # cons 7 (vehicle capacity).
    for k in range(vehicle_number):
        s1.addConstr(quicksum(vim[i][m] * qimk[i][m][k]
                              for i in range(retailer_number)
                              for m in range(product_kind))
                     <= Qk[k] * yik[0][k],
                     name=f"cons7_k{k+1}")

    # cons 8 (degree / flow consistency).
    for i in range(nodes_number):
        for k in range(vehicle_number):
            in_out = LinExpr()
            for j in range(nodes_number):
                if i < j:
                    in_out.addTerms(1.0, xijk[i][j][k])
                if j < i:
                    in_out.addTerms(1.0, xijk[j][i][k])
            s1.addConstr(in_out == 2 * yik[i][k], name=f"cons8_i{i}_k{k+1}")

    # cons 9 (sub-tour elimination via subset enumeration -- same as the original).
    for S in _power_set_binary(list(range(retailer_number))):
        if not S:
            continue
        for g in S:
            for k in range(vehicle_number):
                arc_sum = LinExpr()
                yik_sum = LinExpr()
                for i in S:
                    yik_sum.addTerms(1.0, yik[i+1][k])
                    for j in S:
                        if i < j:
                            arc_sum.addTerms(1.0, xijk[i+1][j+1][k])
                s1.addConstr(arc_sum <= yik_sum - yik[g+1][k],
                             name=f"cons9_S{tuple(S)}_g{g+1}_k{k+1}")

    s1.optimize()

    if s1.SolCount == 0:
        # No feasible solution found.
        return {
            "status": int(s1.Status),
            "stage1_cost": float("nan"),
            "lt_cost": float("nan"),
            "total_cost": float("nan"),
            "stage1_routing_cost": float("nan"),
            "stage1_holding_cost": float("nan"),
            "stage1_vehicle_fixed_cost": float("nan"),
            "stage1_forecast_shortage": float("nan"),
            "stage2_lt_transport_cost": float("nan"),
            "stage2_holding_cost": float("nan"),
            "stage2_remaining_shortage": float("nan"),
            "final_stockout_units": float("nan"),
            "stockout_rate": float("nan"),
            "mip_gap": float("nan"),
            "runtime_seconds": time.time() - t0,
            "post_lt_inventory": None,
            "lt_moves": [],
            "qimk": None,
            "infeasible": True,
        }

    # Extract Stage 1 numeric solution.
    qimk_val = [[[qimk[i][m][k].X for k in range(vehicle_number)]
                 for m in range(product_kind)] for i in range(retailer_number)]
    Iim_val = [[Iim[i][m].X for m in range(product_kind)]
               for i in range(nodes_number)]
    rm_val = [rm[m].X for m in range(product_kind)]
    sdim_val = [[sdim[i, m].X for m in range(product_kind)]
                for i in range(retailer_number)]
    yik_val = [[yik[i][k].X for k in range(vehicle_number)]
               for i in range(nodes_number)]
    xijk_val: Dict[Tuple[int, int, int], float] = {}
    for i in range(nodes_number):
        for j in range(nodes_number):
            if i < j:
                for k in range(vehicle_number):
                    xijk_val[(i, j, k)] = xijk[i][j][k].X

    # Cost breakdown for Stage 1.
    s1_routing = sum(cij[i][j] * xijk_val[(i, j, k)]
                     for i in range(nodes_number)
                     for j in range(nodes_number) if i < j
                     for k in range(vehicle_number))
    s1_vehicle_fixed = sum(bk[k] * yik_val[0][k] for k in range(vehicle_number))
    s1_holding = sum(him[i][m] * Iim_val[i][m]
                     for i in range(nodes_number)
                     for m in range(product_kind))
    s1_forecast_shortage_units = sum(max(0.0, dim[i][m] - sdim_val[i][m])
                                     for i in range(retailer_number)
                                     for m in range(product_kind))
    s1_forecast_shortage_cost = pen1 * s1_forecast_shortage_units
    stage1_cost = s1_routing + s1_vehicle_fixed + s1_holding + s1_forecast_shortage_cost
    stage1_mip_gap = float(s1.MIPGap) if s1.IsMIP else 0.0
    stage1_status = int(s1.Status)

    # ----------------------------------------------------------------------- #
    # Stage 2: lateral transshipment with actual demand revealed              #
    # ----------------------------------------------------------------------- #
    # Pre-compute Stage-1 leftover / shortage at every retailer.
    Irim = {}    # signed leftover inventory after Stage 1 - actual demand
    ksi = {}     # surplus (>=0)
    miu = {}     # pre-LT shortage (>=0)
    for i in range(retailer_number):
        for m in range(product_kind):
            qsum = sum(qimk_val[i][m][k] for k in range(vehicle_number))
            balance = Iim0[i+1][m] + qsum - dim_real[i][m]
            Irim[i, m] = balance
            if balance >= 0:
                ksi[i, m] = balance
                miu[i, m] = 0.0
            else:
                ksi[i, m] = 0.0
                miu[i, m] = -balance

    # If no surplus and no shortage anywhere, skip the Stage-2 MIP.
    total_surplus = sum(ksi.values())
    total_shortage = sum(miu.values())
    if total_surplus < 1e-9 or total_shortage < 1e-9:
        # No feasible LT can change anything: every shortage stays.
        post_inv = [[max(0.0, Irim[i, m]) for m in range(product_kind)]
                    for i in range(retailer_number)]
        final_stockout = sum(miu.values())
        total_demand = sum(dim_real[i][m] for i in range(retailer_number)
                           for m in range(product_kind))
        stockout_rate = final_stockout / total_demand if total_demand > 0 else 0.0
        runtime = time.time() - t0
        return {
            "status": stage1_status,
            "stage1_cost": stage1_cost,
            "lt_cost": 0.0,
            "total_cost": stage1_cost,
            "stage1_routing_cost": s1_routing,
            "stage1_holding_cost": s1_holding,
            "stage1_vehicle_fixed_cost": s1_vehicle_fixed,
            "stage1_forecast_shortage": s1_forecast_shortage_cost,
            "stage1_forecast_shortage_units": s1_forecast_shortage_units,
            "stage2_lt_transport_cost": 0.0,
            "stage2_holding_cost": 0.0,
            "stage2_remaining_shortage": pen2 * final_stockout,
            "stage2_remaining_shortage_units": final_stockout,
            "final_stockout_units": final_stockout,
            "stockout_rate": stockout_rate,
            "mip_gap": stage1_mip_gap,
            "runtime_seconds": runtime,
            "post_lt_inventory": post_inv,
            "post_lt_dc_inventory": [Iim_val[0][m] for m in range(product_kind)],
            "lt_moves": [],
            "qimk": qimk_val,
            "rm": rm_val,
            "infeasible": False,
        }

    s2 = Model("BFP_TC_stage2")
    s2.setParam("OutputFlag", 1 if verbose else 0)
    s2.setParam("TimeLimit", time_limit)
    s2.setParam("MIPGap", mip_gap)
    s2.setParam("Threads", threads)
    s2.setParam("NodefileStart", 0.5)

    # LT decision variables (same shape as in the original Stage-2 check model).
    wijm: Dict[int, Dict[int, Dict[int, "Var"]]] = {}
    for i in range(retailer_number):
        wijm[i] = {}
        for j in range(retailer_number):
            if i != j:
                wijm[i][j] = {}
                for m in range(product_kind):
                    wijm[i][j][m] = s2.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                              name=f"w_{i+1}_{j+1}_{m+1}")
    zij: Dict[int, Dict[int, "Var"]] = {}
    for i in range(retailer_number):
        zij[i] = {}
        for j in range(retailer_number):
            if i != j:
                zij[i][j] = s2.addVar(vtype=GRB.BINARY, name=f"z_{i+1}_{j+1}")

    # Post-LT physical inventory at retailer i for product m (>= 0).
    post_inv_var: Dict[Tuple[int, int], "Var"] = {}
    # Remaining shortage at retailer i product m (>= 0).
    rem_short: Dict[Tuple[int, int], "Var"] = {}
    for i in range(retailer_number):
        for m in range(product_kind):
            post_inv_var[i, m] = s2.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                           name=f"postInv_{i+1}_{m+1}")
            rem_short[i, m] = s2.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                        name=f"remShort_{i+1}_{m+1}")

    # ----- Stage 2 objective: direct total cost -----
    obj2 = LinExpr()
    # LT transportation cost (third-party fleet, paid per used arc).
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                obj2.addTerms(cij_trans[i][j], zij[i][j])
    # Post-LT holding cost.
    for i in range(retailer_number):
        for m in range(product_kind):
            obj2.addTerms(him[i+1][m], post_inv_var[i, m])
    # Remaining stockout penalty.
    for i in range(retailer_number):
        for m in range(product_kind):
            obj2.addTerms(pen2, rem_short[i, m])
    s2.setObjective(obj2, GRB.MINIMIZE)

    # ----- Stage 2 constraints -----

    # (i) outbound LT bounded by surplus (cons 18 in solve_DB_FP).
    for i in range(retailer_number):
        for m in range(product_kind):
            outflow = LinExpr()
            for j in range(retailer_number):
                if i != j:
                    outflow.addTerms(1.0, wijm[i][j][m])
            s2.addConstr(outflow <= ksi[i, m], name=f"out_le_surplus_i{i+1}_m{m+1}")

    # (ii) inbound LT bounded by capacity * (1 if shortage > 0 else 0)
    #      (cons 19 in solve_DB_FP -- using pre-LT shortage `miu` as the cap).
    for i in range(retailer_number):
        for m in range(product_kind):
            inflow = LinExpr()
            for j in range(retailer_number):
                if i != j:
                    inflow.addTerms(1.0, wijm[j][i][m])
            s2.addConstr(inflow <= Ui[i+1] * miu[i, m],
                         name=f"in_le_capacity_when_short_i{i+1}_m{m+1}")

    # (iii) link LT quantity to direction binary z_ij  (cons 17).
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                pair_qty = LinExpr()
                pair_cap = sum(ksi[i, m] for m in range(product_kind))
                for m in range(product_kind):
                    pair_qty.addTerms(1.0, wijm[i][j][m])
                s2.addConstr(pair_qty <= pair_cap * zij[i][j],
                             name=f"w_le_z_i{i+1}_j{j+1}")

    # (iv) z gating by total surplus / shortage at the endpoints (cons 20-21).
    for i in range(retailer_number):
        out_z = LinExpr()
        in_z = LinExpr()
        for j in range(retailer_number):
            if i != j:
                out_z.addTerms(1.0, zij[i][j])
                in_z.addTerms(1.0, zij[j][i])
        ksi_total = sum(ksi[i, m] for m in range(product_kind))
        miu_total = sum(miu[i, m] for m in range(product_kind))
        s2.addConstr(out_z <= ksi_total, name=f"z_out_le_surplus_i{i+1}")
        s2.addConstr(in_z <= miu_total, name=f"z_in_le_shortage_i{i+1}")

    # (v) post-LT capacity feasibility (cons 22 in solve_DB_FP).
    for i in range(retailer_number):
        cap_lhs = LinExpr()
        for m in range(product_kind):
            cap_lhs.addConstant(Irim[i, m] * vim[i][m])
            for j in range(retailer_number):
                if i != j:
                    cap_lhs.addTerms(-vim[i][m], wijm[i][j][m])
                    cap_lhs.addTerms(vim[i][m], wijm[j][i][m])
        s2.addConstr(cap_lhs <= Ui[i+1], name=f"post_lt_capacity_i{i+1}")

    # (vi) post-LT inventory definition:  post_inv >= Irim + inbound - outbound
    #      Together with post_inv >= 0 this lets the holding cost charge the
    #      non-negative leftover.
    for i in range(retailer_number):
        for m in range(product_kind):
            inflow = LinExpr()
            outflow = LinExpr()
            for j in range(retailer_number):
                if i != j:
                    inflow.addTerms(1.0, wijm[j][i][m])
                    outflow.addTerms(1.0, wijm[i][j][m])
            s2.addConstr(post_inv_var[i, m] >= Irim[i, m] + inflow - outflow,
                         name=f"post_inv_def_i{i+1}_m{m+1}")

    # (vii) remaining shortage definition:
    #       rem_short >= miu - inbound,   rem_short >= 0  (the latter via lb=0)
    for i in range(retailer_number):
        for m in range(product_kind):
            inflow = LinExpr()
            for j in range(retailer_number):
                if i != j:
                    inflow.addTerms(1.0, wijm[j][i][m])
            s2.addConstr(rem_short[i, m] >= miu[i, m] - inflow,
                         name=f"rem_short_def_i{i+1}_m{m+1}")

    s2.optimize()

    if s2.SolCount == 0:
        runtime = time.time() - t0
        return {
            "status": int(s2.Status),
            "stage1_cost": stage1_cost,
            "lt_cost": float("nan"),
            "total_cost": float("nan"),
            "stage1_routing_cost": s1_routing,
            "stage1_holding_cost": s1_holding,
            "stage1_vehicle_fixed_cost": s1_vehicle_fixed,
            "stage1_forecast_shortage": s1_forecast_shortage_cost,
            "stage1_forecast_shortage_units": s1_forecast_shortage_units,
            "stage2_lt_transport_cost": float("nan"),
            "stage2_holding_cost": float("nan"),
            "stage2_remaining_shortage": float("nan"),
            "stage2_remaining_shortage_units": float("nan"),
            "final_stockout_units": float("nan"),
            "stockout_rate": float("nan"),
            "mip_gap": stage1_mip_gap,
            "runtime_seconds": runtime,
            "post_lt_inventory": None,
            "post_lt_dc_inventory": [Iim_val[0][m] for m in range(product_kind)],
            "lt_moves": [],
            "qimk": qimk_val,
            "rm": rm_val,
            "infeasible": True,
        }

    # Extract Stage 2 numeric solution.
    wijm_val: Dict[Tuple[int, int, int], float] = {}
    zij_val: Dict[Tuple[int, int], float] = {}
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                zij_val[(i, j)] = zij[i][j].X
                for m in range(product_kind):
                    wijm_val[(i, j, m)] = wijm[i][j][m].X

    post_inv_val = [[post_inv_var[i, m].X for m in range(product_kind)]
                    for i in range(retailer_number)]
    rem_short_val = [[rem_short[i, m].X for m in range(product_kind)]
                     for i in range(retailer_number)]

    s2_lt_transport = sum(cij_trans[i][j] * zij_val[(i, j)]
                          for i in range(retailer_number)
                          for j in range(retailer_number) if i != j)
    s2_holding = sum(him[i+1][m] * post_inv_val[i][m]
                     for i in range(retailer_number)
                     for m in range(product_kind))
    final_stockout_units = sum(rem_short_val[i][m]
                               for i in range(retailer_number)
                               for m in range(product_kind))
    s2_rem_short_cost = pen2 * final_stockout_units
    lt_cost = s2_lt_transport + s2_holding + s2_rem_short_cost

    total_demand = sum(dim_real[i][m] for i in range(retailer_number)
                       for m in range(product_kind))
    stockout_rate = final_stockout_units / total_demand if total_demand > 0 else 0.0

    # Lateral transshipment moves (for diagnostics).
    lt_moves: List[Dict] = []
    for (i, j, m), qty in wijm_val.items():
        if qty > 1e-6:
            lt_moves.append({
                "from_store_idx": i,
                "to_store_idx": j,
                "product_idx": m,
                "qty": qty,
                "unit_cost": cij_trans[i][j] / max(1.0, sum(1 for _m in range(product_kind))),
            })

    runtime = time.time() - t0
    stage2_mip_gap = float(s2.MIPGap) if s2.IsMIP else 0.0
    combined_gap = max(stage1_mip_gap, stage2_mip_gap)

    return {
        "status": int(s2.Status),
        "stage1_cost": stage1_cost,
        "lt_cost": lt_cost,
        "total_cost": stage1_cost + lt_cost,
        "stage1_routing_cost": s1_routing,
        "stage1_holding_cost": s1_holding,
        "stage1_vehicle_fixed_cost": s1_vehicle_fixed,
        "stage1_forecast_shortage": s1_forecast_shortage_cost,
        "stage1_forecast_shortage_units": s1_forecast_shortage_units,
        "stage2_lt_transport_cost": s2_lt_transport,
        "stage2_holding_cost": s2_holding,
        "stage2_remaining_shortage": s2_rem_short_cost,
        "stage2_remaining_shortage_units": final_stockout_units,
        "final_stockout_units": final_stockout_units,
        "stockout_rate": stockout_rate,
        "mip_gap": combined_gap,
        "runtime_seconds": runtime,
        "post_lt_inventory": post_inv_val,
        "post_lt_dc_inventory": [Iim_val[0][m] for m in range(product_kind)],
        "lt_moves": lt_moves,
        "qimk": qimk_val,
        "rm": rm_val,
        "infeasible": False,
    }


# --------------------------------------------------------------------------- #
# Script entry point (mirrors solve_DB_FP.py CLI shape)                       #
# --------------------------------------------------------------------------- #

def _main_from_dat(dat_path: str) -> None:
    if parse_data_file is None:
        raise RuntimeError("revise_data.parse_data_file is unavailable; cannot run as script")

    data = parse_data_file(dat_path)
    res = solve_man_bfp_tc(
        retailer_number=data["num_customers"],
        product_kind=data["num_products"],
        vehicle_number=data["num_vehicles"],
        Ui=data["Ui"],
        cij=data["cij"],
        pim=data["pim"],
        Qk=data["Qk"],
        bk=data["bk"],
        vim=data["vim"],
        him=data["him"],
        Iim0=data["Iim0"],
        dim_real=data["dim_actual"],
        dim=data["dim_predict"],
    )
    print("method: Man-BFP-TC")
    print(f"stage1_cost: {res['stage1_cost']:.4f}")
    print(f"lt_cost: {res['lt_cost']:.4f}")
    print(f"total_cost: {res['total_cost']:.4f}")
    print(f"final_stockout_units: {res['final_stockout_units']:.4f}")
    print(f"stockout_rate: {res['stockout_rate']:.4f}")
    print(f"runtime_seconds: {res['runtime_seconds']:.4f}")
    print(f"mip_gap: {res['mip_gap']:.4f}")
    print(f"status: {res['status']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python solve_BFP_TC.py <dat_file>")
    _main_from_dat(sys.argv[1])
