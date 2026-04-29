"""
solve_TSRFP_TC.py — Man-Joint-TC Oracle Benchmark.

This solves a JOINT two-stage problem where both Stage 1 (routing) and Stage 2
(lateral transshipment) are optimized simultaneously.  Because Stage 1 variables
share the same Gurobi model as Stage 2, the routing decisions can "see" the
realized shocked demand indirectly — making this an oracle / full-information
benchmark, NOT a true TSRFP-TC.

Labelled internally as: "Man-Joint-TC"

Structural alignment with Man-BFP-TC (solve_BFP_TC.py):
  - Exact same constraint set for Stage 1 (cons 2–9 from solve_DB_FP.py)
  - Exact same Stage 2 LT formulation (wijm per unit, zij binary route gate)
  - Same cost components: routing, vehicle fixed, holding, shortage, LT
  - Stage 1 holding/shortage uses dim_forecast (sdim approach)
  - Stage 2 pre-LT balance uses realized demand (ksi/miu as LP variables)
  - LT transport cost = cij * 1.2 per route opened (zij binary), same as BFP-TC

Difference from Man-BFP-TC:
  - Single Gurobi model (joint) instead of two sequential models
  - Stage 1 can indirectly anticipate Stage 2 demand shock → Oracle semantics

Usage:
    from solve_TSRFP_TC import solve_joint_tc
    res = solve_joint_tc(N, M, K, Ui, cij, pim, Qk, bk, vim, him, Iim0,
                         dim_real=..., dim=..., ...)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

from gurobipy import GRB, LinExpr, Model, quicksum


# ============================================================================
# Helper — subtour elimination subsets (mirror of BFP-TC)
# ============================================================================

def _power_set_binary(node_list: Sequence[int]) -> List[List[int]]:
    n = len(node_list)
    out: List[List[int]] = []
    for i in range(2 ** n):
        sub: List[int] = []
        for j in range(n):
            if (i >> j) % 2:
                sub.append(node_list[j])
        out.append(sub)
    return out


# ============================================================================
# Main solver
# ============================================================================

def solve_joint_tc(
    # Dimensions
    retailer_number: int,
    product_kind: int,
    vehicle_number: int,
    # Topology and distances
    Ui: Sequence[float],               # node capacity [0=DC, 1..N=stores]
    cij: Sequence[Sequence[float]],    # distance matrix (N+1)×(N+1)
    pim: Sequence[Sequence[float]],    # product value (API compat, unused in TC obj)
    Qk: Sequence[float],               # vehicle capacity per vehicle
    bk: Sequence[float],               # vehicle fixed cost per vehicle
    vim: Sequence[Sequence[float]],    # volume factor [0=DC or store-0..N-1 per BFP-TC convention]
    him: Sequence[Sequence[float]],    # holding cost [0=DC, 1..N=stores]
    Iim0: Sequence[Sequence[float]],   # initial inventory [0=DC, 1..N=stores] × M
    # Demand
    dim: Sequence[Sequence[float]],          # N×M forecast demand (Stage 1 planning)
    dim_real: Sequence[Sequence[float]],     # N×M realized demand (Stage 2 evaluation)
    # Cost parameters
    shortage_penalty_stage1: float = 2.0,
    shortage_penalty_stage2: float = 2.0,
    lt_cost_multiplier: float = 1.2,         # LT cost = cij * multiplier per route opened
    # Solver settings
    time_limit: int = 600,
    mip_gap: float = 0.01,
    threads: int = 4,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve the joint two-stage Oracle benchmark (Man-Joint-TC).

    Stage 1 decisions (routing, deliveries) and Stage 2 decisions (LT) are
    jointly optimized in one Gurobi model.  Stage 1 objective is based on
    forecast demand (dim); Stage 2 is evaluated on realized demand (dim_real).
    Because they share variables, Stage 1 can anticipate Stage 2 — Oracle.

    Returns the same dict schema as solve_man_bfp_tc for drop-in compatibility.
    """
    N = int(retailer_number)
    M = int(product_kind)
    K = int(vehicle_number)
    nodes_number = N + 1              # node 0 = DC, 1..N = stores
    pen1 = float(shortage_penalty_stage1)
    pen2 = float(shortage_penalty_stage2)
    cij_trans = [[cij[i][j] * lt_cost_multiplier for j in range(nodes_number)]
                 for i in range(nodes_number)]
    eps_eq = 1e-5   # tolerance for inventory balance equalities (BFP-TC convention)
    t0 = time.time()

    m = Model("Man-Joint-TC")
    m.setParam("OutputFlag", 1 if verbose else 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPGap", mip_gap)
    m.setParam("Threads", threads)
    m.setParam("NodefileStart", 0.5)

    # ======================================================================
    # STAGE 1 VARIABLES  (same naming as solve_BFP_TC.py)
    # ======================================================================

    # sdim[i,m]: satisfied demand at store i for product m (≤ dim_forecast)
    sdim: Dict[Tuple[int, int], Any] = {}
    for i in range(N):
        for m_idx in range(M):
            sdim[i, m_idx] = m.addVar(
                lb=0.0, ub=float(dim[i][m_idx]),
                vtype=GRB.CONTINUOUS, name=f"sdim_{i+1}_{m_idx+1}"
            )

    # rm[m]: replenishment ordered by DC from supplier
    rm: Dict[int, Any] = {}
    for m_idx in range(M):
        rm[m_idx] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"r_{m_idx+1}")

    # xijk: edge binary (DC edges integer 0-2, store edges binary)
    xijk: Dict[int, Dict[int, Dict[int, Any]]] = {}
    for i in range(nodes_number):
        xijk[i] = {}
        for j in range(nodes_number):
            if i < j:
                xijk[i][j] = {}
                for k in range(K):
                    if i == 0:
                        xijk[i][j][k] = m.addVar(lb=0, ub=2, vtype=GRB.INTEGER,
                                                  name=f"x_{i}_{j}_{k+1}")
                    else:
                        xijk[i][j][k] = m.addVar(vtype=GRB.BINARY,
                                                  name=f"x_{i}_{j}_{k+1}")

    # yik: vehicle k visits node i
    yik: Dict[int, Dict[int, Any]] = {}
    for i in range(nodes_number):
        yik[i] = {}
        for k in range(K):
            yik[i][k] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k+1}")

    # Iim: Stage 1 end-of-period inventory (forecast-demand based)
    Iim: Dict[int, Dict[int, Any]] = {}
    for i in range(nodes_number):
        Iim[i] = {}
        for m_idx in range(M):
            Iim[i][m_idx] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                     name=f"I_s1_{i}_{m_idx+1}")

    # qimk: quantity of product m delivered to store i by vehicle k
    qimk: Dict[int, Dict[int, Dict[int, Any]]] = {}
    for i in range(N):
        qimk[i] = {}
        for m_idx in range(M):
            qimk[i][m_idx] = {}
            for k in range(K):
                qimk[i][m_idx][k] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                             name=f"q_{i+1}_{m_idx+1}_{k+1}")

    # ======================================================================
    # STAGE 2 VARIABLES  (same naming as solve_BFP_TC.py Stage 2)
    # ======================================================================

    # ksi[i,m]: surplus at store i (= max(0, Iim0+deliveries-dim_real))
    # miu[i,m]: pre-LT shortage at store i (= max(0, dim_real-Iim0-deliveries))
    ksi: Dict[Tuple[int, int], Any] = {}
    miu: Dict[Tuple[int, int], Any] = {}
    for i in range(N):
        for m_idx in range(M):
            ksi[i, m_idx] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                     name=f"ksi_{i+1}_{m_idx+1}")
            miu[i, m_idx] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                     name=f"miu_{i+1}_{m_idx+1}")

    # wijm: LT quantity from store i to store j for product m
    # zij:  binary route gate for LT arc (i→j)
    wijm: Dict[int, Dict[int, Dict[int, Any]]] = {}
    zij: Dict[int, Dict[int, Any]] = {}
    for i in range(N):
        wijm[i] = {}
        zij[i] = {}
        for j in range(N):
            if i != j:
                wijm[i][j] = {}
                for m_idx in range(M):
                    wijm[i][j][m_idx] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                                 name=f"w_{i+1}_{j+1}_{m_idx+1}")
                zij[i][j] = m.addVar(vtype=GRB.BINARY, name=f"z_{i+1}_{j+1}")

    # post_inv[i,m]: inventory at store i after LT (and after dim_real)
    post_inv: Dict[Tuple[int, int], Any] = {}
    rem_short: Dict[Tuple[int, int], Any] = {}
    for i in range(N):
        for m_idx in range(M):
            post_inv[i, m_idx] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                          name=f"postInv_{i+1}_{m_idx+1}")
            rem_short[i, m_idx] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                           name=f"remShort_{i+1}_{m_idx+1}")

    # ======================================================================
    # OBJECTIVE: total cost (both stages)
    # ======================================================================
    obj = LinExpr()

    # Stage 1: routing cost
    for i in range(nodes_number):
        for j in range(nodes_number):
            if i < j:
                for k in range(K):
                    obj.addTerms(cij[i][j], xijk[i][j][k])

    # Stage 1: vehicle fixed cost
    for k in range(K):
        obj.addTerms(bk[k], yik[0][k])

    # Stage 1: holding cost (DC + stores, on forecast-demand-based Iim)
    for i in range(nodes_number):
        for m_idx in range(M):
            obj.addTerms(him[i][m_idx], Iim[i][m_idx])

    # Stage 1: forecast shortage penalty  (pen1 * (dim - sdim))
    # = pen1 * sum(dim) - pen1 * sum(sdim)   → constant + variable terms
    s1_shortage_const = sum(pen1 * float(dim[i][m_idx])
                            for i in range(N) for m_idx in range(M))
    for i in range(N):
        for m_idx in range(M):
            obj.addTerms(-pen1, sdim[i, m_idx])

    # Stage 2: LT transport cost (binary route gate, not per-unit — same as BFP-TC)
    for i in range(N):
        for j in range(N):
            if i != j:
                obj.addTerms(cij_trans[i+1][j+1], zij[i][j])

    # Stage 2: post-LT holding cost
    for i in range(N):
        for m_idx in range(M):
            obj.addTerms(him[i+1][m_idx], post_inv[i, m_idx])

    # Stage 2: remaining shortage penalty
    for i in range(N):
        for m_idx in range(M):
            obj.addTerms(pen2, rem_short[i, m_idx])

    m.setObjective(obj + s1_shortage_const, GRB.MINIMIZE)

    # ======================================================================
    # STAGE 1 CONSTRAINTS (cons 2–9 from solve_DB_FP.py, identical)
    # ======================================================================

    # cons 2: DC inventory balance
    for m_idx in range(M):
        qsum = quicksum(qimk[i][m_idx][k]
                        for i in range(N) for k in range(K))
        m.addConstr(Iim[0][m_idx] - (Iim0[0][m_idx] + rm[m_idx] - qsum) <= eps_eq,
                    name=f"cons2_m{m_idx+1}_ub")
        m.addConstr(Iim[0][m_idx] - (Iim0[0][m_idx] + rm[m_idx] - qsum) >= -eps_eq,
                    name=f"cons2_m{m_idx+1}_lb")

    # cons 3: store inventory balance (uses sdim / forecast demand)
    for i in range(N):
        for m_idx in range(M):
            qsum_k = quicksum(qimk[i][m_idx][k] for k in range(K))
            m.addConstr(
                Iim[i+1][m_idx] - (Iim0[i+1][m_idx] + qsum_k - sdim[i, m_idx]) <= eps_eq,
                name=f"cons3_i{i+1}_m{m_idx+1}_ub"
            )
            m.addConstr(
                Iim[i+1][m_idx] - (Iim0[i+1][m_idx] + qsum_k - sdim[i, m_idx]) >= -eps_eq,
                name=f"cons3_i{i+1}_m{m_idx+1}_lb"
            )

    # cons 4: DC capacity
    dc_init_vol = sum(vim[0][m_idx] * Iim0[0][m_idx] for m_idx in range(M))
    m.addConstr(
        quicksum(vim[0][m_idx] * rm[m_idx] for m_idx in range(M)) + dc_init_vol <= Ui[0],
        name="cons4_dc_cap"
    )

    # cons 5: store capacity (initial + delivered ≤ Ui)
    for i in range(N):
        i0_vol = sum(vim[i][m_idx] * Iim0[i+1][m_idx] for m_idx in range(M))
        m.addConstr(
            quicksum(vim[i][m_idx] * qimk[i][m_idx][k]
                     for m_idx in range(M) for k in range(K))
            + i0_vol <= Ui[i+1],
            name=f"cons5_i{i+1}"
        )

    # cons 6: link delivery to vehicle visit
    for i in range(N):
        for k in range(K):
            m.addConstr(
                quicksum(vim[i][m_idx] * qimk[i][m_idx][k] for m_idx in range(M))
                <= Ui[i+1] * yik[i+1][k],
                name=f"cons6_i{i+1}_k{k+1}"
            )

    # cons 7: vehicle capacity
    for k in range(K):
        m.addConstr(
            quicksum(vim[i][m_idx] * qimk[i][m_idx][k]
                     for i in range(N) for m_idx in range(M))
            <= Qk[k] * yik[0][k],
            name=f"cons7_k{k+1}"
        )

    # cons 8: degree / flow consistency
    for i in range(nodes_number):
        for k in range(K):
            in_out = LinExpr()
            for j in range(nodes_number):
                if i < j:
                    in_out.addTerms(1.0, xijk[i][j][k])
                if j < i:
                    in_out.addTerms(1.0, xijk[j][i][k])
            m.addConstr(in_out == 2 * yik[i][k], name=f"cons8_i{i}_k{k+1}")

    # cons 9: subtour elimination
    for S in _power_set_binary(list(range(N))):
        if not S:
            continue
        for g in S:
            for k in range(K):
                arc_sum = LinExpr()
                yik_sum = LinExpr()
                for i in S:
                    yik_sum.addTerms(1.0, yik[i+1][k])
                    for j in S:
                        if i < j:
                            arc_sum.addTerms(1.0, xijk[i+1][j+1][k])
                m.addConstr(arc_sum <= yik_sum - yik[g+1][k],
                            name=f"cons9_S{tuple(S)}_g{g+1}_k{k+1}")

    # ======================================================================
    # STAGE 2 CONSTRAINTS (identical to BFP-TC Stage 2)
    # ======================================================================

    # Link ksi/miu to deliveries and realized demand:
    #   ksi[i,m] - miu[i,m] = Iim0[i+1][m] + sum_k qimk[i][m][k] - dim_real[i][m]
    # With ksi,miu >= 0 (via lb) and both in the objective with positive cost,
    # the optimizer ensures complementarity: exactly one of {ksi, miu} is positive.
    for i in range(N):
        for m_idx in range(M):
            deliveries = quicksum(qimk[i][m_idx][k] for k in range(K))
            m.addConstr(
                ksi[i, m_idx] - miu[i, m_idx]
                == Iim0[i+1][m_idx] + deliveries - float(dim_real[i][m_idx]),
                name=f"ksi_miu_balance_i{i+1}_m{m_idx+1}"
            )

    # Stage 2 LT: outbound bounded by surplus ksi  (cons 18 from solve_DB_FP)
    for i in range(N):
        for m_idx in range(M):
            outflow = quicksum(wijm[i][j][m_idx] for j in range(N) if j != i)
            m.addConstr(outflow <= ksi[i, m_idx],
                        name=f"lt_out_le_ksi_i{i+1}_m{m_idx+1}")

    # Stage 2 LT: inbound bounded by shortage miu  (cons 19 — simplified to miu)
    for i in range(N):
        for m_idx in range(M):
            inflow = quicksum(wijm[j][i][m_idx] for j in range(N) if j != i)
            m.addConstr(inflow <= miu[i, m_idx],
                        name=f"lt_in_le_miu_i{i+1}_m{m_idx+1}")

    # Stage 2 LT: link quantity to route gate (cons 17)
    for i in range(N):
        for j in range(N):
            if i != j:
                pair_qty = quicksum(wijm[i][j][m_idx] for m_idx in range(M))
                # Use a BigM of sum of all ksi (or just sum_m ksi[i,m])
                # To keep it LP-valid we use an upper bound approximation:
                # ksi_total is bounded by sum of all Iim0 + deliveries
                # We use a safe bigM = sum_m Ui[i+1] / vim[i][m] (generous)
                bigM = max(1.0, sum(Ui[i+1] for _ in range(M)))
                m.addConstr(pair_qty <= bigM * zij[i][j],
                            name=f"lt_qty_le_gate_i{i+1}_j{j+1}")

    # Gating by surplus/shortage totals (cons 20–21)
    for i in range(N):
        out_z = quicksum(zij[i][j] for j in range(N) if j != i)
        in_z  = quicksum(zij[j][i] for j in range(N) if j != i)
        ksi_total = quicksum(ksi[i, m_idx] for m_idx in range(M))
        miu_total = quicksum(miu[i, m_idx] for m_idx in range(M))
        m.addConstr(out_z <= ksi_total, name=f"z_out_le_ksi_i{i+1}")
        m.addConstr(in_z <= miu_total,  name=f"z_in_le_miu_i{i+1}")

    # post-LT capacity (cons 22)
    for i in range(N):
        cap_lhs = LinExpr()
        for m_idx in range(M):
            cap_lhs.addTerms(vim[i][m_idx], ksi[i, m_idx])
            for j in range(N):
                if i != j:
                    cap_lhs.addTerms(-vim[i][m_idx], wijm[i][j][m_idx])
                    cap_lhs.addTerms( vim[i][m_idx], wijm[j][i][m_idx])
        m.addConstr(cap_lhs <= Ui[i+1], name=f"post_lt_cap_i{i+1}")

    # post_inv definition (cons vi from BFP-TC)
    for i in range(N):
        for m_idx in range(M):
            inflow  = quicksum(wijm[j][i][m_idx] for j in range(N) if j != i)
            outflow = quicksum(wijm[i][j][m_idx] for j in range(N) if j != i)
            m.addConstr(
                post_inv[i, m_idx] >= ksi[i, m_idx] + inflow - outflow,
                name=f"post_inv_def_i{i+1}_m{m_idx+1}"
            )

    # rem_short definition (cons vii from BFP-TC)
    for i in range(N):
        for m_idx in range(M):
            inflow = quicksum(wijm[j][i][m_idx] for j in range(N) if j != i)
            m.addConstr(
                rem_short[i, m_idx] >= miu[i, m_idx] - inflow,
                name=f"rem_short_def_i{i+1}_m{m_idx+1}"
            )

    # ======================================================================
    # SOLVE
    # ======================================================================
    m.optimize()

    runtime = time.time() - t0

    # ======================================================================
    # INFEASIBLE / NO SOLUTION
    # ======================================================================
    if m.SolCount == 0:
        return {
            "status": int(m.Status),
            "infeasible": True,
            "stage1_cost": float("nan"),
            "lt_cost": float("nan"),
            "total_cost": float("nan"),
            "stage1_routing_cost": float("nan"),
            "stage1_holding_cost": float("nan"),
            "stage1_vehicle_fixed_cost": float("nan"),
            "stage1_forecast_shortage": float("nan"),
            "stage1_forecast_shortage_units": float("nan"),
            "stage2_lt_transport_cost": float("nan"),
            "stage2_holding_cost": float("nan"),
            "stage2_remaining_shortage": float("nan"),
            "stage2_remaining_shortage_units": float("nan"),
            "final_stockout_units": float("nan"),
            "stockout_rate": float("nan"),
            "mip_gap": float("nan"),
            "runtime_seconds": runtime,
            "post_lt_inventory": None,
            "post_lt_dc_inventory": None,
            "lt_moves": [],
            "ccg_iterations": 0,
            "final_robust_gap": float("nan"),
        }

    # ======================================================================
    # EXTRACT SOLUTION
    # ======================================================================

    # --- Stage 1 ---
    xijk_val: Dict[Tuple[int, int, int], float] = {}
    for i in range(nodes_number):
        for j in range(nodes_number):
            if i < j:
                for k in range(K):
                    xijk_val[i, j, k] = xijk[i][j][k].X

    yik_val = [[yik[i][k].X for k in range(K)] for i in range(nodes_number)]
    qimk_val = [[[qimk[i][m_idx][k].X for k in range(K)]
                 for m_idx in range(M)] for i in range(N)]
    Iim_val = [[Iim[i][m_idx].X for m_idx in range(M)] for i in range(nodes_number)]
    sdim_val = [[sdim[i, m_idx].X for m_idx in range(M)] for i in range(N)]

    s1_routing = sum(cij[i][j] * xijk_val[i, j, k]
                     for i in range(nodes_number)
                     for j in range(nodes_number) if i < j
                     for k in range(K))
    s1_vehicle_fixed = sum(bk[k] * yik_val[0][k] for k in range(K))
    s1_holding = sum(him[i][m_idx] * Iim_val[i][m_idx]
                     for i in range(nodes_number) for m_idx in range(M))
    s1_short_units = sum(max(0.0, float(dim[i][m_idx]) - sdim_val[i][m_idx])
                         for i in range(N) for m_idx in range(M))
    s1_short_cost = pen1 * s1_short_units
    stage1_cost = s1_routing + s1_vehicle_fixed + s1_holding + s1_short_cost

    # --- Stage 2 ---
    ksi_val  = [[ksi[i, m_idx].X  for m_idx in range(M)] for i in range(N)]
    miu_val  = [[miu[i, m_idx].X  for m_idx in range(M)] for i in range(N)]
    post_inv_val  = [[post_inv[i, m_idx].X  for m_idx in range(M)] for i in range(N)]
    rem_short_val = [[rem_short[i, m_idx].X for m_idx in range(M)] for i in range(N)]

    wijm_val: Dict[Tuple[int, int, int], float] = {}
    zij_val: Dict[Tuple[int, int], float] = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                zij_val[i, j] = zij[i][j].X
                for m_idx in range(M):
                    wijm_val[i, j, m_idx] = wijm[i][j][m_idx].X

    s2_lt = sum(cij_trans[i+1][j+1] * zij_val[i, j]
                for i in range(N) for j in range(N) if i != j)
    s2_holding = sum(him[i+1][m_idx] * post_inv_val[i][m_idx]
                     for i in range(N) for m_idx in range(M))
    final_stockout_units = sum(rem_short_val[i][m_idx]
                               for i in range(N) for m_idx in range(M))
    s2_short_cost = pen2 * final_stockout_units
    lt_cost = s2_lt + s2_holding + s2_short_cost

    total_demand = sum(float(dim_real[i][m_idx])
                       for i in range(N) for m_idx in range(M))
    stockout_rate = final_stockout_units / total_demand if total_demand > 0 else 0.0

    lt_moves: List[Dict[str, Any]] = []
    for (i, j, m_idx), qty in wijm_val.items():
        if qty > 1e-6:
            lt_moves.append({
                "from_store_idx": i,
                "to_store_idx": j,
                "product_idx": m_idx,
                "qty": qty,
                "unit_cost": cij_trans[i+1][j+1],
            })

    mip_gap_val = float(m.MIPGap)

    return {
        "status": int(m.Status),
        "infeasible": False,
        "stage1_cost": stage1_cost,
        "lt_cost": lt_cost,
        "total_cost": stage1_cost + lt_cost,
        "stage1_routing_cost": s1_routing,
        "stage1_holding_cost": s1_holding,
        "stage1_vehicle_fixed_cost": s1_vehicle_fixed,
        "stage1_forecast_shortage": s1_short_cost,
        "stage1_forecast_shortage_units": s1_short_units,
        "stage2_lt_transport_cost": s2_lt,
        "stage2_holding_cost": s2_holding,
        "stage2_remaining_shortage": s2_short_cost,
        "stage2_remaining_shortage_units": final_stockout_units,
        "final_stockout_units": final_stockout_units,
        "stockout_rate": stockout_rate,
        "mip_gap": mip_gap_val,
        "runtime_seconds": runtime,
        "post_lt_inventory": post_inv_val,
        "post_lt_dc_inventory": [Iim_val[0][m_idx] for m_idx in range(M)],
        "lt_moves": lt_moves,
        "ccg_iterations": 0,   # placeholder for future C&CG implementation
        "final_robust_gap": float("nan"),
    }
