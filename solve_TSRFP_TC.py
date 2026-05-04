"""
solve_TSRFP_TC.py
=================
Two functions in this file:

1. solve_joint_tc(...)       — Man-Joint-TC Oracle
   Joint two-stage MIP: Stage 1 routing can "see" realized demand indirectly.
   Used as perfect-information lower bound. NOT a true robust method.
   Label: "Man-Joint-TC"

2. solve_tsrfp_tc(...)       — Man-TSRFP-TC with C&CG
   True two-stage robust optimization via Column-and-Constraint Generation.
   Stage 1 uses only nominal/forecast demand. Stage 2 is recourse.
   Uncertainty set (from Man et al.):
       d(ε)(i,m) = d_nom(i,m) × (1 + ε_im × rho)
       0 ≤ ε_im ≤ 1,   Σ_im ε_im ≤ Γ = b × N × M
   C&CG algorithm:
     1. Solve master with recourse scenarios S (Stage 1 + η + Stage 2 for each s)
     2. Adversarial subproblem: given fixed Stage 1 deliveries q*, find d* ∈ U
        that maximizes Stage 2 cost (LP, using Stage 2 dual variables)
     3. If Stage2_cost(d*) − η < tol: converged
     4. Else: add d* to S and repeat
   Final evaluation: apply actual shocked demand (dim_real) to the converged
   Stage 1 solution — same metric as Man-BFP-TC for comparability.
   Label: "Man-TSRFP-TC"

Structural alignment with Man-BFP-TC for all cost components.
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
    # Thesis math model: b_{ij} = 0.01 * alpha * dist (per-unit, not per-arc)
    # alpha=1.0 in the thesis benchmark, so b_{ij} = 0.01 * dist_{ij}
    cij_unit = [[0.01 * cij[i][j] for j in range(nodes_number)]
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

    # DC holding (Iim[0]) only — DC inventory rolls to next period.
    # Store-side holding is counted ONCE at the final post-LT state (Stage 2),
    # to avoid the double-count that BFP-TC incurs when Stage 2 runs.
    for m_idx in range(M):
        obj.addTerms(him[0][m_idx], Iim[0][m_idx])

    # Stage 2: LT transport cost — thesis math model b_{ij} * w_{ijm} (per-unit, not per-arc)
    for i in range(N):
        for j in range(N):
            if i != j:
                for m_idx in range(M):
                    obj.addTerms(cij_unit[i+1][j+1], wijm[i][j][m_idx])

    # Stage 2: post-LT holding cost — the SOLE holding term for stores.
    for i in range(N):
        for m_idx in range(M):
            obj.addTerms(him[i+1][m_idx], post_inv[i, m_idx])

    # Stage 2: remaining (final) shortage penalty — the SOLE shortage term.
    for i in range(N):
        for m_idx in range(M):
            obj.addTerms(pen2, rem_short[i, m_idx])

    m.setObjective(obj, GRB.MINIMIZE)

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
    # Stage 1 holding only includes DC (him[0]=0 in our setup → 0); store
    # holding is reported under Stage 2 to avoid double-counting.
    s1_holding = sum(him[0][m_idx] * Iim_val[0][m_idx] for m_idx in range(M))
    # Stage 1 forecast shortage no longer in objective; report 0 to keep the
    # total breakdown additive (s1 + s2 = total).
    s1_short_units = 0.0
    s1_short_cost = 0.0
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

    s2_lt = sum(
        cij_unit[i+1][j+1] * wijm_val.get((i, j, m_idx), 0.0)
        for i in range(N) for j in range(N) if i != j
        for m_idx in range(M)
    )
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
                "unit_cost": cij_unit[i+1][j+1],
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
        "ccg_iterations": 0,
        "final_robust_gap": float("nan"),
    }


# ============================================================================
# C&CG helpers for solve_tsrfp_tc
# ============================================================================

def _solve_stage2_lp(
    N: int, M: int,
    Ui: Sequence[float],
    cij_trans: Sequence[Sequence[float]],
    vim: Sequence[Sequence[float]],
    him: Sequence[Sequence[float]],
    Iim0: Sequence[Sequence[float]],
    qimk_fixed: Sequence[Sequence[float]],   # [i][m] summed deliveries from Stage 1
    d_scenario: Sequence[Sequence[float]],   # [i][m] scenario demand
    shortage_penalty: float,
    threads: int = 4,
) -> Tuple[float, List[List[float]]]:
    """
    Solve Stage 2 LP relaxation for a fixed Stage 1 delivery plan and a demand scenario.
    Returns (stage2_cost, pi_im) where pi_im are the dual variables on the
    ksi-miu balance constraints (used by adversarial subproblem).
    """
    lp = Model("Stage2LP")
    lp.setParam("OutputFlag", 0)
    lp.setParam("Threads", threads)

    ksi: Dict[Tuple[int, int], Any] = {}
    miu: Dict[Tuple[int, int], Any] = {}
    wijm: Dict[int, Dict[int, Dict[int, Any]]] = {}
    zij: Dict[int, Dict[int, Any]] = {}
    post_inv: Dict[Tuple[int, int], Any] = {}
    rem_short: Dict[Tuple[int, int], Any] = {}

    for i in range(N):
        for m_idx in range(M):
            ksi[i, m_idx] = lp.addVar(lb=0.0, name=f"ksi_{i}_{m_idx}")
            miu[i, m_idx] = lp.addVar(lb=0.0, name=f"miu_{i}_{m_idx}")
            post_inv[i, m_idx] = lp.addVar(lb=0.0, name=f"pi_{i}_{m_idx}")
            rem_short[i, m_idx] = lp.addVar(lb=0.0, name=f"rs_{i}_{m_idx}")

    for i in range(N):
        wijm[i] = {}
        zij[i] = {}
        for j in range(N):
            if i != j:
                wijm[i][j] = {}
                for m_idx in range(M):
                    wijm[i][j][m_idx] = lp.addVar(lb=0.0, name=f"w_{i}_{j}_{m_idx}")
                zij[i][j] = lp.addVar(lb=0.0, ub=1.0, name=f"z_{i}_{j}")   # LP relaxation

    obj = LinExpr()
    for i in range(N):
        for j in range(N):
            if i != j:
                obj.addTerms(cij_trans[i+1][j+1], zij[i][j])
    for i in range(N):
        for m_idx in range(M):
            obj.addTerms(him[i+1][m_idx], post_inv[i, m_idx])
            obj.addTerms(shortage_penalty, rem_short[i, m_idx])
    lp.setObjective(obj, GRB.MINIMIZE)

    balance_constrs: Dict[Tuple[int, int], Any] = {}
    for i in range(N):
        for m_idx in range(M):
            rhs = float(Iim0[i+1][m_idx]) + float(qimk_fixed[i][m_idx]) - float(d_scenario[i][m_idx])
            balance_constrs[i, m_idx] = lp.addConstr(
                ksi[i, m_idx] - miu[i, m_idx] == rhs,
                name=f"bal_{i}_{m_idx}"
            )

    for i in range(N):
        for m_idx in range(M):
            lp.addConstr(
                quicksum(wijm[i][j][m_idx] for j in range(N) if j != i) <= ksi[i, m_idx])
            lp.addConstr(
                quicksum(wijm[j][i][m_idx] for j in range(N) if j != i) <= miu[i, m_idx])

    bigM_val = max(Ui)
    for i in range(N):
        for j in range(N):
            if i != j:
                lp.addConstr(
                    quicksum(wijm[i][j][m_idx] for m_idx in range(M)) <= bigM_val * zij[i][j])

    for i in range(N):
        lp.addConstr(
            quicksum(zij[i][j] for j in range(N) if j != i) <=
            quicksum(ksi[i, m_idx] for m_idx in range(M)))
        lp.addConstr(
            quicksum(zij[j][i] for j in range(N) if j != i) <=
            quicksum(miu[i, m_idx] for m_idx in range(M)))

    for i in range(N):
        cap_lhs = LinExpr()
        for m_idx in range(M):
            cap_lhs.addTerms(vim[i][m_idx], ksi[i, m_idx])
            for j in range(N):
                if i != j:
                    cap_lhs.addTerms(-vim[i][m_idx], wijm[i][j][m_idx])
                    cap_lhs.addTerms( vim[i][m_idx], wijm[j][i][m_idx])
        lp.addConstr(cap_lhs <= Ui[i+1])

    for i in range(N):
        for m_idx in range(M):
            inflow  = quicksum(wijm[j][i][m_idx] for j in range(N) if j != i)
            outflow = quicksum(wijm[i][j][m_idx] for j in range(N) if j != i)
            lp.addConstr(post_inv[i, m_idx] >= ksi[i, m_idx] + inflow - outflow)
            lp.addConstr(rem_short[i, m_idx] >= miu[i, m_idx] - inflow)

    lp.optimize()

    if lp.SolCount == 0:
        return float("inf"), [[0.0] * M for _ in range(N)]

    cost = lp.ObjVal
    pi_im = [[balance_constrs[i, m_idx].Pi for m_idx in range(M)] for i in range(N)]
    return cost, pi_im


def _adversarial_greedy(
    N: int, M: int,
    d_nom: Sequence[Sequence[float]],
    pi_im: Sequence[Sequence[float]],
    rho: float,
    Gamma: float,
) -> List[List[float]]:
    """
    Greedy closed-form solution to:
        max  Σ_im  pi_im × d_nom_im × rho × eps_im
        s.t. Σ_im eps_im ≤ Gamma,  0 ≤ eps_im ≤ 1

    Sort (i,m) pairs by coefficient descending, fill budget greedily.
    Returns d_adversarial[i][m] = d_nom[i][m] × (1 + eps_im × rho).
    """
    coeffs: List[Tuple[float, int, int]] = []
    for i in range(N):
        for m_idx in range(M):
            c = float(pi_im[i][m_idx]) * float(d_nom[i][m_idx]) * rho
            if c > 0:
                coeffs.append((c, i, m_idx))
    coeffs.sort(key=lambda x: x[0], reverse=True)

    eps = [[0.0] * M for _ in range(N)]
    budget = float(Gamma)
    for (coeff, i, m_idx) in coeffs:
        if budget <= 0:
            break
        alloc = min(1.0, budget)
        eps[i][m_idx] = alloc
        budget -= alloc

    d_adv = [
        [float(d_nom[i][m_idx]) * (1.0 + eps[i][m_idx] * rho) for m_idx in range(M)]
        for i in range(N)
    ]
    return d_adv


def _solve_master_with_scenarios(
    N: int, M: int, K: int,
    Ui: Sequence[float],
    cij: Sequence[Sequence[float]],
    cij_trans: Sequence[Sequence[float]],
    Qk: Sequence[float],
    bk: Sequence[float],
    vim: Sequence[Sequence[float]],
    him: Sequence[Sequence[float]],
    Iim0: Sequence[Sequence[float]],
    dim: Sequence[Sequence[float]],
    scenarios: List[List[List[float]]],   # list of [i][m] demand arrays
    shortage_penalty_stage1: float,
    shortage_penalty_stage2: float,
    time_limit: int,
    mip_gap: float,
    threads: int,
    verbose: bool,
) -> Tuple[Dict[str, Any], List[List[float]], float]:
    """
    Solve master problem: Stage 1 MIP + eta (recourse upper bound) +
    Stage 2 LP vars for each scenario in `scenarios`.

    Returns (stage1_info_dict, qimk_summed[i][m], eta_value).
    stage1_info_dict keys: status, stage1_cost, mip_gap, runtime_seconds,
                            post_lt_dc_inventory, lt_moves (empty).
    """
    nodes_number = N + 1
    eps_eq = 1e-5
    pen1 = float(shortage_penalty_stage1)
    pen2 = float(shortage_penalty_stage2)

    mst = Model("TSRFP_Master")
    mst.setParam("OutputFlag", 1 if verbose else 0)
    mst.setParam("TimeLimit", time_limit)
    mst.setParam("MIPGap", mip_gap)
    mst.setParam("Threads", threads)
    mst.setParam("NodefileStart", 0.5)

    # ---- Stage 1 variables ----
    sdim = {(i, m_idx): mst.addVar(lb=0.0, ub=float(dim[i][m_idx]),
                                    vtype=GRB.CONTINUOUS, name=f"sd_{i}_{m_idx}")
            for i in range(N) for m_idx in range(M)}

    rm = {m_idx: mst.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"r_{m_idx}")
          for m_idx in range(M)}

    xijk: Dict[int, Dict[int, Dict[int, Any]]] = {}
    for i in range(nodes_number):
        xijk[i] = {}
        for j in range(nodes_number):
            if i < j:
                xijk[i][j] = {}
                for k in range(K):
                    if i == 0:
                        xijk[i][j][k] = mst.addVar(lb=0, ub=2, vtype=GRB.INTEGER,
                                                    name=f"x_{i}_{j}_{k}")
                    else:
                        xijk[i][j][k] = mst.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    yik = {(i, k): mst.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")
           for i in range(nodes_number) for k in range(K)}

    Iim = {(i, m_idx): mst.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"I_{i}_{m_idx}")
           for i in range(nodes_number) for m_idx in range(M)}

    qimk = {(i, m_idx, k): mst.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_{i}_{m_idx}_{k}")
            for i in range(N) for m_idx in range(M) for k in range(K)}

    # ---- Recourse variable ----
    eta = mst.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="eta")

    # ---- Per-scenario Stage 2 variables ----
    S_count = len(scenarios)
    ksi_s: Dict[Tuple[int, int, int], Any] = {}
    miu_s: Dict[Tuple[int, int, int], Any] = {}
    wijm_s: Dict[Tuple[int, int, int, int], Any] = {}
    zij_s: Dict[Tuple[int, int, int], Any] = {}
    post_inv_s: Dict[Tuple[int, int, int], Any] = {}
    rem_short_s: Dict[Tuple[int, int, int], Any] = {}

    for s in range(S_count):
        for i in range(N):
            for m_idx in range(M):
                ksi_s[s, i, m_idx] = mst.addVar(lb=0.0, name=f"ksi_s{s}_{i}_{m_idx}")
                miu_s[s, i, m_idx] = mst.addVar(lb=0.0, name=f"miu_s{s}_{i}_{m_idx}")
                post_inv_s[s, i, m_idx] = mst.addVar(lb=0.0, name=f"piv_s{s}_{i}_{m_idx}")
                rem_short_s[s, i, m_idx] = mst.addVar(lb=0.0, name=f"rs_s{s}_{i}_{m_idx}")
        for i in range(N):
            for j in range(N):
                if i != j:
                    zij_s[s, i, j] = mst.addVar(lb=0.0, ub=1.0, name=f"z_s{s}_{i}_{j}")
                    for m_idx in range(M):
                        wijm_s[s, i, j, m_idx] = mst.addVar(lb=0.0, name=f"w_s{s}_{i}_{j}_{m_idx}")

    # ---- Objective ----
    obj = LinExpr()

    for i in range(nodes_number):
        for j in range(nodes_number):
            if i < j:
                for k in range(K):
                    obj.addTerms(cij[i][j], xijk[i][j][k])

    for k in range(K):
        obj.addTerms(bk[k], yik[0, k])

    # DC holding only — store holding is captured per-scenario in Stage 2 (eta).
    # Drops Stage 1 store holding and Stage 1 forecast shortage to avoid the
    # double-count present in BFP-TC's two-stage cost decomposition.
    for m_idx in range(M):
        obj.addTerms(him[0][m_idx], Iim[0, m_idx])

    obj.addTerms(1.0, eta)
    mst.setObjective(obj, GRB.MINIMIZE)

    # ---- Stage 1 constraints (same as solve_joint_tc) ----
    for m_idx in range(M):
        qsum = quicksum(qimk[i, m_idx, k] for i in range(N) for k in range(K))
        mst.addConstr(Iim[0, m_idx] - (Iim0[0][m_idx] + rm[m_idx] - qsum) <= eps_eq)
        mst.addConstr(Iim[0, m_idx] - (Iim0[0][m_idx] + rm[m_idx] - qsum) >= -eps_eq)

    for i in range(N):
        for m_idx in range(M):
            qsum_k = quicksum(qimk[i, m_idx, k] for k in range(K))
            mst.addConstr(Iim[i+1, m_idx] - (Iim0[i+1][m_idx] + qsum_k - sdim[i, m_idx]) <= eps_eq)
            mst.addConstr(Iim[i+1, m_idx] - (Iim0[i+1][m_idx] + qsum_k - sdim[i, m_idx]) >= -eps_eq)

    dc_vol = sum(vim[0][m_idx] * Iim0[0][m_idx] for m_idx in range(M))
    mst.addConstr(quicksum(vim[0][m_idx] * rm[m_idx] for m_idx in range(M)) + dc_vol <= Ui[0])

    for i in range(N):
        i0_vol = sum(vim[i][m_idx] * Iim0[i+1][m_idx] for m_idx in range(M))
        mst.addConstr(
            quicksum(vim[i][m_idx] * qimk[i, m_idx, k] for m_idx in range(M) for k in range(K))
            + i0_vol <= Ui[i+1])

    for i in range(N):
        for k in range(K):
            mst.addConstr(
                quicksum(vim[i][m_idx] * qimk[i, m_idx, k] for m_idx in range(M))
                <= Ui[i+1] * yik[i+1, k])

    for k in range(K):
        mst.addConstr(
            quicksum(vim[i][m_idx] * qimk[i, m_idx, k] for i in range(N) for m_idx in range(M))
            <= Qk[k] * yik[0, k])

    for i in range(nodes_number):
        for k in range(K):
            in_out = LinExpr()
            for j in range(nodes_number):
                if i < j:
                    in_out.addTerms(1.0, xijk[i][j][k])
                if j < i:
                    in_out.addTerms(1.0, xijk[j][i][k])
            mst.addConstr(in_out == 2 * yik[i, k])

    for S_sub in _power_set_binary(list(range(N))):
        if not S_sub:
            continue
        for g in S_sub:
            for k in range(K):
                arc_sum = LinExpr()
                yik_sum = LinExpr()
                for i in S_sub:
                    yik_sum.addTerms(1.0, yik[i+1, k])
                    for j in S_sub:
                        if i < j:
                            arc_sum.addTerms(1.0, xijk[i+1][j+1][k])
                mst.addConstr(arc_sum <= yik_sum - yik[g+1, k])

    # ---- Per-scenario Stage 2 constraints + recourse cut ----
    for s in range(S_count):
        d_s = scenarios[s]
        bigM_val = max(Ui)
        s2_obj = LinExpr()

        for i in range(N):
            for j in range(N):
                if i != j:
                    s2_obj.addTerms(cij_trans[i+1][j+1], zij_s[s, i, j])
        for i in range(N):
            for m_idx in range(M):
                s2_obj.addTerms(him[i+1][m_idx], post_inv_s[s, i, m_idx])
                s2_obj.addTerms(pen2, rem_short_s[s, i, m_idx])

        mst.addConstr(eta >= s2_obj, name=f"eta_cut_s{s}")

        for i in range(N):
            for m_idx in range(M):
                deliveries = quicksum(qimk[i, m_idx, k] for k in range(K))
                mst.addConstr(
                    ksi_s[s, i, m_idx] - miu_s[s, i, m_idx]
                    == Iim0[i+1][m_idx] + deliveries - float(d_s[i][m_idx]))

        for i in range(N):
            for m_idx in range(M):
                mst.addConstr(
                    quicksum(wijm_s[s, i, j, m_idx] for j in range(N) if j != i)
                    <= ksi_s[s, i, m_idx])
                mst.addConstr(
                    quicksum(wijm_s[s, j, i, m_idx] for j in range(N) if j != i)
                    <= miu_s[s, i, m_idx])

        for i in range(N):
            for j in range(N):
                if i != j:
                    mst.addConstr(
                        quicksum(wijm_s[s, i, j, m_idx] for m_idx in range(M))
                        <= bigM_val * zij_s[s, i, j])

        for i in range(N):
            mst.addConstr(
                quicksum(zij_s[s, i, j] for j in range(N) if j != i)
                <= quicksum(ksi_s[s, i, m_idx] for m_idx in range(M)))
            mst.addConstr(
                quicksum(zij_s[s, j, i] for j in range(N) if j != i)
                <= quicksum(miu_s[s, i, m_idx] for m_idx in range(M)))

        for i in range(N):
            cap_lhs = LinExpr()
            for m_idx in range(M):
                cap_lhs.addTerms(vim[i][m_idx], ksi_s[s, i, m_idx])
                for j in range(N):
                    if i != j:
                        cap_lhs.addTerms(-vim[i][m_idx], wijm_s[s, i, j, m_idx])
                        cap_lhs.addTerms( vim[i][m_idx], wijm_s[s, j, i, m_idx])
            mst.addConstr(cap_lhs <= Ui[i+1])

        for i in range(N):
            for m_idx in range(M):
                inflow  = quicksum(wijm_s[s, j, i, m_idx] for j in range(N) if j != i)
                outflow = quicksum(wijm_s[s, i, j, m_idx] for j in range(N) if j != i)
                mst.addConstr(post_inv_s[s, i, m_idx] >= ksi_s[s, i, m_idx] + inflow - outflow)
                mst.addConstr(rem_short_s[s, i, m_idx] >= miu_s[s, i, m_idx] - inflow)

    mst.optimize()

    if mst.SolCount == 0:
        return {"status": int(mst.Status), "infeasible": True}, \
               [[0.0] * M for _ in range(N)], float("inf")

    qimk_sum = [[sum(qimk[i, m_idx, k].X for k in range(K)) for m_idx in range(M)]
                for i in range(N)]
    Iim_val = [[Iim[i, m_idx].X for m_idx in range(M)] for i in range(nodes_number)]
    eta_val = eta.X

    s1_routing = sum(cij[i][j] * xijk[i][j][k].X
                     for i in range(nodes_number)
                     for j in range(nodes_number) if i < j
                     for k in range(K))
    s1_vehicle_fixed = sum(bk[k] * yik[0, k].X for k in range(K))
    # DC-only holding (him[0]=0) — store holding is in Stage 2 (eta).
    s1_holding = sum(him[0][m_idx] * Iim_val[0][m_idx] for m_idx in range(M))
    s1_short_units = 0.0   # Stage 1 forecast shortage no longer in objective
    stage1_cost = s1_routing + s1_vehicle_fixed + s1_holding

    info = {
        "status": int(mst.Status),
        "infeasible": False,
        "stage1_cost": stage1_cost,
        "stage1_routing_cost": s1_routing,
        "stage1_holding_cost": s1_holding,
        "stage1_vehicle_fixed_cost": s1_vehicle_fixed,
        "stage1_forecast_shortage_units": s1_short_units,
        "post_lt_dc_inventory": [Iim_val[0][m_idx] for m_idx in range(M)],
        "mip_gap": float(mst.MIPGap),
        "runtime_seconds": 0.0,  # filled by caller
    }
    return info, qimk_sum, eta_val


# ============================================================================
# Main TSRFP-TC solver (C&CG)
# ============================================================================

def solve_tsrfp_tc(
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
    dim: Sequence[Sequence[float]],          # forecast demand (Stage 1)
    dim_real: Sequence[Sequence[float]],     # realized demand (final evaluation only)
    shortage_penalty_stage1: float = 2.0,
    shortage_penalty_stage2: float = 2.0,
    lt_cost_multiplier: float = 1.2,
    b_budget: float = 0.3,                   # fraction of (N×M) as budget Gamma
    rho: float = 0.5,                        # max demand uplift fraction
    max_ccg_iter: int = 20,
    ccg_tol: float = 1e-3,
    time_limit: int = 600,
    mip_gap: float = 0.01,
    threads: int = 4,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    True Two-Stage Robust TSRFP-TC via Column-and-Constraint Generation.

    Uncertainty set:  d(eps)_im = d_nom_im × (1 + eps_im × rho)
                      0 ≤ eps_im ≤ 1,  Σ_im eps_im ≤ Gamma = b × N × M

    Stage 1 sees only dim (forecast). Stage 2 is recourse.
    Final metrics are evaluated on dim_real for comparability with Man-BFP-TC.
    Label: "Man-TSRFP-TC"
    """
    N = int(retailer_number)
    M = int(product_kind)
    K = int(vehicle_number)
    nodes_number = N + 1
    Gamma = b_budget * N * M
    cij_trans = [[cij[i][j] * lt_cost_multiplier for j in range(nodes_number)]
                 for i in range(nodes_number)]
    t0 = time.time()

    # Initialise with nominal demand scenario
    scenarios: List[List[List[float]]] = [
        [[float(dim[i][m_idx]) for m_idx in range(M)] for i in range(N)]
    ]

    master_info: Dict[str, Any] = {}
    qimk_sum: List[List[float]] = [[0.0] * M for _ in range(N)]
    eta_val = 0.0
    ccg_iter = 0
    robust_gap = float("inf")

    for ccg_iter in range(1, max_ccg_iter + 1):
        remaining_time = max(30, time_limit - int(time.time() - t0))
        master_info, qimk_sum, eta_val = _solve_master_with_scenarios(
            N, M, K, Ui, cij, cij_trans, Qk, bk, vim, him, Iim0, dim,
            scenarios,
            shortage_penalty_stage1, shortage_penalty_stage2,
            remaining_time, mip_gap, threads, verbose,
        )
        master_info["runtime_seconds"] = time.time() - t0

        if master_info.get("infeasible"):
            break

        # Adversarial: find worst-case demand given current Stage 1 q*
        s2_cost, pi_im = _solve_stage2_lp(
            N, M, Ui, cij_trans, vim, him, Iim0,
            qimk_sum, dim,   # use nominal (forecast) as base for adversarial
            shortage_penalty_stage2, threads,
        )

        d_adv = _adversarial_greedy(N, M, dim, pi_im, rho, Gamma)

        # Evaluate Stage 2 cost at d_adv
        s2_cost_adv, _ = _solve_stage2_lp(
            N, M, Ui, cij_trans, vim, him, Iim0,
            qimk_sum, d_adv,
            shortage_penalty_stage2, threads,
        )

        robust_gap = abs(s2_cost_adv - eta_val) / max(abs(eta_val), 1.0)

        if verbose:
            print(f"  [C&CG iter {ccg_iter}] eta={eta_val:.4f}  "
                  f"s2_adv={s2_cost_adv:.4f}  gap={robust_gap:.6f}")

        if robust_gap < ccg_tol:
            break

        # Check for duplicate scenario
        is_dup = any(
            all(abs(d_adv[i][m_idx] - sc[i][m_idx]) < 1e-8
                for i in range(N) for m_idx in range(M))
            for sc in scenarios
        )
        if is_dup:
            break

        scenarios.append(d_adv)

        if time.time() - t0 > time_limit - 10:
            break

    # ---- Final evaluation on actual realized demand (dim_real) ----
    s2_final, _ = _solve_stage2_lp(
        N, M, Ui, cij_trans, vim, him, Iim0,
        qimk_sum, dim_real,
        shortage_penalty_stage2, threads,
    )

    # Re-solve Stage 2 at dim_real to extract detailed metrics
    lp_final = Model("Stage2Final")
    lp_final.setParam("OutputFlag", 0)
    lp_final.setParam("Threads", threads)

    ksi_f = {(i, m_idx): lp_final.addVar(lb=0.0) for i in range(N) for m_idx in range(M)}
    miu_f = {(i, m_idx): lp_final.addVar(lb=0.0) for i in range(N) for m_idx in range(M)}
    post_inv_f = {(i, m_idx): lp_final.addVar(lb=0.0) for i in range(N) for m_idx in range(M)}
    rem_short_f = {(i, m_idx): lp_final.addVar(lb=0.0) for i in range(N) for m_idx in range(M)}
    wijm_f: Dict[Tuple[int, int, int], Any] = {}
    zij_f: Dict[Tuple[int, int], Any] = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                zij_f[i, j] = lp_final.addVar(lb=0.0, ub=1.0)
                for m_idx in range(M):
                    wijm_f[i, j, m_idx] = lp_final.addVar(lb=0.0)

    obj_f = LinExpr()
    for i in range(N):
        for j in range(N):
            if i != j:
                obj_f.addTerms(cij_trans[i+1][j+1], zij_f[i, j])
    for i in range(N):
        for m_idx in range(M):
            obj_f.addTerms(him[i+1][m_idx], post_inv_f[i, m_idx])
            obj_f.addTerms(float(shortage_penalty_stage2), rem_short_f[i, m_idx])
    lp_final.setObjective(obj_f, GRB.MINIMIZE)

    bigM_val = max(Ui)
    for i in range(N):
        for m_idx in range(M):
            rhs = float(Iim0[i+1][m_idx]) + float(qimk_sum[i][m_idx]) - float(dim_real[i][m_idx])
            lp_final.addConstr(ksi_f[i, m_idx] - miu_f[i, m_idx] == rhs)

    for i in range(N):
        for m_idx in range(M):
            lp_final.addConstr(
                quicksum(wijm_f[i, j, m_idx] for j in range(N) if j != i) <= ksi_f[i, m_idx])
            lp_final.addConstr(
                quicksum(wijm_f[j, i, m_idx] for j in range(N) if j != i) <= miu_f[i, m_idx])

    for i in range(N):
        for j in range(N):
            if i != j:
                lp_final.addConstr(
                    quicksum(wijm_f[i, j, m_idx] for m_idx in range(M)) <= bigM_val * zij_f[i, j])

    for i in range(N):
        lp_final.addConstr(
            quicksum(zij_f[i, j] for j in range(N) if j != i)
            <= quicksum(ksi_f[i, m_idx] for m_idx in range(M)))
        lp_final.addConstr(
            quicksum(zij_f[j, i] for j in range(N) if j != i)
            <= quicksum(miu_f[i, m_idx] for m_idx in range(M)))

    for i in range(N):
        cap_lhs = LinExpr()
        for m_idx in range(M):
            cap_lhs.addTerms(vim[i][m_idx], ksi_f[i, m_idx])
            for j in range(N):
                if i != j:
                    cap_lhs.addTerms(-vim[i][m_idx], wijm_f[i, j, m_idx])
                    cap_lhs.addTerms( vim[i][m_idx], wijm_f[j, i, m_idx])
        lp_final.addConstr(cap_lhs <= Ui[i+1])

    for i in range(N):
        for m_idx in range(M):
            inflow  = quicksum(wijm_f[j, i, m_idx] for j in range(N) if j != i)
            outflow = quicksum(wijm_f[i, j, m_idx] for j in range(N) if j != i)
            lp_final.addConstr(post_inv_f[i, m_idx] >= ksi_f[i, m_idx] + inflow - outflow)
            lp_final.addConstr(rem_short_f[i, m_idx] >= miu_f[i, m_idx] - inflow)

    lp_final.optimize()

    runtime = time.time() - t0

    if lp_final.SolCount == 0 or master_info.get("infeasible"):
        return {
            "status": master_info.get("status", -1),
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
            "ccg_iterations": ccg_iter,
            "final_robust_gap": robust_gap,
        }

    pen2 = float(shortage_penalty_stage2)
    post_inv_val = [[post_inv_f[i, m_idx].X for m_idx in range(M)] for i in range(N)]
    rem_short_val = [[rem_short_f[i, m_idx].X for m_idx in range(M)] for i in range(N)]
    zij_val = {(i, j): zij_f[i, j].X for i in range(N) for j in range(N) if i != j}
    wijm_val = {(i, j, m_idx): wijm_f[i, j, m_idx].X
                for i in range(N) for j in range(N) if i != j
                for m_idx in range(M)}

    s2_lt = sum(cij_trans[i+1][j+1] * zij_val[i, j] for (i, j) in zij_val)
    s2_holding = sum(him[i+1][m_idx] * post_inv_val[i][m_idx]
                     for i in range(N) for m_idx in range(M))
    final_stockout_units = sum(rem_short_val[i][m_idx]
                               for i in range(N) for m_idx in range(M))
    s2_short_cost = pen2 * final_stockout_units
    lt_cost = s2_lt + s2_holding + s2_short_cost

    stage1_cost = master_info["stage1_cost"]
    total_demand = sum(float(dim_real[i][m_idx]) for i in range(N) for m_idx in range(M))
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

    return {
        "status": master_info["status"],
        "infeasible": False,
        "stage1_cost": stage1_cost,
        "lt_cost": lt_cost,
        "total_cost": stage1_cost + lt_cost,
        "stage1_routing_cost": master_info["stage1_routing_cost"],
        "stage1_holding_cost": master_info["stage1_holding_cost"],
        "stage1_vehicle_fixed_cost": master_info["stage1_vehicle_fixed_cost"],
        "stage1_forecast_shortage": float(shortage_penalty_stage1) * master_info["stage1_forecast_shortage_units"],
        "stage1_forecast_shortage_units": master_info["stage1_forecast_shortage_units"],
        "stage2_lt_transport_cost": s2_lt,
        "stage2_holding_cost": s2_holding,
        "stage2_remaining_shortage": s2_short_cost,
        "stage2_remaining_shortage_units": final_stockout_units,
        "final_stockout_units": final_stockout_units,
        "stockout_rate": stockout_rate,
        "mip_gap": master_info["mip_gap"],
        "runtime_seconds": runtime,
        "post_lt_inventory": post_inv_val,
        "post_lt_dc_inventory": master_info["post_lt_dc_inventory"],
        "lt_moves": lt_moves,
        "ccg_iterations": ccg_iter,
        "final_robust_gap": robust_gap,
    }
