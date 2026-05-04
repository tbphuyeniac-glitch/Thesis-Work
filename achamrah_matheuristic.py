"""
achamrah_matheuristic.py
=========================
GA/SA matheuristic with inner Gurobi LP for delivery quantity optimization.

Follows Achamrah 2022 design:
  - Outer SA/GA loop optimises routes (same as achamrah_ga_sa_baseline.py)
  - Inner Gurobi LP optimises delivery quantities given fixed routes
    (replaces the greedy evaluator in the pure-heuristic version)

Usage:
    from achamrah_matheuristic import GASAMatheuristicSolver, MatheuristicParams
    solver = GASAMatheuristicSolver(inst, params)
    result = solver.solve(initial_routes=gur_routes)   # optional warm-start
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from achamrah_ga_sa_baseline import (
    BaselineGASASolver,
    BaselineInstance,
    GASAParams,
    GASAResult,
    Chromosome,
)


# ──────────────────────────────────────────────────────────────────
# Inner LP: solve delivery quantities for fixed routes
# ──────────────────────────────────────────────────────────────────
def _solve_inner_lp(inst: BaselineInstance,
                    routes: Chromosome,
                    dispatch: set,
                    time_limit: float = 10.0) -> Dict[str, float]:
    """
    Given fixed routes, minimise holding + shortage + ship_cost over delivery
    quantities (LP, routes are parameters not variables).

    Returns dict: total, holding, shortage, shortage_qty, ship.
    Falls back to large-penalty dict on LP failure.
    """
    H = sorted(inst.H)

    try:
        mdl = gp.Model("inner_lp")
        mdl.setParam("OutputFlag", 0)
        mdl.setParam("TimeLimit", time_limit)
        mdl.setParam("Method", 1)   # dual simplex — fastest for small LP

        # q[s,p,t]: delivery quantity (non-zero only for stores on route at t)
        q_keys = set()
        for t in H:
            if t not in dispatch:
                continue
            for v in inst.V:
                for s in routes.get((t, v), []):
                    for p in inst.P:
                        q_keys.add((s, p, t))

        q    = {k: mdl.addVar(lb=0.0) for k in q_keys}
        I_wh = {(p, t): mdl.addVar(lb=0.0) for p in inst.P for t in H}
        I_s  = {(s, p, t): mdl.addVar(lb=0.0) for s in inst.N for p in inst.P for t in H}
        lost = {(s, p, t): mdl.addVar(lb=0.0) for s in inst.N for p in inst.P for t in H}
        mdl.update()

        # ── Objective ───────────────────────────────────────────────
        mdl.setObjective(
            gp.quicksum(float(inst.h.get((p, 0), 0.0)) * I_wh[(p, t)]
                        for p in inst.P for t in H)
            + gp.quicksum(float(inst.h.get((p, s), 0.0)) * I_s[(s, p, t)]
                          for s in inst.N for p in inst.P for t in H)
            + gp.quicksum(float(inst.f.get((p, s), 0.25)) * lost[(s, p, t)]
                          for s in inst.N for p in inst.P for t in H)
            + gp.quicksum(float(inst.ship_cost_cw.get((p, s), 0.0)) * q[(s, p, t)]
                          for (s, p, t) in q_keys),
            GRB.MINIMIZE,
        )

        # ── Warehouse rolling balance ────────────────────────────────
        for p in inst.P:
            for i, t in enumerate(H):
                prev = (float(inst.I0.get((p, 0), 0.0)) if i == 0
                        else I_wh[(p, H[i - 1])])
                shipped = gp.quicksum(q.get((s, p, t), 0.0) for s in inst.N)
                mdl.addConstr(
                    I_wh[(p, t)] == prev + float(inst.g.get((p, t), 0.0)) - shipped
                )

        # ── Store rolling balance ─────────────────────────────────────
        for s in inst.N:
            for p in inst.P:
                for i, t in enumerate(H):
                    prev = (float(inst.I0.get((p, s), 0.0)) if i == 0
                            else I_s[(s, p, H[i - 1])])
                    recv = q.get((s, p, t), 0.0)
                    dem  = float(inst.D.get((p, s, t), 0.0))
                    mdl.addConstr(
                        I_s[(s, p, t)] == prev + recv - dem + lost[(s, p, t)]
                    )

        # ── Vehicle capacity ─────────────────────────────────────────
        for t in H:
            if t not in dispatch:
                continue
            for v in inst.V:
                seq = routes.get((t, v), [])
                if not seq:
                    continue
                mdl.addConstr(
                    gp.quicksum(q.get((s, p, t), 0.0) for s in seq for p in inst.P)
                    <= float(inst.Q)
                )

        # ── Solve ────────────────────────────────────────────────────
        mdl.optimize()

        if mdl.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            return {"total": 1e15, "holding": 0.0, "shortage": 0.0,
                    "shortage_qty": 0.0, "ship": 0.0}

        obj_val  = mdl.objVal
        h_val    = (sum(float(inst.h.get((p, 0), 0.0)) * I_wh[(p, t)].X
                        for p in inst.P for t in H)
                    + sum(float(inst.h.get((p, s), 0.0)) * I_s[(s, p, t)].X
                          for s in inst.N for p in inst.P for t in H))
        sh_val   = sum(float(inst.f.get((p, s), 0.25)) * lost[(s, p, t)].X
                       for s in inst.N for p in inst.P for t in H)
        sh_qty   = sum(lost[(s, p, t)].X for s in inst.N for p in inst.P for t in H)
        ship_val = sum(float(inst.ship_cost_cw.get((p, s), 0.0)) * q[(s, p, t)].X
                       for (s, p, t) in q_keys)

        return {"total": obj_val, "holding": h_val,
                "shortage": sh_val, "shortage_qty": sh_qty, "ship": ship_val}

    except Exception:
        return {"total": 1e15, "holding": 0.0, "shortage": 0.0,
                "shortage_qty": 0.0, "ship": 0.0}


# ──────────────────────────────────────────────────────────────────
# Matheuristic solver: GA/SA outer + inner Gurobi LP
# ──────────────────────────────────────────────────────────────────
class GASAMatheuristicSolver(BaselineGASASolver):
    """
    Subclass of BaselineGASASolver that replaces the greedy delivery
    evaluator with an inner Gurobi LP.  All GA/SA operators are inherited.

    inner_lp_time_limit: per-LP Gurobi time limit in seconds.
    """

    def __init__(self, inst: BaselineInstance,
                 params: Optional[GASAParams] = None,
                 inner_lp_time_limit: float = 5.0):
        super().__init__(inst, params)
        self._inner_tl = inner_lp_time_limit
        self._lp_calls = 0
        self._lp_total_time = 0.0

    def _evaluate_routes(self, routes: Chromosome) -> Dict[str, float]:
        """Override: routing cost from parent logic + inner Gurobi for quantities."""
        inst = self.inst

        # 1) Routing + vehicle fixed cost (deterministic, same as base class)
        routing = 0.0
        veh_fixed = 0.0
        for (t, v), seq in routes.items():
            if not seq:
                continue
            veh_fixed += inst.vehicle_fixed_cost
            prev = 0
            for node in seq:
                routing += inst.alpha * inst.d.get((prev, node), 0.0)
                prev = node
            routing += inst.alpha * inst.d.get((prev, 0), 0.0)

        # 2) Inner LP for holding + shortage + ship_cost
        _t0 = time.perf_counter()
        lp = _solve_inner_lp(inst, routes, self._dispatch,
                             time_limit=self._inner_tl)
        self._lp_calls += 1
        self._lp_total_time += time.perf_counter() - _t0

        total = routing + veh_fixed + lp["total"]
        return {
            "routing":       routing,
            "holding":       lp["holding"],
            "shortage":      lp["shortage"],
            "shortage_qty":  lp.get("shortage_qty", 0.0),
            "ship":          lp["ship"],
            "vehicle_fixed": veh_fixed,
            "total":         total,
        }
