"""
Extended Achamrah IRPT Solver with Simplified LT Mode
=======================================================

Extends achamrah_2022_irpt_matheuristic.AchamrahIRPTSolver to support:
- Mode 1: vehicle_indexed_lt=True  → Original Achamrah (y[p,i,j,v,t])
- Mode 2: vehicle_indexed_lt=False → Simplified pairwise LT (y[p,i,j,t])

Both modes solve ONE integrated MIP (not two-stage).

Does NOT modify original Achamrah files.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from achamrah_2022_irpt_matheuristic import (
    IRPTInstance,
    AchamrahIRPTSolver,
    SolveArtifacts,
)


@dataclass
class ExtendedSolveArtifacts(SolveArtifacts):
    """Extended solve artifacts with cost breakdown and LT tracking."""
    cost_breakdown: Dict[str, float] = None
    lt_moves: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.cost_breakdown is None:
            self.cost_breakdown = {}
        if self.lt_moves is None:
            self.lt_moves = []


class AchamrahIntegratedExtendedSolver(AchamrahIRPTSolver):
    """
    Extended solver supporting both vehicle-indexed and simplified LT modes.
    
    Mode 1: vehicle_indexed_lt=True
    - LT: y[p, i, j, v, t] (indexed by vehicle)
    - Constraint: y[p,i,j,v,t] <= q[p,i,j,v,t] (LT must be carried by flow)
    - This is the original Achamrah formulation
    
    Mode 2: vehicle_indexed_lt=False
    - LT: y[p, i, j, t] (not indexed by vehicle)
    - NO constraint linking y to q
    - LT is pairwise between stores only
    - Still integrated: routing, inventory, shortage, LT all in one model
    """
    
    def __init__(
        self,
        inst: IRPTInstance,
        vehicle_indexed_lt: bool = True,
        **kwargs
    ):
        super().__init__(inst, **kwargs)
        self.vehicle_indexed_lt = vehicle_indexed_lt
        
    def build_model(
        self,
        relaxed: bool = False,
        fixed_routes: Optional[Dict] = None,
        active_nodes_by_period: Optional[Dict[int, List[int]]] = None,
        allow_lateral_transshipment: bool = True,
        use_valid_16_19: bool = True,
        use_valid_20: bool = True,
        add_lazy_21_placeholder: bool = False,
        model_name: str = "IRPT_Extended",
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Build model with support for simplified LT mode.
        
        If vehicle_indexed_lt=False:
        - Changes y to y[p,i,j,t] instead of y[p,i,j,v,t]
        - Removes linking constraint y <= q
        - Keeps y in inventory balance and objective
        """
        
        if self.vehicle_indexed_lt:
            # Use original Achamrah model
            return super().build_model(
                relaxed=relaxed,
                fixed_routes=fixed_routes,
                active_nodes_by_period=active_nodes_by_period,
                allow_lateral_transshipment=allow_lateral_transshipment,
                use_valid_16_19=use_valid_16_19,
                use_valid_20=use_valid_20,
                add_lazy_21_placeholder=add_lazy_21_placeholder,
                model_name=model_name,
            )
        else:
            # Build simplified LT model
            return self._build_model_simplified_lt(
                relaxed=relaxed,
                fixed_routes=fixed_routes,
                active_nodes_by_period=active_nodes_by_period,
                allow_lateral_transshipment=allow_lateral_transshipment,
                use_valid_16_19=use_valid_16_19,
                use_valid_20=use_valid_20,
                add_lazy_21_placeholder=add_lazy_21_placeholder,
                model_name=model_name,
            )
    
    def _build_model_simplified_lt(
        self,
        relaxed: bool = False,
        fixed_routes: Optional[Dict] = None,
        active_nodes_by_period: Optional[Dict[int, List[int]]] = None,
        allow_lateral_transshipment: bool = True,
        use_valid_16_19: bool = True,
        use_valid_20: bool = True,
        add_lazy_21_placeholder: bool = False,
        model_name: str = "IRPT_Simplified_LT",
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Build integrated model with simplified pairwise LT (NOT vehicle-indexed).
        
        Key difference from original:
        - y[p,i,j,t] instead of y[p,i,j,v,t]
        - No linking y <= q constraint
        - LT is pairwise between stores, not assigned to vehicles
        """
        
        inst = self.inst
        N = inst.N
        P = inst.P
        H = inst.H
        V = inst.V
        N0 = inst.N0
        
        m = gp.Model(model_name)
        m.Params.OutputFlag = 0
        
        # Decision variables
        # Inventory at node i for product p at end of period t
        I = {}
        for p in P:
            for i in N0:
                for t in H:
                    I[p, i, t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"I[{p},{i},{t}]")
        
        # Direct shipment from CW
        Qdir = {}
        for p in P:
            for j in N:
                for t in H:
                    Qdir[p, j, t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"Qdir[{p},{j},{t}]")
        
        # Product flow on route arcs (vehicle-indexed)
        q = {}
        for p in P:
            for i in N0:
                for j in N0:
                    if i == j:
                        continue
                    for v in V:
                        for t in H:
                            q[p, i, j, v, t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q[{p},{i},{j},{v},{t}]")
        
        # SIMPLIFIED LT: NOT indexed by vehicle
        y = {}
        for p in P:
            for i in N:
                for j in N:
                    if i == j:
                        continue
                    for t in H:
                        y[p, i, j, t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"y[{p},{i},{j},{t}]")
        
        # Shortage at node i for product p at end of period t
        S = {}
        for p in P:
            for i in N:
                for t in H:
                    S[p, i, t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"S[{p},{i},{t}]")
        
        # Routing variables (same as original)
        x = {}
        u = {}
        z = {}
        for i in N0:
            for j in N0:
                if i == j:
                    continue
                for v in V:
                    for t in H:
                        x[i, j, v, t] = m.addVar(vtype=GRB.BINARY, name=f"x[{i},{j},{v},{t}]")
        
        for v in V:
            for t in H:
                u[v, t] = m.addVar(vtype=GRB.BINARY, name=f"u[{v},{t}]")
        
        for i in N0:
            for v in V:
                for t in H:
                    z[i, v, t] = m.addVar(vtype=GRB.BINARY, name=f"z[{i},{v},{t}]")
        
        m.update()
        
        # OBJECTIVE: routing + holding + shortage + simplified LT cost
        obj = gp.quicksum(
            inst.alpha * inst.d[i, j] * x[i, j, v, t]
            for i in N0 for j in N0 if i != j
            for v in V for t in H
        )
        obj += gp.quicksum(
            inst.h[p, i] * I[p, i, t]
            for p in P for i in N0 for t in H
        )
        obj += gp.quicksum(
            inst.f[p, i] * S[p, i, t]
            for p in P for i in N for t in H
        )
        # SIMPLIFIED LT cost: not indexed by vehicle
        obj += gp.quicksum(
            inst.b[i, j] * y[p, i, j, t]
            for p in P for i in N for j in N if i != j
            for t in H
        )
        m.setObjective(obj, GRB.MINIMIZE)
        
        # CONSTRAINTS
        
        # (1) Inventory balance for CW
        for p in P:
            for t in H:
                lhs = I[p, 0, t]
                prev_inv = inst.I0[p, 0] if t == H[0] else I[p, 0, t-1]
                rhs = prev_inv + inst.g[p, t] - gp.quicksum(Qdir[p, j, t] for j in N)
                m.addConstr(lhs == rhs, name=f"inv_cw[{p},{t}]")
        
        # (2) Inventory balance for POS (WITH SIMPLIFIED LT)
        if allow_lateral_transshipment:
            for p in P:
                for i in N:
                    for t in H:
                        prev_inv = inst.I0[p, i] if t == H[0] else I[p, i, t-1]
                        inbound_direct = Qdir[p, i, t]
                        inbound_lt = gp.quicksum(y[p, j, i, t] for j in N if j != i)
                        demand = inst.D[p, i, t]
                        outbound_lt = gp.quicksum(y[p, i, j, t] for j in N if j != i)
                        
                        lhs = I[p, i, t] + S[p, i, t]
                        rhs = prev_inv + inbound_direct + inbound_lt - demand - outbound_lt
                        m.addConstr(lhs == rhs, name=f"inv_pos[{p},{i},{t}]")
        else:
            for p in P:
                for i in N:
                    for t in H:
                        prev_inv = inst.I0[p, i] if t == H[0] else I[p, i, t-1]
                        inbound = Qdir[p, i, t]
                        demand = inst.D[p, i, t]
                        
                        lhs = I[p, i, t] + S[p, i, t]
                        rhs = prev_inv + inbound - demand
                        m.addConstr(lhs == rhs, name=f"inv_pos_no_lt[{p},{i},{t}]")
        
        # (3) Non-negative inventory (optional tightening)
        for p in P:
            for i in N:
                for t in H:
                    m.addConstr(I[p, i, t] >= 0, name=f"nonneg_inv[{p},{i},{t}]")
        
        # (4) Shortage only at demand nodes
        for p in P:
            for i in N:
                for t in H:
                    m.addConstr(S[p, i, t] <= inst.D[p, i, t], name=f"max_shortage[{p},{i},{t}]")
        
        # (5) Empty vehicle return constraint
        for v in V:
            for t in H:
                if active_nodes_by_period and t in active_nodes_by_period:
                    active = active_nodes_by_period[t]
                    m.addConstr(gp.quicksum(q[p, i, 0, v, t] for p in P for i in active) == 0, name=f"empty_return[{v},{t}]")
                else:
                    m.addConstr(gp.quicksum(q[p, i, 0, v, t] for p in P for i in N) == 0, name=f"empty_return[{v},{t}]")
        
        # (6) Storage capacity
        for i in N0:
            for t in H:
                m.addConstr(gp.quicksum(I[p, i, t] for p in P) <= inst.C[i], name=f"cap_store[{i},{t}]")
        
        # (7) Vehicle capacity on arcs
        for i in N0:
            for j in N0:
                if i == j:
                    continue
                for v in V:
                    for t in H:
                        m.addConstr(gp.quicksum(q[p, i, j, v, t] for p in P) <= inst.Q * x[i, j, v, t], 
                                   name=f"cap_arc[{i},{j},{v},{t}]")
        
        # (8) LT outflow limited by inventory available
        # SIMPLIFIED: y[p,i,j,t] only (no vehicle index)
        for p in P:
            for i in N:
                for t in H:
                    inv_prev = inst.I0[p, i] if t == H[0] else I[p, i, t-1]
                    m.addConstr(
                        gp.quicksum(y[p, i, j, t] for j in N if j != i) <= inv_prev,
                        name=f"trans_inv[{p},{i},{t}]"
                    )
        
        # (9) Flow conservation at customer
        for j in N:
            for v in V:
                for t in H:
                    m.addConstr(
                        gp.quicksum(x[i, j, v, t] for i in N0 if i != j)
                        == gp.quicksum(x[j, i, v, t] for i in N0 if i != j),
                        name=f"route_cons[{j},{v},{t}]"
                    )
        
        # (10) Visit at most once per period
        for j in N:
            for t in H:
                m.addConstr(
                    gp.quicksum(x[i, j, v, t] for i in N0 if i != j for v in V) <= 1,
                    name=f"visit_once[{j},{t}]"
                )
        
        # (11) Vehicle usage iff leaves CW
        for v in V:
            for t in H:
                m.addConstr(gp.quicksum(x[0, j, v, t] for j in N) == u[v, t], name=f"veh_use[{v},{t}]")
        
        # (12) Fleet availability
        for t in H:
            m.addConstr(gp.quicksum(u[v, t] for v in V) <= len(V), name=f"fleet[{t}]")
        
        # (13) Product flow only on routing arcs
        for p in P:
            for i in N0:
                for j in N0:
                    if i == j:
                        continue
                    for v in V:
                        for t in H:
                            m.addConstr(q[p, i, j, v, t] <= inst.Q * x[i, j, v, t], 
                                       name=f"link_q_x[{p},{i},{j},{v},{t}]")
        
        # (14) Direct shipment definition
        for p in P:
            for j in N:
                for t in H:
                    m.addConstr(
                        Qdir[p, j, t] == gp.quicksum(q[p, 0, j, v, t] for v in V),
                        name=f"direct_from_cw[{p},{j},{t}]"
                    )
        
        # SIMPLIFIED LT: NO linking constraint y <= q
        # This is the KEY difference from original Achamrah
        # LT is completely pairwise, not linked to vehicle routing
        
        # (15) Route consistency: z indicators
        for v in V:
            for t in H:
                m.addConstr(z[0, v, t] == u[v, t], name=f"zdef[0,{v},{t}]")
                for i in N:
                    m.addConstr(z[i, v, t] == gp.quicksum(x[j, i, v, t] for j in N0 if j != i), 
                               name=f"zdef[{i},{v},{t}]")
        
        # Valid inequalities (16)-(19) - same as original
        if use_valid_16_19:
            for i in N:
                for v in V:
                    for t in H:
                        m.addConstr(x[0, i, v, t] <= z[i, v, t], name=f"v16[{i},{v},{t}]")
            for i in N:
                for j in N:
                    if i == j:
                        continue
                    for v in V:
                        for t in H:
                            m.addConstr(x[i, j, v, t] <= z[j, v, t], name=f"v17[{i},{j},{v},{t}]")
            for i in N:
                for v in V:
                    for t in H:
                        m.addConstr(z[i, v, t] <= z[0, v, t], name=f"v18[{i},{v},{t}]")
            for idx, v in enumerate(V):
                if idx == 0:
                    continue
                prev_v = V[idx - 1]
                for t in H:
                    m.addConstr(z[0, v, t] <= z[0, prev_v, t], name=f"v19[{v},{t}]")
        
        # Valid inequality (20) - adjusted for simplified LT
        if use_valid_20:
            for p in P:
                for i in N:
                    for t1 in H:
                        for t2 in H:
                            if t2 < t1:
                                continue
                            demand_sum = sum(inst.D[p, i, tau] for tau in range(t1, t2 + 1))
                            if demand_sum <= 0:
                                continue
                            inv_before = inst.I0[p, i] if t1 == H[0] else I[p, i, t1 - 1]
                            # Modified for simplified LT (no vehicle index)
                            lhs = (
                                gp.quicksum(z[i, v, tau] for v in V for tau in range(t1, t2 + 1))
                                + (1.0 / demand_sum)
                                * gp.quicksum(y[p, j, i, tau] for j in N if j != i for tau in range(t1, t2 + 1))
                            )
                            rhs = 1.0 - (1.0 / demand_sum) * inv_before
                            m.addConstr(lhs >= rhs, name=f"v20[{p},{i},{t1},{t2}]")
        
        return m, {
            "I": I,
            "Qdir": Qdir,
            "q": q,
            "y": y,
            "S": S,
            "x": x,
            "u": u,
            "z": z,
        }
    
    def solve_model(
        self,
        relaxed: bool = False,
        fixed_routes: Optional[Dict] = None,
        active_nodes_by_period: Optional[Dict[int, List[int]]] = None,
        time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        allow_lateral_transshipment: bool = True,
        model_name: str = None,
    ) -> ExtendedSolveArtifacts:
        """Solve and extract extended artifacts including cost breakdown."""
        
        if model_name is None:
            model_name = "IRPT_Extended" if not self.vehicle_indexed_lt else "IRPT"
        
        n_periods = len(self.inst.H)
        use_v20 = n_periods <= 20
        
        m, vars_dict = self.build_model(
            relaxed=relaxed,
            fixed_routes=fixed_routes,
            active_nodes_by_period=active_nodes_by_period,
            allow_lateral_transshipment=allow_lateral_transshipment,
            use_valid_20=use_v20,
            model_name=model_name,
        )
        
        if time_limit is not None:
            m.Params.TimeLimit = time_limit
        if mip_gap is not None:
            m.Params.MIPGap = mip_gap
        
        m.optimize()
        
        # Extract solution
        artifacts = ExtendedSolveArtifacts(
            objective=m.ObjVal if m.SolCount > 0 else math.inf,
            status=m.Status,
            runtime=m.Runtime,
            mip_gap=m.MIPGap if m.SolCount > 0 else float("nan"),
            num_vars=m.NumVars,
            num_constrs=m.NumConstrs,
        )
        
        if m.SolCount > 0:
            # Extract cost breakdown
            I = vars_dict["I"]
            S = vars_dict["S"]
            y = vars_dict["y"]
            x = vars_dict["x"]
            
            routing_cost = sum(
                self.inst.alpha * self.inst.d[i, j] * x[i, j, v, t].X
                for i in self.inst.N0 for j in self.inst.N0 if i != j
                for v in self.inst.V for t in self.inst.H
                if (i, j, v, t) in x
            )
            
            holding_cost = sum(
                self.inst.h[p, i] * I[p, i, t].X
                for p in self.inst.P for i in self.inst.N0 for t in self.inst.H
            )
            
            shortage_cost = sum(
                self.inst.f[p, i] * S[p, i, t].X
                for p in self.inst.P for i in self.inst.N for t in self.inst.H
            )
            
            lt_cost = 0.0
            lt_moves = []
            if self.vehicle_indexed_lt:
                # Original y[p,i,j,v,t]
                for p in self.inst.P:
                    for i in self.inst.N:
                        for j in self.inst.N:
                            if i == j:
                                continue
                            for v in self.inst.V:
                                for t in self.inst.H:
                                    if (p, i, j, v, t) in y:
                                        val = y[p, i, j, v, t].X
                                        if val > 1e-6:
                                            lt_cost += self.inst.b[i, j] * val
                                            lt_moves.append({
                                                "period": t,
                                                "product": self.inst.P.index(p),
                                                "from_store": i,
                                                "to_store": j,
                                                "vehicle": v,
                                                "quantity": val,
                                                "cost": self.inst.b[i, j] * val,
                                            })
            else:
                # Simplified y[p,i,j,t]
                for p in self.inst.P:
                    for i in self.inst.N:
                        for j in self.inst.N:
                            if i == j:
                                continue
                            for t in self.inst.H:
                                if (p, i, j, t) in y:
                                    val = y[p, i, j, t].X
                                    if val > 1e-6:
                                        lt_cost += self.inst.b[i, j] * val
                                        lt_moves.append({
                                            "period": t,
                                            "product": self.inst.P.index(p),
                                            "from_store": i,
                                            "to_store": j,
                                            "vehicle": None,  # Not assigned to vehicle
                                            "quantity": val,
                                            "cost": self.inst.b[i, j] * val,
                                        })
            
            artifacts.cost_breakdown = {
                "routing": routing_cost,
                "holding": holding_cost,
                "shortage": shortage_cost,
                "lt": lt_cost,
                "total": routing_cost + holding_cost + shortage_cost + lt_cost,
            }
            artifacts.lt_moves = lt_moves
        
        return artifacts

