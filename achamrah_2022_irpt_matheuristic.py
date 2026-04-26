from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Any
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB


Node = int
Product = int
Period = int
Vehicle = int
Arc = Tuple[int, int]
RouteKey = Tuple[int, int]  # (period, vehicle)
Chromosome = Dict[RouteKey, List[int]]


@dataclass
class IRPTInstance:
    """
    Instance for the Achamrah et al. (2022) inventory-routing problem with transshipment.

    Conventions:
    - Node 0 is the central warehouse (CW)
    - POS nodes are 1..n
    - Products are indexed by integers in P
    - Periods are indexed by integers in H, starting from 1
    - Vehicles are indexed by integers in V, starting from 1
    """

    N: List[Node]                 # POS nodes only
    P: List[Product]              # products
    H: List[Period]               # periods
    V: List[Vehicle]              # vehicles

    alpha: float                  # routing cost per km
    Q: float                      # vehicle capacity

    d: Dict[Arc, float]           # distance matrix over N0 x N0 without self arcs
    b: Dict[Tuple[int, int], float]  # transshipment unit cost between POS only

    h: Dict[Tuple[int, int], float]   # holding cost for (p, i), i in N0
    C: Dict[int, float]               # storage capacity by node in N0
    I0: Dict[Tuple[int, int], float]  # initial inventory (p, i)
    D: Dict[Tuple[int, int, int], float]   # demand (p, i, t), i in N
    g: Dict[Tuple[int, int], float]         # replenishment to CW (p, t)
    f: Dict[Tuple[int, int], float]         # lost-sales cost (p, i), i in N

    name: str = "instance"

    @property
    def N0(self) -> List[int]:
        return [0] + list(self.N)

    @property
    def A(self) -> List[Arc]:
        return [(i, j) for i in self.N0 for j in self.N0 if i != j]

    def validate(self) -> None:
        for i in self.N0:
            if i not in self.C:
                raise ValueError(f"Missing capacity C[{i}]")
        for p in self.P:
            for i in self.N0:
                if (p, i) not in self.h:
                    raise ValueError(f"Missing holding cost h[{p},{i}]")
                if (p, i) not in self.I0:
                    raise ValueError(f"Missing initial inventory I0[{p},{i}]")
            for t in self.H:
                if (p, t) not in self.g:
                    raise ValueError(f"Missing g[{p},{t}]")
                for i in self.N:
                    if (p, i, t) not in self.D:
                        raise ValueError(f"Missing demand D[{p},{i},{t}]")
                    if (p, i) not in self.f:
                        raise ValueError(f"Missing lost-sales cost f[{p},{i}]")
        for i, j in self.A:
            if (i, j) not in self.d:
                raise ValueError(f"Missing distance d[{i},{j}]")
        for i in self.N:
            for j in self.N:
                if i != j and (i, j) not in self.b:
                    raise ValueError(f"Missing transshipment cost b[{i},{j}]")


@dataclass
class HeuristicParams:
    initial_temperature: float = 92.0
    final_temperature: float = 4.2
    cooling_ratio: float = 0.96
    crossover_probability: float = 0.84
    mutation_probability: float = 0.37
    population_size: int = 180
    iterations_per_temp: int = 180
    seed: int = 0
    constructive_time_limit: float = 600.0
    improvement_time_limit: float = 1800.0
    full_time_limit: float = 2400.0
    rmilp_mipgap: float = 0.05
    cluster_mipgap: float = 0.02
    fmilp_mipgap: float = 0.02


@dataclass
class SolveArtifacts:
    objective: float
    status: int
    runtime: float
    model: gp.Model
    x_vals: Dict[Tuple[int, int, int, int], float] = field(default_factory=dict)
    routes: Chromosome = field(default_factory=dict)
    vars: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatheuristicResult:
    best_objective: float
    best_routes: Chromosome
    constructive_solution: Optional[SolveArtifacts]
    final_solution: Optional[SolveArtifacts]
    history: List[Tuple[float, float]]
    constructive_runtime_seconds: float = 0.0
    improvement_runtime_seconds: float = 0.0
    total_runtime_seconds: float = 0.0


class AchamrahIRPTSolver:
    """
    Implements the paper's main MILP and the two-phase matheuristic:
    - Constructive phase: RMILP -> clustering -> clustered MILPs
    - Improvement phase: GA/SA hybrid over route chromosomes evaluated by FMILP

    Notes:
    1) Equations (1)-(20) are implemented directly.
    2) Eq. (21) disjoint-route cuts are included as an optional placeholder callback hook,
       because their practical separation is nontrivial and paper-specific.
    3) The code is written for clarity and reproducibility rather than extreme speed.
    """

    def __init__(self, inst: IRPTInstance, params: Optional[HeuristicParams] = None):
        self.inst = inst
        self.inst.validate()
        self.params = params or HeuristicParams()
        self.rng = random.Random(self.params.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve_full_matheuristic(self) -> MatheuristicResult:
        start = time.time()
        constructive = self.constructive_phase()
        constructive_runtime = time.time() - start
        if constructive is None:
            raise RuntimeError("Constructive phase failed to produce a solution.")

        remaining = max(1.0, self.params.full_time_limit - constructive_runtime)
        improvement_start = time.time()
        final_solution, history = self.improvement_phase(
            initial_routes=constructive.routes,
            time_limit=min(self.params.improvement_time_limit, remaining),
        )
        improvement_runtime = time.time() - improvement_start

        if final_solution is None:
            best_obj = constructive.objective
            best_routes = constructive.routes
        else:
            best_obj = final_solution.objective
            best_routes = final_solution.routes

        return MatheuristicResult(
            best_objective=best_obj,
            best_routes=best_routes,
            constructive_solution=constructive,
            final_solution=final_solution,
            history=history,
            constructive_runtime_seconds=constructive_runtime,
            improvement_runtime_seconds=improvement_runtime,
            total_runtime_seconds=time.time() - start,
        )

    # ------------------------------------------------------------------
    # Core MILP builders
    # ------------------------------------------------------------------
    def build_model(
        self,
        relaxed: bool = False,
        fixed_routes: Optional[Chromosome] = None,
        active_nodes_by_period: Optional[Dict[int, List[int]]] = None,
        use_valid_16_19: bool = True,
        use_valid_20: bool = True,
        allow_lateral_transshipment: bool = True,
        add_lazy_21_placeholder: bool = False,
        model_name: str = "IRPT",
    ) -> Tuple[gp.Model, Dict[str, Any]]:
        inst = self.inst
        m = gp.Model(model_name)
        m.Params.OutputFlag = 0

        N = list(inst.N)
        N0 = list(inst.N0)
        P = list(inst.P)
        H = list(inst.H)
        V = list(inst.V)
        A = [(i, j) for i, j in inst.A if self._arc_allowed(i, j, active_nodes_by_period)]

        # Variables
        x_type = GRB.BINARY
        flow_type = GRB.CONTINUOUS if relaxed else GRB.INTEGER

        I = m.addVars(P, N0, H, vtype=flow_type, lb=0.0, name="I")
        Qdir = m.addVars(P, N, H, vtype=flow_type, lb=0.0, name="Q")
        q = m.addVars(P, N0, N0, V, H, vtype=flow_type, lb=0.0, name="q")
        y = m.addVars(P, N, N, V, H, vtype=flow_type, lb=0.0, name="y")
        if not allow_lateral_transshipment:
            for key in y.keys():
                y[key].ub = 0.0
        S = m.addVars(P, N, H, vtype=flow_type, lb=0.0, name="S")
        x = m.addVars(N0, N0, V, H, vtype=x_type, lb=0.0, ub=1.0, name="x")
        u = m.addVars(V, H, vtype=GRB.BINARY, lb=0.0, ub=1.0, name="u")
        z = m.addVars(N0, V, H, vtype=GRB.BINARY, lb=0.0, ub=1.0, name="z")

        # Disable impossible/self arcs explicitly
        for i in N0:
            for j in N0:
                for v in V:
                    for t in H:
                        if i == j or not self._node_pair_allowed(i, j, t, active_nodes_by_period):
                            x[i, j, v, t].ub = 0.0
                            for p in P:
                                q[p, i, j, v, t].ub = 0.0
                                if i in N and j in N:
                                    y[p, i, j, v, t].ub = 0.0

        # If routes are fixed: set x exactly from chromosome
        if fixed_routes is not None:
            fixed_x = self.chromosome_to_x(fixed_routes)
            for i in N0:
                for j in N0:
                    for v in V:
                        for t in H:
                            val = fixed_x.get((i, j, v, t), 0)
                            x[i, j, v, t].lb = val
                            x[i, j, v, t].ub = val

        # Objective (1)
        m.setObjective(
            gp.quicksum(inst.h[p, i] * I[p, i, t] for p in P for i in N0 for t in H)
            + gp.quicksum(inst.alpha * inst.d[i, j] * x[i, j, v, t] for i in N0 for j in N0 if i != j for v in V for t in H)
            + gp.quicksum(inst.b[i, j] * y[p, i, j, v, t] for p in P for i in N for j in N if i != j for v in V for t in H)
            + gp.quicksum(inst.f[p, i] * S[p, i, t] for p in P for i in N for t in H),
            GRB.MINIMIZE,
        )

        # (2) inventory balance at POS
        for p in P:
            for i in N:
                for t in H:
                    prev_I = inst.I0[p, i] if t == H[0] else I[p, i, t - 1]
                    m.addConstr(
                        I[p, i, t]
                        == prev_I
                        + Qdir[p, i, t]
                        - inst.D[p, i, t]
                        + S[p, i, t]
                        + gp.quicksum(y[p, j, i, v, t] - y[p, i, j, v, t] for v in V for j in N if j != i),
                        name=f"bal_pos[{p},{i},{t}]",
                    )

        # (3) inventory balance at CW
        for p in P:
            for t in H:
                prev_I = inst.I0[p, 0] if t == H[0] else I[p, 0, t - 1]
                m.addConstr(
                    I[p, 0, t] == prev_I - gp.quicksum(Qdir[p, i, t] for i in N) + inst.g[p, t],
                    name=f"bal_cw[{p},{t}]",
                )

        # (4) flow conservation linking direct and transshipped flows to arc flows at POS
        for p in P:
            for j in N:
                for t in H:
                    m.addConstr(
                        Qdir[p, j, t]
                        + gp.quicksum(y[p, i, j, v, t] - y[p, j, i, v, t] for v in V for i in N if i != j)
                        == gp.quicksum(q[p, i, j, v, t] - q[p, j, i, v, t] for v in V for i in N0 if i != j),
                        name=f"flow_pos[{p},{j},{t}]",
                    )

        # (5) vehicles return empty to CW
        for i in N0:
            for v in V:
                for t in H:
                    m.addConstr(gp.quicksum(q[p, i, 0, v, t] for p in P) == 0, name=f"empty_return[{i},{v},{t}]")

        # (6) storage capacity
        for i in N0:
            for t in H:
                m.addConstr(gp.quicksum(I[p, i, t] for p in P) <= inst.C[i], name=f"cap_store[{i},{t}]")

        # (7) vehicle capacity on each arc
        for i in N0:
            for j in N0:
                if i == j:
                    continue
                for v in V:
                    for t in H:
                        m.addConstr(gp.quicksum(q[p, i, j, v, t] for p in P) <= inst.Q * u[v, t], name=f"cap_arc[{i},{j},{v},{t}]")

        # (8) transshipment outflow from POS limited by inventory available at start of period
        for p in P:
            for i in N:
                for t in H:
                    if t == H[0]:
                        inv_prev = inst.I0[p, i]
                    else:
                        inv_prev = I[p, i, t - 1]
                    m.addConstr(
                        gp.quicksum(y[p, i, j, v, t] for v in V for j in N if j != i) <= inv_prev,
                        name=f"trans_inv[{p},{i},{t}]",
                    )

        # (9) flow conservation for routing at customer node
        for j in N:
            for v in V:
                for t in H:
                    m.addConstr(
                        gp.quicksum(x[i, j, v, t] for i in N0 if i != j)
                        == gp.quicksum(x[j, i, v, t] for i in N0 if i != j),
                        name=f"route_cons[{j},{v},{t}]",
                    )

        # (10) each POS visited at most once per period
        for j in N:
            for t in H:
                m.addConstr(
                    gp.quicksum(x[i, j, v, t] for i in N0 if i != j for v in V) <= 1,
                    name=f"visit_once[{j},{t}]",
                )

        # (11) vehicle usage iff it leaves CW
        for v in V:
            for t in H:
                m.addConstr(gp.quicksum(x[0, j, v, t] for j in N) == u[v, t], name=f"veh_use[{v},{t}]")

        # (12) fleet availability
        for t in H:
            m.addConstr(gp.quicksum(u[v, t] for v in V) <= len(V), name=f"fleet[{t}]")

        # (13) q only on used arcs between distinct nodes; paper writes i,j in N, but implementation uses N0
        for p in P:
            for i in N0:
                for j in N0:
                    if i == j:
                        continue
                    for v in V:
                        for t in H:
                            m.addConstr(q[p, i, j, v, t] <= inst.Q * x[i, j, v, t], name=f"link_q_x[{p},{i},{j},{v},{t}]")

        # Direct shipments must be represented by product flow on CW -> POS arcs.
        for p in P:
            for j in N:
                for t in H:
                    m.addConstr(
                        Qdir[p, j, t] == gp.quicksum(q[p, 0, j, v, t] for v in V),
                        name=f"direct_from_cw[{p},{j},{t}]",
                    )

        # Lateral transshipment must be carried by product flow on the same POS -> POS arc.
        for p in P:
            for i in N:
                for j in N:
                    if i == j:
                        continue
                    for v in V:
                        for t in H:
                            m.addConstr(y[p, i, j, v, t] <= q[p, i, j, v, t], name=f"link_y_q[{p},{i},{j},{v},{t}]")

        # Additional route consistency: z indicators
        for v in V:
            for t in H:
                m.addConstr(z[0, v, t] == u[v, t], name=f"zdef[0,{v},{t}]")
                for i in N:
                    m.addConstr(z[i, v, t] == gp.quicksum(x[j, i, v, t] for j in N0 if j != i), name=f"zdef[{i},{v},{t}]")

        # Valid inequalities (16)-(19)
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

        # Valid inequality (20)
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
                            lhs = (
                                gp.quicksum(z[i, v, tau] for v in V for tau in range(t1, t2 + 1))
                                + (1.0 / demand_sum)
                                * gp.quicksum(y[p, j, i, v, tau] for j in N if j != i for v in V for tau in range(t1, t2 + 1))
                            )
                            rhs = 1.0 - (1.0 / demand_sum) * inv_before
                            m.addConstr(lhs >= rhs, name=f"v20[{p},{i},{t1},{t2}]")

        # Optional placeholder for Eq. (21) separation
        if add_lazy_21_placeholder:
            m._enable_disjoint_route_cuts = True
            m.Params.LazyConstraints = 1
        else:
            m._enable_disjoint_route_cuts = False

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
        fixed_routes: Optional[Chromosome] = None,
        active_nodes_by_period: Optional[Dict[int, List[int]]] = None,
        time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        allow_lateral_transshipment: bool = True,
        model_name: str = "IRPT",
    ) -> SolveArtifacts:
        m, vars_dict = self.build_model(
            relaxed=relaxed,
            fixed_routes=fixed_routes,
            active_nodes_by_period=active_nodes_by_period,
            allow_lateral_transshipment=allow_lateral_transshipment,
            model_name=model_name,
        )
        if time_limit is not None:
            m.Params.TimeLimit = time_limit
        if mip_gap is not None:
            m.Params.MIPGap = mip_gap
        m.optimize(self._lazy_callback if getattr(m, "_enable_disjoint_route_cuts", False) else None)

        x_vals = {}
        if m.SolCount > 0:
            x = vars_dict["x"]
            for key in x.keys():
                val = x[key].X
                if val > 1e-6:
                    x_vals[key] = val
            routes = self.x_to_chromosome(x_vals)
            obj = m.ObjVal
        else:
            routes = {}
            obj = math.inf

        return SolveArtifacts(
            objective=obj,
            status=m.Status,
            runtime=m.Runtime,
            model=m,
            x_vals=x_vals,
            routes=routes,
            vars=vars_dict,
        )

    # ------------------------------------------------------------------
    # Constructive phase
    # ------------------------------------------------------------------
    def constructive_phase(self) -> Optional[SolveArtifacts]:
        """
        Paper's Phase 1:
        1) Solve RMILP (continuous non-routing variables, binary x/u/z)
        2) If optimal integral route solution good enough, use it directly
        3) Else build clusters from the RMILP routes
        4) Solve subgraph MILPs and merge the routes
        """
        rmilp = self.solve_model(
            relaxed=True,
            time_limit=self.params.constructive_time_limit,
            mip_gap=self.params.rmilp_mipgap,
            model_name="RMILP",
        )

        if rmilp.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and rmilp.routes:
            # If x is integral enough and already feasible, keep directly
            if self._is_route_integral(rmilp.x_vals):
                fixed_eval = self.solve_model(
                    relaxed=False,
                    fixed_routes=rmilp.routes,
                    time_limit=max(30.0, 0.25 * self.params.constructive_time_limit),
                    mip_gap=self.params.cluster_mipgap,
                    model_name="ConstructiveDirectFMILP",
                )
                if fixed_eval.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and math.isfinite(fixed_eval.objective):
                    return fixed_eval

            clusters = self.build_clusters_from_routes(rmilp.routes)
            merged_routes = self.solve_clusters_and_merge(clusters)
            if merged_routes:
                clustered = self.solve_model(
                    relaxed=False,
                    fixed_routes=merged_routes,
                    time_limit=max(30.0, 0.5 * self.params.constructive_time_limit),
                    mip_gap=self.params.cluster_mipgap,
                    model_name="ClusteredFMILP",
                )
                if clustered.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and math.isfinite(clustered.objective):
                    return clustered

        return rmilp if math.isfinite(rmilp.objective) else None

    def build_clusters_from_routes(self, routes: Chromosome) -> Dict[int, List[List[int]]]:
        """
        Build connected components of POS per period from route arcs, as in the paper's constructive phase.
        """
        clusters_by_period: Dict[int, List[List[int]]] = {}
        for t in self.inst.H:
            adj = defaultdict(set)
            used_nodes = set()
            for v in self.inst.V:
                seq = routes.get((t, v), [])
                if not seq:
                    continue
                for node in seq:
                    used_nodes.add(node)
                prev = 0
                for node in seq:
                    if prev != 0:
                        adj[prev].add(node)
                        adj[node].add(prev)
                    prev = node
            # isolated visited nodes also form singleton clusters
            seen = set()
            comps: List[List[int]] = []
            for node in used_nodes:
                if node in seen:
                    continue
                comp = []
                dq = deque([node])
                seen.add(node)
                while dq:
                    u = dq.popleft()
                    comp.append(u)
                    for w in adj[u]:
                        if w not in seen:
                            seen.add(w)
                            dq.append(w)
                comps.append(sorted(comp))
            if not comps:
                comps = [[i] for i in self.inst.N]
            clusters_by_period[t] = comps
        return clusters_by_period

    def solve_clusters_and_merge(self, clusters_by_period: Dict[int, List[List[int]]]) -> Chromosome:
        merged: Chromosome = {(t, v): [] for t in self.inst.H for v in self.inst.V}
        vehicle_cursor = {t: 0 for t in self.inst.H}

        for t in self.inst.H:
            clusters = clusters_by_period.get(t, [])
            for cluster in clusters:
                if vehicle_cursor[t] >= len(self.inst.V):
                    break
                v = self.inst.V[vehicle_cursor[t]]
                vehicle_cursor[t] += 1

                # Restrict to this cluster for this period only
                active_nodes = {tau: list(self.inst.N) for tau in self.inst.H}
                active_nodes[t] = list(cluster)
                sub = self.solve_model(
                    relaxed=False,
                    active_nodes_by_period=active_nodes,
                    time_limit=max(10.0, self.params.constructive_time_limit / max(1, len(clusters_by_period))),
                    mip_gap=self.params.cluster_mipgap,
                    model_name=f"Cluster_t{t}_v{v}",
                )
                cluster_route = self._pick_best_route_for_period(sub.routes, t)
                merged[(t, v)] = cluster_route
        return merged

    # ------------------------------------------------------------------
    # Improvement phase (GA + SA)
    # ------------------------------------------------------------------
    def improvement_phase(self, initial_routes: Chromosome, time_limit: float) -> Tuple[Optional[SolveArtifacts], List[Tuple[float, float]]]:
        start = time.time()
        T = self.params.initial_temperature
        history: List[Tuple[float, float]] = []

        current_routes = self._copy_chromosome(initial_routes)
        current_eval = self.evaluate_routes(current_routes, max(30.0, 0.15 * time_limit))
        if current_eval is None:
            return None, history
        best_routes = self._copy_chromosome(current_routes)
        best_eval = current_eval

        population = self.generate_initial_population(current_routes)

        while T > self.params.final_temperature and (time.time() - start) < time_limit:
            accepted_pool: List[Tuple[Chromosome, float]] = []
            iter_cap = min(self.params.iterations_per_temp, max(1, len(population)))

            for idx in range(iter_cap):
                if (time.time() - start) >= time_limit:
                    break
                candidate = self._copy_chromosome(population[idx % len(population)])
                candidate_eval = self.evaluate_routes(candidate, max(5.0, 0.02 * time_limit))
                if candidate_eval is None:
                    continue

                delta = candidate_eval.objective - current_eval.objective
                if delta < 0:
                    current_routes = candidate
                    current_eval = candidate_eval
                    accepted_pool.append((self._copy_chromosome(candidate), candidate_eval.objective))
                else:
                    prob = math.exp(-delta / max(1e-9, T))
                    if self.rng.random() < prob:
                        current_routes = candidate
                        current_eval = candidate_eval
                        accepted_pool.append((self._copy_chromosome(candidate), candidate_eval.objective))

                if current_eval.objective < best_eval.objective:
                    best_routes = self._copy_chromosome(current_routes)
                    best_eval = current_eval

            history.append((T, best_eval.objective))
            T *= self.params.cooling_ratio

            # Generate a new GA population from accepted solutions; fallback to current best
            seeds = [r for r, _ in sorted(accepted_pool, key=lambda z: z[1])]
            if not seeds:
                seeds = [self._copy_chromosome(best_routes)]
            population = self.ga_next_population(seeds)

        return best_eval, history

    def generate_initial_population(self, base_routes: Chromosome) -> List[Chromosome]:
        population: List[Chromosome] = [self._copy_chromosome(base_routes)]
        seen = set()
        seen.add(self.chromosome_signature(base_routes))
        max_attempts = max(self.params.population_size * 10, 1)
        attempts = 0
        while len(population) < self.params.population_size and attempts < max_attempts:
            attempts += 1
            cand = self.two_opt_neighbor(base_routes)
            sig = self.chromosome_signature(cand)
            if sig in seen:
                continue
            seen.add(sig)
            population.append(cand)
        return population

    def ga_next_population(self, seeds: List[Chromosome]) -> List[Chromosome]:
        pop = [self._copy_chromosome(s) for s in seeds]
        target = self.params.population_size

        # Cloning best 30%
        elite_count = max(1, int(0.30 * target))
        while len(pop) < elite_count:
            pop.append(self._copy_chromosome(seeds[(len(pop)) % len(seeds)]))

        while len(pop) < target:
            p1 = self.binary_tournament(seeds)
            p2 = self.binary_tournament(seeds)
            c1, c2 = self.crossover(p1, p2) if self.rng.random() < self.params.crossover_probability else (self._copy_chromosome(p1), self._copy_chromosome(p2))
            if self.rng.random() < self.params.mutation_probability:
                c1 = self.mutate(c1)
            if self.rng.random() < self.params.mutation_probability:
                c2 = self.mutate(c2)
            pop.append(c1)
            if len(pop) < target:
                pop.append(c2)
        return pop[:target]

    def binary_tournament(self, population: List[Chromosome]) -> Chromosome:
        a, b = self.rng.sample(population, k=2) if len(population) >= 2 else (population[0], population[0])
        ea = self.fast_route_cost(a)
        eb = self.fast_route_cost(b)
        return self._copy_chromosome(a if ea <= eb else b)

    def crossover(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        c1 = self._copy_chromosome(p1)
        c2 = self._copy_chromosome(p2)

        for key in sorted(c1.keys()):
            seq1 = c1[key][:]
            seq2 = c2[key][:]
            n = min(len(seq1), len(seq2))
            if n < 2:
                continue
            i, j = sorted(self.rng.sample(range(n), 2))
            seg1 = seq1[i:j + 1]
            seg2 = seq2[i:j + 1]
            seq1[i:j + 1] = seg2
            seq2[i:j + 1] = seg1
            c1[key] = self.repair_sequence(seq1)
            c2[key] = self.repair_sequence(seq2)

        c1 = self.repair_chromosome(c1)
        c2 = self.repair_chromosome(c2)
        return c1, c2

    def mutate(self, chrom: Chromosome) -> Chromosome:
        out = self._copy_chromosome(chrom)
        keys = [k for k, seq in out.items() if len(seq) >= 2]
        if not keys:
            return out
        key = self.rng.choice(keys)
        seq = out[key][:]
        i, j = sorted(self.rng.sample(range(len(seq)), 2))
        seq[i:j + 1] = reversed(seq[i:j + 1])
        out[key] = seq
        return self.repair_chromosome(out)

    def two_opt_neighbor(self, chrom: Chromosome) -> Chromosome:
        out = self._copy_chromosome(chrom)
        candidate_keys = [k for k, seq in out.items() if len(seq) >= 2]
        if not candidate_keys:
            return out
        key = self.rng.choice(candidate_keys)
        seq = out[key][:]
        best_seq = seq[:]
        best_cost = self.route_cost(best_seq)

        # intra-route 2-opt / reversal
        for _ in range(min(10, len(seq) * len(seq))):
            i, j = sorted(self.rng.sample(range(len(seq)), 2))
            new_seq = seq[:]
            new_seq[i:j + 1] = reversed(new_seq[i:j + 1])
            c = self.route_cost(new_seq)
            if c < best_cost:
                best_seq, best_cost = new_seq, c

        out[key] = best_seq

        # optional inter-route swap in same period
        same_period_keys = [k for k in out.keys() if k[0] == key[0] and len(out[k]) >= 1]
        if len(same_period_keys) >= 2 and self.rng.random() < 0.5:
            k1, k2 = self.rng.sample(same_period_keys, 2)
            i1 = self.rng.randrange(len(out[k1]))
            i2 = self.rng.randrange(len(out[k2]))
            out[k1][i1], out[k2][i2] = out[k2][i2], out[k1][i1]

        return self.repair_chromosome(out)

    def evaluate_routes(self, routes: Chromosome, time_limit: float) -> Optional[SolveArtifacts]:
        sol = self.solve_model(
            relaxed=False,
            fixed_routes=routes,
            time_limit=time_limit,
            mip_gap=self.params.fmilp_mipgap,
            model_name="FMILP",
        )
        if sol.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and math.isfinite(sol.objective):
            return sol
        return None

    # ------------------------------------------------------------------
    # Route / chromosome utilities
    # ------------------------------------------------------------------
    def x_to_chromosome(self, x_vals: Dict[Tuple[int, int, int, int], float]) -> Chromosome:
        routes: Chromosome = {(t, v): [] for t in self.inst.H for v in self.inst.V}
        for t in self.inst.H:
            for v in self.inst.V:
                succ = {}
                starts = []
                for i in self.inst.N0:
                    for j in self.inst.N0:
                        if i == j:
                            continue
                        if x_vals.get((i, j, v, t), 0.0) > 0.5:
                            succ[i] = j
                            if i == 0:
                                starts.append(j)
                if not starts:
                    continue
                # Follow the route from depot 0 until return to 0
                cur = starts[0]
                visited = set()
                while cur != 0 and cur not in visited:
                    visited.add(cur)
                    routes[(t, v)].append(cur)
                    cur = succ.get(cur, 0)
        return routes

    def chromosome_to_x(self, routes: Chromosome) -> Dict[Tuple[int, int, int, int], int]:
        x_vals: Dict[Tuple[int, int, int, int], int] = {}
        for t in self.inst.H:
            for v in self.inst.V:
                seq = routes.get((t, v), [])
                if not seq:
                    continue
                prev = 0
                for node in seq:
                    x_vals[(prev, node, v, t)] = 1
                    prev = node
                x_vals[(prev, 0, v, t)] = 1
        return x_vals

    def chromosome_signature(self, chrom: Chromosome) -> Tuple[Tuple[RouteKey, Tuple[int, ...]], ...]:
        return tuple(sorted((k, tuple(v)) for k, v in chrom.items()))

    def repair_sequence(self, seq: List[int]) -> List[int]:
        seen = set()
        out = []
        for x in seq:
            if x in self.inst.N and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def repair_chromosome(self, chrom: Chromosome) -> Chromosome:
        """
        Ensure each node appears at most once per period across all vehicles.
        Unassigned nodes are allowed because the paper's routes are not required to visit all POS.
        """
        out = self._copy_chromosome(chrom)
        for t in self.inst.H:
            seen = set()
            for v in self.inst.V:
                cleaned = []
                for node in out.get((t, v), []):
                    if node not in seen:
                        cleaned.append(node)
                        seen.add(node)
                out[(t, v)] = cleaned
        return out

    def _pick_best_route_for_period(self, routes: Chromosome, period: int) -> List[int]:
        best_seq = []
        best_cost = math.inf
        for v in self.inst.V:
            seq = routes.get((period, v), [])
            c = self.route_cost(seq)
            if seq and c < best_cost:
                best_cost = c
                best_seq = seq[:]
        return best_seq

    def route_cost(self, seq: List[int]) -> float:
        if not seq:
            return 0.0
        cost = self.inst.alpha * self.inst.d[0, seq[0]]
        for a, b in zip(seq[:-1], seq[1:]):
            cost += self.inst.alpha * self.inst.d[a, b]
        cost += self.inst.alpha * self.inst.d[seq[-1], 0]
        return cost

    def fast_route_cost(self, chrom: Chromosome) -> float:
        return sum(self.route_cost(seq) for seq in chrom.values())

    def _copy_chromosome(self, chrom: Chromosome) -> Chromosome:
        return {k: v[:] for k, v in chrom.items()}

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def _arc_allowed(self, i: int, j: int, active_nodes_by_period: Optional[Dict[int, List[int]]]) -> bool:
        return i != j

    def _node_pair_allowed(self, i: int, j: int, t: int, active_nodes_by_period: Optional[Dict[int, List[int]]]) -> bool:
        if i == j:
            return False
        if active_nodes_by_period is None:
            return True
        active = set(active_nodes_by_period.get(t, self.inst.N))
        if i == 0:
            return j in active
        if j == 0:
            return i in active
        return i in active and j in active

    def _is_route_integral(self, x_vals: Dict[Tuple[int, int, int, int], float], tol: float = 1e-6) -> bool:
        return all(abs(v - round(v)) <= tol for v in x_vals.values())

    def _lazy_callback(self, model: gp.Model, where: int) -> None:
        # Placeholder only: Eq. (21) separation is highly specialized and instance-dependent.
        # Hook left here so you can add cut separation later without changing the outer API.
        return


# ----------------------------------------------------------------------
# Simple synthetic instance generator and CLI demo
# ----------------------------------------------------------------------

def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return round(math.dist(a, b))


def generate_demo_instance(
    n_pos: int = 8,
    n_products: int = 2,
    n_periods: int = 4,
    n_vehicles: int = 3,
    seed: int = 0,
) -> IRPTInstance:
    rng = random.Random(seed)
    N = list(range(1, n_pos + 1))
    P = list(range(n_products))
    H = list(range(1, n_periods + 1))
    V = list(range(1, n_vehicles + 1))

    coords = {0: (50.0, 50.0)}
    for i in N:
        coords[i] = (rng.uniform(0, 100), rng.uniform(0, 100))

    d = {}
    for i in [0] + N:
        for j in [0] + N:
            if i != j:
                d[i, j] = max(1.0, euclidean_distance(coords[i], coords[j]))

    b = {(i, j): 0.01 * d[i, j] for i in N for j in N if i != j}
    h = {(p, i): rng.uniform(0.03, 0.2) for p in P for i in [0] + N}
    C = {0: 5000.0}
    C.update({i: rng.uniform(80, 150) for i in N})
    I0 = {(p, 0): rng.uniform(400, 700) for p in P}
    for p in P:
        for i in N:
            I0[p, i] = rng.uniform(5, 30)
    D = {(p, i, t): rng.uniform(5, 25) for p in P for i in N for t in H}
    g = {(p, t): rng.uniform(150, 250) for p in P for t in H}
    f = {(p, i): 200 * h[p, i] for p in P for i in N}

    return IRPTInstance(
        N=N,
        P=P,
        H=H,
        V=V,
        alpha=1.0,
        Q=120.0,
        d=d,
        b=b,
        h=h,
        C=C,
        I0=I0,
        D=D,
        g=g,
        f=f,
        name="demo_irpt",
    )


def main() -> None:
    inst = generate_demo_instance()
    params = HeuristicParams(
        initial_temperature=92,
        final_temperature=4.2,
        cooling_ratio=0.96,
        crossover_probability=0.84,
        mutation_probability=0.37,
        population_size=30,
        iterations_per_temp=30,
        constructive_time_limit=20,
        improvement_time_limit=30,
        full_time_limit=60,
        seed=0,
    )
    solver = AchamrahIRPTSolver(inst, params)
    result = solver.solve_full_matheuristic()
    print(f"Best objective: {result.best_objective:.3f}")
    for key in sorted(result.best_routes):
        print(key, result.best_routes[key])
    print("History:")
    for T, obj in result.history[:10]:
        print(f"  T={T:.2f}, best={obj:.3f}")


if __name__ == "__main__":
    main()
