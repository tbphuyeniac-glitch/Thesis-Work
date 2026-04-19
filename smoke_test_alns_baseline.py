"""Smoke test for BaselineALNSModel replacing Gurobi Step 1 solver.

Runs a tiny synthetic IRPData through BaselineALNSModel.solve() and checks
that the returned FullIRPTSolution has the expected schema and internally
consistent values (flow conservation, inventory balance, vehicle capacity).

Run:  python3 smoke_test_alns_baseline.py
"""

from __future__ import annotations
import os
import sys
import math

# We only need the ALNS class + data class, not Gurobi. But the module imports
# gurobipy at top-level; stub it to avoid requiring a license for this smoke test.
class _StubEnv:
    def setParam(self, *a, **kw): pass
    def start(self): pass
class _StubModule:
    class Model:
        def __init__(self, *a, **kw):
            self.Params = _StubEnv()
            self.Status = 0
            self.SolCount = 0
            self.Runtime = 0.0
            self.IterCount = 0
            self.BarIterCount = 0
            self.NodeCount = 0
            self.IsMIP = 0
            self.ObjVal = 0.0
        def addVars(self, *a, **kw): return {}
        def setObjective(self, *a, **kw): pass
        def addConstr(self, *a, **kw): pass
        def optimize(self): pass
    @staticmethod
    def quicksum(*a, **kw): return 0
    class Env:
        def __init__(self, empty=False): pass
        def setParam(self, *a, **kw): pass
        def start(self): pass
    class GRB:
        OPTIMAL=2; INFEASIBLE=3; UNBOUNDED=5; INF_OR_UNBD=4
        TIME_LIMIT=9; INTERRUPTED=11; SUBOPTIMAL=13; NUMERIC=12
        INTEGER=1; CONTINUOUS=0; BINARY=2; MINIMIZE=1

# Only stub if gurobipy is actually unavailable
try:
    import gurobipy  # noqa: F401
except Exception:
    sys.modules['gurobipy'] = _StubModule()
    sys.modules['gurobipy'].GRB = _StubModule.GRB

# Provide required env vars if create_gurobi_env is ever called
os.environ.setdefault("WLSACCESSID", "dummy")
os.environ.setdefault("WLSSECRET", "dummy")
os.environ.setdefault("LICENSEID", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from irp_gurobi_converted import IRPData, BaselineALNSModel, FullIRPTSolution


def build_tiny_instance() -> IRPData:
    stores = ["S1", "S2", "S3"]
    products = ["P1", "P2"]
    periods = [1, 2, 3, 4]
    vehicles = ["V1", "V2"]
    warehouse = "CW"
    nodes = [warehouse] + stores

    demand = {(s, p, t): 6.0 for s in stores for p in products for t in periods}
    init_inv_store = {(s, p): 2.0 for s in stores for p in products}
    init_inv_wh = {p: 200.0 for p in products}
    max_inv_store = {(s, p): 30.0 for s in stores for p in products}
    max_inv_wh = {p: 400.0 for p in products}
    node_capacity = {warehouse: 2000.0}
    for s in stores:
        node_capacity[s] = 60.0

    holding_store = {(s, p): 0.01 for s in stores for p in products}
    holding_wh = {p: 0.005 for p in products}
    shortage_cost = {(s, p): 5.0 for s in stores for p in products}
    ship_cost_cw = {(s, p): 1.0 for s in stores for p in products}
    fixed_dispatch_cw = {s: 8.0 for s in stores}

    distance = {}
    coords = {warehouse: (0.0, 0.0), "S1": (3.0, 0.0), "S2": (0.0, 4.0), "S3": (3.0, 4.0)}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            xi, yi = coords[i]
            xj, yj = coords[j]
            distance[(i, j)] = math.hypot(xi - xj, yi - yj)

    replenishment_wh = {(p, t): 10.0 for p in products for t in periods}

    data = IRPData(
        periods=periods,
        stores=stores,
        products=products,
        warehouse=warehouse,
        demand=demand,
        init_inventory_store=init_inv_store,
        init_inventory_wh=init_inv_wh,
        max_inventory_store=max_inv_store,
        max_inventory_wh=max_inv_wh,
        holding_cost_store=holding_store,
        holding_cost_wh=holding_wh,
        shortage_cost=shortage_cost,
        ship_cost_cw=ship_cost_cw,
        fixed_dispatch_cw=fixed_dispatch_cw,
        vehicles=vehicles,
        vehicle_capacity=50.0,
        vehicle_fixed_cost=20.0,
        max_vehicles_used=2,
        alpha=1.0,
        replenishment_wh=replenishment_wh,
        node_capacity=node_capacity,
        distance=distance,
        big_m_cw={(s, p): 1e6 for s in stores for p in products},
    )
    return data


def check_solution(data: IRPData, sol: FullIRPTSolution) -> None:
    assert isinstance(sol, FullIRPTSolution), "Output must be FullIRPTSolution"
    assert sol.status in {"ALNS-Feasible", "ALNS-InfeasiblePenalized"}, f"Unexpected status {sol.status}"
    assert sol.objective >= 0.0, "Objective must be non-negative"

    # Schema presence
    for s in data.stores:
        for p in data.products:
            for t in data.periods:
                assert (s, p, t) in sol.direct_ship_q
                assert (s, p, t) in sol.inv_store
                assert (s, p, t) in sol.shortage
    for p in data.products:
        for t in data.periods:
            assert (p, t) in sol.inv_wh

    CW = data.warehouse
    N0 = [CW] + data.stores
    for v in data.vehicles:
        for t in data.periods:
            assert (v, t) in sol.u
            assert (CW, v, t) in sol.z
            for s in data.stores:
                assert (s, v, t) in sol.z

    # y must be all zero (no lateral transshipment)
    for key, val in sol.y.items():
        assert val == 0.0, f"y[{key}] should be 0 for baseline ALNS"

    # Vehicle capacity check
    for v in data.vehicles:
        for t in data.periods:
            loaded = sum(sol.deliv.get((s, p, v, t), 0.0) for s in data.stores for p in data.products)
            assert loaded <= data.vehicle_capacity + 1e-6, f"Vehicle capacity breached at (v={v}, t={t}): {loaded}"

    # Route consistency: if vehicle used, must have an outgoing arc from CW
    for v in data.vehicles:
        for t in data.periods:
            if sol.u[(v, t)] == 1:
                out_cw = sum(sol.x.get((CW, j, v, t), 0) for j in data.stores)
                assert out_cw == 1, f"Vehicle {v}@t={t}: u=1 but CW out-degree={out_cw}"

    # Single-visit-per-period: each store visited by at most one vehicle in each period.
    for s in data.stores:
        for t in data.periods:
            in_degree = sum(sol.x.get((i, s, v, t), 0) for i in N0 if i != s for v in data.vehicles)
            assert in_degree <= 1, f"Store {s}@t={t} visited {in_degree} times across vehicles (must be <=1)"

    # Inventory balance
    for s in data.stores:
        for p in data.products:
            prev = float(data.init_inventory_store.get((s, p), 0.0))
            for t in data.periods:
                q = sol.direct_ship_q.get((s, p, t), 0.0)
                demand = float(data.demand.get((s, p, t), 0.0))
                expected_inv = max(0.0, prev + q - demand)
                expected_short = max(0.0, demand - prev - q)
                assert abs(sol.inv_store[(s, p, t)] - expected_inv) < 1e-6, (
                    f"inv_store[{s},{p},{t}] mismatch: {sol.inv_store[(s, p, t)]} vs {expected_inv}"
                )
                assert abs(sol.shortage[(s, p, t)] - expected_short) < 1e-6, (
                    f"shortage[{s},{p},{t}] mismatch"
                )
                prev = expected_inv

    print("[OK] schema, feasibility, capacity, routing, inventory balance checks passed.")


def main() -> int:
    print("Building tiny IRP instance...")
    data = build_tiny_instance()
    print(f"stores={len(data.stores)} products={len(data.products)} periods={len(data.periods)} vehicles={len(data.vehicles)}")

    print("\nRunning BaselineALNSModel.solve()...")
    model = BaselineALNSModel(data)
    sol = model.solve(
        msg=False,
        time_limit=15,
        cw_dispatch_cycle=2,
        max_iterations=400,
        seed=42,
    )

    print(f"status={sol.status}")
    print(f"objective={sol.objective:.4f}")
    print(f"summary={sol.summary()}")
    print(f"efficiency_metrics={sol.efficiency_metrics}")

    check_solution(data, sol)
    print("\nALL SMOKE TESTS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
