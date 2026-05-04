"""
test_2scenario_baseline.py
===========================
Quick 2-scenario local test before running all 30.
  Small: 4 stores, 3 products, 2 vehicles, 60 periods
  Large: 7 stores, 5 products, 2 vehicles, 60 periods

Compares: Gurobi (exact) | ALNS | pure-heuristic GA/SA
Metrics:  Objective value, Runtime, Service Level (%), gap vs Gurobi
"""

from __future__ import annotations
import math
import time
from collections import defaultdict
from typing import Dict, Tuple

from irp_gurobi_converted import (
    DatasetToIRPValidationMapper,
    BaselineALNSModel,
    AchamrahFullIRPTModel,
    IRPData,
    _detect_period_window,
)
from achamrah_ga_sa_baseline import (
    BaselineInstance,
    BaselineGASASolver,
    GASAParams,
)
from achamrah_matheuristic import GASAMatheuristicSolver

DATASET = "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
SEED = 42

# ──────────────────────────────────────────────────────────────────
# Scenarios to test
# ──────────────────────────────────────────────────────────────────
SCENARIOS = [
    {"label": "small_4s3p_60d", "stores": 4, "products": 3, "vehicles": 2, "periods": 60},
    {"label": "large_7s5p_60d", "stores": 7, "products": 5, "vehicles": 2, "periods": 60},
]


# ──────────────────────────────────────────────────────────────────
# Adapter: IRPData → BaselineInstance  (GA/SA input)
# ──────────────────────────────────────────────────────────────────
def _to_gasa_instance(data: IRPData):
    """Returns (BaselineInstance, s_id, t_id, v_id) — index maps exposed for route conversion."""
    stores   = list(data.stores)
    products = list(data.products)
    periods  = list(data.periods)
    vehicles = list(data.vehicles) if data.vehicles else [1, 2]

    s_id = {s: i + 1 for i, s in enumerate(stores)}
    p_id = {p: i + 1 for i, p in enumerate(products)}
    t_id = {t: i + 1 for i, t in enumerate(periods)}

    N  = list(s_id.values())
    P  = list(p_id.values())
    H  = list(t_id.values())
    V  = list(range(1, len(vehicles) + 1))
    N0 = [0] + N

    d: Dict[Tuple[int, int], float] = {}
    for i in N0:
        for j in N0:
            if i == j:
                continue
            si = stores[i - 1] if i > 0 else data.warehouse
            sj = stores[j - 1] if j > 0 else data.warehouse
            raw = data.distance.get((si, sj), data.distance.get((sj, si), None))
            d[(i, j)] = float(raw) if raw is not None else 1.0

    h: Dict[Tuple[int, int], float] = {}
    for p in products:
        h[(p_id[p], 0)] = float(data.holding_cost_wh.get(p, 0.0))
        for s in stores:
            h[(p_id[p], s_id[s])] = float(data.holding_cost_store.get((s, p), 0.0))

    C: Dict[int, float] = {
        0: float(data.max_inventory_wh.get(products[0], 1e9)) * len(products)
    }
    for s in stores:
        C[s_id[s]] = sum(
            float(data.max_inventory_store.get((s, p), 1e9)) for p in products
        )

    I0: Dict[Tuple[int, int], float] = {}
    for p in products:
        I0[(p_id[p], 0)] = float(data.init_inventory_wh.get(p, 0.0))
        for s in stores:
            I0[(p_id[p], s_id[s])] = float(data.init_inventory_store.get((s, p), 0.0))

    D: Dict[Tuple[int, int, int], float] = {}
    for s in stores:
        for p in products:
            for t in periods:
                D[(p_id[p], s_id[s], t_id[t])] = float(data.demand.get((s, p, t), 0.0))

    g: Dict[Tuple[int, int], float] = {}
    for p in products:
        for t in periods:
            g[(p_id[p], t_id[t])] = float(data.replenishment_wh.get((p, t), 0.0))

    f: Dict[Tuple[int, int], float] = {}
    for p in products:
        for s in stores:
            f[(p_id[p], s_id[s])] = float(data.shortage_cost.get((s, p), 0.25))

    ship: Dict[Tuple[int, int], float] = {}
    for p in products:
        for s in stores:
            ship[(p_id[p], s_id[s])] = float(data.ship_cost_cw.get((s, p), 0.0))

    inst = BaselineInstance(
        N=N, P=P, H=H, V=V,
        alpha=float(data.alpha) if data.alpha else 1.0,
        Q=float(data.vehicle_capacity),
        d=d, h=h, C=C, I0=I0, D=D, g=g, f=f,
        ship_cost_cw=ship,
        vehicle_fixed_cost=float(data.vehicle_fixed_cost),
    )
    v_id = {v: i + 1 for i, v in enumerate(vehicles)}
    return inst, s_id, t_id, v_id


# ──────────────────────────────────────────────────────────────────
# Convert Gurobi solution routes → GA/SA chromosome (int-indexed)
# ──────────────────────────────────────────────────────────────────
def _gurobi_to_gasa_routes(gur_sol, data: IRPData,
                            s_id: Dict, t_id: Dict, v_id: Dict) -> dict:
    """
    Extract route visit order from gur_sol.x (arc activations) and convert
    from (string store names, date periods) to (int store idx, int period idx)
    so it matches the GA/SA Chromosome = Dict[(t_idx, v_idx) → List[store_idx]].
    """
    wh = data.warehouse
    # Build arc map per (vehicle, period) — include ALL active arcs
    arcs_by_vt: Dict = {}
    for (i, j, v, t), val in gur_sol.x.items():
        if float(val) > 0.5:
            arcs_by_vt.setdefault((v, t), []).append((i, j))

    # Pre-populate all (t_idx, v_idx) with empty lists so crossover never KeyErrors
    chromosome: dict = {
        (t_id[t], v_id[v]): []
        for t in data.periods if t in t_id
        for v in (data.vehicles if data.vehicles else list(v_id.keys())) if v in v_id
    }
    for (v, t), arcs in arcs_by_vt.items():
        if v not in v_id or t not in t_id:
            continue
        next_map = {i: j for i, j in arcs}
        route = []
        cur = wh
        visited = {cur}
        while cur in next_map:
            nxt = next_map[cur]
            if nxt == wh or nxt in visited:
                break
            if nxt in s_id:
                route.append(s_id[nxt])
            visited.add(nxt)
            cur = nxt
        if route:
            chromosome[(t_id[t], v_id[v])] = route
    return chromosome


# ──────────────────────────────────────────────────────────────────
# Service level:  1 - total_shortage / total_demand
# (forward inventory simulation from solution deliveries)
# ──────────────────────────────────────────────────────────────────
def _service_level(data: IRPData, deliv: dict) -> float:
    """deliv keys may be (s,p,v,t) or (s,p,t) — both accepted."""
    agg: Dict[Tuple, float] = defaultdict(float)
    for k, qty in deliv.items():
        if len(k) == 4:
            s, p, _, t = k
        else:
            s, p, t = k
        agg[(s, p, t)] += float(qty)

    inv = {
        (s, p): float(data.init_inventory_store.get((s, p), 0.0))
        for s in data.stores
        for p in data.products
    }
    total_demand = total_shortage = 0.0
    for t in sorted(data.periods):
        for s in data.stores:
            for p in data.products:
                dem  = float(data.demand.get((s, p, t), 0.0))
                recv = agg.get((s, p, t), 0.0)
                avail = inv[(s, p)] + recv
                shortage = max(0.0, dem - avail)
                inv[(s, p)] = max(0.0, avail - dem)
                total_demand   += dem
                total_shortage += shortage

    return 1.0 - total_shortage / max(1.0, total_demand)


# ──────────────────────────────────────────────────────────────────
# Build IRPData for one scenario config
# ──────────────────────────────────────────────────────────────────
def _build_data(stores: int, products: int, vehicles: int, periods: int) -> IRPData:
    sd, ed = _detect_period_window(DATASET, periods, None)
    mapper = DatasetToIRPValidationMapper(
        excel_path=DATASET, store_limit=stores, sku_limit=products,
        start_date=sd, end_date=ed,
    )
    data, _, _, _ = mapper.build_irp_data(
        cw_replenishment_factor=0.8, holding_cost_rate=0.01,
        shortage_cost_rate=0.25, wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2, store_initial_inventory_multiplier=0.2,
        vehicle_count=vehicles, vehicle_fixed_cost=50.0,
        alpha=1.0, cw_capacity_factor=2.0,
    )
    disp = [t for t in data.periods if (t - min(data.periods)) % 5 == 0]
    data.vehicle_capacity = max(
        500.0,
        round(sum(data.demand.values()) / max(1, len(disp)) / vehicles * 0.85),
    )
    return data


# ──────────────────────────────────────────────────────────────────
# Run one scenario
# ──────────────────────────────────────────────────────────────────
def run_scenario(cfg: dict) -> None:
    label    = cfg["label"]
    stores   = cfg["stores"]
    products = cfg["products"]
    vehicles = cfg["vehicles"]
    periods  = cfg["periods"]

    print(f"\n{'='*72}")
    print(f"  Scenario: {label}  ({stores}s × {products}p × {vehicles}v × {periods}d)")
    print(f"{'='*72}")

    data = _build_data(stores, products, vehicles, periods)
    n_dispatch = len([t for t in data.periods if (t - min(data.periods)) % 5 == 0])
    total_dem  = sum(data.demand.values())
    print(f"  VehCap={data.vehicle_capacity:.0f}  TotalDemand={total_dem:.0f}  "
          f"Dispatches={n_dispatch}")

    results = {}

    # ── Gurobi ────────────────────────────────────────────────────
    print("\n  [Gurobi] solving (no time limit) ...")
    t0 = time.perf_counter()
    gur = AchamrahFullIRPTModel(data).solve(msg=False, allow_lateral_transshipment=False)
    gur_rt = time.perf_counter() - t0
    gur_sl = _service_level(data, gur.deliv)
    results["Gurobi"] = {"obj": gur.objective, "rt": gur_rt, "sl": gur_sl}
    print(f"    obj={gur.objective:.0f}  rt={gur_rt:.1f}s  SL={gur_sl*100:.2f}%  status={gur.status}")

    # ── ALNS ──────────────────────────────────────────────────────
    print("\n  [ALNS] solving (1 000 iter) ...")
    t0 = time.perf_counter()
    alns = BaselineALNSModel(data).solve(
        msg=False, max_iterations=1000, seed=SEED,
        allow_lateral_transshipment=False,
    )
    alns_rt = time.perf_counter() - t0
    alns_sl = _service_level(data, alns.deliv)
    results["ALNS"] = {"obj": alns.objective, "rt": alns_rt, "sl": alns_sl}
    print(f"    obj={alns.objective:.0f}  rt={alns_rt:.1f}s  SL={alns_sl*100:.2f}%")

    # ── GA/SA (Gurobi warm-start routes + SA improvement) ─────────
    print("\n  [GA/SA] solving (Gurobi initial routes + 180s SA) ...")
    inst, s_id, t_id, v_id = _to_gasa_instance(data)
    gur_routes = _gurobi_to_gasa_routes(gur, data, s_id, t_id, v_id)
    n_warm = sum(1 for v in gur_routes.values() if v)
    print(f"    warm-start routes extracted: {n_warm} non-empty (t,v) pairs")
    params = GASAParams(seed=SEED, population_size=30,
                        iterations_per_temp=30, time_limit=180.0,
                        max_iterations=10_000_000)
    t0 = time.perf_counter()
    ga = BaselineGASASolver(inst, params).solve(initial_routes=gur_routes)
    ga_rt = time.perf_counter() - t0

    gasa_sl = 1.0 - ga.shortage_qty / max(1.0, total_dem)
    results["GA/SA"] = {"obj": ga.objective, "rt": ga_rt, "sl": gasa_sl}
    print(f"    obj={ga.objective:.0f}  rt={ga_rt:.1f}s  SL={gasa_sl*100:.2f}%  "
          f"new_best={ga.new_best_count}")

    # ── GA/SA + inner Gurobi LP (matheuristic) ────────────────────
    # Time limit: 60s total, inner LP cap 5s per call
    # Fewer iterations since each call is ~10-100× more expensive
    MATH_TIME_LIMIT = 60.0
    print(f"\n  [Matheuristic] GA/SA + inner Gurobi LP ({MATH_TIME_LIMIT:.0f}s budget) ...")
    math_params = GASAParams(seed=SEED, population_size=30,
                             iterations_per_temp=30, time_limit=MATH_TIME_LIMIT,
                             max_iterations=10_000_000)
    math_solver = GASAMatheuristicSolver(inst, math_params, inner_lp_time_limit=5.0)
    t0 = time.perf_counter()
    math_ga = math_solver.solve(initial_routes=gur_routes)
    math_rt = time.perf_counter() - t0
    math_sl = 1.0 - math_ga.shortage_qty / max(1.0, total_dem)
    avg_lp_t = math_solver._lp_total_time / max(1, math_solver._lp_calls)
    results["Matheuristic"] = {"obj": math_ga.objective, "rt": math_rt, "sl": math_sl}
    print(f"    obj={math_ga.objective:.0f}  rt={math_rt:.1f}s  SL={math_sl*100:.2f}%  "
          f"new_best={math_ga.new_best_count}  "
          f"lp_calls={math_solver._lp_calls}  avg_lp={avg_lp_t*1000:.1f}ms")

    # ── Summary table ─────────────────────────────────────────────
    ref = results["Gurobi"]["obj"]
    print(f"\n  {'Method':<14} {'Objective':>14} {'Runtime':>9} {'SL%':>8} {'Gap%':>8}")
    print(f"  {'-'*57}")
    for method, r in results.items():
        gap = (r["obj"] - ref) / abs(ref) * 100 if abs(ref) > 1e-9 else float("nan")
        gap_s = f"{gap:+.2f}%" if method != "Gurobi" else "  —"
        print(f"  {method:<14} {r['obj']:>14,.0f} {r['rt']:>8.1f}s "
              f"{r['sl']*100:>7.2f}% {gap_s:>8}")



# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for cfg in SCENARIOS:
        run_scenario(cfg)

    print("\n\nDone. Review output and confirm before running all 30 scenarios.")
