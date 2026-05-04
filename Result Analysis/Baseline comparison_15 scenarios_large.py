"""
Baseline comparison_15 scenarios_large.py
==========================================
30-scenario baseline benchmark: 15 large scenarios
Compares 3 methods: Gurobi (exact) | ALNS | GA/SA (pure heuristic)
Output: CSV, JSON, console summary

Metrics:  Objective, Runtime, Service Level (%), Gap vs Gurobi
"""

from __future__ import annotations
import json
import csv
import math
import time
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict

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


DATASET = "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
SEED = 42

# 15 large scenarios: 7-12 stores, 4-7 products, 2-4 vehicles, 60-120 periods
SCENARIOS_LARGE = [
    {"label": "L01", "stores": 7, "products": 4, "vehicles": 2, "periods": 60},
    {"label": "L02", "stores": 8, "products": 5, "vehicles": 3, "periods": 60},
    {"label": "L03", "stores": 9, "products": 5, "vehicles": 3, "periods": 60},
    {"label": "L04", "stores": 9, "products": 6, "vehicles": 3, "periods": 60},
    {"label": "L05", "stores": 10, "products": 5, "vehicles": 3, "periods": 60},
    {"label": "L06", "stores": 10, "products": 6, "vehicles": 3, "periods": 60},
    {"label": "L07", "stores": 10, "products": 7, "vehicles": 3, "periods": 60},
    {"label": "L08", "stores": 12, "products": 5, "vehicles": 3, "periods": 60},
    {"label": "L09", "stores": 8, "products": 5, "vehicles": 3, "periods": 90},
    {"label": "L10", "stores": 9, "products": 5, "vehicles": 3, "periods": 90},
    {"label": "L11", "stores": 10, "products": 5, "vehicles": 3, "periods": 90},
    {"label": "L12", "stores": 10, "products": 6, "vehicles": 4, "periods": 90},
    {"label": "L13", "stores": 12, "products": 5, "vehicles": 3, "periods": 90},
    {"label": "L14", "stores": 12, "products": 6, "vehicles": 3, "periods": 120},
    {"label": "L15", "stores": 12, "products": 7, "vehicles": 4, "periods": 120},
]


@dataclass
class BenchmarkResult:
    scenario: str
    method: str
    objective: float
    runtime: float
    service_level: float
    gap_cost: Optional[float] = None
    gap_time: Optional[float] = None


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


def _gurobi_to_gasa_routes(gur_sol, data: IRPData,
                            s_id: Dict, t_id: Dict, v_id: Dict) -> dict:
    """
    Extract route visit order from gur_sol.x (arc activations) and convert
    to (int_t_idx, int_v_idx) → List[store_idx].
    """
    wh = data.warehouse
    arcs_by_vt: Dict = {}
    for (i, j, v, t), val in gur_sol.x.items():
        if float(val) > 0.5:
            arcs_by_vt.setdefault((v, t), []).append((i, j))

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


def run_scenario(cfg: dict) -> List[BenchmarkResult]:
    """Run one scenario across 3 methods, return list of BenchmarkResult."""
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

    results = []

    # ── Gurobi ────────────────────────────────────────────────────
    print("\n  [Gurobi] solving (no time limit) ...")
    t0 = time.perf_counter()
    gur = AchamrahFullIRPTModel(data).solve(msg=False, allow_lateral_transshipment=False)
    gur_rt = time.perf_counter() - t0
    gur_sl = _service_level(data, gur.deliv)
    results.append(BenchmarkResult(
        scenario=label, method="Gurobi",
        objective=gur.objective, runtime=gur_rt, service_level=gur_sl,
        gap_cost=0.0, gap_time=0.0
    ))
    print(f"    obj={gur.objective:.0f}  rt={gur_rt:.1f}s  SL={gur_sl*100:.2f}%  status={gur.status}")

    # ── ALNS ──────────────────────────────────────────────────────
    print("\n  [ALNS] solving (1,000 iterations) ...")
    t0 = time.perf_counter()
    alns = BaselineALNSModel(data).solve(
        msg=False, max_iterations=1000, seed=SEED,
        allow_lateral_transshipment=False,
    )
    alns_rt = time.perf_counter() - t0
    alns_sl = _service_level(data, alns.deliv)
    gap_cost = (alns.objective - gur.objective) / abs(gur.objective) * 100 if abs(gur.objective) > 1e-9 else 0.0
    gap_time = (alns_rt - gur_rt) / max(1e-9, gur_rt) * 100
    results.append(BenchmarkResult(
        scenario=label, method="ALNS",
        objective=alns.objective, runtime=alns_rt, service_level=alns_sl,
        gap_cost=gap_cost, gap_time=gap_time
    ))
    print(f"    obj={alns.objective:.0f}  rt={alns_rt:.1f}s  SL={alns_sl*100:.2f}%  gap={gap_cost:+.2f}%")

    # ── GA/SA (pure heuristic) ────────────────────────────────────
    print("\n  [GA/SA] solving (pure heuristic, no warm-start) ...")
    inst, s_id, t_id, v_id = _to_gasa_instance(data)
    params = GASAParams(seed=SEED, population_size=30,
                        iterations_per_temp=30, time_limit=None,
                        max_iterations=2250)
    t0 = time.perf_counter()
    ga = BaselineGASASolver(inst, params).solve(initial_routes=None)
    ga_rt = time.perf_counter() - t0
    gasa_sl = 1.0 - ga.shortage_qty / max(1.0, total_dem)
    gap_cost = (ga.objective - gur.objective) / abs(gur.objective) * 100 if abs(gur.objective) > 1e-9 else 0.0
    gap_time = (ga_rt - gur_rt) / max(1e-9, gur_rt) * 100
    results.append(BenchmarkResult(
        scenario=label, method="GA/SA",
        objective=ga.objective, runtime=ga_rt, service_level=gasa_sl,
        gap_cost=gap_cost, gap_time=gap_time
    ))
    print(f"    obj={ga.objective:.0f}  rt={ga_rt:.1f}s  SL={gasa_sl*100:.2f}%  gap={gap_cost:+.2f}%  "
          f"new_best={ga.new_best_count}")

    # ── Summary for this scenario ────────────────────────────────
    print(f"\n  {'Method':<10} {'Objective':>14} {'Runtime':>9} {'SL%':>8} {'Gap%':>8}")
    print(f"  {'-'*57}")
    for r in results:
        gap_s = f"{r.gap_cost:+.2f}%" if r.method != "Gurobi" else "  —"
        print(f"  {r.method:<10} {r.objective:>14,.0f} {r.runtime:>8.1f}s "
              f"{r.service_level*100:>7.2f}% {gap_s:>8}")

    return results


def main():
    all_results = []
    for cfg in SCENARIOS_LARGE:
        results = run_scenario(cfg)
        all_results.extend(results)

    # ── Export to CSV ──────────────────────────────────────────────
    csv_path = "benchmark_results_15scenarios_large.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "method", "objective", "runtime", "service_level", "gap_cost", "gap_time"]
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow(asdict(r))
    print(f"\n✓ Exported CSV: {csv_path}")

    # ── Export to JSON ─────────────────────────────────────────────
    json_path = "benchmark_results_15scenarios_large.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"✓ Exported JSON: {json_path}")

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY: 15 Large Scenarios")
    print(f"{'='*80}\n")
    print(f"{'Scenario':<12} {'Method':<10} {'Objective':>14} {'Runtime':>9} {'SL%':>8} {'Gap%':>8}")
    print(f"{'-'*80}")
    for r in all_results:
        gap_s = f"{r.gap_cost:+.2f}%" if r.method != "Gurobi" else "  —"
        print(f"{r.scenario:<12} {r.method:<10} {r.objective:>14,.0f} {r.runtime:>8.1f}s "
              f"{r.service_level*100:>7.2f}% {gap_s:>8}")

    print("\nDone. All 15 large scenarios completed.\n")


if __name__ == "__main__":
    main()
