"""
test_ga_sa_vs_alns.py
======================

Standalone test: pure-heuristic Achamrah GA/SA (no Gurobi) vs ALNS baseline.
Both run on the same small scenario to validate the Gurobi-free GA/SA module
produces sensible objectives and routes before integrating it into the main
benchmark.

Run:
    python3 test_ga_sa_vs_alns.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Repo imports
from irp_gurobi_converted import (
    DatasetToIRPValidationMapper,
    BaselineALNSModel,
    AchamrahFullIRPTModel,   # Gurobi exact reference (optional)
    IRPData,
)

from achamrah_ga_sa_baseline import (
    BaselineInstance,
    BaselineGASASolver,
    GASAParams,
)


# ===================================================================
# Configuration — supports a single scenario (legacy) or multi-tier sweep
# ===================================================================
DATASET = "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
PERIODS = 60
SEED    = 42
GUROBI_REF = True

# Tiers for the sweep: (label, store_limit, sku_limit)
TIERS: List[Tuple[str, int, int]] = [
    ("small_4s2p",   4, 2),
    ("small_5s2p",   5, 2),
    ("medium_7s3p",  7, 3),
]

# Single-scenario fallback (used when this script is run with --single)
STORE_LIMIT = 4
SKU_LIMIT   = 2


# ===================================================================
# Adapter: IRPData → BaselineInstance (no LT)
# ===================================================================
def irpdata_to_baseline_instance(data: IRPData) -> BaselineInstance:
    stores = list(data.stores)
    products = list(data.products)
    periods = list(data.periods)
    vehicles = list(data.vehicles) if data.vehicles else [1, 2]

    s_id = {s: i + 1 for i, s in enumerate(stores)}
    p_id = {p: i + 1 for i, p in enumerate(products)}
    t_id = {t: i + 1 for i, t in enumerate(periods)}
    v_id = {v: i + 1 for i, v in enumerate(vehicles)}

    N = list(s_id.values())
    P = list(p_id.values())
    H = list(t_id.values())
    V = list(v_id.values())
    N0 = [0] + N

    d: Dict[Tuple[int, int], float] = {}
    for i in N0:
        for j in N0:
            if i == j:
                continue
            s_i = stores[i - 1] if i > 0 else data.warehouse
            s_j = stores[j - 1] if j > 0 else data.warehouse
            raw = data.distance.get((s_i, s_j), data.distance.get((s_j, s_i), None))
            d[(i, j)] = float(raw) if raw is not None else 1.0

    h: Dict[Tuple[int, int], float] = {}
    for p in products:
        h[(p_id[p], 0)] = float(data.holding_cost_wh.get(p, 0.0))
        for s in stores:
            h[(p_id[p], s_id[s])] = float(data.holding_cost_store.get((s, p), 0.0))

    C: Dict[int, float] = {0: float(data.max_inventory_wh.get(products[0], 1e9)) * len(products)}
    for s in stores:
        C[s_id[s]] = sum(float(data.max_inventory_store.get((s, p), 1e9)) for p in products)

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

    return BaselineInstance(
        N=N, P=P, H=H, V=V,
        alpha=float(data.alpha) if data.alpha else 1.0,
        Q=float(data.vehicle_capacity),
        d=d, h=h, C=C, I0=I0, D=D, g=g, f=f,
        ship_cost_cw=ship,
        vehicle_fixed_cost=float(data.vehicle_fixed_cost),
        name="test_no_lt",
    )


# ===================================================================
# Run the test
# ===================================================================
def _run_one_scenario(
    excel_path: Path,
    label: str,
    store_limit: int,
    sku_limit: int,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Run Gurobi + ALNS + pure-heuristic GA/SA on one scenario."""
    print("\n" + "=" * 80)
    print(f"  Scenario: {label}  ({store_limit}s × {sku_limit}p × {PERIODS}d)")
    print("=" * 80)

    mapper = DatasetToIRPValidationMapper(
        excel_path=str(excel_path),
        store_limit=store_limit,
        sku_limit=sku_limit,
        start_date=start_date or None,
        end_date=end_date or None,
    )
    data, _, _, _ = mapper.build_irp_data(
        cw_replenishment_factor=0.8,
        holding_cost_rate=0.01,
        shortage_cost_rate=0.25,
        wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2,
        store_initial_inventory_multiplier=0.2,
        vehicle_count=2,
        vehicle_fixed_cost=50.0,
        alpha=1.0,
        cw_capacity_factor=2.0,
    )
    dispatch_pds = [t for t in data.periods if (t - min(data.periods)) % 5 == 0]
    n_dispatch = max(1, len(dispatch_pds))
    total_dem = max(1.0, sum(data.demand.values()))
    auto_cap = total_dem / n_dispatch / 2 * 0.85
    data.vehicle_capacity = max(500.0, round(auto_cap))

    print(f"  Stores={len(data.stores)}  SKUs={len(data.products)}  "
          f"Periods={len(data.periods)}  Vehicle_cap={data.vehicle_capacity:.0f}  "
          f"Demand={total_dem:.0f}")

    # ALNS
    print("  [ALNS] running 1000 iter ...")
    t0 = time.perf_counter()
    alns_sol = BaselineALNSModel(data).solve(
        msg=False, max_iterations=1000, seed=SEED,
        allow_lateral_transshipment=False,
    )
    alns_rt = time.perf_counter() - t0
    alns_obj = float(alns_sol.objective)
    print(f"    obj={alns_obj:.2f}  rt={alns_rt:.2f}s")

    # Pure-heuristic GA/SA
    print("  [GA/SA pure-heuristic] running ...")
    inst = irpdata_to_baseline_instance(data)
    params = GASAParams(
        seed=SEED,
        population_size=30,
        iterations_per_temp=30,
        time_limit=180.0,
    )
    t0 = time.perf_counter()
    ga = BaselineGASASolver(inst, params).solve()
    ga_rt = time.perf_counter() - t0
    print(f"    obj={ga.objective:.2f}  rt={ga_rt:.2f}s  "
          f"new_best={ga.new_best_count}")
    print(f"    breakdown: routing={ga.routing_cost:.0f}  "
          f"holding={ga.holding_cost:.0f}  shortage={ga.shortage_cost:.0f}  "
          f"ship={ga.ship_cost:.0f}  veh_fixed={ga.vehicle_fixed_cost:.0f}")

    # Gurobi
    gur_obj = float("nan")
    gur_rt = float("nan")
    if GUROBI_REF:
        print("  [Gurobi] exact MIP ...")
        t0 = time.perf_counter()
        try:
            gs = AchamrahFullIRPTModel(data).solve(
                msg=False, allow_lateral_transshipment=False,
            )
            gur_rt = time.perf_counter() - t0
            gur_obj = float(gs.objective)
            print(f"    obj={gur_obj:.2f}  rt={gur_rt:.2f}s  status={gs.status}")
        except Exception as exc:
            print(f"    ERROR: {exc}")

    return {
        "label": label,
        "stores": len(data.stores),
        "skus": len(data.products),
        "periods": len(data.periods),
        "gurobi_obj": gur_obj, "gurobi_rt": gur_rt,
        "alns_obj":   alns_obj, "alns_rt": alns_rt,
        "gasa_obj":   ga.objective, "gasa_rt": ga_rt,
        "gasa_new_best": ga.new_best_count,
    }


def main() -> None:
    repo_root = Path(__file__).parent
    excel_path = repo_root / DATASET

    from irp_gurobi_converted import _detect_period_window
    start_date, end_date = _detect_period_window(str(excel_path), PERIODS, None)
    print(f"Period window: {start_date} → {end_date}")

    results: List[Dict[str, Any]] = []
    for label, store_lim, sku_lim in TIERS:
        r = _run_one_scenario(excel_path, label, store_lim, sku_lim, start_date, end_date)
        results.append(r)

    print("\n\n" + "=" * 100)
    print("PURE-HEURISTIC GA/SA  vs  ALNS  vs  Gurobi (all baseline, no LT)")
    print("=" * 100)
    print(f"{'Scenario':<14} {'Gurobi obj':>14} {'Gur rt':>8} | "
          f"{'ALNS obj':>14} {'ALNS rt':>8} {'gap%':>7} | "
          f"{'GASA obj':>14} {'GASA rt':>8} {'gap%':>7}")
    print("-" * 100)
    for r in results:
        ref = r["gurobi_obj"]
        alns_gap = (r["alns_obj"] - ref) / ref * 100.0 if math.isfinite(ref) and abs(ref) > 1e-9 else float("nan")
        ga_gap   = (r["gasa_obj"] - ref) / ref * 100.0 if math.isfinite(ref) and abs(ref) > 1e-9 else float("nan")
        print(f"{r['label']:<14} "
              f"{r['gurobi_obj']:>14.2f} {r['gurobi_rt']:>7.2f}s | "
              f"{r['alns_obj']:>14.2f} {r['alns_rt']:>7.2f}s "
              f"{alns_gap:>+6.2f}% | "
              f"{r['gasa_obj']:>14.2f} {r['gasa_rt']:>7.2f}s "
              f"{ga_gap:>+6.2f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
