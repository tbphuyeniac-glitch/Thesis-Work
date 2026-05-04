"""
Diagnostic: compare per-(store,product,period) delivery quantities
between Gurobi optimal and ALNS solution on small_4s2p.

Goal: pinpoint WHERE ALNS under-delivers (vs Gurobi).
"""

from __future__ import annotations
import math, time
from collections import defaultdict
from pathlib import Path
import pandas as pd

from irp_gurobi_converted import (
    DatasetToIRPValidationMapper, BaselineALNSModel, AchamrahFullIRPTModel,
    _detect_period_window,
)

DATASET = "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
STORE_LIMIT, SKU_LIMIT, PERIODS = 4, 2, 60
SEED = 42

excel = Path(DATASET)
sd, ed = _detect_period_window(str(excel), PERIODS, None)

mapper = DatasetToIRPValidationMapper(
    excel_path=str(excel), store_limit=STORE_LIMIT, sku_limit=SKU_LIMIT,
    start_date=sd, end_date=ed,
)
data, _, _, _ = mapper.build_irp_data(
    cw_replenishment_factor=0.8, holding_cost_rate=0.01,
    shortage_cost_rate=0.25, wh_inventory_multiplier=0.8,
    store_capacity_multiplier=1.2, store_initial_inventory_multiplier=0.2,
    vehicle_count=2, vehicle_fixed_cost=50.0, alpha=1.0, cw_capacity_factor=2.0,
)
disp = [t for t in data.periods if (t - min(data.periods)) % 5 == 0]
data.vehicle_capacity = max(500.0, round(sum(data.demand.values()) / max(1,len(disp)) / 2 * 0.85))

print(f"Stores={len(data.stores)} SKUs={len(data.products)} Periods={len(data.periods)} "
      f"VehCap={data.vehicle_capacity:.0f} Dispatches={len(disp)}")

# Gurobi
print("\n[Gurobi] solving ...")
gur = AchamrahFullIRPTModel(data).solve(msg=False, allow_lateral_transshipment=False)
print(f"  obj={gur.objective:.0f}")

# ALNS
print("\n[ALNS] solving ...")
alns = BaselineALNSModel(data).solve(msg=False, max_iterations=1000, seed=SEED,
                                       allow_lateral_transshipment=False)
print(f"  obj={alns.objective:.0f}")

# Aggregate per (s, p, t)
def agg_deliv_from_alns(sol):
    out = defaultdict(float)
    for (s, p, v, t), q in sol.deliv.items():
        out[(s, p, t)] += float(q)
    return out

def agg_deliv_from_gurobi(sol):
    out = defaultdict(float)
    for (s, p, v, t), q in sol.deliv.items():
        out[(s, p, t)] += float(q)
    return out

g_q = agg_deliv_from_gurobi(gur)
a_q = agg_deliv_from_alns(alns)

# ==========================================================
# Per-dispatch-period totals
# ==========================================================
print(f"\n{'='*80}")
print(f"{'Period':>7} {'Gurobi total':>15} {'ALNS total':>13} {'Diff':>10} {'ALNS%Gur':>10}")
print(f"{'-'*80}")
total_g, total_a = 0.0, 0.0
for t in sorted(data.periods):
    if t not in disp: continue
    g_t = sum(g_q.get((s,p,t), 0.0) for s in data.stores for p in data.products)
    a_t = sum(a_q.get((s,p,t), 0.0) for s in data.stores for p in data.products)
    total_g += g_t
    total_a += a_t
    pct = a_t / g_t * 100 if g_t > 0 else 0
    print(f"{str(t):>7} {g_t:>15.1f} {a_t:>13.1f} {a_t-g_t:>+10.1f} {pct:>9.1f}%")
print(f"{'-'*80}")
print(f"{'TOTAL':>7} {total_g:>15.1f} {total_a:>13.1f} {total_a-total_g:>+10.1f} "
      f"{total_a/total_g*100:>9.1f}%")

# ==========================================================
# Vehicle capacity utilization per dispatch
# ==========================================================
veh_cap = float(data.vehicle_capacity) * len(data.vehicles)
print(f"\nVehicle total cap per dispatch: {veh_cap:.0f}")
print(f"Gurobi avg utilization: {total_g/len(disp)/veh_cap*100:.1f}%")
print(f"ALNS   avg utilization: {total_a/len(disp)/veh_cap*100:.1f}%")

# ==========================================================
# Per-(store,product) gap, sorted by absolute diff
# ==========================================================
keys = set(g_q.keys()) | set(a_q.keys())
diffs = []
for k in keys:
    diff = a_q.get(k, 0.0) - g_q.get(k, 0.0)
    if abs(diff) > 0.5:
        diffs.append((k, g_q.get(k, 0.0), a_q.get(k, 0.0), diff))
diffs.sort(key=lambda x: x[3])
print(f"\nTop 5 stores where ALNS UNDER-delivers:")
print(f"{'(store,sku,t)':>40} {'Gur':>10} {'ALNS':>10} {'Diff':>10}")
for k, g, a, d in diffs[:5]:
    print(f"  {str(k):>38} {g:>10.1f} {a:>10.1f} {d:>+10.1f}")

print(f"\nTop 5 where ALNS OVER-delivers:")
for k, g, a, d in diffs[-5:][::-1]:
    print(f"  {str(k):>38} {g:>10.1f} {a:>10.1f} {d:>+10.1f}")

# ==========================================================
# WH inventory trajectory comparison
# ==========================================================
print("\n=== WH inventory at each dispatch (computed from solution) ===")
inv_wh_g = {p: float(data.init_inventory_wh.get(p, 0.0)) for p in data.products}
inv_wh_a = {p: float(data.init_inventory_wh.get(p, 0.0)) for p in data.products}
print(f"{'Period':>7} {'Gurobi WH':>20} {'ALNS WH':>20}")
for t in sorted(data.periods):
    for p in data.products:
        inv_wh_g[p] += float(data.replenishment_wh.get((p, t), 0.0))
        inv_wh_a[p] += float(data.replenishment_wh.get((p, t), 0.0))
        for s in data.stores:
            inv_wh_g[p] -= g_q.get((s, p, t), 0.0)
            inv_wh_a[p] -= a_q.get((s, p, t), 0.0)
    if t in disp or t == max(data.periods):
        gw = sum(inv_wh_g.values())
        aw = sum(inv_wh_a.values())
        print(f"{str(t):>7} {gw:>20.1f} {aw:>20.1f}")
