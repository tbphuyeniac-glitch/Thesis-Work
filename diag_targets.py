"""Debug what _target_delivery returns at period 1 (the problematic period)."""
from pathlib import Path
from irp_gurobi_converted import (
    DatasetToIRPValidationMapper, BaselineALNSModel, _detect_period_window,
)

DATASET = "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
excel = Path(DATASET)
sd, ed = _detect_period_window(str(excel), 60, None)

mapper = DatasetToIRPValidationMapper(
    excel_path=str(excel), store_limit=4, sku_limit=2,
    start_date=sd, end_date=ed,
)
data, _, _, _ = mapper.build_irp_data(
    cw_replenishment_factor=0.8, holding_cost_rate=0.01, shortage_cost_rate=0.25,
    wh_inventory_multiplier=0.8, store_capacity_multiplier=1.2,
    store_initial_inventory_multiplier=0.2, vehicle_count=2, vehicle_fixed_cost=50.0,
    alpha=1.0, cw_capacity_factor=2.0,
)
disp = [t for t in data.periods if (t - min(data.periods)) % 5 == 0]
data.vehicle_capacity = max(500.0, round(sum(data.demand.values()) / max(1,len(disp)) / 2 * 0.85))

m = BaselineALNSModel(data)
m._dispatch_periods = m._compute_dispatch_periods(5)

print(f"Vehicle capacity: {data.vehicle_capacity:.0f}")
print(f"Dispatch periods (first 3): {sorted(list(m._dispatch_periods))[:3]}")
print()

t1 = sorted(data.periods)[0]
print(f"=== Period {t1} (first dispatch) ===")
print(f"{'Store':<14} {'Product':<32} {'init':>7} {'max':>8} {'room':>7} {'demand_t':>9} {'5d_demand':>10} {'target':>8}")
print('-'*100)

total_target = 0.0
for s in data.stores:
    for p in data.products:
        init   = float(data.init_inventory_store.get((s, p), 0.0))
        max_i  = float(data.max_inventory_store.get((s, p), float('inf')))
        room   = max_i - init
        d_t    = float(data.demand.get((s, p, t1), 0.0))
        # 5-day cumulative demand
        cum = sum(float(data.demand.get((s, p, t1+k), 0.0)) for k in range(5))
        tgt = m._target_delivery(s, p, t1, init)
        total_target += tgt
        print(f"{s[:14]:<14} {p[:32]:<32} {init:>7.1f} {max_i:>8.1f} {room:>7.1f} "
              f"{d_t:>9.1f} {cum:>10.1f} {tgt:>8.1f}")

print('-'*100)
print(f"Total target across (s,p) at period {t1}: {total_target:.1f}")
print(f"Vehicle total cap available: {2 * data.vehicle_capacity:.0f}")
print(f"Expected utilization (target / capacity): {total_target / (2 * data.vehicle_capacity) * 100:.1f}%")

print(f"\n=== Init WH inventory ===")
for p in data.products:
    print(f"  {p[:30]:<30}  init={data.init_inventory_wh.get(p, 0):>10.1f}")

print(f"\n=== Replenishment at t={t1} ===")
for p in data.products:
    print(f"  {p[:30]:<30}  g({t1})={data.replenishment_wh.get((p, t1), 0):>8.1f}")
