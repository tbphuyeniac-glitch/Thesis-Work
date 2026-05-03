"""
Local quick-look E2 benchmark driver.
=====================================
Runs run_three_way_benchmark with IRP_BENCHMARK_VARIANTS=e2_only on two
instance sizes (small + large), using the 3-epoch local checkpoint.

Outputs:
  - Result_E2_local_benchmark/<size>/comparison_per_run.csv
  - Result_E2_local_benchmark/<size>/comparison_aggregate.csv
  - Result_E2_local_benchmark/<size>/comparison_table.txt
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA_PATH = REPO / "test data.csv"
CKPT = REPO / "GNN" / "trained_models" / "irplt_teacher_E2_filtered_3epochs" / "bigat" / "pairwise_rank" / "best_model.pt"

# SLA penalty configuration MUST be set BEFORE importing irp_gurobi_converted
# because some module-level code may capture defaults.
os.environ["IRP_SLA_PENALTY"] = "on"
# Tăng calibration: μ ×20 (shortage matters more), ν ×5 (surplus còn relevant
# nhưng không áp đảo). Chỉ thay đổi cho benchmark — teacher data + GNN
# checkpoint train ở μ=ν=0.005 nên E2 GNN signal sẽ hơi mismatched, đây
# chỉ là sanity check về magnitude penalty, không phải production benchmark.
os.environ["IRP_SLA_MU"] = "0.10"
os.environ["IRP_SLA_NU"] = "0.025"
os.environ["IRP_SLA_ALPHA"] = "4.0"
os.environ["IRP_SLA_BETA"] = "2.0"
os.environ["IRP_BENCHMARK_VARIANTS"] = "e2_only"
os.environ["IRP_BENCHMARK_FIXED_SHOCK"] = "1"
os.environ["IRP_GNN_SELECTION_MODE"] = "top_frac"
os.environ.setdefault("IRP_GNN_MAX_KEEP_FRAC", "0.30")
os.environ.setdefault("IRP_GNN_MIN_KEEP_FRAC", "0.10")
os.environ.setdefault("IRP_GNN_MIN_KEEP", "5")
os.environ.setdefault("IRP_QUIET", "0")

import irp_gurobi_converted as irp


SIZES = [
    # (label, store_limit, sku_limit, start_date, end_date)
    ("small_4s2sku", 4, 2, "2025-02-01", "2025-06-30"),
    ("large_8s4sku", 8, 4, "2025-02-01", "2025-06-30"),
]

CG_ITER_CAP = 30      # convergence-mode benchmark stops earlier; this is a safety cap
N_REPEATS = 1         # one repeat — quick local check
DEMAND_SHOCK_SEED = 42


def _build_data(store_limit: int, sku_limit: int, start: str, end: str):
    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=str(DATA_PATH),
        sheet_name=None,
        store_limit=store_limit,
        sku_limit=sku_limit,
        start_date=start,
        end_date=end,
    )
    data, _, _, _ = mapper.build_irp_data(
        wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2,
        shortage_cost_rate=0.25,
        holding_cost_rate=0.01,
        cw_ship_cost_flat=1.0,
        lt_ship_cost_flat=0.6,
        fixed_dispatch_cw=8.0,
        fixed_dispatch_lt=2.0,
        vehicle_count=2,
        vehicle_capacity=500.0,
        vehicle_fixed_cost=50.0,
        alpha=1.0,
        cw_replenishment_factor=0.2,
        cw_capacity_factor=2.0,
        store_initial_inventory_multiplier=0.2,
    )
    return data


def _run_one_size(label: str, data, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 80)
    print(f"# E2 LOCAL BENCHMARK — {label}")
    print(f"# stores={len(data.stores)}  products={len(data.products)}  periods={len(data.periods)}")
    print(f"# checkpoint={CKPT}")
    print("#" * 80)

    t0 = time.perf_counter()
    df = irp.run_three_way_benchmark(
        data=data,
        cg_iterations=CG_ITER_CAP,
        time_limit=None,
        bp_max_nodes=15,
        bp_max_depth=6,
        gnn_checkpoint_path=str(CKPT),
        demand_shock_seed=DEMAND_SHOCK_SEED,
        demand_shock_probability=0.85,
        demand_shock_reallocation_fraction=0.60,
        demand_shock_reallocations_per_product_period=3,
        demand_shock_non_dispatch_multiplier=1.8,
        lt_activation_threshold=10.0,
        heuristic_top_k=20,
        enforce_integer_flows=False,
        n_repeats=N_REPEATS,
        results_dir=results_dir,
    )
    wall = time.perf_counter() - t0
    print(f"[{label}] benchmark wall time: {wall:.1f}s")

    # Save a friendly summary table
    cols = [
        "variant", "rmp_objective",
        "realized_cost_with_lt", "lt_cost_with_lt", "shortage_cost_with_lt",
        "cg_iterations", "variant_runtime_seconds", "columns_generated", "columns_added_to_rmp",
    ]
    cols = [c for c in cols if c in df.columns]
    summary = df[cols].copy()
    summary_text = summary.to_string(index=False)
    (results_dir / "comparison_table.txt").write_text(summary_text + f"\n\nwall_seconds={wall:.2f}\n")
    print(summary_text)


def main() -> int:
    if not DATA_PATH.exists():
        print(f"[ERROR] {DATA_PATH} not found", file=sys.stderr)
        return 2
    if not CKPT.exists():
        print(f"[ERROR] {CKPT} not found — run training first", file=sys.stderr)
        return 2

    out_root = REPO / "Result_E2_local_benchmark"
    for label, sl, kl, sd, ed in SIZES:
        try:
            print(f"\n[build_data] {label}: stores<= {sl}, sku<= {kl}, {sd}..{ed}")
            data = _build_data(sl, kl, sd, ed)
        except Exception as exc:
            print(f"[ERROR] build_data failed for {label}: {exc}", file=sys.stderr)
            continue
        try:
            _run_one_size(label, data, out_root / label)
        except Exception as exc:
            print(f"[ERROR] benchmark failed for {label}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
