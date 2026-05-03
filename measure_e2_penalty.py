"""
Quick measurement: compute SLA penalty cost magnitude for A0 vs E2.

Re-runs the small instance with IRP_QUIET=0 so the cost-breakdown print
shows every component (sla_shortage_penalty, sla_surplus_penalty,
shortage_cost_realized, lateral_transshipment_cost_realized, etc.).
Then we can answer: is mu=nu=0.005 too small to bite?
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "test data.csv"
CKPT = REPO / "GNN" / "trained_models" / "irplt_teacher_E2_filtered_3epochs" / "bigat" / "pairwise_rank" / "best_model.pt"

# IMPORTANT: set BEFORE importing irp_gurobi_converted (env captured at import)
os.environ["IRP_SLA_PENALTY"] = "on"
os.environ["IRP_SLA_MU"] = os.environ.get("IRP_SLA_MU", "0.005")
os.environ["IRP_SLA_NU"] = os.environ.get("IRP_SLA_NU", "0.005")
os.environ["IRP_SLA_ALPHA"] = "4.0"
os.environ["IRP_SLA_BETA"] = "2.0"
os.environ["IRP_BENCHMARK_VARIANTS"] = "e2_only"
os.environ["IRP_BENCHMARK_FIXED_SHOCK"] = "1"
os.environ["IRP_GNN_SELECTION_MODE"] = "top_frac"
os.environ.setdefault("IRP_GNN_MAX_KEEP_FRAC", "0.30")
os.environ.setdefault("IRP_GNN_MIN_KEEP_FRAC", "0.10")
os.environ.setdefault("IRP_GNN_MIN_KEEP", "5")
os.environ["IRP_QUIET"] = "0"  # ← Make breakdown verbose

import irp_gurobi_converted as irp


def _build(stores, skus, sd, ed):
    m = irp.DatasetToIRPValidationMapper(
        excel_path=str(DATA), sheet_name=None,
        store_limit=stores, sku_limit=skus,
        start_date=sd, end_date=ed,
    )
    data, _, _, _ = m.build_irp_data(
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


def main():
    print(f"[mu={os.environ['IRP_SLA_MU']} nu={os.environ['IRP_SLA_NU']} alpha={os.environ['IRP_SLA_ALPHA']} beta={os.environ['IRP_SLA_BETA']}]")
    data = _build(4, 2, "2025-02-01", "2025-06-30")
    print(f"\n[data] stores={len(data.stores)} products={len(data.products)} periods={len(data.periods)} active_cells={len(data.stores)*len(data.products)*len(data.periods)}")
    out_dir = REPO / "Result_E2_local_benchmark" / "small_4s2sku_verbose"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = irp.run_three_way_benchmark(
        data=data,
        cg_iterations=30,
        time_limit=None,
        bp_max_nodes=15,
        bp_max_depth=6,
        gnn_checkpoint_path=str(CKPT),
        demand_shock_seed=42,
        demand_shock_probability=0.85,
        demand_shock_reallocation_fraction=0.60,
        demand_shock_reallocations_per_product_period=3,
        demand_shock_non_dispatch_multiplier=1.8,
        lt_activation_threshold=10.0,
        heuristic_top_k=20,
        enforce_integer_flows=False,
        n_repeats=1,
        results_dir=out_dir,
    )
    print("\n=== DONE — check log for [Realized Operating Cost ...] breakdown lines ===")


if __name__ == "__main__":
    sys.exit(main())
