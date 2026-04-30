"""Local smoke benchmark: A0 vs C only, 1 repeat, small instance.

Usage:
  IRP_GNN_CHECKPOINT=GNN/trained_models/irplt_teacher_filtered_smoke/bigat/pairwise_rank/best_model.pt \
  python local_smoke_benchmark.py
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from copy import deepcopy

# ── Force smallest viable instance + short time limits ───────────────
os.environ.setdefault("IRP_STORE_LIMIT", "2")
os.environ.setdefault("IRP_SKU_LIMIT", "2")
os.environ.setdefault("IRP_TIME_LIMIT", "60")
os.environ.setdefault("IRP_CG_ITERATIONS", "5")
os.environ.setdefault("IRP_BP_MAX_NODES", "5")
os.environ.setdefault("IRP_BP_MAX_DEPTH", "3")
os.environ.setdefault("IRP_QUIET", "1")
os.environ.setdefault("IRP_CG_STOPPING_MODE", "convergence")
os.environ.setdefault("IRP_RESULTS_DIR_OVERRIDE", "Results_local_smoke")
os.environ.pop("IRP_SLA_PENALTY", None)  # E1 = no penalty

import irp_gurobi_converted as irp


def main():
    # Build the small dataset
    excel_path = Path(__file__).resolve().parent / "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=str(excel_path),
        sheet_name="Sheet1",
        store_limit=int(os.environ["IRP_STORE_LIMIT"]),
        sku_limit=int(os.environ["IRP_SKU_LIMIT"]),
    )
    data, _, _, _ = mapper.build_irp_data(
        wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2,
        shortage_cost_rate=0.05,
        holding_cost_rate=100,
    )
    print(f"\n[smoke] data: {len(data.stores)} stores × {len(data.products)} skus × "
          f"{len(data.periods)} periods")

    # Solve shared baseline once (so A0 and C compare LT recourse only)
    print("\n[smoke] solving shared ALNS baseline...")
    t0 = time.perf_counter()
    baseline_sol = irp.BaselineALNSModel(data).solve(
        msg=False, time_limit=60,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
    base_t = time.perf_counter() - t0
    print(f"[smoke] baseline obj={float(baseline_sol.objective):.2f}  ({base_t:.1f}s)")

    gnn_ckpt = os.environ.get(
        "IRP_GNN_CHECKPOINT",
        "GNN/trained_models/irplt_teacher_filtered_smoke/bigat/pairwise_rank/best_model.pt",
    )
    if not Path(gnn_ckpt).exists():
        raise SystemExit(f"GNN checkpoint not found: {gnn_ckpt}")
    print(f"[smoke] GNN checkpoint: {gnn_ckpt}")

    # ── A0: full exact pricing, no GNN ───────────────────────────────
    # ── C : GNN-guided pricing, runtime_gnn_mode=True ────────────────
    variants = [
        ("A0_cg_full_exact",
         {"use_gnn": False, "collect_teacher_mode": False,
          "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
          "exact_full_mode": True}),
        ("C_gnn_guided_cg",
         {"use_gnn": True, "collect_teacher_mode": False,
          "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
          "gnn_selection_mode": "cumulative_mass"}),
    ]
    seed = 20260418

    results = []
    for name, kwargs in variants:
        print("\n" + "=" * 70)
        print(f"[smoke] VARIANT {name}")
        print("=" * 70)
        # Apply demand shock (deterministic) on a deep copy
        data_copy = deepcopy(data)
        irp.apply_hidden_local_reallocation_demand_shocks(
            data_copy,
            shock_probability=0.85,
            max_reallocation_fraction=0.60,
            reallocations_per_product_period=3,
            non_dispatch_shock_multiplier=1.8,
            seed=seed,
        )
        baseline_copy = deepcopy(baseline_sol)
        cg = irp.LateralTransshipmentCG(
            data=data_copy, baseline_solution=baseline_copy,
            lt_activation_threshold=10.0,
            gnn_checkpoint=gnn_ckpt if kwargs.get("use_gnn") else None,
            **kwargs,
        )
        t_v = time.perf_counter()
        sol = cg.run_column_generation(max_iter=5, stopping_mode="convergence")
        runtime = time.perf_counter() - t_v
        obj = float(sol.objective) if hasattr(sol, "objective") else float("nan")
        cols = getattr(cg, "total_columns_generated", None) or len(getattr(sol, "patterns", []))
        cg_iters = getattr(cg, "iterations_completed", None)
        results.append({
            "variant": name,
            "objective": obj,
            "runtime_s": runtime,
            "total_columns": cols,
            "cg_iterations": cg_iters,
        })
        print(f"[smoke] {name}: obj={obj:.2f}  runtime={runtime:.1f}s  cols={cols}  iters={cg_iters}")

    print("\n" + "#" * 70)
    print("# RESULT SUMMARY (E1 smoke, 1 repeat, small instance)")
    print("#" * 70)
    print(f"{'variant':<25} {'objective':>12} {'runtime_s':>10} {'cols':>8} {'iters':>6}")
    for r in results:
        print(f"{r['variant']:<25} {r['objective']:>12.2f} {r['runtime_s']:>10.1f} "
              f"{r['total_columns']:>8} {str(r['cg_iterations']):>6}")
    if len(results) == 2:
        a0, c = results[0], results[1]
        print("\n[verdict]")
        print(f"  obj C - A0       = {c['objective'] - a0['objective']:+.2f}  "
              f"(C closer to optimal if negative)")
        print(f"  runtime C / A0   = {c['runtime_s'] / max(1e-6, a0['runtime_s']):.2f}x")
        print(f"  columns C / A0   = {c['total_columns'] / max(1, a0['total_columns']):.2f}x")


if __name__ == "__main__":
    main()
