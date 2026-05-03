"""
Test GNN contribution on a large instance: 10 stores × 6 SKUs × 153 periods.

Candidate pairs per (p,t) ≈ 10×9 = 90 (vs 7×6=42 in current large test set).
Large pool → GNN ranking may actually matter.

Variants (matched regime μ=ν=0.05):
  A0_QUAD  = exact MIP pricing, all neg-RC cols, quadratic pricing
  E2       = pruned_exact + GNN top_frac (30%), quadratic pricing
  E1_k3    = RC-filter only, 3 cols/(p,t), no GNN, no penalty
"""
from __future__ import annotations
import os, time, json, copy
from pathlib import Path

os.environ["IRP_SLA_PENALTY"] = "on"
os.environ["IRP_SLA_MU"] = "0.05"
os.environ["IRP_SLA_NU"] = "0.05"
os.environ["IRP_SLA_ALPHA"] = "4.0"
os.environ["IRP_SLA_BETA"] = "2.0"
os.environ["IRP_QUIET"] = "1"

import irp_gurobi_converted as irp

REPO = Path(__file__).resolve().parent
CKPT = REPO / "GNN" / "trained_models" / "irplt_teacher_E2_filtered_local200_fixed" / "bigat" / "pairwise_rank" / "best_model.pt"
EXCEL = REPO / "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"

STORE_LIMIT = 10
SKU_LIMIT   = 6

def build_data():
    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=str(EXCEL), sheet_name="Sheet1",
        store_limit=STORE_LIMIT, sku_limit=SKU_LIMIT,
    )
    data, *_ = mapper.build_irp_data(
        wh_inventory_multiplier=0.8, store_capacity_multiplier=1.2,
        shortage_cost_rate=0.05, holding_cost_rate=100,
    )
    return data


def run_variant(name, data, baseline_sol, kwargs):
    pipeline = irp.IRPResearchPipeline(data)
    t0 = time.perf_counter()
    results = pipeline.run_lt_recourse_from_baseline(
        baseline_sol,
        shock_summary=None,
        use_random_initial_patterns=True,
        n_initial_patterns_per_product_period=5,
        cg_iterations=30,
        msg=False,
        gnn_checkpoint=str(CKPT) if kwargs.get("use_gnn") else None,
        use_classical_fallback=False,
        gnn_mass_threshold=0.55,
        gnn_max_keep=150,
        gnn_max_keep_fraction=0.30,
        bp_max_nodes=15,
        bp_max_depth=6,
        lt_activation_threshold=10.0,
        diagnostic_verbosity="summary",
        **kwargs,
    )
    runtime = time.perf_counter() - t0
    cg = results.get("cg_solution")
    rmp_obj = float(getattr(cg, "objective", float("nan"))) if cg else float("nan")
    ep_hist = results.get("cg_episode_history") or []
    cg_iters = len(ep_hist) if ep_hist else 0
    return {"variant": name, "rmp_objective": rmp_obj,
            "runtime_seconds": runtime, "cg_iterations": cg_iters}


def main():
    if not CKPT.exists():
        raise SystemExit(f"GNN checkpoint not found: {CKPT}")

    print(f"[build] loading {STORE_LIMIT} stores × {SKU_LIMIT} SKUs ...")
    data = build_data()
    print(f"[build] {len(data.stores)} stores × {len(data.products)} SKUs × {len(data.periods)} periods")
    print(f"[build] candidate pairs per (p,t) ≈ {len(data.stores)*(len(data.stores)-1)}")

    print("\n[baseline] solving ALNS ...")
    t0 = time.perf_counter()
    baseline_sol = irp.BaselineALNSModel(data).solve(
        msg=False, time_limit=120,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
    print(f"[baseline] obj={float(baseline_sol.objective):.2f}  ({time.perf_counter()-t0:.1f}s)")

    print("\n[shock] applying demand shock ...")
    shocked = copy.deepcopy(data)
    irp.apply_hidden_local_reallocation_demand_shocks(
        shocked,
        shock_probability=0.85,
        max_reallocation_fraction=0.60,
        reallocations_per_product_period=3,
        non_dispatch_shock_multiplier=1.8,
        seed=20260502,
    )

    VARIANTS = [
        ("A0_QUAD", dict(
            use_gnn=False, collect_teacher_mode=False,
            runtime_gnn_mode=False, heuristic_top_k_mode=False,
            exact_full_mode=True,
            use_branch_and_price=False,
        )),
        ("E2_GNN", dict(
            use_gnn=True, collect_teacher_mode=False,
            runtime_gnn_mode=True, heuristic_top_k_mode=False,
            pruned_exact_mode=True,
            gnn_selection_mode="top_frac",
            use_branch_and_price=False,
        )),
        ("E1_k3", dict(
            use_gnn=False, collect_teacher_mode=False,
            runtime_gnn_mode=False, heuristic_top_k_mode=False,
            rc_filter_mode=True,
            rc_only_cols_per_pp=3,
            use_branch_and_price=False,
        )),
    ]

    rows = []
    for name, kwargs in VARIANTS:
        print(f"\n[RUN] {name}")
        r = run_variant(name, shocked, copy.deepcopy(baseline_sol), kwargs)
        rows.append(r)
        print(f"  iters={r['cg_iterations']}  runtime={r['runtime_seconds']:.2f}s  obj={r['rmp_objective']:.2f}")

    print("\n" + "=" * 65)
    print(f"{'Variant':<12} {'Iters':>7} {'Runtime':>10} {'RMP Obj':>18}")
    print("=" * 65)
    for r in rows:
        print(f"{r['variant']:<12} {r['cg_iterations']:>7}  {r['runtime_seconds']:>9.2f}s  {r['rmp_objective']:>18.2f}")

    out = REPO / "Result_large_gnn_test" / "large_instance_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
