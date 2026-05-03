"""Local E1 ablation benchmark: A0 vs E1_rc_only (k=1 and k=3) vs E1_rc_gnn.

Tests the fix for the "k=3 same (p,t) overhead" bug:
  - E1_rc_only_k3 (OLD): max_columns_per_product_period=3 → adds 3 cols from
    same (p,t) per iter → RMP grows 3× faster, LP uses only 1 at optimum
  - E1_rc_only_k1 (FIX): rc_only_cols_per_pp=1 → 1 best col per (p,t) per
    iter → diversity from different (p,t) pairs, same as A0's pool_size=1

Usage:
    IRP_GNN_CHECKPOINT=GNN/trained_models/irplt_teacher_filtered_local3/bigat/pairwise_rank/best_model.pt \
    python3 local_e1_benchmark.py
"""
from __future__ import annotations
import os, time, json
from pathlib import Path
from copy import deepcopy

# ── Force small instance + short time limits ─────────────────────────
os.environ.setdefault("IRP_STORE_LIMIT", "3")
os.environ.setdefault("IRP_SKU_LIMIT", "2")
os.environ.setdefault("IRP_TIME_LIMIT", "60")
os.environ.setdefault("IRP_CG_ITERATIONS", "10")
os.environ.setdefault("IRP_BP_MAX_NODES", "5")
os.environ.setdefault("IRP_BP_MAX_DEPTH", "3")
os.environ.setdefault("IRP_QUIET", "1")
os.environ.setdefault("IRP_CG_STOPPING_MODE", "convergence")
os.environ.setdefault("IRP_RESULTS_DIR_OVERRIDE", "Results_local_e1_bench")
os.environ.pop("IRP_SLA_PENALTY", None)  # E1 = no penalty

import irp_gurobi_converted as irp


def run_variant(name, kwargs, data, baseline_sol, gnn_ckpt, seed, n_scenarios=2):
    """Run a variant on n_scenarios different demand shocks; return list of dicts."""
    rows = []
    for s_idx in range(n_scenarios):
        scenario_seed = seed + 10007 * s_idx
        data_copy = deepcopy(data)
        irp.apply_hidden_local_reallocation_demand_shocks(
            data_copy,
            shock_probability=0.85,
            max_reallocation_fraction=0.60,
            reallocations_per_product_period=3,
            non_dispatch_shock_multiplier=1.8,
            seed=scenario_seed,
        )
        baseline_copy = deepcopy(baseline_sol)
        cg = irp.LateralTransshipmentCG(
            data=data_copy,
            baseline_solution=baseline_copy,
            lt_activation_threshold=10.0,
            gnn_checkpoint=gnn_ckpt if kwargs.get("use_gnn") else None,
            **kwargs,
        )
        t0 = time.perf_counter()
        sol = cg.run_column_generation(max_iter=10, stopping_mode="convergence")
        runtime = time.perf_counter() - t0
        obj = float(sol.objective) if hasattr(sol, "objective") else float("nan")
        cg_history = getattr(cg, "cg_history", [])
        cg_iters = max(0, len(cg_history) - 1)  # exclude episode 0 (initial RMP)
        cols = sum(int(h.get("added_columns", 0)) for h in cg_history)
        rows.append({
            "variant": name,
            "scenario": s_idx,
            "seed": scenario_seed,
            "objective": obj,
            "runtime_s": runtime,
            "cg_iterations": cg_iters,
            "total_columns": cols,
        })
        print(f"  [scenario {s_idx} seed={scenario_seed}] obj={obj:.2f}  rt={runtime:.1f}s  iters={cg_iters}  cols={cols}")
    return rows


def main():
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
    print(f"\n[bench] data: {len(data.stores)} stores × {len(data.products)} skus × "
          f"{len(data.periods)} periods")

    print("\n[bench] solving shared ALNS baseline...")
    t0 = time.perf_counter()
    baseline_sol = irp.BaselineALNSModel(data).solve(
        msg=False, time_limit=60,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
    base_t = time.perf_counter() - t0
    print(f"[bench] baseline obj={float(baseline_sol.objective):.2f}  ({base_t:.1f}s)")

    gnn_ckpt = os.environ.get(
        "IRP_GNN_CHECKPOINT",
        "GNN/trained_models/irplt_teacher_filtered_local3/bigat/pairwise_rank/best_model.pt",
    )
    if not Path(gnn_ckpt).exists():
        raise SystemExit(f"GNN checkpoint not found: {gnn_ckpt}")
    print(f"[bench] GNN checkpoint: {gnn_ckpt}")

    variants = [
        # A0: exact Gurobi MIP pricing, 1 column per (p,t) per iter (pool_size=1)
        ("A0_no_penalty", {
            "use_gnn": False, "collect_teacher_mode": False,
            "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
            "exact_full_mode": True,
        }),
        # E1 OLD (k=3): RC filter, 3 cols per (p,t) per iter — demonstrates the
        # overhead bug: RMP grows 3× faster with no quality gain (extra cols λ=0)
        ("E1_rc_only_k3", {
            "use_gnn": False, "collect_teacher_mode": False,
            "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
            "rc_filter_mode": True,
            "rc_only_cols_per_pp": 3,   # old behavior for comparison
        }),
        # E1 FIX (k=1): RC filter, 1 best col per (p,t) per iter — mirrors A0's
        # pool_size=1; diversity across (p,t) pairs, not depth within one pair
        ("E1_rc_only_k1", {
            "use_gnn": False, "collect_teacher_mode": False,
            "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
            "rc_filter_mode": True,
            "rc_only_cols_per_pp": 1,   # default, explicit for clarity
        }),
        # E1+GNN (k=1): GNN ranker on top of k=1 RC filter
        ("E1_rc_gnn_k1", {
            "use_gnn": True, "collect_teacher_mode": False,
            "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
            "rc_filter_mode": True,
            "rc_only_cols_per_pp": 1,
            "gnn_selection_mode": "relative_threshold",
            "gnn_relative_threshold": 0.70,
            "gnn_max_keep_fraction": 0.30,
        }),
    ]
    seed = 20260418
    n_scenarios = 2

    all_rows = []
    for name, kwargs in variants:
        print("\n" + "=" * 70)
        print(f"[bench] VARIANT {name}  (n_scenarios={n_scenarios})")
        print("=" * 70)
        rows = run_variant(name, kwargs, data, baseline_sol, gnn_ckpt, seed, n_scenarios)
        all_rows.extend(rows)

    # Summary
    print("\n" + "#" * 76)
    print("# E1 ABLATION SUMMARY  (mean across scenarios)")
    print("#" * 76)
    print(f"{'variant':<22} {'obj_mean':>12} {'rt_mean':>10} {'iters_mean':>11} {'cols_mean':>10}")
    summary = {}
    for v_name, _ in variants:
        rows = [r for r in all_rows if r["variant"] == v_name]
        n = len(rows) or 1
        s = {
            "obj_mean":   sum(r["objective"] for r in rows) / n,
            "rt_mean":    sum(r["runtime_s"] for r in rows) / n,
            "iters_mean": sum(r["cg_iterations"] for r in rows) / n,
            "cols_mean":  sum(r["total_columns"] for r in rows) / n,
        }
        summary[v_name] = s
        print(f"{v_name:<22} {s['obj_mean']:>12.2f} {s['rt_mean']:>10.1f}s "
              f"{s['iters_mean']:>11.1f} {s['cols_mean']:>10.1f}")

    # Verdict: k=3 bug vs k=1 fix
    a0   = summary.get("A0_no_penalty")
    k3   = summary.get("E1_rc_only_k3")
    k1   = summary.get("E1_rc_only_k1")
    gnn1 = summary.get("E1_rc_gnn_k1")

    if a0:
        print("\n[verdict — vs A0_no_penalty]")
        for tag, s in [("E1_rc_only_k3", k3), ("E1_rc_only_k1", k1), ("E1_rc_gnn_k1", gnn1)]:
            if s is None:
                continue
            d_obj = s["obj_mean"] - a0["obj_mean"]
            d_rt  = s["rt_mean"] / max(1e-6, a0["rt_mean"])
            d_it  = s["iters_mean"] - a0["iters_mean"]
            d_col = s["cols_mean"] - a0["cols_mean"]
            print(f"  {tag:<18}  Δobj={d_obj:+.2f}  rt_ratio={d_rt:.2f}×  Δiters={d_it:+.1f}  Δcols={d_col:+.0f}")

    if k3 and k1:
        print("\n[verdict — k=1 FIX vs k=3 OLD (same pricing path)]")
        d_obj = k1["obj_mean"] - k3["obj_mean"]
        d_rt  = k1["rt_mean"] / max(1e-6, k3["rt_mean"])
        d_it  = k1["iters_mean"] - k3["iters_mean"]
        d_col = k1["cols_mean"] - k3["cols_mean"]
        print(f"  Δobj={d_obj:+.2f}  rt_ratio={d_rt:.2f}×  Δiters={d_it:+.1f}  Δcols={d_col:+.0f}")
        print(f"  (expected: Δiters≈0 or small +, rt_ratio<1.0 = faster, Δcols≈−2/3 per active pp)")

    if k1 and gnn1:
        print("\n[verdict — E1_rc_gnn_k1 vs E1_rc_only_k1 (marginal GNN on k=1)]")
        d_obj = gnn1["obj_mean"] - k1["obj_mean"]
        d_rt  = gnn1["rt_mean"] / max(1e-6, k1["rt_mean"])
        d_it  = gnn1["iters_mean"] - k1["iters_mean"]
        print(f"  Δobj={d_obj:+.2f}  rt_ratio={d_rt:.2f}×  Δiters={d_it:+.1f}")

    out_path = Path("Results_local_e1_bench") / "e1_k1_vs_k3_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": all_rows, "summary": summary}, f, indent=2)
    print(f"\n[bench] results saved → {out_path}")
    print(f"\n[bench] saved {out_path}")


if __name__ == "__main__":
    main()
