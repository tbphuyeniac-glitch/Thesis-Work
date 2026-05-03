"""A0 vs GNN-ranker with WIDE exact-pricing pool (K=15).

Previous benchmark used default K=3 → MIP returned ≤3 cols/(p,t) → GNN had
nothing to filter (min-per-group floor kept everything). With K=15, MIP
returns up to 15 cols/(p,t) so GNN's relative_threshold + max_keep_fraction
actually prunes the candidate pool.

Variants:
  - A0_pool3       : exact MIP K=3, no GNN          (reference, original A0)
  - A0_pool15      : exact MIP K=15, no GNN         (control: more cols, no rank)
  - gnn_rank_pool3 : exact MIP K=3 + GNN ranker     (already showed: no effect)
  - gnn_rank_pool15: exact MIP K=15 + GNN ranker    (← this should show effect)

Usage:
    IRP_GNN_CHECKPOINT=GNN/trained_models/irplt_teacher_filtered_local200_fixed/bigat/pairwise_rank/best_model.pt \
    python3 run_a0_vs_gnn_ranker_pool15.py
"""
from __future__ import annotations
import os, time, json, gzip, pickle
from pathlib import Path
from copy import deepcopy

os.environ.setdefault("IRP_TIME_LIMIT", "240")
os.environ.setdefault("IRP_CG_ITERATIONS", "15")
os.environ.setdefault("IRP_BP_MAX_NODES", "15")
os.environ.setdefault("IRP_BP_MAX_DEPTH", "6")
os.environ.setdefault("IRP_QUIET", "1")
os.environ.setdefault("IRP_CG_STOPPING_MODE", "convergence")
os.environ.setdefault("IRP_RESULTS_DIR_OVERRIDE", "Result_a0_vs_gnn_ranker_pool15")
os.environ.pop("IRP_SLA_PENALTY", None)
os.environ.pop("IRP_DISABLE_RC_FILTER", None)

import irp_gurobi_converted as irp


# (variant, kwargs, env_overrides)
# IRP_EXACT_GNN_RANKER=1 unlocks runtime_gnn_mode for exact_full_mode (default
# behavior force-disables GNN when exact pricing is on; the new env override
# allows GNN to run as a pure ranker on top of exact MIP K-best columns).
VARIANTS = [
    ("A0_pool3", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "exact_full_mode": True, "use_branch_and_price": True,
    }, {"IRP_EXACT_PRICING_POOL_SIZE": "3", "IRP_EXACT_GNN_RANKER": "0"}),
    ("A0_pool15", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "exact_full_mode": True, "use_branch_and_price": True,
    }, {"IRP_EXACT_PRICING_POOL_SIZE": "15", "IRP_EXACT_GNN_RANKER": "0"}),
    ("gnn_rank_pool3", {
        "use_gnn": True, "collect_teacher_mode": False,
        "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
        "exact_full_mode": True,
        "gnn_selection_mode": "relative_threshold",
        "gnn_relative_threshold": 0.70,
        "gnn_max_keep_fraction": 0.30,
        "use_branch_and_price": True,
    }, {"IRP_EXACT_PRICING_POOL_SIZE": "3", "IRP_EXACT_GNN_RANKER": "1"}),
    ("gnn_rank_pool15", {
        "use_gnn": True, "collect_teacher_mode": False,
        "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
        "exact_full_mode": True,
        "gnn_selection_mode": "relative_threshold",
        "gnn_relative_threshold": 0.70,
        "gnn_max_keep_fraction": 0.30,
        "use_branch_and_price": True,
    }, {"IRP_EXACT_PRICING_POOL_SIZE": "15", "IRP_EXACT_GNN_RANKER": "1"}),
]

SCENARIO_SELECTION = [
    ("small",  "base_01__normal_global__seed1373158607"),
    ("medium", "base_09__normal_global__seed1730483679"),
    ("large",  "base_24__normal_global__seed814874364"),
]

GNN_CKPT = os.environ.get(
    "IRP_GNN_CHECKPOINT",
    "GNN/trained_models/irplt_teacher_filtered_local200_fixed/bigat/pairwise_rank/best_model.pt",
)
SCENARIOS_DIR = Path("Test 30 scenarios/test_baselines/scenarios")


def load_scenario(name):
    pkl = SCENARIOS_DIR / f"{name}.pkl.gz"
    with gzip.open(pkl, "rb") as f:
        s = pickle.load(f)
    return s["shocked_data"], s["baseline_sol"], s.get("shock_type", ""), s.get("shock_seed", "")


def run_variant(v_name, kwargs, env_overrides, shocked_data, baseline_sol):
    saved_env = {}
    for k, v in env_overrides.items():
        saved_env[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        data_copy = deepcopy(shocked_data)
        baseline_copy = deepcopy(baseline_sol)
        kwargs = dict(kwargs)
        use_bp = kwargs.pop("use_branch_and_price", False)
        pipeline = irp.IRPResearchPipeline(data_copy)
        t0 = time.perf_counter()
        result = pipeline.run_lt_recourse_from_baseline(
            baseline_copy,
            use_random_initial_patterns=True,
            n_initial_patterns_per_product_period=5,
            cg_iterations=15,
            msg=False,
            gnn_checkpoint=GNN_CKPT if kwargs.get("use_gnn") else None,
            use_classical_fallback=False,
            gnn_max_keep=150,
            use_branch_and_price=use_bp,
            bp_max_nodes=15, bp_max_depth=6,
            lt_activation_threshold=10.0,
            **kwargs,
        )
        rt = time.perf_counter() - t0
        cg_sol = result.get("cg_solution")
        obj = float(getattr(cg_sol, "objective", float("nan")))
        cg_history = result.get("cg_episode_history", []) or []
        cg_iters = max(0, len(cg_history) - 1)
        cols_added = sum(int(h.get("added_columns", 0)) for h in cg_history)
        cols_proposed = sum(int(h.get("proposed_columns", 0)) for h in cg_history)
        bp_nodes = len(result.get("branch_price_history", []) or [])
        return {
            "obj": obj, "rt": rt, "iters": cg_iters,
            "cols_added": cols_added, "cols_proposed": cols_proposed,
            "bp_nodes": bp_nodes, "use_bp": use_bp,
        }
    finally:
        for k, prev in saved_env.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def main():
    if not Path(GNN_CKPT).exists():
        raise SystemExit(f"GNN checkpoint not found: {GNN_CKPT}")
    print(f"[bench] checkpoint: {GNN_CKPT}\n")

    rows = []
    for size_label, name in SCENARIO_SELECTION:
        print("=" * 95)
        print(f"[scenario] {size_label} | {name}")
        print("=" * 95)
        shocked_data, baseline_sol, shock_type, shock_seed = load_scenario(name)
        n_stores = len(shocked_data.stores)
        n_skus = len(shocked_data.products)
        n_periods = len(shocked_data.periods)
        baseline_obj = float(baseline_sol.objective)
        print(f"  size: {n_stores}×{n_skus}×{n_periods}  shock={shock_type}  baseline obj={baseline_obj:.2f}\n")

        for v_name, kwargs_orig, env_orig in VARIANTS:
            r = run_variant(v_name, kwargs_orig, env_orig, shocked_data, baseline_sol)
            row = {
                "size": size_label, "scenario": name,
                "stores": n_stores, "skus": n_skus, "periods": n_periods,
                "shock_type": shock_type, "shock_seed": shock_seed,
                "variant": v_name,
                "obj": r["obj"], "rt_s": r["rt"],
                "iters": r["iters"], "bp_nodes": r["bp_nodes"],
                "cols_added": r["cols_added"], "cols_proposed": r["cols_proposed"],
                "baseline_obj": baseline_obj,
            }
            rows.append(row)
            print(f"  {v_name:<18} obj={r['obj']:.2f}  rt={r['rt']:.2f}s  "
                  f"iters={r['iters']}  bp_nodes={r['bp_nodes']}  "
                  f"cols_proposed={r['cols_proposed']}  cols_added={r['cols_added']}")
        print()

    print("\n" + "#" * 120)
    print("# A0 vs GNN-RANKER, varying exact_pricing_pool_size (K=3 vs K=15)")
    print("#" * 120)
    print(f"{'size':<7} {'scenario':<35} {'A0(K=3)':>9} {'A0(K=15)':>10} {'gnn(K=3)':>10} {'gnn(K=15)':>11}  "
          f"{'spdK15':>7} {'A0K15→gnnK15':>13}  {'cols_K3':>8} {'cols_K15':>9} {'gnn_K15':>8}")
    grouped = {}
    for r in rows:
        grouped.setdefault((r["size"], r["scenario"]), {})[r["variant"]] = r
    for (size, scen), per_v in grouped.items():
        a0_3 = per_v.get("A0_pool3"); a0_15 = per_v.get("A0_pool15")
        gn_3 = per_v.get("gnn_rank_pool3"); gn_15 = per_v.get("gnn_rank_pool15")
        if not all([a0_3, a0_15, gn_3, gn_15]): continue
        spd_K15 = a0_3["rt_s"] / max(1e-6, gn_15["rt_s"])
        spd_a0K15_gnnK15 = a0_15["rt_s"] / max(1e-6, gn_15["rt_s"])
        scen_short = scen[-30:] if len(scen) > 30 else scen
        print(f"{size:<7} {scen_short:<35} {a0_3['rt_s']:>8.1f}s {a0_15['rt_s']:>9.1f}s {gn_3['rt_s']:>9.1f}s {gn_15['rt_s']:>10.1f}s  "
              f"{spd_K15:>6.2f}x {spd_a0K15_gnnK15:>12.2f}x  "
              f"{a0_3['cols_added']:>8} {a0_15['cols_added']:>9} {gn_15['cols_added']:>8}")

    print("\n# AGGREGATE BY SIZE (mean)")
    print("#" * 120)
    print(f"{'size':<10} {'A0_K3':>9} {'A0_K15':>9} {'gnn_K3':>9} {'gnn_K15':>9}  "
          f"{'A0K3→gnnK15':>12} {'A0K15→gnnK15':>13}  "
          f"{'cols_A0K15':>11} {'cols_gnnK15':>12}  {'prune%':>7}")
    by_size = {}
    for r in rows:
        by_size.setdefault(r["size"], []).append(r)
    for size in ["small", "medium", "large"]:
        size_rows = by_size.get(size, [])
        if not size_rows: continue
        def mean_(v, key):
            vs = [r[key] for r in size_rows if r["variant"]==v]
            return sum(vs)/len(vs) if vs else 0
        a0_3 = mean_("A0_pool3","rt_s"); a0_15 = mean_("A0_pool15","rt_s")
        g_3 = mean_("gnn_rank_pool3","rt_s"); g_15 = mean_("gnn_rank_pool15","rt_s")
        c_a0_15 = mean_("A0_pool15","cols_added"); c_g_15 = mean_("gnn_rank_pool15","cols_added")
        prune = (1 - c_g_15 / max(1, c_a0_15)) * 100
        print(f"{size:<10} {a0_3:>8.2f}s {a0_15:>8.2f}s {g_3:>8.2f}s {g_15:>8.2f}s  "
              f"{a0_3/max(1e-6,g_15):>11.2f}x {a0_15/max(1e-6,g_15):>12.2f}x  "
              f"{c_a0_15:>11.0f} {c_g_15:>12.0f}  {prune:>+6.1f}%")

    out_path = Path("Result_a0_vs_gnn_ranker_pool15") / "a0_vs_gnn_ranker_pool15.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "ckpt": GNN_CKPT}, f, indent=2)
    print(f"\n[bench] saved {out_path}")


if __name__ == "__main__":
    main()
