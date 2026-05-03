"""Clean 2-variant benchmark: A0 (exact pricing) vs A0 + GNN ranker.

Both variants use IDENTICAL pricing — exact Gurobi MIP — so any difference
in runtime/objective is purely from GNN's RANKING contribution. This is the
proper ablation for "what does GNN add beyond exact pricing?"

Variants:
  - A0_no_penalty            : exact_full_mode + BP, no GNN
  - E1_gnn_ranker            : exact_full_mode + BP + GNN selects from MIP pool

Both use:
  - exact_full_mode=True (Gurobi MIP pricing per p,t with K-best pool)
  - exact_pricing_pool_size=1 (classical CG: 1 best column per p,t per iteration)
  - use_branch_and_price=True (production setup)

Difference: E1_gnn_ranker has runtime_gnn_mode=True → GNN scores the union
of MIP columns and keeps top-K via relative_threshold + max_keep_fraction.

Usage:
    IRP_GNN_CHECKPOINT=GNN/trained_models/irplt_teacher_filtered_local200_fixed/bigat/pairwise_rank/best_model.pt \
    python3 run_a0_vs_gnn_ranker.py
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
os.environ.setdefault("IRP_RESULTS_DIR_OVERRIDE", "Result_a0_vs_gnn_ranker")
os.environ.pop("IRP_SLA_PENALTY", None)
os.environ.pop("IRP_DISABLE_RC_FILTER", None)

import irp_gurobi_converted as irp


VARIANTS = [
    ("A0_no_penalty", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "exact_full_mode": True,
        "use_branch_and_price": True,
    }),
    ("E1_gnn_ranker", {
        "use_gnn": True, "collect_teacher_mode": False,
        "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
        "exact_full_mode": True,                   # ← SAME exact pricing as A0
        "gnn_selection_mode": "relative_threshold",
        "gnn_relative_threshold": 0.70,
        "gnn_max_keep_fraction": 0.30,
        "use_branch_and_price": True,
    }),
]

SCENARIO_SELECTION = [
    ("small",  "base_01__normal_global__seed1373158607"),
    ("small",  "base_01__sku_spike__seed53710185"),
    ("medium", "base_09__normal_global__seed1730483679"),
    ("medium", "base_09__sku_spike__seed1499242942"),
    ("large",  "base_24__normal_global__seed814874364"),
    ("large",  "base_24__sku_spike__seed992696250"),
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


def run_variant(v_name, kwargs, shocked_data, baseline_sol):
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
        print(f"  size: {n_stores}×{n_skus}×{n_periods}  shock={shock_type} seed={shock_seed}  baseline obj={baseline_obj:.2f}\n")

        for v_name, kwargs_orig in VARIANTS:
            r = run_variant(v_name, kwargs_orig, shocked_data, baseline_sol)
            row = {
                "size": size_label, "scenario": name,
                "stores": n_stores, "skus": n_skus, "periods": n_periods,
                "shock_type": shock_type, "shock_seed": shock_seed,
                "variant": v_name,
                "use_bp": r["use_bp"],
                "obj": r["obj"], "rt_s": r["rt"],
                "iters": r["iters"], "bp_nodes": r["bp_nodes"],
                "cols_added": r["cols_added"], "cols_proposed": r["cols_proposed"],
                "baseline_obj": baseline_obj,
            }
            rows.append(row)
            tag = "BP" if r["use_bp"] else "noBP"
            print(f"  {v_name:<14} [{tag}] obj={r['obj']:.2f}  rt={r['rt']:.2f}s  "
                  f"iters={r['iters']}  bp_nodes={r['bp_nodes']}  "
                  f"cols_proposed={r['cols_proposed']}  cols_added={r['cols_added']}")
        print()

    print("\n" + "#" * 110)
    print("# A0 vs GNN-RANKER BENCHMARK SUMMARY (200-epoch checkpoint, test scenarios)")
    print("# Both variants use IDENTICAL pricing (exact MIP, K=5). Difference: GNN ranker on top.")
    print("#" * 110)
    print(f"{'size':<7} {'scenario':<55} {'A0_rt':>8} {'gnn_rt':>8} {'speedup':>8}  "
          f"{'A0_it':>6} {'gnn_it':>7}  {'A0_obj':>15} {'gnn_obj':>15} {'Δobj%':>8}")
    grouped = {}
    for r in rows:
        key = (r["size"], r["scenario"])
        grouped.setdefault(key, {})[r["variant"]] = r
    for (size, scen), per_v in grouped.items():
        a0 = per_v.get("A0_no_penalty"); gn = per_v.get("E1_gnn_ranker")
        if not (a0 and gn): continue
        speedup = a0["rt_s"] / max(1e-6, gn["rt_s"])
        d_obj = 100 * (gn["obj"] - a0["obj"]) / max(1e-9, abs(a0["obj"]))
        scen_short = scen[-50:] if len(scen) > 50 else scen
        print(f"{size:<7} {scen_short:<55} {a0['rt_s']:>7.1f}s {gn['rt_s']:>7.1f}s {speedup:>7.2f}x  "
              f"{a0['iters']:>6} {gn['iters']:>7}  {a0['obj']:>15.0f} {gn['obj']:>15.0f} {d_obj:>+7.4f}%")

    print("\n# AGGREGATE BY SIZE (mean across scenarios)")
    print("#" * 110)
    print(f"{'size':<10} {'n':>3} {'A0_rt':>10} {'gnn_rt':>10} {'speedup':>9}  "
          f"{'A0_it':>7} {'gnn_it':>7}  {'A0_cols':>9} {'gnn_cols':>9}  {'mean_Δobj%':>10}")
    by_size = {}
    for r in rows:
        by_size.setdefault(r["size"], []).append(r)
    for size in ["small", "medium", "large"]:
        size_rows = by_size.get(size, [])
        n = len([r for r in size_rows if r["variant"] == "A0_no_penalty"])
        if n == 0: continue
        def mean_(v, key):
            vs = [r[key] for r in size_rows if r["variant"]==v]
            return sum(vs)/len(vs) if vs else 0
        a0m, gnm = mean_("A0_no_penalty","rt_s"), mean_("E1_gnn_ranker","rt_s")
        a0i, gni = mean_("A0_no_penalty","iters"), mean_("E1_gnn_ranker","iters")
        a0c, gnc = mean_("A0_no_penalty","cols_added"), mean_("E1_gnn_ranker","cols_added")
        a0o, gno = mean_("A0_no_penalty","obj"), mean_("E1_gnn_ranker","obj")
        d_obj = 100 * (gno - a0o) / max(1e-9, abs(a0o))
        print(f"{size:<10} {n:>3} {a0m:>9.2f}s {gnm:>9.2f}s {a0m/max(1e-6,gnm):>8.2f}x  "
              f"{a0i:>7.1f} {gni:>7.1f}  {a0c:>9.0f} {gnc:>9.0f}  {d_obj:>+9.4f}%")

    out_path = Path("Result_a0_vs_gnn_ranker") / "a0_vs_gnn_ranker.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "ckpt": GNN_CKPT}, f, indent=2)
    print(f"\n[bench] saved {out_path}")


if __name__ == "__main__":
    main()
