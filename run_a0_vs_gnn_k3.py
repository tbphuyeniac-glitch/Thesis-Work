"""A0 vs E1_gnn_only (k=1 and k=3) on 3 pre-computed test scenarios.

E1_gnn_only = NO RC pre-filter; GNN is the sole pruning mechanism.
IRP_DISABLE_RC_FILTER=1 makes _candidate_patterns_rc_only skip the
negative-RC gate so the GNN scores the full (i,j) candidate pool.

  k=1: n_to_build=1 per (p,t) — 1 best column (most-negative-RC seed),
       diversity from different (p,t) pairs  ← correct default
  k=3: n_to_build=3 per (p,t) — old behavior, adds 3 cols from same
       (p,t), RMP grows 3× while LP only uses 1  ← comparison point

Usage:
    IRP_GNN_CHECKPOINT=GNN/trained_models/irplt_teacher_filtered_local200_fixed/bigat/pairwise_rank/best_model.pt \
    python3 run_a0_vs_gnn_k3.py
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
os.environ.pop("IRP_SLA_PENALTY", None)

import irp_gurobi_converted as irp

GNN_CKPT = os.environ.get(
    "IRP_GNN_CHECKPOINT",
    "GNN/trained_models/irplt_teacher_filtered_local200_fixed/bigat/pairwise_rank/best_model.pt",
)
SCENARIOS_DIR = Path("Test 30 scenarios/test_baselines/scenarios")

SCENARIO_SELECTION = [
    ("small",  "base_01__normal_global__seed1373158607"),
    ("medium", "base_09__normal_global__seed1730483679"),
    ("large",  "base_24__normal_global__seed814874364"),
]

# (variant_name, kwargs, env_overrides)
VARIANTS = [
    ("A0_no_penalty", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "exact_full_mode": True, "use_branch_and_price": False,
    }, {}),
    # GNN-only k=1: no RC pre-filter, 1 best col per (p,t), GNN re-ranks
    ("E1_gnn_only_k1", {
        "use_gnn": True, "collect_teacher_mode": False,
        "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
        "rc_filter_mode": True,        # uses _candidate_patterns_rc_only path
        "rc_only_cols_per_pp": 1,      # 1 col per (p,t), most-negative-RC seed
        "gnn_selection_mode": "relative_threshold",
        "gnn_relative_threshold": 0.70,
        "gnn_max_keep_fraction": 0.30,
        "use_branch_and_price": False,
    }, {"IRP_DISABLE_RC_FILTER": "1"}),  # bypass RC gate → GNN sees full pool
    # GNN-only k=3: same but n_to_build=3 per (p,t) — old behavior for comparison
    ("E1_gnn_only_k3", {
        "use_gnn": True, "collect_teacher_mode": False,
        "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
        "rc_filter_mode": True,
        "rc_only_cols_per_pp": 3,      # old: 3 cols per (p,t), RMP grows 3×
        "gnn_selection_mode": "relative_threshold",
        "gnn_relative_threshold": 0.70,
        "gnn_max_keep_fraction": 0.30,
        "use_branch_and_price": False,
    }, {"IRP_DISABLE_RC_FILTER": "1"}),
]


def load_scenario(name):
    pkl = SCENARIOS_DIR / f"{name}.pkl.gz"
    with gzip.open(pkl, "rb") as f:
        s = pickle.load(f)
    return s["shocked_data"], s["baseline_sol"], s.get("shock_type", ""), s.get("shock_seed", "")


def run_variant(v_name, kwargs, env_overrides, shocked_data, baseline_sol):
    # Apply env overrides and restore afterward
    saved_env = {}
    for k, v in env_overrides.items():
        saved_env[k] = os.environ.get(k)
        os.environ[k] = v
    for env_var in ["IRP_DISABLE_RC_FILTER"]:
        if env_var not in env_overrides:
            saved_env.setdefault(env_var, os.environ.get(env_var))
            os.environ.pop(env_var, None)

    try:
        data_copy = deepcopy(shocked_data)
        baseline_copy = deepcopy(baseline_sol)
        kw = dict(kwargs)
        use_bp = kw.pop("use_branch_and_price", False)
        pipeline = irp.IRPResearchPipeline(data_copy)
        t0 = time.perf_counter()
        result = pipeline.run_lt_recourse_from_baseline(
            baseline_copy,
            use_random_initial_patterns=True,
            n_initial_patterns_per_product_period=5,
            cg_iterations=15,
            msg=False,
            gnn_checkpoint=GNN_CKPT if kw.get("use_gnn") else None,
            use_classical_fallback=False,
            gnn_max_keep=150,
            use_branch_and_price=use_bp,
            bp_max_nodes=15, bp_max_depth=6,
            lt_activation_threshold=10.0,
            **kw,
        )
        rt = time.perf_counter() - t0
        cg_sol = result.get("cg_solution")
        obj = float(getattr(cg_sol, "objective", float("nan")))
        history = result.get("cg_episode_history", []) or []
        iters = max(0, len(history) - 1)
        cols_added = sum(int(h.get("added_columns", 0)) for h in history)
        cols_proposed = sum(int(h.get("proposed_columns", 0)) for h in history)
        return {"obj": obj, "rt": rt, "iters": iters,
                "cols_added": cols_added, "cols_proposed": cols_proposed,
                "use_bp": use_bp}
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
        print("=" * 90)
        print(f"[scenario] {size_label} | {name}")
        print("=" * 90)
        shocked_data, baseline_sol, shock_type, shock_seed = load_scenario(name)
        n_s = len(shocked_data.stores)
        n_p = len(shocked_data.products)
        n_t = len(shocked_data.periods)
        baseline_obj = float(baseline_sol.objective)
        print(f"  size: {n_s}×{n_p}×{n_t}  shock={shock_type}  baseline obj={baseline_obj:.2f}\n")

        for v_name, kw_orig, env_orig in VARIANTS:
            r = run_variant(v_name, kw_orig, env_orig, shocked_data, baseline_sol)
            row = {
                "size": size_label, "scenario": name,
                "variant": v_name,
                "obj": r["obj"], "rt_s": r["rt"],
                "iters": r["iters"],
                "cols_added": r["cols_added"],
                "cols_proposed": r["cols_proposed"],
                "baseline_obj": baseline_obj,
            }
            rows.append(row)
            tag = "BP" if r["use_bp"] else "noBP"
            cpi = r["cols_added"] / max(1, r["iters"])
            print(f"  {v_name:<16} [{tag}]  obj={r['obj']:.2f}  rt={r['rt']:.2f}s  "
                  f"iters={r['iters']}  cols={r['cols_added']}  cols/iter={cpi:.1f}")
        print()

    print("\n" + "#" * 90)
    print("# SUMMARY (per scenario)")
    print("#" * 90)
    print(f"{'size':<8} {'scenario':<52} {'variant':<16} {'obj':>14} {'rt':>7} {'iters':>6} {'cols':>6} {'c/i':>6}")
    for r in rows:
        cpi = r["cols_added"] / max(1, r["iters"])
        print(f"{r['size']:<8} {r['scenario'][-50:]:<52} {r['variant']:<16} "
              f"{r['obj']:>14.2f} {r['rt_s']:>6.2f}s {r['iters']:>6} {r['cols_added']:>6} {cpi:>6.1f}")

    print("\n# AGGREGATE (mean across 3 scenarios)")
    print("#" * 90)
    for v_name, _, __ in VARIANTS:
        v_rows = [r for r in rows if r["variant"] == v_name]
        n = len(v_rows) or 1
        m_rt  = sum(r["rt_s"] for r in v_rows) / n
        m_obj = sum(r["obj"] for r in v_rows) / n
        m_it  = sum(r["iters"] for r in v_rows) / n
        m_col = sum(r["cols_added"] for r in v_rows) / n
        print(f"  {v_name:<16}  obj_mean={m_obj:.2f}  rt_mean={m_rt:.2f}s  "
              f"iters_mean={m_it:.1f}  cols_mean={m_col:.0f}")

    # Speed ratio vs A0
    a0_rows = {r["scenario"]: r for r in rows if r["variant"] == "A0_no_penalty"}
    print("\n# SPEED RATIO vs A0")
    for v_name, _, __ in VARIANTS[1:]:
        ratios = []
        for scen, a0r in a0_rows.items():
            vr = next((r for r in rows if r["variant"] == v_name and r["scenario"] == scen), None)
            if vr:
                ratios.append(a0r["rt_s"] / max(1e-6, vr["rt_s"]))
        if ratios:
            print(f"  {v_name:<16}  speedup vs A0: {sum(ratios)/len(ratios):.2f}x  "
                  f"(per scenario: {', '.join(f'{x:.2f}x' for x in ratios)})")

    out = Path("Result_a0_vs_gnn_only") / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"rows": rows, "ckpt": GNN_CKPT}, f, indent=2)
    print(f"\n[bench] saved → {out}")


if __name__ == "__main__":
    main()
