"""Benchmark E1 + GNN-only ablation on test scenarios.

Adds a 4th variant — E1_gnn_only — that DISABLES RC pre-filter, leaving the
GNN as the sole pruning mechanism. Helps answer: "what does the GNN do alone,
without analytical RC filtering as a safety net?"

Variants:
  - A0_no_penalty   : exact_full_mode + BP=True
  - E1_rc_only      : RC filter only
  - E1_rc_gnn       : RC filter + GNN re-rank   (current production E1)
  - E1_gnn_only     : NO RC filter, GNN selects from full candidate pool
                      (set IRP_DISABLE_RC_FILTER=1 inside _candidate_patterns_rc_only)

Usage:
    IRP_GNN_CHECKPOINT=GNN/trained_models/irplt_teacher_filtered_local200_fixed/bigat/pairwise_rank/best_model.pt \
    python3 run_test_scenarios_gnn_only.py
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
os.environ.setdefault("IRP_RESULTS_DIR_OVERRIDE", "Result_E2_local_benchmark_gnn_only")
os.environ.pop("IRP_SLA_PENALTY", None)

import irp_gurobi_converted as irp


# (variant_name, kwargs, env_overrides)
VARIANTS = [
    ("A0_no_penalty", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "exact_full_mode": True, "use_branch_and_price": True,
    }, {}),
    ("E1_rc_only", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "rc_filter_mode": True, "use_branch_and_price": False,
    }, {}),
    ("E1_rc_gnn", {
        "use_gnn": True, "collect_teacher_mode": False,
        "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
        "rc_filter_mode": True,
        "gnn_selection_mode": "relative_threshold",
        "gnn_relative_threshold": 0.70,
        "gnn_max_keep_fraction": 0.30,
        "use_branch_and_price": False,
    }, {}),
    ("E1_gnn_only", {
        "use_gnn": True, "collect_teacher_mode": False,
        "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
        "rc_filter_mode": True,    # uses _candidate_patterns_rc_only path
        "gnn_selection_mode": "relative_threshold",
        "gnn_relative_threshold": 0.70,
        "gnn_max_keep_fraction": 0.30,
        "use_branch_and_price": False,
    }, {"IRP_DISABLE_RC_FILTER": "1"}),  # ← key difference: no RC filter
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


def run_variant(v_name, kwargs, env_overrides, shocked_data, baseline_sol):
    # Apply env overrides (and remember to reset)
    saved_env = {}
    for k, v in env_overrides.items():
        saved_env[k] = os.environ.get(k)
        os.environ[k] = v
    # Reset env vars NOT in overrides
    for env_var in ["IRP_DISABLE_RC_FILTER"]:
        if env_var not in env_overrides:
            saved_env.setdefault(env_var, os.environ.get(env_var))
            os.environ.pop(env_var, None)

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
        # Restore env
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
        print(f"  size: {n_stores}×{n_skus}×{n_periods}  shock={shock_type} seed={shock_seed}  baseline obj={baseline_obj:.2f}\n")

        for v_name, kwargs_orig, env_orig in VARIANTS:
            r = run_variant(v_name, kwargs_orig, env_orig, shocked_data, baseline_sol)
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
                  f"iters={r['iters']}  cols_proposed={r['cols_proposed']}  cols_added={r['cols_added']}")
        print()

    # Summary table
    print("\n" + "#" * 115)
    print("# E1 + GNN-ONLY BENCHMARK SUMMARY (200-epoch checkpoint, test scenarios)")
    print("#" * 115)
    print(f"{'size':<7} {'scenario':<55} {'A0_rt':>7} {'rco_rt':>7} {'rcg_rt':>7} {'gnn_rt':>7}  "
          f"{'A0_it':>5} {'rco':>5} {'rcg':>5} {'gnn':>5}  {'spd_rco':>8} {'spd_rcg':>8} {'spd_gnn':>8}")
    grouped = {}
    for r in rows:
        key = (r["size"], r["scenario"])
        grouped.setdefault(key, {})[r["variant"]] = r
    for (size, scen), per_v in grouped.items():
        a0 = per_v.get("A0_no_penalty"); ro = per_v.get("E1_rc_only")
        rg = per_v.get("E1_rc_gnn"); go = per_v.get("E1_gnn_only")
        if not (a0 and ro and rg and go): continue
        s_ro = a0["rt_s"] / max(1e-6, ro["rt_s"])
        s_rg = a0["rt_s"] / max(1e-6, rg["rt_s"])
        s_go = a0["rt_s"] / max(1e-6, go["rt_s"])
        scen_short = scen[-50:] if len(scen) > 50 else scen
        print(f"{size:<7} {scen_short:<55} {a0['rt_s']:>6.1f}s {ro['rt_s']:>6.1f}s {rg['rt_s']:>6.1f}s {go['rt_s']:>6.1f}s  "
              f"{a0['iters']:>5} {ro['iters']:>5} {rg['iters']:>5} {go['iters']:>5}  "
              f"{s_ro:>7.2f}x {s_rg:>7.2f}x {s_go:>7.2f}x")

    print("\n# AGGREGATE BY SIZE (mean across scenarios)")
    print("#" * 115)
    print(f"{'size':<10} {'n':>3} {'A0_rt':>8} {'rco_rt':>8} {'rcg_rt':>8} {'gnn_rt':>8}  "
          f"{'A0_it':>6} {'rco':>5} {'rcg':>5} {'gnn':>5}  "
          f"{'spd_rco':>8} {'spd_rcg':>8} {'spd_gnn':>8}")
    by_size = {}
    for r in rows:
        by_size.setdefault(r["size"], []).append(r)
    for size in ["small", "medium", "large"]:
        size_rows = by_size.get(size, [])
        n = len([r for r in size_rows if r["variant"] == "A0_no_penalty"])
        if n == 0: continue
        def mean_rt(v): vs = [r["rt_s"] for r in size_rows if r["variant"]==v]; return sum(vs)/len(vs) if vs else 0
        def mean_it(v): vs = [r["iters"] for r in size_rows if r["variant"]==v]; return sum(vs)/len(vs) if vs else 0
        a0m, rom, rgm, gom = mean_rt("A0_no_penalty"), mean_rt("E1_rc_only"), mean_rt("E1_rc_gnn"), mean_rt("E1_gnn_only")
        a0i, roi, rgi, goi = mean_it("A0_no_penalty"), mean_it("E1_rc_only"), mean_it("E1_rc_gnn"), mean_it("E1_gnn_only")
        print(f"{size:<10} {n:>3} {a0m:>7.2f}s {rom:>7.2f}s {rgm:>7.2f}s {gom:>7.2f}s  "
              f"{a0i:>6.1f} {roi:>5.1f} {rgi:>5.1f} {goi:>5.1f}  "
              f"{a0m/max(1e-6,rom):>7.2f}x {a0m/max(1e-6,rgm):>7.2f}x {a0m/max(1e-6,gom):>7.2f}x")

    # Cols proposed/added breakdown
    print("\n# COLS DETAIL (mean across scenarios)")
    print("#" * 115)
    print(f"{'size':<10} {'A0_added':>10} {'rco_prop':>10} {'rco_added':>11} {'rcg_prop':>10} {'rcg_added':>11} {'gnn_prop':>10} {'gnn_added':>11}")
    for size in ["small", "medium", "large"]:
        size_rows = by_size.get(size, [])
        if not size_rows: continue
        def mean_col(v, key):
            vs = [r[key] for r in size_rows if r["variant"]==v]
            return sum(vs)/len(vs) if vs else 0
        print(f"{size:<10} {mean_col('A0_no_penalty','cols_added'):>10.0f} "
              f"{mean_col('E1_rc_only','cols_proposed'):>10.0f} {mean_col('E1_rc_only','cols_added'):>11.0f} "
              f"{mean_col('E1_rc_gnn','cols_proposed'):>10.0f} {mean_col('E1_rc_gnn','cols_added'):>11.0f} "
              f"{mean_col('E1_gnn_only','cols_proposed'):>10.0f} {mean_col('E1_gnn_only','cols_added'):>11.0f}")

    out_path = Path("Result_E2_local_benchmark_gnn_only") / "e1_gnn_only.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "ckpt": GNN_CKPT}, f, indent=2)
    print(f"\n[bench] saved {out_path}")


if __name__ == "__main__":
    main()
