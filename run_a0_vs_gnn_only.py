"""Focused 2-variant benchmark: A0 (classical CG) vs E1_gnn_only.

Both variants solve one Gurobi MIP per active (p,t) pair each CG iteration.
pool_size (IRP_EXACT_PRICING_POOL_SIZE) controls how many solutions Gurobi
extracts per MIP via PoolSolutions + PoolSearchMode=2.

A0_no_penalty  : pool_size=1 → 1 most-negative-RC column per (p,t) per
                 iteration added to RMP (~68 cols/iter for medium).
                 Classical CG baseline, no GNN.

E1_gnn_only    : pool_size=3 → Gurobi extracts 3 candidate columns per (p,t)
                 MIP. GNN scores all 3 within each (p,t) group and selects
                 the 1 highest-GNN-combined-scored pattern per group.
                 Same ~68 cols/iter as A0 but GNN-guided selection.
                 Runtime reported as rt_net = wall-clock - GNN forward-pass
                 overhead, for a fair comparison with A0.

Both converge to the same LP optimal. E1's advantage is solution quality.

Usage:
    python3 run_a0_vs_gnn_only.py
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
os.environ.pop("IRP_DISABLE_RC_FILTER", None)
os.environ.pop("IRP_EXACT_PRICING_POOL_SIZE", None)  # each variant sets its own

import irp_gurobi_converted as irp

# A0: pool_size=1  → Gurobi extracts 1 solution per (p,t) MIP, add 1 col/group/iter
# E1: pool_size=3  → Gurobi extracts 3 solutions per (p,t) MIP, GNN picks top-1/group/iter
VARIANTS = [
    ("A0_no_penalty", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "exact_full_mode": True, "use_branch_and_price": False,
    }, {"IRP_EXACT_PRICING_POOL_SIZE": "1", "IRP_EXACT_GNN_RANKER": "0"}),
    ("E1_gnn_only", {
        "use_gnn": True, "collect_teacher_mode": False,
        "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
        "exact_full_mode": True,
        "gnn_selection_mode": "relative_threshold",
        "gnn_relative_threshold": 0.70,
        "gnn_max_keep_fraction": 0.30,
        "use_branch_and_price": False,
    }, {"IRP_EXACT_PRICING_POOL_SIZE": "3", "IRP_EXACT_GNN_RANKER": "1"}),
]

# 2 per size tier (small / medium-low / medium-high / large), 2 shock types each
SCENARIO_SELECTION = [
    ("small",       "base_01__normal_global__seed1373158607"),
    ("small",       "base_01__sku_spike__seed53710185"),
    ("small-med",   "base_04__normal_global__seed1268073013"),
    ("small-med",   "base_04__sku_spike__seed68252794"),
    ("medium",      "base_09__normal_global__seed1730483679"),
    ("medium",      "base_09__sku_spike__seed1499242942"),
    ("med-large",   "base_21__normal_global__seed1439190227"),
    ("med-large",   "base_21__sku_spike__seed449912920"),
    ("large",       "base_24__normal_global__seed814874364"),
    ("large",       "base_24__sku_spike__seed992696250"),
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
            cg_iterations=150,
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
        bp_nodes = len(result.get("branch_price_history", []) or [])

        # Service level (fill rate) = 1 - residual_shortage / total_realized_demand
        with_lt = result.get("realized_with_lt_cost_breakdown") or {}
        shortage_units = float(with_lt.get("total_realized_shortage_units", float("nan")))
        total_demand = sum(
            float(data_copy.realized_demand.get((s, p, t), 0.0))
            for s in data_copy.stores
            for p in data_copy.products
            for t in data_copy.periods
        )
        sla = (1.0 - shortage_units / total_demand) if total_demand > 1e-9 else float("nan")

        # GNN inference overhead: tracked in efficiency_metrics when runtime_gnn_mode=True.
        # rt_net = wall-clock minus GNN forward-pass time (apples-to-apples vs A0).
        gnn_rt = float((cg_sol.efficiency_metrics or {}).get("gnn_total_runtime", 0.0)) if cg_sol else 0.0
        rt_net = rt - gnn_rt

        return {"obj": obj, "rt": rt, "rt_net": rt_net, "gnn_rt": gnn_rt,
                "iters": cg_iters, "cols_added": cols_added,
                "bp_nodes": bp_nodes, "use_bp": use_bp,
                "shortage_units": shortage_units, "sla": sla, "total_demand": total_demand}
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
        n_stores = len(shocked_data.stores)
        n_skus = len(shocked_data.products)
        n_periods = len(shocked_data.periods)
        baseline_obj = float(baseline_sol.objective)
        print(f"  {n_stores}s×{n_skus}k×{n_periods}p  shock={shock_type}  baseline_obj={baseline_obj:.0f}\n")

        for v_name, kwargs_orig, env_orig in VARIANTS:
            r = run_variant(v_name, kwargs_orig, env_orig, shocked_data, baseline_sol)
            row = {
                "size": size_label, "scenario": name,
                "stores": n_stores, "skus": n_skus, "periods": n_periods,
                "shock_type": shock_type, "shock_seed": shock_seed,
                "variant": v_name,
                "obj": r["obj"], "rt_s": r["rt"], "rt_net_s": r["rt_net"], "gnn_rt_s": r["gnn_rt"],
                "iters": r["iters"], "cols_added": r["cols_added"],
                "bp_nodes": r["bp_nodes"], "baseline_obj": baseline_obj,
                "sla": r["sla"], "shortage_units": r["shortage_units"],
            }
            rows.append(row)
            tag = "[BP]" if r["use_bp"] else "    "
            sla_str = f"{r['sla']:.4%}" if r['sla'] == r['sla'] else "n/a"
            gnn_tag = f"  gnn_overhead={r['gnn_rt']:.2f}s  rt_net={r['rt_net']:.2f}s" if r["gnn_rt"] > 0 else ""
            print(f"  {v_name:<16} {tag}  obj={r['obj']:.0f}  rt={r['rt']:.2f}s{gnn_tag}  "
                  f"iters={r['iters']:>3}  cols={r['cols_added']:>4}  "
                  f"shortage={r['shortage_units']:.1f}  sla={sla_str}")
        print()

    # ── Summary table ──────────────────────────────────────────────────────────
    SEP = "─" * 155
    print(f"\n{SEP}")
    print("  A0 (classical CG, K=1)  vs  E1_gnn_only (GNN per-(p,t) top-1 from K=3 pool)")
    print("  NOTE: E1 rt_net = wall-clock minus GNN forward-pass overhead (apples-to-apples vs A0)")
    print(SEP)
    print(f"  {'size':<9} {'scenario':<44} {'A0_obj':>14} {'gnn_obj':>14} {'Δobj%':>7}  "
          f"{'A0_it':>5} {'gnn_it':>6}  {'A0_rt':>7} {'gnn_net':>8} {'speedup':>8}  "
          f"{'gnn_overhead':>12}  {'A0_col':>6} {'gnn_col':>7}  {'A0_sla':>8} {'gnn_sla':>8} {'Δsla':>7}")
    print(SEP)

    grouped = {}
    for r in rows:
        grouped.setdefault((r["size"], r["scenario"]), {})[r["variant"]] = r

    by_size: dict[str, list] = {}
    for (size, scen), per_v in grouped.items():
        a0 = per_v.get("A0_no_penalty")
        gn = per_v.get("E1_gnn_only")
        if not (a0 and gn):
            continue
        # speedup uses rt_net (E1 minus GNN overhead) for fair comparison
        speedup = a0["rt_s"] / max(1e-6, gn["rt_net_s"])
        d_obj = 100.0 * (gn["obj"] - a0["obj"]) / max(1e-9, abs(a0["obj"]))
        d_sla = (gn["sla"] - a0["sla"]) * 100.0 if (a0["sla"] == a0["sla"] and gn["sla"] == gn["sla"]) else float("nan")
        scen_s = scen[-43:] if len(scen) > 43 else scen
        a0_sla_s = f"{a0['sla']:.3%}" if a0["sla"] == a0["sla"] else "n/a"
        gn_sla_s = f"{gn['sla']:.3%}" if gn["sla"] == gn["sla"] else "n/a"
        d_sla_s = f"{d_sla:+.3f}pp" if d_sla == d_sla else "n/a"
        print(f"  {size:<9} {scen_s:<44} {a0['obj']:>14.0f} {gn['obj']:>14.0f} {d_obj:>+6.3f}%  "
              f"{a0['iters']:>5} {gn['iters']:>6}  {a0['rt_s']:>6.2f}s {gn['rt_net_s']:>7.2f}s {speedup:>7.2f}×  "
              f"{gn['gnn_rt_s']:>11.2f}s  "
              f"{a0['cols_added']:>6} {gn['cols_added']:>7}  {a0_sla_s:>8} {gn_sla_s:>8} {d_sla_s:>7}")
        by_size.setdefault(size, []).append((a0, gn))

    print(f"\n{SEP}")
    print("  AGGREGATE BY SIZE (mean)  |  speedup uses E1 rt_net (wall-clock minus GNN overhead)")
    print(SEP)
    print(f"  {'size':<9} {'n':>3}  {'A0_rt':>8} {'gnn_net':>8} {'speedup':>8}  "
          f"{'gnn_ovhd':>9}  {'A0_it':>6} {'gnn_it':>7}  {'A0_col':>7} {'gnn_col':>8}  {'mean_Δobj%':>10}  "
          f"{'A0_sla':>8} {'gnn_sla':>8} {'Δsla(pp)':>10}")
    for size in ["small", "small-med", "medium", "med-large", "large"]:
        pairs = by_size.get(size, [])
        if not pairs:
            continue
        n = len(pairs)
        a0_rt = sum(a["rt_s"] for a, _ in pairs) / n
        gn_rt_net = sum(g["rt_net_s"] for _, g in pairs) / n
        gn_rt_ovhd = sum(g["gnn_rt_s"] for _, g in pairs) / n
        a0_it = sum(a["iters"] for a, _ in pairs) / n
        gn_it = sum(g["iters"] for _, g in pairs) / n
        a0_col = sum(a["cols_added"] for a, _ in pairs) / n
        gn_col = sum(g["cols_added"] for _, g in pairs) / n
        d_obj = sum(
            100.0 * (g["obj"] - a["obj"]) / max(1e-9, abs(a["obj"]))
            for a, g in pairs
        ) / n
        a0_sla_mean = sum(a["sla"] for a, _ in pairs if a["sla"] == a["sla"]) / n
        gn_sla_mean = sum(g["sla"] for _, g in pairs if g["sla"] == g["sla"]) / n
        d_sla_mean = (gn_sla_mean - a0_sla_mean) * 100.0
        print(f"  {size:<9} {n:>3}  {a0_rt:>7.2f}s {gn_rt_net:>7.2f}s {a0_rt/max(1e-6,gn_rt_net):>7.2f}×  "
              f"{gn_rt_ovhd:>8.2f}s  {a0_it:>6.1f} {gn_it:>7.1f}  {a0_col:>7.0f} {gn_col:>8.0f}  {d_obj:>+9.4f}%  "
              f"{a0_sla_mean:>7.3%} {gn_sla_mean:>7.3%} {d_sla_mean:>+9.4f}pp")

    out_dir = Path("Result_a0_vs_gnn_only")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "a0_vs_gnn_only.json"
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "ckpt": GNN_CKPT}, f, indent=2)
    print(f"\n[bench] saved {out_path}")


if __name__ == "__main__":
    main()
