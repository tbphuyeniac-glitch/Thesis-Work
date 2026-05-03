"""Small-scale local CG test: A0 vs E1_gnn_only (k=1 and k=3), global column budget.

Uses a locally-generated small instance (3 stores × 2 SKUs) so total columns
stay well under 300.  Demand shock is applied before solving, so
data.realized_demand reflects post-shock actuals.

Column budget is GLOBAL per iteration:
  A0     : pool_size=1 per (p,t) MIP, keep 1 globally (most-negative RC)
  E1_k=1 : pool_size=3 per (p,t) MIP, GNN scores all, keep 1 globally
  E1_k=3 : pool_size=3 per (p,t) MIP, GNN scores all, keep 3 globally

No max-CG-iteration cap — convergence stopping only (stop when added==0).

Usage:
    python3 run_small_scale_cg_test.py
"""
from __future__ import annotations
import os, time, json
from pathlib import Path
from copy import deepcopy

# ── Environment (before irp import) ──────────────────────────────────────────
os.environ["IRP_STORE_LIMIT"] = "3"
os.environ["IRP_SKU_LIMIT"]   = "2"
os.environ["IRP_TIME_LIMIT"]  = "120"
os.environ["IRP_QUIET"]       = "1"
os.environ["IRP_CG_STOPPING_MODE"] = "convergence"
os.environ.pop("IRP_SLA_PENALTY",           None)
os.environ.pop("IRP_DISABLE_RC_FILTER",     None)
os.environ.pop("IRP_EXACT_PRICING_POOL_SIZE", None)

import irp_gurobi_converted as irp

GNN_CKPT = os.environ.get(
    "IRP_GNN_CHECKPOINT",
    "GNN/trained_models/irplt_teacher_filtered_local200_fixed/bigat/pairwise_rank/best_model.pt",
)
EXCEL_PATH = Path("1BISCR501V_90100140_20260323-150407111_filtered_sites.csv")

MAX_ITER = 9999   # effectively no cap; stop only when added==0

# 2 independent seeds → 2 scenarios
SHOCK_SEEDS = [20260418, 20260501]

VARIANTS = [
    # (name, kwargs_for_run_lt_recourse, env_overrides, global_col_budget_per_iter)
    ("A0",
     {
         "use_gnn": False, "collect_teacher_mode": False,
         "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
         "exact_full_mode": True, "use_branch_and_price": False,
     },
     {"IRP_EXACT_PRICING_POOL_SIZE": "1", "IRP_EXACT_GNN_RANKER": "0"},
     1,
    ),
    ("E1_k=1",
     {
         "use_gnn": True, "collect_teacher_mode": False,
         "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
         "exact_full_mode": True,
         "gnn_selection_mode": "relative_threshold",
         "gnn_relative_threshold": 0.70,
         "gnn_max_keep_fraction": 0.30,
         "use_branch_and_price": False,
     },
     {"IRP_EXACT_PRICING_POOL_SIZE": "3", "IRP_EXACT_GNN_RANKER": "1"},
     1,
    ),
    ("E1_k=3",
     {
         "use_gnn": True, "collect_teacher_mode": False,
         "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
         "exact_full_mode": True,
         "gnn_selection_mode": "relative_threshold",
         "gnn_relative_threshold": 0.70,
         "gnn_max_keep_fraction": 0.30,
         "use_branch_and_price": False,
     },
     {"IRP_EXACT_PRICING_POOL_SIZE": "3", "IRP_EXACT_GNN_RANKER": "1"},
     3,
    ),
]


N_PERIODS = 14  # 14-day horizon → ~6 active (p,t) pairs after shock

def build_base_data():
    import pandas as pd
    # Discover first N_PERIODS distinct dates in the CSV
    raw = pd.read_csv(str(EXCEL_PATH))
    raw["_dt"] = pd.to_datetime(raw["PERIOD"].astype(str), format="%Y%m%d", errors="coerce")
    dates = sorted(raw["_dt"].dropna().unique())
    if len(dates) < N_PERIODS:
        raise RuntimeError(f"CSV has only {len(dates)} dates, need {N_PERIODS}")
    start_str = str(dates[0])[:10]
    end_str   = str(dates[N_PERIODS - 1])[:10]

    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=str(EXCEL_PATH),
        sheet_name="Sheet1",
        store_limit=int(os.environ["IRP_STORE_LIMIT"]),
        sku_limit=int(os.environ["IRP_SKU_LIMIT"]),
        start_date=start_str,
        end_date=end_str,
    )
    data, _, _, _ = mapper.build_irp_data(
        wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2,
        shortage_cost_rate=0.05,
        holding_cost_rate=100,
    )
    return data


def solve_baseline(data):
    return irp.BaselineALNSModel(data).solve(
        msg=False, time_limit=60,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )


def run_variant(kwargs, env_overrides, global_k, data, baseline_sol):
    saved_env = {}
    for k, v in env_overrides.items():
        saved_env[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        data_copy     = deepcopy(data)          # has realized_demand already set
        baseline_copy = deepcopy(baseline_sol)
        kw = dict(kwargs)
        kw.pop("use_branch_and_price", None)    # always False here

        pipeline = irp.IRPResearchPipeline(data_copy)
        t0 = time.perf_counter()
        result = pipeline.run_lt_recourse_from_baseline(
            baseline_copy,
            use_random_initial_patterns=True,
            n_initial_patterns_per_product_period=5,
            cg_iterations=MAX_ITER,
            msg=False,
            gnn_checkpoint=GNN_CKPT if kw.get("use_gnn") else None,
            use_classical_fallback=False,
            gnn_max_keep=150,
            use_branch_and_price=False,
            lt_activation_threshold=10.0,
            global_col_budget_per_iter=global_k,
            **kw,
        )
        rt = time.perf_counter() - t0

        cg_sol     = result.get("cg_solution")
        obj        = float(getattr(cg_sol, "objective", float("nan")))
        cg_history = result.get("cg_episode_history", []) or []
        cg_iters   = max(0, len(cg_history) - 1)
        cols_added = sum(int(h.get("added_columns", 0)) for h in cg_history)

        with_lt         = result.get("realized_with_lt_cost_breakdown") or {}
        shortage_units  = float(with_lt.get("total_realized_shortage_units", float("nan")))
        total_demand    = sum(
            float(data_copy.realized_demand.get((s, p, t), 0.0))
            for s in data_copy.stores
            for p in data_copy.products
            for t in data_copy.periods
        )
        sla = (1.0 - shortage_units / total_demand) if total_demand > 1e-9 else float("nan")

        gnn_rt  = float((cg_sol.efficiency_metrics or {}).get("gnn_total_runtime", 0.0)) if cg_sol else 0.0
        rt_net  = rt - gnn_rt
        return {
            "obj": obj, "rt": rt, "rt_net": rt_net, "gnn_rt": gnn_rt,
            "iters": cg_iters, "cols_added": cols_added,
            "shortage_units": shortage_units, "sla": sla,
            "total_demand": total_demand,
        }
    finally:
        for k, prev in saved_env.items():
            if prev is None: os.environ.pop(k, None)
            else:            os.environ[k] = prev


def main():
    if not Path(GNN_CKPT).exists():
        raise SystemExit(f"GNN checkpoint not found: {GNN_CKPT}")
    print(f"[test] GNN checkpoint : {GNN_CKPT}")
    print(f"[test] Instance size   : {os.environ['IRP_STORE_LIMIT']} stores × "
          f"{os.environ['IRP_SKU_LIMIT']} SKUs")
    print(f"[test] Stopping mode  : convergence only (max_iter={MAX_ITER}, no hard cap)")
    print(f"[test] Column budget  : global / iter  (A0→1, E1_k=1→1, E1_k=3→3)\n")

    print("[setup] Building base IRP data ...")
    base_data = build_base_data()
    n_stores  = len(base_data.stores)
    n_skus    = len(base_data.products)
    n_periods = len(base_data.periods)
    print(f"  {n_stores} stores × {n_skus} SKUs × {n_periods} periods")

    print("[setup] Solving shared ALNS baseline ...")
    t0 = time.perf_counter()
    baseline_sol = solve_baseline(base_data)
    print(f"  baseline obj={float(baseline_sol.objective):.2f}  ({time.perf_counter()-t0:.1f}s)\n")

    rows = []
    for seed_idx, seed in enumerate(SHOCK_SEEDS):
        print("=" * 80)
        print(f"[scenario {seed_idx}] demand shock seed={seed}")
        print("=" * 80)

        # Apply shock: sets data.realized_demand to post-shock values
        data_shocked = deepcopy(base_data)
        shock_summary = irp.apply_hidden_local_reallocation_demand_shocks(
            data_shocked,
            baseline_solution=deepcopy(baseline_sol),
            shock_probability=0.85,
            max_reallocation_fraction=0.60,
            reallocations_per_product_period=3,
            non_dispatch_shock_multiplier=1.8,
            seed=seed,
        )
        total_realized = sum(
            float(data_shocked.realized_demand.get((s, p, t), 0.0))
            for s in data_shocked.stores
            for p in data_shocked.products
            for t in data_shocked.periods
        )
        print(f"  shock: n_shocked={shock_summary.get('n_shocked_product_periods', '?')}  "
              f"total_realized_demand={total_realized:.1f}\n")

        for v_name, kwargs_orig, env_orig, global_k in VARIANTS:
            r = run_variant(kwargs_orig, env_orig, global_k,
                            data_shocked, baseline_sol)
            row = {
                "scenario": seed_idx, "seed": seed,
                "stores": n_stores, "skus": n_skus, "periods": n_periods,
                "variant": v_name, "global_k": global_k,
                "obj": r["obj"],
                "rt_s": r["rt"], "rt_net_s": r["rt_net"], "gnn_rt_s": r["gnn_rt"],
                "iters": r["iters"], "cols_added": r["cols_added"],
                "shortage_units": r["shortage_units"],
                "sla": r["sla"],
                "baseline_obj": float(baseline_sol.objective),
            }
            rows.append(row)
            sla_str  = f"{r['sla']:.4%}" if r["sla"] == r["sla"] else "n/a"
            gnn_tag  = f"  gnn={r['gnn_rt']:.2f}s  rt_net={r['rt_net']:.2f}s" \
                       if r["gnn_rt"] > 0 else ""
            print(f"  {v_name:<10} global_k={global_k}"
                  f"  obj={r['obj']:.0f}"
                  f"  rt={r['rt']:.2f}s{gnn_tag}"
                  f"  iters={r['iters']:>3}  cols={r['cols_added']:>3}"
                  f"  shortage={r['shortage_units']:.1f}  sla={sla_str}")
        print()

    # ── Summary comparison ────────────────────────────────────────────────────
    SEP = "─" * 115
    print(f"\n{SEP}")
    print("  COMPARISON  (A0 vs E1_k=1 vs E1_k=3 — global column budget per iteration)")
    print(SEP)
    print(f"  {'scen':>4}  {'variant':<10}  {'k':>2}  {'obj':>12}  "
          f"{'iters':>5}  {'cols':>4}  {'rt':>7}  {'rt_net':>7}  "
          f"{'sla':>9}  {'Δobj%':>8}")
    print(SEP)

    a0_by_scen: dict = {}
    for r in rows:
        if r["variant"] == "A0":
            a0_by_scen[r["scenario"]] = r

    for r in rows:
        a0 = a0_by_scen.get(r["scenario"])
        d_obj    = 100.0*(r["obj"]-a0["obj"])/max(1e-9,abs(a0["obj"])) if a0 else float("nan")
        sla_str  = f"{r['sla']:.4%}" if r["sla"]==r["sla"] else "n/a"
        d_str    = f"{d_obj:+.3f}%" if d_obj==d_obj else "n/a"
        net_str  = f"{r['rt_net_s']:>6.2f}s" if r["gnn_rt_s"]>0 else "    —  "
        print(f"  {r['scenario']:>4}  {r['variant']:<10}  {r['global_k']:>2}  "
              f"{r['obj']:>12.0f}  {r['iters']:>5}  {r['cols_added']:>4}  "
              f"{r['rt_s']:>6.2f}s  {net_str}  {sla_str:>9}  {d_str:>8}")

    print(SEP)
    print(f"\n  [note] rt_net = wall-clock minus GNN forward-pass (apples-to-apples vs A0)")
    print(f"  [note] realized_demand is post-shock (apply_hidden_local_reallocation_demand_shocks)")
    print(f"  [note] convergence stopping: CG stops when no new negative-RC column found\n")

    out_dir  = Path("Result_small_scale_cg_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "small_scale_results.json"
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "ckpt": GNN_CKPT, "max_iter": MAX_ITER,
                   "stores": n_stores, "skus": n_skus, "periods": n_periods}, f, indent=2)
    print(f"[test] saved → {out_path}")


if __name__ == "__main__":
    main()
