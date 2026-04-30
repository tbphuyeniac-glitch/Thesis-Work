"""E1 sweep matching PRODUCTION benchmark setup.

Critical differences vs local_e1_sweep.py:
  - Goes through IRPResearchPipeline.run_lt_recourse_from_baseline (not LateralTransshipmentCG directly)
  - A0 uses use_branch_and_price=True (default in production)
  - E1_rc_only / E1_rc_gnn use use_branch_and_price=False
  - Generates 5 initial patterns per (product, period) like production

Usage:
    IRP_GNN_CHECKPOINT=GNN/trained_models/irplt_teacher_filtered_local3/bigat/pairwise_rank/best_model.pt \
    python3 local_e1_sweep_realistic.py
"""
from __future__ import annotations
import os, time, json
from pathlib import Path
from copy import deepcopy

os.environ.setdefault("IRP_TIME_LIMIT", "240")
os.environ.setdefault("IRP_CG_ITERATIONS", "15")
os.environ.setdefault("IRP_BP_MAX_NODES", "15")  # production default
os.environ.setdefault("IRP_BP_MAX_DEPTH", "6")   # production default
os.environ.setdefault("IRP_QUIET", "1")
os.environ.setdefault("IRP_CG_STOPPING_MODE", "convergence")
os.environ.setdefault("IRP_RESULTS_DIR_OVERRIDE", "Results_local_e1_sweep_realistic")
os.environ.pop("IRP_SLA_PENALTY", None)

import irp_gurobi_converted as irp


VARIANTS = [
    ("A0", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "exact_full_mode": True,
        "use_branch_and_price": True,   # ← production default for A0
    }),
    ("E1_rc_only", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "rc_filter_mode": True,
        "use_branch_and_price": False,  # ← production setup
    }),
    ("E1_rc_gnn", {
        "use_gnn": True, "collect_teacher_mode": False,
        "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
        "rc_filter_mode": True,
        "gnn_selection_mode": "relative_threshold",
        "gnn_relative_threshold": 0.70,
        "gnn_max_keep_fraction": 0.30,
        "use_branch_and_price": False,  # ← production setup
    }),
]

INSTANCES = [
    (5, 3),
    (10, 5),
]

SHOCKS = [
    ("medium", 0.85, 0.6, 1.8),
]

EXCEL_PATH = Path(__file__).resolve().parent / "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
GNN_CKPT = os.environ.get(
    "IRP_GNN_CHECKPOINT",
    "GNN/trained_models/irplt_teacher_filtered_local3/bigat/pairwise_rank/best_model.pt",
)
SEED = 20260418


def build_data(stores, skus):
    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=str(EXCEL_PATH),
        sheet_name="Sheet1",
        store_limit=stores,
        sku_limit=skus,
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
        msg=False, time_limit=120,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )


def run_variant(name, kwargs, data, baseline_sol, shock_p, shock_f, shock_m, seed):
    data_copy = deepcopy(data)
    irp.apply_hidden_local_reallocation_demand_shocks(
        data_copy,
        shock_probability=shock_p,
        max_reallocation_fraction=shock_f,
        reallocations_per_product_period=3,
        non_dispatch_shock_multiplier=shock_m,
        seed=seed,
    )
    baseline_copy = deepcopy(baseline_sol)
    pipeline = irp.IRPResearchPipeline(data_copy)
    use_bp = kwargs.pop("use_branch_and_price", False)
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
    cols_gen = sum(int(h.get("proposed_columns", 0)) for h in cg_history)
    bp_nodes = len(result.get("branch_price_history", []) or [])
    return {
        "obj": obj, "rt": rt, "iters": cg_iters,
        "cols_gen": cols_gen, "cols_added": cols_added,
        "bp_nodes": bp_nodes,
        "use_bp": use_bp,
    }


def main():
    if not Path(GNN_CKPT).exists():
        raise SystemExit(f"GNN checkpoint not found: {GNN_CKPT}")
    print(f"[realistic] checkpoint: {GNN_CKPT}\n")

    rows = []
    for stores, skus in INSTANCES:
        print("=" * 80)
        print(f"[realistic] BUILD instance {stores} stores × {skus} skus")
        print("=" * 80)
        data = build_data(stores, skus)
        print(f"[realistic] data: {len(data.stores)} × {len(data.products)} × {len(data.periods)} periods")

        t0 = time.perf_counter()
        baseline = solve_baseline(data)
        base_t = time.perf_counter() - t0
        print(f"[realistic] baseline obj={float(baseline.objective):.2f}  ({base_t:.1f}s)\n")

        for shock_name, sp, sf, sm in SHOCKS:
            print(f"--- shock={shock_name} ---")
            for v_name, kwargs_orig in VARIANTS:
                kwargs = dict(kwargs_orig)  # don't mutate original
                r = run_variant(v_name, kwargs, data, baseline, sp, sf, sm, SEED)
                row = {
                    "instance": f"{stores}x{skus}",
                    "stores": stores, "skus": skus,
                    "shock": shock_name,
                    "variant": v_name,
                    "use_bp": r["use_bp"],
                    "obj": r["obj"], "rt_s": r["rt"],
                    "iters": r["iters"],
                    "bp_nodes": r["bp_nodes"],
                    "cols_gen": r["cols_gen"], "cols_added": r["cols_added"],
                }
                rows.append(row)
                bp_label = "BP" if r["use_bp"] else "noBP"
                print(f"  {v_name:<12} [{bp_label}] obj={r['obj']:.2f}  rt={r['rt']:.1f}s  "
                      f"iters={r['iters']}  bp_nodes={r['bp_nodes']}  cols_added={r['cols_added']}")
            print()

    # Summary
    print("\n" + "#" * 100)
    print("# REALISTIC SWEEP SUMMARY (production-style: A0=BP, E1=noBP)")
    print("#" * 100)
    print(f"{'instance':<10} {'shock':<8} {'A0_rt(BP)':>12} {'rcOnly_rt':>12} {'rcGnn_rt':>12}  "
          f"{'A0_it':>6} {'rcOnly_it':>10} {'rcGnn_it':>9}  {'rc_speedup':>11} {'gnn_speedup':>12}")
    grouped = {}
    for r in rows:
        key = (r["instance"], r["shock"])
        grouped.setdefault(key, {})[r["variant"]] = r
    for (inst, shock), per_v in grouped.items():
        a0 = per_v["A0"]
        ro = per_v["E1_rc_only"]
        rg = per_v["E1_rc_gnn"]
        rc_speedup  = a0["rt_s"] / max(1e-6, ro["rt_s"])
        gnn_speedup = a0["rt_s"] / max(1e-6, rg["rt_s"])
        print(f"{inst:<10} {shock:<8} {a0['rt_s']:>12.1f} {ro['rt_s']:>12.1f} {rg['rt_s']:>12.1f}  "
              f"{a0['iters']:>6} {ro['iters']:>10} {rg['iters']:>9}  "
              f"{rc_speedup:>10.2f}x {gnn_speedup:>11.2f}x")

    out_path = Path("Results_local_e1_sweep_realistic") / "e1_realistic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "ckpt": GNN_CKPT}, f, indent=2)
    print(f"\n[realistic] saved {out_path}")


if __name__ == "__main__":
    main()
