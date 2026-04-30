"""E1 ablation sweep: vary instance size × demand-shock intensity.

Usage:
    IRP_GNN_CHECKPOINT=GNN/trained_models/irplt_teacher_filtered_local3/bigat/pairwise_rank/best_model.pt \
    python3 local_e1_sweep.py
"""
from __future__ import annotations
import os, time, json
from pathlib import Path
from copy import deepcopy

os.environ.setdefault("IRP_TIME_LIMIT", "120")
os.environ.setdefault("IRP_CG_ITERATIONS", "15")
os.environ.setdefault("IRP_BP_MAX_NODES", "5")
os.environ.setdefault("IRP_BP_MAX_DEPTH", "3")
os.environ.setdefault("IRP_QUIET", "1")
os.environ.setdefault("IRP_CG_STOPPING_MODE", "convergence")
os.environ.setdefault("IRP_RESULTS_DIR_OVERRIDE", "Results_local_e1_sweep")
os.environ.pop("IRP_SLA_PENALTY", None)

import irp_gurobi_converted as irp


VARIANTS = [
    ("A0", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "exact_full_mode": True,
    }),
    ("E1_rc_only", {
        "use_gnn": False, "collect_teacher_mode": False,
        "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
        "rc_filter_mode": True,
    }),
    ("E1_rc_gnn", {
        "use_gnn": True, "collect_teacher_mode": False,
        "runtime_gnn_mode": True, "heuristic_top_k_mode": False,
        "rc_filter_mode": True,
        "gnn_selection_mode": "relative_threshold",
        "gnn_relative_threshold": 0.70,
        "gnn_max_keep_fraction": 0.30,
    }),
]

# Instance sizes (stores, skus)
INSTANCES = [
    (10, 5),   # baseline reference (already covered)
    (12, 6),
    (15, 7),
]

# Shock intensities — only medium for extended sweep (most representative)
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
    cg = irp.LateralTransshipmentCG(
        data=data_copy, baseline_solution=baseline_copy,
        lt_activation_threshold=10.0,
        gnn_checkpoint=GNN_CKPT if kwargs.get("use_gnn") else None,
        **kwargs,
    )
    t0 = time.perf_counter()
    sol = cg.run_column_generation(max_iter=15, stopping_mode="convergence")
    rt = time.perf_counter() - t0
    obj = float(sol.objective)
    cg_history = getattr(cg, "cg_history", [])
    iters = max(0, len(cg_history) - 1)
    cols = sum(int(h.get("added_columns", 0)) for h in cg_history)
    rt_per_iter = rt / max(1, iters)
    return {"obj": obj, "rt": rt, "iters": iters, "cols": cols, "rt_per_iter": rt_per_iter}


def main():
    if not Path(GNN_CKPT).exists():
        raise SystemExit(f"GNN checkpoint not found: {GNN_CKPT}")
    print(f"[sweep] checkpoint: {GNN_CKPT}\n")

    rows = []
    for stores, skus in INSTANCES:
        print("=" * 80)
        print(f"[sweep] BUILD instance {stores} stores × {skus} skus")
        print("=" * 80)
        t0 = time.perf_counter()
        data = build_data(stores, skus)
        print(f"[sweep] data: {len(data.stores)} × {len(data.products)} × {len(data.periods)} periods  ({time.perf_counter()-t0:.1f}s build)")

        t0 = time.perf_counter()
        baseline = solve_baseline(data)
        base_t = time.perf_counter() - t0
        base_obj = float(baseline.objective)
        print(f"[sweep] baseline obj={base_obj:.2f}  ({base_t:.1f}s)\n")

        for shock_name, sp, sf, sm in SHOCKS:
            print(f"--- shock={shock_name}  prob={sp}  frac={sf}  mult={sm} ---")
            for v_name, kwargs in VARIANTS:
                r = run_variant(v_name, kwargs, data, baseline, sp, sf, sm, SEED)
                row = {
                    "instance": f"{stores}x{skus}",
                    "stores": stores, "skus": skus,
                    "shock": shock_name,
                    "variant": v_name,
                    "obj": r["obj"], "rt_s": r["rt"],
                    "iters": r["iters"], "cols": r["cols"],
                    "rt_per_iter": r["rt_per_iter"],
                    "baseline_obj": base_obj,
                    "delta_vs_baseline": r["obj"] - base_obj,
                }
                rows.append(row)
                print(f"  {v_name:<12} obj={r['obj']:.2f}  rt={r['rt']:.2f}s  iters={r['iters']}  cols={r['cols']}  rt/it={r['rt_per_iter']:.3f}s")
            print()

    # ================== SUMMARY TABLE ==================
    print("\n" + "#" * 100)
    print("# SWEEP SUMMARY — runtime (s) and CG iterations per cell")
    print("#" * 100)
    print(f"{'instance':<10} {'shock':<8} {'A0_rt':>8} {'rcOnly_rt':>10} {'rcGnn_rt':>10}  "
          f"{'A0_rt/it':>10} {'rc_rt/it':>10} {'gnn_rt/it':>10}  "
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
        print(f"{inst:<10} {shock:<8} {a0['rt_s']:>8.2f} {ro['rt_s']:>10.2f} {rg['rt_s']:>10.2f}  "
              f"{a0['rt_per_iter']:>10.3f} {ro['rt_per_iter']:>10.3f} {rg['rt_per_iter']:>10.3f}  "
              f"{a0['iters']:>6} {ro['iters']:>10} {rg['iters']:>9}  "
              f"{rc_speedup:>10.2f}x {gnn_speedup:>11.2f}x")

    # Objective deltas
    print("\n" + "#" * 100)
    print("# OBJECTIVE QUALITY — Δ vs A0 (lower is better; absolute units = cost)")
    print("#" * 100)
    print(f"{'instance':<10} {'shock':<8} {'A0_obj':>20} {'rcOnly_Δ':>12} {'rcGnn_Δ':>12}  "
          f"{'rcOnly_pct':>11} {'rcGnn_pct':>10}")
    for (inst, shock), per_v in grouped.items():
        a0 = per_v["A0"]
        ro = per_v["E1_rc_only"]
        rg = per_v["E1_rc_gnn"]
        d_ro = ro["obj"] - a0["obj"]
        d_rg = rg["obj"] - a0["obj"]
        pct_ro = 100.0 * d_ro / max(1e-9, a0["obj"])
        pct_rg = 100.0 * d_rg / max(1e-9, a0["obj"])
        print(f"{inst:<10} {shock:<8} {a0['obj']:>20.2f} {d_ro:>+12.2f} {d_rg:>+12.2f}  "
              f"{pct_ro:>+10.6f}% {pct_rg:>+9.6f}%")

    # Identify GNN wins
    print("\n" + "#" * 100)
    print("# GNN-WINS-ANALYSIS: cells where E1_rc_gnn beats A0 on runtime AND matches obj")
    print("#" * 100)
    wins = []
    for (inst, shock), per_v in grouped.items():
        a0 = per_v["A0"]
        rg = per_v["E1_rc_gnn"]
        ro = per_v["E1_rc_only"]
        gnn_faster_than_a0 = rg["rt_s"] < a0["rt_s"]
        gnn_obj_close = abs(rg["obj"] - a0["obj"]) / max(1e-9, a0["obj"]) < 1e-5
        gnn_faster_than_rc_only = rg["rt_s"] < ro["rt_s"]
        flags = []
        if gnn_faster_than_a0: flags.append("GNN<A0")
        if gnn_obj_close: flags.append("obj=A0")
        if gnn_faster_than_rc_only: flags.append("GNN<rc_only")
        if gnn_faster_than_a0 and gnn_obj_close:
            wins.append((inst, shock, flags))
        print(f"  {inst:<10} {shock:<8}  flags={flags}")
    if wins:
        print(f"\n[wins] GNN beats A0 on {len(wins)} cells: {wins}")
    else:
        print("\n[wins] GNN does NOT beat A0 in this sweep (likely needs more training epochs).")

    out_path = Path("Results_local_e1_sweep") / "e1_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "ckpt": GNN_CKPT}, f, indent=2)
    print(f"\n[sweep] saved {out_path}")


if __name__ == "__main__":
    main()
