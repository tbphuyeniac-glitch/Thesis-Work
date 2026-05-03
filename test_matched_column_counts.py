"""
Test: A0 vs E2 with MATCHED column counts per iteration.

Both add 1 column per (p,t) per iteration:
  A0 (with penalty): exact MIP, picks most-negative-RC column per (p,t)
  E2: pruned exact + GNN ranking, but keeps ALL priced columns (no GNN filtering)

The only difference: GNN inference overhead vs no GNN.
RMP grows at the same rate for both → fair runtime comparison.

Test scenarios (small + medium):
  base_01: 4 stores × 2 SKUs (small)
  base_04: 4 stores × 3 SKUs (small)
  base_21: 7 stores × 3 SKUs (medium)
  base_24: 7 stores × 4 SKUs (medium)
"""
from __future__ import annotations
import os
import time
import gzip
import pickle
import json
from pathlib import Path
from copy import deepcopy

# A0 with penalty, no adaptive pruning (focus on column-count comparison)
os.environ["IRP_SLA_PENALTY"] = "on"
os.environ["IRP_SLA_MU"] = "0.05"
os.environ["IRP_SLA_NU"] = "0.05"
os.environ["IRP_ADAPTIVE_PRUNING"] = "0"  # disable to isolate GNN cost
os.environ["IRP_QUIET"] = "1"

import irp_gurobi_converted as irp

REPO = Path(__file__).resolve().parent
SCEN_DIR = REPO / "Test 30 scenarios" / "test_baselines" / "scenarios"
CKPT = REPO / "GNN" / "trained_models" / "irplt_teacher_E2_filtered_local200_fixed" / "bigat" / "pairwise_rank" / "best_model.pt"

# 2 small + 2 medium scenarios
SCENARIOS = [
    ("small_4s2sku",   "base_01__normal_global__seed1373158607"),
    ("small_4s3sku",   "base_04__normal_global__seed1268073013"),
    ("medium_7s3sku",  "base_21__normal_global__seed1439190227"),
    ("medium_7s4sku",  "base_24__normal_global__seed525727462"),
]


def _load_scenario(name: str):
    path = SCEN_DIR / f"{name}.pkl.gz"
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _run_variant(variant_name: str, scen):
    data = deepcopy(scen["shocked_data"])
    base = deepcopy(scen["baseline_sol"])

    if variant_name == "A0_penalty_k1":
        kwargs = dict(
            use_gnn=False, collect_teacher_mode=False,
            runtime_gnn_mode=False, heuristic_top_k_mode=False,
            exact_full_mode=True,            # A0: exact MIP, 1 col per (p,t)
            use_branch_and_price=False,
        )
        ckpt = None
    elif variant_name == "E2_GNN_keep_all":
        kwargs = dict(
            use_gnn=True, collect_teacher_mode=False,
            runtime_gnn_mode=True, heuristic_top_k_mode=False,
            pruned_exact_mode=True,          # E2: pruned exact + GNN
            gnn_selection_mode="top_frac",
            gnn_max_keep_fraction=1.0,       # KEEP ALL — no GNN filtering
            gnn_max_keep=10000,              # large cap
            use_branch_and_price=False,
        )
        ckpt = str(CKPT)
    else:
        raise ValueError(variant_name)

    pipeline = irp.IRPResearchPipeline(data)
    t0 = time.perf_counter()
    results = pipeline.run_lt_recourse_from_baseline(
        base,
        shock_summary=None,
        use_random_initial_patterns=True,
        n_initial_patterns_per_product_period=5,
        cg_iterations=30,
        msg=False,
        gnn_checkpoint=ckpt,
        use_classical_fallback=False,
        gnn_mass_threshold=0.55,
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

    # Count columns added per iteration
    # ep_hist[0] = initial RMP, ep_hist[i] for i>=1 = iteration i
    cols_per_iter = []
    if len(ep_hist) > 1:
        for i in range(1, len(ep_hist)):
            added = int(ep_hist[i].get("added_columns", 0)) if isinstance(ep_hist[i], dict) else 0
            cols_per_iter.append(added)
    total_cols = sum(cols_per_iter)
    avg_cols = total_cols / max(len(cols_per_iter), 1)

    return {
        "variant": variant_name,
        "rmp_objective": rmp_obj,
        "runtime_seconds": runtime,
        "cg_iterations": cg_iters,
        "cols_per_iter": cols_per_iter,
        "total_cols_added": total_cols,
        "avg_cols_per_iter": avg_cols,
    }


def main():
    if not CKPT.exists():
        print(f"ERROR: GNN checkpoint not found at {CKPT}")
        return

    print("=" * 80)
    print("MATCHED COLUMN COUNTS: A0 (k=1 per p,t) vs E2 (GNN ranks, keep all)")
    print("=" * 80)
    print(f"Both variants:")
    print(f"  - SLA penalty: ON (μ=ν=0.05)")
    print(f"  - Adaptive pruning: OFF (IRP_ADAPTIVE_PRUNING=0)")
    print(f"  - 1 column per (p,t) per iteration (no GNN filtering for E2)")
    print()

    all_results = []
    for label, scen_name in SCENARIOS:
        print("\n" + "#" * 80)
        print(f"# {label}  ({scen_name})")
        print("#" * 80)
        scen = _load_scenario(scen_name)
        d = scen["shocked_data"]
        print(f"  stores={len(d.stores)} products={len(d.products)} periods={len(d.periods)}")
        n_pt = len(d.products) * len(d.periods)
        print(f"  Total (p,t) combinations = {n_pt}")

        for variant in ("A0_penalty_k1", "E2_GNN_keep_all"):
            print(f"\n[RUN] {label} / {variant}")
            r = _run_variant(variant, scen)
            r["scenario_label"] = label
            r["scenario_name"] = scen_name
            r["stores"] = len(d.stores)
            r["products"] = len(d.products)
            r["periods"] = len(d.periods)
            all_results.append(r)
            print(f"  obj={r['rmp_objective']:.2f}")
            print(f"  iters={r['cg_iterations']}  runtime={r['runtime_seconds']:.2f}s")
            print(f"  total_cols={r['total_cols_added']}  avg_cols/iter={r['avg_cols_per_iter']:.1f}")

    # Summary
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Scenario':<18} {'Variant':<20} {'Iters':>5} {'Runtime':>10} {'TotCols':>8} {'AvgCols':>8} {'RMP Obj':>22}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['scenario_label']:<18} {r['variant']:<20} {r['cg_iterations']:>5} "
              f"{r['runtime_seconds']:>9.2f}s {r['total_cols_added']:>8} {r['avg_cols_per_iter']:>8.1f} "
              f"{r['rmp_objective']:>22.2f}")

    # Per-scenario analysis
    print("\n" + "=" * 100)
    print("PER-SCENARIO ANALYSIS")
    print("=" * 100)
    for label, _ in SCENARIOS:
        a0_rows = [r for r in all_results if r["scenario_label"] == label and r["variant"] == "A0_penalty_k1"]
        e2_rows = [r for r in all_results if r["scenario_label"] == label and r["variant"] == "E2_GNN_keep_all"]
        if not (a0_rows and e2_rows):
            continue
        a0, e2 = a0_rows[0], e2_rows[0]
        d_obj = e2["rmp_objective"] - a0["rmp_objective"]
        d_rt_ratio = e2["runtime_seconds"] / max(a0["runtime_seconds"], 1e-6)
        d_iters = e2["cg_iterations"] - a0["cg_iterations"]
        d_cols = e2["total_cols_added"] - a0["total_cols_added"]
        print(f"\n[{label}]  stores={a0['stores']}, products={a0['products']}, periods={a0['periods']}")
        print(f"  RMP delta:        {d_obj:+.2f} ({100*d_obj/max(abs(a0['rmp_objective']),1e-9):+.4f}%)")
        print(f"  Runtime ratio:    {d_rt_ratio:.2f}× (E2/A0)")
        print(f"  Iterations delta: {d_iters:+d}")
        print(f"  Total cols delta: {d_cols:+d}")
        print(f"  Per-iter cols:")
        print(f"    A0: {a0['cols_per_iter']}")
        print(f"    E2: {e2['cols_per_iter']}")

    # Save results
    out_path = REPO / "Result_matched_column_counts" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
