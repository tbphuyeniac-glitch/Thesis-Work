"""
Test 1 case: A0 (pool=1, no penalty) vs E2 (pool=5, no penalty, GNN globally)

A0: exact_full_mode, pool_size=1, NO penalty
    → 1 col per (p,t) → adds N cols/iter
E2: exact_full_mode, pool_size=5, NO penalty, GNN ranks globally
    → 5N candidate cols/iter, GNN keeps top 1/5 = N cols globally
    → IRP_GNN_MIN_PER_GROUP=0 (no per-(p,t) quota)

Both: same N cols added per iter.
A0 picks 1 best per (p,t) by RC.
E2 picks N globally by GNN score (could be multiple from same p,t, 0 from others).

Test scenario: base_21 medium (7 stores × 3 SKUs × 153 periods)
"""
from __future__ import annotations
import os
import time
import gzip
import pickle
import json
from pathlib import Path
from copy import deepcopy

# NO SLA penalty for both
os.environ.pop("IRP_SLA_PENALTY", None)
os.environ["IRP_ADAPTIVE_PRUNING"] = "0"      # disable adaptive pruning
os.environ["IRP_GNN_MIN_PER_GROUP"] = "0"     # disable per-(p,t) quota → global
os.environ["IRP_QUIET"] = "1"

import irp_gurobi_converted as irp

REPO = Path(__file__).resolve().parent
SCEN_DIR = REPO / "Test 30 scenarios" / "test_baselines" / "scenarios"
CKPT = REPO / "GNN" / "trained_models" / "irplt_teacher_E2_filtered_local200_fixed" / "bigat" / "pairwise_rank" / "best_model.pt"

POOL_SIZE_E2 = 5
SCENARIO_NAME = "base_21__normal_global__seed1439190227"


def _load_scenario(name: str):
    path = SCEN_DIR / f"{name}.pkl.gz"
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _run_variant(variant_name: str, scen):
    data = deepcopy(scen["shocked_data"])
    base = deepcopy(scen["baseline_sol"])

    if variant_name == "A0_pool1":
        os.environ["IRP_EXACT_PRICING_POOL_SIZE"] = "1"
        kwargs = dict(
            use_gnn=False, collect_teacher_mode=False,
            runtime_gnn_mode=False, heuristic_top_k_mode=False,
            exact_full_mode=True,
            use_branch_and_price=False,
        )
        ckpt = None
    elif variant_name == "E2_pool5_GNN_global":
        os.environ["IRP_EXACT_PRICING_POOL_SIZE"] = str(POOL_SIZE_E2)
        kwargs = dict(
            use_gnn=True, collect_teacher_mode=False,
            runtime_gnn_mode=True, heuristic_top_k_mode=False,
            exact_full_mode=True,                       # full exact, but pool_size=5
            gnn_selection_mode="top_frac",
            gnn_max_keep_fraction=1.0 / POOL_SIZE_E2,   # keep top 1/5 globally
            gnn_max_keep=10000,                         # fraction wins
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

    cols_per_iter = []
    if len(ep_hist) > 1:
        for i in range(1, len(ep_hist)):
            added = int(ep_hist[i].get("added_columns", 0)) if isinstance(ep_hist[i], dict) else 0
            cols_per_iter.append(added)
    total_cols = sum(cols_per_iter)
    avg_cols = total_cols / max(len(cols_per_iter), 1)

    realized = results.get("realized_with_lt_cost_breakdown") or {}
    realized_cost = float(realized.get("total_realized_operating_cost", float("nan")))

    return {
        "variant": variant_name,
        "rmp_objective": rmp_obj,
        "realized_cost": realized_cost,
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
    print("TEST: A0 (pool=1, NO penalty) vs E2 (pool=5, NO penalty, GNN global)")
    print("=" * 80)
    print(f"Scenario: {SCENARIO_NAME}")
    print(f"Configuration:")
    print(f"  IRP_SLA_PENALTY:        OFF")
    print(f"  IRP_ADAPTIVE_PRUNING:   0")
    print(f"  IRP_GNN_MIN_PER_GROUP:  0  (no per-(p,t) quota)")
    print(f"  E2 GNN keep fraction:   {1.0/POOL_SIZE_E2:.2f} (top 1/{POOL_SIZE_E2} globally)")
    print()

    scen = _load_scenario(SCENARIO_NAME)
    d = scen["shocked_data"]
    print(f"Data: {len(d.stores)} stores × {len(d.products)} products × {len(d.periods)} periods")
    print()

    results = []
    for variant in ("A0_pool1", "E2_pool5_GNN_global"):
        print("\n" + "#" * 80)
        print(f"# {variant}")
        print("#" * 80)
        r = _run_variant(variant, scen)
        results.append(r)
        print(f"  obj={r['rmp_objective']:.2f}")
        print(f"  realized_cost={r['realized_cost']:.2f}")
        print(f"  iters={r['cg_iterations']}  runtime={r['runtime_seconds']:.2f}s")
        print(f"  total_cols={r['total_cols_added']}  avg_cols/iter={r['avg_cols_per_iter']:.1f}")
        print(f"  cols_per_iter: {r['cols_per_iter']}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    a0, e2 = results[0], results[1]
    print(f"{'Metric':<25} {'A0_pool1':>22} {'E2_pool5_GNN':>22}")
    print("-" * 80)
    print(f"{'RMP objective':<25} {a0['rmp_objective']:>22.2f} {e2['rmp_objective']:>22.2f}")
    print(f"{'Realized cost':<25} {a0['realized_cost']:>22.2f} {e2['realized_cost']:>22.2f}")
    print(f"{'CG iterations':<25} {a0['cg_iterations']:>22d} {e2['cg_iterations']:>22d}")
    print(f"{'Runtime (s)':<25} {a0['runtime_seconds']:>22.2f} {e2['runtime_seconds']:>22.2f}")
    print(f"{'Total cols added':<25} {a0['total_cols_added']:>22d} {e2['total_cols_added']:>22d}")
    print(f"{'Avg cols/iter':<25} {a0['avg_cols_per_iter']:>22.1f} {e2['avg_cols_per_iter']:>22.1f}")

    # Deltas
    d_obj = e2['rmp_objective'] - a0['rmp_objective']
    d_obj_pct = 100 * d_obj / max(abs(a0['rmp_objective']), 1e-9)
    d_real = e2['realized_cost'] - a0['realized_cost']
    d_real_pct = 100 * d_real / max(abs(a0['realized_cost']), 1e-9)
    rt_ratio = e2['runtime_seconds'] / max(a0['runtime_seconds'], 1e-6)

    print()
    print("DELTA (E2 - A0):")
    print(f"  RMP obj:       {d_obj:+.2f} ({d_obj_pct:+.4f}%)")
    print(f"  Realized cost: {d_real:+.2f} ({d_real_pct:+.4f}%)")
    print(f"  Runtime ratio: {rt_ratio:.2f}× (E2/A0)")

    if abs(d_obj_pct) < 0.001:
        print("  → RMP equivalent")
    elif d_obj_pct < 0:
        print("  → E2 RMP BETTER")
    else:
        print("  → A0 RMP BETTER")

    out_path = REPO / "Result_pool_size_compare" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
