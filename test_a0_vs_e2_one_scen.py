"""Single-scenario A0 vs E2 RMP-objective parity check (post-SLA-removal).

Runs A0 (exact_full_mode) and E2 (pruned_exact_mode + GNN top_frac) on the
smallest test scenario and prints both RMP objectives. After SLA penalty
removal, both variants should converge to the same LP optimum.
"""
from __future__ import annotations
import copy, gzip, os, pickle, time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCEN = REPO / "Test 30 scenarios" / "test_baselines" / "scenarios" / "base_01__gamma_store__seed239081664.pkl.gz"
CKPT = REPO / "GNN" / "trained_models" / "irplt_teacher_E2_filtered_local200_fixed" / "bigat" / "pairwise_rank" / "best_model.pt"

os.environ["IRP_GNN_SELECTION_MODE"] = "top_frac"
os.environ.setdefault("IRP_GNN_MAX_KEEP_FRAC", "0.30")
os.environ.setdefault("IRP_GNN_MIN_KEEP_FRAC", "0.10")
os.environ.setdefault("IRP_GNN_MIN_KEEP", "5")
os.environ["IRP_QUIET"] = "1"

import irp_gurobi_converted as irp


def run(variant: str, scen):
    data = copy.deepcopy(scen["shocked_data"])
    base = copy.deepcopy(scen["baseline_sol"])

    if variant == "A0":
        kwargs = dict(
            use_gnn=False, collect_teacher_mode=False,
            runtime_gnn_mode=False, heuristic_top_k_mode=False,
            exact_full_mode=True, use_branch_and_price=False,
        )
    elif variant == "E2":
        kwargs = dict(
            use_gnn=True, collect_teacher_mode=False,
            runtime_gnn_mode=True, heuristic_top_k_mode=False,
            pruned_exact_mode=True, gnn_selection_mode="top_frac",
            use_branch_and_price=False,
        )
    else:
        raise ValueError(variant)

    pipeline = irp.IRPResearchPipeline(data)
    t0 = time.perf_counter()
    results = pipeline.run_lt_recourse_from_baseline(
        base, shock_summary=None, use_random_initial_patterns=True,
        n_initial_patterns_per_product_period=5, cg_iterations=30,
        msg=False, gnn_checkpoint=str(CKPT), use_classical_fallback=False,
        gnn_mass_threshold=0.55, gnn_max_keep=150, gnn_max_keep_fraction=0.30,
        bp_max_nodes=15, bp_max_depth=6, lt_activation_threshold=10.0,
        diagnostic_verbosity="summary", **kwargs,
    )
    runtime = time.perf_counter() - t0

    cg = results.get("cg_solution")
    rmp_obj = float(cg.objective) if cg else float("nan")
    ep_hist = results.get("cg_episode_history") or []
    return {"variant": variant, "rmp_obj": rmp_obj, "runtime_s": runtime,
            "iters": len(ep_hist)}


def main():
    with gzip.open(SCEN, "rb") as f:
        scen = pickle.load(f)
    d = scen["shocked_data"]
    print(f"Scenario: stores={len(d.stores)} products={len(d.products)} periods={len(d.periods)}")
    print(f"Checkpoint exists: {CKPT.exists()}")

    rows = []
    for v in ("A0", "E2"):
        print(f"\n--- Running {v} ---")
        r = run(v, scen)
        rows.append(r)
        print(f"  {v}: rmp_obj={r['rmp_obj']:.6f} runtime={r['runtime_s']:.2f}s iters={r['iters']}")

    print("\n" + "=" * 70)
    a0, e2 = rows[0], rows[1]
    diff = e2["rmp_obj"] - a0["rmp_obj"]
    rel = abs(diff) / abs(a0["rmp_obj"]) if a0["rmp_obj"] else 0.0
    print(f"A0 RMP obj : {a0['rmp_obj']:.6f}")
    print(f"E2 RMP obj : {e2['rmp_obj']:.6f}")
    print(f"diff        : {diff:+.6f}  (relative {rel:.2e})")
    if rel < 1e-6:
        print("RESULT: RMP objectives MATCH (within numerical tolerance)")
    else:
        print("RESULT: RMP objectives DIFFER")


if __name__ == "__main__":
    main()
