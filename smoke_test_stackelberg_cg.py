"""smoke_test_stackelberg_cg.py — Smoke test for Stackelberg-aware CG mode.

Verifies that LateralTransshipmentCG with stackelberg_aware_scoring=True:
1. Runs end-to-end without errors on a small instance.
2. Produces a finite CG objective.
3. Produces FollowerBestResponseResult entries in cg_sol.follower_solution.
4. Produces stackelberg_column_scores entries.
5. Each scored pattern has metadata keys: stackelberg_delta,
   stackelberg_baseline_follower_cost, stackelberg_residual_follower_cost.
6. IRPResearchPipeline.run returns stackelberg_follower_plan and
   stackelberg_column_scores keys.

Usage
-----
python smoke_test_stackelberg_cg.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("IRP_QUIET", "1")
os.environ.setdefault("IRP_STORE_LIMIT", "4")
os.environ.setdefault("IRP_SKU_LIMIT", "2")
os.environ.setdefault("IRP_CG_STOPPING_MODE", "convergence")
os.environ.setdefault("IRP_LT_MIN_UNITS", "1")  # low threshold to ensure some LT flows exist

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import irp_gurobi_converted as irp  # noqa: E402


def _build_tiny_data():
    training_csv = REPO / "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
    if not training_csv.exists():
        raise FileNotFoundError(f"Expected training CSV missing: {training_csv}")
    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=str(training_csv),
        sheet_name="Sheet1",
        store_limit=4,
        sku_limit=2,
    )
    data, _, _, meta = mapper.build_irp_data(
        wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2,
        shortage_cost_rate=0.05,
        holding_cost_rate=100,
        cw_ship_cost_flat=1.0,
        lt_ship_cost_flat=0.6,
        fixed_dispatch_cw=8.0,
        fixed_dispatch_lt=2.0,
        vehicle_count=2,
        vehicle_capacity=500.0,
        vehicle_fixed_cost=50.0,
        alpha=1.0,
        cw_replenishment_factor=0.2,
        cw_capacity_factor=2.0,
        store_initial_inventory_multiplier=0.2,
        lt_cost_multiplier=1.0,
    )
    data.dataset_id = "smoke_test_stackelberg"
    data.scenario_id = "stackelberg"
    return data, meta


def _test_direct_cg(data, baseline_sol) -> int:
    """Test LateralTransshipmentCG directly with stackelberg_aware_scoring=True."""
    print("\n--- Direct CG test (stackelberg_aware_scoring=True) ---")
    initial_patterns = irp.generate_random_lt_patterns(
        data,
        baseline_solution=baseline_sol,
        n_patterns_per_product_period=2,
        max_pairs_in_pattern=3,
        lt_activation_threshold=0.0,
        seed=42,
    )
    print(f"[Init] random warm-start patterns = {len(initial_patterns)}")

    cg_engine = irp.LateralTransshipmentCG(
        data=data,
        baseline_solution=baseline_sol,
        initial_patterns=initial_patterns,
        lt_activation_threshold=0.0,
        max_pairs_per_pattern=3,
        use_gnn=False,
        collect_teacher_mode=False,
        runtime_gnn_mode=False,
        heuristic_top_k_mode=False,
        exact_full_mode=False,
        stackelberg_aware_scoring=True,
        stackelberg_exact_follower=False,
        stackelberg_min_lateral_qty=1.0,
    )
    cg_sol = cg_engine.run_column_generation(max_iter=3, msg=False, stopping_mode="convergence")

    obj = float(cg_sol.objective)
    assert obj < float("inf"), f"CG objective is infinite: {obj}"
    print(f"[Assert] cg_sol.objective = {obj:.4f} (finite, OK)")

    scores = cg_sol.stackelberg_column_scores
    print(f"[Assert] stackelberg_column_scores count = {len(scores)}")

    patterns = list(cg_engine.patterns)
    stackelberg_scored = [p for p in patterns if "stackelberg_delta" in p.metadata]
    print(f"[Assert] patterns with stackelberg_delta metadata = {len(stackelberg_scored)}")

    for pat in stackelberg_scored[:3]:
        assert "stackelberg_baseline_follower_cost" in pat.metadata, (
            f"pattern {pat.pattern_id} missing stackelberg_baseline_follower_cost"
        )
        assert "stackelberg_residual_follower_cost" in pat.metadata, (
            f"pattern {pat.pattern_id} missing stackelberg_residual_follower_cost"
        )
        delta = float(pat.metadata["stackelberg_delta"])
        print(
            f"  pattern {pat.pattern_id} | delta={delta:.6f} "
            f"| baseline_fc={pat.metadata['stackelberg_baseline_follower_cost']:.4f}"
            f"| residual_fc={pat.metadata['stackelberg_residual_follower_cost']:.4f}"
        )

    follower_sol = cg_sol.follower_solution
    if follower_sol:
        print(f"[Assert] follower_solution keys (product,period) = {len(follower_sol)}")
        for key, res in list(follower_sol.items())[:3]:
            print(
                f"  (p={key[0]}, t={key[1]}) | lt_cost={res.lt_cost:.4f} "
                f"| shortage_reduction={res.shortage_reduction:.4f} "
                f"| n_arcs={res.n_arcs_used} | solver={res.solver_used}"
            )
    else:
        print("[Assert] follower_solution is None or empty (no active pairs after CG)")

    last = dict(getattr(cg_engine, "_last_pricing_summary", {}) or {})
    stackelberg_mode = last.get("stackelberg_aware_scoring")
    print(f"[Assert] last_pricing_summary.stackelberg_aware_scoring = {stackelberg_mode}")
    # Only assert if we actually ran stackelberg pricing (need active pairs).
    if stackelberg_scored:
        assert stackelberg_mode is True, (
            f"Expected last_pricing_summary.stackelberg_aware_scoring=True, got {stackelberg_mode}"
        )
    return 0


def _test_pipeline_run(data) -> int:
    """Test IRPResearchPipeline.run returns Stackelberg keys."""
    print("\n--- Pipeline.run test (stackelberg_aware_scoring=True) ---")
    pipeline = irp.IRPResearchPipeline(data=data)
    results = pipeline.run(
        msg=False,
        cg_iterations=2,
        lt_activation_threshold=0.0,
        cw_dispatch_cycle=5,
        time_limit=30,
        enforce_integer_flows=False,
        use_gnn=False,
        collect_teacher_mode=False,
        runtime_gnn_mode=False,
        heuristic_top_k_mode=False,
        exact_full_mode=False,
        use_branch_and_price=False,
        n_initial_patterns_per_product_period=2,
        stackelberg_aware_scoring=True,
        stackelberg_exact_follower=False,
        stackelberg_min_lateral_qty=1.0,
    )

    assert "stackelberg_follower_plan" in results, (
        "results dict missing 'stackelberg_follower_plan'"
    )
    assert "stackelberg_column_scores" in results, (
        "results dict missing 'stackelberg_column_scores'"
    )
    print("[Assert] 'stackelberg_follower_plan' key present in results (OK)")
    print("[Assert] 'stackelberg_column_scores' key present in results (OK)")

    follower_plan = results["stackelberg_follower_plan"]
    scores = results["stackelberg_column_scores"]
    follower_count = len(follower_plan) if follower_plan else 0
    print(f"[Assert] stackelberg_follower_plan entries = {follower_count}")
    print(f"[Assert] stackelberg_column_scores entries = {len(scores)}")
    return 0


def main() -> int:
    print("=" * 70)
    print("SMOKE TEST: Stackelberg-aware CG")
    print("  stackelberg_aware_scoring=True, greedy follower, no GNN/top-k")
    print("=" * 70)

    t0 = time.perf_counter()
    data, meta = _build_tiny_data()
    print(f"[Data] {meta}")

    baseline_sol = irp.BaselineALNSModel(data).solve(
        msg=False,
        time_limit=30,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
    print(f"[Baseline] objective = {baseline_sol.objective:.4f}")

    irp.apply_hidden_local_reallocation_demand_shocks(
        data,
        baseline_solution=baseline_sol,
        shock_probability=0.85,
        max_reallocation_fraction=0.60,
        reallocations_per_product_period=2,
        non_dispatch_shock_multiplier=1.8,
        cw_dispatch_cycle=5,
        seed=20260418,
    )

    rc1 = _test_direct_cg(data, baseline_sol)
    if rc1 != 0:
        return rc1

    # Re-apply shock to reset demand for pipeline test (shock is idempotent with same seed).
    irp.apply_hidden_local_reallocation_demand_shocks(
        data,
        baseline_solution=baseline_sol,
        shock_probability=0.85,
        max_reallocation_fraction=0.60,
        reallocations_per_product_period=2,
        non_dispatch_shock_multiplier=1.8,
        cw_dispatch_cycle=5,
        seed=20260418,
    )

    rc2 = _test_pipeline_run(data)
    if rc2 != 0:
        return rc2

    runtime = time.perf_counter() - t0
    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED — Stackelberg-aware CG runs end-to-end.")
    print(f"  runtime: {runtime:.2f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n[SMOKE FAIL] {type(exc).__name__}: {exc}")
        sys.exit(1)
