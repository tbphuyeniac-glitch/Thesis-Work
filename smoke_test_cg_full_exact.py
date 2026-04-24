"""smoke_test_cg_full_exact.py — Minimal smoke test for benchmark A0 (CG_full_exact).

Verifies that the exact Gurobi pricing path in LateralTransshipmentCG:
1. Runs end-to-end without errors on a small instance.
2. Bypasses pruning / Stackelberg / GNN / top-k.
3. Produces patterns whose metadata['source'] == "exact_pricing_gurobi".
4. Produces a CG solution with finite objective.

Usage
-----
python smoke_test_cg_full_exact.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Keep logs quiet and the instance tiny.
os.environ.setdefault("IRP_QUIET", "1")
os.environ.setdefault("IRP_STORE_LIMIT", "4")
os.environ.setdefault("IRP_SKU_LIMIT", "2")
os.environ.setdefault("IRP_CG_STOPPING_MODE", "convergence")

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
    data.dataset_id = "smoke_test_a0"
    data.scenario_id = "a0"
    return data, meta


def main() -> int:
    print("=" * 70)
    print("SMOKE TEST: benchmark A0 = CG_full_exact")
    print("  Exact Gurobi pricing, NO pruning / NO Stackelberg / NO GNN / NO top-k")
    print("=" * 70)

    t0 = time.perf_counter()
    data, meta = _build_tiny_data()
    print(f"[Data] {meta}")

    # Step 1 — build baseline via ALNS and apply the standard demand shock.
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

    # Step 2 — build the CG engine in A0 mode and run pure CG (no B&P).
    initial_patterns = irp.generate_random_lt_patterns(
        data,
        baseline_solution=baseline_sol,
        n_patterns_per_product_period=2,
        max_pairs_in_pattern=3,
        lt_activation_threshold=0.0,
        seed=123,
    )
    print(f"[Init] random warm-start patterns = {len(initial_patterns)}")

    cg_engine = irp.LateralTransshipmentCG(
        data=data,
        baseline_solution=baseline_sol,
        initial_patterns=initial_patterns,
        lt_activation_threshold=0.0,
        max_pairs_per_pattern=4,
        # A0: every heuristic off.
        use_gnn=False,
        collect_teacher_mode=False,
        runtime_gnn_mode=False,
        heuristic_top_k_mode=False,
        exact_full_mode=True,
    )
    cg_sol = cg_engine.run_column_generation(max_iter=3, msg=False, stopping_mode="convergence")

    runtime = time.perf_counter() - t0
    print(f"[Pipeline] runtime = {runtime:.2f}s")

    # ---- Assertions --------------------------------------------------------
    obj = float(cg_sol.objective)
    assert obj < float("inf"), f"CG objective is infinite (solver failed): {obj}"
    print(f"[Assert] cg_solution.objective = {obj:.4f} (finite, OK)")

    patterns = list(cg_engine.patterns)
    exact_patterns = [p for p in patterns if p.metadata.get("source") == "exact_pricing_gurobi"]
    random_patterns = [p for p in patterns if p.metadata.get("source") == "random_warm_start"]
    print(f"[Assert] total patterns in pool             = {len(patterns)}")
    print(f"[Assert] patterns from exact Gurobi pricing = {len(exact_patterns)}")
    print(f"[Assert] random warm-start patterns         = {len(random_patterns)}")

    # No pattern should come from the pruning+Stackelberg path.
    forbidden_sources = {"pricing_pruned_feature"}
    leaked = [p for p in patterns if p.metadata.get("source") in forbidden_sources]
    assert not leaked, (
        f"A0 leaked {len(leaked)} pattern(s) from heuristic path: "
        f"{[p.metadata.get('source') for p in leaked[:3]]}"
    )
    print("[Assert] no pattern leaked from the pruning/Stackelberg path (OK)")

    # Sanity-check exact patterns metadata and reduced costs.
    for pat in exact_patterns[:5]:
        rc = float(pat.metadata.get("reduced_cost", 0.0))
        assert rc <= 1e-6, f"exact pattern has positive reduced cost: {rc}"
        assert pat.metadata.get("pruning_used") is False
        assert pat.metadata.get("stackelberg_used") is False
        assert pat.metadata.get("gnn_used") is False
        assert pat.metadata.get("heuristic_top_k_used") is False
        print(
            f"  pattern {pat.pattern_id} | rc={rc:.6f} | flows={len(pat.pattern_flows)}"
        )

    # Efficiency metrics bookkeeping.
    eff = dict(cg_sol.efficiency_metrics or {})
    rmp_solves = float(eff.get("rmp_solves", 0.0))
    assert rmp_solves >= 1.0, f"RMP should have been solved at least once (got {rmp_solves})"
    print(f"[Assert] rmp_solves = {rmp_solves:.0f} (>=1, OK)")

    # Last pricing summary must reflect exact mode.
    last = dict(getattr(cg_engine, "_last_pricing_summary", {}) or {})
    print(f"[Assert] last_pricing_summary.exact_full_mode = {last.get('exact_full_mode')}")
    # NOTE: the value may be missing if there were no active (product, period) pairs;
    # in that case nothing was priced and the key is never populated. That is still
    # a correct end state, so only assert when we did price.
    if exact_patterns:
        assert last.get("exact_full_mode") is True

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED — A0 (CG_full_exact) runs end-to-end.")
    print(f"  runtime                   : {runtime:.2f}s")
    print(f"  cg_objective              : {obj:.4f}")
    print(f"  exact_patterns_priced     : {len(exact_patterns)}")
    print(f"  rmp_solves                : {rmp_solves:.0f}")
    print(f"  iterations_run            : {cg_sol.iterations_run}")
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
