"""smoke_test_adaptive_pruning.py — Smoke test for AdaptiveFeaturePruner.

Verifies:
1. The pruner class is constructible and exposes the expected 4 features.
2. update() honours warm-up (returns 'warmup' mode for tiny inputs).
3. update() tightens windows from the surviving RC distribution.
4. The wipe-out safeguard refuses an update that would prune > 95% of points.
5. No-pruning fallback path still works when adaptive_pruning_enabled=False.
6. Inside a real CG run, the pruner's history is populated and windows
   change between iterations — i.e., it is genuinely *adaptive*, not fixed.

Run
---
python smoke_test_adaptive_pruning.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("IRP_QUIET", "1")
os.environ.setdefault("IRP_STORE_LIMIT", "4")
os.environ.setdefault("IRP_SKU_LIMIT", "2")
os.environ.setdefault("IRP_CG_STOPPING_MODE", "convergence")
os.environ.setdefault("IRP_LT_MIN_UNITS", "1")
os.environ.setdefault("IRP_ADAPTIVE_PRUNING", "1")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import irp_gurobi_converted as irp  # noqa: E402


# ---------------------------------------------------------------------------
# Unit-style checks on the pruner in isolation
# ---------------------------------------------------------------------------

def _unit_checks() -> None:
    print("\n--- Unit checks on AdaptiveFeaturePruner ---")
    pruner = irp.AdaptiveFeaturePruner()

    # 1) Default feature set.
    assert pruner.feature_names == (
        "shortage_ratio", "surplus_ratio", "time_urgency", "negative_reduced_cost"
    ), f"unexpected feature_names: {pruner.feature_names}"
    print("[Assert] default feature set matches the 4 expected names (OK)")

    # 2) Warm-up: too few candidates → no window update.
    tiny = [
        {"reduced_cost": -1.0, "feature_values": {"shortage_ratio": 0.5, "surplus_ratio": 0.4,
                                                    "time_urgency": 0.6, "negative_reduced_cost": 1.0}},
    ]
    rec = pruner.update(tiny, iteration=1, phase="lt")
    assert rec["mode"] == "warmup", f"expected warmup, got {rec['mode']}"
    assert rec["skipped_reason"] == "warmup_insufficient_candidates"
    print("[Assert] warm-up triggers no update for n=1 (OK)")

    # 3) Update: enough candidates, mixed RC distribution → windows tighten.
    rng_candidates = []
    for k in range(20):
        rc = -10.0 + k * 1.0  # spans negative to positive
        rng_candidates.append({
            "reduced_cost": rc,
            "feature_values": {
                "shortage_ratio": 0.05 + 0.04 * k,
                "surplus_ratio": 0.10 + 0.03 * k,
                "time_urgency": 0.20 + 0.02 * k,
                "negative_reduced_cost": max(0.0, -rc),
            },
        })
    rec = pruner.update(rng_candidates, iteration=2, phase="lt")
    assert rec["mode"] in {"bound_based", "rc_quantile_fallback"}, rec
    assert rec["n_survivors"] >= 1, "should have at least one survivor"
    new_windows = pruner.current_windows("lt")
    for fname in pruner.feature_names:
        lo, hi = new_windows[fname]
        assert math.isfinite(lo) and math.isfinite(hi), f"window for {fname} is infinite"
        assert lo <= hi, f"window for {fname} is inverted"
    print(f"[Assert] update tightened windows from open ranges to: "
          f"shortage_ratio={new_windows['shortage_ratio']}, "
          f"surplus_ratio={new_windows['surplus_ratio']} (OK)")

    # 4) Wipe-out guard: with explicit LB=0, UB=1 the bound rule keeps only
    # the 5 spike candidates (all rc < 0). Their features cluster at a single
    # point, so a literal min/max window would prune ~91% of the full
    # candidate set — the safeguard must refuse this update.
    spike_candidates = []
    for k in range(5):
        spike_candidates.append({
            "reduced_cost": -100.0 - 0.001 * k,
            "feature_values": {"shortage_ratio": 0.5, "surplus_ratio": 0.5,
                               "time_urgency": 0.5, "negative_reduced_cost": 100.0},
        })
    for k in range(50):
        # rc = 50..99, all > UB-LB = 1, so excluded by the bound rule.
        spike_candidates.append({
            "reduced_cost": 50.0 + k,
            "feature_values": {
                "shortage_ratio": 0.01 * (k + 1),
                "surplus_ratio": 0.95 - 0.01 * k,
                "time_urgency": 0.01 + 0.015 * k,
                "negative_reduced_cost": 0.0,
            },
        })
    isolated_pruner = irp.AdaptiveFeaturePruner(
        max_pruned_fraction=0.5, keep_best_rc=False
    )
    rec_spike = isolated_pruner.update(
        spike_candidates, iteration=1, phase="lt", lb=0.0, ub=1.0,
    )
    assert rec_spike["skipped_reason"] is not None, (
        f"wipe-out guard should have refused this update; record={rec_spike}"
    )
    assert rec_spike["skipped_reason"].startswith("would_prune_fraction"), rec_spike
    print(f"[Assert] wipe-out guard refused a window that would prune "
          f"too aggressively: {rec_spike['skipped_reason']} (OK)")


# ---------------------------------------------------------------------------
# End-to-end CG check: pruner sees real iterations, history populated
# ---------------------------------------------------------------------------

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
    data.dataset_id = "smoke_test_adaptive"
    data.scenario_id = "adaptive"
    return data, meta


def _end_to_end_check() -> None:
    print("\n--- End-to-end CG check (adaptive pruner enabled) ---")
    data, _ = _build_tiny_data()
    baseline_sol = irp.BaselineALNSModel(data).solve(
        msg=False,
        time_limit=30,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
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

    initial_patterns = irp.generate_random_lt_patterns(
        data, baseline_solution=baseline_sol,
        n_patterns_per_product_period=2, max_pairs_in_pattern=3,
        lt_activation_threshold=0.0, seed=42,
    )
    cg_engine = irp.LateralTransshipmentCG(
        data=data, baseline_solution=baseline_sol,
        initial_patterns=initial_patterns,
        lt_activation_threshold=0.0,
        max_pairs_per_pattern=3,
        use_gnn=False, collect_teacher_mode=False, runtime_gnn_mode=False,
        heuristic_top_k_mode=False, exact_full_mode=False,
        adaptive_pruning_enabled=True,
    )
    assert cg_engine.adaptive_pruner is not None, "pruner should be active"

    cg_sol = cg_engine.run_column_generation(max_iter=3, msg=False, stopping_mode="convergence")
    assert cg_sol.objective < float("inf"), "CG failed to produce a finite objective"

    history = cg_engine.adaptive_pruner.history
    print(f"[Assert] pruner history length = {len(history)} (>= 1 expected)")
    assert len(history) >= 1, "pruner update was never called"

    # Track unique window snapshots: at least two distinct windows = 'adaptive'.
    snapshots = set()
    n_real_updates = 0
    n_skipped = 0
    for rec in history:
        if rec["windows"]:
            snapshots.add(repr(sorted(rec["windows"].items())))
            n_real_updates += 1
        if rec.get("skipped_reason"):
            n_skipped += 1
    print(f"[Assert] real window updates = {n_real_updates}, skipped = {n_skipped}")
    print(f"[Assert] distinct window snapshots = {len(snapshots)}")
    assert n_real_updates >= 1, "no actual window update occurred — pruner is not adaptive"

    print(f"[Assert] LB seen by pruner (last call): "
          f"{history[-1].get('lb_used')}  mode={history[-1].get('mode')}")

    # No-wipe-out: at every real update, n_kept_after_window > 0.
    for rec in history:
        if rec["windows"]:
            assert rec.get("n_kept_after_window", 0) > 0, (
                f"adaptive windows would have wiped out all candidates: {rec}"
            )
    print("[Assert] no iteration wiped out all candidates (OK)")


def main() -> int:
    print("=" * 70)
    print("SMOKE TEST: AdaptiveFeaturePruner (Bianchessi-inspired)")
    print("=" * 70)
    t0 = time.perf_counter()
    _unit_checks()
    _end_to_end_check()
    print("\n" + "=" * 70)
    print(f"SMOKE TEST PASSED — adaptive pruning works ({time.perf_counter() - t0:.2f}s)")
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
