#!/usr/bin/env python3
"""Diagnostic: why does V1 prune almost nothing?

Two hypotheses:
  H1 (parameter): static thresholds (shortage_ratio>=0.30, surplus_ratio>=0.30,
       time_urgency>=0.10) are too lenient for store_limit=7.
  H2 (logic): adaptive pruning is ON by default (IRP_ADAPTIVE_PRUNING=1) and its
       auto-learned windows override the static feature_ranges with wider bounds.

This script:
  1. Disables adaptive pruning.
  2. Inspects the feature value distribution on the medium instance.
  3. Runs V0 vs V1 across three threshold profiles (loose/medium/tight) and
     reports pairs_after_pruning_unique / fill / wall.
"""
from __future__ import annotations

import copy
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

# Force adaptive pruning off so the static feature_ranges actually apply.
os.environ["IRP_ADAPTIVE_PRUNING"] = "0"
os.environ["IRP_QUIET"] = "2"  # suppress per-(p,t) lines, keep summaries
os.environ["IRP_CG_STOPPING_MODE"] = "convergence"
os.environ.setdefault("IRP_BENCHMARK_FIXED_SHOCK", "1")

import pandas as pd

from cg_ablation_benchmark import (
    THREE_FEATURE_RANGES,
    UNBOUNDED_FOUR_FEATURE_RANGES,
    _shock_params_from_env,
    _stackelberg_params,
    load_medium_data,
)
from irp_gurobi_converted import (
    BaselineALNSModel,
    LateralTransshipmentCG,
    apply_hidden_local_reallocation_demand_shocks,
    build_lt_plan_df_from_cg,
    build_post_shock_inventory_state,
    build_realized_operating_cost_breakdown,
    generate_random_lt_patterns,
)


PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    "loose (current)": {
        "shortage_ratio": {"min": 0.30, "max": 1.00},
        "surplus_ratio":  {"min": 0.30, "max": 1.00},
        "time_urgency":   {"min": 0.10, "max": 1.00},
    },
    "medium": {
        "shortage_ratio": {"min": 0.50, "max": 1.00},
        "surplus_ratio":  {"min": 0.50, "max": 1.00},
        "time_urgency":   {"min": 0.30, "max": 1.00},
    },
    "tight": {
        "shortage_ratio": {"min": 0.70, "max": 1.00},
        "surplus_ratio":  {"min": 0.70, "max": 1.00},
        "time_urgency":   {"min": 0.50, "max": 1.00},
    },
    # The OR-semantics in _prune_pairs_by_feature means time_urgency (saturated
    # at min=0.501) admits 100% of pairs alone. Dropping it and tightening the
    # remaining two features should make pruning bite under OR-semantics.
    "no_urgency_loose": {
        "shortage_ratio": {"min": 0.30, "max": 1.00},
        "surplus_ratio":  {"min": 0.30, "max": 1.00},
    },
    "no_urgency_medium": {
        "shortage_ratio": {"min": 0.50, "max": 1.00},
        "surplus_ratio":  {"min": 0.50, "max": 1.00},
    },
    "no_urgency_tight": {
        "shortage_ratio": {"min": 0.70, "max": 1.00},
        "surplus_ratio":  {"min": 0.70, "max": 1.00},
    },
}


def _prep_repeat(base_data, shared_baseline, shock_params, seed):
    repeat_data = copy.deepcopy(base_data)
    baseline_for_run = copy.deepcopy(shared_baseline)
    apply_hidden_local_reallocation_demand_shocks(
        repeat_data,
        baseline_solution=baseline_for_run,
        shock_probability=shock_params["demand_shock_probability"],
        max_reallocation_fraction=shock_params["demand_shock_reallocation_fraction"],
        reallocations_per_product_period=shock_params["demand_shock_reallocations_per_product_period"],
        non_dispatch_shock_multiplier=shock_params["demand_shock_non_dispatch_multiplier"],
        cw_dispatch_cycle=5,
        seed=seed,
    )
    build_post_shock_inventory_state(repeat_data, baseline_for_run)
    return repeat_data, baseline_for_run


def _inspect_feature_distribution(cg, repeat_data, baseline_for_run) -> None:
    """Print feature value distribution across all active (p,t) pairs."""
    need, surplus = cg._build_need_and_surplus_proxies(master_solution=None)
    active_pts = cg._compute_active_product_periods(need, surplus)
    print(f"\n[Feature distribution] active (p,t) pairs: {len(active_pts)}")

    all_shortage: List[float] = []
    all_surplus: List[float] = []
    all_urgency: List[float] = []
    n_pairs = 0
    for p, t in active_pts:
        fmap = cg._feature_value_map(p=p, t=t, need=need, surplus=surplus)
        for (i, j), fvals in fmap.items():
            all_shortage.append(fvals["shortage_ratio"])
            all_surplus.append(fvals["surplus_ratio"])
            all_urgency.append(fvals["time_urgency"])
            n_pairs += 1

    def _quants(xs, name):
        if not xs:
            print(f"  {name}: empty")
            return
        xs = sorted(xs)
        n = len(xs)
        p25 = xs[n // 4]
        p50 = xs[n // 2]
        p75 = xs[(3 * n) // 4]
        p10 = xs[n // 10] if n >= 10 else xs[0]
        p90 = xs[(9 * n) // 10] if n >= 10 else xs[-1]
        print(f"  {name:14s}  min={min(xs):.3f}  p10={p10:.3f}  p25={p25:.3f}  "
              f"p50={p50:.3f}  p75={p75:.3f}  p90={p90:.3f}  max={max(xs):.3f}  "
              f"mean={statistics.mean(xs):.3f}")

    print(f"  total candidate pairs across all active (p,t): {n_pairs}")
    _quants(all_shortage, "shortage_ratio")
    _quants(all_surplus, "surplus_ratio")
    _quants(all_urgency, "time_urgency")

    for prof_name, ranges in PROFILES.items():
        kept_or = 0
        kept_and = 0
        for s_r, su_r, u_r in zip(all_shortage, all_surplus, all_urgency):
            checks = []
            if "shortage_ratio" in ranges:
                checks.append(ranges["shortage_ratio"]["min"] <= s_r <= ranges["shortage_ratio"]["max"])
            if "surplus_ratio" in ranges:
                checks.append(ranges["surplus_ratio"]["min"] <= su_r <= ranges["surplus_ratio"]["max"])
            if "time_urgency" in ranges:
                checks.append(ranges["time_urgency"]["min"] <= u_r <= ranges["time_urgency"]["max"])
            if any(checks):  # OR-semantics: matches the actual codebase
                kept_or += 1
            if all(checks):  # AND-semantics: hypothetical alternative
                kept_and += 1
        rate_or = (1.0 - kept_or / max(n_pairs, 1)) * 100.0
        rate_and = (1.0 - kept_and / max(n_pairs, 1)) * 100.0
        print(f"  profile={prof_name:20s}  OR_kept={kept_or:>4d}/{n_pairs} ({rate_or:.1f}% pruned) | "
              f"AND_kept={kept_and:>4d}/{n_pairs} ({rate_and:.1f}% pruned)")


def _run_one(label, base_data, shared_baseline, shock_params, seed, kwargs,
             initial_patterns_seed_pp=5, lt_threshold=10.0, max_iter=15) -> dict:
    repeat_data, baseline_for_run = _prep_repeat(base_data, shared_baseline, shock_params, seed)
    init_patterns = generate_random_lt_patterns(
        repeat_data, baseline_solution=baseline_for_run,
        n_patterns_per_product_period=initial_patterns_seed_pp,
        max_pairs_in_pattern=4, lt_activation_threshold=lt_threshold, seed=123,
    )
    cg = LateralTransshipmentCG(
        data=repeat_data, baseline_solution=baseline_for_run,
        initial_patterns=init_patterns, lt_activation_threshold=lt_threshold,
        max_pairs_per_pattern=4, top_pairs_per_feature=20, top_patterns_per_feature=5,
        stackelberg_params=_stackelberg_params(), diagnostic_verbosity="summary",
        **kwargs,
    )
    t0 = time.time()
    cg_sol = cg.run_column_generation(max_iter=max_iter, msg=False)
    wall = time.time() - t0
    lt_plan_df = build_lt_plan_df_from_cg(cg_sol, cg.patterns, repeat_data)
    pre_lt = build_realized_operating_cost_breakdown(repeat_data, baseline_for_run, lt_plan_df=None)
    post_lt = build_realized_operating_cost_breakdown(repeat_data, baseline_for_run, lt_plan_df=lt_plan_df)
    pre = float(pre_lt["total_realized_shortage_units"]); post = float(post_lt["total_realized_shortage_units"])
    fill = 1.0 - (post / pre) if pre > 0 else 0.0
    diags = getattr(cg, "cg_episode_diagnostics", []) or []
    if kwargs.get("pruned_exact_mode"):
        unique = sum(int(d.get("pairs_after_pruning_unique", 0) or 0) for d in diags)
    else:
        unique = sum(int(d.get("pairs_after_pruning", 0) or 0) for d in diags)
    cand = sum(int(d.get("candidate_pairs_before_pruning", 0) or 0) for d in diags)
    return {
        "label": label,
        "wall": round(wall, 2),
        "fill": round(fill, 4),
        "iters": int(cg_sol.iterations_run),
        "candidates": cand,
        "after_pruning_unique": unique,
        "prune_rate": round((1.0 - unique / max(cand, 1)) * 100.0, 1),
        "obj": round(float(cg_sol.objective), 2),
    }


def main() -> int:
    print("=" * 80)
    print("DIAGNOSTIC: V0 vs V1 with IRP_ADAPTIVE_PRUNING=0 across threshold profiles")
    print("=" * 80)
    base_data = load_medium_data()
    shock_params = _shock_params_from_env()
    seed = shock_params["demand_shock_seed"]

    print("\n[ALNS] solving baseline...")
    t0 = time.time()
    shared_baseline = BaselineALNSModel(base_data).solve(
        msg=False, time_limit=None, enforce_integer_flows=False,
        add_valid_16_20=True, allow_lateral_transshipment=False, cw_dispatch_cycle=5,
    )
    print(f"[ALNS] baseline_obj={shared_baseline.objective:.4f}  ({time.time()-t0:.1f}s)")

    # Inspect feature distribution once (via a temporary CG with V1 setup)
    repeat_data, baseline_for_run = _prep_repeat(base_data, shared_baseline, shock_params, seed)
    init_patterns = generate_random_lt_patterns(
        repeat_data, baseline_solution=baseline_for_run,
        n_patterns_per_product_period=5, max_pairs_in_pattern=4,
        lt_activation_threshold=10.0, seed=123,
    )
    inspect_cg = LateralTransshipmentCG(
        data=repeat_data, baseline_solution=baseline_for_run,
        initial_patterns=init_patterns, lt_activation_threshold=10.0,
        max_pairs_per_pattern=4, top_pairs_per_feature=20, top_patterns_per_feature=5,
        stackelberg_params=_stackelberg_params(), diagnostic_verbosity="summary",
        use_gnn=False, heuristic_top_k_mode=False, exact_full_mode=False,
        pruned_exact_mode=True, stackelberg_aware_scoring=False,
        feature_ranges=THREE_FEATURE_RANGES,
    )
    _inspect_feature_distribution(inspect_cg, repeat_data, baseline_for_run)

    rows: List[dict] = []
    print("\n[V0] reference (no pruning):")
    rows.append(_run_one(
        label="V0_no_prune",
        base_data=base_data, shared_baseline=shared_baseline,
        shock_params=shock_params, seed=seed,
        kwargs=dict(
            use_gnn=False, heuristic_top_k_mode=False,
            exact_full_mode=True, stackelberg_aware_scoring=False,
            feature_ranges=UNBOUNDED_FOUR_FEATURE_RANGES,
        ),
    ))
    for prof_name, ranges in PROFILES.items():
        print(f"\n[V1] profile={prof_name}")
        rows.append(_run_one(
            label=f"V1_{prof_name}",
            base_data=base_data, shared_baseline=shared_baseline,
            shock_params=shock_params, seed=seed,
            kwargs=dict(
                use_gnn=False, heuristic_top_k_mode=False,
                exact_full_mode=False, pruned_exact_mode=True,
                stackelberg_aware_scoring=False, feature_ranges=ranges,
            ),
        ))

    df = pd.DataFrame(rows)
    print("\n=== RESULT ===")
    print(df.to_string(index=False))
    out_path = Path("Results/Analysis/cg_ablation/diag_v0_v1.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[CSV] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
