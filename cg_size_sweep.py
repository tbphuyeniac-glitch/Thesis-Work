#!/usr/bin/env python3
"""
cg_size_sweep.py
================
Compare 4 CG variants across 9 instance scenarios (3 small, 3 medium, 3 large).

Variants:
  V0_no_pruning_no_stack    exact_full_mode=True (oracle, no pruning)
  V1_no_time_urgency        2-feature pruning (shortage+surplus) + exact MIP
  V1_service_pruning_only   3-feature pruning (incl. saturated time_urgency) + exact MIP
  V2_no_time_urgency        2-feature pruning + exact MIP + Stackelberg delta filter

Scenarios (3 per tier):
  small:  s1=3x2, s2=4x2, s3=5x2
  medium: m1=6x3, m2=7x3, m3=8x4
  large:  l1=9x4, l2=10x5, l3=12x5

Key implementation notes:
  - IRP_ADAPTIVE_PRUNING=0 forced so the static feature_ranges actually applies.
    With adaptive pruning the auto-learned windows override the static ranges
    and hide the effect of feature selection.
  - V2's stackelberg_min_lateral_qty is auto-scaled by instance shortage volume:
    MOQ = clip(shortage * 0.005, 5, 500). Without scaling, fixed MOQ=200 starves
    the leader on small instances (follower greedy already covers baseline
    shortage so all leader columns have delta>=0 and are rejected).

Outputs (Results/Analysis/cg_ablation/):
  size_sweep.csv         one row per (scenario, variant)
  size_sweep_chart.png   3 rows (tiers) x 3 cols (wall, fill, prune_rate)

Run:
  python3 cg_size_sweep.py
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Force static feature_ranges to apply (not adaptive windows).
os.environ["IRP_ADAPTIVE_PRUNING"] = "0"
os.environ.setdefault("IRP_QUIET", "2")
os.environ.setdefault("IRP_BENCHMARK_FIXED_SHOCK", "1")
os.environ.setdefault("IRP_CG_STOPPING_MODE", "convergence")

import pandas as pd
import matplotlib.pyplot as plt

from cg_ablation_benchmark import (
    THREE_FEATURE_RANGES,
    UNBOUNDED_FOUR_FEATURE_RANGES,
    _resolve_excel_path,
    _shock_params_from_env,
    _stackelberg_params,
)
from irp_gurobi_converted import (
    BaselineALNSModel,
    DatasetToIRPValidationMapper,
    LateralTransshipmentCG,
    apply_hidden_local_reallocation_demand_shocks,
    build_lt_plan_df_from_cg,
    build_post_shock_inventory_state,
    build_realized_operating_cost_breakdown,
    generate_random_lt_patterns,
)


RESULTS_DIR = Path("Results/Analysis/cg_ablation")

# Two-feature pruning: drops the saturated time_urgency feature so the OR-bucket
# admission gate becomes meaningful. shortage_ratio>=0.30 OR surplus_ratio>=0.30.
TWO_FEATURE_RANGES_LOOSE: Dict[str, Dict[str, float]] = {
    "shortage_ratio": {"min": 0.30, "max": 1.00},
    "surplus_ratio":  {"min": 0.30, "max": 1.00},
}

VARIANTS: List[str] = [
    "V0_no_pruning_no_stack",
    "V1_no_time_urgency",
    "V1_service_pruning_only",
    "V2_no_time_urgency",
]

SHORT_LABEL = {
    "V0_no_pruning_no_stack":  "V0",
    "V1_no_time_urgency":      "V1_no_urg",
    "V1_service_pruning_only": "V1",
    "V2_no_time_urgency":      "V2_no_urg",
}

# 9 scenarios across 3 tiers. Tier groupings reflect typical CG runtime regimes:
#   small  = ~1-5s per CG variant (a few thousand candidate pairs total)
#   medium = ~5-15s per CG variant (tens of thousands of pairs)
#   large  = ~20-60s per CG variant (hundreds of thousands of pairs)
SCENARIOS: List[Dict] = [
    # 10 large scenarios only (store_limit <= 12, sku_limit <= 46)
    # Group A: store scaling, SKU=5 fixed  (pairs ~ n_stores^2 * n_skus)
    {"name": "l1",  "tier": "large", "store_limit": 9,  "sku_limit": 4},
    {"name": "l2",  "tier": "large", "store_limit": 10, "sku_limit": 5},
    {"name": "l3",  "tier": "large", "store_limit": 12, "sku_limit": 5},
    {"name": "l4",  "tier": "large", "store_limit": 9,  "sku_limit": 5},
    {"name": "l5",  "tier": "large", "store_limit": 11, "sku_limit": 5},
    # Group B: SKU scaling, stores=12 fixed
    {"name": "l6",  "tier": "large", "store_limit": 12, "sku_limit": 4},
    {"name": "l7",  "tier": "large", "store_limit": 12, "sku_limit": 6},
    {"name": "l8",  "tier": "large", "store_limit": 12, "sku_limit": 7},
    # Group C: mixed (both dimensions grow)
    {"name": "l9",  "tier": "large", "store_limit": 10, "sku_limit": 6},
    {"name": "l10", "tier": "large", "store_limit": 11, "sku_limit": 6},
]


def _total_realized_demand(data) -> float:
    """Total demand after shock application. Mirrors kaggle_sensitivity_analysis._total_demand."""
    realized = getattr(data, "realized_demand", None)
    if realized:
        return float(sum(realized.values()))
    return float(sum(data.demand.values()))


def _service_level(shortage: float, demand: float) -> float:
    """Classic service level: 1 - shortage/demand, clipped to [0,1].
    Same as kaggle_sensitivity_analysis._fill_rate."""
    if demand <= 1e-9:
        return 1.0
    return max(0.0, min(1.0, 1.0 - shortage / demand))


def _moq_for_shortage(total_shortage: float) -> float:
    """Heuristic Stackelberg MOQ scaled by instance shortage volume.

    Reasoning: with a fixed MOQ, small instances suffer because the follower's
    greedy can already cover most baseline shortage at small batch sizes,
    making delta>=0 for all leader patterns (~all rejected). Scaling by
    total shortage keeps the follower constrained enough that leader patterns
    add real value. Clipped to [5, 500] to bound extreme cases.
    """
    return float(max(5.0, min(500.0, total_shortage * 0.005)))


def _variant_kwargs(variant: str, moq_override: Optional[float] = None) -> dict:
    if variant == "V0_no_pruning_no_stack":
        return dict(
            use_gnn=False, heuristic_top_k_mode=False,
            exact_full_mode=True, stackelberg_aware_scoring=False,
            feature_ranges=UNBOUNDED_FOUR_FEATURE_RANGES,
        )
    if variant == "V1_no_time_urgency":
        return dict(
            use_gnn=False, heuristic_top_k_mode=False,
            exact_full_mode=False, pruned_exact_mode=True,
            stackelberg_aware_scoring=False,
            feature_ranges=TWO_FEATURE_RANGES_LOOSE,
        )
    if variant == "V1_service_pruning_only":
        return dict(
            use_gnn=False, heuristic_top_k_mode=False,
            exact_full_mode=False, pruned_exact_mode=True,
            stackelberg_aware_scoring=False,
            feature_ranges=THREE_FEATURE_RANGES,
        )
    if variant == "V2_no_time_urgency":
        moq = moq_override if moq_override is not None else float(
            os.environ.get("IRP_STACKELBERG_MIN_QTY", "200")
        )
        return dict(
            use_gnn=False, heuristic_top_k_mode=False,
            exact_full_mode=False, pruned_exact_mode=True,
            stackelberg_aware_scoring=True,
            feature_ranges=TWO_FEATURE_RANGES_LOOSE,
            stackelberg_min_lateral_qty=moq,
        )
    raise ValueError(f"Unknown variant: {variant}")


def _load_data_for_size(store_limit: int, sku_limit: int):
    excel_path = _resolve_excel_path()
    mapper = DatasetToIRPValidationMapper(
        excel_path=excel_path,
        sheet_name=os.environ.get("IRP_SHEET_NAME", "Sheet1"),
        store_limit=store_limit, sku_limit=sku_limit,
        start_date=os.environ.get("IRP_START_DATE"),
        end_date=os.environ.get("IRP_END_DATE"),
    )
    data, _, _, _ = mapper.build_irp_data(
        wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2,
        shortage_cost_rate=0.05,
        holding_cost_rate=0.01,
        cw_ship_cost_flat=1.0,
        lt_ship_cost_flat=0.6,
        fixed_dispatch_cw=8.0,
        fixed_dispatch_lt=2.0,
        vehicle_count=2,
        vehicle_capacity=float(os.environ.get("IRP_VEHICLE_CAPACITY", "900.0")),
        vehicle_fixed_cost=50.0,
        alpha=1.0,
        cw_replenishment_factor=0.8,
        cw_capacity_factor=2.0,
        store_initial_inventory_multiplier=float(
            os.environ.get("IRP_STORE_INIT_MULTIPLIER", "0.2")
        ),
        lt_cost_multiplier=float(os.environ.get("IRP_LT_COST_MULTIPLIER", "1.0")),
    )
    return data


def _compute_baseline_shortage(base_data, shared_baseline, shock_params: dict, seed: int) -> float:
    """Pre-compute total post-shock shortage (no LT) for MOQ scaling."""
    repeat_data = copy.deepcopy(base_data)
    baseline_for_run = copy.deepcopy(shared_baseline)
    apply_hidden_local_reallocation_demand_shocks(
        repeat_data, baseline_solution=baseline_for_run,
        shock_probability=shock_params["demand_shock_probability"],
        max_reallocation_fraction=shock_params["demand_shock_reallocation_fraction"],
        reallocations_per_product_period=shock_params["demand_shock_reallocations_per_product_period"],
        non_dispatch_shock_multiplier=shock_params["demand_shock_non_dispatch_multiplier"],
        cw_dispatch_cycle=5, seed=seed,
    )
    build_post_shock_inventory_state(repeat_data, baseline_for_run)
    pre_lt = build_realized_operating_cost_breakdown(repeat_data, baseline_for_run, lt_plan_df=None)
    return float(pre_lt["total_realized_shortage_units"])


def _run_one(variant: str, scenario: dict, base_data, shared_baseline,
             shock_params: dict, seed: int, lt_threshold: float, max_iter: int,
             n_init_per_pp: int, moq_for_v2: Optional[float]) -> dict:
    repeat_data = copy.deepcopy(base_data)
    baseline_for_run = copy.deepcopy(shared_baseline)
    apply_hidden_local_reallocation_demand_shocks(
        repeat_data, baseline_solution=baseline_for_run,
        shock_probability=shock_params["demand_shock_probability"],
        max_reallocation_fraction=shock_params["demand_shock_reallocation_fraction"],
        reallocations_per_product_period=shock_params["demand_shock_reallocations_per_product_period"],
        non_dispatch_shock_multiplier=shock_params["demand_shock_non_dispatch_multiplier"],
        cw_dispatch_cycle=5, seed=seed,
    )
    build_post_shock_inventory_state(repeat_data, baseline_for_run)
    init_patterns = generate_random_lt_patterns(
        repeat_data, baseline_solution=baseline_for_run,
        n_patterns_per_product_period=n_init_per_pp,
        max_pairs_in_pattern=4, lt_activation_threshold=lt_threshold,
        seed=int(os.environ.get("IRP_PATTERN_INIT_SEED", "123")),
    )
    cg = LateralTransshipmentCG(
        data=repeat_data, baseline_solution=baseline_for_run,
        initial_patterns=init_patterns,
        lt_activation_threshold=lt_threshold,
        max_pairs_per_pattern=4, top_pairs_per_feature=20, top_patterns_per_feature=5,
        stackelberg_params=_stackelberg_params(),
        diagnostic_verbosity="summary",
        **_variant_kwargs(variant, moq_override=moq_for_v2),
    )
    t0 = time.time()
    cg_sol = cg.run_column_generation(max_iter=max_iter, msg=False)
    wall = time.time() - t0
    lt_plan_df = build_lt_plan_df_from_cg(cg_sol, cg.patterns, repeat_data)
    pre_lt = build_realized_operating_cost_breakdown(repeat_data, baseline_for_run, lt_plan_df=None)
    post_lt = build_realized_operating_cost_breakdown(repeat_data, baseline_for_run, lt_plan_df=lt_plan_df)
    pre = float(pre_lt["total_realized_shortage_units"])
    post = float(post_lt["total_realized_shortage_units"])
    lt_shortage_recovery_ratio = 1.0 - (post / pre) if pre > 0 else 0.0
    post_demand = _total_realized_demand(repeat_data)
    post_lt_service_level = _service_level(post, post_demand)
    diags = getattr(cg, "cg_episode_diagnostics", []) or []
    kwargs = _variant_kwargs(variant, moq_override=moq_for_v2)
    is_pruned_exact = kwargs.get("pruned_exact_mode", False)
    cand = sum(int(d.get("candidate_pairs_before_pruning", 0) or 0) for d in diags)
    if is_pruned_exact:
        unique = sum(int(d.get("pairs_after_pruning_unique", 0) or 0) for d in diags)
    else:
        unique = sum(int(d.get("pairs_after_pruning", 0) or 0) for d in diags)
    has_stack = kwargs.get("stackelberg_aware_scoring", False)
    n_stack = (sum(int(d.get("pairs_accepted_stackelberg", 0) or 0) for d in diags)
               if has_stack else 0)
    # Patterns built by pricing MIP (input to Stackelberg filter for V2;
    # input directly to RMP gate for V0/V1). Read-only from diagnostics.
    n_patterns_built = sum(int(d.get("patterns_built_before_gnn", 0) or 0) for d in diags)
    # Patterns that actually entered the RMP pool after dedup/empty-flow gate.
    n_patterns_added_to_rmp = sum(int(d.get("patterns_added_to_pool", 0) or 0) for d in diags)
    # Stackelberg acceptance rate at pattern level: fraction of patterns built
    # that pass the delta<0 filter. Only meaningful for V2; V0/V1 set to 0.0.
    stack_acceptance_rate_pct = (
        round((n_stack / n_patterns_built) * 100.0, 2)
        if (has_stack and n_patterns_built > 0) else 0.0
    )
    return {
        "scenario": scenario["name"],
        "tier": scenario["tier"],
        "n_stores": len(repeat_data.stores),
        "n_skus": len(repeat_data.products),
        "n_periods": len(repeat_data.periods),
        "variant": variant,
        "moq_used": float(kwargs.get("stackelberg_min_lateral_qty", 0.0)),
        "wall": round(wall, 3),
        "lt_shortage_recovery_ratio": round(lt_shortage_recovery_ratio, 4),
        "post_lt_service_level": round(post_lt_service_level, 6),
        "post_realized_demand": round(post_demand, 2),
        "iters": int(cg_sol.iterations_run),
        "candidates": cand,
        "after_pruning_unique": unique,
        "prune_rate_pct": round((1.0 - unique / max(cand, 1)) * 100.0, 1),
        "n_stack_accepted": n_stack,
        "n_patterns_built": n_patterns_built,
        "n_patterns_added_to_rmp": n_patterns_added_to_rmp,
        "stack_acceptance_rate_pct": stack_acceptance_rate_pct,
        "rmp_objective_value": round(float(cg_sol.objective), 2),
        "shortage_before_lt": round(pre, 2),
        "shortage_after_lt": round(post, 2),
    }


def _make_chart(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("wall", "CG Wall Time (s)"),
        ("lt_shortage_recovery_ratio", "LT Shortage Recovery Ratio"),
        ("post_lt_service_level", "Post-LT Service Level"),
        ("prune_rate_pct", "Effective Prune Rate (%)"),
    ]
    tiers = ["small", "medium", "large"]
    fig, axes = plt.subplots(len(tiers), len(metrics), figsize=(18, 12))
    colors = ["#9aa0a6", "#4a90d9", "#e07a5f", "#2e7d32"]
    n_var = len(VARIANTS)
    width = 0.8 / n_var

    for r, tier in enumerate(tiers):
        sub = df[df["tier"] == tier]
        scenarios_in_tier = sorted(sub["scenario"].unique().tolist())
        n_sc = len(scenarios_in_tier)
        for c, (col, title) in enumerate(metrics):
            ax = axes[r, c]
            for v_i, variant in enumerate(VARIANTS):
                vals = []
                for sc in scenarios_in_tier:
                    cell = sub.loc[(sub["scenario"] == sc) & (sub["variant"] == variant), col]
                    vals.append(float(cell.mean()) if not cell.empty else 0.0)
                x_positions = [i + v_i * width - 0.4 + width / 2 for i in range(n_sc)]
                ax.bar(x_positions, vals, width=width,
                       color=colors[v_i % len(colors)], label=SHORT_LABEL[variant])
            ax.set_xticks(range(n_sc))
            ax.set_xticklabels(scenarios_in_tier)
            ax.set_title(f"{tier} | {title}")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle("CG Ablation: 4 variants x 9 scenarios (adaptive pruning OFF)", y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] -> {out_path}")


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cg_max_iter = int(os.environ.get("IRP_CG_ITERATIONS", "15"))
    lt_threshold = float(os.environ.get("IRP_LT_ACTIVATION_THRESHOLD", "10.0"))
    n_init_per_pp = int(os.environ.get("IRP_N_INITIAL_PATTERNS", "5"))
    shock_params = _shock_params_from_env()
    seed = shock_params["demand_shock_seed"]

    rows: List[dict] = []
    for scenario in SCENARIOS:
        print("\n" + "=" * 78)
        print(f"[Scenario] {scenario['name']} ({scenario['tier']}): "
              f"store_limit={scenario['store_limit']} sku_limit={scenario['sku_limit']}")
        print("=" * 78)
        try:
            base_data = _load_data_for_size(scenario["store_limit"], scenario["sku_limit"])
        except Exception as e:
            print(f"  data load failed: {e}")
            rows.append({"scenario": scenario["name"], "tier": scenario["tier"],
                         "variant": "(load_failed)", "error": str(e)[:200]})
            continue
        print(f"[Data] |stores|={len(base_data.stores)} "
              f"|skus|={len(base_data.products)} |periods|={len(base_data.periods)}")
        print(f"[ALNS] solving baseline for {scenario['name']}...")
        t0 = time.time()
        shared_baseline = BaselineALNSModel(base_data).solve(
            msg=False, time_limit=None, enforce_integer_flows=False,
            add_valid_16_20=True, allow_lateral_transshipment=False, cw_dispatch_cycle=5,
        )
        print(f"[ALNS] obj={shared_baseline.objective:.2f}  ({time.time()-t0:.1f}s)")

        # Pre-compute baseline shortage to scale V2 MOQ.
        shortage_total = _compute_baseline_shortage(base_data, shared_baseline, shock_params, seed)
        moq_for_v2 = _moq_for_shortage(shortage_total)
        print(f"[MOQ] baseline_shortage={shortage_total:.0f} -> V2 MOQ={moq_for_v2:.0f}")

        for variant in VARIANTS:
            print(f"\n[Run] scenario={scenario['name']} variant={variant}")
            try:
                row = _run_one(
                    variant=variant, scenario=scenario,
                    base_data=base_data, shared_baseline=shared_baseline,
                    shock_params=shock_params, seed=seed,
                    lt_threshold=lt_threshold, max_iter=cg_max_iter,
                    n_init_per_pp=n_init_per_pp,
                    moq_for_v2=moq_for_v2,
                )
                rows.append(row)
                print(f"  -> wall={row['wall']:.2f}s "
                      f"recovery={row['lt_shortage_recovery_ratio']:.4f} "
                      f"service={row['post_lt_service_level']:.4f} "
                      f"iters={row['iters']} prune={row['prune_rate_pct']}% "
                      f"built={row['n_patterns_built']} "
                      f"stack={row['n_stack_accepted']} "
                      f"stack_rate={row['stack_acceptance_rate_pct']}% "
                      f"in_rmp={row['n_patterns_added_to_rmp']} "
                      f"rmp_obj={row['rmp_objective_value']:.2e}")
            except Exception as e:
                import traceback
                print(f"  FAILED: {e}")
                traceback.print_exc()
                rows.append({
                    "scenario": scenario["name"], "tier": scenario["tier"],
                    "variant": variant, "error": str(e)[:200],
                })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "size_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[CSV] -> {out_csv}")
    print("\n=== ALL RESULTS ===")
    if "wall" in df.columns:
        cols_to_show = [c for c in [
            "scenario", "tier", "n_stores", "n_skus", "variant", "moq_used",
            "wall", "lt_shortage_recovery_ratio", "post_lt_service_level",
            "post_realized_demand",
            "iters", "candidates", "after_pruning_unique",
            "prune_rate_pct",
            "n_patterns_built", "n_stack_accepted",
            "stack_acceptance_rate_pct", "n_patterns_added_to_rmp",
            "rmp_objective_value",
            "shortage_before_lt", "shortage_after_lt",
        ] if c in df.columns]
        print(df[cols_to_show].to_string(index=False))
    else:
        print(df.to_string(index=False))

    if not df.empty and "wall" in df.columns and df["wall"].notna().any():
        _make_chart(df.dropna(subset=["wall"]), RESULTS_DIR / "size_sweep_chart.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
