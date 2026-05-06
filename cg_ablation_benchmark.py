#!/usr/bin/env python3
"""
cg_ablation_benchmark.py
========================
Ablation benchmark to validate the Column Generation method (pruning + Stackelberg).

Three CG variants are run on the same medium-tier instance (store_limit=7, sku_limit=3),
the same ALNS baseline, and the same demand-shock realisation per repeat.
No GNN is used in any variant.

  V0_no_pruning_no_stack         exact_full_mode=True,  stack=False  (oracle CG)
  V1_service_pruning_only        3-feature pruning,     stack=False
  V2_service_pruning_plus_stack  3-feature pruning,     stack=True

Outputs (Results/Analysis/cg_ablation/):
  per_run.csv       one row per (variant, repeat)
  aggregate.csv     mean +/- std per variant
  summary_chart.png 3 bar charts: wall_time / fill_rate / cols_after_pruning

Run:
  python cg_ablation_benchmark.py
  N_REPEATS=3 EXCEL_PATH=data/your.csv python cg_ablation_benchmark.py
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# All public symbols come from irp_gurobi_converted.py (do NOT modify).
# ---------------------------------------------------------------------------
from irp_gurobi_converted import (
    DatasetToIRPValidationMapper,
    BaselineALNSModel,
    LateralTransshipmentCG,
    StackelbergParams,
    apply_hidden_local_reallocation_demand_shocks,
    build_post_shock_inventory_state,
    build_lt_plan_df_from_cg,
    build_realized_operating_cost_breakdown,
    generate_random_lt_patterns,
)


RESULTS_DIR = Path("Results/Analysis/cg_ablation")

VARIANTS = [
    "V0_no_pruning_no_stack",
    "V1_service_pruning_only",
    "V2_service_pruning_plus_stack",
]

SHORT_LABEL = {
    "V0_no_pruning_no_stack": "V0",
    "V1_service_pruning_only": "V1",
    "V2_service_pruning_plus_stack": "V2",
}

THREE_FEATURE_RANGES: Dict[str, Dict[str, float]] = {
    "shortage_ratio": {"min": 0.30, "max": 1.00},
    "surplus_ratio":  {"min": 0.30, "max": 1.00},
    "time_urgency":   {"min": 0.10, "max": 1.00},
}

# V0 keeps all four features but with unbounded ranges so nothing is pruned;
# this matches the "no pruning" intent while staying compatible with
# exact_full_mode (which itself bypasses the feature gate).
UNBOUNDED_FOUR_FEATURE_RANGES: Dict[str, Dict[str, float]] = {
    "shortage_ratio":         {"min": 0.0, "max": math.inf},
    "surplus_ratio":          {"min": 0.0, "max": math.inf},
    "time_urgency":            {"min": 0.0, "max": math.inf},
    "negative_reduced_cost":  {"min": 0.0, "max": math.inf},
}


def _variant_kwargs(variant: str) -> dict:
    if variant == "V0_no_pruning_no_stack":
        return dict(
            use_gnn=False,
            heuristic_top_k_mode=False,
            exact_full_mode=True,
            stackelberg_aware_scoring=False,
            feature_ranges=UNBOUNDED_FOUR_FEATURE_RANGES,
        )
    if variant == "V1_service_pruning_only":
        return dict(
            use_gnn=False,
            heuristic_top_k_mode=False,
            exact_full_mode=False,
            pruned_exact_mode=True,
            stackelberg_aware_scoring=False,
            feature_ranges=THREE_FEATURE_RANGES,
        )
    if variant == "V2_service_pruning_plus_stack":
        # pruned_exact_mode=True + stackelberg_aware_scoring=True activates the
        # combined branch in pricing_step(): feature pruning → exact Gurobi MIP
        # → Stackelberg delta filter (admit only patterns where delta < 0).
        # stackelberg_min_lateral_qty=200: follower requires ≥200-unit shipments
        # to activate an arc. With default=5 the follower covers most shortage
        # in baseline already → delta ≈ 0 → all patterns rejected (~47% fill).
        # 200 reflects the assumption that stores cannot self-coordinate small
        # batches without the central LT plan. ONLY affects follower solver.
        return dict(
            use_gnn=False,
            heuristic_top_k_mode=False,
            exact_full_mode=False,
            pruned_exact_mode=True,
            stackelberg_aware_scoring=True,
            feature_ranges=THREE_FEATURE_RANGES,
            stackelberg_min_lateral_qty=float(
                os.environ.get("IRP_STACKELBERG_MIN_QTY", "200")
            ),
        )
    raise ValueError(f"Unknown variant: {variant}")


def _resolve_excel_path() -> str:
    explicit = os.environ.get("EXCEL_PATH") or os.environ.get("IRP_DATASET_PATH")
    if explicit:
        return explicit
    default_csv = Path(__file__).resolve().parent / "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
    return str(default_csv)


def load_medium_data():
    excel_path = _resolve_excel_path()
    print(f"[Data] Loading from {excel_path}")
    mapper = DatasetToIRPValidationMapper(
        excel_path=excel_path,
        sheet_name=os.environ.get("IRP_SHEET_NAME", "Sheet1"),
        store_limit=int(os.environ.get("STORE_LIMIT", os.environ.get("IRP_STORE_LIMIT", "7"))),
        sku_limit=int(os.environ.get("SKU_LIMIT", os.environ.get("IRP_SKU_LIMIT", "3"))),
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
        store_initial_inventory_multiplier=float(os.environ.get("IRP_STORE_INIT_MULTIPLIER", "0.2")),
        lt_cost_multiplier=float(os.environ.get("IRP_LT_COST_MULTIPLIER", "1.0")),
    )
    print(f"[Data] |stores|={len(data.stores)} |skus|={len(data.products)} |periods|={len(data.periods)}")
    return data


def _shock_params_from_env() -> dict:
    return dict(
        demand_shock_probability=float(os.environ.get("IRP_DEMAND_SHOCK_PROBABILITY", "0.85")),
        demand_shock_reallocation_fraction=float(os.environ.get("IRP_DEMAND_SHOCK_REALLOCATION_FRACTION", "0.60")),
        demand_shock_reallocations_per_product_period=int(os.environ.get("IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD", "3")),
        demand_shock_non_dispatch_multiplier=float(os.environ.get("IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER", "1.8")),
        demand_shock_seed=int(os.environ.get("IRP_DEMAND_SHOCK_SEED", "20260418")),
    )


def _stackelberg_params() -> StackelbergParams:
    # Same parameters used by IRPResearchPipeline.run_lt_recourse_from_baseline.
    return StackelbergParams(
        donor_accept_threshold=0.0,
        receiver_accept_threshold=0.0,
        donor_risk_weight=1.2,
        donor_ship_burden_weight=1.0,
        donor_service_loss_weight=1.0,
        receiver_shortage_reduction_weight=2.0,
        receiver_service_gain_weight=1.0,
        receiver_handling_weight=0.5,
        min_compensation=0.0,
        compensation_cap=50.0,
        acceptance_score_weight=0.6,
        economic_score_weight=0.4,
        top_k_after_game_per_feature=5,
    )


def run_one(
    variant: str,
    repeat_idx: int,
    seed_for_repeat: int,
    shared_baseline_sol,
    base_data,
    shock_params: dict,
    lt_activation_threshold: float,
    cg_max_iter: int,
    n_initial_patterns_per_pp: int,
) -> dict:
    print(f"\n[Run] variant={variant} repeat={repeat_idx} seed={seed_for_repeat}")

    repeat_data = copy.deepcopy(base_data)
    baseline_for_run = copy.deepcopy(shared_baseline_sol)

    apply_hidden_local_reallocation_demand_shocks(
        repeat_data,
        baseline_solution=baseline_for_run,
        shock_probability=shock_params["demand_shock_probability"],
        max_reallocation_fraction=shock_params["demand_shock_reallocation_fraction"],
        reallocations_per_product_period=shock_params["demand_shock_reallocations_per_product_period"],
        non_dispatch_shock_multiplier=shock_params["demand_shock_non_dispatch_multiplier"],
        cw_dispatch_cycle=5,
        seed=seed_for_repeat,
    )
    build_post_shock_inventory_state(repeat_data, baseline_for_run)

    initial_patterns = generate_random_lt_patterns(
        repeat_data,
        baseline_solution=baseline_for_run,
        n_patterns_per_product_period=n_initial_patterns_per_pp,
        max_pairs_in_pattern=4,
        lt_activation_threshold=lt_activation_threshold,
        seed=int(os.environ.get("IRP_PATTERN_INIT_SEED", "123")),
    )

    cg = LateralTransshipmentCG(
        data=repeat_data,
        baseline_solution=baseline_for_run,
        initial_patterns=initial_patterns,
        lt_activation_threshold=lt_activation_threshold,
        max_pairs_per_pattern=4,
        top_pairs_per_feature=20,
        top_patterns_per_feature=5,
        stackelberg_params=_stackelberg_params(),
        diagnostic_verbosity="summary",
        **_variant_kwargs(variant),
    )

    os.environ["IRP_CG_STOPPING_MODE"] = "convergence"

    t0 = time.time()
    cg_sol = cg.run_column_generation(max_iter=cg_max_iter, msg=False)
    cg_wall_time_sec = time.time() - t0

    lt_plan_df = build_lt_plan_df_from_cg(cg_sol, cg.patterns, repeat_data)
    pre_lt = build_realized_operating_cost_breakdown(repeat_data, baseline_for_run, lt_plan_df=None)
    post_lt = build_realized_operating_cost_breakdown(repeat_data, baseline_for_run, lt_plan_df=lt_plan_df)
    pre_lt_shortage = float(pre_lt["total_realized_shortage_units"])
    post_lt_shortage = float(post_lt["total_realized_shortage_units"])
    fill_rate = 1.0 - (post_lt_shortage / pre_lt_shortage) if pre_lt_shortage > 0 else 0.0

    # Pair-count metrics live on cg.cg_episode_diagnostics (one entry per episode);
    # cg_sol.efficiency_metrics only carries RMP-level metrics. Sum across episodes.
    #
    # V0 (exact_full): pairs_after_pruning is a unique count (no pruning, = before).
    # V1/V2 (pruned_exact): _candidate_patterns_pruned_exact tracks pairs_after_pruning_unique
    # separately from the total (which would be inflated by the feature loop).
    kwargs = _variant_kwargs(variant)
    is_pruned_exact = kwargs.get("pruned_exact_mode", False)

    diags = getattr(cg, "cg_episode_diagnostics", []) or []
    n_cols_generated = sum(int(d.get("candidate_pairs_before_pruning", 0) or 0) for d in diags)
    if is_pruned_exact:
        # Use the deduplicated unique count produced by _candidate_patterns_pruned_exact.
        n_cols_after_pruning = sum(int(d.get("pairs_after_pruning_unique", 0) or 0) for d in diags)
    else:
        # V0 exact_full: pairs_after_pruning == pairs_before (unique, no pruning).
        n_cols_after_pruning = sum(int(d.get("pairs_after_pruning", 0) or 0) for d in diags)
    # pairs_accepted_stackelberg is set to delta<0 accepted count for V2 (exact path)
    # and to 0 for V0/V1. Force 0 for non-Stackelberg variants for clarity.
    has_stackelberg = kwargs.get("stackelberg_aware_scoring", False)
    n_cols_after_stack = (
        sum(int(d.get("pairs_accepted_stackelberg", 0) or 0) for d in diags)
        if has_stackelberg else 0
    )

    return {
        "variant": variant,
        "repeat": repeat_idx,
        "seed": seed_for_repeat,
        "cg_wall_time_sec": round(cg_wall_time_sec, 4),
        "n_columns_generated": float(n_cols_generated),
        "n_columns_after_pruning": float(n_cols_after_pruning),
        "n_columns_after_stackelberg": float(n_cols_after_stack),
        "total_lt_cost": round(float(cg_sol.objective), 6),
        "total_shortage_before_lt": round(pre_lt_shortage, 6),
        "total_shortage_after_lt": round(post_lt_shortage, 6),
        "shortage_fill_rate": round(fill_rate, 6),
        "cg_iterations": int(cg_sol.iterations_run),
    }


def _aggregate(per_run_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "cg_wall_time_sec",
        "n_columns_generated",
        "n_columns_after_pruning",
        "n_columns_after_stackelberg",
        "total_lt_cost",
        "total_shortage_before_lt",
        "total_shortage_after_lt",
        "shortage_fill_rate",
        "cg_iterations",
    ]
    grouped = per_run_df.groupby("variant")[numeric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(c).rstrip("_") for c in grouped.columns]
    return grouped


def _make_chart(per_run_df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("cg_wall_time_sec", "CG Wall Time (s)"),
        ("shortage_fill_rate", "Shortage Fill Rate"),
        ("n_columns_after_pruning", "Columns After Pruning\n(unique est., all iters)"),
    ]
    short_labels = [SHORT_LABEL[v] for v in VARIANTS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#9aa0a6", "#4a90d9", "#2e7d32"]
    for ax, (col, title) in zip(axes, metrics):
        means = [per_run_df.loc[per_run_df.variant == v, col].mean() for v in VARIANTS]
        stds = [per_run_df.loc[per_run_df.variant == v, col].std() for v in VARIANTS]
        stds = [0.0 if (s is None or pd.isna(s)) else float(s) for s in stds]
        ax.bar(short_labels, means, yerr=stds, capsize=6, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("CG Ablation: Pruning + Stackelberg", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] saved -> {out_path}")


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_repeats = int(os.environ.get("N_REPEATS", "1"))
    cg_max_iter = int(os.environ.get("IRP_CG_ITERATIONS", "15"))
    lt_activation_threshold = float(os.environ.get("IRP_LT_ACTIVATION_THRESHOLD", "10.0"))
    n_initial_patterns_per_pp = int(os.environ.get("IRP_N_INITIAL_PATTERNS", "5"))

    # Same shock per repeat across all variants -> identical demand realisation.
    os.environ.setdefault("IRP_BENCHMARK_FIXED_SHOCK", "1")

    base_data = load_medium_data()
    shock_params = _shock_params_from_env()

    print("\n[ALNS] solving baseline (no LT)...")
    env_time_limit = os.environ.get("IRP_TIME_LIMIT")
    t0 = time.time()
    shared_baseline_sol = BaselineALNSModel(base_data).solve(
        msg=False,
        time_limit=int(env_time_limit) if env_time_limit else None,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
    print(f"[ALNS] baseline objective={shared_baseline_sol.objective:.4f}  ({time.time()-t0:.1f}s)")

    fixed_shock = os.environ.get("IRP_BENCHMARK_FIXED_SHOCK", "1").lower() not in {"0", "false", "no", ""}
    base_seed = shock_params["demand_shock_seed"]
    if fixed_shock:
        seeds = [base_seed] * n_repeats
    else:
        seeds = [base_seed + 10007 * r for r in range(n_repeats)]
    print(f"\n[Benchmark] variants={len(VARIANTS)} repeats={n_repeats} fixed_shock={fixed_shock} seeds={seeds}")

    rows: List[dict] = []
    for variant in VARIANTS:
        for r, seed_for_repeat in enumerate(seeds):
            row = run_one(
                variant=variant,
                repeat_idx=r,
                seed_for_repeat=seed_for_repeat,
                shared_baseline_sol=shared_baseline_sol,
                base_data=base_data,
                shock_params=shock_params,
                lt_activation_threshold=lt_activation_threshold,
                cg_max_iter=cg_max_iter,
                n_initial_patterns_per_pp=n_initial_patterns_per_pp,
            )
            rows.append(row)
            print(
                f"  -> wall={row['cg_wall_time_sec']:.2f}s  "
                f"fill={row['shortage_fill_rate']:.3f}  "
                f"cols_gen={row['n_columns_generated']:.0f}  "
                f"cols_after_prune={row['n_columns_after_pruning']:.0f}  "
                f"iters={row['cg_iterations']}"
            )

    per_run_df = pd.DataFrame(rows)
    per_run_path = RESULTS_DIR / "per_run.csv"
    per_run_df.to_csv(per_run_path, index=False)
    print(f"\n[CSV] per_run -> {per_run_path}")

    agg_df = _aggregate(per_run_df)
    agg_path = RESULTS_DIR / "aggregate.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"[CSV] aggregate -> {agg_path}")

    _make_chart(per_run_df, RESULTS_DIR / "summary_chart.png")

    print("\n=== AGGREGATE SUMMARY ===")
    print(agg_df.to_string(index=False))
    return 0


def sweep_stackelberg_min_qty() -> int:
    """Quick diagnostic sweep: run V2 with different stackelberg_min_lateral_qty values.

    ALNS baseline is solved once; the shock seed is fixed. Results go to
    Results/Analysis/cg_ablation/sweep_stackelberg_min_qty.csv.

    Run:
      python cg_ablation_benchmark.py --sweep-stackelberg
    """
    SWEEP_VALUES = [5, 20, 50, 100, 200]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("IRP_BENCHMARK_FIXED_SHOCK", "1")
    os.environ["IRP_CG_STOPPING_MODE"] = "convergence"

    cg_max_iter = int(os.environ.get("IRP_CG_ITERATIONS", "15"))
    lt_activation_threshold = float(os.environ.get("IRP_LT_ACTIVATION_THRESHOLD", "10.0"))
    n_initial_patterns_per_pp = int(os.environ.get("IRP_N_INITIAL_PATTERNS", "5"))
    shock_params = _shock_params_from_env()
    seed = shock_params["demand_shock_seed"]

    base_data = load_medium_data()
    print("\n[ALNS] solving baseline (sweep mode)...")
    env_time_limit = os.environ.get("IRP_TIME_LIMIT")
    t0 = time.time()
    shared_baseline_sol = BaselineALNSModel(base_data).solve(
        msg=False,
        time_limit=int(env_time_limit) if env_time_limit else None,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
    print(f"[ALNS] baseline objective={shared_baseline_sol.objective:.4f}  ({time.time()-t0:.1f}s)")

    rows: List[dict] = []
    for min_qty in SWEEP_VALUES:
        repeat_data = copy.deepcopy(base_data)
        baseline_for_run = copy.deepcopy(shared_baseline_sol)

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

        initial_patterns = generate_random_lt_patterns(
            repeat_data,
            baseline_solution=baseline_for_run,
            n_patterns_per_product_period=n_initial_patterns_per_pp,
            max_pairs_in_pattern=4,
            lt_activation_threshold=lt_activation_threshold,
            seed=int(os.environ.get("IRP_PATTERN_INIT_SEED", "123")),
        )

        cg = LateralTransshipmentCG(
            data=repeat_data,
            baseline_solution=baseline_for_run,
            initial_patterns=initial_patterns,
            lt_activation_threshold=lt_activation_threshold,
            max_pairs_per_pattern=4,
            top_pairs_per_feature=20,
            top_patterns_per_feature=5,
            stackelberg_params=_stackelberg_params(),
            diagnostic_verbosity="summary",
            use_gnn=False,
            heuristic_top_k_mode=False,
            exact_full_mode=False,
            pruned_exact_mode=True,
            stackelberg_aware_scoring=True,
            feature_ranges=THREE_FEATURE_RANGES,
            stackelberg_min_lateral_qty=float(min_qty),
        )

        t0 = time.time()
        cg_sol = cg.run_column_generation(max_iter=cg_max_iter, msg=False)
        wall = time.time() - t0

        lt_plan_df = build_lt_plan_df_from_cg(cg_sol, cg.patterns, repeat_data)
        pre_lt = build_realized_operating_cost_breakdown(repeat_data, baseline_for_run, lt_plan_df=None)
        post_lt = build_realized_operating_cost_breakdown(repeat_data, baseline_for_run, lt_plan_df=lt_plan_df)
        pre_lt_shortage = float(pre_lt["total_realized_shortage_units"])
        post_lt_shortage = float(post_lt["total_realized_shortage_units"])
        fill_rate = 1.0 - (post_lt_shortage / pre_lt_shortage) if pre_lt_shortage > 0 else 0.0

        diags = getattr(cg, "cg_episode_diagnostics", []) or []
        n_stack_accepted = sum(int(d.get("pairs_accepted_stackelberg", 0) or 0) for d in diags)

        row = {
            "stackelberg_min_lateral_qty": min_qty,
            "cg_wall_time_sec": round(wall, 4),
            "shortage_fill_rate": round(fill_rate, 6),
            "total_shortage_before_lt": round(pre_lt_shortage, 6),
            "total_shortage_after_lt": round(post_lt_shortage, 6),
            "n_columns_after_stackelberg": float(n_stack_accepted),
            "cg_iterations": int(cg_sol.iterations_run),
        }
        rows.append(row)
        print(
            f"  min_qty={min_qty:>4d} | wall={wall:.1f}s  fill={fill_rate:.3f}  "
            f"stack_accepted={n_stack_accepted}  iters={cg_sol.iterations_run}"
        )

    sweep_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "sweep_stackelberg_min_qty.csv"
    sweep_df.to_csv(out_path, index=False)
    print(f"\n[Sweep] saved -> {out_path}")
    print(sweep_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    if "--sweep-stackelberg" in sys.argv:
        sys.exit(sweep_stackelberg_min_qty())
    sys.exit(main())
