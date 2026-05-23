#!/usr/bin/env python3
"""
sensitivity_analysis.py
=======================
Sensitivity analysis on three parameters for the IRP-LT pipeline:

  1) demand_shock_reallocation_fraction  (V1: 3-feature pruning, NO Stackelberg)
  2) shortage/holding cost ratio (pi/h)   (V1: 3-feature pruning, NO Stackelberg)
  3) stackelberg_min_lateral_qty          (V2: 3-feature pruning + Stackelberg)

All variants use exact Gurobi MIP pricing (pruned_exact_mode=True). No GNN, no
heuristic pricing. Pruning features: shortage_ratio, surplus_ratio, time_urgency.

Pipeline per run:
  ALNS baseline (no LT)  ->  apply_hidden_local_reallocation_demand_shocks
                         ->  CG with the chosen variant (V1 or V2)

Outputs (Results/Analysis/sensitivity/<sweep_name>/):
  per_run.csv        one row per (param_value, repeat)
  summary_table.csv  mean across repeats per param_value with breakdown
  line_chart.png     2-panel line chart (cost, service level) with pre/post-shock

Run:
  python sensitivity_analysis.py                  # all three sweeps
  python sensitivity_analysis.py --sweep shock    # only shock fraction
  python sensitivity_analysis.py --sweep pih      # only pi/h ratio
  python sensitivity_analysis.py --sweep moq      # only min_lateral_qty

Env overrides:
  N_REPEATS (default 1), IRP_TIME_LIMIT, EXCEL_PATH
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

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


RESULTS_ROOT = Path("Results/Analysis/sensitivity")

THREE_FEATURE_RANGES: Dict[str, Dict[str, float]] = {
    "shortage_ratio": {"min": 0.30, "max": 1.00},
    "surplus_ratio":  {"min": 0.30, "max": 1.00},
    "time_urgency":   {"min": 0.10, "max": 1.00},
}

# Fixed instance for ALL three sweeps — matches scenario L1 in cg_size_sweep.py
# (9 stores x 4 SKUs, large tier) so service level numbers are directly comparable.
SCENARIO = dict(
    store_limit=9,
    sku_limit=4,
    vehicle_count=2,
    vehicle_capacity=900.0,
    wh_inventory_multiplier=0.8,
    store_capacity_multiplier=1.2,
    cw_ship_cost_flat=1.0,
    lt_ship_cost_flat=0.6,
    fixed_dispatch_cw=8.0,
    fixed_dispatch_lt=2.0,
    vehicle_fixed_cost=50.0,
    alpha=1.0,
    cw_replenishment_factor=0.8,
    cw_capacity_factor=2.0,
    store_initial_inventory_multiplier=0.2,
    lt_cost_multiplier=1.0,
)

# Sweep values
SHOCK_FRACTION_VALUES = [0.15, 0.30, 0.45, 0.60, 0.75]
PI_H_RATIO_VALUES = [1, 3, 5, 10, 20]   # shortage_rate = ratio * holding_rate (holding=0.01)
HOLDING_COST_RATE = 0.01
MIN_LATERAL_QTY_VALUES = [50, 100, 200, 350, 500]

# Default shock params (held constant when not the swept variable).
DEFAULT_SHOCK_PROBABILITY = 0.85
DEFAULT_SHOCK_FRACTION = 0.60
DEFAULT_REALLOCATIONS_PER_PP = 3
DEFAULT_NON_DISPATCH_MULTIPLIER = 1.8
DEFAULT_SHOCK_SEED = 20260418

# CG control
CG_MAX_ITER = int(os.environ.get("IRP_CG_ITERATIONS", "15"))
LT_ACTIVATION_THRESHOLD = float(os.environ.get("IRP_LT_ACTIVATION_THRESHOLD", "10.0"))
N_INITIAL_PATTERNS_PER_PP = int(os.environ.get("IRP_N_INITIAL_PATTERNS", "5"))
PATTERN_INIT_SEED = int(os.environ.get("IRP_PATTERN_INIT_SEED", "123"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_excel_path() -> str:
    explicit = os.environ.get("EXCEL_PATH") or os.environ.get("IRP_DATASET_PATH")
    if explicit:
        return explicit
    default_csv = (
        Path(__file__).resolve().parent
        / "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
    )
    return str(default_csv)


def load_data(shortage_cost_rate: float, holding_cost_rate: float):
    """Build IRPData with a specified shortage/holding cost rate.

    For pi/h sweep this is rebuilt per ratio value. For other sweeps it is
    built once with default rates (0.05 / 0.01 -> ratio = 5).
    """
    excel_path = _resolve_excel_path()
    mapper = DatasetToIRPValidationMapper(
        excel_path=excel_path,
        sheet_name=os.environ.get("IRP_SHEET_NAME", "Sheet1"),
        store_limit=SCENARIO["store_limit"],
        sku_limit=SCENARIO["sku_limit"],
        start_date=os.environ.get("IRP_START_DATE"),
        end_date=os.environ.get("IRP_END_DATE"),
    )
    data, _, _, _ = mapper.build_irp_data(
        wh_inventory_multiplier=SCENARIO["wh_inventory_multiplier"],
        store_capacity_multiplier=SCENARIO["store_capacity_multiplier"],
        shortage_cost_rate=shortage_cost_rate,
        holding_cost_rate=holding_cost_rate,
        cw_ship_cost_flat=SCENARIO["cw_ship_cost_flat"],
        lt_ship_cost_flat=SCENARIO["lt_ship_cost_flat"],
        fixed_dispatch_cw=SCENARIO["fixed_dispatch_cw"],
        fixed_dispatch_lt=SCENARIO["fixed_dispatch_lt"],
        vehicle_count=SCENARIO["vehicle_count"],
        vehicle_capacity=SCENARIO["vehicle_capacity"],
        vehicle_fixed_cost=SCENARIO["vehicle_fixed_cost"],
        alpha=SCENARIO["alpha"],
        cw_replenishment_factor=SCENARIO["cw_replenishment_factor"],
        cw_capacity_factor=SCENARIO["cw_capacity_factor"],
        store_initial_inventory_multiplier=SCENARIO["store_initial_inventory_multiplier"],
        lt_cost_multiplier=SCENARIO["lt_cost_multiplier"],
    )
    return data


def solve_baseline(data) -> Tuple[object, float]:
    env_time_limit = os.environ.get("IRP_TIME_LIMIT")
    t0 = time.time()
    sol = BaselineALNSModel(data).solve(
        msg=False,
        time_limit=int(env_time_limit) if env_time_limit else None,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
    return sol, time.time() - t0


def _stackelberg_params() -> StackelbergParams:
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


def _v1_kwargs() -> dict:
    return dict(
        use_gnn=False,
        heuristic_top_k_mode=False,
        exact_full_mode=False,
        pruned_exact_mode=True,
        stackelberg_aware_scoring=False,
        feature_ranges=THREE_FEATURE_RANGES,
    )


def _v2_kwargs(min_lateral_qty: float) -> dict:
    return dict(
        use_gnn=False,
        heuristic_top_k_mode=False,
        exact_full_mode=False,
        pruned_exact_mode=True,
        stackelberg_aware_scoring=True,
        feature_ranges=THREE_FEATURE_RANGES,
        stackelberg_min_lateral_qty=float(min_lateral_qty),
    )


def _total_demand(data) -> float:
    """Total demand for fill-rate denominator.

    Uses realized_demand if shock has been applied (sum matches numerator's
    realized shortage). Falls back to forecast demand otherwise.
    """
    realized = getattr(data, "realized_demand", None)
    if realized:
        return float(sum(realized.values()))
    return float(sum(data.demand.values()))


def _fill_rate(shortage_units: float, demand_units: float) -> float:
    if demand_units <= 1e-9:
        return 1.0
    return max(0.0, min(1.0, 1.0 - shortage_units / demand_units))


# ---------------------------------------------------------------------------
# Single-run engine
# ---------------------------------------------------------------------------

def run_one(
    *,
    sweep_name: str,
    param_value: float,
    repeat_idx: int,
    seed_for_repeat: int,
    base_data,
    baseline_sol,
    pre_breakdown: Dict[str, float],
    pre_total_demand: float,
    shock_fraction: float,
    variant_kwargs_fn,
) -> Dict[str, object]:
    """Apply shock, run CG, collect pre/post metrics + breakdown."""
    repeat_data = copy.deepcopy(base_data)
    baseline_for_run = copy.deepcopy(baseline_sol)

    apply_hidden_local_reallocation_demand_shocks(
        repeat_data,
        baseline_solution=baseline_for_run,
        shock_probability=DEFAULT_SHOCK_PROBABILITY,
        max_reallocation_fraction=shock_fraction,
        reallocations_per_product_period=DEFAULT_REALLOCATIONS_PER_PP,
        non_dispatch_shock_multiplier=DEFAULT_NON_DISPATCH_MULTIPLIER,
        cw_dispatch_cycle=5,
        seed=seed_for_repeat,
    )
    build_post_shock_inventory_state(repeat_data, baseline_for_run)

    initial_patterns = generate_random_lt_patterns(
        repeat_data,
        baseline_solution=baseline_for_run,
        n_patterns_per_product_period=N_INITIAL_PATTERNS_PER_PP,
        max_pairs_in_pattern=4,
        lt_activation_threshold=LT_ACTIVATION_THRESHOLD,
        seed=PATTERN_INIT_SEED,
    )

    cg = LateralTransshipmentCG(
        data=repeat_data,
        baseline_solution=baseline_for_run,
        initial_patterns=initial_patterns,
        lt_activation_threshold=LT_ACTIVATION_THRESHOLD,
        max_pairs_per_pattern=4,
        top_pairs_per_feature=20,
        top_patterns_per_feature=5,
        stackelberg_params=_stackelberg_params(),
        diagnostic_verbosity="summary",
        **variant_kwargs_fn(),
    )

    os.environ["IRP_CG_STOPPING_MODE"] = "convergence"

    t0 = time.time()
    cg_sol = cg.run_column_generation(max_iter=CG_MAX_ITER, msg=False)
    cg_wall_time_sec = time.time() - t0

    lt_plan_df = build_lt_plan_df_from_cg(cg_sol, cg.patterns, repeat_data)
    post_breakdown = build_realized_operating_cost_breakdown(
        repeat_data, baseline_for_run, lt_plan_df=lt_plan_df
    )
    post_total_demand = _total_demand(repeat_data)

    # Pair / column counts (V1 + V2 share the pruned_exact path).
    diags = getattr(cg, "cg_episode_diagnostics", []) or []
    n_cols_generated = sum(int(d.get("candidate_pairs_before_pruning", 0) or 0) for d in diags)
    n_cols_after_pruning = sum(int(d.get("pairs_after_pruning_unique", 0) or 0) for d in diags)
    has_stack = variant_kwargs_fn().get("stackelberg_aware_scoring", False)
    n_cols_after_stack = (
        sum(int(d.get("pairs_accepted_stackelberg", 0) or 0) for d in diags)
        if has_stack else 0
    )

    pre_total_cost = float(pre_breakdown["total_realized_operating_cost"])
    post_total_cost = float(post_breakdown["total_realized_operating_cost"])
    post_shortage = float(post_breakdown["total_realized_shortage_units"])

    return {
        "sweep": sweep_name,
        "param_value": param_value,
        "repeat": repeat_idx,
        "seed": seed_for_repeat,
        # Pre-shock cost (baseline solved on original demand, no LT) — for cost-chart reference only
        "pre_total_cost": round(pre_total_cost, 4),
        # Post-shock + LT (CG solution executed on shocked demand)
        "post_total_cost": round(post_total_cost, 4),
        "post_shortage_units": round(post_shortage, 4),
        "post_total_demand": round(post_total_demand, 4),
        "post_fill_rate": round(_fill_rate(post_shortage, post_total_demand), 6),
        # Cost breakdown (post-shock, with LT)
        "direct_cw_unit_cost": float(post_breakdown["direct_cw_unit_cost_executed_plan"]),
        "store_holding_cost": float(post_breakdown["store_holding_cost_realized"]),
        "warehouse_holding_cost": float(post_breakdown["warehouse_holding_cost_executed_plan"]),
        "route_distance_cost": float(post_breakdown["route_distance_cost_executed_plan"]),
        "vehicle_fixed_cost": float(post_breakdown["vehicle_fixed_cost_executed_plan"]),
        "lateral_transshipment_cost": float(post_breakdown["lateral_transshipment_cost_realized"]),
        "shortage_cost": float(post_breakdown["shortage_cost_realized"]),
        # CG performance
        "n_columns_generated": int(n_cols_generated),
        "n_columns_after_pruning": int(n_cols_after_pruning),
        "n_columns_after_stackelberg": int(n_cols_after_stack),
        "cg_iterations": int(cg_sol.iterations_run),
        "cg_wall_time_sec": round(cg_wall_time_sec, 4),
    }


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def _baseline_pre_metrics(base_data, baseline_sol) -> Tuple[Dict[str, float], float]:
    """Realized cost + total demand for the baseline solution on ORIGINAL data."""
    pre_breakdown = build_realized_operating_cost_breakdown(
        base_data, baseline_sol, lt_plan_df=None
    )
    pre_total_demand = _total_demand(base_data)
    return pre_breakdown, pre_total_demand


def sweep_shock_fraction(n_repeats: int) -> pd.DataFrame:
    """V1 + vary demand_shock_reallocation_fraction in {0.15..0.75}."""
    print("\n" + "=" * 70)
    print("[Sweep 1] demand_shock_reallocation_fraction (variant V1, no Stackelberg)")
    print("=" * 70)

    base_data = load_data(shortage_cost_rate=0.05, holding_cost_rate=HOLDING_COST_RATE)
    print(f"[Data] |stores|={len(base_data.stores)} |skus|={len(base_data.products)} "
          f"|periods|={len(base_data.periods)}")
    print("[ALNS] solving baseline (no LT, original demand)...")
    baseline_sol, t_baseline = solve_baseline(base_data)
    print(f"[ALNS] baseline objective={baseline_sol.objective:.4f}  ({t_baseline:.1f}s)")

    pre_breakdown, pre_total_demand = _baseline_pre_metrics(base_data, baseline_sol)
    print(f"[Pre]  baseline_cost={pre_breakdown['total_realized_operating_cost']:.4f}")

    rows: List[dict] = []
    for value in SHOCK_FRACTION_VALUES:
        for r in range(n_repeats):
            seed = DEFAULT_SHOCK_SEED + 10007 * r
            print(f"\n[Run] shock_fraction={value:.2f}  repeat={r}  seed={seed}")
            row = run_one(
                sweep_name="shock_fraction",
                param_value=value,
                repeat_idx=r,
                seed_for_repeat=seed,
                base_data=base_data,
                baseline_sol=baseline_sol,
                pre_breakdown=pre_breakdown,
                pre_total_demand=pre_total_demand,
                shock_fraction=value,
                variant_kwargs_fn=_v1_kwargs,
            )
            rows.append(row)
            print(f"  -> post_cost={row['post_total_cost']:.2f}  "
                  f"post_fill={row['post_fill_rate']:.4f}  "
                  f"cols_after_prune={row['n_columns_after_pruning']}  "
                  f"iters={row['cg_iterations']}  "
                  f"wall={row['cg_wall_time_sec']:.2f}s")
    return pd.DataFrame(rows)


def sweep_pi_h_ratio(n_repeats: int) -> pd.DataFrame:
    """V1 + vary shortage/holding ratio. Rebuilds data + re-solves baseline per ratio."""
    print("\n" + "=" * 70)
    print("[Sweep 2] shortage/holding cost ratio (variant V1, no Stackelberg)")
    print("=" * 70)

    rows: List[dict] = []
    for ratio in PI_H_RATIO_VALUES:
        shortage_rate = ratio * HOLDING_COST_RATE
        print(f"\n[Stage] pi/h ratio={ratio} (shortage_rate={shortage_rate:.4f}, "
              f"holding_rate={HOLDING_COST_RATE:.4f})")
        base_data = load_data(shortage_cost_rate=shortage_rate, holding_cost_rate=HOLDING_COST_RATE)
        print("[ALNS] solving baseline (no LT, original demand)...")
        baseline_sol, t_baseline = solve_baseline(base_data)
        print(f"[ALNS] baseline objective={baseline_sol.objective:.4f}  ({t_baseline:.1f}s)")

        pre_breakdown, pre_total_demand = _baseline_pre_metrics(base_data, baseline_sol)
        print(f"[Pre]  total_cost={pre_breakdown['total_realized_operating_cost']:.4f}  "
              f"shortage={pre_breakdown['total_realized_shortage_units']:.4f}  "
              f"fill_rate={_fill_rate(pre_breakdown['total_realized_shortage_units'], pre_total_demand):.4f}")

        for r in range(n_repeats):
            seed = DEFAULT_SHOCK_SEED + 10007 * r
            print(f"\n[Run] pi/h={ratio}  repeat={r}  seed={seed}")
            row = run_one(
                sweep_name="pi_h_ratio",
                param_value=ratio,
                repeat_idx=r,
                seed_for_repeat=seed,
                base_data=base_data,
                baseline_sol=baseline_sol,
                pre_breakdown=pre_breakdown,
                pre_total_demand=pre_total_demand,
                shock_fraction=DEFAULT_SHOCK_FRACTION,
                variant_kwargs_fn=_v1_kwargs,
            )
            rows.append(row)
            print(f"  -> post_cost={row['post_total_cost']:.2f}  "
                  f"post_fill={row['post_fill_rate']:.4f}  "
                  f"cols_after_prune={row['n_columns_after_pruning']}  "
                  f"iters={row['cg_iterations']}  "
                  f"wall={row['cg_wall_time_sec']:.2f}s")
    return pd.DataFrame(rows)


def sweep_min_lateral_qty(n_repeats: int) -> pd.DataFrame:
    """V2 + vary stackelberg_min_lateral_qty in {50..500}."""
    print("\n" + "=" * 70)
    print("[Sweep 3] stackelberg_min_lateral_qty (variant V2, with Stackelberg)")
    print("=" * 70)

    base_data = load_data(shortage_cost_rate=0.05, holding_cost_rate=HOLDING_COST_RATE)
    print(f"[Data] |stores|={len(base_data.stores)} |skus|={len(base_data.products)} "
          f"|periods|={len(base_data.periods)}")
    print("[ALNS] solving baseline (no LT, original demand)...")
    baseline_sol, t_baseline = solve_baseline(base_data)
    print(f"[ALNS] baseline objective={baseline_sol.objective:.4f}  ({t_baseline:.1f}s)")

    pre_breakdown, pre_total_demand = _baseline_pre_metrics(base_data, baseline_sol)
    print(f"[Pre]  baseline_cost={pre_breakdown['total_realized_operating_cost']:.4f}")

    rows: List[dict] = []
    for value in MIN_LATERAL_QTY_VALUES:
        for r in range(n_repeats):
            seed = DEFAULT_SHOCK_SEED + 10007 * r
            print(f"\n[Run] min_lateral_qty={value}  repeat={r}  seed={seed}")
            row = run_one(
                sweep_name="min_lateral_qty",
                param_value=value,
                repeat_idx=r,
                seed_for_repeat=seed,
                base_data=base_data,
                baseline_sol=baseline_sol,
                pre_breakdown=pre_breakdown,
                pre_total_demand=pre_total_demand,
                shock_fraction=DEFAULT_SHOCK_FRACTION,
                variant_kwargs_fn=lambda v=value: _v2_kwargs(v),
            )
            rows.append(row)
            print(f"  -> post_cost={row['post_total_cost']:.2f}  "
                  f"post_fill={row['post_fill_rate']:.4f}  "
                  f"cols_after_prune={row['n_columns_after_pruning']}  "
                  f"cols_after_stack={row['n_columns_after_stackelberg']}  "
                  f"iters={row['cg_iterations']}  "
                  f"wall={row['cg_wall_time_sec']:.2f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting (per-sweep): summary table + 2-panel line chart
# ---------------------------------------------------------------------------

SUMMARY_COLS_MEAN = [
    "pre_total_cost",
    "post_total_cost", "post_fill_rate",
    "direct_cw_unit_cost", "store_holding_cost", "warehouse_holding_cost",
    "route_distance_cost", "vehicle_fixed_cost", "lateral_transshipment_cost",
    "shortage_cost",
    "n_columns_generated", "n_columns_after_pruning", "n_columns_after_stackelberg",
    "cg_iterations", "cg_wall_time_sec",
]


def _aggregate(per_run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = per_run_df.groupby("param_value", sort=True)[SUMMARY_COLS_MEAN].mean().reset_index()
    return grouped


def _make_line_chart(
    summary_df: pd.DataFrame,
    sweep_name: str,
    x_label: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    x = summary_df["param_value"].tolist()

    # Panel 1: total cost (pre + post)
    ax = axes[0]
    ax.plot(x, summary_df["pre_total_cost"], marker="s", linestyle="--",
            color="#9aa0a6", label="Pre-shock (baseline)")
    ax.plot(x, summary_df["post_total_cost"], marker="o", linestyle="-",
            color="#2e7d32", label="Post-shock + LT (CG)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Total realized cost")
    ax.set_title(f"Total Cost vs {x_label}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Panel 2: post-shock service level only (pre-shock metric removed —
    # pre uses MILP B var which is inflated by capacity binding; not comparable)
    ax = axes[1]
    ax.plot(x, summary_df["post_fill_rate"], marker="o", linestyle="-",
            color="#4a90d9", label="Post-shock + LT (CG)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Service level (fill rate)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Service Level vs {x_label}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.suptitle(f"Sensitivity: {sweep_name}", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] saved -> {out_path}")


def report_sweep(per_run_df: pd.DataFrame, sweep_name: str, x_label: str) -> None:
    out_dir = RESULTS_ROOT / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_path = out_dir / "per_run.csv"
    per_run_df.to_csv(per_run_path, index=False)
    print(f"\n[CSV] per_run -> {per_run_path}")

    summary_df = _aggregate(per_run_df)
    summary_path = out_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[CSV] summary -> {summary_path}")

    _make_line_chart(summary_df, sweep_name, x_label, out_dir / "line_chart.png")

    print(f"\n=== SUMMARY: {sweep_name} ===")
    print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        choices=["all", "shock", "pih", "moq"],
        default="all",
        help="Which sweep to run (default: all three).",
    )
    args = parser.parse_args()

    n_repeats = int(os.environ.get("N_REPEATS", "1"))
    print(f"[Config] N_REPEATS={n_repeats}  CG_MAX_ITER={CG_MAX_ITER}  "
          f"LT_ACTIVATION_THRESHOLD={LT_ACTIVATION_THRESHOLD}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.sweep in ("all", "shock"):
        df = sweep_shock_fraction(n_repeats)
        report_sweep(df, "shock_fraction", "Demand shock reallocation fraction")

    if args.sweep in ("all", "pih"):
        df = sweep_pi_h_ratio(n_repeats)
        report_sweep(df, "pi_h_ratio", "Shortage / holding cost ratio")

    if args.sweep in ("all", "moq"):
        df = sweep_min_lateral_qty(n_repeats)
        report_sweep(df, "min_lateral_qty", "Stackelberg min_lateral_qty")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
