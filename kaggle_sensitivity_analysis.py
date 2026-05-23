# %% [markdown]
# # IRP-LT Sensitivity Analysis (3 Parameters)
# Run all cells top-to-bottom.
# Ensure Kaggle secrets `WLSACCESSID` / `WLSSECRET` / `LICENSEID` are set for Gurobi.

# %% [code]
# =========================================================
# 0) Install & imports
# =========================================================
import os, sys, time, copy, math, subprocess, shutil, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# 1) Clone repo
# =========================================================
REPO_URL  = "https://github.com/tbphuyeniac-glitch/Thesis-Work.git"
REPO_ROOT = Path("/kaggle/working/Thesis-Work")

if REPO_ROOT.exists():
    shutil.rmtree(REPO_ROOT)

subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_ROOT)],
               check=True, capture_output=True)
print(f"[Repo] cloned -> {REPO_ROOT}")

sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# =========================================================
# 2) Gurobi licence (from Kaggle secrets)
# =========================================================
from kaggle_secrets import UserSecretsClient
_secrets = UserSecretsClient()
os.environ["WLSACCESSID"] = _secrets.get_secret("WLSACCESSID")
os.environ["WLSSECRET"]   = _secrets.get_secret("WLSSECRET")
os.environ["LICENSEID"]   = _secrets.get_secret("LICENSEID")

import gurobipy  # noqa: F401 — confirms licence OK
print(f"[Gurobi] gurobipy {gurobipy.gurobi.version()} loaded")

# =========================================================
# 3) Core imports from repo
# =========================================================
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
print("[Import] irp_gurobi_converted OK")

# %% [code]
# =========================================================
# 4) Configuration — edit here if needed
# =========================================================
RESULTS_ROOT = Path("/kaggle/working/sensitivity")

DATA_PATH = str(REPO_ROOT / "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv")

# Fixed scenario — matches scenario L1 in cg_size_sweep.py (9 stores x 4 SKUs,
# large tier) so service level numbers are directly comparable across files.
SCENARIO = dict(
    store_limit=9, sku_limit=4,
    vehicle_count=2, vehicle_capacity=900.0,
    wh_inventory_multiplier=0.8,
    store_capacity_multiplier=1.2,
    cw_ship_cost_flat=1.0, lt_ship_cost_flat=0.6,
    fixed_dispatch_cw=8.0, fixed_dispatch_lt=2.0,
    vehicle_fixed_cost=50.0, alpha=1.0,
    cw_replenishment_factor=0.8, cw_capacity_factor=2.0,
    store_initial_inventory_multiplier=0.2,
    lt_cost_multiplier=1.0,
)

# Sweep values
SHOCK_FRACTION_VALUES  = [0.15, 0.30, 0.45, 0.60, 0.75]
PI_H_RATIO_VALUES      = [1, 3, 5, 10, 20]  # shortage_rate = ratio * 0.01
HOLDING_COST_RATE      = 0.01
MIN_LATERAL_QTY_VALUES = [50, 100, 200, 350, 500]

# 3-feature pruning thresholds (same as cg_ablation V1/V2)
THREE_FEATURE_RANGES = {
    "shortage_ratio": {"min": 0.30, "max": 1.00},
    "surplus_ratio":  {"min": 0.30, "max": 1.00},
    "time_urgency":   {"min": 0.10, "max": 1.00},
}

# CG control
os.environ["IRP_CG_STOPPING_MODE"] = "convergence"
os.environ["IRP_ADAPTIVE_PRUNING"]  = "0"   # avoid 51% overhead on small instances
CG_MAX_ITER             = 15
LT_ACTIVATION_THRESHOLD = 10.0
N_INITIAL_PATTERNS_PP   = 5
PATTERN_INIT_SEED       = 123

# Shock defaults (held fixed when not the swept variable)
DEFAULT_SHOCK_PROB       = 0.85
DEFAULT_SHOCK_FRACTION   = 0.60
DEFAULT_REALLOCS_PP      = 3
DEFAULT_NON_DISPATCH_MUL = 1.8
DEFAULT_SHOCK_SEED       = 20260418

print("[Config] OK")
print(f"  scenario: stores={SCENARIO['store_limit']}  skus={SCENARIO['sku_limit']}  "
      f"vehicles={SCENARIO['vehicle_count']}  capacity={SCENARIO['vehicle_capacity']}")

# %% [code]
# =========================================================
# 5) Shared helpers
# =========================================================

def load_data(shortage_cost_rate: float, holding_cost_rate: float):
    mapper = DatasetToIRPValidationMapper(
        excel_path=DATA_PATH,
        sheet_name="Sheet1",
        store_limit=SCENARIO["store_limit"],
        sku_limit=SCENARIO["sku_limit"],
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


def solve_baseline(data):
    t0 = time.time()
    sol = BaselineALNSModel(data).solve(
        msg=False,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
    print(f"  [ALNS] objective={sol.objective:.4f}  ({time.time()-t0:.1f}s)")
    return sol


def _stackelberg_params():
    return StackelbergParams(
        donor_accept_threshold=0.0, receiver_accept_threshold=0.0,
        donor_risk_weight=1.2, donor_ship_burden_weight=1.0,
        donor_service_loss_weight=1.0,
        receiver_shortage_reduction_weight=2.0, receiver_service_gain_weight=1.0,
        receiver_handling_weight=0.5,
        min_compensation=0.0, compensation_cap=50.0,
        acceptance_score_weight=0.6, economic_score_weight=0.4,
        top_k_after_game_per_feature=5,
    )


def _v1_kwargs():
    return dict(
        use_gnn=False, heuristic_top_k_mode=False,
        exact_full_mode=False, pruned_exact_mode=True,
        stackelberg_aware_scoring=False,
        feature_ranges=THREE_FEATURE_RANGES,
    )


def _v2_kwargs(min_lateral_qty: float):
    return dict(
        use_gnn=False, heuristic_top_k_mode=False,
        exact_full_mode=False, pruned_exact_mode=True,
        stackelberg_aware_scoring=True,
        feature_ranges=THREE_FEATURE_RANGES,
        stackelberg_min_lateral_qty=float(min_lateral_qty),
    )


def _total_demand(data) -> float:
    """Use realized_demand if shock has been applied; else forecast demand."""
    realized = getattr(data, "realized_demand", None)
    if realized:
        return float(sum(realized.values()))
    return float(sum(data.demand.values()))


def _fill_rate(shortage: float, demand: float) -> float:
    if demand <= 1e-9:
        return 1.0
    return max(0.0, min(1.0, 1.0 - shortage / demand))


def pre_shock_metrics(base_data, baseline_sol):
    bd = build_realized_operating_cost_breakdown(base_data, baseline_sol, lt_plan_df=None)
    return bd, _total_demand(base_data)


def run_one(*, sweep_name, param_value, seed, base_data, baseline_sol,
            pre_bd, pre_demand, shock_fraction, variant_kwargs):
    repeat_data    = copy.deepcopy(base_data)
    baseline_copy  = copy.deepcopy(baseline_sol)

    apply_hidden_local_reallocation_demand_shocks(
        repeat_data, baseline_solution=baseline_copy,
        shock_probability=DEFAULT_SHOCK_PROB,
        max_reallocation_fraction=shock_fraction,
        reallocations_per_product_period=DEFAULT_REALLOCS_PP,
        non_dispatch_shock_multiplier=DEFAULT_NON_DISPATCH_MUL,
        cw_dispatch_cycle=5, seed=seed,
    )
    build_post_shock_inventory_state(repeat_data, baseline_copy)

    patterns = generate_random_lt_patterns(
        repeat_data, baseline_solution=baseline_copy,
        n_patterns_per_product_period=N_INITIAL_PATTERNS_PP,
        max_pairs_in_pattern=4,
        lt_activation_threshold=LT_ACTIVATION_THRESHOLD,
        seed=PATTERN_INIT_SEED,
    )

    cg = LateralTransshipmentCG(
        data=repeat_data, baseline_solution=baseline_copy,
        initial_patterns=patterns,
        lt_activation_threshold=LT_ACTIVATION_THRESHOLD,
        max_pairs_per_pattern=4, top_pairs_per_feature=20,
        top_patterns_per_feature=5,
        stackelberg_params=_stackelberg_params(),
        diagnostic_verbosity="summary",
        **variant_kwargs,
    )

    t0 = time.time()
    cg_sol = cg.run_column_generation(max_iter=CG_MAX_ITER, msg=False)
    cg_time = time.time() - t0

    lt_plan = build_lt_plan_df_from_cg(cg_sol, cg.patterns, repeat_data)
    post_bd = build_realized_operating_cost_breakdown(
        repeat_data, baseline_copy, lt_plan_df=lt_plan
    )
    post_demand = _total_demand(repeat_data)

    diags = getattr(cg, "cg_episode_diagnostics", []) or []
    n_gen   = sum(int(d.get("candidate_pairs_before_pruning", 0) or 0) for d in diags)
    n_prune = sum(int(d.get("pairs_after_pruning_unique", 0) or 0) for d in diags)
    n_stack = (
        sum(int(d.get("pairs_accepted_stackelberg", 0) or 0) for d in diags)
        if variant_kwargs.get("stackelberg_aware_scoring") else 0
    )

    pre_cost     = float(pre_bd["total_realized_operating_cost"])
    post_cost    = float(post_bd["total_realized_operating_cost"])
    post_shortage= float(post_bd["total_realized_shortage_units"])

    return {
        "sweep": sweep_name,
        "param_value": param_value,
        "seed": seed,
        # Pre-shock cost (baseline ALNS, no LT) — for cost-chart reference
        "pre_total_cost":  round(pre_cost, 4),
        "post_total_cost": round(post_cost, 4),
        "post_fill_rate":  round(_fill_rate(post_shortage, post_demand), 6),
        "direct_cw_unit_cost":        float(post_bd["direct_cw_unit_cost_executed_plan"]),
        "store_holding_cost":          float(post_bd["store_holding_cost_realized"]),
        "warehouse_holding_cost":      float(post_bd["warehouse_holding_cost_executed_plan"]),
        "route_distance_cost":         float(post_bd["route_distance_cost_executed_plan"]),
        "vehicle_fixed_cost":          float(post_bd["vehicle_fixed_cost_executed_plan"]),
        "lateral_transshipment_cost":  float(post_bd["lateral_transshipment_cost_realized"]),
        "shortage_cost":               float(post_bd["shortage_cost_realized"]),
        "n_cols_generated":    int(n_gen),
        "n_cols_after_pruning":int(n_prune),
        "n_cols_after_stack":  int(n_stack),
        "cg_iterations":       int(cg_sol.iterations_run),
        "cg_wall_time_sec":    round(cg_time, 4),
    }


def save_and_plot(rows: List[dict], sweep_name: str, x_label: str):
    out_dir = RESULTS_ROOT / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_run.csv", index=False)

    # Summary (one row per param value — single seed so same as raw)
    num_cols = [c for c in df.columns if c not in ("sweep", "param_value", "seed")]
    summary  = df.groupby("param_value", sort=True)[num_cols].mean().reset_index()
    summary.to_csv(out_dir / "summary_table.csv", index=False)

    # Line chart: cost + service level
    x = summary["param_value"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(x, summary["pre_total_cost"],  marker="s", ls="--",
            color="#9aa0a6", label="Pre-shock (baseline, no LT)")
    ax.plot(x, summary["post_total_cost"], marker="o", ls="-",
            color="#2e7d32", label="Post-shock + LT (CG)")
    ax.set_xlabel(x_label); ax.set_ylabel("Total realized cost")
    ax.set_title(f"Total Cost vs {x_label}")
    ax.grid(True, ls="--", alpha=0.4); ax.legend()

    ax = axes[1]
    ax.plot(x, summary["post_fill_rate"], marker="o", ls="-",
            color="#4a90d9", label="Post-shock + LT (CG)")
    ax.set_xlabel(x_label); ax.set_ylabel("Service level (fill rate)")
    ax.set_ylim(0.0, 1.05); ax.set_title(f"Service Level vs {x_label}")
    ax.grid(True, ls="--", alpha=0.4); ax.legend()

    fig.suptitle(f"Sensitivity: {sweep_name}", y=1.02)
    plt.tight_layout()
    chart_path = out_dir / "line_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)

    print(f"\n[{sweep_name}] per_run  -> {out_dir/'per_run.csv'}")
    print(f"[{sweep_name}] summary  -> {out_dir/'summary_table.csv'}")
    print(f"[{sweep_name}] chart    -> {chart_path}")
    print(f"\n--- Summary ---\n{summary.to_string(index=False)}")
    return df, summary


print("[Helpers] OK")

# %% [code]
# =========================================================
# 6) Sweep 1: demand_shock_reallocation_fraction  (V1)
# =========================================================
print("\n" + "="*65)
print("Sweep 1: demand_shock_reallocation_fraction  [V1, no Stackelberg]")
print("="*65)

base_data_s1 = load_data(shortage_cost_rate=0.05, holding_cost_rate=HOLDING_COST_RATE)
print(f"[Data] stores={len(base_data_s1.stores)}  skus={len(base_data_s1.products)}  "
      f"periods={len(base_data_s1.periods)}")
print("[ALNS] solving baseline ...")
baseline_s1 = solve_baseline(base_data_s1)
pre_bd_s1, pre_dem_s1 = pre_shock_metrics(base_data_s1, baseline_s1)

rows_s1 = []
for val in SHOCK_FRACTION_VALUES:
    print(f"\n  shock_fraction={val:.2f}")
    row = run_one(
        sweep_name="shock_fraction", param_value=val,
        seed=DEFAULT_SHOCK_SEED,
        base_data=base_data_s1, baseline_sol=baseline_s1,
        pre_bd=pre_bd_s1, pre_demand=pre_dem_s1,
        shock_fraction=val,
        variant_kwargs=_v1_kwargs(),
    )
    rows_s1.append(row)
    print(f"  post_cost={row['post_total_cost']:.2f}  fill={row['post_fill_rate']:.4f}  "
          f"iters={row['cg_iterations']}  time={row['cg_wall_time_sec']:.1f}s")

df_s1, summary_s1 = save_and_plot(rows_s1, "shock_fraction",
                                   "Demand shock reallocation fraction")

# %% [code]
# =========================================================
# 7) Sweep 2: shortage / holding cost ratio  (V1)
# =========================================================
print("\n" + "="*65)
print("Sweep 2: shortage/holding cost ratio pi/h  [V1, no Stackelberg]")
print("="*65)

rows_s2 = []
for ratio in PI_H_RATIO_VALUES:
    shortage_rate = ratio * HOLDING_COST_RATE
    print(f"\n  pi/h ratio={ratio}  (shortage_rate={shortage_rate:.4f})")
    base_data_s2 = load_data(shortage_cost_rate=shortage_rate,
                              holding_cost_rate=HOLDING_COST_RATE)
    print("[ALNS] solving baseline ...")
    baseline_s2 = solve_baseline(base_data_s2)
    pre_bd_s2, pre_dem_s2 = pre_shock_metrics(base_data_s2, baseline_s2)

    row = run_one(
        sweep_name="pi_h_ratio", param_value=ratio,
        seed=DEFAULT_SHOCK_SEED,
        base_data=base_data_s2, baseline_sol=baseline_s2,
        pre_bd=pre_bd_s2, pre_demand=pre_dem_s2,
        shock_fraction=DEFAULT_SHOCK_FRACTION,
        variant_kwargs=_v1_kwargs(),
    )
    rows_s2.append(row)
    print(f"  post_cost={row['post_total_cost']:.2f}  fill={row['post_fill_rate']:.4f}  "
          f"iters={row['cg_iterations']}  time={row['cg_wall_time_sec']:.1f}s")

df_s2, summary_s2 = save_and_plot(rows_s2, "pi_h_ratio",
                                   "Shortage / holding cost ratio (π/h)")

# %% [code]
# =========================================================
# 8) Sweep 3: stackelberg_min_lateral_qty  (V2)
# =========================================================
print("\n" + "="*65)
print("Sweep 3: stackelberg_min_lateral_qty  [V2, with Stackelberg]")
print("="*65)

base_data_s3 = load_data(shortage_cost_rate=0.05, holding_cost_rate=HOLDING_COST_RATE)
print(f"[Data] stores={len(base_data_s3.stores)}  skus={len(base_data_s3.products)}  "
      f"periods={len(base_data_s3.periods)}")
print("[ALNS] solving baseline ...")
baseline_s3 = solve_baseline(base_data_s3)
pre_bd_s3, pre_dem_s3 = pre_shock_metrics(base_data_s3, baseline_s3)

rows_s3 = []
for val in MIN_LATERAL_QTY_VALUES:
    print(f"\n  min_lateral_qty={val}")
    row = run_one(
        sweep_name="min_lateral_qty", param_value=val,
        seed=DEFAULT_SHOCK_SEED,
        base_data=base_data_s3, baseline_sol=baseline_s3,
        pre_bd=pre_bd_s3, pre_demand=pre_dem_s3,
        shock_fraction=DEFAULT_SHOCK_FRACTION,
        variant_kwargs=_v2_kwargs(val),
    )
    rows_s3.append(row)
    print(f"  post_cost={row['post_total_cost']:.2f}  fill={row['post_fill_rate']:.4f}  "
          f"stack_accepted={row['n_cols_after_stack']}  "
          f"iters={row['cg_iterations']}  time={row['cg_wall_time_sec']:.1f}s")

df_s3, summary_s3 = save_and_plot(rows_s3, "min_lateral_qty",
                                   "Stackelberg min_lateral_qty (MOQ)")

# %% [code]
# =========================================================
# 9) Save combined results + print paths
# =========================================================
all_rows = rows_s1 + rows_s2 + rows_s3
pd.DataFrame(all_rows).to_csv(RESULTS_ROOT / "all_sweeps.csv", index=False)

print("\n" + "="*65)
print("All done. Output files:")
for p in sorted(RESULTS_ROOT.rglob("*")):
    if p.is_file():
        size_kb = p.stat().st_size // 1024
        print(f"  {p.relative_to(RESULTS_ROOT)}  ({size_kb} KB)")
