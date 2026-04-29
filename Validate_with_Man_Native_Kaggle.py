"""
Validate_with_Man_Native_Kaggle.py
====================================
Compare Thesis C vs Man-BFP-TC using **Man et al.'s native data configuration**
on **single-period** scenarios — the natural scope of Man's BFP model.

Key differences from Validate_with_Man_Kaggle.py (harmonized config):
  - window_length = 1  (single period, Man's native scope)
  - demand_shock_sigma = 0.5  (Man's "medium error" from revise_data.py)
  - vehicle_capacity  = 2 × total_period_demand / num_vehicles  (demand-scaled)
  - bk (vehicle fixed cost) = vehicle_capacity / 10  (Man's native, NOT zero)
  - him ∈ [0.02, 0.20] randomly per (store, sku)     (Man's native range)
  - vim ∈ [0.10, 1.50] randomly per (store, sku)     (volume factor)
  - Thesis C holding cost overridden to match Man's him values for fair comparison

This file is used to answer: "When both methods face the same single-period
horizon and the same cost structure, how do they differ?"

Usage:
    python Validate_with_Man_Native_Kaggle.py [--debug] [--scenarios_per_size 5]
                                              [--output_dir Results/man_native]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Re-use everything from the harmonized-config validator
from Validate_with_Man_Kaggle import (
    # data classes & shared infra
    ScenarioSpec,
    ScenarioResult,
    ManBFPTCResult,
    ManScenarioGenerator,
    SharedDataBuilder,
    ManComparisonEngine,
    ManOutputWriter,
    ManRunner,
    # runners we'll subclass
    ManBFPTCRunner,
    ManThesisCRunner,
    _ShockedDataProxy,
    generate_demand_shock,
    # constants
    STORE_LIMITS,
    SCENARIOS_PER_SIZE,
    SKU_LIMIT,
    PERIOD_GRANULARITY,
    WINDOW_LENGTH_PERIODS,
    DEBUG_N_SCENARIOS,
    THESIS_C_TIME_LIMIT,
    HOLDING_COST_RATE,
    SHORTAGE_COST_RATE,
    LT_COST_FLAT,
    ROUTING_ALPHA,
    VEHICLE_COUNT,
    VEHICLE_CAPACITY,
    ALLOW_GNN_FALLBACK,
    DEFAULT_DATA_CSV,
    DEFAULT_CHECKPOINT,
    DIST_REL_PATH,
    MAN_TIME_LIMIT,
    MAN_MIP_GAP,
)

# ======================================================================
# NATIVE-CONFIG CONSTANTS
# ======================================================================

NATIVE_WINDOW_LENGTH:     int   = 1      # single period — Man's native scope
NATIVE_SHOCK_SIGMA:       float = 0.5    # Man's "medium error"
NATIVE_HIM_LO:            float = 0.02
NATIVE_HIM_HI:            float = 0.20
NATIVE_VIM_LO:            float = 0.10
NATIVE_VIM_HI:            float = 1.50
NATIVE_VEHICLE_CAP_SCALE: float = 2.0   # capacity = scale × total_demand / num_vehicles
DEFAULT_OUTPUT_DIR_NATIVE = "Results/man_native_single_period"


# ======================================================================
# NATIVE-CONFIG PARAMETER BUILDER
# ======================================================================

def build_native_params(
    stores: List[str],
    skus: List[str],
    df_slice: pd.DataFrame,
    vehicle_count: int,
    scenario_seed: int,
) -> Dict[str, Any]:
    """Generate Man et al.'s native random cost/capacity parameters for one scenario.

    Returns a dict with:
      him_native   – {(store, sku): holding_cost}   ∈ [0.02, 0.20]
      vim_native   – {(store, sku): volume_factor}  ∈ [0.10, 1.50]
      vehicle_cap  – scalar float (demand-scaled)
      bk_native    – list[float] = vehicle_cap / 10 per vehicle
    """
    rng = random.Random(scenario_seed ^ 0xC0FFEE)

    # him and vim: random per (store, sku)
    him_native: Dict[Tuple[str, str], float] = {}
    vim_native: Dict[Tuple[str, str], float] = {}
    for store in stores:
        for sku in skus:
            him_native[(store, sku)] = round(rng.uniform(NATIVE_HIM_LO, NATIVE_HIM_HI), 3)
            vim_native[(store, sku)] = round(rng.uniform(NATIVE_VIM_LO, NATIVE_VIM_HI), 3)

    # vehicle capacity = 2 × total single-period demand / num_vehicles
    # Use the median period to avoid outliers
    periods = sorted(df_slice["period_date"].unique())
    mid_period = periods[len(periods) // 2]
    period_rows = df_slice[
        df_slice["store"].isin(stores) &
        df_slice["sku"].isin(skus) &
        (df_slice["period_date"] == mid_period)
    ]
    total_demand = float(period_rows["sale_qty"].sum())
    if total_demand <= 0:
        total_demand = 100.0 * len(stores) * len(skus)
    vehicle_cap = max(
        200.0,
        NATIVE_VEHICLE_CAP_SCALE * total_demand / max(vehicle_count, 1),
    )
    bk_native = [round(vehicle_cap / 10.0, 2) for _ in range(vehicle_count)]

    return {
        "him_native":  him_native,
        "vim_native":  vim_native,
        "vehicle_cap": vehicle_cap,
        "bk_native":   bk_native,
    }


# ======================================================================
# NATIVE-CONFIG Man-BFP-TC RUNNER
# ======================================================================

class ManNativeBFPTCRunner(ManBFPTCRunner):
    """ManBFPTCRunner with Man et al.'s native cost/capacity parameters.

    Overrides _build_static_inputs to inject:
      - demand-scaled vehicle capacity
      - bk = capacity / 10 (vehicle fixed cost)
      - him per (store, sku) ∈ [0.02, 0.20]
      - vim per (store, sku) ∈ [0.10, 1.50]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_params: Optional[Dict] = None  # set per scenario

    def set_native_params(self, params: Optional[Dict]):
        self._native_params = params

    def _build_static_inputs(self, stores, skus, dist_dict, df_slice):
        base = super()._build_static_inputs(stores, skus, dist_dict, df_slice)

        p = self._native_params
        if p is None:
            return base

        N, M = base["N"], base["M"]
        store_to_idx = base["store_to_idx"]
        sku_to_idx   = base["sku_to_idx"]

        # Override vehicle capacity and fixed cost
        Qk = [p["vehicle_cap"] for _ in range(self.vehicle_count)]
        bk = list(p["bk_native"])

        # Override holding cost matrix: index 0 = DC (0.0), 1..N = stores
        him = [[0.0 for _ in range(M)]]   # DC holding = 0
        for store in stores:
            row = []
            for sku in skus:
                row.append(p["him_native"].get((store, sku), HOLDING_COST_RATE))
            him.append(row)

        # Override volume factor matrix: N rows × M cols
        vim = []
        for store in stores:
            row = []
            for sku in skus:
                row.append(p["vim_native"].get((store, sku), 1.0))
            vim.append(row)

        base["Qk"] = Qk
        base["bk"] = bk
        base["him"] = him
        base["vim"] = vim
        return base


# ======================================================================
# NATIVE-CONFIG Thesis C RUNNER
# ======================================================================

class ManNativeThesisCRunner(ManThesisCRunner):
    """ManThesisCRunner with Man et al.'s native holding costs and vehicle capacity.

    Overrides _build_irp_data to:
      - Apply per-(store,sku) holding costs matching ManNativeBFPTCRunner's him
      - Override vehicle capacity to demand-scaled value
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_params: Optional[Dict] = None

    def set_native_params(self, params: Optional[Dict]):
        self._native_params = params

    def _build_irp_data(self, scenario, df_slice, dist_dict):
        data = super()._build_irp_data(scenario, df_slice, dist_dict)

        p = self._native_params
        if p is None:
            return data

        # Override per-store-product holding cost to match Man's native him
        for s in data.stores:
            for prod in data.products:
                cost = p["him_native"].get((str(s), str(prod)), HOLDING_COST_RATE)
                data.holding_cost_store[(s, prod)] = float(cost)

        # Override vehicle capacity
        data.vehicle_capacity = float(p["vehicle_cap"])

        return data


# ======================================================================
# NATIVE OUTPUT WRITER  (separate file prefix)
# ======================================================================

class ManNativeOutputWriter(ManOutputWriter):
    """Same as ManOutputWriter but uses 'validate_man_native_' prefix."""

    def _save(self, df: pd.DataFrame, filename: str):
        native_filename = filename.replace("validate_man_vs_C_", "validate_man_native_")
        return super()._save(df, native_filename)


# ======================================================================
# NATIVE VALIDATION ORCHESTRATOR
# ======================================================================

class ManNativeValidationOrchestrator:
    """Run Thesis C vs Man-BFP-TC on single-period scenarios with Man's native config."""

    def __init__(
        self,
        data_csv: str,
        checkpoint_path: str,
        output_dir: str,
        dist_path: Optional[str] = None,
        repo_root: Optional[str] = None,
        store_limits: Optional[List[int]] = None,
        scenarios_per_size: int = SCENARIOS_PER_SIZE,
        demand_shock_sigma: float = NATIVE_SHOCK_SIGMA,
        debug: bool = False,
        seed: int = 42,
    ):
        self.data_csv           = data_csv
        self.output_dir         = output_dir
        self.demand_shock_sigma = float(demand_shock_sigma)
        self.debug              = debug
        self.seed               = seed

        _scenarios_per_size = DEBUG_N_SCENARIOS if debug else scenarios_per_size
        _store_limits       = store_limits or STORE_LIMITS

        # Single-period generator (ManScenarioGenerator relaxes n_periods >= 1)
        self.scenario_gen = ManScenarioGenerator(
            data_csv=data_csv,
            store_limits=_store_limits,
            scenarios_per_size=_scenarios_per_size,
            sku_limit=SKU_LIMIT,
            period_granularity=PERIOD_GRANULARITY,
            window_length=NATIVE_WINDOW_LENGTH,
            base_seed=seed,
        )
        self.data_builder = SharedDataBuilder(
            data_csv=data_csv,
            dist_path=dist_path,
            period_granularity=PERIOD_GRANULARITY,
        )
        self.thesis_runner = ManNativeThesisCRunner(
            checkpoint_path=checkpoint_path,
            repo_root=repo_root,
            cg_iterations=1,
            time_limit=THESIS_C_TIME_LIMIT,
        )
        self.man_runner = ManNativeBFPTCRunner(
            time_limit=MAN_TIME_LIMIT,
            mip_gap=MAN_MIP_GAP,
            vehicle_count=VEHICLE_COUNT,
            vehicle_capacity=VEHICLE_CAPACITY,   # will be overridden per scenario
            holding_cost_rate=HOLDING_COST_RATE,
            shortage_cost_rate=SHORTAGE_COST_RATE,
        )
        self.writer = ManNativeOutputWriter(output_dir)

    def run(self) -> None:
        t_total = time.time()

        ckpt_ok = self.thesis_runner.check_checkpoint()
        if not ckpt_ok and not ALLOW_GNN_FALLBACK:
            raise FileNotFoundError(f"GNN checkpoint not found.")

        scenarios = self.scenario_gen.generate()
        self.writer.write_scenario_manifest(scenarios)

        config = {
            "data_csv":               self.data_csv,
            "checkpoint_valid":       ckpt_ok,
            "n_scenarios":            len(scenarios),
            "window_length_periods":  NATIVE_WINDOW_LENGTH,
            "demand_shock_sigma":     self.demand_shock_sigma,
            "native_him_range":       [NATIVE_HIM_LO, NATIVE_HIM_HI],
            "native_vim_range":       [NATIVE_VIM_LO, NATIVE_VIM_HI],
            "native_vehicle_cap_scale": NATIVE_VEHICLE_CAP_SCALE,
            "bk_vehicle_fixed_cost":  "vehicle_capacity / 10  (Man's native)",
            "shortage_cost_rate":     SHORTAGE_COST_RATE,
            "vehicle_count":          VEHICLE_COUNT,
            "debug":                  self.debug,
        }
        self.writer.write_config(config)

        all_results:  List[Dict] = []
        all_lt_moves: List[Dict] = []
        gap_rows:     List[Dict] = []
        partial_path = Path(self.output_dir) / "validate_man_native_per_scenario.csv"

        print(f"\n{'='*70}")
        print(f"Validate_with_Man_Native_Kaggle.py — {len(scenarios)} scenarios")
        print(f"Single-period | σ={self.demand_shock_sigma} | Man native config")
        print(f"{'='*70}\n")

        for idx, scenario in enumerate(scenarios, start=1):
            print(f"\n[{idx:02d}/{len(scenarios)}] {scenario.scenario_id}  "
                  f"stores={scenario.store_limit}  skus={scenario.sku_limit}  "
                  f"period={scenario.start_date}")

            df_slice  = self.data_builder.slice_for_scenario(scenario)
            dist_dict, dist_src = self.data_builder.get_distances(scenario.selected_stores)
            if idx == 1:
                print(f"  Distance source: {dist_src}")

            if df_slice.empty:
                print(f"  WARNING: empty data slice; skipping.")
                continue

            # ── Native params (shared between both methods this scenario) ──
            native_p = build_native_params(
                scenario.selected_stores, scenario.selected_skus,
                df_slice, VEHICLE_COUNT, scenario.random_seed,
            )
            print(f"  [NativeConfig]  vehicle_cap={native_p['vehicle_cap']:.0f}  "
                  f"bk={native_p['bk_native'][0]:.1f}  "
                  f"mean_him={sum(native_p['him_native'].values())/max(len(native_p['him_native']),1):.3f}  "
                  f"mean_vim={sum(native_p['vim_native'].values())/max(len(native_p['vim_native']),1):.3f}")

            # ── Demand shock ──
            shock = generate_demand_shock(scenario, scenario.n_periods,
                                          self.demand_shock_sigma)
            if shock:
                mean_eps = sum(shock.values()) / len(shock)
                print(f"  [Shock]  σ={self.demand_shock_sigma}  mean_ε={mean_eps:.3f}")

            # ── Thesis C ──
            print(f"  [Thesis C]  running...")
            self.thesis_runner.set_native_params(native_p)
            self.thesis_runner.set_shock(shock)
            try:
                c_result = self.thesis_runner.run_scenario(scenario, df_slice, dist_dict)
            finally:
                self.thesis_runner.set_shock(None)
                self.thesis_runner.set_native_params(None)
            print(f"  [Thesis C]  status={c_result.status}  "
                  f"cost={c_result.total_cost:.2f}  "
                  f"shortage={c_result.shortage_qty:.0f}  "
                  f"runtime={c_result.runtime_seconds:.1f}s")

            # ── Man-BFP-TC ──
            print(f"  [Man-BFP-TC]  running...")
            self.man_runner.set_native_params(native_p)
            try:
                man_result, lt_moves = self.man_runner.run_scenario(
                    scenario, df_slice, dist_dict, demand_shock=shock or None
                )
            finally:
                self.man_runner.set_native_params(None)
            print(f"  [Man-BFP-TC]  status={man_result.status}  "
                  f"cost={man_result.total_cost:.2f}  "
                  f"shortage={man_result.shortage_qty:.0f}  "
                  f"mip_gap={man_result.mip_gap:.4f}  "
                  f"runtime={man_result.runtime_seconds:.1f}s")

            # ── Annotate & collect ──
            for result in [c_result, man_result]:
                d = result.to_dict()
                d["size_label"]         = scenario.size_label
                d["store_limit"]        = scenario.store_limit
                d["sku_limit"]          = scenario.sku_limit
                d["n_periods"]          = scenario.n_periods
                d["start_date"]         = scenario.start_date
                d["end_date"]           = scenario.end_date
                d["random_seed"]        = scenario.random_seed
                d["native_vehicle_cap"] = native_p["vehicle_cap"]
                d["native_bk"]          = native_p["bk_native"][0]
                all_results.append(d)

            all_lt_moves.extend(lt_moves)

            gap = ManComparisonEngine.compute_per_scenario_gaps(c_result, man_result)
            gap["size_label"] = scenario.size_label
            gap_rows.append(gap)

            if c_result.success and man_result.success:
                print(f"  [Gap]  cost_gap={gap.get('cost_gap_pct',float('nan')):.2f}%  "
                      f"→ {gap.get('interpretation','')}")

            if all_results:
                pd.DataFrame(all_results).to_csv(partial_path, index=False)

        # ── Final outputs ──
        print(f"\n{'='*70}\nWriting final outputs...")
        results_df = pd.DataFrame(all_results)
        summary_df = ManComparisonEngine.compute_summary(pd.DataFrame(gap_rows), results_df)

        self.writer.write_per_scenario(all_results)
        self.writer.write_summary(summary_df)
        self.writer.write_cost_breakdown(all_results)
        self.writer.write_gaps(gap_rows)
        self.writer.write_failures(all_results)

        total_runtime = time.time() - t_total
        print(f"\n{'='*70}")
        print(f"COMPLETE  total_runtime={total_runtime:.0f}s  scenarios={len(scenarios)}")
        print(f"{'='*70}")

        if not summary_df.empty:
            print(summary_df[[
                "method", "size_label", "n_scenarios", "n_success",
                "mean_runtime", "mean_total_cost", "mean_service_level",
            ]].to_string(index=False))

        if not results_df.empty:
            valid_gaps = [g for g in gap_rows
                          if not math.isnan(g.get("cost_gap_pct", float("nan")))]
            c_sl  = results_df[results_df["method"]=="Thesis_C_ALNS_CG_GNN"]["service_level"].mean()
            m_sl  = results_df[results_df["method"]=="Man-BFP-TC"]["service_level"].mean()
            print(f"\n  Mean service level — Thesis C: {c_sl:.4f}  Man-BFP-TC: {m_sl:.4f}")
            if valid_gaps:
                mean_gap = sum(g["cost_gap_pct"] for g in valid_gaps) / len(valid_gaps)
                print(f"  Mean cost gap (C vs Man): {mean_gap:+.2f}%")
        print(f"  Outputs: {self.output_dir}")


# ======================================================================
# CLI
# ======================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Thesis C vs Man-BFP-TC — Man native config, single-period"
    )
    p.add_argument("--data_csv",    default=DEFAULT_DATA_CSV)
    p.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT)
    p.add_argument("--output_dir",  default=DEFAULT_OUTPUT_DIR_NATIVE)
    p.add_argument("--dist_path",   default=None)
    p.add_argument("--repo_root",   default=None)
    p.add_argument("--store_limits",  nargs="+", type=int, default=STORE_LIMITS)
    p.add_argument("--scenarios_per_size", type=int, default=SCENARIOS_PER_SIZE)
    p.add_argument("--demand_shock_sigma", type=float, default=NATIVE_SHOCK_SIGMA,
                   help=f"Demand shock σ (default: {NATIVE_SHOCK_SIGMA} = Man's medium error)")
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    return p


def main() -> None:
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    data_csv   = os.environ.get("VALIDATE_DATA_CSV",        args.data_csv)
    checkpoint = os.environ.get("CHECKPOINT_PATH",          args.checkpoint)
    output_dir = os.environ.get("VALIDATE_MAN_NATIVE_DIR",  args.output_dir)
    repo_root  = args.repo_root

    if repo_root is None:
        for candidate in ["/kaggle/working/Thesis-Work", "/kaggle/working/repo",
                          str(Path(__file__).parent)]:
            if Path(candidate).exists():
                repo_root = candidate
                break

    if not Path(checkpoint).exists() and repo_root:
        for candidate in [
            f"{repo_root}/Results/gnn_training/best_valid_prauc.pt",
            "/kaggle/working/Results/gnn_training/best_valid_prauc.pt",
        ]:
            if candidate and Path(candidate).exists():
                checkpoint = candidate
                break

    dist_path = args.dist_path
    if dist_path is None and repo_root:
        dp = Path(repo_root) / DIST_REL_PATH
        if dp.exists():
            dist_path = str(dp)

    print(f"\n{'='*70}")
    print("Validate_with_Man_Native_Kaggle.py — single-period, Man native config")
    print(f"{'='*70}")
    print(f"  window_length:      {NATIVE_WINDOW_LENGTH}  (single period)")
    print(f"  demand_shock_sigma: {args.demand_shock_sigma}  (Man medium error)")
    print(f"  vehicle_cap:        demand-scaled  (2 × demand / num_vehicles)")
    print(f"  bk_fixed_cost:      vehicle_cap / 10  (Man's native)")
    print(f"  him_range:          [{NATIVE_HIM_LO}, {NATIVE_HIM_HI}]")
    print(f"  vim_range:          [{NATIVE_VIM_LO}, {NATIVE_VIM_HI}]")
    print(f"  store_limits:       {args.store_limits}")
    print(f"  scenarios_per_size: {args.scenarios_per_size}")
    print(f"  debug:              {args.debug}")
    print(f"{'='*70}\n")

    ManNativeValidationOrchestrator(
        data_csv=data_csv,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        dist_path=dist_path,
        repo_root=repo_root,
        store_limits=args.store_limits,
        scenarios_per_size=args.scenarios_per_size,
        demand_shock_sigma=args.demand_shock_sigma,
        debug=args.debug,
        seed=args.seed,
    ).run()


if __name__ == "__main__":
    main()
