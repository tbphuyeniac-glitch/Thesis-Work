"""
Validate_with_Man_Kaggle.py
============================
Compare Thesis Variant C (ALNS + Column Generation + GNN-guided selection)
against the Man et al. 2025 BFP-TC benchmark (deterministic two-stage,
single-period BFP rolled period-by-period over the same scenario window).

Run independently from `Validate_with_Achamrah_Kaggle.py` — the two
benchmarks are NEVER executed in the same scenario loop. This script
exists so the user can run "C vs Man-BFP-TC" without paying the cost
of also running Achamrah on the same data.

Usage:
    python Validate_with_Man_Kaggle.py [--data_csv /path/to/test data.csv]
                                       [--checkpoint /path/to/best_valid_prauc.pt]
                                       [--output_dir /kaggle/working/Results/man_validation]
                                       [--n_scenarios 30]
                                       [--debug]

Design:
    - 30 scenarios = 3 store sizes × 10 scenarios each (same generator
      and same random seed as the Achamrah validator, so scenarios are
      directly comparable across the two benchmarks if needed offline).
    - Thesis C: identical pipeline / checkpoint as the Achamrah validator.
    - Man-BFP-TC: a rolling-horizon driver around `solve_man_bfp_tc`
      (see `man_bfp_tc_benchmark.ManBFPTCRunner`).
    - Outputs written under a `validate_man_vs_C_*` prefix to keep them
      separate from the Achamrah outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# -- Re-use shared infrastructure from the Achamrah validator without -------- #
# -- modifying that file. Import-only; we do NOT trigger its `main()`.        #
from Validate_with_Achamrah_Kaggle import (
    # data classes
    ScenarioSpec,
    ScenarioResult,
    # pipeline pieces we re-use as-is
    ScenarioGenerator,
    SharedDataBuilder,
    ThesisCRunner,
    # constants we want to keep aligned with the Achamrah validator
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
)

# -- Man-BFP-TC adapter ------------------------------------------------------ #
from man_bfp_tc_benchmark import ManBFPTCRunner, ManBFPTCResult

# -- Man-Joint-TC (Oracle) and Man-TSRFP-TC (C&CG robust) adapters ----------- #
from tsrfp_runner import JointTCRunner, JointTCResult, TSRFPTCRunner, TSRFPTCResult


# ======================================================================
# DEMAND-SHOCKED DATA PROXY  (used by ManThesisCRunner only)
# ======================================================================

class _ShockedDataProxy:
    """Wraps an IRPData object and intercepts `realized_demand` assignment.

    When ThesisCRunner does `data.realized_demand = dict(data.demand)` inside
    run_scenario(), this proxy applies the per-(store,sku,period) shock
    multipliers before storing the value, so Stage-2 (CG/LT) evaluates cost
    against shocked demand while Stage-1 routing still planned with forecast.

    All other attribute reads/writes are forwarded transparently to the wrapped
    IRPData, so duck-typed IRP solver code is unaffected.
    """

    def __init__(self, wrapped, shock: Dict[Tuple[str, str, int], float]):
        object.__setattr__(self, "_w", wrapped)
        object.__setattr__(self, "_shock", shock)
        object.__setattr__(self, "_realized", None)

    @property
    def realized_demand(self):
        return object.__getattribute__(self, "_realized")

    @realized_demand.setter
    def realized_demand(self, value: Dict):
        shock = object.__getattribute__(self, "_shock")
        object.__setattr__(
            self, "_realized",
            {k: v * shock.get(k, 1.0) for k, v in value.items()} if shock else value,
        )

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_w"), name)

    def __setattr__(self, name: str, value):
        # Properties on the proxy class (e.g. `realized_demand`) must
        # dispatch to their setter, otherwise the unconditional else-branch
        # below bypasses the shock multiplier and writes the unshocked value
        # straight onto the wrapped IRPData. (This was a silent bug — every
        # prior shock run actually evaluated Thesis C on forecast demand.)
        cls_attr = getattr(type(self), name, None)
        if isinstance(cls_attr, property):
            if cls_attr.fset is None:
                raise AttributeError(f"can't set attribute {name!r}")
            cls_attr.fset(self, value)
            return
        if name in ("_w", "_shock", "_realized"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_w"), name, value)


# ======================================================================
# THESIS-C RUNNER WITH SHOCK SUPPORT  (Man-validator-only subclass)
# ======================================================================

class ManThesisCRunner(ThesisCRunner):
    """Thin subclass of ThesisCRunner that supports per-scenario demand shock.

    Adds two things on top of ThesisCRunner:
      1. `_build_irp_data` is overridden to apply a per-scenario demand shock
         via `_ShockedDataProxy`.
      2. `run_scenario` runs an additional "no-routing trial" (iter0) by
         calling JointTCRunner with vehicle_count=0 — i.e. solving the Stage 2
         LP (LT-only redistribution) given zero DC deliveries. If that beats
         the standard ALNS+CG+LT result, we adopt it. This closes the gap to
         the Oracle in scenarios where ALNS over-routes.

    Usage:
        runner.set_shock(shock_dict)
        result = runner.run_scenario(scenario, df_slice, dist_dict)
        runner.set_shock(None)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_shock: Optional[Dict] = None
        self._no_routing_trial_enabled: bool = (
            os.environ.get("THESIS_C_NO_ROUTING_TRIAL", "1") == "1"
        )
        self._disable_phase1_mip: bool = (
            os.environ.get("THESIS_C_DISABLE_PHASE1_MIP", "0") == "1"
        )

    def set_shock(self, shock: Optional[Dict]):
        self._active_shock = shock or None

    def _build_irp_data(self, scenario, df_slice, dist_dict):
        data = super()._build_irp_data(scenario, df_slice, dist_dict)
        shock = self._active_shock
        if shock:
            return _ShockedDataProxy(data, shock)
        return data

    def _no_routing_trial(
        self,
        scenario: "ScenarioSpec",
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
    ) -> Tuple[Optional["JointTCResult"], List[Dict[str, Any]]]:
        """Solve Stage 2 LP with zero DC routing — returns LT-only optimum.

        Implemented by reusing JointTCRunner with vehicle_count=0. The
        Stage 1 routing structure becomes empty, so the joint MIP collapses
        to the Stage 2 LP that redistributes existing store/DC inventory via
        LT to meet realized demand. This is the "iter0" trial.
        """
        try:
            helper = JointTCRunner(
                time_limit=180,
                mip_gap=0.02,
                threads=4,
                vehicle_count=0,        # KEY: forbid DC routing
                vehicle_capacity=0.0,
                holding_cost_rate=HOLDING_COST_RATE,
                shortage_cost_rate=SHORTAGE_COST_RATE,
                lt_cost_multiplier=1.2,
            )
            return helper.run_scenario(
                scenario, df_slice, dist_dict,
                demand_shock=self._active_shock or None,
            )
        except Exception as exc:
            print(f"  [ThesisC iter0] no-routing trial errored: {exc}")
            return None, []

    def run_scenario(
        self,
        scenario: "ScenarioSpec",
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
    ) -> ScenarioResult:
        import irp_gurobi_converted as _irp

        # ---- Optional: disable Phase 1 (Gurobi MIP exact routing) ----
        original_mip_solve = None
        if self._disable_phase1_mip:
            original_mip_solve = _irp.AchamrahFullIRPTModel.solve

            def _stub_solve(self, *args, **kwargs):
                return None
            _irp.AchamrahFullIRPTModel.solve = _stub_solve
            print(f"  [ThesisC]   Phase 1 (Gurobi MIP exact) DISABLED — ALNS only")

        # ---- Shock-propagation patch: auto-build post_shock_inventory ----
        # _ShockedDataProxy alone is insufficient — it intercepts
        # data.realized_demand, but cost evaluation reads
        # data.post_shock_inventory / data.post_shock_shortage which are
        # empty dicts unless build_post_shock_inventory_state(data, sol) is
        # called explicitly. The training pipeline calls it; the benchmark
        # validator does not, so shock never reaches the Phase 3 shortage
        # gate. Monkey-patch build_realized_operating_cost_breakdown so it
        # auto-populates the post-shock state on first call.
        original_breakdown = _irp.build_realized_operating_cost_breakdown
        shock_active = bool(self._active_shock)

        def _patched_breakdown(data, dc_solution, lt_plan_df=None):
            if shock_active:
                rd = getattr(data, "realized_demand", None)
                psi = getattr(data, "post_shock_inventory", None) or {}
                if rd and not psi:
                    _irp.build_post_shock_inventory_state(data, dc_solution)
            return original_breakdown(data, dc_solution, lt_plan_df=lt_plan_df)

        _irp.build_realized_operating_cost_breakdown = _patched_breakdown

        try:
            # ---- Standard ALNS + CG + LT path ----
            result_A = super().run_scenario(scenario, df_slice, dist_dict)
        finally:
            _irp.build_realized_operating_cost_breakdown = original_breakdown
            if original_mip_solve is not None:
                _irp.AchamrahFullIRPTModel.solve = original_mip_solve

        if not self._no_routing_trial_enabled:
            return result_A
        if not getattr(result_A, "success", False):
            return result_A

        # ---- iter0: no-routing trial (LT-only) ----
        result_B, lt_moves_B = self._no_routing_trial(scenario, df_slice, dist_dict)

        if (result_B is None or not getattr(result_B, "success", False)
                or not (result_B.total_cost < result_A.total_cost - 1e-6)):
            print(f"  [ThesisC iter0]   no-routing={getattr(result_B, 'total_cost', float('nan')):.2f}"
                  f"  ALNS+CG={result_A.total_cost:.2f}  → keep ALNS")
            return result_A

        print(f"  [ThesisC iter0]   no-routing={result_B.total_cost:.2f} "
              f"< ALNS+CG={result_A.total_cost:.2f}  → adopt iter0 (saved "
              f"{result_A.total_cost - result_B.total_cost:.2f})")

        # Overlay iter0 numbers onto the ScenarioResult shell. Keep status,
        # method, runtime, etc. from result_A; replace the cost/quantity
        # fields with the iter0 winners.
        result_A.total_cost          = float(result_B.total_cost)
        result_A.routing_cost        = 0.0
        result_A.holding_cost        = float(result_B.holding_cost)
        result_A.transshipment_cost  = float(result_B.transshipment_cost)
        result_A.shortage_cost       = float(result_B.shortage_cost)
        result_A.shortage_qty        = float(result_B.shortage_qty)
        result_A.service_level       = float(result_B.service_level)
        result_A.lt_total_qty        = float(result_B.lt_total_qty)
        result_A.n_lt_moves          = int(result_B.n_lt_moves)
        result_A.n_routes            = 0
        return result_A


# ======================================================================
# SCENARIO GENERATOR — relaxed minimum-period guard
# ======================================================================

class ManScenarioGenerator(ScenarioGenerator):
    """Same as ScenarioGenerator but allows n_periods >= 1 (not >= 2).

    Required so --window_length 1 doesn't produce an empty scenario list.
    The parent hard-codes `if n_periods < 2: continue`; this subclass
    replaces that check with `if n_periods < 1: continue`.
    """

    MIN_PERIODS: int = 1

    def generate(self) -> List["ScenarioSpec"]:
        df         = self._load_and_aggregate()
        top_skus   = self._select_top_skus(df, self.sku_limit)
        all_stores = sorted(df["store"].unique())
        windows    = self._get_windows(df)

        if not windows:
            raise ValueError("[ScenarioGen] No valid period windows found in data.")

        scenarios: list = []
        size_labels = {5: "small", 8: "medium", 10: "large"}

        for store_limit in self.store_limits:
            if store_limit > len(all_stores):
                print(f"[ScenarioGen] WARNING: store_limit={store_limit} > available stores "
                      f"({len(all_stores)}); using {len(all_stores)}")
                store_limit = len(all_stores)

            size_label = size_labels.get(store_limit, f"size{store_limit}")

            for scenario_idx in range(self.scenarios_per_size):
                rng  = random.Random(self.base_seed + store_limit * 1000 + scenario_idx)
                seed = self.base_seed + store_limit * 1000 + scenario_idx

                selected_stores = sorted(
                    rng.sample(list(all_stores), k=min(store_limit, len(all_stores)))
                )
                selected_skus = top_skus[: self.sku_limit]

                win_idx             = scenario_idx % len(windows)
                start_date, end_date = windows[win_idx]

                mask = (
                    df["store"].isin(selected_stores)
                    & df["sku"].isin(selected_skus)
                    & (df["period_date"].astype(str) >= start_date)
                    & (df["period_date"].astype(str) <= end_date)
                )
                n_periods = df[mask]["period_date"].nunique()

                if n_periods < self.MIN_PERIODS:
                    print(
                        f"[ScenarioGen] WARNING: scenario {scenario_idx} for size "
                        f"{store_limit} has only {n_periods} periods; skipping."
                    )
                    continue

                scenario_id = f"{size_label}_s{store_limit:02d}_sc{scenario_idx:02d}"

                scenarios.append(
                    ScenarioSpec(
                        scenario_id=scenario_id,
                        store_limit=store_limit,
                        sku_limit=len(selected_skus),
                        selected_stores=selected_stores,
                        selected_skus=selected_skus,
                        start_date=start_date,
                        end_date=end_date,
                        period_granularity=self.period_granularity,
                        n_periods=n_periods,
                        random_seed=seed,
                        size_label=size_label,
                        window_index=win_idx,
                    )
                )

        print(
            f"[ScenarioGen] Generated {len(scenarios)} scenarios "
            f"({self.scenarios_per_size} per store size × {len(self.store_limits)} sizes)"
        )
        return scenarios


# ======================================================================
# CONFIGURATION
# ======================================================================

MAN_TIME_LIMIT:        int   = int(os.environ.get("MAN_TIME_LIMIT",        "1200"))
MAN_MIP_GAP:           float = float(os.environ.get("MAN_MIP_GAP",           "0.01"))
DEMAND_SHOCK_SIGMA:    float = float(os.environ.get("DEMAND_SHOCK_SIGMA",    "0.20"))

DEFAULT_OUTPUT_DIR = "Results/man_validation"


# ======================================================================
# DEMAND SHOCK GENERATOR
# ======================================================================

def generate_demand_shock(
    scenario: "ScenarioSpec",
    n_periods: int,
    sigma: float,
) -> Dict[Tuple[str, str, int], float]:
    """Return {(store, sku, period_int): multiplier} where period_int is 1-based.

    Each multiplier is drawn from N(1, sigma^2), clipped to [0.5, 2.0].
    sigma=0 returns {} (no shock — deterministic baseline).
    The RNG is seeded deterministically so the same scenario always gets the
    same shock regardless of run order.
    """
    if sigma <= 0.0:
        return {}
    rng = random.Random(scenario.random_seed ^ 0xF00DCAFE)
    shock: Dict[Tuple[str, str, int], float] = {}
    for t in range(1, n_periods + 1):
        for store in scenario.selected_stores:
            for sku in scenario.selected_skus:
                eps = rng.gauss(1.0, sigma)
                eps = max(0.5, min(2.0, eps))
                shock[(store, sku, t)] = eps
    return shock


# ======================================================================
# MAN RUNNER (thin orchestrator-side wrapper around ManBFPTCRunner)
# ======================================================================

class ManRunner:
    """Mirror the surface of `AchamrahRunner.run_scenario` but call into
    `ManBFPTCRunner`. Returns a `ManBFPTCResult` (already aligned with the
    `ScenarioResult` fields used by the orchestrator)."""

    METHOD_NAME = "Man-BFP-TC"

    def __init__(
        self,
        time_limit: int = MAN_TIME_LIMIT,
        mip_gap: float = MAN_MIP_GAP,
        threads: int = 4,
    ):
        self.time_limit = int(time_limit)
        self.mip_gap    = float(mip_gap)
        self.threads    = int(threads)

        self._runner = ManBFPTCRunner(
            time_limit=self.time_limit,
            mip_gap=self.mip_gap,
            threads=self.threads,
            # Match the shared cost / fleet calibration so total_cost is
            # comparable to Thesis C.
            vehicle_count=VEHICLE_COUNT,
            vehicle_capacity=VEHICLE_CAPACITY,
            holding_cost_rate=HOLDING_COST_RATE,
            shortage_cost_rate=SHORTAGE_COST_RATE,
        )
        self.source_name = self.METHOD_NAME

    def run_scenario(
        self,
        scenario: ScenarioSpec,
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
        demand_shock: Optional[Dict[Tuple[str, str, int], float]] = None,
    ) -> Tuple[ManBFPTCResult, List[Dict[str, Any]]]:
        return self._runner.run_scenario(scenario, df_slice, dist_dict,
                                         demand_shock=demand_shock)


# ======================================================================
# JOINT-TC RUNNER WRAPPER
# ======================================================================

class JointRunner:
    """Thin wrapper around JointTCRunner matching the ManRunner interface."""

    METHOD_NAME = "Man-Joint-TC"

    def __init__(
        self,
        time_limit: int = MAN_TIME_LIMIT,
        mip_gap: float = MAN_MIP_GAP,
        threads: int = 4,
    ):
        # Bound dc_capacity exactly to total initial DC stock (5 SKUs × 10k =
        # 50k). This forces replenishment rm to 0 across the rolling horizon.
        # Without this bound, rm has no cost penalty in the objective, so
        # Gurobi picks rm = max(allowed) = dc_capacity − dc_init_stock, which
        # leaves later periods with a depleted-per-SKU DC and no replenishment
        # headroom → infeasible. With rm=0 forced, the model becomes a pure
        # closed-system rolling-horizon problem (matches Man et al. 2025).
        self._runner = JointTCRunner(
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            vehicle_count=VEHICLE_COUNT,
            vehicle_capacity=VEHICLE_CAPACITY,
            holding_cost_rate=HOLDING_COST_RATE,
            shortage_cost_rate=SHORTAGE_COST_RATE,
            dc_capacity=55000.0,                     # 5 SKUs × 10k init + 5k buffer
            dc_initial_stock_per_product=10000.0,
        )
        self.source_name = self.METHOD_NAME

    def run_scenario(
        self,
        scenario: ScenarioSpec,
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
        demand_shock: Optional[Dict[Tuple[str, str, int], float]] = None,
    ) -> Tuple[JointTCResult, List[Dict[str, Any]]]:
        return self._runner.run_scenario(scenario, df_slice, dist_dict,
                                         demand_shock=demand_shock)


class TSRFPRunner:
    """Thin wrapper around TSRFPTCRunner matching the ManRunner interface."""

    METHOD_NAME = "Man-TSRFP-TC"

    def __init__(
        self,
        time_limit: int = MAN_TIME_LIMIT,
        mip_gap: float = MAN_MIP_GAP,
        threads: int = 4,
    ):
        self._runner = TSRFPTCRunner(
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            vehicle_count=VEHICLE_COUNT,
            vehicle_capacity=VEHICLE_CAPACITY,
            holding_cost_rate=HOLDING_COST_RATE,
            shortage_cost_rate=SHORTAGE_COST_RATE,
        )
        self.source_name = self.METHOD_NAME

    def run_scenario(
        self,
        scenario: ScenarioSpec,
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
        demand_shock: Optional[Dict[Tuple[str, str, int], float]] = None,
    ) -> Tuple[TSRFPTCResult, List[Dict[str, Any]]]:
        return self._runner.run_scenario(scenario, df_slice, dist_dict,
                                         demand_shock=demand_shock)


# ======================================================================
# COMPARISON ENGINE  (C vs Man)
# ======================================================================

class ManComparisonEngine:
    """Per-scenario gap and aggregate statistics for the C-vs-Man comparison."""

    @staticmethod
    def compute_per_scenario_gaps(
        thesis_result: ScenarioResult,
        man_result:    ManBFPTCResult,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "scenario_id":      thesis_result.scenario_id,
            "thesis_success":   thesis_result.success,
            "man_success":      man_result.success,
            "thesis_status":    thesis_result.status,
            "man_status":       man_result.status,
        }
        if thesis_result.success and man_result.success:
            tc = thesis_result.total_cost
            mc = man_result.total_cost
            gap = (tc - mc) / max(abs(mc), 1e-9) * 100.0
            row.update({
                "thesis_total_cost":         tc,
                "man_total_cost":            mc,
                "cost_gap_pct":              round(gap, 4),       # +ve = C more expensive
                "cost_gap_abs":              round(tc - mc, 4),
                "runtime_C":                 thesis_result.runtime_seconds,
                "runtime_Man":               man_result.runtime_seconds,
                "runtime_ratio_C_vs_Man":    round(
                    thesis_result.runtime_seconds / max(man_result.runtime_seconds, 1e-3), 4
                ),
                "thesis_service_level":      thesis_result.service_level,
                "man_service_level":         man_result.service_level,
                "service_level_diff":        round(
                    thesis_result.service_level - man_result.service_level, 6
                ),
                "lt_qty_diff":               round(
                    (thesis_result.lt_total_qty or 0.0) - (man_result.lt_total_qty or 0.0), 4
                ),
                "shortage_cost_diff":        round(
                    (thesis_result.shortage_cost or 0.0) - (man_result.shortage_cost or 0.0), 4
                ),
                "interpretation": (
                    "C cheaper than Man-BFP-TC" if gap < -0.5
                    else "Man-BFP-TC cheaper than C" if gap > 0.5
                    else "C and Man-BFP-TC within 0.5% of each other"
                ),
            })
        else:
            for k in [
                "thesis_total_cost", "man_total_cost",
                "cost_gap_pct", "cost_gap_abs",
                "runtime_C", "runtime_Man", "runtime_ratio_C_vs_Man",
                "thesis_service_level", "man_service_level", "service_level_diff",
                "lt_qty_diff", "shortage_cost_diff",
            ]:
                row[k] = float("nan")
            row["interpretation"] = "comparison not possible (one or both methods failed)"
        return row

    @staticmethod
    def compute_per_scenario_gaps_joint(
        thesis_result: ScenarioResult,
        joint_result:  "JointTCResult",
    ) -> Dict[str, Any]:
        """Same structure as compute_per_scenario_gaps but for C vs Man-Joint-TC."""
        row: Dict[str, Any] = {
            "scenario_id":       thesis_result.scenario_id,
            "thesis_success":    thesis_result.success,
            "joint_success":     joint_result.success,
            "thesis_status":     thesis_result.status,
            "joint_status":      joint_result.status,
        }
        if thesis_result.success and joint_result.success:
            tc = thesis_result.total_cost
            jc = joint_result.total_cost
            gap = (tc - jc) / max(abs(jc), 1e-9) * 100.0
            row.update({
                "thesis_total_cost":          tc,
                "joint_total_cost":           jc,
                "cost_gap_pct":               round(gap, 4),
                "cost_gap_abs":               round(tc - jc, 4),
                "runtime_C":                  thesis_result.runtime_seconds,
                "runtime_Joint":              joint_result.runtime_seconds,
                "runtime_ratio_C_vs_Joint":   round(
                    thesis_result.runtime_seconds / max(joint_result.runtime_seconds, 1e-3), 4
                ),
                "thesis_service_level":       thesis_result.service_level,
                "joint_service_level":        joint_result.service_level,
                "service_level_diff":         round(
                    thesis_result.service_level - joint_result.service_level, 6
                ),
                "interpretation": (
                    "C cheaper than Oracle" if gap < -0.5
                    else "Oracle cheaper than C" if gap > 0.5
                    else "C and Oracle within 0.5%"
                ),
            })
        else:
            for k in [
                "thesis_total_cost", "joint_total_cost",
                "cost_gap_pct", "cost_gap_abs",
                "runtime_C", "runtime_Joint", "runtime_ratio_C_vs_Joint",
                "thesis_service_level", "joint_service_level", "service_level_diff",
            ]:
                row[k] = float("nan")
            row["interpretation"] = "comparison not possible (one or both methods failed)"
        return row

    @staticmethod
    def compute_per_scenario_gaps_tsrfp(
        thesis_result: ScenarioResult,
        tsrfp_result:  "TSRFPTCResult",
    ) -> Dict[str, Any]:
        """Same structure as compute_per_scenario_gaps but for C vs Man-TSRFP-TC."""
        row: Dict[str, Any] = {
            "scenario_id":       thesis_result.scenario_id,
            "thesis_success":    thesis_result.success,
            "tsrfp_success":     tsrfp_result.success,
            "thesis_status":     thesis_result.status,
            "tsrfp_status":      tsrfp_result.status,
        }
        if thesis_result.success and tsrfp_result.success:
            tc = thesis_result.total_cost
            rc = tsrfp_result.total_cost
            gap = (tc - rc) / max(abs(rc), 1e-9) * 100.0
            row.update({
                "thesis_total_cost":           tc,
                "tsrfp_total_cost":            rc,
                "cost_gap_pct":                round(gap, 4),
                "cost_gap_abs":                round(tc - rc, 4),
                "runtime_C":                   thesis_result.runtime_seconds,
                "runtime_TSRFP":               tsrfp_result.runtime_seconds,
                "runtime_ratio_C_vs_TSRFP":    round(
                    thesis_result.runtime_seconds / max(tsrfp_result.runtime_seconds, 1e-3), 4
                ),
                "thesis_service_level":        thesis_result.service_level,
                "tsrfp_service_level":         tsrfp_result.service_level,
                "service_level_diff":          round(
                    thesis_result.service_level - tsrfp_result.service_level, 6
                ),
                "ccg_iterations":              tsrfp_result.ccg_iterations,
                "final_robust_gap":            tsrfp_result.final_robust_gap,
                "interpretation": (
                    "C cheaper than TSRFP" if gap < -0.5
                    else "TSRFP cheaper than C" if gap > 0.5
                    else "C and TSRFP within 0.5%"
                ),
            })
        else:
            for k in [
                "thesis_total_cost", "tsrfp_total_cost",
                "cost_gap_pct", "cost_gap_abs",
                "runtime_C", "runtime_TSRFP", "runtime_ratio_C_vs_TSRFP",
                "thesis_service_level", "tsrfp_service_level", "service_level_diff",
                "ccg_iterations", "final_robust_gap",
            ]:
                row[k] = float("nan")
            row["interpretation"] = "comparison not possible (one or both methods failed)"
        return row

    @staticmethod
    def compute_summary(
        per_scenario_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:
        methods = results_df["method"].unique()
        rows: List[Dict] = []
        for method in methods:
            for label in ["small", "medium", "large", "all"]:
                if label == "all":
                    sub = results_df[results_df["method"] == method]
                else:
                    mask = results_df["method"] == method
                    if "size_label" in results_df.columns:
                        mask &= results_df["size_label"] == label
                    sub = results_df[mask]
                if sub.empty:
                    continue
                success_sub = sub[sub["success"]]
                row = {
                    "method":       method,
                    "size_label":   label,
                    "n_scenarios":  len(sub),
                    "n_success":    len(success_sub),
                    "n_timeout":    int((sub["status"] == "timeout").sum()),
                    "n_failed":     int((sub["status"] == "failed").sum()),
                    "success_rate": round(len(success_sub) / max(len(sub), 1), 4),
                    "timeout_rate": round((sub["status"] == "timeout").sum() / max(len(sub), 1), 4),
                }
                for col, stat_name in [
                    ("runtime_seconds", "runtime"),
                    ("total_cost",      "total_cost"),
                    ("shortage_cost",   "shortage_cost"),
                    ("service_level",   "service_level"),
                    ("lt_total_qty",    "lt_total_qty"),
                ]:
                    vals = pd.to_numeric(success_sub[col], errors="coerce").dropna()
                    row[f"mean_{stat_name}"]   = float(vals.mean())   if len(vals) else float("nan")
                    row[f"median_{stat_name}"] = float(vals.median()) if len(vals) else float("nan")
                    row[f"std_{stat_name}"]    = float(vals.std())    if len(vals) else float("nan")
                    row[f"min_{stat_name}"]    = float(vals.min())    if len(vals) else float("nan")
                    row[f"max_{stat_name}"]    = float(vals.max())    if len(vals) else float("nan")
                rows.append(row)
        return pd.DataFrame(rows)


# ======================================================================
# OUTPUT WRITER
# ======================================================================

class ManOutputWriter:
    """Writes all output files for the C-vs-Man-BFP-TC comparison.

    Mirror of `OutputWriter` in the Achamrah validator, with a
    `validate_man_vs_C_*` filename prefix so outputs are isolated."""

    def __init__(self, output_dir: str):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def _save(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.out / filename
        df.to_csv(path, index=False)
        print(f"[Output] Saved {filename} ({len(df)} rows)")
        return path

    def write_per_scenario(self, rows: List[Dict]) -> Path:
        return self._save(pd.DataFrame(rows), "validate_man_vs_C_per_scenario.csv")

    def write_summary(self, df: pd.DataFrame) -> Path:
        return self._save(df, "validate_man_vs_C_summary.csv")

    def write_cost_breakdown(self, rows: List[Dict]) -> Path:
        cols = ["scenario_id", "method", "success", "total_cost",
                "routing_cost", "holding_cost", "transshipment_cost",
                "shortage_cost", "lt_total_qty", "service_level",
                # Man-specific extras (silently ignored if absent)
                "stage1_cost", "lt_cost", "final_stockout", "stockout_rate"]
        df = pd.DataFrame(rows)
        df = df[[c for c in cols if c in df.columns]]
        return self._save(df, "validate_man_vs_C_cost_breakdown.csv")

    def write_runtime(self, rows: List[Dict]) -> Path:
        cols = ["scenario_id", "method", "status", "runtime_seconds",
                "n_columns_generated", "n_columns_selected",
                "num_vars", "num_constrs", "mip_gap"]
        df = pd.DataFrame(rows)
        df = df[[c for c in cols if c in df.columns]]
        return self._save(df, "validate_man_vs_C_runtime.csv")

    def write_lt_plan(self, lt_moves: List[Dict]) -> Path:
        if not lt_moves:
            df = pd.DataFrame(columns=[
                "scenario_id", "method", "period",
                "from_store", "to_store", "sku",
                "lt_qty", "lt_unit_cost", "lt_total_cost", "vehicle"
            ])
        else:
            df = pd.DataFrame(lt_moves)
        return self._save(df, "validate_man_vs_C_lt_plan.csv")

    def write_gaps(self, gap_rows: List[Dict]) -> Path:
        return self._save(pd.DataFrame(gap_rows), "validate_man_vs_C_gaps.csv")

    def write_failures(self, rows: List[Dict]) -> Path:
        fail_rows = [r for r in rows if not r.get("success", True)]
        if not fail_rows:
            df = pd.DataFrame(columns=["scenario_id", "method", "status", "error_message"])
        else:
            df = pd.DataFrame(fail_rows)
            df = df[[c for c in ["scenario_id", "method", "status", "runtime_seconds",
                                   "error_message", "error_traceback"] if c in df.columns]]
        return self._save(df, "validate_man_vs_C_failures.csv")

    def write_config(self, config: Dict) -> Path:
        path = self.out / "validate_man_vs_C_config.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"[Output] Saved validate_man_vs_C_config.json")
        return path

    def write_scenario_manifest(self, scenarios: List[ScenarioSpec]) -> Path:
        df = pd.DataFrame([s.to_dict() for s in scenarios])
        return self._save(df, "validate_man_vs_C_scenario_manifest.csv")

    def write_joint_gaps(self, gap_rows: List[Dict]) -> Path:
        return self._save(pd.DataFrame(gap_rows), "validate_man_vs_C_joint_gaps.csv")

    def write_tsrfp_gaps(self, gap_rows: List[Dict]) -> Path:
        return self._save(pd.DataFrame(gap_rows), "validate_man_vs_C_tsrfp_gaps.csv")


# ======================================================================
# VALIDATION ORCHESTRATOR
# ======================================================================

class ManValidationOrchestrator:
    """
    1. Generate scenarios
    2. For each scenario: run Thesis C and Man-BFP-TC
    3. Save partial results after each scenario
    4. Compute gaps and summary statistics
    5. Export all output files
    """

    def __init__(
        self,
        data_csv: str,
        checkpoint_path: str,
        output_dir: str,
        dist_path: Optional[str] = None,
        repo_root: Optional[str] = None,
        store_limits: Optional[List[int]] = None,
        scenarios_per_size: int = SCENARIOS_PER_SIZE,
        window_length: int = WINDOW_LENGTH_PERIODS,
        demand_shock_sigma: float = DEMAND_SHOCK_SIGMA,
        include_joint_tc: bool = False,
        include_tsrfp_tc: bool = False,
        debug: bool = False,
        seed: int = 42,
    ):
        self.data_csv           = data_csv
        self.checkpoint         = checkpoint_path
        self.output_dir         = output_dir
        self.dist_path          = dist_path
        self.repo_root          = repo_root
        self.window_length      = int(window_length)
        self.demand_shock_sigma = float(demand_shock_sigma)
        self.include_joint_tc   = bool(include_joint_tc)
        self.include_tsrfp_tc   = bool(include_tsrfp_tc)
        self.debug              = debug
        self.seed               = seed

        _scenarios_per_size = DEBUG_N_SCENARIOS if debug else scenarios_per_size
        _store_limits       = store_limits or STORE_LIMITS

        self.scenario_gen = ManScenarioGenerator(
            data_csv=data_csv,
            store_limits=_store_limits,
            scenarios_per_size=_scenarios_per_size,
            sku_limit=SKU_LIMIT,
            period_granularity=PERIOD_GRANULARITY,
            window_length=self.window_length,
            base_seed=seed,
        )
        self.data_builder = SharedDataBuilder(
            data_csv=data_csv,
            dist_path=dist_path,
            period_granularity=PERIOD_GRANULARITY,
        )
        self.thesis_runner = ManThesisCRunner(
            checkpoint_path=checkpoint_path,
            repo_root=repo_root,
            cg_iterations=1,
            time_limit=THESIS_C_TIME_LIMIT,
        )
        self.man_runner = ManRunner(
            time_limit=MAN_TIME_LIMIT,
            mip_gap=MAN_MIP_GAP,
        )
        self.joint_runner = JointRunner(
            time_limit=MAN_TIME_LIMIT,
            mip_gap=MAN_MIP_GAP,
        ) if self.include_joint_tc else None
        self.tsrfp_runner = TSRFPRunner(
            time_limit=MAN_TIME_LIMIT,
            mip_gap=MAN_MIP_GAP,
        ) if self.include_tsrfp_tc else None
        self.writer = ManOutputWriter(output_dir)

    def run(self) -> None:
        t_total = time.time()

        # ── Checkpoint validation ──────────────────────────────────────
        ckpt_ok = self.thesis_runner.check_checkpoint()
        if not ckpt_ok and not ALLOW_GNN_FALLBACK:
            raise FileNotFoundError(
                f"GNN checkpoint not found: {self.checkpoint}\n"
                f"Run Step 7b training first, or set ALLOW_GNN_FALLBACK=1 for heuristic-only fallback."
            )

        # ── Scenario generation ────────────────────────────────────────
        scenarios = self.scenario_gen.generate()
        self.writer.write_scenario_manifest(scenarios)

        config_record = {
            "data_csv":               self.data_csv,
            "checkpoint_path":        self.checkpoint,
            "checkpoint_valid":       ckpt_ok,
            "n_scenarios":            len(scenarios),
            "store_limits":           STORE_LIMITS,
            "scenarios_per_size":     SCENARIOS_PER_SIZE,
            "sku_limit":              SKU_LIMIT,
            "period_granularity":     PERIOD_GRANULARITY,
            "window_length_periods":  self.window_length,
            "demand_shock_sigma":     self.demand_shock_sigma,
            "man_time_limit":         MAN_TIME_LIMIT,
            "man_mip_gap":            MAN_MIP_GAP,
            "thesis_c_time_limit":    THESIS_C_TIME_LIMIT,
            "allow_gnn_fallback":     ALLOW_GNN_FALLBACK,
            "holding_cost_rate":      HOLDING_COST_RATE,
            "shortage_cost_rate":     SHORTAGE_COST_RATE,
            "lt_cost_flat":           LT_COST_FLAT,
            "routing_alpha":          ROUTING_ALPHA,
            "vehicle_count":          VEHICLE_COUNT,
            "vehicle_capacity":       VEHICLE_CAPACITY,
            "include_joint_tc":       self.include_joint_tc,
            "include_tsrfp_tc":       self.include_tsrfp_tc,
            "debug":                  self.debug,
            "seed":                   self.seed,
        }
        self.writer.write_config(config_record)

        # ── Per-scenario loop ──────────────────────────────────────────
        all_results:     List[Dict] = []
        all_lt_moves:    List[Dict] = []
        gap_rows:        List[Dict] = []       # C vs Man-BFP-TC
        joint_gap_rows:  List[Dict] = []       # C vs Man-Joint-TC
        tsrfp_gap_rows:  List[Dict] = []       # C vs Man-TSRFP-TC
        partial_path = Path(self.output_dir) / "validate_man_vs_C_per_scenario.csv"

        methods_str = f"Thesis_C  vs  {self.man_runner.source_name}"
        if self.include_joint_tc:
            methods_str += f"  vs  {self.joint_runner.source_name}"
        if self.include_tsrfp_tc:
            methods_str += f"  vs  {self.tsrfp_runner.source_name}"
        print(f"\n{'='*70}")
        print(f"Starting validation: {len(scenarios)} scenarios")
        print(f"Methods: {methods_str}")
        print(f"{'='*70}\n")

        for idx, scenario in enumerate(scenarios, start=1):
            print(f"\n[{idx:02d}/{len(scenarios)}] Scenario: {scenario.scenario_id}  "
                  f"stores={scenario.store_limit}  skus={scenario.sku_limit}  "
                  f"periods={scenario.n_periods}  window={scenario.start_date}→{scenario.end_date}")

            # Build shared data slice
            df_slice  = self.data_builder.slice_for_scenario(scenario)
            dist_dict, dist_src = self.data_builder.get_distances(scenario.selected_stores)

            if idx == 1:
                print(f"  Distance source: {dist_src}")

            if df_slice.empty:
                print(f"  WARNING: empty data slice; skipping scenario.")
                continue

            # ─ Demand shock — generated once, shared by both methods ─────
            shock = generate_demand_shock(scenario, scenario.n_periods,
                                          self.demand_shock_sigma)
            if shock:
                print(f"  [Shock]  σ={self.demand_shock_sigma}  "
                      f"mean_ε={sum(shock.values())/len(shock):.3f}")

            # ─ Run Thesis C ─────────────────────────────────────────────
            print(f"  [Thesis C]  running...")
            self.thesis_runner.set_shock(shock)
            try:
                c_result = self.thesis_runner.run_scenario(scenario, df_slice, dist_dict)
            finally:
                self.thesis_runner.set_shock(None)
            print(f"  [Thesis C]  status={c_result.status}  "
                  f"cost={c_result.total_cost:.2f}  runtime={c_result.runtime_seconds:.1f}s")

            # ─ Run Man-BFP-TC ───────────────────────────────────────────
            print(f"  [Man-BFP-TC]  running...")
            man_result, lt_moves = self.man_runner.run_scenario(
                scenario, df_slice, dist_dict, demand_shock=shock or None
            )
            print(f"  [Man-BFP-TC]  status={man_result.status}  "
                  f"cost={man_result.total_cost:.2f}  "
                  f"mip_gap={man_result.mip_gap:.4f}  runtime={man_result.runtime_seconds:.1f}s")

            # ─ Run Man-Joint-TC (Oracle) — optional ─────────────────────
            joint_result = None
            joint_lt_moves: List[Dict[str, Any]] = []
            if self.include_joint_tc and self.joint_runner is not None:
                print(f"  [Man-Joint-TC]  running...")
                joint_result, joint_lt_moves = self.joint_runner.run_scenario(
                    scenario, df_slice, dist_dict, demand_shock=shock or None
                )
                print(f"  [Man-Joint-TC]  status={joint_result.status}  "
                      f"cost={joint_result.total_cost:.2f}  "
                      f"mip_gap={joint_result.mip_gap:.4f}  "
                      f"runtime={joint_result.runtime_seconds:.1f}s")

            # ─ Run Man-TSRFP-TC (C&CG robust) — optional ────────────────
            tsrfp_result = None
            tsrfp_lt_moves: List[Dict[str, Any]] = []
            if self.include_tsrfp_tc and self.tsrfp_runner is not None:
                print(f"  [Man-TSRFP-TC]  running...")
                tsrfp_result, tsrfp_lt_moves = self.tsrfp_runner.run_scenario(
                    scenario, df_slice, dist_dict, demand_shock=shock or None
                )
                print(f"  [Man-TSRFP-TC]  status={tsrfp_result.status}  "
                      f"cost={tsrfp_result.total_cost:.2f}  "
                      f"ccg_iters={tsrfp_result.ccg_iterations}  "
                      f"mip_gap={tsrfp_result.mip_gap:.4f}  "
                      f"runtime={tsrfp_result.runtime_seconds:.1f}s")

            # ─ Annotate results with scenario metadata ──────────────────
            results_this_scenario = [c_result, man_result]
            if joint_result is not None:
                results_this_scenario.append(joint_result)
            if tsrfp_result is not None:
                results_this_scenario.append(tsrfp_result)

            for result in results_this_scenario:
                d = result.to_dict()
                d["size_label"]  = scenario.size_label
                d["store_limit"] = scenario.store_limit
                d["sku_limit"]   = scenario.sku_limit
                d["n_periods"]   = scenario.n_periods
                d["start_date"]  = scenario.start_date
                d["end_date"]    = scenario.end_date
                d["random_seed"] = scenario.random_seed
                all_results.append(d)

            all_lt_moves.extend(lt_moves)
            if joint_lt_moves:
                all_lt_moves.extend(joint_lt_moves)
            if tsrfp_lt_moves:
                all_lt_moves.extend(tsrfp_lt_moves)

            # ─ Compute C vs Man-BFP-TC gap ──────────────────────────────
            gap = ManComparisonEngine.compute_per_scenario_gaps(c_result, man_result)
            gap["size_label"] = scenario.size_label
            gap_rows.append(gap)

            if c_result.success and man_result.success:
                print(f"  [Gap C vs BFP-TC]  {gap.get('cost_gap_pct', float('nan')):.2f}%  "
                      f"→ {gap.get('interpretation', '')}")

            # ─ Compute C vs Man-Joint-TC gap ────────────────────────────
            if joint_result is not None:
                jgap = ManComparisonEngine.compute_per_scenario_gaps_joint(
                    c_result, joint_result
                )
                jgap["size_label"] = scenario.size_label
                joint_gap_rows.append(jgap)
                if c_result.success and joint_result.success:
                    print(f"  [Gap C vs Joint-TC]  {jgap.get('cost_gap_pct', float('nan')):.2f}%  "
                          f"→ {jgap.get('interpretation', '')}")

            # ─ Compute C vs Man-TSRFP-TC gap ────────────────────────────
            if tsrfp_result is not None:
                rgap = ManComparisonEngine.compute_per_scenario_gaps_tsrfp(
                    c_result, tsrfp_result
                )
                rgap["size_label"] = scenario.size_label
                tsrfp_gap_rows.append(rgap)
                if c_result.success and tsrfp_result.success:
                    print(f"  [Gap C vs TSRFP-TC]  {rgap.get('cost_gap_pct', float('nan')):.2f}%  "
                          f"→ {rgap.get('interpretation', '')}")

            # ─ Partial save ─────────────────────────────────────────────
            if all_results:
                pd.DataFrame(all_results).to_csv(partial_path, index=False)

        # ── Aggregate outputs ──────────────────────────────────────────
        print(f"\n{'='*70}")
        print("Writing final outputs...")

        results_df = pd.DataFrame(all_results)
        summary_df = ManComparisonEngine.compute_summary(pd.DataFrame(gap_rows), results_df)

        self.writer.write_per_scenario(all_results)
        self.writer.write_summary(summary_df)
        self.writer.write_cost_breakdown(all_results)
        self.writer.write_runtime(all_results)
        self.writer.write_lt_plan(all_lt_moves)
        self.writer.write_gaps(gap_rows)
        self.writer.write_failures(all_results)
        if joint_gap_rows:
            self.writer.write_joint_gaps(joint_gap_rows)
        if tsrfp_gap_rows:
            self.writer.write_tsrfp_gaps(tsrfp_gap_rows)

        # ── Print summary table ────────────────────────────────────────
        total_runtime = time.time() - t_total
        print(f"\n{'='*70}")
        print(f"VALIDATION COMPLETE  total_runtime={total_runtime:.0f}s")
        print(f"{'='*70}")

        if not summary_df.empty:
            print(summary_df[[
                "method", "size_label", "n_scenarios", "n_success",
                "success_rate", "timeout_rate",
                "mean_runtime", "mean_total_cost", "mean_service_level",
            ]].to_string(index=False))

        # ── Thesis-ready interpretation notes ──────────────────────────
        print(f"\n{'='*70}")
        print("INTERPRETATION NOTES FOR THESIS:")
        print(f"{'='*70}")
        if not results_df.empty:
            c_success   = results_df[results_df["method"] == "Thesis_C_ALNS_CG_GNN"]["success"].mean()
            man_success = results_df[results_df["method"] == self.man_runner.source_name]["success"].mean()
            print(f"  Thesis C success rate:    {c_success*100:.1f}%")
            print(f"  Man-BFP-TC success rate:  {man_success*100:.1f}%")
            gap_valid = [g for g in gap_rows
                         if not math.isnan(g.get("cost_gap_pct", float("nan")))]
            if gap_valid:
                mean_gap = sum(g["cost_gap_pct"] for g in gap_valid) / len(gap_valid)
                print(f"  Mean cost gap (C vs Man-BFP-TC): {mean_gap:+.2f}%  "
                      f"(positive = C more expensive, negative = C cheaper)")
            if joint_gap_rows:
                jgap_valid = [g for g in joint_gap_rows
                              if not math.isnan(g.get("cost_gap_pct", float("nan")))]
                if jgap_valid:
                    mean_jgap = sum(g["cost_gap_pct"] for g in jgap_valid) / len(jgap_valid)
                    print(f"  Mean cost gap (C vs Oracle):    {mean_jgap:+.2f}%  "
                          f"(Oracle = perfect-info lower bound)")
                    joint_rows = results_df[results_df["method"] == "Man-Joint-TC"]
                    if not joint_rows.empty:
                        j_success = joint_rows["success"].mean()
                        print(f"  Man-Joint-TC success rate: {j_success*100:.1f}%")
        print(f"  Outputs saved to: {self.output_dir}")


# ======================================================================
# CLI ENTRY POINT
# ======================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate Thesis C vs Man-BFP-TC benchmark on test data.csv"
    )
    p.add_argument("--data_csv",    default=DEFAULT_DATA_CSV,
                   help="Path to test data.csv (default: 'test data.csv')")
    p.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT,
                   help="Path to trained GNN checkpoint (best_valid_prauc.pt)")
    p.add_argument("--output_dir",  default=DEFAULT_OUTPUT_DIR,
                   help="Directory for validation outputs")
    p.add_argument("--dist_path",   default=None,
                   help="Optional: path to thesis distance matrix CSV")
    p.add_argument("--repo_root",   default=None,
                   help="Optional: path to cloned repo root (for irp_gurobi_converted import)")
    p.add_argument("--n_scenarios", type=int, default=30,
                   help="Total number of scenarios (default: 30 = 3 sizes × 10)")
    p.add_argument("--store_limits", nargs="+", type=int, default=STORE_LIMITS,
                   help="Store counts to benchmark (default: 5 8 10)")
    p.add_argument("--scenarios_per_size", type=int, default=SCENARIOS_PER_SIZE,
                   help="Scenarios per store size (default: 10)")
    p.add_argument("--window_length", type=int, default=WINDOW_LENGTH_PERIODS,
                   help=("Number of consecutive periods per scenario window "
                         f"(default: {WINDOW_LENGTH_PERIODS}). "
                         "Set to 1 to test single-period rolling — Man-BFP-TC's "
                         "native scope, where the multi-period horizon advantage "
                         "of Thesis C disappears."))
    p.add_argument("--demand_shock_sigma", type=float, default=DEMAND_SHOCK_SIGMA,
                   help=("Std-dev of the N(1,σ²) demand shock applied between "
                         "Stage-1 (routing) and Stage-2 (LT/CG) for BOTH methods "
                         f"(default: {DEMAND_SHOCK_SIGMA}). "
                         "Set to 0 for deterministic baseline (no shock)."))
    p.add_argument("--joint_tc",    action="store_true",
                   help="Also run Man-Joint-TC (Oracle) alongside BFP-TC and Thesis C")
    p.add_argument("--tsrfp_tc",    action="store_true",
                   help="Also run Man-TSRFP-TC (C&CG robust) alongside BFP-TC and Thesis C")
    p.add_argument("--debug",       action="store_true",
                   help="Quick debug run: 3 scenarios per size instead of 10")
    p.add_argument("--seed",        type=int, default=42)
    return p


def main() -> None:
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()  # ignore Jupyter/Papermill kernel args

    # Kaggle env-var overrides
    data_csv   = os.environ.get("VALIDATE_DATA_CSV",       args.data_csv)
    checkpoint = os.environ.get("CHECKPOINT_PATH",         args.checkpoint)
    output_dir = os.environ.get("VALIDATE_MAN_OUTPUT_DIR", args.output_dir)
    repo_root  = os.environ.get("REPO_ROOT",               args.repo_root)

    if repo_root is None:
        for candidate in [
            "/kaggle/working/Thesis-Work",
            "/kaggle/working/repo",
            str(Path(__file__).parent),
        ]:
            if Path(candidate).exists():
                repo_root = candidate
                break

    if not Path(checkpoint).exists() and repo_root:
        bundle_dir = os.environ.get("IRP_RESUME_BUNDLE_DIR", "")
        for candidate in [
            f"{bundle_dir}/Results/gnn_training/best_valid_prauc.pt",
            f"{repo_root}/Results/gnn_training/best_valid_prauc.pt",
            "/kaggle/working/Results/gnn_training/best_valid_prauc.pt",
        ]:
            if candidate and Path(candidate).exists():
                checkpoint = candidate
                print(f"[Main] Auto-detected checkpoint: {checkpoint}")
                break

    dist_path = args.dist_path
    if dist_path is None and repo_root:
        dp = Path(repo_root) / DIST_REL_PATH
        if dp.exists():
            dist_path = str(dp)
            print(f"[Main] Auto-detected distance matrix: {dist_path}")

    print(f"\n{'='*70}")
    print("Validate_with_Man_Kaggle.py")
    print(f"{'='*70}")
    print(f"  data_csv:           {data_csv}")
    print(f"  checkpoint:         {checkpoint}")
    print(f"  output_dir:         {output_dir}")
    print(f"  dist_path:          {dist_path or 'synthetic (fallback)'}")
    print(f"  repo_root:          {repo_root or 'current dir'}")
    print(f"  store_limits:       {args.store_limits}")
    print(f"  scenarios_per_size: {args.scenarios_per_size}")
    print(f"  window_length:      {args.window_length}  "
          f"({'single-period (Man native scope)' if args.window_length == 1 else 'multi-period'})")
    print(f"  demand_shock_sigma: {args.demand_shock_sigma}"
          f"  ({'no shock — deterministic' if args.demand_shock_sigma == 0 else 'N(1,σ²) shock active'})")
    print(f"  joint_tc:           {args.joint_tc}  (Oracle benchmark)")
    print(f"  tsrfp_tc:           {args.tsrfp_tc}  (C&CG robust benchmark)")
    print(f"  debug:              {args.debug}")
    print(f"  man_time_limit:     {MAN_TIME_LIMIT}s")
    print(f"  man_mip_gap:        {MAN_MIP_GAP}")
    print(f"  thesis_c_time_limit: {THESIS_C_TIME_LIMIT}s")
    print(f"{'='*70}\n")

    orchestrator = ManValidationOrchestrator(
        data_csv=data_csv,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        dist_path=dist_path,
        repo_root=repo_root,
        store_limits=args.store_limits,
        scenarios_per_size=args.scenarios_per_size,
        window_length=args.window_length,
        demand_shock_sigma=args.demand_shock_sigma,
        include_joint_tc=args.joint_tc,
        include_tsrfp_tc=args.tsrfp_tc,
        debug=args.debug,
        seed=args.seed,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
