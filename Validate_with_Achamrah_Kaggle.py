"""
Validate_with_Achamrah_Kaggle.py
==================================
Compare Thesis Variant C (ALNS + Column Generation + GNN-guided selection)
against Achamrah Integrated benchmark on controlled scenarios generated from
the same test data.csv.

Usage:
    python Validate_with_Achamrah_Kaggle.py [--data_csv /path/to/test data.csv]
                                            [--checkpoint /path/to/best_valid_prauc.pt]
                                            [--output_dir /kaggle/working/Results/achamrah_validation]
                                            [--n_scenarios 30]
                                            [--debug]

Design:
    - 30 scenarios = 3 store sizes × 10 scenarios each
    - Both methods run on IDENTICAL store subset, SKU subset, period window, demand data
    - Weekly aggregation, 8-period windows
    - Achamrah: 1200s time limit per scenario, failure logged not crash
    - Thesis C: uses trained GNN checkpoint from Step 7b
    - All outputs are thesis-comparable (same column names)
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ======================================================================
# CONFIGURATION
# ======================================================================

STORE_LIMITS: List[int] = [5, 8, 10]
SCENARIOS_PER_SIZE: int = 10
SKU_LIMIT: int = 5
PERIOD_GRANULARITY: str = "weekly"
WINDOW_LENGTH_PERIODS: int = 8

ACHAMRAH_TIME_LIMIT: int = int(os.environ.get("ACHAMRAH_TIME_LIMIT", "1200"))
THESIS_C_TIME_LIMIT:  int = int(os.environ.get("THESIS_C_TIME_LIMIT",  "1200"))
ACHAMRAH_MIP_GAP: float  = 0.01
ALLOW_GNN_FALLBACK: bool = (
    os.environ.get("ALLOW_GNN_FALLBACK", "0").lower() in {"1", "true", "yes"}
)

VEHICLE_COUNT:    int   = 2
VEHICLE_CAPACITY: float = 500.0

# Cost parameters — must be IDENTICAL for both methods.
# CRITICAL: shortage_cost MUST exceed LT_COST_FLAT (0.6) for LT to be
# beneficial. With shortage < LT cost, CG finds zero improving columns
# (instant termination) and ALNS prefers shortages over deliveries.
# Using shortage=2.0 and holding=0.1 ensures LT and deliveries are both
# economically meaningful in the optimization.
HOLDING_COST_RATE:   float = 0.1
SHORTAGE_COST_RATE:  float = 2.0
LT_COST_FLAT:        float = 0.6
ROUTING_ALPHA:       float = 1.0

DEBUG_N_SCENARIOS: int = 3   # used when --debug is passed

DEFAULT_DATA_CSV   = "test data.csv"
DEFAULT_OUTPUT_DIR = "Results/achamrah_validation"
# Must match irp.DEFAULT_GNN_CHECKPOINT exactly so run_lt_recourse_from_baseline can load it
DEFAULT_CHECKPOINT = "GNN/trained_models/irplt_teacher/bigat/pairwise_rank/best_model.pt"
DIST_REL_PATH      = Path("Distance data") / "mm_megamarket_distance_matrix_clean.csv"

# ======================================================================
# SCENARIO SPECIFICATION
# ======================================================================

@dataclass
class ScenarioSpec:
    """Defines one comparable benchmark scenario."""
    scenario_id:        str
    store_limit:        int
    sku_limit:          int
    selected_stores:    List[str]
    selected_skus:      List[str]
    start_date:         str
    end_date:           str
    period_granularity: str
    n_periods:          int
    random_seed:        int
    size_label:         str          # "small" | "medium" | "large"
    window_index:       int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["selected_stores"] = json.dumps(self.selected_stores)
        d["selected_skus"]   = json.dumps(self.selected_skus)
        return d


@dataclass
class ScenarioResult:
    """Result for one method on one scenario."""
    scenario_id:        str
    method:             str
    status:             str          # "success" | "timeout" | "failed" | "skipped"
    success:            bool
    runtime_seconds:    float
    total_cost:         float
    holding_cost:       float
    routing_cost:       float
    transshipment_cost: float
    shortage_cost:      float
    shortage_qty:       float
    service_level:      float
    lt_total_qty:       float
    n_lt_moves:         int
    n_routes:           int
    # thesis-C extras
    n_columns_generated: int = 0
    n_columns_selected:  int = 0
    gnn_checkpoint_used: str = ""
    # achamrah extras
    achamrah_vehicle_indexed_lt: Optional[bool] = None
    num_vars:            int = 0
    num_constrs:         int = 0
    mip_gap:             float = float("nan")
    error_message:       str = ""
    error_traceback:     str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ======================================================================
# SCENARIO GENERATOR
# ======================================================================

class ScenarioGenerator:
    """Generates N_SCENARIOS controlled benchmark scenarios from test data.csv."""

    REQUIRED_COLS = ["SITE_NAME", "ART_SV_NAME_ENG", "SALE_QTY", "END_QTY", "PERIOD"]

    def __init__(
        self,
        data_csv: str,
        store_limits: List[int] = None,
        scenarios_per_size: int = SCENARIOS_PER_SIZE,
        sku_limit: int = SKU_LIMIT,
        period_granularity: str = PERIOD_GRANULARITY,
        window_length: int = WINDOW_LENGTH_PERIODS,
        base_seed: int = 42,
    ):
        self.data_csv         = data_csv
        self.store_limits     = store_limits or STORE_LIMITS
        self.scenarios_per_size = scenarios_per_size
        self.sku_limit        = sku_limit
        self.period_granularity = period_granularity
        self.window_length    = window_length
        self.base_seed        = base_seed
        self._df_weekly: Optional[pd.DataFrame] = None

    def _load_and_aggregate(self) -> pd.DataFrame:
        """Load CSV and aggregate to weekly periods."""
        if self._df_weekly is not None:
            return self._df_weekly

        path = Path(self.data_csv)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path, sheet_name=0)

        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"[ScenarioGen] Missing required columns: {missing}")

        df = df[self.REQUIRED_COLS].copy()
        df.columns = ["store", "sku", "sale_qty", "end_qty", "period_raw"]
        df["store"]    = df["store"].astype(str).str.strip()
        df["sku"]      = df["sku"].astype(str).str.strip()
        df["sale_qty"] = pd.to_numeric(df["sale_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["end_qty"]  = pd.to_numeric(df["end_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["date"]     = pd.to_datetime(df["period_raw"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"])

        gran = self.period_granularity.lower()
        if gran == "weekly":
            df["period_date"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
        elif gran == "biweekly":
            ref = df["date"].min()
            df["period_date"] = df["date"].apply(
                lambda d: ref + pd.Timedelta(weeks=((d - ref).days // 14) * 2)
            )
        elif gran == "monthly":
            df["period_date"] = df["date"].dt.to_period("M").apply(lambda p: p.start_time)
        else:
            df["period_date"] = df["date"]

        agg = (
            df.groupby(["store", "sku", "period_date"], as_index=False)
              .agg(sale_qty=("sale_qty", "sum"), end_qty=("end_qty", "last"))
        )
        self._df_weekly = agg.sort_values(["store", "sku", "period_date"]).reset_index(drop=True)
        return self._df_weekly

    def _select_top_skus(self, df: pd.DataFrame, k: int) -> List[str]:
        """Select top k SKUs by total demand across all stores."""
        sku_demand = df.groupby("sku")["sale_qty"].sum().sort_values(ascending=False)
        return list(sku_demand.head(k).index)

    def _get_windows(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Get rolling windows of window_length consecutive periods."""
        all_periods = sorted(df["period_date"].unique())
        if len(all_periods) < self.window_length:
            return [(str(all_periods[0].date()), str(all_periods[-1].date()))]
        windows = []
        for i in range(len(all_periods) - self.window_length + 1):
            start = all_periods[i]
            end   = all_periods[i + self.window_length - 1]
            windows.append((str(start.date()), str(end.date())))
        return windows

    def generate(self) -> List[ScenarioSpec]:
        """Generate all benchmark scenarios."""
        df = self._load_and_aggregate()
        top_skus = self._select_top_skus(df, self.sku_limit)
        all_stores = sorted(df["store"].unique())
        windows    = self._get_windows(df)

        if not windows:
            raise ValueError("[ScenarioGen] No valid period windows found in data.")

        scenarios: List[ScenarioSpec] = []
        size_labels = {5: "small", 8: "medium", 10: "large"}

        for store_limit in self.store_limits:
            if store_limit > len(all_stores):
                print(f"[ScenarioGen] WARNING: store_limit={store_limit} > available stores "
                      f"({len(all_stores)}); using {len(all_stores)}")
                store_limit = len(all_stores)

            size_label = size_labels.get(store_limit, f"size{store_limit}")

            for scenario_idx in range(self.scenarios_per_size):
                rng = random.Random(self.base_seed + store_limit * 1000 + scenario_idx)
                seed = self.base_seed + store_limit * 1000 + scenario_idx

                # Select stores: randomised subset using deterministic seed
                selected_stores = sorted(rng.sample(list(all_stores), k=min(store_limit, len(all_stores))))

                # Select SKUs
                selected_skus = top_skus[:self.sku_limit]

                # Select window: cycle through available windows
                win_idx    = scenario_idx % len(windows)
                start_date, end_date = windows[win_idx]

                # Count actual periods in window for the selected stores/SKUs
                mask = (
                    df["store"].isin(selected_stores)
                    & df["sku"].isin(selected_skus)
                    & (df["period_date"].astype(str) >= start_date)
                    & (df["period_date"].astype(str) <= end_date)
                )
                n_periods = df[mask]["period_date"].nunique()

                if n_periods < 2:
                    print(f"[ScenarioGen] WARNING: scenario {scenario_idx} for size {store_limit} "
                          f"has only {n_periods} periods; skipping.")
                    continue

                scenario_id = f"{size_label}_s{store_limit:02d}_sc{scenario_idx:02d}"

                scenarios.append(ScenarioSpec(
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
                ))

        print(f"[ScenarioGen] Generated {len(scenarios)} scenarios "
              f"({self.scenarios_per_size} per store size × {len(self.store_limits)} sizes)")
        return scenarios


# ======================================================================
# SHARED DATA BUILDER — identical data slice for both methods
# ======================================================================

class SharedDataBuilder:
    """
    Builds a filtered DataFrame slice that is IDENTICAL for both Thesis C
    and Achamrah so the comparison is fully controlled.

    Also loads the distance matrix from the thesis distance file if available,
    and generates synthetic Euclidean distances as fallback.
    """

    def __init__(
        self,
        data_csv: str,
        dist_path: Optional[str] = None,
        period_granularity: str = PERIOD_GRANULARITY,
    ):
        self.data_csv          = data_csv
        self.dist_path         = dist_path
        self.period_granularity = period_granularity
        self._df_full: Optional[pd.DataFrame] = None
        self._dist_df: Optional[pd.DataFrame] = None
        self._dist_source: str = "synthetic"

    def _load_full(self) -> pd.DataFrame:
        if self._df_full is not None:
            return self._df_full
        path = Path(self.data_csv)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path, sheet_name=0)
        df = df[["SITE_NAME", "ART_SV_NAME_ENG", "SALE_QTY", "END_QTY", "PERIOD"]].copy()
        df.columns = ["store", "sku", "sale_qty", "end_qty", "period_raw"]
        df["store"]    = df["store"].astype(str).str.strip()
        df["sku"]      = df["sku"].astype(str).str.strip()
        df["sale_qty"] = pd.to_numeric(df["sale_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["end_qty"]  = pd.to_numeric(df["end_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["date"]     = pd.to_datetime(df["period_raw"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"])

        gran = self.period_granularity.lower()
        if gran == "weekly":
            df["period_date"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
        elif gran == "biweekly":
            ref = df["date"].min()
            df["period_date"] = df["date"].apply(
                lambda d: ref + pd.Timedelta(weeks=((d - ref).days // 14) * 2)
            )
        elif gran == "monthly":
            df["period_date"] = df["date"].dt.to_period("M").apply(lambda p: p.start_time)
        else:
            df["period_date"] = df["date"]

        self._df_full = (
            df.groupby(["store", "sku", "period_date"], as_index=False)
              .agg(sale_qty=("sale_qty", "sum"), end_qty=("end_qty", "last"))
        )
        return self._df_full

    def _load_distances(self, stores: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Load distances from thesis distance matrix CSV if available.
        Falls back to synthetic Euclidean coordinates (seed=42 for reproducibility).
        Store→Node mapping: node 0 = warehouse, nodes 1..N = stores.
        Returns: dict {(store_name_i, store_name_j): dist_km}
        """
        if self.dist_path and Path(self.dist_path).exists():
            try:
                dist_df = pd.read_csv(self.dist_path)
                # Expected columns: from_site, to_site, distance_km (or similar)
                col_candidates = [c for c in dist_df.columns if "dist" in c.lower() or "km" in c.lower()]
                from_col = next((c for c in dist_df.columns if "from" in c.lower() or c == "SITE_FROM"), None)
                to_col   = next((c for c in dist_df.columns if "to" in c.lower()   or c == "SITE_TO"),   None)
                dist_col = col_candidates[0] if col_candidates else None

                if from_col and to_col and dist_col:
                    dist_lookup: Dict[Tuple[str, str], float] = {}
                    for _, row in dist_df.iterrows():
                        fi = str(row[from_col]).strip()
                        ti = str(row[to_col]).strip()
                        dist_lookup[(fi, ti)] = float(row[dist_col])
                    self._dist_source = f"thesis_distance_file:{Path(self.dist_path).name}"
                    return dist_lookup
            except Exception as e:
                print(f"[SharedData] Distance file load failed ({e}); using synthetic distances.")

        # Synthetic fallback: reproducible Euclidean, seed=42
        n_nodes = len(stores) + 1  # +1 for warehouse
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 100, size=(n_nodes, 2))
        dist_lookup = {}
        store_names = ["__WAREHOUSE__"] + list(stores)
        for i, si in enumerate(store_names):
            for j, sj in enumerate(store_names):
                if i != j:
                    dist_lookup[(si, sj)] = float(np.linalg.norm(coords[i] - coords[j]))
        self._dist_source = "synthetic_euclidean_seed42"
        return dist_lookup

    def slice_for_scenario(self, scenario: ScenarioSpec) -> pd.DataFrame:
        """Return filtered DataFrame for the scenario (identical for both methods)."""
        df = self._load_full()
        mask = (
            df["store"].isin(scenario.selected_stores)
            & df["sku"].isin(scenario.selected_skus)
            & (df["period_date"].astype(str) >= scenario.start_date)
            & (df["period_date"].astype(str) <= scenario.end_date)
        )
        return df[mask].copy().reset_index(drop=True)

    def get_distances(
        self,
        stores: List[str],
    ) -> Tuple[Dict[Tuple[str, str], float], str]:
        """Returns (distance_dict, source_description)."""
        dist = self._load_distances(stores)
        return dist, self._dist_source


# ======================================================================
# THESIS C RUNNER
# ======================================================================

class ThesisCRunner:
    """
    Runs thesis variant C (ALNS baseline + CG + GNN-guided column selection)
    on a given scenario. Uses the trained GNN checkpoint from Step 7b.
    """

    def __init__(
        self,
        checkpoint_path: str,
        repo_root: Optional[str] = None,
        cg_iterations: int = 50,
        time_limit: int = THESIS_C_TIME_LIMIT,
        gnn_selection_mode: str = "adaptive_gap",
        allow_gnn_fallback: bool = ALLOW_GNN_FALLBACK,
    ):
        self.checkpoint_path   = checkpoint_path
        self.repo_root         = repo_root
        self.cg_iterations     = cg_iterations
        self.time_limit        = time_limit
        self.gnn_selection_mode = gnn_selection_mode
        self.allow_gnn_fallback = allow_gnn_fallback
        self._irp              = None
        self._checkpoint_ok    = False

    def check_checkpoint(self) -> bool:
        """Validate that checkpoint exists and is loadable."""
        ckpt = Path(self.checkpoint_path)
        if not ckpt.exists():
            print(f"[ThesisC] ERROR: checkpoint not found at {ckpt}")
            return False
        try:
            import torch
            payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            assert "state_dict" in payload, "Checkpoint missing state_dict"
            prauc = payload.get("best_valid_prauc") or payload.get("valid_metrics", {}).get("pr_auc", "?")
            print(f"[ThesisC] Checkpoint OK: {ckpt.name}  "
                  f"epoch={payload.get('last_epoch', '?')}  "
                  f"valid_pr_auc={prauc}")
            self._checkpoint_ok = True
            return True
        except Exception as e:
            print(f"[ThesisC] ERROR loading checkpoint: {e}")
            return False

    def _load_irp(self) -> Any:
        """Import irp_gurobi_converted (from repo root if given)."""
        if self._irp is not None:
            return self._irp
        if self.repo_root:
            sys.path.insert(0, str(self.repo_root))
        try:
            import irp_gurobi_converted as irp
            self._irp = irp
        except ImportError as e:
            raise ImportError(
                f"[ThesisC] Cannot import irp_gurobi_converted. "
                f"Make sure repo is cloned and repo_root is set. ({e})"
            )
        return self._irp

    def _build_irp_data(
        self,
        scenario: ScenarioSpec,
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
    ) -> Any:
        """
        Build an IRPData from the scenario slice using DatasetToIRPValidationMapper.
        Writes a temp CSV in the exact format the mapper expects.
        """
        irp = self._load_irp()

        # Resolve distance matrix path
        dist_path = None
        if self.repo_root:
            dp = Path(self.repo_root) / DIST_REL_PATH
            if dp.exists():
                dist_path = str(dp)

        # Write a temp CSV with the exact columns the mapper requires:
        # SITE_NAME, NORMAL_PRICE, ART_SV_NAME_ENG, SALE_QTY, END_QTY, PERIOD
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
        df_out = df_slice.copy()
        df_out["PERIOD"] = df_out["period_date"].dt.strftime("%Y%m%d").astype(int)
        # Use existing price column if present, else a neutral placeholder (1.0)
        if "price" in df_out.columns:
            df_out["NORMAL_PRICE"] = df_out["price"].fillna(1.0)
        else:
            df_out["NORMAL_PRICE"] = 1.0
        df_out = df_out.rename(columns={
            "store":    "SITE_NAME",
            "sku":      "ART_SV_NAME_ENG",
            "sale_qty": "SALE_QTY",
            "end_qty":  "END_QTY",
        })
        df_out[["SITE_NAME", "NORMAL_PRICE", "ART_SV_NAME_ENG", "SALE_QTY", "END_QTY", "PERIOD"]].to_csv(
            tmp.name, index=False
        )
        tmp.close()

        try:
            # Data is pre-filtered to exactly the right stores/SKUs/periods,
            # so pass store_limit=None / sku_limit=None to avoid re-selection.
            mapper = irp.DatasetToIRPValidationMapper(
                excel_path=tmp.name,
                store_limit=None,
                sku_limit=None,
            )
            data, *_ = mapper.build_irp_data(
                vehicle_count=VEHICLE_COUNT,
                vehicle_capacity=VEHICLE_CAPACITY,
                shortage_cost_rate=SHORTAGE_COST_RATE,
                holding_cost_rate=HOLDING_COST_RATE,
                lt_ship_cost_flat=LT_COST_FLAT,
                distance_matrix_path=dist_path,
            )
            # ── Cost normalisation for fair comparison with Achamrah ──────
            # Achamrah has no CW shipping cost, no vehicle fixed cost, no
            # warehouse holding cost, and uses flat unit rates. Override to
            # match. shortage_cost=2.0 > LT_COST_FLAT=0.6 so LT is
            # economically beneficial (saves 1.4/unit vs unmet shortage).
            for s in data.stores:
                for p in data.products:
                    data.holding_cost_store[(s, p)] = HOLDING_COST_RATE   # 0.1 flat
                    data.shortage_cost[(s, p)]       = SHORTAGE_COST_RATE  # 2.0 flat
                    data.ship_cost_cw[(s, p)]        = 0.0
            for p in data.products:
                data.holding_cost_wh[p] = 0.0
            data.vehicle_fixed_cost = 0.0
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        return data

    def run_scenario(
        self,
        scenario: ScenarioSpec,
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
    ) -> ScenarioResult:
        """Run thesis variant C on the scenario and return metrics."""
        t0 = time.time()

        if not self._checkpoint_ok and not self.allow_gnn_fallback:
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                method="Thesis_C_ALNS_CG_GNN",
                status="skipped",
                success=False,
                runtime_seconds=0.0,
                total_cost=float("nan"), holding_cost=float("nan"),
                routing_cost=float("nan"), transshipment_cost=float("nan"),
                shortage_cost=float("nan"), shortage_qty=float("nan"),
                service_level=float("nan"), lt_total_qty=float("nan"),
                n_lt_moves=0, n_routes=0,
                gnn_checkpoint_used=self.checkpoint_path,
                error_message="GNN checkpoint not found or invalid; ALLOW_GNN_FALLBACK=False",
            )

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                irp = self._load_irp()

                # Build IRPData
                data = self._build_irp_data(scenario, df_slice, dist_dict)

                # Demand shock disabled — realized demand equals forecast demand.
                # This ensures both methods face identical deterministic demand.
                data.realized_demand = dict(data.demand)

                # Run ALNS baseline
                baseline_sol = irp.BaselineALNSModel(data).solve(
                    msg=False,
                    time_limit=max(60, self.time_limit // 4),
                    allow_lateral_transshipment=False,
                )

                # Set env vars for GNN selection
                os.environ["IRP_GNN_SELECTION_MODE"] = self.gnn_selection_mode
                os.environ["IRP_GNN_MIN_KEEP"]        = "3"
                os.environ["IRP_GNN_MIN_KEEP_FRAC"]   = "0.05"
                os.environ["IRP_GNN_MAX_KEEP_FRAC"]   = "0.50"

                # Run CG + GNN (variant C).
                # lt_activation_threshold=1.0 ensures CG activates even for
                # small (≥1 unit) shortages; default 10.0 is too coarse for
                # small scenarios.
                pipeline = irp.IRPResearchPipeline(data)
                results  = pipeline.run_lt_recourse_from_baseline(
                    baseline_sol,
                    use_random_initial_patterns=True,
                    n_initial_patterns_per_product_period=5,
                    cg_iterations=self.cg_iterations,
                    msg=False,
                    gnn_checkpoint=self.checkpoint_path,
                    use_gnn=True,
                    runtime_gnn_mode=True,
                    gnn_selection_mode=self.gnn_selection_mode,
                    use_classical_fallback=False,
                    gnn_max_keep=100,
                    gnn_max_keep_fraction=0.50,
                    lt_activation_threshold=1.0,
                )

            runtime = time.time() - t0

            # Extract metrics from results
            cg_sol   = results.get("cg_solution")
            realized = results.get("realized_with_lt_cost_breakdown", {}) or {}
            realized_no_lt = results.get("realized_no_lt_cost_breakdown", {}) or {}
            cg_history = results.get("cg_episode_history") or results.get("cg_history", [])

            total_cost    = float(realized.get("total_realized_operating_cost", float("nan")))
            holding_cost  = float(realized.get("store_holding_cost_realized", 0.0))
            routing_cost  = float(realized_no_lt.get("route_distance_cost_executed_plan", 0.0))
            lt_cost       = float(realized.get("lateral_transshipment_cost_realized", 0.0))
            shortage_cost = float(realized.get("shortage_cost_realized", 0.0))
            shortage_qty  = float(realized.get("total_realized_shortage_units", 0.0))

            # Service level = 1 - shortage_qty / total_demand
            total_demand = float(realized.get("total_demand", 1.0)) or 1.0
            service_level = max(0.0, 1.0 - shortage_qty / total_demand)

            # LT plan from results
            lt_plan = results.get("lt_plan", pd.DataFrame())
            if isinstance(lt_plan, pd.DataFrame) and not lt_plan.empty:
                lt_qty_col  = next((c for c in lt_plan.columns if "qty" in c.lower()), None)
                lt_total_qty = float(lt_plan[lt_qty_col].sum()) if lt_qty_col else float("nan")
                n_lt_moves   = len(lt_plan)
            else:
                lt_total_qty, n_lt_moves = 0.0, 0

            # CG stats
            n_cols_gen  = int(results.get("gnn_candidates_before", 0))
            n_cols_sel  = int(results.get("gnn_selected_columns", 0))
            n_cols_gen  = n_cols_gen or sum(int(ep.get("generated_columns", 0)) for ep in cg_history)

            print(f"  [ThesisC breakdown] routing={routing_cost:.2f}  "
                  f"holding={holding_cost:.2f}  shortage={shortage_cost:.2f}  "
                  f"lt={lt_cost:.2f}  total={total_cost:.2f}")

            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                method="Thesis_C_ALNS_CG_GNN",
                status="success",
                success=True,
                runtime_seconds=runtime,
                total_cost=total_cost,
                holding_cost=holding_cost,
                routing_cost=routing_cost,
                transshipment_cost=lt_cost,
                shortage_cost=shortage_cost,
                shortage_qty=shortage_qty,
                service_level=service_level,
                lt_total_qty=lt_total_qty,
                n_lt_moves=n_lt_moves,
                n_routes=len(cg_sol.selected_patterns) if cg_sol else 0,
                n_columns_generated=n_cols_gen,
                n_columns_selected=n_cols_sel,
                gnn_checkpoint_used=self.checkpoint_path,
            )

        except Exception as exc:
            runtime = time.time() - t0
            tb = traceback.format_exc()
            status = "timeout" if runtime >= self.time_limit * 0.98 else "failed"
            print(f"[ThesisC] {scenario.scenario_id} {status}: {exc}")
            print(f"[ThesisC TRACEBACK]:\n{tb}")
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                method="Thesis_C_ALNS_CG_GNN",
                status=status,
                success=False,
                runtime_seconds=runtime,
                total_cost=float("nan"), holding_cost=float("nan"),
                routing_cost=float("nan"), transshipment_cost=float("nan"),
                shortage_cost=float("nan"), shortage_qty=float("nan"),
                service_level=float("nan"), lt_total_qty=float("nan"),
                n_lt_moves=0, n_routes=0,
                gnn_checkpoint_used=self.checkpoint_path,
                error_message=str(exc),
                error_traceback=tb,
            )


# ======================================================================
# ACHAMRAH RUNNER
# ======================================================================

class AchamrahRunner:
    """
    Runs Achamrah Integrated benchmark (vehicle_indexed_lt mode) on a scenario,
    using EXACTLY the same demand data as ThesisCRunner.
    """

    def __init__(
        self,
        vehicle_indexed_lt: bool = False,
        time_limit: int = ACHAMRAH_TIME_LIMIT,
        mip_gap: float = ACHAMRAH_MIP_GAP,
        threads: int = 4,
    ):
        self.vehicle_indexed_lt = vehicle_indexed_lt
        self.time_limit         = time_limit
        self.mip_gap            = mip_gap
        self.threads            = threads
        self.source_name = (
            "Achamrah_Matheuristic_VehicleLT"
            if vehicle_indexed_lt
            else "Achamrah_Matheuristic_SimplifiedLT"
        )

    def _build_achamrah_instance(
        self,
        scenario: ScenarioSpec,
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
    ):
        """Build IRPTInstance from scenario data — same demand as thesis."""
        from achamrah_2022_irpt_matheuristic import IRPTInstance

        stores   = scenario.selected_stores
        skus     = scenario.selected_skus
        periods_dt = sorted(df_slice["period_date"].unique())
        H = list(range(1, len(periods_dt) + 1))
        period_map = {p: h for h, p in enumerate(periods_dt, start=1)}

        store_to_id = {s: i + 1 for i, s in enumerate(stores)}
        sku_to_id   = {p: i     for i, p in enumerate(skus)}
        N = list(store_to_id.values())
        P = list(sku_to_id.values())
        V = list(range(1, VEHICLE_COUNT + 1))

        # Build demand D[p, i, t]
        D: Dict = {}
        for _, row in df_slice.iterrows():
            if row["store"] not in store_to_id or row["sku"] not in sku_to_id:
                continue
            p_id = sku_to_id[row["sku"]]
            i_id = store_to_id[row["store"]]
            h    = period_map.get(row["period_date"])
            if h is not None:
                D[(p_id, i_id, h)] = float(row["sale_qty"])

        # Fill missing zeros
        for p in P:
            for i in N:
                for t in H:
                    D.setdefault((p, i, t), 0.0)

        # Initial inventory from first period end_qty
        I0: Dict = {}
        for p_name, p_id in sku_to_id.items():
            for s_name, i_id in store_to_id.items():
                mask = (df_slice["store"] == s_name) & (df_slice["sku"] == p_name)
                I0[(p_id, i_id)] = float(df_slice[mask]["end_qty"].iloc[0]) if mask.any() else 0.0
        for p in P:
            I0[(p, 0)] = 10000.0   # warehouse: unlimited

        # Build distance matrix in Achamrah format (node_id based)
        wh_name = "__WAREHOUSE__"
        node_names = [wh_name] + stores   # 0 = warehouse
        d_achamrah: Dict = {}
        for i_idx, ni in enumerate(node_names):
            for j_idx, nj in enumerate(node_names):
                if i_idx == j_idx:
                    continue
                key = (ni, nj)
                if key in dist_dict:
                    d_achamrah[(i_idx, j_idx)] = dist_dict[key]
                else:
                    # Fill missing with max known + penalty
                    d_achamrah[(i_idx, j_idx)] = 100.0  # synthetic fallback

        # Cost dictionaries
        h_cost = {(p, i): HOLDING_COST_RATE  for p in P for i in N}
        h_cost.update({(p, 0): 0.0 for p in P})
        f_cost = {(p, i): SHORTAGE_COST_RATE for p in P for i in N}
        f_cost.update({(p, 0): 0.0 for p in P})
        b_cost = {(i, j): LT_COST_FLAT for i in N for j in N if i != j}

        # Storage capacities: size from data so the constraint never spuriously
        # binds. Stores: 5× initial inventory + buffer for LT/delivery inflows.
        # Warehouse: must absorb g_rep[(p,t)] inflows across the horizon, since
        # Qdir is bounded by V*Q per period (~1000) << g (~100000). Set to a
        # large constant so the CW balance + capacity is always feasible.
        C_cap = {}
        for i in N:
            init_total_i = sum(I0.get((p, i), 0.0) for p in P)
            C_cap[i] = max(10000.0, init_total_i * 5.0 + 5000.0)
        C_cap[0] = 1e12   # warehouse: effectively unbounded

        g_rep  = {(p, t): 100000.0 for p in P for t in H}

        instance = IRPTInstance(
            N=N, P=P, H=H, V=V,
            alpha=ROUTING_ALPHA,
            Q=VEHICLE_CAPACITY,
            d=d_achamrah,
            b=b_cost, h=h_cost, C=C_cap, I0=I0, D=D, g=g_rep, f=f_cost,
            name=scenario.scenario_id,
        )
        return instance, store_to_id, sku_to_id, periods_dt

    def run_scenario(
        self,
        scenario: ScenarioSpec,
        df_slice: pd.DataFrame,
        dist_dict: Dict[Tuple[str, str], float],
    ) -> Tuple[ScenarioResult, List[Dict]]:
        """Run Achamrah on the scenario; return (ScenarioResult, lt_moves)."""
        t0 = time.time()
        lt_moves_out: List[Dict] = []

        try:
            from achamrah_2022_irpt_matheuristic import (
                AchamrahIRPTSolver, HeuristicParams,
            )
            from achamrah_integrated_extended_solver import AchamrahIntegratedExtendedSolver
            from gurobipy import GRB

            instance, store_to_id, sku_to_id, periods_dt = self._build_achamrah_instance(
                scenario, df_slice, dist_dict
            )
            id_to_store = {v: k for k, v in store_to_id.items()}
            id_to_sku   = {v: k for k, v in sku_to_id.items()}

            # ── Phase 1+2: run the real Achamrah 2022 matheuristic ────────
            # Constructive phase: RMILP → cluster MILPs → initial routes
            # Improvement phase:  GA+SA hybrid, each candidate evaluated by FMILP
            params = HeuristicParams(
                full_time_limit=float(self.time_limit),
                constructive_time_limit=float(self.time_limit) * 0.25,
                improvement_time_limit=float(self.time_limit) * 0.75,
                rmilp_mipgap=self.mip_gap,
                cluster_mipgap=self.mip_gap,
                fmilp_mipgap=self.mip_gap,
            )
            math_solver = AchamrahIRPTSolver(instance, params=params)
            math_result = math_solver.solve_full_matheuristic()

            best_routes = math_result.best_routes

            # ── Phase 3: evaluate best routes with extended solver ────────
            # Re-solve the FMILP with best_routes fixed to extract cost
            # breakdown and LT moves (using simplified LT mode for thesis match).
            eval_solver = AchamrahIntegratedExtendedSolver(
                instance,
                vehicle_indexed_lt=self.vehicle_indexed_lt,
            )
            artifacts = eval_solver.solve_model(
                time_limit=max(120.0, float(self.time_limit) * 0.15),
                mip_gap=self.mip_gap,
                allow_lateral_transshipment=True,
                fixed_routes=best_routes,
            )

            runtime = time.time() - t0

            # Status: if matheuristic produced a solution use it; otherwise
            # fall back to the evaluation solve status.
            math_obj = math_result.best_objective
            if math.isfinite(math_obj):
                status = "success"
            elif artifacts.status == GRB.OPTIMAL:
                status = "success"
            elif artifacts.status == GRB.TIME_LIMIT and not math.isinf(artifacts.objective):
                status = "success"
            elif artifacts.status == GRB.TIME_LIMIT:
                status = "timeout"
            else:
                status = "failed"

            success = status == "success"

            cb = artifacts.cost_breakdown or {}
            routing_cost      = cb.get("routing",  0.0)
            holding_cost      = cb.get("holding",  0.0)
            shortage_cost     = cb.get("shortage", 0.0)
            lt_cost           = cb.get("lt",       0.0)
            # Prefer matheuristic objective (from GA+SA); fall back to eval
            total_cost = math_obj if math.isfinite(math_obj) else cb.get("total", artifacts.objective)

            # Compute shortage qty from raw model output
            lt_moves_raw = artifacts.lt_moves or []
            lt_total_qty = sum(m.get("quantity", 0) for m in lt_moves_raw)

            # Shortage qty (approximate from demand vs inventory, not directly tracked)
            total_demand = sum(
                instance.D.get((p, i, t), 0.0)
                for p in instance.P for i in instance.N for t in instance.H
            )
            shortage_qty = shortage_cost / max(SHORTAGE_COST_RATE, 1e-9)
            service_level = max(0.0, 1.0 - shortage_qty / max(total_demand, 1.0))

            # Annotate LT moves with store/SKU names
            for mv in lt_moves_raw:
                from_name = id_to_store.get(mv.get("from_store"), str(mv.get("from_store")))
                to_name   = id_to_store.get(mv.get("to_store"),   str(mv.get("to_store")))
                sku_name  = id_to_sku.get(mv.get("product"),      str(mv.get("product")))
                lt_moves_out.append({
                    "scenario_id":  scenario.scenario_id,
                    "method":       self.source_name,
                    "period":       mv.get("period"),
                    "from_store":   from_name,
                    "to_store":     to_name,
                    "sku":          sku_name,
                    "lt_qty":       mv.get("quantity", 0.0),
                    "lt_unit_cost": LT_COST_FLAT,
                    "lt_total_cost": mv.get("cost", 0.0),
                    "vehicle": mv.get("vehicle", "N/A") or "N/A",
                    "achamrah_vehicle_indexed_lt": self.vehicle_indexed_lt,
                })

            if success:
                print(f"  [Achamrah breakdown] routing={routing_cost:.2f}  "
                      f"holding={holding_cost:.2f}  shortage={shortage_cost:.2f}  "
                      f"lt={lt_cost:.2f}  total={total_cost:.2f}")

            result = ScenarioResult(
                scenario_id=scenario.scenario_id,
                method=self.source_name,
                status=status,
                success=success,
                runtime_seconds=runtime,
                total_cost=total_cost if success else float("nan"),
                holding_cost=holding_cost if success else float("nan"),
                routing_cost=routing_cost if success else float("nan"),
                transshipment_cost=lt_cost if success else float("nan"),
                shortage_cost=shortage_cost if success else float("nan"),
                shortage_qty=shortage_qty if success else float("nan"),
                service_level=service_level if success else float("nan"),
                lt_total_qty=lt_total_qty if success else float("nan"),
                n_lt_moves=len(lt_moves_raw) if success else 0,
                n_routes=0,
                achamrah_vehicle_indexed_lt=self.vehicle_indexed_lt,
                num_vars=artifacts.num_vars,
                num_constrs=artifacts.num_constrs,
                mip_gap=artifacts.mip_gap,
            )
            return result, lt_moves_out

        except Exception as exc:
            runtime = time.time() - t0
            tb = traceback.format_exc()
            status = "timeout" if runtime >= self.time_limit * 0.98 else "failed"
            print(f"[Achamrah] {scenario.scenario_id} {status}: {exc}")
            print(f"[Achamrah TRACEBACK]:\n{tb}")
            result = ScenarioResult(
                scenario_id=scenario.scenario_id,
                method=self.source_name,
                status=status,
                success=False,
                runtime_seconds=runtime,
                total_cost=float("nan"), holding_cost=float("nan"),
                routing_cost=float("nan"), transshipment_cost=float("nan"),
                shortage_cost=float("nan"), shortage_qty=float("nan"),
                service_level=float("nan"), lt_total_qty=float("nan"),
                n_lt_moves=0, n_routes=0,
                achamrah_vehicle_indexed_lt=self.vehicle_indexed_lt,
                error_message=str(exc),
                error_traceback=tb,
            )
            return result, []


# ======================================================================
# COMPARISON ENGINE
# ======================================================================

class ComparisonEngine:
    """Compute per-scenario gaps and aggregate statistics."""

    @staticmethod
    def compute_per_scenario_gaps(
        thesis_result: ScenarioResult,
        achamrah_result: ScenarioResult,
    ) -> Dict[str, Any]:
        """Compute cost gap, runtime ratio, and other differences."""
        row: Dict[str, Any] = {
            "scenario_id": thesis_result.scenario_id,
            "thesis_success":   thesis_result.success,
            "achamrah_success": achamrah_result.success,
            "thesis_status":    thesis_result.status,
            "achamrah_status":  achamrah_result.status,
        }

        if thesis_result.success and achamrah_result.success:
            tc  = thesis_result.total_cost
            ac  = achamrah_result.total_cost
            gap = (tc - ac) / max(abs(ac), 1e-9) * 100.0

            row.update({
                "thesis_total_cost":    tc,
                "achamrah_total_cost":  ac,
                "cost_gap_pct":         round(gap, 4),        # positive = thesis more expensive
                "cost_gap_abs":         round(tc - ac, 4),
                "runtime_C":            thesis_result.runtime_seconds,
                "runtime_Achamrah":     achamrah_result.runtime_seconds,
                "runtime_ratio_C_vs_Achamrah": (
                    round(thesis_result.runtime_seconds / max(achamrah_result.runtime_seconds, 1e-3), 4)
                ),
                "thesis_service_level":    thesis_result.service_level,
                "achamrah_service_level":  achamrah_result.service_level,
                "service_level_diff":      round(thesis_result.service_level - achamrah_result.service_level, 6),
                "lt_qty_diff":             round(
                    (thesis_result.lt_total_qty or 0.0) - (achamrah_result.lt_total_qty or 0.0), 4
                ),
                "shortage_cost_diff":      round(
                    (thesis_result.shortage_cost or 0.0) - (achamrah_result.shortage_cost or 0.0), 4
                ),
                "interpretation": (
                    "C cheaper than Achamrah incumbent"
                    if gap < -0.5
                    else "Achamrah cheaper than C"
                    if gap > 0.5
                    else "C and Achamrah within 0.5% of each other"
                ),
            })
        else:
            for k in ["thesis_total_cost", "achamrah_total_cost", "cost_gap_pct", "cost_gap_abs",
                      "runtime_C", "runtime_Achamrah", "runtime_ratio_C_vs_Achamrah",
                      "thesis_service_level", "achamrah_service_level", "service_level_diff",
                      "lt_qty_diff", "shortage_cost_diff"]:
                row[k] = float("nan")
            row["interpretation"] = "comparison not possible (one or both methods failed)"

        return row

    @staticmethod
    def compute_summary(
        per_scenario_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate statistics by method and size_label."""
        methods  = results_df["method"].unique()
        size_labels = []
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
                    "method":          method,
                    "size_label":      label,
                    "n_scenarios":     len(sub),
                    "n_success":       len(success_sub),
                    "n_timeout":       int((sub["status"] == "timeout").sum()),
                    "n_failed":        int((sub["status"] == "failed").sum()),
                    "success_rate":    round(len(success_sub) / max(len(sub), 1), 4),
                    "timeout_rate":    round((sub["status"] == "timeout").sum() / max(len(sub), 1), 4),
                }

                for col, stat_name in [
                    ("runtime_seconds", "runtime"),
                    ("total_cost",       "total_cost"),
                    ("shortage_cost",    "shortage_cost"),
                    ("service_level",    "service_level"),
                    ("lt_total_qty",     "lt_total_qty"),
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

class OutputWriter:
    """Writes all 7 output files for the Achamrah vs C comparison."""

    def __init__(self, output_dir: str):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def _save(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.out / filename
        df.to_csv(path, index=False)
        print(f"[Output] Saved {filename} ({len(df)} rows)")
        return path

    def write_per_scenario(self, rows: List[Dict]) -> Path:
        return self._save(pd.DataFrame(rows), "validate_achamrah_vs_C_per_scenario.csv")

    def write_summary(self, df: pd.DataFrame) -> Path:
        return self._save(df, "validate_achamrah_vs_C_summary.csv")

    def write_cost_breakdown(self, rows: List[Dict]) -> Path:
        cols = ["scenario_id", "method", "success", "total_cost",
                "routing_cost", "holding_cost", "transshipment_cost",
                "shortage_cost", "lt_total_qty", "service_level"]
        df = pd.DataFrame(rows)
        df = df[[c for c in cols if c in df.columns]]
        return self._save(df, "validate_achamrah_vs_C_cost_breakdown.csv")

    def write_runtime(self, rows: List[Dict]) -> Path:
        cols = ["scenario_id", "method", "status", "runtime_seconds",
                "n_columns_generated", "n_columns_selected", "num_vars", "num_constrs", "mip_gap"]
        df = pd.DataFrame(rows)
        df = df[[c for c in cols if c in df.columns]]
        return self._save(df, "validate_achamrah_vs_C_runtime.csv")

    def write_lt_plan(self, lt_moves: List[Dict]) -> Path:
        if not lt_moves:
            df = pd.DataFrame(columns=["scenario_id", "method", "period",
                                        "from_store", "to_store", "sku",
                                        "lt_qty", "lt_unit_cost", "lt_total_cost", "vehicle"])
        else:
            df = pd.DataFrame(lt_moves)
        return self._save(df, "validate_achamrah_vs_C_lt_plan.csv")

    def write_gaps(self, gap_rows: List[Dict]) -> Path:
        return self._save(pd.DataFrame(gap_rows), "validate_achamrah_vs_C_gaps.csv")

    def write_failures(self, rows: List[Dict]) -> Path:
        fail_rows = [r for r in rows if not r.get("success", True)]
        if not fail_rows:
            df = pd.DataFrame(columns=["scenario_id", "method", "status", "error_message"])
        else:
            df = pd.DataFrame(fail_rows)
            df = df[[c for c in ["scenario_id", "method", "status", "runtime_seconds",
                                   "error_message", "error_traceback"] if c in df.columns]]
        return self._save(df, "validate_achamrah_vs_C_failures.csv")

    def write_config(self, config: Dict) -> Path:
        path = self.out / "validate_achamrah_vs_C_config.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"[Output] Saved validate_achamrah_vs_C_config.json")
        return path

    def write_scenario_manifest(self, scenarios: List[ScenarioSpec]) -> Path:
        df = pd.DataFrame([s.to_dict() for s in scenarios])
        return self._save(df, "validate_achamrah_vs_C_scenario_manifest.csv")


# ======================================================================
# VALIDATION ORCHESTRATOR
# ======================================================================

class ValidationOrchestrator:
    """
    Main orchestrator:
    1. Generate scenarios
    2. For each scenario: run Thesis C and Achamrah
    3. Save partial results after each scenario
    4. Compute gaps and summary statistics
    5. Export all 7 output files
    """

    def __init__(
        self,
        data_csv: str,
        checkpoint_path: str,
        output_dir: str,
        dist_path: Optional[str] = None,
        repo_root: Optional[str] = None,
        n_scenarios: int = 30,
        store_limits: Optional[List[int]] = None,
        scenarios_per_size: int = SCENARIOS_PER_SIZE,
        achamrah_vehicle_indexed_lt: bool = False,
        debug: bool = False,
        seed: int = 42,
    ):
        self.data_csv    = data_csv
        self.checkpoint  = checkpoint_path
        self.output_dir  = output_dir
        self.dist_path   = dist_path
        self.repo_root   = repo_root
        self.debug       = debug
        self.seed        = seed

        _scenarios_per_size = DEBUG_N_SCENARIOS if debug else scenarios_per_size
        _store_limits       = store_limits or STORE_LIMITS

        self.scenario_gen = ScenarioGenerator(
            data_csv=data_csv,
            store_limits=_store_limits,
            scenarios_per_size=_scenarios_per_size,
            sku_limit=SKU_LIMIT,
            period_granularity=PERIOD_GRANULARITY,
            window_length=WINDOW_LENGTH_PERIODS,
            base_seed=seed,
        )
        self.data_builder = SharedDataBuilder(
            data_csv=data_csv,
            dist_path=dist_path,
            period_granularity=PERIOD_GRANULARITY,
        )
        self.thesis_runner = ThesisCRunner(
            checkpoint_path=checkpoint_path,
            repo_root=repo_root,
            cg_iterations=50,
            time_limit=THESIS_C_TIME_LIMIT,
        )
        self.achamrah_runner = AchamrahRunner(
            vehicle_indexed_lt=achamrah_vehicle_indexed_lt,
            time_limit=ACHAMRAH_TIME_LIMIT,
            mip_gap=ACHAMRAH_MIP_GAP,
        )
        self.writer = OutputWriter(output_dir)

    def run(self) -> None:
        """Execute the full comparison."""
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
            "data_csv":                  self.data_csv,
            "checkpoint_path":           self.checkpoint,
            "checkpoint_valid":          ckpt_ok,
            "n_scenarios":               len(scenarios),
            "store_limits":              STORE_LIMITS,
            "scenarios_per_size":        SCENARIOS_PER_SIZE,
            "sku_limit":                 SKU_LIMIT,
            "period_granularity":        PERIOD_GRANULARITY,
            "window_length_periods":     WINDOW_LENGTH_PERIODS,
            "achamrah_time_limit":       ACHAMRAH_TIME_LIMIT,
            "thesis_c_time_limit":       THESIS_C_TIME_LIMIT,
            "achamrah_vehicle_indexed_lt": self.achamrah_runner.vehicle_indexed_lt,
            "allow_gnn_fallback":        ALLOW_GNN_FALLBACK,
            "holding_cost_rate":         HOLDING_COST_RATE,
            "shortage_cost_rate":        SHORTAGE_COST_RATE,
            "lt_cost_flat":              LT_COST_FLAT,
            "routing_alpha":             ROUTING_ALPHA,
            "vehicle_count":             VEHICLE_COUNT,
            "vehicle_capacity":          VEHICLE_CAPACITY,
            "debug":                     self.debug,
            "seed":                      self.seed,
        }
        self.writer.write_config(config_record)

        # ── Per-scenario loop ──────────────────────────────────────────
        all_results: List[Dict]  = []
        all_lt_moves: List[Dict] = []
        gap_rows: List[Dict]     = []
        partial_path = Path(self.output_dir) / "validate_achamrah_vs_C_per_scenario.csv"

        print(f"\n{'='*70}")
        print(f"Starting validation: {len(scenarios)} scenarios")
        print(f"Methods: Thesis_C_ALNS_CG_GNN  vs  {self.achamrah_runner.source_name}")
        print(f"{'='*70}\n")

        for idx, scenario in enumerate(scenarios, start=1):
            print(f"\n[{idx:02d}/{len(scenarios)}] Scenario: {scenario.scenario_id}  "
                  f"stores={scenario.store_limit}  skus={scenario.sku_limit}  "
                  f"periods={scenario.n_periods}  window={scenario.start_date}→{scenario.end_date}")

            # Build shared data slice
            df_slice   = self.data_builder.slice_for_scenario(scenario)
            dist_dict, dist_src = self.data_builder.get_distances(scenario.selected_stores)

            if idx == 1:
                print(f"  Distance source: {dist_src}")

            if df_slice.empty:
                print(f"  WARNING: empty data slice; skipping scenario.")
                continue

            # ─ Run Thesis C ──────────────────────────────────────────
            print(f"  [Thesis C]  running...")
            c_result = self.thesis_runner.run_scenario(scenario, df_slice, dist_dict)
            print(f"  [Thesis C]  status={c_result.status}  "
                  f"cost={c_result.total_cost:.2f}  runtime={c_result.runtime_seconds:.1f}s")

            # ─ Run Achamrah ───────────────────────────────────────────
            print(f"  [Achamrah]  running (vehicle_indexed_lt={self.achamrah_runner.vehicle_indexed_lt})...")
            ach_result, lt_moves = self.achamrah_runner.run_scenario(scenario, df_slice, dist_dict)
            print(f"  [Achamrah]  status={ach_result.status}  "
                  f"cost={ach_result.total_cost:.2f}  "
                  f"mip_gap={ach_result.mip_gap:.4f}  runtime={ach_result.runtime_seconds:.1f}s")

            # ─ Annotate results with scenario metadata ────────────────
            for result in [c_result, ach_result]:
                d = result.to_dict()
                d["size_label"]   = scenario.size_label
                d["store_limit"]  = scenario.store_limit
                d["sku_limit"]    = scenario.sku_limit
                d["n_periods"]    = scenario.n_periods
                d["start_date"]   = scenario.start_date
                d["end_date"]     = scenario.end_date
                d["random_seed"]  = scenario.random_seed
                all_results.append(d)

            all_lt_moves.extend(lt_moves)

            # ─ Compute gap ───────────────────────────────────────────
            gap = ComparisonEngine.compute_per_scenario_gaps(c_result, ach_result)
            gap["size_label"] = scenario.size_label
            gap_rows.append(gap)

            if c_result.success and ach_result.success:
                print(f"  [Gap]  cost_gap={gap.get('cost_gap_pct', 'n/a'):.2f}%  "
                      f"runtime_ratio={gap.get('runtime_ratio_C_vs_Achamrah', 'n/a'):.2f}x  "
                      f"→ {gap.get('interpretation', '')}")

            # ─ Partial save ──────────────────────────────────────────
            if all_results:
                pd.DataFrame(all_results).to_csv(partial_path, index=False)

        # ── Aggregate outputs ──────────────────────────────────────────
        print(f"\n{'='*70}")
        print("Writing final outputs...")

        results_df = pd.DataFrame(all_results)
        summary_df = ComparisonEngine.compute_summary(pd.DataFrame(gap_rows), results_df)

        self.writer.write_per_scenario(all_results)
        self.writer.write_summary(summary_df)
        self.writer.write_cost_breakdown(all_results)
        self.writer.write_runtime(all_results)
        self.writer.write_lt_plan(all_lt_moves)
        self.writer.write_gaps(gap_rows)
        self.writer.write_failures(all_results)

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

        # ── Thesis-ready interpretation notes ─────────────────────────
        print(f"\n{'='*70}")
        print("INTERPRETATION NOTES FOR THESIS:")
        print(f"{'='*70}")
        if not results_df.empty:
            c_success   = results_df[results_df["method"] == "Thesis_C_ALNS_CG_GNN"]["success"].mean()
            ach_success = results_df[results_df["method"] == self.achamrah_runner.source_name]["success"].mean()
            print(f"  Thesis C success rate:   {c_success*100:.1f}%")
            print(f"  Achamrah success rate:   {ach_success*100:.1f}%")
            gap_valid = [g for g in gap_rows
                         if not math.isnan(g.get("cost_gap_pct", float("nan")))]
            if gap_valid:
                mean_gap = sum(g["cost_gap_pct"] for g in gap_valid) / len(gap_valid)
                print(f"  Mean cost gap (C vs Achamrah): {mean_gap:+.2f}%  "
                      f"(positive = C more expensive, negative = C cheaper)")
        print(f"  Outputs saved to: {self.output_dir}")


# ======================================================================
# CLI ENTRY POINT
# ======================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate Thesis C vs Achamrah benchmark on test data.csv"
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
    p.add_argument("--vehicle_indexed_lt", action="store_true", default=False,
                   help="Use original Achamrah (vehicle-indexed LT) instead of simplified")
    p.add_argument("--debug",       action="store_true",
                   help="Quick debug run: 3 scenarios per size instead of 10")
    p.add_argument("--seed",        type=int, default=42)
    return p


def main() -> None:
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()  # ignore Jupyter/Papermill kernel args

    # Kaggle env-var overrides
    data_csv    = os.environ.get("VALIDATE_DATA_CSV",    args.data_csv)
    checkpoint  = os.environ.get("CHECKPOINT_PATH",      args.checkpoint)
    output_dir  = os.environ.get("VALIDATE_OUTPUT_DIR",  args.output_dir)
    repo_root   = os.environ.get("REPO_ROOT",            args.repo_root)

    # Try to auto-detect repo root on Kaggle
    if repo_root is None:
        for candidate in [
            "/kaggle/working/Thesis-Work",
            "/kaggle/working/repo",
            str(Path(__file__).parent),
        ]:
            if Path(candidate).exists():
                repo_root = candidate
                break

    # Auto-detect checkpoint from bundle if not found
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

    # Auto-detect distance matrix
    dist_path = args.dist_path
    if dist_path is None and repo_root:
        dp = Path(repo_root) / DIST_REL_PATH
        if dp.exists():
            dist_path = str(dp)
            print(f"[Main] Auto-detected distance matrix: {dist_path}")

    print(f"\n{'='*70}")
    print("Validate_with_Achamrah_Kaggle.py")
    print(f"{'='*70}")
    print(f"  data_csv:         {data_csv}")
    print(f"  checkpoint:       {checkpoint}")
    print(f"  output_dir:       {output_dir}")
    print(f"  dist_path:        {dist_path or 'synthetic (fallback)'}")
    print(f"  repo_root:        {repo_root or 'current dir'}")
    print(f"  store_limits:     {args.store_limits}")
    print(f"  scenarios_per_size: {args.scenarios_per_size}")
    print(f"  vehicle_indexed_lt: {args.vehicle_indexed_lt}")
    print(f"  debug:            {args.debug}")
    print(f"  achamrah_time_limit: {ACHAMRAH_TIME_LIMIT}s")
    print(f"  thesis_c_time_limit: {THESIS_C_TIME_LIMIT}s")
    print(f"{'='*70}\n")

    orchestrator = ValidationOrchestrator(
        data_csv=data_csv,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        dist_path=dist_path,
        repo_root=repo_root,
        store_limits=args.store_limits,
        scenarios_per_size=args.scenarios_per_size,
        achamrah_vehicle_indexed_lt=args.vehicle_indexed_lt,
        debug=args.debug,
        seed=args.seed,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
