"""
compare_models_6scenarios.py
==============================
Benchmark ALNS-CG (Thesis Variant C) vs Achamrah (2022) matheuristic on
6 hand-picked (store, sku, period) scenarios.

Goal: prove that ALNS-CG handles instance sizes that Achamrah's flat MIP
cannot — by reporting num_vars / num_constrs / runtime for both methods.

The 6 scenarios:
    (4,3,30), (6,4,30), (7,5,60), (10,5,60), (7,5,120), (10,5,120)

Modes:
    --mode count   : just BUILD models, report num_vars/num_constrs (fast).
    --mode quick   : also solve with 60s time limit each (sanity check).
    --mode full    : solve with full time limits (Achamrah=1200s, ALNS-CG=600s).

Usage:
    python compare_models_6scenarios.py --mode count
    python compare_models_6scenarios.py --mode quick
    python compare_models_6scenarios.py --mode full --output_dir Results/6scenarios
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG: 6 scenarios. (stores, skus, periods).
# ---------------------------------------------------------------------------
SCENARIOS: List[Tuple[str, int, int, int]] = [
    ("S1_4s3p30",   4, 3,  30),
    ("S2_6s4p30",   6, 4,  30),
    ("S3_7s5p60",   7, 5,  60),
    ("S4_10s5p60", 10, 5,  60),
    ("S5_7s5p120",  7, 5, 120),
    ("S6_10s5p120",10, 5, 120),
]

# Reuse the cost / vehicle config from Validate_with_Achamrah_Kaggle.py so
# numbers stay comparable with the existing benchmark run.
HOLDING_COST_RATE  = 0.1
SHORTAGE_COST_RATE = 2.0
LT_COST_FLAT       = 0.6
ROUTING_ALPHA      = 1.0
VEHICLE_CAPACITY   = 1500.0


def vehicle_count_for_stores(n_stores: int) -> int:
    if n_stores <= 5:  return 2
    if n_stores <= 8:  return 3
    return 4


# ---------------------------------------------------------------------------
# DATA SLICE: top-N stores, top-N SKUs, first T daily periods
# ---------------------------------------------------------------------------
def load_daily_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[["SITE_NAME", "ART_SV_NAME_ENG", "SALE_QTY", "END_QTY", "PERIOD"]].copy()
    df.columns = ["store", "sku", "sale_qty", "end_qty", "period_raw"]
    df["store"]    = df["store"].astype(str).str.strip()
    df["sku"]      = df["sku"].astype(str).str.strip()
    df["sale_qty"] = pd.to_numeric(df["sale_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["end_qty"]  = pd.to_numeric(df["end_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["date"]     = pd.to_datetime(df["period_raw"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    # Daily aggregation
    df["period_date"] = df["date"]
    return (
        df.groupby(["store", "sku", "period_date"], as_index=False)
          .agg(sale_qty=("sale_qty", "sum"), end_qty=("end_qty", "last"))
    )


def slice_for_scenario(
    df_full: pd.DataFrame,
    n_stores: int,
    n_skus: int,
    n_periods: int,
) -> Tuple[pd.DataFrame, List[str], List[str], List[pd.Timestamp]]:
    """Select top-N stores by total volume, top-N SKUs, first T daily dates."""
    # Top stores by total sale_qty
    store_demand = df_full.groupby("store")["sale_qty"].sum().sort_values(ascending=False)
    selected_stores = list(store_demand.head(n_stores).index)
    # Top SKUs
    sku_demand = (
        df_full[df_full["store"].isin(selected_stores)]
        .groupby("sku")["sale_qty"].sum().sort_values(ascending=False)
    )
    selected_skus = list(sku_demand.head(n_skus).index)
    # First N consecutive daily periods
    all_dates = sorted(df_full["period_date"].unique())
    selected_dates = all_dates[:n_periods]
    mask = (
        df_full["store"].isin(selected_stores)
        & df_full["sku"].isin(selected_skus)
        & df_full["period_date"].isin(selected_dates)
    )
    return df_full[mask].copy().reset_index(drop=True), selected_stores, selected_skus, selected_dates


def synthetic_distances(stores: List[str], seed: int = 42) -> Dict[Tuple[str, str], float]:
    n = len(stores) + 1
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n, 2))
    nodes = ["__WAREHOUSE__"] + list(stores)
    out: Dict[Tuple[str, str], float] = {}
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i != j:
                out[(ni, nj)] = float(np.linalg.norm(coords[i] - coords[j]))
    return out


# ---------------------------------------------------------------------------
# ACHAMRAH builder
# ---------------------------------------------------------------------------
def build_achamrah_instance(
    df_slice: pd.DataFrame,
    selected_stores: List[str],
    selected_skus: List[str],
    selected_dates: List[pd.Timestamp],
    dist_dict: Dict[Tuple[str, str], float],
    scenario_id: str,
):
    from achamrah_2022_irpt_matheuristic import IRPTInstance

    H = list(range(1, len(selected_dates) + 1))
    period_map = {d: h for h, d in enumerate(selected_dates, start=1)}
    store_to_id = {s: i + 1 for i, s in enumerate(selected_stores)}
    sku_to_id   = {p: i     for i, p in enumerate(selected_skus)}
    N = list(store_to_id.values())
    P = list(sku_to_id.values())
    V = list(range(1, vehicle_count_for_stores(len(selected_stores)) + 1))

    # Demand
    D: Dict = {(p, i, t): 0.0 for p in P for i in N for t in H}
    for _, row in df_slice.iterrows():
        if row["store"] not in store_to_id or row["sku"] not in sku_to_id:
            continue
        h = period_map.get(row["period_date"])
        if h is None:
            continue
        D[(sku_to_id[row["sku"]], store_to_id[row["store"]], h)] = float(row["sale_qty"])

    # I0
    I0: Dict = {}
    for p_name, p_id in sku_to_id.items():
        for s_name, i_id in store_to_id.items():
            mask = (df_slice["store"] == s_name) & (df_slice["sku"] == p_name)
            I0[(p_id, i_id)] = float(df_slice[mask]["end_qty"].iloc[0]) if mask.any() else 0.0
    for p in P:
        I0[(p, 0)] = 10000.0

    # Distances (node-id form)
    nodes = ["__WAREHOUSE__"] + selected_stores
    d_ach: Dict = {}
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i == j:
                continue
            d_ach[(i, j)] = dist_dict.get((ni, nj), 100.0)

    h_cost = {(p, i): HOLDING_COST_RATE  for p in P for i in N}
    h_cost.update({(p, 0): 0.0 for p in P})
    f_cost = {(p, i): SHORTAGE_COST_RATE for p in P for i in N}
    f_cost.update({(p, 0): 0.0 for p in P})
    b_cost = {(i, j): LT_COST_FLAT for i in N for j in N if i != j}

    C_cap: Dict = {}
    for i in N:
        init_total = sum(I0.get((p, i), 0.0) for p in P)
        C_cap[i] = max(10000.0, init_total * 5.0 + 5000.0)
    C_cap[0] = 1e12

    g_rep = {(p, t): 100000.0 for p in P for t in H}

    return IRPTInstance(
        N=N, P=P, H=H, V=V,
        alpha=ROUTING_ALPHA, Q=VEHICLE_CAPACITY,
        d=d_ach, b=b_cost, h=h_cost, C=C_cap, I0=I0,
        D=D, g=g_rep, f=f_cost,
        name=scenario_id,
    )


def measure_achamrah(instance, *, solve: bool, time_limit: float) -> Dict[str, Any]:
    """Build the Achamrah MIP. Optionally solve. Always report var/constr counts."""
    from achamrah_2022_irpt_matheuristic import AchamrahIRPTSolver, HeuristicParams

    out: Dict[str, Any] = {
        "build_seconds": 0.0,
        "solve_seconds": 0.0,
        "num_vars": 0,
        "num_int_vars": 0,
        "num_constrs": 0,
        "objective": float("nan"),
        "status": "build_only",
        "mip_gap": float("nan"),
        "use_v20": (len(instance.H) <= 20),
    }

    solver = AchamrahIRPTSolver(instance, params=HeuristicParams())

    # Build the full MIP only (no solve) — same path solve_model uses.
    n_periods = len(instance.H)
    use_v20 = n_periods <= 20

    t0 = time.time()
    m, _ = solver.build_model(
        relaxed=False,
        fixed_routes=None,
        active_nodes_by_period=None,
        allow_lateral_transshipment=True,
        use_valid_16_19=True,
        use_valid_20=use_v20,
        model_name="IRPT_size_check",
    )
    m.update()
    out["build_seconds"] = time.time() - t0
    out["num_vars"] = int(m.NumVars)
    out["num_int_vars"] = int(m.NumIntVars)
    out["num_constrs"] = int(m.NumConstrs)

    if not solve:
        return out

    # Run the full matheuristic (constructive + improvement) — this is the
    # method actually used in the comparison script.
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            params = HeuristicParams(
                full_time_limit=float(time_limit),
                constructive_time_limit=float(time_limit) * 0.25,
                improvement_time_limit=float(time_limit) * 0.75,
            )
            mh_solver = AchamrahIRPTSolver(instance, params=params)
            res = mh_solver.solve_full_matheuristic()
        out["solve_seconds"] = time.time() - t0
        out["objective"] = float(res.best_objective) if math.isfinite(res.best_objective) else float("inf")
        out["status"] = "solved" if math.isfinite(out["objective"]) else "no_solution"
        if res.final_solution is not None and getattr(res.final_solution, "model", None) is not None:
            out["mip_gap"] = float(getattr(res.final_solution.model, "MIPGap", float("nan")))
    except Exception as e:
        out["solve_seconds"] = time.time() - t0
        out["status"] = f"error:{type(e).__name__}"
        out["error_msg"] = str(e)[:200]
    return out


# ---------------------------------------------------------------------------
# ALNS-CG builder & runner
# ---------------------------------------------------------------------------
def build_irp_data(
    df_slice: pd.DataFrame,
    selected_stores: List[str],
    selected_skus: List[str],
    dist_dict: Dict[Tuple[str, str], float],
    scenario_id: str,
):
    """Convert slice into IRPData using DatasetToIRPValidationMapper."""
    import irp_gurobi_converted as irp

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    df_out = df_slice.copy()
    df_out["PERIOD"] = df_out["period_date"].dt.strftime("%Y%m%d").astype(int)
    df_out["NORMAL_PRICE"] = 1.0
    df_out = df_out.rename(columns={
        "store":    "SITE_NAME",
        "sku":      "ART_SV_NAME_ENG",
        "sale_qty": "SALE_QTY",
        "end_qty":  "END_QTY",
    })
    df_out[["SITE_NAME", "NORMAL_PRICE", "ART_SV_NAME_ENG",
            "SALE_QTY", "END_QTY", "PERIOD"]].to_csv(tmp.name, index=False)
    tmp.close()
    try:
        mapper = irp.DatasetToIRPValidationMapper(
            excel_path=tmp.name, store_limit=None, sku_limit=None,
        )
        data, *_ = mapper.build_irp_data(
            vehicle_count=vehicle_count_for_stores(len(selected_stores)),
            vehicle_capacity=VEHICLE_CAPACITY,
            shortage_cost_rate=SHORTAGE_COST_RATE,
            holding_cost_rate=HOLDING_COST_RATE,
            lt_ship_cost_flat=LT_COST_FLAT,
            distance_matrix_path=None,
        )
        # Fairness overrides (mirror Validate_with_Achamrah_Kaggle.py)
        for s in data.stores:
            for p in data.products:
                data.holding_cost_store[(s, p)] = HOLDING_COST_RATE
                data.shortage_cost[(s, p)]      = SHORTAGE_COST_RATE
                data.ship_cost_cw[(s, p)]       = 0.0
        for p in data.products:
            data.holding_cost_wh[p] = 0.0
        data.vehicle_fixed_cost = 0.0

        # Distance override to match Achamrah's synthetic distances.
        wh_alias = "__WAREHOUSE__"
        all_nodes = [data.warehouse] + data.stores
        for i in all_nodes:
            for j in all_nodes:
                if i == j:
                    continue
                key = (wh_alias if i == data.warehouse else i,
                       wh_alias if j == data.warehouse else j)
                if key in dist_dict:
                    data.distance[(i, j)] = float(dist_dict[key])
        data.realized_demand = dict(data.demand)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return data


def measure_alns_cg_sizes(data) -> Dict[str, Any]:
    """Build ALNS-CG's two Gurobi models (Phase 1 routing MIP + a sample CG
    pricing subproblem) and report their sizes.

    Phase 1 MIP = AchamrahFullIRPTModel with allow_lateral_transshipment=False
                  (LT vars created but ub forced to 0 → effective binary count
                  is just routing).
    CG pricing  = exact MIP solved per (product, period). We use S×(S-1) as
                  the worst-case pair count.
    """
    import gurobipy as gp
    from gurobipy import GRB
    import irp_gurobi_converted as irp

    out: Dict[str, Any] = {}

    # ---- Phase 1 routing MIP size ----
    # We piggy-back on AchamrahFullIRPTModel's solve() but cut it off
    # right after model construction.
    # Easiest: construct the model ourselves using its same env.
    # Instead, just call solve(time_limit=1) with no LT — Gurobi will start
    # solving but we read NumVars/NumConstrs immediately afterwards.
    # Cleaner approach: call the underlying build (replicated below).
    # For honesty, we directly build the standalone Achamrah MIP again with
    # LT disabled, since irp.AchamrahFullIRPTModel composes constraints in a
    # similar way.
    # NOTE: We avoid re-importing the Achamrah builder; instead we just re-use
    # Achamrah's IRPTInstance counts which match.
    # But the truer-to-thesis-C approach is to build irp.AchamrahFullIRPTModel
    # via Gurobi callbacks. We instead just record S, P, T, V analytically:
    n_stores   = len(data.stores)
    n_products = len(data.products)
    n_periods  = len(data.periods)
    n_vehicles = len(data.vehicles)

    # ---- CG pricing subproblem worst case ----
    # From irp_gurobi_converted.py:5223-5224:
    #   q = mdl.addVars(pairs, lb=0.0, vtype=CONTINUOUS)
    #   y = mdl.addVars(pairs, lb=0.0, ub=1.0, vtype=BINARY)
    # |pairs| ≤ S × (S - 1)
    pricing_max_pairs    = n_stores * (n_stores - 1)
    pricing_binary_vars  = pricing_max_pairs
    pricing_total_vars   = 2 * pricing_max_pairs  # q (cont) + y (bin)
    pricing_calls_per_iter = n_products * n_periods

    out["n_stores"]   = n_stores
    out["n_products"] = n_products
    out["n_periods"]  = n_periods
    out["n_vehicles"] = n_vehicles
    out["pricing_max_pairs"]      = pricing_max_pairs
    out["pricing_max_binary"]     = pricing_binary_vars
    out["pricing_max_total_vars"] = pricing_total_vars
    out["pricing_calls_per_cg_iter"] = pricing_calls_per_iter

    # ---- Phase 1 MIP via AchamrahFullIRPTModel ----
    # Build the model using a tiny helper that calls Gurobi and returns the
    # built (un-optimized) Gurobi model.
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            phase1 = irp.AchamrahFullIRPTModel(data)
            # Tiny hack: monkey-patch m.optimize to a no-op so we can read
            # NumVars/NumConstrs without solving.
            orig_opt = gp.Model.optimize
            captured: Dict[str, Any] = {}
            def _cap(self_m, *a, **kw):
                self_m.update()
                captured["NumVars"]    = int(self_m.NumVars)
                captured["NumIntVars"] = int(self_m.NumIntVars)
                captured["NumConstrs"] = int(self_m.NumConstrs)
                # Don't actually call the solver
                raise RuntimeError("__SIZE_CAPTURED__")
            gp.Model.optimize = _cap
            try:
                phase1.solve(
                    msg=False, time_limit=1,
                    allow_lateral_transshipment=False,
                    cw_dispatch_cycle=1,
                )
            except RuntimeError as e:
                if "__SIZE_CAPTURED__" not in str(e):
                    raise
            finally:
                gp.Model.optimize = orig_opt
        out["phase1_num_vars"]    = captured.get("NumVars", -1)
        out["phase1_num_int_vars"]= captured.get("NumIntVars", -1)
        out["phase1_num_constrs"] = captured.get("NumConstrs", -1)
    except Exception as e:
        out["phase1_num_vars"] = -1
        out["phase1_num_constrs"] = -1
        out["phase1_error"] = f"{type(e).__name__}:{str(e)[:120]}"
    out["phase1_build_seconds"] = time.time() - t0
    return out


def run_alns_cg(data, *, time_limit: float, gnn_checkpoint: Optional[str]) -> Dict[str, Any]:
    """Run a simplified ALNS-CG pipeline: Phase 1 routing MIP → ALNS → CG.
    No GNN / no iter-2 feedback — we just want a usable runtime + objective.
    """
    import irp_gurobi_converted as irp

    out: Dict[str, Any] = {
        "phase1_seconds": 0.0,
        "alns_seconds":   0.0,
        "cg_seconds":     0.0,
        "total_seconds":  0.0,
        "objective":      float("nan"),
        "cg_iterations":  0,
        "columns_generated": 0,
        "status":         "ok",
    }

    mip_budget  = max(15.0, time_limit * 0.25)
    alns_budget = max(10.0, time_limit * 0.10)
    cg_budget   = max(15.0, time_limit * 0.50)

    t_start = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            # Phase 1 — MIP routing without LT.
            t0 = time.time()
            mip_sol = None
            try:
                mip_sol = irp.AchamrahFullIRPTModel(data).solve(
                    msg=False,
                    time_limit=int(mip_budget),
                    allow_lateral_transshipment=False,
                    cw_dispatch_cycle=1,
                )
            except Exception:
                mip_sol = None
            out["phase1_seconds"] = time.time() - t0
            mip_ok = mip_sol is not None and getattr(mip_sol, "objective", float("inf")) < float("inf")

            # Phase 2 — ALNS warm-started from MIP (or greedy).
            t0 = time.time()
            bsol = irp.BaselineALNSModel(data).solve(
                msg=False,
                time_limit=int(alns_budget),
                allow_lateral_transshipment=False,
                cw_dispatch_cycle=1,
                max_iterations=5000,
                seed=42,
                initial_solution=mip_sol if mip_ok else None,
            )
            out["alns_seconds"] = time.time() - t0

            # Phase 3 — CG for LT recourse.
            t0 = time.time()
            pipe = irp.IRPResearchPipeline(data)
            res = pipe.run_lt_recourse_from_baseline(
                bsol,
                use_random_initial_patterns=True,
                n_initial_patterns_per_product_period=5,
                cg_iterations=1,           # single CG iteration is enough for size demo
                msg=False,
                gnn_checkpoint=gnn_checkpoint,
                use_gnn=bool(gnn_checkpoint),
                runtime_gnn_mode=bool(gnn_checkpoint),
                collect_teacher_mode=False,
                use_classical_fallback=True,
                lt_activation_threshold=1.0,
            )
            out["cg_seconds"] = time.time() - t0

            cg_sol = res.get("cg_solution")
            if cg_sol is not None:
                out["cg_iterations"]     = int(getattr(cg_sol, "n_iterations", 0))
                out["columns_generated"] = int(getattr(cg_sol, "columns_generated", 0))

            with_lt_breakdown = res.get("realized_with_lt_cost_breakdown", {})
            obj = with_lt_breakdown.get("total_realized_operating_cost")
            if obj is not None and math.isfinite(obj):
                out["objective"] = float(obj)
    except Exception as e:
        out["status"] = f"error:{type(e).__name__}"
        out["error_msg"] = str(e)[:200]
        out["traceback"] = traceback.format_exc()[-500:]
    out["total_seconds"] = time.time() - t_start
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["count", "quick", "full"], default="count",
                   help="count = build only; quick = solve 60s; full = real time limits")
    p.add_argument("--data_csv", default="test data.csv")
    p.add_argument("--output_dir", default="Results/6scenarios")
    p.add_argument("--gnn_checkpoint",
                   default="GNN/trained_models/irplt_teacher_E2_filtered_3epochs/bigat/pairwise_rank/best_model.pt",
                   help="Optional GNN checkpoint for the ALNS-CG run.")
    p.add_argument("--scenarios", default="all",
                   help="Comma-separated scenario names to run (default: all). "
                        "Options: " + ", ".join(s[0] for s in SCENARIOS))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] Mode: {args.mode}")
    print(f"[main] Loading {args.data_csv} (daily granularity)…")
    df_full = load_daily_data(args.data_csv)
    print(f"[main] {df_full['store'].nunique()} stores, "
          f"{df_full['sku'].nunique()} SKUs, "
          f"{df_full['period_date'].nunique()} daily periods available.")

    if args.mode == "quick":
        ach_tl, alns_tl = 60.0, 60.0
    elif args.mode == "full":
        ach_tl, alns_tl = 1200.0, 600.0
    else:
        ach_tl = alns_tl = 0.0

    selected_names = (
        [s[0] for s in SCENARIOS] if args.scenarios == "all"
        else [s.strip() for s in args.scenarios.split(",")]
    )

    rows: List[Dict[str, Any]] = []
    for name, S, P, T in SCENARIOS:
        if name not in selected_names:
            continue
        print(f"\n========== {name}  (S={S}, P={P}, T={T}) ==========")
        slice_df, sel_st, sel_sk, sel_dt = slice_for_scenario(df_full, S, P, T)
        if slice_df.empty:
            print(f"  SKIP — no data for this slice")
            continue

        dist = synthetic_distances(sel_st)

        # ---- Achamrah ----
        try:
            ach_inst = build_achamrah_instance(slice_df, sel_st, sel_sk, sel_dt, dist, name)
            ach_res  = measure_achamrah(ach_inst, solve=(args.mode != "count"), time_limit=ach_tl)
            print(f"  [Achamrah]  num_vars={ach_res['num_vars']:>9,d}   "
                  f"num_int_vars={ach_res['num_int_vars']:>7,d}   "
                  f"num_constrs={ach_res['num_constrs']:>9,d}   "
                  f"build={ach_res['build_seconds']:.1f}s")
            if args.mode != "count":
                print(f"              status={ach_res['status']}  "
                      f"obj={ach_res['objective']:.2f}  "
                      f"solve={ach_res['solve_seconds']:.1f}s  "
                      f"mip_gap={ach_res.get('mip_gap', float('nan')):.3f}")
        except Exception as e:
            ach_res = {"status": f"build_error:{type(e).__name__}", "error_msg": str(e)[:200]}
            print(f"  [Achamrah]  BUILD ERROR: {e}")

        # ---- ALNS-CG ----
        try:
            data = build_irp_data(slice_df, sel_st, sel_sk, dist, name)
            sizes = measure_alns_cg_sizes(data)
            print(f"  [ALNS-CG]   phase1_num_vars={sizes.get('phase1_num_vars', '?')}   "
                  f"phase1_num_int_vars={sizes.get('phase1_num_int_vars', '?')}   "
                  f"phase1_num_constrs={sizes.get('phase1_num_constrs', '?')}")
            print(f"              CG pricing per call (worst-case): "
                  f"{sizes['pricing_max_binary']} binary, "
                  f"{sizes['pricing_max_total_vars']} total vars")
            print(f"              CG pricing calls per iter: {sizes['pricing_calls_per_cg_iter']}")
            cg_res = {}
            if args.mode != "count":
                cg_res = run_alns_cg(data, time_limit=alns_tl,
                                      gnn_checkpoint=(args.gnn_checkpoint
                                                      if Path(args.gnn_checkpoint).exists()
                                                      else None))
                print(f"              status={cg_res['status']}  "
                      f"obj={cg_res['objective']:.2f}  "
                      f"phase1={cg_res['phase1_seconds']:.1f}s  "
                      f"alns={cg_res['alns_seconds']:.1f}s  "
                      f"cg={cg_res['cg_seconds']:.1f}s  "
                      f"total={cg_res['total_seconds']:.1f}s  "
                      f"cg_iters={cg_res['cg_iterations']}  "
                      f"columns={cg_res['columns_generated']}")
        except Exception as e:
            sizes  = {"phase1_num_vars": -1, "phase1_num_constrs": -1}
            cg_res = {"status": f"build_error:{type(e).__name__}", "error_msg": str(e)[:200]}
            print(f"  [ALNS-CG]   BUILD ERROR: {e}")

        rows.append({
            "scenario":    name,
            "stores":      S,
            "products":    P,
            "periods":     T,
            "vehicles":    vehicle_count_for_stores(S),
            # Achamrah
            "ach_num_vars":      ach_res.get("num_vars"),
            "ach_num_int_vars":  ach_res.get("num_int_vars"),
            "ach_num_constrs":   ach_res.get("num_constrs"),
            "ach_use_v20":       ach_res.get("use_v20"),
            "ach_build_s":       ach_res.get("build_seconds"),
            "ach_solve_s":       ach_res.get("solve_seconds"),
            "ach_objective":     ach_res.get("objective"),
            "ach_status":        ach_res.get("status"),
            "ach_mip_gap":       ach_res.get("mip_gap"),
            # ALNS-CG
            "cg_phase1_num_vars":     sizes.get("phase1_num_vars"),
            "cg_phase1_num_int_vars": sizes.get("phase1_num_int_vars"),
            "cg_phase1_num_constrs":  sizes.get("phase1_num_constrs"),
            "cg_pricing_max_binary":  sizes.get("pricing_max_binary"),
            "cg_pricing_max_total":   sizes.get("pricing_max_total_vars"),
            "cg_pricing_calls":       sizes.get("pricing_calls_per_cg_iter"),
            "cg_phase1_s":            cg_res.get("phase1_seconds"),
            "cg_alns_s":              cg_res.get("alns_seconds"),
            "cg_cg_s":                cg_res.get("cg_seconds"),
            "cg_total_s":             cg_res.get("total_seconds"),
            "cg_objective":           cg_res.get("objective"),
            "cg_status":              cg_res.get("status"),
            "cg_iterations":          cg_res.get("cg_iterations"),
            "cg_columns_gen":         cg_res.get("columns_generated"),
        })

    # ---- save outputs ----
    df_out = pd.DataFrame(rows)
    csv_path = out_dir / f"comparison_{args.mode}.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n[main] Wrote {csv_path}")

    def _fmt_int(v): return f"{v:,}" if isinstance(v, (int, float)) and v is not None and not (isinstance(v, float) and math.isnan(v)) else "—"
    def _fmt_flt(v, p=2): return f"{v:.{p}f}" if isinstance(v, (int, float)) and v is not None and not (isinstance(v, float) and math.isnan(v)) else "—"

    md_path = out_dir / f"comparison_{args.mode}.md"
    with open(md_path, "w") as f:
        f.write(f"# ALNS-CG vs Achamrah — 6 scenarios ({args.mode} mode)\n\n")
        f.write("## Variable counts (Gurobi MIP size)\n\n")
        f.write("| Scenario | (S,P,T) | Achamrah NumVars | Achamrah Bin | "
                "Achamrah Constrs | v20 active | CG Phase1 NumVars | "
                "CG Pricing Bin/call | CG Pricing calls/iter |\n")
        f.write("|---|---|---:|---:|---:|:-:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['scenario']} | ({r['stores']},{r['products']},{r['periods']}) "
                    f"| {_fmt_int(r.get('ach_num_vars'))} | {_fmt_int(r.get('ach_num_int_vars'))} "
                    f"| {_fmt_int(r.get('ach_num_constrs'))} | {'✓' if r.get('ach_use_v20') else '✗'} "
                    f"| {_fmt_int(r.get('cg_phase1_num_vars'))} "
                    f"| {_fmt_int(r.get('cg_pricing_max_binary'))} "
                    f"| {_fmt_int(r.get('cg_pricing_calls'))} |\n")
        if args.mode != "count":
            f.write("\n## Runtime / objective\n\n")
            f.write("| Scenario | Achamrah obj | Achamrah solve(s) | gap | "
                    "ALNS-CG obj | ALNS-CG total(s) | CG iters |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(f"| {r['scenario']} | "
                        f"{_fmt_flt(r.get('ach_objective'))} | {_fmt_flt(r.get('ach_solve_s'), 1)} | "
                        f"{_fmt_flt(r.get('ach_mip_gap'), 3)} | "
                        f"{_fmt_flt(r.get('cg_objective'))} | "
                        f"{_fmt_flt(r.get('cg_total_s'), 1)} | "
                        f"{_fmt_int(r.get('cg_iterations'))} |\n")
    print(f"[main] Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
