"""
IRP / IRPT thesis prototype with validation split
=================================================

Contents (top-to-bottom)
------------------------
- DatasetToIRPValidationMapper        : Excel/CSV → IRPData + validation target
- IRPData / CGSolution / FullIRPTSolution : dataclasses
- BaselineALNSModel                   : ALNS baseline (DC→stores→DC, no LT)
- AchamrahFullIRPTModel               : Legacy Gurobi baseline (kept for reference)
- LateralTransshipmentCG              : CG + RMP + pricing + B&P, optional GNN/heuristic scorer
- IRPResearchPipeline.run             : full pipeline orchestrator (baseline → post-shock shock → CG LT)
- Post-processing helpers             : LT plan, cost breakdowns, realized inventory
- Teacher/GNN bridge                  : run_teacher_graph_and_gnn_training (offline train + offline test)
- run_three_way_benchmark             : Classical vs Heuristic vs GNN benchmark runner
- __main__                            : reads env vars, runs pipeline or benchmark

Inputs
------
- Excel/CSV dataset at EXCEL_PATH (set in __main__)
- Optional distance matrix CSV (DatasetToIRPValidationMapper)

Outputs (written under Results/)
--------------------------------
- irp_validation_target.csv, irp_baseline_routes.csv, irp_solver_efficiency_metrics.csv
- irp_baseline_cost_breakdown.csv, irp_realized_operating_cost_breakdown.csv
- irp_lt_plan.csv, irp_hidden_demand_shock_summary.csv, irp_alns_history.csv
- irp_gnn_training_history.csv, irp_gnn_offline_test.csv, chart_*.png
- irp_benchmark_comparison.csv (only if IRP_RUN_BENCHMARK=1)

Environment variables read by __main__ (all optional, all have defaults)
-----------------------------------------------------------------------
- IRP_CG_ITERATIONS, IRP_GNN_TRAIN_EPOCHS, IRP_TIME_LIMIT
- IRP_USE_BRANCH_AND_PRICE, IRP_BP_MAX_NODES, IRP_BP_MAX_DEPTH
- IRP_COLLECT_TEACHER_MODE, IRP_RUNTIME_GNN_MODE
- IRP_BUILD_TEACHER_GRAPHS, IRP_TRAIN_GNN_AFTER_TEACHER, IRP_RESUME_GNN_CHECKPOINT
- IRP_GNN_CHECKPOINT
- IRP_LT_ACTIVATION_THRESHOLD
- IRP_DEMAND_SHOCK_PROBABILITY, IRP_DEMAND_SHOCK_REALLOCATION_FRACTION,
  IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD,
  IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER, IRP_DEMAND_SHOCK_SEED
- IRP_STORE_INIT_MULTIPLIER, IRP_CLEAN_RESULTS
- IRP_LT_COST_MULTIPLIER                 : sensitivity multiplier on transship_unit_cost
- IRP_ENFORCE_INTEGER                    : force integer delivery + inventory + LT flows
- IRP_RUN_BENCHMARK                      : run A0/A/B/C benchmark and exit
- IRP_HEURISTIC_TOP_K                    : top-k cut-off for heuristic variant B
- IRP_ONLINE_INFERENCE                   : set to 1 to run Stage 2 — loads "test data.csv" + pre-trained GNN, no retraining
- IRP_ONLINE_LEARNING                    : set to 1 (alongside IRP_ONLINE_INFERENCE=1) to enable solver-supervised online
                                           fine-tuning after each inference run; collects new teacher rows from the inference
                                           CG episodes and fine-tunes the checkpoint in-place (resume_checkpoint=True).
                                           Defaults to 0 — offline training + pure inference is preferred for thesis
                                           reproducibility; online learning should only be activated after the model has
                                           been validated in inference-only mode first.
- IRP_ONLINE_LEARNING_EPOCHS             : fine-tuning epochs for online learning (default 2; keep small to avoid overfitting
                                           to a single inference run)
- IRP_DATASET_PATH                       : explicit override for the input CSV/Excel path (overrides both stage defaults)

Dependencies
------------
pip install pandas openpyxl gurobipy torch
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any
import datetime
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import pprint
import subprocess
import sys
import time
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    raise ImportError("Install gurobipy first: pip install gurobipy") from e

GLOBAL_GUROBI_ENV: Optional[gp.Env] = None


def _read_gurobi_lic() -> dict:
    """Read WLSACCESSID/WLSSECRET/LICENSEID from ~/gurobi.lic if not in env."""
    import pathlib
    lic_path = pathlib.Path.home() / "gurobi.lic"
    result = {}
    if lic_path.exists():
        for line in lic_path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                result[k.strip().upper()] = v.strip()
    return result


def create_gurobi_env() -> gp.Env:
    env = gp.Env(empty=True)
    wls_id = os.environ.get("WLSACCESSID")
    wls_secret = os.environ.get("WLSSECRET")
    license_id = os.environ.get("LICENSEID")
    if not (wls_id and wls_secret and license_id):
        lic = _read_gurobi_lic()
        wls_id = wls_id or lic.get("WLSACCESSID")
        wls_secret = wls_secret or lic.get("WLSSECRET")
        license_id = license_id or lic.get("LICENSEID")
    if wls_id and wls_secret and license_id:
        env.setParam("WLSAccessID", wls_id)
        env.setParam("WLSSecret", wls_secret)
        env.setParam("LicenseID", int(license_id))
    env.start()
    return env


def get_gurobi_env() -> gp.Env:
    global GLOBAL_GUROBI_ENV
    if GLOBAL_GUROBI_ENV is None:
        GLOBAL_GUROBI_ENV = create_gurobi_env()
    return GLOBAL_GUROBI_ENV

Store = str
Product = str
Period = int
Node = str
Vehicle = str


def _grb_status_name(status: int) -> str:
    mapping = {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.UNBOUNDED: "Unbounded",
        GRB.INF_OR_UNBD: "InfOrUnbd",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.INTERRUPTED: "Interrupted",
        GRB.SUBOPTIMAL: "Suboptimal",
        GRB.NUMERIC: "Numeric",
    }
    return mapping.get(status, f"Status_{status}")


def _has_solution(model: gp.Model) -> bool:
    try:
        return model.SolCount > 0
    except Exception:
        return False


def _safe_obj_value(model: gp.Model) -> float:
    return float(model.ObjVal) if _has_solution(model) else math.inf


def _model_efficiency_metrics(model: gp.Model) -> Dict[str, float]:
    node_count = float(getattr(model, "NodeCount", 0.0))
    is_mip = bool(getattr(model, "IsMIP", 0))
    return {
        "gurobi_runtime_seconds": float(getattr(model, "Runtime", 0.0)),
        "lp_iterations": float(getattr(model, "IterCount", 0.0)),
        "barrier_iterations": float(getattr(model, "BarIterCount", 0.0)),
        "nodes_explored": node_count,
        "lp_relaxations_solved_estimate": node_count if is_mip else 1.0,
    }


def _add_efficiency_metrics(target: Dict[str, float], source: Dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = float(target.get(key, 0.0)) + float(value)


def print_efficiency_metrics(title: str, metrics: Dict[str, float]) -> None:
    if _is_quiet():
        if not metrics:
            print(f"[{title}] (empty)")
            return
        rt = metrics.get("gurobi_runtime_seconds")
        solves = metrics.get("rmp_solves")
        parts = [f"[{title}]"]
        if rt is not None:
            parts.append(f"runtime={float(rt):.2f}s")
        if solves is not None:
            parts.append(f"rmp_solves={float(solves):.0f}")
        parts.append(f"keys={len(metrics)}")
        print(" ".join(parts))
        return
    print(f"\n[{title}]")
    if not metrics:
        print("  No efficiency metrics available.")
        return
    for key in [
        "gurobi_runtime_seconds",
        "lp_iterations",
        "barrier_iterations",
        "nodes_explored",
        "lp_relaxations_solved_estimate",
        "rmp_solves",
        "branch_price_nodes_explored",
        "branch_price_nodes_remaining",
    ]:
        if key in metrics:
            print(f"  {key}: {float(metrics[key]):.6f}")


def _safe_var_value(model: gp.Model, var: gp.Var) -> float:
    return float(var.X) if var is not None and _has_solution(model) else 0.0


DEFAULT_GNN_CHECKPOINT = "GNN/trained_models/irplt_teacher/bigat/pairwise_rank/best_model.pt"
RESULTS_DIR = Path(os.environ.get("IRP_RESULTS_DIR") or Path(__file__).resolve().parent / "Results")


# ============================================================================
# RESULTS LAYOUT
# ============================================================================
# Every pipeline invocation is tagged by a "phase label" that controls which
# sub-folder its artifacts land in. This keeps train / valid / test / benchmark
# outputs cleanly separated so the Results/ tree is self-describing.
#
#     Results/
#       run_manifest.json          — per-run index (dataset, scenario, phase, files)
#       seed_manifest.json         — resolved seeds
#       scenarios/                 — teacher-data scenario generator outputs
#       teacher/                   — single-run teacher rows (fallback mode)
#       graphs/                    — teacher-graph dataset summary (mirror of GNN/data/<…>/dataset_summary.json)
#       gnn/                       — BiGAT training + offline test
#       phase1_offline_baseline/   — Phase 1 = teacher-collection / classical CG
#       phase2_online_inference/   — Phase 2 = pre-trained GNN scoring during CG
#       phase3_online_learning/    — Phase 3 = fine-tune + inference
#       benchmark/                 — A0/A/B/C benchmark comparison outputs
#       thesis_summary/            — headline tables for reporting
#       charts/                    — all PNGs in one place, phase-suffixed
#
# Phase label is derived from env:
#   IRP_PHASE_LABEL               — explicit override (set by Kaggle notebook)
#   IRP_ONLINE_LEARNING=1         → phase3_online_learning
#   IRP_ONLINE_INFERENCE=1        → phase2_online_inference
#   otherwise                     → phase1_offline_baseline

SCENARIOS_SUBDIR       = "scenarios"
TEACHER_SUBDIR         = "teacher"
GRAPHS_SUBDIR          = "graphs"
GNN_SUBDIR             = "gnn"
BENCHMARK_SUBDIR       = "benchmark"
CHARTS_SUBDIR          = "charts"
THESIS_SUMMARY_SUBDIR  = "thesis_summary"
BENCHMARK_VARIANT_ORDER = [
    "A0_cg_full_exact",
    "A_classical_cg",
    "B_heuristic_cg",
    "C_gnn_guided_cg",
]

_KNOWN_PHASE_LABELS = (
    "phase1_offline_baseline",
    "phase2_online_inference",
    "phase3_online_learning",
    "phase2_deploy_after_teacher",
    "scenario",
    "smoke_test",
)


def resolve_phase_label() -> str:
    explicit = os.environ.get("IRP_PHASE_LABEL", "").strip()
    if explicit:
        return explicit
    if os.environ.get("IRP_ONLINE_LEARNING", "0").lower() not in {"0", "false", "no", ""}:
        return "phase3_online_learning"
    if os.environ.get("IRP_ONLINE_INFERENCE", "0").lower() not in {"0", "false", "no", ""}:
        return "phase2_online_inference"
    return "phase1_offline_baseline"


# ---------------------------------------------------------------------------
# Integer enforcement for final operational outputs
# ---------------------------------------------------------------------------
# The CG LP relaxation produces fractional lambda values and therefore fractional
# shipment quantities; the demand-shock reallocation also produces fractional
# demand deltas. For a thesis-grade IRP result, the *final* operational plan
# (what to ship, how many units, resulting inventory, shortage) must be integer.
# These helpers round final output DataFrames at save time while preserving the
# LP-relaxed values in a parallel <phase>/debug/ folder so examiners can audit
# the relaxation gap.
INTEGER_FINAL_OUTPUT_COLUMNS = {
    "lt_qty",
    "predicted_end_qty",
    "predicted_end_qty_realized",
    "predicted_end_qty_forecast",
    "actual_end_qty",
    "error",
    "forecast_error",
    "shortage",
    "post_shock_shortage",
    "total_realized_demand",
    "fulfilled_demand",
    "forecast_demand",
    "forecast_fulfilled_demand",
    "forecast_shortage",
    "total_direct_qty",
    "total_lt_qty",
    "load_departure",
}


def _integer_final_outputs_enabled() -> bool:
    """Return True when final CSVs should show integer qty/inv/shortage values.

    Default ON. Set IRP_INTEGER_FINAL_OUTPUTS=0 to keep raw LP-relaxed floats
    in the primary output files (debug copies are always written).
    """
    return os.environ.get("IRP_INTEGER_FINAL_OUTPUTS", "1").lower() not in {"0", "false", "no", ""}


def _is_quiet() -> bool:
    """Suppress per-iteration pricing/RMP/pattern prints when IRP_QUIET=1.

    Kaggle's __notebook__.ipynb balloons past 1 GB when multi-scenario runs
    emit tens of thousands of CG iteration lines. With IRP_QUIET=1 the high-
    frequency stdout is silenced here; the same content still appears in
    Results/run.log via subprocess/stdout tee.
    """
    return os.environ.get("IRP_QUIET", "0").lower() not in {"0", "false", "no", ""}


def _qprint(*args, **kwargs) -> None:
    """print() that is a no-op when IRP_QUIET=1."""
    if not _is_quiet():
        print(*args, **kwargs)


def _cg_partial_flush_dir() -> Optional[Path]:
    """Where to drop partial CG diagnostics each iteration (for resume audit).

    Set IRP_CG_PARTIAL_DIR=<path> to enable; unset disables. Written files:
      <dir>/cg_episode_history.partial.json
      <dir>/cg_episode_diagnostics.partial.json
    """
    raw = os.environ.get("IRP_CG_PARTIAL_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


def _flush_cg_partials(cg_history: List[Dict[str, Any]],
                      cg_diags: List[Dict[str, Any]]) -> None:
    """Atomic dump of partial CG progress — survives kernel death."""
    partial_dir = _cg_partial_flush_dir()
    if partial_dir is None:
        return
    for name, payload in (
        ("cg_episode_history.partial.json", cg_history),
        ("cg_episode_diagnostics.partial.json", cg_diags),
    ):
        try:
            target = partial_dir / name
            tmp = target.with_suffix(target.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
            tmp.replace(target)
        except OSError:
            pass


def _round_integer_columns(df: "pd.DataFrame", extra_cols: Optional[Iterable[str]] = None) -> "pd.DataFrame":
    """Return a copy of df with all known integer-valued columns rounded to int.

    Uses banker's rounding (np.rint) then casts to nullable Int64 so missing
    values remain representable. Non-present columns are skipped silently.
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    cols = set(INTEGER_FINAL_OUTPUT_COLUMNS)
    if extra_cols:
        cols.update(extra_cols)
    for c in cols:
        if c in out.columns:
            numeric = pd.to_numeric(out[c], errors="coerce")
            rounded = numeric.round(0)
            # Use nullable Int64 so NaN survives a round-trip through CSV.
            out[c] = rounded.astype("Int64")
    return out


class ResultsLayout:
    """Owns every output path the pipeline writes to.

    Call `build_results_layout()` once at the start of a run; pass the
    resulting object to any helper that needs to write an artifact. Every
    sub-folder is created lazily on first access via `ensure()`.
    """

    def __init__(self, root: Path, phase_label: str):
        self.root = Path(root)
        self.phase_label = phase_label
        self.run_manifest_path = self.root / "run_manifest.json"
        self.seed_manifest_path = self.root / "seed_manifest.json"
        self.scenarios  = self.root / SCENARIOS_SUBDIR
        self.teacher    = self.root / TEACHER_SUBDIR
        self.graphs     = self.root / GRAPHS_SUBDIR
        self.gnn        = self.root / GNN_SUBDIR
        self.gnn_test   = self.gnn / "offline_test"
        self.benchmark  = self.root / BENCHMARK_SUBDIR
        self.charts     = self.root / CHARTS_SUBDIR
        self.thesis     = self.root / THESIS_SUMMARY_SUBDIR
        self.phase      = self.root / phase_label

    def ensure(self, *paths: Path) -> None:
        for p in paths:
            Path(p).mkdir(parents=True, exist_ok=True)

    def phase_file(self, name: str) -> Path:
        self.ensure(self.phase)
        return self.phase / name

    def chart_file(self, name: str, phase_label: Optional[str] = None) -> Path:
        self.ensure(self.charts)
        suffix = phase_label or self.phase_label
        stem, _, ext = name.rpartition(".")
        if ext:
            return self.charts / f"{stem}_{suffix}.{ext}"
        return self.charts / f"{name}_{suffix}"


def build_results_layout(
    root: Optional[Path] = None,
    phase_label: Optional[str] = None,
) -> ResultsLayout:
    root = Path(root) if root is not None else RESULTS_DIR
    phase = phase_label or resolve_phase_label()
    layout = ResultsLayout(root=root, phase_label=phase)
    layout.ensure(layout.root)
    return layout


def resolve_seed_manifest(results_dir: Path | str = RESULTS_DIR) -> Dict[str, int]:
    """Resolve all per-component seeds from a single IRP_MASTER_SEED.

    If IRP_MASTER_SEED is set, deterministically derive per-component seeds for
    demand shocks, ALNS, pattern initialisation, CG pricing, and GNN training.
    Any per-component env var that is already set overrides the derived value
    (so existing scripts that pin individual seeds still work).

    Writes the resolved manifest to Results/seed_manifest.json so every run's
    seed state is auditable alongside the outputs. Returns the manifest dict.
    """
    master_raw = os.environ.get("IRP_MASTER_SEED")
    if master_raw is not None and str(master_raw).strip() != "":
        master_seed = int(master_raw)
        rng = random.Random(master_seed)
        derived = {
            "demand_shock_seed": rng.randrange(1, 2**31 - 1),
            "alns_seed": rng.randrange(1, 2**31 - 1),
            "pattern_init_seed": rng.randrange(1, 2**31 - 1),
            "cg_pricing_seed": rng.randrange(1, 2**31 - 1),
            "gnn_train_seed": rng.randrange(1, 2**31 - 1),
            "gnn_build_seed": rng.randrange(1, 2**31 - 1),
        }
    else:
        master_seed = None
        derived = {
            "demand_shock_seed": int(os.environ.get("IRP_DEMAND_SHOCK_SEED", "20260418")),
            "alns_seed": int(os.environ.get("IRP_ALNS_SEED", "42")),
            "pattern_init_seed": int(os.environ.get("IRP_PATTERN_INIT_SEED", "123")),
            "cg_pricing_seed": int(os.environ.get("IRP_CG_PRICING_SEED", "0")),
            "gnn_train_seed": int(os.environ.get("IRP_GNN_TRAIN_SEED", "0")),
            "gnn_build_seed": int(os.environ.get("IRP_GNN_BUILD_SEED", "0")),
        }
    # Per-component env values take precedence (so users who pin a specific
    # seed for one subsystem can still do so).
    for key, env_name in [
        ("demand_shock_seed", "IRP_DEMAND_SHOCK_SEED"),
        ("alns_seed", "IRP_ALNS_SEED"),
        ("pattern_init_seed", "IRP_PATTERN_INIT_SEED"),
        ("cg_pricing_seed", "IRP_CG_PRICING_SEED"),
        ("gnn_train_seed", "IRP_GNN_TRAIN_SEED"),
        ("gnn_build_seed", "IRP_GNN_BUILD_SEED"),
    ]:
        override = os.environ.get(env_name)
        if override is not None and str(override).strip() != "":
            derived[key] = int(override)
    # Make derived values available to subprocesses (GNN scripts etc.) that
    # read per-component env vars directly.
    for key, env_name in [
        ("demand_shock_seed", "IRP_DEMAND_SHOCK_SEED"),
        ("alns_seed", "IRP_ALNS_SEED"),
        ("pattern_init_seed", "IRP_PATTERN_INIT_SEED"),
        ("cg_pricing_seed", "IRP_CG_PRICING_SEED"),
        ("gnn_train_seed", "IRP_GNN_TRAIN_SEED"),
        ("gnn_build_seed", "IRP_GNN_BUILD_SEED"),
    ]:
        os.environ.setdefault(env_name, str(derived[key]))

    manifest = {
        "master_seed": master_seed,
        "derived_seeds": derived,
        "resolved_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    out_dir = Path(results_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "seed_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except OSError as exc:
        # Never fatal — the seeds are still usable; only the manifest file failed.
        print(f"[Seed] Warning: could not write seed_manifest.json ({exc})")
    print(f"[Seed] master_seed={master_seed} derived={derived}")
    return manifest


# Sub-folders under Results/ that the pipeline owns end-to-end. clean_managed_outputs()
# wipes them before a fresh run so stale artifacts from an earlier run can't
# shadow new ones. `scenarios/` is intentionally EXCLUDED (scenario generator
# appends across invocations), as are `seed_manifest.json` and `run_manifest.json`
# (rewritten fresh on every run but not destroyed before seed_manifest is
# regenerated).
_MANAGED_SUBDIRS: Tuple[str, ...] = (
    TEACHER_SUBDIR,
    GRAPHS_SUBDIR,
    GNN_SUBDIR,
    BENCHMARK_SUBDIR,
    CHARTS_SUBDIR,
    THESIS_SUMMARY_SUBDIR,
    "phase1_offline_baseline",
    "phase2_online_inference",
    "phase2_deploy_after_teacher",
    "phase3_online_learning",
    "smoke_test",
)


def clean_managed_outputs(
    results_dir: Path | str = RESULTS_DIR,
    keep_scenarios: bool = True,
) -> List[str]:
    """Remove managed output sub-folders so a fresh run starts clean.

    Returns the list of removed paths. Safe to call even if the directory
    does not yet exist. `scenarios/` is preserved by default because the
    teacher-scenario generator appends across runs.
    """
    import shutil
    removed: List[str] = []
    root = Path(results_dir)
    if not root.exists():
        return removed
    for sub in _MANAGED_SUBDIRS:
        target = root / sub
        if target.exists():
            try:
                shutil.rmtree(target)
                removed.append(str(target))
            except OSError:
                pass
    if not keep_scenarios:
        target = root / SCENARIOS_SUBDIR
        if target.exists():
            try:
                shutil.rmtree(target)
                removed.append(str(target))
            except OSError:
                pass
    # Legacy stray CSVs from the old flat layout, if a stale tree still
    # contains them — wipe quietly.
    for entry in list(root.iterdir()) if root.exists() else []:
        if entry.is_file() and (
            entry.name.startswith("irp_")
            or entry.name.startswith("cg_")
            or entry.name.startswith("column_pool_")
            or entry.name.startswith("chart_")
        ):
            try:
                entry.unlink()
                removed.append(str(entry))
            except OSError:
                pass
    return removed


def _project_path(path: str) -> Path:
    value = Path(str(path).strip()).expanduser()
    if value.is_absolute():
        return value
    return Path(__file__).resolve().parent / value


def _preferred_teacher_rows_path(path: str | Path) -> Path:
    """Prefer a compact binary teacher export when it exists next to a CSV."""
    value = _project_path(str(path))
    if value.name.lower().endswith(".csv"):
        for candidate in (
            value.with_suffix(".pkl.gz"),
            value.with_suffix(".pickle.gz"),
            value.with_suffix(".pkl"),
            value.with_suffix(".pickle"),
            value.with_suffix(".parquet"),
        ):
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
    return value


def _write_teacher_dataset_exports(df: pd.DataFrame, csv_path: str | Path) -> Path:
    """Write teacher rows as CSV for inspection and as pkl.gz for robust graph builds."""
    resolved_csv_path = _project_path(str(csv_path))
    resolved_csv_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_pickle_path = resolved_csv_path.with_suffix(".pkl.gz")
    write_csv = os.environ.get("IRP_WRITE_TEACHER_CSV", "1").lower() not in {"0", "false", "no"}

    df.to_pickle(resolved_pickle_path)
    if write_csv:
        df.to_csv(resolved_csv_path, index=False)
        print(f"Saved CG teacher dataset CSV to: {resolved_csv_path}")
    else:
        print(f"Skipped full teacher CSV export because IRP_WRITE_TEACHER_CSV=0: {resolved_csv_path}")
    print(f"Saved CG teacher dataset pickle to: {resolved_pickle_path}")
    return resolved_pickle_path


def load_gnn_training_history(checkpoint_path: str = DEFAULT_GNN_CHECKPOINT) -> List[Dict[str, Any]]:
    history_path = _project_path(checkpoint_path).parent / "training_history.json"
    if not history_path.exists():
        return []
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_gnn_training_history(history: List[Dict[str, Any]], checkpoint_path: str = DEFAULT_GNN_CHECKPOINT) -> None:
    if _is_quiet():
        if not history:
            print("[GNN Training History] (none)")
            return
        last = history[-1]
        print(
            f"[GNN Training History] epochs={len(history)} "
            f"last_epoch={int(last.get('epoch', last.get('episode', 0))):03d} "
            f"last_train={float(last.get('train_loss', math.nan)):.4f} "
            f"last_valid={float(last.get('valid_loss', math.nan)):.4f}"
        )
        return
    print("\n[GNN Training Loss By Episode]")
    if not history:
        print("  No GNN training history found. Build teacher graphs, then run GNN/03_train_bigat.py for final training.")
        return
    for row in history:
        print(
            f"  epoch={int(row.get('epoch', row.get('episode', 0))):03d} "
            f"| train_loss={float(row.get('train_loss', math.nan)):.6f} "
            f"| valid_loss={float(row.get('valid_loss', math.nan)):.6f} "
            f"| mrr={float(row.get('valid_mrr', row.get('ranking_valid_mrr', math.nan))):.4f} "
            f"| top1={float(row.get('valid_top1', row.get('ranking_valid_top1', math.nan))):.4f} "
            f"| top3={float(row.get('ranking_valid_top3', math.nan)):.4f} "
            f"| f1={float(row.get('valid_f1', row.get('binary_valid_f1', math.nan))):.4f}"
        )
    chart_path = _project_path(checkpoint_path).parent / "training_loss_curve.png"
    if chart_path.exists():
        print(f"  loss_chart={chart_path}")


def run_teacher_graph_and_gnn_training(
    teacher_csv_path: str,
    build_graphs: bool = True,
    train_gnn: bool = True,
    train_epochs: int = 5,
    resume_checkpoint: bool = False,
    checkpoint_path: str = DEFAULT_GNN_CHECKPOINT,
    training_history_csv_path: Optional[str] = None,
    run_offline_test: bool = True,
) -> List[Dict[str, Any]]:
    """Build teacher graph samples and train the BiGAT scorer.

    Returns the training history (list of per-epoch dicts). If
    `training_history_csv_path` is provided, also writes the history as a CSV
    at that path. If it's left as None, the CSV is written next to the project
    Results directory as `irp_gnn_training_history.csv` (so Kaggle exports
    automatically produce the file without needing extra code downstream).
    """
    requested_teacher_path = _project_path(teacher_csv_path)
    teacher_rows_path = _preferred_teacher_rows_path(requested_teacher_path)
    if not teacher_rows_path.exists() or teacher_rows_path.stat().st_size <= 1:
        print(f"[Teacher/GNN] No non-empty teacher rows file found; skip graph/GNN update: {teacher_rows_path}")
        return []
    if teacher_rows_path != requested_teacher_path:
        print(f"[Teacher/GNN] Using compact teacher rows file for graph build: {teacher_rows_path}")

    if build_graphs:
        print("\n[Teacher Graph Dataset Update]")
        subprocess.run(
            [
                sys.executable,
                "GNN/build_teacher_graph_dataset.py",
                "--teacher-csv",
                str(teacher_rows_path),
                "--out-dir",
                "GNN/data/irplt_teacher",
                "--overwrite",
            ],
            cwd=str(Path(__file__).resolve().parent),
            check=True,
        )

    history: List[Dict[str, Any]] = []
    if train_gnn:
        cmd = [
            sys.executable,
            "GNN/03_train_bigat.py",
            "--data-dir",
            "GNN/data/irplt_teacher",
            "--dataset-type",
            "teacher",
            "--epochs",
            str(int(train_epochs)),
            "--objective",
            "pairwise_rank",
        ]
        checkpoint = _project_path(checkpoint_path)
        if resume_checkpoint and checkpoint.exists():
            cmd.extend(["--resume-checkpoint", str(checkpoint)])
            print("\n[Teacher BiGAT Training Update] (resume from existing checkpoint)")
        else:
            if not resume_checkpoint and checkpoint.exists():
                print(f"\n[Teacher BiGAT Training Update] Found existing checkpoint at {checkpoint}; "
                      f"training fresh because resume_checkpoint=False. Pass resume_checkpoint=True to fine-tune.")
            else:
                print("\n[Teacher BiGAT Training Update] (fresh training)")
        try:
            subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[Teacher/GNN] Training failed (non-fatal): {exc}")
            print("[Teacher/GNN] Continuing without updated checkpoint — teacher rows were collected successfully.")

        history = load_gnn_training_history(checkpoint_path)
        if not history:
            print(
                "[Teacher/GNN] Training subprocess exited cleanly but wrote no history rows. "
                "This usually means build_teacher_graph_dataset.py produced zero samples — "
                "check constraint_features_json diversity or group_key collapse."
            )

        gnn_dir = Path(RESULTS_DIR) / GNN_SUBDIR
        gnn_dir.mkdir(parents=True, exist_ok=True)
        resolved_csv_path = (
            Path(training_history_csv_path)
            if training_history_csv_path
            else gnn_dir / "training_history.csv"
        )
        resolved_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(resolved_csv_path, index=False)
        print(f"[Teacher/GNN] Wrote training history CSV ({len(history)} rows) to: {resolved_csv_path}")

        # Mirror the canonical training loss chart + JSON history into Results/gnn/
        # so everything for the thesis is colocated under Results/.
        _ckpt_dir = _project_path(checkpoint_path).parent
        for src_name, dst_name in [
            ("training_loss_curve.png", "training_loss_curve.png"),
            ("training_history.json",   "training_history.json"),
            ("training_summary.json",   "training_summary.json"),
        ]:
            src = _ckpt_dir / src_name
            if src.exists():
                try:
                    import shutil as _shutil
                    _shutil.copy2(src, gnn_dir / dst_name)
                except OSError:
                    pass

        # ---- Validation summary (best epoch by MRR) ----
        if history:
            best_row = max(
                history,
                key=lambda r: float(r.get("valid_mrr", r.get("ranking_valid_mrr", float("-inf"))) or float("-inf")),
            )
            print("\n[GNN Validation Summary — best epoch by MRR]")
            print(
                f"  epoch={int(best_row.get('epoch', 0)):03d} | "
                f"train_loss={float(best_row.get('train_loss', math.nan)):.4f} | "
                f"valid_loss={float(best_row.get('valid_loss', math.nan)):.4f} | "
                f"mrr={float(best_row.get('valid_mrr', best_row.get('ranking_valid_mrr', math.nan))):.4f} | "
                f"top1={float(best_row.get('valid_top1', best_row.get('ranking_valid_top1', math.nan))):.4f} | "
                f"top3={float(best_row.get('ranking_valid_top3', math.nan)):.4f} | "
                f"f1={float(best_row.get('valid_f1', best_row.get('binary_valid_f1', math.nan))):.4f}"
            )

        # ---- Offline test evaluation on strictly held-out split ----
        if run_offline_test:
            checkpoint_path_obj = _project_path(checkpoint_path)
            test_split_dir = _project_path("GNN/data/irplt_teacher/test")
            # Strict mode — fail loudly instead of silently skipping. A "silent
            # skip" here is the worst failure mode for thesis evaluation: the
            # pipeline prints "OK" but never actually runs held-out testing.
            # Activate with env IRP_STRICT_OFFLINE_TEST=1.
            _strict_offline_test = os.environ.get("IRP_STRICT_OFFLINE_TEST", "0").strip().lower() not in {"0", "false", "no", ""}
            if not checkpoint_path_obj.exists():
                msg = f"[Teacher/GNN] offline test — checkpoint not found: {checkpoint_path_obj}"
                if _strict_offline_test:
                    raise RuntimeError(msg + " (IRP_STRICT_OFFLINE_TEST=1)")
                print(f"Skipping {msg}")
            elif not test_split_dir.exists() or not any(test_split_dir.iterdir()):
                msg = (
                    f"[Teacher/GNN] offline test — empty test split at {test_split_dir}. "
                    f"Collect teacher rows from more source_instance values to enable instance-level split."
                )
                if _strict_offline_test:
                    raise RuntimeError(msg + " (IRP_STRICT_OFFLINE_TEST=1)")
                print(f"Skipping {msg}")
            else:
                gnn_test_dir = Path(RESULTS_DIR) / GNN_SUBDIR / "offline_test"
                gnn_test_dir.mkdir(parents=True, exist_ok=True)
                test_out_csv = gnn_test_dir / "test_per_sample.csv"
                test_cmd = [
                    sys.executable,
                    "GNN/04_test.py",
                    "--data-dir", "GNN/data/irplt_teacher",
                    "--checkpoint", str(checkpoint_path_obj),
                    "--split", "test",
                    "--out-file", str(test_out_csv),
                ]
                print("\n[Teacher/GNN] Running offline test evaluation on strictly held-out split")
                try:
                    subprocess.run(test_cmd, cwd=str(Path(__file__).resolve().parent), check=True)
                    print(f"[Teacher/GNN] Offline test metrics saved to: {test_out_csv}")
                    # Aggregate by mass_threshold → test_summary.json
                    try:
                        _df = pd.read_csv(test_out_csv)
                        if not _df.empty and "mass_threshold" in _df.columns:
                            _agg = _df.groupby("mass_threshold").mean(numeric_only=True).reset_index()
                            _summary = {
                                "n_samples": int(_df["sample_id"].nunique()) if "sample_id" in _df.columns else int(len(_df)),
                                "thresholds": _agg.to_dict(orient="records"),
                            }
                            with open(gnn_test_dir / "test_summary.json", "w", encoding="utf-8") as f:
                                json.dump(_summary, f, indent=2)
                    except Exception as agg_exc:
                        print(f"[Teacher/GNN] Could not aggregate test summary: {agg_exc}")
                except subprocess.CalledProcessError as exc:
                    # Surface the failure so silent regressions are caught (see code review Task).
                    print(f"[Teacher/GNN] Offline test FAILED (non-zero exit): {exc}")

    return history


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IRPData:
    periods: List[Period]
    stores: List[Store]
    products: List[Product]
    warehouse: str = "CW"

    # Identity tags for downstream logging / teacher-row tagging. Populated by
    # DatasetToIRPValidationMapper when it builds the instance so that scenario
    # generators can pass through a stable dataset_id for the manifest.
    dataset_id: str = ""
    scenario_id: str = ""

    demand: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)
    realized_demand: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)
    init_inventory_store: Dict[Tuple[Store, Product], float] = field(default_factory=dict)
    init_inventory_wh: Dict[Product, float] = field(default_factory=dict)

    max_inventory_store: Dict[Tuple[Store, Product], float] = field(default_factory=dict)
    max_inventory_wh: Dict[Product, float] = field(default_factory=dict)

    holding_cost_store: Dict[Tuple[Store, Product], float] = field(default_factory=dict)
    holding_cost_wh: Dict[Product, float] = field(default_factory=dict)
    shortage_cost: Dict[Tuple[Store, Product], float] = field(default_factory=dict)

    ship_cost_cw: Dict[Tuple[Store, Product], float] = field(default_factory=dict)
    ship_cost_lt: Dict[Tuple[Store, Store, Product], float] = field(default_factory=dict)

    fixed_dispatch_cw: Dict[Store, float] = field(default_factory=dict)
    fixed_dispatch_lt: Dict[Tuple[Store, Store], float] = field(default_factory=dict)

    big_m_cw: Dict[Tuple[Store, Product], float] = field(default_factory=dict)
    big_m_lt: Dict[Tuple[Store, Store, Product], float] = field(default_factory=dict)

    # ===== Added for Achamrah-style IRPT =====
    vehicles: List[Vehicle] = field(default_factory=list)
    vehicle_capacity: float = 120.0
    vehicle_fixed_cost: float = 0.0
    max_vehicles_used: int = 3
    alpha: float = 1.0

    # replenishment to warehouse g_{p,t}
    replenishment_wh: Dict[Tuple[Product, Period], float] = field(default_factory=dict)

    # aggregate node capacity C_i in paper
    node_capacity: Dict[Node, float] = field(default_factory=dict)

    # routing distance / cost base d_{i,j}
    distance: Dict[Tuple[Node, Node], float] = field(default_factory=dict)

    # LT unit cost b_{i,j}
    transship_unit_cost: Dict[Tuple[Store, Store], float] = field(default_factory=dict)

    # State after forecast-based DC shipment and hidden realized-demand shock.
    post_shock_inventory: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)
    post_shock_shortage: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)


@dataclass
class BaselineIRPSolution:
    status: str
    objective: float
    ship_cw: Dict[Tuple[Store, Product, Period], float]
    activate_cw: Dict[Tuple[Store, Period], int]
    inv_store: Dict[Tuple[Store, Product, Period], float]
    inv_wh: Dict[Tuple[Product, Period], float]
    shortage: Dict[Tuple[Store, Product, Period], float]

    def summary(self) -> Dict:
        return {
            "status": self.status,
            "objective": self.objective,
            "total_ship_from_cw": sum(self.ship_cw.values()),
            "total_shortage": sum(self.shortage.values()),
        }


@dataclass
class FullIRPTSolution:
    status: str
    objective: float

    direct_ship_q: Dict[Tuple[Store, Product, Period], float]
    inv_store: Dict[Tuple[Store, Product, Period], float]
    inv_wh: Dict[Tuple[Product, Period], float]
    shortage: Dict[Tuple[Store, Product, Period], float]

    x: Dict[Tuple[Node, Node, Vehicle, Period], int]
    u: Dict[Tuple[Vehicle, Period], int]
    z: Dict[Tuple[Node, Vehicle, Period], int]
    q: Dict[Tuple[Product, Node, Node, Vehicle, Period], float]
    y: Dict[Tuple[Store, Store, Product, Vehicle, Period], float]
    deliv: Dict[Tuple[Store, Product, Vehicle, Period], float] = field(default_factory=dict)
    load: Dict[Tuple[Node, Vehicle, Period], float] = field(default_factory=dict)
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    # Per-iteration trace populated by ALNS; empty for MIP solvers.
    alns_history: List[Dict[str, float]] = field(default_factory=list)
    # Attached by the pipeline after LT recourse is computed so validation compares
    # against the inventory that actually lands on shelves (baseline + post-shock + LT).
    realized_inventory_after_lt: Optional[Dict[Tuple[Store, Product, Period], float]] = None

    def summary(self) -> Dict:
        return {
            "status": self.status,
            "objective": self.objective,
            "total_direct_shipments": sum(self.direct_ship_q.values()),
            "total_shortage": sum(self.shortage.values()),
            "total_transshipment": sum(self.y.values()),
            "active_route_arcs": sum(self.x.values()),
            "vehicles_used": sum(self.u.values()),
            "efficiency_metrics": self.efficiency_metrics,
        }


@dataclass
class LTPattern:
    pattern_id: str
    period: Period
    product: Product
    pattern_flows: Dict[Tuple[Store, Store], float]
    column_cost: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class StackelbergParams:
    donor_accept_threshold: float = 0.0
    receiver_accept_threshold: float = 0.0

    donor_risk_weight: float = 1.2
    donor_ship_burden_weight: float = 1.0
    donor_service_loss_weight: float = 1.0

    receiver_shortage_reduction_weight: float = 2.0
    receiver_service_gain_weight: float = 1.0
    receiver_handling_weight: float = 0.5

    min_compensation: float = 0.0
    compensation_cap: float = 999999.0
    acceptance_score_weight: float = 0.6
    economic_score_weight: float = 0.4
    top_k_after_game_per_feature: int = 5
    allow_pricing_fallback_when_no_acceptance: bool = True
    fallback_top_k_after_game_per_feature: int = 5
    fallback_requires_negative_reduced_cost: bool = True


@dataclass
class StackelbergDecision:
    accepted: bool
    compensation: float
    donor_utility: float
    receiver_utility: float
    acceptance_score: float
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class FollowerBestResponseResult:
    """Follower-optimal LT plan for one (product, period).

    The follower observes post-delivery, post-shock store inventories and
    minimises LT shipping cost + shortage penalty on residual unmet need +
    holding cost on remaining surplus (units that could not be moved away).
    """
    product: Any
    period: Any
    flows: Dict[Tuple[Any, Any], float]  # (donor, receiver) -> qty shipped
    lt_cost: float                       # total LT shipping + fixed cost
    shortage_reduction: float            # units of need covered by LT
    remaining_need: float                # uncovered need after follower LT
    remaining_surplus: float             # unused surplus after follower LT
    shortage_penalty: float              # penalty on remaining_need
    holding_cost: float                  # holding cost on remaining_surplus
    total_cost: float                    # lt_cost + shortage_penalty + holding_cost
    n_arcs_used: int
    solver_used: str = "greedy"          # "greedy" | "lp"


@dataclass
class CGSolution:
    status: str
    objective: float
    lambda_values: Dict[str, float]
    selected_patterns: List[str]
    implied_net_lt: Dict[Tuple[Store, Product, Period], float]
    dual_need: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)
    dual_surplus: Dict[Tuple[Store, Product, Period], float] = field(default_factory=dict)
    active_product_periods: Set[Tuple[Product, Period]] = field(default_factory=set)
    iterations_run: int = 0
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    branch_summary: Dict[str, Any] = field(default_factory=dict)
    # Stackelberg follower best-response computed after CG converges.
    # Keys: (product, period); Values: FollowerBestResponseResult.
    follower_solution: Optional[Dict[Tuple[Any, Any], "FollowerBestResponseResult"]] = None
    # Per-pattern system-cost delta recorded during follower-aware pricing.
    stackelberg_column_scores: List[Dict[str, Any]] = field(default_factory=list)
    # Post-convergence check: were there unselected patterns that would still improve cost?
    stackelberg_validation: Optional[Dict[str, Any]] = None

    def summary(self) -> Dict:
        payload = {
            "status": self.status,
            "objective": self.objective,
            "selected_patterns": self.selected_patterns,
            "n_selected_patterns": len(self.selected_patterns),
            "n_active_product_periods": len(self.active_product_periods),
            "iterations_run": self.iterations_run,
            "efficiency_metrics": self.efficiency_metrics,
        }
        if self.branch_summary:
            payload["branch_summary"] = self.branch_summary
        return payload


# ============================================================================
# DATA MAPPER
# ============================================================================

class DatasetToIRPValidationMapper:
    REQUIRED_COLUMNS = [
        "SITE_NAME", "NORMAL_PRICE", "ART_SV_NAME_ENG",
        "SALE_QTY", "END_QTY", "PERIOD"
    ]

    def __init__(
        self,
        excel_path: str,
        sheet_name: Optional[str] = None,
        store_limit: Optional[int] = None,
        sku_limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.excel_path = str(excel_path).strip()
        self.sheet_name = sheet_name
        self.store_limit = store_limit
        self.sku_limit = sku_limit
        self.start_date = start_date
        self.end_date = end_date

    def load_raw(self) -> pd.DataFrame:
        input_path = _project_path(self.excel_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {input_path}")

        suffix = input_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(input_path)
        elif suffix in {".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"}:
            df = pd.read_excel(input_path, sheet_name=self.sheet_name or 0)
        else:
            raise ValueError(
                f"Unsupported dataset file extension '{suffix}'. "
                "Use .csv, .xls, .xlsx, .xlsm, .xlsb, or .ods."
            )

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[self.REQUIRED_COLUMNS].copy()
        df = df.rename(columns={
            "SITE_NAME": "store",
            "NORMAL_PRICE": "price",
            "ART_SV_NAME_ENG": "sku",
            "SALE_QTY": "sale_qty",
            "END_QTY": "end_qty",
            "PERIOD": "period_raw",
        })

        df["store"] = df["store"].astype(str).str.strip()
        df["sku"] = df["sku"].astype(str).str.strip()
        df["sale_qty"] = pd.to_numeric(df["sale_qty"], errors="coerce").fillna(0.0)
        df["end_qty"] = pd.to_numeric(df["end_qty"], errors="coerce").fillna(0.0)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["period_date"] = pd.to_datetime(df["period_raw"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["period_date"]).copy()

        if self.start_date is not None:
            df = df[df["period_date"] >= pd.to_datetime(self.start_date)]
        if self.end_date is not None:
            df = df[df["period_date"] <= pd.to_datetime(self.end_date)]

        return df

    def preprocess(self) -> pd.DataFrame:
        df = self.load_raw()
        grp = (
            df.groupby(["store", "sku", "period_date"], as_index=False)
              .agg(
                  sale_qty=("sale_qty", "sum"),
                  end_qty=("end_qty", "sum"),
                  price=("price", "median"),
              )
        )

        if self.store_limit is not None:
            top_stores = (
                grp.groupby("store")["sale_qty"].sum()
                .sort_values(ascending=False)
                .head(self.store_limit)
                .index.tolist()
            )
            grp = grp[grp["store"].isin(top_stores)].copy()

        if self.sku_limit is not None:
            top_skus = (
                grp.groupby("sku")["sale_qty"].sum()
                .sort_values(ascending=False)
                .head(self.sku_limit)
                .index.tolist()
            )
            grp = grp[grp["sku"].isin(top_skus)].copy()

        grp = grp.sort_values(["store", "sku", "period_date"]).reset_index(drop=True)
        unique_dates = sorted(grp["period_date"].drop_duplicates().tolist())
        date_to_period = {dt: i + 1 for i, dt in enumerate(unique_dates)}
        grp["period"] = grp["period_date"].map(date_to_period)
        return grp

    def _populate_routing_distances(
        self,
        data: IRPData,
        stores: List[Store],
        distance_matrix_path: Optional[str],
    ) -> None:
        all_nodes = [data.warehouse] + stores
        distance_lookup: Dict[Tuple[Node, Node], float] = {}
        matrix_depot = "South Ambient DC"

        if distance_matrix_path:
            matrix_path = _project_path(distance_matrix_path)
            if matrix_path.exists():
                matrix_df = pd.read_csv(matrix_path)
                node_col = matrix_df.columns[0]
                matrix_df[node_col] = matrix_df[node_col].astype(str).str.strip()
                matrix_nodes = [str(c).strip() for c in matrix_df.columns[1:]]
                for _, row in matrix_df.iterrows():
                    src = str(row[node_col]).strip()
                    for dst in matrix_nodes:
                        value = pd.to_numeric(row.get(dst), errors="coerce")
                        if pd.notna(value):
                            distance_lookup[(src, dst)] = float(value)
                depot_candidates = [n for n in matrix_df[node_col].tolist() if "DC" in str(n).upper()]
                if depot_candidates:
                    matrix_depot = str(depot_candidates[0]).strip()

        def matrix_name(node: Node) -> Node:
            return matrix_depot if node == data.warehouse else node

        def fallback_distance(i: Node, j: Node) -> float:
            if i == j:
                return 0.0
            i_idx = 0 if i == data.warehouse else stores.index(i) + 1
            j_idx = 0 if j == data.warehouse else stores.index(j) + 1
            if i == data.warehouse or j == data.warehouse:
                return round(10.0 + 1.7 * max(i_idx, j_idx), 1)
            return round(6.0 + 2.3 * abs(i_idx - j_idx) + 0.4 * ((i_idx + j_idx) % 5), 1)

        for i in all_nodes:
            for j in all_nodes:
                src = matrix_name(i)
                dst = matrix_name(j)
                data.distance[(i, j)] = distance_lookup.get((src, dst), fallback_distance(i, j))

    def build_irp_data(
        self,
        wh_inventory_multiplier: float = 2.5,
        store_capacity_multiplier: float = 1.5,
        shortage_cost_rate: float = 0.25,
        holding_cost_rate: float = 0.01,
        cw_ship_cost_flat: float = 1.0,
        lt_ship_cost_flat: float = 0.6,
        fixed_dispatch_cw: float = 8.0,
        fixed_dispatch_lt: float = 2.0,
        vehicle_count: int = 2,
        vehicle_capacity: float = 120.0,
        vehicle_fixed_cost: float = 50.0,
        alpha: float = 1.0,
        cw_replenishment_factor: float = 0.6,
        cw_capacity_factor: float = 2.0,
        distance_matrix_path: Optional[str] = "Distance data/mm_megamarket_distance_matrix_clean.csv",
        store_initial_inventory_multiplier: float = 1.0,
        lt_cost_multiplier: float = 1.0,
    ):
        df = self.preprocess()

        stores = sorted(df["store"].unique().tolist())
        products = sorted(df["sku"].unique().tolist())
        periods = sorted(df["period"].unique().tolist())
        data = IRPData(periods=periods, stores=stores, products=products)

        full_index = pd.MultiIndex.from_product(
            [stores, products, periods], names=["store", "sku", "period"]
        )
        base = (
            df.set_index(["store", "sku", "period"])[["sale_qty", "end_qty", "price", "period_date"]]
              .reindex(full_index)
              .reset_index()
        )
        base["sale_qty"] = base["sale_qty"].fillna(0.0)
        base["end_qty"] = base["end_qty"].fillna(0.0)

        sku_price = base.groupby("sku")["price"].median()
        global_price = float(base["price"].median()) if base["price"].notna().any() else 1.0
        base["price"] = base.apply(
            lambda r: sku_price.get(r["sku"], global_price) if pd.isna(r["price"]) else r["price"],
            axis=1
        )
        base["price"] = base["price"].fillna(global_price)

        # Demand from sales for all days
        for _, row in base.iterrows():
            s, p, t = row["store"], row["sku"], int(row["period"])
            data.demand[(s, p, t)] = float(row["sale_qty"])
            data.realized_demand[(s, p, t)] = float(row["sale_qty"])

        # ONLY first-day END_QTY as initial inventory
        first_period = min(periods)
        first_df = base[base["period"] == first_period].copy()
        store_initial_inventory_multiplier = max(0.0, float(store_initial_inventory_multiplier))
        for _, row in first_df.iterrows():
            s, p = row["store"], row["sku"]
            data.init_inventory_store[(s, p)] = max(0.0, store_initial_inventory_multiplier * float(row["end_qty"]))

        for s, p in itertools.product(stores, products):
            data.init_inventory_store.setdefault((s, p), 0.0)

        # Capacity from first-day init and historical maxima only as rough cap proxy
        for s, p in itertools.product(stores, products):
            obs = base[(base["store"] == s) & (base["sku"] == p)]["end_qty"]
            obs_max = float(obs.max()) if not obs.empty else 0.0
            init_inv = data.init_inventory_store[(s, p)]
            data.max_inventory_store[(s, p)] = max(5.0, store_capacity_multiplier * max(obs_max, init_inv, 1.0))

        total_demand_by_sku = base.groupby("sku")["sale_qty"].sum().to_dict()
        for p in products:
            total_dem = float(total_demand_by_sku.get(p, 0.0))
            data.init_inventory_wh[p] = max(0.0, wh_inventory_multiplier * total_dem)
            data.max_inventory_wh[p] = max(data.init_inventory_wh[p], 1.2 * data.init_inventory_wh[p])

        price_by_store_sku = base.groupby(["store", "sku"])["price"].median().to_dict()
        for s, p in itertools.product(stores, products):
            price = float(price_by_store_sku.get((s, p), global_price))
            data.holding_cost_store[(s, p)] = max(0.05, holding_cost_rate * price)
            data.shortage_cost[(s, p)] = max(1.0, shortage_cost_rate * price)
            data.ship_cost_cw[(s, p)] = cw_ship_cost_flat
            data.big_m_cw[(s, p)] = max(
                data.max_inventory_store[(s, p)],
                sum(data.demand[(s, p, t)] for t in periods) + data.max_inventory_store[(s, p)]
            )

        for p in products:
            med_price = float(base.loc[base["sku"] == p, "price"].median()) if (base["sku"] == p).any() else global_price
            data.holding_cost_wh[p] = max(0.02, holding_cost_rate * 0.5 * med_price)

        for i, j, p in itertools.product(stores, stores, products):
            if i == j:
                continue
            data.ship_cost_lt[(i, j, p)] = lt_ship_cost_flat
            data.big_m_lt[(i, j, p)] = max(5.0, 0.5 * sum(data.demand[(j, p, t)] for t in periods))

        for s in stores:
            data.fixed_dispatch_cw[s] = fixed_dispatch_cw
        for i, j in itertools.product(stores, stores):
            if i == j:
                continue
            data.fixed_dispatch_lt[(i, j)] = fixed_dispatch_lt

        # ===== Added Achamrah-style fields =====
        data.vehicles = [f"V{v}" for v in range(1, vehicle_count + 1)]
        data.vehicle_capacity = vehicle_capacity
        data.vehicle_fixed_cost = vehicle_fixed_cost
        data.max_vehicles_used = vehicle_count
        data.alpha = alpha

        # replenishment to warehouse by period/product
        # cw_replenishment_factor controls the per-cycle top-up size relative to the
        # cycle's forecast demand: factor=1.0 covers exactly one cycle's demand,
        # factor<1.0 under-stocks (forces tighter planning + real LT opportunities),
        # factor>1.0 over-stocks the warehouse.
        demand_by_sku_period = base.groupby(["sku", "period"])["sale_qty"].sum().to_dict()
        replenishment_cycle = 7   # replenish from DC every 7 days
        cw_replenishment_factor = max(0.0, float(cw_replenishment_factor))
        for p in products:
            for t in periods:
                if (t - 1) % replenishment_cycle == 0:
                    cycle_demand = sum(
                        float(demand_by_sku_period.get((p, tt), 0.0))
                        for tt in periods
                        if t <= tt < t + replenishment_cycle
                    )
                    data.replenishment_wh[(p, t)] = max(
                        0.0,
                        cw_replenishment_factor * cycle_demand,
                    )
                else:
                    data.replenishment_wh[(p, t)] = 0.0

        # aggregate node capacity for stores
        for s in stores:
            data.node_capacity[s] = sum(data.max_inventory_store[(s, p)] for p in products)

        # aggregate capacity for CW
        data.node_capacity[data.warehouse] = cw_capacity_factor * sum(data.init_inventory_wh[p] for p in products)

        self._populate_routing_distances(data, stores, distance_matrix_path)

        # paper-style LT unit cost b_ij
        for i in stores:
            for j in stores:
                if i == j:
                    continue
                data.transship_unit_cost[(i, j)] = 0.01 * alpha * data.distance[(i, j)] * max(0.0, lt_cost_multiplier)

        # Validation target = actual END_QTY for periods >= 2
        validation_target = base[base["period"] > first_period][["store", "sku", "period", "end_qty"]].copy()
        validation_target = validation_target.rename(columns={"end_qty": "actual_end_qty"})

        metadata = {
            "n_rows_processed": len(base),
            "n_stores": len(stores),
            "n_products": len(products),
            "n_periods": len(periods),
            "validation_rows": len(validation_target),
            "first_period_used_as_initial_inventory": first_period,
            "n_vehicles": len(data.vehicles),
            "vehicle_capacity": data.vehicle_capacity,
            "vehicle_fixed_cost": data.vehicle_fixed_cost,
            "store_initial_inventory_multiplier": store_initial_inventory_multiplier,
        }
        return data, base, validation_target, metadata


# ============================================================================
# ORIGINAL BASELINE MODEL (KEPT)
# ============================================================================

class BaselineIRPModel:
    def __init__(self, data: IRPData):
        self.data = data

    def solve(self, msg: bool = False) -> BaselineIRPSolution:
        d = self.data
        mdl = gp.Model("Baseline_IRP", env=get_gurobi_env())
        mdl.Params.OutputFlag = 1 if msg else 0

        q_cw_keys = [(s, p, t) for s in d.stores for p in d.products for t in d.periods]
        y_cw_keys = [(s, t) for s in d.stores for t in d.periods]
        I_s_keys = [(s, p, t) for s in d.stores for p in d.products for t in d.periods]
        I_w_keys = [(p, t) for p in d.products for t in d.periods]
        B_keys = [(s, p, t) for s in d.stores for p in d.products for t in d.periods]

        q_cw = mdl.addVars(q_cw_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="q_cw")
        y_cw = mdl.addVars(y_cw_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="y_cw")
        I_s = mdl.addVars(I_s_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="I_s")
        I_w = mdl.addVars(I_w_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="I_w")
        B = mdl.addVars(B_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="B")

        mdl.setObjective(
            gp.quicksum(d.ship_cost_cw[(s, p)] * q_cw[(s, p, t)] for s, p, t in q_cw_keys)
            + gp.quicksum(d.fixed_dispatch_cw[s] * y_cw[(s, t)] for s, t in y_cw_keys)
            + gp.quicksum(d.holding_cost_store[(s, p)] * I_s[(s, p, t)] for s, p, t in I_s_keys)
            + gp.quicksum(d.holding_cost_wh[p] * I_w[(p, t)] for p, t in I_w_keys)
            + gp.quicksum(d.shortage_cost[(s, p)] * B[(s, p, t)] for s, p, t in B_keys),
            GRB.MINIMIZE,
        )

        for s, p in itertools.product(d.stores, d.products):
            t0 = d.periods[0]
            mdl.addConstr(
                I_s[(s, p, t0)] == d.init_inventory_store[(s, p)] + q_cw[(s, p, t0)] - d.demand[(s, p, t0)] + B[(s, p, t0)]
            )
            for t_prev, t in zip(d.periods[:-1], d.periods[1:]):
                mdl.addConstr(
                    I_s[(s, p, t)] == I_s[(s, p, t_prev)] + q_cw[(s, p, t)] - d.demand[(s, p, t)] + B[(s, p, t)]
                )

        for p in d.products:
            t0 = d.periods[0]
            mdl.addConstr(I_w[(p, t0)] == d.init_inventory_wh[p] - gp.quicksum(q_cw[(s, p, t0)] for s in d.stores))
            for t_prev, t in zip(d.periods[:-1], d.periods[1:]):
                mdl.addConstr(I_w[(p, t)] == I_w[(p, t_prev)] - gp.quicksum(q_cw[(s, p, t)] for s in d.stores))

        for s, p, t in itertools.product(d.stores, d.products, d.periods):
            mdl.addConstr(I_s[(s, p, t)] <= d.max_inventory_store[(s, p)])
            mdl.addConstr(q_cw[(s, p, t)] <= d.big_m_cw[(s, p)] * y_cw[(s, t)])
        for p, t in itertools.product(d.products, d.periods):
            mdl.addConstr(I_w[(p, t)] <= d.max_inventory_wh[p])

        mdl.optimize()

        return BaselineIRPSolution(
            status=_grb_status_name(mdl.Status),
            objective=_safe_obj_value(mdl),
            ship_cw={(s, p, t): _safe_var_value(mdl, q_cw[(s, p, t)]) for s, p, t in q_cw_keys},
            activate_cw={(s, t): int(round(_safe_var_value(mdl, y_cw[(s, t)]))) for s, t in y_cw_keys},
            inv_store={(s, p, t): _safe_var_value(mdl, I_s[(s, p, t)]) for s, p, t in I_s_keys},
            inv_wh={(p, t): _safe_var_value(mdl, I_w[(p, t)]) for p, t in I_w_keys},
            shortage={(s, p, t): _safe_var_value(mdl, B[(s, p, t)]) for s, p, t in B_keys},
        )


# ============================================================================
# BASELINE ALNS MODEL (Song et al. 2023 adaptation, DC->stores only, no LT)
# ============================================================================
#
# Replaces the Gurobi-based Step 1 solver. The ALNS produces a FullIRPTSolution
# with the exact same field schema (direct_ship_q, inv_store, inv_wh, shortage,
# x, u, z, q, y, deliv, load, efficiency_metrics) so downstream Step 1B / CG /
# LT stages remain unchanged. Lateral transshipment (y) is always zero here;
# inter-store rebalancing happens only in later stages via column generation.


@dataclass
class _ALNSState:
    """ALNS solution state: per-period vehicle routes + per-(store, product, vehicle, period) deliveries."""
    routes: Dict[Tuple[Period, Vehicle], List[Store]] = field(default_factory=dict)
    deliv: Dict[Tuple[Store, Product, Vehicle, Period], float] = field(default_factory=dict)

    def clone(self) -> "_ALNSState":
        return _ALNSState(
            routes={k: list(v) for k, v in self.routes.items()},
            deliv=dict(self.deliv),
        )


class BaselineALNSModel:
    """
    Adaptive Large Neighborhood Search baseline (DC -> stores -> DC, no LT).

    Keeps the IRPT baseline's constraints and objective from AchamrahFullIRPTModel
    when allow_lateral_transshipment=False:
      * inventory balance (warehouse, stores) with shortage B,
      * vehicle capacity and single-visit-per-store-per-period routing,
      * warehouse/store node capacities,
      * cw_dispatch_cycle periodicity,
      * min_visit_activity_qty,
      * max_vehicles_used.

    Objective components (identical to the Gurobi baseline with y=0):
      direct shipping cost  + store holding + warehouse holding
      + alpha * distance (routing) + vehicle fixed cost
      + shortage cost.
    """

    SIGMA_NEW_BEST = 33.0
    SIGMA_BETTER = 13.0
    SIGMA_ACCEPTED = 9.0

    def __init__(self, data: IRPData):
        self.data = data

    # ------------------------------------------------------------------ public

    def solve(
        self,
        msg: bool = False,
        time_limit: Optional[int] = None,
        enforce_integer_flows: bool = False,
        add_valid_16_20: bool = True,
        allow_lateral_transshipment: bool = False,
        min_visit_activity_qty: float = 1.0,
        min_visit_delivery_qty: float = 0.0,
        cw_dispatch_cycle: Optional[int] = 5,
        max_iterations: int = 2000,
        seed: int = 20260419,
        initial_temperature: Optional[float] = None,
        cooling_rate: float = 0.998,
        segment_size: int = 40,
        reaction_factor: float = 0.1,
    ) -> FullIRPTSolution:
        # allow_lateral_transshipment is accepted for API parity; LT is never produced here.
        if allow_lateral_transshipment:
            if msg:
                print("[ALNS] allow_lateral_transshipment=True ignored: baseline ALNS keeps y=0 by design.")
        _ = add_valid_16_20  # kept for signature parity with Gurobi baseline
        self._enforce_integer = bool(enforce_integer_flows)
        self._min_visit_activity_qty = max(0.0, float(min_visit_activity_qty))
        self._min_visit_delivery_qty = max(0.0, float(min_visit_delivery_qty))
        self._msg = bool(msg)
        self._rng = random.Random(int(seed))

        d = self.data
        self._dispatch_periods = self._compute_dispatch_periods(cw_dispatch_cycle)

        t0 = time.perf_counter()
        deadline = (t0 + float(time_limit)) if time_limit is not None else None

        current = self._build_greedy_initial_solution()
        curr_cost, curr_feasible = self._evaluate(current)
        best = current.clone()
        best_cost = curr_cost
        best_feasible = curr_feasible

        temperature = float(initial_temperature) if initial_temperature is not None else max(1.0, abs(curr_cost) * 0.05 + 1.0)

        destroy_ops = [
            ("destroy_random_delivery", self._destroy_random_delivery),
            ("destroy_worst_delivery", self._destroy_worst_delivery),
            ("destroy_random_route", self._destroy_random_route),
            ("destroy_random_period", self._destroy_random_period),
            ("destroy_shaw_stores", self._destroy_shaw_stores),
            ("destroy_low_demand_stores", self._destroy_low_demand_stores),
        ]
        repair_ops = [
            ("repair_greedy", self._repair_greedy),
            ("repair_regret2", self._repair_regret2),
            ("repair_random", self._repair_random),
        ]
        w_destroy = [1.0] * len(destroy_ops)
        w_repair = [1.0] * len(repair_ops)
        pi_destroy = [0.0] * len(destroy_ops)
        pi_repair = [0.0] * len(repair_ops)
        theta_destroy = [0] * len(destroy_ops)
        theta_repair = [0] * len(repair_ops)

        n_accept = 0
        n_improve = 0
        n_new_best = 0
        history: List[Dict[str, float]] = [{
            "iteration": 0,
            "current_cost": float(curr_cost),
            "candidate_cost": float(curr_cost),
            "best_cost": float(best_cost),
            "temperature": float(temperature),
            "accepted": 1.0,
            "new_best": 1.0,
            "destroy_op": "initial",
            "repair_op": "initial",
            "feasible": 1.0 if curr_feasible else 0.0,
        }]

        for iteration in range(1, int(max_iterations) + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                break

            d_idx = self._roulette_pick(w_destroy)
            r_idx = self._roulette_pick(w_repair)
            candidate = current.clone()

            # Intensity: remove between 5% and 25% of served store-period assignments
            served_pairs = [key for key, qty in candidate.deliv.items() if qty > 1e-9]
            n_remove = max(1, min(len(served_pairs), int(round(self._rng.uniform(0.05, 0.25) * max(1, len(served_pairs))))))

            destroy_ops[d_idx][1](candidate, n_remove)
            repair_ops[r_idx][1](candidate)

            cand_cost, cand_feasible = self._evaluate(candidate)
            theta_destroy[d_idx] += 1
            theta_repair[r_idx] += 1

            delta = cand_cost - curr_cost
            accepted = False
            if cand_feasible and (not curr_feasible or delta <= 0.0):
                accepted = True
            elif cand_feasible and self._rng.random() < math.exp(-delta / max(temperature, 1e-9)):
                accepted = True

            score = 0.0
            if cand_feasible and (cand_cost < best_cost - 1e-9 or (not best_feasible)):
                best = candidate.clone()
                best_cost = cand_cost
                best_feasible = True
                score = self.SIGMA_NEW_BEST
                n_new_best += 1
                n_improve += 1
                current = candidate
                curr_cost = cand_cost
                curr_feasible = True
                n_accept += 1
            elif accepted:
                n_accept += 1
                if delta < 0.0:
                    score = self.SIGMA_BETTER
                    n_improve += 1
                else:
                    score = self.SIGMA_ACCEPTED
                current = candidate
                curr_cost = cand_cost
                curr_feasible = cand_feasible

            pi_destroy[d_idx] += score
            pi_repair[r_idx] += score

            history.append({
                "iteration": int(iteration),
                "current_cost": float(curr_cost),
                "candidate_cost": float(cand_cost),
                "best_cost": float(best_cost),
                "temperature": float(temperature),
                "accepted": 1.0 if accepted else 0.0,
                "new_best": 1.0 if score == self.SIGMA_NEW_BEST else 0.0,
                "destroy_op": destroy_ops[d_idx][0],
                "repair_op": repair_ops[r_idx][0],
                "feasible": 1.0 if cand_feasible else 0.0,
            })

            temperature = max(1e-6, temperature * cooling_rate)

            if iteration % segment_size == 0:
                for i in range(len(w_destroy)):
                    if theta_destroy[i] > 0:
                        w_destroy[i] = (1.0 - reaction_factor) * w_destroy[i] + reaction_factor * (pi_destroy[i] / theta_destroy[i])
                        w_destroy[i] = max(0.05, w_destroy[i])
                    pi_destroy[i] = 0.0
                    theta_destroy[i] = 0
                for i in range(len(w_repair)):
                    if theta_repair[i] > 0:
                        w_repair[i] = (1.0 - reaction_factor) * w_repair[i] + reaction_factor * (pi_repair[i] / theta_repair[i])
                        w_repair[i] = max(0.05, w_repair[i])
                    pi_repair[i] = 0.0
                    theta_repair[i] = 0
                if self._msg:
                    print(f"[ALNS] iter={iteration} best={best_cost:.4f} curr={curr_cost:.4f} T={temperature:.3f}")

        runtime = time.perf_counter() - t0
        solution = self._build_full_irpt_solution(
            best,
            runtime_seconds=runtime,
            iterations=iteration,
            n_accept=n_accept,
            n_improve=n_improve,
            n_new_best=n_new_best,
            final_cost=best_cost,
            feasible=best_feasible,
        )
        solution.alns_history = history
        return solution

    # ---------------------------------------------------------- initial build

    def _compute_dispatch_periods(self, cw_dispatch_cycle: Optional[int]) -> Set[Period]:
        d = self.data
        if cw_dispatch_cycle is None or int(cw_dispatch_cycle) <= 1:
            return set(d.periods)
        cycle = int(cw_dispatch_cycle)
        t0 = min(d.periods) if d.periods else 0
        return {t for t in d.periods if (t - t0) % cycle == 0}

    def _target_delivery(self, s: Store, p: Product, t: Period, prev_inv: float) -> float:
        """How much to push to (s,p) in period t: cover current demand + small safety buffer, bounded by max store inv."""
        d = self.data
        demand = float(d.demand.get((s, p, t), 0.0))
        # Look-ahead: if the next period is not a dispatch period, add its demand too.
        future_need = 0.0
        remaining_periods = [tau for tau in d.periods if tau > t]
        for tau in remaining_periods:
            future_need += float(d.demand.get((s, p, tau), 0.0))
            if tau in self._dispatch_periods:
                break
        need = max(0.0, demand + future_need - prev_inv)
        max_room = max(0.0, float(d.max_inventory_store.get((s, p), float("inf"))) - prev_inv)
        return min(need, max_room)

    def _build_greedy_initial_solution(self) -> _ALNSState:
        d = self.data
        state = _ALNSState()
        inv_store = {(s, p): float(d.init_inventory_store.get((s, p), 0.0)) for s in d.stores for p in d.products}
        inv_wh = {p: float(d.init_inventory_wh.get(p, 0.0)) for p in d.products}

        for t in d.periods:
            for p in d.products:
                inv_wh[p] = inv_wh[p] + float(d.replenishment_wh.get((p, t), 0.0))

            if t not in self._dispatch_periods:
                for s in d.stores:
                    for p in d.products:
                        demand = float(d.demand.get((s, p, t), 0.0))
                        inv_store[(s, p)] = max(0.0, inv_store[(s, p)] - demand)
                continue

            # Compute targets per (store, product); subject to WH availability.
            targets: Dict[Tuple[Store, Product], float] = {}
            for s in d.stores:
                for p in d.products:
                    tgt = self._target_delivery(s, p, t, inv_store[(s, p)])
                    if tgt > 1e-9:
                        targets[(s, p)] = tgt
            # Cap by warehouse inventory per product
            for p in d.products:
                total = sum(q for (s, pp), q in targets.items() if pp == p)
                if total > inv_wh[p] + 1e-9 and total > 0:
                    scale = inv_wh[p] / total
                    for key in list(targets.keys()):
                        if key[1] == p:
                            targets[key] *= scale

            # Assign to vehicles via nearest-neighbor, capacity-respecting
            stores_need = sorted({s for (s, p) in targets if sum(targets.get((s, q), 0.0) for q in d.products) > 1e-9},
                                 key=lambda s: -sum(targets.get((s, q), 0.0) for q in d.products))
            remaining_stores = list(stores_need)
            for v in d.vehicles:
                if not remaining_stores:
                    break
                capacity = float(d.vehicle_capacity)
                route: List[Store] = []
                current_node: Node = d.warehouse
                while remaining_stores:
                    # pick nearest store whose total target fits
                    candidates = []
                    for s in remaining_stores:
                        load_s = sum(targets.get((s, q), 0.0) for q in d.products)
                        if load_s <= capacity + 1e-9:
                            dist = float(d.distance.get((current_node, s), 0.0))
                            candidates.append((dist, load_s, s))
                    if not candidates:
                        break
                    candidates.sort(key=lambda item: (item[0], -item[1]))
                    _, load_s, s = candidates[0]
                    route.append(s)
                    capacity -= load_s
                    current_node = s
                    remaining_stores.remove(s)
                    for p in d.products:
                        q = float(targets.get((s, p), 0.0))
                        if q > 1e-9:
                            state.deliv[(s, p, v, t)] = q
                if route:
                    state.routes[(t, v)] = route

            # Update inventories after scheduled dispatches in period t
            for p in d.products:
                shipped = sum(state.deliv.get((s, p, v, t), 0.0) for s in d.stores for v in d.vehicles)
                inv_wh[p] = max(0.0, inv_wh[p] - shipped)
            for s in d.stores:
                for p in d.products:
                    demand = float(d.demand.get((s, p, t), 0.0))
                    qdir = sum(state.deliv.get((s, p, v, t), 0.0) for v in d.vehicles)
                    inv_store[(s, p)] = max(0.0, inv_store[(s, p)] + qdir - demand)
        return state

    # ------------------------------------------------------------ evaluation

    def _evaluate(self, state: _ALNSState) -> Tuple[float, bool]:
        d = self.data
        cost = 0.0
        feasible = True

        inv_store = {(s, p): float(d.init_inventory_store.get((s, p), 0.0)) for s in d.stores for p in d.products}
        inv_wh = {p: float(d.init_inventory_wh.get(p, 0.0)) for p in d.products}

        for t in d.periods:
            for p in d.products:
                inv_wh[p] += float(d.replenishment_wh.get((p, t), 0.0))

            # Aggregate direct shipments
            qdir_pt: Dict[Tuple[Store, Product], float] = {}
            for (s, p, v, tau), qty in state.deliv.items():
                if tau == t and qty > 0.0:
                    qdir_pt[(s, p)] = qdir_pt.get((s, p), 0.0) + qty

            # Dispatch cycle check
            if t not in self._dispatch_periods:
                if any(q > 1e-6 for q in qdir_pt.values()):
                    feasible = False
                    cost += 1e6 * sum(qdir_pt.values())

            # Warehouse balance
            for p in d.products:
                ship_p = sum(q for (s, pp), q in qdir_pt.items() if pp == p)
                inv_wh[p] -= ship_p
                if inv_wh[p] < -1e-6:
                    feasible = False
                    cost += 1e6 * (-inv_wh[p])
                    inv_wh[p] = 0.0
                cost += float(d.holding_cost_wh.get(p, 0.0)) * max(0.0, inv_wh[p])
                # Direct shipping cost
                for s in d.stores:
                    q = qdir_pt.get((s, p), 0.0)
                    cost += float(d.ship_cost_cw.get((s, p), 0.0)) * q

            # Warehouse aggregate capacity
            wh_total = sum(inv_wh[p] for p in d.products)
            cap_wh = float(d.node_capacity.get(d.warehouse, float("inf")))
            if wh_total > cap_wh + 1e-6:
                feasible = False
                cost += 1e5 * (wh_total - cap_wh)

            # Store balance
            for s in d.stores:
                for p in d.products:
                    demand = float(d.demand.get((s, p, t), 0.0))
                    q = qdir_pt.get((s, p), 0.0)
                    new_inv = inv_store[(s, p)] + q - demand
                    shortage = max(0.0, -new_inv)
                    inv_store[(s, p)] = max(0.0, new_inv)
                    cost += float(d.holding_cost_store.get((s, p), 0.0)) * inv_store[(s, p)]
                    cost += float(d.shortage_cost.get((s, p), 0.0)) * shortage
                    if inv_store[(s, p)] > float(d.max_inventory_store.get((s, p), float("inf"))) + 1e-6:
                        feasible = False
                        cost += 1e5 * (inv_store[(s, p)] - float(d.max_inventory_store.get((s, p), 0.0)))
                # Node capacity
                cap_s = float(d.node_capacity.get(s, float("inf")))
                total_inv_s = sum(inv_store[(s, q)] for q in d.products)
                if total_inv_s > cap_s + 1e-6:
                    feasible = False
                    cost += 1e5 * (total_inv_s - cap_s)

            # Single-visit-per-period: each store must appear in at most one vehicle's route in period t.
            visit_counts: Dict[Store, int] = {}
            for (tt, vv), rt in state.routes.items():
                if tt != t:
                    continue
                for s in rt:
                    visit_counts[s] = visit_counts.get(s, 0) + 1
            for s, cnt in visit_counts.items():
                if cnt > 1:
                    feasible = False
                    cost += 1e5 * (cnt - 1)

            # Routing cost and vehicle/route feasibility per period
            vehicles_used = 0
            for v in d.vehicles:
                route = list(state.routes.get((t, v), []))
                load_total = sum(state.deliv.get((s, p, v, t), 0.0) for s in route for p in d.products)
                if not route and load_total <= 1e-9:
                    continue
                if load_total > float(d.vehicle_capacity) + 1e-6:
                    feasible = False
                    cost += 1e5 * (load_total - float(d.vehicle_capacity))
                if load_total > 1e-9 or route:
                    vehicles_used += 1
                    cost += float(d.vehicle_fixed_cost)
                # Distance cost along CW -> s1 -> ... -> sk -> CW
                path = [d.warehouse] + route + [d.warehouse]
                for i, j in zip(path[:-1], path[1:]):
                    cost += float(d.alpha) * float(d.distance.get((i, j), 0.0))
                # min_visit_activity_qty: each visited store must receive at least the threshold
                if self._min_visit_activity_qty > 0:
                    for s in route:
                        got = sum(state.deliv.get((s, p, v, t), 0.0) for p in d.products)
                        if got < self._min_visit_activity_qty - 1e-6:
                            feasible = False
                            cost += 1e4 * (self._min_visit_activity_qty - got)
                if self._min_visit_delivery_qty > 0:
                    for s in route:
                        got = sum(state.deliv.get((s, p, v, t), 0.0) for p in d.products)
                        if got < self._min_visit_delivery_qty - 1e-6:
                            feasible = False
                            cost += 1e4 * (self._min_visit_delivery_qty - got)
            if vehicles_used > int(d.max_vehicles_used):
                feasible = False
                cost += 1e5 * (vehicles_used - int(d.max_vehicles_used))

        return cost, feasible

    # ---------------------------------------------------------- destroy ops

    def _served_keys(self, state: _ALNSState) -> List[Tuple[Store, Product, Vehicle, Period]]:
        return [k for k, v in state.deliv.items() if v > 1e-9]

    def _drop_empty_route_entries(self, state: _ALNSState, t: Period, v: Vehicle) -> None:
        route = state.routes.get((t, v), [])
        new_route = [s for s in route if any(state.deliv.get((s, p, v, t), 0.0) > 1e-9 for p in self.data.products)]
        if new_route:
            state.routes[(t, v)] = new_route
        else:
            state.routes.pop((t, v), None)

    def _destroy_random_delivery(self, state: _ALNSState, k: int) -> None:
        keys = self._served_keys(state)
        if not keys:
            return
        self._rng.shuffle(keys)
        for key in keys[:k]:
            state.deliv[key] = 0.0
            state.deliv.pop(key, None)
            _, _, v, t = key
            self._drop_empty_route_entries(state, t, v)

    def _destroy_worst_delivery(self, state: _ALNSState, k: int) -> None:
        d = self.data
        keys = self._served_keys(state)
        if not keys:
            return
        # score = direct_ship_cost * qty / (demand + 1)  (expensive deliveries relative to demand)
        scored = []
        for (s, p, v, t) in keys:
            qty = state.deliv[(s, p, v, t)]
            dem = float(d.demand.get((s, p, t), 1.0)) + 1.0
            score = float(d.ship_cost_cw.get((s, p), 0.0)) * qty / dem
            scored.append((score, (s, p, v, t)))
        scored.sort(reverse=True)
        for _, key in scored[:k]:
            state.deliv.pop(key, None)
            _, _, v, t = key
            self._drop_empty_route_entries(state, t, v)

    def _destroy_random_route(self, state: _ALNSState, k: int) -> None:
        route_keys = list(state.routes.keys())
        if not route_keys:
            return
        self._rng.shuffle(route_keys)
        for (t, v) in route_keys[: max(1, k // 4)]:
            route = state.routes.pop((t, v), [])
            for s in route:
                for p in self.data.products:
                    state.deliv.pop((s, p, v, t), None)

    def _destroy_random_period(self, state: _ALNSState, k: int) -> None:
        periods = sorted({t for (t, _) in state.routes.keys()})
        if not periods:
            return
        t = self._rng.choice(periods)
        keys_to_drop = [key for key in state.routes if key[0] == t]
        for key in keys_to_drop:
            _, v = key
            for s in state.routes.pop(key, []):
                for p in self.data.products:
                    state.deliv.pop((s, p, v, t), None)

    def _destroy_shaw_stores(self, state: _ALNSState, k: int) -> None:
        """Remove geographically close stores across a single period."""
        d = self.data
        periods_with_routes = sorted({t for (t, _) in state.routes.keys()})
        if not periods_with_routes:
            return
        t = self._rng.choice(periods_with_routes)
        visited = [s for (tt, v), route in state.routes.items() if tt == t for s in route]
        if not visited:
            return
        seed_store = self._rng.choice(visited)
        scored = sorted(visited, key=lambda s: float(d.distance.get((seed_store, s), 0.0)))
        remove = set(scored[: max(1, k // 3)])
        for (tt, v) in list(state.routes.keys()):
            if tt != t:
                continue
            new_route = [s for s in state.routes[(tt, v)] if s not in remove]
            if new_route:
                state.routes[(tt, v)] = new_route
            else:
                state.routes.pop((tt, v), None)
            for s in remove:
                for p in d.products:
                    state.deliv.pop((s, p, v, tt), None)

    def _destroy_low_demand_stores(self, state: _ALNSState, k: int) -> None:
        d = self.data
        keys = self._served_keys(state)
        if not keys:
            return
        scored = []
        for (s, p, v, t) in keys:
            dem = float(d.demand.get((s, p, t), 0.0))
            scored.append((dem, (s, p, v, t)))
        scored.sort()
        for _, key in scored[: max(1, k // 2)]:
            state.deliv.pop(key, None)
            _, _, v, t = key
            self._drop_empty_route_entries(state, t, v)

    # ---------------------------------------------------------- repair ops

    def _snap_qty(self, qty: float) -> float:
        """Floor to integer when integer flows are enforced; identity otherwise."""
        if getattr(self, "_enforce_integer", False):
            return float(math.floor(qty + 1e-9))
        return float(qty)

    def _collect_unserved_requests(self, state: _ALNSState) -> List[Tuple[Store, Product, Period, float]]:
        d = self.data
        requests: List[Tuple[Store, Product, Period, float]] = []
        inv_store = {(s, p): float(d.init_inventory_store.get((s, p), 0.0)) for s in d.stores for p in d.products}
        inv_wh = {p: float(d.init_inventory_wh.get(p, 0.0)) for p in d.products}
        for t in d.periods:
            for p in d.products:
                inv_wh[p] += float(d.replenishment_wh.get((p, t), 0.0))
            for p in d.products:
                for s in d.stores:
                    qdir = sum(state.deliv.get((s, p, v, t), 0.0) for v in d.vehicles)
                    inv_wh[p] = max(0.0, inv_wh[p] - qdir)
            for s in d.stores:
                for p in d.products:
                    qdir = sum(state.deliv.get((s, p, v, t), 0.0) for v in d.vehicles)
                    demand = float(d.demand.get((s, p, t), 0.0))
                    ending = inv_store[(s, p)] + qdir - demand
                    if t in self._dispatch_periods and ending < -1e-6:
                        shortfall = -ending
                        max_room = max(0.0, float(d.max_inventory_store.get((s, p), float("inf"))) - (inv_store[(s, p)] + qdir))
                        addable = min(shortfall, max_room, inv_wh.get(p, 0.0))
                        addable = self._snap_qty(addable)
                        if addable > 1e-9:
                            requests.append((s, p, t, addable))
                    inv_store[(s, p)] = max(0.0, ending)
        return requests

    def _insertion_cost(self, state: _ALNSState, s: Store, p: Product, t: Period, v: Vehicle, qty: float) -> Optional[float]:
        """Return incremental cost of inserting (s,p,t,qty) on vehicle v; None if infeasible."""
        d = self.data
        route = state.routes.get((t, v), [])
        # Single-visit-per-period constraint: store s must not be on another vehicle's route in period t.
        for (tt, vv), other_route in state.routes.items():
            if tt == t and vv != v and s in other_route:
                return None
        load_total = sum(state.deliv.get((ss, pp, v, t), 0.0) for ss in route for pp in d.products)
        if load_total + qty > float(d.vehicle_capacity) + 1e-6:
            return None
        ship_cost = float(d.ship_cost_cw.get((s, p), 0.0)) * qty
        if s in route:
            return ship_cost  # no new arc cost
        # Finding best insertion position in route
        best_delta = None
        path = [d.warehouse] + route + [d.warehouse]
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            delta = float(d.distance.get((a, s), 0.0)) + float(d.distance.get((s, b), 0.0)) - float(d.distance.get((a, b), 0.0))
            if best_delta is None or delta < best_delta:
                best_delta = delta
        added_vehicle_cost = 0.0 if route else float(d.vehicle_fixed_cost)
        return ship_cost + float(d.alpha) * (best_delta or 0.0) + added_vehicle_cost

    def _apply_insertion(self, state: _ALNSState, s: Store, p: Product, t: Period, v: Vehicle, qty: float) -> None:
        d = self.data
        route = state.routes.get((t, v), [])
        if s not in route:
            path = [d.warehouse] + route + [d.warehouse]
            best_pos = 0
            best_delta = None
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                delta = float(d.distance.get((a, s), 0.0)) + float(d.distance.get((s, b), 0.0)) - float(d.distance.get((a, b), 0.0))
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_pos = i
            route.insert(best_pos, s)
            state.routes[(t, v)] = route
        state.deliv[(s, p, v, t)] = state.deliv.get((s, p, v, t), 0.0) + self._snap_qty(qty)

    def _repair_greedy(self, state: _ALNSState) -> None:
        d = self.data
        requests = self._collect_unserved_requests(state)
        self._rng.shuffle(requests)
        for (s, p, t, qty) in requests:
            if t not in self._dispatch_periods:
                continue
            best_v = None
            best_c = None
            for v in d.vehicles:
                c = self._insertion_cost(state, s, p, t, v, qty)
                if c is None:
                    continue
                if best_c is None or c < best_c:
                    best_c = c
                    best_v = v
            if best_v is None:
                continue
            # Respect max_vehicles per period
            active_vehicles = {vv for (tt, vv) in state.routes if tt == t}
            if best_v not in active_vehicles and len(active_vehicles) >= int(d.max_vehicles_used):
                continue
            self._apply_insertion(state, s, p, t, best_v, qty)

    def _repair_regret2(self, state: _ALNSState) -> None:
        d = self.data
        requests = self._collect_unserved_requests(state)
        while requests:
            best_req_idx = None
            best_regret = -1.0
            best_v = None
            best_insert_cost = None
            for idx, (s, p, t, qty) in enumerate(requests):
                if t not in self._dispatch_periods:
                    continue
                costs = []
                for v in d.vehicles:
                    c = self._insertion_cost(state, s, p, t, v, qty)
                    if c is not None:
                        costs.append((c, v))
                if not costs:
                    continue
                costs.sort()
                cheapest = costs[0][0]
                second = costs[1][0] if len(costs) > 1 else cheapest + 1e6
                regret = second - cheapest
                if regret > best_regret:
                    best_regret = regret
                    best_req_idx = idx
                    best_v = costs[0][1]
                    best_insert_cost = cheapest
            if best_req_idx is None:
                break
            (s, p, t, qty) = requests.pop(best_req_idx)
            active_vehicles = {vv for (tt, vv) in state.routes if tt == t}
            if best_v not in active_vehicles and len(active_vehicles) >= int(d.max_vehicles_used):
                continue
            self._apply_insertion(state, s, p, t, best_v, qty)

    def _repair_random(self, state: _ALNSState) -> None:
        d = self.data
        requests = self._collect_unserved_requests(state)
        self._rng.shuffle(requests)
        for (s, p, t, qty) in requests:
            if t not in self._dispatch_periods:
                continue
            feasible_vs = [v for v in d.vehicles if self._insertion_cost(state, s, p, t, v, qty) is not None]
            if not feasible_vs:
                continue
            v = self._rng.choice(feasible_vs)
            active_vehicles = {vv for (tt, vv) in state.routes if tt == t}
            if v not in active_vehicles and len(active_vehicles) >= int(d.max_vehicles_used):
                continue
            self._apply_insertion(state, s, p, t, v, qty)

    # ------------------------------------------------------- adaptive weights

    def _roulette_pick(self, weights: List[float]) -> int:
        total = sum(weights)
        if total <= 0.0:
            return self._rng.randint(0, len(weights) - 1)
        r = self._rng.uniform(0.0, total)
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                return i
        return len(weights) - 1

    # ----------------------------------------------- FullIRPTSolution builder

    def _build_full_irpt_solution(
        self,
        state: _ALNSState,
        runtime_seconds: float,
        iterations: int,
        n_accept: int,
        n_improve: int,
        n_new_best: int,
        final_cost: float,
        feasible: bool,
    ) -> FullIRPTSolution:
        d = self.data
        N = d.stores
        P = d.products
        T = d.periods
        V = d.vehicles
        CW = d.warehouse
        N0 = [CW] + N

        # Initialize all decision dicts to zero
        x = {(i, j, v, t): 0 for i in N0 for j in N0 if i != j for v in V for t in T}
        u = {(v, t): 0 for v in V for t in T}
        z = {(i, v, t): 0 for i in N0 for v in V for t in T}
        q = {(p, i, j, v, t): 0.0 for p in P for i in N0 for j in N0 if i != j for v in V for t in T}
        y = {(i, j, p, v, t): 0.0 for i in N for j in N if i != j for p in P for v in V for t in T}
        deliv = {(s, p, v, t): 0.0 for s in N for p in P for v in V for t in T}
        load = {(i, v, t): 0.0 for i in N0 for v in V for t in T}
        direct_ship_q = {(s, p, t): 0.0 for s in N for p in P for t in T}
        inv_store = {(s, p, t): 0.0 for s in N for p in P for t in T}
        inv_wh = {(p, t): 0.0 for p in P for t in T}
        shortage = {(s, p, t): 0.0 for s in N for p in P for t in T}

        # Fill deliv and direct_ship_q
        for (s, p, v, t), qty in state.deliv.items():
            if qty > 0.0:
                deliv[(s, p, v, t)] = float(qty)
                direct_ship_q[(s, p, t)] = direct_ship_q.get((s, p, t), 0.0) + float(qty)

        # Fill routing: x, u, z, q, load
        for (t, v), route in state.routes.items():
            if not route:
                continue
            u[(v, t)] = 1
            z[(CW, v, t)] = 1
            path = [CW] + list(route) + [CW]
            for i, j in zip(path[:-1], path[1:]):
                x[(i, j, v, t)] = 1
                if j != CW:
                    z[(j, v, t)] = 1
            # q: cumulative remaining deliveries on each outbound arc
            # Walk forward, starting with total load at CW and subtracting delivery at each stop.
            for p in P:
                remaining = sum(deliv.get((s, p, v, t), 0.0) for s in route)
                load[(CW, v, t)] = load.get((CW, v, t), 0.0) + sum(deliv.get((s, pp, v, t), 0.0) for pp in P if pp == p)
                prev = CW
                for s in route:
                    q[(p, prev, s, v, t)] = float(remaining)
                    remaining -= float(deliv.get((s, p, v, t), 0.0))
                    prev = s
                q[(p, prev, CW, v, t)] = max(0.0, float(remaining))
            # load per node = cumulative deliveries still to be made at that node
            total_load = sum(deliv.get((s, p, v, t), 0.0) for s in route for p in P)
            load[(CW, v, t)] = float(total_load)
            running = total_load
            for s in route:
                running -= sum(deliv.get((s, p, v, t), 0.0) for p in P)
                load[(s, v, t)] = max(0.0, float(running))

        # Forward simulate inventories and shortages
        cur_inv_store = {(s, p): float(d.init_inventory_store.get((s, p), 0.0)) for s in N for p in P}
        cur_inv_wh = {p: float(d.init_inventory_wh.get(p, 0.0)) for p in P}
        for t in T:
            for p in P:
                cur_inv_wh[p] += float(d.replenishment_wh.get((p, t), 0.0))
                ship = sum(direct_ship_q.get((s, p, t), 0.0) for s in N)
                cur_inv_wh[p] = max(0.0, cur_inv_wh[p] - ship)
                inv_wh[(p, t)] = float(cur_inv_wh[p])
            for s in N:
                for p in P:
                    qdir = direct_ship_q.get((s, p, t), 0.0)
                    demand = float(d.demand.get((s, p, t), 0.0))
                    new_inv = cur_inv_store[(s, p)] + qdir - demand
                    if getattr(self, "_enforce_integer", False):
                        new_inv = float(math.floor(new_inv + 1e-9))
                    shortage[(s, p, t)] = float(max(0.0, -new_inv))
                    cur_inv_store[(s, p)] = max(0.0, new_inv)
                    inv_store[(s, p, t)] = float(cur_inv_store[(s, p)])

        efficiency_metrics = {
            "alns_runtime_seconds": float(runtime_seconds),
            "alns_iterations": float(iterations),
            "alns_accepts": float(n_accept),
            "alns_improvements": float(n_improve),
            "alns_new_best": float(n_new_best),
            # Parity keys used by print_efficiency_metrics
            "gurobi_runtime_seconds": float(runtime_seconds),
            "lp_iterations": 0.0,
            "barrier_iterations": 0.0,
            "nodes_explored": float(iterations),
            "lp_relaxations_solved_estimate": 1.0,
        }

        return FullIRPTSolution(
            status="ALNS-Feasible" if feasible else "ALNS-InfeasiblePenalized",
            objective=float(final_cost),
            direct_ship_q=direct_ship_q,
            inv_store=inv_store,
            inv_wh=inv_wh,
            shortage=shortage,
            x=x,
            u=u,
            z=z,
            q=q,
            y=y,
            deliv=deliv,
            load=load,
            efficiency_metrics=efficiency_metrics,
        )


# ============================================================================
# ACHAMRAH-STYLE FULLER IRPT MODEL  (kept for reference / other experiments)
# ============================================================================

class AchamrahFullIRPTModel:
    """
    Practical implementation of the paper-style IRPT model.

    Included:
    - Base constraints (2)-(15)
    - Valid inequalities (16)-(20)

    Not fully included:
    - Constraint (21), because the paper separates those cuts dynamically in branch-and-cut.
    """

    def __init__(self, data: IRPData):
        self.data = data

    def solve(
        self,
        msg: bool = False,
        time_limit: Optional[int] = None,
        enforce_integer_flows: bool = False,
        add_valid_16_20: bool = True,
        allow_lateral_transshipment: bool = True,
        min_visit_activity_qty: float = 1.0,
        min_visit_delivery_qty: float = 0.0,
        cw_dispatch_cycle: Optional[int] = 5,
    ) -> FullIRPTSolution:
        d = self.data
        N = d.stores
        P = d.products
        T = d.periods
        V = d.vehicles
        CW = d.warehouse
        N0 = [CW] + N

        mdl = gp.Model("Achamrah_Full_IRPT", env=get_gurobi_env())
        mdl.Params.OutputFlag = 1 if msg else 0
        if time_limit is not None:
            mdl.Params.TimeLimit = time_limit

        flow_vtype = GRB.INTEGER if enforce_integer_flows else GRB.CONTINUOUS

        I_s_keys = [(s, p, t) for s in N for p in P for t in T]
        I_w_keys = [(p, t) for p in P for t in T]
        Qdir_keys = [(s, p, t) for s in N for p in P for t in T]
        q_keys = [(p, i, j, v, t) for p in P for i in N0 for j in N0 if i != j for v in V for t in T]
        y_keys = [(i, j, p, v, t) for i in N for j in N if i != j for p in P for v in V for t in T]
        deliv_keys = [(s, p, v, t) for s in N for p in P for v in V for t in T]
        load_keys = [(i, v, t) for i in N0 for v in V for t in T]
        ordv_keys = [(s, v, t) for s in N for v in V for t in T]
        B_keys = [(s, p, t) for s in N for p in P for t in T]
        x_keys = [(i, j, v, t) for i in N0 for j in N0 if i != j for v in V for t in T]
        u_keys = [(v, t) for v in V for t in T]
        z_keys = [(i, v, t) for i in N0 for v in V for t in T]

        I_s = mdl.addVars(I_s_keys, lb=0.0, vtype=flow_vtype, name="I_s")
        I_w = mdl.addVars(I_w_keys, lb=0.0, vtype=flow_vtype, name="I_w")
        Qdir = mdl.addVars(Qdir_keys, lb=0.0, vtype=flow_vtype, name="Qdir")
        q = mdl.addVars(q_keys, lb=0.0, vtype=flow_vtype, name="q")
        y = mdl.addVars(y_keys, lb=0.0, vtype=flow_vtype, name="y")
        deliv = mdl.addVars(deliv_keys, lb=0.0, vtype=flow_vtype, name="deliv")
        load = mdl.addVars(load_keys, lb=0.0, ub=d.vehicle_capacity, vtype=flow_vtype, name="load")
        ordv = mdl.addVars(ordv_keys, lb=0.0, ub=len(N), vtype=GRB.CONTINUOUS, name="ordv")
        if not allow_lateral_transshipment:
            for key in y_keys:
                y[key].UB = 0.0
        B = mdl.addVars(B_keys, lb=0.0, vtype=flow_vtype, name="B")
        x = mdl.addVars(x_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")
        u = mdl.addVars(u_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="u")
        z = mdl.addVars(z_keys, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="z")

        mdl.setObjective(
            gp.quicksum(d.ship_cost_cw[(s, p)] * Qdir[(s, p, t)] for s, p, t in Qdir_keys)
            + gp.quicksum(d.holding_cost_store[(s, p)] * I_s[(s, p, t)] for s, p, t in I_s_keys)
            + gp.quicksum(d.holding_cost_wh[p] * I_w[(p, t)] for p, t in I_w_keys)
            + gp.quicksum(d.alpha * d.distance[(i, j)] * x[(i, j, v, t)] for i, j, v, t in x_keys)
            + gp.quicksum(d.vehicle_fixed_cost * u[(v, t)] for v, t in u_keys)
            + gp.quicksum(d.transship_unit_cost[(i, j)] * y[(i, j, p, v, t)] for i, j, p, v, t in y_keys)
            + gp.quicksum(d.shortage_cost[(s, p)] * B[(s, p, t)] for s, p, t in B_keys),
            GRB.MINIMIZE,
        )

        first_t = min(T)
        restricted_cw_dispatch_periods: Optional[Set[Period]] = None
        if cw_dispatch_cycle is not None and int(cw_dispatch_cycle) > 1:
            cycle = int(cw_dispatch_cycle)
            restricted_cw_dispatch_periods = {t for t in T if (t - first_t) % cycle == 0}

        for s in N:
            for p in P:
                for t in T:
                    prev = d.init_inventory_store[(s, p)] if t == first_t else I_s[(s, p, t - 1)]
                    mdl.addConstr(
                        I_s[(s, p, t)]
                        == prev
                        + Qdir[(s, p, t)]
                        - d.demand[(s, p, t)]
                        + B[(s, p, t)]
                        + gp.quicksum(y[(j, s, p, v, t)] for j in N if j != s for v in V)
                        - gp.quicksum(y[(s, j, p, v, t)] for j in N if j != s for v in V)
                    )

        for p in P:
            for t in T:
                prev = d.init_inventory_wh[p] if t == first_t else I_w[(p, t - 1)]
                mdl.addConstr(
                    I_w[(p, t)]
                    == prev
                    - gp.quicksum(Qdir[(s, p, t)] for s in N)
                    + d.replenishment_wh[(p, t)]
                )

        for s in N:
            for p in P:
                for t in T:
                    mdl.addConstr(Qdir[(s, p, t)] == gp.quicksum(deliv[(s, p, v, t)] for v in V))
                    if restricted_cw_dispatch_periods is not None and t not in restricted_cw_dispatch_periods:
                        mdl.addConstr(Qdir[(s, p, t)] == 0.0)
                    for v in V:
                        mdl.addConstr(
                            deliv[(s, p, v, t)]
                            + gp.quicksum(y[(i, s, p, v, t)] for i in N if i != s)
                            - gp.quicksum(y[(s, j, p, v, t)] for j in N if j != s)
                            == gp.quicksum(q[(p, i, s, v, t)] for i in N0 if i != s)
                            - gp.quicksum(q[(p, s, j, v, t)] for j in N0 if j != s)
                        )
                        mdl.addConstr(deliv[(s, p, v, t)] <= d.vehicle_capacity * z[(s, v, t)])

        for i in N:
            for v in V:
                for t in T:
                    mdl.addConstr(gp.quicksum(q[(p, i, CW, v, t)] for p in P) == 0)

        for s in N:
            for t in T:
                mdl.addConstr(gp.quicksum(I_s[(s, p, t)] for p in P) <= d.node_capacity[s])
        for t in T:
            mdl.addConstr(gp.quicksum(I_w[(p, t)] for p in P) <= d.node_capacity[CW])

        for i in N0:
            for j in N0:
                if i == j:
                    continue
                for v in V:
                    for t in T:
                        mdl.addConstr(gp.quicksum(q[(p, i, j, v, t)] for p in P) <= d.vehicle_capacity * x[(i, j, v, t)])

        for s in N:
            for p in P:
                for t in T:
                    begin_inv = d.init_inventory_store[(s, p)] if t == first_t else I_s[(s, p, t - 1)]
                    mdl.addConstr(gp.quicksum(y[(s, j, p, v, t)] for j in N if j != s for v in V) <= begin_inv)

        for j in N:
            for v in V:
                for t in T:
                    mdl.addConstr(
                        gp.quicksum(x[(i, j, v, t)] for i in N0 if i != j)
                        == gp.quicksum(x[(j, i, v, t)] for i in N0 if i != j)
                    )

        for j in N:
            for t in T:
                mdl.addConstr(gp.quicksum(x[(i, j, v, t)] for i in N0 if i != j for v in V) <= 1)

        for v in V:
            for t in T:
                mdl.addConstr(gp.quicksum(x[(CW, j, v, t)] for j in N) == u[(v, t)])

        for t in T:
            mdl.addConstr(gp.quicksum(u[(v, t)] for v in V) <= d.max_vehicles_used)

        for i in N0:
            for v in V:
                for t in T:
                    if i == CW:
                        mdl.addConstr(z[(i, v, t)] == u[(v, t)])
                    else:
                        mdl.addConstr(z[(i, v, t)] == gp.quicksum(x[(j, i, v, t)] for j in N0 if j != i))

        min_visit_activity_qty = max(0.0, float(min_visit_activity_qty))
        if min_visit_activity_qty > 0.0:
            for s in N:
                for v in V:
                    for t in T:
                        store_activity = (
                            gp.quicksum(deliv[(s, p, v, t)] for p in P)
                            + gp.quicksum(y[(i, s, p, v, t)] for i in N if i != s for p in P)
                            + gp.quicksum(y[(s, j, p, v, t)] for j in N if j != s for p in P)
                        )
                        mdl.addConstr(store_activity >= min_visit_activity_qty * z[(s, v, t)])

        min_visit_delivery_qty = max(0.0, float(min_visit_delivery_qty))
        if min_visit_delivery_qty > 0.0:
            for s in N:
                for v in V:
                    for t in T:
                        mdl.addConstr(
                            gp.quicksum(deliv[(s, p, v, t)] for p in P)
                            >= min_visit_delivery_qty * z[(s, v, t)]
                        )

        for p in P:
            for i in N0:
                for j in N0:
                    if i == j:
                        continue
                    for v in V:
                        for t in T:
                            mdl.addConstr(q[(p, i, j, v, t)] <= d.vehicle_capacity * x[(i, j, v, t)])

        for v in V:
            for t in T:
                mdl.addConstr(
                    load[(CW, v, t)]
                    == gp.quicksum(deliv[(s, p, v, t)] for s in N for p in P)
                )
                mdl.addConstr(load[(CW, v, t)] <= d.vehicle_capacity * u[(v, t)])

        for s in N:
            for v in V:
                for t in T:
                    mdl.addConstr(
                        load[(s, v, t)]
                        <= d.vehicle_capacity * gp.quicksum(x[(s, j, v, t)] for j in N0 if j != s)
                    )

        for i in N0:
            for j in N:
                if i == j:
                    continue
                for v in V:
                    for t in T:
                        delivered_at_j = gp.quicksum(deliv[(j, p, v, t)] for p in P)
                        mdl.addConstr(
                            load[(j, v, t)]
                            <= load[(i, v, t)] - delivered_at_j + d.vehicle_capacity * (1 - x[(i, j, v, t)])
                        )
                        mdl.addConstr(
                            load[(j, v, t)]
                            >= load[(i, v, t)] - delivered_at_j - d.vehicle_capacity * (1 - x[(i, j, v, t)])
                        )

        for i in N:
            for v in V:
                for t in T:
                    mdl.addConstr(load[(i, v, t)] >= gp.quicksum(q[(p, i, j, v, t)] for p in P for j in N0 if j != i))

        for s in N:
            for v in V:
                for t in T:
                    mdl.addConstr(ordv[(s, v, t)] <= len(N) * z[(s, v, t)])
                    mdl.addConstr(ordv[(s, v, t)] >= z[(s, v, t)])

        for i in N:
            for j in N:
                if i == j:
                    continue
                for v in V:
                    for t in T:
                        mdl.addConstr(
                            ordv[(i, v, t)] - ordv[(j, v, t)] + len(N) * x[(i, j, v, t)]
                            <= len(N) - 1
                        )

        for i in N:
            for j in N:
                if i == j:
                    continue
                for p in P:
                    for v in V:
                        for t in T:
                            mdl.addConstr(y[(i, j, p, v, t)] <= q[(p, i, j, v, t)])

        if add_valid_16_20:
            for i in N:
                for v in V:
                    for t in T:
                        mdl.addConstr(x[(CW, i, v, t)] <= z[(i, v, t)])

            for i in N:
                for j in N:
                    if i == j:
                        continue
                    for v in V:
                        for t in T:
                            mdl.addConstr(x[(i, j, v, t)] <= z[(j, v, t)])

            for i in N:
                for v in V:
                    for t in T:
                        mdl.addConstr(z[(i, v, t)] <= z[(CW, v, t)])

            for idx_v in range(1, len(V)):
                v = V[idx_v]
                v_prev = V[idx_v - 1]
                for t in T:
                    mdl.addConstr(z[(CW, v, t)] <= z[(CW, v_prev, t)])

            for s in N:
                for p in P:
                    for t1 in T:
                        for t2 in T:
                            if t2 < t1:
                                continue
                            total_dem = sum(d.demand[(s, p, tau)] for tau in T if t1 <= tau <= t2)
                            if total_dem <= 1e-9:
                                continue
                            init_term = d.init_inventory_store[(s, p)] if t1 == first_t else I_s[(s, p, t1 - 1)]
                            lhs = (
                                gp.quicksum(z[(s, v, tau)] for v in V for tau in T if t1 <= tau <= t2)
                                + (1.0 / total_dem) * gp.quicksum(
                                    y[(j, s, p, v, tau)]
                                    for j in N if j != s
                                    for v in V
                                    for tau in T if t1 <= tau <= t2
                                )
                            )
                            rhs = (total_dem - init_term) / total_dem
                            mdl.addConstr(lhs >= rhs)

        mdl.optimize()

        return FullIRPTSolution(
            status=_grb_status_name(mdl.Status),
            objective=_safe_obj_value(mdl),
            direct_ship_q={(s, p, t): _safe_var_value(mdl, Qdir[(s, p, t)]) for s in N for p in P for t in T},
            inv_store={(s, p, t): _safe_var_value(mdl, I_s[(s, p, t)]) for s in N for p in P for t in T},
            inv_wh={(p, t): _safe_var_value(mdl, I_w[(p, t)]) for p in P for t in T},
            shortage={(s, p, t): _safe_var_value(mdl, B[(s, p, t)]) for s in N for p in P for t in T},
            x={(i, j, v, t): int(round(_safe_var_value(mdl, x[(i, j, v, t)]))) for i in N0 for j in N0 if i != j for v in V for t in T},
            u={(v, t): int(round(_safe_var_value(mdl, u[(v, t)]))) for v in V for t in T},
            z={(i, v, t): int(round(_safe_var_value(mdl, z[(i, v, t)]))) for i in N0 for v in V for t in T},
            q={(p, i, j, v, t): _safe_var_value(mdl, q[(p, i, j, v, t)]) for p in P for i in N0 for j in N0 if i != j for v in V for t in T},
            y={(i, j, p, v, t): _safe_var_value(mdl, y[(i, j, p, v, t)]) for i in N for j in N if i != j for p in P for v in V for t in T},
            deliv={(s, p, v, t): _safe_var_value(mdl, deliv[(s, p, v, t)]) for s in N for p in P for v in V for t in T},
            load={(i, v, t): _safe_var_value(mdl, load[(i, v, t)]) for i in N0 for v in V for t in T},
            efficiency_metrics=_model_efficiency_metrics(mdl),
        )


# ============================================================================
# COLUMN GENERATION FOR LATERAL TRANSSHIPMENT
# ============================================================================

def generate_random_lt_patterns(
    data: IRPData,
    baseline_solution,
    n_patterns_per_product_period: int = 5,
    max_pairs_in_pattern: int = 4,
    max_qty_per_pair: int = 10,
    lt_activation_threshold: float = 0.0,
    seed: int = 123,
) -> List[LTPattern]:
    """
    Feasible random warm-start patterns. Only generated for active (product, period)
    pairs that pass the minimum LT activation threshold.
    """
    cg = LateralTransshipmentCG(
        data=data,
        baseline_solution=baseline_solution,
        initial_patterns=None,
        lt_activation_threshold=lt_activation_threshold,
    )
    need, surplus = cg._build_need_and_surplus_proxies()
    active_pt = cg._compute_active_product_periods(need, surplus)

    rng = random.Random(seed)
    patterns = []
    for p, t in sorted(active_pt):
        donors = [s for s in data.stores if surplus[(s, p, t)] > 1e-9]
        receivers = [s for s in data.stores if need[(s, p, t)] > 1e-9]
        if not donors or not receivers:
            continue

        candidate_pairs = [(i, j) for i in donors for j in receivers if i != j]
        if not candidate_pairs:
            continue

        for idx in range(1, n_patterns_per_product_period + 1):
            rng.shuffle(candidate_pairs)
            chosen_pairs = candidate_pairs[:rng.randint(1, min(max_pairs_in_pattern, len(candidate_pairs)))]
            donor_left = {i: surplus[(i, p, t)] for i in donors}
            recv_left = {j: need[(j, p, t)] for j in receivers}

            flows: Dict[Tuple[Store, Store], float] = {}
            total_cost = 0.0
            for i, j in chosen_pairs:
                ub = min(donor_left[i], recv_left[j], float(max_qty_per_pair))
                if ub <= 1e-9:
                    continue
                qty = float(rng.uniform(1.0, ub))
                flows[(i, j)] = qty
                donor_left[i] -= qty
                recv_left[j] -= qty
                total_cost += qty * data.ship_cost_lt[(i, j, p)] + data.fixed_dispatch_lt[(i, j)]

            if flows:
                patterns.append(LTPattern(
                    pattern_id=f"LT_{p}_T{t}_{idx}",
                    period=t,
                    product=p,
                    pattern_flows=flows,
                    column_cost=round(total_cost, 6),
                    metadata={"source": "random_warm_start"},
                ))
    return patterns


def format_pattern_detail(pat):
    flow_text = ", ".join(
        [f"{i}->{j}:{qty:.2f}" for (i, j), qty in pat.pattern_flows.items()]
    )
    return (
        f"pattern_id={pat.pattern_id} | "
        f"product={pat.product} | period={pat.period} | "
        f"cost={pat.column_cost:.4f} | flows=[{flow_text}]"
    )


# ============================================================================
# FOLLOWER BEST-RESPONSE SOLVER
# ============================================================================

class FollowerBestResponseSolver:
    """Solves the store's optimal lateral transshipment for a given post-shock inventory state.

    For a specific (product, period) the solver decides store-to-store
    transfers to minimise:

        Σ_{i,j} [ship_cost[i,j,p] * q[i,j] + fixed[i,j] * 1{q[i,j]>0}]
        + Σ_j  shortage_cost[j,p] * max(0, need[j] - Σ_i q[i,j])
        + Σ_i  holding_cost[i,p] * max(0, surplus[i] - Σ_j q[i,j])

    subject to:
        Σ_j q[i,j] ≤ surplus[i]              (donor capacity)
        Σ_i q[i,j] ≤ need[j]                 (receiver capacity)
        q[i,j] ≥ min_lateral_qty  if arc used (minimum shipment MOQ)
        q[i,j] ≥ 0

    Used by the follower-aware LT pricing step to score CG candidate columns
    by total system cost impact before admitting them to the RMP.

    Implemented as an O(n²) greedy heuristic (default) — fast enough for
    scoring hundreds of candidate columns per CG iteration — or as an LP
    relaxation via Gurobi (drops fixed charges; used for final validation).

    The greedy prioritises arcs by *net unit benefit*:
        net_benefit = shortage_cost[j,p] - ship_cost[i,j,p]
    An arc is activated only when:
        net_benefit > 0  AND
        effective quantity ≥ min_lateral_qty  AND
        fixed-cost break-even is satisfied.
    """

    def __init__(
        self,
        data: "IRPData",
        min_lateral_qty: float = 5.0,
        use_exact_lp: bool = False,
    ) -> None:
        self.data = data
        self.min_lateral_qty = max(0.0, float(min_lateral_qty))
        self.use_exact_lp = use_exact_lp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_for_product_period(
        self,
        p: Any,
        t: Any,
        need: Dict[Any, float],
        surplus: Dict[Any, float],
    ) -> "FollowerBestResponseResult":
        """Solve follower best-response for one (product, period).

        `need` and `surplus` are keyed as (store, product, period).
        """
        d = self.data
        donors = [s for s in d.stores if surplus.get((s, p, t), 0.0) > 1e-9]
        receivers = [s for s in d.stores if need.get((s, p, t), 0.0) > 1e-9]
        if not donors or not receivers:
            _sp = sum(d.shortage_cost.get((s, p), 0.0) * max(0.0, need.get((s, p, t), 0.0)) for s in d.stores)
            _hc = sum(d.holding_cost_store.get((s, p), 0.0) * max(0.0, float(surplus.get((s, p, t), 0.0))) for s in d.stores)
            return FollowerBestResponseResult(
                product=p, period=t, flows={},
                lt_cost=0.0, shortage_reduction=0.0,
                remaining_need=sum(max(0.0, need.get((s, p, t), 0.0)) for s in d.stores),
                remaining_surplus=sum(max(0.0, float(surplus.get((s, p, t), 0.0))) for s in d.stores),
                shortage_penalty=_sp,
                holding_cost=_hc,
                total_cost=_sp + _hc,
                n_arcs_used=0,
                solver_used="greedy_empty",
            )

        # Build arc table: (donor, receiver, avail_surplus, avail_need,
        #                    ship_cost, fixed_cost, shortage_benefit, net_unit_benefit)
        arcs = []
        for i in donors:
            s_i = float(surplus.get((i, p, t), 0.0))
            if s_i <= 1e-9:
                continue
            for j in receivers:
                if i == j:
                    continue
                n_j = float(need.get((j, p, t), 0.0))
                if n_j <= 1e-9:
                    continue
                ship = float(d.ship_cost_lt.get((i, j, p), 0.0))
                fixed = float(d.fixed_dispatch_lt.get((i, j), 0.0))
                spen = float(d.shortage_cost.get((j, p), 0.0))
                net_unit = spen - ship  # benefit per unit
                arcs.append((i, j, s_i, n_j, ship, fixed, spen, net_unit))

        if not arcs:
            return self._empty_result(p, t, need, surplus)

        if self.use_exact_lp:
            return self._solve_lp(p, t, need, surplus, arcs)
        return self._solve_greedy(p, t, need, surplus, arcs)

    def solve_all_active(
        self,
        need: Dict[Any, float],
        surplus: Dict[Any, float],
        active_product_periods: Optional[Set[Any]] = None,
    ) -> Dict[Tuple[Any, Any], "FollowerBestResponseResult"]:
        """Solve follower best-response for every active (product, period).

        Returns a dict keyed (product, period) → FollowerBestResponseResult.
        """
        results: Dict[Tuple[Any, Any], FollowerBestResponseResult] = {}
        d = self.data
        pts = active_product_periods or {(p, t) for p in d.products for t in d.periods}
        for p, t in pts:
            results[(p, t)] = self.solve_for_product_period(p, t, need, surplus)
        return results

    # ------------------------------------------------------------------
    # Internal solvers
    # ------------------------------------------------------------------

    def _solve_greedy(
        self,
        p: Any,
        t: Any,
        need: Dict[Any, float],
        surplus: Dict[Any, float],
        arcs: List[tuple],
    ) -> "FollowerBestResponseResult":
        """Greedy arc-by-arc assignment sorted by net unit benefit."""
        d = self.data
        min_qty = self.min_lateral_qty
        # Mutable residuals
        rem_need = {s: max(0.0, float(need.get((s, p, t), 0.0))) for s in d.stores}
        rem_surplus = {s: max(0.0, float(surplus.get((s, p, t), 0.0))) for s in d.stores}

        # Sort arcs: prefer high net_unit_benefit; break ties by descending qty_cap
        arcs_sorted = sorted(
            arcs,
            key=lambda a: (a[7], min(a[2], a[3])),  # net_unit, then qty cap
            reverse=True,
        )

        flows: Dict[Tuple[Any, Any], float] = {}
        lt_cost = 0.0
        shortage_reduction = 0.0

        for i, j, _, _, ship, fixed, spen, net_unit in arcs_sorted:
            avail_s = rem_surplus[i]
            avail_n = rem_need[j]
            qty = min(avail_s, avail_n)
            if qty < min_qty:
                continue
            # Check fixed-cost break-even: fixed < net_unit * qty
            if fixed > 0 and net_unit * qty <= fixed:
                continue
            # Only ship if economically beneficial (saves more shortage cost than it costs)
            if net_unit <= 0:
                continue
            flows[(i, j)] = qty
            lt_cost += ship * qty + fixed
            shortage_reduction += qty
            rem_surplus[i] = max(0.0, avail_s - qty)
            rem_need[j] = max(0.0, avail_n - qty)

        remaining_need = sum(max(0.0, rem_need[s]) for s in d.stores)
        remaining_surplus = sum(max(0.0, rem_surplus[s]) for s in d.stores)
        shortage_penalty = sum(
            float(d.shortage_cost.get((s, p), 0.0)) * max(0.0, rem_need[s])
            for s in d.stores
        )
        holding_cost = sum(
            float(d.holding_cost_store.get((s, p), 0.0)) * max(0.0, rem_surplus[s])
            for s in d.stores
        )
        return FollowerBestResponseResult(
            product=p, period=t,
            flows=flows,
            lt_cost=lt_cost,
            shortage_reduction=shortage_reduction,
            remaining_need=remaining_need,
            remaining_surplus=remaining_surplus,
            shortage_penalty=shortage_penalty,
            holding_cost=holding_cost,
            total_cost=lt_cost + shortage_penalty + holding_cost,
            n_arcs_used=len(flows),
            solver_used="greedy",
        )

    def _solve_lp(
        self,
        p: Any,
        t: Any,
        need: Dict[Any, float],
        surplus: Dict[Any, float],
        arcs: List[tuple],
    ) -> "FollowerBestResponseResult":
        """LP relaxation via Gurobi (fixed charges dropped, used for final validation)."""
        d = self.data
        pairs = [(a[0], a[1]) for a in arcs]
        arc_map = {(a[0], a[1]): a for a in arcs}

        mdl = gp.Model("FollowerLP", env=get_gurobi_env())
        mdl.Params.OutputFlag = 0
        q = mdl.addVars(pairs, lb=0.0, name="q")
        resid = mdl.addVars(
            [s for s in d.stores if need.get((s, p, t), 0.0) > 1e-9],
            lb=0.0, name="resid",
        )

        # Donor capacity
        for i in {a[0] for a in arcs}:
            out = [q[i, j] for (ii, j) in pairs if ii == i]
            if out:
                mdl.addConstr(gp.quicksum(out) <= float(surplus.get((i, p, t), 0.0)))
        # Receiver coverage
        for j in {a[1] for a in arcs}:
            inn = [q[i, j] for (i, jj) in pairs if jj == j]
            n_j = float(need.get((j, p, t), 0.0))
            if inn:
                mdl.addConstr(gp.quicksum(inn) + resid[j] >= n_j)
                mdl.addConstr(gp.quicksum(inn) <= n_j)
            else:
                resid[j].LB = n_j
        # Objective: shipping cost + shortage penalty on residual need
        ship_obj = gp.quicksum(float(arc_map[k][4]) * q[k] for k in pairs)
        pen_obj = gp.quicksum(
            float(d.shortage_cost.get((j, p), 0.0)) * resid[j]
            for j in resid
        )
        mdl.setObjective(ship_obj + pen_obj, GRB.MINIMIZE)
        mdl.optimize()

        flows: Dict[Tuple[Any, Any], float] = {}
        lt_cost = 0.0
        shortage_reduction = 0.0
        if mdl.Status == GRB.OPTIMAL:
            for k in pairs:
                val = float(q[k].X)
                if val > 1e-9:
                    flows[k] = val
                    lt_cost += float(arc_map[k][4]) * val
                    shortage_reduction += val
        remaining_need = sum(max(0.0, float(resid[j].X)) for j in resid) if mdl.Status == GRB.OPTIMAL else sum(
            max(0.0, need.get((s, p, t), 0.0)) for s in d.stores
        )
        remaining_surplus = sum(
            max(0.0, float(surplus.get((s, p, t), 0.0)) - sum(flows.get((s, jj), 0.0) for jj in d.stores))
            for s in d.stores
        )
        shortage_penalty = sum(
            float(d.shortage_cost.get((s, p), 0.0)) * max(0.0, need.get((s, p, t), 0.0) - sum(flows.get((ii, s), 0.0) for ii in d.stores))
            for s in d.stores
        )
        holding_cost = sum(
            float(d.holding_cost_store.get((s, p), 0.0)) * max(0.0, float(surplus.get((s, p, t), 0.0)) - sum(flows.get((s, jj), 0.0) for jj in d.stores))
            for s in d.stores
        )
        return FollowerBestResponseResult(
            product=p, period=t, flows=flows,
            lt_cost=lt_cost, shortage_reduction=shortage_reduction,
            remaining_need=remaining_need, remaining_surplus=remaining_surplus,
            shortage_penalty=shortage_penalty, holding_cost=holding_cost,
            total_cost=lt_cost + shortage_penalty + holding_cost,
            n_arcs_used=len(flows), solver_used="lp",
        )

    def _empty_result(self, p, t, need, surplus) -> "FollowerBestResponseResult":
        d = self.data
        pen = sum(d.shortage_cost.get((s, p), 0.0) * max(0.0, need.get((s, p, t), 0.0)) for s in d.stores)
        hc = sum(d.holding_cost_store.get((s, p), 0.0) * max(0.0, float(surplus.get((s, p, t), 0.0))) for s in d.stores)
        return FollowerBestResponseResult(
            product=p, period=t, flows={},
            lt_cost=0.0, shortage_reduction=0.0,
            remaining_need=sum(max(0.0, need.get((s, p, t), 0.0)) for s in d.stores),
            remaining_surplus=sum(max(0.0, float(surplus.get((s, p, t), 0.0))) for s in d.stores),
            shortage_penalty=pen, holding_cost=hc, total_cost=pen + hc,
            n_arcs_used=0, solver_used="greedy_empty",
        )


def apply_hidden_local_reallocation_demand_shocks(
    data: IRPData,
    baseline_solution: Optional[FullIRPTSolution] = None,
    shock_probability: float = 0.5,
    max_reallocation_fraction: float = 0.35,
    reallocations_per_product_period: int = 3,
    non_dispatch_shock_multiplier: float = 1.8,
    cw_dispatch_cycle: Optional[int] = 5,
    seed: int = 20260418,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    shock_probability = min(1.0, max(0.0, float(shock_probability)))
    max_reallocation_fraction = min(1.0, max(0.0, float(max_reallocation_fraction)))
    reallocations_per_product_period = max(1, int(reallocations_per_product_period))
    non_dispatch_shock_multiplier = max(1.0, float(non_dispatch_shock_multiplier))
    data.realized_demand = {key: float(value) for key, value in data.demand.items()}

    n_shocked = 0
    shocked_product_periods: Set[Tuple[Product, Period]] = set()
    total_reallocated = 0.0
    first_t = min(data.periods) if data.periods else 0
    cycle = int(cw_dispatch_cycle) if cw_dispatch_cycle is not None else 0
    for p in data.products:
        for t in data.periods:
            is_dispatch_period = True
            if cycle > 1:
                is_dispatch_period = ((t - first_t) % cycle == 0)
            period_multiplier = 1.0 if is_dispatch_period else non_dispatch_shock_multiplier
            effective_probability = min(1.0, shock_probability * period_multiplier)
            if rng.random() > effective_probability:
                continue
            candidates = [
                s for s in data.stores
                if float(data.demand.get((s, p, t), 0.0)) > 1e-9
            ]
            if len(candidates) < 2:
                continue

            for _ in range(reallocations_per_product_period):
                def receiver_score(store: Store) -> float:
                    forecast = float(data.demand.get((store, p, t), 0.0))
                    if baseline_solution is None:
                        return forecast
                    ending_inv = float(baseline_solution.inv_store.get((store, p, t), 0.0))
                    shortage = float(baseline_solution.shortage.get((store, p, t), 0.0))
                    fragility = shortage + max(0.0, forecast - ending_inv)
                    return fragility * period_multiplier + 0.01 * forecast

                receiver = max(candidates, key=receiver_score)
                donor_candidates = [s for s in candidates if s != receiver]
                if not donor_candidates:
                    continue

                def donor_score(store: Store) -> float:
                    realized = float(data.realized_demand.get((store, p, t), 0.0))
                    if baseline_solution is None:
                        return realized
                    ending_inv = float(baseline_solution.inv_store.get((store, p, t), 0.0))
                    shortage = float(baseline_solution.shortage.get((store, p, t), 0.0))
                    return ending_inv - shortage + 0.01 * realized

                donor = max(donor_candidates, key=donor_score)
                receiver_forecast = float(data.demand.get((receiver, p, t), 0.0))
                donor_realized = float(data.realized_demand.get((donor, p, t), 0.0))
                delta = min(
                    period_multiplier * max_reallocation_fraction * max(receiver_forecast, 1.0),
                    max_reallocation_fraction * donor_realized,
                )
                if delta <= 1e-9:
                    continue

                data.realized_demand[(receiver, p, t)] = data.realized_demand.get((receiver, p, t), 0.0) + delta
                data.realized_demand[(donor, p, t)] = max(0.0, donor_realized - delta)
                n_shocked += 1
                shocked_product_periods.add((p, t))
                total_reallocated += delta

    return {
        "shock_model": "hidden_local_reallocation",
        "shock_probability": shock_probability,
        "max_reallocation_fraction": max_reallocation_fraction,
        "reallocations_per_product_period": reallocations_per_product_period,
        "non_dispatch_shock_multiplier": non_dispatch_shock_multiplier,
        "cw_dispatch_cycle": cw_dispatch_cycle,
        "shock_seed": seed,
        "n_store_sku_period_reallocations": n_shocked,
        "n_shocked_product_periods": len(shocked_product_periods),
        "total_reallocated_units": round(total_reallocated, 6),
    }


def build_post_shock_inventory_state(
    data: IRPData,
    baseline_solution: FullIRPTSolution,
) -> Dict[str, Any]:
    data.post_shock_inventory = {}
    data.post_shock_shortage = {}
    total_shortage = 0.0
    total_realized_demand = 0.0
    total_forecast_demand = 0.0

    for s in data.stores:
        for p in data.products:
            prev_inventory = float(data.init_inventory_store.get((s, p), 0.0))
            for t in sorted(data.periods):
                forecast = float(data.demand.get((s, p, t), 0.0))
                realized = float(data.realized_demand.get((s, p, t), forecast))
                shipment = float(baseline_solution.direct_ship_q.get((s, p, t), 0.0))
                available = prev_inventory + shipment
                shortage = max(0.0, realized - available)
                ending_inventory = max(0.0, available - realized)
                data.post_shock_shortage[(s, p, t)] = shortage
                data.post_shock_inventory[(s, p, t)] = ending_inventory
                prev_inventory = ending_inventory
                total_shortage += shortage
                total_realized_demand += realized
                total_forecast_demand += forecast

    return {
        "total_forecast_demand": round(total_forecast_demand, 6),
        "total_realized_demand": round(total_realized_demand, 6),
        "total_post_shock_shortage": round(total_shortage, 6),
        "total_post_shock_inventory": round(sum(data.post_shock_inventory.values()), 6),
    }


def build_post_shock_lt_diagnostics(
    data: IRPData,
    baseline_solution: FullIRPTSolution,
    lt_activation_threshold: float = 0.0,
) -> Dict[str, Any]:
    cg = LateralTransshipmentCG(
        data=data,
        baseline_solution=baseline_solution,
        initial_patterns=None,
        lt_activation_threshold=lt_activation_threshold,
    )
    need, surplus = cg._build_need_and_surplus_proxies()
    active_product_periods = cg._compute_active_product_periods(need, surplus)
    return {
        "n_active_product_periods_after_shock": len(active_product_periods),
        "total_need_after_shock": round(sum(need.values()), 6),
        "total_surplus_after_shock": round(sum(surplus.values()), 6),
        "max_need_after_shock": round(max(need.values()) if need else 0.0, 6),
        "max_surplus_after_shock": round(max(surplus.values()) if surplus else 0.0, 6),
    }


# ============================================================================
# ADAPTIVE FEATURE PRUNER
#
# Inspired by Bianchessi, Gschwind & Irnich (2024), "Resource-Window Reduction
# by Reduced Costs in Path-Based Formulations for Routing and Scheduling
# Problems" (INFORMS J. Comp.). The paper tightens vertex/arc resource windows
# using the reduced costs of paths visiting them: a value v is eliminated
# whenever LB(π) + min_{p has v} c̃_p(π) > UB. We adapt the same idea to the
# four bilateral-pair features used in our LT pricing step:
#   {shortage_ratio, surplus_ratio, time_urgency, negative_reduced_cost}.
#
# Each iteration we:
#   1) Score every (donor, receiver) pair with a reduced-cost proxy.
#   2) Pick "surviving" pairs by the bound rule when LB/UB are available, else
#      fall back to a quantile of the current iteration's RC distribution.
#   3) Set [L_k, U_k] for each feature k = [min, max] over surviving pairs
#      (with a small epsilon buffer to avoid over-tightening).
#   4) In subsequent iterations, prune candidates whose feature values fall
#      outside [L_k, U_k]. Safeguards keep the best-RC candidate, refuse
#      wipeouts, and warm-start without pruning until enough data is seen.
# ============================================================================

class AdaptiveFeaturePruner:
    """Adaptive admissible windows for the four LT pricing features.

    Maintains windows [L_k, U_k] per feature and per phase. Windows are
    tightened from the reduced-cost distribution of each pricing iteration's
    surviving candidates — never from a fixed numeric threshold.
    """

    DEFAULT_FEATURES: Tuple[str, ...] = (
        "shortage_ratio",
        "surplus_ratio",
        "time_urgency",
        "negative_reduced_cost",
    )

    def __init__(
        self,
        feature_names: Optional[Tuple[str, ...]] = None,
        epsilon: float = 0.05,
        warmup_min_candidates: int = 8,
        rc_fallback_quantile: float = 0.5,
        max_pruned_fraction: float = 0.95,
        keep_best_rc: bool = True,
    ):
        self.feature_names: Tuple[str, ...] = tuple(feature_names or self.DEFAULT_FEATURES)
        self.epsilon = float(epsilon)
        self.warmup_min_candidates = int(warmup_min_candidates)
        self.rc_fallback_quantile = float(rc_fallback_quantile)
        self.max_pruned_fraction = float(max_pruned_fraction)
        self.keep_best_rc = bool(keep_best_rc)
        # Per-phase windows; default phase "lt" is created lazily.
        self._windows: Dict[str, Dict[str, Tuple[float, float]]] = {}
        # Audit log of every iteration the pruner saw.
        self.history: List[Dict[str, Any]] = []

    # ---- Window access -----------------------------------------------------
    def current_windows(self, phase: str = "lt") -> Dict[str, Tuple[float, float]]:
        """Return current admissible windows; full open ranges if no update yet."""
        return self._windows.get(phase, {
            name: (-math.inf, math.inf) for name in self.feature_names
        })

    def is_admissible(
        self,
        feature_values: Dict[str, float],
        phase: str = "lt",
    ) -> bool:
        windows = self.current_windows(phase)
        for name in self.feature_names:
            lo, hi = windows.get(name, (-math.inf, math.inf))
            v = float(feature_values.get(name, 0.0))
            if v < lo or v > hi:
                return False
        return True

    # ---- Update -----------------------------------------------------------
    def update(
        self,
        candidates: List[Dict[str, Any]],
        *,
        iteration: int = 0,
        phase: str = "lt",
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Recompute admissible windows from the surviving candidates.

        candidates: list of dicts, each with keys 'reduced_cost' and 'feature_values'.
        Survivors are picked by LB+rc<=UB when both bounds are available, else by
        the bottom rc_fallback_quantile of the iteration's RC distribution.
        """
        n_total = len(candidates)
        record: Dict[str, Any] = {
            "iteration": int(iteration),
            "phase": str(phase),
            "n_candidates": n_total,
            "lb_used": None if lb is None else float(lb),
            "ub_used": None if ub is None else float(ub),
            "mode": "warmup",
            "n_survivors": 0,
            "windows": {},
            "skipped_reason": None,
        }

        if n_total < self.warmup_min_candidates:
            record["skipped_reason"] = "warmup_insufficient_candidates"
            self.history.append(record)
            return record

        rcs = [float(c.get("reduced_cost", 0.0)) for c in candidates]

        # Bound-based rule when LB and UB are both finite and consistent.
        survivors: List[Dict[str, Any]] = []
        if lb is not None and ub is not None and math.isfinite(lb) and math.isfinite(ub) and ub >= lb:
            gap = ub - lb
            survivors = [c for c, rc in zip(candidates, rcs) if rc <= gap]
            record["mode"] = "bound_based"
        if not survivors:
            # Fallback: use the lower quantile of the RC distribution.
            sorted_rcs = sorted(rcs)
            cutoff_idx = max(0, int(math.ceil(self.rc_fallback_quantile * n_total)) - 1)
            cutoff_idx = min(cutoff_idx, n_total - 1)
            cutoff_rc = sorted_rcs[cutoff_idx]
            survivors = [c for c, rc in zip(candidates, rcs) if rc <= cutoff_rc]
            record["mode"] = "rc_quantile_fallback"
            record["fallback_quantile"] = self.rc_fallback_quantile
            record["fallback_cutoff_rc"] = float(cutoff_rc)

        # Always keep best-RC candidate so windows include at least one improving column.
        if self.keep_best_rc and candidates:
            best = min(candidates, key=lambda c: float(c.get("reduced_cost", 0.0)))
            if best not in survivors:
                survivors.append(best)

        if not survivors:
            record["skipped_reason"] = "no_survivors_after_filter"
            self.history.append(record)
            return record

        # Compute new windows: [min - eps, max + eps] over surviving features.
        proposed: Dict[str, Tuple[float, float]] = {}
        for name in self.feature_names:
            vals = [float(c["feature_values"].get(name, 0.0)) for c in survivors]
            lo = min(vals)
            hi = max(vals)
            span = max(hi - lo, 1e-9)
            proposed[name] = (lo - self.epsilon * span, hi + self.epsilon * span)

        # Wipe-out check: how many of the *current* candidates would the new
        # windows reject? If too aggressive, refuse this update.
        n_kept = sum(
            1 for c in candidates
            if all(
                proposed[name][0] <= float(c["feature_values"].get(name, 0.0)) <= proposed[name][1]
                for name in self.feature_names
            )
        )
        pruned_fraction = 1.0 - (n_kept / float(n_total))
        if pruned_fraction > self.max_pruned_fraction:
            record["skipped_reason"] = (
                f"would_prune_fraction={pruned_fraction:.3f}>cap={self.max_pruned_fraction}"
            )
            self.history.append(record)
            return record

        self._windows[phase] = proposed
        record["n_survivors"] = len(survivors)
        record["windows"] = {
            name: (round(lo, 6), round(hi, 6)) for name, (lo, hi) in proposed.items()
        }
        record["n_kept_after_window"] = int(n_kept)
        record["pruned_fraction"] = round(pruned_fraction, 6)
        self.history.append(record)
        return record


class LateralTransshipmentCG:
    def __init__(
        self,
        data: IRPData,
        baseline_solution,
        initial_patterns: Optional[List[LTPattern]] = None,
        lt_activation_threshold: float = 0.0,
        safety_stock_units: float = 0.0,
        need_lookahead_periods: int = 2,
        surplus_reserve_periods: int = 1,
        rebalance_need_penalty: float = 1.0,
        max_pairs_per_pattern: int = 4,
        max_columns_per_product_period: int = 3,
        top_pairs_per_feature: int = 20,
        top_patterns_per_feature: int = 5,
        feature_ranges: Optional[Dict[str, Dict[str, float]]] = None,
        stackelberg_params: Optional[StackelbergParams] = None,
        use_gnn: bool = False,
        collect_teacher_mode: bool = False,
        runtime_gnn_mode: Optional[bool] = None,
        gnn_checkpoint: Optional[str] = None,
        use_classical_fallback: bool = True,
        gnn_selection_mode: str = "cumulative_mass",
        gnn_mass_threshold: float = 0.55,
        gnn_relative_threshold: float = 0.85,
        gnn_min_keep: int = 1,
        gnn_max_keep: Optional[int] = 150,
        gnn_max_keep_fraction: float = 0.30,
        gnn_root: str = "GNN",
        branch_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        diagnostic_verbosity: str = "summary",
        heuristic_top_k_mode: bool = False,
        heuristic_top_k: int = 20,
        exact_full_mode: bool = False,
        exact_pricing_pool_size: int = 3,
        exact_pricing_time_limit: Optional[int] = 30,
        # ── Follower-aware LT column pricing ──────────────────────────────
        # When stackelberg_aware_scoring=True the pricing step scores every
        # candidate pattern by total system cost after follower best-response
        # (LT cost + shortage penalty + holding cost on remaining surplus).
        # A column is admitted only when its combined cost delta is strictly
        # negative. Returning no column signals follower-aware convergence.
        stackelberg_aware_scoring: bool = False,
        # Use exact LP for the follower's subproblem (Gurobi, slower but
        # tighter). Default: greedy heuristic (fast, no extra Gurobi calls).
        stackelberg_exact_follower: bool = False,
        # Minimum LT shipment size enforced inside the follower solver.
        # None → read IRP_LT_MIN_UNITS (default 5).
        stackelberg_min_lateral_qty: Optional[float] = None,
        # ── Adaptive feature pruning ──────────────────────────────────────
        # When True, replace the static `feature_ranges` admissibility test
        # with iteration-by-iteration adaptive windows derived from the
        # surviving candidates' reduced-cost distribution. Inspired by
        # Bianchessi et al. (2024) resource-window reduction. Default reads
        # IRP_ADAPTIVE_PRUNING (default "1" = enabled).
        adaptive_pruning_enabled: Optional[bool] = None,
        source_instance: Optional[str] = None,
    ):
        self.data = data
        self.baseline = baseline_solution
        self.patterns = initial_patterns[:] if initial_patterns else []
        self.lt_activation_threshold = max(0.0, float(lt_activation_threshold))
        self.safety_stock_units = float(safety_stock_units)
        self.need_lookahead_periods = max(1, int(need_lookahead_periods))
        self.surplus_reserve_periods = max(1, int(surplus_reserve_periods))
        self.rebalance_need_penalty = float(rebalance_need_penalty)
        self.max_pairs_per_pattern = int(max_pairs_per_pattern)
        self.max_columns_per_product_period = int(max_columns_per_product_period)
        self.top_pairs_per_feature = int(top_pairs_per_feature)
        self.top_patterns_per_feature = int(top_patterns_per_feature)
        # Strict benchmark mode — disable every search-space-restricting heuristic
        # so the classical CG baseline sees the full candidate pool. All of these
        # are speed heuristics only; relaxing them changes the effective search
        # space, not solver correctness, so benchmark runs that need apples-to-
        # apples comparisons (Run A: classical vs Run C: GNN) must opt in.
        # Activate with env IRP_STRICT_BENCHMARK=1 or by passing strict_benchmark=True.
        _strict_env = os.environ.get("IRP_STRICT_BENCHMARK", "0").strip().lower() not in {"0", "false", "no", ""}
        self.strict_benchmark_mode = bool(_strict_env)
        if self.strict_benchmark_mode:
            _UNCAPPED = 10**9
            print("[StrictBenchmark] IRP_STRICT_BENCHMARK=1 — relaxing all pruning caps:")
            print(f"  max_columns_per_product_period  : {self.max_columns_per_product_period} -> {_UNCAPPED}")
            print(f"  top_pairs_per_feature           : {self.top_pairs_per_feature} -> {_UNCAPPED}")
            print(f"  top_patterns_per_feature        : {self.top_patterns_per_feature} -> {_UNCAPPED}")
            self.max_columns_per_product_period = _UNCAPPED
            self.top_pairs_per_feature = _UNCAPPED
            self.top_patterns_per_feature = _UNCAPPED
        self.feature_ranges = feature_ranges or {
            "shortage_ratio": {"min": 0.00, "max": 1.00},
            "surplus_ratio": {"min": 0.00, "max": 1.00},
            "time_urgency": {"min": 0.00, "max": 1.00},
            "negative_reduced_cost": {"min": 0.00, "max": math.inf},
        }
        self.stackelberg_params = stackelberg_params or StackelbergParams()
        self.collect_teacher_mode = bool(collect_teacher_mode)
        requested_runtime_gnn = bool(use_gnn if runtime_gnn_mode is None else runtime_gnn_mode)
        self.runtime_gnn_mode = False if self.collect_teacher_mode else requested_runtime_gnn
        self.use_gnn = self.runtime_gnn_mode
        self.gnn_checkpoint = gnn_checkpoint or DEFAULT_GNN_CHECKPOINT
        self.gnn_selection_mode = str(gnn_selection_mode)
        self.gnn_mass_threshold = float(gnn_mass_threshold)
        self.gnn_relative_threshold = float(gnn_relative_threshold)
        self.gnn_min_keep = int(gnn_min_keep)
        self.gnn_max_keep = int(gnn_max_keep) if gnn_max_keep is not None else None
        self.gnn_max_keep_fraction = float(gnn_max_keep_fraction)
        if getattr(self, "strict_benchmark_mode", False):
            print(f"  gnn_max_keep                    : {self.gnn_max_keep} -> None")
            print(f"  gnn_max_keep_fraction           : {self.gnn_max_keep_fraction} -> 1.0")
            self.gnn_max_keep = None
            self.gnn_max_keep_fraction = 1.0
        self.gnn_prob_weight = 0.60
        self.gnn_rc_gain_weight = 0.30
        self.gnn_acceptance_weight = 0.10
        self.use_classical_fallback = bool(use_classical_fallback)
        self.gnn_root = gnn_root
        # Run-B (benchmark) — rank candidate patterns by |reduced_cost| and keep top-k
        # before they enter RMP. Active only when use_gnn=False and collect_teacher_mode=False.
        self.heuristic_top_k_mode = bool(heuristic_top_k_mode)
        self.heuristic_top_k = max(1, int(heuristic_top_k))
        # Run-A0 (benchmark) — exact CG pricing via Gurobi MIP per active
        # (product, period), with no feature pruning, Stackelberg game, GNN, or
        # top-k ranking anywhere in the path. Pool search returns up to
        # exact_pricing_pool_size distinct negative-RC columns per subproblem
        # per iteration (PoolSearchMode=2), matching the multi-column generation
        # rate of classical CG for a fair runtime comparison.
        self.exact_full_mode = bool(exact_full_mode)
        self.exact_pricing_pool_size = max(1, int(exact_pricing_pool_size))
        self.exact_pricing_time_limit = (
            int(exact_pricing_time_limit) if exact_pricing_time_limit is not None else None
        )
        if self.exact_full_mode:
            # Exact mode overrides all heuristic filters unconditionally.
            self.collect_teacher_mode = False
            self.runtime_gnn_mode = False
            self.use_gnn = False
            self.heuristic_top_k_mode = False
        # Follower-aware system-level column scoring (enabled when stackelberg_aware_scoring=True).
        self.stackelberg_aware_scoring = bool(stackelberg_aware_scoring)
        self.stackelberg_exact_follower = bool(stackelberg_exact_follower)
        self._stackelberg_min_lateral_qty_override: Optional[float] = (
            float(stackelberg_min_lateral_qty) if stackelberg_min_lateral_qty is not None else None
        )
        # Per-iteration Stackelberg evaluation log (pattern_id, delta, ...)
        self.stackelberg_column_score_log: List[Dict[str, Any]] = []
        # ── Adaptive feature pruning (Bianchessi-inspired) ────────────────
        if adaptive_pruning_enabled is None:
            adaptive_pruning_enabled = os.environ.get(
                "IRP_ADAPTIVE_PRUNING", "1"
            ).lower() not in {"0", "false", "no", ""}
        self.adaptive_pruning_enabled = bool(adaptive_pruning_enabled)
        self.adaptive_pruner: Optional[AdaptiveFeaturePruner] = (
            AdaptiveFeaturePruner() if self.adaptive_pruning_enabled else None
        )
        # Best LT-cost UB seen across pricing iterations (initialized lazily).
        self._adaptive_pruning_lb: Optional[float] = None
        self._adaptive_pruning_ub: Optional[float] = None
        # Buffer of candidate dicts collected within a single CG iteration —
        # flushed to the pruner once per iteration, not once per (p, t) call.
        self._adaptive_pending_candidates: List[Dict[str, Any]] = []
        self._gnn_loaded = False
        self._gnn_unavailable_reason: Optional[str] = None
        self._gnn_model = None
        self._gnn_checkpoint_payload = None
        self._gnn_build_graph = None
        self._gnn_graph_to_tensors = None
        self._gnn_normalize_dataset = None
        self._gnn_adaptive_select_indices = None
        self.gnn_selection_history: List[Dict[str, Any]] = []
        self.teacher_dataset_rows: List[Dict[str, Any]] = []
        # Per-engine teacher export diagnostics. The graph builder reads these
        # via results["teacher_export_diagnostics"] so the GNN training pipeline
        # can fail loudly if too many batches were dropped for missing constraint
        # features (silent skips were the failure mode that motivated this).
        self.teacher_export_diagnostics: Dict[str, int] = {
            "batches_total": 0,
            "batches_with_constraint_features": 0,
            "batches_skipped_no_constraint_features": 0,
            "batches_skipped_no_column_features": 0,
            "batches_skipped_no_edges": 0,
            "rows_written": 0,
        }
        self.cg_history: List[Dict[str, Any]] = []
        self.cg_history_all_nodes: List[Dict[str, Any]] = []
        self.branch_history: List[Dict[str, Any]] = []
        self.column_pool_diagnostics: List[Dict[str, Any]] = []
        self.cg_episode_diagnostics: List[Dict[str, Any]] = []
        self.branch_bounds: Dict[str, Tuple[float, float]] = dict(branch_bounds or {})
        self.current_episode = 0
        self.current_branch_node_id: Optional[int] = None
        self.last_duplicate_rejects = 0
        self.last_signature_rejects = 0
        self.last_empty_rejects = 0
        self.last_added_patterns = 0
        self._last_candidate_pair_count = 0
        self._last_pricing_summary: Dict[str, Any] = {}
        verbosity = str(diagnostic_verbosity or "summary").strip().lower()
        if verbosity not in {"summary", "full"}:
            verbosity = "summary"
        self.diagnostic_verbosity = verbosity
        self._log_candidate_pairs = verbosity == "full"
        # source_instance tags every exported teacher row so
        # build_teacher_graph_dataset.py can split at the instance level.
        # Default: env override → auto-timestamped unique tag. Never reuse
        # "irplt_cg" across runs; doing so collapses every run into one
        # source_instance and forces group-level split (leakage warning).
        if source_instance is None:
            source_instance = os.environ.get("IRP_SOURCE_INSTANCE") or (
                f"irplt_cg__{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_{os.getpid()}"
            )
        self.source_instance = str(source_instance)

    def _compute_combined_column_scores(self, patterns: List[LTPattern], probs: List[float]) -> List[Dict[str, float]]:
        rc_gains = []
        acceptance_scores = []
        for pat in patterns:
            try:
                rc = float(pat.metadata.get("reduced_cost", 0.0))
            except (TypeError, ValueError):
                rc = 0.0
            rc_gains.append(max(0.0, -rc))
            try:
                acceptance = float(pat.metadata.get("mean_acceptance_score", pat.metadata.get("acceptance_score", 0.0)))
            except (TypeError, ValueError):
                acceptance = 0.0
            acceptance_scores.append(min(max(acceptance, 0.0), 1.0))

        max_rc_gain = max(rc_gains) if rc_gains else 0.0
        rows = []
        for idx, pat in enumerate(patterns):
            rc_gain_norm = rc_gains[idx] / max(max_rc_gain, 1e-9) if max_rc_gain > 1e-12 else 0.0
            combined_score = (
                self.gnn_prob_weight * float(probs[idx])
                + self.gnn_rc_gain_weight * rc_gain_norm
                + self.gnn_acceptance_weight * acceptance_scores[idx]
            )
            rows.append({
                "idx": float(idx),
                "gnn_prob": float(probs[idx]),
                "rc_gain": float(rc_gains[idx]),
                "normalized_rc_gain": float(rc_gain_norm),
                "acceptance_score": float(acceptance_scores[idx]),
                "combined_score": float(max(0.0, combined_score)),
                "reduced_cost": float(pat.metadata.get("reduced_cost", 0.0) or 0.0),
            })
        return rows

    @staticmethod
    def _pattern_signature_key(pat: LTPattern) -> Tuple[Product, Period, Tuple[Tuple[Store, Store, float], ...]]:
        return (
            pat.product,
            pat.period,
            tuple(sorted((i, j, round(float(qty), 6)) for (i, j), qty in pat.pattern_flows.items())),
        )

    def _deduplicate_priced_patterns(self, patterns: List[LTPattern], episode: int) -> List[LTPattern]:
        if not patterns:
            self._last_pricing_summary["patterns_built_before_dedup"] = 0
            self._last_pricing_summary["patterns_deduplicated_before_gnn"] = 0
            return patterns

        best_by_signature: Dict[Tuple[Product, Period, Tuple[Tuple[Store, Store, float], ...]], LTPattern] = {}
        existing_signatures = {self._pattern_signature_key(pat) for pat in self.patterns}
        rejected_duplicates: List[LTPattern] = []
        for pat in patterns:
            signature = self._pattern_signature_key(pat)
            if not pat.pattern_flows or signature in existing_signatures:
                rejected_duplicates.append(pat)
                continue
            current = best_by_signature.get(signature)
            if current is None:
                best_by_signature[signature] = pat
                continue

            pat_rc = float(pat.metadata.get("reduced_cost", 0.0) or 0.0)
            current_rc = float(current.metadata.get("reduced_cost", 0.0) or 0.0)
            pat_acceptance = float(pat.metadata.get("mean_acceptance_score", 0.0) or 0.0)
            current_acceptance = float(current.metadata.get("mean_acceptance_score", 0.0) or 0.0)
            pat_is_better = (pat_rc, -pat_acceptance) < (current_rc, -current_acceptance)
            if pat_is_better:
                rejected_duplicates.append(current)
                best_by_signature[signature] = pat
            else:
                rejected_duplicates.append(pat)

        deduped = list(best_by_signature.values())
        for pat in rejected_duplicates:
            self._record_pattern_pool_diagnostic(
                pat=pat,
                stage="pricing_dedup_rejected_before_gnn",
                pattern_reduced_cost=pat.metadata.get("reduced_cost"),
                duplicate_signature_reject=True,
                added_to_pool=False,
                signature=str(self._pattern_signature_key(pat)),
            )

        self._last_pricing_summary["patterns_built_before_dedup"] = len(patterns)
        self._last_pricing_summary["patterns_deduplicated_before_gnn"] = len(patterns) - len(deduped)
        return deduped

    def _adaptive_select_columns(
        self,
        score_rows: List[Dict[str, float]],
        patterns: Optional[List[LTPattern]] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        if not score_rows:
            return [], {"selection_mode": self.gnn_selection_mode, "adaptive_k": 0}

        scores = [row["combined_score"] for row in score_rows]
        if sum(max(0.0, float(score)) for score in scores) <= 1e-12:
            scores = [row["gnn_prob"] for row in score_rows]
        selector = self._gnn_adaptive_select_indices
        if selector is None:
            raise RuntimeError("GNN adaptive selector helper is not loaded")
        max_keep = self.gnn_max_keep
        if self.gnn_max_keep_fraction > 0:
            fraction_cap = max(self.gnn_min_keep, int(math.ceil(len(score_rows) * self.gnn_max_keep_fraction)))
            max_keep = min(max_keep, fraction_cap) if max_keep is not None else fraction_cap
        selected_idx, info = selector(
            scores,
            tie_breaker=[row["gnn_prob"] for row in score_rows],
            selection_mode=self.gnn_selection_mode,
            mass_threshold=self.gnn_mass_threshold,
            relative_threshold=self.gnn_relative_threshold,
            min_keep=self.gnn_min_keep,
            max_keep=max_keep,
        )
        info["max_keep"] = max_keep
        info["max_keep_fraction"] = self.gnn_max_keep_fraction

        # ---- Fairness protection: per (product, period) minimum quota --------
        # Global top-k selection can starve product-period groups whose best
        # column scores are uniformly below the globally top-k-th column. That
        # hides valid pricing subproblems from the RMP. We enforce a minimum
        # of `gnn_min_per_group` columns per *active* group (a group with at
        # least one negative-reduced-cost candidate). When a group is starved,
        # force-add its highest-scoring column(s) on top of the global picks.
        # Configurable via env IRP_GNN_MIN_PER_GROUP (default 1). Set to 0 to
        # disable and recover the old behavior.
        if patterns is not None and len(patterns) == len(score_rows):
            min_per_group = int(os.environ.get("IRP_GNN_MIN_PER_GROUP", "1") or 0)
            group_keys: List[Tuple[Any, Any]] = [
                (getattr(pat, "product", None), getattr(pat, "period", None))
                for pat in patterns
            ]
            groups: Dict[Tuple[Any, Any], List[int]] = {}
            for idx, key in enumerate(group_keys):
                groups.setdefault(key, []).append(idx)
            # Only groups that actually contain a candidate with negative RC
            # need protection — empty groups have nothing to contribute.
            active_groups = {
                key: idxs
                for key, idxs in groups.items()
                if any(
                    float(patterns[i].metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
                    for i in idxs
                )
            }
            selected_set = set(selected_idx)
            group_forced: Dict[str, int] = {}
            if min_per_group > 0 and active_groups:
                for key, idxs in active_groups.items():
                    already = sum(1 for i in idxs if i in selected_set)
                    if already >= min_per_group:
                        continue
                    # Pick the best-scoring indices for this group that are not
                    # already selected. Ties broken by gnn_prob, then reduced_cost.
                    group_ranked = sorted(
                        idxs,
                        key=lambda i: (
                            -float(score_rows[i].get("combined_score", 0.0) or 0.0),
                            -float(score_rows[i].get("gnn_prob", 0.0) or 0.0),
                            float(patterns[i].metadata.get("reduced_cost", 0.0) or 0.0),
                        ),
                    )
                    need = min_per_group - already
                    added_for_group = 0
                    for i in group_ranked:
                        if i in selected_set:
                            continue
                        selected_idx.append(i)
                        selected_set.add(i)
                        added_for_group += 1
                        if added_for_group >= need:
                            break
                    if added_for_group > 0:
                        group_forced[f"{key[0]}__t{key[1]}"] = added_for_group

            # Per-group diagnostics — written into `info` so the caller can log
            # them in gnn_selection_history and the benchmark CSV.
            group_diag = []
            for key, idxs in groups.items():
                n_candidates = len(idxs)
                n_selected = sum(1 for i in idxs if i in selected_set)
                n_negative_rc = sum(
                    1 for i in idxs
                    if float(patterns[i].metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
                )
                group_diag.append({
                    "product": key[0],
                    "period": key[1],
                    "n_candidates": n_candidates,
                    "n_negative_rc_candidates": n_negative_rc,
                    "n_selected_after_gnn": n_selected,
                    "selection_rate": (n_selected / n_candidates) if n_candidates else 0.0,
                    "forced_by_quota": int(group_forced.get(f"{key[0]}__t{key[1]}", 0)),
                })
            info["per_group_diagnostics"] = group_diag
            info["min_per_group"] = min_per_group
            info["groups_forced_count"] = sum(group_forced.values())
        return selected_idx, info

    def _apply_classical_fallback(self, patterns: List[LTPattern], selected_idx: List[int]) -> Tuple[List[int], Optional[int]]:
        if not self.use_classical_fallback or not patterns:
            return selected_idx, None

        has_negative_selected = any(
            float(patterns[idx].metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
            for idx in selected_idx
        )
        if selected_idx and has_negative_selected:
            return selected_idx, None

        feasible = [
            (idx, float(pat.metadata.get("reduced_cost", 0.0) or 0.0))
            for idx, pat in enumerate(patterns)
            if float(pat.metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
        ]
        if not feasible:
            return selected_idx, None

        best_idx = min(feasible, key=lambda item: item[1])[0]
        if best_idx not in selected_idx:
            selected_idx = [best_idx] + selected_idx
        return selected_idx, best_idx

    def _record_teacher_rows(
        self,
        *,
        episode: int,
        patterns: List[LTPattern],
        score_rows: List[Dict[str, float]],
        raw_graph: Optional[Dict[str, Any]],
        gnn_selected_idx: List[int],
        final_selected_idx: List[int],
        adaptive_info: Dict[str, Any],
        fallback_idx: Optional[int],
    ) -> None:
        gnn_selected_set = set(gnn_selected_idx)
        final_selected_set = set(final_selected_idx)
        fallback_set = {fallback_idx} if fallback_idx is not None else set()
        score_by_idx = {int(row["idx"]): row for row in score_rows}
        column_features = raw_graph.get("column_features") if raw_graph else None
        constraint_features = raw_graph.get("constraint_features") if raw_graph else None
        edge_index = raw_graph.get("edge_index_col_to_con") if raw_graph else None
        edge_attr = raw_graph.get("edge_attr_col_to_con") if raw_graph else None

        # ------------------------------------------------------------------
        # Validate the batch BEFORE writing any rows.  Earlier code wrote rows
        # with empty constraint_features_json whenever raw_graph was missing or
        # the graph builder returned a degenerate graph; the graph builder then
        # silently dropped those groups via `propagate_constraint_features_json`
        # finding no non-empty value to fill in.  We now refuse to write a batch
        # that lacks constraint features (or column features, or edges) and
        # increment the corresponding skip counter so downstream tooling can
        # detect when teacher coverage degraded.
        # ------------------------------------------------------------------
        self.teacher_export_diagnostics["batches_total"] += 1
        if constraint_features is None or len(constraint_features) == 0:
            self.teacher_export_diagnostics["batches_skipped_no_constraint_features"] += 1
            return
        if column_features is None or len(column_features) == 0:
            self.teacher_export_diagnostics["batches_skipped_no_column_features"] += 1
            return
        if edge_index is None or edge_attr is None or len(edge_attr) == 0:
            self.teacher_export_diagnostics["batches_skipped_no_edges"] += 1
            return
        self.teacher_export_diagnostics["batches_with_constraint_features"] += 1

        edge_constraint_ids_by_col: Dict[int, List[int]] = {idx: [] for idx in range(len(patterns))}
        edge_attrs_by_col: Dict[int, List[List[float]]] = {idx: [] for idx in range(len(patterns))}
        col_indices = edge_index[0].tolist()
        con_indices = edge_index[1].tolist()
        edge_attrs = edge_attr.tolist()
        for edge_pos, col_idx in enumerate(col_indices):
            col_idx = int(col_idx)
            edge_constraint_ids_by_col.setdefault(col_idx, []).append(int(con_indices[edge_pos]))
            edge_attrs_by_col.setdefault(col_idx, []).append([float(value) for value in edge_attrs[edge_pos]])
        constraint_features_json = json.dumps(constraint_features.tolist())
        # `constraint_features_json` is identical for every pattern exported by this
        # batch (same branch node / episode / product / period / RMP state), but it
        # can be ~20 KB per row. Write it only on the first row of the batch and leave
        # subsequent rows empty — build_teacher_graph_dataset.py fills missing values
        # from the first non-empty row in each group. This shrinks the teacher CSV by
        # roughly the number of patterns per batch (55 MB → a few MB on realistic runs).
        constraint_state_hash = (
            hashlib.sha1(constraint_features_json.encode("utf-8")).hexdigest()[:12]
            if constraint_features_json
            else "no_constraints"
        )
        full_negative_rc_count = sum(
            1 for pat in patterns
            if float(pat.metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
        )
        for idx, pat in enumerate(patterns):
            row = score_by_idx.get(idx, {})
            per_row_constraint_json = constraint_features_json if idx == 0 else ""
            self.teacher_dataset_rows.append({
                "source_instance": self.source_instance,
                "branch_node_id": self.current_branch_node_id,
                "episode": episode,
                "column_index": idx,
                "pattern_id": pat.pattern_id,
                "product": pat.product,
                "period": pat.period,
                "pattern_flows_json": json.dumps([
                    {"donor": donor, "receiver": receiver, "quantity": float(qty)}
                    for (donor, receiver), qty in pat.pattern_flows.items()
                ]),
                "column_features_json": json.dumps(column_features[idx].tolist()) if column_features is not None else "",
                "constraint_features_json": per_row_constraint_json,
                "constraint_state_hash": constraint_state_hash,
                "edge_constraint_indices_json": json.dumps(edge_constraint_ids_by_col.get(idx, [])),
                "edge_attrs_json": json.dumps(edge_attrs_by_col.get(idx, [])),
                "gnn_prob": row.get("gnn_prob"),
                "combined_score": row.get("combined_score"),
                "normalized_rc_gain": row.get("normalized_rc_gain"),
                "reduced_cost": pat.metadata.get("reduced_cost"),
                "mean_acceptance_score": pat.metadata.get("mean_acceptance_score"),
                "selected_by_gnn": idx in gnn_selected_set,
                "selected_by_classical_fallback": idx in fallback_set,
                "passed_to_rmp": idx in final_selected_set,
                "adaptive_k_star": adaptive_info.get("adaptive_k"),
                "selection_mode": adaptive_info.get("selection_mode"),
                "full_negative_reduced_cost_columns": full_negative_rc_count,
                "classical_fallback_enabled": self.use_classical_fallback,
                "classical_fallback_used": fallback_idx is not None,
                "label_source": "teacher_rmp",
                "selected_in_rmp": False,
                "lambda_value": 0.0,
                "teacher_label": 0,
                "teacher_score": 0.0,
                "cg_objective_after_reopt": None,
                "cg_objective_improvement": None,
            })

    def _annotate_teacher_rows_after_reopt(self, episode: int, sol: CGSolution, improvement: float) -> None:
        selected_by_gnn_in_rmp = 0
        for row in self.teacher_dataset_rows:
            if row.get("episode") != episode:
                continue
            lam = float(sol.lambda_values.get(row["pattern_id"], 0.0))
            reduced_cost = float(row.get("reduced_cost") or 0.0)
            row["lambda_value"] = lam
            row["selected_in_rmp"] = lam > 1e-6
            row["teacher_label"] = 1 if lam > 1e-6 else 0
            if row.get("selected_by_gnn") and row["selected_in_rmp"]:
                selected_by_gnn_in_rmp += 1
            row["teacher_score"] = float(lam * max(0.0, -reduced_cost))
            row["cg_objective_after_reopt"] = float(sol.objective)
            row["cg_objective_improvement"] = float(improvement)
        for row in self.gnn_selection_history:
            if row.get("episode") == episode:
                row["selected_by_gnn_in_rmp"] = selected_by_gnn_in_rmp

    def _mark_teacher_rows_passed_to_rmp(self, episode: int) -> None:
        pattern_ids_in_pool = {pat.pattern_id for pat in self.patterns}
        for row in self.teacher_dataset_rows:
            if row.get("episode") != episode:
                continue
            row["passed_to_rmp"] = row.get("pattern_id") in pattern_ids_in_pool

    def _load_gnn_utilities(self) -> bool:
        gnn_root = _project_path(self.gnn_root)
        if str(gnn_root) not in sys.path:
            sys.path.insert(0, str(gnn_root))
        try:
            utilities_module = importlib.import_module("utilities")
            self._gnn_build_graph = getattr(utilities_module, "build_bigraph_for_patterns")
            self._gnn_graph_to_tensors = getattr(utilities_module, "graph_to_tensors")
            self._gnn_normalize_dataset = getattr(utilities_module, "normalize_dataset")
            self._gnn_adaptive_select_indices = getattr(utilities_module, "adaptive_select_indices")
            return True
        except Exception as exc:
            self._gnn_unavailable_reason = str(exc)
            print(f"[GNN] Could not load graph utilities ({exc})")
            return False

    def _load_gnn_if_needed(self) -> bool:
        if not self.use_gnn:
            return False
        if self._gnn_loaded:
            return self._gnn_model is not None

        self._gnn_loaded = True
        checkpoint_path = _project_path(self.gnn_checkpoint)

        try:
            if not self._load_gnn_utilities():
                return False

            if not checkpoint_path.exists():
                self._gnn_unavailable_reason = f"checkpoint not found: {checkpoint_path}"
                print(f"[GNN] Model disabled, but teacher graph export remains enabled: {self._gnn_unavailable_reason}")
                return False

            torch = importlib.import_module("torch")
            model_module = importlib.import_module("models.attention.model")
            BiGATColumnScorer = getattr(model_module, "BiGATColumnScorer")

            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")

            config = dict(checkpoint.get("config", {}))
            dropout = float(config.pop("dropout", 0.0))
            model = BiGATColumnScorer(**config, dropout=dropout)
            model_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in checkpoint["state_dict"].items()
                if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
            }
            skipped = sorted(set(checkpoint["state_dict"]) - set(compatible_state))
            model.load_state_dict(compatible_state, strict=False)
            if skipped:
                print(f"[GNN] Skipped {len(skipped)} checkpoint tensors with incompatible shapes.")
            model.eval()

            self._gnn_model = model
            self._gnn_checkpoint_payload = checkpoint
            print(f"[GNN] BiGAT checkpoint loaded: {checkpoint_path}")
            return True
        except Exception as exc:
            self._gnn_unavailable_reason = str(exc)
            print(f"[GNN] Disabled: could not load BiGAT checkpoint ({exc})")
            self._gnn_model = None
            return False

    def _select_patterns_with_gnn(
        self,
        patterns: List[LTPattern],
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
    ) -> List[LTPattern]:
        if not patterns:
            return patterns
        model_available = self._load_gnn_if_needed()

        try:
            if self._gnn_build_graph is None:
                return patterns

            raw_graph = self._gnn_build_graph(
                patterns=patterns,
                data=self.data,
                need=need,
                surplus=surplus,
                dual_need=dual_need,
                dual_surplus=dual_surplus,
            )
            if (
                not model_available
                or self._gnn_model is None
                or self._gnn_graph_to_tensors is None
                or self._gnn_normalize_dataset is None
            ):
                probs = [0.0 for _ in patterns]
                score_rows = self._compute_combined_column_scores(patterns, probs)
                gnn_selected_idx = []
                selected_idx = list(range(len(patterns)))
                fallback_idx = None
                adaptive_info = {
                    "selection_mode": "runtime_no_model_pass_through",
                    "adaptive_k": len(patterns),
                    "mass_threshold": float(self.gnn_mass_threshold),
                    "relative_threshold": float(self.gnn_relative_threshold),
                    "score_mass_total": 0.0,
                }
            else:
                torch = importlib.import_module("torch")
                graph = self._gnn_graph_to_tensors(raw_graph)
                stats = (self._gnn_checkpoint_payload or {}).get("normalization")
                if stats:
                    normalized_graphs, _ = self._gnn_normalize_dataset([graph], stats)
                    graph = normalized_graphs[0]

                with torch.no_grad():
                    logits = self._gnn_model(
                        graph["column_features"],
                        graph["constraint_features"],
                        graph["edge_index_col_to_con"],
                        graph["edge_attr_col_to_con"],
                    )
                    probs_tensor = torch.sigmoid(logits).detach().cpu()

                probs = [float(value) for value in probs_tensor.tolist()]
                score_rows = self._compute_combined_column_scores(patterns, probs)
                gnn_selected_idx, adaptive_info = self._adaptive_select_columns(
                    score_rows, patterns=patterns
                )
                selected_idx, fallback_idx = self._apply_classical_fallback(patterns, gnn_selected_idx)
            gnn_selected_set = set(gnn_selected_idx)
            selected = [patterns[idx] for idx in selected_idx]
            full_negative_rc_count = sum(
                1 for pat in patterns
                if float(pat.metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
            )

            episode = self.current_episode or (len(self.gnn_selection_history) + 1)
            selected_rows = []
            for idx, pat in enumerate(patterns):
                score = float(score_rows[idx]["gnn_prob"])
                combined_score = float(score_rows[idx]["combined_score"])
                pat.metadata["gnn_score"] = round(score, 6)
                pat.metadata["gnn_combined_score"] = round(combined_score, 6)
                pat.metadata["gnn_selected"] = idx in gnn_selected_set
                pat.metadata["gnn_selected_by_fallback"] = idx == fallback_idx
                pat.metadata["adaptive_k_star"] = adaptive_info.get("adaptive_k")
                self._record_pattern_pool_diagnostic(
                    pat=pat,
                    stage="gnn_scored",
                    pattern_reduced_cost=pat.metadata.get("reduced_cost"),
                    duplicate_id_reject=False,
                    duplicate_signature_reject=False,
                    empty_flow_reject=False,
                    added_to_pool=False,
                    signature=pat.metadata.get("pair_signature", ""),
                )
            for rank, idx in enumerate(selected_idx, start=1):
                pat = patterns[idx]
                score = float(score_rows[idx]["gnn_prob"])
                combined_score = float(score_rows[idx]["combined_score"])
                pat.metadata["gnn_score"] = round(score, 6)
                pat.metadata["gnn_combined_score"] = round(combined_score, 6)
                pat.metadata["gnn_rank"] = rank
                pat.metadata["gnn_selected"] = idx in gnn_selected_set
                selected_rows.append({
                    "rank": rank,
                    "pattern_id": pat.pattern_id,
                    "gnn_prob": score,
                    "combined_score": combined_score,
                    "reduced_cost": pat.metadata.get("reduced_cost"),
                    "column_cost": pat.column_cost,
                    "selected_by_gnn": idx in gnn_selected_set,
                    "selected_by_classical_fallback": idx == fallback_idx,
                })

            self._record_teacher_rows(
                episode=episode,
                patterns=patterns,
                score_rows=score_rows,
                raw_graph=raw_graph,
                gnn_selected_idx=gnn_selected_idx,
                final_selected_idx=selected_idx,
                adaptive_info=adaptive_info,
                fallback_idx=fallback_idx,
            )

            self.gnn_selection_history.append({
                "branch_node_id": self.current_branch_node_id,
                "episode": episode,
                "n_candidates": len(patterns),
                "n_selected": len(selected),
                "selection_mode": adaptive_info.get("selection_mode"),
                "adaptive_k_star": adaptive_info.get("adaptive_k"),
                "negative_reduced_cost_candidates": full_negative_rc_count,
                "classical_fallback_enabled": self.use_classical_fallback,
                "classical_fallback_used": fallback_idx is not None,
                "fallback_idx": fallback_idx,
                "selected": selected_rows,
                # Fairness diagnostics — per (product, period) candidates vs
                # picks. Written by _adaptive_select_columns so the benchmark
                # CSV and irp_gnn_selected_columns.json can audit starvation.
                "per_group_diagnostics": adaptive_info.get("per_group_diagnostics", []),
                "gnn_min_per_group": adaptive_info.get("min_per_group"),
                "groups_forced_by_quota": adaptive_info.get("groups_forced_count", 0),
            })
            print(
                f"[GNN] Episode {episode}: scored {len(patterns)} priced columns, "
                f"adaptive_k={adaptive_info.get('adaptive_k')} kept={len(selected)} "
                f"negative_rc={full_negative_rc_count} fallback_used={fallback_idx is not None}"
            )
            for row in selected_rows[:10]:
                print(
                    f"  rank={row['rank']} | prob={row['gnn_prob']:.6f} "
                    f"| combined={row['combined_score']:.6f} "
                    f"| pattern={row['pattern_id']} | rc={row['reduced_cost']} "
                    f"| column_cost={row['column_cost']:.6f}"
                )
            return selected
        except Exception as exc:
            # Track scoring failures so the benchmark can detect when Run C
            # silently degraded to the all-patterns fallback (was previously a
            # blind passthrough with no accounting).
            self._gnn_scoring_failures = int(getattr(self, "_gnn_scoring_failures", 0)) + 1
            print(
                f"[GNN] Scoring skipped for this pricing episode (failure "
                f"#{self._gnn_scoring_failures}): {exc}"
            )
            return patterns

    def _collect_teacher_batch_without_gnn_prefilter(
        self,
        patterns: List[LTPattern],
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
    ) -> List[LTPattern]:
        if not patterns:
            return patterns
        if not self._load_gnn_utilities() or self._gnn_build_graph is None:
            return patterns
        try:
            raw_graph = self._gnn_build_graph(
                patterns=patterns,
                data=self.data,
                need=need,
                surplus=surplus,
                dual_need=dual_need,
                dual_surplus=dual_surplus,
            )
            probs = [0.0 for _ in patterns]
            score_rows = self._compute_combined_column_scores(patterns, probs)
            selected_idx = list(range(len(patterns)))
            adaptive_info = {
                "selection_mode": "teacher_full_batch_no_gnn_prefilter",
                "adaptive_k": len(patterns),
                "mass_threshold": None,
                "relative_threshold": None,
                "score_mass_total": 0.0,
            }
            for idx, pat in enumerate(patterns):
                pat.metadata["gnn_score"] = None
                pat.metadata["gnn_combined_score"] = None
                pat.metadata["gnn_selected"] = False
                pat.metadata["gnn_selected_by_fallback"] = False
                pat.metadata["adaptive_k_star"] = len(patterns)
                self._record_pattern_pool_diagnostic(
                    pat=pat,
                    stage="teacher_full_batch_recorded_before_rmp",
                    pattern_reduced_cost=pat.metadata.get("reduced_cost"),
                    signature=pat.metadata.get("pair_signature", ""),
                )
            self._record_teacher_rows(
                episode=self.current_episode or (len(self.gnn_selection_history) + 1),
                patterns=patterns,
                score_rows=score_rows,
                raw_graph=raw_graph,
                gnn_selected_idx=[],
                final_selected_idx=selected_idx,
                adaptive_info=adaptive_info,
                fallback_idx=None,
            )
            self.gnn_selection_history.append({
                "branch_node_id": self.current_branch_node_id,
                "episode": self.current_episode or (len(self.gnn_selection_history) + 1),
                "n_candidates": len(patterns),
                "n_selected": len(patterns),
                "selection_mode": adaptive_info["selection_mode"],
                "adaptive_k_star": len(patterns),
                "negative_reduced_cost_candidates": sum(
                    1 for pat in patterns
                    if float(pat.metadata.get("reduced_cost", 0.0) or 0.0) < -1e-6
                ),
                "classical_fallback_enabled": False,
                "classical_fallback_used": False,
                "fallback_idx": None,
                "selected": [],
                # Teacher-only path (no GNN prefilter) — no quota decision to
                # record, but keep schema consistent so downstream log joins
                # don't explode on missing keys.
                "per_group_diagnostics": [],
                "gnn_min_per_group": None,
                "groups_forced_by_quota": 0,
            })
            print(
                f"[Teacher] Episode {self.current_episode}: recorded full priced batch "
                f"without GNN prefilter ({len(patterns)} columns)."
            )
        except Exception as exc:
            # Count and surface these failures — silently losing teacher batches
            # silently shrinks the training set and skews the learned ranker.
            self._teacher_record_failures = getattr(self, "_teacher_record_failures", 0) + 1
            print(
                f"[Teacher] Could not record full priced batch graph: {exc} "
                f"(cumulative failures={self._teacher_record_failures})"
            )
        return patterns

    def add_patterns(self, new_patterns: Iterable[LTPattern]) -> int:
        existing_ids = {p.pattern_id for p in self.patterns}
        existing_signatures = {
            (p.product, p.period, tuple(sorted((i, j, round(q, 6)) for (i, j), q in p.pattern_flows.items())))
            for p in self.patterns
        }
        added = 0
        self.last_duplicate_rejects = 0
        self.last_signature_rejects = 0
        self.last_empty_rejects = 0
        self.last_added_patterns = 0
        for pat in new_patterns:
            signature = (
                pat.product,
                pat.period,
                tuple(sorted((i, j, round(q, 6)) for (i, j), q in pat.pattern_flows.items())),
            )
            duplicate_id = pat.pattern_id in existing_ids
            duplicate_signature = signature in existing_signatures
            empty_flow = not pat.pattern_flows
            added_to_pool = False
            if duplicate_id:
                self.last_duplicate_rejects += 1
            if duplicate_signature:
                self.last_signature_rejects += 1
            if empty_flow:
                self.last_empty_rejects += 1
            if duplicate_id or duplicate_signature or empty_flow:
                self._record_pattern_pool_diagnostic(
                    pat=pat,
                    stage="add_patterns_rejected",
                    pattern_reduced_cost=pat.metadata.get("reduced_cost"),
                    duplicate_id_reject=duplicate_id,
                    duplicate_signature_reject=duplicate_signature,
                    empty_flow_reject=empty_flow,
                    added_to_pool=False,
                    signature=str(signature),
                )
                continue
            self.patterns.append(pat)
            existing_ids.add(pat.pattern_id)
            existing_signatures.add(signature)
            added += 1
            self.last_added_patterns += 1
            added_to_pool = True
            self._record_pattern_pool_diagnostic(
                pat=pat,
                stage="add_patterns_added",
                pattern_reduced_cost=pat.metadata.get("reduced_cost"),
                duplicate_id_reject=False,
                duplicate_signature_reject=False,
                empty_flow_reject=False,
                added_to_pool=added_to_pool,
                signature=str(signature),
            )
        return added

    def _record_pattern_pool_diagnostic(
        self,
        pat: LTPattern,
        stage: str,
        pattern_reduced_cost: Optional[float] = None,
        duplicate_id_reject: bool = False,
        duplicate_signature_reject: bool = False,
        empty_flow_reject: bool = False,
        added_to_pool: bool = False,
        signature: str = "",
    ) -> None:
        if pat.pattern_flows:
            items = list(pat.pattern_flows.items())
        else:
            items = [((None, None), 0.0)]
        for (donor, receiver), qty in items:
            self.column_pool_diagnostics.append({
                "episode": self.current_episode,
                "stage": stage,
                "product": pat.product,
                "period": pat.period,
                "feature_name": pat.metadata.get("feature_name"),
                "donor_store": donor,
                "receiver_store": receiver,
                "qty_cap": qty,
                "reduced_cost_proxy": None,
                "stackelberg_accepted": None,
                "acceptance_score": pat.metadata.get("mean_acceptance_score"),
                "compensation": pat.metadata.get("mean_compensation"),
                "pattern_id": pat.pattern_id,
                "pattern_reduced_cost": pattern_reduced_cost,
                "gnn_score": pat.metadata.get("gnn_score"),
                "gnn_combined_score": pat.metadata.get("gnn_combined_score"),
                "gnn_selected": pat.metadata.get("gnn_selected"),
                "gnn_selected_by_fallback": pat.metadata.get("gnn_selected_by_fallback"),
                "adaptive_k_star": pat.metadata.get("adaptive_k_star"),
                "duplicate_id_reject": duplicate_id_reject,
                "duplicate_signature_reject": duplicate_signature_reject,
                "empty_flow_reject": empty_flow_reject,
                "added_to_pool": added_to_pool,
                "signature": signature,
            })

    def _build_need_and_surplus_proxies(self, master_solution: Optional[CGSolution] = None):
        d = self.data
        need, surplus = {}, {}
        sorted_periods = sorted(d.periods)
        period_pos = {t: idx for idx, t in enumerate(sorted_periods)}
        for s, p, t in itertools.product(d.stores, d.products, d.periods):
            idx = period_pos[t]
            need_window = sorted_periods[idx:idx + self.need_lookahead_periods]
            reserve_window = sorted_periods[idx:idx + self.surplus_reserve_periods]
            demand_cover_target = sum(
                float(d.realized_demand.get((s, p, tau), d.demand.get((s, p, tau), 0.0)))
                for tau in need_window
            )
            reserve_target = max(
                self.safety_stock_units,
                sum(
                    float(d.realized_demand.get((s, p, tau), d.demand.get((s, p, tau), 0.0)))
                    for tau in reserve_window
                ),
            )
            ending_inventory = max(0.0, float(
                d.post_shock_inventory.get((s, p, t), self.baseline.inv_store[(s, p, t)])
            ))
            shortage = max(0.0, float(
                d.post_shock_shortage.get((s, p, t), self.baseline.shortage[(s, p, t)])
            ))
            need[(s, p, t)] = max(shortage, demand_cover_target - ending_inventory)
            surplus[(s, p, t)] = max(0.0, ending_inventory - reserve_target)
        if master_solution is not None:
            for key, net_lt in master_solution.implied_net_lt.items():
                if key not in need:
                    continue
                net_lt = float(net_lt)
                if net_lt > 1e-9:
                    need[key] = max(0.0, need[key] - net_lt)
                elif net_lt < -1e-9:
                    surplus[key] = max(0.0, surplus[key] + net_lt)
        return need, surplus

    def _compute_active_product_periods(self, need, surplus) -> Set[Tuple[Product, Period]]:
        active = set()
        for p in self.data.products:
            for t in self.data.periods:
                total_need = sum(need[(s, p, t)] for s in self.data.stores)
                total_surplus = sum(surplus[(s, p, t)] for s in self.data.stores)
                min_signal = max(self.lt_activation_threshold, 1e-9)
                if total_need > min_signal and total_surplus > min_signal:
                    active.add((p, t))
        return active

    def _feature_value_map(
        self,
        p: Product,
        t: Period,
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
    ) -> Dict[Tuple[Store, Store], Dict[str, float]]:
        total_need = sum(need[(s, p, t)] for s in self.data.stores)
        total_surplus = sum(surplus[(s, p, t)] for s in self.data.stores)

        feature_map: Dict[Tuple[Store, Store], Dict[str, float]] = {}
        for i in self.data.stores:
            for j in self.data.stores:
                if i == j:
                    continue
                donor_surplus = surplus[(i, p, t)]
                recv_need = need[(j, p, t)]
                if donor_surplus <= 1e-9 or recv_need <= 1e-9:
                    continue
                feature_map[(i, j)] = {
                    "shortage_ratio": recv_need / max(total_need, 1e-9),
                    "surplus_ratio": donor_surplus / max(total_surplus, 1e-9),
                    "time_urgency": 1.0 / (1.0 + self._estimate_days_until_stockout(j, p, t, recv_need)),
                }
        return feature_map

    def _estimate_days_until_stockout(self, store: Store, product: Product, period: Period, receiver_need: float) -> float:
        demand_now = max(0.0, float(
            self.data.realized_demand.get(
                (store, product, period),
                self.data.demand.get((store, product, period), 0.0),
            )
        ))
        if receiver_need > 1e-9:
            if demand_now <= 1e-9:
                return 0.0
            serviceable_qty_before_stockout = max(0.0, demand_now - receiver_need)
            return serviceable_qty_before_stockout / demand_now

        ending_inventory = max(0.0, float(
            self.data.post_shock_inventory.get(
                (store, product, period),
                self.baseline.inv_store.get((store, product, period), 0.0),
            )
        ))
        future_demands = [
            max(0.0, float(
                self.data.realized_demand.get(
                    (store, product, tau),
                    self.data.demand.get((store, product, tau), 0.0),
                )
            ))
            for tau in self.data.periods
            if tau >= period
        ]
        positive_future_demands = [value for value in future_demands if value > 1e-9]
        if not positive_future_demands:
            return math.inf
        avg_future_demand = sum(positive_future_demands) / len(positive_future_demands)
        return ending_inventory / max(avg_future_demand, 1e-9)

    def _prune_pairs_by_feature(
        self,
        p: Product,
        t: Period,
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
        episode: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        feature_values = self._feature_value_map(p=p, t=t, need=need, surplus=surplus)
        self._last_candidate_pair_count = len(feature_values)
        pruned: Dict[str, List[Dict[str, Any]]] = {feature: [] for feature in self.feature_ranges.keys()}

        # Collect every viable candidate first so the adaptive pruner sees the
        # full reduced-cost distribution at the end of this (p, t) call.
        adaptive_candidates: List[Dict[str, Any]] = []

        # Choose admissibility check: adaptive windows when the pruner is on,
        # otherwise the static feature_ranges fall-through (fixed thresholds).
        use_adaptive = bool(self.adaptive_pruning_enabled and self.adaptive_pruner is not None)
        if use_adaptive:
            adaptive_windows = self.adaptive_pruner.current_windows(phase="lt")
            best_rc_pair: Optional[Tuple[Store, Store]] = None
            best_rc_value = math.inf

        def _is_admissible_under_windows(fvals_local: Dict[str, float]) -> bool:
            for fname, (lo, hi) in adaptive_windows.items():
                v = float(fvals_local.get(fname, 0.0))
                if v < lo or v > hi:
                    return False
            return True

        for (i, j), fvals in feature_values.items():
            donor_surplus = surplus[(i, p, t)]
            recv_need = need[(j, p, t)]
            qty_cap = min(donor_surplus, recv_need)
            if qty_cap <= 1e-9:
                continue
            unit_cost = self.data.ship_cost_lt[(i, j, p)]
            fixed_cost = self.data.fixed_dispatch_lt[(i, j)]
            dual_score = dual_need.get((j, p, t), 0.0) + dual_surplus.get((i, p, t), 0.0)
            reduced_cost_proxy = fixed_cost + unit_cost * qty_cap - dual_score * qty_cap
            fvals["negative_reduced_cost"] = max(0.0, -reduced_cost_proxy)
            base_rank_score = dual_score - unit_cost

            payload = {
                "pair": (i, j),
                "qty_cap": qty_cap,
                "fixed_cost": fixed_cost,
                "unit_cost": unit_cost,
                "dual_score": dual_score,
                "reduced_cost_proxy": reduced_cost_proxy,
                "base_rank_score": base_rank_score,
                "feature_values": fvals,
            }

            adaptive_candidates.append({
                "pair": (i, j),
                "reduced_cost": reduced_cost_proxy,
                "feature_values": dict(fvals),
            })
            if use_adaptive and reduced_cost_proxy < best_rc_value:
                best_rc_value = reduced_cost_proxy
                best_rc_pair = (i, j)

            for feature_name, bounds in self.feature_ranges.items():
                fval = fvals[feature_name]
                # Adaptive admission: window from pruner. Static fallback:
                # the (effectively open) bounds in self.feature_ranges.
                if use_adaptive:
                    admit = _is_admissible_under_windows(fvals)
                    # Safeguard: always keep best-RC pair so CG can progress.
                    if not admit and (i, j) == best_rc_pair:
                        admit = True
                else:
                    admit = bounds["min"] <= fval <= bounds["max"]
                if admit:
                    feature_bonus = 0.05 * fval
                    payload_copy = dict(payload)
                    payload_copy["feature_name"] = feature_name
                    payload_copy["feature_score"] = payload["base_rank_score"] + feature_bonus
                    pruned[feature_name].append(payload_copy)
                    if self._log_candidate_pairs:
                        self.column_pool_diagnostics.append({
                            "episode": episode,
                            "stage": "candidate_pair_before_stackelberg",
                            "product": p,
                            "period": t,
                            "feature_name": feature_name,
                            "donor_store": i,
                            "receiver_store": j,
                            "qty_cap": qty_cap,
                            "reduced_cost_proxy": reduced_cost_proxy,
                            "stackelberg_accepted": None,
                            "acceptance_score": None,
                            "compensation": None,
                            "pattern_id": "",
                            "pattern_reduced_cost": None,
                            "gnn_score": None,
                            "gnn_combined_score": None,
                            "gnn_selected": None,
                            "gnn_selected_by_fallback": None,
                            "adaptive_k_star": None,
                            "duplicate_id_reject": False,
                            "duplicate_signature_reject": False,
                            "empty_flow_reject": False,
                            "added_to_pool": False,
                            "signature": "",
                        })

        for feature_name, rows in pruned.items():
            rows.sort(key=lambda x: x["feature_score"], reverse=True)
            pruned[feature_name] = rows[:self.top_pairs_per_feature]

        # Adaptive pruning: buffer this (p, t)'s candidates. The accumulated
        # batch is flushed to the pruner once per CG iteration (in
        # _candidate_patterns_from_duals) so windows tighten on the full
        # iteration's distribution, not the tiny per-(p, t) slice.
        if self.adaptive_pruning_enabled and self.adaptive_pruner is not None:
            self._adaptive_pending_candidates.extend(adaptive_candidates)
        return pruned

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _solve_stackelberg_for_pair(
        self,
        *,
        p: Product,
        t: Period,
        row: Dict[str, Any],
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
    ) -> StackelbergDecision:
        params = self.stackelberg_params

        i, j = row["pair"]
        q = float(max(0.0, row["qty_cap"]))

        if q <= 1e-9:
            return StackelbergDecision(
                accepted=False,
                compensation=0.0,
                donor_utility=-1e9,
                receiver_utility=-1e9,
                acceptance_score=0.0,
                details={"reason": -1.0},
            )

        donor_surplus = max(0.0, surplus[(i, p, t)])
        receiver_need = max(0.0, need[(j, p, t)])

        shortage_risk_increase = q / max(donor_surplus, 1e-9)
        service_level_loss = q / max(donor_surplus + 1.0, 1e-9)
        ship_burden = row["unit_cost"] * q + row["fixed_cost"]

        donor_noncomp_cost = (
            params.donor_risk_weight * shortage_risk_increase
            + params.donor_ship_burden_weight * ship_burden
            + params.donor_service_loss_weight * service_level_loss
        )
        donor_required_comp = params.donor_accept_threshold + donor_noncomp_cost

        shortage_reduction = min(q, receiver_need) / max(receiver_need, 1e-9)
        service_gain = min(q, receiver_need) / max(receiver_need + 1.0, 1e-9)
        handling_cost = 0.25 * row["unit_cost"] * q

        receiver_benefit_before_comp = (
            params.receiver_shortage_reduction_weight * shortage_reduction
            + params.receiver_service_gain_weight * service_gain
            - params.receiver_handling_weight * handling_cost
        )
        receiver_max_comp = receiver_benefit_before_comp - params.receiver_accept_threshold

        compensation = max(params.min_compensation, donor_required_comp)

        accepted = (
            compensation <= receiver_max_comp
            and compensation <= params.compensation_cap
        )

        donor_utility = compensation - donor_noncomp_cost
        receiver_utility = receiver_benefit_before_comp - compensation

        acceptance_score = 0.5 * (
            self._sigmoid(donor_utility) + self._sigmoid(receiver_utility)
        )

        return StackelbergDecision(
            accepted=accepted,
            compensation=float(compensation),
            donor_utility=float(donor_utility),
            receiver_utility=float(receiver_utility),
            acceptance_score=float(acceptance_score if accepted else 0.0),
            details={
                "q": float(q),
                "donor_surplus": float(donor_surplus),
                "receiver_need": float(receiver_need),
                "shortage_risk_increase": float(shortage_risk_increase),
                "service_level_loss": float(service_level_loss),
                "ship_burden": float(ship_burden),
                "shortage_reduction": float(shortage_reduction),
                "service_gain": float(service_gain),
                "handling_cost": float(handling_cost),
                "donor_required_comp": float(donor_required_comp),
                "receiver_max_comp": float(receiver_max_comp),
            },
        )

    def _apply_stackelberg_game_to_pairs(
        self,
        p: Product,
        t: Period,
        pruned_pairs_by_feature: Dict[str, List[Dict[str, Any]]],
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        episode: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        params = self.stackelberg_params
        accepted_by_feature: Dict[str, List[Dict[str, Any]]] = {
            feature_name: [] for feature_name in pruned_pairs_by_feature.keys()
        }

        for feature_name, rows in pruned_pairs_by_feature.items():
            rejected_candidates: List[Dict[str, Any]] = []
            for row in rows:
                decision = self._solve_stackelberg_for_pair(
                    p=p,
                    t=t,
                    row=row,
                    need=need,
                    surplus=surplus,
                )

                row2 = dict(row)
                row2["stackelberg_accepted"] = decision.accepted
                row2["compensation"] = decision.compensation
                row2["donor_utility"] = decision.donor_utility
                row2["receiver_utility"] = decision.receiver_utility
                row2["acceptance_score"] = decision.acceptance_score
                row2["stackelberg_details"] = decision.details
                i, j = row["pair"]
                self.column_pool_diagnostics.append({
                    "episode": episode,
                    "stage": "after_stackelberg",
                    "product": p,
                    "period": t,
                    "feature_name": feature_name,
                    "donor_store": i,
                    "receiver_store": j,
                    "qty_cap": row.get("qty_cap"),
                    "reduced_cost_proxy": row.get("reduced_cost_proxy"),
                    "stackelberg_accepted": decision.accepted,
                    "acceptance_score": decision.acceptance_score,
                    "compensation": decision.compensation,
                    "pattern_id": "",
                    "pattern_reduced_cost": None,
                    "gnn_score": None,
                    "gnn_combined_score": None,
                    "gnn_selected": None,
                    "gnn_selected_by_fallback": None,
                    "adaptive_k_star": None,
                    "duplicate_id_reject": False,
                    "duplicate_signature_reject": False,
                    "empty_flow_reject": False,
                    "added_to_pool": False,
                    "signature": "",
                })

                if not decision.accepted:
                    fallback_economic_score = max(0.0, -float(row.get("reduced_cost_proxy", 0.0)))
                    row2["post_game_score"] = fallback_economic_score
                    row2["stackelberg_fallback_used"] = False
                    rejected_candidates.append(row2)
                    continue

                combined_score = (
                    params.acceptance_score_weight * decision.acceptance_score
                    + params.economic_score_weight * row["base_rank_score"]
                )
                row2["post_game_score"] = combined_score
                row2["stackelberg_fallback_used"] = False
                accepted_by_feature[feature_name].append(row2)

            if not accepted_by_feature[feature_name] and params.allow_pricing_fallback_when_no_acceptance:
                fallback_rows = []
                for row in rejected_candidates:
                    if (
                        params.fallback_requires_negative_reduced_cost
                        and float(row.get("reduced_cost_proxy", 0.0)) >= -1e-9
                    ):
                        continue
                    fallback_row = dict(row)
                    fallback_row["stackelberg_fallback_used"] = True
                    fallback_row["acceptance_score"] = max(
                        float(fallback_row.get("acceptance_score", 0.0)),
                        0.10,
                    )
                    fallback_row["post_game_score"] = (
                        max(0.0, -float(fallback_row.get("reduced_cost_proxy", 0.0)))
                        + 0.01 * float(fallback_row.get("base_rank_score", 0.0))
                    )
                    fallback_rows.append(fallback_row)
                fallback_rows.sort(key=lambda x: x["post_game_score"], reverse=True)
                fallback_rows = fallback_rows[:max(1, int(params.fallback_top_k_after_game_per_feature))]
                accepted_by_feature[feature_name].extend(fallback_rows)
                for row in fallback_rows:
                    i, j = row["pair"]
                    self.column_pool_diagnostics.append({
                        "episode": episode,
                        "stage": "stackelberg_pricing_fallback",
                        "product": p,
                        "period": t,
                        "feature_name": feature_name,
                        "donor_store": i,
                        "receiver_store": j,
                        "qty_cap": row.get("qty_cap"),
                        "reduced_cost_proxy": row.get("reduced_cost_proxy"),
                        "stackelberg_accepted": False,
                        "acceptance_score": row.get("acceptance_score"),
                        "compensation": row.get("compensation"),
                        "pattern_id": "",
                        "pattern_reduced_cost": None,
                        "gnn_score": None,
                        "gnn_combined_score": None,
                        "gnn_selected": None,
                        "gnn_selected_by_fallback": None,
                        "adaptive_k_star": None,
                        "duplicate_id_reject": False,
                        "duplicate_signature_reject": False,
                        "empty_flow_reject": False,
                        "added_to_pool": False,
                        "signature": "",
                    })

            accepted_by_feature[feature_name].sort(
                key=lambda x: x["post_game_score"], reverse=True
            )
            accepted_by_feature[feature_name] = accepted_by_feature[feature_name][
                :params.top_k_after_game_per_feature
            ]

        return accepted_by_feature

    def _build_patterns_from_pruned_pairs(
        self,
        p: Product,
        t: Period,
        pruned_pairs_by_feature: Dict[str, List[Dict[str, Any]]],
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        rc_tol: float,
        episode: int = 0,
    ) -> List[LTPattern]:
        new_patterns: List[LTPattern] = []
        for feature_name, rows in pruned_pairs_by_feature.items():
            if not rows:
                continue
            built_here = 0
            for start_idx in range(min(len(rows), self.top_patterns_per_feature)):
                donor_work = {s: surplus[(s, p, t)] for s in self.data.stores}
                recv_work = {s: need[(s, p, t)] for s in self.data.stores}
                flows: Dict[Tuple[Store, Store], float] = {}
                pattern_cost = 0.0
                reduced_cost = 0.0

                ordered_rows = rows[start_idx:] + rows[:start_idx]
                for row in ordered_rows:
                    if len(flows) >= self.max_pairs_per_pattern:
                        break
                    i, j = row["pair"]
                    qty = min(donor_work.get(i, 0.0), recv_work.get(j, 0.0), row["qty_cap"])
                    if qty <= 1e-9:
                        continue
                    pair_rc = row["fixed_cost"] + row["unit_cost"] * qty - row["dual_score"] * qty
                    if pair_rc >= -1e-9 and flows:
                        continue
                    flows[(i, j)] = qty
                    donor_work[i] -= qty
                    recv_work[j] -= qty
                    pattern_cost += row["fixed_cost"] + row["unit_cost"] * qty
                    reduced_cost += pair_rc

                if flows and reduced_cost < rc_tol:
                    built_here += 1
                    pair_signature = tuple(sorted((i, j, round(qty, 6)) for (i, j), qty in flows.items()))
                    fallback_pair_count = sum(1 for r in rows if r.get("stackelberg_fallback_used"))
                    new_patterns.append(
                        LTPattern(
                            pattern_id=f"PRICED_E{episode}_{feature_name}_{p}_T{t}_{built_here}",
                            period=t,
                            product=p,
                            pattern_flows=flows,
                            column_cost=round(pattern_cost, 6),
                            metadata={
                                "source": "pricing_pruned_feature",
                                "feature_name": feature_name,
                                "reduced_cost": round(reduced_cost, 6),
                                "pruned_pair_count": len(rows),
                                "feature_range": self.feature_ranges[feature_name],
                                "stackelberg_used": True,
                                "stackelberg_fallback_used": fallback_pair_count > 0,
                                "stackelberg_fallback_pair_count": fallback_pair_count,
                                "mean_acceptance_score": round(
                                    sum(r.get("acceptance_score", 0.0) for r in rows) / max(len(rows), 1), 6
                                ),
                                "mean_compensation": round(
                                    sum(r.get("compensation", 0.0) for r in rows) / max(len(rows), 1), 6
                                ),
                                "pair_signature": str(pair_signature),
                            },
                        )
                    )
                elif flows:
                    self.column_pool_diagnostics.append({
                        "episode": episode,
                        "stage": "pattern_rejected_reduced_cost",
                        "product": p,
                        "period": t,
                        "feature_name": feature_name,
                        "donor_store": "",
                        "receiver_store": "",
                        "qty_cap": sum(flows.values()),
                        "reduced_cost_proxy": None,
                        "stackelberg_accepted": None,
                        "acceptance_score": None,
                        "compensation": None,
                        "pattern_id": f"REJECTED_E{episode}_{feature_name}_{p}_T{t}_{start_idx + 1}",
                        "pattern_reduced_cost": round(reduced_cost, 6),
                        "gnn_score": None,
                        "gnn_combined_score": None,
                        "gnn_selected": None,
                        "gnn_selected_by_fallback": None,
                        "adaptive_k_star": None,
                        "duplicate_id_reject": False,
                        "duplicate_signature_reject": False,
                        "empty_flow_reject": False,
                        "added_to_pool": False,
                        "signature": str(tuple(sorted((i, j, round(qty, 6)) for (i, j), qty in flows.items()))),
                    })
        return new_patterns

    def _solve_exact_pricing_subproblem(
        self,
        p: Product,
        t: Period,
        need: Dict[Tuple[Store, Product, Period], float],
        surplus: Dict[Tuple[Store, Product, Period], float],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
        rc_tol: float,
        episode: int,
    ) -> List[LTPattern]:
        """Exact CG pricing subproblem solved by Gurobi for one (product, period).

        Solves the fixed-charge MIP

            min  sum_{i,j} [f_ij * y_ij + (c_ij - dual_need_j - dual_surplus_i) * q_ij]
            s.t. sum_j q_ij <= surplus_i            (donor capacity)
                 sum_i q_ij <= need_j               (receiver capacity)
                 q_ij <= min(surplus_i, need_j) * y_ij  (fixed-charge)
                 sum y_ij <= max_pairs_per_pattern
                 y_ij in {0,1}, q_ij >= 0

        Uses Gurobi PoolSearchMode=2 (systematic K-best search) to return up to
        `exact_pricing_pool_size` distinct negative-RC columns per subproblem per
        iteration. This gives A0 a comparable multi-column generation rate to the
        classical CG (which generates many columns per iteration via enumerate-
        and-filter), making the runtime comparison fair.

        NO heuristic filter (pruning / Stackelberg / GNN / top-k) is applied
        anywhere in this path.
        """
        d = self.data
        donors = [s for s in d.stores if float(surplus.get((s, p, t), 0.0)) > 1e-9]
        receivers = [s for s in d.stores if float(need.get((s, p, t), 0.0)) > 1e-9]
        pairs = [(i, j) for i in donors for j in receivers if i != j]
        if not pairs:
            return []

        mdl = gp.Model(f"ExactPricing_{p}_T{t}", env=get_gurobi_env())
        mdl.Params.OutputFlag = 0
        # Request K-best pool: PoolSearchMode=2 systematically finds up to
        # PoolSolutions solutions ordered by objective value.
        pool_size = self.exact_pricing_pool_size
        mdl.Params.PoolSearchMode = 2
        mdl.Params.PoolSolutions = pool_size
        if self.exact_pricing_time_limit is not None:
            mdl.Params.TimeLimit = float(self.exact_pricing_time_limit)

        q = mdl.addVars(pairs, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
        y = mdl.addVars(pairs, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="y")

        for (i, j) in pairs:
            qty_cap = min(float(surplus[(i, p, t)]), float(need[(j, p, t)]))
            if qty_cap <= 1e-9:
                mdl.addConstr(q[(i, j)] == 0.0)
                mdl.addConstr(y[(i, j)] == 0.0)
                continue
            mdl.addConstr(q[(i, j)] <= qty_cap * y[(i, j)])

        for i in donors:
            outgoing = [(i, j) for (ii, j) in pairs if ii == i]
            if outgoing:
                mdl.addConstr(
                    gp.quicksum(q[key] for key in outgoing) <= float(surplus[(i, p, t)])
                )
        for j in receivers:
            incoming = [(i, j) for (i, jj) in pairs if jj == j]
            if incoming:
                mdl.addConstr(
                    gp.quicksum(q[key] for key in incoming) <= float(need[(j, p, t)])
                )

        if self.max_pairs_per_pattern > 0:
            mdl.addConstr(
                gp.quicksum(y[key] for key in pairs) <= int(self.max_pairs_per_pattern)
            )

        obj_expr = gp.quicksum(
            float(d.fixed_dispatch_lt[(i, j)]) * y[(i, j)]
            + (
                float(d.ship_cost_lt[(i, j, p)])
                - float(dual_need.get((j, p, t), 0.0))
                - float(dual_surplus.get((i, p, t), 0.0))
            ) * q[(i, j)]
            for (i, j) in pairs
        )
        mdl.setObjective(obj_expr, GRB.MINIMIZE)
        mdl.optimize()

        if mdl.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or mdl.SolCount == 0:
            return []

        # Extract all pool solutions with negative reduced cost.
        results: List[LTPattern] = []
        for sol_idx in range(mdl.SolCount):
            mdl.Params.SolutionNumber = sol_idx
            sol_obj = float(mdl.PoolObjVal)
            if sol_obj >= rc_tol:
                break  # pool is ordered ascending; no need to check further
            flows: Dict[Tuple[Store, Store], float] = {}
            pattern_cost = 0.0
            for (i, j) in pairs:
                q_val = float(q[(i, j)].Xn)
                y_val = float(y[(i, j)].Xn)
                if q_val > 1e-9 and y_val > 0.5:
                    flows[(i, j)] = q_val
                    pattern_cost += float(d.fixed_dispatch_lt[(i, j)]) + float(
                        d.ship_cost_lt[(i, j, p)]
                    ) * q_val
            if not flows:
                continue
            results.append(
                LTPattern(
                    pattern_id=f"EXACT_E{episode}_{p}_T{t}_P{sol_idx}",
                    period=t,
                    product=p,
                    pattern_flows=flows,
                    column_cost=round(pattern_cost, 6),
                    metadata={
                        "source": "exact_pricing_gurobi",
                        "reduced_cost": round(sol_obj, 6),
                        "pruning_used": False,
                        "stackelberg_used": False,
                        "gnn_used": False,
                        "heuristic_top_k_used": False,
                        "feature_name": "exact_full",
                        "exact_pool_index": sol_idx,
                    },
                )
            )
        return results

    def _candidate_patterns_exact_full(
        self,
        need,
        surplus,
        active_product_periods: Set[Tuple[Product, Period]],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
        rc_tol: float,
        episode: int,
    ) -> List[LTPattern]:
        """Generate A0 benchmark candidates via exact Gurobi pricing only.

        One exact pricing MIP is solved per active (product, period). No
        pruning, Stackelberg game, GNN filtering, or top-k ranking is applied.
        """
        new_patterns: List[LTPattern] = []
        pairs_enumerated = 0
        subproblems_with_negative_rc = 0
        for p, t in sorted(active_product_periods):
            donors = [s for s in self.data.stores if float(surplus.get((s, p, t), 0.0)) > 1e-9]
            receivers = [s for s in self.data.stores if float(need.get((s, p, t), 0.0)) > 1e-9]
            pairs_enumerated += sum(1 for i in donors for j in receivers if i != j)
            priced = self._solve_exact_pricing_subproblem(
                p=p,
                t=t,
                need=need,
                surplus=surplus,
                dual_need=dual_need,
                dual_surplus=dual_surplus,
                rc_tol=rc_tol,
                episode=episode,
            )
            if priced:
                subproblems_with_negative_rc += 1
            new_patterns.extend(priced)
        patterns_built_before_dedup = len(new_patterns)
        new_patterns = self._deduplicate_priced_patterns(new_patterns, episode=episode)
        self._last_pricing_summary = {
            "candidate_pairs_before_pruning": pairs_enumerated,
            "pairs_after_pruning": pairs_enumerated,  # no pruning in exact mode
            "pairs_accepted_stackelberg": 0,
            "pairs_recovered_stackelberg_fallback": 0,
            "patterns_built_before_dedup": patterns_built_before_dedup,
            "patterns_deduplicated_before_gnn": patterns_built_before_dedup - len(new_patterns),
            "patterns_removed_by_product_period_cap": 0,
            "patterns_built_before_gnn": len(new_patterns),
            "max_columns_per_product_period": self.max_columns_per_product_period,
            "exact_full_mode": True,
            "exact_subproblems_with_negative_rc": subproblems_with_negative_rc,
            "exact_subproblems_solved": len(active_product_periods),
        }
        return new_patterns

    def _candidate_patterns_from_duals(
        self,
        need,
        surplus,
        active_product_periods: Set[Tuple[Product, Period]],
        dual_need: Dict[Tuple[Store, Product, Period], float],
        dual_surplus: Dict[Tuple[Store, Product, Period], float],
        rc_tol: float = -1e-6,
        episode: int = 0,
    ) -> List[LTPattern]:
        new_patterns: List[LTPattern] = []
        candidate_pairs_before_pruning = 0
        pairs_after_pruning = 0
        pairs_accepted_stackelberg = 0
        pairs_recovered_stackelberg_fallback = 0

        # Reset the adaptive-pruning buffer so the iteration sees only this
        # call's candidates.
        if self.adaptive_pruning_enabled and self.adaptive_pruner is not None:
            self._adaptive_pending_candidates = []

        for p, t in sorted(active_product_periods):
            pruned_pairs_by_feature = self._prune_pairs_by_feature(
                p=p,
                t=t,
                need=need,
                surplus=surplus,
                dual_need=dual_need,
                dual_surplus=dual_surplus,
                episode=episode,
            )
            candidate_pairs_before_pruning += self._last_candidate_pair_count
            pairs_after_pruning += sum(len(rows) for rows in pruned_pairs_by_feature.values())
            accepted_pairs_by_feature = self._apply_stackelberg_game_to_pairs(
                p=p,
                t=t,
                pruned_pairs_by_feature=pruned_pairs_by_feature,
                need=need,
                surplus=surplus,
                episode=episode,
            )
            pairs_accepted_stackelberg += sum(
                1
                for rows in accepted_pairs_by_feature.values()
                for row in rows
                if row.get("stackelberg_accepted") and not row.get("stackelberg_fallback_used")
            )
            pairs_recovered_stackelberg_fallback += sum(
                1
                for rows in accepted_pairs_by_feature.values()
                for row in rows
                if row.get("stackelberg_fallback_used")
            )
            feature_patterns = self._build_patterns_from_pruned_pairs(
                p=p,
                t=t,
                pruned_pairs_by_feature=accepted_pairs_by_feature,
                need=need,
                surplus=surplus,
                rc_tol=rc_tol,
                episode=episode,
            )
            new_patterns.extend(feature_patterns)

        # Flush per-iteration adaptive-pruning update with the full pooled
        # candidate distribution from every (p, t).
        if (
            self.adaptive_pruning_enabled
            and self.adaptive_pruner is not None
            and self._adaptive_pending_candidates
        ):
            update_record = self.adaptive_pruner.update(
                self._adaptive_pending_candidates,
                iteration=int(episode or self.current_episode or 0),
                phase="lt",
                lb=self._adaptive_pruning_lb,
                ub=self._adaptive_pruning_ub,
            )
            if self._log_candidate_pairs:
                self.column_pool_diagnostics.append({
                    "episode": episode,
                    "stage": "adaptive_pruning_update",
                    "product": "",
                    "period": "",
                    "feature_name": "_adaptive_windows",
                    "donor_store": "",
                    "receiver_store": "",
                    "qty_cap": 0.0,
                    "reduced_cost_proxy": None,
                    "stackelberg_accepted": None,
                    "acceptance_score": None,
                    "compensation": None,
                    "pattern_id": "",
                    "pattern_reduced_cost": None,
                    "gnn_score": None,
                    "gnn_combined_score": None,
                    "gnn_selected": None,
                    "gnn_selected_by_fallback": None,
                    "adaptive_k_star": None,
                    "duplicate_id_reject": False,
                    "duplicate_signature_reject": False,
                    "empty_flow_reject": False,
                    "added_to_pool": False,
                    "signature": json.dumps(update_record, default=str),
                })
            self._adaptive_pending_candidates = []

        patterns_built_before_dedup = len(new_patterns)
        new_patterns = self._deduplicate_priced_patterns(new_patterns, episode=episode)
        patterns_after_dedup = len(new_patterns)
        if self.max_columns_per_product_period > 0:
            capped_patterns = []
            by_product_period: Dict[Tuple[Product, Period], List[LTPattern]] = {}
            for pat in new_patterns:
                by_product_period.setdefault((pat.product, pat.period), []).append(pat)
            for key in sorted(by_product_period):
                rows = by_product_period[key]
                rows.sort(key=lambda pat: float(pat.metadata.get("reduced_cost", 0.0) or 0.0))
                capped_patterns.extend(rows[:self.max_columns_per_product_period])
            new_patterns = capped_patterns
        self._last_pricing_summary = {
            "candidate_pairs_before_pruning": candidate_pairs_before_pruning,
            "pairs_after_pruning": pairs_after_pruning,
            "pairs_accepted_stackelberg": pairs_accepted_stackelberg,
            "pairs_recovered_stackelberg_fallback": pairs_recovered_stackelberg_fallback,
            "patterns_built_before_dedup": patterns_built_before_dedup,
            "patterns_deduplicated_before_gnn": patterns_built_before_dedup - patterns_after_dedup,
            "patterns_removed_by_product_period_cap": patterns_after_dedup - len(new_patterns),
            "patterns_built_before_gnn": len(new_patterns),
            "max_columns_per_product_period": self.max_columns_per_product_period,
        }
        return new_patterns

    def solve_rmp(self, msg: bool = False, return_model: bool = False):
        d = self.data
        need, surplus = self._build_need_and_surplus_proxies()
        active_product_periods = self._compute_active_product_periods(need, surplus)
        mdl = gp.Model("LT_RMP", env=get_gurobi_env())
        mdl.Params.OutputFlag = 1 if msg else 0

        pattern_map = {pat.pattern_id: pat for pat in self.patterns if (pat.product, pat.period) in active_product_periods}
        lam = mdl.addVars(list(pattern_map.keys()), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="lambda")
        infeasible_branch = False
        for pid, (lb, ub) in self.branch_bounds.items():
            lb = max(0.0, float(lb))
            ub = min(1.0, float(ub))
            if pid not in pattern_map:
                if lb > 1e-9:
                    infeasible_branch = True
                continue
            lam[pid].LB = lb
            lam[pid].UB = ub
        residual_need_keys = [(s, p, t) for s in d.stores for p in d.products for t in d.periods]
        residual_need = mdl.addVars(residual_need_keys, lb=0.0, vtype=GRB.CONTINUOUS, name="residual_need")

        baseline_shortage_component = sum(
            d.shortage_cost[(s, p)] * float(
                d.post_shock_shortage.get((s, p, t), self.baseline.shortage[(s, p, t)])
            )
            for s, p, t in itertools.product(d.stores, d.products, d.periods)
        )
        baseline_without_shortage = float(self.baseline.objective) - baseline_shortage_component
        need_penalty = {
            (s, p, t): (
                d.shortage_cost[(s, p)]
                if float(d.post_shock_shortage.get((s, p, t), self.baseline.shortage[(s, p, t)])) > 1e-9
                else self.rebalance_need_penalty
            )
            for s, p, t in residual_need_keys
        }

        mdl.setObjective(
            baseline_without_shortage
            + gp.quicksum(pat.column_cost * lam[pat.pattern_id] for pat in pattern_map.values())
            + gp.quicksum(need_penalty[(s, p, t)] * residual_need[(s, p, t)] for s, p, t in residual_need_keys),
            GRB.MINIMIZE,
        )

        need_constraints = {}
        surplus_constraints = {}

        for s, p, t in residual_need_keys:
            relevant = [pat for pat in pattern_map.values() if pat.product == p and pat.period == t]
            inflow = gp.quicksum(
                qty * lam[pat.pattern_id]
                for pat in relevant
                for (i, j), qty in pat.pattern_flows.items()
                if j == s
            )
            con = mdl.addConstr(residual_need[(s, p, t)] + inflow >= need[(s, p, t)], name=f"need_cover__{len(need_constraints)}")
            need_constraints[(s, p, t)] = con

            if (p, t) not in active_product_periods:
                mdl.addConstr(residual_need[(s, p, t)] == need[(s, p, t)], name=f"inactive_fix__{s}__{t}__{len(need_constraints)}")

        for s, p, t in residual_need_keys:
            if (p, t) not in active_product_periods:
                continue
            relevant = [pat for pat in pattern_map.values() if pat.product == p and pat.period == t]
            outbound = gp.quicksum(
                qty * lam[pat.pattern_id]
                for pat in relevant
                for (i, j), qty in pat.pattern_flows.items()
                if i == s
            )
            con = mdl.addConstr(outbound <= surplus[(s, p, t)], name=f"surplus_cap__{len(surplus_constraints)}")
            surplus_constraints[(s, p, t)] = con

        if infeasible_branch:
            mdl.addConstr(gp.LinExpr(0.0) >= 1.0, name="infeasible_missing_forced_branch_pattern")

        mdl.optimize()

        lambda_values = {pid: _safe_var_value(mdl, lam[pid]) for pid in pattern_map.keys()}
        selected = [k for k, v in lambda_values.items() if v > 1e-6]

        dual_need = {
            key: float(con.Pi) if mdl.Status == GRB.OPTIMAL else 0.0
            for key, con in need_constraints.items()
        }
        dual_surplus = {
            key: float(con.Pi) if mdl.Status == GRB.OPTIMAL else 0.0
            for key, con in surplus_constraints.items()
        }

        implied_net_lt = {}
        for s, p, t in residual_need_keys:
            net = 0.0
            for pat in pattern_map.values():
                if pat.product != p or pat.period != t:
                    continue
                coeff = lambda_values[pat.pattern_id]
                for (i, j), qty in pat.pattern_flows.items():
                    if j == s:
                        net += qty * coeff
                    if i == s:
                        net -= qty * coeff
            implied_net_lt[(s, p, t)] = net

        sol = CGSolution(
            status=_grb_status_name(mdl.Status),
            objective=_safe_obj_value(mdl),
            lambda_values=lambda_values,
            selected_patterns=selected,
            implied_net_lt=implied_net_lt,
            dual_need=dual_need,
            dual_surplus=dual_surplus,
            active_product_periods=active_product_periods,
            efficiency_metrics=_model_efficiency_metrics(mdl),
        )
        if return_model:
            return sol, mdl
        return sol

    # ──────────────────────────────────────────────────────────────────────
    # Follower-aware column scoring helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_stackelberg_min_lateral_qty(self) -> float:
        """Return the minimum LT shipment size for follower best-response."""
        if self._stackelberg_min_lateral_qty_override is not None:
            return self._stackelberg_min_lateral_qty_override
        try:
            return float(os.environ.get("IRP_LT_MIN_UNITS", "5"))
        except ValueError:
            return 5.0

    def _build_stackelberg_follower_solver(self) -> FollowerBestResponseSolver:
        return FollowerBestResponseSolver(
            data=self.data,
            min_lateral_qty=self._get_stackelberg_min_lateral_qty(),
            use_exact_lp=self.stackelberg_exact_follower,
        )

    def _score_patterns_by_follower_response(
        self,
        patterns: List[LTPattern],
        need: Dict[Tuple, float],
        surplus: Dict[Tuple, float],
        follower: FollowerBestResponseSolver,
        rc_tol: float = -1e-6,
    ) -> List[Tuple[LTPattern, float]]:
        """Score each candidate pattern by total system cost delta after follower.

        For each pattern c targeting (product p, period t):
          1. Compute baseline follower cost for (p, t) without c.
          2. Apply c's flows to produce updated need/surplus.
          3. Re-run follower on updated state to get residual follower cost.
          4. delta = c.column_cost + residual_follower.total_cost
                     - baseline_follower.total_cost
             Negative delta → pattern reduces total system cost → good column.

        The baseline is cached per (p, t) so each (p, t) pair is solved at most
        once per call regardless of how many patterns share the same (p, t).

        Returns list of (pattern, delta) pairs, sorted ascending by delta.
        """
        # Step 1: cache baseline follower response per active (p, t)
        active_pts: Set[Tuple] = {(pat.product, pat.period) for pat in patterns}
        baseline_cache: Dict[Tuple, FollowerBestResponseResult] = {}
        for p, t in active_pts:
            baseline_cache[(p, t)] = follower.solve_for_product_period(p, t, need, surplus)

        scored: List[Tuple[LTPattern, float]] = []
        for pat in patterns:
            p, t = pat.product, pat.period
            baseline = baseline_cache.get((p, t))
            if baseline is None:
                # Safety: treat as pure RC-based — only accept if negative RC
                rc = float(pat.metadata.get("reduced_cost", 0.0) or 0.0)
                scored.append((pat, rc))
                continue

            # Step 2: apply pattern flows → updated need/surplus
            upd_need = dict(need)
            upd_surplus = dict(surplus)
            for (i, j), qty in pat.pattern_flows.items():
                key_j = (j, p, t)
                key_i = (i, p, t)
                upd_need[key_j] = max(0.0, upd_need.get(key_j, 0.0) - qty)
                upd_surplus[key_i] = max(0.0, upd_surplus.get(key_i, 0.0) - qty)

            # Step 3: residual follower after this pattern is activated
            residual = follower.solve_for_product_period(p, t, upd_need, upd_surplus)

            # Step 4: total system cost delta
            delta = pat.column_cost + residual.total_cost - baseline.total_cost

            pat.metadata["stackelberg_delta"] = round(delta, 6)
            pat.metadata["stackelberg_baseline_follower_cost"] = round(baseline.total_cost, 6)
            pat.metadata["stackelberg_residual_follower_cost"] = round(residual.total_cost, 6)
            pat.metadata["stackelberg_follower_shortage_reduction"] = round(
                baseline.shortage_reduction - residual.shortage_reduction, 6
            )
            scored.append((pat, delta))

            self.stackelberg_column_score_log.append({
                "episode": self.current_episode,
                "pattern_id": pat.pattern_id,
                "product": p,
                "period": t,
                "column_cost": pat.column_cost,
                "baseline_follower_cost": baseline.total_cost,
                "residual_follower_cost": residual.total_cost,
                "delta": delta,
                "rc": float(pat.metadata.get("reduced_cost", 0.0) or 0.0),
            })

        scored.sort(key=lambda x: x[1])
        return scored

    def _pricing_step_stackelberg_aware(
        self,
        master_solution: CGSolution,
        rc_tol: float = -1e-6,
    ) -> List[LTPattern]:
        """Follower-aware LT column pricing step.

        Generates candidate columns via the classical path (feature pruning +
        bilateral pair scoring), then scores each column by its total system
        cost impact after the follower's best-response (LT cost + shortage
        penalty + holding cost).  Only columns with strictly negative system-
        cost delta are admitted to the RMP.

        Returning an empty list when no column improves total system cost
        signals convergence to the CG loop — this is the correct stopping
        criterion for follower-aware column selection: the leader has no
        further column that the follower's recourse cannot neutralise.

        The column score is state-dependent: the same pattern has different
        value at different (product, period, inventory, shock) states because
        the follower's best-response changes with every CG iteration.
        """
        need, surplus = self._build_need_and_surplus_proxies(master_solution=master_solution)
        active_product_periods = self._compute_active_product_periods(need, surplus)

        # Generate all candidate patterns via the classical path (pruning + bilateral game)
        raw_patterns = self._candidate_patterns_from_duals(
            need=need,
            surplus=surplus,
            active_product_periods=active_product_periods,
            dual_need=master_solution.dual_need,
            dual_surplus=master_solution.dual_surplus,
            rc_tol=rc_tol,
            episode=self.current_episode,
        )

        if not raw_patterns:
            return []

        # Score by system-level follower response
        follower = self._build_stackelberg_follower_solver()
        scored = self._score_patterns_by_follower_response(
            raw_patterns, need, surplus, follower, rc_tol=rc_tol,
        )

        # Primary selection: patterns that reduce total system cost
        selected = [pat for pat, delta in scored if delta < 0]

        # Optionally cap to top-k (reuse heuristic_top_k if mode is on)
        if self.heuristic_top_k_mode and len(selected) > self.heuristic_top_k:
            selected = selected[: self.heuristic_top_k]

        # Update pricing summary
        self._last_pricing_summary.update({
            "active_product_period_count": len(active_product_periods),
            "patterns_kept_after_gnn": len(selected),
            "stackelberg_aware_scoring": True,
            "stackelberg_total_scored": len(scored),
            "stackelberg_negative_delta": sum(1 for _, d in scored if d < 0),
            "collect_teacher_mode": False,
            "runtime_gnn_mode": False,
        })
        return selected

    def _compute_final_follower_solution(
        self,
        cg_solution: CGSolution,
    ) -> Dict[Tuple, FollowerBestResponseResult]:
        """After CG converges, compute the follower's best-response on the
        final need/surplus state (reflecting all selected columns).

        This is the authoritative follower plan: it shows what the stores
        would do given the leader's final column selection and accounts for
        the minimum-shipment MOQ constraint.

        Uses the LP solver when stackelberg_exact_follower=True, otherwise
        the greedy heuristic.
        """
        need, surplus = self._build_need_and_surplus_proxies(master_solution=cg_solution)
        active_pts = self._compute_active_product_periods(need, surplus)
        follower = self._build_stackelberg_follower_solver()
        return follower.solve_all_active(need, surplus, active_pts)

    def _validate_no_better_column_after_follower(
        self,
        cg_solution: CGSolution,
        follower_solution: Dict[Tuple, FollowerBestResponseResult],
        tolerance: float = 1e-4,
    ) -> Dict[str, Any]:
        """Optional validation: check no unselected column would give a better
        total system cost after follower response.

        Returns a dict with:
          - 'passed': True if CG is Stackelberg-optimal
          - 'n_patterns_checked': number of unselected patterns examined
          - 'n_improving': number of patterns that would improve total cost
          - 'best_improving_delta': the most negative delta found (None if passed)
        """
        need, surplus = self._build_need_and_surplus_proxies(master_solution=cg_solution)
        follower = self._build_stackelberg_follower_solver()
        selected_ids = set(cg_solution.selected_patterns)
        candidates = [p for p in self.patterns if p.pattern_id not in selected_ids]

        scored = self._score_patterns_by_follower_response(
            candidates, need, surplus, follower
        )
        improving = [(pat, d) for pat, d in scored if d < -tolerance]
        return {
            "passed": len(improving) == 0,
            "n_patterns_checked": len(candidates),
            "n_improving": len(improving),
            "best_improving_delta": improving[0][1] if improving else None,
            "best_improving_pattern": improving[0][0].pattern_id if improving else None,
        }

    def pricing_step(self, master_solution: CGSolution, rc_tol: float = -1e-6) -> List[LTPattern]:
        need, surplus = self._build_need_and_surplus_proxies(master_solution=master_solution)
        active_product_periods = self._compute_active_product_periods(need, surplus)
        # Make LB/UB visible to the adaptive pruner. LB = current RMP LP value
        # (this iteration's lower bound on the LP relaxation). UB is left None
        # unless an integer incumbent has been recorded — the pruner falls back
        # to its RC-quantile rule in that case.
        self._adaptive_pruning_lb = float(master_solution.objective) if math.isfinite(
            float(master_solution.objective)
        ) else None
        if self.exact_full_mode:
            # A0 benchmark: exact Gurobi pricing, no pruning / Stackelberg / GNN / top-k.
            selected_patterns = self._candidate_patterns_exact_full(
                need=need,
                surplus=surplus,
                active_product_periods=active_product_periods,
                dual_need=master_solution.dual_need,
                dual_surplus=master_solution.dual_surplus,
                rc_tol=rc_tol,
                episode=self.current_episode,
            )
            self._last_pricing_summary["active_product_period_count"] = len(active_product_periods)
            self._last_pricing_summary["patterns_kept_after_gnn"] = len(selected_patterns)
            self._last_pricing_summary["collect_teacher_mode"] = False
            self._last_pricing_summary["runtime_gnn_mode"] = False
            return selected_patterns
        if self.stackelberg_aware_scoring:
            # Follower-aware mode: admit only columns that reduce total system
            # cost (LT + shortage + holding) after the follower's best-response.
            return self._pricing_step_stackelberg_aware(master_solution, rc_tol=rc_tol)
        new_patterns = self._candidate_patterns_from_duals(
            need=need,
            surplus=surplus,
            active_product_periods=active_product_periods,
            dual_need=master_solution.dual_need,
            dual_surplus=master_solution.dual_surplus,
            rc_tol=rc_tol,
            episode=self.current_episode,
        )
        if self.collect_teacher_mode:
            selected_patterns = self._collect_teacher_batch_without_gnn_prefilter(
                patterns=new_patterns,
                need=need,
                surplus=surplus,
                dual_need=master_solution.dual_need,
                dual_surplus=master_solution.dual_surplus,
            )
        elif self.runtime_gnn_mode:
            selected_patterns = self._select_patterns_with_gnn(
                patterns=new_patterns,
                need=need,
                surplus=surplus,
                dual_need=master_solution.dual_need,
                dual_surplus=master_solution.dual_surplus,
            )
        elif self.heuristic_top_k_mode:
            ranked = sorted(
                new_patterns,
                key=lambda p: abs(float(p.metadata.get("reduced_cost", 0.0) or 0.0)),
                reverse=True,
            )
            selected_patterns = ranked[: self.heuristic_top_k]
        else:
            selected_patterns = new_patterns
        self._last_pricing_summary["active_product_period_count"] = len(active_product_periods)
        self._last_pricing_summary["patterns_kept_after_gnn"] = len(selected_patterns)
        self._last_pricing_summary["collect_teacher_mode"] = self.collect_teacher_mode
        self._last_pricing_summary["runtime_gnn_mode"] = self.runtime_gnn_mode and not self.collect_teacher_mode
        return selected_patterns

    def run_column_generation(
        self,
        max_iter: int = 10,
        improvement_tol: float = 1e-5,
        rc_tol: float = -1e-6,
        msg: bool = False,
        stopping_mode: Optional[str] = None,
    ) -> CGSolution:
        # stopping_mode controls when the CG loop terminates:
        #   "fixed_budget" → stop at max_iter regardless of RC / improvement
        #   "convergence"  → stop ONLY when added == 0 (no new negative-RC column
        #                    was added to the RMP pool). improvement_tol is
        #                    ignored. max_iter becomes a safety cap only.
        #   "hybrid" (default) → original behavior: stop on added==0 OR
        #                    improvement <= improvement_tol OR iter > max_iter.
        # Benchmark runs should use "convergence" to avoid prematurely halting
        # the LP just because the RMP improvement stalled for one iteration.
        if stopping_mode is None:
            stopping_mode = os.environ.get("IRP_CG_STOPPING_MODE", "hybrid").strip().lower()
        if stopping_mode not in {"fixed_budget", "convergence", "hybrid"}:
            raise ValueError(
                f"Unknown stopping_mode={stopping_mode!r}; expected one of "
                "'fixed_budget', 'convergence', 'hybrid'."
            )
        self._cg_stopping_mode = stopping_mode
        best_sol = self.solve_rmp(msg=msg)
        best_sol.iterations_run = 0
        rmp_metrics_total = dict(best_sol.efficiency_metrics)
        rmp_metrics_total["rmp_solves"] = 1.0
        best_sol.efficiency_metrics = dict(rmp_metrics_total)
        prev_obj = best_sol.objective
        self.current_episode = 0
        self.cg_history = [{
            "episode": 0,
            "total_cost": float(best_sol.objective),
            "improvement": 0.0,
            "proposed_columns": 0,
            "added_columns": 0,
            "selected_patterns": len(best_sol.selected_patterns),
        }]

        _qprint("\n[CG] Active (product, period) pairs with positive need and surplus:")
        if not best_sol.active_product_periods:
            _qprint("  None. RMP is not activated for any product-period.")
            self.cg_episode_diagnostics.append({
                "episode": 0,
                "active_product_period_count": 0,
                "candidate_pairs_before_pruning": 0,
                "pairs_after_pruning": 0,
                "pairs_accepted_stackelberg": 0,
                "pairs_recovered_stackelberg_fallback": 0,
                "patterns_built_before_dedup": 0,
                "patterns_deduplicated_before_gnn": 0,
                "patterns_built_before_gnn": 0,
                "patterns_kept_after_gnn": 0,
                "duplicate_id_rejects": 0,
                "duplicate_signature_rejects": 0,
                "empty_flow_rejects": 0,
                "patterns_added_to_pool": 0,
                "selected_patterns": len(best_sol.selected_patterns),
                "objective": float(best_sol.objective),
                "improvement": 0.0,
            })
            self._print_cg_episode_history()
            best_sol.efficiency_metrics = dict(rmp_metrics_total)
            return best_sol
        for p, t in sorted(best_sol.active_product_periods):
            _qprint(f"  product={p} | period={t}")

        _qprint("\n[RMP] Initially selected LT patterns:")
        if not best_sol.selected_patterns:
            _qprint("  None")
        elif not _is_quiet():
            for pat_id in best_sol.selected_patterns:
                pat = next(p for p in self.patterns if p.pattern_id == pat_id)
                print("  " + format_pattern_detail(pat) + f" | lambda={best_sol.lambda_values[pat_id]:.4f}")

        for it in range(1, max_iter + 1):
            self.current_episode = it
            new_patterns = self.pricing_step(best_sol, rc_tol=rc_tol)
            added = self.add_patterns(new_patterns)
            self._mark_teacher_rows_passed_to_rmp(it)
            episode_summary = dict(self._last_pricing_summary)
            episode_summary.update({
                "episode": it,
                "duplicate_id_rejects": self.last_duplicate_rejects,
                "duplicate_signature_rejects": self.last_signature_rejects,
                "empty_flow_rejects": self.last_empty_rejects,
                "patterns_added_to_pool": added,
                "selected_patterns": len(best_sol.selected_patterns),
                "objective": float(best_sol.objective),
                "improvement": 0.0,
            })

            if _is_quiet():
                # One concise progress line per iter — safe to leave in notebook.
                print(f"[Pricing] iter={it} proposed={len(new_patterns)} added={added}")
            else:
                print(f"\n[Pricing] Iter {it}: proposed={len(new_patterns)}, added={added}")
                for pat in new_patterns[:10]:
                    print(
                        "  " + format_pattern_detail(pat)
                        + f" | feature={pat.metadata.get('feature_name')}"
                        + f" | rc={pat.metadata.get('reduced_cost')}"
                        + f" | mean_acceptance={pat.metadata.get('mean_acceptance_score')}"
                        + f" | mean_comp={pat.metadata.get('mean_compensation')}"
                        + f" | gnn_score={pat.metadata.get('gnn_score')}"
                    )

            if added == 0 and stopping_mode != "fixed_budget":
                best_sol.iterations_run = it - 1
                best_sol.efficiency_metrics = dict(rmp_metrics_total)
                self.cg_history.append({
                    "episode": it,
                    "total_cost": float(best_sol.objective),
                    "improvement": 0.0,
                    "proposed_columns": len(new_patterns),
                    "added_columns": added,
                    "selected_patterns": len(best_sol.selected_patterns),
                })
                self.cg_episode_diagnostics.append(episode_summary)
                _flush_cg_partials(self.cg_history, self.cg_episode_diagnostics)
                if new_patterns:
                    print(
                        f"[CG] {len(new_patterns)} negative-RC column(s) priced but all already in pool "
                        f"(degenerate cycling). Stop. [stopping_mode={stopping_mode}]"
                    )
                else:
                    print(f"[CG] No negative reduced-cost columns found. LP optimal. Stop. [stopping_mode={stopping_mode}]")
                self._print_cg_episode_history()
                best_sol.stackelberg_column_scores = list(self.stackelberg_column_score_log)
                if self.stackelberg_aware_scoring:
                    best_sol.follower_solution = self._compute_final_follower_solution(best_sol)
                    best_sol.stackelberg_validation = self._validate_no_better_column_after_follower(
                        best_sol, best_sol.follower_solution
                    )
                return best_sol
            if added == 0 and stopping_mode == "fixed_budget":
                # Log the would-be-convergence event but continue until max_iter
                # so benchmark runs share a fixed iteration budget regardless of
                # when LP optimality is reached.
                print(f"[CG] added==0 at iter {it} but stopping_mode=fixed_budget — continuing until max_iter={max_iter}.")

            sol = self.solve_rmp(msg=msg)
            sol.iterations_run = it
            _add_efficiency_metrics(rmp_metrics_total, sol.efficiency_metrics)
            rmp_metrics_total["rmp_solves"] = float(rmp_metrics_total.get("rmp_solves", 0.0)) + 1.0
            sol.efficiency_metrics = dict(rmp_metrics_total)
            improvement = prev_obj - sol.objective
            self._annotate_teacher_rows_after_reopt(it, sol, improvement)
            episode_summary["selected_patterns"] = len(sol.selected_patterns)
            episode_summary["objective"] = float(sol.objective)
            episode_summary["improvement"] = float(improvement)
            self.cg_episode_diagnostics.append(episode_summary)
            self.cg_history.append({
                "episode": it,
                "total_cost": float(sol.objective),
                "improvement": float(improvement),
                "proposed_columns": len(new_patterns),
                "added_columns": added,
                "selected_patterns": len(sol.selected_patterns),
            })
            _flush_cg_partials(self.cg_history, self.cg_episode_diagnostics)
            if _is_quiet():
                print(f"[CG] iter={it} obj={sol.objective:.4f} delta={improvement:.4f} selected={len(sol.selected_patterns)}")
            else:
                print(f"[CG] Iter {it}: objective = {sol.objective:.6f}, improvement = {improvement:.6f}")
                print("[RMP] Selected LT patterns after re-optimization:")
                if not sol.selected_patterns:
                    print("  None")
                else:
                    for pat_id in sol.selected_patterns:
                        pat = next(p for p in self.patterns if p.pattern_id == pat_id)
                        print("  " + format_pattern_detail(pat) + f" | lambda={sol.lambda_values[pat_id]:.4f}")

            if stopping_mode == "hybrid" and improvement <= improvement_tol:
                print(f"[CG] Improvement {improvement:.2e} <= tol {improvement_tol:.2e}. Converged. [stopping_mode=hybrid]")
                self._print_cg_episode_history()
                sol.efficiency_metrics = dict(rmp_metrics_total)
                sol.stackelberg_column_scores = list(self.stackelberg_column_score_log)
                if self.stackelberg_aware_scoring:
                    sol.follower_solution = self._compute_final_follower_solution(sol)
                    sol.stackelberg_validation = self._validate_no_better_column_after_follower(
                        sol, sol.follower_solution
                    )
                return sol
            # In "convergence" and "fixed_budget" modes we do NOT stop on
            # improvement_tol — the only convergence signal is added==0 (handled
            # above). This keeps strict benchmark runs from halting early when
            # the RMP objective happens to flatten but negative-RC columns still
            # exist, which would bias the runtime comparison.
            prev_obj = sol.objective
            best_sol = sol

        self._print_cg_episode_history()
        best_sol.efficiency_metrics = dict(rmp_metrics_total)
        best_sol.stackelberg_column_scores = list(self.stackelberg_column_score_log)
        if self.stackelberg_aware_scoring:
            best_sol.follower_solution = self._compute_final_follower_solution(best_sol)
            best_sol.stackelberg_validation = self._validate_no_better_column_after_follower(
                best_sol, best_sol.follower_solution
            )
        return best_sol

    @staticmethod
    def _fractional_lambda_values(sol: CGSolution, int_tol: float = 1e-5) -> List[Tuple[str, float]]:
        fractional = []
        for pid, value in sol.lambda_values.items():
            value = float(value)
            if int_tol < value < 1.0 - int_tol:
                fractional.append((pid, value))
        fractional.sort(key=lambda item: abs(item[1] - 0.5))
        return fractional

    def run_branch_and_price(
        self,
        max_iter: int = 10,
        improvement_tol: float = 1e-5,
        rc_tol: float = -1e-6,
        msg: bool = False,
        max_nodes: int = 15,
        max_depth: int = 6,
        int_tol: float = 1e-5,
        stopping_mode: Optional[str] = None,
    ) -> CGSolution:
        original_branch_bounds = dict(self.branch_bounds)
        self.branch_history = []
        self.cg_history_all_nodes = []
        best_integer_sol: Optional[CGSolution] = None
        best_relaxation_sol: Optional[CGSolution] = None
        best_bound = math.inf
        nodes_explored = 0
        nodes_pruned_by_bound = 0
        nodes_pruned_by_integrality = 0
        nodes_pruned_by_depth = 0
        nodes_infeasible = 0
        pending_nodes: List[Dict[str, Any]] = [{
            "node_id": 0,
            "parent_id": None,
            "depth": 0,
            "branch_var": "",
            "branch_sense": "root",
            "branch_value": None,
            "bounds": dict(original_branch_bounds),
        }]
        next_node_id = 1

        print("\n" + "=" * 80)
        print("STEP 3B - Branch-and-price on fractional RMP columns")
        print("=" * 80)

        try:
            while pending_nodes and nodes_explored < max_nodes:
                node = pending_nodes.pop()
                self.branch_bounds = dict(node["bounds"])
                self.current_branch_node_id = int(node["node_id"])
                nodes_explored += 1
                print(
                    f"\n[B&P] Node {node['node_id']} depth={node['depth']} "
                    f"| branch={node['branch_var'] or 'root'} {node['branch_sense']}"
                )
                sol = self.run_column_generation(
                    max_iter=max_iter,
                    improvement_tol=improvement_tol,
                    rc_tol=rc_tol,
                    msg=msg,
                    stopping_mode=stopping_mode,
                )
                for row in self.cg_history:
                    row_with_node = dict(row)
                    row_with_node["branch_node_id"] = node["node_id"]
                    row_with_node["branch_depth"] = node["depth"]
                    row_with_node["branch_sense_from_parent"] = node["branch_sense"]
                    row_with_node["branch_var_from_parent"] = node["branch_var"]
                    self.cg_history_all_nodes.append(row_with_node)
                fractional = self._fractional_lambda_values(sol, int_tol=int_tol)
                if math.isfinite(sol.objective):
                    best_bound = min(best_bound, float(sol.objective))
                    if best_relaxation_sol is None or sol.objective < best_relaxation_sol.objective:
                        best_relaxation_sol = sol

                branch_pid = fractional[0][0] if fractional else ""
                branch_value = fractional[0][1] if fractional else None
                node_status = "open"

                if not math.isfinite(sol.objective) or sol.status in {"Infeasible", "InfOrUnbd", "Unbounded"}:
                    nodes_infeasible += 1
                    node_status = "infeasible"
                elif best_integer_sol is not None and sol.objective >= best_integer_sol.objective - improvement_tol:
                    nodes_pruned_by_bound += 1
                    node_status = "pruned_by_bound"
                elif not fractional:
                    nodes_pruned_by_integrality += 1
                    node_status = "integer_incumbent"
                    if best_integer_sol is None or sol.objective < best_integer_sol.objective:
                        best_integer_sol = sol
                elif node["depth"] >= max_depth:
                    nodes_pruned_by_depth += 1
                    node_status = "pruned_by_depth"
                else:
                    left_bounds = dict(node["bounds"])
                    left_bounds[branch_pid] = (0.0, 0.0)
                    right_bounds = dict(node["bounds"])
                    right_bounds[branch_pid] = (1.0, 1.0)
                    pending_nodes.append({
                        "node_id": next_node_id,
                        "parent_id": node["node_id"],
                        "depth": node["depth"] + 1,
                        "branch_var": branch_pid,
                        "branch_sense": "<= 0",
                        "branch_value": 0.0,
                        "bounds": left_bounds,
                    })
                    next_node_id += 1
                    pending_nodes.append({
                        "node_id": next_node_id,
                        "parent_id": node["node_id"],
                        "depth": node["depth"] + 1,
                        "branch_var": branch_pid,
                        "branch_sense": ">= 1",
                        "branch_value": 1.0,
                        "bounds": right_bounds,
                    })
                    next_node_id += 1
                    node_status = "branched"

                self.branch_history.append({
                    "node_id": node["node_id"],
                    "parent_id": node["parent_id"],
                    "depth": node["depth"],
                    "status": node_status,
                    "objective": float(sol.objective) if math.isfinite(sol.objective) else math.inf,
                    "fractional_lambda_count": len(fractional),
                    "branch_var": branch_pid,
                    "branch_lambda_value": branch_value,
                    "branch_sense_from_parent": node["branch_sense"],
                    "branch_value_from_parent": node["branch_value"],
                    "incumbent_objective": (
                        float(best_integer_sol.objective)
                        if best_integer_sol is not None and math.isfinite(best_integer_sol.objective)
                        else math.inf
                    ),
                })
                print(
                    f"[B&P] Node {node['node_id']} status={node_status} "
                    f"| obj={sol.objective:.6f} | fractional_lambdas={len(fractional)}"
                )
                if branch_pid and node_status == "branched":
                    print(f"[B&P] Branch on lambda[{branch_pid}] = {branch_value:.6f}")

            chosen_sol = best_integer_sol or best_relaxation_sol
            if chosen_sol is None:
                chosen_sol = CGSolution(
                    status="Infeasible",
                    objective=math.inf,
                    lambda_values={},
                    selected_patterns=[],
                    implied_net_lt={},
                )
            gap = 0.0
            if best_integer_sol is not None and math.isfinite(best_bound):
                gap = max(0.0, (best_integer_sol.objective - best_bound) / max(abs(best_integer_sol.objective), 1e-9))
            chosen_sol.branch_summary = {
                "status": "integer_incumbent" if best_integer_sol is not None else "no_integer_incumbent_returned_best_relaxation",
                "nodes_explored": nodes_explored,
                "nodes_remaining": len(pending_nodes),
                "nodes_pruned_by_bound": nodes_pruned_by_bound,
                "nodes_pruned_by_integrality": nodes_pruned_by_integrality,
                "nodes_pruned_by_depth": nodes_pruned_by_depth,
                "nodes_infeasible": nodes_infeasible,
                "best_bound": best_bound,
                "incumbent_objective": best_integer_sol.objective if best_integer_sol is not None else math.inf,
                "relative_gap": gap,
                "max_nodes": max_nodes,
                "max_depth": max_depth,
                "integer_incumbent_found": best_integer_sol is not None,
            }
            chosen_sol.efficiency_metrics["branch_price_nodes_explored"] = float(nodes_explored)
            chosen_sol.efficiency_metrics["branch_price_nodes_remaining"] = float(len(pending_nodes))
            print("\n[B&P Summary]")
            pprint.pprint(chosen_sol.branch_summary)
            return chosen_sol
        finally:
            self.current_branch_node_id = None
            self.branch_bounds = original_branch_bounds

    def _print_cg_episode_history(self) -> None:
        if _is_quiet():
            if not self.cg_history:
                print("[CG Episode History] (empty)")
                return
            last = self.cg_history[-1]
            print(
                f"[CG Episode History] episodes={len(self.cg_history)} "
                f"final_cost={float(last['total_cost']):.4f} "
                f"final_selected={int(last['selected_patterns'])}"
            )
            return
        print("\n[CG Total Cost By Episode]")
        if not self.cg_history:
            print("  No CG episode history recorded.")
            return
        for row in self.cg_history:
            print(
                f"  episode={int(row['episode']):03d} "
                f"| total_cost={float(row['total_cost']):.6f} "
                f"| improvement={float(row['improvement']):.6f} "
                f"| proposed={int(row['proposed_columns'])} "
                f"| added={int(row['added_columns'])} "
                f"| selected_patterns={int(row['selected_patterns'])}"
            )


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def build_predicted_inventory_df(solution) -> pd.DataFrame:
    """Emit per-(store, sku, period) predicted end-of-period inventory.

    Prefers `realized_inventory_after_lt` (baseline + post-shock + LT recourse)
    when the pipeline has attached it; otherwise falls back to the forecast-plan
    `inv_store`. The realized path is what should be compared against dataset
    `actual_end_qty`, since that is the quantity that actually lands on shelves.
    """
    rows = []
    inventory_source = None
    inventory_tag = "inv_store_forecast"
    realized = getattr(solution, "realized_inventory_after_lt", None)
    if realized:
        inventory_source = realized
        inventory_tag = "realized_after_lt"
    elif hasattr(solution, "inv_store"):
        inventory_source = solution.inv_store
    else:
        raise ValueError("Solution object does not contain inv_store")
    for (s, p, t), inv in inventory_source.items():
        rows.append({
            "store": s,
            "sku": p,
            "period": t,
            "predicted_end_qty": float(inv),
            "inventory_source": inventory_tag,
        })
    return pd.DataFrame(rows)


def compute_realized_inventory_after_lt(
    data: IRPData,
    dc_solution: FullIRPTSolution,
    lt_plan_df: Optional[pd.DataFrame] = None,
) -> Dict[Tuple[Store, Product, Period], float]:
    """Replay baseline + post-shock state, then net LT flows per (store, sku, period).

    Returns the same dict that `build_realized_operating_cost_breakdown` uses
    internally, so the validation comparison sees the exact inventory trajectory
    that produced the realized operating cost.
    """
    lt_net: Dict[Tuple[Store, Product, Period], float] = {
        (s, p, t): 0.0
        for s in data.stores
        for p in data.products
        for t in data.periods
    }
    if lt_plan_df is not None and not lt_plan_df.empty:
        for _, row in lt_plan_df.iterrows():
            p = str(row["sku"])
            t = int(row["period"])
            i = str(row["from_store"])
            j = str(row["to_store"])
            qty = float(row["lt_qty"])
            if (i, p, t) in lt_net:
                lt_net[(i, p, t)] -= qty
            if (j, p, t) in lt_net:
                lt_net[(j, p, t)] += qty
    realized: Dict[Tuple[Store, Product, Period], float] = {}
    for s in data.stores:
        for p in data.products:
            for t in data.periods:
                post_shock_inv = float(
                    data.post_shock_inventory.get((s, p, t), dc_solution.inv_store.get((s, p, t), 0.0))
                )
                realized[(s, p, t)] = max(0.0, post_shock_inv + lt_net.get((s, p, t), 0.0))
    return realized


def build_forecast_fulfillment_df(data: IRPData, solution: FullIRPTSolution) -> pd.DataFrame:
    """Pre-shock baseline fulfillment using forecast demand. Almost always 100%
    because the baseline is planned against forecast demand with ample capacity —
    included for comparison against the post-shock realization only."""
    rows = []
    for s in data.stores:
        for t in data.periods:
            total_demand = sum(float(data.demand.get((s, p, t), 0.0)) for p in data.products)
            total_shortage = sum(float(solution.shortage.get((s, p, t), 0.0)) for p in data.products)
            fulfilled_demand = max(0.0, total_demand - total_shortage)
            fulfillment_rate = fulfilled_demand / total_demand if total_demand > 1e-9 else 1.0
            rows.append({
                "store": s,
                "period": t,
                "forecast_demand": round(total_demand, 6),
                "forecast_fulfilled_demand": round(fulfilled_demand, 6),
                "forecast_shortage": round(total_shortage, 6),
                "forecast_fulfillment_rate": round(fulfillment_rate, 6),
            })
    return pd.DataFrame(rows)


# Back-compat alias. Do not remove — keeps existing callers working while making
# the pre-shock semantics explicit in code that's been updated.
build_demand_fulfillment_df = build_forecast_fulfillment_df


def build_post_shock_fulfillment_df(data: IRPData) -> pd.DataFrame:
    rows = []
    for s in data.stores:
        for t in data.periods:
            total_demand = sum(float(data.realized_demand.get((s, p, t), 0.0)) for p in data.products)
            total_shortage = sum(float(data.post_shock_shortage.get((s, p, t), 0.0)) for p in data.products)
            fulfilled_demand = max(0.0, total_demand - total_shortage)
            fulfillment_rate = fulfilled_demand / total_demand if total_demand > 1e-9 else 1.0
            rows.append({
                "store": s,
                "period": t,
                "total_realized_demand": round(total_demand, 6),
                "fulfilled_demand": round(fulfilled_demand, 6),
                "post_shock_shortage": round(total_shortage, 6),
                "post_shock_fulfillment_rate": round(fulfillment_rate, 6),
            })
    return pd.DataFrame(rows)


def print_demand_fulfillment(fulfillment_df: pd.DataFrame) -> None:
    print("\n[Demand Fulfillment Rate By Store-Period]")
    if fulfillment_df.empty:
        print("  No demand fulfillment rows available.")
        return
    for _, row in fulfillment_df.sort_values(["period", "store"]).iterrows():
        print(
            f"  period={row['period']} | store={row['store']} "
            f"| demand={row['total_demand']:.6f} "
            f"| fulfilled={row['fulfilled_demand']:.6f} "
            f"| shortage={row['shortage']:.6f} "
            f"| fulfillment_rate={row['demand_fulfillment_rate']:.2%}"
        )


def print_post_shock_fulfillment(fulfillment_df: pd.DataFrame) -> None:
    if _is_quiet():
        if fulfillment_df.empty:
            print("[Post-Shock Demand Fulfillment] (empty)")
            return
        mean_rate = float(fulfillment_df["post_shock_fulfillment_rate"].mean())
        print(f"[Post-Shock Demand Fulfillment] rows={len(fulfillment_df)} mean_rate={mean_rate:.4f}")
        return
    print("\n[Post-Shock Demand Fulfillment Rate By Store-Period]")
    if fulfillment_df.empty:
        print("  No post-shock demand fulfillment rows available.")
        return
    for _, row in fulfillment_df.sort_values(["period", "store"]).iterrows():
        print(
            f"  period={row['period']} | store={row['store']} "
            f"| realized_demand={row['total_realized_demand']:.6f} "
            f"| fulfilled={row['fulfilled_demand']:.6f} "
            f"| post_shock_shortage={row['post_shock_shortage']:.6f} "
            f"| fulfillment_rate={row['post_shock_fulfillment_rate']:.2%}"
        )


def build_full_irpt_cost_breakdown(data: IRPData, solution: FullIRPTSolution) -> Dict[str, float]:
    direct_cw_unit_cost = sum(
        data.ship_cost_cw[(s, p)] * float(solution.direct_ship_q.get((s, p, t), 0.0))
        for s in data.stores
        for p in data.products
        for t in data.periods
    )
    store_holding_cost = sum(
        data.holding_cost_store[(s, p)] * float(solution.inv_store.get((s, p, t), 0.0))
        for s in data.stores
        for p in data.products
        for t in data.periods
    )
    warehouse_holding_cost = sum(
        data.holding_cost_wh[p] * float(solution.inv_wh.get((p, t), 0.0))
        for p in data.products
        for t in data.periods
    )
    route_distance_cost = sum(
        data.alpha * data.distance[(i, j)] * float(solution.x.get((i, j, v, t), 0.0))
        for i in [data.warehouse] + data.stores
        for j in [data.warehouse] + data.stores
        if i != j
        for v in data.vehicles
        for t in data.periods
    )
    vehicle_fixed_cost = sum(
        data.vehicle_fixed_cost * float(solution.u.get((v, t), 0.0))
        for v in data.vehicles
        for t in data.periods
    )
    lateral_transshipment_cost = sum(
        data.transship_unit_cost[(i, j)] * float(solution.y.get((i, j, p, v, t), 0.0))
        for i in data.stores
        for j in data.stores
        if i != j
        for p in data.products
        for v in data.vehicles
        for t in data.periods
    )
    shortage_cost = sum(
        data.shortage_cost[(s, p)] * float(solution.shortage.get((s, p, t), 0.0))
        for s in data.stores
        for p in data.products
        for t in data.periods
    )
    objective_recomputed = (
        direct_cw_unit_cost
        + store_holding_cost
        + warehouse_holding_cost
        + route_distance_cost
        + vehicle_fixed_cost
        + lateral_transshipment_cost
        + shortage_cost
    )
    return {
        "direct_cw_unit_cost": round(direct_cw_unit_cost, 6),
        "store_holding_cost": round(store_holding_cost, 6),
        "warehouse_holding_cost": round(warehouse_holding_cost, 6),
        "route_distance_cost": round(route_distance_cost, 6),
        "vehicle_fixed_cost": round(vehicle_fixed_cost, 6),
        "lateral_transshipment_cost": round(lateral_transshipment_cost, 6),
        "shortage_cost": round(shortage_cost, 6),
        "objective_recomputed": round(objective_recomputed, 6),
        "gurobi_objective": round(float(solution.objective), 6),
    }


def build_realized_operating_cost_breakdown(
    data: IRPData,
    dc_solution: FullIRPTSolution,
    lt_plan_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Evaluate executed DC plan under realized demand and optional LT recourse."""
    direct_cw_unit_cost = sum(
        data.ship_cost_cw[(s, p)] * float(dc_solution.direct_ship_q.get((s, p, t), 0.0))
        for s in data.stores
        for p in data.products
        for t in data.periods
    )
    warehouse_holding_cost = sum(
        data.holding_cost_wh[p] * float(dc_solution.inv_wh.get((p, t), 0.0))
        for p in data.products
        for t in data.periods
    )
    route_distance_cost = sum(
        data.alpha * data.distance[(i, j)] * float(dc_solution.x.get((i, j, v, t), 0.0))
        for i in [data.warehouse] + data.stores
        for j in [data.warehouse] + data.stores
        if i != j
        for v in data.vehicles
        for t in data.periods
    )
    vehicle_fixed_cost = sum(
        data.vehicle_fixed_cost * float(dc_solution.u.get((v, t), 0.0))
        for v in data.vehicles
        for t in data.periods
    )

    lt_net: Dict[Tuple[Store, Product, Period], float] = {
        (s, p, t): 0.0
        for s in data.stores
        for p in data.products
        for t in data.periods
    }
    lateral_transshipment_cost = 0.0
    if lt_plan_df is not None and not lt_plan_df.empty:
        lateral_transshipment_cost = float(pd.to_numeric(lt_plan_df["lt_total_cost"], errors="coerce").fillna(0.0).sum())
        for _, row in lt_plan_df.iterrows():
            p = str(row["sku"])
            t = int(row["period"])
            i = str(row["from_store"])
            j = str(row["to_store"])
            qty = float(row["lt_qty"])
            if (i, p, t) in lt_net:
                lt_net[(i, p, t)] -= qty
            if (j, p, t) in lt_net:
                lt_net[(j, p, t)] += qty

    realized_inventory_after_lt: Dict[Tuple[Store, Product, Period], float] = {}
    realized_shortage_after_lt: Dict[Tuple[Store, Product, Period], float] = {}
    for s in data.stores:
        for p in data.products:
            for t in data.periods:
                post_shock_inv = float(data.post_shock_inventory.get((s, p, t), dc_solution.inv_store.get((s, p, t), 0.0)))
                post_shock_shortage = float(data.post_shock_shortage.get((s, p, t), dc_solution.shortage.get((s, p, t), 0.0)))
                adjusted = post_shock_inv + lt_net.get((s, p, t), 0.0)
                realized_inventory_after_lt[(s, p, t)] = max(0.0, adjusted)
                realized_shortage_after_lt[(s, p, t)] = max(0.0, post_shock_shortage - max(0.0, lt_net.get((s, p, t), 0.0)))
                if adjusted < 0.0:
                    realized_shortage_after_lt[(s, p, t)] += -adjusted

    store_holding_cost = sum(
        data.holding_cost_store[(s, p)] * inv
        for (s, p, _), inv in realized_inventory_after_lt.items()
    )
    shortage_cost = sum(
        data.shortage_cost[(s, p)] * shortage
        for (s, p, _), shortage in realized_shortage_after_lt.items()
    )
    realized_operating_cost = (
        direct_cw_unit_cost
        + store_holding_cost
        + warehouse_holding_cost
        + route_distance_cost
        + vehicle_fixed_cost
        + lateral_transshipment_cost
        + shortage_cost
    )
    return {
        "direct_cw_unit_cost_executed_plan": round(direct_cw_unit_cost, 6),
        "store_holding_cost_realized": round(store_holding_cost, 6),
        "warehouse_holding_cost_executed_plan": round(warehouse_holding_cost, 6),
        "route_distance_cost_executed_plan": round(route_distance_cost, 6),
        "vehicle_fixed_cost_executed_plan": round(vehicle_fixed_cost, 6),
        "lateral_transshipment_cost_realized": round(lateral_transshipment_cost, 6),
        "shortage_cost_realized": round(shortage_cost, 6),
        "total_realized_operating_cost": round(realized_operating_cost, 6),
        "total_realized_shortage_units": round(sum(realized_shortage_after_lt.values()), 6),
        "total_realized_store_inventory_units": round(sum(realized_inventory_after_lt.values()), 6),
    }


def print_cost_breakdown(title: str, cost_breakdown: Dict[str, float]) -> None:
    if _is_quiet():
        total = sum(float(v) for v in cost_breakdown.values())
        print(f"[{title}] total={total:.4f} keys={len(cost_breakdown)}")
        return
    print(f"\n[{title}]")
    for key, value in cost_breakdown.items():
        print(f"  {key}: {float(value):.6f}")


LT_PLAN_COLUMNS = [
    "source",
    "period",
    "vehicle",
    "from_store",
    "to_store",
    "sku",
    "lt_qty",
    "lt_unit_cost",
    "lt_fixed_cost",
    "lt_total_cost",
    "pattern_id",
    "lambda_value",
]


def build_lt_plan_df_from_solution(
    solution: FullIRPTSolution,
    data: IRPData,
    source: str = "FullIRPT",
) -> pd.DataFrame:
    rows = []
    for (i, j, p, v, t), qty in sorted(solution.y.items(), key=lambda item: (item[0][4], item[0][3], item[0][0], item[0][1], item[0][2])):
        if qty <= 1e-9:
            continue
        unit_cost = float(data.transship_unit_cost.get((i, j), data.ship_cost_lt.get((i, j, p), 0.0)))
        rows.append({
            "source": source,
            "period": t,
            "vehicle": v,
            "from_store": i,
            "to_store": j,
            "sku": p,
            "lt_qty": round(float(qty), 6),
            "lt_unit_cost": round(unit_cost, 6),
            "lt_fixed_cost": 0.0,
            "lt_total_cost": round(unit_cost * float(qty), 6),
            "pattern_id": "",
            "lambda_value": 1.0,
        })
    return pd.DataFrame(rows, columns=LT_PLAN_COLUMNS)


def build_lt_plan_df_from_cg(cg_solution: CGSolution, patterns: List[LTPattern], data: IRPData) -> pd.DataFrame:
    """Aggregate the LP-relaxed CG solution into an integer LT plan.

    Two post-processing steps are always applied to the raw fractional flows
    Σ_p λ_p · pattern_flows[i,j]:

      1. **Aggregate by (period, sku, from, to)** so multiple patterns that
         touch the same arc collapse into one shipment row.
      2. **Round + threshold**: each aggregated qty is rounded to integer
         (default ON via IRP_INTEGER_FINAL_OUTPUTS) and dropped if it falls
         below `IRP_LT_MIN_UNITS` (default 5). This MOQ filter eliminates LP
         relaxation artefacts — fractional λ values that aggregate to 1-2 units
         after rounding — while keeping all economically meaningful shipments.
         Set IRP_LT_MIN_UNITS=1 to disable the filter.
    """
    pattern_by_id = {pat.pattern_id: pat for pat in patterns}
    integer_outputs = _integer_final_outputs_enabled()
    try:
        min_units = float(os.environ.get("IRP_LT_MIN_UNITS", "5"))
    except ValueError:
        min_units = 1.0
    if min_units < 0:
        min_units = 0.0

    # Step 1 — aggregate fractional flows per arc.
    agg: Dict[Tuple[int, str, int, int], Dict[str, Any]] = {}
    for pat_id in sorted(cg_solution.selected_patterns):
        pat = pattern_by_id.get(pat_id)
        if pat is None:
            continue
        lam = float(cg_solution.lambda_values.get(pat_id, 0.0))
        if lam <= 1e-9:
            continue
        for i, j in sorted(pat.pattern_flows):
            raw_qty = float(pat.pattern_flows[(i, j)]) * lam
            if raw_qty <= 1e-9:
                continue
            key = (pat.period, pat.product, i, j)
            unit_cost = float(data.ship_cost_lt.get((i, j, pat.product), data.transship_unit_cost.get((i, j), 0.0)))
            fixed_share = float(data.fixed_dispatch_lt.get((i, j), 0.0)) * lam
            cell = agg.setdefault(key, {
                "qty": 0.0, "unit_cost": unit_cost, "fixed_cost": 0.0,
                "pattern_ids": [], "lambda_total": 0.0,
            })
            cell["qty"] += raw_qty
            cell["fixed_cost"] += fixed_share
            cell["pattern_ids"].append(pat.pattern_id)
            cell["lambda_total"] += lam

    # Step 2 — round + threshold.
    rows: List[Dict[str, Any]] = []
    for (period, sku, i, j), cell in agg.items():
        qty = cell["qty"]
        if integer_outputs:
            qty = float(round(qty))
        if qty < min_units:
            continue
        unit_cost = cell["unit_cost"]
        # Fixed cost: only paid if at least one unit ships on this arc/period.
        fixed_cost = cell["fixed_cost"] if qty > 0 else 0.0
        rows.append({
            "source": "CG_LT",
            "period": period,
            "vehicle": "",
            "from_store": i,
            "to_store": j,
            "sku": sku,
            "lt_qty": int(qty) if integer_outputs else round(qty, 6),
            "lt_unit_cost": round(unit_cost, 6),
            "lt_fixed_cost": round(fixed_cost, 6),
            "lt_total_cost": round(unit_cost * qty + fixed_cost, 6),
            "pattern_id": ",".join(str(pid) for pid in cell["pattern_ids"]),
            "lambda_value": round(min(1.0, cell["lambda_total"]), 6),
        })
    return pd.DataFrame(rows, columns=LT_PLAN_COLUMNS)


def print_lt_plan(lt_plan_df: pd.DataFrame, title: str = "Lateral Transshipment Plan") -> None:
    if _is_quiet():
        if lt_plan_df.empty:
            print(f"[{title}] (empty)")
            return
        total_cost = float(lt_plan_df["lt_total_cost"].sum()) if "lt_total_cost" in lt_plan_df else 0.0
        print(f"[{title}] rows={len(lt_plan_df)} total_cost={total_cost:.4f}")
        return
    print(f"\n[{title}]")
    if lt_plan_df.empty:
        print("  No lateral transshipment moves selected.")
        return
    for _, row in lt_plan_df.sort_values(["period", "sku", "from_store", "to_store", "vehicle", "pattern_id"]).iterrows():
        vehicle = row["vehicle"] if str(row["vehicle"]) else "-"
        print(
            f"  source={row['source']} | period={row['period']} | vehicle={vehicle} "
            f"| {row['from_store']} -> {row['to_store']} | sku={row['sku']} "
            f"| qty={row['lt_qty']:.6f} | unit_cost={row['lt_unit_cost']:.6f} "
            f"| fixed_cost={row['lt_fixed_cost']:.6f} | total_cost={row['lt_total_cost']:.6f} "
            f"| pattern={row['pattern_id']} | lambda={row['lambda_value']:.6f}"
        )


def _mpl_agg():
    """Return (matplotlib, pyplot) with Agg backend, or (None, None) if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except Exception:
        return None, None


def _save_fig(plt, path: str, tight: bool = True) -> str:
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def save_pipeline_charts(
    results: Dict[str, Any],
    out_dir: Path,
    refreshed_gnn_history: Optional[List[Dict[str, Any]]] = None,
    phase_comparison_df: Optional[pd.DataFrame] = None,
    phase_label: Optional[str] = None,
) -> List[str]:
    """Generate all pipeline visualisation charts and write them to *out_dir*.

    Chart filenames are suffixed with *phase_label* (default: "phase1") so
    multiple pipeline phases can share the same charts/ directory without
    clobbering each other. Returns a list of paths that were actually written.
    """
    mpl, plt = _mpl_agg()
    if plt is None:
        print("[Charts] matplotlib not available; skipping all charts.")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = phase_label or "phase1"

    def _chart(name: str) -> str:
        stem, _, ext = name.rpartition(".")
        return str(out_dir / (f"{stem}_{suffix}.{ext}" if ext else f"{name}_{suffix}"))

    saved: List[str] = []

    # ------------------------------------------------------------------
    # 1. GNN Training Loss Curve
    # ------------------------------------------------------------------
    gnn_history = refreshed_gnn_history or results.get("gnn_training_history") or []
    if gnn_history:
        try:
            epochs = [int(r.get("epoch", r.get("episode", i))) for i, r in enumerate(gnn_history)]
            train_loss = [float(r.get("train_loss", float("nan"))) for r in gnn_history]
            valid_loss = [float(r.get("valid_loss", float("nan"))) for r in gnn_history]
            valid_f1   = [float(r.get("valid_f1",   float("nan"))) for r in gnn_history]
            valid_top1 = [float(r.get("valid_top1", float("nan"))) for r in gnn_history]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            ax1, ax2 = axes

            ax1.plot(epochs, train_loss, marker="o", label="Train Loss", linewidth=2)
            ax1.plot(epochs, valid_loss, marker="s", linestyle="--", label="Valid Loss", linewidth=2)
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("GNN Training Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(epochs, valid_f1,   marker="^", label="Valid F1",   linewidth=2, color="green")
            ax2.plot(epochs, valid_top1, marker="D", linestyle="--", label="Valid Top-1", linewidth=2, color="darkorange")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Score")
            ax2.set_title("GNN Validation Metrics")
            ax2.set_ylim(0, 1.05)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            path = _chart("gnn_training.png")
            saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] gnn_training: {exc}")

    # ------------------------------------------------------------------
    # 2. CG Cost Convergence (all B&P nodes)
    # ------------------------------------------------------------------
    cg_all = results.get("cg_episode_history") or []
    if cg_all:
        try:
            cg_df = pd.DataFrame(cg_all)
            fig, ax = plt.subplots(figsize=(10, 4))
            node_ids = sorted(cg_df["branch_node_id"].unique()) if "branch_node_id" in cg_df.columns else [0]
            cmap = mpl.colormaps.get_cmap("tab10")
            for idx, nid in enumerate(node_ids):
                sub = cg_df[cg_df["branch_node_id"] == nid] if "branch_node_id" in cg_df.columns else cg_df
                label = f"Node {nid}"
                ax.plot(sub["episode"], sub["total_cost"] / 1e6,
                        marker="o", linewidth=1.5, color=cmap(idx % 10), label=label, markersize=4)
            ax.set_xlabel("CG Episode (within node)")
            ax.set_ylabel("Total Cost (M)")
            ax.set_title("Column Generation Cost Convergence per B&P Node")
            if len(node_ids) <= 10:
                ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            path = _chart("cg_convergence.png")
            saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] cg_convergence: {exc}")

    # ------------------------------------------------------------------
    # 3. Cost Breakdown Comparison: baseline | without LT | with CG LT
    # ------------------------------------------------------------------
    no_lt  = results.get("realized_no_lt_cost_breakdown")  or {}
    with_lt = results.get("realized_with_lt_cost_breakdown") or {}
    baseline_bd = results.get("baseline_cost_breakdown") or {}
    if no_lt and with_lt:
        try:
            components = [
                ("DC Ship",      "direct_cw_unit_cost",                  "direct_cw_unit_cost_executed_plan"),
                ("Store Hold",   "store_holding_cost",                   "store_holding_cost_realized"),
                ("WH Hold",      "warehouse_holding_cost",               "warehouse_holding_cost_executed_plan"),
                ("Route",        "route_distance_cost",                  "route_distance_cost_executed_plan"),
                ("Vehicle",      "vehicle_fixed_cost",                   "vehicle_fixed_cost_executed_plan"),
                ("LT Cost",      None,                                   "lateral_transshipment_cost_realized"),
                ("Shortage",     "shortage_cost",                        "shortage_cost_realized"),
            ]
            labels = [c[0] for c in components]
            baseline_vals = [float(baseline_bd.get(c[1], 0.0)) / 1e6 if c[1] else 0.0 for c in components]
            nolt_vals    = [float(no_lt.get(c[2],  0.0)) / 1e6 for c in components]
            withlt_vals  = [float(with_lt.get(c[2], 0.0)) / 1e6 for c in components]

            x = range(len(labels))
            width = 0.25
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar([i - width for i in x], baseline_vals, width, label="Baseline (forecast)", color="#4C72B0")
            ax.bar([i         for i in x], nolt_vals,    width, label="Realized — No LT",    color="#DD8452")
            ax.bar([i + width for i in x], withlt_vals,  width, label="Realized — With LT",  color="#55A868")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("Cost (M)")
            ax.set_title("Cost Breakdown: Baseline vs Realized Without/With LT")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            path = _chart("cost_breakdown.png")
            saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] cost_breakdown: {exc}")

    # ------------------------------------------------------------------
    # 4. Shortage Reduction: before / after LT
    # ------------------------------------------------------------------
    if no_lt and with_lt:
        try:
            categories = ["Shortage Units", "Shortage Cost (k)"]
            before = [
                float(no_lt.get("total_realized_shortage_units", 0.0)),
                float(no_lt.get("shortage_cost_realized", 0.0)) / 1e3,
            ]
            after = [
                float(with_lt.get("total_realized_shortage_units", 0.0)),
                float(with_lt.get("shortage_cost_realized", 0.0)) / 1e3,
            ]
            x = range(len(categories))
            width = 0.35
            fig, ax = plt.subplots(figsize=(7, 4))
            bars1 = ax.bar([i - width/2 for i in x], before, width, label="Without LT", color="#DD8452")
            bars2 = ax.bar([i + width/2 for i in x], after,  width, label="With LT",    color="#55A868")
            for bar in bars1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{bar.get_height():,.1f}", ha="center", va="bottom", fontsize=8)
            for bar in bars2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{bar.get_height():,.1f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(list(x))
            ax.set_xticklabels(categories)
            ax.set_title("Shortage Reduction via Lateral Transshipment")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            path = _chart("shortage_reduction.png")
            saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] shortage_reduction: {exc}")

    # ------------------------------------------------------------------
    # 5. Demand Fulfillment Rate by Store (post-shock, before vs after LT)
    # ------------------------------------------------------------------
    post_shock_df = results.get("post_shock_demand_fulfillment")
    lt_plan_df    = results.get("lt_plan")
    if post_shock_df is not None and not post_shock_df.empty and no_lt and with_lt:
        try:
            # Aggregate by store (mean fulfillment rate)
            rate_col = "post_shock_fulfillment_rate" if "post_shock_fulfillment_rate" in post_shock_df.columns else "demand_fulfillment_rate"
            store_rates = post_shock_df.groupby("store")[rate_col].mean().sort_values()
            stores = list(store_rates.index)

            # Compute per-store post-LT fulfillment from realized breakdown
            shortage_no_lt  = float(no_lt.get("total_realized_shortage_units", 0.0))
            shortage_with_lt = float(with_lt.get("total_realized_shortage_units", 0.0))
            total_demand_all = float(post_shock_df["total_realized_demand"].sum()) if "total_realized_demand" in post_shock_df.columns else 0.0

            fig, ax = plt.subplots(figsize=(max(8, len(stores)), 5))
            x = range(len(stores))
            pre_lt_vals  = [float(store_rates[s]) * 100 for s in stores]

            # Post-LT per store: approximation — distribute LT benefit proportionally to pre-LT shortage
            if lt_plan_df is not None and not lt_plan_df.empty and "to_store" in lt_plan_df.columns:
                lt_received = lt_plan_df.groupby("to_store")["lt_qty"].sum()
                post_lt_vals = []
                for s in stores:
                    shortage_s = float(post_shock_df[post_shock_df["store"] == s]["post_shock_shortage" if "post_shock_shortage" in post_shock_df.columns else "shortage"].sum()) if ("post_shock_shortage" in post_shock_df.columns or "shortage" in post_shock_df.columns) else 0.0
                    demand_s   = float(post_shock_df[post_shock_df["store"] == s]["total_realized_demand"].sum()) if "total_realized_demand" in post_shock_df.columns else 1.0
                    lt_gain    = float(lt_received.get(s, 0.0))
                    shortage_after = max(0.0, shortage_s - lt_gain)
                    fulfilled_after = max(0.0, demand_s - shortage_after)
                    rate_after = fulfilled_after / demand_s if demand_s > 1e-9 else 1.0
                    post_lt_vals.append(min(rate_after * 100, 100.0))
            else:
                post_lt_vals = [min(v + (shortage_no_lt - shortage_with_lt) / max(total_demand_all, 1) * 100, 100.0) for v in pre_lt_vals]

            width = 0.4
            ax.barh([i - width/2 for i in x], pre_lt_vals,  width, label="Post-Shock (before LT)", color="#DD8452")
            ax.barh([i + width/2 for i in x], post_lt_vals, width, label="After LT",               color="#55A868")
            ax.set_yticks(list(x))
            ax.set_yticklabels(stores, fontsize=8)
            ax.set_xlabel("Fulfillment Rate (%)")
            ax.set_title("Demand Fulfillment Rate by Store: Before vs After LT Recourse")
            ax.axvline(100, linestyle="--", color="black", alpha=0.4, linewidth=1)
            ax.legend()
            ax.grid(True, axis="x", alpha=0.3)
            path = _chart("fulfillment_by_store.png")
            saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] fulfillment_by_store: {exc}")

    # ------------------------------------------------------------------
    # 6. LT Flow Heatmap (from_store × to_store, aggregated by qty)
    # ------------------------------------------------------------------
    if lt_plan_df is not None and not lt_plan_df.empty:
        try:
            lt_agg = lt_plan_df.groupby(["from_store", "to_store"])["lt_qty"].sum().reset_index()
            all_stores = sorted(set(lt_agg["from_store"]) | set(lt_agg["to_store"]))
            n = len(all_stores)
            idx_map = {s: i for i, s in enumerate(all_stores)}
            matrix = [[0.0] * n for _ in range(n)]
            for _, row in lt_agg.iterrows():
                r, c = idx_map[row["from_store"]], idx_map[row["to_store"]]
                matrix[r][c] = float(row["lt_qty"])

            import numpy as np
            mat = np.array(matrix)
            fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
            im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(n)); ax.set_xticklabels(all_stores, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(n)); ax.set_yticklabels(all_stores, fontsize=7)
            ax.set_xlabel("Receiver Store")
            ax.set_ylabel("Donor Store")
            ax.set_title("LT Flow Heatmap (total qty transferred)")
            plt.colorbar(im, ax=ax, label="Units")
            for i in range(n):
                for j in range(n):
                    if mat[i, j] > 1e-9:
                        ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                                fontsize=6, color="black" if mat[i, j] < mat.max() * 0.6 else "white")
            path = _chart("lt_flow_heatmap.png")
            saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] lt_flow_heatmap: {exc}")

    # ------------------------------------------------------------------
    # 7. Branch-and-Price Bound Progression
    # ------------------------------------------------------------------
    bp_history = results.get("branch_price_history") or []
    if bp_history:
        try:
            bp_df = pd.DataFrame(bp_history)
            bp_df = bp_df[bp_df["objective"] < 1e17]  # filter inf
            status_colors = {
                "integer_incumbent": "#55A868",
                "pruned_by_bound":   "#C44E52",
                "pruned_by_depth":   "#DD8452",
                "pruned_by_integrality": "#8172B2",
                "infeasible":        "#937860",
                "branched":          "#4C72B0",
                "open":              "#64B5CD",
            }
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: objective per node coloured by status
            ax = axes[0]
            for status, grp in bp_df.groupby("status"):
                ax.scatter(grp["node_id"], grp["objective"] / 1e6,
                           label=status, color=status_colors.get(status, "gray"),
                           s=60, zorder=3)
            if "incumbent_objective" in bp_df.columns:
                inc = bp_df[bp_df["incumbent_objective"] < 1e17].copy()
                if not inc.empty:
                    ax.step(inc["node_id"], inc["incumbent_objective"] / 1e6,
                            where="post", linestyle="--", color="black", linewidth=1.5, label="Incumbent bound")
            ax.set_xlabel("B&P Node ID")
            ax.set_ylabel("Objective (M)")
            ax.set_title("B&P Node Objectives")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            # Right: node status distribution
            ax2 = axes[1]
            counts = bp_df["status"].value_counts()
            colors = [status_colors.get(s, "gray") for s in counts.index]
            ax2.bar(range(len(counts)), counts.values, color=colors)
            ax2.set_xticks(range(len(counts)))
            ax2.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=8)
            ax2.set_ylabel("Node Count")
            ax2.set_title("B&P Node Status Distribution")
            ax2.grid(True, axis="y", alpha=0.3)

            path = _chart("branch_price.png")
            saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] branch_price: {exc}")

    # ------------------------------------------------------------------
    # 8. CG Pricing Funnel per Episode (at root node)
    # ------------------------------------------------------------------
    cg_diag = results.get("cg_episode_diagnostics") or []
    if cg_diag:
        try:
            diag_df = pd.DataFrame(cg_diag)
            if "branch_node_id" in diag_df.columns:
                diag_df = diag_df[diag_df["branch_node_id"] == 0]
            diag_df = diag_df[diag_df["episode"] > 0].reset_index(drop=True)
            if not diag_df.empty:
                eps = diag_df["episode"].tolist()
                fig, ax = plt.subplots(figsize=(10, 4))
                cols_labels = [
                    ("candidate_pairs_before_pruning", "Candidate pairs"),
                    ("pairs_after_pruning",             "After feature pruning"),
                    ("pairs_accepted_stackelberg",      "After Stackelberg"),
                    ("patterns_built_before_gnn",       "Patterns built"),
                    ("patterns_added_to_pool",          "Added to pool"),
                ]
                for col, label in cols_labels:
                    if col in diag_df.columns:
                        ax.plot(eps, diag_df[col], marker="o", linewidth=2, label=label)
                ax.set_xlabel("CG Episode (root node)")
                ax.set_ylabel("Count")
                ax.set_title("Pricing Funnel: Candidate Pairs → Pool per CG Episode")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                path = _chart("pricing_funnel.png")
                saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] pricing_funnel: {exc}")

    # ------------------------------------------------------------------
    # 10. ALNS Convergence (effectiveness of baseline search)
    # ------------------------------------------------------------------
    alns_history = results.get("alns_history") or []
    if alns_history:
        try:
            hist_df = pd.DataFrame(alns_history)
            if not hist_df.empty and "iteration" in hist_df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
                ax = axes[0]
                ax.plot(hist_df["iteration"], hist_df["best_cost"] / 1e6,
                        color="#55A868", linewidth=2, label="Best (incumbent)")
                ax.plot(hist_df["iteration"], hist_df["current_cost"] / 1e6,
                        color="#4C72B0", linewidth=0.8, alpha=0.6, label="Current")
                if "candidate_cost" in hist_df.columns:
                    ax.scatter(hist_df["iteration"], hist_df["candidate_cost"] / 1e6,
                               s=3, color="#DD8452", alpha=0.25, label="Candidate")
                ax.set_xlabel("ALNS Iteration")
                ax.set_ylabel("Cost (M)")
                ax.set_title("ALNS Cost Trajectory")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

                ax2 = axes[1]
                # Acceptance rate + new-best events (rolling window)
                window = max(20, len(hist_df) // 50)
                if "accepted" in hist_df.columns:
                    accept_rate = hist_df["accepted"].rolling(window, min_periods=1).mean()
                    ax2.plot(hist_df["iteration"], accept_rate * 100,
                             color="#4C72B0", linewidth=2, label=f"Accept rate ({window}-iter avg)")
                if "new_best" in hist_df.columns:
                    new_best_mask = hist_df["new_best"] > 0
                    ax2.scatter(hist_df.loc[new_best_mask, "iteration"],
                                [100] * int(new_best_mask.sum()),
                                marker="v", color="#55A868", s=40, label="New best")
                ax2.set_xlabel("ALNS Iteration")
                ax2.set_ylabel("Accept Rate (%)")
                ax2.set_ylim(-5, 110)
                ax2.set_title("ALNS Acceptance Dynamics")
                ax2.legend(fontsize=8, loc="lower left")
                ax2.grid(True, alpha=0.3)

                path = _chart("alns_convergence.png")
                saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] alns_convergence: {exc}")

    # ------------------------------------------------------------------
    # 11. Cost Breakdown Log-Scale (so small components stay visible)
    # ------------------------------------------------------------------
    if no_lt and with_lt:
        try:
            components = [
                ("DC Ship",      "direct_cw_unit_cost_executed_plan"),
                ("Store Hold",   "store_holding_cost_realized"),
                ("WH Hold",      "warehouse_holding_cost_executed_plan"),
                ("Route",        "route_distance_cost_executed_plan"),
                ("Vehicle",      "vehicle_fixed_cost_executed_plan"),
                ("LT Cost",      "lateral_transshipment_cost_realized"),
                ("Shortage",     "shortage_cost_realized"),
            ]
            labels = [c[0] for c in components]
            nolt_vals   = [max(1e-3, float(no_lt.get(c[1],  0.0))) for c in components]
            withlt_vals = [max(1e-3, float(with_lt.get(c[1], 0.0))) for c in components]
            x = range(len(labels))
            width = 0.4
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar([i - width/2 for i in x], nolt_vals,   width, label="Realized — No LT",   color="#DD8452")
            ax.bar([i + width/2 for i in x], withlt_vals, width, label="Realized — With LT", color="#55A868")
            ax.set_yscale("log")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("Cost (raw, log scale)")
            ax.set_title("Cost Breakdown After LT Recourse (log scale — all components visible)")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3, which="both")
            # Value labels so LT Cost is readable even when dwarfed
            for i, (n, w) in enumerate(zip(nolt_vals, withlt_vals)):
                ax.text(i - width/2, n, f"{n:,.0f}", ha="center", va="bottom", fontsize=7, rotation=0)
                ax.text(i + width/2, w, f"{w:,.0f}", ha="center", va="bottom", fontsize=7, rotation=0)
            path = str(out_dir / "chart_11_cost_breakdown_log.png")
            saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] chart_11_cost_breakdown_log: {exc}")

    # ------------------------------------------------------------------
    # 9. Phase Comparison: Classical CG vs GNN-Deployed CG (optional)
    # ------------------------------------------------------------------
    if phase_comparison_df is not None and not phase_comparison_df.empty:
        try:
            metrics_keys = [
                ("realized_operating_cost_without_lt", "Cost (no LT, M)"),
                ("realized_operating_cost_with_cg_lt", "Cost (with LT, M)"),
                ("realized_cost_delta_without_minus_with_lt", "LT Savings (M)"),
            ]
            phases = phase_comparison_df["phase"].tolist() if "phase" in phase_comparison_df.columns else list(range(len(phase_comparison_df)))
            x = range(len(metrics_keys))
            width = 0.8 / max(len(phases), 1)
            cmap2 = mpl.colormaps.get_cmap("Set2")
            fig, ax = plt.subplots(figsize=(10, 5))
            for pi, phase in enumerate(phases):
                row = phase_comparison_df[phase_comparison_df["phase"] == phase].iloc[0] if "phase" in phase_comparison_df.columns else phase_comparison_df.iloc[pi]
                vals = [float(row.get(k, 0.0)) / 1e6 for k, _ in metrics_keys]
                offsets = [i + (pi - len(phases) / 2 + 0.5) * width for i in x]
                bars = ax.bar(offsets, vals, width * 0.9, label=str(phase), color=cmap2(pi))
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                            f"{v:.1f}M", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(list(x))
            ax.set_xticklabels([lbl for _, lbl in metrics_keys])
            ax.set_ylabel("Value (M)")
            ax.set_title("Phase Comparison: Classical CG vs GNN-Deployed CG")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            path = str(out_dir / "phase_comparison.png")  # cross-phase chart — no suffix
            saved.append(_save_fig(plt, path))
        except Exception as exc:
            print(f"[Charts] phase_comparison: {exc}")

    return saved


def save_cg_cost_curve(cg_history: List[Dict[str, Any]], output_path: str) -> Optional[str]:
    if not cg_history:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = [int(row["episode"]) for row in cg_history]
        total_cost = [float(row["total_cost"]) for row in cg_history]
        plt.figure(figsize=(7, 4))
        plt.plot(episodes, total_cost, marker="o", linewidth=2)
        plt.xlabel("CG episode")
        plt.ylabel("Total cost")
        plt.title("Column Generation Total Cost")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return output_path
    except Exception as exc:
        print(f"[CG] Could not save total-cost chart: {exc}")
        return None


def compute_validation_metrics(comp: pd.DataFrame) -> Dict:
    df = comp.copy()
    df["abs_error"] = (df["predicted_end_qty"] - df["actual_end_qty"]).abs()
    df["sq_error"] = (df["predicted_end_qty"] - df["actual_end_qty"]) ** 2
    df["pct_error"] = df.apply(
        lambda r: abs(r["predicted_end_qty"] - r["actual_end_qty"]) / abs(r["actual_end_qty"])
        if r["actual_end_qty"] not in [0, 0.0] else math.nan,
        axis=1
    )
    mae = float(df["abs_error"].mean()) if len(df) else math.nan
    rmse = float(math.sqrt(df["sq_error"].mean())) if len(df) else math.nan
    bias = float((df["predicted_end_qty"] - df["actual_end_qty"]).mean()) if len(df) else math.nan
    mape = float(df["pct_error"].dropna().mean()) if df["pct_error"].notna().any() else math.nan
    return {"MAE": mae, "RMSE": rmse, "Bias": bias, "MAPE": mape}


# ============================================================================
# ROUTE EXTRACTION HELPERS
# ============================================================================

def extract_routes_from_solution(solution: FullIRPTSolution, warehouse: str = "CW") -> List[Dict[str, Any]]:
    routes: List[Dict[str, Any]] = []
    active_by_vt: Dict[Tuple[Vehicle, Period], List[Tuple[Node, Node]]] = {}
    for (i, j, v, t), val in solution.x.items():
        if val > 0.5:
            active_by_vt.setdefault((v, t), []).append((i, j))

    for (v, t), arcs in sorted(active_by_vt.items(), key=lambda x: (x[0][1], x[0][0])):
        outgoing: Dict[Node, List[Node]] = {}
        incoming: Dict[Node, List[Node]] = {}
        for i, j in arcs:
            outgoing.setdefault(i, []).append(j)
            incoming.setdefault(j, []).append(i)
        degree_warnings = []
        for node in sorted(set(outgoing) | set(incoming)):
            out_deg = len(outgoing.get(node, []))
            in_deg = len(incoming.get(node, []))
            if out_deg > 1 or in_deg > 1:
                degree_warnings.append(f"{node}:in={in_deg},out={out_deg}")
        next_map = {i: js[0] for i, js in outgoing.items() if js}
        if warehouse not in next_map:
            routes.append({
                "period": t,
                "vehicle": v,
                "route": [f"UNRESOLVED_ARCS::{arcs}"],
                "arcs": arcs,
                "total_direct_qty": 0.0,
                "total_lt_qty": 0.0,
                "load_departure": round(float(solution.load.get((warehouse, v, t), 0.0)), 6),
                "load_by_node": {},
                "direct_qty_by_node": {},
                "service_qty_by_node": {},
                "zero_service_nodes": [],
                "degree_warnings": degree_warnings,
                "unvisited_arcs": arcs,
                "product_flow_summary": {},
            })
            continue

        route = [warehouse]
        visited = set()
        cur = warehouse
        while cur in next_map and (cur, next_map[cur]) not in visited:
            nxt = next_map[cur]
            visited.add((cur, nxt))
            route.append(nxt)
            cur = nxt
            if cur == warehouse:
                break
        unvisited_arcs = [arc for arc in arcs if arc not in visited]

        product_flow_summary: Dict[Product, float] = {}
        total_direct_qty = sum(
            float(qty)
            for (s, p, vv, tt), qty in solution.deliv.items()
            if vv == v and tt == t and qty > 1e-9
        )
        total_lt_qty = 0.0
        for i, j in arcs:
            for (p, ii, jj, vv, tt), qty in solution.q.items():
                if ii == i and jj == j and vv == v and tt == t and qty > 1e-9:
                    product_flow_summary[p] = product_flow_summary.get(p, 0.0) + float(qty)
            for (ii, jj, p, vv, tt), qty in solution.y.items():
                if ii == i and jj == j and vv == v and tt == t and qty > 1e-9:
                    total_lt_qty += float(qty)
        load_by_node = {
            node: round(float(solution.load.get((node, v, t), 0.0)), 6)
            for node in route
            if (node, v, t) in solution.load
        }
        direct_qty_by_node = {
            node: round(
                sum(
                    float(qty)
                    for (s, p, vv, tt), qty in solution.deliv.items()
                    if s == node and vv == v and tt == t and qty > 1e-9
                ),
                6,
            )
            for node in route
            if node != warehouse
        }
        service_qty_by_node = {
            node: round(
                direct_qty_by_node.get(node, 0.0)
                + sum(
                    float(qty)
                    for (src, dst, p, vv, tt), qty in solution.y.items()
                    if vv == v and tt == t and qty > 1e-9 and (src == node or dst == node)
                ),
                6,
            )
            for node in route
            if node != warehouse
        }
        zero_service_nodes = [
            node for node, qty in service_qty_by_node.items()
            if qty <= 1e-6
        ]

        routes.append({
            "period": t,
            "vehicle": v,
            "route": route,
            "arcs": arcs,
            "total_direct_qty": round(total_direct_qty, 6),
            "total_lt_qty": round(total_lt_qty, 6),
            "load_departure": round(float(solution.load.get((warehouse, v, t), 0.0)), 6),
            "load_by_node": load_by_node,
            "direct_qty_by_node": direct_qty_by_node,
            "service_qty_by_node": service_qty_by_node,
            "zero_service_nodes": zero_service_nodes,
            "degree_warnings": degree_warnings,
            "unvisited_arcs": unvisited_arcs,
            "product_flow_summary": {k: round(vv, 6) for k, vv in product_flow_summary.items()},
        })
    return routes


def print_routes(routes: List[Dict[str, Any]]) -> None:
    print("\n[Baseline Routing Output]")
    if not routes:
        print("  No active routes found.")
        return
    for row in routes:
        route_str = " -> ".join(row["route"])
        print(
            f"  period={row['period']} | vehicle={row['vehicle']} | route={route_str} "
            f"| direct_qty={row['total_direct_qty']:.2f} | lt_qty={row['total_lt_qty']:.2f} "
            f"| load_departure={row.get('load_departure', 0.0):.2f} "
            f"| product_flow={row['product_flow_summary']}"
        )
        if row.get("degree_warnings") or row.get("unvisited_arcs"):
            print(
                f"    route_warning degree={row.get('degree_warnings', [])} "
                f"unvisited_arcs={row.get('unvisited_arcs', [])}"
            )
        if row.get("zero_service_nodes"):
            print(f"    zero_service_nodes={row['zero_service_nodes']}")



# ============================================================================
# PIPELINE
# ============================================================================

class IRPResearchPipeline:
    def __init__(self, data: IRPData):
        self.data = data

    def run(
        self,
        use_random_initial_patterns: bool = True,
        n_initial_patterns_per_product_period: int = 5,
        cg_iterations: int = 15,
        msg: bool = True,
        time_limit: Optional[int] = None,
        enforce_integer_flows: bool = False,
        cw_dispatch_cycle: Optional[int] = 5,
        use_gnn: bool = False,
        collect_teacher_mode: bool = True,
        runtime_gnn_mode: Optional[bool] = None,
        gnn_checkpoint: str = DEFAULT_GNN_CHECKPOINT,
        use_classical_fallback: bool = True,
        gnn_selection_mode: str = "cumulative_mass",
        gnn_mass_threshold: float = 0.55,
        gnn_relative_threshold: float = 0.85,
        gnn_max_keep: Optional[int] = 150,
        gnn_max_keep_fraction: float = 0.30,
        use_branch_and_price: bool = True,
        bp_max_nodes: int = 15,
        bp_max_depth: int = 6,
        lt_activation_threshold: float = 10.0,
        demand_shock_probability: float = 0.85,
        demand_shock_reallocation_fraction: float = 0.60,
        demand_shock_reallocations_per_product_period: int = 3,
        demand_shock_non_dispatch_multiplier: float = 1.8,
        demand_shock_seed: int = 20260418,
        diagnostic_verbosity: str = "summary",
        heuristic_top_k_mode: bool = False,
        heuristic_top_k: int = 20,
        exact_full_mode: bool = False,
        stackelberg_aware_scoring: bool = False,
        stackelberg_exact_follower: bool = False,
        stackelberg_min_lateral_qty: Optional[float] = None,
    ) -> Dict:
        pipeline_started_at = time.perf_counter()
        print("=" * 80)
        print("STEP 1 - Solve baseline IRPT  (ALNS, DC->stores->DC, no LT)")
        print("=" * 80)
        # --- DEPRECATED GUROBI BASELINE (kept commented for reference; ALNS replaces it) ---
        # baseline_sol = AchamrahFullIRPTModel(self.data).solve(
        #     msg=msg,
        #     time_limit=time_limit,
        #     enforce_integer_flows=enforce_integer_flows,
        #     add_valid_16_20=True,
        #     allow_lateral_transshipment=False,
        #     cw_dispatch_cycle=cw_dispatch_cycle,
        # )
        baseline_sol = BaselineALNSModel(self.data).solve(
            msg=msg,
            time_limit=time_limit,
            enforce_integer_flows=enforce_integer_flows,
            add_valid_16_20=True,
            allow_lateral_transshipment=False,
            cw_dispatch_cycle=cw_dispatch_cycle,
        )
        if _is_quiet():
            _summary = baseline_sol.summary()
            print(f"[Baseline Summary] obj={_summary.get('objective'):.4f} keys={len(_summary)}" if isinstance(_summary, dict) else "[Baseline Summary] printed")
        else:
            pprint.pprint(baseline_sol.summary())
        print_efficiency_metrics("Baseline Solver Efficiency", baseline_sol.efficiency_metrics)
        baseline_cost_breakdown = build_full_irpt_cost_breakdown(self.data, baseline_sol)
        print_cost_breakdown("Baseline Full IRPT Cost Breakdown", baseline_cost_breakdown)
        baseline_routes = extract_routes_from_solution(baseline_sol, warehouse=self.data.warehouse)

        print("\n" + "=" * 80)
        print("STEP 1B - Apply hidden realized-demand shock after DC shipment")
        print("=" * 80)
        shock_summary = apply_hidden_local_reallocation_demand_shocks(
            self.data,
            baseline_solution=baseline_sol,
            shock_probability=demand_shock_probability,
            max_reallocation_fraction=demand_shock_reallocation_fraction,
            reallocations_per_product_period=demand_shock_reallocations_per_product_period,
            non_dispatch_shock_multiplier=demand_shock_non_dispatch_multiplier,
            cw_dispatch_cycle=cw_dispatch_cycle,
            seed=demand_shock_seed,
        )
        post_shock_summary = build_post_shock_inventory_state(self.data, baseline_sol)
        post_shock_lt_diagnostics = build_post_shock_lt_diagnostics(
            self.data,
            baseline_sol,
            lt_activation_threshold=lt_activation_threshold,
        )
        forecast_fulfillment_df = build_forecast_fulfillment_df(self.data, baseline_sol)
        demand_fulfillment_df = build_post_shock_fulfillment_df(self.data)
        if _is_quiet():
            print(f"[Hidden Demand Shock Summary] keys={len(shock_summary) if isinstance(shock_summary, dict) else 'n/a'}")
            print(f"[Post-Shock Inventory State Summary] keys={len(post_shock_summary) if isinstance(post_shock_summary, dict) else 'n/a'}")
            print(f"[Post-Shock LT Diagnostics] keys={len(post_shock_lt_diagnostics) if isinstance(post_shock_lt_diagnostics, dict) else 'n/a'}")
        else:
            print("[Hidden Demand Shock Summary]")
            pprint.pprint(shock_summary)
            print("[Post-Shock Inventory State Summary]")
            pprint.pprint(post_shock_summary)
            print("[Post-Shock LT Diagnostics]")
            pprint.pprint(post_shock_lt_diagnostics)
            print_post_shock_fulfillment(demand_fulfillment_df)

        initial_patterns = []
        if use_random_initial_patterns:
            print("\n" + "=" * 80)
            print("STEP 2 - Create demo LT patterns")
            print("=" * 80)
            initial_patterns = generate_random_lt_patterns(
                self.data,
                baseline_solution=baseline_sol,
                n_patterns_per_product_period=n_initial_patterns_per_product_period,
                max_pairs_in_pattern=4,
                lt_activation_threshold=lt_activation_threshold,
                seed=int(os.environ.get("IRP_PATTERN_INIT_SEED", "123")),
            )
            print(f"Generated {len(initial_patterns)} initial LT patterns")

        print("\n" + "=" * 80)
        print("STEP 3 - Run column generation with RMP + pricing + dual loop")
        print("=" * 80)
        requested_runtime_gnn = bool(use_gnn if runtime_gnn_mode is None else runtime_gnn_mode)
        effective_runtime_gnn = False if collect_teacher_mode else requested_runtime_gnn
        gnn_training_history = load_gnn_training_history(gnn_checkpoint) if effective_runtime_gnn else []
        if effective_runtime_gnn:
            print_gnn_training_history(gnn_training_history, gnn_checkpoint)
        if collect_teacher_mode:
            print("[Teacher] collect_teacher_mode=True: priced batches will bypass GNN filtering before RMP.")
            if requested_runtime_gnn:
                print("[Teacher] runtime_gnn_mode request ignored during teacher collection to avoid self-filtered labels.")

        stackelberg_params = StackelbergParams(
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

        cg_engine = LateralTransshipmentCG(
            data=self.data,
            baseline_solution=baseline_sol,
            initial_patterns=initial_patterns,
            lt_activation_threshold=lt_activation_threshold,
            max_pairs_per_pattern=4,
            top_pairs_per_feature=20,
            top_patterns_per_feature=5,
            stackelberg_params=stackelberg_params,
            use_gnn=effective_runtime_gnn,
            collect_teacher_mode=collect_teacher_mode,
            runtime_gnn_mode=effective_runtime_gnn,
            gnn_checkpoint=gnn_checkpoint,
            use_classical_fallback=use_classical_fallback,
            gnn_selection_mode=gnn_selection_mode,
            gnn_mass_threshold=gnn_mass_threshold,
            gnn_relative_threshold=gnn_relative_threshold,
            gnn_max_keep=gnn_max_keep,
            gnn_max_keep_fraction=gnn_max_keep_fraction,
            diagnostic_verbosity=diagnostic_verbosity,
            heuristic_top_k_mode=heuristic_top_k_mode,
            heuristic_top_k=heuristic_top_k,
            exact_full_mode=exact_full_mode,
            stackelberg_aware_scoring=stackelberg_aware_scoring,
            stackelberg_exact_follower=stackelberg_exact_follower,
            stackelberg_min_lateral_qty=stackelberg_min_lateral_qty,
        )
        if use_branch_and_price:
            cg_sol = cg_engine.run_branch_and_price(
                max_iter=cg_iterations,
                msg=msg,
                max_nodes=bp_max_nodes,
                max_depth=bp_max_depth,
            )
        else:
            cg_sol = cg_engine.run_column_generation(max_iter=cg_iterations, msg=msg)
        if _is_quiet():
            _cg_summary = cg_sol.summary()
            print(f"[CG Summary] obj={cg_sol.objective:.4f} selected={len(cg_sol.selected_patterns)} keys={len(_cg_summary) if isinstance(_cg_summary, dict) else 'n/a'}")
        else:
            pprint.pprint(cg_sol.summary())
        print_efficiency_metrics("CG RMP Solver Efficiency", cg_sol.efficiency_metrics)
        lt_plan_df = build_lt_plan_df_from_cg(cg_sol, cg_engine.patterns, self.data)
        print_lt_plan(lt_plan_df, title="CG Lateral Transshipment Plan")
        realized_no_lt_cost_breakdown = build_realized_operating_cost_breakdown(
            self.data,
            baseline_sol,
            lt_plan_df=None,
        )
        realized_with_lt_cost_breakdown = build_realized_operating_cost_breakdown(
            self.data,
            baseline_sol,
            lt_plan_df=lt_plan_df,
        )
        print_cost_breakdown("Realized Operating Cost Without LT", realized_no_lt_cost_breakdown)
        print_cost_breakdown("Realized Operating Cost With CG LT", realized_with_lt_cost_breakdown)

        # Expose the inventory trajectory that produced the "with LT" realized cost
        # so downstream validation compares against the actual shelf state (baseline
        # forecast + post-shock reallocation + LT recourse), not just the forecast
        # inv_store. build_predicted_inventory_df automatically picks this up.
        baseline_sol.realized_inventory_after_lt = compute_realized_inventory_after_lt(
            self.data,
            baseline_sol,
            lt_plan_df=lt_plan_df,
        )

        pipeline_runtime_seconds = time.perf_counter() - pipeline_started_at
        comparison = {
            "forecast_dc_plan_objective": baseline_sol.objective,
            "realized_operating_cost_without_lt": realized_no_lt_cost_breakdown["total_realized_operating_cost"],
            "realized_operating_cost_with_cg_lt": realized_with_lt_cost_breakdown["total_realized_operating_cost"],
            "realized_cost_delta_without_minus_with_lt": (
                realized_no_lt_cost_breakdown["total_realized_operating_cost"]
                - realized_with_lt_cost_breakdown["total_realized_operating_cost"]
            ),
            "cg_rmp_surrogate_objective": cg_sol.objective,
            "comparison_note": (
                "DC planning uses forecast demand. Realized operating cost keeps executed DC "
                "shipment/routing/vehicle costs fixed, then evaluates store holding, shortage, "
                "and LT recourse costs after hidden realized-demand shocks."
            ),
            "pipeline_runtime_seconds": pipeline_runtime_seconds,
            "lt_plan_source": (
                "cg_selected_patterns_with_follower_filter"
                if stackelberg_aware_scoring else "cg_selected_patterns"
            ),
            "follower_aware_validation": cg_sol.stackelberg_validation,
            "baseline_efficiency_metrics": baseline_sol.efficiency_metrics,
            "cg_rmp_efficiency_metrics": cg_sol.efficiency_metrics,
        }
        print("\n" + "=" * 80)
        print("STEP 4 - Comparison")
        print("=" * 80)
        if _is_quiet():
            print(
                f"[Comparison] forecast_plan={comparison['forecast_dc_plan_objective']:.4f} "
                f"realized_no_lt={comparison['realized_operating_cost_without_lt']:.4f} "
                f"realized_with_lt={comparison['realized_operating_cost_with_cg_lt']:.4f} "
                f"delta={comparison['realized_cost_delta_without_minus_with_lt']:.4f} "
                f"runtime={pipeline_runtime_seconds:.2f}s"
            )
        else:
            pprint.pprint(comparison)

        return {
            "baseline_solution": baseline_sol,
            "baseline_routes": baseline_routes,
            "demand_fulfillment": demand_fulfillment_df,
            "forecast_demand_fulfillment": forecast_fulfillment_df,
            "post_shock_demand_fulfillment": demand_fulfillment_df,
            "baseline_cost_breakdown": baseline_cost_breakdown,
            "realized_no_lt_cost_breakdown": realized_no_lt_cost_breakdown,
            "realized_with_lt_cost_breakdown": realized_with_lt_cost_breakdown,
            "demand_shock_summary": shock_summary,
            "post_shock_summary": post_shock_summary,
            "post_shock_lt_diagnostics": post_shock_lt_diagnostics,
            "cg_solution": cg_sol,
            "lt_plan": lt_plan_df,
            "comparison": comparison,
            "gnn_training_history": gnn_training_history,
            "gnn_selection_history": cg_engine.gnn_selection_history,
            "cg_episode_history": cg_engine.cg_history_all_nodes or cg_engine.cg_history,
            "branch_price_history": cg_engine.branch_history,
            "cg_episode_diagnostics": cg_engine.cg_episode_diagnostics,
            "column_pool_diagnostics": cg_engine.column_pool_diagnostics,
            "teacher_dataset_rows": cg_engine.teacher_dataset_rows,
            "alns_history": list(getattr(baseline_sol, "alns_history", []) or []),
            "gnn_scoring_failures": int(getattr(cg_engine, "_gnn_scoring_failures", 0)),
            "lt_plan_source": (
                "cg_selected_patterns_with_follower_filter"
                if stackelberg_aware_scoring else "cg_selected_patterns"
            ),
            "stackelberg_follower_plan": cg_sol.follower_solution,
            "stackelberg_column_scores": cg_sol.stackelberg_column_scores,
            "stackelberg_validation": cg_sol.stackelberg_validation,
        }


# ============================================================================
# 3-WAY BENCHMARK RUNNER (IRP_RUN_BENCHMARK=1)
# ============================================================================

def _collect_benchmark_metrics(variant: str, results: Dict[str, Any],
                               runtime_seconds: float,
                               run_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract the comparison-table row from a completed pipeline run.

    The row includes the fields thesis benchmarking needs: identity (run label,
    source_instance, seed), runtime breakdown (phase-1 baseline, phase-2 CG,
    GNN inference, optional post-run fine-tune), solver stopping reason,
    objective/cost, iteration count, candidate counts, and selected counts.
    """
    baseline_sol = results.get("baseline_solution")
    cg_sol = results.get("cg_solution")
    realized_no_lt = results.get("realized_no_lt_cost_breakdown") or {}
    realized_with_lt = results.get("realized_with_lt_cost_breakdown") or {}
    # cg_episode_history rows carry high-level objective curve (episode, total_cost,
    # proposed_columns, added_columns). cg_episode_diagnostics rows carry the
    # fine-grained per-episode column counts (patterns_built_before_gnn,
    # patterns_kept_after_gnn, patterns_added_to_pool). The two files never share
    # schemas — earlier code mistakenly pulled the diagnostic keys from
    # cg_episode_history and got zeros.
    cg_history = results.get("cg_episode_history") or []
    cg_diags = results.get("cg_episode_diagnostics") or []
    total_cols_generated = sum(int(r.get("patterns_built_before_gnn", 0) or 0) for r in cg_diags)
    total_cols_added = sum(int(r.get("patterns_added_to_pool", 0) or 0) for r in cg_diags)
    total_cols_selected_by_gnn = sum(int(r.get("patterns_kept_after_gnn", 0) or 0) for r in cg_diags)
    # Fallback: if diagnostics were not recorded, use cg_history's proposed/added
    # fields so the row is never silently zero.
    if total_cols_generated == 0 and cg_history:
        total_cols_generated = sum(int(r.get("proposed_columns", 0) or 0) for r in cg_history)
    if total_cols_added == 0 and cg_history:
        total_cols_added = sum(int(r.get("added_columns", 0) or 0) for r in cg_history)
    # Final pool size = number of columns remaining in the RMP; selected = columns
    # with lambda > 0 in the final solution. Both come directly from cg_solution.
    pool_size = len(getattr(cg_sol, "lambda_values", {}) or {}) if cg_sol is not None else 0
    selected_in_rmp = len(getattr(cg_sol, "selected_patterns", []) or []) if cg_sol is not None else 0
    pool_util = (selected_in_rmp / pool_size) if pool_size > 0 else 0.0
    gnn_scoring_failures = int(results.get("gnn_scoring_failures", 0) or 0)
    # Baseline ALNS runtime lives in efficiency_metrics["alns_runtime_seconds"]
    # (there is no baseline_sol.runtime_seconds attribute — earlier code was
    # always getting NaN here).
    baseline_eff = getattr(baseline_sol, "efficiency_metrics", {}) or {}
    baseline_runtime = float(
        baseline_eff.get("alns_runtime_seconds",
                         baseline_eff.get("gurobi_runtime_seconds", float("nan")))
    )
    # cg_solution.efficiency_metrics["gurobi_runtime_seconds"] is the sum of
    # RMP-solve runtimes (_add_efficiency_metrics accumulates it across solves);
    # there is no "rmp_total_runtime" key.
    cg_runtime = float(
        (cg_sol.efficiency_metrics or {}).get("gurobi_runtime_seconds", float("nan"))
        if cg_sol is not None else float("nan")
    )
    # GNN inference runtime is reported via comparison → cg_rmp_efficiency_metrics
    # when runtime_gnn_mode was enabled. Fall back to 0 when it wasn't tracked.
    gnn_inference_runtime = float((cg_sol.efficiency_metrics or {}).get("gnn_total_runtime", 0.0)) if cg_sol else 0.0
    # Stopping reason: infer from last CG episode — added==0 means convergence,
    # else budget exhausted.
    last_ep = cg_history[-1] if cg_history else {}
    stopping_reason = (
        "convergence_no_negative_rc" if int(last_ep.get("added_columns", 0) or 0) == 0
        else "budget_exhausted"
    )
    run_context = run_context or {}
    return {
        "variant": variant,
        "run_label": run_context.get("run_label", variant),
        "source_instance": run_context.get("source_instance", ""),
        "dataset_id": run_context.get("dataset_id", ""),
        "scenario_id": run_context.get("scenario_id", ""),
        "seed": run_context.get("seed", ""),
        "benchmark_mode": run_context.get("benchmark_mode", ""),
        "rmp_objective": float(cg_sol.objective) if cg_sol is not None else float("nan"),
        # build_realized_operating_cost_breakdown returns these exact keys —
        # earlier code read total_cost / lateral_transshipment_cost / shortage_cost
        # which never existed in the breakdown dict, producing NaN/0 in every
        # benchmark CSV row.
        "realized_cost_no_lt": float(realized_no_lt.get("total_realized_operating_cost", float("nan"))),
        "realized_cost_with_lt": float(realized_with_lt.get("total_realized_operating_cost", float("nan"))),
        "final_objective_or_total_cost": float(realized_with_lt.get("total_realized_operating_cost", float("nan"))),
        "lt_cost_with_lt": float(realized_with_lt.get("lateral_transshipment_cost_realized", 0.0)),
        "shortage_cost_with_lt": float(realized_with_lt.get("shortage_cost_realized", 0.0)),
        "cg_iterations": int(len(cg_history)),
        "stopping_reason": stopping_reason,
        "total_runtime_seconds": float(runtime_seconds),
        "phase1_baseline_runtime_seconds": baseline_runtime,
        "phase2_cg_runtime_seconds": cg_runtime,
        "gnn_inference_runtime_seconds": gnn_inference_runtime,
        "post_run_fine_tune_runtime_seconds": float(run_context.get("fine_tune_runtime_seconds", 0.0)),
        "columns_generated": total_cols_generated,
        "columns_selected_by_gnn": total_cols_selected_by_gnn,
        "columns_added_to_rmp": total_cols_added,
        "column_pool_utilization": float(pool_util),
        "gnn_scoring_failures": gnn_scoring_failures,
    }


def save_phase_artifacts(
    results: Dict[str, Any],
    layout: ResultsLayout,
    phase_label: Optional[str] = None,
    *,
    validation_target: Optional[pd.DataFrame] = None,
    save_routes: Optional[bool] = None,
) -> Dict[str, str]:
    """Write every per-phase CSV/JSON artifact into Results/<phase>/ with short names.

    Returns a mapping {artifact_name: path} describing what was actually written,
    so the run manifest can list it and the runtime summary can print it.
    Filenames do NOT repeat the phase label — the containing folder signals it.
    """
    phase_label = phase_label or layout.phase_label
    phase_dir = layout.root / phase_label
    phase_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = phase_dir / "debug"
    if save_routes is None:
        save_routes = os.environ.get("IRP_SAVE_ROUTES", "0").lower() not in {"0", "false", "no", ""}
    written: Dict[str, str] = {}
    apply_integer_rounding = _integer_final_outputs_enabled()

    def _put(name: str, path: Path) -> None:
        written[name] = str(path.relative_to(layout.root))

    def _csv(name: str, df: pd.DataFrame) -> None:
        """Write df as CSV. If any INTEGER_FINAL_OUTPUT_COLUMNS column is
        present and IRP_INTEGER_FINAL_OUTPUTS is enabled (default), write the
        rounded-integer version to the main path and keep the LP-relaxed floats
        under <phase>/debug/<name> for auditing."""
        path = phase_dir / name
        has_qty_cols = any(c in df.columns for c in INTEGER_FINAL_OUTPUT_COLUMNS)
        if apply_integer_rounding and has_qty_cols:
            debug_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(debug_dir / name, index=False)
            _round_integer_columns(df).to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)
        _put(name, path)

    # Cost breakdowns
    try:
        _csv("baseline_cost_breakdown.csv", pd.DataFrame([results["baseline_cost_breakdown"]]))
    except Exception as exc:
        print(f"[artifacts/{phase_label}] baseline_cost_breakdown: {exc}")

    try:
        no_lt = results.get("realized_no_lt_cost_breakdown") or {}
        with_lt = results.get("realized_with_lt_cost_breakdown") or {}
        delta = {k: round(float(no_lt.get(k, 0.0)) - float(with_lt.get(k, 0.0)), 6) for k in no_lt}
        _csv("realized_cost_breakdown.csv", pd.DataFrame([
            {"scenario": "without_lt", **no_lt},
            {"scenario": "with_cg_lt", **with_lt},
            {"scenario": "delta_no_lt_minus_with_lt", **delta},
        ]))
    except Exception as exc:
        print(f"[artifacts/{phase_label}] realized_cost_breakdown: {exc}")

    # Solver efficiency
    try:
        _csv("solver_efficiency.csv", pd.DataFrame([
            {"model": "baseline_irpt", **results["baseline_solution"].efficiency_metrics},
            {"model": "cg_rmp_total",  **results["cg_solution"].efficiency_metrics},
        ]))
    except Exception as exc:
        print(f"[artifacts/{phase_label}] solver_efficiency: {exc}")

    # Demand fulfillment (forecast + post-shock in one file, tagged by mode)
    try:
        parts: List[pd.DataFrame] = []
        fc = results.get("forecast_demand_fulfillment")
        ps = results.get("post_shock_demand_fulfillment")
        if isinstance(fc, pd.DataFrame) and not fc.empty:
            tagged = fc.copy(); tagged["mode"] = "pre_shock_forecast"; parts.append(tagged)
        if isinstance(ps, pd.DataFrame) and not ps.empty:
            tagged = ps.copy(); tagged["mode"] = "post_shock_realized"; parts.append(tagged)
        if parts:
            _csv("demand_fulfillment.csv", pd.concat(parts, ignore_index=True, sort=False))
    except Exception as exc:
        print(f"[artifacts/{phase_label}] demand_fulfillment: {exc}")

    # LT plan
    try:
        lt_plan = results.get("lt_plan")
        if isinstance(lt_plan, pd.DataFrame):
            _csv("lt_plan.csv", lt_plan)
    except Exception as exc:
        print(f"[artifacts/{phase_label}] lt_plan: {exc}")

    # Demand shock summary
    try:
        shock_row = {
            **(results.get("demand_shock_summary") or {}),
            **(results.get("post_shock_summary") or {}),
            **(results.get("post_shock_lt_diagnostics") or {}),
        }
        if shock_row:
            _csv("demand_shock_summary.csv", pd.DataFrame([shock_row]))
    except Exception as exc:
        print(f"[artifacts/{phase_label}] demand_shock_summary: {exc}")

    # CG / B&P histories (per-phase so phase1 and phase2 don't overwrite each other)
    for key, fname in [
        ("cg_episode_history",     "cg_episode_history.csv"),
        ("branch_price_history",   "branch_price_history.csv"),
        ("cg_episode_diagnostics", "cg_episode_diagnostics.csv"),
        ("column_pool_diagnostics","column_pool_diagnostics.csv"),
    ]:
        v = results.get(key)
        if isinstance(v, list) and v:
            try:
                _csv(fname, pd.DataFrame(v))
            except Exception as exc:
                print(f"[artifacts/{phase_label}] {fname}: {exc}")
        elif isinstance(v, pd.DataFrame) and not v.empty:
            try:
                _csv(fname, v)
            except Exception as exc:
                print(f"[artifacts/{phase_label}] {fname}: {exc}")

    # ALNS history — only if populated
    alns = results.get("alns_history") or []
    if alns:
        try:
            _csv("alns_history.csv", pd.DataFrame(alns))
        except Exception as exc:
            print(f"[artifacts/{phase_label}] alns_history: {exc}")

    # Selected-column history (JSON)
    try:
        sel = results.get("gnn_selection_history")
        if sel is not None:
            path = phase_dir / "selected_columns.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sel, f, indent=2)
            _put("selected_columns.json", path)
    except Exception as exc:
        print(f"[artifacts/{phase_label}] selected_columns: {exc}")

    # Optional verbose routes dump — off by default (very wide rows)
    if save_routes:
        try:
            rows = [
                {
                    "period": r["period"],
                    "vehicle": r["vehicle"],
                    "route": " -> ".join(r["route"]),
                    "total_direct_qty": r["total_direct_qty"],
                    "total_lt_qty": r["total_lt_qty"],
                    "load_departure": r.get("load_departure", 0.0),
                }
                for r in results.get("baseline_routes", [])
            ]
            if rows:
                _csv("baseline_routes.csv", pd.DataFrame(rows))
        except Exception as exc:
            print(f"[artifacts/{phase_label}] baseline_routes: {exc}")

    # Predicted inventory + validation comparison (only if validation_target given).
    #
    # Emits BOTH the forecast-only inventory (`predicted_end_qty_forecast`) and
    # the realized-after-LT inventory (`predicted_end_qty_realized`) alongside
    # `actual_end_qty`, so examiners can see which trajectory matches reality.
    # A validation_diagnostics.json captures the merge statistics and a clear
    # reason string when the comparison is effectively unavailable (e.g. the
    # test data has no overlapping (store, sku, period) keys).
    if validation_target is not None:
        try:
            baseline_sol = results["baseline_solution"]
            forecast_inv = getattr(baseline_sol, "inv_store", {}) or {}
            realized_inv = getattr(baseline_sol, "realized_inventory_after_lt", None) or {}
            rows = []
            for (s, p, t), inv in forecast_inv.items():
                rows.append({
                    "store": s, "sku": p, "period": t,
                    "predicted_end_qty_forecast": float(inv),
                    "predicted_end_qty_realized": float(realized_inv.get((s, p, t), inv)),
                })
            predicted = pd.DataFrame(rows)
            cmp = predicted.merge(
                validation_target, on=["store", "sku", "period"], how="inner"
            )
            # Keep the legacy `predicted_end_qty` / `error` columns so downstream
            # reporting (compute_validation_metrics) still works; they mirror the
            # realized-after-LT trajectory which is what the model claims lands
            # on shelves.
            if not cmp.empty:
                cmp["predicted_end_qty"] = cmp["predicted_end_qty_realized"]
                cmp["error"] = cmp["predicted_end_qty_realized"] - cmp["actual_end_qty"]
                cmp["forecast_error"] = cmp["predicted_end_qty_forecast"] - cmp["actual_end_qty"]
            _csv("validation_comparison.csv", cmp)
            # Diagnostics: where did rows come from / why are they zero?
            diag = {
                "rows_validation_target": int(len(validation_target)),
                "rows_predicted_forecast": int(len(predicted)),
                "rows_merged": int(len(cmp)),
                "n_predicted_zero_realized": int((cmp["predicted_end_qty_realized"] == 0).sum()) if not cmp.empty else 0,
                "n_predicted_zero_forecast": int((cmp["predicted_end_qty_forecast"] == 0).sum()) if not cmp.empty else 0,
                "mean_actual_end_qty": float(cmp["actual_end_qty"].mean()) if not cmp.empty else float("nan"),
                "mean_predicted_realized": float(cmp["predicted_end_qty_realized"].mean()) if not cmp.empty else float("nan"),
                "mean_predicted_forecast": float(cmp["predicted_end_qty_forecast"].mean()) if not cmp.empty else float("nan"),
                "mean_abs_error_realized": float((cmp["predicted_end_qty_realized"] - cmp["actual_end_qty"]).abs().mean()) if not cmp.empty else float("nan"),
                "reason": (
                    "ok" if not cmp.empty
                    else "empty_merge: validation_target and predicted share no (store, sku, period) keys"
                ),
            }
            diag_path = phase_dir / "validation_diagnostics.json"
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2, default=str)
            _put("validation_diagnostics.json", diag_path)
        except Exception as exc:
            print(f"[artifacts/{phase_label}] validation_comparison: {exc}")
            # Emit an NA-with-reason file so examiners know validation ran and failed,
            # rather than silently missing.
            try:
                diag_path = phase_dir / "validation_diagnostics.json"
                with open(diag_path, "w", encoding="utf-8") as f:
                    json.dump({"rows_merged": 0, "reason": f"exception: {exc}"}, f, indent=2)
                _put("validation_diagnostics.json", diag_path)
            except Exception:
                pass

    # CG cost curve → charts/ with phase suffix
    try:
        chart_path = save_cg_cost_curve(
            results.get("cg_episode_history") or [],
            str(layout.chart_file("cg_cost_curve.png", phase_label=phase_label)),
        )
        if chart_path:
            written["cg_cost_curve.png"] = str(Path(chart_path).relative_to(layout.root))
    except Exception as exc:
        print(f"[artifacts/{phase_label}] cg_cost_curve: {exc}")

    return written


# ============================================================================
# THESIS EFFECTIVENESS REPORT + RESULTS/README.md
# ============================================================================


def build_effectiveness_report(
    results_dir: Path,
    *,
    phase_summaries: Optional[List[Dict[str, Any]]] = None,
    gnn_training_history: Optional[List[Dict[str, Any]]] = None,
    gnn_training_summary: Optional[Dict[str, Any]] = None,
    gnn_offline_test_csv: Optional[Path] = None,
    benchmark_aggregate_csv: Optional[Path] = None,
    runtime_breakdown: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Consolidate every 'is this working?' signal into two files examiners
    actually open first:
      - thesis_summary/effectiveness_report.csv  (tabular per-stage KPIs)
      - thesis_summary/effectiveness_report.json (same data, structured)
      - charts/effectiveness_overview.png        (bar chart)

    Returns a mapping {artifact: relative_path}. Missing inputs are skipped
    silently — the report degrades gracefully when a pipeline stage wasn't run.
    """
    results_dir = Path(results_dir)
    thesis_dir = results_dir / THESIS_SUMMARY_SUBDIR
    thesis_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = results_dir / CHARTS_SUBDIR
    charts_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    # Offline training (Phase 1) — baseline solver without GNN
    if phase_summaries:
        for summary in phase_summaries:
            rows.append({
                "stage": summary.get("phase", ""),
                "metric": "realized_operating_cost_no_lt",
                "value": summary.get("cost_no_lt_M", float("nan")) * 1e6,
                "unit": "currency",
            })
            rows.append({
                "stage": summary.get("phase", ""),
                "metric": "realized_operating_cost_with_lt",
                "value": summary.get("cost_with_lt_M", float("nan")) * 1e6,
                "unit": "currency",
            })
            rows.append({
                "stage": summary.get("phase", ""),
                "metric": "lt_saving",
                "value": summary.get("lt_saving_M", float("nan")) * 1e6,
                "unit": "currency",
            })
            rows.append({
                "stage": summary.get("phase", ""),
                "metric": "runtime_end_to_end",
                "value": summary.get("runtime_sec", float("nan")),
                "unit": "seconds",
            })

    # GNN training (offline) — train/valid best-epoch summary
    if gnn_training_summary:
        for key in ("best_epoch", "best_valid_mrr", "best_train_loss",
                    "best_valid_loss", "final_train_loss", "final_valid_loss"):
            if key in gnn_training_summary:
                rows.append({
                    "stage": "gnn_offline_training",
                    "metric": key,
                    "value": gnn_training_summary[key],
                    "unit": "metric",
                })
    if gnn_training_history:
        last = gnn_training_history[-1]
        for key in ("train_loss", "valid_loss", "valid_mrr", "epoch"):
            if key in last:
                rows.append({
                    "stage": "gnn_offline_training",
                    "metric": f"final_{key}",
                    "value": last[key],
                    "unit": "metric",
                })

    # GNN offline held-out test
    if gnn_offline_test_csv is not None and Path(gnn_offline_test_csv).exists():
        try:
            df = pd.read_csv(gnn_offline_test_csv)
            if not df.empty:
                group_col = next((c for c in ["mass_threshold", "threshold"] if c in df.columns), None)
                if group_col:
                    agg = df.groupby(group_col).mean(numeric_only=True).reset_index()
                    for _, row in agg.iterrows():
                        tag = row[group_col]
                        for metric in ("mrr", "ndcg", "top1_hit", "top3_hit", "adaptive_f1"):
                            if metric in row:
                                rows.append({
                                    "stage": "gnn_offline_test",
                                    "metric": f"{metric}@{tag}",
                                    "value": float(row[metric]),
                                    "unit": "metric",
                                })
                else:
                    for metric in ("mrr", "ndcg", "top1_hit", "top3_hit", "adaptive_f1"):
                        if metric in df.columns:
                            rows.append({
                                "stage": "gnn_offline_test",
                                "metric": metric,
                                "value": float(df[metric].mean()),
                                "unit": "metric",
                            })
        except Exception as exc:
            print(f"[EffectivenessReport] offline test: {exc}")

    # External benchmark on test data — the aggregate mean/std
    if benchmark_aggregate_csv is not None and Path(benchmark_aggregate_csv).exists():
        try:
            df = pd.read_csv(benchmark_aggregate_csv)
            if "variant" in df.columns:
                df["variant"] = pd.Categorical(
                    df["variant"],
                    categories=BENCHMARK_VARIANT_ORDER,
                    ordered=True,
                )
                df = df.sort_values("variant").reset_index(drop=True)
            for _, row in df.iterrows():
                variant = row.get("variant", "?")
                for metric in ("realized_cost_with_lt_mean",
                               "realized_cost_with_lt_std",
                               "total_runtime_seconds_mean",
                               "total_runtime_seconds_std",
                               "columns_generated_mean",
                               "columns_added_to_rmp_mean",
                               "cg_iterations_mean"):
                    if metric in df.columns:
                        rows.append({
                            "stage": f"external_benchmark_{variant}",
                            "metric": metric,
                            "value": float(row[metric]) if pd.notna(row[metric]) else float("nan"),
                            "unit": "mean_std" if metric.endswith(("_mean", "_std")) else "metric",
                        })
        except Exception as exc:
            print(f"[EffectivenessReport] benchmark aggregate: {exc}")

    if runtime_breakdown:
        for label, seconds in runtime_breakdown.items():
            try:
                rows.append({
                    "stage": "runtime_breakdown",
                    "metric": str(label),
                    "value": float(seconds),
                    "unit": "seconds",
                })
            except (TypeError, ValueError):
                continue

    report_df = pd.DataFrame(rows)
    csv_path = thesis_dir / "effectiveness_report.csv"
    report_df.to_csv(csv_path, index=False)
    json_path = thesis_dir / "effectiveness_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    out: Dict[str, str] = {
        "effectiveness_report.csv":  str(csv_path.relative_to(results_dir)),
        "effectiveness_report.json": str(json_path.relative_to(results_dir)),
    }

    # Bar chart of a small, examiner-friendly subset.
    try:
        _, plt = _mpl_agg()
        if plt is not None and not report_df.empty:
            highlights = report_df[report_df["metric"].isin([
                "realized_operating_cost_with_lt",
                "realized_cost_with_lt_mean",
                "runtime_end_to_end",
                "total_runtime_seconds_mean",
            ])]
            if not highlights.empty:
                plt.figure(figsize=(10, 5))
                labels = highlights.apply(lambda r: f"{r['stage']}\n{r['metric']}", axis=1)
                values = pd.to_numeric(highlights["value"], errors="coerce").fillna(0.0)
                plt.bar(range(len(values)), values.values)
                plt.xticks(range(len(values)), labels, rotation=30, ha="right", fontsize=7)
                plt.ylabel("value")
                plt.title("Pipeline Effectiveness Overview")
                chart_path = charts_dir / "effectiveness_overview.png"
                _save_fig(plt, str(chart_path))
                out["effectiveness_overview.png"] = str(chart_path.relative_to(results_dir))
        if plt is not None and benchmark_aggregate_csv is not None and Path(benchmark_aggregate_csv).exists():
            bench_df = pd.read_csv(benchmark_aggregate_csv)
            if not bench_df.empty and "variant" in bench_df.columns:
                bench_df["variant"] = pd.Categorical(
                    bench_df["variant"],
                    categories=BENCHMARK_VARIANT_ORDER,
                    ordered=True,
                )
                bench_df = bench_df.sort_values("variant").reset_index(drop=True)
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                plotted = False
                if "realized_cost_with_lt_mean" in bench_df.columns:
                    axes[0].bar(
                        bench_df["variant"].astype(str),
                        pd.to_numeric(bench_df["realized_cost_with_lt_mean"], errors="coerce").fillna(0.0) / 1e6,
                        color="#4472C4",
                    )
                    axes[0].set_title("Benchmark Cost With LT")
                    axes[0].set_ylabel("Cost (M)")
                    axes[0].tick_params(axis="x", rotation=20)
                    plotted = True
                else:
                    axes[0].axis("off")
                if "total_runtime_seconds_mean" in bench_df.columns:
                    axes[1].bar(
                        bench_df["variant"].astype(str),
                        pd.to_numeric(bench_df["total_runtime_seconds_mean"], errors="coerce").fillna(0.0),
                        color="#70AD47",
                    )
                    axes[1].set_title("Benchmark Runtime")
                    axes[1].set_ylabel("Seconds")
                    axes[1].tick_params(axis="x", rotation=20)
                    plotted = True
                else:
                    axes[1].axis("off")
                if plotted:
                    chart_path = charts_dir / "benchmark_overview.png"
                    _save_fig(plt, str(chart_path))
                    out["benchmark_overview.png"] = str(chart_path.relative_to(results_dir))
    except Exception as exc:
        print(f"[EffectivenessReport] chart skipped: {exc}")

    print(f"[EffectivenessReport] wrote {csv_path}")
    return out


def write_results_readme(
    results_dir: Path,
    *,
    source_files: Optional[Dict[str, Any]] = None,
    split_counts: Optional[Dict[str, int]] = None,
    scenario_counts: Optional[Dict[str, Any]] = None,
    phase_summaries: Optional[List[Dict[str, Any]]] = None,
    runtime_breakdown: Optional[Dict[str, Any]] = None,
) -> Path:
    """Emit Results/README.md documenting where each output lives and how
    to read the primary reporting files. Always overwrites."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    readme = results_dir / "README.md"
    lines: List[str] = []
    lines.append("# IRP-LT + BiGAT — Results Layout")
    lines.append("")
    lines.append("This folder is the single source of truth for a pipeline run.")
    lines.append("Each sub-folder corresponds to one pipeline stage. File names inside")
    lines.append("a folder are short — the folder already tells you which stage/split.")
    lines.append("")
    if source_files:
        lines.append("## Source files")
        for k, v in source_files.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")
    if split_counts:
        lines.append("## Graph dataset splits (from `graphs/graph_dataset_summary.json`)")
        for split, n in split_counts.items():
            lines.append(f"- **{split}**: {n} samples")
        lines.append("")
    if scenario_counts:
        lines.append("## Teacher scenarios")
        for k, v in scenario_counts.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    if phase_summaries:
        lines.append("## Phase-level headline metrics")
        def _fmt(v: Any, spec: str) -> str:
            try:
                return format(float(v), spec)
            except (TypeError, ValueError):
                return "NA"
        for s in phase_summaries:
            lines.append(
                f"- **{s.get('phase', '?')}**  "
                f"cost_no_lt_M={_fmt(s.get('cost_no_lt_M'), '.3f')}  "
                f"cost_with_lt_M={_fmt(s.get('cost_with_lt_M'), '.3f')}  "
                f"lt_saving_M={_fmt(s.get('lt_saving_M'), '.3f')}  "
                f"runtime_sec={_fmt(s.get('runtime_sec'), '.1f')}"
            )
        lines.append("")
    if runtime_breakdown:
        lines.append("## Runtime categories (do not mix)")
        for k, v in runtime_breakdown.items():
            try:
                lines.append(f"- **{k}**: {float(v):.2f} s")
            except (TypeError, ValueError):
                lines.append(f"- **{k}**: {v}")
        lines.append("")
    lines.append("## Directory map")
    lines.append("")
    lines.append("| Folder | What it contains | How to read it |")
    lines.append("|---|---|---|")
    lines.append("| `scenarios/` | teacher-collection runs (one CG solve per scenario) | `scenarios_manifest.json` lists every scenario; `aggregate_teacher_rows.csv` is the concatenated teacher table |")
    lines.append("| `teacher/` | canonical teacher CSV feeding GNN graph builder | `teacher_rows.csv` |")
    lines.append("| `graphs/` | GNN graph dataset split sizes | `graph_dataset_summary.json` |")
    lines.append("| `gnn/` | BiGAT training artifacts | `training_history.csv`, `training_summary.json` |")
    lines.append("| `gnn/offline_test/` | Offline held-out GNN ranking metrics | `test_per_sample.csv` (ranking-only, **not** solver runtime) |")
    lines.append("| `phase1_offline_baseline/` | Phase 1: ALNS + classical CG on training data | `baseline_cost_breakdown.csv`, `realized_cost_breakdown.csv`, `lt_plan.csv`, `validation_comparison.csv`, `validation_diagnostics.json` |")
    lines.append("| `phase2_online_inference/` | Phase 2: pre-trained GNN scores columns during CG on test data | same file set as phase 1 |")
    lines.append("| `phase3_online_learning/` | Phase 3 (optional): collects NEW teacher rows on test data + fine-tunes checkpoint | same file set |")
    lines.append("| `benchmark/` | external A0/A/B/C benchmark on held-out test data | `comparison_per_run.csv` (per repeat), `comparison_aggregate.csv` (mean/std) |")
    lines.append("| `charts/` | every PNG the pipeline produces | filenames suffix the phase label |")
    lines.append("| `thesis_summary/` | the files examiners read first | `phase_comparison.csv`, `benchmark_comparison.csv` (aggregate), `effectiveness_report.csv` |")
    lines.append("| `<phase>/debug/` | LP-relaxed (fractional) copies of every CSV whose main version was integer-rounded | same filenames; diff for audit |")
    lines.append("")
    lines.append("## Which file answers which question?")
    lines.append("")
    lines.append("| Question | File |")
    lines.append("|---|---|")
    lines.append("| Is GNN training converging? | `gnn/training_history.csv` + `charts/gnn_training_loss_*.png` |")
    lines.append("| How well does GNN rank columns on held-out graphs? | `gnn/offline_test/test_per_sample.csv` (MRR, NDCG, top-k, adaptive-F1) |")
    lines.append("| Does GNN-guided CG improve cost vs classical CG on test data? | `benchmark/comparison_aggregate.csv` (mean/std over repeats) |")
    lines.append("| How much runtime do we pay for GNN guidance? | `benchmark/comparison_aggregate.csv` → `total_runtime_seconds_{mean,std}` |")
    lines.append("| Does Phase 2 (GNN embedded) match actual inventory? | `phase2_online_inference/validation_comparison.csv` + `validation_diagnostics.json` |")
    lines.append("| Final A0/A/B/C end-to-end comparison | `thesis_summary/benchmark_comparison.csv` (aggregate of per-run) |")
    lines.append("| Single examiner-readable scoreboard | `thesis_summary/effectiveness_report.csv` |")
    lines.append("")
    lines.append("## Runtime reporting — three categories, never mixed")
    lines.append("")
    lines.append("1. **Offline GNN test runtime** — model forward-pass only, no solver. Reported by `GNN/04_test.py` and saved as `gnn/offline_test/test_runtime_seconds.json`.")
    lines.append("2. **External online inference runtime** — end-to-end Phase 2 solve on test data with GNN embedded in CG. Reported under `phase2_online_inference/` efficiency metrics and in `thesis_summary/runtime_breakdown.json`.")
    lines.append("3. **External A0/A/B/C benchmark runtime** — wall-clock of each variant's end-to-end solve on the test data. Reported in `benchmark/comparison_per_run.csv` (`total_runtime_seconds`) and aggregated in `benchmark/comparison_aggregate.csv`.")
    lines.append("")
    lines.append("## Integer vs LP-relaxed outputs")
    lines.append("")
    lines.append("Primary CSVs (`lt_plan.csv`, `validation_comparison.csv`, etc.) show **integer** quantities for shipment, inventory, shortage, demand. The fractional LP-relaxed copies are preserved under `<phase>/debug/` so examiners can audit the relaxation gap. Set `IRP_INTEGER_FINAL_OUTPUTS=0` to disable rounding.")
    lines.append("")
    readme.write_text("\n".join(lines))
    print(f"[README] wrote {readme}")
    return readme


def run_three_way_benchmark(
    data: "IRPData",
    *,
    cg_iterations: int,
    time_limit: Optional[int],
    bp_max_nodes: int,
    bp_max_depth: int,
    gnn_checkpoint_path: str,
    demand_shock_seed: int,
    demand_shock_probability: float,
    demand_shock_reallocation_fraction: float,
    demand_shock_reallocations_per_product_period: int,
    demand_shock_non_dispatch_multiplier: float,
    lt_activation_threshold: float,
    heuristic_top_k: int = 20,
    enforce_integer_flows: bool = False,
    n_repeats: int = 3,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run A0/A/B/C benchmark variants on the same data.

    Each variant runs `n_repeats` times with a different demand-shock seed per
    repeat (derived deterministically from the base seed). Outputs:
      - <results_dir>/benchmark/comparison_per_run.csv  (one row per (variant, repeat))
      - <results_dir>/benchmark/comparison_aggregate.csv  (mean/std per variant)
      - <results_dir>/thesis_summary/benchmark_comparison.csv  (the aggregate, for reporting)
      - <results_dir>/benchmark/variant_<name>_summary.json     (one per variant)
    Returns the per-run DataFrame.
    CG stopping mode is forced to "convergence" so each variant stops only when
    no negative reduced-cost column can be priced — a fixed iteration budget
    would bias runtime comparisons.
    """
    if results_dir is None:
        results_dir = Path(RESULTS_DIR)
    results_dir = Path(results_dir)
    # Force convergence-based CG termination for the entire benchmark so no
    # variant is artificially clipped by max_iter. Restore on exit.
    _prior_stop_mode = os.environ.get("IRP_CG_STOPPING_MODE")
    os.environ["IRP_CG_STOPPING_MODE"] = "convergence"
    try:
        variants: List[Tuple[str, Dict[str, Any]]] = [
            # A0: pure CG with exact Gurobi pricing — no pruning, no Stackelberg,
            # no GNN, no top-k heuristic. One exact pricing MIP is solved per
            # active (product, period); this is the reference variant that A/B/C
            # are measured against.
            ("A0_cg_full_exact", {"use_gnn": False, "collect_teacher_mode": False,
                                    "runtime_gnn_mode": False, "heuristic_top_k_mode": False,
                                    "exact_full_mode": True}),
            ("A_classical_cg",  {"use_gnn": False, "collect_teacher_mode": False,
                                  "runtime_gnn_mode": False, "heuristic_top_k_mode": False}),
            ("B_heuristic_cg",  {"use_gnn": False, "collect_teacher_mode": False,
                                  "runtime_gnn_mode": False, "heuristic_top_k_mode": True,
                                  "heuristic_top_k": heuristic_top_k}),
            ("C_gnn_guided_cg", {"use_gnn": True,  "collect_teacher_mode": False,
                                  "runtime_gnn_mode": True,  "heuristic_top_k_mode": False}),
        ]
        # Same seed sequence is reused across variants so variant differences
        # aren't confounded by different demand realizations.
        n_repeats = max(1, int(n_repeats))
        # IRP_BENCHMARK_FIXED_SHOCK=1 pins the SAME demand-shock seed across
        # every repeat (and therefore every variant), guaranteeing the
        # benchmark_comparison.csv compares algorithms on bit-identical
        # realized-demand instances. Used for online-inference test runs.
        fixed_shock = os.environ.get("IRP_BENCHMARK_FIXED_SHOCK", "0").lower() not in {"0", "false", "no", ""}
        if fixed_shock:
            seeds = [int(demand_shock_seed)] * n_repeats
            print(f"\n[Benchmark] FIXED SHOCK MODE — single seed {demand_shock_seed} "
                  f"reused across all {n_repeats} repeats")
        else:
            seeds = [int(demand_shock_seed) + 10007 * r for r in range(n_repeats)]
        print(f"\n[Benchmark] {len(variants)} variants × {n_repeats} repeats — "
              f"seeds={seeds}  stopping_mode=convergence  fixed_shock={fixed_shock}")
        rows: List[Dict[str, Any]] = []
        for variant_name, variant_kwargs in variants:
            for repeat_idx, seed in enumerate(seeds):
                run_label = f"{variant_name}__repeat{repeat_idx + 1}"
                print("\n" + "#" * 80)
                print(f"# BENCHMARK VARIANT: {variant_name}  repeat {repeat_idx + 1}/{n_repeats}  seed={seed}")
                print("#" * 80)
                t0 = time.perf_counter()
                try:
                    variant_results = IRPResearchPipeline(data).run(
                        use_random_initial_patterns=True,
                        n_initial_patterns_per_product_period=5,
                        cg_iterations=cg_iterations,
                        msg=False,
                        time_limit=time_limit,
                        enforce_integer_flows=enforce_integer_flows,
                        gnn_checkpoint=gnn_checkpoint_path,
                        use_classical_fallback=False,
                        gnn_mass_threshold=0.55,
                        gnn_max_keep=150,
                        gnn_max_keep_fraction=0.30,
                        use_branch_and_price=True,
                        bp_max_nodes=bp_max_nodes,
                        bp_max_depth=bp_max_depth,
                        lt_activation_threshold=lt_activation_threshold,
                        demand_shock_probability=demand_shock_probability,
                        demand_shock_reallocation_fraction=demand_shock_reallocation_fraction,
                        demand_shock_reallocations_per_product_period=demand_shock_reallocations_per_product_period,
                        demand_shock_non_dispatch_multiplier=demand_shock_non_dispatch_multiplier,
                        demand_shock_seed=seed,
                        **variant_kwargs,
                    )
                except Exception as exc:
                    print(f"[Benchmark] Variant {run_label} FAILED: {exc}")
                    rows.append({
                        "variant": variant_name, "run_label": run_label,
                        "repeat": repeat_idx + 1,
                        "source_instance": os.environ.get("IRP_SOURCE_INSTANCE", ""),
                        "seed": seed,
                        "benchmark_mode": "strict" if os.environ.get("IRP_STRICT_BENCHMARK", "0").lower() not in {"0","false","no",""} else "default",
                        "rmp_objective": float("nan"),
                        "realized_cost_no_lt": float("nan"), "realized_cost_with_lt": float("nan"),
                        "lt_cost_with_lt": float("nan"), "shortage_cost_with_lt": float("nan"),
                        "cg_iterations": 0, "total_runtime_seconds": time.perf_counter() - t0,
                        "columns_generated": 0, "columns_added_to_rmp": 0,
                        "column_pool_utilization": 0.0, "error": str(exc),
                        "stopping_reason": "failed",
                    })
                    continue
                runtime = time.perf_counter() - t0
                run_context = {
                    "run_label": run_label,
                    "source_instance": os.environ.get("IRP_SOURCE_INSTANCE", ""),
                    "dataset_id": getattr(data, "dataset_id", "") or "",
                    "scenario_id": str(seed),
                    "seed": seed,
                    "benchmark_mode": "strict" if os.environ.get("IRP_STRICT_BENCHMARK", "0").lower() not in {"0","false","no",""} else "default",
                    "fine_tune_runtime_seconds": 0.0,
                }
                row = _collect_benchmark_metrics(variant_name, variant_results, runtime, run_context=run_context)
                row["repeat"] = repeat_idx + 1
                rows.append(row)
                print(f"[Benchmark] {run_label}: obj={row['rmp_objective']:.2f} "
                      f"cost_with_lt={row['realized_cost_with_lt']:.2f} runtime={runtime:.1f}s "
                      f"cols_gen={row['columns_generated']} cols_added={row['columns_added_to_rmp']}")

        per_run_df = pd.DataFrame(rows)
        if not per_run_df.empty and "variant" in per_run_df.columns:
            per_run_df["variant"] = pd.Categorical(
                per_run_df["variant"],
                categories=BENCHMARK_VARIANT_ORDER,
                ordered=True,
            )
            sort_cols = ["variant"] + [c for c in ("repeat", "run_label") if c in per_run_df.columns]
            per_run_df = per_run_df.sort_values(sort_cols).reset_index(drop=True)
        benchmark_dir = results_dir / BENCHMARK_SUBDIR
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        per_run_path = benchmark_dir / "comparison_per_run.csv"
        per_run_df.to_csv(per_run_path, index=False)

        # Aggregate: mean + std per variant for numeric columns.
        numeric_cols = [
            "rmp_objective", "realized_cost_no_lt", "realized_cost_with_lt",
            "lt_cost_with_lt", "shortage_cost_with_lt",
            "cg_iterations", "total_runtime_seconds",
            "phase1_baseline_runtime_seconds", "phase2_cg_runtime_seconds",
            "gnn_inference_runtime_seconds",
            "columns_generated", "columns_selected_by_gnn", "columns_added_to_rmp",
            "column_pool_utilization", "gnn_scoring_failures",
        ]
        numeric_cols = [c for c in numeric_cols if c in per_run_df.columns]
        agg_rows: List[Dict[str, Any]] = []
        for variant_name, grp in per_run_df.groupby("variant", sort=False, observed=False):
            row = {"variant": variant_name, "n_repeats_successful": int(len(grp))}
            for c in numeric_cols:
                numeric = pd.to_numeric(grp[c], errors="coerce")
                row[f"{c}_mean"] = float(numeric.mean())
                row[f"{c}_std"]  = float(numeric.std(ddof=0)) if len(numeric) else float("nan")
                row[f"{c}_min"]  = float(numeric.min()) if len(numeric) else float("nan")
                row[f"{c}_max"]  = float(numeric.max()) if len(numeric) else float("nan")
            agg_rows.append(row)
        aggregate_df = pd.DataFrame(agg_rows)
        if not aggregate_df.empty and "variant" in aggregate_df.columns:
            aggregate_df["variant"] = pd.Categorical(
                aggregate_df["variant"],
                categories=BENCHMARK_VARIANT_ORDER,
                ordered=True,
            )
            aggregate_df = aggregate_df.sort_values("variant").reset_index(drop=True)
        aggregate_path = benchmark_dir / "comparison_aggregate.csv"
        aggregate_df.to_csv(aggregate_path, index=False)

        # Also keep the legacy filename so existing readers don't break.
        per_run_df.to_csv(benchmark_dir / "comparison.csv", index=False)

        thesis_dir = results_dir / THESIS_SUMMARY_SUBDIR
        thesis_dir.mkdir(parents=True, exist_ok=True)
        aggregate_df.to_csv(thesis_dir / "benchmark_comparison.csv", index=False)

        # Per-variant JSON summaries for quick inspection
        for variant_name, grp in per_run_df.groupby("variant", sort=False, observed=False):
            summary = {
                "variant": variant_name,
                "n_repeats": int(len(grp)),
                "runs": grp.to_dict(orient="records"),
            }
            with open(benchmark_dir / f"variant_{variant_name}_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)

        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON SUMMARY (per-run)")
        print("=" * 80)
        print(per_run_df.to_string(index=False))
        print("\n[Aggregate]")
        print(aggregate_df.to_string(index=False))
        print(f"\nSaved per-run comparison to : {per_run_path}")
        print(f"Saved aggregate comparison to: {aggregate_path}")
        return per_run_df
    finally:
        if _prior_stop_mode is None:
            os.environ.pop("IRP_CG_STOPPING_MODE", None)
        else:
            os.environ["IRP_CG_STOPPING_MODE"] = _prior_stop_mode


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # ---- Three-stage pipeline -----------------------------------------------
    #
    # Stage 1  OFFLINE TRAINING  (default)
    #   IRP_ONLINE_INFERENCE=0
    #   Dataset : training CSV (1BISCR501V_...)
    #   Flow    : CG episodes → collect teacher rows → build graph samples
    #             → train BiGAT from scratch → offline test on held-out split
    #   GNN role: not used during CG; model produced as output
    #
    # Stage 2  ONLINE INFERENCE  (IRP_ONLINE_INFERENCE=1)
    #   Dataset : test data.csv
    #   Flow    : load pre-trained checkpoint → GNN scores columns during CG
    #             → no new teacher collection, no retraining
    #   Why this is the thesis default for Stage 2: the offline-trained model
    #   is evaluated on a fixed checkpoint for reproducible examiner comparison.
    #   Online learning introduces non-determinism and should only be activated
    #   after the model has been validated in inference-only mode first.
    #
    # Stage 3  POST-RUN FINE-TUNE   (IRP_ONLINE_INFERENCE=1 + IRP_ONLINE_LEARNING=1)
    #   Historical note: the env flag is named IRP_ONLINE_LEARNING for backward
    #   compatibility, but this stage is NOT true in-loop online learning — GNN
    #   weights are frozen throughout CG and updated exactly once, in a batch
    #   fine-tune pass at the end of the run. True in-loop online learning was
    #   deliberately not implemented because (a) reproducibility for thesis
    #   evaluation requires a frozen model across CG iterations, and (b) the
    #   per-episode gradient signal is extremely noisy.
    #   Dataset : test data.csv
    #   Supervision : solver-supervised — teacher labels come from RMP dual
    #                 variables (lambda > 1e-6 → label=1) and reduced costs
    #                 during inference CG episodes; no human annotation needed.
    #   Sample collection: same teacher-row export as Stage 1, but on inference
    #                 instances. Samples are appended to the existing teacher CSV
    #                 so the fine-tune sees both old and new distributions.
    #   Update frequency: once per full pipeline run (end-of-run fine-tune).
    #                 Use IRP_ONLINE_LEARNING_EPOCHS (default 2) to control
    #                 how many gradient steps are taken; keep small (1-3) to
    #                 avoid overfitting to a single inference run.
    #   Checkpoint: resume_checkpoint=True — weights are updated in-place;
    #                 the original offline checkpoint is overwritten, so back
    #                 it up manually before enabling this mode.
    # -------------------------------------------------------------------------
    _TRAINING_DATASET = Path(__file__).with_name("1BISCR501V_90100140_20260323-150407111_filtered_sites.csv")
    _INFERENCE_DATASET = Path(__file__).with_name("test data.csv")

    _online_inference = os.environ.get("IRP_ONLINE_INFERENCE", "0").lower() not in {"0", "false", "no"}
    _online_learning  = (
        _online_inference
        and os.environ.get("IRP_ONLINE_LEARNING", "0").lower() not in {"0", "false", "no"}
    )
    EXCEL_PATH = Path(os.environ.get("IRP_DATASET_PATH", str(_INFERENCE_DATASET if _online_inference else _TRAINING_DATASET)))

    if _online_learning:
        print(
            "[Mode] POST-RUN FINE-TUNE (legacy name: ONLINE LEARNING) — test data.csv; "
            "GNN scores during CG AND new teacher rows are collected; model is fine-tuned "
            "(solver-supervised, resume_checkpoint=True) at END of run. This is NOT true "
            "in-loop online backprop — weights are frozen during CG and updated in one "
            "batch pass after the final iteration. See docstring above for rationale."
        )
    elif _online_inference:
        print("[Mode] ONLINE INFERENCE — test data.csv; pre-trained GNN checkpoint loaded, no retraining during CG.")
    else:
        print("[Mode] OFFLINE TRAINING — training dataset; CG → teacher rows → GNN train/validate.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("IRP_CLEAN_RESULTS", "1").lower() not in {"0", "false", "no"}:
        removed = clean_managed_outputs(RESULTS_DIR)
        if removed:
            print(f"[Cleanup] Removed {len(removed)} stale output sub-folders/files in {RESULTS_DIR}")
        else:
            print(f"[Cleanup] No stale managed outputs found in {RESULTS_DIR}")

    # Resolve layout (root + phase) once so every output helper shares the
    # same target folders. Phase label is derived from the mode env vars
    # (see resolve_phase_label()) unless IRP_PHASE_LABEL overrides it.
    layout = build_results_layout(RESULTS_DIR)
    print(f"[Layout] Results root = {layout.root}  |  phase = {layout.phase_label}")

    # Resolve all per-component seeds from IRP_MASTER_SEED (or individual env
    # overrides) and write Results/seed_manifest.json so every run's seed
    # state is auditable. This runs *after* clean_managed_outputs so the
    # manifest is not deleted, and *before* mapper construction so downstream
    # subsystems see the resolved seed env vars.
    _seed_manifest = resolve_seed_manifest(RESULTS_DIR)

    mapper = DatasetToIRPValidationMapper(
        excel_path=EXCEL_PATH,
        sheet_name="Sheet1",
        store_limit=int(os.environ.get("IRP_STORE_LIMIT", "10")),
        sku_limit=int(os.environ.get("IRP_SKU_LIMIT", "5")),
        start_date=os.environ.get("IRP_START_DATE"),
        end_date=os.environ.get("IRP_END_DATE"),
    )

    data, base_df, validation_target, meta = mapper.build_irp_data(
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
        store_initial_inventory_multiplier=float(os.environ.get("IRP_STORE_INIT_MULTIPLIER", "0.2")),
        lt_cost_multiplier=float(os.environ.get("IRP_LT_COST_MULTIPLIER", "1.0")),
    )

    # Attach identity tags so downstream logging / benchmark CSV / teacher rows
    # all agree on which base dataset + scenario this run belongs to. Set via
    # env by the scenario generator; fall back to the dataset filename stem.
    data.dataset_id = os.environ.get("IRP_DATASET_ID") or EXCEL_PATH.stem
    data.scenario_id = os.environ.get("IRP_SCENARIO_ID", "")

    print("Mapped dataset metadata:")
    pprint.pprint(meta)
    print(f"[Identity] dataset_id={data.dataset_id} | scenario_id={data.scenario_id}")

    # Cost unit diagnostic — helps detect scale mismatch between cost components.
    # Shortage cost and LT cost should be on the same economic scale for valid optimisation.
    if data.stores and data.products and data.periods:
        _s0, _p0, _t0 = next(iter(data.stores)), next(iter(data.products)), next(iter(data.periods))
        _shortage_unit = data.shortage_cost.get((_s0, _p0), float("nan"))
        _holding_unit  = data.holding_cost_store.get((_s0, _p0), float("nan"))
        _lt_pairs = [(i, j) for i in data.stores for j in data.stores if i != j]
        _lt_unit = float(sum(data.transship_unit_cost.get((i, j), 0.0) for i, j in _lt_pairs) / max(1, len(_lt_pairs)))
        _lt_cost_mult = float(os.environ.get("IRP_LT_COST_MULTIPLIER", "1.0"))
        print(
            f"\n[Cost Unit Diagnostic] (representative values — sample store={_s0}, sku={_p0})\n"
            f"  shortage_unit_cost      = {_shortage_unit:.4f}  (shortage_cost_rate * price)\n"
            f"  holding_unit_cost       = {_holding_unit:.4f}  (holding_cost_rate * price)\n"
            f"  lt_unit_cost (avg)      = {_lt_unit:.4f}  (0.01*alpha*distance*lt_cost_multiplier)\n"
            f"  lt_cost_multiplier      = {_lt_cost_mult:.2f}x  (env IRP_LT_COST_MULTIPLIER)\n"
            f"  lt/shortage ratio       = {_lt_unit / max(1e-12, _shortage_unit):.6f}\n"
            f"  *** If lt/shortage << 1, LT is effectively free — examiners will challenge results. ***\n"
            f"  *** For thesis sensitivity analysis, re-run with IRP_LT_COST_MULTIPLIER=5,10,25,50. ***"
        )

    # validation_target is held in memory; it is merged into each phase's
    # validation_comparison.csv by save_phase_artifacts(). No standalone copy is
    # written (used to be Results/irp_validation_target.csv) — the target is a
    # direct slice of the input CSV and adds no information on its own.

    # Flag defaults per mode:
    #   offline training : collect=T  runtime_gnn=F  train=T  resume=F  graphs=T
    #   online inference : collect=F  runtime_gnn=T  train=F  resume=F  graphs=F
    #   online learning  : collect=T  runtime_gnn=T  train=T  resume=T  graphs=T
    if _online_learning:
        _default_collect    = "1"
        _default_runtime_gnn = "1"
        _default_train_after = "1"
        _default_build_graphs = "1"
        _default_resume     = "1"   # fine-tune in-place, never retrain from scratch
    elif _online_inference:
        _default_collect    = "0"
        _default_runtime_gnn = "1"
        _default_train_after = "0"
        _default_build_graphs = "0"
        _default_resume     = "0"
    else:  # offline training
        _default_collect    = "1"
        _default_runtime_gnn = "0"
        _default_train_after = "1"
        _default_build_graphs = "1"
        _default_resume     = "0"

    collect_teacher_mode  = os.environ.get("IRP_COLLECT_TEACHER_MODE",    _default_collect).lower()     not in {"0", "false", "no"}
    runtime_gnn_mode      = os.environ.get("IRP_RUNTIME_GNN_MODE",        _default_runtime_gnn).lower() not in {"0", "false", "no"}
    use_branch_and_price  = os.environ.get("IRP_USE_BRANCH_AND_PRICE",    "1").lower()                  not in {"0", "false", "no"}
    build_teacher_graphs  = os.environ.get("IRP_BUILD_TEACHER_GRAPHS",    _default_build_graphs).lower() not in {"0", "false", "no"}
    train_gnn_after_teacher = os.environ.get("IRP_TRAIN_GNN_AFTER_TEACHER", _default_train_after).lower() not in {"0", "false", "no"}
    # Online learning uses fewer epochs (1-3) to avoid overfitting to a single run.
    _default_epochs = str(int(os.environ.get("IRP_ONLINE_LEARNING_EPOCHS", "2"))) if _online_learning else "5"
    gnn_train_epochs = int(os.environ.get("IRP_GNN_TRAIN_EPOCHS", _default_epochs))
    lt_activation_threshold = float(os.environ.get("IRP_LT_ACTIVATION_THRESHOLD", "10.0"))
    demand_shock_probability = float(os.environ.get("IRP_DEMAND_SHOCK_PROBABILITY", "0.85"))
    demand_shock_reallocation_fraction = float(os.environ.get("IRP_DEMAND_SHOCK_REALLOCATION_FRACTION", "0.60"))
    demand_shock_reallocations_per_product_period = int(os.environ.get("IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD", "3"))
    demand_shock_non_dispatch_multiplier = float(os.environ.get("IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER", "1.8"))
    demand_shock_seed = int(os.environ.get("IRP_DEMAND_SHOCK_SEED", "20260418"))
    resume_gnn_checkpoint = os.environ.get("IRP_RESUME_GNN_CHECKPOINT", _default_resume).lower() not in {"0", "false", "no"}
    gnn_checkpoint_path = os.environ.get("IRP_GNN_CHECKPOINT", DEFAULT_GNN_CHECKPOINT)

    env_time_limit = os.environ.get("IRP_TIME_LIMIT")
    enforce_integer_flows = os.environ.get("IRP_ENFORCE_INTEGER", "0").lower() not in {"0", "false", "no"}

    # ----- A0/A/B/C benchmark short-circuit -----
    if os.environ.get("IRP_RUN_BENCHMARK", "0").lower() not in {"0", "false", "no"}:
        heuristic_top_k_env = int(os.environ.get("IRP_HEURISTIC_TOP_K", "20"))
        run_three_way_benchmark(
            data=data,
            cg_iterations=int(os.environ.get("IRP_CG_ITERATIONS", "15")),
            time_limit=int(env_time_limit) if env_time_limit else None,
            bp_max_nodes=int(os.environ.get("IRP_BP_MAX_NODES", "15")),
            bp_max_depth=int(os.environ.get("IRP_BP_MAX_DEPTH", "6")),
            gnn_checkpoint_path=gnn_checkpoint_path,
            demand_shock_seed=demand_shock_seed,
            demand_shock_probability=demand_shock_probability,
            demand_shock_reallocation_fraction=demand_shock_reallocation_fraction,
            demand_shock_reallocations_per_product_period=demand_shock_reallocations_per_product_period,
            demand_shock_non_dispatch_multiplier=demand_shock_non_dispatch_multiplier,
            lt_activation_threshold=lt_activation_threshold,
            heuristic_top_k=heuristic_top_k_env,
            enforce_integer_flows=enforce_integer_flows,
        )
        print("\n[Benchmark] IRP_RUN_BENCHMARK=1 — exiting after A0/A/B/C comparison.")
        sys.exit(0)

    results = IRPResearchPipeline(data).run(
        use_random_initial_patterns=True,
        n_initial_patterns_per_product_period=5,
        cg_iterations=int(os.environ.get("IRP_CG_ITERATIONS", "15")),
        msg=False,
        time_limit=int(env_time_limit) if env_time_limit else None,
        enforce_integer_flows=enforce_integer_flows,
        use_gnn=runtime_gnn_mode,
        collect_teacher_mode=collect_teacher_mode,
        runtime_gnn_mode=runtime_gnn_mode,
        gnn_checkpoint=gnn_checkpoint_path,
        use_classical_fallback=False,
        gnn_mass_threshold=0.55,
        gnn_max_keep=150,
        gnn_max_keep_fraction=0.30,
        use_branch_and_price=use_branch_and_price,
        bp_max_nodes=int(os.environ.get("IRP_BP_MAX_NODES", "15")),
        bp_max_depth=int(os.environ.get("IRP_BP_MAX_DEPTH", "6")),
        lt_activation_threshold=lt_activation_threshold,
        demand_shock_probability=demand_shock_probability,
        demand_shock_reallocation_fraction=demand_shock_reallocation_fraction,
        demand_shock_reallocations_per_product_period=demand_shock_reallocations_per_product_period,
        demand_shock_non_dispatch_multiplier=demand_shock_non_dispatch_multiplier,
        demand_shock_seed=demand_shock_seed,
    )

    # ------------------------------------------------------------------
    # Phase 1 per-phase artifacts
    # ------------------------------------------------------------------
    # save_phase_artifacts() writes every CSV/JSON for this phase into
    # Results/<phase_label>/ using short, folder-scoped filenames. The phase
    # folder itself identifies the split, so filenames stay neutral
    # (baseline_cost_breakdown.csv, lt_plan.csv, ...). It also emits the CG
    # cost-curve PNG into Results/charts/cg_cost_curve_<phase>.png.
    phase1_artifacts = save_phase_artifacts(
        results,
        layout,
        layout.phase_label,
        validation_target=validation_target,
    )
    print(
        f"[Phase 1] Wrote {len(phase1_artifacts)} artifacts to "
        f"{(layout.root / layout.phase_label)}"
    )

    refreshed_history: List[Dict[str, Any]] = []
    phase2_artifacts: Optional[Dict[str, str]] = None
    teacher_dataset_path = layout.teacher / "teacher_rows.csv"
    layout.ensure(layout.teacher)
    teacher_dataset_df = pd.DataFrame(results["teacher_dataset_rows"])
    if teacher_dataset_df.empty:
        print(
            "No CG teacher dataset rows were generated in this run; "
            "skipping teacher CSV overwrite, graph rebuild, and GNN training update."
        )
    else:
        teacher_graph_input_path = _write_teacher_dataset_exports(teacher_dataset_df, teacher_dataset_path)
        if collect_teacher_mode:
            refreshed_history = run_teacher_graph_and_gnn_training(
                teacher_csv_path=str(teacher_graph_input_path),
                build_graphs=build_teacher_graphs,
                train_gnn=train_gnn_after_teacher,
                train_epochs=gnn_train_epochs,
                resume_checkpoint=resume_gnn_checkpoint,
                checkpoint_path=gnn_checkpoint_path,
            )
            print("Refreshed GNN history rows:", len(refreshed_history))

            deploy_gnn_after_training = os.environ.get(
                "IRP_DEPLOY_GNN_AFTER_TRAINING", "1"
            ).lower() not in {"0", "false", "no"}
            checkpoint_ready = _project_path(gnn_checkpoint_path).exists()
            if deploy_gnn_after_training and refreshed_history and checkpoint_ready:
                print("\n" + "=" * 80)
                print("PHASE 2 - Deploy trained BiGAT in CG (use_gnn=True, collect_teacher_mode=False)")
                print("=" * 80)
                gnn_data, _, _, _ = mapper.build_irp_data(
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
                    store_initial_inventory_multiplier=float(
                        os.environ.get("IRP_STORE_INIT_MULTIPLIER", "0.2")
                    ),
                    lt_cost_multiplier=float(os.environ.get("IRP_LT_COST_MULTIPLIER", "1.0")),
                )
                gnn_results = IRPResearchPipeline(gnn_data).run(
                    use_random_initial_patterns=True,
                    n_initial_patterns_per_product_period=5,
                    cg_iterations=int(os.environ.get("IRP_CG_ITERATIONS", "15")),
                    msg=False,
                    time_limit=int(env_time_limit) if env_time_limit else None,
                    enforce_integer_flows=enforce_integer_flows,
                    use_gnn=True,
                    collect_teacher_mode=False,
                    runtime_gnn_mode=True,
                    gnn_checkpoint=gnn_checkpoint_path,
                    use_classical_fallback=True,
                    gnn_mass_threshold=0.55,
                    gnn_max_keep=150,
                    gnn_max_keep_fraction=0.30,
                    use_branch_and_price=use_branch_and_price,
                    bp_max_nodes=int(os.environ.get("IRP_BP_MAX_NODES", "15")),
                    bp_max_depth=int(os.environ.get("IRP_BP_MAX_DEPTH", "6")),
                    lt_activation_threshold=lt_activation_threshold,
                    demand_shock_probability=demand_shock_probability,
                    demand_shock_reallocation_fraction=demand_shock_reallocation_fraction,
                    demand_shock_reallocations_per_product_period=demand_shock_reallocations_per_product_period,
                    demand_shock_non_dispatch_multiplier=demand_shock_non_dispatch_multiplier,
                    demand_shock_seed=demand_shock_seed,
                )

                phase2_label = "phase2_deploy_after_teacher"
                phase2_artifacts = save_phase_artifacts(
                    gnn_results,
                    layout,
                    phase2_label,
                    validation_target=validation_target,
                )
                print(
                    f"[Phase 2] Wrote {len(phase2_artifacts)} artifacts to "
                    f"{(layout.root / phase2_label)}"
                )

                # Classical-vs-GNN phase comparison → thesis_summary/ only
                # (used to live as Results/irp_phase_comparison.csv in the flat
                # layout). The charts reader below picks it up from here.
                layout.ensure(layout.thesis)
                phase_cmp_out = layout.thesis / "phase_comparison.csv"
                pd.DataFrame([
                    {"phase": "teacher_collection_classical_cg", **results["comparison"]},
                    {"phase": "gnn_deployed_cg",                 **gnn_results["comparison"]},
                ]).to_csv(phase_cmp_out, index=False)
                print(f"Saved phase comparison (classical vs GNN) to: {phase_cmp_out}")
            elif deploy_gnn_after_training and not checkpoint_ready:
                print(
                    f"[Phase 2] Skipped GNN deployment: checkpoint not found at "
                    f"{_project_path(gnn_checkpoint_path)}"
                )
            elif deploy_gnn_after_training and not refreshed_history:
                print(
                    "[Phase 2] Skipped GNN deployment: training produced no history rows "
                    "(teacher graph dataset was likely empty)."
                )

    # Validation metrics from the Phase 1 validation_comparison.csv (already
    # written inside save_phase_artifacts). We recompute the comparison frame
    # here only to print summary stats to stdout — no duplicate file.
    try:
        predicted_df = build_predicted_inventory_df(results["baseline_solution"])
        comparison_df = predicted_df.merge(
            validation_target, on=["store", "sku", "period"], how="inner"
        )
        comparison_df["error"] = (
            comparison_df["predicted_end_qty"] - comparison_df["actual_end_qty"]
        )
        metrics = compute_validation_metrics(comparison_df)
        print("\nValidation metrics:")
        pprint.pprint(metrics)
    except Exception as exc:
        print(f"[Validation] metrics unavailable: {exc}")
        metrics = {}

    # ------------------------------------------------------------------
    # Pipeline charts
    # ------------------------------------------------------------------
    # Charts land in Results/charts/ with filenames suffixed by phase_label,
    # so Phase 1 and Phase 2 runs in the same Results root never clobber each
    # other. The phase_comparison chart is only produced when we actually ran
    # Phase 2 (thesis_summary/phase_comparison.csv exists).
    phase_cmp_df: Optional[pd.DataFrame] = None
    phase_cmp_path = layout.thesis / "phase_comparison.csv"
    if phase_cmp_path.exists():
        try:
            phase_cmp_df = pd.read_csv(phase_cmp_path)
        except Exception as exc:
            print(f"[Charts] Could not read {phase_cmp_path}: {exc}.")
            phase_cmp_df = None

    saved_charts = save_pipeline_charts(
        results=results,
        out_dir=layout.charts,
        refreshed_gnn_history=refreshed_history if collect_teacher_mode else None,
        phase_comparison_df=phase_cmp_df,
        phase_label=layout.phase_label,
    )
    if saved_charts:
        print(f"\nSaved {len(saved_charts)} pipeline charts to: {layout.charts}")
        for cp in saved_charts:
            print(f"  {Path(cp).name}")
    else:
        print("\n[Charts] No charts were generated (matplotlib missing or data empty).")

    # ------------------------------------------------------------------
    # Run manifest + runtime summary
    # ------------------------------------------------------------------
    # One manifest per run indexes every artifact by phase so examiners /
    # Kaggle orchestrators can locate a file without guessing the folder.
    artifacts_by_phase: Dict[str, Dict[str, str]] = {
        layout.phase_label: phase1_artifacts,
    }
    if phase2_artifacts:
        artifacts_by_phase["phase2_deploy_after_teacher"] = phase2_artifacts

    run_manifest = {
        "run_id": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "phase_label": layout.phase_label,
        "dataset_id": data.dataset_id,
        "scenario_id": data.scenario_id,
        "dataset_path": str(EXCEL_PATH),
        "mode": (
            "phase3_online_learning" if _online_learning
            else "phase2_online_inference" if _online_inference
            else "phase1_offline_baseline"
        ),
        "results_root": str(layout.root),
        "artifacts_by_phase": artifacts_by_phase,
        "validation_metrics": metrics,
        "charts": [str(Path(p).relative_to(layout.root)) for p in saved_charts],
    }
    try:
        with open(layout.run_manifest_path, "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2)
        print(f"[Manifest] Wrote {layout.run_manifest_path}")
    except OSError as exc:
        print(f"[Manifest] Could not write run_manifest.json: {exc}")

    # Concise per-phase file listing for the examiner — filename alone already
    # says what it is; the folder says which split/phase it belongs to.
    print("\n" + "=" * 72)
    print(f"Runtime summary — Results root: {layout.root}")
    print("=" * 72)
    for phase, files in artifacts_by_phase.items():
        print(f"  [{phase}]  ({len(files)} files)")
        for name in sorted(files):
            print(f"    - {files[name]}")
    if saved_charts:
        print(f"  [charts]  ({len(saved_charts)} files)")
        for cp in saved_charts:
            print(f"    - charts/{Path(cp).name}")
    print("=" * 72)

    print("\nDone.")
