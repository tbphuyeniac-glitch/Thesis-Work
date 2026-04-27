# %% [markdown]
# # Achamrah 2022 IRP-T Matheuristic (GA+SA) — Kaggle Run
# Implements the full Achamrah et al. (2022) matheuristic:
#   Phase 1: RMILP constructive heuristic (LP relaxation → cluster sub-MILPs)
#   Phase 2: GA hybridised with SA improvement
# Lateral transshipment is enabled. Outputs match thesis pipeline format.
#
# Prerequisites (Kaggle secrets): WLSACCESSID, WLSSECRET, LICENSEID
# Run all cells top-to-bottom.

# %% [code]
# !pip -q install gurobipy   # uncomment if gurobipy is not pre-installed

# =========================================================
# 0) Imports
# =========================================================
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import pprint
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = print  # type: ignore[assignment]


# =========================================================
# 1) Run configuration
# =========================================================

REPO_URL  = "https://github.com/tbphuyeniac-glitch/Thesis-Work.git"
REPO_ROOT = Path("/kaggle/working/Thesis-Work")
REFRESH_WORKING_REPO = True   # force fresh clone for clean end-to-end run

# ── Data file ────────────────────────────────────────────
DATA_FILE = "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
DATA_PATH = REPO_ROOT / DATA_FILE

RESULTS_DIR = Path("/kaggle/working/Results")

# ── Instance scope (set None for full dataset) ───────────
STORE_LIMIT: Optional[int] = 4    # reduce for faster smoke test; None = all stores
SKU_LIMIT:   Optional[int] = 5    # reduce for faster smoke test; None = all SKUs
START_DATE:  Optional[str] = "2025-08-01"
END_DATE:    Optional[str] = "2025-09-30"

# ── Solver parameters (Achamrah 2022 Table 2 calibration) ─
INITIAL_TEMPERATURE     = 92.0
FINAL_TEMPERATURE       = 4.2
COOLING_RATIO           = 0.96
CROSSOVER_PROBABILITY   = 0.84
MUTATION_PROBABILITY    = 0.37
POPULATION_SIZE         = 30
ITERATIONS_PER_TEMP     = 30
CONSTRUCTIVE_TIME_LIMIT = 120    # seconds — RMILP constructive phase
IMPROVEMENT_TIME_LIMIT  = 300    # seconds — GA+SA improvement phase
FULL_TIME_LIMIT         = 600    # seconds — fallback full MILP time limit
SEED                    = 42

# ── Instance mapping parameters ──────────────────────────
VEHICLE_COUNT              = 2
VEHICLE_CAPACITY           = 120.0
WH_INVENTORY_MULTIPLIER    = 2.5
STORE_CAPACITY_MULTIPLIER  = 1.5
SHORTAGE_COST_RATE         = 0.25
HOLDING_COST_RATE          = 0.01
ALPHA                      = 1.0
CW_REPLENISHMENT_FACTOR    = 0.6
CW_CAPACITY_FACTOR         = 2.0


# =========================================================
# 2) Repository clone
# =========================================================

def clone_repo() -> None:
    if REFRESH_WORKING_REPO and REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    if not REPO_ROOT.exists():
        stable_cwd = Path("/kaggle/working")
        r = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_ROOT)],
            capture_output=True, text=True, cwd=str(stable_cwd),
        )
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise RuntimeError("git clone failed")
        print("Cloned to:", REPO_ROOT)
    else:
        print("Using existing repo:", REPO_ROOT)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    print("sys.path updated → repo on path")


# =========================================================
# 3) Gurobi WLS license
# =========================================================

def load_gurobi_wls_secrets() -> Tuple[str, str, str]:
    """Load WLS credentials from Kaggle secrets, set env vars, and write ~/gurobi.lic."""
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()

        def _get(c, *names):
            for n in names:
                try:
                    v = c.get_secret(n)
                    if v and str(v).strip():
                        return str(v).strip()
                except Exception:
                    pass
            raise RuntimeError(f"Kaggle secret not found — tried: {names}")

        access_id  = _get(client, "WLSACCESSID",  "GRB_WLSACCESSID")
        secret     = _get(client, "WLSSECRET",    "GRB_WLSSECRET")
        license_id = _get(client, "LICENSEID",    "GRB_LICENSEID")
    except ImportError:
        # Outside Kaggle: fall back to existing ~/gurobi.lic
        lic_path = Path.home() / "gurobi.lic"
        if not lic_path.exists():
            raise RuntimeError(
                "Not on Kaggle and ~/gurobi.lic not found. "
                "Set WLSACCESSID/WLSSECRET/LICENSEID environment variables."
            )
        creds: Dict[str, str] = {}
        for line in lic_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                creds[k.strip().upper()] = v.strip()
        access_id  = creds["WLSACCESSID"]
        secret     = creds["WLSSECRET"]
        license_id = creds["LICENSEID"]

    for k, v in [("WLSACCESSID", access_id), ("WLSSECRET", secret), ("LICENSEID", license_id)]:
        os.environ[k] = v

    lic_content = (
        "# Gurobi WLS license\n"
        f"WLSACCESSID={access_id}\n"
        f"WLSSECRET={secret}\n"
        f"LICENSEID={license_id}\n"
    )
    lic_path = Path.home() / "gurobi.lic"
    lic_path.write_text(lic_content)
    print(f"Gurobi WLS credentials loaded → wrote {lic_path}")
    return access_id, secret, license_id


def verify_gurobi() -> bool:
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("WLSAccessID", os.environ["WLSACCESSID"])
        env.setParam("WLSSecret",   os.environ["WLSSECRET"])
        env.setParam("LicenseID",   int(os.environ["LICENSEID"]))
        env.setParam("OutputFlag",  0)
        env.start()
        m = gp.Model(env=env)
        m.addVar(); m.optimize()
        env.dispose()
        print("Gurobi license verified")
        return True
    except Exception as e:
        print(f"Gurobi license check failed: {e}")
        return False


# =========================================================
# 4) Main pipeline
# =========================================================

def run_achamrah_pipeline() -> Tuple[Dict[str, Any], Any]:
    from achamrah_2022_thesis_format_wrapper import (
        DatasetToAchamrahMapper,
        AchamrahThesisFormatPipeline,
    )
    from achamrah_2022_irpt_matheuristic import HeuristicParams

    print(f"\nData file : {DATA_PATH}")
    print(f"Stores    : {STORE_LIMIT or 'all'}")
    print(f"SKUs      : {SKU_LIMIT or 'all'}")
    print(f"Dates     : {START_DATE} → {END_DATE}")

    mapper = DatasetToAchamrahMapper(
        excel_path=str(DATA_PATH),
        store_limit=STORE_LIMIT,
        sku_limit=SKU_LIMIT,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print("\nBuilding IRPTInstance from dataset...")
    mapping = mapper.build_instance(
        wh_inventory_multiplier=WH_INVENTORY_MULTIPLIER,
        store_capacity_multiplier=STORE_CAPACITY_MULTIPLIER,
        shortage_cost_rate=SHORTAGE_COST_RATE,
        holding_cost_rate=HOLDING_COST_RATE,
        vehicle_count=VEHICLE_COUNT,
        vehicle_capacity=VEHICLE_CAPACITY,
        alpha=ALPHA,
        cw_replenishment_factor=CW_REPLENISHMENT_FACTOR,
        cw_capacity_factor=CW_CAPACITY_FACTOR,
    )

    print("\nInstance metadata:")
    pprint.pprint(mapping.metadata)

    params = HeuristicParams(
        initial_temperature=INITIAL_TEMPERATURE,
        final_temperature=FINAL_TEMPERATURE,
        cooling_ratio=COOLING_RATIO,
        crossover_probability=CROSSOVER_PROBABILITY,
        mutation_probability=MUTATION_PROBABILITY,
        population_size=POPULATION_SIZE,
        iterations_per_temp=ITERATIONS_PER_TEMP,
        constructive_time_limit=CONSTRUCTIVE_TIME_LIMIT,
        improvement_time_limit=IMPROVEMENT_TIME_LIMIT,
        full_time_limit=FULL_TIME_LIMIT,
        seed=SEED,
    )

    print("\nRunning Achamrah GA+SA matheuristic (LT enabled)...")
    t0 = time.time()
    pipeline = AchamrahThesisFormatPipeline(
        mapping,
        params=params,
        allow_lateral_transshipment=True,
    )
    outputs = pipeline.run()
    elapsed = time.time() - t0
    print(f"Pipeline finished in {elapsed:.1f}s")

    return outputs, pipeline


# =========================================================
# 5) Thesis-format output writers
# =========================================================

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def write_phase_artifacts(
    outputs: Dict[str, Any],
    pipeline: Any,
    stage_dir: Path,
) -> Dict[str, str]:
    """Write per-stage artifacts under Results/achamrah_matheuristic/."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    summary = outputs["summary"]

    # Core solver outputs
    outputs["predicted_inventory"].to_csv(stage_dir / "predicted_inventory.csv", index=False)
    paths["predicted_inventory"] = str(stage_dir / "predicted_inventory.csv")

    outputs["routes"].to_csv(stage_dir / "routes.csv", index=False)
    paths["routes"] = str(stage_dir / "routes.csv")

    outputs["lt_detail"].to_csv(stage_dir / "lt_detail.csv", index=False)
    paths["lt_detail"] = str(stage_dir / "lt_detail.csv")

    outputs["lt_plan"].to_csv(stage_dir / "lt_plan.csv", index=False)
    paths["lt_plan"] = str(stage_dir / "lt_plan.csv")

    outputs["validation_comparison"].to_csv(stage_dir / "validation_comparison.csv", index=False)
    paths["validation_comparison"] = str(stage_dir / "validation_comparison.csv")

    pipeline.mapping.validation_target.to_csv(stage_dir / "validation_target.csv", index=False)
    paths["validation_target"] = str(stage_dir / "validation_target.csv")

    # Stage-level JSON summary
    stage_summary = {
        "stage": "achamrah_matheuristic",
        "status": summary["status"],
        "objective": _safe_float(summary["objective"]),
        "best_objective": _safe_float(summary["best_objective"]),
        "constructive_objective": _safe_float(summary["constructive_objective"]),
        "final_objective": _safe_float(summary["final_objective"]),
        "lt_enabled": bool(summary["allow_lateral_transshipment"]),
        "matheuristic_used": bool(summary["matheuristic_used"]),
        "history_length": int(summary["history_length"]),
        "n_routes": int(summary["n_routes"]),
        "n_lt_moves": int(summary["n_lt_moves"]),
        "constructive_runtime_seconds": _safe_float(summary["constructive_runtime_seconds"]),
        "improvement_runtime_seconds": _safe_float(summary["improvement_runtime_seconds"]),
        "total_runtime_seconds": _safe_float(summary["total_runtime_seconds"]),
        "objective_breakdown": {k: _safe_float(v) for k, v in summary["objective_breakdown"].items()},
        "validation_metrics": {k: _safe_float(v) for k, v in summary["validation_metrics"].items()},
        "instance_metadata": pipeline.mapping.metadata,
    }
    stage_json = stage_dir / "achamrah_matheuristic_summary.json"
    stage_json.write_text(json.dumps(stage_summary, indent=2, default=str))
    paths["stage_summary_json"] = str(stage_json)

    print(f"[achamrah_matheuristic] Wrote {len(paths)} artifacts to {stage_dir}")
    return paths


def write_irp_compat_artifacts(
    outputs: Dict[str, Any],
    results_dir: Path,
) -> None:
    """Write files that mirror the IRP pipeline naming convention for cross-tool compatibility."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # These filenames match what the main thesis pipeline produces
    outputs["predicted_inventory"].to_csv(results_dir / "irp_predicted_inventory.csv", index=False)
    outputs["routes"].to_csv(results_dir / "irp_baseline_routes.csv", index=False)
    outputs["lt_plan"].to_csv(results_dir / "irp_lt_plan.csv", index=False)
    outputs["validation_comparison"].to_csv(results_dir / "irp_validation_comparison.csv", index=False)

    # Achamrah-namespaced copies
    outputs["routes"].to_csv(results_dir / "achamrah_routes.csv", index=False)
    outputs["predicted_inventory"].to_csv(results_dir / "achamrah_predicted_inventory.csv", index=False)
    outputs["lt_plan"].to_csv(results_dir / "achamrah_lt_plan.csv", index=False)
    outputs["validation_comparison"].to_csv(results_dir / "achamrah_validation_comparison.csv", index=False)
    pipeline_mapping_target = outputs.get("validation_comparison")
    if pipeline_mapping_target is not None:
        pipeline_mapping_target.to_csv(results_dir / "achamrah_validation_target.csv", index=False)


def build_phase_comparison(
    outputs: Dict[str, Any],
    thesis_dir: Path,
) -> pd.DataFrame:
    """Build Results/thesis_summary/phase_comparison.csv matching the main pipeline format."""
    summary = outputs["summary"]
    breakdown = summary["objective_breakdown"]
    metrics = summary["validation_metrics"]
    lt_df = outputs.get("lt_plan", pd.DataFrame())
    lt_qty = 0.0
    if isinstance(lt_df, pd.DataFrame) and "lt_qty" in lt_df.columns:
        lt_qty = float(pd.to_numeric(lt_df["lt_qty"], errors="coerce").fillna(0).sum())

    objective = _safe_float(summary["best_objective"])
    shortage_cost = _safe_float(breakdown.get("shortage_cost", 0.0))
    holding_cost = _safe_float(breakdown.get("holding_cost", 0.0))
    routing_cost = _safe_float(breakdown.get("routing_cost", 0.0))
    lt_cost = _safe_float(breakdown.get("transshipment_cost", 0.0))

    row = {
        "phase":              "achamrah_matheuristic",
        "cost_no_lt_M":       (objective + lt_cost) / 1e6,
        "cost_with_lt_M":     objective / 1e6,
        "lt_saving_M":        lt_cost / 1e6,
        "shortage_no_lt":     shortage_cost,
        "shortage_with_lt":   shortage_cost,
        "lt_qty":             lt_qty,
        "lt_cost":            lt_cost,
        "teacher_rows":       0,
        "gnn_failures":       0,
        "runtime_sec":        _safe_float(summary["total_runtime_seconds"]),
        "forecast_objective": objective,
        "cg_objective":       0.0,
        # Extra Achamrah columns for thesis comparison
        "constructive_objective": _safe_float(summary["constructive_objective"]),
        "final_objective":        _safe_float(summary["final_objective"]),
        "ga_sa_history_steps":    int(summary["history_length"]),
        "n_routes":               int(summary["n_routes"]),
        "n_lt_moves":             int(summary["n_lt_moves"]),
        "holding_cost":           holding_cost,
        "routing_cost":           routing_cost,
        "mae":                    _safe_float(metrics.get("MAE")),
        "rmse":                   _safe_float(metrics.get("RMSE")),
        "mape":                   _safe_float(metrics.get("MAPE")),
        "bias":                   _safe_float(metrics.get("Bias")),
    }
    df = pd.DataFrame([row])
    thesis_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(thesis_dir / "phase_comparison.csv", index=False)
    return df


def build_effectiveness_report(
    outputs: Dict[str, Any],
    thesis_dir: Path,
) -> pd.DataFrame:
    """Build Results/thesis_summary/effectiveness_report.csv — one-file scoreboard."""
    summary = outputs["summary"]
    breakdown = summary["objective_breakdown"]
    metrics = summary["validation_metrics"]

    row = {
        "solver":                      "Achamrah_2022_GA_SA",
        "lt_enabled":                  True,
        "best_objective":              _safe_float(summary["best_objective"]),
        "constructive_objective":      _safe_float(summary["constructive_objective"]),
        "final_objective":             _safe_float(summary["final_objective"]),
        "holding_cost":                _safe_float(breakdown.get("holding_cost")),
        "routing_cost":                _safe_float(breakdown.get("routing_cost")),
        "transshipment_cost":          _safe_float(breakdown.get("transshipment_cost")),
        "shortage_cost":               _safe_float(breakdown.get("shortage_cost")),
        "n_routes":                    int(summary["n_routes"]),
        "n_lt_moves":                  int(summary["n_lt_moves"]),
        "ga_sa_history_steps":         int(summary["history_length"]),
        "mae":                         _safe_float(metrics.get("MAE")),
        "rmse":                        _safe_float(metrics.get("RMSE")),
        "mape":                        _safe_float(metrics.get("MAPE")),
        "bias":                        _safe_float(metrics.get("Bias")),
        "constructive_runtime_sec":    _safe_float(summary["constructive_runtime_seconds"]),
        "improvement_runtime_sec":     _safe_float(summary["improvement_runtime_seconds"]),
        "total_runtime_sec":           _safe_float(summary["total_runtime_seconds"]),
        "gurobi_status":               int(summary["status"]),
    }
    df = pd.DataFrame([row])
    thesis_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(thesis_dir / "effectiveness_report.csv", index=False)
    return df


def write_runtime_breakdown(
    outputs: Dict[str, Any],
    thesis_dir: Path,
    wall_seconds: float,
) -> Dict[str, Any]:
    """Build Results/thesis_summary/runtime_breakdown.json matching the main pipeline format."""
    summary = outputs["summary"]
    rb: Dict[str, Any] = {
        "constructive_phase_seconds":  _safe_float(summary["constructive_runtime_seconds"]),
        "improvement_phase_seconds":   _safe_float(summary["improvement_runtime_seconds"]),
        "total_matheuristic_seconds":  _safe_float(summary["total_runtime_seconds"]),
        "pipeline_wall_seconds":       wall_seconds,
        # Zeros for fields expected by main pipeline but not applicable here
        "offline_gnn_test_subprocess_wall_seconds":  0,
        "online_inference_phase2_wall_seconds":      0,
        "external_benchmark_total_wall_seconds":     0,
    }
    thesis_dir.mkdir(parents=True, exist_ok=True)
    (thesis_dir / "runtime_breakdown.json").write_text(json.dumps(rb, indent=2, default=str))
    return rb


def write_readme(
    results_dir: Path,
    outputs: Dict[str, Any],
    pipeline: Any,
    runtime_breakdown: Dict[str, Any],
) -> None:
    """Write Results/README.md — human-readable directory map."""
    summary = outputs["summary"]
    meta = pipeline.mapping.metadata
    lines = [
        "# Achamrah 2022 IRP-T Matheuristic — Results",
        "",
        f"Generated: {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Instance",
        f"- Stores: {meta.get('n_stores')}  SKUs: {meta.get('n_products')}  Periods: {meta.get('n_periods')}",
        f"- Vehicles: {meta.get('n_vehicles')}  Vehicle capacity: {meta.get('vehicle_capacity')}",
        f"- Data file: {DATA_FILE}  Date range: {START_DATE} → {END_DATE}",
        "",
        "## Key Results",
        f"- Best objective: {_safe_float(summary['best_objective']):.4f}",
        f"- Constructive objective: {_safe_float(summary['constructive_objective']):.4f}",
        f"- Final (GA+SA) objective: {_safe_float(summary['final_objective']):.4f}",
        f"- LT moves: {summary['n_lt_moves']}   Routes: {summary['n_routes']}",
        f"- MAE={_safe_float(summary['validation_metrics'].get('MAE')):.4f}  "
        f"RMSE={_safe_float(summary['validation_metrics'].get('RMSE')):.4f}  "
        f"MAPE={_safe_float(summary['validation_metrics'].get('MAPE')):.4f}",
        "",
        "## Runtime",
        f"- Constructive phase : {_safe_float(summary['constructive_runtime_seconds']):.1f}s",
        f"- GA+SA improvement  : {_safe_float(summary['improvement_runtime_seconds']):.1f}s",
        f"- Total (matheuristic): {_safe_float(summary['total_runtime_seconds']):.1f}s",
        f"- Wall time (pipeline): {runtime_breakdown.get('pipeline_wall_seconds', 0):.1f}s",
        "",
        "## File Map",
        "",
        "Open these first for reporting:",
        "| File | Description |",
        "|------|-------------|",
        "| `thesis_summary/phase_comparison.csv` | Phase-by-phase headline KPIs |",
        "| `thesis_summary/effectiveness_report.csv` | One-file scoreboard: costs + metrics + runtimes |",
        "| `thesis_summary/runtime_breakdown.json` | Constructive vs GA+SA vs wall time |",
        "| `run_manifest.json` | Structured index of all artifacts |",
        "",
        "Stage folders:",
        "| Folder | Contents |",
        "|--------|----------|",
        "| `achamrah_matheuristic/` | Per-stage solver artifacts (routes, LT plan, inventory, comparison) |",
        "| `thesis_summary/` | Aggregated reporting files |",
        "",
    ]
    (results_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_manifest(
    results_dir: Path,
    runtime_breakdown: Dict[str, Any],
    artifacts_by_stage: Dict[str, List[str]],
    pipeline: Any,
) -> Path:
    """Write Results/run_manifest.json."""
    manifest = {
        "run_id": _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "notebook": "GA_SA_Achamrah_Kaggle.py",
        "phases_run": ["achamrah_matheuristic"],
        "solver": "Achamrah_2022_GA_SA",
        "lt_enabled": True,
        "store_limit": STORE_LIMIT,
        "sku_limit": SKU_LIMIT,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "seed": SEED,
        "ga_sa_params": {
            "initial_temperature":   INITIAL_TEMPERATURE,
            "final_temperature":     FINAL_TEMPERATURE,
            "cooling_ratio":         COOLING_RATIO,
            "crossover_probability": CROSSOVER_PROBABILITY,
            "mutation_probability":  MUTATION_PROBABILITY,
            "population_size":       POPULATION_SIZE,
            "iterations_per_temp":   ITERATIONS_PER_TEMP,
            "constructive_time_limit_sec": CONSTRUCTIVE_TIME_LIMIT,
            "improvement_time_limit_sec":  IMPROVEMENT_TIME_LIMIT,
            "full_time_limit_sec":         FULL_TIME_LIMIT,
        },
        "instance_metadata": pipeline.mapping.metadata,
        "runtime_breakdown": runtime_breakdown,
        "artifacts_by_stage": artifacts_by_stage,
    }
    manifest_path = results_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest_path


# =========================================================
# 6) Display helpers
# =========================================================

def show_df(title: str, df: pd.DataFrame) -> None:
    print(f"\n[{title}]")
    display(df)


def print_thesis_summary_files(results_dir: Path) -> None:
    thesis_summary_files = [
        ("README.md",                               "Human-readable results directory map"),
        ("thesis_summary/phase_comparison.csv",     "Phase-by-phase headline KPIs"),
        ("thesis_summary/effectiveness_report.csv", "One-file scoreboard: costs + GNN metrics + runtimes"),
        ("thesis_summary/runtime_breakdown.json",   "Constructive vs GA+SA vs wall time"),
        ("run_manifest.json",                       "Structured index of all artifacts"),
    ]
    print("\n[Thesis-summary files] — open these first for reporting")
    for rel, desc in thesis_summary_files:
        p = results_dir / rel
        status = f"{p.stat().st_size / 1024:>8.1f} KB" if p.exists() else "  (missing)"
        print(f"  {rel:<48s}  {status}   {desc}")


def print_results_tree(results_dir: Path) -> None:
    print("\n[Results/ tree by stage]")
    root_files = [p for p in results_dir.iterdir() if p.is_file()] if results_dir.exists() else []
    if root_files:
        print("\n  [root files]")
        for p in sorted(root_files):
            print(f"    {p.name:<70s}  {p.stat().st_size / 1024:>7.1f} KB")

    stage_dirs = ["achamrah_matheuristic", "thesis_summary"]
    for stage in stage_dirs:
        d = results_dir / stage
        if not d.exists():
            continue
        files = [p for p in d.rglob("*") if p.is_file()]
        if not files:
            continue
        print(f"\n  [{stage}]  ({len(files)} file{'s' if len(files) != 1 else ''})")
        for p in sorted(files):
            print(f"    {str(p.relative_to(results_dir)):<70s}  {p.stat().st_size / 1024:>7.1f} KB")


# =========================================================
# 7) Entry point
# =========================================================

if __name__ == "__main__":
    wall_start = time.time()

    # --- Step 1: clone repo and put it on sys.path ---
    clone_repo()

    # --- Step 2: load and verify Gurobi WLS license ---
    load_gurobi_wls_secrets()
    if not verify_gurobi():
        raise RuntimeError("Gurobi license invalid — cannot proceed.")

    # --- Step 3: run the matheuristic pipeline ---
    outputs, pipeline = run_achamrah_pipeline()
    wall_seconds = time.time() - wall_start

    # --- Step 4: write thesis-format output tree ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    thesis_dir   = RESULTS_DIR / "thesis_summary"
    stage_dir    = RESULTS_DIR / "achamrah_matheuristic"

    # 4a: per-stage artifacts
    stage_paths = write_phase_artifacts(outputs, pipeline, stage_dir)

    # 4b: IRP-compat filenames at Results/ root for cross-tool use
    write_irp_compat_artifacts(outputs, RESULTS_DIR)

    # 4c: thesis_summary/ files
    phase_comparison_df  = build_phase_comparison(outputs, thesis_dir)
    effectiveness_df     = build_effectiveness_report(outputs, thesis_dir)
    runtime_breakdown    = write_runtime_breakdown(outputs, thesis_dir, wall_seconds)

    # 4d: README + run manifest
    write_readme(RESULTS_DIR, outputs, pipeline, runtime_breakdown)

    artifacts_by_stage: Dict[str, List[str]] = {}
    for stage in ["achamrah_matheuristic", "thesis_summary"]:
        d = RESULTS_DIR / stage
        if d.exists():
            artifacts_by_stage[stage] = sorted(
                str(p.relative_to(RESULTS_DIR))
                for p in d.rglob("*") if p.is_file()
            )
    manifest_path = write_run_manifest(RESULTS_DIR, runtime_breakdown, artifacts_by_stage, pipeline)

    # --- Step 5: display results ---
    from achamrah_2022_thesis_format_wrapper import print_lt_plan

    summary = outputs["summary"]

    print("\n" + "=" * 70)
    print("ACHAMRAH 2022 IRP-T MATHEURISTIC — RESULTS")
    print("=" * 70)
    print(f"  Status                  : {summary['status']}")
    print(f"  Objective (best)        : {summary['best_objective']:.4f}")
    print(f"  Constructive objective  : "
          + (f"{summary['constructive_objective']:.4f}" if math.isfinite(summary['constructive_objective']) else "N/A"))
    print(f"  Final (GA+SA) objective : "
          + (f"{summary['final_objective']:.4f}" if math.isfinite(summary['final_objective']) else "N/A"))
    print(f"  LT enabled              : {summary['allow_lateral_transshipment']}")
    print(f"  Matheuristic used       : {summary['matheuristic_used']}")
    print(f"  History steps (SA)      : {summary['history_length']}")
    print(f"  Routes                  : {summary['n_routes']}")
    print(f"  LT moves                : {summary['n_lt_moves']}")
    print(f"  Constructive runtime    : {summary['constructive_runtime_seconds']:.1f}s")
    print(f"  Improvement runtime     : {summary['improvement_runtime_seconds']:.1f}s")
    print(f"  Total runtime           : {summary['total_runtime_seconds']:.1f}s")
    print(f"  Wall time               : {wall_seconds:.1f}s")

    print("\nObjective breakdown:")
    pprint.pprint(summary["objective_breakdown"])

    print("\nValidation metrics:")
    pprint.pprint(summary["validation_metrics"])

    show_df("Phase Comparison", phase_comparison_df)
    show_df("Effectiveness Report", effectiveness_df)

    print("\n[Runtime breakdown]")
    for k, v in runtime_breakdown.items():
        print(f"  {k:<56s}  {v}")

    print_lt_plan(outputs["lt_plan"], title="Achamrah LT Plan")

    print("\nRoutes (first 20 rows):")
    display(outputs["routes"].head(20))

    print("\nPredicted inventory (first 20 rows):")
    display(outputs["predicted_inventory"].head(20))

    # --- Step 6: thesis summary file status ---
    print_thesis_summary_files(RESULTS_DIR)

    # --- Step 7: Results/ tree ---
    print_results_tree(RESULTS_DIR)

    print(f"\n[Manifest] wrote {manifest_path}")
    print("\n=== PIPELINE COMPLETE ===")
    print(f"  Results root      : {RESULTS_DIR}")
    print(f"  LT enabled        : True")
    print(f"  Solver            : Achamrah 2022 GA+SA")
    print(f"  Wall time         : {wall_seconds:.1f}s")
    print("\n  Runtime (seconds):")
    for k, v in runtime_breakdown.items():
        print(f"    {k:<56s}  {v}")
