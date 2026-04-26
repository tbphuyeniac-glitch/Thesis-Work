# %% [markdown]
# # IRP-LT + BiGAT — Kaggle Small End-to-End Pipeline
# Fast profile for checking the full Kaggle process and refreshed outputs.
# It intentionally keeps the same stage flow as kaggle_irp_pipeline_clean.py
# while using tiny train/test slices, few teacher scenarios, few CG iterations,
# few GNN epochs, and one benchmark repeat.

# %% [code]
# !pip -q install gurobipy   # run once if not pre-installed

# =========================================================
# 0) Imports
# =========================================================
from __future__ import annotations

import importlib
import json
import os
import pprint
import shutil
import subprocess
import sys
import time 
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
try:
    from IPython.display import display  # noqa: F401  — kept for parity with prior notebook API
except Exception:  # pragma: no cover — outside notebook
    display = print  # type: ignore[assignment]


# =========================================================
# 1) Run configuration
# =========================================================

REPO_URL  = "https://github.com/tbphuyeniac-glitch/Thesis-Work.git"
REPO_ROOT = Path("/kaggle/working/Thesis-Work")
REFRESH_WORKING_REPO = False  # resume-friendly after Kaggle timeout/kernel death

# ── Data files (relative to REPO_ROOT) ───────────────────
TRAIN_DATA_FILE = "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
TEST_DATA_FILE  = "test data.csv"
DIST_REL_PATH   = Path("Distance data") / "mm_megamarket_distance_matrix_clean.csv"

TRAIN_DATA_PATH = REPO_ROOT / TRAIN_DATA_FILE
TEST_DATA_PATH  = REPO_ROOT / TEST_DATA_FILE
DIST_PATH       = REPO_ROOT / DIST_REL_PATH
RESULTS_DIR     = Path("/kaggle/working/Results_small_e2e")

# ── Master reproducibility seed ──────────────────────────
MASTER_SEED = 42

# ── Small end-to-end run scope ───────────────────────────
# This file is intentionally NOT thesis-scale. It is for a quick Kaggle sanity
# run that exercises the entire flow and writes the current output layout.
STORE_LIMIT: Optional[int] = None
SKU_LIMIT:   Optional[int] = 5
TRAIN_START_DATE: Optional[str] = None
TRAIN_END_DATE:   Optional[str] = None
TEST_START_DATE:  Optional[str] = None
TEST_END_DATE:    Optional[str] = None

# Teacher-data scenario generation still needs a bounded canonical date window
# so BASE_SPECS can be constructed deterministically even when Phase 1/2 run on
# the full dataset without date slicing.
DEFAULT_TEACHER_START_DATE = "2025-08-01"
DEFAULT_TEACHER_END_DATE   = "2025-08-14"

# ── Solver / GNN ─────────────────────────────────────────
# CG always stops on convergence; these iteration counts are safety caps, not
# fixed iteration budgets. Keep them high enough that the run can reach
# convergence instead of stopping early at the cap.
CG_ITERATIONS    = 100
CG_STOPPING_MODE = "convergence"
BP_MAX_NODES     = 4
BP_MAX_DEPTH     = 3
GNN_TRAIN_EPOCHS = 3
LT_COST_MULTIPLIER = 1.0        # sensitivity: 5, 10, 25, 50 for thesis


# ── Teacher / scenario generation ────────────────────────
# Set MULTI_SCENARIO_MODE=True to run generate_teacher_scenarios.py for
# multi-instance diversity (required for instance-level train/valid/test split).
# Set False for a fast single-run smoke test.
MULTI_SCENARIO_MODE   = True
# Use more base topologies and more shock realizations per base so the teacher
# dataset has enough distinct source_instance values for GNN training.
SCENARIOS_PER_BASE    = 5       # scenarios per base dataset
CG_ITERATIONS_TEACHER = 100      # safety cap; generator stops on convergence
TIME_LIMIT_TEACHER    = 300      # seconds per base ALNS baseline
SMALL_BASE_SPEC_COUNT = 10       # enough for train/valid/test with split_by=base


def infer_date_range_from_csv(
    csv_path: Path,
    fallback_start: str = DEFAULT_TEACHER_START_DATE,
    fallback_end: str = DEFAULT_TEACHER_END_DATE,
) -> Tuple[str, str]:
    """Infer YYYY-MM-DD start/end dates from the raw demand CSV PERIOD column.

    Safe by design:
    - if the repo/data file is not present yet, return the fallback window
    - if PERIOD is missing or unparsable, return the fallback window
    """
    try:
        if not csv_path.exists():
            print(f"[Teacher base window] CSV not available yet, using fallback: {fallback_start} -> {fallback_end}")
            return fallback_start, fallback_end

        period_only = pd.read_csv(csv_path, usecols=["PERIOD"])
        if "PERIOD" not in period_only.columns or period_only.empty:
            print(f"[Teacher base window] PERIOD column unavailable/empty, using fallback: {fallback_start} -> {fallback_end}")
            return fallback_start, fallback_end

        parsed = pd.to_datetime(
            period_only["PERIOD"].astype(str).str.strip(),
            format="%Y%m%d",
            errors="coerce",
        ).dropna()
        if parsed.empty:
            print(f"[Teacher base window] PERIOD values unparsable, using fallback: {fallback_start} -> {fallback_end}")
            return fallback_start, fallback_end

        start_date = parsed.min().strftime("%Y-%m-%d")
        end_date = parsed.max().strftime("%Y-%m-%d")
        print(f"[Teacher base window] Inferred from TRAIN_DATA_PATH: {start_date} -> {end_date}")
        return start_date, end_date
    except Exception as exc:
        print(
            "[Teacher base window] Could not infer date range from "
            f"{csv_path} ({exc}); using fallback: {fallback_start} -> {fallback_end}"
        )
        return fallback_start, fallback_end


def build_normalized_base_specs(
    start_date: Optional[str],
    end_date: Optional[str],
    target_count: int = 30,
) -> List[str]:
    """Build a reproducible set of base-dataset specs from the raw demand CSV.

    Each base is a normalized slice of the same source dataset defined by:
    - a store_limit
    - a sku_limit
    - a date window inside the master train window
    """
    resolved_start = start_date or DEFAULT_TEACHER_START_DATE
    resolved_end = end_date or DEFAULT_TEACHER_END_DATE
    start_dt = datetime.strptime(resolved_start, "%Y-%m-%d")
    end_dt = datetime.strptime(resolved_end, "%Y-%m-%d")
    horizon_days = max(1, (end_dt - start_dt).days + 1)

    # Five topology scales × three SKU granularities × two time windows = 30 bases.
    store_limits = [4, 5, 6, 7, 8]
    sku_limits = [2, 3, 4]

    full_window = (start_dt, end_dt)
    normalized_days = max(7, horizon_days - 4)
    normalized_start = start_dt + timedelta(days=min(2, max(0, horizon_days - normalized_days)))
    normalized_end = min(end_dt, normalized_start + timedelta(days=normalized_days - 1))
    date_windows = [full_window, (normalized_start, normalized_end)]

    specs: List[str] = []
    base_idx = 0
    for store_limit in store_limits:
        for sku_limit in sku_limits:
            for window_start, window_end in date_windows:
                base_idx += 1
                specs.append(
                    f"base_{base_idx:02d}:{store_limit}:{sku_limit}:"
                    f"{window_start.strftime('%Y-%m-%d')}:"
                    f"{window_end.strftime('%Y-%m-%d')}"
                )

    if target_count < len(specs):
        return specs[:target_count]

    if len(specs) != target_count:
        raise ValueError(
            f"Expected exactly {target_count} normalized base specs, got {len(specs)}"
        )
    return specs

# Base specs are resolved after prepare_working_repo() so TRAIN_DATA_PATH
# definitely exists before we infer the teacher date window.
TEACHER_START_DATE: Optional[str] = None
TEACHER_END_DATE: Optional[str] = None
BASE_SPECS: List[str] = []
TEACHER_SCENARIO_OUT_DIR = str(RESULTS_DIR / "scenarios")

# ── Optional stages ───────────────────────────────────────
# This script is configured for a small single end-to-end run:
# teacher generation -> GNN train/reuse -> Phase 1 -> Phase 2 -> A0/A/B/C benchmark.
RUN_PHASE_2          = True
# Phase 3 is intentionally disabled for this Kaggle pipeline.
# Keep it off unless you explicitly want checkpoint fine-tuning on test data.
RUN_ONLINE_LEARNING  = False
ONLINE_LEARNING_EPOCHS = 2
RUN_BENCHMARK        = True
# Final clean run: wipe old Results/ before any artifacts or logs are written.
CLEAR_RESULTS_DIR    = False
# Resume-friendly retry: reuse an existing checkpoint if the previous run got
# through GNN training before the kernel died.
REUSE_EXISTING_CHECKPOINT = True
BENCHMARK_N_REPEATS  = 3       # small validation run only
HEURISTIC_TOP_K      = 5
DEMAND_SHOCK_SEED    = 42

# Output integerization knobs (LP-relaxed CG → integer operational plan).
# The CG LP produces fractional λ values and therefore fractional implied
# shipment quantities. For thesis-grade reporting these are aggregated per
# (period, sku, from, to) arc and then:
#   - rounded to integer when IRP_INTEGER_FINAL_OUTPUTS=1 (default ON)
#   - dropped if rounded qty < IRP_LT_MIN_UNITS (default 5, i.e. 5-unit MOQ)
LT_MIN_UNITS         = 5        # minimum units per LT shipment arc (MOQ); eliminates LP artefact tiny moves
INTEGER_FINAL_OUTPUTS = True    # round lt_qty / inventory / shortage in final CSVs

# Benchmark fairness: when True, every variant (A0/A/B/C) runs on the
# bit-identical realized-demand realization (same shock seed reused across
# all repeats), so benchmark_comparison.csv compares algorithms only — not
# different demand worlds. Recommended for online-inference test runs.
BENCHMARK_FIXED_SHOCK = True


# =========================================================
# 2) Setup helpers
# =========================================================

def require_path(path: Path, label: str, fatal: bool = True) -> bool:
    exists = path.exists()
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {label}: {path}")
    if not exists and fatal:
        raise FileNotFoundError(f"Required path missing: {label} → {path}")
    return exists


def prepare_working_repo() -> None:
    stable_cwd = REPO_ROOT.parent
    stable_cwd.mkdir(parents=True, exist_ok=True)
    try:
        current_cwd = Path.cwd()
    except FileNotFoundError:
        current_cwd = None
    if current_cwd is None or REPO_ROOT == current_cwd or REPO_ROOT in current_cwd.parents:
        os.chdir(stable_cwd)
    if REPO_ROOT.exists() and REFRESH_WORKING_REPO:
        print("Removing old clone:", REPO_ROOT)
        shutil.rmtree(REPO_ROOT)
    if REPO_ROOT.exists():
        print("Using existing repo:", REPO_ROOT)
        return
    print("Cloning:", REPO_URL)
    r = subprocess.run(["git", "clone", REPO_URL, str(REPO_ROOT)],
                       capture_output=True, text=True, cwd=str(stable_cwd))
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise RuntimeError("git clone failed")
    print("Cloned to:", REPO_ROOT)


def load_gurobi_wls_secrets() -> Tuple[str, str, str]:
    """Load WLS credentials from Kaggle secrets, set env vars, and write ~/gurobi.lic.

    Subprocesses (scenario generator, GNN trainer, offline tester) all inherit
    env vars, but Gurobi's auto-discovery reads ~/gurobi.lic instead.  Writing
    both ensures every code path works.
    """
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()

        def _get(client, *names):
            for n in names:
                try:
                    v = client.get_secret(n)
                    if v and str(v).strip():
                        return str(v).strip()
                except Exception:
                    pass
            raise RuntimeError(f"Kaggle secret not found — tried: {names}")

        access_id  = _get(client, "WLSACCESSID",  "GRB_WLSACCESSID")
        secret     = _get(client, "WLSSECRET",    "GRB_WLSSECRET")
        license_id = _get(client, "LICENSEID",    "GRB_LICENSEID")
    except ImportError:
        # Outside Kaggle: read from existing ~/gurobi.lic or raise
        lic_path = Path.home() / "gurobi.lic"
        if not lic_path.exists():
            raise RuntimeError("Not on Kaggle and ~/gurobi.lic not found. "
                               "Set WLSACCESSID/WLSSECRET/LICENSEID environment variables.")
        creds: Dict[str, str] = {}
        for line in lic_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                creds[k.strip().upper()] = v.strip()
        access_id  = creds["WLSACCESSID"]
        secret     = creds["WLSSECRET"]
        license_id = creds["LICENSEID"]

    # Set env vars (direct gurobipy calls in this process)
    for k, v in [("WLSACCESSID", access_id), ("WLSSECRET", secret), ("LICENSEID", license_id)]:
        os.environ[k] = v

    # Write ~/gurobi.lic (Gurobi auto-discovery in subprocess calls)
    lic_content = (
        "# Gurobi WLS license file\n"
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
        print("✓ Gurobi license verified")
        return True
    except Exception as e:
        print(f"✗ Gurobi license check failed: {e}")
        return False


def show_df(title: str, df: pd.DataFrame, max_rows: int = 10) -> None:
    # run_infra is initialized below — we rely on module-level globals set in section 5.
    log_path = globals().get("RUN_LOG_PATH")
    _rinfra = globals().get("run_infrastructure")
    if _rinfra is not None:
        _rinfra.safe_preview(title, df, rows=max_rows, log_path=log_path)
        return
    print(f"=== {title} ===")
    if df is None or df.empty:
        print("  (empty)")
        return
    print(df.head(max_rows).to_string(index=False))


def show_json(title: str, obj: Any) -> None:
    log_path = globals().get("RUN_LOG_PATH")
    _rinfra = globals().get("run_infrastructure")
    if _rinfra is not None:
        _rinfra.safe_print_json(title, obj, log_path=log_path)
        return
    print(f"=== {title} ===")
    pprint.pprint(obj)


# =========================================================
# 3) Output / summary helpers
# =========================================================

def phase_summary(results: Dict[str, Any], label: str) -> Dict[str, Any]:
    no_lt = results.get("realized_no_lt_cost_breakdown", {})
    wlt   = results.get("realized_with_lt_cost_breakdown", {})
    comp  = results.get("comparison", {})
    lt_df = results.get("lt_plan", pd.DataFrame())

    lt_qty = 0.0
    if isinstance(lt_df, pd.DataFrame) and "lt_qty" in lt_df.columns:
        lt_qty = float(pd.to_numeric(lt_df["lt_qty"], errors="coerce").fillna(0).sum())

    c_no  = float(no_lt.get("total_realized_operating_cost", 0))
    c_wlt = float(wlt.get("total_realized_operating_cost", 0))
    return {
        "phase":              label,
        "cost_no_lt_M":       c_no  / 1e6,
        "cost_with_lt_M":     c_wlt / 1e6,
        "lt_saving_M":        (c_no - c_wlt) / 1e6,
        "shortage_no_lt":     float(no_lt.get("total_realized_shortage_units", 0)),
        "shortage_with_lt":   float(wlt.get("total_realized_shortage_units", 0)),
        "lt_qty":             lt_qty,
        "lt_cost":            float(wlt.get("lateral_transshipment_cost_realized", 0)),
        "teacher_rows":       len(results.get("teacher_dataset_rows", [])),
        "gnn_failures":       int(results.get("gnn_scoring_failures", 0)),
        "runtime_sec":        float(comp.get("pipeline_runtime_seconds", 0)),
        "forecast_objective": float(comp.get("forecast_dc_plan_objective", 0)),
        "cg_objective":       float(comp.get("cg_rmp_surrogate_objective", 0)),
    }


def save_phase_outputs(
    results: Dict[str, Any],
    phase_label: str,
    *,
    layout=None,
    validation_target: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """Thin wrapper so Kaggle cells keep the old name but write to the new layout.

    Delegates to `irp.save_phase_artifacts` which writes everything for one
    phase to `Results/<phase_label>/` with short filenames. Returns the
    artifact map {relative_path_name: relative_path}.
    """
    import irp_gurobi_converted as _irp
    layout = layout or _irp.build_results_layout(RESULTS_DIR, phase_label=phase_label)
    artifacts = _irp.save_phase_artifacts(
        results,
        layout,
        phase_label,
        validation_target=validation_target,
    )
    print(f"[{phase_label}] Wrote {len(artifacts)} artifacts to {(layout.root / phase_label)}")
    return artifacts


def show_offline_test_results(test_csv_path: Path) -> None:
    if not test_csv_path.exists():
        print(f"  [Offline Test] result file not found: {test_csv_path}")
        return
    df = pd.read_csv(test_csv_path)
    if df.empty:
        print("  [Offline Test] empty result file")
        return
    group_col = next((c for c in ["mass_threshold", "threshold"] if c in df.columns), None)
    if group_col:
        agg = df.groupby(group_col).mean(numeric_only=True).reset_index()
    else:
        agg = df
    show_df("GNN Offline Test Metrics (held-out split)", agg)


# =========================================================
# 4) Pipeline helpers
# =========================================================

def build_data(irp: Any, data_path: Path,
               store_limit: Optional[int] = None, sku_limit: Optional[int] = None,
               start_date: Optional[str] = None, end_date: Optional[str] = None):
    if store_limit is None:
        store_limit = STORE_LIMIT
    if sku_limit is None:
        sku_limit = SKU_LIMIT
    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=str(data_path),
        sheet_name=None,
        store_limit=store_limit,
        sku_limit=sku_limit,
        start_date=start_date,
        end_date=end_date,
    )
    data, base_df, validation_target, meta = mapper.build_irp_data(
        wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2,
        shortage_cost_rate=0.25,
        holding_cost_rate=0.01,
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
        distance_matrix_path=str(DIST_PATH) if DIST_PATH.exists() else None,
        store_initial_inventory_multiplier=0.2,
        lt_cost_multiplier=LT_COST_MULTIPLIER,
    )
    return mapper, data, base_df, validation_target, meta


def run_phase(irp: Any, data: Any, *, use_gnn: bool, collect_teacher: bool,
              heuristic_top_k_mode: bool = False) -> Dict[str, Any]:
    return irp.IRPResearchPipeline(data).run(
        use_random_initial_patterns=True,
        n_initial_patterns_per_product_period=10,
        cg_iterations=CG_ITERATIONS,
        msg=False,
        time_limit=None,
        enforce_integer_flows=False,
        cw_dispatch_cycle=5,
        use_gnn=use_gnn,
        collect_teacher_mode=collect_teacher,
        runtime_gnn_mode=use_gnn,
        gnn_checkpoint=irp.DEFAULT_GNN_CHECKPOINT,
        use_classical_fallback=use_gnn,
        gnn_selection_mode="cumulative_mass",
        gnn_mass_threshold=0.55,
        gnn_relative_threshold=0.85,
        gnn_max_keep=150,
        gnn_max_keep_fraction=0.30,
        heuristic_top_k_mode=heuristic_top_k_mode,
        heuristic_top_k=HEURISTIC_TOP_K,
        use_branch_and_price=True,
        bp_max_nodes=BP_MAX_NODES,
        bp_max_depth=BP_MAX_DEPTH,
        lt_activation_threshold=10.0,
        demand_shock_probability=0.85,
        demand_shock_reallocation_fraction=0.60,
        demand_shock_reallocations_per_product_period=3,
        demand_shock_non_dispatch_multiplier=1.8,
        demand_shock_seed=DEMAND_SHOCK_SEED,
        diagnostic_verbosity="summary",
        stackelberg_aware_scoring=True,
        stackelberg_exact_follower=False,
    )


# =========================================================
# 5) Repo setup + imports
# =========================================================

prepare_working_repo()

print("\n[Path check]")
require_path(REPO_ROOT,       "REPO_ROOT")
require_path(TRAIN_DATA_PATH, "TRAIN_DATA_PATH")
require_path(TEST_DATA_PATH,  "TEST_DATA_PATH")
require_path(DIST_PATH,       "DIST_PATH", fatal=False)

# Resolve teacher generation window only after the repo clone / refresh step,
# so TRAIN_DATA_PATH is guaranteed to exist for a final clean run.
TEACHER_START_DATE, TEACHER_END_DATE = infer_date_range_from_csv(TRAIN_DATA_PATH)
BASE_SPECS = build_normalized_base_specs(
    start_date=TEACHER_START_DATE,
    end_date=TEACHER_END_DATE,
    target_count=SMALL_BASE_SPEC_COUNT,
)

print("[Run scope]")
print(f"  store_limit={STORE_LIMIT}  sku_limit={SKU_LIMIT}")
print(f"  train_window={TRAIN_START_DATE}..{TRAIN_END_DATE}")
print(f"  test_window ={TEST_START_DATE}..{TEST_END_DATE}")
print(f"  teacher_window={TEACHER_START_DATE}..{TEACHER_END_DATE}  base_specs={len(BASE_SPECS)}")
print(f"  cg_iterations={CG_ITERATIONS}  bp_nodes={BP_MAX_NODES}  bp_depth={BP_MAX_DEPTH}")
print(f"  gnn_epochs={GNN_TRAIN_EPOCHS}  scenarios_per_base={SCENARIOS_PER_BASE}  benchmark_repeats={BENCHMARK_N_REPEATS}")
print(f"  lt_min_units={LT_MIN_UNITS}  integer_outputs={INTEGER_FINAL_OUTPUTS}  benchmark_fixed_shock={BENCHMARK_FIXED_SHOCK}")
print(f"  clear_results_dir={CLEAR_RESULTS_DIR}  reuse_existing_checkpoint={REUSE_EXISTING_CHECKPOINT}")

if CLEAR_RESULTS_DIR and RESULTS_DIR.exists():
    print(f"[Clean run] Removing old Results directory: {RESULTS_DIR}")
    shutil.rmtree(RESULTS_DIR)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Propagate to all subprocesses (scenario generator, GNN trainer, graph
# builder, offline tester) so they write into the canonical RESULTS_DIR.
os.environ["IRP_RESULTS_DIR"] = str(RESULTS_DIR)
# Force a strict CG convergence stop rule across direct runs and subprocesses.
os.environ["IRP_CG_STOPPING_MODE"] = CG_STOPPING_MODE
# Kaggle-safety defaults: quiet CG/pipeline prints, flush partial CG diagnostics
# to disk, and auto-resume BiGAT training from last_model.pt. Each can still be
# overridden from outside if the user has already set the variable.
os.environ.setdefault("IRP_QUIET", "1")
os.environ.setdefault("IRP_CG_PARTIAL_DIR", str(RESULTS_DIR / "cg_partials"))
os.environ.setdefault("IRP_TRAINING_AUTO_RESUME", "1")
# Integer/threshold knobs for final operational outputs (LT plan).
os.environ["IRP_INTEGER_FINAL_OUTPUTS"] = "1" if INTEGER_FINAL_OUTPUTS else "0"
os.environ["IRP_LT_MIN_UNITS"] = str(LT_MIN_UNITS)
# Benchmark fairness: same demand-shock realization across all variants/repeats.
os.environ["IRP_BENCHMARK_FIXED_SHOCK"] = "1" if BENCHMARK_FIXED_SHOCK else "0"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "GNN") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "GNN"))

os.chdir(REPO_ROOT)

load_gurobi_wls_secrets()
if not verify_gurobi():
    raise RuntimeError("Gurobi license verification failed — fix credentials before proceeding.")

import irp_gurobi_converted as irp
irp = importlib.reload(irp)

# --- Kaggle robustness wiring -------------------------------------------------
# RunLogger tees stdout/stderr into Results/run.log so a kernel crash still
# leaves a readable execution trail. RunState records phase completion so a
# rerun can skip any stage whose artifacts already exist on disk.
import run_infrastructure  # noqa: E402
RUN_LOGGER = run_infrastructure.RunLogger(RESULTS_DIR)
RUN_LOG_PATH = RUN_LOGGER.activate()
RUN_STATE = run_infrastructure.RunState(RESULTS_DIR)
print(f"[run_infrastructure] logging to {RUN_LOG_PATH}")
print(f"[run_infrastructure] IRP_QUIET={os.environ.get('IRP_QUIET')} "
      f"auto_resume={os.environ.get('IRP_TRAINING_AUTO_RESUME')} "
      f"cg_partial_dir={os.environ.get('IRP_CG_PARTIAL_DIR')}")


# =========================================================
# 6) Teacher data collection
# =========================================================
# Strategy:
#   MULTI_SCENARIO_MODE=True  → run generate_teacher_scenarios.py with multiple
#     base specs (different store/SKU limits) to generate ≥4 distinct
#     source_instance values → instance-level train/valid/test split
#   MULTI_SCENARIO_MODE=False → single CG run (fast smoke test, no held-out test)

print("\n" + "=" * 70)
print("TEACHER DATA COLLECTION")
print("=" * 70)

teacher_csv_path: Optional[Path] = None

if MULTI_SCENARIO_MODE:
    print(f"\n[Multi-scenario mode]  bases={len(BASE_SPECS)}  scenarios_per_base={SCENARIOS_PER_BASE}")
    print(f"  Expected distinct source_instances: {len(BASE_SPECS) * SCENARIOS_PER_BASE}")

    if len(BASE_SPECS) < 3:
        raise ValueError(
            f"Need ≥ 3 base specs for a meaningful instance-level split (train/valid/test). "
            f"Got {len(BASE_SPECS)}.  Add more entries to BASE_SPECS."
        )

    scenario_cmd = [
        sys.executable,
        str(REPO_ROOT / "GNN" / "generate_teacher_scenarios.py"),
        "--master-csv",          str(TRAIN_DATA_PATH),
        "--bases",               *BASE_SPECS,
        "--scenarios-per-base",  str(SCENARIOS_PER_BASE),
        "--cg-iterations",       str(CG_ITERATIONS_TEACHER),
        "--time-limit",          str(TIME_LIMIT_TEACHER),
        "--master-seed",         str(MASTER_SEED),
        "--out-dir",             TEACHER_SCENARIO_OUT_DIR,
        "--continue-on-failure",
    ]
    agg_csv_probe = Path(TEACHER_SCENARIO_OUT_DIR) / "aggregate_teacher_rows.csv"
    if RUN_STATE.is_done("scenario_generation") and agg_csv_probe.exists() and agg_csv_probe.stat().st_size > 100:
        print(f"[scenario_generation] already done — reusing {agg_csv_probe}")
        rc_scenarios = 0
    else:
        print("\n[Running scenario generator...]")
        RUN_STATE.mark_start("scenario_generation", command=list(scenario_cmd))
        rc_scenarios = run_infrastructure.quiet_subprocess(
            scenario_cmd,
            log_path=RUN_LOG_PATH,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            tag="scenario_generator",
            heartbeat_every=500,
        )
        if rc_scenarios == 0:
            RUN_STATE.mark_done("scenario_generation", returncode=rc_scenarios)
        else:
            print(f"[WARNING] Scenario generator exited with code {rc_scenarios}. "
                  "Some scenarios may have failed — continuing with collected rows.")
            RUN_STATE.update("scenario_generation", returncode=rc_scenarios, partial_ok=True)

    agg_csv = Path(TEACHER_SCENARIO_OUT_DIR) / "aggregate_teacher_rows.csv"
    if not agg_csv.exists() or agg_csv.stat().st_size < 100:
        raise RuntimeError(
            f"Scenario generation produced no aggregate CSV at {agg_csv}. "
            "Check the scenario generator log for errors."
        )

    agg_df = pd.read_csv(agg_csv)
    print(f"\n[Scenario generation complete]")
    print(f"  aggregate rows    : {len(agg_df)}")
    n_instances = agg_df["source_instance"].nunique() if "source_instance" in agg_df.columns else 0
    print(f"  unique instances  : {n_instances}")
    if n_instances < 3:
        print(f"  [WARNING] Only {n_instances} distinct source_instance values — "
              "instance-level test split may be empty. Increase BASE_SPECS diversity.")

    # Copy aggregate to Results/teacher/ under the canonical name so the
    # GNN graph builder finds it at the standard location.
    teacher_dir = RESULTS_DIR / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    teacher_csv_path = teacher_dir / "teacher_rows.csv"
    shutil.copy2(agg_csv, teacher_csv_path)
    print(f"  teacher CSV       : {teacher_csv_path}")

    # Manifest summary
    manifest_path = Path(TEACHER_SCENARIO_OUT_DIR) / "scenarios_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        scenarios_run   = len(manifest.get("scenarios", []))
        split_assign    = manifest.get("split_assignment", {})
        for split in ["train", "valid", "test"]:
            n = sum(1 for v in split_assign.values() if v == split)
            print(f"  {split:<6} scenarios : {n}")

else:
    # Single-run mode: collect teacher rows from one CG run
    print("\n[Single-run mode — collecting teacher rows from one CG run]")
    print("  WARNING: only 1 source_instance — test split will be empty.")
    print("  Switch MULTI_SCENARIO_MODE=True for a real train/valid/test split.")

    _, single_data, _, _, single_meta = build_data(
        irp,
        TRAIN_DATA_PATH,
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
    )
    show_json("Dataset metadata", single_meta)

    p1_results = run_phase(irp, single_data, use_gnn=False, collect_teacher=True)

    teacher_df = pd.DataFrame(p1_results.get("teacher_dataset_rows", []))
    print(f"\n[Teacher rows collected: {len(teacher_df)}]")
    if teacher_df.empty:
        raise RuntimeError(
            "No teacher rows generated. Increase CG_ITERATIONS_TEACHER or check CG convergence."
        )

    teacher_dir = RESULTS_DIR / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    teacher_pkl = teacher_dir / "teacher_rows.pkl.gz"
    teacher_df.to_pickle(teacher_pkl)
    teacher_csv_path = teacher_dir / "teacher_rows.csv"
    teacher_df.to_csv(teacher_csv_path, index=False)
    save_phase_outputs(p1_results, "phase1_offline_baseline")
    print(f"  saved: {teacher_pkl.name}  ({teacher_pkl.stat().st_size / 1024:.0f} KB)")

    # Single-run mode ran a full ALNS+CG+B&P pipeline above — identical to what
    # section 8 would run with collect_teacher=False.  Mark Phase 1 complete now
    # so section 8 skips the redundant re-solve (avoids ~2× ALNS+CG runtime).
    _p1_dedup_path = RESULTS_DIR / "thesis_summary" / "phase1_offline_baseline_summary.json"
    _p1_dedup_path.parent.mkdir(parents=True, exist_ok=True)
    run_infrastructure.write_json_atomic(
        _p1_dedup_path,
        phase_summary(p1_results, "phase1_offline_baseline"),
    )
    RUN_STATE.mark_done("phase1")
    print(f"  [Phase 1] marked done (single-run dedup) — section 8 will reuse this result")


# =========================================================
# 7) Build teacher graphs → train BiGAT → offline test
# =========================================================

print("\n" + "=" * 70)
print("GNN — BUILD GRAPHS → TRAIN → OFFLINE TEST")
print("=" * 70)

graph_dir  = REPO_ROOT / "GNN" / "data" / "irplt_teacher"
checkpoint = REPO_ROOT / irp.DEFAULT_GNN_CHECKPOINT
gnn_dir = RESULTS_DIR / "gnn"
gnn_dir.mkdir(parents=True, exist_ok=True)
train_hist_csv   = gnn_dir / "training_history.csv"
offline_test_dir = gnn_dir / "offline_test"
offline_test_dir.mkdir(parents=True, exist_ok=True)
offline_test_csv = offline_test_dir / "test_per_sample.csv"

# -- 7a: Build graph dataset ---------------------------------------------------
print("\n[Step 7a] Building teacher graph dataset...")
build_result = irp.run_teacher_graph_and_gnn_training(
    teacher_csv_path=str(teacher_csv_path),
    build_graphs=True,
    train_gnn=False,
    checkpoint_path=irp.DEFAULT_GNN_CHECKPOINT,
    run_offline_test=False,
)

# Report split sizes
for split in ["train", "valid", "test"]:
    split_dir = graph_dir / split
    n = len(list(split_dir.glob("*.pkl"))) if split_dir.exists() else 0
    print(f"  {split:<6} samples : {n}")

if (graph_dir / "dataset_summary.json").exists():
    with open(graph_dir / "dataset_summary.json") as f:
        ds_summary = json.load(f)
    show_json("Graph Dataset Summary", ds_summary)
    graphs_dir = RESULTS_DIR / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(graph_dir / "dataset_summary.json", graphs_dir / "graph_dataset_summary.json")

train_samples = list((graph_dir / "train").glob("*.pkl")) if (graph_dir / "train").exists() else []
valid_samples = list((graph_dir / "valid").glob("*.pkl")) if (graph_dir / "valid").exists() else []
test_samples  = list((graph_dir / "test").glob("*.pkl"))  if (graph_dir / "test").exists()  else []

if not train_samples:
    raise RuntimeError(
        "No training graph samples built. Check that teacher CSV has "
        "constraint_features_json populated for at least one group."
    )
if not valid_samples:
    raise RuntimeError(
        "No validation graph samples built. Need ≥ 2 distinct source_instance values. "
        "Increase BASE_SPECS or SCENARIOS_PER_BASE."
    )
if not test_samples:
    raise RuntimeError(
        "No held-out test graph samples built. Final run requires a non-empty test split. "
        "Increase BASE_SPECS / SCENARIOS_PER_BASE or inspect source_instance diversity."
    )

# -- 7b: Train BiGAT -----------------------------------------------------------
print(f"\n[Step 7b] Training BiGAT  epochs={GNN_TRAIN_EPOCHS}, objective=pairwise_rank...")
train_cmd = [
    sys.executable, "GNN/03_train_bigat.py",
    "--data-dir",     str(graph_dir),
    "--dataset-type", "teacher",
    "--epochs",       str(GNN_TRAIN_EPOCHS),
    "--patience",     str(GNN_TRAIN_EPOCHS),   # rely on MRR early-stopping
    "--objective",    "pairwise_rank",
    "--auto-resume",
]
# Skip retraining if a reusable checkpoint already exists. `run_state` is still
# honored when present, but the checkpoint file itself is the stronger signal:
# it lets later Kaggle runs reuse the model even if run_state.json is missing.
if checkpoint.exists() and REUSE_EXISTING_CHECKPOINT:
    print(f"[gnn_training] checkpoint already exists — reusing {checkpoint}")
    if RUN_STATE.is_done("gnn_training"):
        print("[gnn_training] run_state confirms prior training completion.")
    rc_train = 0
else:
    RUN_STATE.mark_start("gnn_training", epochs=GNN_TRAIN_EPOCHS)
    rc_train = run_infrastructure.quiet_subprocess(
        train_cmd,
        log_path=RUN_LOG_PATH,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        tag="bigat_train",
        heartbeat_every=200,
    )
    if rc_train == 0:
        RUN_STATE.mark_done("gnn_training", returncode=rc_train)
    else:
        print(f"[WARNING] BiGAT training exited with code {rc_train} — will retry on next run")
        RUN_STATE.update("gnn_training", returncode=rc_train, partial_ok=True)

gnn_history = irp.load_gnn_training_history(irp.DEFAULT_GNN_CHECKPOINT)
pd.DataFrame(gnn_history).to_csv(train_hist_csv, index=False)
training_summary_src = checkpoint.parent / "training_summary.json"
if training_summary_src.exists():
    shutil.copy2(training_summary_src, gnn_dir / "training_summary.json")
irp.print_gnn_training_history(gnn_history)
print(f"\n  checkpoint exists : {checkpoint.exists()}")
print(f"  history rows      : {len(gnn_history)}")

# -- 7c: Offline held-out test -------------------------------------------------
# Runtime category: OFFLINE GNN TEST ONLY — forward pass on held-out graphs,
# no solver involved. `04_test.py` additionally writes forward-pass timing to
# gnn/offline_test/test_runtime_seconds.json. The subprocess wall time below
# bounds that with Python startup + I/O overhead.
offline_gnn_test_wall_seconds: Optional[float] = None
if checkpoint.exists():
    print(f"\n[Step 7c] Offline test on {len(test_samples)} held-out samples...")
    test_cmd = [
        sys.executable, "GNN/04_test.py",
        "--data-dir",   str(graph_dir),
        "--checkpoint", str(checkpoint),
        "--split",      "test",
        "--out-file",   str(offline_test_csv),
    ]
    # Skip re-running if we already have a complete offline-test CSV.
    if RUN_STATE.is_done("offline_gnn_test") and offline_test_csv.exists() and offline_test_csv.stat().st_size > 50:
        print(f"[offline_gnn_test] already done — reusing {offline_test_csv}")
        offline_gnn_test_wall_seconds = float(RUN_STATE.get("offline_gnn_test").get("wall_seconds") or 0.0)
        rc_offline = 0
    else:
        RUN_STATE.mark_start("offline_gnn_test", n_samples=len(test_samples))
        _t_off = time.perf_counter()
        rc_offline = run_infrastructure.quiet_subprocess(
            test_cmd,
            log_path=RUN_LOG_PATH,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            tag="bigat_test",
            heartbeat_every=200,
        )
        offline_gnn_test_wall_seconds = time.perf_counter() - _t_off
        if rc_offline == 0:
            RUN_STATE.mark_done("offline_gnn_test", returncode=rc_offline,
                                wall_seconds=offline_gnn_test_wall_seconds)
        else:
            print(f"[WARNING] Offline test exited with code {rc_offline} — will retry on next run")
            RUN_STATE.update("offline_gnn_test", returncode=rc_offline, partial_ok=True)
    if rc_offline == 0:
        show_offline_test_results(offline_test_csv)
    print(f"  [Offline GNN test] subprocess wall time: {offline_gnn_test_wall_seconds:.2f} s")
else:
    print(f"\n[Step 7c] Offline test skipped — "
          f"test_samples={len(test_samples)}, checkpoint={checkpoint.exists()}")


# =========================================================
# 8) Phase 1 — offline baseline (training dataset)
# =========================================================

print("\n" + "=" * 70)
print("PHASE 1 — OFFLINE BASELINE  (training data, no GNN)")
print("=" * 70)

phase_summaries: List[Dict[str, Any]] = []
p1_results: Optional[Dict[str, Any]] = None
gnn_results: Optional[Dict[str, Any]] = None
ol_results: Optional[Dict[str, Any]] = None

_p1_summary_json = RESULTS_DIR / "thesis_summary" / "phase1_offline_baseline_summary.json"
if RUN_STATE.is_done("phase1") and _p1_summary_json.exists():
    with open(_p1_summary_json) as _f:
        p1_summary = json.load(_f)
    phase_summaries.append(p1_summary)
    _p1_skip_reason = (
        "single-run dedup: section 6 already ran the full ALNS+CG+B&P pipeline"
        if not MULTI_SCENARIO_MODE
        else "prior run: artifacts already on disk"
    )
    print(f"[phase1] skipped ({_p1_skip_reason})")
    print(f"  summary loaded from: {_p1_summary_json}")
    print(f"  cost_with_lt_M={p1_summary.get('cost_with_lt_M', 'n/a'):.4f}  "
          f"lt_saving_M={p1_summary.get('lt_saving_M', 'n/a'):.4f}  "
          f"runtime_sec={p1_summary.get('runtime_sec', 'n/a'):.1f}s")
else:
    RUN_STATE.mark_start("phase1")
    _, p1_data, _, p1_val_target, p1_meta = build_data(
        irp,
        TRAIN_DATA_PATH,
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
    )
    show_json("Phase 1 dataset metadata", p1_meta)

    # Cost-scale diagnostic (LT vs shortage — thesis examiners will ask)
    if p1_data.stores and p1_data.products:
        _s, _p = next(iter(p1_data.stores)), next(iter(p1_data.products))
        _pairs = [(i, j) for i in p1_data.stores for j in p1_data.stores if i != j]
        _lt_avg = sum(p1_data.transship_unit_cost.get(pr, 0.0) for pr in _pairs) / max(1, len(_pairs))
        _srt = p1_data.shortage_cost.get((_s, _p), float("nan"))
        ratio = _lt_avg / max(float(_srt), 1e-12)
        print(f"\n[Cost Diagnostic]  shortage_unit={float(_srt):.4f}  lt_avg={_lt_avg:.4f}  "
              f"ratio={ratio:.4f}  multiplier={LT_COST_MULTIPLIER}x")
        if ratio < 0.1:
            print("  *** LT near-free vs shortage — consider LT_COST_MULTIPLIER ≥ 5 for thesis ***")

    # Multi-scenario mode: teacher rows already collected by scenario generator
    # (SMALL_BASE_SPEC_COUNT × SCENARIOS_PER_BASE sub-instances,
    # CG_ITERATIONS_TEACHER each). Phase 1 here runs the small canonical
    # instance at the configured CG_ITERATIONS +
    # B&P) with collect_teacher=False — no overlap with teacher gen, no redundant
    # teacher export overhead.  This is the thesis benchmark reference solution.
    print(f"[phase1] Running canonical benchmark solve  "
          f"(MULTI_SCENARIO_MODE={MULTI_SCENARIO_MODE}, collect_teacher=False, "
          f"cg_iterations={CG_ITERATIONS}, bp_nodes={BP_MAX_NODES})")
    p1_results = run_phase(irp, p1_data, use_gnn=False, collect_teacher=False)

    p1_summary = phase_summary(p1_results, "phase1_offline_baseline")
    show_df("Phase 1 Summary", pd.DataFrame([p1_summary]))
    save_phase_outputs(p1_results, "phase1_offline_baseline", validation_target=p1_val_target)
    _p1_summary_json.parent.mkdir(parents=True, exist_ok=True)
    run_infrastructure.write_json_atomic(_p1_summary_json, p1_summary)
    phase_summaries.append(p1_summary)
    RUN_STATE.mark_done("phase1")


# =========================================================
# 9) Phase 2 — online inference (test data, GNN scoring)
# =========================================================

online_inference_wall_seconds: Optional[float] = None
_p2_summary_json = RESULTS_DIR / "thesis_summary" / "phase2_online_inference_summary.json"
if RUN_PHASE_2 and checkpoint.exists() and gnn_history:
    if RUN_STATE.is_done("phase2") and _p2_summary_json.exists():
        with open(_p2_summary_json) as _f:
            p2_summary = json.load(_f)
        phase_summaries.append(p2_summary)
        online_inference_wall_seconds = float(RUN_STATE.get("phase2").get("wall_seconds") or 0.0)
        print(f"[phase2] already done — reused summary from {_p2_summary_json}")
    else:
        RUN_STATE.mark_start("phase2")
        print("\n" + "=" * 70)
        print("PHASE 2 — ONLINE INFERENCE  (test data, GNN scoring)")
        print("=" * 70)

        require_path(TEST_DATA_PATH, "TEST_DATA_PATH (Phase 2)")
        _, p2_data, _, p2_val_target, p2_meta = build_data(
            irp,
            TEST_DATA_PATH,
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
        )
        show_json("Phase 2 test data metadata", p2_meta)

        # Runtime category: ONLINE INFERENCE END-TO-END — full CG solve with GNN
        # embedded inside the pricing loop. Separate from offline GNN test.
        _t_p2 = time.perf_counter()
        gnn_results = run_phase(irp, p2_data, use_gnn=True, collect_teacher=False)
        online_inference_wall_seconds = time.perf_counter() - _t_p2

        p2_summary = phase_summary(gnn_results, "phase2_online_inference")
        phase_summaries.append(p2_summary)
        show_df("Phase 2 Summary", pd.DataFrame([p2_summary]))
        save_phase_outputs(gnn_results, "phase2_online_inference", validation_target=p2_val_target)
        run_infrastructure.write_json_atomic(_p2_summary_json, p2_summary)
        RUN_STATE.mark_done("phase2", wall_seconds=online_inference_wall_seconds)
        print(f"  [Online inference] Phase 2 end-to-end wall time: {online_inference_wall_seconds:.2f} s")

elif RUN_PHASE_2:
    print(f"\n[Phase 2 skipped]  checkpoint={checkpoint.exists()}  "
          f"gnn_history_rows={len(gnn_history)}")


# =========================================================
# 10) Phase 3 — online learning (disabled in this notebook)
# =========================================================
# This notebook keeps Phase 3 off so the checkpoint is not fine-tuned on
# test-data runs. Re-enable only if you explicitly want online learning.

_p3_summary_json = RESULTS_DIR / "thesis_summary" / "phase3_online_learning_summary.json"
if RUN_ONLINE_LEARNING and checkpoint.exists() and RUN_STATE.is_done("phase3") and _p3_summary_json.exists():
    with open(_p3_summary_json) as _f:
        p3_summary = json.load(_f)
    phase_summaries.append(p3_summary)
    print(f"[phase3] already done — reused summary from {_p3_summary_json}")
elif RUN_ONLINE_LEARNING and checkpoint.exists():
    RUN_STATE.mark_start("phase3")
    print("\n" + "=" * 70)
    print("PHASE 3 — ONLINE LEARNING  (test data, fine-tune checkpoint)")
    print(f"  epochs={ONLINE_LEARNING_EPOCHS}  resume_checkpoint=True")
    print("=" * 70)

    require_path(TEST_DATA_PATH, "TEST_DATA_PATH (Phase 3)")
    _, ol_data, _, ol_val_target, _ = build_data(
        irp,
        TEST_DATA_PATH,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
    )

    ol_results = run_phase(irp, ol_data, use_gnn=True, collect_teacher=True)

    ol_teacher_df = pd.DataFrame(ol_results.get("teacher_dataset_rows", []))
    print(f"[Online teacher rows collected: {len(ol_teacher_df)}]")

    if not ol_teacher_df.empty:
        ol_teacher_pkl = RESULTS_DIR / "teacher" / "teacher_rows_online.pkl.gz"
        ol_teacher_pkl.parent.mkdir(parents=True, exist_ok=True)
        ol_teacher_df.to_pickle(ol_teacher_pkl)
        irp.run_teacher_graph_and_gnn_training(
            teacher_csv_path=str(ol_teacher_pkl),
            build_graphs=True,
            train_gnn=True,
            train_epochs=ONLINE_LEARNING_EPOCHS,
            resume_checkpoint=True,
            checkpoint_path=irp.DEFAULT_GNN_CHECKPOINT,
            training_history_csv_path=str(gnn_dir / "online_learning_history.csv"),
            run_offline_test=False,
        )
        print("[Online learning] checkpoint updated.")
    else:
        print("[Online learning] No teacher rows — fine-tune skipped.")

    p3_summary = phase_summary(ol_results, "phase3_online_learning")
    phase_summaries.append(p3_summary)
    show_df("Phase 3 Summary", pd.DataFrame([p3_summary]))
    save_phase_outputs(ol_results, "phase3_online_learning", validation_target=ol_val_target)
    run_infrastructure.write_json_atomic(_p3_summary_json, p3_summary)
    RUN_STATE.mark_done("phase3")

elif RUN_ONLINE_LEARNING:
    print("[Phase 3 skipped] checkpoint not found — run and validate Phase 2 first.")
else:
    print("\n[Phase 3 disabled] RUN_ONLINE_LEARNING=False — online learning is not executed.")


# =========================================================
# 11) A0/A/B/C benchmark (optional)
# =========================================================

benchmark_df: Optional[pd.DataFrame] = None
benchmark_wall_seconds: Optional[float] = None

if RUN_BENCHMARK:
    print("\n" + "=" * 70)
    print(f"A0/A/B/C BENCHMARK  (Exact / Classical / Heuristic / GNN-Guided)  on TEST data  ×  {BENCHMARK_N_REPEATS} repeats")
    print("=" * 70)

    # The external A0/A/B/C benchmark runs on the TEST holdout, not training data.
    # Training data is reserved for teacher generation + offline GNN train/valid/test.
    # `run_three_way_benchmark` internally writes comparison_per_run.csv +
    # comparison_aggregate.csv (mean/std) and mirrors the aggregate to
    # thesis_summary/benchmark_comparison.csv.
    require_path(TEST_DATA_PATH, "TEST_DATA_PATH (Benchmark)")
    bench_agg_path = RESULTS_DIR / "benchmark" / "comparison_aggregate.csv"
    bench_per_run_path = RESULTS_DIR / "benchmark" / "comparison_per_run.csv"
    if RUN_STATE.is_done("benchmark") and bench_agg_path.exists() and bench_per_run_path.exists():
        print(f"[benchmark] already done — reusing {bench_per_run_path}")
        benchmark_df = pd.read_csv(bench_per_run_path)
        benchmark_wall_seconds = float(RUN_STATE.get("benchmark").get("wall_seconds") or 0.0)
    else:
        RUN_STATE.mark_start("benchmark", n_repeats=BENCHMARK_N_REPEATS)
        _, bm_data, _, _, _ = build_data(
            irp,
            TEST_DATA_PATH,
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
        )
        _t_bm = time.perf_counter()
        try:
            benchmark_df = irp.run_three_way_benchmark(
                data=bm_data,
                cg_iterations=CG_ITERATIONS,
                time_limit=None,
                bp_max_nodes=BP_MAX_NODES,
                bp_max_depth=BP_MAX_DEPTH,
                gnn_checkpoint_path=irp.DEFAULT_GNN_CHECKPOINT,
                demand_shock_seed=DEMAND_SHOCK_SEED,
                demand_shock_probability=0.85,
                demand_shock_reallocation_fraction=0.60,
                demand_shock_reallocations_per_product_period=3,
                demand_shock_non_dispatch_multiplier=1.8,
                lt_activation_threshold=10.0,
                heuristic_top_k=HEURISTIC_TOP_K,
                enforce_integer_flows=False,
                n_repeats=BENCHMARK_N_REPEATS,
                results_dir=RESULTS_DIR,
            )
            benchmark_wall_seconds = time.perf_counter() - _t_bm
            RUN_STATE.mark_done("benchmark", wall_seconds=benchmark_wall_seconds)
        except Exception as _exc:
            benchmark_wall_seconds = time.perf_counter() - _t_bm
            print(f"[WARNING] Benchmark failed: {_exc} — will retry on next run")
            RUN_STATE.update("benchmark", wall_seconds=benchmark_wall_seconds, error=str(_exc))
            benchmark_df = None
    show_df("A0/A/B/C Benchmark Comparison (per-run)", benchmark_df)
    if bench_agg_path.exists():
        show_df("A0/A/B/C Benchmark Comparison (aggregate mean/std)",
                pd.read_csv(bench_agg_path))
    print(f"  [External benchmark] total wall time: {benchmark_wall_seconds:.2f} s")


# =========================================================
# 12) Phase comparison + charts
# =========================================================

thesis_dir = RESULTS_DIR / "thesis_summary"
thesis_dir.mkdir(parents=True, exist_ok=True)
phase_comparison_df = pd.DataFrame(phase_summaries)
phase_comparison_df.to_csv(thesis_dir / "phase_comparison.csv", index=False)
show_df("Phase Comparison", phase_comparison_df)

charts_dir = RESULTS_DIR / "charts"
try:
    saved_charts = irp.save_pipeline_charts(
        results=p1_results,
        out_dir=charts_dir,
        refreshed_gnn_history=gnn_history,
        phase_comparison_df=phase_comparison_df,
        phase_label="phase1_offline_baseline",
    )
    print(f"\n[Charts] saved {len(saved_charts)}")
    for c in saved_charts:
        print(f"  {Path(c).name}")
except Exception as exc:
    print(f"[Charts] skipped: {exc}")

for phase_label, phase_res in [
    ("phase1_offline_baseline", p1_results),
    ("phase2_online_inference", gnn_results),
    ("phase3_online_learning",  ol_results),
]:
    if phase_res is None:
        continue
    try:
        irp.save_cg_cost_curve(
            phase_res.get("cg_episode_history", []),
            str(charts_dir / f"cg_cost_curve_{phase_label}.png"),
        )
    except Exception as exc:
        print(f"[Chart] CG cost curve {phase_label} skipped: {exc}")


# =========================================================
# 13) Final output summary + run manifest
# =========================================================
# Results/ now has one folder per pipeline stage. The summary below prints
# exactly which file belongs to which stage and writes run_manifest.json as
# a structured index so downstream scripts don't have to guess filenames.

print("\n" + "=" * 70)
print("FINAL OUTPUT FILES")
print("=" * 70)

import datetime as _dt

# --- Runtime breakdown --------------------------------------------------------
# Three runtime categories — never mixed:
#   1. offline GNN test: model forward-pass only (GNN/04_test.py)
#   2. online inference : end-to-end Phase 2 CG solve with GNN embedded
#   3. external benchmark: wall time of A0/A/B/C 4-way benchmark across repeats
runtime_breakdown: Dict[str, Any] = {
    "offline_gnn_test_subprocess_wall_seconds": offline_gnn_test_wall_seconds,
    "online_inference_phase2_wall_seconds":     online_inference_wall_seconds,
    "external_benchmark_total_wall_seconds":    benchmark_wall_seconds,
}
# Pull the GNN forward-pass timing produced inside 04_test.py (more granular).
_offline_runtime_json = offline_test_dir / "test_runtime_seconds.json"
if _offline_runtime_json.exists():
    try:
        with open(_offline_runtime_json) as _f:
            _offline_payload = json.load(_f)
        runtime_breakdown["offline_gnn_test_forward_pass_total_seconds"] = \
            _offline_payload.get("forward_pass_seconds_total")
        runtime_breakdown["offline_gnn_test_forward_pass_mean_seconds"] = \
            _offline_payload.get("forward_pass_seconds_mean")
    except Exception as _exc:
        print(f"[Runtime] could not read {_offline_runtime_json}: {_exc}")

thesis_dir = RESULTS_DIR / "thesis_summary"
thesis_dir.mkdir(parents=True, exist_ok=True)
with open(thesis_dir / "runtime_breakdown.json", "w") as _f:
    json.dump(runtime_breakdown, _f, indent=2, default=str)
print("\n[Runtime breakdown]")
for _k, _v in runtime_breakdown.items():
    print(f"  {_k:<56s}  {_v}")

# --- Effectiveness report + Results/README.md --------------------------------
# Consolidate per-stage KPIs into a single examiner-facing file and write a
# human-readable directory map so reviewers can navigate Results/ unaided.
_training_summary_json = gnn_dir / "training_summary.json"
_training_summary_obj: Optional[Dict[str, Any]] = None
if _training_summary_json.exists():
    try:
        with open(_training_summary_json) as _f:
            _training_summary_obj = json.load(_f)
    except Exception as _exc:
        print(f"[Effectiveness] could not read training_summary: {_exc}")

try:
    irp.build_effectiveness_report(
        RESULTS_DIR,
        phase_summaries=phase_summaries,
        gnn_training_history=gnn_history,
        gnn_training_summary=_training_summary_obj,
        gnn_offline_test_csv=offline_test_csv,
        benchmark_aggregate_csv=(RESULTS_DIR / "benchmark" / "comparison_aggregate.csv")
            if (RESULTS_DIR / "benchmark" / "comparison_aggregate.csv").exists() else None,
        runtime_breakdown=runtime_breakdown,
    )
except Exception as _exc:
    print(f"[Effectiveness] report skipped: {_exc}")

try:
    _split_counts = {}
    _graph_summary_path = graph_dir / "dataset_summary.json"
    if _graph_summary_path.exists():
        with open(_graph_summary_path) as _f:
            _ds = json.load(_f)
        for _k in ("train", "valid", "test"):
            if _k in _ds:
                _split_counts[_k] = int(_ds[_k].get("num_samples", 0)) if isinstance(_ds[_k], dict) else int(_ds[_k])
    _scenario_counts: Dict[str, Any] = {}
    _manifest_path = Path(TEACHER_SCENARIO_OUT_DIR) / "scenarios_manifest.json"
    if _manifest_path.exists():
        with open(_manifest_path) as _f:
            _manifest = json.load(_f)
        _scenario_counts["n_scenarios"] = len(_manifest.get("scenarios", []))
        _split_assign = _manifest.get("split_assignment", {})
        for _sp in ("train", "valid", "test"):
            _scenario_counts[f"{_sp}_scenarios"] = sum(1 for _v in _split_assign.values() if _v == _sp)
    irp.write_results_readme(
        RESULTS_DIR,
        source_files={
            "training_csv (teacher + phase1)":   str(TRAIN_DATA_PATH),
            "test_csv (phase2 + phase3 + benchmark)": str(TEST_DATA_PATH),
            "distance_matrix":                   str(DIST_PATH),
        },
        split_counts=_split_counts,
        scenario_counts=_scenario_counts,
        phase_summaries=phase_summaries,
        runtime_breakdown=runtime_breakdown,
    )
except Exception as _exc:
    print(f"[README] skipped: {_exc}")

# Core thesis outputs — ONE reporting file per headline question.
# If any of these is missing, the thesis report has a gap.
thesis_summary_files = [
    ("run.log",                                 "Full execution log captured outside notebook output"),
    ("run_state.json",                          "Phase completion / resume state"),
    ("README.md",                               "Human-readable results directory map"),
    ("thesis_summary/phase_comparison.csv",     "Phase-by-phase headline KPIs"),
    ("thesis_summary/benchmark_comparison.csv", "A0/A/B/C benchmark aggregate (mean/std across repeats)"),
    ("thesis_summary/effectiveness_report.csv", "One-file scoreboard: solver costs + GNN metrics + runtimes"),
    ("thesis_summary/runtime_breakdown.json",   "Offline-test vs online-inference vs benchmark wall times"),
    ("benchmark/comparison_per_run.csv",        "Per-repeat benchmark rows (for variance check)"),
    ("benchmark/comparison_aggregate.csv",      "Per-variant mean/std/min/max"),
    ("charts/benchmark_overview.png",           "Benchmark cost/runtime chart for A0/A/B/C"),
    ("gnn/training_history.csv",                "GNN training loss curve"),
    ("gnn/training_summary.json",               "GNN best-epoch summary"),
    ("gnn/offline_test/test_per_sample.csv",    "Offline held-out test metrics"),
    ("gnn/offline_test/test_runtime_seconds.json", "Offline GNN forward-pass runtime (model only)"),
    ("graphs/graph_dataset_summary.json",       "Graph dataset split sizes"),
    ("scenarios/scenarios_manifest.json",       "Scenario-generation manifest"),
    ("scenarios/aggregate_teacher_rows.csv",    "Aggregated teacher rows"),
]

print("\n[Thesis-summary files] — open these first for reporting")
for rel, desc in thesis_summary_files:
    p = RESULTS_DIR / rel
    status = f"{p.stat().st_size / 1024:>8.1f} KB" if p.exists() else " (missing)"
    print(f"  {rel:<48s}  {status}   {desc}")

print("\n[Results/ tree by stage]")
root_files = [p for p in RESULTS_DIR.iterdir() if p.is_file()] if RESULTS_DIR.exists() else []
if root_files:
    print("\n  [root files]")
    for p in sorted(root_files):
        print(f"    {p.name:<70s}  {p.stat().st_size / 1024:>7.1f} KB")
stage_dirs = [
    "scenarios", "teacher", "graphs", "gnn",
    "phase1_offline_baseline", "phase2_online_inference",
    "phase2_deploy_after_teacher", "phase3_online_learning",
    "benchmark", "charts", "thesis_summary",
]
for stage in stage_dirs:
    d = RESULTS_DIR / stage
    if not d.exists():
        continue
    files = [p for p in d.rglob("*") if p.is_file()]
    if not files:
        continue
    print(f"\n  [{stage}]  ({len(files)} file{'s' if len(files) != 1 else ''})")
    for p in sorted(files):
        print(f"    {str(p.relative_to(RESULTS_DIR)):<70s}  {p.stat().st_size / 1024:>7.1f} KB")

# --- Run manifest (single JSON index of this notebook run) -------------------
# One file per run that lists every artifact grouped by phase. Useful for
# (a) copying a run into thesis appendices, (b) diffing two runs.
artifacts_by_phase: Dict[str, List[str]] = {}
for stage in stage_dirs:
    d = RESULTS_DIR / stage
    if d.exists():
        artifacts_by_phase[stage] = sorted(
            str(p.relative_to(RESULTS_DIR))
            for p in d.rglob("*") if p.is_file()
        )

run_manifest = {
    "run_id": _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    "notebook": "kaggle_irp_pipeline_small_e2e.py",
    "phases_run": [s["phase"] for s in phase_summaries],
    "multi_scenario_mode": MULTI_SCENARIO_MODE,
    "gnn_checkpoint": str(checkpoint),
    "gnn_checkpoint_exists": checkpoint.exists(),
    "benchmark_variants": (
        sorted(set(benchmark_df["variant"]))
        if benchmark_df is not None and not benchmark_df.empty else []
    ),
    "benchmark_n_repeats": BENCHMARK_N_REPEATS if RUN_BENCHMARK else 0,
    "benchmark_data_source": "TEST_DATA_PATH" if RUN_BENCHMARK else None,
    "benchmark_fixed_shock": BENCHMARK_FIXED_SHOCK if RUN_BENCHMARK else None,
    "lt_min_units": LT_MIN_UNITS,
    "integer_final_outputs": INTEGER_FINAL_OUTPUTS,
    "runtime_breakdown": runtime_breakdown,
    "artifacts_by_stage": artifacts_by_phase,
}
manifest_path = RESULTS_DIR / "run_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(run_manifest, f, indent=2)
print(f"\n[Manifest] wrote {manifest_path}")

print("\n=== PIPELINE COMPLETE ===")
print(f"  Results root      : {RESULTS_DIR}")
print(f"  CG stopping mode  : {CG_STOPPING_MODE}")
print(f"  Phases run        : {[s['phase'] for s in phase_summaries]}")
if benchmark_df is not None and not benchmark_df.empty:
    print(f"  Benchmark variants: {sorted(set(benchmark_df['variant']))}  "
          f"(repeats per variant: {BENCHMARK_N_REPEATS}, source=TEST_DATA_PATH, "
          f"fixed_shock={BENCHMARK_FIXED_SHOCK})")
print(f"  GNN checkpoint    : {checkpoint}  (exists={checkpoint.exists()})")
print("\n  Runtime (seconds):")
for _k, _v in runtime_breakdown.items():
    print(f"    {_k:<56s}  {_v}")
