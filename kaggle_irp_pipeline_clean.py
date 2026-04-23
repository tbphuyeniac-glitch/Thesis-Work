# %% [markdown]
# # IRP-LT + BiGAT — Kaggle End-to-End Pipeline
# Run all cells top-to-bottom.  First-time setup: ensure gurobipy is installed
# and Kaggle secrets WLSACCESSID / WLSSECRET / LICENSEID are set.

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from IPython.display import display


# =========================================================
# 1) Run configuration
# =========================================================

REPO_URL  = "https://github.com/tbphuyeniac-glitch/Thesis-Work.git"
REPO_ROOT = Path("/kaggle/working/Thesis-Work")
REFRESH_WORKING_REPO = False  # set True once after updating the repo to force a fresh clone

# ── Data files (relative to REPO_ROOT) ───────────────────
TRAIN_DATA_FILE = "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
TEST_DATA_FILE  = "test data.csv"
DIST_REL_PATH   = Path("Distance data") / "mm_megamarket_distance_matrix_clean.csv"

TRAIN_DATA_PATH = REPO_ROOT / TRAIN_DATA_FILE
TEST_DATA_PATH  = REPO_ROOT / TEST_DATA_FILE
DIST_PATH       = REPO_ROOT / DIST_REL_PATH
RESULTS_DIR     = Path("/kaggle/working/Results")

# ── Master reproducibility seed ──────────────────────────
MASTER_SEED = 42

# ── Scope limits for single-run mode ─────────────────────
STORE_LIMIT: Optional[int] = None
SKU_LIMIT:   Optional[int] = 5
START_DATE:  Optional[str] = None
END_DATE:    Optional[str] = None

# ── Solver / GNN ─────────────────────────────────────────
CG_ITERATIONS    = 20
BP_MAX_NODES     = 15
BP_MAX_DEPTH     = 6
# First-run recommendation: 60–100 epochs to validate the pipeline end-to-end
# without burning Kaggle GPU quota.  Bump to 500 only after Phase 1 + Phase 2
# produce sensible numbers.
GNN_TRAIN_EPOCHS = 80
LT_COST_MULTIPLIER = 1.0        # sensitivity: 5, 10, 25, 50 for thesis


# ── Teacher / scenario generation ────────────────────────
# Set MULTI_SCENARIO_MODE=True to run generate_teacher_scenarios.py for
# multi-instance diversity (required for instance-level train/valid/test split).
# Set False for a fast single-run smoke test.
MULTI_SCENARIO_MODE   = True
SCENARIOS_PER_BASE    = 5       # scenarios per base dataset
CG_ITERATIONS_TEACHER = 10      # CG iters per scenario run (keep low for speed)
TIME_LIMIT_TEACHER    = 300     # seconds per scenario run
# Base specs: "name:store_limit:sku_limit[:start_date[:end_date]]"
# Vary store/SKU limits to produce topologically distinct instances.
BASE_SPECS = [
    "base_a:4:2",
    "base_b:6:3",
    "base_c:8:4",
    "base_d:5:2",
]
TEACHER_SCENARIO_OUT_DIR = str(RESULTS_DIR / "scenarios")

# ── Optional stages ───────────────────────────────────────
# Only enable RUN_BENCHMARK after Phase 1 + Phase 2 results look correct.
# Benchmark runs 3 CG variants — adds significant wall-clock time.
RUN_PHASE_2          = True
RUN_ONLINE_LEARNING  = False
ONLINE_LEARNING_EPOCHS = 2
RUN_BENCHMARK        = True     # flip to False on first run; enable after Phase 1 + Phase 2 validated
HEURISTIC_TOP_K      = 20
DEMAND_SHOCK_SEED    = 42


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
    if REPO_ROOT.exists() and REFRESH_WORKING_REPO:
        print("Removing old clone:", REPO_ROOT)
        shutil.rmtree(REPO_ROOT)
    if REPO_ROOT.exists():
        print("Using existing repo:", REPO_ROOT)
        return
    print("Cloning:", REPO_URL)
    r = subprocess.run(["git", "clone", REPO_URL, str(REPO_ROOT)],
                       capture_output=True, text=True)
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
    print(f"\n=== {title} ===")
    if df is None or df.empty:
        print("  (empty)")
        return
    display(df.head(max_rows))


def show_json(title: str, obj: Any) -> None:
    print(f"\n=== {title} ===")
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
               store_limit=STORE_LIMIT, sku_limit=SKU_LIMIT,
               start_date=START_DATE, end_date=END_DATE):
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
        lt_activation_threshold=0.0,
        demand_shock_probability=0.85,
        demand_shock_reallocation_fraction=0.60,
        demand_shock_reallocations_per_product_period=3,
        demand_shock_non_dispatch_multiplier=1.8,
        demand_shock_seed=DEMAND_SHOCK_SEED,
        diagnostic_verbosity="summary",
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Propagate to all subprocesses (scenario generator, GNN trainer, graph
# builder, offline tester) so they write into the canonical RESULTS_DIR.
os.environ["IRP_RESULTS_DIR"] = str(RESULTS_DIR)

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
    print("\n[Running scenario generator...]")
    r = subprocess.run(scenario_cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print(f"[WARNING] Scenario generator exited with code {r.returncode}. "
              "Some scenarios may have failed — continuing with collected rows.")

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

    _, single_data, _, _, single_meta = build_data(irp, TRAIN_DATA_PATH)
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

# -- 7b: Train BiGAT -----------------------------------------------------------
print(f"\n[Step 7b] Training BiGAT  epochs={GNN_TRAIN_EPOCHS}, objective=pairwise_rank...")
train_cmd = [
    sys.executable, "GNN/03_train_bigat.py",
    "--data-dir",     str(graph_dir),
    "--dataset-type", "teacher",
    "--epochs",       str(GNN_TRAIN_EPOCHS),
    "--patience",     str(GNN_TRAIN_EPOCHS),   # rely on MRR early-stopping
    "--objective",    "pairwise_rank",
]
r = subprocess.run(train_cmd, cwd=str(REPO_ROOT))
if r.returncode != 0:
    print(f"[WARNING] BiGAT training exited with code {r.returncode}")

gnn_history = irp.load_gnn_training_history(irp.DEFAULT_GNN_CHECKPOINT)
pd.DataFrame(gnn_history).to_csv(train_hist_csv, index=False)
training_summary_src = checkpoint.parent / "training_summary.json"
if training_summary_src.exists():
    shutil.copy2(training_summary_src, gnn_dir / "training_summary.json")
irp.print_gnn_training_history(gnn_history)
print(f"\n  checkpoint exists : {checkpoint.exists()}")
print(f"  history rows      : {len(gnn_history)}")

# -- 7c: Offline held-out test -------------------------------------------------
if test_samples and checkpoint.exists():
    print(f"\n[Step 7c] Offline test on {len(test_samples)} held-out samples...")
    test_cmd = [
        sys.executable, "GNN/04_test.py",
        "--data-dir",   str(graph_dir),
        "--checkpoint", str(checkpoint),
        "--split",      "test",
        "--out-file",   str(offline_test_csv),
    ]
    r = subprocess.run(test_cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print(f"[WARNING] Offline test exited with code {r.returncode}")
    else:
        show_offline_test_results(offline_test_csv)
else:
    print(f"\n[Step 7c] Offline test skipped — "
          f"test_samples={len(test_samples)}, checkpoint={checkpoint.exists()}")


# =========================================================
# 8) Phase 1 — offline baseline (training dataset)
# =========================================================

print("\n" + "=" * 70)
print("PHASE 1 — OFFLINE BASELINE  (training data, no GNN)")
print("=" * 70)

_, p1_data, _, p1_val_target, p1_meta = build_data(irp, TRAIN_DATA_PATH)
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

# validation_target is no longer dumped to disk — it is merged into each
# phase's Results/<phase>/validation_comparison.csv inside save_phase_artifacts.
p1_results = run_phase(irp, p1_data, use_gnn=False, collect_teacher=False)

p1_summary = phase_summary(p1_results, "phase1_offline_baseline")
show_df("Phase 1 Summary", pd.DataFrame([p1_summary]))
save_phase_outputs(p1_results, "phase1_offline_baseline", validation_target=p1_val_target)

phase_summaries: List[Dict[str, Any]] = [p1_summary]
gnn_results: Optional[Dict[str, Any]] = None
ol_results:  Optional[Dict[str, Any]] = None


# =========================================================
# 9) Phase 2 — online inference (test data, GNN scoring)
# =========================================================

if RUN_PHASE_2 and checkpoint.exists() and gnn_history:
    print("\n" + "=" * 70)
    print("PHASE 2 — ONLINE INFERENCE  (test data, GNN scoring)")
    print("=" * 70)

    require_path(TEST_DATA_PATH, "TEST_DATA_PATH (Phase 2)")
    _, p2_data, _, p2_val_target, p2_meta = build_data(irp, TEST_DATA_PATH)
    show_json("Phase 2 test data metadata", p2_meta)

    gnn_results = run_phase(irp, p2_data, use_gnn=True, collect_teacher=False)

    p2_summary = phase_summary(gnn_results, "phase2_online_inference")
    phase_summaries.append(p2_summary)
    show_df("Phase 2 Summary", pd.DataFrame([p2_summary]))
    save_phase_outputs(gnn_results, "phase2_online_inference", validation_target=p2_val_target)

elif RUN_PHASE_2:
    print(f"\n[Phase 2 skipped]  checkpoint={checkpoint.exists()}  "
          f"gnn_history_rows={len(gnn_history)}")


# =========================================================
# 10) Phase 3 — online learning (optional fine-tune)
# =========================================================
# GNN scores columns AND collects new teacher rows simultaneously.
# Fine-tunes the checkpoint in-place (resume_checkpoint=True).
# Keep OFF by default; enable after Phase 2 is validated.

if RUN_ONLINE_LEARNING and checkpoint.exists():
    print("\n" + "=" * 70)
    print("PHASE 3 — ONLINE LEARNING  (test data, fine-tune checkpoint)")
    print(f"  epochs={ONLINE_LEARNING_EPOCHS}  resume_checkpoint=True")
    print("=" * 70)

    require_path(TEST_DATA_PATH, "TEST_DATA_PATH (Phase 3)")
    _, ol_data, _, ol_val_target, _ = build_data(irp, TEST_DATA_PATH)

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

elif RUN_ONLINE_LEARNING:
    print("[Phase 3 skipped] checkpoint not found — run and validate Phase 2 first.")


# =========================================================
# 11) 3-way benchmark (optional)
# =========================================================

benchmark_df: Optional[pd.DataFrame] = None

if RUN_BENCHMARK:
    print("\n" + "=" * 70)
    print("3-WAY BENCHMARK  (Classical / Heuristic / GNN-Guided)")
    print("=" * 70)

    _, bm_data, _, _, _ = build_data(irp, TRAIN_DATA_PATH)
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
        lt_activation_threshold=0.0,
        heuristic_top_k=HEURISTIC_TOP_K,
        enforce_integer_flows=False,
    )
    show_df("3-Way Benchmark Comparison", benchmark_df)
    bench_dir = RESULTS_DIR / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(bench_dir / "comparison.csv", index=False)
    # Mirror into thesis_summary/ for reporting alongside phase_comparison.csv
    thesis_dir = RESULTS_DIR / "thesis_summary"
    thesis_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(thesis_dir / "benchmark_comparison.csv", index=False)


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

# Core thesis outputs — ONE reporting file per headline question.
# If any of these is missing, the thesis report has a gap.
thesis_summary_files = [
    ("thesis_summary/phase_comparison.csv",    "Phase-by-phase headline KPIs"),
    ("thesis_summary/benchmark_comparison.csv","3-way benchmark (classical/heuristic/GNN)"),
    ("gnn/training_history.csv",               "GNN training loss curve"),
    ("gnn/training_summary.json",              "GNN best-epoch summary"),
    ("gnn/offline_test/test_per_sample.csv",   "Offline held-out test metrics"),
    ("graphs/graph_dataset_summary.json",      "Graph dataset split sizes"),
    ("scenarios/scenarios_manifest.json",      "Scenario-generation manifest"),
    ("scenarios/aggregate_teacher_rows.csv",   "Aggregated teacher rows"),
]

print("\n[Thesis-summary files] — open these first for reporting")
for rel, desc in thesis_summary_files:
    p = RESULTS_DIR / rel
    status = f"{p.stat().st_size / 1024:>8.1f} KB" if p.exists() else " (missing)"
    print(f"  {rel:<48s}  {status}   {desc}")

print("\n[Results/ tree by stage]")
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
    "notebook": "kaggle_irp_pipeline_clean.py",
    "phases_run": [s["phase"] for s in phase_summaries],
    "multi_scenario_mode": MULTI_SCENARIO_MODE,
    "gnn_checkpoint": str(checkpoint),
    "gnn_checkpoint_exists": checkpoint.exists(),
    "benchmark_variants": (
        list(benchmark_df["variant"]) if benchmark_df is not None and not benchmark_df.empty else []
    ),
    "artifacts_by_stage": artifacts_by_phase,
}
manifest_path = RESULTS_DIR / "run_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(run_manifest, f, indent=2)
print(f"\n[Manifest] wrote {manifest_path}")

print("\n=== PIPELINE COMPLETE ===")
print(f"  Results root      : {RESULTS_DIR}")
print(f"  Phases run        : {[s['phase'] for s in phase_summaries]}")
if benchmark_df is not None and not benchmark_df.empty:
    print(f"  Benchmark variants: {list(benchmark_df['variant'])}")
print(f"  GNN checkpoint    : {checkpoint}  (exists={checkpoint.exists()})")
