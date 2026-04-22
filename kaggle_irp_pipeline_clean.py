# If gurobipy is not installed in the Kaggle notebook, run this in a separate cell first:
# !pip -q install gurobipy

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
from typing import Any, Dict, List, Optional

import pandas as pd
from IPython.display import display
from kaggle_secrets import UserSecretsClient


# =========================================================
# 0) Run configuration
# =========================================================

REPO_URL  = "https://github.com/tbphuyeniac-glitch/Thesis-Work.git"
REPO_ROOT = Path("/kaggle/working/Thesis-Work")
REFRESH_WORKING_REPO = False

# Training dataset (offline Stage 1)
DATA_FILE_NAME = "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
# Inference / online-learning dataset (Stage 2 / 3)
# Ensure "test data.csv" is committed to the repo root.
TEST_DATA_FILE_NAME = "test data.csv"

DIST_REL_PATH = Path("Distance data") / "mm_megamarket_distance_matrix_clean.csv"

DATA_PATH      = REPO_ROOT / DATA_FILE_NAME
TEST_DATA_PATH = REPO_ROOT / TEST_DATA_FILE_NAME
DIST_PATH      = REPO_ROOT / DIST_REL_PATH
RESULTS_DIR    = REPO_ROOT / "Results"

# ── Scope limits ──────────────────────────────────────────
STORE_LIMIT: Optional[int] = None
SKU_LIMIT:   Optional[int] = 5
START_DATE:  Optional[str] = None
END_DATE:    Optional[str] = None

# ── Solver / GNN ─────────────────────────────────────────
CG_ITERATIONS    = 20
BP_MAX_NODES     = 15
BP_MAX_DEPTH     = 6
GNN_TRAIN_EPOCHS = 500
LT_COST_MULTIPLIER = 1.0          # sensitivity: try 5, 10, 25, 50 for thesis

# ── Optional stages ───────────────────────────────────────
# Run Phase 2 (online inference on test data.csv) after training.
RUN_PHASE_2 = True
# Run Phase 3 (online learning: GNN scores AND fine-tunes on test data.csv).
# Enable only after Phase 2 has been validated.  Overwrites checkpoint.
RUN_ONLINE_LEARNING   = False
ONLINE_LEARNING_EPOCHS = 2        # keep <=3 to avoid overfitting a single run
# Run 3-way benchmark (Classical / Heuristic / GNN) instead of the full pipeline.
RUN_BENCHMARK    = False
HEURISTIC_TOP_K  = 20

DEMAND_SHOCK_SEED = 42


# =========================================================
# 1) Setup helpers
# =========================================================

def require_path(path: Path, label: str) -> None:
    print(f"  {label}: exists={path.exists()}  {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def prepare_working_repo() -> None:
    if REPO_ROOT.exists() and REFRESH_WORKING_REPO:
        print("Removing old clone:", REPO_ROOT)
        shutil.rmtree(REPO_ROOT)
    if REPO_ROOT.exists():
        print("Using existing repo:", REPO_ROOT)
        return
    print("Cloning:", REPO_URL)
    result = subprocess.run(
        ["git", "clone", REPO_URL, str(REPO_ROOT)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stdout, result.stderr)
        raise RuntimeError("git clone failed")
    print("Cloned to:", REPO_ROOT)


def load_gurobi_wls_secrets() -> None:
    def _get(client, *names):
        for n in names:
            try:
                v = client.get_secret(n)
                if v and str(v).strip():
                    return str(v).strip()
            except Exception:
                pass
        raise RuntimeError(f"Gurobi WLS secret not found. Tried: {names}")

    client = UserSecretsClient()
    access_id  = _get(client, "WLSACCESSID",  "GRB_WLSACCESSID")
    secret     = _get(client, "WLSSECRET",    "GRB_WLSSECRET")
    license_id = _get(client, "LICENSEID",    "GRB_LICENSEID")
    for k, v in [("WLSACCESSID", access_id), ("WLSSECRET", secret), ("LICENSEID", license_id),
                 ("GRB_WLSACCESSID", access_id), ("GRB_WLSSECRET", secret), ("GRB_LICENSEID", license_id)]:
        os.environ[k] = v
    print("Gurobi WLS secrets loaded.")


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
# 2) Runtime print patches (concise Kaggle output)
# =========================================================

def patch_irp_printing(irp: Any) -> None:
    def _post_shock(df):
        print("\n[Post-Shock Fulfillment]")
        if df.empty:
            print("  (empty)"); return
        td = float(df["total_realized_demand"].sum())
        ts = float(df["post_shock_shortage"].sum())
        print(f"  rows={len(df)}  demand={td:.1f}  shortage={ts:.1f}  "
              f"rate={1 - ts/max(td,1e-9):.2%}")
        worst = df.sort_values("post_shock_shortage", ascending=False).head(10)
        cols = [c for c in ["period","store","total_realized_demand",
                             "fulfilled_demand","post_shock_shortage",
                             "post_shock_fulfillment_rate"] if c in worst.columns]
        display(worst[cols])

    def _lt_plan(df, title="LT Plan"):
        print(f"\n[{title}]")
        if df.empty:
            print("  No LT moves."); return
        qty  = float(pd.to_numeric(df.get("lt_qty",  pd.Series()), errors="coerce").fillna(0).sum())
        cost = float(pd.to_numeric(df.get("lt_total_cost", pd.Series()), errors="coerce").fillna(0).sum())
        print(f"  rows={len(df)}  total_qty={qty:.1f}  total_cost={cost:.1f}")
        cols = [c for c in ["source","period","from_store","to_store","sku",
                             "lt_qty","lt_total_cost","lambda_value"] if c in df.columns]
        display(df.sort_values("lt_qty", ascending=False)[cols].head(20))

    def _gnn_history(history, checkpoint_path=None):
        print("\n[GNN Training Summary]")
        if not history:
            print("  No history."); return
        df = pd.DataFrame(history)
        mrr_col = next((c for c in ["ranking_valid_mrr","valid_mrr"] if c in df.columns), None)
        if mrr_col and pd.to_numeric(df[mrr_col], errors="coerce").notna().any():
            best = df.loc[pd.to_numeric(df[mrr_col], errors="coerce").idxmax()]
            print(f"  best_epoch (MRR): {int(best['epoch'])}")
        else:
            best = df.loc[pd.to_numeric(df["valid_loss"], errors="coerce").idxmin()]
            print(f"  best_epoch (loss): {int(best['epoch'])}")
        keep = [c for c in ["epoch","train_loss","valid_loss","ranking_valid_mrr","valid_mrr",
                             "ranking_valid_top1","ranking_valid_top3","binary_valid_f1","score_gap"]
                if c in df.columns]
        display(pd.DataFrame([best, df.iloc[-1]])[keep])

    irp.print_post_shock_fulfillment = _post_shock
    irp.print_lt_plan                = _lt_plan
    irp.print_gnn_training_history   = _gnn_history


# =========================================================
# 3) Output / summary helpers
# =========================================================

def save_df(value: Any, path: Path) -> None:
    if isinstance(value, pd.DataFrame):
        value.to_csv(path, index=False)
    elif isinstance(value, (list, dict)):
        pd.DataFrame(value if isinstance(value, list) else [value]).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def phase_summary(results: Dict[str, Any], label: str) -> Dict[str, Any]:
    no_lt  = results.get("realized_no_lt_cost_breakdown", {})
    wlt    = results.get("realized_with_lt_cost_breakdown", {})
    comp   = results.get("comparison", {})
    lt_df  = results.get("lt_plan", pd.DataFrame())
    lt_qty = 0.0
    if isinstance(lt_df, pd.DataFrame) and "lt_qty" in lt_df.columns:
        lt_qty = float(pd.to_numeric(lt_df["lt_qty"], errors="coerce").fillna(0).sum())
    c_no   = float(no_lt.get("total_realized_operating_cost", 0))
    c_wlt  = float(wlt.get("total_realized_operating_cost", 0))
    return {
        "phase": label,
        "cost_no_lt_M":     c_no  / 1e6,
        "cost_with_lt_M":   c_wlt / 1e6,
        "lt_saving_M":     (c_no - c_wlt) / 1e6,
        "shortage_no_lt":   float(no_lt.get("total_realized_shortage_units", 0)),
        "shortage_with_lt": float(wlt.get("total_realized_shortage_units", 0)),
        "lt_qty":           lt_qty,
        "lt_cost":          float(wlt.get("lateral_transshipment_cost_realized", 0)),
        "teacher_rows":     len(results.get("teacher_dataset_rows", [])),
        "gnn_failures":     int(results.get("gnn_scoring_failures", 0)),
        "runtime_sec":      float(comp.get("pipeline_runtime_seconds", 0)),
    }


def save_phase_outputs(results: Dict[str, Any], validation_target: pd.DataFrame,
                       suffix: str) -> Optional[pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR

    # Routes
    rows = [{
        "period": r.get("period"), "vehicle": r.get("vehicle"),
        "route":  " -> ".join(r.get("route", [])),
        "total_direct_qty": r.get("total_direct_qty", 0),
        "total_lt_qty": r.get("total_lt_qty", 0),
        "product_flow_summary": str(r.get("product_flow_summary", {})),
    } for r in results.get("baseline_routes", [])]
    pd.DataFrame(rows).to_csv(p / f"irp_baseline_routes_{suffix}.csv", index=False)

    for key, name in [
        ("forecast_demand_fulfillment",    f"irp_forecast_demand_fulfillment_{suffix}.csv"),
        ("post_shock_demand_fulfillment",  f"irp_post_shock_demand_fulfillment_{suffix}.csv"),
        ("lt_plan",                        f"irp_lt_plan_{suffix}.csv"),
        ("cg_episode_history",             f"irp_cg_episode_history_{suffix}.csv"),
        ("cg_episode_diagnostics",         f"cg_episode_diagnostics_{suffix}.csv"),
        ("column_pool_diagnostics",        f"column_pool_diagnostics_{suffix}.csv"),
        ("branch_price_history",           f"irp_branch_price_history_{suffix}.csv"),
        ("alns_history",                   f"irp_alns_history_{suffix}.csv"),
    ]:
        save_df(results.get(key), p / name)

    pd.DataFrame([
        {"model": "baseline_alns", **results["baseline_solution"].efficiency_metrics},
        {"model": "cg_rmp_total",  **results["cg_solution"].efficiency_metrics},
    ]).to_csv(p / f"irp_solver_efficiency_metrics_{suffix}.csv", index=False)

    pd.DataFrame([results["baseline_cost_breakdown"]]).to_csv(
        p / f"irp_baseline_cost_breakdown_{suffix}.csv", index=False)

    pd.DataFrame([
        {"scenario": "without_lt", **results["realized_no_lt_cost_breakdown"]},
        {"scenario": "with_cg_lt", **results["realized_with_lt_cost_breakdown"]},
    ]).to_csv(p / f"irp_realized_cost_breakdown_{suffix}.csv", index=False)

    pd.DataFrame([{
        **results.get("demand_shock_summary", {}),
        **results.get("post_shock_summary", {}),
        **results.get("post_shock_lt_diagnostics", {}),
    }]).to_csv(p / f"irp_demand_shock_summary_{suffix}.csv", index=False)

    with open(p / f"irp_gnn_selected_columns_{suffix}.json", "w") as f:
        json.dump(results.get("gnn_selection_history", []), f, indent=2)

    comparison_df = None
    try:
        import irp_gurobi_converted as _irp
        predicted_df = _irp.build_predicted_inventory_df(results["baseline_solution"])
        predicted_df.to_csv(p / f"irp_predicted_inventory_{suffix}.csv", index=False)
        comparison_df = predicted_df.merge(validation_target, on=["store","sku","period"], how="inner")
        comparison_df["error"] = comparison_df["predicted_end_qty"] - comparison_df["actual_end_qty"]
        comparison_df.to_csv(p / f"irp_validation_comparison_{suffix}.csv", index=False)
        print(f"\n[Validation {suffix}]")
        pprint.pprint(_irp.compute_validation_metrics(comparison_df))
    except Exception as exc:
        print(f"[Validation] skipped for {suffix}: {exc}")

    return comparison_df


def show_offline_test_results(test_csv_path: Path) -> None:
    """Display the offline held-out test metrics produced by 04_test.py."""
    if not test_csv_path.exists():
        print("  [Offline Test] result file not found:", test_csv_path)
        return
    df = pd.read_csv(test_csv_path)
    if df.empty:
        print("  [Offline Test] empty result file"); return
    agg = df.groupby("mass_threshold").mean(numeric_only=True).reset_index()
    show_df("GNN Offline Test Metrics (held-out split, mean per threshold)", agg)


# =========================================================
# 4) Pipeline helpers
# =========================================================

def build_data(irp: Any, data_path: Path = DATA_PATH):
    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=str(data_path),
        sheet_name=None,
        store_limit=STORE_LIMIT,
        sku_limit=SKU_LIMIT,
        start_date=START_DATE,
        end_date=END_DATE,
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
# 5) Setup working repo and import project
# =========================================================

prepare_working_repo()

print("\n[Path check]")
require_path(REPO_ROOT,      "REPO_ROOT")
require_path(DATA_PATH,      "DATA_PATH (training)")
require_path(TEST_DATA_PATH, "TEST_DATA_PATH (inference)")
# DIST_PATH is optional — real distances improve routing quality
if not DIST_PATH.exists():
    print(f"  DIST_PATH not found (will use synthetic distances): {DIST_PATH}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "GNN") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "GNN"))

os.chdir(REPO_ROOT)
load_gurobi_wls_secrets()

import irp_gurobi_converted as irp
irp = importlib.reload(irp)
patch_irp_printing(irp)


# =========================================================
# 6) Phase 1 — offline training: ALNS baseline + CG teacher collection
# =========================================================

print("\n" + "=" * 70)
print("PHASE 1 — OFFLINE TRAINING  (training dataset, no GNN yet)")
print("=" * 70)

mapper, data, base_df, validation_target, meta = build_data(irp, DATA_PATH)

show_json("Dataset metadata", meta)

# Cost-scale diagnostic (LT vs shortage — thesis examiners will ask)
if data.stores and data.products:
    _s, _p = next(iter(data.stores)), next(iter(data.products))
    _lt_pairs = [(i, j) for i in data.stores for j in data.stores if i != j]
    _lt_avg = sum(data.transship_unit_cost.get(p, 0.0) for p in _lt_pairs) / max(1, len(_lt_pairs))
    _srt = data.shortage_cost.get((_s, _p), float("nan"))
    print(f"\n[Cost Diagnostic]  shortage_unit={_srt:.4f}  lt_unit_avg={_lt_avg:.4f}  "
          f"lt/shortage={_lt_avg/max(_srt,1e-12):.4f}  lt_cost_multiplier={LT_COST_MULTIPLIER}x")
    if _lt_avg / max(_srt, 1e-12) < 0.1:
        print("  *** LT is near-free vs shortage — consider LT_COST_MULTIPLIER >= 5 ***")

validation_target.to_csv(RESULTS_DIR / "irp_validation_target.csv", index=False)

p1_results = run_phase(irp, data, use_gnn=False, collect_teacher=True)

p1_summary = phase_summary(p1_results, "phase1_offline_training")
show_df("Phase 1 Summary", pd.DataFrame([p1_summary]))
save_phase_outputs(p1_results, validation_target, "phase1")


# =========================================================
# 7) Save teacher rows
# =========================================================

teacher_df = pd.DataFrame(p1_results.get("teacher_dataset_rows", []))
print(f"\n[Teacher rows: {len(teacher_df)}]")

if teacher_df.empty:
    raise RuntimeError(
        "No teacher rows generated — CG did not export any. "
        "Increase CG_ITERATIONS or check collect_teacher_mode."
    )

teacher_pkl = RESULTS_DIR / "cg_teacher_dataset_kaggle.pkl.gz"
teacher_csv_out = RESULTS_DIR / "cg_teacher_dataset_kaggle.csv"
teacher_df.to_pickle(teacher_pkl)
teacher_df.to_csv(teacher_csv_out, index=False)
print(f"  saved: {teacher_pkl.name}  ({teacher_pkl.stat().st_size/1024:.0f} KB)")

show_df("Teacher Label Distribution",
        teacher_df.groupby(["teacher_label"], dropna=False).size().reset_index(name="rows"))


# =========================================================
# 8) Build teacher graphs → train BiGAT → offline test evaluation
# =========================================================

print("\n" + "=" * 70)
print("GNN TRAINING  (build graphs → train BiGAT → offline test)")
print("=" * 70)

graph_dir    = REPO_ROOT / "GNN" / "data" / "irplt_teacher"
checkpoint   = REPO_ROOT / irp.DEFAULT_GNN_CHECKPOINT
train_hist_csv = RESULTS_DIR / "irp_gnn_training_history_kaggle.csv"
offline_test_csv = RESULTS_DIR / "irp_gnn_offline_test.csv"

# -- Step A: build graph dataset
irp.run_teacher_graph_and_gnn_training(
    teacher_csv_path=str(teacher_pkl),
    build_graphs=True,
    train_gnn=False,
    train_epochs=GNN_TRAIN_EPOCHS,
    resume_checkpoint=False,
    checkpoint_path=irp.DEFAULT_GNN_CHECKPOINT,
    training_history_csv_path=str(train_hist_csv),
    run_offline_test=False,
)

train_samples = list((graph_dir / "train").glob("*.pkl")) if (graph_dir / "train").exists() else []
valid_samples = list((graph_dir / "valid").glob("*.pkl")) if (graph_dir / "valid").exists() else []
test_samples  = list((graph_dir / "test").glob("*.pkl"))  if (graph_dir / "test").exists()  else []
print(f"\n[Graph samples]  train={len(train_samples)}  valid={len(valid_samples)}  test={len(test_samples)}")

if (graph_dir / "dataset_summary.json").exists():
    with open(graph_dir / "dataset_summary.json") as f:
        show_json("Graph Dataset Summary", json.load(f))

if not (train_samples and valid_samples):
    raise RuntimeError(
        "Not enough graph samples for train+valid splits. "
        "Collect teacher rows from more source_instance values."
    )

# -- Step B: train BiGAT (MRR-based early stopping)
print("\n[Training BiGAT — objective=pairwise_rank, early stopping on MRR]")
subprocess.run([
    sys.executable, "GNN/03_train_bigat.py",
    "--data-dir",       str(graph_dir),
    "--dataset-type",   "teacher",
    "--epochs",         str(GNN_TRAIN_EPOCHS),
    "--patience",       str(GNN_TRAIN_EPOCHS),   # let early-stopping decide
    "--objective",      "pairwise_rank",
], cwd=str(REPO_ROOT), check=True)

gnn_history: List[Dict[str, Any]] = []
hist_json = checkpoint.parent / "training_history.json"
if hist_json.exists():
    with open(hist_json) as f:
        gnn_history = json.load(f)
pd.DataFrame(gnn_history).to_csv(train_hist_csv, index=False)

irp.print_gnn_training_history(gnn_history)
print(f"\n[Checkpoint]  exists={checkpoint.exists()}  {checkpoint}")

# -- Step C: offline held-out test evaluation
if test_samples and checkpoint.exists():
    print("\n[Offline test — strictly held-out split]")
    subprocess.run([
        sys.executable, "GNN/04_test.py",
        "--data-dir",   str(graph_dir),
        "--checkpoint", str(checkpoint),
        "--split",      "test",
        "--out-file",   str(offline_test_csv),
    ], cwd=str(REPO_ROOT), check=True)
    show_offline_test_results(offline_test_csv)
else:
    print(f"[Offline test skipped]  test_samples={len(test_samples)}  checkpoint={checkpoint.exists()}")


# =========================================================
# 9) Phase 2 — online inference on test data.csv
# =========================================================

phase_summaries: List[Dict[str, Any]] = [p1_summary]
gnn_results = None

if RUN_PHASE_2 and checkpoint.exists() and gnn_history:
    print("\n" + "=" * 70)
    print("PHASE 2 — ONLINE INFERENCE  (test data.csv, no retraining)")
    print("=" * 70)

    _, gnn_data, _, gnn_val_target, gnn_meta = build_data(irp, TEST_DATA_PATH)
    show_json("Test data metadata", gnn_meta)

    gnn_results = run_phase(irp, gnn_data, use_gnn=True, collect_teacher=False)

    p2_summary = phase_summary(gnn_results, "phase2_online_inference")
    phase_summaries.append(p2_summary)
    show_df("Phase 2 Summary", pd.DataFrame([p2_summary]))
    save_phase_outputs(gnn_results, gnn_val_target, "phase2_gnn")
elif RUN_PHASE_2:
    print(f"[Phase 2 skipped]  checkpoint={checkpoint.exists()}  "
          f"gnn_history_rows={len(gnn_history)}")


# =========================================================
# 10) Phase 3 — online learning (optional, fine-tunes checkpoint)
# =========================================================

# Online learning: solver-supervised.
# Teacher labels come from RMP duals (lambda > 1e-6 → label=1) and reduced costs
# collected during the inference CG episodes — same export mechanism as Phase 1.
# The checkpoint is updated in-place (resume_checkpoint=True).
# Justification for defaulting to OFF: offline training + pure inference gives
# a fixed, reproducible model for thesis comparison. Enable only after Phase 2
# has been validated; keep ONLINE_LEARNING_EPOCHS <= 3.

ol_results = None

if RUN_ONLINE_LEARNING and checkpoint.exists():
    print("\n" + "=" * 70)
    print("PHASE 3 — ONLINE LEARNING  (test data.csv, fine-tune in-place)")
    print(f"  supervision : solver-supervised (RMP duals + reduced costs)")
    print(f"  collection  : teacher rows from this inference run")
    print(f"  update freq : once per run (end-of-run fine-tune)")
    print(f"  epochs      : {ONLINE_LEARNING_EPOCHS}  (resume_checkpoint=True)")
    print("=" * 70)

    _, ol_data, _, ol_val_target, _ = build_data(irp, TEST_DATA_PATH)

    # GNN scores columns AND collects new teacher rows simultaneously
    ol_results = run_phase(irp, ol_data, use_gnn=True, collect_teacher=True)

    ol_teacher_df = pd.DataFrame(ol_results.get("teacher_dataset_rows", []))
    print(f"[Online teacher rows collected: {len(ol_teacher_df)}]")

    if not ol_teacher_df.empty:
        ol_teacher_pkl = RESULTS_DIR / "cg_teacher_dataset_online.pkl.gz"
        ol_teacher_df.to_pickle(ol_teacher_pkl)

        # Fine-tune checkpoint in-place
        irp.run_teacher_graph_and_gnn_training(
            teacher_csv_path=str(ol_teacher_pkl),
            build_graphs=True,
            train_gnn=True,
            train_epochs=ONLINE_LEARNING_EPOCHS,
            resume_checkpoint=True,   # fine-tune, never retrain from scratch
            checkpoint_path=irp.DEFAULT_GNN_CHECKPOINT,
            training_history_csv_path=str(RESULTS_DIR / "irp_gnn_online_learning_history.csv"),
            run_offline_test=False,   # skip held-out test after fine-tune
        )
        print("[Online learning] checkpoint updated.")
    else:
        print("[Online learning] No teacher rows — fine-tune skipped.")

    p3_summary = phase_summary(ol_results, "phase3_online_learning")
    phase_summaries.append(p3_summary)
    show_df("Phase 3 Summary", pd.DataFrame([p3_summary]))
    save_phase_outputs(ol_results, ol_val_target, "phase3_online_learning")

elif RUN_ONLINE_LEARNING:
    print("[Phase 3 skipped] checkpoint not found — run Phase 2 first.")


# =========================================================
# 11) 3-way benchmark (optional)
# =========================================================

benchmark_df = None

if RUN_BENCHMARK:
    print("\n" + "=" * 70)
    print("3-WAY BENCHMARK  (Classical CG / Heuristic CG / GNN-Guided CG)")
    print("=" * 70)

    _, bm_data, _, _, _ = build_data(irp, DATA_PATH)

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
    benchmark_df.to_csv(RESULTS_DIR / "irp_benchmark_comparison.csv", index=False)


# =========================================================
# 12) Phase comparison + charts
# =========================================================

phase_comparison_df = pd.DataFrame(phase_summaries)
phase_comparison_df.to_csv(RESULTS_DIR / "irp_phase_comparison_kaggle.csv", index=False)
show_df("Phase Comparison", phase_comparison_df)

try:
    saved_charts = irp.save_pipeline_charts(
        results=p1_results,
        out_dir=RESULTS_DIR / "charts",
        refreshed_gnn_history=gnn_history,
        phase_comparison_df=phase_comparison_df,
    )
    print(f"\n[Charts] saved {len(saved_charts)}")
    for c in saved_charts:
        print(f"  {Path(c).name}")
except Exception as exc:
    print(f"[Charts] skipped: {exc}")

# CG cost curve per phase
for phase_label, phase_res in [("phase1", p1_results), ("phase2_gnn", gnn_results),
                                ("phase3_online", ol_results)]:
    if phase_res is None:
        continue
    try:
        irp.save_cg_cost_curve(
            phase_res.get("cg_episode_history", []),
            str(RESULTS_DIR / f"irp_cg_cost_curve_{phase_label}.png"),
        )
    except Exception as exc:
        print(f"[Chart] CG cost curve {phase_label} skipped: {exc}")


# =========================================================
# 13) Final output summary
# =========================================================

print("\n" + "=" * 70)
print("FINAL OUTPUT FILES")
print("=" * 70)

priority = [
    "irp_phase_comparison_kaggle.csv",
    "irp_benchmark_comparison.csv",
    "irp_realized_cost_breakdown_phase1.csv",
    "irp_realized_cost_breakdown_phase2_gnn.csv",
    "irp_realized_cost_breakdown_phase3_online_learning.csv",
    "irp_gnn_training_history_kaggle.csv",
    "irp_gnn_offline_test.csv",
    "irp_gnn_online_learning_history.csv",
    "cg_teacher_dataset_kaggle.pkl.gz",
    "irp_lt_plan_phase1.csv",
    "irp_lt_plan_phase2_gnn.csv",
]
print("\n[Priority files]")
for name in priority:
    path = RESULTS_DIR / name
    if path.exists():
        print(f"  {name:<65} {path.stat().st_size/1024:>8.1f} KB")

print("\n[All files]")
for path in sorted(RESULTS_DIR.rglob("*")):
    if path.is_file():
        print(f"  {str(path.relative_to(RESULTS_DIR)):<75} {path.stat().st_size/1024:>8.1f} KB")

print("\n=== PIPELINE COMPLETE ===")
print(f"  Phases run: {[s['phase'] for s in phase_summaries]}")
if benchmark_df is not None:
    print(f"  Benchmark variants: {list(benchmark_df['variant'])}")
