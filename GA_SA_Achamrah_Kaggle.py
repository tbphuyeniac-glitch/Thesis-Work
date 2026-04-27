# %% [markdown]
# # Achamrah 2022 IRP-T Matheuristic (GA+SA) — Kaggle Run
# Implements the full Achamrah et al. (2022) matheuristic:
#   Phase 1: RMILP constructive heuristic (LP relaxation → cluster sub-MILPs)
#   Phase 2: GA hybridised with SA improvement
# Lateral transshipment is enabled. Outputs match thesis format.
#
# Prerequisites (Kaggle secrets): WLSACCESSID, WLSSECRET, LICENSEID
# Run all cells top-to-bottom.

# %% [code]
# !pip -q install gurobipy   # uncomment if gurobipy is not pre-installed

# =========================================================
# 0) Imports
# =========================================================
from __future__ import annotations

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

RESULTS_DIR = Path("/kaggle/working/Achamrah_Results")

# ── Instance scope (set None for full dataset) ───────────
STORE_LIMIT: Optional[int] = 4    # reduce for faster smoke test; None = all stores
SKU_LIMIT:   Optional[int] = 5    # reduce for faster smoke test; None = all SKUs
START_DATE:  Optional[str] = "2025-08-01"
END_DATE:    Optional[str] = "2025-09-30"

# ── Solver parameters (Achamrah 2022 Table 2 calibration) ─
# These follow the paper's recommended settings.
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

def run_achamrah_pipeline() -> Dict[str, Any]:
    from achamrah_2022_thesis_format_wrapper import (
        DatasetToAchamrahMapper,
        AchamrahThesisFormatPipeline,
        print_lt_plan,
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


def save_and_display(outputs: Dict[str, Any], pipeline: Any) -> None:
    from achamrah_2022_thesis_format_wrapper import print_lt_plan

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    export_paths = pipeline.export_outputs(outputs, str(RESULTS_DIR))

    summary = outputs["summary"]

    print("\n" + "=" * 60)
    print("ACHAMRAH 2022 IRP-T MATHEURISTIC — RESULTS")
    print("=" * 60)
    print(f"  Status                  : {summary['status']}")
    print(f"  Objective (best)        : {summary['best_objective']:.4f}")
    print(f"  Constructive objective  : {summary['constructive_objective']:.4f}"
          if math.isfinite(summary['constructive_objective']) else
          f"  Constructive objective  : N/A")
    print(f"  Final objective         : {summary['final_objective']:.4f}"
          if math.isfinite(summary['final_objective']) else
          f"  Final objective         : N/A")
    print(f"  LT enabled              : {summary['allow_lateral_transshipment']}")
    print(f"  Matheuristic used       : {summary['matheuristic_used']}")
    print(f"  History steps           : {summary['history_length']}")
    print(f"  Routes                  : {summary['n_routes']}")
    print(f"  LT moves                : {summary['n_lt_moves']}")
    print(f"  Constructive runtime    : {summary['constructive_runtime_seconds']:.1f}s")
    print(f"  Improvement runtime     : {summary['improvement_runtime_seconds']:.1f}s")
    print(f"  Total runtime           : {summary['total_runtime_seconds']:.1f}s")

    print("\nObjective breakdown:")
    pprint.pprint(summary["objective_breakdown"])

    print("\nValidation metrics:")
    pprint.pprint(summary["validation_metrics"])

    print("\nLateral Transshipment Plan:")
    print_lt_plan(outputs["lt_plan"], title="Achamrah LT Plan")

    print("\nRoutes (first 20 rows):")
    display(outputs["routes"].head(20))

    print("\nPredicted inventory (first 20 rows):")
    display(outputs["predicted_inventory"].head(20))

    # Save full summary as JSON
    json_path = RESULTS_DIR / "achamrah_summary.json"
    serialisable_summary = {
        k: (v if isinstance(v, (int, float, bool, str, list, type(None))) else str(v))
        for k, v in summary.items()
        if k != "objective_breakdown"
    }
    serialisable_summary["objective_breakdown"] = {
        k: float(v) for k, v in summary["objective_breakdown"].items()
    }
    json_path.write_text(json.dumps(serialisable_summary, indent=2))

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Files written:")
    for key, path in export_paths.items():
        size = Path(path).stat().st_size if Path(path).exists() else 0
        print(f"  {key:35s} {path}  ({size:,} bytes)")
    print(f"  {'summary_json':35s} {json_path}")


# =========================================================
# 5) Entry point
# =========================================================

if __name__ == "__main__":
    # Step 1: clone repo and put it on sys.path
    clone_repo()

    # Step 2: load and verify Gurobi WLS license
    load_gurobi_wls_secrets()
    if not verify_gurobi():
        raise RuntimeError("Gurobi license invalid — cannot proceed.")

    # Step 3: run the matheuristic pipeline
    outputs, pipeline = run_achamrah_pipeline()

    # Step 4: display and save results
    save_and_display(outputs, pipeline)
