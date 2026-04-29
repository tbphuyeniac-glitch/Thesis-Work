"""
E1 — Teacher row regeneration (NO penalty, pure CG-baseline regime).
====================================================================

Use this on a Kaggle GPU T4×2 notebook (or any host with Gurobi license).
Drives the existing kaggle_irp_pipeline_clean.py via env-var injection so
no code in that file is modified — only env vars steer the run:

  IRP_SLA_PENALTY = off                  (E1 baseline, no SLA penalty)
  IRP_TEACHER_USE_EXACT_PRICING = 1      (use A0-style exact pricing for labels)
  IRP_ALLOW_COLLECT_WITH_EXACT  = 1      (export teacher rows from exact pricing)
  IRP_GNN_FEATURE_MASK = rc_only         (we'll train rc-only model on E1 rows)

Output: aggregate_teacher_rows.csv, scenarios_manifest.json, etc., under
RESULTS_DIR_E1 = /kaggle/working/Results_E1_no_penalty/.

Run:
    python run_e1_teacher_gen.py
or in a Kaggle cell:
    !python /kaggle/working/Thesis-Work/run_e1_teacher_gen.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent
    target = repo / "kaggle_irp_pipeline_clean.py"
    if not target.exists():
        print(f"[E1] ERROR: {target} not found")
        return 2

    # ── E1 configuration: baseline NO-PENALTY teacher dataset ────────
    env = os.environ.copy()
    # SLA penalty OFF — bit-identical objective with legacy pipeline.
    env.pop("IRP_SLA_PENALTY", None)
    env.pop("IRP_SLA_MU", None)
    env.pop("IRP_SLA_NU", None)
    # Tell the engine to use A0 exact pricing during teacher generation
    # (so columns match A0's ground truth) AND to export per-column rows.
    env["IRP_TEACHER_USE_EXACT_PRICING"] = "1"
    env["IRP_ALLOW_COLLECT_WITH_EXACT"] = "1"
    # Enable B&P for teacher collection — adds ~50-100% wall-clock per
    # scenario but yields ~3x more teacher rows + ~2.5x more graph groups
    # + meaningful constraint-hash repetition (vs 0% without B&P). Audit
    # of the legacy 150-scenario dataset confirmed B&P is the source of
    # most ranking-relevant diversity. Override with IRP_USE_BRANCH_AND_PRICE=0
    # in the calling notebook to disable for a fast smoke run.
    env.setdefault("IRP_USE_BRANCH_AND_PRICE", "1")
    env.setdefault("IRP_BP_MAX_NODES", "15")
    env.setdefault("IRP_BP_MAX_DEPTH", "6")
    # CG_ITERATIONS_TEACHER safety cap. Set a generous value because
    # stopping_mode="convergence" already terminates the loop the moment
    # CG finds no negative-RC column (LP optimum). The cap only matters as
    # a guard against runaway loops; under SLA penalty more iterations may
    # be needed to rebalance, so 50 leaves comfortable headroom over the
    # ~4–10 iterations old runs needed.
    env.setdefault("IRP_CG_ITERATIONS_TEACHER", "50")
    # Run separate output directory so E1 rows don't overwrite legacy or E2
    # results. IRP_RESULTS_DIR_OVERRIDE is honored by the patched pipeline
    # (kaggle_irp_pipeline_clean.py:50). Default Kaggle path used when unset.
    e1_results = repo / "Results_E1_no_penalty"
    env["IRP_RESULTS_DIR_OVERRIDE"] = str(e1_results)
    env["IRP_PHASE_LABEL"] = "scenario"
    # Resume support: if Kaggle session was interrupted mid-run, set
    # IRP_RESUME_E1=1 in the calling notebook to PRESERVE the partial Results
    # folder and skip phases that already finished (run_state.json is the
    # source of truth, see RunState).
    if os.environ.get("IRP_RESUME_E1", "0").lower() not in {"0", "false", "no", ""}:
        env["IRP_RESUME_EXISTING_RUN"] = "1"
        env["IRP_CLEAR_RESULTS_DIR"] = "0"
        print("[E1] RESUME mode: keeping existing Results_E1_no_penalty/")
    else:
        env.setdefault("IRP_CLEAR_RESULTS_DIR", "1")
    env.setdefault("IRP_QUIET", "1")
    print("[E1] env (SLA-related): IRP_SLA_PENALTY=<unset> (off)")
    print(f"[E1] results dir: {e1_results}")

    cmd = [sys.executable, "-u", str(target)]
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
