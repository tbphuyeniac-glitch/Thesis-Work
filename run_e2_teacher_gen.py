"""
E2 — Teacher row regeneration WITH SLA penalty (multi-feature regime).
======================================================================

Run AFTER mu* has been calibrated (see calibrate_sla_mu.py).
This is the teacher dataset that will train the multi-feature C variant.

  IRP_SLA_PENALTY = on
  IRP_SLA_MU      = <mu*>           (default 0.005)
  IRP_SLA_NU      = <mu*>           (default 0.005)
  IRP_TEACHER_USE_PRUNED_EXACT_PRICING = 1
  IRP_ALLOW_COLLECT_WITH_EXACT  = 1

Output: under RESULTS_DIR_E2 = /kaggle/working/Results_E2_with_penalty/.
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
        print(f"[E2] ERROR: {target} not found")
        return 2

    # ── E2 configuration: SLA-penalty teacher dataset ────────────────
    env = os.environ.copy()
    env["IRP_SLA_PENALTY"] = os.environ.get("IRP_SLA_PENALTY", "on")
    env["IRP_SLA_MU"] = os.environ.get("IRP_SLA_MU", "0.005")
    env["IRP_SLA_NU"] = os.environ.get("IRP_SLA_NU", "0.005")
    env["IRP_SLA_ALPHA"] = os.environ.get("IRP_SLA_ALPHA", "4.0")
    env["IRP_SLA_BETA"] = os.environ.get("IRP_SLA_BETA", "2.0")
    # E2 learns the multi-feature regime: first screen candidate pairs by the
    # four pricing features, then solve exact pricing over that screened set.
    env["IRP_TEACHER_USE_EXACT_PRICING"] = "0"
    env["IRP_TEACHER_USE_PRUNED_EXACT_PRICING"] = "1"
    env["IRP_ALLOW_COLLECT_WITH_EXACT"] = "1"
    # B&P enabled by default for teacher generation richness — same rationale
    # as run_e1_teacher_gen.py. Disable via IRP_USE_BRANCH_AND_PRICE=0 if needed.
    env.setdefault("IRP_USE_BRANCH_AND_PRICE", "1")
    env.setdefault("IRP_BP_MAX_NODES", "15")
    env.setdefault("IRP_BP_MAX_DEPTH", "6")
    # Same generous CG safety cap as E1 — stopping_mode="convergence"
    # determines the actual stop condition.
    env.setdefault("IRP_CG_ITERATIONS_TEACHER", "50")
    e2_results = repo / "Results_E2_with_penalty"
    env["IRP_RESULTS_DIR_OVERRIDE"] = str(e2_results)
    env["IRP_PHASE_LABEL"] = "scenario"
    # Same resume handshake as run_e1_teacher_gen.py — set IRP_RESUME_E2=1 in
    # the Kaggle notebook before re-launching after a timeout to preserve
    # whatever scenarios already finished.
    if os.environ.get("IRP_RESUME_E2", "0").lower() not in {"0", "false", "no", ""}:
        env["IRP_RESUME_EXISTING_RUN"] = "1"
        env["IRP_CLEAR_RESULTS_DIR"] = "0"
        print("[E2] RESUME mode: keeping existing Results_E2_with_penalty/")
    else:
        env.setdefault("IRP_CLEAR_RESULTS_DIR", "1")
    env.setdefault("IRP_QUIET", "1")

    print(f"[E2] SLA penalty: ON  mu={env['IRP_SLA_MU']}  nu={env['IRP_SLA_NU']}")
    print(f"[E2] alpha={env['IRP_SLA_ALPHA']}  beta={env['IRP_SLA_BETA']}")
    print(f"[E2] results dir: {e2_results}")

    cmd = [sys.executable, "-u", str(target)]
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
