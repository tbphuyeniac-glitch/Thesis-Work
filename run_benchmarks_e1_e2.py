"""
Benchmark runners for E1 (no-penalty) and E2 (with-penalty) experiments.
========================================================================

E1 benchmark:  A0_no_penalty   vs E1_rc_only / E1_rc_gnn  ("vanilla CG ablation")
E2 benchmark:  A0_with_penalty vs E2_pruned_gnn           ("SLA-aware CG acceleration")

Both A0 and E2_pruned_gnn solve the SAME exact MIQP pricing under the SLA
penalty; E2 just pre-screens donor/receiver pairs by four pricing features
and re-ranks the resulting column pool with the trained BiGAT. No
Stackelberg in either E1 or E2.

Both run on the 30 test scenarios from scenarios_manifest.json. The two
experiments are kept separate so each is internally consistent (train regime
matches test regime).

The script drives kaggle_irp_pipeline_clean.py via env vars; no editing of
that file is required.

Run:
    python run_benchmarks_e1_e2.py --experiment e1
    python run_benchmarks_e1_e2.py --experiment e2
    python run_benchmarks_e1_e2.py --experiment both     # sequential
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _run_one(experiment: str, mu: str, nu: str, gnn_ckpt: str, feature_mask: str,
             results_dir: Path, n_repeats: int) -> int:
    env = os.environ.copy()

    # Force the C variant to use the right GNN checkpoint AND feature mask.
    env["IRP_GNN_CHECKPOINT"] = str(gnn_ckpt)
    env["IRP_GNN_FEATURE_MASK"] = feature_mask
    # Use proper selection mode that actually prunes.
    env["IRP_GNN_SELECTION_MODE"] = "top_frac"
    env.setdefault("IRP_GNN_MAX_KEEP_FRAC", "0.30")
    env.setdefault("IRP_GNN_MIN_KEEP_FRAC", "0.10")
    env.setdefault("IRP_GNN_MIN_KEEP", "5")

    if experiment == "e1":
        # E1: NO penalty for both A0 and E1_rc_only / E1_rc_gnn.
        env.pop("IRP_SLA_PENALTY", None)
        env.pop("IRP_SLA_MU", None)
        env.pop("IRP_SLA_NU", None)
        env["IRP_BENCHMARK_VARIANTS"] = "e1_only"
        print("[benchmark E1] SLA penalty: OFF — variants=A0_no_penalty / E1_rc_only / E1_rc_gnn")
    elif experiment == "e2":
        env["IRP_SLA_PENALTY"] = "on"
        env["IRP_SLA_MU"] = mu
        env["IRP_SLA_NU"] = nu
        env.setdefault("IRP_SLA_ALPHA", "4.0")
        env.setdefault("IRP_SLA_BETA", "2.0")
        env["IRP_BENCHMARK_VARIANTS"] = "e2_only"
        print(f"[benchmark E2] SLA penalty: ON  mu={mu}  nu={nu} — variants=A0_with_penalty / E2_pruned_gnn")
    else:
        raise ValueError(f"unknown experiment {experiment!r}")

    env["IRP_RESULTS_DIR"] = str(results_dir)
    env["IRP_N_REPEATS_BENCHMARK"] = str(n_repeats)
    env["IRP_PHASE_LABEL"] = "phase2_online_inference"
    env.setdefault("IRP_QUIET", "1")
    # Skip teacher generation phase — we already have rows + checkpoints.
    env["IRP_BENCHMARK_ONLY"] = "1"

    target = REPO / "kaggle_irp_pipeline_clean.py"
    cmd = [sys.executable, "-u", str(target)]
    print(f"[benchmark {experiment.upper()}] launching {target.name} → {results_dir}")
    return subprocess.run(cmd, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["e1", "e2", "both"], default="both")
    parser.add_argument("--mu", default=os.environ.get("IRP_SLA_MU", "0.005"))
    parser.add_argument("--nu", default=os.environ.get("IRP_SLA_NU", "0.005"))
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--e1-gnn-ckpt",
                        default=str(REPO / "Results_E1_no_penalty" / "gnn_training" / "best_valid_prauc.pt"))
    parser.add_argument("--e2-gnn-ckpt",
                        default=str(REPO / "Results_E2_with_penalty" / "gnn_training" / "best_valid_prauc.pt"))
    args = parser.parse_args()

    rc = 0
    if args.experiment in {"e1", "both"}:
        rc = max(rc, _run_one(
            experiment="e1",
            mu="0", nu="0",
            gnn_ckpt=args.e1_gnn_ckpt,
            feature_mask="rc_only",
            results_dir=REPO / "Results_Benchmarks" / "E1_vanilla_a0_vs_c_rc",
            n_repeats=args.n_repeats,
        ))
    if args.experiment in {"e2", "both"}:
        rc = max(rc, _run_one(
            experiment="e2",
            mu=args.mu, nu=args.nu,
            gnn_ckpt=args.e2_gnn_ckpt,
            feature_mask="all",
            results_dir=REPO / "Results_Benchmarks" / "E2_penalty_a0_vs_c_multi",
            n_repeats=args.n_repeats,
        ))
    return rc


if __name__ == "__main__":
    sys.exit(main())
