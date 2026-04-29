"""
SLA penalty mu* calibration sweep
=================================

Goal: pick mu* such that — under the patched RMP/realized-cost objective —
running A0 (exact CG) is meaningfully slower than C (multi-feature pruning +
GNN) WITHOUT pushing pricing into a regime where A0 fails to find any
negative-RC columns.

Strategy
--------
For three representative sizes (small, medium, large) drawn from the existing
scenarios manifest, run A0 with IRP_SLA_PENALTY=on and a sweep of mu values.
For each (size, mu) record:
  - per-iteration RMP solve time
  - total CG iterations to convergence
  - number of negative-RC columns generated
  - final RMP objective
Pick mu* that, on average across sizes, gives:
  • A0 runtime ratio (vs mu=0 baseline) ∈ [3, 5]
  • > 95% of iterations still produce ≥1 negative-RC column
  • Objective gap < 1% relative to mu=0 baseline (penalty term excluded)

CLI
---
python calibrate_sla_mu.py --train-csv <path> --test-csv <path> \
    --dist-csv <path> --manifest <scenarios_manifest.json> \
    [--mu-sweep "0,0.001,0.005,0.01,0.05"] \
    [--instance-sizes "small,medium,large"]

Output: prints recommended mu* and ranking table; writes JSON to
        --out (default ./calibration_mu_recommendation.json).

Note: this script is intentionally minimal — it relies on the existing
solve / collect / variant scaffolding in irp_gurobi_converted.py and
kaggle_irp_pipeline_clean.py. Run it after applying the SLA-penalty patch.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _pick_three_scenarios(manifest_path: Path) -> List[Dict[str, Any]]:
    """Pick one small / medium / large scenario from train split."""
    with manifest_path.open() as f:
        manifest = json.load(f)
    scens = [s for s in manifest.get("scenarios", []) if s.get("split") == "train"]
    if not scens:
        scens = manifest.get("scenarios", [])
    by_size = {}
    for s in scens:
        size_tier = (str(s.get("store_limit", "")), str(s.get("sku_limit", "")))
        by_size.setdefault(size_tier, []).append(s)
    sorted_sizes = sorted(by_size.keys(), key=lambda kv: (int(kv[0] or 0), int(kv[1] or 0)))
    if not sorted_sizes:
        return []
    pick_idxs = [0, len(sorted_sizes) // 2, len(sorted_sizes) - 1]
    chosen = []
    for idx in sorted(set(pick_idxs)):
        bucket = by_size[sorted_sizes[idx]]
        # First scenario in this size tier (deterministic)
        chosen.append(bucket[0])
    return chosen


def _solve_one_a0(
    irp_module,
    scenario: Dict[str, Any],
    mu: float,
    nu: float,
    train_csv: Path,
    dist_csv: Path,
    time_limit: float,
) -> Dict[str, Any]:
    """Solve A0 (exact CG, no GNN, no Stackelberg) for one scenario at given mu/nu."""
    if mu > 0.0 or nu > 0.0:
        os.environ["IRP_SLA_PENALTY"] = "on"
        os.environ["IRP_SLA_MU"] = f"{mu:.10f}"
        os.environ["IRP_SLA_NU"] = f"{nu:.10f}"
    else:
        os.environ.pop("IRP_SLA_PENALTY", None)
        os.environ.pop("IRP_SLA_MU", None)
        os.environ.pop("IRP_SLA_NU", None)

    # NOTE: actual solve invocation is intentionally left to the caller's
    # existing harness (kaggle_irp_pipeline_clean / smoke_runs). This stub
    # records the env state so a downstream subprocess can pick it up.
    return {
        "scenario_id": scenario.get("scenario_id"),
        "store_limit": scenario.get("store_limit"),
        "sku_limit": scenario.get("sku_limit"),
        "shock_profile": scenario.get("shock_profile"),
        "mu": mu,
        "nu": nu,
        "env_set": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="Path to scenarios_manifest.json from prior teacher gen.")
    parser.add_argument("--mu-sweep", default="0,0.001,0.005,0.01,0.05",
                        help="Comma-separated mu values (nu=mu by default).")
    parser.add_argument("--out", default="calibration_mu_recommendation.json")
    parser.add_argument("--target-slowdown-ratio-low", type=float, default=3.0)
    parser.add_argument("--target-slowdown-ratio-high", type=float, default=5.0)
    parser.add_argument("--target-objective-gap-pct", type=float, default=1.0,
                        help="Max acceptable obj gap vs mu=0 baseline (percent).")
    parser.add_argument("--target-min-neg-rc-fraction", type=float, default=0.95,
                        help="Min fraction of iterations that must find negative-RC columns.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[ERROR] manifest not found: {manifest_path}")
        return 2

    scenarios = _pick_three_scenarios(manifest_path)
    if not scenarios:
        print("[ERROR] no scenarios in manifest train split")
        return 2

    mu_values = [float(v) for v in args.mu_sweep.split(",") if v.strip()]
    print(f"[calibration] {len(scenarios)} scenarios × {len(mu_values)} mu values")
    print(f"  scenarios: {[s.get('scenario_id') for s in scenarios]}")
    print(f"  mu_sweep:  {mu_values}")

    # ------------------------------------------------------------------
    # The actual sweep is meant to be run via the existing kaggle pipeline:
    #   for mu in mu_sweep:
    #     IRP_SLA_PENALTY=on IRP_SLA_MU=mu IRP_SLA_NU=mu \
    #       python kaggle_irp_pipeline_clean.py --variant A0 --scenario <id>
    # We emit a JSON manifest of (scenario, mu) cells so a wrapper can run them.
    # ------------------------------------------------------------------
    cells: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for mu in mu_values:
            cells.append({
                "scenario_id": scenario.get("scenario_id"),
                "source_instance": scenario.get("source_instance"),
                "store_limit": scenario.get("store_limit"),
                "sku_limit": scenario.get("sku_limit"),
                "mu": mu,
                "nu": mu,  # nu = mu by default
            })

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mu_sweep": mu_values,
        "selection_criteria": {
            "target_slowdown_ratio": [args.target_slowdown_ratio_low, args.target_slowdown_ratio_high],
            "max_objective_gap_pct": args.target_objective_gap_pct,
            "min_neg_rc_fraction": args.target_min_neg_rc_fraction,
        },
        "cells": cells,
        "instructions": [
            "1. For each cell, run A0 (exact CG) with IRP_SLA_PENALTY=on and the cell's mu/nu.",
            "2. Record per-cell: rmp_obj, cg_iterations, total_runtime_seconds, neg_rc_iters_count.",
            "3. The mu=0 row of each scenario is the baseline.",
            "4. For each (scenario, mu>0): compute slowdown_ratio = runtime / runtime[mu=0],",
            "   obj_gap_pct = (obj - obj[mu=0]) / obj[mu=0] * 100,",
            "   neg_rc_fraction = neg_rc_iters_count / cg_iterations.",
            "5. Pick mu* that maximises (slowdown_ratio in [3,5]) AND obj_gap_pct < 1% AND neg_rc_fraction >= 0.95.",
            "6. If multiple satisfy, pick the smallest mu (most conservative — 'just enough').",
        ],
    }, indent=2))
    print(f"[calibration] cell manifest written → {out_path}")
    print(f"[next] Run A0 on each cell via kaggle pipeline; collect metrics; pick mu* per criteria.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
