"""
SLA penalty μ* calibration sweep — RUNNABLE version.
====================================================

Goal: pick μ* such that under the SLA-penalty regime
  • A0 (full exact pricing + penalty) is **meaningfully slower** than the
    no-penalty baseline (target slowdown 3–5×),
  • but A0 still finds negative-RC columns in **every CG iteration**
    (otherwise pricing terminates prematurely and the comparison is
    no longer LP-fair),
  • and the optimal RMP objective is within ~1% of the μ=0 baseline
    (so A0 and C optimize comparable problems).

This script actually RUNS A0 across a μ sweep on a small base instance
drawn from the master demand CSV, prints a tabular summary, and writes a
JSON recommendation. Designed to fit in <10 minutes of local compute.

CLI
---
  python calibrate_sla_mu.py \\
      --master-csv "1BISCR501V_..._filtered_sites.csv" \\
      [--store-limit 5] [--sku-limit 3] \\
      [--start-date 2025-08-01] [--end-date 2025-08-14] \\
      [--mu-sweep "0,0.001,0.002,0.005,0.01,0.02,0.05"] \\
      [--shock-seed 42] \\
      [--time-limit-alns 60] \\
      [--cg-iterations 30] \\
      [--out calibration_mu_recommendation.json]

Notes
-----
A0 is invoked via `_candidate_patterns_exact_full` (exact_full_mode=True),
NOT the pruned-exact path. The pruned-exact path is only used by E2
TEACHER generation, not by the benchmark, and the calibration target is
the BENCHMARK A0 vs C runtime gap.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _build_data_and_baseline(
    irp,
    master_csv: Path,
    store_limit: int,
    sku_limit: int,
    start_date: str,
    end_date: str,
    time_limit_alns: int,
):
    """Construct an IRPData instance + solve ALNS baseline. Mirrors the recipe
    in GNN/generate_teacher_scenarios.py:_build_base_instance so calibration
    measurements match the production teacher-gen path exactly."""
    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=str(master_csv),
        sheet_name=None,
        store_limit=int(store_limit),
        sku_limit=int(sku_limit),
        start_date=start_date,
        end_date=end_date,
    )
    data, _, _, _ = mapper.build_irp_data(
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
        store_initial_inventory_multiplier=0.2,
        lt_cost_multiplier=1.0,
    )
    data.dataset_id = f"calib_{store_limit}x{sku_limit}"
    data.scenario_id = "calibration"
    baseline_sol = irp.BaselineALNSModel(data).solve(
        msg=False,
        time_limit=time_limit_alns,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
    return data, baseline_sol


def _run_cg_with_mu(
    irp,
    base_data,
    baseline_sol,
    shock_seed: int,
    mu: float,
    nu: float,
    cg_iterations: int,
    *,
    mode: str = "exact_full",  # "exact_full" (A0) or "pruned_exact" (E2)
) -> Dict[str, Any]:
    """Run one CG pass at given μ in either A0 (exact_full) or E2 (pruned_exact)
    mode. Returns runtime + per-iteration MIP size statistics that let us
    compare A0's per-(p,t) MIP load against E2's pruned MIP load directly."""
    if mu > 0.0 or nu > 0.0:
        os.environ["IRP_SLA_PENALTY"] = "on"
        os.environ["IRP_SLA_MU"] = f"{mu:.10f}"
        os.environ["IRP_SLA_NU"] = f"{nu:.10f}"
    else:
        for k in ("IRP_SLA_PENALTY", "IRP_SLA_MU", "IRP_SLA_NU"):
            os.environ.pop(k, None)

    data = copy.deepcopy(base_data)
    irp.apply_hidden_local_reallocation_demand_shocks(
        data,
        baseline_solution=baseline_sol,
        shock_probability=0.85,
        max_reallocation_fraction=0.60,
        reallocations_per_product_period=3,
        non_dispatch_shock_multiplier=1.8,
        cw_dispatch_cycle=5,
        seed=int(shock_seed),
    )
    initial_patterns = irp.generate_random_lt_patterns(
        data,
        baseline_solution=baseline_sol,
        n_patterns_per_product_period=5,
        max_pairs_in_pattern=3,
        lt_activation_threshold=10.0,
        seed=int(shock_seed),
    )
    cg_engine = irp.LateralTransshipmentCG(
        data=data,
        baseline_solution=baseline_sol,
        initial_patterns=initial_patterns,
        lt_activation_threshold=10.0,
        max_pairs_per_pattern=3,
        use_gnn=False,
        collect_teacher_mode=False,
        runtime_gnn_mode=False,
        heuristic_top_k_mode=False,
        exact_full_mode=(mode == "exact_full"),
        pruned_exact_mode=(mode == "pruned_exact"),
    )

    t0 = time.perf_counter()
    cg_sol = cg_engine.run_column_generation(
        max_iter=cg_iterations,
        msg=False,
        stopping_mode="convergence",
    )
    elapsed = time.perf_counter() - t0

    cg_history = list(cg_engine.cg_history or [])
    n_iters = len(cg_history) - 1  # exclude episode 0 (root RMP solve)
    n_iters_neg_rc = sum(
        1 for r in cg_history[1:] if int(r.get("added_columns", 0) or 0) > 0
    )
    n_columns_total = sum(
        int(r.get("added_columns", 0) or 0) for r in cg_history[1:]
    )

    # Per-(p,t) MIP-size telemetry — strongest evidence of the filter benefit.
    # `cg_episode_diagnostics` is populated by the engine each iteration and
    # carries `pairs_after_pruning_unique` for pruned_exact mode and the
    # implicit |donors|*|receivers| pair count for exact_full mode.
    diagnostics = list(cg_engine.cg_episode_diagnostics or [])
    pairs_per_pricing_call: List[int] = []
    for ep in diagnostics:
        if mode == "pruned_exact":
            ppt = ep.get("pruned_exact_per_pt") or []
            for entry in ppt:
                if entry.get("exact_subproblem_solved"):
                    pairs_per_pricing_call.append(int(entry.get("pairs_after_pruning_unique", 0)))
        else:
            # exact_full: every (p,t) MIP processes |donors|×|receivers| pairs.
            n_solved = int(ep.get("exact_subproblems_solved", 0) or 0)
            n_pairs_total = int(ep.get("candidate_pairs_before_pruning", 0) or 0)
            if n_solved > 0:
                pairs_per_pricing_call.extend([n_pairs_total // max(n_solved, 1)] * n_solved)

    avg_pairs = (sum(pairs_per_pricing_call) / len(pairs_per_pricing_call)) if pairs_per_pricing_call else 0
    max_pairs = max(pairs_per_pricing_call) if pairs_per_pricing_call else 0

    return {
        "mode": mode,
        "mu": mu,
        "nu": nu,
        "elapsed_sec": elapsed,
        "rmp_objective": float(cg_sol.objective),
        "cg_iterations": int(n_iters),
        "iterations_with_neg_rc": int(n_iters_neg_rc),
        "total_columns_added": int(n_columns_total),
        "neg_rc_fraction": (n_iters_neg_rc / n_iters) if n_iters > 0 else 0.0,
        "n_pricing_calls": len(pairs_per_pricing_call),
        "avg_pairs_per_mip": round(avg_pairs, 1),
        "max_pairs_per_mip": int(max_pairs),
        "stopping_reason": cg_history[-1].get("stopping_reason", "convergence")
        if cg_history else "n/a",
    }


# Backward-compat alias — old script entry-point name.
_run_a0_with_mu = _run_cg_with_mu


def _pick_mu_star(rows: List[Dict[str, Any]],
                  target_low: float, target_high: float,
                  obj_gap_max_pct: float, neg_rc_min: float) -> Dict[str, Any]:
    """Return the recommended μ* and reasoning."""
    baseline = next((r for r in rows if r["mu"] == 0.0), rows[0])
    base_obj = baseline["rmp_objective"]
    base_time = baseline["elapsed_sec"]

    candidates: List[Dict[str, Any]] = []
    for r in rows:
        if r["mu"] == 0.0:
            continue
        slowdown = r["elapsed_sec"] / max(base_time, 1e-9)
        obj_gap_pct = abs(r["rmp_objective"] - base_obj) / abs(max(base_obj, 1e-9)) * 100.0
        meets_slowdown = target_low <= slowdown <= target_high
        meets_obj = obj_gap_pct <= obj_gap_max_pct
        meets_neg_rc = r["neg_rc_fraction"] >= neg_rc_min
        score = (
            int(meets_slowdown) * 10 + int(meets_obj) * 5 + int(meets_neg_rc) * 5
            - abs(slowdown - (target_low + target_high) / 2)
        )
        candidates.append({
            **r,
            "slowdown_ratio": slowdown,
            "obj_gap_pct": obj_gap_pct,
            "meets_slowdown": meets_slowdown,
            "meets_obj_gap": meets_obj,
            "meets_neg_rc": meets_neg_rc,
            "score": score,
        })

    candidates.sort(key=lambda c: -c["score"])
    return {
        "baseline": baseline,
        "candidates": candidates,
        "recommended_mu": candidates[0]["mu"] if candidates else 0.0,
        "recommendation_meets_all": (
            candidates[0]["meets_slowdown"]
            and candidates[0]["meets_obj_gap"]
            and candidates[0]["meets_neg_rc"]
        ) if candidates else False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--store-limit", type=int, default=5)
    parser.add_argument("--sku-limit", type=int, default=3)
    parser.add_argument("--start-date", default="2025-08-01")
    parser.add_argument("--end-date", default="2025-08-14")
    parser.add_argument("--mu-sweep", default="0,0.001,0.002,0.005,0.01,0.02,0.05")
    parser.add_argument("--shock-seed", type=int, default=42)
    parser.add_argument("--time-limit-alns", type=int, default=60)
    parser.add_argument("--cg-iterations", type=int, default=30)
    parser.add_argument("--out", default="calibration_mu_recommendation.json")
    parser.add_argument("--target-slowdown-low", type=float, default=3.0)
    parser.add_argument("--target-slowdown-high", type=float, default=5.0)
    parser.add_argument("--obj-gap-max-pct", type=float, default=1.0)
    parser.add_argument("--neg-rc-min-fraction", type=float, default=0.95)
    args = parser.parse_args()

    print(f"\n{'=' * 78}\nSLA penalty μ* calibration\n{'=' * 78}")
    print(f"  master-csv:        {args.master_csv}")
    print(f"  base size:         store_limit={args.store_limit}  sku_limit={args.sku_limit}")
    print(f"  date window:       {args.start_date} → {args.end_date}")
    print(f"  μ sweep:           {args.mu_sweep}")
    print(f"  shock seed:        {args.shock_seed}")
    print(f"  CG iter cap:       {args.cg_iterations} (stops on convergence)")
    print(f"  target slowdown:   [{args.target_slowdown_low:.1f}×, {args.target_slowdown_high:.1f}×]")
    print(f"  max obj gap:       {args.obj_gap_max_pct:.1f}%")
    print(f"  min neg-rc frac:   {args.neg_rc_min_fraction:.0%}")

    import irp_gurobi_converted as irp

    print(f"\n[1/3] Building data + ALNS baseline once (shared across all μ)...")
    t0 = time.perf_counter()
    base_data, baseline_sol = _build_data_and_baseline(
        irp,
        master_csv=Path(args.master_csv),
        store_limit=args.store_limit,
        sku_limit=args.sku_limit,
        start_date=args.start_date,
        end_date=args.end_date,
        time_limit_alns=args.time_limit_alns,
    )
    print(f"      done in {time.perf_counter() - t0:.1f}s | "
          f"baseline obj={float(baseline_sol.objective):.2f} | "
          f"stores={len(base_data.stores)} skus={len(base_data.products)} periods={len(base_data.periods)}")

    mu_values = [float(v) for v in args.mu_sweep.split(",") if v.strip()]
    print(f"\n[2/3] Sweeping μ × {{A0 (exact_full), E2 (pruned_exact)}} on the same instance...")
    rows: List[Dict[str, Any]] = []
    for i, mu in enumerate(mu_values, start=1):
        for mode in ("exact_full", "pruned_exact"):
            mode_label = "A0" if mode == "exact_full" else "E2"
            print(f"      [{i}/{len(mu_values)}] μ={mu:.4f} mode={mode_label} ...",
                  end="", flush=True)
            try:
                r = _run_cg_with_mu(
                    irp, base_data, baseline_sol,
                    shock_seed=args.shock_seed,
                    mu=mu, nu=mu,
                    cg_iterations=args.cg_iterations,
                    mode=mode,
                )
                rows.append(r)
                print(
                    f" t={r['elapsed_sec']:.2f}s  iters={r['cg_iterations']}  "
                    f"avg_pairs/MIP={r['avg_pairs_per_mip']:.1f}  "
                    f"obj={r['rmp_objective']:,.0f}",
                    flush=True,
                )
            except Exception as exc:
                print(f" FAILED: {exc}", flush=True)
                rows.append({"mode": mode, "mu": mu, "nu": mu, "error": str(exc)})

    print(f"\n[3/3] Comparison summary — A0 vs E2 across μ values\n")

    # Per-mode tabular summary (A0 first, E2 second) so the operator can
    # compare runtime/MIP-size at the same μ side-by-side.
    valid = [r for r in rows if "error" not in r]
    a0_rows = sorted([r for r in valid if r["mode"] == "exact_full"], key=lambda r: r["mu"])
    e2_rows = sorted([r for r in valid if r["mode"] == "pruned_exact"], key=lambda r: r["mu"])

    print(f"{'μ':>7} | {'A0_time':>8} {'A0_iters':>8} {'A0_avgPairs':>11} {'A0_obj':>14} | "
          f"{'E2_time':>8} {'E2_iters':>8} {'E2_avgPairs':>11} {'E2_obj':>14} | "
          f"{'speedup':>8} {'objgap%':>8}")
    print("-" * 130)
    for a, e in zip(a0_rows, e2_rows):
        speedup = a["elapsed_sec"] / max(e["elapsed_sec"], 1e-9)
        obj_gap = abs(e["rmp_objective"] - a["rmp_objective"]) / max(abs(a["rmp_objective"]), 1e-9) * 100.0
        print(
            f"{a['mu']:>7.4f} | "
            f"{a['elapsed_sec']:>8.2f} {a['cg_iterations']:>8d} {a['avg_pairs_per_mip']:>11.1f} "
            f"{a['rmp_objective']:>14,.0f} | "
            f"{e['elapsed_sec']:>8.2f} {e['cg_iterations']:>8d} {e['avg_pairs_per_mip']:>11.1f} "
            f"{e['rmp_objective']:>14,.0f} | "
            f"{speedup:>7.2f}× {obj_gap:>7.2f}%"
        )

    print(f"\n[3/3] Picking μ* per criteria (using A0 vs μ=0 baseline as the reference)...")
    pick = _pick_mu_star(
        a0_rows,
        target_low=args.target_slowdown_low,
        target_high=args.target_slowdown_high,
        obj_gap_max_pct=args.obj_gap_max_pct,
        neg_rc_min=args.neg_rc_min_fraction,
    )

    print()
    if pick["candidates"]:
        rec = pick["candidates"][0]
        if pick["recommendation_meets_all"]:
            print(f"⭐ RECOMMENDED μ* = {rec['mu']:.4f}  "
                  f"(slowdown {rec['slowdown_ratio']:.2f}×, obj gap {rec['obj_gap_pct']:.2f}%, "
                  f"neg-rc {rec['neg_rc_fraction']:.0%})")
        else:
            print(f"⚠️  Best candidate μ = {rec['mu']:.4f} doesn't meet ALL criteria.")
            if not rec["meets_slowdown"]:
                print(f"     • slowdown={rec['slowdown_ratio']:.2f}× outside "
                      f"[{args.target_slowdown_low}, {args.target_slowdown_high}]")
            if not rec["meets_obj_gap"]:
                print(f"     • obj_gap={rec['obj_gap_pct']:.2f}% > {args.obj_gap_max_pct}%")
            if not rec["meets_neg_rc"]:
                print(f"     • neg_rc_fraction={rec['neg_rc_fraction']:.0%} < {args.neg_rc_min_fraction:.0%}")
            print(f"     → Consider widening the μ-sweep range or relaxing target criteria.")
    else:
        print("❌ No μ candidate produced a finite result. Check master CSV / data window.")
        return 2

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "store_limit": args.store_limit, "sku_limit": args.sku_limit,
            "start_date": args.start_date, "end_date": args.end_date,
            "shock_seed": args.shock_seed,
            "cg_iterations": args.cg_iterations,
            "mu_sweep": mu_values,
        },
        "criteria": {
            "target_slowdown_range": [args.target_slowdown_low, args.target_slowdown_high],
            "obj_gap_max_pct": args.obj_gap_max_pct,
            "neg_rc_min_fraction": args.neg_rc_min_fraction,
        },
        "rows": rows,
        "baseline_mu0": pick["baseline"],
        "candidates_ranked": pick["candidates"],
        "recommended_mu": pick["candidates"][0]["mu"] if pick["candidates"] else None,
        "recommendation_meets_all_criteria": pick["recommendation_meets_all"],
    }, indent=2, default=str))
    print(f"\n[done] full report → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
