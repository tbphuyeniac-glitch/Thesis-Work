"""
E1 ablation on large-scale local scenarios: A0 vs E1_k=1 vs E1_k=3

Variants (all NO SLA penalty — pure E1 regime):
  A0      = exact Gurobi MIP pricing, all neg-RC columns/iter
  E1_k1   = RC-filter only (no MIP), 1 col per (product,period) per iter
  E1_k3   = RC-filter only (no MIP), 3 cols per (product,period) per iter

Scenarios: med_7s3sku (7 stores × 3 SKU) + large_7s4sku (7 stores × 4 SKU)
           — the two largest size classes in the 30-test-scenario set.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parent
SCEN_DIR = REPO / "Test 30 scenarios" / "test_baselines" / "scenarios"
OUT_DIR = REPO / "Result_e1_large_scale"

# E1 = NO SLA penalty
os.environ.pop("IRP_SLA_PENALTY", None)
os.environ["IRP_QUIET"] = "1"

import irp_gurobi_converted as irp

PICKS = [
    ("med_7s3sku",    "base_21__gamma_store__seed573330499"),
    ("med_7s3sku",    "base_21__gamma_store__seed762938026"),
    ("med_7s3sku",    "base_21__normal_global__seed1439190227"),
    ("med_7s3sku",    "base_21__normal_global__seed794957573"),
    ("med_7s3sku",    "base_21__sku_spike__seed449912920"),
    ("large_7s4sku",  "base_24__gamma_store__seed350904184"),
    ("large_7s4sku",  "base_24__gamma_store__seed579708538"),
    ("large_7s4sku",  "base_24__normal_global__seed525727462"),
    ("large_7s4sku",  "base_24__normal_global__seed814874364"),
    ("large_7s4sku",  "base_24__sku_spike__seed992696250"),
]


def _load(name: str):
    path = SCEN_DIR / f"{name}.pkl.gz"
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _run(variant: str, scen: Dict[str, Any]) -> Dict[str, Any]:
    data = copy.deepcopy(scen["shocked_data"])
    base = copy.deepcopy(scen["baseline_sol"])

    if variant == "A0":
        kwargs = dict(
            use_gnn=False, collect_teacher_mode=False,
            runtime_gnn_mode=False, heuristic_top_k_mode=False,
            exact_full_mode=True,
            use_branch_and_price=False,
        )
    elif variant == "E1_k1":
        kwargs = dict(
            use_gnn=False, collect_teacher_mode=False,
            runtime_gnn_mode=False, heuristic_top_k_mode=False,
            rc_filter_mode=True,
            rc_only_cols_per_pp=1,
            use_branch_and_price=False,
        )
    elif variant == "E1_k3":
        kwargs = dict(
            use_gnn=False, collect_teacher_mode=False,
            runtime_gnn_mode=False, heuristic_top_k_mode=False,
            rc_filter_mode=True,
            rc_only_cols_per_pp=3,
            use_branch_and_price=False,
        )
    else:
        raise ValueError(variant)

    pipeline = irp.IRPResearchPipeline(data)
    t0 = time.perf_counter()
    results = pipeline.run_lt_recourse_from_baseline(
        base,
        shock_summary=None,
        use_random_initial_patterns=True,
        n_initial_patterns_per_product_period=5,
        cg_iterations=30,
        msg=False,
        gnn_checkpoint=None,
        use_classical_fallback=False,
        gnn_mass_threshold=0.55,
        gnn_max_keep=150,
        gnn_max_keep_fraction=0.30,
        bp_max_nodes=15,
        bp_max_depth=6,
        lt_activation_threshold=10.0,
        diagnostic_verbosity="summary",
        **kwargs,
    )
    runtime = time.perf_counter() - t0

    cg = results.get("cg_solution")
    rmp_obj = float(getattr(cg, "objective", float("nan"))) if cg else float("nan")
    ep_hist = results.get("cg_episode_history") or []
    cg_iters = len(ep_hist) if ep_hist else 0

    realized = results.get("realized_with_lt_cost_breakdown") or {}
    return {
        "variant": variant,
        "rmp_objective": rmp_obj,
        "runtime_seconds": runtime,
        "cg_iterations": cg_iters,
        "total_realized_cost": float(realized.get("total_realized_operating_cost", float("nan"))),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, scen_name in PICKS:
        print("\n" + "#" * 80)
        print(f"# {label}  ({scen_name})")
        print("#" * 80)
        scen = _load(scen_name)
        d = scen["shocked_data"]
        print(f"  stores={len(d.stores)} products={len(d.products)} periods={len(d.periods)}")

        for variant in ("A0", "E1_k1", "E1_k3"):
            print(f"\n[RUN] {label} / {variant}")
            try:
                r = _run(variant, scen)
            except Exception as exc:
                import traceback; traceback.print_exc()
                r = {"variant": variant, "error": str(exc)}
            r.update({"scenario_label": label, "scenario_name": scen_name,
                       "stores": len(d.stores), "products": len(d.products), "periods": len(d.periods)})
            rows.append(r)
            if "error" not in r:
                print(f"  obj={r['rmp_objective']:.2f}  runtime={r['runtime_seconds']:.2f}s  iters={r['cg_iterations']}")

    # ── Aggregate summary per size class × variant ──
    print("\n" + "=" * 95)
    print(f"{'Size':<16} {'Variant':<10} {'Avg Iters':>10} {'Avg Runtime':>12} {'Avg RMP Obj':>16}  {'Wins':>5}")
    print("=" * 95)

    for label in ("med_7s3sku", "large_7s4sku"):
        label_rows = [r for r in rows if r.get("scenario_label") == label and "error" not in r]
        scen_names = list(dict.fromkeys(r["scenario_name"] for r in label_rows))
        variants = ("A0", "E1_k1", "E1_k3")
        avg = {}
        for v in variants:
            vr = [r for r in label_rows if r["variant"] == v]
            n = len(vr) or 1
            avg[v] = {
                "iters": sum(r["cg_iterations"] for r in vr) / n,
                "rt": sum(r["runtime_seconds"] for r in vr) / n,
                "obj": sum(r["rmp_objective"] for r in vr) / n,
            }
        # count wins (lowest runtime) per scenario
        wins = {v: 0 for v in variants}
        for sn in scen_names:
            sv = {v: next((r for r in label_rows if r["variant"] == v and r["scenario_name"] == sn), None) for v in variants}
            valid = {v: sv[v] for v in variants if sv[v] and "error" not in sv[v]}
            if valid:
                best = min(valid, key=lambda v: valid[v]["runtime_seconds"])
                wins[best] += 1
        for v in variants:
            a = avg[v]
            print(f"{label:<16} {v:<10} {a['iters']:>10.1f} {a['rt']:>11.2f}s {a['obj']:>16.2f}  {wins[v]:>5}/5")
        print()

    out_json = OUT_DIR / "e1_k1_vs_k3_vs_a0.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"[SAVED] {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
