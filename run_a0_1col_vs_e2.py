"""
A0_QUAD_1COL vs E2 — does restricting A0 to 1 column/iter make GNN filter competitive?

Three variants on 1 representative scenario per size class (6 total):

  A0_QUAD     = exact_full_mode + quadratic pricing, ALL neg-RC columns/iter
  A0_QUAD_1COL= exact_full_mode + quadratic pricing, only TOP-1 column/iter (most neg RC)
  E2          = pruned_exact + GNN top-frac + quadratic pricing (~30% of cols/iter)

Hypothesis: A0_QUAD_1COL needs many more iterations than A0_QUAD (adding 1 col/iter is slow
to converge). E2's GNN keeps ~30% so it's between the two in terms of RMP growth rate but
with better column quality. Report: CG iterations, runtime, RMP objective.
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
OUT_DIR = REPO / "Result_a0_1col_vs_e2"
CKPT = REPO / "GNN" / "trained_models" / "irplt_teacher_E2_filtered_local200_fixed" / "bigat" / "pairwise_rank" / "best_model.pt"

# MATCHED regime μ=ν=0.05
os.environ["IRP_SLA_PENALTY"] = "on"
os.environ["IRP_SLA_MU"] = "0.05"
os.environ["IRP_SLA_NU"] = "0.05"
os.environ["IRP_SLA_ALPHA"] = "4.0"
os.environ["IRP_SLA_BETA"] = "2.0"
os.environ["IRP_GNN_SELECTION_MODE"] = "top_frac"
os.environ["IRP_GNN_MAX_KEEP_FRAC"] = "0.30"
os.environ["IRP_GNN_MIN_KEEP_FRAC"] = "0.10"
os.environ["IRP_GNN_MIN_KEEP"] = "5"
os.environ["IRP_QUIET"] = "1"

import irp_gurobi_converted as irp

# ── Monkey-patch: cap pricing_step return to IRP_MAX_COLS_PER_ITER (0 = no cap) ──
_orig_pricing_step = irp.LateralTransshipmentCG.pricing_step

def _patched_pricing_step(self, *args, **kwargs):
    patterns = _orig_pricing_step(self, *args, **kwargs)
    _max = int(os.environ.get("IRP_MAX_COLS_PER_ITER", "0") or 0)
    if _max > 0 and len(patterns) > _max:
        patterns = sorted(
            patterns,
            key=lambda p: float(p.metadata.get("reduced_cost", 0.0) or 0.0),
        )[:_max]
    return patterns

irp.LateralTransshipmentCG.pricing_step = _patched_pricing_step


# One representative scenario per size class
PICKS = [
    ("small_4s2sku",  "base_01__gamma_store__seed239081664"),
    ("med_5s2sku",    "base_08__gamma_store__seed1629526406"),
    ("med_4s3sku",    "base_04__gamma_store__seed201209006"),
    ("med_5s3sku",    "base_09__gamma_store__seed342865763"),
    ("med_7s3sku",    "base_21__gamma_store__seed573330499"),
    ("large_7s4sku",  "base_24__gamma_store__seed350904184"),
]


def _load_scenario(name: str):
    path = SCEN_DIR / f"{name}.pkl.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _run_variant(variant: str, scen: Dict[str, Any]) -> Dict[str, Any]:
    data = copy.deepcopy(scen["shocked_data"])
    base = copy.deepcopy(scen["baseline_sol"])

    if variant == "A0_QUAD":
        os.environ["IRP_MAX_COLS_PER_ITER"] = "0"
        kwargs = dict(
            use_gnn=False, collect_teacher_mode=False,
            runtime_gnn_mode=False, heuristic_top_k_mode=False,
            exact_full_mode=True,
            use_branch_and_price=False,
        )
    elif variant == "A0_QUAD_1COL":
        os.environ["IRP_MAX_COLS_PER_ITER"] = "1"
        kwargs = dict(
            use_gnn=False, collect_teacher_mode=False,
            runtime_gnn_mode=False, heuristic_top_k_mode=False,
            exact_full_mode=True,
            use_branch_and_price=False,
        )
    elif variant == "E2":
        os.environ["IRP_MAX_COLS_PER_ITER"] = "0"
        kwargs = dict(
            use_gnn=True, collect_teacher_mode=False,
            runtime_gnn_mode=True, heuristic_top_k_mode=False,
            pruned_exact_mode=True,
            gnn_selection_mode="top_frac",
            use_branch_and_price=False,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    pipeline = irp.IRPResearchPipeline(data)
    t0 = time.perf_counter()
    results = pipeline.run_lt_recourse_from_baseline(
        base,
        shock_summary=None,
        use_random_initial_patterns=True,
        n_initial_patterns_per_product_period=5,
        cg_iterations=30,
        msg=False,
        gnn_checkpoint=str(CKPT),
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
        "lt_cost_realized": float(realized.get("lateral_transshipment_cost_realized", float("nan"))),
        "shortage_cost_realized": float(realized.get("shortage_cost_realized", float("nan"))),
        "total_realized_cost": float(realized.get("total_realized_operating_cost", float("nan"))),
    }


def main() -> int:
    if not CKPT.exists():
        print(f"[ERROR] checkpoint not found: {CKPT}", file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, scen_name in PICKS:
        print("\n" + "#" * 80)
        print(f"# SCENARIO: {label}  ({scen_name})")
        print("#" * 80)
        try:
            scen = _load_scenario(scen_name)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        d = scen["shocked_data"]
        print(f"  stores={len(d.stores)} products={len(d.products)} periods={len(d.periods)}")

        for variant in ("A0_QUAD", "A0_QUAD_1COL", "E2"):
            print(f"\n[RUN] {label} / {variant}")
            try:
                r = _run_variant(variant, scen)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                r = {"variant": variant, "error": str(exc)}
            r["scenario_label"] = label
            r["scenario_name"] = scen_name
            r["stores"] = len(d.stores)
            r["products"] = len(d.products)
            r["periods"] = len(d.periods)
            rows.append(r)
            obj = r.get("rmp_objective", float("nan"))
            rt  = r.get("runtime_seconds", float("nan"))
            it  = r.get("cg_iterations", "?")
            try:
                print(f"  obj={obj:.2f}  runtime={rt:.2f}s  iters={it}")
            except (TypeError, ValueError):
                print(f"  obj={obj}  runtime={rt}  iters={it}")

    # ── Summary table ──
    print("\n" + "=" * 90)
    print(f"{'Size':<16} {'Variant':<16} {'Iters':>6} {'Runtime':>9} {'RMP Obj':>14}")
    print("=" * 90)
    for r in rows:
        if "error" in r:
            print(f"{r['scenario_label']:<16} {r['variant']:<16}  ERROR: {r['error']}")
        else:
            print(
                f"{r['scenario_label']:<16} {r['variant']:<16} "
                f"{r['cg_iterations']:>6}  "
                f"{r['runtime_seconds']:>8.2f}s  "
                f"{r['rmp_objective']:>14.2f}"
            )

    out_json = OUT_DIR / "a0_1col_vs_e2.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n[SAVED] {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
