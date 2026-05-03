"""
Ablation: pruning strength — does aggressive pair-pruning add runtime value?

Two new variants on all 30 test scenarios at MATCHED training regime μ=ν=0.05:

  E2_PRUNE5 = E2 with top_pairs_per_feature=5  (moderate pruning, ~30-60% drop)
  E2_PRUNE3 = E2 with top_pairs_per_feature=3  (aggressive pruning, ~50-80% drop)

Compare against (already in Result_E2_test30_benchmark/ablation_pricing_vs_gnn.json):
  A0_QUAD   = quadratic pricing, no GNN, no aggressive pruning   (baseline ref)
  E2        = quadratic pricing + GNN, top_pairs_per_feature=20  (no effective pruning)

Goal: see if pair-level pruning (independent of GNN ranking) reduces runtime
or hurts (drops critical pairs → more CG iters).
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
OUT_DIR = REPO / "Result_E2_test30_benchmark"
CKPT = REPO / "GNN" / "trained_models" / "irplt_teacher_E2_filtered_local200_fixed" / "bigat" / "pairwise_rank" / "best_model.pt"

# MATCHED regime
os.environ["IRP_SLA_PENALTY"] = "on"
os.environ.setdefault("IRP_SLA_MU", "0.05")
os.environ.setdefault("IRP_SLA_NU", "0.05")
os.environ["IRP_SLA_ALPHA"] = "4.0"
os.environ["IRP_SLA_BETA"] = "2.0"
os.environ["IRP_GNN_SELECTION_MODE"] = "top_frac"
os.environ.setdefault("IRP_GNN_MAX_KEEP_FRAC", "0.30")
os.environ.setdefault("IRP_GNN_MIN_KEEP_FRAC", "0.10")
os.environ.setdefault("IRP_GNN_MIN_KEEP", "5")
os.environ["IRP_QUIET"] = "1"

# ── Monkey-patch LateralTransshipmentCG.__init__ to honor env var ──
import irp_gurobi_converted as irp

_orig_init = irp.LateralTransshipmentCG.__init__
def _patched_init(self, *args, **kwargs):
    _val = os.environ.get("IRP_TOP_PAIRS_PER_FEATURE", "").strip()
    if _val:
        kwargs["top_pairs_per_feature"] = int(_val)
    return _orig_init(self, *args, **kwargs)
irp.LateralTransshipmentCG.__init__ = _patched_init


SCENARIO_GROUPS = {
    "small_4s2sku": ["base_01__gamma_store__seed239081664",
                     "base_01__gamma_store__seed590620972",
                     "base_01__normal_global__seed1373158607",
                     "base_01__normal_global__seed1592467582",
                     "base_01__sku_spike__seed53710185"],
    "med_5s2sku":   ["base_08__gamma_store__seed1629526406",
                     "base_08__gamma_store__seed1738238662",
                     "base_08__normal_global__seed13955984",
                     "base_08__normal_global__seed597409993",
                     "base_08__sku_spike__seed1866808230"],
    "med_4s3sku":   ["base_04__gamma_store__seed201209006",
                     "base_04__gamma_store__seed906070221",
                     "base_04__normal_global__seed1268073013",
                     "base_04__normal_global__seed63989048",
                     "base_04__sku_spike__seed68252794"],
    "med_5s3sku":   ["base_09__gamma_store__seed342865763",
                     "base_09__gamma_store__seed730682428",
                     "base_09__normal_global__seed1730483679",
                     "base_09__normal_global__seed907557513",
                     "base_09__sku_spike__seed1499242942"],
    "med_7s3sku":   ["base_21__gamma_store__seed573330499",
                     "base_21__gamma_store__seed762938026",
                     "base_21__normal_global__seed1439190227",
                     "base_21__normal_global__seed794957573",
                     "base_21__sku_spike__seed449912920"],
    "large_7s4sku": ["base_24__gamma_store__seed350904184",
                     "base_24__gamma_store__seed579708538",
                     "base_24__normal_global__seed525727462",
                     "base_24__normal_global__seed814874364",
                     "base_24__sku_spike__seed992696250"],
}
PICKS = [(label, name) for label, names in SCENARIO_GROUPS.items() for name in names]


def _load_scenario(name: str):
    path = SCEN_DIR / f"{name}.pkl.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _run_variant(variant: str, scen: Dict[str, Any]) -> Dict[str, Any]:
    data = copy.deepcopy(scen["shocked_data"])
    base = copy.deepcopy(scen["baseline_sol"])

    # Set pruning strength via env var (read by patched __init__)
    if variant == "E2_PRUNE5":
        os.environ["IRP_TOP_PAIRS_PER_FEATURE"] = "5"
    elif variant == "E2_PRUNE3":
        os.environ["IRP_TOP_PAIRS_PER_FEATURE"] = "3"
    else:
        os.environ["IRP_TOP_PAIRS_PER_FEATURE"] = "20"

    kwargs = dict(
        use_gnn=True, collect_teacher_mode=False,
        runtime_gnn_mode=True, heuristic_top_k_mode=False,
        pruned_exact_mode=True,
        gnn_selection_mode="top_frac",
        use_branch_and_price=False,
    )

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

    realized_with_lt = results.get("realized_with_lt_cost_breakdown") or {}
    return {
        "variant": variant,
        "rmp_objective": rmp_obj,
        "runtime_seconds": runtime,
        "cg_iterations": cg_iters,
        "lt_cost_realized": float(realized_with_lt.get("lateral_transshipment_cost_realized", float("nan"))),
        "shortage_cost_realized": float(realized_with_lt.get("shortage_cost_realized", float("nan"))),
        "sla_shortage_penalty": float(realized_with_lt.get("sla_shortage_penalty", float("nan"))),
        "sla_surplus_penalty": float(realized_with_lt.get("sla_surplus_penalty", float("nan"))),
        "total_realized_cost": float(realized_with_lt.get("total_realized_operating_cost", float("nan"))),
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

        for variant in ("E2_PRUNE5", "E2_PRUNE3"):
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
            print(f"  obj={r.get('rmp_objective'):.2f}  runtime={r.get('runtime_seconds'):.2f}s  iters={r.get('cg_iterations')}")

    out_json = OUT_DIR / "ablation_pruning_strength.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n[SAVED] {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
