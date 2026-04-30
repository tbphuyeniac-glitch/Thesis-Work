"""
generate_test_baselines.py
==========================
Pre-computes ALNS baselines AND demand-shocked scenarios for the 30 E2 test
scenarios (6 test bases × 5 scenarios each, as identified in E2_run.txt).

Two-level output under --out-dir:
    {out_dir}/bases/{base_id}.pkl.gz
        Contains: {base_spec, base_data (unshocked), baseline_sol}
        One file per base (6 total).  Useful for re-generating scenario pkls
        with different shock params without re-running ALNS.

    {out_dir}/scenarios/{source_instance}.pkl.gz
        Contains: {source_instance, base_dataset_id, shock_type, shock_seed,
                   shock_params, shocked_data (demand modified), baseline_sol}
        One file per scenario (30 total).  benchmark runners (A0, E1_rc_only,
        E1_rc_gnn, E2) can load this and start directly from the CG step.

    {out_dir}/manifest.json
        Index of all 30 scenarios with their pkl paths.

Usage (Kaggle):
    python generate_test_baselines.py \\
        --master-csv /kaggle/working/Thesis-Work/1BISCR501V_90100140_20260323-150407111_filtered_sites.csv \\
        --out-dir    /kaggle/working/Thesis-Work/test_baselines \\
        --time-limit 300 \\
        --repo-root  /kaggle/working/Thesis-Work

Runtime: ~30-40 min on Kaggle T4  (6 ALNS solves + 30 deepcopy+shock, no CG)
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Test base specs (from E2_run.txt CMD line) ────────────────────────────────
# Format: base_id:store_limit:sku_limit:start_date:end_date
TEST_BASES: List[Dict[str, str]] = [
    {"base_dataset_id": "base_01", "store_limit": "4", "sku_limit": "2",
     "start_date": "2025-08-01", "end_date": "2025-12-31"},
    {"base_dataset_id": "base_04", "store_limit": "4", "sku_limit": "3",
     "start_date": "2025-08-03", "end_date": "2025-12-29"},
    {"base_dataset_id": "base_08", "store_limit": "5", "sku_limit": "2",
     "start_date": "2025-08-03", "end_date": "2025-12-29"},
    {"base_dataset_id": "base_09", "store_limit": "5", "sku_limit": "3",
     "start_date": "2025-08-01", "end_date": "2025-12-31"},
    {"base_dataset_id": "base_21", "store_limit": "7", "sku_limit": "3",
     "start_date": "2025-08-01", "end_date": "2025-12-31"},
    {"base_dataset_id": "base_24", "store_limit": "7", "sku_limit": "4",
     "start_date": "2025-08-03", "end_date": "2025-12-29"},
]

# ── Shock parameters (from generate_teacher_scenarios.py:build_scenarios) ────
_SHOCK_PARAMS: Dict[str, Dict[str, str]] = {
    "normal_global": {
        "shock_probability":                    "0.85",
        "max_reallocation_fraction":            "0.60",
        "reallocations_per_product_period":     "3",
        "non_dispatch_shock_multiplier":        "1.8",
    },
    "gamma_store": {
        "shock_probability":                    "0.75",
        "max_reallocation_fraction":            "0.80",
        "reallocations_per_product_period":     "4",
        "non_dispatch_shock_multiplier":        "2.2",
    },
    "sku_spike": {
        "shock_probability":                    "0.95",
        "max_reallocation_fraction":            "0.40",
        "reallocations_per_product_period":     "2",
        "non_dispatch_shock_multiplier":        "2.8",
    },
}

# ── 30 test scenarios (source_instance, shock_type, shock_seed) ──────────────
# Extracted from E2_run.txt scenario split listing.
TEST_SCENARIOS: List[Dict[str, str]] = []
for _base_id, _entries in [
    ("base_01", [
        ("normal_global", "1373158607"),
        ("gamma_store",   "239081664"),
        ("sku_spike",     "53710185"),
        ("normal_global", "1592467582"),
        ("gamma_store",   "590620972"),
    ]),
    ("base_04", [
        ("normal_global", "1268073013"),
        ("gamma_store",   "906070221"),
        ("sku_spike",     "68252794"),
        ("normal_global", "63989048"),
        ("gamma_store",   "201209006"),
    ]),
    ("base_08", [
        ("normal_global", "597409993"),
        ("gamma_store",   "1738238662"),
        ("sku_spike",     "1866808230"),
        ("normal_global", "13955984"),
        ("gamma_store",   "1629526406"),
    ]),
    ("base_09", [
        ("normal_global", "1730483679"),
        ("gamma_store",   "342865763"),
        ("sku_spike",     "1499242942"),
        ("normal_global", "907557513"),
        ("gamma_store",   "730682428"),
    ]),
    ("base_21", [
        ("normal_global", "794957573"),
        ("gamma_store",   "762938026"),
        ("sku_spike",     "449912920"),
        ("normal_global", "1439190227"),
        ("gamma_store",   "573330499"),
    ]),
    ("base_24", [
        ("normal_global", "525727462"),
        ("gamma_store",   "350904184"),
        ("sku_spike",     "992696250"),
        ("normal_global", "814874364"),
        ("gamma_store",   "579708538"),
    ]),
]:
    for _shock_type, _seed in _entries:
        TEST_SCENARIOS.append({
            "base_dataset_id": _base_id,
            "source_instance": f"{_base_id}__{_shock_type}__seed{_seed}",
            "shock_type": _shock_type,
            "shock_seed": _seed,
        })


def _pkl_save(path: Path, obj: Any) -> None:
    """Atomic gzip-pickle write."""
    tmp = path.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.rename(path)


def _build_baseline(
    irp: Any,
    master_csv: str,
    base: Dict[str, str],
    time_limit: int,
) -> Tuple[Any, Any]:
    """Build IRPData and solve ALNS baseline (no shock, no LT, no CG)."""
    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=master_csv,
        sheet_name=None,
        store_limit=int(base["store_limit"]),
        sku_limit=int(base["sku_limit"]),
        start_date=base.get("start_date"),
        end_date=base.get("end_date"),
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
        lt_cost_multiplier=float(os.environ.get("IRP_LT_COST_MULTIPLIER", "1.0")),
    )
    data.dataset_id = base["base_dataset_id"]
    data.scenario_id = "baseline"

    t0 = time.perf_counter()
    baseline_sol = irp.BaselineALNSModel(data).solve(
        msg=False,
        time_limit=time_limit,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
        cw_dispatch_cycle=5,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"  [baseline] {base['base_dataset_id']:8s}  "
        f"obj={float(baseline_sol.objective):.2f}  "
        f"stores={len(data.stores)}  skus={len(data.products)}  "
        f"t={elapsed:.1f}s",
        flush=True,
    )
    return data, baseline_sol


def _apply_shock(
    irp: Any,
    base_data: Any,
    baseline_sol: Any,
    scenario: Dict[str, str],
) -> Any:
    """Deep-copy base_data and apply demand shock. Returns shocked IRPData."""
    shock_type = scenario["shock_type"]
    shock_seed = int(scenario["shock_seed"])
    params = _SHOCK_PARAMS[shock_type]

    data = copy.deepcopy(base_data)
    data.dataset_id = scenario["base_dataset_id"]
    data.scenario_id = f"{shock_type}__seed{shock_seed}"

    irp.apply_hidden_local_reallocation_demand_shocks(
        data,
        baseline_solution=baseline_sol,
        shock_probability=float(params["shock_probability"]),
        max_reallocation_fraction=float(params["max_reallocation_fraction"]),
        reallocations_per_product_period=int(params["reallocations_per_product_period"]),
        non_dispatch_shock_multiplier=float(params["non_dispatch_shock_multiplier"]),
        cw_dispatch_cycle=5,
        seed=shock_seed,
    )
    # realized_demand = shocked demand (deterministic, no further noise).
    # Benchmark runners must NOT overwrite this.
    data.realized_demand = dict(data.demand)
    return data


def main() -> int:
    p = argparse.ArgumentParser(
        description="Pre-compute ALNS baselines + shocked scenarios for 30 E2 test scenarios"
    )
    p.add_argument("--master-csv", required=True)
    p.add_argument("--out-dir", default="test_baselines")
    p.add_argument("--time-limit", type=int, default=300,
                   help="ALNS time limit per base (seconds, default 300)")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing pkl.gz files")
    args = p.parse_args()

    repo_root = args.repo_root or str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import irp_gurobi_converted as irp

    out_dir = Path(args.out_dir)
    bases_dir = out_dir / "bases"
    scenarios_dir = out_dir / "scenarios"
    bases_dir.mkdir(parents=True, exist_ok=True)
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[test_baselines] Generating baselines + shocked scenarios")
    print(f"  master_csv   : {args.master_csv}")
    print(f"  out_dir      : {out_dir}")
    print(f"  time_limit   : {args.time_limit}s/base  (ALNS only, no CG)")
    print(f"  bases        : {len(TEST_BASES)}  (test only)")
    print(f"  scenarios    : {len(TEST_SCENARIOS)}  (5/base × 6 bases)")
    print()

    manifest_scenarios: List[Dict] = []
    n_base_ok = 0
    n_scen_ok = 0

    # ── Phase 1: ALNS baseline per base ──────────────────────────────────────
    print("[Phase 1] ALNS baselines", flush=True)
    base_cache: Dict[str, Tuple[Any, Any]] = {}

    for base in TEST_BASES:
        base_id = base["base_dataset_id"]
        base_pkl = bases_dir / f"{base_id}.pkl.gz"

        if base_pkl.exists() and not args.force:
            print(f"  [skip] {base_id} base pkl already exists — loading for phase 2", flush=True)
            try:
                with gzip.open(base_pkl, "rb") as f:
                    cached = pickle.load(f)
                base_cache[base_id] = (cached["base_data"], cached["baseline_sol"])
                n_base_ok += 1
                continue
            except Exception as e:
                print(f"  [warn] {base_id} cached pkl unreadable ({e}), re-solving", flush=True)

        try:
            base_data, baseline_sol = _build_baseline(irp, args.master_csv, base, args.time_limit)
            _pkl_save(base_pkl, {
                "base_spec":    base,
                "base_data":    base_data,
                "baseline_sol": baseline_sol,
            })
            print(f"  [saved] {base_id} → bases/{base_id}.pkl.gz "
                  f"({base_pkl.stat().st_size // 1024} KB)", flush=True)
            base_cache[base_id] = (base_data, baseline_sol)
            n_base_ok += 1
        except Exception as exc:
            print(f"  [FAILED] {base_id}: {exc}", flush=True)
            base_cache[base_id] = (None, None)

    # ── Phase 2: apply demand shock → save per-scenario pkl ──────────────────
    print(f"\n[Phase 2] Demand shock + save ({len(TEST_SCENARIOS)} scenarios)", flush=True)

    for scen in TEST_SCENARIOS:
        source_instance = scen["source_instance"]
        base_id = scen["base_dataset_id"]
        scen_pkl = scenarios_dir / f"{source_instance}.pkl.gz"

        if scen_pkl.exists() and not args.force:
            print(f"  [skip] {source_instance}", flush=True)
            manifest_scenarios.append({
                "source_instance": source_instance,
                "base_dataset_id": base_id,
                "shock_type": scen["shock_type"],
                "shock_seed": scen["shock_seed"],
                "shock_params": _SHOCK_PARAMS[scen["shock_type"]],
                "pkl_path": str(scen_pkl),
                "status": "cached",
            })
            n_scen_ok += 1
            continue

        base_data, baseline_sol = base_cache.get(base_id, (None, None))
        if base_data is None:
            print(f"  [skip] {source_instance} — base {base_id} failed", flush=True)
            manifest_scenarios.append({
                "source_instance": source_instance,
                "base_dataset_id": base_id,
                "status": f"skipped: base {base_id} failed",
            })
            continue

        try:
            t0 = time.perf_counter()
            shocked_data = _apply_shock(irp, base_data, baseline_sol, scen)
            elapsed = time.perf_counter() - t0

            _pkl_save(scen_pkl, {
                "source_instance": source_instance,
                "base_dataset_id": base_id,
                "shock_type":      scen["shock_type"],
                "shock_seed":      scen["shock_seed"],
                "shock_params":    _SHOCK_PARAMS[scen["shock_type"]],
                "shocked_data":    shocked_data,   # IRPData with demand = shocked demand
                "baseline_sol":    baseline_sol,   # FullIRPTSolution from unshocked ALNS
            })
            size_kb = scen_pkl.stat().st_size // 1024
            print(f"  [saved] {source_instance}  t={elapsed:.2f}s  ({size_kb} KB)", flush=True)
            manifest_scenarios.append({
                "source_instance": source_instance,
                "base_dataset_id": base_id,
                "shock_type":      scen["shock_type"],
                "shock_seed":      scen["shock_seed"],
                "shock_params":    _SHOCK_PARAMS[scen["shock_type"]],
                "pkl_path":        str(scen_pkl),
                "status":          "ok",
            })
            n_scen_ok += 1
        except Exception as exc:
            print(f"  [FAILED] {source_instance}: {exc}", flush=True)
            manifest_scenarios.append({
                "source_instance": source_instance,
                "base_dataset_id": base_id,
                "status": f"failed: {exc}",
            })

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest = {
        "generated_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "master_csv":    args.master_csv,
        "time_limit":    args.time_limit,
        "n_bases":       len(TEST_BASES),
        "n_scenarios":   len(TEST_SCENARIOS),
        "bases_dir":     str(bases_dir),
        "scenarios_dir": str(scenarios_dir),
        "usage": (
            "Load scenarios/{source_instance}.pkl.gz → "
            "{'shocked_data': IRPData, 'baseline_sol': FullIRPTSolution}. "
            "shocked_data.demand and realized_demand are already set. "
            "Pass directly to run_lt_recourse_from_baseline() for A0/E1/E2 CG."
        ),
        "scenarios": manifest_scenarios,
    }
    manifest_path = out_dir / "manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.rename(manifest_path)

    print(f"\n[test_baselines] manifest → {manifest_path}")
    print(f"[test_baselines] done.  bases={n_base_ok}/{len(TEST_BASES)}  "
          f"scenarios={n_scen_ok}/{len(TEST_SCENARIOS)}")
    return 0 if (n_base_ok == len(TEST_BASES) and n_scen_ok == len(TEST_SCENARIOS)) else 1


if __name__ == "__main__":
    sys.exit(main())
