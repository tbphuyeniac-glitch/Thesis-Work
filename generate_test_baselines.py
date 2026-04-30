"""
generate_test_baselines.py
==========================
Pre-computes ALNS baseline solutions for the 30 E2 test scenarios
(6 test bases × 5 scenarios each, as identified in E2_run.txt).

Saves one pkl.gz per base to:
    {out_dir}/{base_id}.pkl.gz   — contains {base, base_data, baseline_sol}
    {out_dir}/manifest.json      — scenario_id → base_id + shock params

Usage (Kaggle):
    python generate_test_baselines.py \
        --master-csv /kaggle/working/Thesis-Work/1BISCR501V_90100140_20260323-150407111_filtered_sites.csv \
        --out-dir /kaggle/working/Thesis-Work/test_baselines \
        --time-limit 300 \
        --repo-root /kaggle/working/Thesis-Work

Why:
    generate_teacher_scenarios.py solves ALNS once per base and keeps it only
    in memory (base_cache).  After the E2 run ends those solutions are lost.
    This script re-computes just the 6 test-base ALNS solutions and persists
    them so the benchmark runner can skip re-solving ALNS for every A0/C trial.

    6 bases × ~5 min/base ≈ 30 min on Kaggle T4.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Test base specs (extracted from E2_run.txt CMD line) ─────────────────────
# Format comes from kaggle_irp_pipeline_clean.py's build_normalized_base_specs()
# which serialises as:  base_id:store_limit:sku_limit:start_date:end_date
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

# ── 30 test scenarios (source_instance, shock_seed, shock_params) ────────────
# Seeds and shock types extracted from E2_run.txt.
# Shock params come from generate_teacher_scenarios.py:build_scenarios().
_SHOCK_PARAMS = {
    "normal_global": {
        "IRP_DEMAND_SHOCK_PROBABILITY": "0.85",
        "IRP_DEMAND_SHOCK_REALLOCATION_FRACTION": "0.60",
        "IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD": "3",
        "IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER": "1.8",
    },
    "gamma_store": {
        "IRP_DEMAND_SHOCK_PROBABILITY": "0.75",
        "IRP_DEMAND_SHOCK_REALLOCATION_FRACTION": "0.80",
        "IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD": "4",
        "IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER": "2.2",
    },
    "sku_spike": {
        "IRP_DEMAND_SHOCK_PROBABILITY": "0.95",
        "IRP_DEMAND_SHOCK_REALLOCATION_FRACTION": "0.40",
        "IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD": "2",
        "IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER": "2.8",
    },
}

# source_instance → {base_id, shock_type, shock_seed, shock_params}
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
            **_SHOCK_PARAMS[_shock_type],
        })


def _build_baseline(irp: Any, master_csv: str, base: Dict[str, str], time_limit: int) -> Tuple[Any, Any]:
    """Build IRPData and solve ALNS baseline for one base spec (no shocks, no LT)."""
    store_limit = int(base["store_limit"])
    sku_limit = int(base["sku_limit"])
    start_date = base.get("start_date")
    end_date = base.get("end_date")
    lt_cost_multiplier = float(os.environ.get("IRP_LT_COST_MULTIPLIER", "1.0"))

    mapper = irp.DatasetToIRPValidationMapper(
        excel_path=master_csv,
        sheet_name=None,
        store_limit=store_limit,
        sku_limit=sku_limit,
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
        lt_cost_multiplier=lt_cost_multiplier,
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


def main() -> int:
    p = argparse.ArgumentParser(
        description="Pre-compute ALNS baselines for the 30 E2 test scenarios"
    )
    p.add_argument("--master-csv", required=True, help="Path to training data CSV")
    p.add_argument("--out-dir", default="test_baselines", help="Output directory for pkl.gz files")
    p.add_argument("--time-limit", type=int, default=300, help="ALNS time limit per base (seconds)")
    p.add_argument("--repo-root", default=None, help="Path to cloned repo root")
    p.add_argument("--force", action="store_true", help="Overwrite existing pkl.gz files")
    args = p.parse_args()

    repo_root = args.repo_root or str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import irp_gurobi_converted as irp

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_bases = len(TEST_BASES)
    n_scenarios = len(TEST_SCENARIOS)
    print(f"\n[test_baselines] Pre-computing ALNS baselines")
    print(f"  master_csv : {args.master_csv}")
    print(f"  out_dir    : {out_dir}")
    print(f"  time_limit : {args.time_limit}s per base")
    print(f"  bases      : {n_bases}  (test only: base_01,04,08,09,21,24)")
    print(f"  scenarios  : {n_scenarios}  ({n_bases} bases × 5 scenarios each)")
    print()

    base_status: Dict[str, str] = {}
    for base in TEST_BASES:
        base_id = base["base_dataset_id"]
        pkl_path = out_dir / f"{base_id}.pkl.gz"

        if pkl_path.exists() and not args.force:
            size_kb = pkl_path.stat().st_size // 1024
            print(f"  [skip] {base_id} — already exists ({size_kb} KB)  pass --force to overwrite",
                  flush=True)
            base_status[base_id] = "cached"
            continue

        try:
            base_data, baseline_sol = _build_baseline(irp, args.master_csv, base, args.time_limit)
            # Atomic write: write to .tmp then rename
            tmp_path = pkl_path.with_suffix(".gz.tmp")
            with gzip.open(tmp_path, "wb") as f:
                pickle.dump(
                    {"base_spec": base, "base_data": base_data, "baseline_sol": baseline_sol},
                    f,
                    protocol=4,
                )
            tmp_path.rename(pkl_path)
            size_kb = pkl_path.stat().st_size // 1024
            print(f"  [saved] {base_id} → {pkl_path.name} ({size_kb} KB)", flush=True)
            base_status[base_id] = "ok"
        except Exception as exc:
            print(f"  [FAILED] {base_id}: {exc}", flush=True)
            base_status[base_id] = f"failed: {exc}"

    # Save manifest: scenario_id → {base_id, shock_seed, shock_params, pkl_path}
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "master_csv": args.master_csv,
        "out_dir": str(out_dir),
        "time_limit_per_base": args.time_limit,
        "bases": {
            b["base_dataset_id"]: {
                **b,
                "pkl_path": str(out_dir / f"{b['base_dataset_id']}.pkl.gz"),
                "status": base_status.get(b["base_dataset_id"], "unknown"),
            }
            for b in TEST_BASES
        },
        "scenarios": {
            s["source_instance"]: {
                "base_dataset_id": s["base_dataset_id"],
                "shock_type": s["shock_type"],
                "shock_seed": s["shock_seed"],
                "shock_params": {
                    k: v for k, v in s.items()
                    if k.startswith("IRP_DEMAND_SHOCK")
                },
                "pkl_path": str(out_dir / f"{s['base_dataset_id']}.pkl.gz"),
            }
            for s in TEST_SCENARIOS
        },
    }
    manifest_path = out_dir / "manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2))
    tmp_manifest.rename(manifest_path)
    print(f"\n[test_baselines] manifest → {manifest_path}")

    n_ok = sum(1 for s in base_status.values() if s in ("ok", "cached"))
    n_fail = sum(1 for s in base_status.values() if s.startswith("failed"))
    print(f"[test_baselines] done.  {n_ok}/{n_bases} bases OK  {n_fail} failed")
    if n_fail:
        print(f"  Failed bases: {[k for k, v in base_status.items() if v.startswith('failed')]}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
