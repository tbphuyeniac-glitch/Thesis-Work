from __future__ import annotations

"""GNN/generate_teacher_scenarios.py — teacher-data scenario enumerator.

Runs the IRP-LT pipeline once per (base_dataset, scenario) combination and
aggregates the exported teacher rows so the downstream GNN pipeline has enough
diversity for a real instance-level train / valid / test split.

A "base dataset" is a filtered slice of the master retail CSV (by store /
SKU / date). A "scenario" is one shock profile applied to a base. Together
they form the unit of identity used for splitting:

    source_instance := "<base_dataset_id>__<scenario_id>"

Architecture (v2 — shared ALNS baseline)
-----------------------------------------
Previously each scenario spawned a fresh subprocess that re-ran the full
baseline ALNS → shock → CG chain independently.  With S=5 scenarios per
base and B=30 bases, that meant 150 identical baseline ALNS solves.

v2 replaces the per-scenario subprocess with an in-process loop:

    for each unique base spec:
        build IRPData once
        solve ALNS baseline ONCE          ← single solve per base
        for each scenario (shock seed):
            data_copy = deepcopy(base_data)   ← O(ms) copy
            apply shock to data_copy
            run CG with collect_teacher_mode=True
            collect teacher rows

ALNS solves drop from B×S (150) to B (30). All other solver work is
unchanged: CG runs one full execution per scenario as before.

Output layout
-------------
    Results/scenarios/
        scenarios_manifest.json       — full list of (base, scenario) runs
        aggregate_teacher_rows.csv    — concatenated teacher rows across runs
        run_<NNN>__<base>__<scenario>/
            cg_teacher_dataset.csv    — raw per-run teacher rows
            run_metadata.json

Split semantics
---------------
split_by='base' (default): all scenarios from the same base stay in the same
split — prevents topology leakage across train/valid/test.

CLI
---
--master-csv                Master retail CSV path.
--bases                     List of base-dataset specs "name:store_limit:sku_limit[:start:end]".
--scenarios-per-base        Number of shock scenarios per base (default 10).
--cg-iterations             CG iterations per run (default 5).
--time-limit                Per-run time limit seconds for the ALNS baseline (default 300).
--shock-distributions       Comma-separated shock profiles (default all three, rotated).
--out-dir                   Output directory (default Results/scenarios).
--split-by                  "base" (default) or "scenario".
--train-ratio / --valid-ratio
                            Base-dataset split ratios (default 0.60 / 0.20).
--master-seed               Reproducibility master seed.
--dry-run                   Print the plan and exit without running.
--continue-on-failure       Skip failing scenarios/bases instead of aborting.
"""

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _raise_csv_field_size_limit() -> int:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit = int(limit / 10)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    _raise_csv_field_size_limit()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: set = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_typed_rows_to_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Like write_csv_rows but accepts mixed-type values (converts to str)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: set = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else str(v)) for k, v in row.items()})


def _rows_to_str_dicts(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert mixed-type teacher row dicts to all-string dicts for CSV aggregation."""
    return [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in rows]


# ---------------------------------------------------------------------------
# Scenario / split construction (unchanged from v1)
# ---------------------------------------------------------------------------

def parse_base_spec(spec: str) -> Dict[str, str]:
    parts = spec.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"base spec must be 'name:store_limit:sku_limit[:start_date[:end_date]]', got {spec!r}"
        )
    return {
        "base_dataset_id": parts[0].strip() or "base",
        "store_limit": parts[1].strip() or "10",
        "sku_limit": parts[2].strip() or "5",
        "start_date": parts[3].strip() if len(parts) >= 4 else "None",
        "end_date": parts[4].strip() if len(parts) >= 5 else "None",
    }


def build_scenarios(
    bases: List[Dict[str, str]],
    scenarios_per_base: int,
    shock_distributions: List[str],
    master_seed: int,
) -> List[Dict[str, str]]:
    """Return list of scenario dicts. Each scenario = one CG teacher run."""
    rng = random.Random(master_seed)
    scenarios: List[Dict[str, str]] = []
    for base in bases:
        for scenario_idx in range(scenarios_per_base):
            shock_profile = shock_distributions[scenario_idx % len(shock_distributions)]
            seed = rng.randrange(1, 2**31 - 1)
            if shock_profile == "normal_global":
                shock_params = {
                    "IRP_DEMAND_SHOCK_PROBABILITY": "0.85",
                    "IRP_DEMAND_SHOCK_REALLOCATION_FRACTION": "0.60",
                    "IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD": "3",
                    "IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER": "1.8",
                }
            elif shock_profile == "gamma_store":
                shock_params = {
                    "IRP_DEMAND_SHOCK_PROBABILITY": "0.75",
                    "IRP_DEMAND_SHOCK_REALLOCATION_FRACTION": "0.80",
                    "IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD": "4",
                    "IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER": "2.2",
                }
            elif shock_profile == "sku_spike":
                shock_params = {
                    "IRP_DEMAND_SHOCK_PROBABILITY": "0.95",
                    "IRP_DEMAND_SHOCK_REALLOCATION_FRACTION": "0.40",
                    "IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD": "2",
                    "IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER": "2.8",
                }
            else:
                raise ValueError(f"unknown shock profile {shock_profile!r}")
            scenario_id = f"{shock_profile}__seed{seed}"
            source_instance = f"{base['base_dataset_id']}__{scenario_id}"
            scenarios.append({
                **base,
                "scenario_idx": scenario_idx,
                "scenario_id": scenario_id,
                "shock_profile": shock_profile,
                "shock_seed": seed,
                "source_instance": source_instance,
                **shock_params,
            })
    return scenarios


def assign_split(
    scenarios: List[Dict[str, str]],
    split_by: str,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Dict[str, str]:
    """Return mapping source_instance -> 'train' / 'valid' / 'test'."""
    rng = random.Random(seed)
    if split_by == "base":
        bases = sorted({scen["base_dataset_id"] for scen in scenarios})
        shuffled = list(bases)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_valid = max(1, int(round(n * valid_ratio))) if n >= 3 else 0
        if n_train + n_valid >= n:
            n_valid = max(0, n - n_train - 1)
        train_set = set(shuffled[:n_train])
        valid_set = set(shuffled[n_train:n_train + n_valid])
        assignment: Dict[str, str] = {}
        for scen in scenarios:
            key = scen["source_instance"]
            base = scen["base_dataset_id"]
            if base in train_set:
                assignment[key] = "train"
            elif base in valid_set:
                assignment[key] = "valid"
            else:
                assignment[key] = "test"
        return assignment

    # per-scenario split
    keys = [scen["source_instance"] for scen in scenarios]
    shuffled = list(keys)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio)))
    n_valid = max(1, int(round(n * valid_ratio))) if n >= 3 else 0
    if n_train + n_valid >= n:
        n_valid = max(0, n - n_train - 1)
    train_set = set(shuffled[:n_train])
    valid_set = set(shuffled[n_train:n_train + n_valid])
    return {
        key: ("train" if key in train_set else "valid" if key in valid_set else "test")
        for key in keys
    }


# ---------------------------------------------------------------------------
# In-process IRP helpers (v2 — replaces subprocess run_pipeline)
# ---------------------------------------------------------------------------

def _load_irp_module(project_root: Path) -> Any:
    """Import irp_gurobi_converted from the project root (once per process)."""
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module("irp_gurobi_converted")


def _build_base_instance(
    irp: Any,
    master_csv: str,
    base: Dict[str, str],
    time_limit: int,
) -> Tuple[Any, Any]:
    """Build IRPData and solve ALNS baseline for one base spec.

    The returned (base_data, baseline_sol) are read-only from the caller's
    perspective: each scenario must deepcopy(base_data) before applying a
    shock so the original stays clean for the next scenario.

    baseline_sol is shared across all scenarios of this base — it is computed
    from the original (unshocked) demand and does not depend on the shock seed.
    """
    store_limit = int(base["store_limit"])
    sku_limit = int(base["sku_limit"])
    start_date = base.get("start_date") if base.get("start_date") not in (None, "None") else None
    end_date = base.get("end_date") if base.get("end_date") not in (None, "None") else None
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
        f"  [baseline] {base['base_dataset_id']:24s}  "
        f"obj={float(baseline_sol.objective):.2f}  "
        f"stores={len(data.stores)}  skus={len(data.products)}  "
        f"t={elapsed:.1f}s"
    )
    return data, baseline_sol


def _run_scenario_inprocess(
    irp: Any,
    base_data: Any,
    baseline_sol: Any,
    scenario: Dict[str, str],
    run_dir: Path,
    cg_iterations: int,
) -> Tuple[bool, List[Dict[str, Any]], float]:
    """Run CG teacher collection for one scenario, fully in-process.

    Steps:
      1. deepcopy(base_data) — isolates this scenario from others
      2. apply_hidden_local_reallocation_demand_shocks on the copy
      3. run LateralTransshipmentCG with collect_teacher_mode=True
      4. tag every teacher row with source_instance
      5. write per-run CSV + metadata for resume detection

    Returns (ok, teacher_rows_as_dicts, runtime_seconds).
    baseline_sol is shared (read-only) across all scenarios of the same base.

    Design decisions vs the old subprocess path (__main__ defaults):
    - lt_activation_threshold=10.0  matches Phase 2 inference conditions so
      the GNN trains on the same pair distribution it prices at runtime.
    - n_initial_patterns_per_product_period=5  matches __main__ default.
    - use_branch_and_price=False  intentionally omitted: B&P was ON by default
      in the old path (IRP_USE_BRANCH_AND_PRICE=1) but added up to 15×CG-iter
      of extra work per scenario.  With 150 scenarios the runtime cost outweighs
      the marginal gain from B&P-node pricing examples; root CG rows at the
      correct threshold already cover the inference distribution.
    """
    source_instance = scenario["source_instance"]
    shock_seed = int(scenario["shock_seed"])
    # Match the production lt_activation_threshold so teacher rows cover the
    # same pair distribution the GNN will encounter during Phase 2 pricing.
    lt_activation_threshold = float(
        os.environ.get("IRP_LT_ACTIVATION_THRESHOLD", "10.0")
    )
    t0 = time.perf_counter()
    try:
        # Isolate this scenario — shock modifies demand in-place.
        data = copy.deepcopy(base_data)
        data.dataset_id = scenario["base_dataset_id"]
        data.scenario_id = scenario["scenario_id"]

        irp.apply_hidden_local_reallocation_demand_shocks(
            data,
            baseline_solution=baseline_sol,
            shock_probability=float(scenario.get("IRP_DEMAND_SHOCK_PROBABILITY", 0.85)),
            max_reallocation_fraction=float(scenario.get("IRP_DEMAND_SHOCK_REALLOCATION_FRACTION", 0.60)),
            reallocations_per_product_period=int(scenario.get("IRP_DEMAND_SHOCK_REALLOCATIONS_PER_PRODUCT_PERIOD", 3)),
            non_dispatch_shock_multiplier=float(scenario.get("IRP_DEMAND_SHOCK_NON_DISPATCH_MULTIPLIER", 1.8)),
            cw_dispatch_cycle=5,
            seed=shock_seed,
        )

        initial_patterns = irp.generate_random_lt_patterns(
            data,
            baseline_solution=baseline_sol,
            n_patterns_per_product_period=5,
            max_pairs_in_pattern=3,
            lt_activation_threshold=lt_activation_threshold,
            seed=shock_seed,
        )

        cg_engine = irp.LateralTransshipmentCG(
            data=data,
            baseline_solution=baseline_sol,
            initial_patterns=initial_patterns,
            lt_activation_threshold=lt_activation_threshold,
            max_pairs_per_pattern=3,
            use_gnn=False,
            collect_teacher_mode=True,
            runtime_gnn_mode=False,
            heuristic_top_k_mode=False,
            exact_full_mode=False,
        )
        cg_sol = cg_engine.run_column_generation(
            max_iter=cg_iterations,
            msg=False,
            stopping_mode="convergence",
        )

        rows: List[Dict[str, Any]] = list(cg_engine.teacher_dataset_rows or [])
        # Tag every row with the source_instance so downstream dedup and
        # graph splitting work correctly (old path did this via IRP_SOURCE_INSTANCE env var).
        for row in rows:
            row["source_instance"] = source_instance
            row.setdefault("base_dataset_id", scenario["base_dataset_id"])
            row.setdefault("scenario_id", scenario["scenario_id"])

        runtime = time.perf_counter() - t0
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_typed_rows_to_csv(run_dir / "cg_teacher_dataset.csv", rows)
        with open(run_dir / "run_metadata.json", "w") as mf:
            json.dump({
                "source_instance": source_instance,
                "base_dataset_id": scenario["base_dataset_id"],
                "scenario_id": scenario["scenario_id"],
                "shock_seed": shock_seed,
                "shock_profile": scenario.get("shock_profile", ""),
                "n_teacher_rows": len(rows),
                "runtime_seconds": runtime,
                "cg_objective": float(cg_sol.objective) if cg_sol else None,
                "cg_iterations_run": int(getattr(cg_sol, "iterations_run", 0)),
                "baseline_shared": True,
                "lt_activation_threshold": lt_activation_threshold,
                "n_initial_patterns_per_product_period": 5,
                "branch_and_price_used": False,
            }, mf, indent=2)

        print(
            f"  [scenario] {source_instance:<56s}  "
            f"rows={len(rows):4d}  obj={float(cg_sol.objective):.2f}  "
            f"t={runtime:.1f}s"
        )
        return True, rows, runtime

    except Exception as exc:
        runtime = time.perf_counter() - t0
        print(f"  [scenario] {source_instance} FAILED after {runtime:.1f}s: {exc}")
        traceback.print_exc()
        return False, [], runtime


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    default_bases = [
        "base_a:10:5:None:None",
        "base_b:15:4:None:None",
        "base_c:8:6:None:None",
    ]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--master-csv",
        default=str(
            Path(__file__).resolve().parents[1]
            / "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"
        ),
    )
    parser.add_argument("--bases", nargs="+", default=default_bases)
    parser.add_argument("--scenarios-per-base", type=int, default=10)
    parser.add_argument("--cg-iterations", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=300,
                        help="ALNS baseline time limit per base spec (seconds).")
    parser.add_argument("--shock-distributions", default="normal_global,gamma_store,sku_spike")
    parser.add_argument("--out-dir", default="Results/scenarios")
    parser.add_argument("--split-by", choices=["base", "scenario"], default="base")
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--valid-ratio", type=float, default=0.20)
    parser.add_argument("--master-seed", type=int, default=20260423)
    # --pipeline-script is no longer used (in-process replaced subprocess) but
    # kept for CLI backwards-compatibility so existing invocations don't break.
    parser.add_argument("--pipeline-script", default="irp_gurobi_converted.py",
                        help="[DEPRECATED] Ignored. Scenarios now run in-process.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bases = [parse_base_spec(spec) for spec in args.bases]
    scenarios = build_scenarios(
        bases=bases,
        scenarios_per_base=args.scenarios_per_base,
        shock_distributions=[s.strip() for s in args.shock_distributions.split(",") if s.strip()],
        master_seed=args.master_seed,
    )
    split_assignment = assign_split(
        scenarios=scenarios,
        split_by=args.split_by,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.master_seed,
    )

    manifest: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "master_csv": args.master_csv,
        "master_seed": args.master_seed,
        "split_by": args.split_by,
        "split_ratios": {
            "train": args.train_ratio,
            "valid": args.valid_ratio,
            "test": round(1.0 - args.train_ratio - args.valid_ratio, 4),
        },
        "bases": bases,
        "scenarios": [],
        "split_assignment": split_assignment,
        "run_directory": str(out_dir),
        "aggregate_csv": str(out_dir / "aggregate_teacher_rows.csv"),
        "baseline_shared_per_base": True,
        "build_teacher_graph_command": (
            f"python GNN/build_teacher_graph_dataset.py "
            f"--teacher-csv {out_dir / 'aggregate_teacher_rows.csv'} "
            f"--out-dir GNN/data/irplt_teacher"
        ),
    }

    n_bases = len(bases)
    n_scenarios = len(scenarios)
    print(f"[generate] bases={n_bases}  scenarios={n_scenarios}  split_by={args.split_by}")
    print(f"[generate] ALNS solves: {n_bases} (shared baseline, was {n_scenarios})")
    for scen in scenarios:
        print(f"  {scen['source_instance']:<60s} → {split_assignment[scen['source_instance']]}")

    if args.dry_run:
        manifest_path = out_dir / "scenarios_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[generate] dry-run — wrote plan to {manifest_path}")
        return

    # ------------------------------------------------------------------
    # Load irp module once — shared across all in-process scenario runs.
    # ------------------------------------------------------------------
    print(f"\n[generate] importing irp_gurobi_converted from {project_root}")
    try:
        irp = _load_irp_module(project_root)
    except Exception as exc:
        print(f"[generate] FATAL: could not import irp_gurobi_converted: {exc}")
        raise

    # ------------------------------------------------------------------
    # Resume state: load existing aggregate rows and track done sources.
    # ------------------------------------------------------------------
    aggregate_rows: List[Dict[str, str]] = []
    aggregate_csv = out_dir / "aggregate_teacher_rows.csv"
    if aggregate_csv.exists():
        aggregate_rows = read_csv_rows(aggregate_csv)
        print(f"[generate] resuming — aggregate already has {len(aggregate_rows)} rows")
    done_sources = {row.get("source_instance", "") for row in aggregate_rows if row.get("source_instance")}
    aggregate_rows = [row for row in aggregate_rows if row.get("source_instance") in done_sources]

    # ------------------------------------------------------------------
    # Phase 1: solve ALNS baseline ONCE per base spec.
    # Skip bases whose scenarios are all already done.
    # ------------------------------------------------------------------
    print(f"\n[generate] Phase 1 — ALNS baseline  ({n_bases} base spec(s))")
    base_cache: Dict[str, Tuple[Optional[Any], Optional[Any]]] = {}
    for base in bases:
        base_id = base["base_dataset_id"]
        if base_id in base_cache:
            continue
        base_scenarios = [s for s in scenarios if s["base_dataset_id"] == base_id]
        pending = [s for s in base_scenarios if s["source_instance"] not in done_sources]
        if not pending:
            print(f"  [baseline] {base_id}: all {len(base_scenarios)} scenario(s) done — skip ALNS")
            base_cache[base_id] = (None, None)
            continue
        print(
            f"  [baseline] {base_id}: {len(pending)}/{len(base_scenarios)} pending — "
            f"solving ALNS baseline..."
        )
        try:
            base_data, baseline_sol = _build_base_instance(irp, args.master_csv, base, args.time_limit)
            base_cache[base_id] = (base_data, baseline_sol)
        except Exception as exc:
            print(f"  [baseline] {base_id} FAILED: {exc}")
            traceback.print_exc()
            base_cache[base_id] = (None, None)
            if not args.continue_on_failure:
                print("[generate] aborting — pass --continue-on-failure to skip failed bases.")
                return

    # ------------------------------------------------------------------
    # Phase 2: fan out scenarios in-process, reusing cached baselines.
    # ------------------------------------------------------------------
    print(f"\n[generate] Phase 2 — CG teacher collection  ({n_scenarios} scenario(s))")
    for idx, scenario in enumerate(scenarios, start=1):
        safe_id = (
            f"{scenario['base_dataset_id']}__{scenario['scenario_id']}"
        ).replace("/", "-").replace(" ", "_")
        run_dir = out_dir / f"run_{idx:03d}__{safe_id}"
        per_run_csv = run_dir / "cg_teacher_dataset.csv"
        source_instance = scenario["source_instance"]

        already_done = (
            source_instance in done_sources
            or (per_run_csv.exists() and per_run_csv.stat().st_size > 0)
        )
        if already_done:
            if source_instance not in done_sources and per_run_csv.exists():
                existing = [
                    row for row in read_csv_rows(per_run_csv)
                    if row.get("source_instance") == source_instance
                ]
                if existing:
                    aggregate_rows.extend(existing)
                    done_sources.add(source_instance)
                    write_csv_rows(aggregate_csv, aggregate_rows)
            print(f"[generate] skip idx={idx} {source_instance} (already done)")
            manifest["scenarios"].append({
                **scenario,
                "run_idx": idx,
                "run_dir": str(run_dir),
                "teacher_rows_written": sum(
                    1 for r in aggregate_rows if r.get("source_instance") == source_instance
                ),
                "runtime_seconds": 0.0,
                "ok": True,
                "resumed": True,
                "split": split_assignment[source_instance],
            })
            manifest_path = out_dir / "scenarios_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            continue

        base_data, baseline_sol = base_cache.get(scenario["base_dataset_id"], (None, None))
        if base_data is None:
            print(
                f"[generate] skip idx={idx} {source_instance}: "
                f"baseline unavailable (base failed or all done)"
            )
            manifest["scenarios"].append({
                **scenario,
                "run_idx": idx,
                "run_dir": str(run_dir),
                "teacher_rows_written": 0,
                "runtime_seconds": 0.0,
                "ok": False,
                "resumed": False,
                "error": "baseline_not_available",
                "split": split_assignment[source_instance],
            })
            manifest_path = out_dir / "scenarios_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            if not args.continue_on_failure:
                print("[generate] aborting — pass --continue-on-failure to skip failed scenarios.")
                return
            continue

        ok, rows, runtime = _run_scenario_inprocess(
            irp, base_data, baseline_sol, scenario, run_dir, args.cg_iterations,
        )

        str_rows = _rows_to_str_dicts(rows)
        if ok and str_rows:
            aggregate_rows.extend(str_rows)
            done_sources.add(source_instance)
            write_csv_rows(aggregate_csv, aggregate_rows)

        manifest["scenarios"].append({
            **scenario,
            "run_idx": idx,
            "run_dir": str(run_dir),
            "teacher_rows_written": len(rows),
            "runtime_seconds": runtime,
            "ok": ok,
            "resumed": False,
            "split": split_assignment[source_instance],
        })
        manifest_path = out_dir / "scenarios_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if not ok and not args.continue_on_failure:
            print("[generate] aborting — pass --continue-on-failure to skip failed scenarios.")
            return

    # ------------------------------------------------------------------
    # Final manifest + integrity hash
    # ------------------------------------------------------------------
    sha = hashlib.sha1()
    with open(aggregate_csv, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    manifest["aggregate_sha1"] = sha.hexdigest()
    manifest["aggregate_rows"] = len(aggregate_rows)
    manifest_path = out_dir / "scenarios_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    n_done = sum(1 for s in manifest["scenarios"] if s.get("ok"))
    print(f"\n[generate] done.  {len(aggregate_rows)} teacher rows  "
          f"({n_done}/{n_scenarios} scenarios OK)")
    print(f"[generate] ALNS baseline solves: {n_bases}  (saved ~{n_scenarios - n_bases} re-solves)")
    print(f"[generate] next step: {manifest['build_teacher_graph_command']}")


if __name__ == "__main__":
    main()
