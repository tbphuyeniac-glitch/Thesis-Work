from __future__ import annotations

"""GNN/generate_teacher_scenarios.py — teacher-data scenario enumerator.

Runs `irp_gurobi_converted.py` once per (base_dataset, scenario) combination
and aggregates the exported teacher rows so the downstream GNN pipeline has
enough diversity for a real instance-level train / valid / test split.

A "base dataset" is a filtered slice of the master retail CSV (by store /
SKU / date). A "scenario" is one shock profile applied to a base. Together
they form the unit of identity used for splitting:

    source_instance := "<base_dataset_id>__<scenario_id>"

Output layout
-------------
    Results/scenarios/
        scenarios_manifest.json       — full list of (base, scenario) runs
        aggregate_teacher_rows.csv    — concatenated teacher rows across runs
        run_<NNN>__<base>__<scenario>/
            cg_teacher_dataset.csv    — raw per-run teacher rows
            run_metadata.json

After this script finishes, call

    python GNN/build_teacher_graph_dataset.py \
        --teacher-csv Results/scenarios/aggregate_teacher_rows.csv \
        --out-dir GNN/data/irplt_teacher

to produce the final graph splits. build_teacher_graph_dataset.py's
instance-level split will then correctly produce held-out test data because
each scenario now carries a distinct source_instance.

Split semantics (decided here, enforced by build_teacher_graph_dataset.py)
-------------------------------------------------------------------------
This script writes a `base_dataset_split` field in the manifest that assigns
each *base dataset* (not each scenario) to train / valid / test. All
scenarios from the same base stay in the same split to prevent topology
leakage (same store+SKU set seen in both train and test).

If you pass --split-by scenario the assignment is per scenario instead
(looser — allow the topology across splits but different shocks).

CLI
---
--master-csv                Master retail CSV path.
--bases                     List of base-dataset specs, each "name:store_limit:sku_limit[:start:end]".
--scenarios-per-base        Number of shock scenarios per base (default 10).
--cg-iterations             CG iterations per run (default 5).
--time-limit                Per-run time limit seconds (default 300).
--shock-distributions       "normal,gamma,sku_spike" (default all three, rotated).
--out-dir                   Output directory (default Results/scenarios).
--split-by                  "base" (default) or "scenario".
--train-ratio / --valid-ratio
                            Base-dataset split ratios (default 0.60/0.20).
--master-seed               Reproducibility master seed.
--dry-run                   Print the plan and exit without running the pipeline.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


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


def parse_base_spec(spec: str) -> Dict[str, str]:
    parts = spec.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"base spec must be 'name:store_limit:sku_limit[:start_date[:end_date]]', got {spec!r}"
        )
    base: Dict[str, str] = {
        "base_dataset_id": parts[0].strip() or "base",
        "store_limit": parts[1].strip() or "10",
        "sku_limit": parts[2].strip() or "5",
        "start_date": parts[3].strip() if len(parts) >= 4 else "None",
        "end_date": parts[4].strip() if len(parts) >= 5 else "None",
    }
    return base


def build_scenarios(
    bases: List[Dict[str, str]],
    scenarios_per_base: int,
    shock_distributions: List[str],
    master_seed: int,
) -> List[Dict[str, str]]:
    """Return list of scenario dicts. Each scenario = one pipeline run."""
    rng = random.Random(master_seed)
    scenarios: List[Dict[str, str]] = []
    for base in bases:
        for scenario_idx in range(scenarios_per_base):
            shock_profile = shock_distributions[scenario_idx % len(shock_distributions)]
            seed = rng.randrange(1, 2**31 - 1)
            # Shock parameters differ per profile. These map to existing env
            # knobs consumed by the main pipeline — keeping the generator
            # contract minimal.
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
    """Return mapping scenario_key -> 'train' / 'valid' / 'test'.

    When split_by='base', all scenarios from the same base_dataset_id land
    in the same split. When split_by='scenario', each scenario is assigned
    independently.
    """
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


def run_pipeline(
    scenario: Dict[str, str],
    project_root: Path,
    run_dir: Path,
    cg_iterations: int,
    time_limit: int,
    master_csv: str,
    pipeline_script: str,
) -> Tuple[bool, Path, float]:
    """Invoke the main pipeline once for one scenario. Returns (ok, csv_path, runtime)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Each scenario gets its own isolated Results tree inside run_dir/. This
    # prevents the top-level Results/ from being clobbered by the last run
    # and lets us cleanly pull the single teacher CSV we care about afterwards.
    base_env = {
        "IRP_SOURCE_INSTANCE": scenario["source_instance"],
        "IRP_DATASET_ID": scenario["base_dataset_id"],
        "IRP_SCENARIO_ID": scenario["scenario_id"],
        "IRP_DATASET_PATH": master_csv,
        "IRP_STORE_LIMIT": str(scenario.get("store_limit", "10")),
        "IRP_SKU_LIMIT": str(scenario.get("sku_limit", "5")),
        "IRP_DEMAND_SHOCK_SEED": str(scenario["shock_seed"]),
        "IRP_CG_ITERATIONS": str(cg_iterations),
        "IRP_TIME_LIMIT": str(time_limit),
        "IRP_COLLECT_TEACHER_MODE": "1",
        "IRP_RUNTIME_GNN_MODE": "0",
        "IRP_CLEAN_RESULTS": "0",  # never delete prior runs' aggregates
        "IRP_RESULTS_DIR": str(run_dir),
        "IRP_PHASE_LABEL": "scenario",
        # Skip the Phase-2 GNN redeploy during scenario enumeration; we only
        # want teacher rows here, not a second CG pass per scenario.
        "IRP_DEPLOY_GNN_AFTER_TRAINING": "0",
        "IRP_TRAIN_GNN_AFTER_TEACHER": "0",
        "IRP_BUILD_TEACHER_GRAPHS": "0",
    }
    if scenario.get("start_date") not in (None, "None"):
        base_env["IRP_START_DATE"] = scenario["start_date"]
    if scenario.get("end_date") not in (None, "None"):
        base_env["IRP_END_DATE"] = scenario["end_date"]
    env.update(base_env)
    env.update({
        k: v for k, v in scenario.items()
        if k.startswith("IRP_DEMAND_SHOCK_")
    })
    t0 = time.perf_counter()
    print(f"\n[generate] running scenario source_instance={scenario['source_instance']}")
    try:
        subprocess.run(
            [sys.executable, pipeline_script],
            cwd=str(project_root), env=env, check=True,
        )
    except subprocess.CalledProcessError as exc:
        runtime = time.perf_counter() - t0
        print(f"[generate] scenario {scenario['source_instance']} FAILED after {runtime:.1f}s: {exc}")
        return False, run_dir / "cg_teacher_dataset.csv", runtime
    runtime = time.perf_counter() - t0
    # New layout: each run writes to run_dir/teacher/teacher_rows.csv (set via
    # IRP_RESULTS_DIR above). Copy the teacher rows to run_dir root under the
    # canonical per-run name so aggregation doesn't have to know the layout.
    candidates = [
        run_dir / "teacher" / "teacher_rows.csv",
        run_dir / "cg_teacher_dataset.csv",  # legacy — pre-refactor runs
    ]
    dst_csv = run_dir / "cg_teacher_dataset.csv"
    src_csv = next((p for p in candidates if p.exists()), None)
    if src_csv and src_csv != dst_csv:
        shutil.copy2(src_csv, dst_csv)
    return True, dst_csv, runtime


def main() -> None:
    default_bases = [
        "base_a:10:5:None:None",
        "base_b:15:4:None:None",
        "base_c:8:6:None:None",
    ]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--master-csv",
        default=str(Path(__file__).resolve().parents[1] / "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"),
        help="Master retail CSV path.",
    )
    parser.add_argument("--bases", nargs="+", default=default_bases,
                        help="Base specs 'name:store_limit:sku_limit[:start:end]'")
    parser.add_argument("--scenarios-per-base", type=int, default=10)
    parser.add_argument("--cg-iterations", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--shock-distributions", default="normal_global,gamma_store,sku_spike")
    parser.add_argument("--out-dir", default="Results/scenarios")
    parser.add_argument("--split-by", choices=["base", "scenario"], default="base")
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--valid-ratio", type=float, default=0.20)
    parser.add_argument("--master-seed", type=int, default=20260423)
    parser.add_argument("--pipeline-script", default="irp_gurobi_converted.py")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="If set, skip failing scenarios instead of aborting.")
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

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "master_csv": args.master_csv,
        "master_seed": args.master_seed,
        "split_by": args.split_by,
        "split_ratios": {"train": args.train_ratio, "valid": args.valid_ratio,
                          "test": round(1.0 - args.train_ratio - args.valid_ratio, 4)},
        "bases": bases,
        "scenarios": [],
        "split_assignment": split_assignment,
        "run_directory": str(out_dir),
        "aggregate_csv": str(out_dir / "aggregate_teacher_rows.csv"),
        "build_teacher_graph_command": (
            f"python GNN/build_teacher_graph_dataset.py "
            f"--teacher-csv {out_dir / 'aggregate_teacher_rows.csv'} "
            f"--out-dir GNN/data/irplt_teacher"
        ),
    }
    print(f"[generate] bases={len(bases)} scenarios={len(scenarios)} split_by={args.split_by}")
    for scen in scenarios:
        print(f"  {scen['source_instance']:<60s} → {split_assignment[scen['source_instance']]}")

    if args.dry_run:
        manifest_path = out_dir / "scenarios_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[generate] dry-run wrote plan to {manifest_path}")
        return

    aggregate_rows: List[Dict[str, str]] = []
    aggregate_csv = out_dir / "aggregate_teacher_rows.csv"
    if aggregate_csv.exists():
        aggregate_rows = read_csv_rows(aggregate_csv)
        print(f"[generate] resuming — aggregate already has {len(aggregate_rows)} rows")
    # Dedup key: every row carries source_instance (written by the teacher
    # export). A scenario is "already in aggregate" if its source_instance
    # appears there, so a rerun can skip it entirely without re-appending.
    done_sources = {row.get("source_instance", "") for row in aggregate_rows if row.get("source_instance")}
    # Drop any legacy aggregate rows missing source_instance — we cannot
    # safely resume around them, so strip them to keep dedup clean.
    aggregate_rows = [row for row in aggregate_rows if row.get("source_instance") in done_sources]

    for idx, scenario in enumerate(scenarios, start=1):
        # Include base + scenario id in the run-dir name so a quick `ls`
        # of Results/scenarios/ tells you which scenario is which without
        # having to crack open the manifest.
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
            print(f"[generate] skip idx={idx} source_instance={source_instance} (already done)")
            manifest["scenarios"].append({
                **scenario,
                "run_idx": idx,
                "run_dir": str(run_dir),
                "teacher_rows_written": sum(1 for row in aggregate_rows if row.get("source_instance") == source_instance),
                "runtime_seconds": 0.0,
                "ok": True,
                "resumed": True,
                "split": split_assignment[source_instance],
            })
            manifest_path = out_dir / "scenarios_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            continue

        ok, csv_path, runtime = run_pipeline(
            scenario=scenario,
            project_root=project_root,
            run_dir=run_dir,
            cg_iterations=args.cg_iterations,
            time_limit=args.time_limit,
            master_csv=args.master_csv,
            pipeline_script=args.pipeline_script,
        )
        rows = read_csv_rows(csv_path) if ok else []
        # Guard against any stray rows carrying a different source_instance
        # (shouldn't happen, but keeps dedup invariants honest).
        rows = [row for row in rows if row.get("source_instance", source_instance) == source_instance]
        if ok and rows:
            aggregate_rows.extend(rows)
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

    # Quick sanity hash of the aggregate so we can detect silent corruption.
    sha = hashlib.sha1()
    with open(aggregate_csv, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    manifest["aggregate_sha1"] = sha.hexdigest()
    manifest["aggregate_rows"] = len(aggregate_rows)
    manifest_path = out_dir / "scenarios_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[generate] done. {len(aggregate_rows)} teacher rows aggregated.")
    print(f"[generate] next step: {manifest['build_teacher_graph_command']}")


if __name__ == "__main__":
    main()
