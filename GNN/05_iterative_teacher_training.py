from __future__ import annotations

"""Iterative teacher-supervised training loop for the IRP-LT BiGAT scorer.

Each cycle runs the CG pipeline to export teacher rows, appends those rows to an
aggregate teacher CSV, rebuilds graph samples, and fine-tunes the GNN checkpoint.
This is intentionally offline/iterative rather than online backprop inside the
Gurobi loop, which keeps the experiment easier to audit and reproduce.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(cmd: List[str], cwd: Path, env: Dict[str, str] | None = None) -> None:
    print("\n[iterative] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated collect -> build teacher graphs -> fine-tune cycles for the IRP-LT GNN."
        )
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--epochs-per-cycle", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--objective", choices=["binary", "pairwise_rank", "score_regression"], default="pairwise_rank")
    parser.add_argument("--data-dir", default="GNN/data/irplt_teacher_iterative")
    parser.add_argument("--out-dir", default="GNN/trained_models/irplt_teacher/bigat/pairwise_rank")
    parser.add_argument("--work-dir", default="GNN/results/iterative_teacher_training")
    parser.add_argument("--teacher-csv", default="Results/cg_teacher_dataset.csv")
    parser.add_argument("--pipeline-script", default="irp_gurobi_converted.py")
    parser.add_argument("--cg-iterations", type=int, default=5)
    parser.add_argument("--pipeline-time-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    work_dir = project_root / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    aggregate_csv = work_dir / "cg_teacher_dataset_all.csv"
    manifest_path = work_dir / "manifest.json"

    all_rows: List[Dict[str, str]] = read_csv_rows(aggregate_csv)
    manifest = {
        "cycles_requested": args.cycles,
        "epochs_per_cycle": args.epochs_per_cycle,
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "cg_iterations": args.cg_iterations,
        "pipeline_time_limit": args.pipeline_time_limit,
        "aggregate_csv": str(aggregate_csv),
        "cycles": [],
    }

    checkpoint = project_root / args.out_dir / "best_model.pt"
    for cycle in range(1, args.cycles + 1):
        cycle_dir = work_dir / f"cycle_{cycle:04d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        pipeline_env = os.environ.copy()
        pipeline_env["IRP_CG_ITERATIONS"] = str(args.cg_iterations)
        pipeline_env["IRP_TIME_LIMIT"] = str(args.pipeline_time_limit)
        pipeline_env["IRP_COLLECT_TEACHER_MODE"] = "1"
        pipeline_env["IRP_RUNTIME_GNN_MODE"] = "0"
        run_command([sys.executable, args.pipeline_script], cwd=project_root, env=pipeline_env)

        teacher_csv = project_root / args.teacher_csv
        cycle_csv = cycle_dir / "cg_teacher_dataset.csv"
        if teacher_csv.exists():
            shutil.copy2(teacher_csv, cycle_csv)

        rows = read_csv_rows(cycle_csv)
        for row in rows:
            source_instance = row.get("source_instance") or "irplt_cg"
            row["source_instance"] = f"cycle_{cycle:04d}__{source_instance}"
            row["training_cycle"] = str(cycle)
        if rows:
            all_rows.extend(rows)
            write_csv_rows(aggregate_csv, all_rows)
        else:
            print(f"[iterative] cycle={cycle} produced no teacher rows; skipping dataset/training update")
            manifest["cycles"].append({
                "cycle": cycle,
                "teacher_rows": 0,
                "trained": False,
                "reason": "no_teacher_rows",
            })
            continue

        run_command(
            [
                sys.executable,
                "GNN/build_teacher_graph_dataset.py",
                "--teacher-csv",
                str(aggregate_csv),
                "--out-dir",
                args.data_dir,
                "--overwrite",
            ],
            cwd=project_root,
        )

        train_cmd = [
            sys.executable,
            "GNN/03_train_bigat.py",
            "--data-dir",
            args.data_dir,
            "--dataset-type",
            "teacher",
            "--objective",
            args.objective,
            "--epochs",
            str(args.epochs_per_cycle),
            "--out-dir",
            args.out_dir,
            "--device",
            args.device,
            "--seed",
            str(args.seed + cycle),
        ]
        if checkpoint.exists():
            train_cmd.extend(["--resume-checkpoint", str(checkpoint)])
        run_command(train_cmd, cwd=project_root)

        manifest["cycles"].append({
            "cycle": cycle,
            "teacher_rows": len(rows),
            "aggregate_teacher_rows": len(all_rows),
            "trained": True,
            "checkpoint": str(checkpoint),
        })
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\n[iterative] done")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
