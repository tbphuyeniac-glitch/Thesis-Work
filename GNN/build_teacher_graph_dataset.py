from __future__ import annotations

"""Build teacher-supervised BiGAT graph samples from CG teacher CSV rows."""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import utilities


GroupKey = Tuple[str, str, str, str, str, str]


def group_key(row: Dict[str, Any]) -> GroupKey:
    source_instance = str(row.get("source_instance") or row.get("instance_id") or "default")
    branch_node = str(row.get("branch_node_id") or row.get("node_id") or "root")
    episode = str(row.get("episode") or row.get("episode_id") or "0")
    product = str(row.get("product") or row.get("sku") or "unknown_product")
    period = str(row.get("period") or row.get("time_period") or "unknown_period")
    constraint_json = str(row.get("constraint_features_json") or "")
    constraint_state = hashlib.sha1(constraint_json.encode("utf-8")).hexdigest()[:12] if constraint_json else "no_constraints"
    return source_instance, branch_node, episode, product, period, constraint_state


def read_teacher_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_groups(
    keys: List[GroupKey],
    train_ratio: float,
    valid_ratio: float,
    rng: random.Random,
) -> Dict[str, List[GroupKey]]:
    shuffled = list(keys)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    if n and n_train == 0:
        n_train = 1
    if n >= 2 and n_valid == 0:
        n_valid = 1
    if n - n_train - n_valid < 0:
        n_valid = max(0, n - n_train)
    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test": shuffled[n_train + n_valid:],
    }


def write_samples(
    grouped_rows: Dict[GroupKey, List[Dict[str, Any]]],
    split_keys: Iterable[GroupKey],
    split_dir: Path,
) -> Tuple[int, List[str], Dict[str, Any]]:
    split_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped: List[str] = []
    n_columns: List[int] = []
    n_positive: List[int] = []
    adaptive_k_values: List[float] = []
    for key in split_keys:
        source_instance, branch_node, episode, product, period, constraint_state = key
        try:
            sample = utilities.build_training_sample_from_exported_teacher_rows(
                grouped_rows[key],
                episode_id=episode,
                source_instance=source_instance,
                product=product,
                period=period,
                branch_node_id=branch_node,
                decision_state_id=constraint_state,
            )
        except Exception as exc:
            skipped.append(
                f"{source_instance}/branch={branch_node}/episode={episode}/"
                f"product={product}/period={period}/state={constraint_state}: {exc}"
            )
            continue
        written += 1
        n_columns.append(int(sample["column_features"].shape[0]))
        n_positive.append(int((sample["labels_binary"] > 0.5).sum()))
        adaptive_values = [
            utilities._float_value(row.get("adaptive_k_star"), float("nan"))
            for row in grouped_rows[key]
            if str(row.get("adaptive_k_star", "")).strip() not in {"", "None", "nan"}
        ]
        if adaptive_values:
            adaptive_k_values.append(float(adaptive_values[0]))
        utilities.save_graph_sample(sample, split_dir / f"sample_{written:05d}.pkl")
    diagnostics = {
        "samples_written": written,
        "total_columns": int(sum(n_columns)),
        "total_positive_columns": int(sum(n_positive)),
        "column_count_min": int(min(n_columns)) if n_columns else 0,
        "column_count_mean": float(sum(n_columns) / len(n_columns)) if n_columns else 0.0,
        "column_count_max": int(max(n_columns)) if n_columns else 0,
        "positive_count_min": int(min(n_positive)) if n_positive else 0,
        "positive_count_mean": float(sum(n_positive) / len(n_positive)) if n_positive else 0.0,
        "positive_count_max": int(max(n_positive)) if n_positive else 0,
        "adaptive_k_star_min": float(min(adaptive_k_values)) if adaptive_k_values else None,
        "adaptive_k_star_mean": float(sum(adaptive_k_values) / len(adaptive_k_values)) if adaptive_k_values else None,
        "adaptive_k_star_max": float(max(adaptive_k_values)) if adaptive_k_values else None,
    }
    return written, skipped, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CG teacher CSV rows into train/valid/test graph samples for teacher-supervised BiGAT training."
    )
    parser.add_argument("--teacher-csv", default="Results/cg_teacher_dataset.csv")
    parser.add_argument("--out-dir", default="GNN/data/irplt_teacher")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=utilities.valid_seed, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    teacher_csv = Path(args.teacher_csv)
    if not teacher_csv.exists():
        raise FileNotFoundError(f"Teacher CSV not found: {teacher_csv}")

    rows = read_teacher_rows(teacher_csv)
    if not rows:
        raise RuntimeError(
            f"No teacher rows found in {teacher_csv}. Run the CG pipeline first so it exports rich teacher rows."
        )

    grouped_rows: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[group_key(row)].append(row)

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = split_groups(
        list(grouped_rows.keys()),
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        rng=random.Random(args.seed),
    )

    summary: Dict[str, Any] = {
        "teacher_csv": str(teacher_csv),
        "out_dir": str(out_dir),
        "n_raw_rows": len(rows),
        "n_groups": len(grouped_rows),
        "group_key_fields": ["source_instance", "branch_node_id", "episode", "product", "period", "constraint_state_hash"],
        "splits": {},
        "skipped_groups": [],
    }
    for split, keys in split_map.items():
        written, skipped, diagnostics = write_samples(grouped_rows, keys, out_dir / split)
        summary["splits"][split] = {"groups": len(keys), **diagnostics}
        summary["skipped_groups"].extend(skipped)

    total_written = sum(item["samples_written"] for item in summary["splits"].values())
    if total_written == 0:
        raise RuntimeError(
            "No teacher graph samples were written. The CSV likely came from an older export without graph JSON fields."
        )

    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote teacher-supervised graph dataset to {out_dir}")


if __name__ == "__main__":
    main()
