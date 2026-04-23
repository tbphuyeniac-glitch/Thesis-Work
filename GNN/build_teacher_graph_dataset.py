from __future__ import annotations

"""
GNN/build_teacher_graph_dataset.py
==================================
Convert CG teacher CSV rows into train/valid/test graph samples.

Inputs
------
- Results/cg_teacher_dataset.csv (or .pkl.gz for Kaggle) produced during
  CG episodes where collect_teacher_mode=True.

Outputs
-------
- GNN/data/irplt_teacher/{train,valid,test}/sample_*.pkl
- GNN/data/irplt_teacher/dataset_summary.json

Split semantics
---------------
When more than one distinct `source_instance` is present, the split is
performed at the instance level (default --split-by-instance) so that no
single instance contributes groups to more than one split. With only one
instance, the code falls back to group-level random split and prints a
warning that the test split is NOT a strict generalization check.

CLI
---
--teacher-csv, --out-dir, --train-ratio, --valid-ratio, --seed,
--overwrite/--no-overwrite, --split-by-instance/--no-split-by-instance
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
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
    # Prefer the precomputed `constraint_state_hash` column written by recent
    # teacher exports — those leave `constraint_features_json` blank on all
    # rows except the first in each batch to save CSV size. Fall back to
    # hashing the JSON for legacy CSVs that still carry it on every row.
    stored_hash = str(row.get("constraint_state_hash") or "").strip()
    if stored_hash:
        constraint_state = stored_hash
    else:
        constraint_json = str(row.get("constraint_features_json") or "")
        constraint_state = (
            hashlib.sha1(constraint_json.encode("utf-8")).hexdigest()[:12]
            if constraint_json
            else "no_constraints"
        )
    return source_instance, branch_node, episode, product, period, constraint_state


def propagate_constraint_features_json(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Within each (group_key) bucket, fill the empty `constraint_features_json`
    entries with the first non-empty value in the group. The teacher exporter
    writes the full JSON only on the first row of each batch to keep the CSV
    small; consumers that previously read the JSON per-row still get a full
    value after this pass."""
    first_by_group: Dict[GroupKey, str] = {}
    for row in rows:
        key = group_key(row)
        value = str(row.get("constraint_features_json") or "").strip()
        if value and key not in first_by_group:
            first_by_group[key] = value
    for row in rows:
        if not str(row.get("constraint_features_json") or "").strip():
            row["constraint_features_json"] = first_by_group.get(group_key(row), "")
    return rows


def _raise_csv_field_size_limit() -> int:
    """Allow large JSON payloads in legacy teacher CSV fields."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit = int(limit / 10)


def _dataframe_to_records(df: Any) -> List[Dict[str, Any]]:
    import pandas as pd

    df = df.where(pd.notna(df), "")
    return df.to_dict(orient="records")


def read_teacher_rows(path: Path) -> List[Dict[str, Any]]:
    name = path.name.lower()
    if name.endswith((".pkl", ".pickle", ".pkl.gz", ".pickle.gz")):
        import pandas as pd

        return _dataframe_to_records(pd.read_pickle(path))
    if name.endswith(".parquet"):
        import pandas as pd

        return _dataframe_to_records(pd.read_parquet(path))
    if name.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if name.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return payload["rows"]
        raise ValueError(f"Unsupported teacher JSON shape in {path}; expected a row list or {{'rows': [...]}}")

    _raise_csv_field_size_limit()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_groups(
    keys: List[GroupKey],
    train_ratio: float,
    valid_ratio: float,
    rng: random.Random,
    split_by_instance: bool = True,
) -> Dict[str, List[GroupKey]]:
    """Split group keys into train/valid/test.

    When `split_by_instance=True` (default) and more than one distinct
    `source_instance` is present across keys, the split is performed at the
    instance level so that no single source_instance contributes groups to
    more than one split — eliminating cross-split data leakage. When only
    one instance exists, we fall back to group-level random split and warn
    the caller that the test split is NOT a strict generalization check.
    """
    if not keys:
        return {"train": [], "valid": [], "test": []}

    instances = sorted({key[0] for key in keys})
    if split_by_instance and len(instances) > 1:
        shuffled_instances = list(instances)
        rng.shuffle(shuffled_instances)
        n_inst = len(shuffled_instances)
        n_train_i = max(1, int(round(n_inst * train_ratio)))
        n_valid_i = max(1, int(round(n_inst * valid_ratio))) if n_inst >= 3 else 0
        if n_train_i + n_valid_i >= n_inst:
            n_valid_i = max(0, n_inst - n_train_i - 1)
        train_i = set(shuffled_instances[:n_train_i])
        valid_i = set(shuffled_instances[n_train_i:n_train_i + n_valid_i])
        test_i = set(shuffled_instances[n_train_i + n_valid_i:])
        buckets: Dict[str, List[GroupKey]] = {"train": [], "valid": [], "test": []}
        for key in keys:
            inst = key[0]
            if inst in train_i:
                buckets["train"].append(key)
            elif inst in valid_i:
                buckets["valid"].append(key)
            else:
                buckets["test"].append(key)
        return buckets

    # Fallback: only one instance → group-level random split (leakage warning).
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
    parser.add_argument(
        "--teacher-csv",
        default="Results/cg_teacher_dataset.csv",
        help="Teacher rows file. CSV is supported for legacy runs; .pkl.gz is preferred for large Kaggle exports.",
    )
    parser.add_argument("--out-dir", default="GNN/data/irplt_teacher")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=utilities.valid_seed, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--split-by-instance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When multiple source_instance values exist, split at the instance "
             "level so the test set is strictly unseen. Disable only for debugging.",
    )
    args = parser.parse_args()

    teacher_csv = Path(args.teacher_csv)
    if not teacher_csv.exists():
        raise FileNotFoundError(f"Teacher CSV not found: {teacher_csv}")

    rows = read_teacher_rows(teacher_csv)
    if not rows:
        raise RuntimeError(
            f"No teacher rows found in {teacher_csv}. Run the CG pipeline first so it exports rich teacher rows."
        )

    # Deduplicated teacher exports blank out `constraint_features_json` on all
    # rows except the first of each batch. Fill them back in before grouping
    # so `build_training_sample_from_exported_teacher_rows` has what it needs.
    rows = propagate_constraint_features_json(rows)

    grouped_rows: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[group_key(row)].append(row)

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_keys = list(grouped_rows.keys())
    distinct_instances = sorted({key[0] for key in all_keys})
    split_mode = "instance-level" if (args.split_by_instance and len(distinct_instances) > 1) else "group-level"
    if split_mode == "group-level" and args.split_by_instance:
        print(
            f"[Teacher Split] Only {len(distinct_instances)} distinct source_instance found. "
            "Falling back to group-level random split. The test split is NOT a strict "
            "generalization check — consider collecting teacher rows from more CG runs."
        )

    split_map = split_groups(
        all_keys,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        rng=random.Random(args.seed),
        split_by_instance=args.split_by_instance,
    )

    summary: Dict[str, Any] = {
        "teacher_csv": str(teacher_csv),
        "out_dir": str(out_dir),
        "n_raw_rows": len(rows),
        "n_groups": len(grouped_rows),
        "n_distinct_instances": len(distinct_instances),
        "split_mode": split_mode,
        "group_key_fields": ["source_instance", "branch_node_id", "episode", "product", "period", "constraint_state_hash"],
        "splits": {},
        "skipped_groups": [],
    }
    summary["instances_per_split"] = {
        split_name: sorted({k[0] for k in keys_list})
        for split_name, keys_list in split_map.items()
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

    # Mirror the summary into Results/graphs/ so the thesis reporting tree
    # has a single place to inspect graph-dataset stats without digging into
    # GNN/data/. Only runs when IRP_RESULTS_DIR is set (i.e. inside a
    # pipeline run), so standalone invocations stay self-contained.
    import os as _os
    _results_dir = _os.environ.get("IRP_RESULTS_DIR")
    if _results_dir:
        try:
            mirror = Path(_results_dir) / "graphs"
            mirror.mkdir(parents=True, exist_ok=True)
            with open(mirror / "graph_dataset_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Mirrored graph dataset summary to: {mirror / 'graph_dataset_summary.json'}")
        except OSError as exc:
            print(f"[graphs] mirror skipped ({exc})")

    print(json.dumps(summary, indent=2))
    print(f"Wrote teacher-supervised graph dataset to {out_dir}")


if __name__ == "__main__":
    main()
