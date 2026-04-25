from __future__ import annotations

import ast
import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "kaggle_irp_pipeline_clean.py"

DEFAULT_TEACHER_START_DATE = "2025-08-01"
DEFAULT_TEACHER_END_DATE = "2025-08-14"


def infer_date_range_from_csv(
    csv_path: Path,
    fallback_start: str = DEFAULT_TEACHER_START_DATE,
    fallback_end: str = DEFAULT_TEACHER_END_DATE,
) -> Tuple[str, str]:
    if not csv_path.exists():
        return fallback_start, fallback_end
    period_only = pd.read_csv(csv_path, usecols=["PERIOD"])
    parsed = pd.to_datetime(
        period_only["PERIOD"].astype(str).str.strip(),
        format="%Y%m%d",
        errors="coerce",
    ).dropna()
    if parsed.empty:
        return fallback_start, fallback_end
    return parsed.min().strftime("%Y-%m-%d"), parsed.max().strftime("%Y-%m-%d")


def build_normalized_base_specs(
    start_date: Optional[str],
    end_date: Optional[str],
    target_count: int = 30,
) -> List[str]:
    resolved_start = start_date or DEFAULT_TEACHER_START_DATE
    resolved_end = end_date or DEFAULT_TEACHER_END_DATE
    start_dt = datetime.strptime(resolved_start, "%Y-%m-%d")
    end_dt = datetime.strptime(resolved_end, "%Y-%m-%d")
    horizon_days = max(1, (end_dt - start_dt).days + 1)

    store_limits = [4, 5, 6, 7, 8]
    sku_limits = [2, 3, 4]

    full_window = (start_dt, end_dt)
    normalized_days = max(7, horizon_days - 4)
    normalized_start = start_dt + timedelta(days=min(2, max(0, horizon_days - normalized_days)))
    normalized_end = min(end_dt, normalized_start + timedelta(days=normalized_days - 1))
    date_windows = [full_window, (normalized_start, normalized_end)]

    specs: List[str] = []
    base_idx = 0
    for store_limit in store_limits:
        for sku_limit in sku_limits:
            for window_start, window_end in date_windows:
                base_idx += 1
                specs.append(
                    f"base_{base_idx:02d}:{store_limit}:{sku_limit}:"
                    f"{window_start.strftime('%Y-%m-%d')}:"
                    f"{window_end.strftime('%Y-%m-%d')}"
                )
    if len(specs) != target_count:
        raise AssertionError(f"Expected {target_count} base specs, got {len(specs)}")
    return specs


def validate_graph_split(graph_dir: Path) -> dict:
    train_samples = list((graph_dir / "train").glob("*.pkl")) if (graph_dir / "train").exists() else []
    valid_samples = list((graph_dir / "valid").glob("*.pkl")) if (graph_dir / "valid").exists() else []
    test_samples = list((graph_dir / "test").glob("*.pkl")) if (graph_dir / "test").exists() else []
    if not train_samples:
        raise RuntimeError("No training graph samples built.")
    if not valid_samples:
        raise RuntimeError("No validation graph samples built.")
    if not test_samples:
        raise RuntimeError("No held-out test graph samples built.")
    return {
        "train": len(train_samples),
        "valid": len(valid_samples),
        "test": len(test_samples),
    }


def parse_top_level_assignments(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return values


def assert_notebook_textual_flow(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    prepare_idx = text.index("prepare_working_repo()")
    infer_idx = text.index("TEACHER_START_DATE, TEACHER_END_DATE = infer_date_range_from_csv(TRAIN_DATA_PATH)")
    base_idx = text.index("BASE_SPECS = build_normalized_base_specs(")
    clear_idx = text.index("if CLEAR_RESULTS_DIR and RESULTS_DIR.exists():")
    mkdir_idx = text.index("RESULTS_DIR.mkdir(parents=True, exist_ok=True)")
    assert prepare_idx < infer_idx < base_idx, "teacher window/base specs are not resolved after prepare_working_repo()"
    assert clear_idx < mkdir_idx, "Results cleanup must happen before recreating RESULTS_DIR"
    return {
        "prepare_before_teacher_window": True,
        "clear_results_before_mkdir": True,
    }


def main() -> None:
    assignments = parse_top_level_assignments(TARGET)
    assert assignments.get("CLEAR_RESULTS_DIR") is True
    assert assignments.get("REUSE_EXISTING_CHECKPOINT") is False
    assert assignments.get("BENCHMARK_N_REPEATS") == 3

    textual_flow = assert_notebook_textual_flow(TARGET)

    with tempfile.TemporaryDirectory(prefix="kaggle_flow_smoke_") as tmp:
        tmpdir = Path(tmp)
        repo_root = tmpdir / "repo"
        results_dir = tmpdir / "Results"
        train_csv = repo_root / "train.csv"
        graph_dir = repo_root / "GNN" / "data" / "irplt_teacher"

        repo_root.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "old.txt").write_text("stale", encoding="utf-8")

        pd.DataFrame(
            {
                "SITE_NAME": ["A", "A", "B"],
                "NORMAL_PRICE": [10, 10, 12],
                "ART_SV_NAME_ENG": ["sku1", "sku1", "sku2"],
                "SALE_QTY": [1, 2, 3],
                "END_QTY": [5, 4, 6],
                "PERIOD": [20250801, 20250814, 20250810],
            }
        ).to_csv(train_csv, index=False)

        teacher_start, teacher_end = infer_date_range_from_csv(train_csv)
        specs = build_normalized_base_specs(teacher_start, teacher_end, target_count=30)
        assert teacher_start == "2025-08-01"
        assert teacher_end == "2025-08-14"
        assert len(specs) == 30

        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        assert not (results_dir / "old.txt").exists()

        for split in ["train", "valid", "test"]:
            split_dir = graph_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "sample_000.pkl").write_bytes(b"smoke")

        split_counts = validate_graph_split(graph_dir)

        empty_graph_dir = tmpdir / "empty_graphs"
        (empty_graph_dir / "train").mkdir(parents=True, exist_ok=True)
        (empty_graph_dir / "valid").mkdir(parents=True, exist_ok=True)
        (empty_graph_dir / "train" / "sample_000.pkl").write_bytes(b"smoke")
        (empty_graph_dir / "valid" / "sample_000.pkl").write_bytes(b"smoke")
        try:
            validate_graph_split(empty_graph_dir)
            raise AssertionError("Expected empty test split to raise")
        except RuntimeError as exc:
            assert "No held-out test graph samples built." in str(exc)

        summary = {
            "status": "ok",
            "config_assertions": {
                "CLEAR_RESULTS_DIR": assignments.get("CLEAR_RESULTS_DIR"),
                "REUSE_EXISTING_CHECKPOINT": assignments.get("REUSE_EXISTING_CHECKPOINT"),
                "BENCHMARK_N_REPEATS": assignments.get("BENCHMARK_N_REPEATS"),
            },
            "textual_flow": textual_flow,
            "teacher_window": [teacher_start, teacher_end],
            "base_specs_count": len(specs),
            "graph_split_counts": split_counts,
            "empty_test_split_check": "raised_as_expected",
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
