"""
analyze_man_results.py
======================
Post-process outputs from Validate_with_Man_Kaggle.py and produce a
full analysis table with:

  Phase 1 cost  = routing_cost          (DC deliveries planned on forecast)
  Phase 2 cost  = holding_cost
                + transshipment_cost    (lateral transshipment)
                + shortage_cost         (evaluated on shocked demand)
  Total cost    = Phase 1 + Phase 2

Outputs (written to the same input_dir):
  - full_analysis_per_scenario.csv   per-row with phase1/phase2 columns
  - summary_by_method_size.csv       aggregated means ± std
  - comparison_gap_table.csv         pairwise gap (Thesis C vs each benchmark)
  - analysis_report.txt              human-readable console table

Usage:
    python analyze_man_results.py --input_dir Results/man_validation/sigma0_20_20260501_1234
    python analyze_man_results.py --input_dir Results/man_validation/sigma0_20_20260501_1234 --sigma 0.20
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np


# ── Pretty-print helpers ─────────────────────────────────────────────────────

def _fmt(v, decimals=2, pct=False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  —  "
    if pct:
        return f"{v*100:.1f}%"
    return f"{v:.{decimals}f}"


def _hline(widths: List[int], char="─") -> str:
    return "─┼─".join(char * w for w in widths)


def _header_row(cols: List[str], widths: List[int]) -> str:
    return " │ ".join(c.ljust(w) for c, w in zip(cols, widths))


def _data_row(vals: List[str], widths: List[int]) -> str:
    return " │ ".join(str(v).rjust(w) for v, w in zip(vals, widths))


# ── Phase decomposition ──────────────────────────────────────────────────────

def add_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add phase1_cost and phase2_cost columns.

    Phase 1 = routing_cost            (Stage 1: DC routing on forecast)
    Phase 2 = holding + LT + shortage (Stage 2: evaluated on shocked demand)
    """
    df = df.copy()
    routing      = pd.to_numeric(df.get("routing_cost",      pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    holding      = pd.to_numeric(df.get("holding_cost",      pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    transship    = pd.to_numeric(df.get("transshipment_cost",pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    shortage     = pd.to_numeric(df.get("shortage_cost",     pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    df["phase1_cost"] = routing
    df["phase2_cost"] = holding + transship + shortage
    # Recompute total as sanity check; keep original total_cost too
    df["total_cost_check"] = df["phase1_cost"] + df["phase2_cost"]
    return df


# ── Summary aggregation ──────────────────────────────────────────────────────

NUMERIC_COLS = [
    "runtime_seconds",
    "total_cost",
    "phase1_cost",
    "phase2_cost",
    "routing_cost",
    "holding_cost",
    "transshipment_cost",
    "shortage_cost",
    "shortage_qty",
    "service_level",
    "lt_total_qty",
    "n_lt_moves",
    "n_routes",
    "mip_gap",
]

SIZE_ORDER = ["small", "medium", "large", "ALL"]
METHOD_ORDER = ["Thesis_C_ALNS_CG_GNN", "Man-BFP-TC", "Man-Joint-TC", "Man-TSRFP-TC"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods  = [m for m in METHOD_ORDER if m in df["method"].unique()]
    methods += [m for m in df["method"].unique() if m not in METHOD_ORDER]

    for method in methods:
        for size in SIZE_ORDER:
            if size == "ALL":
                sub = df[df["method"] == method]
            else:
                sub = df[(df["method"] == method) & (df.get("size_label", pd.Series("", index=df.index)) == size)]
            if sub.empty:
                continue
            ok  = sub[sub["success"].astype(bool)]
            row: Dict = {
                "method":       method,
                "size_label":   size,
                "n_scenarios":  len(sub),
                "n_success":    len(ok),
                "success_rate": len(ok) / max(len(sub), 1),
                "n_timeout":    int((sub.get("status", pd.Series("", index=sub.index)) == "timeout").sum()),
                "n_failed":     int((sub.get("status", pd.Series("", index=sub.index)) == "failed").sum()),
            }
            for col in NUMERIC_COLS:
                if col not in ok.columns:
                    continue
                vals = pd.to_numeric(ok[col], errors="coerce").dropna()
                if len(vals) == 0:
                    row[f"mean_{col}"] = float("nan")
                    row[f"std_{col}"]  = float("nan")
                    row[f"min_{col}"]  = float("nan")
                    row[f"max_{col}"]  = float("nan")
                else:
                    row[f"mean_{col}"] = float(vals.mean())
                    row[f"std_{col}"]  = float(vals.std())
                    row[f"min_{col}"]  = float(vals.min())
                    row[f"max_{col}"]  = float(vals.max())
            rows.append(row)
    return pd.DataFrame(rows)


# ── Gap table (pairwise vs Thesis C) ─────────────────────────────────────────

def compute_gap_table(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """For each (size, benchmark_method), compute mean gap vs Thesis C."""
    thesis_key = "Thesis_C_ALNS_CG_GNN"
    rows = []
    benchmarks = [m for m in df["method"].unique() if m != thesis_key]
    for bench in benchmarks:
        for size in SIZE_ORDER:
            if size == "ALL":
                tc_sub = df[df["method"] == thesis_key]
                bm_sub = df[df["method"] == bench]
            else:
                sl = df.get("size_label", pd.Series("", index=df.index))
                tc_sub = df[(df["method"] == thesis_key) & (sl == size)]
                bm_sub = df[(df["method"] == bench) & (sl == size)]
            if tc_sub.empty or bm_sub.empty:
                continue
            # Merge on scenario_id for paired comparison
            merged = tc_sub[["scenario_id", "total_cost", "success"]].rename(
                columns={"total_cost": "c_cost", "success": "c_ok"}
            ).merge(
                bm_sub[["scenario_id", "total_cost", "success"]].rename(
                    columns={"total_cost": "b_cost", "success": "b_ok"}
                ),
                on="scenario_id", how="inner"
            )
            valid = merged[merged["c_ok"].astype(bool) & merged["b_ok"].astype(bool)]
            if valid.empty:
                continue
            gaps = (valid["c_cost"] - valid["b_cost"]) / valid["b_cost"].abs().clip(lower=1e-9) * 100.0
            rows.append({
                "benchmark":        bench,
                "size_label":       size,
                "n_paired":         len(valid),
                "mean_gap_pct":     float(gaps.mean()),    # +ve = C more expensive
                "std_gap_pct":      float(gaps.std()),
                "median_gap_pct":   float(gaps.median()),
                "n_C_wins":         int((gaps < -0.5).sum()),
                "n_tie":            int((gaps.abs() <= 0.5).sum()),
                "n_bench_wins":     int((gaps > 0.5).sum()),
                "c_mean_cost":      float(valid["c_cost"].mean()),
                "b_mean_cost":      float(valid["b_cost"].mean()),
            })
    return pd.DataFrame(rows)


# ── Console report ────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, summary: pd.DataFrame, gap_df: pd.DataFrame,
                 sigma: float, output_dir: str) -> str:
    lines = []
    SEP = "=" * 110

    lines.append(SEP)
    lines.append("  MAN ET AL. vs THESIS C vs ORACLE — FULL VALIDATION ANALYSIS")
    lines.append(f"  Demand shock σ = {sigma}  |  Output dir: {output_dir}")
    lines.append(SEP)

    # ── Table 1: Summary by method × size ──────────────────────────────────
    lines.append("")
    lines.append("TABLE 1: SUMMARY BY METHOD × STORE SIZE")
    lines.append("  Phase 1 = routing cost (forecast-based DC routing)")
    lines.append("  Phase 2 = holding + lateral transshipment + shortage (post-shock evaluation)")
    lines.append("")

    cols = ["Method", "Size", "N", "OK", "Runtime(s)", "Phase1\n(Routing)", "Phase2\n(H+LT+Short)",
            "TotalCost", "Routing", "Holding", "LT(Trans)", "Shortage", "SvcLvl%", "LT_Qty", "MIP_Gap"]
    col_keys = [
        ("method", False), ("size_label", False), ("n_scenarios", False), ("n_success", False),
        ("mean_runtime_seconds", False), ("mean_phase1_cost", False), ("mean_phase2_cost", False),
        ("mean_total_cost", False), ("mean_routing_cost", False), ("mean_holding_cost", False),
        ("mean_transshipment_cost", False), ("mean_shortage_cost", False),
        ("mean_service_level", True), ("mean_lt_total_qty", False), ("mean_mip_gap", False),
    ]
    widths = [26, 7, 3, 3, 10, 12, 14, 10, 9, 9, 10, 9, 8, 8, 8]

    header_row = " │ ".join(c.split("\n")[0].ljust(w) for c, w in zip(cols, widths))
    lines.append("  " + header_row)
    lines.append("  " + "─┼─".join("─" * w for w in widths))

    prev_method = None
    for _, row in summary.iterrows():
        if row["method"] != prev_method and prev_method is not None:
            lines.append("  " + " │ ".join(" " * w for w in widths))
        prev_method = row["method"]

        vals = []
        for (key, is_pct), w in zip(col_keys, widths):
            v = row.get(key, float("nan"))
            if key in ("method", "size_label"):
                vals.append(str(v).ljust(w)[:w])
            elif key in ("n_scenarios", "n_success"):
                vals.append(str(int(v)) if not (isinstance(v, float) and math.isnan(v)) else "—")
            elif is_pct:
                vals.append(_fmt(v, 1, pct=True))
            elif key == "mean_mip_gap":
                vals.append(_fmt(v, 4))
            else:
                vals.append(_fmt(v, 1))
        lines.append("  " + " │ ".join(str(v).rjust(w) for v, w in zip(vals, widths)))

    lines.append("")

    # ── Table 2: Per-scenario phase breakdown (sampled: first 15 rows) ──────
    lines.append("TABLE 2: PER-SCENARIO PHASE 1 / PHASE 2 BREAKDOWN  (first 30 rows)")
    lines.append("")
    cols2   = ["Scenario", "Method", "Size", "Status", "Runtime", "Phase1", "Phase2",
               "Total", "Routing", "Holding", "LT", "Shortage", "SvcLvl%"]
    widths2 = [26, 26, 7, 9, 8, 10, 10, 10, 9, 9, 9, 9, 8]
    header2 = " │ ".join(c.ljust(w) for c, w in zip(cols2, widths2))
    lines.append("  " + header2)
    lines.append("  " + "─┼─".join("─" * w for w in widths2))

    for _, row in df.head(30).iterrows():
        vals2 = [
            str(row.get("scenario_id", ""))[:26].ljust(26),
            str(row.get("method", ""))[:26].ljust(26),
            str(row.get("size_label", ""))[:7].ljust(7),
            str(row.get("status", ""))[:9].ljust(9),
            _fmt(row.get("runtime_seconds")),
            _fmt(row.get("phase1_cost")),
            _fmt(row.get("phase2_cost")),
            _fmt(row.get("total_cost")),
            _fmt(row.get("routing_cost")),
            _fmt(row.get("holding_cost")),
            _fmt(row.get("transshipment_cost")),
            _fmt(row.get("shortage_cost")),
            _fmt(row.get("service_level"), 3, pct=True),
        ]
        lines.append("  " + " │ ".join(str(v).rjust(w) for v, w in zip(vals2, widths2)))

    lines.append("")

    # ── Table 3: Gap table (Thesis C vs benchmarks) ──────────────────────────
    if not gap_df.empty:
        lines.append("TABLE 3: PAIRWISE GAP — Thesis C vs Benchmarks")
        lines.append("  cost_gap_pct = (C_cost − bench_cost) / |bench_cost| × 100")
        lines.append("  +ve = Thesis C is more expensive  |  −ve = Thesis C is cheaper")
        lines.append("")
        cols3   = ["Benchmark", "Size", "N", "MeanGap%", "StdGap%", "MedianGap%",
                   "C_Wins", "Ties", "Bench_Wins", "C_AvgCost", "Bench_AvgCost"]
        widths3 = [22, 7, 4, 10, 10, 12, 7, 6, 11, 12, 14]
        header3 = " │ ".join(c.ljust(w) for c, w in zip(cols3, widths3))
        lines.append("  " + header3)
        lines.append("  " + "─┼─".join("─" * w for w in widths3))

        prev_bench = None
        for _, row in gap_df.iterrows():
            if row["benchmark"] != prev_bench and prev_bench is not None:
                lines.append("  " + " │ ".join(" " * w for w in widths3))
            prev_bench = row["benchmark"]
            vals3 = [
                str(row["benchmark"])[:22].ljust(22),
                str(row["size_label"])[:7].ljust(7),
                str(int(row["n_paired"])),
                _fmt(row.get("mean_gap_pct"), 2),
                _fmt(row.get("std_gap_pct"),  2),
                _fmt(row.get("median_gap_pct"), 2),
                str(int(row.get("n_C_wins", 0))),
                str(int(row.get("n_tie", 0))),
                str(int(row.get("n_bench_wins", 0))),
                _fmt(row.get("c_mean_cost"), 1),
                _fmt(row.get("b_mean_cost"), 1),
            ]
            lines.append("  " + " │ ".join(str(v).rjust(w) for v, w in zip(vals3, widths3)))
        lines.append("")

    lines.append(SEP)
    lines.append("  END OF REPORT")
    lines.append(SEP)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Analyze Validate_with_Man_Kaggle.py outputs")
    p.add_argument("--input_dir", required=True,
                   help="Directory containing validate_man_vs_C_*.csv files")
    p.add_argument("--sigma", type=float, default=0.20,
                   help="Demand shock sigma used in the run (for labelling only)")
    p.add_argument("--output_prefix", default="",
                   help="Optional prefix for output files (default: same as input_dir)")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        print(f"[Error] Input directory not found: {in_dir}")
        sys.exit(1)

    # ── Load per-scenario CSV ────────────────────────────────────────────────
    # Prefer per_scenario.csv — it has more columns (status, runtime, mip_gap, etc.)
    per_scenario_path = in_dir / "validate_man_vs_C_per_scenario.csv"
    breakdown_path    = in_dir / "validate_man_vs_C_cost_breakdown.csv"

    src_path = per_scenario_path if per_scenario_path.exists() else breakdown_path
    if not src_path.exists():
        print(f"[Error] Neither cost_breakdown nor per_scenario CSV found in {in_dir}")
        sys.exit(1)

    print(f"[Analyze] Loading {src_path.name} ...")
    df = pd.read_csv(src_path)

    # Ensure success column is boolean
    if "success" in df.columns:
        df["success"] = df["success"].astype(str).str.lower().isin({"true", "1", "yes"})

    # Add phase columns
    df = add_phase_columns(df)

    # ── Compute summary ──────────────────────────────────────────────────────
    summary = summarize(df)

    # ── Compute gap table ────────────────────────────────────────────────────
    gap_df = compute_gap_table(df, summary)

    # ── Print report ─────────────────────────────────────────────────────────
    report = print_report(df, summary, gap_df, args.sigma, str(in_dir))
    print(report)

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_prefix = args.output_prefix or str(in_dir)

    analysis_csv    = in_dir / "full_analysis_per_scenario.csv"
    summary_csv     = in_dir / "summary_by_method_size.csv"
    gap_csv         = in_dir / "comparison_gap_table.csv"
    report_txt      = in_dir / "analysis_report.txt"

    # Per-scenario with phase columns
    phase_cols = [
        "scenario_id", "method", "size_label", "store_limit", "sku_limit", "n_periods",
        "start_date", "end_date", "status", "success", "runtime_seconds",
        "phase1_cost", "phase2_cost", "total_cost",
        "routing_cost", "holding_cost", "transshipment_cost", "shortage_cost",
        "shortage_qty", "service_level", "lt_total_qty", "n_lt_moves", "n_routes",
        "mip_gap", "n_columns_generated", "n_columns_selected",
    ]
    out_cols = [c for c in phase_cols if c in df.columns]
    df[out_cols].to_csv(analysis_csv, index=False)
    print(f"\n[Analyze] Saved: {analysis_csv.name}")

    summary.to_csv(summary_csv, index=False)
    print(f"[Analyze] Saved: {summary_csv.name}")

    if not gap_df.empty:
        gap_df.to_csv(gap_csv, index=False)
        print(f"[Analyze] Saved: {gap_csv.name}")

    with open(report_txt, "w") as f:
        f.write(report)
    print(f"[Analyze] Saved: {report_txt.name}")

    # ── Quick sanity check ───────────────────────────────────────────────────
    print(f"\n[Analyze] Phase cost sanity check (success rows):")
    ok = df[df["success"].astype(bool)]
    if not ok.empty:
        diff = (ok["total_cost"] - ok["total_cost_check"]).abs()
        max_diff = diff.max()
        print(f"  max |total_cost − (phase1+phase2)| = {max_diff:.4f}"
              f"  {'OK' if max_diff < 1.0 else 'WARNING: large discrepancy'}")


if __name__ == "__main__":
    main()
