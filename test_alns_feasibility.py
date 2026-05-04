"""
test_alns_feasibility.py
========================

Demo: using ALNSFeasibilityValidator in a benchmark run.

Usage:
    python3 test_alns_feasibility.py

This script:
  1. Loads a small scenario
  2. Runs ALNS
  3. Validates the solution with ALNSFeasibilityValidator
  4. Prints detailed and aggregate reports
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

from irp_gurobi_converted import (
    DatasetToIRPValidationMapper,
    BaselineALNSModel,
)

from alns_feasibility_validator import (
    ALNSFeasibilityValidator,
    ALNSFeasibilityReport,
    aggregate_feasibility_reports,
    print_feasibility_report,
    print_benchmark_stats,
)


def run_scenario_with_validation(
    excel_path: Path,
    store_limit: int,
    sku_limit: int,
    label: str,
) -> ALNSFeasibilityReport:
    """Run ALNS on one scenario and validate feasibility."""
    print(f"\n{'='*80}")
    print(f"  Scenario: {label} ({store_limit}s × {sku_limit}p)")
    print(f"{'='*80}")

    # Build scenario
    mapper = DatasetToIRPValidationMapper(
        excel_path=str(excel_path),
        store_limit=store_limit,
        sku_limit=sku_limit,
    )
    data, _, _, _ = mapper.build_irp_data(
        cw_replenishment_factor=0.8,
        holding_cost_rate=0.01,
        shortage_cost_rate=0.25,
        wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2,
        store_initial_inventory_multiplier=0.2,
        vehicle_count=2,
        vehicle_fixed_cost=50.0,
        alpha=1.0,
        cw_capacity_factor=2.0,
    )

    # Auto-size vehicle capacity
    total_dem = max(1.0, sum(data.demand.values()))
    dispatch_pds = [t for t in data.periods if (t - min(data.periods)) % 5 == 0]
    n_dispatch = max(1, len(dispatch_pds))
    auto_cap = total_dem / n_dispatch / 2 * 0.85
    data.vehicle_capacity = max(500.0, round(auto_cap))

    print(f"  Stores={len(data.stores)}  SKUs={len(data.products)}  "
          f"Periods={len(data.periods)}  Vehicle_cap={data.vehicle_capacity:.0f}")

    # Run ALNS
    print(f"  [ALNS] running ...")
    t0 = time.perf_counter()
    alns_sol = BaselineALNSModel(data).solve(
        msg=False, max_iterations=500, seed=42,
        allow_lateral_transshipment=False,
    )
    alns_rt = time.perf_counter() - t0
    print(f"    obj={alns_sol.objective:.2f}  rt={alns_rt:.2f}s")

    # Validate feasibility
    validator = ALNSFeasibilityValidator(data)
    report = validator.validate(alns_sol.state, solution_id=label)

    return report


def main():
    """Run multi-scenario feasibility benchmark."""
    excel_path = Path("1BISCR501V_90100140_20260323-150407111_filtered_sites.csv")
    if not excel_path.exists():
        print(f"Error: {excel_path} not found")
        return 1

    # Multi-tier scenarios: increasing complexity
    tiers: List[tuple] = [
        ("tiny_3s2p", 3, 2),
        ("small_4s2p", 4, 2),
        ("small_5s3p", 5, 3),
        ("medium_6s4p", 6, 4),
    ]

    reports: List[ALNSFeasibilityReport] = []

    for label, store_limit, sku_limit in tiers:
        try:
            report = run_scenario_with_validation(
                excel_path, store_limit, sku_limit, label
            )
            reports.append(report)
            print_feasibility_report(report, verbose=False)
        except Exception as e:
            print(f"  ✗ Scenario failed: {e}")

    # Aggregate and print summary
    if reports:
        stats = aggregate_feasibility_reports(reports)
        print_benchmark_stats(stats)

        if stats.is_reliable:
            print("✓ SUCCESS: ALNS is ready for production benchmarks")
            return 0
        else:
            print("✗ FAILURE: ALNS needs constraint tuning before benchmarking")
            return 1
    else:
        print("✗ No scenarios ran successfully")
        return 1


if __name__ == "__main__":
    sys.exit(main())
