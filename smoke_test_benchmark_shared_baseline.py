"""Smoke test for shared-baseline A0/A/B/C benchmark behavior.

This test monkeypatches the expensive solver pieces so it can run without
Gurobi. It verifies the benchmark orchestration, not optimization quality.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import irp_gurobi_converted as irp  # noqa: E402


class FakeBaselineSolution:
    def __init__(self) -> None:
        self.objective = 123.0
        self.direct_ship_q = {("S1", "P1", 1): 10.0}
        self.inv_store = {("S1", "P1", 1): 5.0}
        self.inv_wh = {("P1", 1): 20.0}
        self.shortage = {("S1", "P1", 1): 0.0}
        self.x = {}
        self.u = {}
        self.z = {}
        self.q = {}
        self.y = {}
        self.efficiency_metrics = {"alns_runtime_seconds": 7.5}
        self.alns_history = []
        self.realized_inventory_after_lt = None

    def summary(self) -> dict:
        return {"objective": self.objective}


class FakeBaselineALNSModel:
    solve_calls = 0

    def __init__(self, data) -> None:
        self.data = data

    def solve(self, **kwargs) -> FakeBaselineSolution:
        FakeBaselineALNSModel.solve_calls += 1
        return FakeBaselineSolution()


class FakeCGSolution:
    objective = 42.0
    lambda_values = {"c1": 1.0}
    selected_patterns = ["c1"]
    efficiency_metrics = {"gurobi_runtime_seconds": 0.25, "gnn_total_runtime": 0.0}
    stackelberg_validation = {}
    follower_solution = None
    stackelberg_column_scores = []


def main() -> int:
    old_baseline_model = irp.BaselineALNSModel
    old_apply_shock = irp.apply_hidden_local_reallocation_demand_shocks
    old_run_lt = irp.IRPResearchPipeline.run_lt_recourse_from_baseline
    old_fixed = os.environ.get("IRP_BENCHMARK_FIXED_SHOCK")

    shock_calls = []
    variant_state_snapshots = []

    def fake_apply_shock(data, *, baseline_solution, seed, **kwargs):
        shock_calls.append(seed)
        data.realized_demand = {("S1", "P1", 1): float(seed)}
        data.post_shock_inventory = {("S1", "P1", 1): 5.0}
        data.post_shock_shortage = {("S1", "P1", 1): 0.0}
        return {"shock_seed": seed}

    def fake_run_lt(self, baseline_sol, *, shock_summary=None, **kwargs):
        variant_state_snapshots.append(dict(getattr(self.data, "realized_demand", {})))
        baseline_sol.realized_inventory_after_lt = {("S1", "P1", 1): 6.0}
        return {
            "baseline_solution": baseline_sol,
            "cg_solution": FakeCGSolution(),
            "realized_no_lt_cost_breakdown": {"total_realized_operating_cost": 100.0},
            "realized_with_lt_cost_breakdown": {
                "total_realized_operating_cost": 90.0,
                "lateral_transshipment_cost_realized": 2.0,
                "shortage_cost_realized": 1.0,
            },
            "cg_episode_history": [{"added_columns": 0}],
            "cg_episode_diagnostics": [{
                "patterns_built_before_gnn": 3,
                "patterns_kept_after_gnn": 2,
                "patterns_added_to_pool": 1,
            }],
            "gnn_scoring_failures": 0,
        }

    try:
        irp.BaselineALNSModel = FakeBaselineALNSModel
        irp.apply_hidden_local_reallocation_demand_shocks = fake_apply_shock
        irp.IRPResearchPipeline.run_lt_recourse_from_baseline = fake_run_lt
        os.environ["IRP_BENCHMARK_FIXED_SHOCK"] = "1"

        data = SimpleNamespace(dataset_id="benchmark_shared_baseline_smoke")
        with tempfile.TemporaryDirectory(prefix="benchmark_shared_baseline_") as tmp:
            df = irp.run_three_way_benchmark(
                data,
                cg_iterations=1,
                time_limit=None,
                bp_max_nodes=1,
                bp_max_depth=1,
                gnn_checkpoint_path="unused.pt",
                demand_shock_seed=42,
                demand_shock_probability=0.85,
                demand_shock_reallocation_fraction=0.60,
                demand_shock_reallocations_per_product_period=1,
                demand_shock_non_dispatch_multiplier=1.8,
                lt_activation_threshold=0.0,
                heuristic_top_k=2,
                enforce_integer_flows=False,
                n_repeats=3,
                results_dir=Path(tmp),
            )

        if FakeBaselineALNSModel.solve_calls != 1:
            raise AssertionError(f"baseline solve calls = {FakeBaselineALNSModel.solve_calls}, expected 1")
        if shock_calls != [42]:
            raise AssertionError(f"fixed-shock calls = {shock_calls}, expected [42]")
        if len(variant_state_snapshots) != 12:
            raise AssertionError(f"variant runs = {len(variant_state_snapshots)}, expected 12")
        if len({repr(s) for s in variant_state_snapshots}) != 1:
            raise AssertionError("variants/repeats did not receive identical fixed-shock state")

        required_cols = {
            "variant_runtime_seconds",
            "shared_baseline_runtime_seconds",
            "total_runtime_with_shared_baseline_seconds",
            "total_runtime_seconds",
        }
        missing = required_cols.difference(df.columns)
        if missing:
            raise AssertionError(f"missing runtime columns: {sorted(missing)}")
        delta = (
            df["total_runtime_seconds"].astype(float)
            - df["total_runtime_with_shared_baseline_seconds"].astype(float)
        ).abs().max()
        if float(delta) > 1e-9:
            raise AssertionError("total_runtime_seconds is not shared-baseline + variant runtime")

        print({
            "status": "ok",
            "baseline_solve_calls": FakeBaselineALNSModel.solve_calls,
            "shock_calls": shock_calls,
            "rows": int(len(df)),
        })
        return 0
    finally:
        irp.BaselineALNSModel = old_baseline_model
        irp.apply_hidden_local_reallocation_demand_shocks = old_apply_shock
        irp.IRPResearchPipeline.run_lt_recourse_from_baseline = old_run_lt
        if old_fixed is None:
            os.environ.pop("IRP_BENCHMARK_FIXED_SHOCK", None)
        else:
            os.environ["IRP_BENCHMARK_FIXED_SHOCK"] = old_fixed


if __name__ == "__main__":
    raise SystemExit(main())
