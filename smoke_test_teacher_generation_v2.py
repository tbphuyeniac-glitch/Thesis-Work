"""smoke_test_teacher_generation_v2.py — Small smoke test for shared-baseline
teacher scenario generation.

What it checks
--------------
1. Builds a tiny retail CSV locally in a temp directory.
2. Uses GNN/generate_teacher_scenarios.py v2 helpers directly:
   - `_build_base_instance()` once per base
   - `_run_scenario_inprocess()` once per scenario
3. Prints concrete outputs in VS Code:
   - tiny CSV path
   - scenario split assignment
   - baseline objective/runtime
   - teacher row counts per scenario
   - sample teacher-row keys / sample values

Usage
-----
python smoke_test_teacher_generation_v2.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

os.environ.setdefault("IRP_QUIET", "1")
os.environ.setdefault("IRP_LT_COST_MULTIPLIER", "1.0")

ROOT = Path(__file__).resolve().parent
SMOKE_ROOT = ROOT / "smoke_runs" / "teacher_generation_v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GNN"))

import irp_gurobi_converted as irp  # noqa: E402
import GNN.generate_teacher_scenarios as gen  # noqa: E402


def _build_tiny_csv(out_path: Path) -> Path:
    rng = random.Random(7)
    stores = ["Store_A", "Store_B", "Store_C"]
    skus = [
        ("SKU_RICE", 25.0),
        ("SKU_COOKIE", 35.0),
    ]
    periods = pd.date_range("2025-08-01", periods=8, freq="D")

    rows = []
    for store_idx, store in enumerate(stores):
        for sku_idx, (sku, price) in enumerate(skus):
            for t_idx, dt in enumerate(periods):
                sale_qty = 2 + ((store_idx + sku_idx + t_idx) % 4) + rng.randint(0, 1)
                end_qty = max(0, 4 + ((store_idx * 2 + sku_idx + t_idx) % 5) - rng.randint(0, 2))
                rows.append(
                    {
                        "SITE_NAME": store,
                        "NORMAL_PRICE": price,
                        "ART_SV_NAME_ENG": sku,
                        "SALE_QTY": sale_qty,
                        "END_QTY": end_qty,
                        "PERIOD": dt.strftime("%Y%m%d"),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path


@contextmanager
def _fake_cg_backend():
    """Fallback used only when local Gurobi WLS is unavailable.

    This keeps the smoke test focused on control flow:
      shared baseline -> per-scenario shock -> teacher-row tagging/writing.
    It does NOT validate the real RMP/CG math.
    """
    original = irp.LateralTransshipmentCG.run_column_generation

    def fake_run(self, max_iter=1, msg=False, stopping_mode="convergence"):
        product = self.data.products[0] if self.data.products else "P"
        period = self.data.periods[0] if self.data.periods else 1
        self.teacher_dataset_rows = [{
            "pattern_id": f"fake_{self.data.scenario_id}_{product}_{period}",
            "product": product,
            "period": period,
            "reduced_cost": -1.0,
            "passed_to_rmp": True,
            "constraint_state_hash": f"fake_{self.data.scenario_id}",
        }]
        return SimpleNamespace(
            objective=123.456,
            iterations_run=1,
        )

    irp.LateralTransshipmentCG.run_column_generation = fake_run
    try:
        yield
    finally:
        irp.LateralTransshipmentCG.run_column_generation = original


def main() -> int:
    print("=" * 72)
    print("SMOKE TEST: shared-baseline teacher generation v2")
    print("=" * 72)

    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    master_csv = _build_tiny_csv(SMOKE_ROOT / "tiny_teacher_master.csv")
    out_dir = SMOKE_ROOT / "scenario_runs"

    print(f"[Output] master_csv = {master_csv}")
    print(f"[Output] out_dir    = {out_dir}")

    bases = [gen.parse_base_spec("tiny_base:3:2:2025-08-01:2025-08-08")]
    scenarios = gen.build_scenarios(
        bases=bases,
        scenarios_per_base=2,
        shock_distributions=["normal_global", "gamma_store"],
        master_seed=42,
    )
    split_assignment = gen.assign_split(
        scenarios=scenarios,
        split_by="scenario",
        train_ratio=0.5,
        valid_ratio=0.0,
        seed=42,
    )

    print(f"[Scenario] built {len(scenarios)} scenario(s)")
    for scen in scenarios:
        print(
            f"  {scen['source_instance']:<50s} -> {split_assignment[scen['source_instance']]} "
            f"(shock_seed={scen['shock_seed']})"
        )

    t0 = time.perf_counter()
    base_data, baseline_sol = gen._build_base_instance(
        irp=irp,
        master_csv=str(master_csv),
        base=bases[0],
        time_limit=10,
    )
    baseline_elapsed = time.perf_counter() - t0
    print(
        f"[Baseline] objective={float(baseline_sol.objective):.4f} "
        f"stores={len(base_data.stores)} skus={len(base_data.products)} "
        f"periods={len(base_data.periods)} runtime={baseline_elapsed:.2f}s"
    )

        # The shared-baseline part is Gurobi-free, but scenario CG still needs
        # a live Gurobi environment for the RMP. In local setups without token
        # access, report that clearly instead of hard-failing the smoke test.
    try:
        env = irp.create_gurobi_env()
        env.dispose()
        gurobi_ready = True
    except Exception as exc:
        gurobi_ready = False
        print(f"[Gurobi Preflight] unavailable for CG teacher step: {exc}")

    if not gurobi_ready:
        print("[Gurobi Preflight] switching to fake CG backend for local flow smoke test")

    cg_ctx = _fake_cg_backend() if not gurobi_ready else nullcontext()
    total_rows = 0
    with cg_ctx:
        for idx, scenario in enumerate(scenarios, start=1):
            run_dir = out_dir / f"run_{idx:03d}__{scenario['source_instance']}"
            ok, rows, runtime = gen._run_scenario_inprocess(
                irp=irp,
                base_data=base_data,
                baseline_sol=baseline_sol,
                scenario=scenario,
                run_dir=run_dir,
                cg_iterations=1,
            )
            print(
                f"[Scenario Result] idx={idx} ok={ok} rows={len(rows)} runtime={runtime:.2f}s "
                f"csv_exists={(run_dir / 'cg_teacher_dataset.csv').exists()}"
            )
            total_rows += len(rows)
            if rows:
                sample = rows[0]
                keep = {
                    k: sample.get(k)
                    for k in [
                        "source_instance",
                        "pattern_id",
                        "product",
                        "period",
                        "reduced_cost",
                        "passed_to_rmp",
                        "constraint_state_hash",
                    ]
                    if k in sample
                }
                print(f"  sample_row_keys={sorted(sample.keys())[:12]} ... total_keys={len(sample)}")
                print(f"  sample_row={json.dumps(keep, ensure_ascii=False)}")

    print(f"[Summary] total_teacher_rows={total_rows}")
    assert total_rows > 0, "Expected at least one teacher row from tiny scenarios"
    print(f"[Summary] persistent outputs kept under: {SMOKE_ROOT}")

    print("=" * 72)
    print("SMOKE TEST PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"\n[SMOKE FAIL] {type(exc).__name__}: {exc}")
        raise SystemExit(1)
