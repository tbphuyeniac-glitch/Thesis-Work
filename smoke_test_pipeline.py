"""Full pipeline smoke test.

Checks each stage of the IRP-LT + GNN pipeline without requiring a Gurobi
license or a pre-trained checkpoint.  Run with:

    python3 smoke_test_pipeline.py

Exit code 0 = all checks passed.  Each check prints PASS or FAIL with a
one-line reason so failures are easy to trace.
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

# ── Gurobipy stub (avoids license requirement) ────────────────────────────────
class _StubGRB:
    OPTIMAL = 2; INFEASIBLE = 3; UNBOUNDED = 5; INF_OR_UNBD = 4
    TIME_LIMIT = 9; INTERRUPTED = 11; SUBOPTIMAL = 13; NUMERIC = 12
    INTEGER = 1; CONTINUOUS = 0; BINARY = 2; MINIMIZE = 1

class _StubModel:
    def __init__(self, *a, **kw):
        self.Params = type("P", (), {"__setattr__": lambda s, k, v: None})()
        self.Status = 2; self.SolCount = 1; self.Runtime = 0.0
        self.IterCount = 0; self.BarIterCount = 0; self.NodeCount = 0
        self.IsMIP = 0; self.ObjVal = 0.0
    def addVars(self, *a, **kw): return {}
    def addVar(self, *a, **kw):
        v = type("V", (), {"X": 0.0, "Obj": 0.0, "VarName": "x"})()
        return v
    def setObjective(self, *a, **kw): pass
    def addConstr(self, *a, **kw): pass
    def addConstrs(self, *a, **kw): return {}
    def optimize(self): pass
    def update(self): pass
    def getVars(self): return []
    def getConstrByName(self, *a): return None
    def write(self, *a): pass

class _StubGurobipy:
    Model = _StubModel
    GRB = _StubGRB
    class Env:
        def __init__(self, empty=False): pass
        def setParam(self, *a, **kw): pass
        def start(self): pass
    @staticmethod
    def quicksum(iterable=(), *a, **kw):
        try:
            return sum(iterable)
        except Exception:
            return 0

try:
    import gurobipy  # noqa: F401 — real Gurobi available, use it
except Exception:
    sys.modules["gurobipy"] = _StubGurobipy()  # type: ignore[assignment]
    sys.modules["gurobipy"].GRB = _StubGRB()   # type: ignore[assignment]

os.environ.setdefault("WLSACCESSID", "dummy")
os.environ.setdefault("WLSSECRET",   "dummy")
os.environ.setdefault("LICENSEID",   "0")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GNN"))

# ── Helpers ───────────────────────────────────────────────────────────────────
_passed = 0
_failed = 0

def check(name: str, fn):
    global _passed, _failed
    try:
        fn()
        print(f"  PASS  {name}")
        _passed += 1
    except Exception as exc:
        print(f"  FAIL  {name}")
        print(f"        {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=4)
        _failed += 1


def section(title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Imports
# ─────────────────────────────────────────────────────────────────────────────
section("1  Imports")

def _import_main():
    import irp_gurobi_converted as m
    assert hasattr(m, "IRPData")
    assert hasattr(m, "BaselineALNSModel")
    assert hasattr(m, "LateralTransshipmentCG")
    assert hasattr(m, "DatasetToIRPValidationMapper")
    assert hasattr(m, "IRPResearchPipeline")
    assert hasattr(m, "run_teacher_graph_and_gnn_training")
    assert hasattr(m, "run_three_way_benchmark")

check("irp_gurobi_converted imports cleanly", _import_main)

def _import_gnn_utilities():
    import utilities as u
    assert hasattr(u, "build_training_sample_from_exported_teacher_rows")
    assert hasattr(u, "normalize_dataset")
    assert hasattr(u, "load_split")
    assert hasattr(u, "save_graph_sample")
    assert hasattr(u, "adaptive_select_indices")

check("GNN/utilities imports cleanly", _import_gnn_utilities)

def _import_build_dataset():
    import build_teacher_graph_dataset as b
    assert hasattr(b, "split_groups")
    assert hasattr(b, "group_key")
    assert hasattr(b, "read_teacher_rows")
    assert hasattr(b, "propagate_constraint_features_json")

check("GNN/build_teacher_graph_dataset imports cleanly", _import_build_dataset)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Dataset loading
# ─────────────────────────────────────────────────────────────────────────────
section("2  Dataset loading")

from irp_gurobi_converted import DatasetToIRPValidationMapper

_TEST_CSV   = ROOT / "test data.csv"
_TRAIN_CSV  = ROOT / "1BISCR501V_90100140_20260323-150407111_filtered_sites.csv"

def _test_csv_exists():
    assert _TEST_CSV.exists(), f"Not found: {_TEST_CSV}"

def _train_csv_exists():
    assert _TRAIN_CSV.exists(), f"Not found: {_TRAIN_CSV}"

check("test data.csv file exists", _test_csv_exists)
check("training CSV file exists",  _train_csv_exists)

def _test_csv_columns():
    mapper = DatasetToIRPValidationMapper(excel_path=str(_TEST_CSV), store_limit=2, sku_limit=2)
    df = mapper.load_raw()
    assert len(df) > 0, "No rows loaded from test data.csv"
    required = {"store", "sku", "sale_qty", "end_qty", "period_raw", "price"}
    missing = required - set(df.columns)
    assert not missing, f"Missing mapped columns: {missing}"

check("test data.csv loads and maps required columns", _test_csv_columns)

def _train_csv_columns():
    mapper = DatasetToIRPValidationMapper(excel_path=str(_TRAIN_CSV), store_limit=2, sku_limit=2)
    df = mapper.load_raw()
    assert len(df) > 0, "No rows loaded from training CSV"

check("training CSV loads required columns", _train_csv_columns)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — IRPData construction from test data.csv
# ─────────────────────────────────────────────────────────────────────────────
section("3  IRPData construction (test data.csv)")

from irp_gurobi_converted import IRPData

_irp_data_from_csv = None

def _build_irpdata_from_test_csv():
    global _irp_data_from_csv
    mapper = DatasetToIRPValidationMapper(
        excel_path=str(_TEST_CSV),
        store_limit=3,
        sku_limit=2,
        start_date="2025-02-01",
        end_date="2025-02-07",
    )
    data, base_df, validation_target, meta = mapper.build_irp_data(
        wh_inventory_multiplier=0.8,
        store_capacity_multiplier=1.2,
        shortage_cost_rate=0.05,
        holding_cost_rate=100,
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
        lt_cost_multiplier=1.0,
    )
    assert isinstance(data, IRPData), "build_irp_data must return IRPData"
    assert data.stores,   "IRPData.stores is empty"
    assert data.products, "IRPData.products is empty"
    assert data.periods,  "IRPData.periods is empty"
    assert data.demand,   "IRPData.demand is empty"
    assert data.shortage_cost, "IRPData.shortage_cost is empty"
    assert data.transship_unit_cost, "IRPData.transship_unit_cost is empty"
    assert isinstance(validation_target, __import__("pandas").DataFrame)
    assert len(meta) > 0
    _irp_data_from_csv = data

check("build_irp_data returns valid IRPData from test data.csv", _build_irpdata_from_test_csv)

def _irpdata_cost_scales():
    assert _irp_data_from_csv is not None, "IRPData not built (previous check failed)"
    data = _irp_data_from_csv
    s0, p0 = next(iter(data.stores)), next(iter(data.products))
    shortage = data.shortage_cost.get((s0, p0), 0.0)
    lt_pairs = [(i, j) for i in data.stores for j in data.stores if i != j]
    lt_avg = sum(data.transship_unit_cost.get(p, 0.0) for p in lt_pairs) / max(1, len(lt_pairs))
    assert shortage > 0, "shortage_cost should be > 0"
    assert lt_avg >= 0,  "transship_unit_cost should be >= 0"

check("cost scales are positive and non-negative", _irpdata_cost_scales)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Pipeline mode flags (IRP_ONLINE_INFERENCE logic)
# ─────────────────────────────────────────────────────────────────────────────
section("4  Pipeline mode flags")

def _mode_offline_defaults():
    # Simulate __main__ env-var reading in offline-training mode
    os.environ["IRP_ONLINE_INFERENCE"] = "0"
    online = os.environ.get("IRP_ONLINE_INFERENCE", "0").lower() not in {"0", "false", "no"}
    assert not online, "IRP_ONLINE_INFERENCE=0 should give online=False"
    _dc = "0" if online else "1"
    _rg = "1" if online else "0"
    _ta = "0" if online else "1"
    collect = os.environ.get("IRP_COLLECT_TEACHER_MODE", _dc).lower() not in {"0", "false", "no"}
    runtime_gnn = os.environ.get("IRP_RUNTIME_GNN_MODE", _rg).lower() not in {"0", "false", "no"}
    train_after = os.environ.get("IRP_TRAIN_GNN_AFTER_TEACHER", _ta).lower() not in {"0", "false", "no"}
    assert collect,      "offline mode: collect_teacher_mode should be True"
    assert not runtime_gnn, "offline mode: runtime_gnn_mode should be False"
    assert train_after,  "offline mode: train_gnn_after_teacher should be True"

check("offline training: collect=True, gnn_runtime=False, train=True", _mode_offline_defaults)

def _mode_online_defaults():
    os.environ["IRP_ONLINE_INFERENCE"] = "1"
    online = os.environ.get("IRP_ONLINE_INFERENCE", "0").lower() not in {"0", "false", "no"}
    assert online, "IRP_ONLINE_INFERENCE=1 should give online=True"
    _dc = "0" if online else "1"
    _rg = "1" if online else "0"
    _ta = "0" if online else "1"
    collect = os.environ.get("IRP_COLLECT_TEACHER_MODE", _dc).lower() not in {"0", "false", "no"}
    runtime_gnn = os.environ.get("IRP_RUNTIME_GNN_MODE", _rg).lower() not in {"0", "false", "no"}
    train_after = os.environ.get("IRP_TRAIN_GNN_AFTER_TEACHER", _ta).lower() not in {"0", "false", "no"}
    assert not collect,  "online mode: collect_teacher_mode should be False"
    assert runtime_gnn,  "online mode: runtime_gnn_mode should be True"
    assert not train_after, "online mode: train_gnn_after_teacher should be False"

check("online inference: collect=False, gnn_runtime=True, train=False", _mode_online_defaults)

def _mode_dataset_path():
    os.environ["IRP_ONLINE_INFERENCE"] = "1"
    online = os.environ.get("IRP_ONLINE_INFERENCE", "0").lower() not in {"0", "false", "no"}
    _inf  = Path(__file__).with_name("test data.csv")
    _train = Path(__file__).with_name("1BISCR501V_90100140_20260323-150407111_filtered_sites.csv")
    path = Path(os.environ.get("IRP_DATASET_PATH", str(_inf if online else _train)))
    assert path.name == "test data.csv", f"Online mode should select test data.csv, got {path.name}"

check("online mode selects test data.csv as EXCEL_PATH", _mode_dataset_path)

def _mode_online_learning_defaults():
    os.environ["IRP_ONLINE_INFERENCE"] = "1"
    os.environ["IRP_ONLINE_LEARNING"]  = "1"
    online   = os.environ.get("IRP_ONLINE_INFERENCE", "0").lower() not in {"0", "false", "no"}
    learning = online and os.environ.get("IRP_ONLINE_LEARNING", "0").lower() not in {"0", "false", "no"}
    assert learning, "IRP_ONLINE_INFERENCE=1 + IRP_ONLINE_LEARNING=1 should give learning=True"
    # Simulate defaults
    if learning:
        _dc, _rg, _ta, _res = "1", "1", "1", "1"
    elif online:
        _dc, _rg, _ta, _res = "0", "1", "0", "0"
    else:
        _dc, _rg, _ta, _res = "1", "0", "1", "0"
    collect    = os.environ.get("IRP_COLLECT_TEACHER_MODE", _dc).lower() not in {"0", "false", "no"}
    runtime_gnn = os.environ.get("IRP_RUNTIME_GNN_MODE",   _rg).lower() not in {"0", "false", "no"}
    train_after = os.environ.get("IRP_TRAIN_GNN_AFTER_TEACHER", _ta).lower() not in {"0", "false", "no"}
    resume      = os.environ.get("IRP_RESUME_GNN_CHECKPOINT", _res).lower() not in {"0", "false", "no"}
    assert collect,     "online learning: collect_teacher_mode should be True"
    assert runtime_gnn, "online learning: runtime_gnn_mode should be True (GNN already deployed)"
    assert train_after, "online learning: train_gnn_after_teacher should be True (fine-tune)"
    assert resume,      "online learning: resume_gnn_checkpoint should be True (fine-tune, not retrain)"
    # Online learning epochs default = 2
    default_epochs = str(int(os.environ.get("IRP_ONLINE_LEARNING_EPOCHS", "2")))
    epochs = int(os.environ.get("IRP_GNN_TRAIN_EPOCHS", default_epochs))
    assert epochs <= 3, f"Online learning epochs should be small (<=3), got {epochs}"

check("online learning: collect=True, gnn_runtime=True, train=True, resume=True, epochs<=3", _mode_online_learning_defaults)

# Clean up env vars we set for the checks above
for _k in ("IRP_ONLINE_INFERENCE", "IRP_ONLINE_LEARNING", "IRP_COLLECT_TEACHER_MODE",
           "IRP_RUNTIME_GNN_MODE", "IRP_TRAIN_GNN_AFTER_TEACHER", "IRP_RESUME_GNN_CHECKPOINT"):
    os.environ.pop(_k, None)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ALNS baseline on CSV-derived IRPData
# ─────────────────────────────────────────────────────────────────────────────
section("5  ALNS baseline solve (test data.csv, 2 stores × 2 SKUs)")

from irp_gurobi_converted import BaselineALNSModel, FullIRPTSolution

_alns_sol_from_csv = None

def _alns_solve_on_csv_data():
    global _alns_sol_from_csv
    assert _irp_data_from_csv is not None, "IRPData not built (section 3 failed)"
    model = BaselineALNSModel(_irp_data_from_csv)
    sol = model.solve(msg=False, time_limit=10, max_iterations=200, seed=99)
    _alns_sol_from_csv = sol
    assert isinstance(sol, FullIRPTSolution)
    assert sol.status in {"ALNS-Feasible", "ALNS-InfeasiblePenalized"}, f"Bad status: {sol.status}"
    assert sol.objective >= 0.0
    # Schema completeness
    data = _irp_data_from_csv
    for s in data.stores:
        for p in data.products:
            for t in data.periods:
                assert (s, p, t) in sol.direct_ship_q, f"Missing direct_ship_q[{s},{p},{t}]"
                assert (s, p, t) in sol.inv_store,     f"Missing inv_store[{s},{p},{t}]"
                assert (s, p, t) in sol.shortage,      f"Missing shortage[{s},{p},{t}]"
    # No lateral transshipment in baseline
    for val in sol.y.values():
        assert val == 0.0, "Baseline ALNS must have y=0 (no LT)"

check("ALNS baseline solves and returns valid FullIRPTSolution", _alns_solve_on_csv_data)

def _alns_vehicle_capacity():
    assert _irp_data_from_csv is not None
    data = _irp_data_from_csv
    model = BaselineALNSModel(data)
    sol = model.solve(msg=False, time_limit=8, max_iterations=100, seed=7)
    for v in data.vehicles:
        for t in data.periods:
            loaded = sum(sol.deliv.get((s, p, v, t), 0.0)
                         for s in data.stores for p in data.products)
            assert loaded <= data.vehicle_capacity + 1e-6, (
                f"Vehicle capacity breached: v={v} t={t} loaded={loaded:.2f} cap={data.vehicle_capacity}"
            )

check("ALNS solution respects vehicle capacity", _alns_vehicle_capacity)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GNN data split logic
# ─────────────────────────────────────────────────────────────────────────────
section("6  GNN data-split logic (build_teacher_graph_dataset)")

from build_teacher_graph_dataset import split_groups

GroupKey = tuple

def _make_keys(n_instances: int, groups_per: int) -> List[GroupKey]:
    keys = []
    import random as _rnd
    for i in range(n_instances):
        for g in range(groups_per):
            keys.append((f"inst_{i}", f"node_{g}", "0", "P1", "1", f"hash_{g}"))
    return keys

def _split_instance_level():
    keys = _make_keys(n_instances=6, groups_per=4)
    rng = __import__("random").Random(42)
    buckets = split_groups(keys, train_ratio=0.70, valid_ratio=0.15,
                           rng=rng, split_by_instance=True)
    assert set(buckets) == {"train", "valid", "test"}
    total = sum(len(v) for v in buckets.values())
    assert total == len(keys), "All keys must appear in exactly one split"
    # No instance should appear in more than one split
    for split_a, keys_a in buckets.items():
        for split_b, keys_b in buckets.items():
            if split_a >= split_b:
                continue
            insts_a = {k[0] for k in keys_a}
            insts_b = {k[0] for k in keys_b}
            overlap = insts_a & insts_b
            assert not overlap, f"Instance leakage between {split_a} and {split_b}: {overlap}"

check("instance-level split: no cross-split instance leakage", _split_instance_level)

def _split_single_instance_fallback():
    keys = _make_keys(n_instances=1, groups_per=10)
    rng = __import__("random").Random(0)
    buckets = split_groups(keys, train_ratio=0.70, valid_ratio=0.15,
                           rng=rng, split_by_instance=True)
    total = sum(len(v) for v in buckets.values())
    assert total == len(keys), "All keys must appear with single-instance fallback"
    assert len(buckets["train"]) >= 1

check("single-instance fallback: group-level split, all keys covered", _split_single_instance_fallback)

def _split_ratios_respected():
    keys = _make_keys(n_instances=10, groups_per=5)
    rng = __import__("random").Random(1)
    buckets = split_groups(keys, train_ratio=0.70, valid_ratio=0.15,
                           rng=rng, split_by_instance=True)
    n = len(keys)
    train_frac = len(buckets["train"]) / n
    assert 0.50 <= train_frac <= 0.90, f"train fraction {train_frac:.2f} out of expected range"
    assert len(buckets["test"]) >= 1, "test split must have at least one key"

check("split ratios roughly respected (train 50-90%)", _split_ratios_respected)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — GNN utilities smoke
# ─────────────────────────────────────────────────────────────────────────────
section("7  GNN utilities")

import utilities as gnn_utils

def _adaptive_select_smoke():
    probs = [0.9, 0.6, 0.4, 0.2, 0.1, 0.05]
    selected, info = gnn_utils.adaptive_select_indices(
        probs, selection_mode="cumulative_mass", mass_threshold=0.70, min_keep=1
    )
    assert isinstance(selected, list)
    assert len(selected) >= 1
    assert "adaptive_k" in info
    assert info["adaptive_k"] >= 1

check("adaptive_select_indices returns valid selection", _adaptive_select_smoke)

def _build_training_sample_smoke():
    import json
    # 7-dim per constraint: [dual, rhs_amount, covered, residual, urgency, is_need, is_surplus]
    constraint_json = json.dumps([
        [0.3, 1.0, 0.6, 0.4, 0.8, 1.0, 0.0],
        [0.1, 2.0, 1.5, 0.5, 0.4, 0.0, 1.0],
    ])
    rows = [
        {
            "column_id": "col_0",
            "reduced_cost": -0.5,
            "teacher_label": 1,
            "teacher_score": 0.8,
            "column_features_json": json.dumps([1.0, 0.2, 0.3]),
            "constraint_features_json": constraint_json,
            "edge_pairs_json": json.dumps([[0, 0]]),
            "edge_weights_json": json.dumps([1.0]),
        },
        {
            "column_id": "col_1",
            "reduced_cost": 0.1,
            "teacher_label": 0,
            "teacher_score": 0.1,
            "column_features_json": json.dumps([0.5, 0.1, 0.9]),
            "constraint_features_json": "",
            "edge_pairs_json": json.dumps([[0, 0]]),
            "edge_weights_json": json.dumps([0.5]),
        },
    ]
    sample = gnn_utils.build_training_sample_from_exported_teacher_rows(
        rows,
        episode_id="ep0",
        source_instance="inst0",
        product="P1",
        period="1",
        branch_node_id="root",
        decision_state_id="state0",
    )
    required_keys = {"column_features", "constraint_features",
                     "edge_index_col_to_con", "edge_attr_col_to_con",
                     "labels_binary"}
    missing = required_keys - set(sample.keys())
    assert not missing, f"Sample missing keys: {missing}"
    assert sample["column_features"].shape[0] == 2, "Expected 2 columns"
    assert sample["labels_binary"].shape[0] == 2

check("build_training_sample_from_exported_teacher_rows returns valid sample", _build_training_sample_smoke)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — LateralTransshipmentCG init (no Gurobi solve)
# ─────────────────────────────────────────────────────────────────────────────
section("8  LateralTransshipmentCG init")

from irp_gurobi_converted import LateralTransshipmentCG

def _cg_init():
    assert _irp_data_from_csv is not None
    assert _alns_sol_from_csv is not None, "ALNS solution not built (section 5 failed)"
    engine = LateralTransshipmentCG(
        data=_irp_data_from_csv,
        baseline_solution=_alns_sol_from_csv,
        use_gnn=False,
        collect_teacher_mode=False,
        heuristic_top_k_mode=False,
    )
    assert hasattr(engine, "data")
    # _gnn_scoring_failures is lazily set on first failure; getattr default = 0
    assert getattr(engine, "_gnn_scoring_failures", 0) == 0

check("LateralTransshipmentCG initialises with CSV-derived IRPData", _cg_init)

def _cg_heuristic_mode_init():
    assert _irp_data_from_csv is not None
    assert _alns_sol_from_csv is not None
    engine = LateralTransshipmentCG(
        data=_irp_data_from_csv,
        baseline_solution=_alns_sol_from_csv,
        use_gnn=False,
        collect_teacher_mode=False,
        heuristic_top_k_mode=True,
        heuristic_top_k=10,
    )
    assert engine.heuristic_top_k_mode is True
    assert engine.heuristic_top_k == 10

check("LateralTransshipmentCG heuristic mode flag stored correctly", _cg_heuristic_mode_init)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Results:  {_passed} passed,  {_failed} failed")
print(f"{'='*70}\n")
sys.exit(0 if _failed == 0 else 1)
