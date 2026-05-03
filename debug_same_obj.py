"""Debug: A0 vs E1_gnn_only trên 1 scenario, cùng initial patterns.

Prints per-iteration objective để xem diverge từ iteration nào.
Nếu initial patterns giống nhau (seed fixed), kiểm tra điểm phân kỳ.
"""
from __future__ import annotations
import os, gzip, pickle, time
from pathlib import Path
from copy import deepcopy

os.environ["IRP_QUIET"] = "0"           # verbose để thấy chi tiết từng iter
os.environ["IRP_CG_STOPPING_MODE"] = "convergence"
os.environ["IRP_PATTERN_INIT_SEED"] = "42"   # fix seed rõ ràng
os.environ.pop("IRP_SLA_PENALTY", None)
os.environ.pop("IRP_DISABLE_RC_FILTER", None)

import irp_gurobi_converted as irp

GNN_CKPT = "GNN/trained_models/irplt_teacher_filtered_local200_fixed/bigat/pairwise_rank/best_model.pt"
SCENARIO  = "Test 30 scenarios/test_baselines/scenarios/base_01__normal_global__seed1373158607.pkl.gz"

with gzip.open(SCENARIO, "rb") as f:
    s = pickle.load(f)
shocked_data = s["shocked_data"]
baseline_sol = s["baseline_sol"]

SHARED_KWARGS = dict(
    use_random_initial_patterns=True,
    n_initial_patterns_per_product_period=5,
    cg_iterations=30,
    msg=False,
    use_classical_fallback=False,
    gnn_max_keep=150,
    use_branch_and_price=False,
    bp_max_nodes=15, bp_max_depth=6,
    lt_activation_threshold=10.0,
)

VARIANTS = [
    ("A0_K1", dict(
        use_gnn=False, collect_teacher_mode=False,
        runtime_gnn_mode=False, heuristic_top_k_mode=False,
        exact_full_mode=True,
        gnn_checkpoint=None,
    ), {"IRP_EXACT_PRICING_POOL_SIZE": "1", "IRP_EXACT_GNN_RANKER": "0"}),
    ("E1_K3_GNN", dict(
        use_gnn=True, collect_teacher_mode=False,
        runtime_gnn_mode=True, heuristic_top_k_mode=False,
        exact_full_mode=True,
        gnn_selection_mode="relative_threshold",
        gnn_relative_threshold=0.70,
        gnn_max_keep_fraction=0.30,
        gnn_checkpoint=GNN_CKPT,
    ), {"IRP_EXACT_PRICING_POOL_SIZE": "3", "IRP_EXACT_GNN_RANKER": "1"}),
    ("E1_K3_noGNN", dict(
        use_gnn=False, collect_teacher_mode=False,
        runtime_gnn_mode=False, heuristic_top_k_mode=False,
        exact_full_mode=True,
        gnn_checkpoint=None,
    ), {"IRP_EXACT_PRICING_POOL_SIZE": "3", "IRP_EXACT_GNN_RANKER": "0"}),
]

for v_name, kw, env in VARIANTS:
    saved = {}
    for k, v in env.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v

    data_copy     = deepcopy(shocked_data)
    baseline_copy = deepcopy(baseline_sol)
    pipeline = irp.IRPResearchPipeline(data_copy)
    t0 = time.perf_counter()
    result = pipeline.run_lt_recourse_from_baseline(
        baseline_copy, **SHARED_KWARGS, **kw,
    )
    rt = time.perf_counter() - t0

    cg_sol = result.get("cg_solution")
    obj    = float(getattr(cg_sol, "objective", float("nan")))
    hist   = result.get("cg_episode_history", []) or []
    iters  = max(0, len(hist) - 1)
    cols   = sum(int(h.get("added_columns", 0)) for h in hist)

    with_lt = result.get("realized_with_lt_cost_breakdown") or {}
    shortage_units = float(with_lt.get("total_realized_shortage_units", float("nan")))
    total_demand = sum(
        float(data_copy.realized_demand.get((s, p, t), 0.0))
        for s in data_copy.stores
        for p in data_copy.products
        for t in data_copy.periods
    )
    sla = (1.0 - shortage_units / total_demand) if total_demand > 1e-9 else float("nan")
    sla_str = f"{sla:.4%}" if sla == sla else "n/a"

    print(f"\n{'='*70}")
    print(f"  {v_name}  |  obj={obj:.2f}  iters={iters}  cols_added={cols}  rt={rt:.2f}s")
    print(f"  shortage={shortage_units:.2f} units  sla={sla_str}  total_demand={total_demand:.2f}")
    print(f"{'='*70}")
    for h in hist:
        ep   = h.get("episode", "?")
        cost = h.get("total_cost", float("nan"))
        added = h.get("added_columns", 0)
        prop  = h.get("proposed_columns", 0) if "proposed_columns" in h else "?"
        print(f"  ep={ep:>2}  obj={cost:.2f}  proposed={prop}  added={added}")

    for k, prev in saved.items():
        if prev is None: os.environ.pop(k, None)
        else:            os.environ[k] = prev
