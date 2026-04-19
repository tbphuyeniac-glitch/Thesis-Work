# Step 1 Baseline: Gurobi MILP → ALNS Replacement

## Scope

Only Step 1 (DC → stores → DC routing + inventory, no lateral transshipment) has been
converted from the exact Gurobi solver to an ALNS metaheuristic adapted from
**Song et al. 2023** (multi-period IRP with lateral transshipment). Steps 1B, 2, 3
(post-shock state, demand-shock generation, column generation, LT rebalancing, GNN
ranking) are **untouched** — they keep reading `baseline_sol` exactly as before.

## Files changed

- **`irp_gurobi_converted.py`** (only file modified)
  - Added `_ALNSState` dataclass and `BaselineALNSModel` class (~540 lines) just above
    `AchamrahFullIRPTModel`. Produces a fully populated `FullIRPTSolution`.
  - At `IRPResearchPipeline.run()` Step 1, the `AchamrahFullIRPTModel(...).solve(
    allow_lateral_transshipment=False, ...)` call is **commented out**; it is replaced
    with `BaselineALNSModel(self.data).solve(...)` using the same signature. The old
    Gurobi call is kept as a comment block for reference.
  - `AchamrahFullIRPTModel` itself is **not deleted** — it is still imported/constructed
    nowhere else in the active pipeline, but kept for diagnostic runs or ablation.

## No changes elsewhere

Grep confirms no active file outside `irp_gurobi_converted.py` imports or uses
`AchamrahFullIRPTModel`, `BaselineIRPModel`, or `BaselineIRPSolution`. Downstream
consumers use `baseline_sol`'s public fields (`direct_ship_q`, `inv_store`,
`inv_wh`, `shortage`, `x`, `u`, `z`, `q`, `y`, `deliv`, `load`,
`efficiency_metrics`) — the ALNS output fills all of them. `y` is zero throughout
(as before, since `allow_lateral_transshipment=False`).

## Constraints preserved (from Achamrah baseline, LT disabled)

| Constraint                                               | ALNS enforcement |
|----------------------------------------------------------|------------------|
| Store inventory balance with shortage B                  | Forward simulation in `_evaluate` and solution builder |
| Warehouse inventory balance with replenishment g_{p,t}   | Forward simulation; big penalty if WH goes negative |
| Vehicle capacity Σ deliv ≤ vehicle_capacity              | Enforced in insertion and re-checked in evaluator |
| Store node capacity Σ_p I_s ≤ node_capacity[s]           | Penalty in evaluator |
| Warehouse aggregate capacity Σ_p I_w ≤ node_capacity[CW] | Penalty in evaluator |
| `cw_dispatch_cycle` periodicity                          | Non-dispatch periods have deliv=0 |
| `min_visit_activity_qty` / `min_visit_delivery_qty`      | Penalty in evaluator |
| `max_vehicles_used` per period                           | Penalty in evaluator; guarded in repair ops |
| Single visit per store per period per vehicle            | Route = ordered list of distinct stores |

## Objective preserved (identical terms to Gurobi baseline with y=0)

```
  Σ ship_cost_cw[s,p] · Qdir[s,p,t]
+ Σ holding_cost_store[s,p] · I_s[s,p,t]
+ Σ holding_cost_wh[p] · I_w[p,t]
+ Σ alpha · distance[i,j] · x[i,j,v,t]
+ Σ vehicle_fixed_cost · u[v,t]
+ Σ shortage_cost[s,p] · B[s,p,t]
```

## ALNS algorithm summary (Song et al. 2023 adaptation)

- **Solution representation** — `_ALNSState(routes, deliv)`:
  - `routes[(t, v)]` = ordered list of store visits (CW implicit at both ends).
  - `deliv[(s, p, v, t)]` = delivered quantity.
- **Initial solution** — greedy: for each dispatch period, compute per-(s,p)
  target (cover own-period demand + look-ahead to next dispatch), clip by
  warehouse availability, then nearest-neighbor insertion into vehicles under
  capacity.
- **6 destroy operators**: random delivery removal, worst-cost removal,
  random-route removal, random-period removal, Shaw (geographic) removal,
  low-demand removal.
- **3 repair operators**: greedy insertion, regret-2 insertion, random
  insertion.
- **Operator selection** — roulette wheel with adaptive weights:
  `w_i ← (1-r)·w_i + r·(π_i / θ_i)` (reaction_factor `r=0.1`, updated every 40
  iterations). Scores σ1=33 (new best), σ2=13 (better than current), σ3=9
  (accepted).
- **Acceptance** — simulated annealing: always accept non-worsening feasible
  moves; accept worse with probability `exp(-Δ/T)`. Temperature cools
  geometrically (factor 0.998 per iteration by default).
- **Termination** — `time_limit` (optional) or `max_iterations=2000`.
- **Infeasibility** — all constraint violations (capacity, dispatch cycle, etc.)
  are large linear penalties in the evaluator; the repair operators try to stay
  feasible, but the SA accept rule only takes feasible candidates.

## `FullIRPTSolution` reconstruction

After ALNS returns the best state, the solver fills every tensor:

- `direct_ship_q[s,p,t] = Σ_v deliv[s,p,v,t]`
- `x[i,j,v,t]`, `u[v,t]`, `z[i,v,t]` from each route traversal.
- `q[p,i,j,v,t]` = remaining product p to deliver on arc (i,j) (cumulative
  along the route starting with the total load at CW).
- `load[i,v,t]` = running load after each stop, matching the Gurobi MTZ
  load-flow equations.
- `y[...] = 0` everywhere (LT disabled here, same as the prior Gurobi run).
- `inv_store`, `inv_wh`, `shortage` from deterministic forward simulation
  using the same balance equations as the Gurobi model.
- `efficiency_metrics` contains ALNS-specific keys plus the parity keys
  `gurobi_runtime_seconds`, `nodes_explored`, etc. so
  `print_efficiency_metrics` keeps working.

## Smoke tests

Run:

```bash
python3 smoke_test_alns_baseline.py
```

Builds a tiny synthetic instance (3 stores, 2 products, 4 periods, 2
vehicles) and verifies:

- Output type is `FullIRPTSolution`.
- All required dict keys are populated for every (s,p,t), (v,t), (i,v,t), etc.
- `y` is identically zero.
- Vehicle capacity is respected.
- Every used vehicle has exactly one outgoing CW arc.
- `inv_store` and `shortage` match the deterministic forward-simulation values
  for the reported `direct_ship_q`.

The current pass reports (tiny instance):

```
status=ALNS-Feasible
objective≈308.32
runtime≈0.04s
```

## Run command (full pipeline)

The `__main__` entrypoint is unchanged. The same command as before now runs
ALNS in Step 1:

```bash
# from the Current Code directory
python3 irp_gurobi_converted.py
```

Useful environment overrides (unchanged from before):

```bash
IRP_TIME_LIMIT=1800 \
IRP_CG_ITERATIONS=15 \
IRP_USE_BRANCH_AND_PRICE=1 \
IRP_COLLECT_TEACHER_MODE=1 \
python3 irp_gurobi_converted.py
```

ALNS-specific knobs (if you want to tune without code edits, expose them via
env vars — currently they are constructor args only: `max_iterations`,
`cooling_rate`, `initial_temperature`, `segment_size`, `reaction_factor`,
`seed`). Current defaults follow the Song et al. parameters informally
(reaction factor 0.1, scores 33/13/9, geometric cooling).

## What is NOT changed

- `AchamrahFullIRPTModel`, `BaselineIRPModel`, `BaselineIRPSolution`,
  `FullIRPTSolution` classes and all their fields.
- `apply_hidden_local_reallocation_demand_shocks`,
  `build_post_shock_inventory_state`, `build_post_shock_lt_diagnostics`.
- `LateralTransshipmentCG`, `generate_random_lt_patterns`, all CG / RMP /
  pricing / branch-and-price code.
- GNN training/evaluation pipeline and teacher CSV export format.
- All Results/*.csv output paths and schemas.

## Do any other files need change?

No. Grep results:

```
$ grep -n 'AchamrahFullIRPTModel\|BaselineIRPModel\|BaselineIRPSolution' --include='*.py' .
# hits only in irp_gurobi_converted.py (active) and in legacy/inactive files
# (`thesis code before GNN.py`, `turn off game.py`) per memory/project_overview.md
```

The GNN pipeline files under `GNN/` never import these classes — they
consume the teacher CSV, which the CG loop writes after Step 1. If you
want me to expose the ALNS knobs via environment variables in the
`__main__` block, say the word; otherwise the default parameter set runs
out of the box.
