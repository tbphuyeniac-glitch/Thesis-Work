# SLA Penalty (L2 Quadratic) — Archived Logic

**Status:** Removed from codebase on 2026-05-02. This document preserves the
full design and equations so the work can be cited / discussed in the thesis
even though the mechanism did not produce the expected E2 vs A0 differentiation
and was retired.

**Origin:** introduced in the column-generation (CG) pipeline of
`irp_gurobi_converted.py` to create a defendable runtime gap between the
exact-pricing baseline (A0) and the GNN-filtered pruned-exact variant (E2),
under matched LP-objective values. The penalty turned the master into a QP and
the pricing MIP into an MIQP, so A0 (full exact pricing) was provably slower
than the linear MIP under L1 — providing the runtime gap E2 needed to
demonstrate filter benefit.

**Why removed:** in practice the penalty did not yield a clean separation in
realized objective between A0 and E2 (RMP objectives matched to numerical
precision regardless of variant). Because the RC duals already encoded the
marginal penalty effect, adding the same penalty term to pricing
double-counted, and the A0 objective ended up matching E2's at convergence
within float tolerance. The mechanism is documented here in case future work
wants to revisit it with a different penalty structure or weighting.

---

## 1. Activation and configuration

Default OFF — bit-identical fallback for any caller that does not opt in.

| Env var          | Type  | Default | Purpose                                           |
| ---------------- | ----- | ------- | ------------------------------------------------- |
| `IRP_SLA_PENALTY`| flag  | `off`   | Master switch (`on`/`1`/`true`/`yes` to enable)   |
| `IRP_SLA_MU`     | float | `0.0`   | μ — shortage penalty weight                        |
| `IRP_SLA_NU`     | float | `0.0`   | ν — surplus-leftover penalty weight                |
| `IRP_SLA_ALPHA`  | float | `4.0`   | α — scale on time_urgency for ω(j,p,t)            |
| `IRP_SLA_BETA`   | float | `2.0`   | β — scale on surplus_ratio for ρ(i,p,t)           |

If `μ ≤ 0` AND `ν ≤ 0` the helper short-circuits to disabled, leaving every
optimisation path bit-identical to the legacy linear-objective code.

Per-variant control via the `pricing_use_sla_quadratic: bool = True` parameter
on `LateralTransshipmentCG.__init__`. When `True` the pricing subproblem became
an MIQP that mirrored the master's penalty function. When `False` pricing
stayed linear and only saw the SLA penalty implicitly via RMP duals — used for
the `A0_with_penalty` variant in the E2 benchmark.

### Calibration regimes used in experiments

| Regime               | μ     | ν     | α   | β   | Notes                                 |
| -------------------- | ----- | ----- | --- | --- | ------------------------------------- |
| Disabled (default)   | 0     | 0     | —   | —   | Bit-identical to legacy CG             |
| E1 teacher data      | 0.005 | 0.005 | 4.0 | 2.0 | Initial small-magnitude calibration    |
| E2 teacher / matched | 0.05  | 0.05  | 4.0 | 2.0 | Production E2 training regime          |
| E2 local benchmark   | 0.10  | 0.025 | 4.0 | 2.0 | μ×20, ν×5 — magnitude sanity check     |

---

## 2. Weight functions

### Time urgency ω(j,p,t)

```
time_urgency(t) = (rank(t) − 0)  /  (n_periods − 1)            ∈ [0, 1]
ω(j,p,t)        = 1  +  α · time_urgency(t)
```

`time_urgency` is 0 at the start of the horizon and 1 in the last period —
late-period shortages are penalised more heavily, matching realistic SLA
weight patterns where stocking out close to a deadline is worse than early.

### Surplus ratio ρ(i,p,t)

```
surplus_ratio(i,p,t)  = clamp(donor_surplus(i,p,t) / total_surplus(p,t), 0, 1)
ρ(i,p,t)              = 1  +  β · surplus_ratio(i,p,t)
```

`total_surplus(p,t) = Σ_s donor_surplus(s,p,t)`. Stores holding a larger share
of the surplus pool are penalised more for leaving inventory unused (donor
imbalance correction).

---

## 3. Master / RMP objective (L2 quadratic)

Full quadratic master:

```
min   baseline_without_shortage
    + Σ_p∈patterns  c_p · λ_p
    + Σ_(s,p,t)     need_penalty(s,p,t) · residual_need(s,p,t)
    + μ · Σ_(s,p,t) ω(s,p,t) · residual_need(s,p,t)²        ← shortage penalty
    + ν · Σ_(s,p,t) ρ(s,p,t) · surplus_unused(s,p,t)²       ← surplus penalty
```

`residual_need(s,p,t) ≥ 0` and `surplus_unused(s,p,t) ≥ 0` are continuous
master-level slack/excess variables.

Coupling constraints (active only when `(p,t) ∈ active_product_periods`):

```
Σ_pat∈relevant  inflow_pat(s) · λ_p   +   residual_need(s,p,t)  ≥  need(s,p,t)

Σ_pat∈relevant  outflow_pat(s) · λ_p  +   surplus_unused(s,p,t) =  surplus(s,p,t)   ← EQUALITY when SLA on
                                          (or  ≤  when SLA off, the legacy form)
```

`surplus_unused(s,p,t)` is bounded by `[0, surplus(s,p,t)]`. The `==` form is
introduced only when SLA is enabled and the store has positive surplus —
otherwise the constraint stays `≤`.

### Why quadratic (L2) and not L1

L1 penalty with one-sided non-negative variables (`residual_need`,
`surplus_unused`, both ≥ 0) collapses to a linear cost coefficient and never
makes the MIP/LP harder. L2 turns the master into a QP and the pricing MIP
into an MIQP, so A0's exact pricing is provably slower under penalty —
defendable as the source of E2's runtime advantage at matched LP-objective.

---

## 4. Pricing subproblem (MIQP form)

Per-(p,t) pricing MIP, augmented with the same quadratic penalty:

```
min  Σ_(i,j) c_ij · y_ij  +  F_lt · Σ_ij z_ij                  ← linear ship + fixed
   − π_conv                                                     ← convexity dual
   − Σ_j π_need(j,p,t)   · frac_j  · Σ_i y_ij                   ← need-coverage dual
   − Σ_i π_surplus(i,p,t)· frac_i  · Σ_j y_ij                   ← surplus-cap dual
   + μ · Σ_j  ω(j,p,t) · (need_j − Σ_i y_ij)²                   ← shortage MIQP penalty
   + ν · Σ_i  ρ(i,p,t) · (surplus_i − Σ_j y_ij)²                ← surplus MIQP penalty
```

`frac_j` and `frac_i` are linearisation fractions derived from
`implied_net_lt`, capping the dual rebate to feasible coverage levels.
The squared terms expand to `need² − 2·need·inflow + inflow²` and similarly
for surplus, which Gurobi handles natively as `gp.QuadExpr`.

### Duality double-count caveat

Under standard CG, RMP duals already encode the marginal penalty effect, so
adding the same penalty term to pricing double-counts. The original design
accepted this controlled bias for three reasons:

1. The same penalty applies to BOTH A0 and C/E2 → fair per-iteration
   comparison.
2. The resulting LP optimum is invariant of the bias direction (it just
   changes WHICH columns get added when), so the final RMP objective converges
   to the same value.
3. It makes the pricing MIQP genuinely harder, which is the mechanism by
   which "A0 is slow under penalty" — exactly the thesis claim being
   demonstrated.

In practice (3) was the only effective consequence — the LP did indeed
converge to the same value, validating point (2), but the runtime gap turned
out to be smaller than the variance across instances and the GNN filter did
not consistently beat the linear-pricing baseline.

---

## 5. Realized cost breakdown

To keep the realized-cost reporting consistent with the optimisation objective
when SLA was active, the `build_realized_operating_cost_breakdown` function
included the same quadratic terms on the realized post-LT inventory state:

```
sla_shortage_penalty = μ · Σ_(s,p,t) (1 + α · time_urgency(t)) · realized_shortage(s,p,t)²

sla_surplus_penalty  = ν · Σ_(p,t) Σ_s ρ(s,p,t) · realized_inventory(s,p,t)²
                       where ρ = 1 + β · surplus_ratio(s, realized_inv_share)

realized_operating_cost = direct_cw_unit_cost
                        + store_holding_cost
                        + warehouse_holding_cost
                        + route_distance_cost
                        + vehicle_fixed_cost
                        + lateral_transshipment_cost
                        + shortage_cost
                        + sla_shortage_penalty
                        + sla_surplus_penalty
```

The realized "surplus_unused" was approximated by the realized per-store
inventory level itself — the natural per-store quantity that the master's
`surplus_unused` variable tracks.

---

## 6. Benchmark variants that exercised the penalty

| Variant            | `pricing_use_sla_quadratic` | Notes                                      |
| ------------------ | --------------------------- | ------------------------------------------ |
| `A0_with_penalty`  | `False`                     | Linear-pricing CG; penalty only via duals  |
| `E2_pruned_gnn`    | `True`                      | Full MIQP pricing on a pruned pair set     |
| `A0_QUAD`          | `True`                      | Full exact MIQP, all neg-RC columns/iter   |
| `A0_QUAD_1COL`     | `True`                      | MIQP, top-1 RC column/iter                 |

Δ(A0_with_penalty, E2_pruned_gnn) was intended to measure the runtime cost of
MIQP + 4-feature pruning + GNN against linear-MIP A0. No Stackelberg; B&P off
for clean CG-only comparison.

---

## 7. Files where SLA logic lived (pre-removal)

For traceability if anyone needs to recover the removed code from git history:

- `irp_gurobi_converted.py`
  - `_get_sla_penalty_config`, `_sla_time_urgency`, `_sla_surplus_ratio`
  - `LateralTransshipmentCG.__init__` (`pricing_use_sla_quadratic` param)
  - `_solve_exact_pricing_subproblem` (MIQP terms)
  - `solve_rmp` (master QP terms + `surplus_unused` variables, `==` constraint)
  - `build_realized_operating_cost_breakdown` (realized penalty)
  - `IRPRecourseSolver.run_lt_recourse` (factory plumb-through)
  - `run_three_way_benchmark` `e2_only` variant config
- Consumer scripts that set `IRP_SLA_*` env vars or passed
  `pricing_use_sla_quadratic`:
  `run_e1_large_scale.py`, `run_e1_teacher_gen.py`, `run_a0_vs_gnn_k3.py`,
  `local_e1_benchmark.py`, `local_e1_sweep_realistic.py`,
  `run_e2_teacher_gen.py`, `calibrate_sla_mu.py`, `run_a0_1col_vs_e2.py`,
  `run_benchmarks_e1_e2.py`, `run_e2_ablation_pricing.py`,
  `test_matched_column_counts.py`, `run_small_scale_cg_test.py`,
  `measure_e2_penalty.py`, `run_e2_ablation_pruning.py`,
  `run_test_scenarios_gnn_only.py`, `run_large_instance_gnn_test.py`,
  `run_test_scenarios_benchmark.py`, `run_a0_vs_gnn_ranker_pool15.py`,
  `run_a0_vs_gnn_ranker.py`, `run_e2_test_30_benchmark.py`,
  `run_e2_local_benchmark.py`, `run_a0_vs_gnn_only.py`, `debug_same_obj.py`,
  `local_e1_sweep.py`, `local_smoke_benchmark.py`,
  `GNN/train_bipat_from_aggregate.py`.

The `pricing_use_sla_quadratic=...` kwargs were stripped from 7 active scripts
during the cleanup. The `IRP_SLA_*` env-var assignments were left in place
because they are now silent no-ops — but they can be cleaned up when those
scripts are next touched.

---

## 8. Why this design did not deliver the runtime story

Empirical observation that motivated removal:

1. **A0 and E2 RMP objectives matched to numerical precision** at convergence,
   even with `pricing_use_sla_quadratic=True` on the E2 side. The QP/MIQP
   structure did not produce a meaningful objective gap — only WHICH columns
   were generated changed, not the final LP value (consistent with the duality
   argument in §4).
2. **Runtime gap was instance-dependent and small.** The MIQP did slow down
   pricing per-iteration, but the iteration count compensated, and the
   variance across shocked scenarios swamped the mean gap.
3. **GNN filter benefit could not be cleanly attributed to the penalty.**
   With the penalty active the pricing MIP was harder, but the ranking
   advantage from the GNN was shared between "skipping bad columns" (the
   intended mechanism) and "avoiding hard MIQP solves on filtered pairs"
   (a confound). Removing the penalty makes the ablation cleaner.

The penalty is documented here for completeness. Future work could revisit
with: a one-sided L2 only on shortage (drop the surplus term), or a CVaR-style
shortfall penalty applied at the master only (not in pricing) to avoid the
double-count.
