# Verification: Test scenario coverage + pricing alignment mechanism

Saved from interactive session 2026-05-01. Tool output `s5fkpz`.

## 1. Test scenario coverage (verifying 30/30 user-provided scenarios)

```
Total: 30 scenarios across 6 groups
  small_4s2sku: 5 scenarios
  med_5s2sku: 5 scenarios
  med_4s3sku: 5 scenarios
  med_5s3sku: 5 scenarios
  med_7s3sku: 5 scenarios
  large_7s4sku: 5 scenarios

Script references 30 | Dir contains 30 pkl.gz files
In dir but NOT in script (skipped): 0
```

Confirmed: `run_e2_test_30_benchmark.py` runs all 30 user-provided scenarios from
`Test 30 scenarios/test_baselines/scenarios/`. No scenarios dropped or substituted.

## 2. `pricing_use_sla_quadratic` code reference (mechanism citation)

`grep -n "pricing_use_sla_quadratic" irp_gurobi_converted.py`:

| Line | Context |
|------|---------|
| 3506 | `pricing_use_sla_quadratic: bool = True,` (LateralTransshipmentCG.__init__ default) |
| 3510 | `self.pricing_use_sla_quadratic = bool(pricing_use_sla_quadratic)` |
| 5292 | `if sla_on_p and (sla_mu_p > 0.0 or sla_nu_p > 0.0) and self.pricing_use_sla_quadratic:` |
| 5297–5328 | quadratic SLA terms added to pricing MIP objective |
| 8152 | `pricing_use_sla_quadratic: bool = True,` (run_lt_recourse_from_baseline param) |
| 8238 | wired through to LateralTransshipmentCG |
| 9296 | A0_with_penalty variant: `"pricing_use_sla_quadratic": False` |
| 9311 | E2_pruned_gnn variant: `"pricing_use_sla_quadratic": True` |

## 3. Author's own comment at lines 5270–5289 (mechanism intent)

```
# ── SLA penalty in pricing (L2 quadratic) ────────────────────────
# When IRP_SLA_PENALTY=on, the pricing subproblem penalises the
# column's contribution to residual shortage / unused surplus by the
# SAME quadratic form used in solve_rmp(). Adding penalty here
# promotes the master's L2 RMP into a properly aligned pricing MIQP:
# A0 (full exact) now solves a quadratic-objective MIP per (p, t),
# which is provably slower than the linear MIP under L1 — providing
# the runtime gap E2 (pruned) needs to demonstrate filter benefit.
#
# Note on duality: under standard CG, RMP duals already encode the
# marginal penalty effect, so adding the same penalty term to pricing
# double-counts. We accept this controlled bias because:
#   (a) the same penalty applies to BOTH A0 and C/E2 → fair
#       per-iteration comparison;
#   (b) the resulting LP optimum is invariant of the bias direction
#       (it just changes WHICH columns get added when), so the final
#       RMP objective converges to the same value;
#   (c) it makes the pricing MIQP genuinely harder, which is the
#       mechanism by which "A0 is slow under penalty" — exactly the
#       thesis claim being demonstrated.
# Bit-identical to legacy pricing when sla_on=False.
```

## 4. Implication for E2 vs A0 iter-count gap

Two-mechanism explanation (to be confirmed by ablation A0_quadratic):

1. **Pricing-RMP cost alignment (likely dominant):** A0 uses linear pricing
   while RMP uses quadratic SLA penalty. A0's pricing proposes columns whose
   reduced-cost looks negative under linear cost but RMP, after adding +
   re-optimizing, only marginally exploits them due to quadratic diminishing
   returns. Duals shift; pricing must re-iterate. E2's pricing matches RMP's
   quadratic objective → columns stable from iteration 1.

2. **GNN top-fraction filter (secondary):** drops near-zero-RC columns that
   would cause LP basis churn in RMP, keeping ~30% top-ranked.

Ablation needed: add A0_QUAD variant (`exact_full_mode=True` +
`pricing_use_sla_quadratic=True`, no GNN) to isolate mechanism 1.
