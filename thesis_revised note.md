# IRP + LT Column Generation + BiGAT GNN
## Fix Notes and Recommended Revisions

This note summarizes the main issues identified in the current implementation and the concrete changes recommended for the next revision of the code.

---

## 1. Current Flow in the Code

The current pipeline runs in the following order:

1. Solve the baseline Achamrah-style full IRPT model with `allow_lateral_transshipment=False`.
2. Use the baseline solution to build LT need/surplus proxies.
3. Generate random initial LT patterns.
4. Solve the restricted master problem (RMP).
5. Run pricing from dual values.
6. Apply Stackelberg acceptance logic.
7. Build candidate LT patterns.
8. Use the BiGAT GNN to score and keep the top-k priced patterns.
9. Add non-duplicate patterns to the pattern pool.
10. Re-optimize the RMP and repeat.

This means the GNN is currently used only as a **selector/ranker over already-generated priced columns**. It does not generate columns itself.

---

## 2. Main Problems Identified

### 2.1 Frozen need/surplus proxy across CG iterations

#### Current behavior
The function `_build_need_and_surplus_proxies()` always uses:
- `self.baseline.shortage`
- `self.baseline.inv_store`

These values come from the initial no-LT baseline solution and stay unchanged across CG iterations.

#### Why this is a problem
Even if dual values change after each RMP re-optimization, the pricing routine still sees almost the same primal proxy state. As a result, it tends to regenerate very similar candidate patterns in later episodes.

#### Recommended fix
Update the pricing proxy after each iteration using the current master solution, not only the original baseline.

Possible directions:
- Use `master_solution.implied_net_lt` to update effective surplus/need.
- Recompute residual shortage after applying currently selected LT patterns.
- Maintain an iteration-dependent proxy state instead of a fixed baseline-only proxy.

#### Expected benefit
This makes pricing responsive to the evolving RMP solution and increases the chance of generating genuinely new columns across episodes.

---

### 2.2 LT activation threshold is too restrictive for small instances

#### Current behavior
A product-period pair becomes active only if:
- total need >= `lt_activation_threshold`
- total surplus >= `lt_activation_threshold`

The threshold is currently set to `10.0`.

#### Why this is a problem
For small instances with only a few stores and SKUs, this threshold may deactivate many `(product, period)` pairs before CG even starts.

#### Recommended fix
Test smaller thresholds, such as:
- `2.0`
- `5.0`

Also export the number of active `(product, period)` pairs per run.

#### Expected benefit
This opens more pricing opportunities and gives the GNN a larger candidate pool to work on.

---

### 2.3 Candidate pool is too small before the GNN step

#### Current behavior
The current configuration is very restrictive:
- `store_limit = 3`
- `sku_limit = 2`
- `n_initial_patterns_per_product_period = 2`
- `max_pairs_per_pattern = 3`
- `top_pairs_per_feature = 8`
- `top_patterns_per_feature = 2`
- `cg_iterations = 3`

#### Why this is a problem
The pool of feasible donor-receiver combinations is already very small in the small instance. After pruning and Stackelberg filtering, only a few patterns remain. Then the GNN is asked to select from a tiny set, so it cannot materially change the outcome.

#### Recommended fix
Increase candidate diversity in pricing. Suggested test values:
- `top_pairs_per_feature: 8 -> 20`
- `top_patterns_per_feature: 2 -> 5`
- `max_pairs_per_pattern: 3 -> 4` or `5`
- `n_initial_patterns_per_product_period: 2 -> 5`
- `cg_iterations: 3 -> 5` or more for experiments

#### Expected benefit
A larger and more diverse candidate set allows the GNN to act as a real ranking mechanism rather than a near-pass-through filter.

---

### 2.4 Pattern IDs are not unique across pricing episodes

#### Current behavior
Pattern IDs are generated using:

```python
pattern_id=f"PRICED_{feature_name}_{p}_T{t}_{built_here}"
```

The variable `built_here` resets every time the function is called.

#### Why this is a problem
Across multiple pricing episodes, the same feature/product/period combination can receive the same ID again. Then `add_patterns()` rejects the new pattern because the ID already exists.

#### Recommended fix
Make pattern IDs globally unique across episodes.

Suggested format:

```python
pattern_id=f"PRICED_E{episode}_{feature_name}_{p}_T{t}_{built_here}"
```

Alternative:
- Use a global pattern counter stored in the CG engine.
- Or append a hash/signature suffix.

#### Expected benefit
This prevents false duplicate rejection caused purely by repeated IDs.

---

### 2.5 Duplicate filtering may be rejecting too many new patterns

#### Current behavior
`add_patterns()` rejects new patterns if:
- `pattern_id` already exists, or
- the pattern signature already exists, or
- the pattern flow dictionary is empty

#### Why this is a problem
If the frozen proxy causes very similar patterns to be priced in later episodes, the signature filter may reject nearly all new patterns.

#### Recommended fix
Keep duplicate filtering, but add diagnostics that explicitly report the rejection reason:
- duplicate ID
- duplicate signature
- empty flow

Also log the pattern signature.

#### Expected benefit
This will show whether the real bottleneck is repeated pricing output or something else.

---

### 2.6 GNN top-k may be too large relative to the number of priced patterns

#### Current behavior
The code uses:
- `gnn_top_k = 5`

But if pricing only generates 3 to 5 patterns in a given episode, the GNN effectively keeps all of them.

#### Why this is a problem
In that case the GNN does score patterns, but it does not actually filter the set, so its online optimization effect becomes negligible.

#### Recommended fix
When the candidate pool is still small, use a stricter GNN filter, for example:
- `gnn_top_k = 2`
- `gnn_top_k = 3`

Also compare:
- raw pricing output count
- count after GNN selection

#### Expected benefit
This helps reveal whether the GNN is materially changing the candidate pool.

---

### 2.7 Reduced-cost acceptance is strict for a small-instance prototype

#### Current behavior
A pattern is only accepted if:

```python
if flows and reduced_cost < rc_tol:
```

with:
- `rc_tol = -1e-6`

There is also pair-level filtering that can skip non-attractive pairs once the pattern already contains at least one flow.

#### Why this is a problem
In a small instance with fixed LT costs and small quantities, many patterns may fail to achieve sufficiently negative reduced cost.

#### Recommended fix
Keep the negative reduced-cost logic, but perform sensitivity tests on:
- fixed LT cost
- unit LT cost
- activation threshold
- pattern size limits
- rc tolerance used only for diagnostics

Also log the reduced cost of every candidate pattern before rejection.

#### Expected benefit
This reveals whether the issue is economic infeasibility or simply over-restrictive filtering.

---

### 2.8 Selected-column history may be empty because pricing never produced enough usable patterns

#### Current behavior
`gnn_selection_history` is populated only if `_select_patterns_with_gnn()` receives non-empty priced patterns and runs successfully.

#### Why this is a problem
An empty selected-column file does not necessarily mean the GNN failed. It may simply mean:
- no active product-period pairs,
- no acceptable patterns built,
- no negative reduced-cost columns,
- or GNN loading/scoring was skipped.

#### Recommended fix
Log the following per episode:
- number of active product-period pairs
- number of candidate pairs before pruning
- number after pruning
- number accepted after Stackelberg
- number of patterns built before GNN
- number of patterns kept by GNN
- number of truly new patterns added to the pool

#### Expected benefit
This makes the bottleneck visible and avoids ambiguous interpretation of an empty JSON output.

---

## 3. Priority Ranking of Bottlenecks

The issues should be prioritized in the following order:

1. **Frozen need/surplus proxy across iterations**
2. **Candidate pool too small due to threshold and pruning limits**
3. **Pattern ID duplication across episodes**
4. **Duplicate filtering rejecting nearly identical repeated patterns**
5. **GNN top-k too loose relative to pool size**
6. **Reduced-cost acceptance too strict for current small instance**

This means the current flat CG behavior is not mainly a GNN problem. It is mostly a pricing-space and flow-design problem upstream of the GNN.

---

## 4. Concrete Code Changes Recommended

### 4.1 Update proxy state by iteration

Revise `_build_need_and_surplus_proxies()` so it can take the current master solution as input.

Suggested direction:
- add an optional argument like `master_solution: Optional[CGSolution] = None`
- if `master_solution` is provided, adjust store-level need/surplus using current selected LT effect
- use this updated state inside `pricing_step()`

---

### 4.2 Add episode-aware pattern IDs

Add an episode counter in the CG engine, then pass it into `_build_patterns_from_pruned_pairs()`.

Suggested pattern ID:

```python
PRICED_E{episode}_{feature_name}_{product}_T{period}_{local_idx}
```

---

### 4.3 Relax activation and diversity parameters for experiments

Recommended test configuration for small instances:

```python
lt_activation_threshold = 2.0 or 5.0
top_pairs_per_feature = 20
top_patterns_per_feature = 5
max_pairs_per_pattern = 4
gnn_top_k = 2 or 3
n_initial_patterns_per_product_period = 5
cg_iterations = 5
```

These are experimental settings for validation, not necessarily final production values.

---

### 4.4 Add detailed diagnostics export

Create a new CSV such as `column_pool_diagnostics.csv` with fields like:

- episode
- product
- period
- feature_name
- donor_store
- receiver_store
- qty_cap
- reduced_cost_proxy
- stackelberg_accepted
- acceptance_score
- compensation
- pattern_id
- pattern_reduced_cost
- gnn_score
- gnn_selected
- duplicate_id_reject
- duplicate_signature_reject
- added_to_pool

This file will be essential for validation and thesis presentation.

---

### 4.5 Add episode-level summary logging

For each CG episode, record:

- active product-period count
- candidate pairs before pruning
- pairs after pruning by feature
- pairs accepted after Stackelberg
- patterns built before GNN
- patterns kept after GNN
- patterns rejected as duplicates
- patterns added to pool
- selected patterns in RMP
- objective value
- improvement from previous episode

This can be saved as `cg_episode_diagnostics.csv`.

---

## 5. Suggested Experimental Validation Plan

After the code changes, compare at least the following settings:

### Setting A — LT + CG without GNN
Use the revised pricing pipeline but disable the GNN.

### Setting B — LT + CG + GNN
Use the revised pricing pipeline and enable BiGAT selection.

### Setting C — LT + CG + simple heuristic ranking
Use a simpler selector such as top reduced-cost patterns only.

### Setting D — No LT baseline
Use the original no-LT baseline model for reference.

Compare:
- total cost
- number of CG episodes
- number of proposed columns
- number of added columns
- number of selected patterns
- runtime
- inventory validation metrics

This will make it possible to isolate the contribution of the GNN from the contribution of the LT-CG framework itself.

---

## 6. Interpretation for the Current Results

Based on the current implementation, the flat total-cost curve and zero added columns are most likely explained by the following combination:

- the pricing proxy is effectively frozen at the baseline no-LT state,
- the active LT space is already small,
- pruning and Stackelberg filtering shrink it further,
- repeated pattern IDs and duplicate signatures block pattern growth,
- and the GNN only scores a pool that is already too small and too similar.

Therefore, the current results should not be interpreted as evidence that the GNN is ineffective. Instead, they indicate that the upstream pricing space is too constrained for the GNN to demonstrate meaningful impact.

---

## 7. Immediate Next Steps

The next revision should do these first:

1. Make pattern IDs unique across episodes.
2. Add rejection-reason diagnostics in `add_patterns()`.
3. Lower the LT activation threshold for small-instance tests.
4. Increase candidate diversity before the GNN stage.
5. Reduce `gnn_top_k` so the GNN actually filters.
6. Update need/surplus proxies using the current master solution rather than only the initial baseline.
7. Export detailed diagnostics at pair-level and pattern-level.

If these changes are implemented, the next run will make it much easier to identify whether the remaining bottleneck is in pricing, Stackelberg filtering, duplicate suppression, or the GNN ranking itself.

---

## 8. Final Conclusion

The main issue in the current code is not that the BiGAT GNN is fundamentally wrong. The main issue is that the column-generation flow becomes too restricted before the GNN can meaningfully influence the search.

In other words:

- the GNN is downstream,
- but the pricing space is already too narrow upstream.

So the next development focus should be on reopening and instrumenting the pricing pipeline first, then reevaluating the GNN contribution under a healthier candidate-generation process.

