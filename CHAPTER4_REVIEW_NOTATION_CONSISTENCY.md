# Chapter 4 Mathematical Models — Comprehensive Notation & Consistency Review

**Date:** 2026-05-05  
**Scope:** Sections 3.1–3.4 (Column Generation) + full cross-section audit  
**Status:** 8 critical issues, 12 minor issues identified

---

## EXECUTIVE SUMMARY

The mathematical formulation is **generally sound**, but notation is **inconsistent and ambiguous** in several places:

1. **Column flow notation** shifts from `q_{ijpvt}` → `q^c_{ij}` without clear distinction
2. **Dual variables μ, ν** used before formal definition in pricing section
3. **Symbol reuse** (`π` for both shortage penalty AND ALNS rewards)
4. **Missing parameter definitions** (`K^{max}`, `θ_{LT}`, `r_{s,p,t}`)
5. **Index inconsistencies** in constraints (especially constraint 10, 20)
6. **Realized demand notation** (`d^{real}_{s,p,τ}`) introduced without definition in 3.1

The formulas match code implementation, but documentation needs **clearer indexing and scope declarations**.

---

## DETAILED FINDINGS

### 🔴 CRITICAL ISSUE 1: Column Flow Notation Ambiguity (3.2 vs 3.4)

**Location:** Sections 3.2 (line 409) and 3.4 (line 460)

**Problem:**
```
Section 3.2:  c = { (i,j) → q^c_ij | ... }
Section 3.4:  min Σ_{(i,j)} [ f_ij·y_ij + (...) q_ij ]
```

- **Section 3.2** uses `q^c_{ij}` (superscript c, no product/period indices)
- **Section 3.4** uses `q_{ij}` (no superscript, no product/period indices)
- **Question:** Are these the same variable or different?
- **Code match:** In `irp_gurobi_converted.py`, both contexts work on a fixed (p,t) pair, so indices are implicit

**Impact:** Readers cannot tell if pricing MIP variables are new or if they become column flows.

**Fix:**

Rewrite 3.2 Column Definition:
```markdown
### 3.2 LT Column Definition

A column c for product p and period t is a set of **donor-to-receiver arc flows**,
where each arc (i,j) carries quantity q^c_ij of product p in period t:

  c = { (i,j) : q^c_ij ≥ 0 | i ∈ stores, j ∈ stores, i ≠ j }

**Implicit scope:** All arcs in column c share the same (p,t) pair. Product p and 
period t are **fixed when the column is generated**; they do not appear as explicit 
indices on q^c_ij but are implicit from column context.

**Cost of column c:**

  cost_c = Σ_{(i,j) ∈ c} [ f_ij · 𝟙[q^c_ij > 0] + b_ij · q^c_ij ]
```

Rewrite 3.4 Pricing Problem:
```markdown
### 3.4 Pricing Problem and Reduced Cost

**Scope:** For a fixed active product-period pair (p,t), solve:

  min_{q,y} Σ_{(i,j)} [ f_ij·y_ij + (b_ij - μ_{j,p,t} - ν_{i,p,t})·q_ij ]

**Variable notation:** In this subproblem:
- q_ij ≡ q^c_ij (quantity of product p from donor i to receiver j in period t)
- y_ij ∈ {0,1} (binary: arc (i,j) is used in this column)
- Implicit indices: product p and period t are fixed; vehicle index v is not relevant
  at column generation stage

**Constraints:**
  Donor capacity:     Σ_j q_ij ≤ surplus_{i,p,t}     ∀ donor i
  Receiver capacity:  Σ_i q_ij ≤ need_{j,p,t}        ∀ receiver j
  Fixed-charge link:  q_ij ≤ min(surplus_i, need_j)·y_ij
  Arc cardinality:    Σ_{(i,j)} y_ij ≤ K^max
```

---

### 🔴 CRITICAL ISSUE 2: Dual Variables Introduced Late (3.3 vs 3.4)

**Location:** Sections 3.3 (line 421, 427) and 3.4 (line 442–455)

**Problem:**

The RMP first introduces `μ_{s,p,t}` and `ν_{s,p,t}` as duals:
- Line 421: "dual variable μ_{s,p,t} ≥ 0" (for need-cover constraint RMP-1)
- Line 427: "dual variable ν_{s,p,t} ≥ 0" (for surplus-capacity constraint RMP-2)

But section 3.4 then **re-introduces** them with caveats:
- Line 442–444: "Gurobi returns non-positive value for ≤ constraint..."
- Line 449: Uses in per-pair reduced cost WITHOUT restating which constraint each dual comes from
- Line 454–455: Uses ambiguous notation `Σ_j μ_{j,p,t}·q^c_{ij}` — unclear summation scope

**Code behavior:**  
In `achamrah_2022_irpt_matheuristic.py` and `irp_gurobi_converted.py`:
- `μ_{j,p,t}` comes from RMP constraint (RMP-1): need-cover at receiver j
- `ν_{i,p,t}` comes from RMP constraint (RMP-2): surplus-capacity at donor i
- Code correctly retrieves Gurobi duals and applies in pricing

**Fix:**

Insert before section 3.4:

```markdown
### 3.3.5 Dual Variable Recap

Recall from the RMP:
- **μ_{j,p,t} ≥ 0**: Dual of constraint (RMP-1), the need-cover constraint for 
  receiver j, product p, period t. Marginal value of additional need coverage.
- **ν_{i,p,t} ≥ 0**: Dual of constraint (RMP-2), the surplus-capacity constraint for 
  donor i, product p, period t. Extracted as |Gurobi.Pi| since constraint (RMP-2) is ≤.
```

Then in 3.4 reduced cost formula (line 449), add:

```markdown
where:
- f_ij = fixed dispatch cost for arc (i,j)
- b_ij = unit lateral transshipment cost
- μ_{j,p,t} = dual variable (benefit of covering receiver j's need)
- ν_{i,p,t} = dual variable (cost of using donor i's surplus)
- q = transferred quantity in this arc

The full column reduced cost is the sum over all arcs in the column:

  c̄(c) = Σ_{(i,j) ∈ c} [ f_ij·𝟙[q^c_ij > 0] + (b_ij - μ_{j,p,t} - ν_{i,p,t})·q^c_ij ]
```

---

### 🔴 CRITICAL ISSUE 3: Symbol Reuse — π Used for Two Different Concepts

**Location:** Lines 50 (section 1.2) vs. lines 362–365 (section 2.5)

**Problem:**

- **Line 50:** `π_{sp}` = **shortage penalty cost** (parameter, units: $ per unmet unit)
- **Lines 362–365:** `π_d`, `π_r` = **accumulated reward scores** in ALNS weight update (unitless, internal state)

This is a **notation conflict**: same symbol for unrelated quantities.

**Impact:**
- Reader confusion when seeing `π` in different sections
- Cannot apply dimensional analysis
- Inconsistent with standard MIP formulation conventions

**Fix:**

Change ALNS section (2.5) to use different symbol:

```markdown
### 2.5 Adaptive Weight Update

Operator weights are updated every η iterations using reaction factor ρ ∈ (0,1):

  w_d ← (1-ρ)w_d + ρ·(R_d/θ_d);    w_r ← (1-ρ)w_r + ρ·(R_r/θ_r)

where:
- **R_d, R_r** = accumulated reward scores ($\sigma_1$ for new global best, 
  $\sigma_2$ for improvement, $\sigma_3$ for accepted non-improving)
- **θ_d, θ_r** = count of operator usage in the segment
```

---

### 🔴 CRITICAL ISSUE 4: Missing Parameter Definitions (1.2)

**Location:** Lines 400, 418, 463

**Problem — Parameter not in 1.2:**

| Parameter | First Use | Definition |
|-----------|-----------|------------|
| `θ_{LT}` | Line 400 (3.1) | Activation threshold for CG; never defined in 1.2 |
| `K^{max}` | Line 463 (3.4) | Max arcs per column; never defined in 1.2 |
| `S_0` | Line 387 (3.1) | Safety stock floor; mentioned but not in 1.2 |
| `L` | Line 383 (3.1) | Lookahead periods; defined as "`need_lookahead_periods`" only in prose |
| `R` | Line 387 (3.1) | Reserve periods; defined as "`surplus_reserve_periods`" only in prose |

**Code mapping:**

```python
# In irp_gurobi_converted.py:
self.lt_activation_threshold  # θ_LT
self.need_lookahead_periods   # L
self.surplus_reserve_periods  # R
self.safety_stock_units       # S_0
# K^max: not explicitly parameterized; hardcoded in pricing subproblem constraints
```

**Fix:**

Expand section 1.2 Parameters table with:

| Symbol | Description |
|---|---|
| `θ_{LT}` | Activation threshold: min total need or surplus (in any store) to trigger CG for a (p,t) pair |
| `L` | Lookahead period count for demand-cover horizon (code: `need_lookahead_periods`, default 2) |
| `R` | Reserve period count for surplus-capacity horizon (code: `surplus_reserve_periods`, default 1) |
| `S_0` | Safety-stock floor for reserve target (code: `safety_stock_units`, default 0.0) |
| `K^{max}` | Maximum donor-receiver arc cardinality per column (constraint in pricing MIP) |

---

### 🔴 CRITICAL ISSUE 5: RMP Slack Variable Not Defined (3.3)

**Location:** Lines 418, 423

**Problem:**

Equation (RMP) and constraint (RMP-1) use `r_{s,p,t}`, but it is **never formally defined** as a decision variable.

```markdown
min  Z^base + Σ_c cost_c·λ_c + Σ_{s,p,t} π_{s,p,t}·r_{s,p,t}    ← r appears in objective
     
Need-cover:  r_{s,p,t} + Σ_c a^need_{c,s,p,t}·λ_c ≥ need_{s,p,t}  ← r appears in constraint
```

**Code implementation:**
In Gurobi models, `r` is added as a variable representing unmet need (recourse) and incurs penalty `π`.

**Fix:**

Before the RMP, add definition:

```markdown
### 3.3 Restricted Master Problem (RMP)

**Decision variables:**
- **λ_c ∈ [0,1]**: Convex weight for column c; indicates whether/how much column c is selected
- **r_{s,p,t} ≥ 0**: Slack (recourse) variable representing unmet demand at store s, product p, 
  period t. Incurs shortage penalty π_{s,p,t} per unit.

**Objective:**

  min Z^base + Σ_c cost_c·λ_c + Σ_{s,p,t} π_{s,p,t}·r_{s,p,t}

where:
- Z^base = objective value from Stage 1 baseline solution
- cost_c = cost of operating column c (fixed + variable transshipment cost)
- π_{s,p,t} = shortage penalty per unit of unmet need at (s,p,t)
```

---

### 🟠 MAJOR ISSUE 6: Realized Demand Notation (3.1)

**Location:** Lines 383, 387 (used), but **never formally introduced**

**Problem:**

Section 3.1 uses `d^{real}_{s,p,τ}` (realized demand) without saying:
- Whether it's a parameter or derived from demand shock
- Relationship to forecast `d_{spt}` from section 1.2
- When/how it's computed (after Stage 1 baseline)

**Code flow:**
```
1. baseline_solution = ALNS(d_spt)          # Stage 1: forecast demand
2. realized_demand = apply_demand_shocks()  # Hidden realized demand shock
3. need, surplus = build_need_surplus(d^real)  # Stage 2: use realized demand
```

**Fix:**

Add to 3.1 opening:

```markdown
### 3.1 Post-Shock State Definitions

After the baseline plan from Stage 1 is executed, the realized market demand **deviates 
from forecast**. We denote:

**Realized demand** d^{real}_{s,p,t}: The actual demand observed at store s for product p 
in period t, after a hidden demand shock has been applied. In Stage 1, realized demand 
equals forecast: d^{real}_{s,p,t} = d_{spt}. In Stage 2 (CG), realized demand incorporates 
a **local reallocation shock** that redistributes demand among stores within each (p,t) 
pair based on fragility heuristics.

**Post-shock inventory** Î^{post}_{s,p,t}: Ending inventory at store s after executing 
the baseline plan and absorbing the hidden demand shock.

**Post-shock shortage** B̂^{post}_{s,p,t}: Unmet demand (backorder/stockout) at store s 
after demand shock, given only the baseline shipments.
```

---

### 🟠 MAJOR ISSUE 7: Undefined Variable I^{beg}_{sp,t} (Constraint 8)

**Location:** Lines 168–172 (constraint 8)

**Problem:**

Constraint (8) references `I^{beg}_{sp,t}` with a piecewise definition:
```
I^{beg}_{sp,t} = I^0_{sp}  if t=1, else I_{sp,t-1}
```

But `I^{beg}_{sp,t}` is **not listed in section 1.3 (Decision Variables)** nor in 1.2 (Parameters).

**Code equivalent:**
It's computed implicitly in inventory balance loops (beginning-of-period inventory).

**Fix:**

Add to section 1.3 Decision Variables, or create a "Derived Values" subsection:

```markdown
### 1.3.A Derived Reference Values

| Symbol | Description |
|---|---|
| I^{beg}_{sp,t} | Beginning-of-period inventory (reference, not a decision variable). Defined as: I^0_{sp} if t is first period, else I_{sp,t-1}. Used in constraint (8) to enforce LT source feasibility. |
```

---

### 🟠 MAJOR ISSUE 8: Constraint (10) Missing Quantifier

**Location:** Line 185

**Problem:**

```
$$\sum_{v \in V} \sum_{\substack{i \in N_0 \\ i \neq j}} x_{ijvt} \leq 1$$
```

The constraint header doesn't include `∀ j ∈ N, v, t`. It's implied but should be explicit.

**Fix:**

```markdown
**Constraint (10) — Single-visit limit per store per period:**

$$\sum_{v \in V} \sum_{\substack{i \in N_0 \\ i \neq j}} x_{ijvt} \leq 1,
\quad \forall j \in N,\; v, t$$
```

---

### 🟡 MINOR ISSUE 9: Index Notation Ambiguity in Constraint (4b) and (2)

**Location:** Lines 122–146

**Problem:**

Constraint (2) uses `y_{jspvt}` and `y_{sjpvt}` (with j, s, p, v, t explicitly named), while section 1.3 defines `y_{ijpvt}` generically. This works mathematically but is confusing.

```markdown
Constraint (2): I_{spt} = ... + Σ_j Σ_v y_{jspvt} - Σ_j Σ_v y_{sjpvt}
Definition (1.3): y_{ijpvt} (i → j direction)
```

A reader must reverse-engineer that in constraint (2), j plays the source role and s the destination.

**Fix:**

Add note after 1.3 variable table:

```markdown
**Index convention for y_{ijpvt}:** The notation follows source→destination order: 
i is the sending store (source), j is the receiving store (destination). Thus:
- y_{jspvt} in constraint (2) represents inflow to s (j sends to s)
- y_{sjpvt} in constraint (2) represents outflow from s (s sends to j)
```

---

### 🟡 MINOR ISSUE 10: Notation Inconsistency — π_{sp} vs π_{s,p,t}

**Location:** Line 50 (1.2) vs. line 418 (RMP objective)

**Problem:**

- **1.2, line 50:** `π_{sp}` (product-store pair; **no time index**)
- **RMP, line 418:** `π_{s,p,t}` (product-store-time triplet; **includes time**)

Formulation 1 uses static penalties, but RMP uses dynamic per-period penalties.

**Code:** Uses dynamic per-period penalties throughout.

**Fix:**

Clarify in RMP intro:

```markdown
**Dynamic shortage penalties:** While Part 1 (full model) may assume static penalties 
π_{sp}, the RMP uses **dynamic per-period penalties** π_{s,p,t} to reflect time-varying 
urgency of unmet demand. These can be derived from π_{sp} or set independently.
```

---

### 🟡 MINOR ISSUE 11: Constraint (20) Index Notation τ vs t

**Location:** Lines 230–236

**Problem:**

Constraint (20) introduces τ (tau) as a summation index over time windows, but doesn't explicitly state:
- τ ranges within window [t₁, t₂]
- τ ≠ t necessarily
- What t₁, t₂ represent (they're quantified ∀ with unclear scope)

```markdown
Σ_{v} Σ_{τ=t₁}^{t₂} z_{svτ} + ...  ≥  ...
```

Is this a constraint for *every possible window* [t₁, t₂], or just *some* windows?

**Fix:**

Clarify constraint statement:

```markdown
**Constraint (20) — Inventory-driven visit validity** (∀ store s, product p, time window [t₁, t₂]):

Cumulative demand coverage is linked with store visits and incoming transshipment 
over any **non-empty time window [t₁, t₂] ⊆ T**:

  [formula]
```

---

### 🟡 MINOR ISSUE 12: Reduced Cost Notation — Per-Pair vs Full Column

**Location:** Lines 449–455

**Problem:**

Line 449 gives a per-pair reduced cost formula:
```
c̄_{ij} = f_{ij} + (b_{ij} - μ_{j,p,t} - ν_{i,p,t})·q
```

But then says "the full column reduced cost is the sum of pair contributions" and 
gives a different formula (lines 453–455). These should be shown as equivalent.

**Fix:**

```markdown
**Per-arc reduced cost** (for donor-receiver pair (i,j) with flow q):

  c̄_{ij} = f_{ij}·𝟙[q>0] + (b_{ij} - μ_{j,p,t} - ν_{i,p,t})·q

**Column reduced cost** (sum of all arcs in column c):

  c̄(c) = Σ_{(i,j) ∈ c} c̄_{ij}
       = Σ_{(i,j) ∈ c} [f_{ij}·𝟙[q^c_ij>0] + (b_{ij} - μ_{j,p,t} - ν_{i,p,t})·q^c_ij]
       = cost_c - Σ_{(i,j) ∈ c} μ_{j,p,t}·q^c_ij - Σ_{(i,j) ∈ c} ν_{i,p,t}·q^c_ij
```

---

## CONSISTENCY CHECK: Code vs. Chapter 4

### ✅ Verified: Demand Windows & Need/Surplus Formulas

| Formula | Chapter 4 | Code Location | Match? |
|---------|-----------|---------------|--------|
| D^{cover}_{s,p,t} = Σ_τ d^real_{s,p,τ} | Line 383 | irp_gurobi_converted.py:4676–4679 | ✅ |
| D^{reserve}_{s,p,t} = max(S_0, Σ_τ ...) | Line 387 | irp_gurobi_converted.py:4679–4686 | ✅ |
| need_{s,p,t} = max(B̂, D^c - Î) | Line 391 | irp_gurobi_converted.py:4693 | ✅ |
| surplus_{s,p,t} = max(0, Î - D^r) | Line 393 | irp_gurobi_converted.py:4694 | ✅ |

### ✅ Verified: Post-Shock Inventory Calculation

```python
# Code: irp_gurobi_converted.py:3246–3280
ending_inventory = max(0.0, prev_inventory + shipment - realized_demand)
post_shock_shortage = max(0.0, realized_demand - prev_inventory - shipment)
```

Matches constraint (2) and inventory balance logic ✅

### ✅ Verified: RMP Constraint Structure

Code correctly implements:
- Need-cover constraint (RMP-1) with μ duals
- Surplus-capacity constraint (RMP-2) with ν duals
- Convex combination λ_c ∈ [0,1]

✅

### ⚠️ Code Detail: K^{max} Hard-Coded

The parameter `K^{max}` (arc cardinality) is **not exposed in the parameter interface**. 
It appears hardcoded in pricing subproblem constraints:

```python
# irp_gurobi_converted.py (approx line 5500–5600, pricing MIP)
# m.addConstr(gp.quicksum(...) <= K_MAX)  # Arc limit
```

**Recommendation:** Add to CG initialization parameters.

---

## IMPLEMENTATION NOTES (from Code Review)

The code implementation in `irp_gurobi_converted.py` is **correct** despite documentation inconsistencies:

1. **Demand shock** (`apply_hidden_local_reallocation_demand_shocks`): 
   - Applies hidden local reallocation shocks to realized demand
   - Probabilistic, with reallocation heuristics based on forecast fragility
   - **NOT documented in Chapter 4** (is this intentional?)

2. **Need/Surplus calculation** (`_build_need_and_surplus_proxies`):
   - Correctly implements windowed horizons with lookahead/reserve periods
   - Falls back to baseline inventory if post-shock not yet computed
   - Matches formulas exactly ✅

3. **RMP structure** (`LateralTransshipmentCG` class):
   - Correctly builds convex combination RMP
   - Uses Gurobi dual extraction (with sign handling for ≤ constraints)
   - Calls pricing MIP per active (p,t) pair
   - **Implicit product-period scoping** matches our recommended notation fix

4. **Pricing MIP** scope:
   - Solves for fixed (p,t) per call
   - Indices in pricing MIP are implicitly (p,t)-scoped
   - Matches our recommendation to clarify this ✅

---

---

## 🔴 MISSING ALGORITHMIC COMPONENTS (Not in Chapter 4 v3)

These are implemented in code but **not documented** in Chapter 4:

### **Missing 1: Inactive (p,t) Pair Constraint**

**Code location:** `irp_gurobi_converted.py:6006–6007`

```python
if (p,t) not in active_product_periods:
    mdl.addConstr(residual_need[(s,p,t)] == need[(s,p,t)])
```

**What it does:** For product-period pairs that don't meet the activation threshold `θ_{LT}`, 
RMP adds a **hard constraint** forcing `r_{s,p,t} = need_{s,p,t}`. This means:
- Zero LT columns can be selected for that (p,t)
- All unmet need incurs the full shortage penalty
- No columns are priced for inactive pairs

**Missing in Chapter 4:** Section 3.3 (RMP) should document this constraint:

```markdown
**Inactive (p,t) constraint** (added if (p,t) ∉ active_product_periods):

For all store s:
  r_{s,p,t} = need_{s,p,t}

This enforces that **no LT is allowed** for inactive product-period pairs; 
all unmet demand incurs shortage penalty.
```

---

### **Missing 2: Heuristic vs Exact Pricing Duality**

**Code location:** `irp_gurobi_converted.py:4812–4813` (heuristic) vs. section 3.4 (exact MIP)

**Heuristic pair-level reduced cost proxy:**
```python
dual_score = dual_need.get((j,p,t), 0.0) + dual_surplus.get((i,p,t), 0.0)
reduced_cost_proxy = fixed_cost + unit_cost * qty_cap - dual_score * qty_cap
```

**Exact pricing MIP reduced cost:**
```
c̄(c) = cost_c - Σ μ_{j,p,t}·q_{ij} - Σ ν_{i,p,t}·q_{ij}
```

**The issue:** Chapter 4 documents **only the exact pricing MIP** (section 3.4). 
But code also implements a **heuristic variant** that:
1. Scores donor-receiver pairs using a simplified dual-score proxy (not full column optimization)
2. Uses ranking/pruning heuristics to select top-k pairs
3. Falls back to this when `IRP_USE_EXACT_PRICING=False`

**Missing in Chapter 4:** Section 3.4 should distinguish:

```markdown
### 3.4 Pricing Problem — Exact vs Heuristic Variants

#### 3.4.1 Exact Pricing (Default)

[Current section 3.4 content]

For a fixed active (p,t) pair, solve the full MIP:

  min_{q,y} Σ_{(i,j)} [ f_ij·y_ij + (b_ij - μ_{j,p,t} - ν_{i,p,t})·q_ij ]
  
This returns the globally-optimal column with minimum reduced cost.

#### 3.4.2 Heuristic Pricing (Optional)

As a faster alternative, the code also supports a pair-level heuristic:

For each donor-receiver pair (i,j), compute a proxy reduced cost:

  c̄_proxy(i,j) = f_ij + (b_ij - μ_{j,p,t} - ν_{i,p,t})·qty_cap

where qty_cap = min(surplus_i, need_j).

Then rank pairs by this proxy and select top-k without solving a full column MIP.
This is faster but sacrifices optimality (may miss good multi-arc columns).

**Use case:** Heuristic mode is faster for large instances; exact mode is slower but 
returns proven-optimal columns per (p,t).
```

---

### **Missing 3: Pattern Deduplication & Column Cardinality Cap**

**Code location:** `irp_gurobi_converted.py:3829–3835` (dedup function) and 
lines 5443, 5586 (application)

**What happens:**

After pricing produces a set of candidate patterns, two filters apply:

1. **Deduplication** (`_deduplicate_priced_patterns`):
   - Remove duplicate patterns (same set of flows)
   - Signature-based dedup by pattern.flows dict
   - Keeps only first occurrence

2. **Column cardinality cap** (`max_columns_per_product_period`):
   - Default: 3 new columns per (p,t) per iteration
   - If > 3 patterns survive dedup, keep only top-3 by score
   - Can be disabled (set to 0 for unlimited)

**Missing in Chapter 4:** Section 3.8 (CG Algorithm) should document:

```markdown
#### Step 6.5 — Pattern Deduplication

After pricing returns candidate LT patterns for active (p,t) pairs:

1. Remove duplicate patterns: For patterns c₁, c₂ with same donor-receiver flows 
   (ignoring cost differences), keep only one.

2. Column cardinality limit: Retain at most K_col = `max_columns_per_product_period` 
   patterns per (p,t) per iteration, ranked by BiGAT score (or reduced cost if 
   BiGAT is disabled).

This prevents RMP from growing unboundedly and reduces numerical issues.
```

---

### **Missing 4: Fallback for Missing Post-Shock Inventory**

**Code location:** `irp_gurobi_converted.py:4685–4686`

```python
ending_inventory = max(0.0, float(
    d.post_shock_inventory.get((s,p,t), self.baseline.inv_store[(s,p,t)])
))
```

**What it does:** In `_build_need_and_surplus_proxies`, if `post_shock_inventory` 
hasn't been populated yet (e.g., during Stage 1), code **falls back to baseline inventory** 
from Stage 1 solution.

**Chapter 4 assumption:** Section 3.1 is written as if Î^{post}_{s,p,t} is always 
available (computed after demand shock). But in practice:
- Before demand shock engine runs, post_shock_inventory is empty
- CG can be called in debug/testing mode before shock
- Code gracefully falls back instead of crashing

**Missing in Chapter 4:** Section 3.1 should note:

```markdown
**Inventory source:** Î^{post}_{s,p,t} is computed by the demand-shock engine 
after Stage 1 baseline execution. If not yet available (e.g., during warm-start 
RMP initialization), the code uses baseline inventory I_{s,p,t} as a fallback.
```

---

## SUMMARY OF FIXES NEEDED

### Critical (Notation & Missing Algorithms)
1. **Missing 1 — Section 3.3:** Add inactive (p,t) constraint: r_{s,p,t} = need_{s,p,t} when (p,t) ∉ active pairs
2. **Missing 2 — Section 3.4:** Distinguish exact pricing MIP vs heuristic pair-level proxy
3. **Missing 3 — Section 3.8:** Document pattern deduplication and max_columns_per_product_period cap
4. **Missing 4 — Section 3.1:** Clarify fallback to baseline inventory when post_shock_inventory unavailable
5. **Section 3.2:** Clarify column flow notation and product-period scope
6. **Section 3.4:** Explain relationship between pricing MIP variables and column flows
7. **Section 3.3.5:** Add dual variable recap before pricing section
8. **Section 2.5:** Rename π_d, π_r to avoid conflict with shortage penalty π_{sp}
9. **Section 1.2:** Add parameter definitions: θ_{LT}, K^{max}, L, R, S_0

### Important (Should Fix)
10. **Section 3.3:** Define r_{s,p,t} (slack variable) before use
11. **Section 3.1:** Clarify d^{real}_{s,p,t} notation and demand shock relationship
12. **Section 1.3:** Add I^{beg}_{sp,t} as derived value

### Recommended (Polish)
13. Fix constraint (10) quantifier
14. Add index convention note for constraint (2) y_{jspvt}
15. Clarify π_{sp} vs π_{s,p,t} dynamic vs static
16. Improve constraint (20) window notation

---

## CODE ALIGNMENT VERDICT

⚠️ **The code implementation is mathematically sound BUT Chapter 4 documents only the "happy path".**

### Formulas Match ✅
- Post-shock inventory balance: matches exactly
- Need/surplus windows: matches exactly
- RMP structure & constraints: matches exactly
- Pricing MIP reduced cost: matches exactly

### Missing Algorithmic Details ⚠️
The following are **implemented in code but NOT documented in Chapter 4 v3**:

| Algorithm | Code Location | Impact |
|-----------|---------------|--------|
| Inactive (p,t) hard constraint | irp_gurobi_converted.py:6006 | Forces r = need when (p,t) below threshold |
| Heuristic vs exact pricing duality | irp_gurobi_converted.py:4812 | Two modes: exact MIP or fast pair-level proxy |
| Pattern deduplication | irp_gurobi_converted.py:3829 | Removes duplicate flows before RMP |
| Column cardinality cap | irp_gurobi_converted.py:3493 | Limits to max_columns_per_product_period per (p,t) per iter |
| Post-shock fallback | irp_gurobi_converted.py:4685 | Uses baseline inventory if post_shock unavailable |

**Severity:** These are **important safeguards** in code (not bugs), but readers following Chapter 4 alone would miss them.

**Recommendation:** Add section 3.3.6 (Inactive Pair Handling), section 3.4.2 (Heuristic Variant), 
and expand section 3.8 (CG Algorithm) to include dedup & column cap logic, plus fallback note in 3.1.

---

*End of Review*
