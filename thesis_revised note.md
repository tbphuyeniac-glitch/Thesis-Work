# Baseline Routing Code Review and Improvement Plan

## File reviewed
`irp_gurobi_converted.py`

This note focuses only on the **baseline model** currently implemented in:
- `AchamrahFullIRPTModel.solve(...)`

The goal is to answer one practical question:

> Is the current baseline already a fully correct routing model, and if not, what should be fixed in code?

---

## 1. Final conclusion

The current baseline is **not yet a fully completed routing model**.

It already contains some routing elements:
- depot/CW,
- vehicles,
- binary route arcs,
- vehicle usage variables,
- arc-capacity linking,
- route extraction,
- store and warehouse inventory balance.

However, it is still better described as a **routing-flavored network-flow IRPT prototype** rather than a fully correct multi-stop delivery routing model.

The main weaknesses are:
1. **direct delivery is tied only to arc `CW -> store`**, so multi-stop delivery is not modeled correctly;
2. **vehicle load progression along the route is missing**;
3. **subtour elimination is incomplete**;
4. **vehicle usage cost is not explicitly charged in the objective**;
5. **synthetic distance matrix is too simple**, so routes may look artificially easy and solver runtime becomes misleadingly fast.

---

## 2. What the current code already does correctly

### 2.1 Depot and vehicles exist
The baseline defines:
- `CW` as the warehouse/depot,
- `V` as the vehicle set,
- `x[(i,j,v,t)]` as binary route decisions,
- `u[(v,t)]` as vehicle activation,
- `z[(i,v,t)]` as node-visit indicators.

This means the model is **not** a pure inventory-only model. It does contain a route skeleton.

### 2.2 Vehicle capacity is partially enforced
The code already includes:

```python
mdl.addConstr(gp.quicksum(q[(p, i, j, v, t)] for p in P) <= d.vehicle_capacity * u[(v, t)])
```

and also:

```python
mdl.addConstr(q[(p, i, j, v, t)] <= d.vehicle_capacity * x[(i, j, v, t)])
```

So the model does consider capacity, but only as an **arc-level carrying bound**, not as a true load evolution throughout the route.

### 2.3 Route start from CW is enforced
The code has:

```python
mdl.addConstr(gp.quicksum(x[(CW, j, v, t)] for j in N) == u[(v, t)])
```

So if a vehicle is active, it must leave the warehouse exactly once.

### 2.4 Flow conservation at stores exists
For each store `j`, vehicle `v`, period `t`, the code enforces:

```python
gp.quicksum(x[(i, j, v, t)] for i in N0 if i != j)
==
gp.quicksum(x[(j, i, v, t)] for i in N0 if i != j)
```

This gives in-flow = out-flow at each visited store.

---

## 3. Main modeling problems and required fixes

---

## Problem 1. Direct delivery is modeled incorrectly for multi-stop routes

### 3.1 Current code
The baseline currently defines direct shipment by:

```python
mdl.addConstr(Qdir[(s, p, t)] == gp.quicksum(q[(p, CW, s, v, t)] for v in V))
```

### 3.2 Why this is a problem
This means product delivered to store `s` is counted **only** if it travels on arc `CW -> s`.

So if a route is:

`CW -> A -> B -> CW`

then:
- delivery to `A` can be captured,
- but delivery to `B` is **not** naturally modeled as goods carried from CW through A and then unloaded at B.

That is not how a real IRP/VRP delivery route works.

### 3.3 Consequence
The current model behaves much closer to:
- direct shipment from CW to each served store,
- plus store-to-store transshipment flow,
- rather than a true delivery tour where a vehicle departs with load and unloads progressively along the path.

This is the single most important reason the current baseline is **not yet a fully correct routing formulation**.

### 3.4 Required fix
Replace the current `Qdir` logic with an explicit **delivery/unloading variable at each store**.

Suggested new variable:

```python
deliv = mdl.addVars([(s, p, v, t) for s in N for p in P for v in V for t in T],
                    lb=0.0,
                    vtype=flow_vtype,
                    name="deliv")
```

Then define total direct delivery as:

```python
mdl.addConstr(
    Qdir[(s, p, t)] == gp.quicksum(deliv[(s, p, v, t)] for v in V)
)
```

And use `deliv` inside vehicle load balance instead of tying delivery to only `CW -> s` arcs.

### 3.5 Priority
**Critical / must fix first**.

---

## Problem 2. Vehicle load progression along the route is missing

### 4.1 Current code
The model uses arc-flow variables `q[(p,i,j,v,t)]`, but it does not explicitly track:
- how much load vehicle `v` leaves CW with,
- how much remains after visiting each store,
- how the load decreases after deliveries.

### 4.2 Why this is a problem
A correct routing delivery model usually needs one of these:
- commodity flow with proper depot-origin semantics,
- or load propagation variables,
- or cumulative load/order variables.

Right now, `q` behaves more like a general network flow over arcs.

### 4.3 Consequence
This can create solutions that are mathematically feasible in the network sense, but operationally not very realistic.

It also contributes to the feeling that:
- routes solve too fast,
- vehicles appear frequently full,
- deliveries do not behave like normal truck unloading.

### 4.4 Required fix
Add explicit load variables, for example:

```python
load = mdl.addVars([(i, v, t) for i in N0 for v in V for t in T],
                   lb=0.0,
                   ub=d.vehicle_capacity,
                   vtype=flow_vtype,
                   name="load")
```

Then impose load transition constraints, for example using big-M:

```python
for i in N0:
    for j in N:
        if i == j:
            continue
        for v in V:
            for t in T:
                delivered_at_j = gp.quicksum(deliv[(j, p, v, t)] for p in P)
                mdl.addConstr(
                    load[(j, v, t)] <= load[(i, v, t)] - delivered_at_j + d.vehicle_capacity * (1 - x[(i, j, v, t)])
                )
                mdl.addConstr(
                    load[(j, v, t)] >= load[(i, v, t)] - delivered_at_j - d.vehicle_capacity * (1 - x[(i, j, v, t)])
                )
```

At depot:

```python
for v in V:
    for t in T:
        mdl.addConstr(load[(CW, v, t)] <= d.vehicle_capacity * u[(v, t)])
```

### 4.5 Priority
**Critical / same priority as Problem 1**.

---

## Problem 3. Subtour elimination is not complete

### 5.1 Current code status
The file itself states that true disjoint path inequalities / branch-and-cut are **not fully implemented**.

The model includes some strengthening inequalities `(16)–(20)`, but does **not** fully guarantee classical subtour elimination in all cases.

### 5.2 Why this matters
A route model without proper subtour elimination may still produce:
- disconnected cycles,
- route fragments,
- or mathematically valid but operationally meaningless loops.

Even if this does not always appear in small instances, it remains a formulation weakness.

### 5.3 Required fix
Choose one of these two options.

#### Option A. MTZ-style subtour elimination
Add node ordering variables per vehicle and period:

```python
ordv = mdl.addVars([(s, v, t) for s in N for v in V for t in T],
                   lb=0,
                   ub=len(N),
                   vtype=GRB.CONTINUOUS,
                   name="ordv")
```

Then add MTZ constraints:

```python
for i in N:
    for j in N:
        if i == j:
            continue
        for v in V:
            for t in T:
                mdl.addConstr(
                    ordv[(i, v, t)] - ordv[(j, v, t)] + len(N) * x[(i, j, v, t)] <= len(N) - 1
                )
```

#### Option B. SEC / callback
If later you want stronger performance and a more academic formulation, migrate to lazy subtour cuts with callbacks.

### 5.4 Priority
**High**.

---

## Problem 4. Vehicle usage is not directly penalized in the objective

### 6.1 Current code
The objective includes:
- store holding cost,
- warehouse holding cost,
- distance cost on arcs,
- LT unit cost,
- shortage cost.

But there is **no explicit fixed vehicle usage cost** like:

```python
gp.quicksum(vehicle_fixed_cost * u[(v,t)] for v in V for t in T)
```

### 6.2 Why this matters
Without an explicit vehicle-activation cost:
- solver decisions are driven mostly by arc distance and shortage penalties,
- the model may over-pack used vehicles,
- or route structure may look distorted compared with real operating logic.

### 6.3 Consequence
This is one reason why you often observe quantities close to full capacity.

That behavior is not automatically wrong, but the cost structure strongly encourages consolidation.

### 6.4 Required fix
Add a vehicle fixed cost parameter:

In `IRPData`:

```python
vehicle_fixed_cost: float = 0.0
```

In mapper:

```python
data.vehicle_fixed_cost = 50.0  # example, calibrate later
```

In the objective:

```python
+ gp.quicksum(d.vehicle_fixed_cost * u[(v, t)] for v in V for t in T)
```

### 6.5 Priority
**Medium to high**.

---

## Problem 5. Distance matrix is too synthetic and too uniform

### 7.1 Current code
The mapper creates synthetic distances:
- `CW <-> store = 10`
- `store <-> store = 6`

### 7.2 Why this is a problem
Such a flat distance matrix makes routing artificially easy:
- many route structures become nearly equivalent,
- solver runtime becomes misleadingly short,
- route quality is not operationally informative.

### 7.3 Required fix
If possible, replace the synthetic matrix with:
- real store coordinates,
- geodesic distance,
- or at least a more heterogeneous proxy matrix.

For example, if coordinates exist:

```python
from math import radians, sin, cos, sqrt, atan2
```

Then compute pairwise distances and populate `data.distance[(i,j)]` accordingly.

### 7.4 Priority
**Medium** for code correctness, **high** for thesis realism.

---

## Problem 6. Arc-capacity is linked to `u[(v,t)]` instead of only route activation

### 8.1 Current code
The code uses:

```python
gp.quicksum(q[(p, i, j, v, t)] for p in P) <= d.vehicle_capacity * u[(v, t)]
```

This is valid as an upper bound, but weak.

### 8.2 Why this is weak
If vehicle `v` is activated, then every arc for that vehicle gets the same broad upper bound, even if a specific arc is not traversed.

A tighter formulation is:

```python
gp.quicksum(q[(p, i, j, v, t)] for p in P) <= d.vehicle_capacity * x[(i, j, v, t)]
```

The file already has product-level arc linking:

```python
q[(p, i, j, v, t)] <= d.vehicle_capacity * x[(i, j, v, t)]
```

but the aggregate form should also be tightened.

### 8.3 Required fix
Replace or supplement the current aggregate capacity constraint with:

```python
for i in N0:
    for j in N0:
        if i == j:
            continue
        for v in V:
            for t in T:
                mdl.addConstr(
                    gp.quicksum(q[(p, i, j, v, t)] for p in P)
                    <= d.vehicle_capacity * x[(i, j, v, t)]
                )
```

### 8.4 Priority
**Medium**.

---

## Problem 7. Route extraction assumes a clean single next-node structure

### 9.1 Current code
The extraction helper uses:

```python
next_map = {i: j for i, j in arcs}
```

### 9.2 Why this is risky
If due to weak subtour elimination or formulation looseness a node has multiple outgoing arcs, this helper will silently overwrite keys and may produce misleading route reports.

### 9.3 Required fix
Strengthen route extraction by:
- validating out-degree and in-degree per active vehicle-period,
- warning if more than one outgoing arc exists from any node,
- printing unresolved route fragments separately.

### 9.4 Priority
**Medium** for debugging, low for formulation itself.

---

## 4. Suggested code changes by priority

## Priority 1 — must fix now

### A. Introduce explicit delivery variables
Add:

```python
deliv[(s,p,v,t)]
```

Use this instead of tying `Qdir` only to `q[(p,CW,s,v,t)]`.

### B. Add vehicle load progression
Add:

```python
load[(i,v,t)]
```

Track remaining truck load through the route.

### C. Rebuild direct-delivery flow logic
The vehicle should depart from CW with a load, then unload at stores along the route.

---

## Priority 2 — strongly recommended

### D. Add subtour elimination
Use MTZ first because it is easier to code.

### E. Tighten arc-capacity constraints
Use `x[(i,j,v,t)]` instead of only `u[(v,t)]` for tighter linking.

### F. Add vehicle fixed cost
This improves realism and helps avoid distorted consolidation patterns.

---

## Priority 3 — improve realism and debugging

### G. Replace synthetic distances
Use real or at least more heterogeneous distances.

### H. Improve route extraction diagnostics
Detect fragmented or ambiguous routes.

### I. Add sanity-check reports after solve
For each vehicle-period, export:
- total load leaving CW,
- total delivered quantity,
- remaining load after each visited node,
- degree in/out by node,
- whether a clean cycle exists.

---

## 5. Recommended revised baseline architecture

If you want the baseline to become a thesis-grade routing formulation, the clean structure should be:

### Inventory variables
- `I_s[(s,p,t)]`
- `I_w[(p,t)]`
- `B[(s,p,t)]`

### Routing variables
- `x[(i,j,v,t)]`
- `u[(v,t)]`
- optional `z[(i,v,t)]`
- subtour/order variables if MTZ is used

### Delivery variables
- `deliv[(s,p,v,t)]`

### Vehicle load variables
- `load[(i,v,t)]`

### LT variables
If baseline is run without LT, keep LT shut off exactly as you already do:

```python
allow_lateral_transshipment=False
```

That part is fine for a pure baseline run.

---

## 6. Why the current solver runtime may look too fast

The model may solve fast because of all of the following together:
- synthetic and simple distances,
- incomplete subtour handling,
- no full delivery-on-route logic,
- no full load-propagation structure,
- continuous product flows by default.

So fast runtime should **not** be interpreted as proof that the routing baseline is already correct.

---

## 7. Why vehicles often appear close to full capacity

This can happen because:
1. shortage cost encourages large replenishment;
2. route cost is mostly distance-based, encouraging consolidation;
3. no explicit fixed vehicle cost calibration exists yet;
4. delivery logic is still simplified;
5. truck load does not decrease along route in an explicitly modeled way.

Therefore, the “full-capacity-looking” result is not enough to conclude the routing is correct.
It may actually be a symptom that the current formulation is still too loose or too stylized.

---

## 8. Practical patch plan for your code

## Patch block 1 — new variables
Inside `AchamrahFullIRPTModel.solve(...)`, add:

```python
deliv_keys = [(s, p, v, t) for s in N for p in P for v in V for t in T]
load_keys = [(i, v, t) for i in N0 for v in V for t in T]


deliv = mdl.addVars(deliv_keys, lb=0.0, vtype=flow_vtype, name="deliv")
load = mdl.addVars(load_keys, lb=0.0, ub=d.vehicle_capacity, vtype=flow_vtype, name="load")
```

---

## Patch block 2 — redefine direct shipment
Replace:

```python
mdl.addConstr(Qdir[(s, p, t)] == gp.quicksum(q[(p, CW, s, v, t)] for v in V))
```

with:

```python
mdl.addConstr(
    Qdir[(s, p, t)] == gp.quicksum(deliv[(s, p, v, t)] for v in V)
)
```

---

## Patch block 3 — vehicle initial load at depot
Add:

```python
for v in V:
    for t in T:
        mdl.addConstr(
            load[(CW, v, t)] == gp.quicksum(deliv[(s, p, v, t)] for s in N for p in P)
        )
        mdl.addConstr(load[(CW, v, t)] <= d.vehicle_capacity * u[(v, t)])
```

This is the simplest first version.

---

## Patch block 4 — load progression after each visited node
For each traveled arc, propagate truck load after delivery.

This part needs careful coding with big-M and node visit logic. Start simple and debug on small instances.

---

## Patch block 5 — MTZ constraints
Add ordering variables and subtour elimination constraints.

---

## Patch block 6 — objective update
Add:

```python
+ gp.quicksum(d.vehicle_fixed_cost * u[(v, t)] for v in V for t in T)
```

---

## Patch block 7 — diagnostic export
After solving, export for each vehicle-period:
- total route distance,
- total delivered quantity,
- total load leaving CW,
- route sequence,
- any degree violations,
- any disconnected cycles detected.

---

## 9. Best interpretation for thesis writing

If you describe the current code in the thesis, the most accurate statement is:

> The current baseline already integrates inventory balance with a vehicle-routing skeleton, including depot departure, arc decisions, and vehicle-capacity linkage. However, it does not yet represent a fully detailed multi-stop delivery routing model because direct deliveries are still tied to depot-to-store arcs, explicit truck load propagation is absent, and complete subtour elimination is not yet enforced.

That wording is accurate and defensible.

---

## 10. Recommended next action

The best next coding step is:

1. fix direct delivery logic;
2. add truck load progression;
3. add MTZ subtour elimination;
4. then validate again on a very small instance and inspect routes manually.

Only after these are stable should you treat the baseline as a true routing baseline for comparison with LT / CG / GNN layers.
