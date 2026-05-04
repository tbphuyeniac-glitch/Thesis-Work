# Chapter 4 — Mathematical Models Reference
## IRP with Sharing Framework and Lateral Transshipment

---

> **Structural note.** Section 4.1 presents the **full integrated mathematical model**
> covering both DC-to-store replenishment and lateral transshipment. The ALNS (Part 2)
> solves the Stage 1 sub-problem by fixing all LT variables to zero ($y = 0$). The
> Column Generation module (Part 3) then handles Stage 2 LT decisions. Constraints
> referencing $y_{ijpvt}$ — specifically constraints (2), (8), and (15) — belong to the
> full integrated model and are trivially satisfied (but structurally inactive) during
> the ALNS baseline solve.

---

## Part 1 — Full Integrated Mathematical Model

### 1.1 Indices and Sets

| Symbol | Description |
|---|---|
| $N$ | Set of retail stores (Points of Sale) |
| $P$ | Set of products / SKUs |
| $T$ | Set of planning periods |
| $V$ | Set of vehicles |
| $CW$ | Central warehouse node |
| $N_0 = \{CW\} \cup N$ | All nodes in the network |
| $i, j \in N_0$ | Generic network nodes |
| $s \in N$ | A retail store |
| $p \in P$ | A product / SKU |
| $t \in T$ | A planning period |
| $v \in V$ | A vehicle |

---

### 1.2 Parameters

| Symbol | Description |
|---|---|
| $d_{spt}$ | Demand of product $p$ at store $s$ in period $t$ |
| $I^0_{sp}$ | Initial inventory of product $p$ at store $s$ |
| $I^{0W}_{p}$ | Initial inventory of product $p$ at warehouse |
| $R_{pt}$ | Replenishment received by warehouse for product $p$ in period $t$ |
| $C_s$ | Aggregate storage capacity of store $s$ |
| $C_{CW}$ | Aggregate storage capacity of central warehouse |
| $Q_V$ | Vehicle capacity |
| $U_{max}$ | Maximum number of vehicles usable per period |
| $h_{sp}$ | Holding cost of product $p$ at store $s$ |
| $h^W_p$ | Holding cost of product $p$ at warehouse |
| $\pi_{sp}$ | Shortage penalty cost of product $p$ at store $s$ |
| $c^{ship}_{sp}$ | Unit shipping cost of product $p$ from CW to store $s$ |
| $dist_{ij}$ | Routing distance / cost base from node $i$ to node $j$ |
| $\alpha$ | Route-cost multiplier (transportation unit cost) |
| $b_{ij}$ | Lateral transshipment unit cost from store $i$ to store $j$ |
| $f_{ij}$ | Fixed dispatch cost for activating a lateral transshipment from store $i$ to store $j$ (enters via column cost in Part 3) |
| $F_v$ | Fixed cost for activating vehicle $v$ in a period |

---

### 1.3 Decision Variables

| Symbol | Type | Description |
|---|---|---|
| $I_{spt}$ | $\geq 0$ | Ending inventory of product $p$ at store $s$ in period $t$ |
| $I^W_{pt}$ | $\geq 0$ | Ending inventory of product $p$ at warehouse in period $t$ |
| $Q^{dir}_{spt}$ | $\geq 0$ | Direct shipment quantity of product $p$ from CW to store $s$ in period $t$ |
| $B_{spt}$ | $\geq 0$ | Shortage / backorder at store $s$, product $p$, period $t$ |
| $q_{pijvt}$ | $\geq 0$ | Quantity of product $p$ flowing on arc $(i,j)$ by vehicle $v$ in period $t$ |
| $y_{ijpvt}$ | $\geq 0$ | LT quantity of product $p$ from store $i$ to store $j$ by vehicle $v$ in period $t$, $i,j \in N$, $i \neq j$ *(Stage 2 only; fixed to 0 in Stage 1 baseline)* |
| $deliv_{spvt}$ | $\geq 0$ | Delivery quantity of product $p$ by vehicle $v$ to store $s$ in period $t$ |
| $load_{ivt}$ | $\geq 0$ | Cumulative load on vehicle $v$ at node $i$ in period $t$ (MTZ) |
| $ord_{svt}$ | $\geq 0$ | Visit-order position of store $s$ on vehicle $v$ in period $t$ (MTZ) |
| $x_{ijvt}$ | $\in \{0,1\}$ | 1 if vehicle $v$ uses arc $(i,j)$ in period $t$ |
| $u_{vt}$ | $\in \{0,1\}$ | 1 if vehicle $v$ is used in period $t$ |
| $z_{ivt}$ | $\in \{0,1\}$ | 1 if node $i$ is visited by vehicle $v$ in period $t$ |

**Derived visit indicators:**

$$z_{CW,vt} = u_{vt}$$

$$z_{ivt} = \sum_{j \in N_0,\, j \neq i} x_{jivt}, \quad \forall i \in N$$

---

### 1.4 Objective Function

Minimize total system cost over all periods:

$$\min Z = \sum_{s,p,t} c^{ship}_{sp}\, Q^{dir}_{spt}
+ \sum_{s,p,t} h_{sp}\, I_{spt}
+ \sum_{p,t} h^W_p\, I^W_{pt}
+ \sum_{i,j,v,t} \alpha\, dist_{ij}\, x_{ijvt}
+ \sum_{v,t} F_v\, u_{vt}
+ \sum_{i,j,p,v,t} b_{ij}\, y_{ijpvt}
+ \sum_{s,p,t} \pi_{sp}\, B_{spt}
\tag{1}$$

The seven cost components are: (1) direct CW-to-store unit shipping cost, (2) store
inventory holding cost, (3) warehouse holding cost, (4) vehicle travel cost,
(5) vehicle fixed activation cost, (6) lateral transshipment unit cost, (7) shortage penalty.

> **Note (Stage 1 / ALNS).** All $y$ variables are fixed to zero in the ALNS baseline,
> so the LT term $\sum b_{ij} y_{ijpvt}$ vanishes. The ALNS objective therefore evaluates
> only components (1)–(5) and (7).
>
> **Note (fixed LT activation cost).** Equation (1) deliberately omits the
> non-linear fixed-charge term $\sum_{(i,j)} f_{ij} \cdot \mathbb{1}\!\bigl[\sum_{p,v,t} y_{ijpvt} > 0\bigr]$,
> which would be required for an exact monolithic MIP. In this thesis the integrated
> Gurobi reference model (`AchamrahFullIRPTModel`) implements only the variable LT cost
> $b_{ij} y_{ijpvt}$, while the fixed activation cost $f_{ij}$ is **linearised inside
> each LT column's cost coefficient $cost_c$ in Part 3** (see §3.2). When CG selects
> column $c$ with $\lambda_c > 0$, every donor-receiver arc $(i,j)$ that the column
> uses incurs $f_{ij}$ exactly once via $cost_c \cdot \lambda_c$ in the RMP objective.

---

### 1.5 Constraints

**Constraint (2) — Store inventory balance.** Ending inventory at each store is determined
by prior inventory, direct replenishment, net LT flows, demand, and shortage:

$$I_{spt} = I_{sp,t-1} + Q^{dir}_{spt} - d_{spt} + B_{spt}
+ \sum_{\substack{j \in N \\ j \neq s}} \sum_{v \in V} y_{jspvt}
- \sum_{\substack{j \in N \\ j \neq s}} \sum_{v \in V} y_{sjpvt}
\tag{2}$$

> In Stage 1, the two LT summation terms are zero, simplifying to the standard
> balance: $I_{spt} = I_{sp,t-1} + Q^{dir}_{spt} - d_{spt} + B_{spt}$.

**Constraint (3) — Warehouse inventory balance:**

$$I^W_{pt} = I^W_{p,t-1} - \sum_{s \in N} Q^{dir}_{spt} + R_{pt}
\tag{3}$$

**Constraints (4a–4b) — Flow consistency.** Net delivered quantity to each store
matches net product flow on the transportation network:

$$Q^{dir}_{spt} = \sum_{v \in V} deliv_{spvt}, \quad \forall s, p, t
\tag{4a}$$

$$deliv_{spvt} + \sum_{\substack{i \in N \\ i \neq s}} y_{ispvt}
- \sum_{\substack{j \in N \\ j \neq s}} y_{sjpvt}
= \sum_{\substack{i \in N_0 \\ i \neq s}} q_{pisvt}
- \sum_{\substack{j \in N_0 \\ j \neq s}} q_{psjvt},
\quad \forall s, p, v, t
\tag{4b}$$

**Constraint (5) — Empty-return condition.** Vehicles carry no product back to CW:

$$\sum_{p \in P} q_{p,i,CW,v,t} = 0
\tag{5}$$

**Constraint (6) — Storage capacity limits:**

$$\sum_{p \in P} I_{spt} \leq C_s; \qquad \sum_{p \in P} I^W_{pt} \leq C_{CW}
\tag{6}$$

**Constraint (7) — Vehicle capacity on arc:**

$$\sum_{p \in P} q_{pijvt} \leq Q_V\, x_{ijvt}
\tag{7}$$

**Constraint (8) — Inventory feasibility of outbound LT.** *(Full model only — trivially
satisfied when $y = 0$ in Stage 1.)* A store cannot transship more than its
beginning-of-period inventory $I^{beg}_{sp,t}$, defined as the inventory at the start
of period $t$:

$$I^{beg}_{sp,t} \;=\;
\begin{cases}
I^0_{sp} & \text{if } t = 1, \\
I_{sp,\,t-1} & \text{otherwise}
\end{cases}$$

$$\sum_{\substack{j \in N \\ j \neq s}} \sum_{v \in V} y_{sjpvt} \leq I^{beg}_{sp,t}
\tag{8}$$

**Constraint (9) — Route continuity at stores** (CW flow balance is handled by Eq. (11)):

$$\sum_{\substack{i \in N_0 \\ i \neq j}} x_{ijvt} = \sum_{\substack{i \in N_0 \\ i \neq j}} x_{jivt},
\quad \forall j \in N,\; v, t
\tag{9}$$

**Constraint (10) — Single-visit limit per store per period:**

$$\sum_{v \in V} \sum_{\substack{i \in N_0 \\ i \neq j}} x_{ijvt} \leq 1
\tag{10}$$

**Constraint (11) — Vehicle activation definition:**

$$\sum_{j \in N} x_{CW,j,v,t} = u_{vt}
\tag{11}$$

**Constraint (12) — Fleet-size limit:**

$$\sum_{v \in V} u_{vt} \leq U_{max}
\tag{12}$$

**Constraint (13) — Product flow only on used arcs:**

$$q_{pijvt} \leq Q_V\, x_{ijvt}
\tag{13}$$

**Constraint (14) — Direct-shipment definition** *(derived: implied by (4a)+(4b)+(5); not added explicitly to the Gurobi model):*

$$Q^{dir}_{spt} = \sum_{v \in V} q_{p,CW,s,v,t}
\tag{14}$$

**Constraint (15) — LT-linking constraint.** *(Full model only — trivially satisfied when
$y = 0$ in Stage 1.)* LT quantity cannot exceed the physical product flow on the
corresponding arc:

$$y_{ijpvt} \leq q_{pijvt}
\tag{15}$$

**Constraints (16)–(18) — Arc usage and node-visit consistency (valid inequalities):**

$$x_{CW,i,v,t} \leq z_{ivt}, \quad \forall i \in N,\; v, t \tag{16}$$

$$x_{ijvt} \leq z_{jvt}, \quad \forall i, j \in N,\; i \neq j,\; v, t \tag{17}$$

$$z_{ivt} \leq z_{CW,vt}, \quad \forall i \in N,\; v, t \tag{18}$$

**Constraint (19) — Vehicle symmetry breaking:**

$$z_{CW,v,t} \leq z_{CW,v-1,t} \tag{19}$$

**Constraint (20) — Inventory-driven visit validity.** Cumulative demand coverage is
linked with store visits and incoming transshipment over any window $[t_1, t_2]$:

$$\sum_{v \in V} \sum_{\tau=t_1}^{t_2} z_{sv\tau}
+ \frac{1}{\sum_{\tau=t_1}^{t_2} d_{sp\tau}}
  \sum_{\substack{j \in N \\ j \neq s}} \sum_{v \in V} \sum_{\tau=t_1}^{t_2} y_{jspv\tau}
\;\geq\;
\frac{\displaystyle\sum_{\tau=t_1}^{t_2} d_{sp\tau} - I^{base}_{sp,t_1}}
     {\displaystyle\sum_{\tau=t_1}^{t_2} d_{sp\tau}}
\tag{20}$$

where $I^{base}_{sp,t_1}$ denotes the inventory at the *beginning* of period $t_1$
(i.e., $I_{sp,t_1-1}$, with the initial parameter $I^0_{sp}$ used when $t_1$ is the first period).
The LT source set is $j \in N$ because $y$ is defined for store-to-store transshipment only.

---

### 1.6 Subtour Elimination — Load Tracking (MTZ-style)

**Departure load at the depot:**

$$load_{CW,v,t} = \sum_{s \in N} \sum_{p \in P} deliv_{spvt} \tag{VI-a}$$

**Depot-load capacity link** (only carry units when the vehicle is used):

$$load_{CW,v,t} \leq Q_V \cdot u_{vt} \tag{VI-a$'$}$$

**Load propagation along used arcs** ($\forall i \in N_0,\, j \in N,\, i \neq j,\, v, t$):

$$load_{j,v,t} \leq load_{i,v,t} - \sum_p deliv_{jpvt} + Q_V(1 - x_{ijvt}) \tag{VI-b}$$

$$load_{j,v,t} \geq load_{i,v,t} - \sum_p deliv_{jpvt} - Q_V(1 - x_{ijvt}) \tag{VI-c}$$

**Store-load activity bound** (load at $s$ is zero unless the vehicle leaves $s$):

$$load_{s,v,t} \leq Q_V \sum_{\substack{j \in N_0 \\ j \neq s}} x_{sjvt}, \qquad \forall s \in N,\; v, t \tag{VI-c$'$}$$

**Outflow lower bound on load** (load at $i$ must cover all outgoing product flow):

$$load_{i,v,t} \geq \sum_{p \in P} \sum_{\substack{j \in N_0 \\ j \neq i}} q_{pijvt}, \qquad \forall i \in N,\; v, t \tag{VI-c$''$}$$

**Visit-order bounds** ($\forall s \in N,\, v, t$):

$$z_{svt} \leq ord_{svt} \leq |N| \cdot z_{svt} \tag{VI-d}$$

**Order-based MTZ subtour elimination among stores:**

$$ord_{ivt} - ord_{jvt} + |N| \cdot x_{ijvt} \leq |N| - 1, \qquad \forall\, i \neq j \in N,\; v, t \tag{VI-e}$$

---

## Part 2 — Stage 1 Solution: Adaptive Large Neighborhood Search (ALNS)

The ALNS solves the baseline DC-to-stores problem with all $y_{ijpvt} = 0$.

### 2.1 Solution State Representation

$$\mathcal{S} = (deliv,\; routes)$$

where $deliv_{spvt}$ records the delivery quantity of product $p$ by vehicle $v$ to store
$s$ in period $t$, and $routes_{vt}$ records the ordered sequence of stores visited by
vehicle $v$ in period $t$.

### 2.2 ALNS Objective Function

$$f(\mathcal{S}) = \sum_{s,p,t} c^{ship}_{sp}\, Q^{dir}_{spt}(\mathcal{S})
+ \sum_{s,p,t} h_{sp}\, I_{spt}(\mathcal{S})
+ \sum_{p,t} h^W_p\, I^W_{pt}(\mathcal{S})
+ \sum_{i,j,v,t} \alpha\, dist_{ij}\, x_{ijvt}(\mathcal{S})
+ \sum_{v,t} F_v\, u_{vt}(\mathcal{S})
+ \sum_{s,p,t} \pi_{sp}\, B_{spt}(\mathcal{S})$$

All variables are derived by forward-simulating the inventory trajectory from state $\mathcal{S}$.

---

### 2.3 Destroy Operators

At each iteration, $k \sim \text{Uniform}(0.05|\mathcal{S}|,\; 0.25|\mathcal{S}|)$ delivery decisions are removed.

**D1 — Random Delivery Removal.** Selects $k$ delivery decisions $(s, p, v, t)$ uniformly at random.

**D2 — Worst Delivery Removal.** Removes the $k$ decisions with the highest cost-per-demand ratio:

$$\text{score}(s,p,v,t) = \frac{c^{ship}_{sp} \cdot deliv_{spvt}}{d_{spt} + 1}$$

**D3 — Random Route Removal.** Randomly selects $\lfloor k/4 \rfloor$ vehicle-period routes and removes all deliveries on those routes.

**D4 — Random Period Removal.** Selects a period $t^*$ uniformly at random and removes all delivery decisions for that period.

**D5 — Shaw Store Removal.** Selects a seed store and removes the $\lfloor k/3 \rfloor$ most geographically proximate stores in that period based on:

$$Similarity_{ij} = dist_{ij}$$

**D6 — Low Demand Store Removal.** Removes the $\lfloor k/2 \rfloor$ delivery decisions with the lowest local demand value $d_{spt}$.

---

### 2.4 Repair Operators

An **unserved request** is a $(s, p, t)$ triple where ending inventory would fall below
zero under forward simulation of the current partial state.

**Insertion cost** of adding delivery of product $p$ to store $s$ in period $t$ via vehicle $v$:

$$\Delta f(s, p, t, v) = c^{ship}_{sp} \cdot qty + \alpha \cdot \Delta_{route}(s, t, v) + F_v \cdot \mathbb{1}[\text{new route}]$$

where the **minimum arc detour cost** of inserting store $s$ into existing route
$[s_0, s_1, \ldots, s_m]$ (with $s_0 = s_{m+1} = CW$) is:

$$\Delta_{route}(s, t, v) = \min_{0 \leq k \leq m}\;\bigl[ dist(s_k,\, s) + dist(s,\, s_{k+1}) - dist(s_k,\, s_{k+1}) \bigr]$$

If $s$ is already on vehicle $v$'s route in period $t$, no detour is incurred:
$\Delta_{route}(s, t, v) = 0$ and the new-route indicator is $0$ — the insertion cost reduces to
$\Delta f = c^{ship}_{sp} \cdot qty$ (additional units piggyback on the existing visit).

A request is infeasible for vehicle $v$ if inserting it would violate $Q_V$, or if $s$
is already visited by another vehicle in period $t$ (single-visit-per-period rule from Eq. (10)).

**R1 — Greedy Insertion.** For each unserved request (in random order):

$$v^* = \arg\min_{v \in V} \Delta f(s, p, t, v)$$

**R2 — Regret-2 Insertion.** The request with the highest regret is inserted first:

$$regret(s, p, t) = \Delta f^{(2)}(s, p, t) - \Delta f^{(1)}(s, p, t)$$

**R3 — Random Insertion.** Randomly selects any feasible vehicle and inserts.

---

### 2.5 Adaptive Weight Update

Operator weights are updated every $\eta$ iterations using reaction factor $\rho \in (0,1)$:

$$w_d \leftarrow (1 - \rho)\, w_d + \rho \cdot \frac{\pi_d}{\theta_d}; \qquad
w_r \leftarrow (1 - \rho)\, w_r + \rho \cdot \frac{\pi_r}{\theta_r}$$

where $\pi_d$, $\pi_r$ accumulate reward scores ($\sigma_1$ for new global best,
$\sigma_2$ for improvement, $\sigma_3$ for accepted non-improving) and $\theta_d$, $\theta_r$
count operator usage in the segment. Operators are selected by roulette wheel;
candidates are accepted via Simulated Annealing cooling $T \leftarrow T \cdot \gamma$.

---

## Part 3 — Stage 2: Column Generation for Lateral Transshipment

### 3.1 Post-Shock State Definitions

After the baseline plan is executed, realized demand deviates from forecast. Let
$\hat{I}^{post}_{s,p,t}$ denote the post-shock ending inventory at store $s$ for product $p$
in period $t$, and $\hat{B}^{post}_{s,p,t}$ the corresponding post-shock shortage. The CG
module uses *windowed* coverage and reserve targets rather than single-period demand:

**Demand-cover horizon** ($L = $ `need_lookahead_periods`):

$$D^{cover}_{s,p,t} = \sum_{\tau=t}^{t+L-1} d^{real}_{s,p,\tau}$$

**Reserve-stock horizon** ($R = $ `surplus_reserve_periods`, with safety-stock floor $S_0$):

$$D^{reserve}_{s,p,t} = \max\!\left(S_0,\; \sum_{\tau=t}^{t+R-1} d^{real}_{s,p,\tau}\right)$$

**Need and surplus proxies:**

$$need_{s,p,t} = \max\!\left(\hat{B}^{post}_{s,p,t},\; D^{cover}_{s,p,t} - \hat{I}^{post}_{s,p,t}\right)$$

$$surplus_{s,p,t} = \max\!\left(0,\; \hat{I}^{post}_{s,p,t} - D^{reserve}_{s,p,t}\right)$$

The need formula is floored at the realized post-shock shortage so that any current
stockout is always reflected, regardless of the lookahead window.

A **(product, period)** pair is **active for LT** if:

$$\sum_{s \in N} need_{s,p,t} \geq \theta_{LT} \quad \text{and} \quad
\sum_{s \in N} surplus_{s,p,t} \geq \theta_{LT}$$

---

### 3.2 LT Column Definition

A column $c$ for product $p$ in period $t$ is a set of donor-receiver arc flows:

$$c = \bigl\{ (i,j) \mapsto q^c_{ij} \;\big|\; i \in \text{donors},\; j \in \text{receivers},\; i \neq j \bigr\}$$

$$cost_c = \sum_{(i,j) \in c} f_{ij} \cdot \mathbb{1}[q^c_{ij} > 0] + \sum_{(i,j) \in c} b_{ij} \cdot q^c_{ij}$$

---

### 3.3 Restricted Master Problem (RMP)

$$\min_{\lambda,\, r} \quad Z^{base} + \sum_{c \in \mathcal{C}} cost_c \cdot \lambda_c
+ \sum_{s,p,t} \pi_{s,p,t} \cdot r_{s,p,t}
\tag{RMP}$$

**Need-cover constraints** (dual variable $\mu_{s,p,t} \geq 0$):

$$r_{s,p,t} + \sum_{c \in \mathcal{C}} a^{need}_{c,s,p,t} \cdot \lambda_c \;\geq\; need_{s,p,t},
\quad \forall\, s, p, t
\tag{RMP-1}$$

**Surplus-capacity constraints** (dual variable $\nu_{s,p,t} \geq 0$):

$$\sum_{c \in \mathcal{C}} a^{surplus}_{c,s,p,t} \cdot \lambda_c \;\leq\; surplus_{s,p,t},
\quad \forall\, s, p, t
\tag{RMP-2}$$

$$\lambda_c \in [0, 1], \quad r_{s,p,t} \geq 0$$

where $a^{need}_{c,s,p,t}$ is the total inflow to receiver $s$ under column $c$, and
$a^{surplus}_{c,s,p,t}$ is the total outflow from donor $s$ under column $c$.

---

### 3.4 Pricing Problem and Reduced Cost

Let $\mu_{s,p,t} \geq 0$ denote the dual of (RMP-1) and $\nu_{s,p,t} \geq 0$ the dual of
(RMP-2), both expressed as non-negative magnitudes (the Gurobi `con.Pi` value for the
$\leq$ surplus-cap constraint is non-positive; we take its absolute value as $\nu$).

For a donor-receiver pair $(i,j)$ transferring quantity $q$ of product $p$ in period $t$,
the per-pair reduced cost contribution is:

$$\bar{c}_{ij} = f_{ij} + \bigl(b_{ij} - \mu_{j,p,t} - \nu_{i,p,t}\bigr) \cdot q$$

The reduced cost of the full column $c$ is the sum of pair contributions, equivalently:

$$\bar{c}(c) = cost_c
- \sum_{j:\,\text{receiver}} \mu_{j,p,t} \cdot q^c_{ij}
- \sum_{i:\,\text{donor}} \nu_{i,p,t} \cdot q^c_{ij}$$

A column is added to the RMP if $\bar{c}(c) < 0$. CG converges when no such column exists
across all active $(p,t)$ pairs. The exact pricing MIP solved per active $(p,t)$ pair is:

$$\min_{q,y}\; \sum_{(i,j)} \Bigl[ f_{ij}\, y_{ij} + (b_{ij} - \mu_{j,p,t} - \nu_{i,p,t})\, q_{ij} \Bigr]$$
subject to donor capacity $\sum_j q_{ij} \leq surplus_{i,p,t}$, receiver capacity
$\sum_i q_{ij} \leq need_{j,p,t}$, fixed-charge link $q_{ij} \leq \min(surplus_i, need_j)\, y_{ij}$,
and a per-pattern arc cap $\sum_{(i,j)} y_{ij} \leq K^{max}$.

---

### 3.5 Feature-Based Candidate Pruning

> **Correction note (mismatch with original report).** Feature $\phi_2$ below reflects
> the code implementation, which defines donor surplus strength as the donor's *share
> of total system surplus*. The original report formula (*remaining surplus after
> transfer / transfer quantity*) is inconsistent with the code and should be updated
> in the thesis text to match.

**Feature 1 — Demand–inventory imbalance severity (shortage ratio):**

$$\phi_1(c) = \frac{need_{j,p,t}}{\displaystyle\sum_{s \in N} need_{s,p,t}}$$

**Feature 2 — Donor surplus strength (surplus ratio):** *(corrected to match code)*

$$\phi_2(c) = \frac{surplus_{i,p,t}}{\displaystyle\sum_{s \in N} surplus_{s,p,t}}$$

**Feature 3 — Time urgency:**

$$\phi_3(c) = \frac{1}{1 + \text{days until stockout at receiver}}$$

**Feature 4 — Negative reduced cost (economic attractiveness):**

$$\phi_4(c) = \max\!\bigl(0,\; -\bar{c}(c)\bigr)$$

---

### 3.6 Stackelberg Acceptance Filter

**Donor (Follower $i$) — required minimum compensation:**

$$comp^*_i = \theta_{donor}
+ w^{risk} \cdot \frac{q}{surplus_{i,p,t}}
+ w^{burden} \cdot (b_{ij} \cdot q + f_{ij})
+ w^{svc} \cdot \frac{q}{surplus_{i,p,t} + 1}$$

**Donor local utility:**

$$U_i = compensation - \left(
w^{risk} \cdot \frac{q}{surplus_{i,p,t}}
+ w^{burden} \cdot (b_{ij} \cdot q + f_{ij})
+ w^{svc} \cdot \frac{q}{surplus_{i,p,t} + 1}
\right)$$

**Receiver (Follower $j$) — maximum acceptable compensation:** the receiver's handling
cost is internally scaled by $\tfrac{1}{4}$ of the per-unit ship cost:

$$comp^{max}_j =
w^{shortage} \cdot \frac{\min(q,\, need_{j,p,t})}{need_{j,p,t}}
+ w^{svc\_gain} \cdot \frac{\min(q,\, need_{j,p,t})}{need_{j,p,t}+1}
- w^{handling} \cdot \tfrac{1}{4} \cdot b_{ij} \cdot q
- \theta_{receiver}$$

**Compensation chosen by the leader** (donor's required price, floored at $\underline{c}$):

$$compensation = \max\!\bigl(\underline{c},\; comp^*_i\bigr)$$

**Acceptance condition:**

$$comp^*_i \leq comp^{max}_j \quad \text{and} \quad compensation \leq cap_{comp}$$

**Acceptance score** (equally-weighted sigmoid of both utilities; set to $0$ if rejected):

$$score_{accept}(i,j) = \tfrac{1}{2}\,\sigma(U_i) + \tfrac{1}{2}\,\sigma(U_j)$$

where $\sigma(\cdot)$ is the sigmoid function.

---

### 3.7 BiGAT-Guided Column Ranking

#### 3.7.1 Bipartite Graph Construction

$$G = (V_c,\; V_r,\; E)$$

**Column node feature vector $x^{col}_i \in \mathbb{R}^{10}$** *(corrected: 10 features,
no `acceptance_score` while Stackelberg is disabled)*:

| Index | Feature |
|---|---|
| 0 | Reduced cost $\bar{c}(c)$ |
| 1 | Total transfer quantity |
| 2 | Number of donor-receiver pairs |
| 3 | Total need covered |
| 4 | Total surplus consumed |
| 5 | Avg. shortage ratio $\phi_1$ across pairs |
| 6 | Avg. surplus ratio $\phi_2$ across pairs |
| 7 | Avg. time urgency $\phi_3$ across pairs |
| 8 | Avg. negative reduced cost $\phi_4$ across pairs |
| 9 | Total column cost $cost_c$ |

> When Stackelberg is re-enabled, `acceptance_score` and `compensation_mean` are
> restored as indices 9–10, and `column_cost` shifts from its current position
> (index 9 in the 10-feature schema) to index 11 — returning the schema to 12 features
> indexed 0–11.

**Constraint node feature vector $x^{con}_j \in \mathbb{R}^{7}$** *(corrected: 7 features)*:

| Index | Feature |
|---|---|
| 0 | Dual value $\mu_j$ or $\nu_j$ |
| 1 | RHS amount (need or surplus) |
| 2 | Covered amount |
| 3 | Residual slack |
| 4 | Urgency or scarcity |
| 5 | $\mathbb{1}[\text{need-cover constraint}]$ |
| 6 | $\mathbb{1}[\text{surplus-cap constraint}]$ |

**Edge feature vector $a_{ij} \in \mathbb{R}^{3}$:**

| Index | Feature |
|---|---|
| 0 | Transfer quantity relevant to this constraint |
| 1 | Fraction of need covered or surplus consumed |
| 2 | $dual_j \times quantity$ (economic weight) |

#### 3.7.2 BiGAT Message Passing

Each directed pass uses three projections of dimension $d$: a source projection $W_c$,
a destination projection $W_r$, and an edge projection $W_e$. The destination embedding
is updated by **concatenating** (not adding) the original destination state with the
attention-weighted aggregate of edge-augmented messages, followed by a linear+ReLU head
$W_o$. A residual connection and LayerNorm wrap each pass.

**Raw attention score** for column-constraint pair $(i,j)$:

$$e_{ij} = \text{LeakyReLU}\!\left(
\mathbf{a}^\top \left[ W_c h_i \;\|\; W_r h_j \;\|\; W_e a_{ij} \right]
\right)$$

**Normalized attention weights** over the destination's neighborhood:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\displaystyle\sum_{k \in \mathcal{N}(j)} \exp(e_{kj})}$$

**Edge-augmented message** sent from source $i$ to destination $j$:

$$m_{i \to j} = W_c h_i + W_e a_{ij}$$

**Layer 1 — Update constraint embeddings** (column $\to$ constraint pass):

$$\tilde{h}_j = \text{ReLU}\!\left( W_o^{\text{c}\to\text{r}}\!\left[\, h_j \;\Big\|\; \sum_{i \in \mathcal{N}(j)} \alpha_{ij}\, m_{i \to j}\, \right] \right),
\qquad h'_j = \text{LayerNorm}\!\bigl( h_j + \tilde{h}_j \bigr)$$

**Layer 1 — Update column embeddings** (constraint $\to$ column pass), using a
symmetric layer with its own projections and weights $\beta_{ij}$:

$$\tilde{h}_i = \text{ReLU}\!\left( W_o^{\text{r}\to\text{c}}\!\left[\, h_i \;\Big\|\; \sum_{j \in \mathcal{N}(i)} \beta_{ij}\, \bigl(W_r h'_j + W_e a_{ij}\bigr)\, \right] \right),
\qquad h'_i = \text{LayerNorm}\!\bigl( h_i + \tilde{h}_i \bigr)$$

Two such layer-pairs are stacked. The final embedding $h^{(L)}_i$ encodes both the
column's local features and its interaction with the current RMP constraint state.

#### 3.7.3 Scoring and Top-k Selection

$$s_i = \phi\!\left(h^{(L)}_i\right)$$

where $\phi$ is a two-layer MLP scoring head with ReLU activation. Columns are sorted
descending by $s_i$ and the top-$k$ subset $\mathcal{S}$ is selected adaptively
(cumulative score-mass, relative threshold, adaptive gap cut, or fixed top-$k$).

---

### 3.8 Full Column Generation Algorithm

```
Input:  IRPData, baseline solution, demand shock state
Output: Selected LT columns, final RMP solution, LT plan

Step 1:  Solve Stage 1 via ALNS (y = 0) → baseline inventory trajectories
Step 2:  Apply demand shock → compute need_{s,p,t} and surplus_{s,p,t}
Step 3:  Initialize RMP with warm-start LT patterns
Step 4:  Solve RMP → obtain λ, dual values μ, ν

Repeat:
  Step 5:  Identify active (p,t) pairs [need > θ_LT and surplus > θ_LT]
  Step 6:  For each active (p,t):
             a. Enumerate donor-receiver pairs
             b. Compute features φ1–φ4 for each pair
             c. Apply feature-range pruning → retain promising pairs
             d. Apply Stackelberg acceptance filter → accepted pairs only
             e. Construct LT patterns with reduced cost c̄(c) < 0
  Step 7:  Build bipartite graph G = (V_c, V_r, E)
  Step 8:  Run BiGAT message passing (2 layers, bidirectional)
  Step 9:  Score and rank candidate columns → s_i = φ(h_i^(L))
  Step 10: Select top-k columns
  Step 11: Add selected columns to RMP pool
  Step 12: Re-solve RMP → update λ, μ, ν
Until:  no column with c̄(c) < 0 found, or max iterations reached

Step 13: Post-process → build LT plan from selected columns
Step 14: Compute realized operating cost breakdown
```

---

*End of Chapter 4 Mathematical Models Reference*
