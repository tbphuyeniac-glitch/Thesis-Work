# Achamrah Integrated Benchmark Implementation

## Overview

Two new files created for thesis-comparable Achamrah benchmark with two integrated modes:

### Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `achamrah_integrated_extended_solver.py` | 555 | Extended Achamrah solver with dual-mode support |
| `achamrah_integrated_benchmark.py` | 474 | Benchmark wrapper with data mapping and output generation |

**Original Achamrah files remain UNCHANGED:**
- `achamrah_2022_irpt_matheuristic.py` (1,035 lines) - NOT modified
- `achamrah_2022_thesis_format_wrapper.py` (842 lines) - NOT modified

## Mode Selection

### Mode 1: `vehicle_indexed_lt=True` (Default)
- **Name:** Achamrah_Original_Integrated
- **LT Variable:** `y[p,i,j,v,t]` (indexed by vehicle)
- **Constraint:** `y[p,i,j,v,t] <= q[p,i,j,v,t]` (LT linked to vehicle flow)
- **Routing:** Integrated with routing decisions (x variables)
- **This:** Closest to original Achamrah paper formulation

### Mode 2: `vehicle_indexed_lt=False`
- **Name:** Achamrah_Integrated_Simplified_LT
- **LT Variable:** `y[p,i,j,t]` (NOT indexed by vehicle)
- **Constraint:** NO linking to vehicle flow
- **Routing:** Kept in model, but LT is independent pairwise transfers
- **This:** Still ONE integrated MIP (not two-stage), but LT is simplified

## Usage

### Run Both Modes (Comparison)
```python
from achamrah_integrated_benchmark import run_achamrah_benchmark_both_modes

results = run_achamrah_benchmark_both_modes(
    dataset_path="test data.csv",
    store_limit=5,
    sku_limit=3,
    period_granularity="daily",
    vehicle_count=2,
    vehicle_capacity=500.0,
    time_limit=3600,  # 1 hour per solve
    mip_gap=0.01,     # 1% gap
    output_dir="Results/achamrah_benchmark",
)

# Access results
original_metrics = results["original"]
simplified_metrics = results["simplified_lt"]
```

### Run Single Mode
```python
from achamrah_integrated_benchmark import AchamrahIntegratedBenchmark, AchamrahBenchmarkConfig

config = AchamrahBenchmarkConfig(
    dataset_path="test data.csv",
    vehicle_indexed_lt=False,  # Simplified LT mode
    store_limit=5,
    sku_limit=3,
    output_dir="Results/achamrah_simplified",
)

benchmark = AchamrahIntegratedBenchmark(config)
metrics = benchmark.run()

print(f"Objective: {metrics.objective:.2f}")
print(f"LT Cost: {metrics.lt_cost:.2f}")
print(f"Runtime: {metrics.runtime_seconds:.1f}s")
```

## Configuration Parameters

```python
AchamrahBenchmarkConfig(
    # Data
    dataset_path: str                    # Path to CSV/Excel file
    store_limit: Optional[int]           # Max stores to use
    sku_limit: Optional[int]             # Max SKUs to use
    start_date: Optional[str]            # YYYY-MM-DD
    end_date: Optional[str]              # YYYY-MM-DD
    period_granularity: str              # "daily" | "weekly" | "biweekly" | "monthly"
    
    # Fleet
    vehicle_count: int = 2               # Number of vehicles
    vehicle_capacity: float = 500.0      # Vehicle capacity (units)
    
    # Mode selection
    vehicle_indexed_lt: bool = True      # True=original, False=simplified
    
    # Costs (per unit or per km)
    alpha: float = 1.0                   # Routing cost per km
    holding_cost_base: float = 0.01      # Holding cost per unit per period
    shortage_cost_base: float = 0.25     # Shortage cost per unit
    lt_cost_base: float = 0.6            # LT cost per unit
    
    # Solver
    time_limit: int = 3600               # Seconds
    mip_gap: float = 0.01                # Relative gap (0.01 = 1%)
    threads: int = 4                     # Gurobi threads
    
    # Output
    output_dir: Optional[Path] = None    # Export directory
    verbose: bool = True                 # Print progress
)
```

## Input Data Format

Required CSV/Excel columns:
- `SITE_NAME` → store
- `ART_SV_NAME_ENG` → SKU/product
- `SALE_QTY` → demand (sale quantity)
- `END_QTY` → ending inventory (for initial state)
- `PERIOD` → date (YYYYMMDD format)

Example:
```csv
SITE_NAME,ART_SV_NAME_ENG,SALE_QTY,END_QTY,PERIOD
MM AN PHU,PRODUCT_A,100.5,250.0,20250101
MM AN PHU,PRODUCT_B,50.2,125.3,20250101
MM BINH PHU,PRODUCT_A,75.0,180.5,20250101
```

## Output Files

### Summary (`achamrah_integrated_summary.json` / `.csv`)
```json
{
  "source": "Achamrah_Original_Integrated | Achamrah_Integrated_Simplified_LT",
  "vehicle_indexed_lt": true / false,
  "dataset_name": "test data",
  "num_stores": 5,
  "num_skus": 3,
  "num_periods": 30,
  "num_vehicles": 2,
  "objective": 45678.50,
  "routing_cost": 1234.50,
  "holding_cost": 12345.00,
  "shortage_cost": 5678.90,
  "lt_cost": 25720.10,
  "total_cost": 45678.50,
  "status": "OPTIMAL",
  "mip_gap": 0.0,
  "runtime_seconds": 127.4,
  "num_lt_moves": 45,
  "total_lt_qty": 1250.5
}
```

### LT Plan (`achamrah_integrated_lt_plan.csv`)
```csv
source,period,from_store,to_store,sku,lt_qty,lt_cost,vehicle
Achamrah_Original_Integrated,1,MM AN PHU,MM BINH PHU,PRODUCT_A,50.0,30.0,1
Achamrah_Original_Integrated,2,MM BINH PHU,MM AN PHU,PRODUCT_B,25.5,15.3,2
Achamrah_Integrated_Simplified_LT,1,MM AN PHU,MM BINH PHU,PRODUCT_A,55.0,33.0,N/A
```

## Comparison with Thesis Work

### Comparable Columns
All outputs include columns that match thesis work for easy comparison:
- `source` / `method` - Achamrah vs thesis
- `period`, `store`, `sku` - Matching keys
- `routing_cost`, `holding_cost`, `shortage_cost`, `lt_cost`, `total_cost`
- `lt_qty`, `lt_unit_cost` - LT quantities and costs
- `runtime_seconds` - Solver runtime
- `status`, `mip_gap` - Solution quality

### Join/Comparison Workflow
```python
# Load both outputs
achamrah_summary = pd.read_csv("Results/achamrah_benchmark/achamrah_original/achamrah_integrated_summary.csv")
thesis_summary = pd.read_csv("Results/thesis_summary/comparison_aggregate.csv")

# Compare by instance
comparison = pd.merge(
    achamrah_summary[["source", "objective", "lt_cost", "runtime_seconds"]],
    thesis_summary[["method", "objective", "lt_cost", "runtime_seconds"]],
    left_on="source",
    right_on="method",
    how="outer"
)
```

## Model Formulation

### Shared Components (Both Modes)

**Variables:**
- `I[p,i,t]` - Inventory at node i for product p at end of period t
- `S[p,i,t]` - Shortage at node i, product p, period t
- `Qdir[p,j,t]` - Direct shipment from CW to store j
- `q[p,i,j,v,t]` - Product p flow on arc (i,j) by vehicle v, period t
- `x[i,j,v,t]` - Routing (1 if vehicle v uses arc (i,j) in period t)
- `u[v,t]` - Vehicle usage indicator
- `z[i,v,t]` - Visit indicator for store i by vehicle v

**Objective:**
```
min: routing_cost + holding_cost + shortage_cost + lt_cost
   = α·Σ d[i,j]·x[i,j,v,t]
   + Σ h[p,i]·I[p,i,t]
   + Σ f[p,i]·S[p,i,t]
   + Σ b[i,j]·y[p,i,j,t or y[p,i,j,v,t]]
```

### Mode 1 Specific: `vehicle_indexed_lt=True`

**LT Variable:**
- `y[p,i,j,v,t]` - LT quantity of product p from store i to j via vehicle v

**Key Constraint:**
```
y[p,i,j,v,t] <= q[p,i,j,v,t]
→ LT must be carried by vehicle flow
```

**Inventory Balance (POS):**
```
I[p,i,t] + S[p,i,t] = I[p,i,t-1]
                      + Qdir[p,i,t]
                      + Σ_j Σ_v y[p,j,i,v,t]  [inbound LT]
                      - demand[p,i,t]
                      - Σ_j Σ_v y[p,i,j,v,t]  [outbound LT]
```

### Mode 2 Specific: `vehicle_indexed_lt=False`

**LT Variable:**
- `y[p,i,j,t]` - LT quantity of product p from store i to j (NO vehicle index)

**Key Constraint:**
```
NO constraint: y[p,i,j,t] is independent from q
→ LT not linked to vehicle routing
```

**Inventory Balance (POS):**
```
I[p,i,t] + S[p,i,t] = I[p,i,t-1]
                      + Qdir[p,i,t]
                      + Σ_j y[p,j,i,t]  [inbound LT, no vehicle]
                      - demand[p,i,t]
                      - Σ_j y[p,i,j,t]  [outbound LT, no vehicle]
```

**Constraint Reduction:**
- Removed: `y[p,i,j,v,t] <= q[p,i,j,v,t]` for all v
- Added: `Σ_j y[p,i,j,t] <= I[p,i,t-1]` (LT limited by available inventory)

## Implementation Details

### Extended Solver (`achamrah_integrated_extended_solver.py`)

**Class:** `AchamrahIntegratedExtendedSolver(AchamrahIRPTSolver)`

**Key Methods:**
- `build_model(vehicle_indexed_lt)` - Dispatches to mode-specific builder
- `_build_model_simplified_lt()` - Constructs simplified LT model
- `solve_model()` - Solves and extracts cost breakdown + LT moves

**Cost Breakdown Extraction:**
- Routing cost: sum of `α·d[i,j]·x[i,j,v,t]`
- Holding cost: sum of `h[p,i]·I[p,i,t]`
- Shortage cost: sum of `f[p,i]·S[p,i,t]`
- LT cost: sum of `b[i,j]·y[...]`

### Benchmark Wrapper (`achamrah_integrated_benchmark.py`)

**Classes:**
- `AchamrahDatasetMapper` - Maps thesis CSV → Achamrah instance
- `AchamrahIntegratedBenchmark` - Orchestrates prepare → solve → export
- `AchamrahSolutionMetrics` - Stores solution data

**Output Generation:**
- Summary: JSON + CSV
- LT Plan: CSV with store/SKU names
- Metadata: Instance size, solver stats, runtimes

## Notes

1. **Distance Matrix:** Currently synthetic Euclidean coordinates (seed=42 for reproducibility)
   - To use real distances: override `build_distance_matrix()` in mapper

2. **Cost Parameters:** Configurable via `AchamrahBenchmarkConfig`
   - Defaults: α=1.0, holding=0.01, shortage=0.25, lt_cost=0.6
   - Adjust as needed for your data

3. **Solver:** Uses Gurobi (requires license)
   - Time limit, MIP gap, threads configurable

4. **Validation:** Data format automatically validated against required columns

5. **Comparison:** All outputs include `source` column for thesis work joining

## Comparison Checklist

- ✅ Both modes solve ONE integrated model (not two-stage)
- ✅ Input format compatible with thesis dataset
- ✅ Output columns match thesis work
- ✅ Source/method column for easy joining
- ✅ Cost breakdown (routing, holding, shortage, LT)
- ✅ LT diagnostics (quantity, cost, moves)
- ✅ Solver stats (runtime, MIP gap, vars, constraints)
- ✅ Original Achamrah code NOT modified
