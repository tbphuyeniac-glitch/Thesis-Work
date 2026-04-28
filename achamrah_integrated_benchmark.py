"""
Achamrah Integrated Benchmark Wrapper
======================================

Two modes for Achamrah IRPT model:
1. vehicle_indexed_lt=True:  Original Achamrah with LT indexed by vehicle
2. vehicle_indexed_lt=False: Simplified pairwise LT without vehicle indexing

Both modes solve ONE integrated MIP (not two-stage).

Output format: Comparable with thesis work via source/method column.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

from achamrah_2022_irpt_matheuristic import IRPTInstance
from achamrah_integrated_extended_solver import AchamrahIntegratedExtendedSolver


@dataclass
class AchamrahBenchmarkConfig:
    """Configuration for Achamrah benchmark run."""
    dataset_path: str
    store_limit: Optional[int] = None
    sku_limit: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period_granularity: str = "daily"
    
    vehicle_count: int = 2
    vehicle_capacity: float = 500.0
    
    vehicle_indexed_lt: bool = True  # Mode selector
    
    # Cost parameters
    alpha: float = 1.0  # routing cost per km
    holding_cost_base: float = 0.01  # base holding cost
    shortage_cost_base: float = 0.25  # base shortage cost
    lt_cost_base: float = 0.6  # base LT cost per unit
    
    # Gurobi tuning
    time_limit: int = 3600
    mip_gap: float = 0.01
    threads: int = 4
    
    # Output
    output_dir: Optional[Path] = None
    verbose: bool = True


@dataclass
class AchamrahSolutionMetrics:
    """Solution metrics from Achamrah benchmark."""
    source: str = ""
    vehicle_indexed_lt: bool = True
    dataset_name: str = ""
    
    # Instance size
    num_stores: int = 0
    num_skus: int = 0
    num_periods: int = 0
    num_vehicles: int = 0
    store_limit: Optional[int] = None
    sku_limit: Optional[int] = None
    start_date: str = ""
    end_date: str = ""
    period_granularity: str = ""
    
    # Model stats
    num_vars: int = 0
    num_constrs: int = 0
    
    # Objective and costs
    objective: float = 0.0
    routing_cost: float = 0.0
    holding_cost: float = 0.0
    shortage_cost: float = 0.0
    lt_cost: float = 0.0
    total_cost: float = 0.0
    
    # Solution quality
    status: str = ""
    mip_gap: float = float("nan")
    runtime_seconds: float = 0.0
    
    # LT diagnostics
    num_lt_moves: int = 0
    total_lt_qty: float = 0.0
    max_shortage_qty: float = 0.0
    total_shortage_qty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


class AchamrahDatasetMapper:
    """Map thesis dataset format to Achamrah IRPT instance."""
    
    REQUIRED_COLUMNS = [
        "SITE_NAME", "ART_SV_NAME_ENG", "SALE_QTY", "END_QTY", "PERIOD"
    ]
    
    def __init__(self, config: AchamrahBenchmarkConfig):
        self.config = config
        self.df_raw = None
        self.df_processed = None
        self.store_to_id = {}
        self.id_to_store = {}
        self.product_to_id = {}
        self.id_to_product = {}
        self.period_dates = {}
        self.distance_matrix = {}
        self.metadata = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate dataset."""
        path = Path(self.config.dataset_path)
        
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, sheet_name=0)
        
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        df = df[[*self.REQUIRED_COLUMNS]].copy()
        df = df.rename(columns={
            "SITE_NAME": "store",
            "ART_SV_NAME_ENG": "sku",
            "SALE_QTY": "sale_qty",
            "END_QTY": "end_qty",
            "PERIOD": "period_raw",
        })
        
        df["store"] = df["store"].astype(str).str.strip()
        df["sku"] = df["sku"].astype(str).str.strip()
        df["sale_qty"] = pd.to_numeric(df["sale_qty"], errors="coerce").fillna(0.0)
        df["end_qty"] = pd.to_numeric(df["end_qty"], errors="coerce").fillna(0.0)
        df["period_date"] = pd.to_datetime(df["period_raw"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["period_date"])
        
        if self.config.start_date:
            df = df[df["period_date"] >= pd.to_datetime(self.config.start_date)]
        if self.config.end_date:
            df = df[df["period_date"] <= pd.to_datetime(self.config.end_date)]
        
        self.df_raw = df
        return df
    
    def preprocess(self) -> pd.DataFrame:
        """Aggregate to period granularity."""
        if self.df_raw is None:
            self.load_data()
        
        df = self.df_raw.copy()
        
        gran = self.config.period_granularity.lower()
        if gran == "weekly":
            df["period_date"] = df["period_date"].dt.to_period("W").apply(lambda p: p.start_time)
        elif gran == "biweekly":
            ref = df["period_date"].min()
            df["period_date"] = df["period_date"].apply(
                lambda d: ref + pd.Timedelta(weeks=((d - ref).days // 14) * 2)
            )
        elif gran == "monthly":
            df["period_date"] = df["period_date"].dt.to_period("M").apply(lambda p: p.start_time)
        
        grp = (
            df.groupby(["store", "sku", "period_date"], as_index=False)
              .agg(sale_qty=("sale_qty", "sum"), end_qty=("end_qty", "last"))
        )
        grp = grp.sort_values(["store", "sku", "period_date"]).reset_index(drop=True)
        
        if self.config.store_limit:
            stores = grp["store"].unique()[:self.config.store_limit]
            grp = grp[grp["store"].isin(stores)]
        
        if self.config.sku_limit:
            skus = grp["sku"].unique()[:self.config.sku_limit]
            grp = grp[grp["sku"].isin(skus)]
        
        self.df_processed = grp
        return grp
    
    def build_mappings(self) -> None:
        """Build store/product/period mappings."""
        if self.df_processed is None:
            self.preprocess()
        
        df = self.df_processed
        
        stores = sorted(df["store"].unique())
        self.store_to_id = {s: i + 1 for i, s in enumerate(stores)}
        self.id_to_store = {v: k for k, v in self.store_to_id.items()}
        
        products = sorted(df["sku"].unique())
        self.product_to_id = {p: i for i, p in enumerate(products)}
        self.id_to_product = {v: k for k, v in self.product_to_id.items()}
        
        periods = sorted(df["period_date"].unique())
        self.period_dates = {i + 1: p for i, p in enumerate(periods)}
        
        self.metadata = {
            "num_stores": len(self.store_to_id),
            "num_products": len(self.product_to_id),
            "num_periods": len(self.period_dates),
            "stores": stores,
            "products": products,
            "periods": periods,
        }
    
    def build_distance_matrix(self) -> Dict[Tuple[int, int], float]:
        """Build synthetic Euclidean distance matrix."""
        num_nodes = len(self.store_to_id) + 1
        np.random.seed(42)
        
        coords = np.random.uniform(0, 100, size=(num_nodes, 2))
        distances = {}
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    distances[(i, j)] = float(np.linalg.norm(coords[i] - coords[j]))
        
        self.distance_matrix = distances
        return distances
    
    def build_instance(self) -> IRPTInstance:
        """Construct IRPTInstance for Achamrah solver."""
        self.build_mappings()
        self.build_distance_matrix()
        
        df = self.df_processed
        N = list(self.store_to_id.values())
        P = list(range(len(self.product_to_id)))
        H = list(range(1, len(self.period_dates) + 1))
        V = list(range(1, self.config.vehicle_count + 1))
        
        # Build demand
        D = {}
        for _, row in df.iterrows():
            p_id = self.product_to_id[row["sku"]]
            i_id = self.store_to_id[row["store"]]
            for t_idx, t in enumerate(H, start=1):
                if self.period_dates[t] == row["period_date"]:
                    D[(p_id, i_id, t)] = row["sale_qty"]
                else:
                    D[(p_id, i_id, t)] = 0.0
        
        for p in P:
            for i in N:
                for t in H:
                    if (p, i, t) not in D:
                        D[(p, i, t)] = 0.0
        
        # Initial inventory
        I0 = {}
        for p in P:
            for i in N:
                mask = (df["sku"] == self.id_to_product[p]) & (df["store"] == self.id_to_store[i])
                I0[(p, i)] = df[mask]["end_qty"].iloc[0] if mask.any() else 0.0
        
        for p in P:
            I0[(p, 0)] = 10000.0
        
        # Cost dicts
        h = {(p, i): self.config.holding_cost_base for p in P for i in N}
        h[(P[0], 0)] = 0.0
        
        C = {i: 1000.0 for i in N}
        C[0] = 100000.0
        
        f = {(p, i): self.config.shortage_cost_base for p in P for i in N}
        f[(P[0], 0)] = 0.0
        
        b = {(i, j): self.config.lt_cost_base for i in N for j in N if i != j}
        
        g = {(p, t): 100000.0 for p in P for t in H}
        
        instance = IRPTInstance(
            N=N, P=P, H=H, V=V,
            alpha=self.config.alpha,
            Q=self.config.vehicle_capacity,
            d=self.distance_matrix,
            b=b, h=h, C=C, I0=I0, D=D, g=g, f=f,
            name=Path(self.config.dataset_path).stem,
        )
        
        instance.validate()
        return instance


class AchamrahIntegratedBenchmark:
    """Run Achamrah benchmark in integrated mode."""
    
    def __init__(self, config: AchamrahBenchmarkConfig):
        self.config = config
        self.mapper = AchamrahDatasetMapper(config)
        self.instance = None
        self.solver = None
        self.artifacts = None
        self.metrics = None
        
    def prepare(self) -> None:
        """Load data and build instance."""
        print(f"[Achamrah] Preparing dataset (mode: vehicle_indexed_lt={self.config.vehicle_indexed_lt})...")
        self.instance = self.mapper.build_instance()
        
        if self.config.verbose:
            print(f"  Stores: {self.mapper.metadata['num_stores']}")
            print(f"  Products: {self.mapper.metadata['num_products']}")
            print(f"  Periods: {self.mapper.metadata['num_periods']}")
            print(f"  Vehicles: {self.config.vehicle_count}")
    
    def solve(self) -> None:
        """Solve the instance using extended Achamrah."""
        print(f"[Achamrah] Solving (time_limit={self.config.time_limit}s, mip_gap={self.config.mip_gap})...")
        
        t0 = time.time()
        
        self.solver = AchamrahIntegratedExtendedSolver(
            self.instance,
            vehicle_indexed_lt=self.config.vehicle_indexed_lt,
        )
        
        self.artifacts = self.solver.solve_model(
            time_limit=self.config.time_limit,
            mip_gap=self.config.mip_gap,
            allow_lateral_transshipment=True,
        )
        
        runtime = time.time() - t0
        self._build_metrics(runtime)
    
    def _build_metrics(self, runtime: float) -> None:
        """Extract solution metrics."""
        m = self.artifacts
        
        self.metrics = AchamrahSolutionMetrics(
            source="Achamrah_Original_Integrated" if self.config.vehicle_indexed_lt else "Achamrah_Integrated_Simplified_LT",
            vehicle_indexed_lt=self.config.vehicle_indexed_lt,
            dataset_name=Path(self.config.dataset_path).stem,
            num_stores=self.mapper.metadata["num_stores"],
            num_skus=self.mapper.metadata["num_products"],
            num_periods=self.mapper.metadata["num_periods"],
            num_vehicles=self.config.vehicle_count,
            store_limit=self.config.store_limit,
            sku_limit=self.config.sku_limit,
            period_granularity=self.config.period_granularity,
            num_vars=m.num_vars,
            num_constrs=m.num_constrs,
            objective=m.objective,
            status="OPTIMAL" if m.status == 2 else "SUBOPTIMAL" if m.status == 9 else str(m.status),
            mip_gap=m.mip_gap,
            runtime_seconds=runtime,
        )
        
        if m.cost_breakdown:
            self.metrics.routing_cost = m.cost_breakdown.get("routing", 0.0)
            self.metrics.holding_cost = m.cost_breakdown.get("holding", 0.0)
            self.metrics.shortage_cost = m.cost_breakdown.get("shortage", 0.0)
            self.metrics.lt_cost = m.cost_breakdown.get("lt", 0.0)
            self.metrics.total_cost = m.cost_breakdown.get("total", m.objective)
        
        self.metrics.num_lt_moves = len(m.lt_moves)
        self.metrics.total_lt_qty = sum(move.get("quantity", 0) for move in m.lt_moves)
    
    def export_results(self) -> Dict[str, Path]:
        """Export solution to thesis-comparable files."""
        if not self.config.output_dir:
            self.config.output_dir = Path("Results") / "achamrah_benchmark"
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        exports = {}
        
        # Summary JSON
        summary_path = self.config.output_dir / "achamrah_integrated_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2, default=str)
        exports["summary"] = summary_path
        
        # Summary CSV
        summary_csv_path = self.config.output_dir / "achamrah_integrated_summary.csv"
        pd.DataFrame([self.metrics.to_dict()]).to_csv(summary_csv_path, index=False)
        exports["summary_csv"] = summary_csv_path
        
        # LT Plan
        if self.artifacts and self.artifacts.lt_moves:
            lt_df = pd.DataFrame(self.artifacts.lt_moves)
            lt_df.columns = ["period", "product_id", "from_store_id", "to_store_id", "vehicle", "lt_qty", "lt_cost"]
            
            # Add store/product names
            lt_df["from_store"] = lt_df["from_store_id"].map(self.mapper.id_to_store)
            lt_df["to_store"] = lt_df["to_store_id"].map(self.mapper.id_to_store)
            lt_df["sku"] = lt_df["product_id"].map(self.mapper.id_to_product)
            lt_df["source"] = self.metrics.source
            
            lt_path = self.config.output_dir / "achamrah_integrated_lt_plan.csv"
            lt_df[[
                "source", "period", "from_store", "to_store", "sku", 
                "lt_qty", "lt_cost", "vehicle"
            ]].to_csv(lt_path, index=False)
            exports["lt_plan"] = lt_path
        
        print(f"[Achamrah] Results exported to {self.config.output_dir}")
        return exports
    
    def run(self) -> AchamrahSolutionMetrics:
        """Execute full benchmark."""
        self.prepare()
        self.solve()
        self.export_results()
        return self.metrics


def run_achamrah_benchmark_both_modes(
    dataset_path: str,
    output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, AchamrahSolutionMetrics]:
    """Run Achamrah benchmark in both modes for comparison."""
    results = {}
    
    for vehicle_indexed_lt in [True, False]:
        mode_name = "original" if vehicle_indexed_lt else "simplified_lt"
        print(f"\n{'='*70}")
        print(f"Mode: {mode_name} (vehicle_indexed_lt={vehicle_indexed_lt})")
        print(f"{'='*70}")
        
        config = AchamrahBenchmarkConfig(
            dataset_path=dataset_path,
            vehicle_indexed_lt=vehicle_indexed_lt,
            output_dir=Path(output_dir) / f"achamrah_{mode_name}" if output_dir else None,
            **kwargs
        )
        
        benchmark = AchamrahIntegratedBenchmark(config)
        metrics = benchmark.run()
        results[mode_name] = metrics
    
    # Comparison
    print(f"\n{'='*70}")
    print("ACHAMRAH BENCHMARK COMPARISON")
    print(f"{'='*70}")
    
    df_comp = pd.DataFrame([results["original"].to_dict(), results["simplified_lt"].to_dict()])
    print(df_comp[[
        "source", "objective", "routing_cost", "lt_cost", "runtime_seconds", "status"
    ]].to_string(index=False))
    
    return results


if __name__ == "__main__":
    results = run_achamrah_benchmark_both_modes(
        dataset_path="test data.csv",
        store_limit=5,
        sku_limit=3,
        period_granularity="daily",
        vehicle_count=2,
        vehicle_capacity=500.0,
        output_dir="Results/achamrah_benchmark",
    )
