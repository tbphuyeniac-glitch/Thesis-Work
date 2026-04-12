from __future__ import annotations

import math
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
from gurobipy import GRB

from achamrah_2022_irpt_matheuristic import (
    IRPTInstance,
    HeuristicParams,
    AchamrahIRPTSolver,
    SolveArtifacts,
)


Store = str
Product = str
Period = int
NodeId = int
ProductId = int


@dataclass
class MappingBundle:
    instance: IRPTInstance
    base_df: pd.DataFrame
    validation_target: pd.DataFrame
    metadata: Dict[str, Any]
    store_to_id: Dict[Store, NodeId]
    id_to_store: Dict[NodeId, Store]
    product_to_id: Dict[Product, ProductId]
    id_to_product: Dict[ProductId, Product]
    period_dates: Dict[Period, pd.Timestamp]


class DatasetToAchamrahMapper:
    REQUIRED_COLUMNS = [
        "SITE_NAME", "NORMAL_PRICE", "ART_SV_NAME_ENG",
        "SALE_QTY", "END_QTY", "PERIOD"
    ]

    def __init__(
        self,
        excel_path: str,
        sheet_name: Optional[str] = None,
        store_limit: Optional[int] = None,
        sku_limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.store_limit = store_limit
        self.sku_limit = sku_limit
        self.start_date = start_date
        self.end_date = end_date

    def load_raw(self) -> pd.DataFrame:
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name or 0)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[self.REQUIRED_COLUMNS].copy()
        df = df.rename(columns={
            "SITE_NAME": "store",
            "NORMAL_PRICE": "price",
            "ART_SV_NAME_ENG": "sku",
            "SALE_QTY": "sale_qty",
            "END_QTY": "end_qty",
            "PERIOD": "period_raw",
        })

        df["store"] = df["store"].astype(str).str.strip()
        df["sku"] = df["sku"].astype(str).str.strip()
        df["sale_qty"] = pd.to_numeric(df["sale_qty"], errors="coerce").fillna(0.0)
        df["end_qty"] = pd.to_numeric(df["end_qty"], errors="coerce").fillna(0.0)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["period_date"] = pd.to_datetime(df["period_raw"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["period_date"]).copy()

        if self.start_date is not None:
            df = df[df["period_date"] >= pd.to_datetime(self.start_date)]
        if self.end_date is not None:
            df = df[df["period_date"] <= pd.to_datetime(self.end_date)]

        return df

    def preprocess(self) -> pd.DataFrame:
        df = self.load_raw()
        grp = (
            df.groupby(["store", "sku", "period_date"], as_index=False)
              .agg(
                  sale_qty=("sale_qty", "sum"),
                  end_qty=("end_qty", "sum"),
                  price=("price", "median"),
              )
        )

        if self.store_limit is not None:
            top_stores = (
                grp.groupby("store")["sale_qty"].sum()
                .sort_values(ascending=False)
                .head(self.store_limit)
                .index.tolist()
            )
            grp = grp[grp["store"].isin(top_stores)].copy()

        if self.sku_limit is not None:
            top_skus = (
                grp.groupby("sku")["sale_qty"].sum()
                .sort_values(ascending=False)
                .head(self.sku_limit)
                .index.tolist()
            )
            grp = grp[grp["sku"].isin(top_skus)].copy()

        grp = grp.sort_values(["store", "sku", "period_date"]).reset_index(drop=True)
        unique_dates = sorted(grp["period_date"].drop_duplicates().tolist())
        date_to_period = {dt: i + 1 for i, dt in enumerate(unique_dates)}
        grp["period"] = grp["period_date"].map(date_to_period)
        return grp

    def build_instance(
        self,
        wh_inventory_multiplier: float = 2.5,
        store_capacity_multiplier: float = 1.5,
        shortage_cost_rate: float = 0.25,
        holding_cost_rate: float = 0.01,
        vehicle_count: int = 2,
        vehicle_capacity: float = 120.0,
        alpha: float = 1.0,
        cw_replenishment_factor: float = 0.6,
        cw_capacity_factor: float = 2.0,
        synthetic_cw_distance: float = 10.0,
        synthetic_store_distance: float = 6.0,
    ) -> MappingBundle:
        df = self.preprocess()

        stores = sorted(df["store"].unique().tolist())
        products = sorted(df["sku"].unique().tolist())
        periods = sorted(df["period"].unique().tolist())
        if not stores or not products or not periods:
            raise ValueError("No valid data after preprocessing.")

        store_to_id = {s: i + 1 for i, s in enumerate(stores)}
        id_to_store = {i: s for s, i in store_to_id.items()}
        product_to_id = {p: i for i, p in enumerate(products)}
        id_to_product = {i: p for p, i in product_to_id.items()}
        period_dates = dict(df[["period", "period_date"]].drop_duplicates().values.tolist())

        full_index = pd.MultiIndex.from_product(
            [stores, products, periods], names=["store", "sku", "period"]
        )
        base = (
            df.set_index(["store", "sku", "period"])[["sale_qty", "end_qty", "price", "period_date"]]
              .reindex(full_index)
              .reset_index()
        )
        base["sale_qty"] = base["sale_qty"].fillna(0.0)
        base["end_qty"] = base["end_qty"].fillna(0.0)

        sku_price = base.groupby("sku")["price"].median()
        global_price = float(base["price"].median()) if base["price"].notna().any() else 1.0
        base["price"] = base.apply(
            lambda r: sku_price.get(r["sku"], global_price) if pd.isna(r["price"]) else r["price"],
            axis=1,
        )
        base["price"] = base["price"].fillna(global_price)

        P = [product_to_id[p] for p in products]
        N = [store_to_id[s] for s in stores]
        H = periods[:]
        V = list(range(1, vehicle_count + 1))

        D: Dict[Tuple[int, int, int], float] = {}
        for _, row in base.iterrows():
            D[(product_to_id[row["sku"]], store_to_id[row["store"]], int(row["period"]))] = float(row["sale_qty"])

        first_period = min(periods)
        first_df = base[base["period"] == first_period].copy()

        I0: Dict[Tuple[int, int], float] = {}
        for p in P:
            I0[(p, 0)] = 0.0
        for _, row in first_df.iterrows():
            s_id = store_to_id[row["store"]]
            p_id = product_to_id[row["sku"]]
            I0[(p_id, s_id)] = max(0.0, float(row["end_qty"]))
        for s in N:
            for p in P:
                I0.setdefault((p, s), 0.0)

        max_inventory_store: Dict[Tuple[int, int], float] = {}
        for s_name in stores:
            for sku in products:
                obs = base[(base["store"] == s_name) & (base["sku"] == sku)]["end_qty"]
                obs_max = float(obs.max()) if not obs.empty else 0.0
                init_inv = I0[(product_to_id[sku], store_to_id[s_name])]
                max_inventory_store[(store_to_id[s_name], product_to_id[sku])] = max(
                    5.0,
                    store_capacity_multiplier * max(obs_max, init_inv, 1.0),
                )

        total_demand_by_sku = base.groupby("sku")["sale_qty"].sum().to_dict()
        max_inventory_wh: Dict[int, float] = {}
        for sku in products:
            p_id = product_to_id[sku]
            total_dem = float(total_demand_by_sku.get(sku, 0.0))
            I0[(p_id, 0)] = max(0.0, wh_inventory_multiplier * total_dem)
            max_inventory_wh[p_id] = max(I0[(p_id, 0)], 1.2 * I0[(p_id, 0)])

        h: Dict[Tuple[int, int], float] = {}
        f: Dict[Tuple[int, int], float] = {}
        price_by_store_sku = base.groupby(["store", "sku"])["price"].median().to_dict()
        for s_name in stores:
            s_id = store_to_id[s_name]
            for sku in products:
                p_id = product_to_id[sku]
                price = float(price_by_store_sku.get((s_name, sku), global_price))
                h[(p_id, s_id)] = max(0.05, holding_cost_rate * price)
                f[(p_id, s_id)] = max(1.0, shortage_cost_rate * price)
        for sku in products:
            p_id = product_to_id[sku]
            med_price = float(base.loc[base["sku"] == sku, "price"].median()) if (base["sku"] == sku).any() else global_price
            h[(p_id, 0)] = max(0.02, holding_cost_rate * 0.5 * med_price)

        C: Dict[int, float] = {}
        for s_name in stores:
            s_id = store_to_id[s_name]
            C[s_id] = sum(max_inventory_store[(s_id, product_to_id[sku])] for sku in products)
        C[0] = cw_capacity_factor * sum(I0[(product_to_id[sku], 0)] for sku in products)

        g: Dict[Tuple[int, int], float] = {}
        demand_by_sku_period = base.groupby(["sku", "period"])["sale_qty"].sum().to_dict()
        for sku in products:
            p_id = product_to_id[sku]
            for t in periods:
                g[(p_id, t)] = max(0.0, cw_replenishment_factor * float(demand_by_sku_period.get((sku, t), 0.0)))

        d: Dict[Tuple[int, int], float] = {}
        node_ids = [0] + N
        for i in node_ids:
            for j in node_ids:
                if i == j:
                    continue
                if i == 0 or j == 0:
                    d[(i, j)] = float(synthetic_cw_distance)
                else:
                    d[(i, j)] = float(synthetic_store_distance)

        b = {(i, j): 0.01 * alpha * d[(i, j)] for i in N for j in N if i != j}

        inst = IRPTInstance(
            N=N,
            P=P,
            H=H,
            V=V,
            alpha=alpha,
            Q=vehicle_capacity,
            d=d,
            b=b,
            h=h,
            C=C,
            I0=I0,
            D=D,
            g=g,
            f=f,
            name=Path(self.excel_path).stem,
        )

        validation_target = base[base["period"] > first_period][["store", "sku", "period", "end_qty"]].copy()
        validation_target = validation_target.rename(columns={"end_qty": "actual_end_qty"})

        metadata = {
            "n_rows_processed": int(len(base)),
            "n_stores": int(len(stores)),
            "n_products": int(len(products)),
            "n_periods": int(len(periods)),
            "validation_rows": int(len(validation_target)),
            "first_period_used_as_initial_inventory": int(first_period),
            "n_vehicles": int(len(V)),
            "vehicle_capacity": float(vehicle_capacity),
            "instance_name": inst.name,
        }

        return MappingBundle(
            instance=inst,
            base_df=base,
            validation_target=validation_target,
            metadata=metadata,
            store_to_id=store_to_id,
            id_to_store=id_to_store,
            product_to_id=product_to_id,
            id_to_product=id_to_product,
            period_dates=period_dates,
        )


def mapping_from_irp_data(
    data: Any,
    base_df: Optional[pd.DataFrame] = None,
    validation_target: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MappingBundle:
    stores = list(data.stores)
    products = list(data.products)
    periods = list(data.periods)
    warehouse = getattr(data, "warehouse", "CW")

    store_to_id = {s: i + 1 for i, s in enumerate(stores)}
    id_to_store = {i: s for s, i in store_to_id.items()}
    product_to_id = {p: i for i, p in enumerate(products)}
    id_to_product = {i: p for p, i in product_to_id.items()}

    P = [product_to_id[p] for p in products]
    N = [store_to_id[s] for s in stores]
    H = periods[:]
    V = list(range(1, len(getattr(data, "vehicles", [])) + 1)) or [1]

    D = {
        (product_to_id[p], store_to_id[s], t): float(data.demand[(s, p, t)])
        for s in stores for p in products for t in periods
    }

    I0: Dict[Tuple[int, int], float] = {}
    for p in products:
        p_id = product_to_id[p]
        I0[(p_id, 0)] = float(data.init_inventory_wh[p])
        for s in stores:
            I0[(p_id, store_to_id[s])] = float(data.init_inventory_store[(s, p)])

    h: Dict[Tuple[int, int], float] = {}
    f: Dict[Tuple[int, int], float] = {}
    for p in products:
        p_id = product_to_id[p]
        h[(p_id, 0)] = float(data.holding_cost_wh[p])
        for s in stores:
            s_id = store_to_id[s]
            h[(p_id, s_id)] = float(data.holding_cost_store[(s, p)])
            f[(p_id, s_id)] = float(data.shortage_cost[(s, p)])

    C: Dict[int, float] = {0: float(data.node_capacity[warehouse])}
    for s in stores:
        s_id = store_to_id[s]
        C[s_id] = float(data.node_capacity.get(
            s,
            sum(data.max_inventory_store[(s, p)] for p in products),
        ))

    g = {
        (product_to_id[p], t): float(data.replenishment_wh[(p, t)])
        for p in products for t in periods
    }

    d: Dict[Tuple[int, int], float] = {}
    for i_label, i_id in [(warehouse, 0)] + [(s, store_to_id[s]) for s in stores]:
        for j_label, j_id in [(warehouse, 0)] + [(s, store_to_id[s]) for s in stores]:
            if i_id == j_id:
                continue
            d[(i_id, j_id)] = float(data.distance[(i_label, j_label)])

    b = {
        (store_to_id[i], store_to_id[j]): float(data.transship_unit_cost[(i, j)])
        for i in stores for j in stores if i != j
    }

    inst = IRPTInstance(
        N=N,
        P=P,
        H=H,
        V=V,
        alpha=float(data.alpha),
        Q=float(data.vehicle_capacity),
        d=d,
        b=b,
        h=h,
        C=C,
        I0=I0,
        D=D,
        g=g,
        f=f,
        name=metadata.get("instance_name", "irp_data") if metadata else "irp_data",
    )

    if base_df is None:
        base_df = pd.DataFrame()
    if validation_target is None:
        validation_target = pd.DataFrame(columns=["store", "sku", "period", "actual_end_qty"])

    period_dates = {}
    if not base_df.empty and {"period", "period_date"}.issubset(base_df.columns):
        period_dates = dict(base_df[["period", "period_date"]].drop_duplicates().values.tolist())

    meta = dict(metadata or {})
    meta.update({
        "n_stores": int(len(stores)),
        "n_products": int(len(products)),
        "n_periods": int(len(periods)),
        "n_vehicles": int(len(V)),
        "vehicle_capacity": float(data.vehicle_capacity),
        "instance_name": inst.name,
        "source": "IRPData",
    })

    return MappingBundle(
        instance=inst,
        base_df=base_df,
        validation_target=validation_target,
        metadata=meta,
        store_to_id=store_to_id,
        id_to_store=id_to_store,
        product_to_id=product_to_id,
        id_to_product=id_to_product,
        period_dates=period_dates,
    )


def artifact_to_predicted_inventory_df(artifact: SolveArtifacts, mapping: MappingBundle) -> pd.DataFrame:
    I = artifact.vars["I"]
    rows = []
    for p_id in mapping.instance.P:
        sku = mapping.id_to_product[p_id]
        for s_id in mapping.instance.N:
            store = mapping.id_to_store[s_id]
            for t in mapping.instance.H:
                rows.append({
                    "store": store,
                    "sku": sku,
                    "period": t,
                    "predicted_end_qty": float(I[p_id, s_id, t].X if artifact.model.SolCount > 0 else 0.0),
                })
    return pd.DataFrame(rows)


def artifact_to_route_df(artifact: SolveArtifacts, mapping: MappingBundle) -> pd.DataFrame:
    q = artifact.vars["q"]
    y = artifact.vars["y"]
    x = artifact.vars["x"]
    rows = []

    active_by_vt: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    if artifact.model.SolCount > 0:
        for (i, j, v, t) in x.keys():
            if float(x[i, j, v, t].X) > 0.5:
                active_by_vt.setdefault((v, t), []).append((i, j))

    for (v, t), arcs in sorted(active_by_vt.items(), key=lambda item: (item[0][1], item[0][0])):
        next_map = {i: j for i, j in arcs}
        if 0 not in next_map:
            route_nodes = [f"UNRESOLVED_ARCS::{arcs}"]
        else:
            route_nodes = [0]
            visited_arcs = set()
            cur = 0
            while cur in next_map and (cur, next_map[cur]) not in visited_arcs:
                nxt = next_map[cur]
                visited_arcs.add((cur, nxt))
                route_nodes.append(nxt)
                cur = nxt
                if cur == 0:
                    break

        route_labels = [
            "CW" if n == 0 else mapping.id_to_store[n] if isinstance(n, int) else str(n)
            for n in route_nodes
        ]
        direct_qty = 0.0
        lt_qty = 0.0
        product_flow_summary: Dict[str, float] = {}

        for i, j in arcs:
            for p_id in mapping.instance.P:
                sku = mapping.id_to_product[p_id]
                q_val = float(q[p_id, i, j, v, t].X if artifact.model.SolCount > 0 else 0.0)
                if q_val > 1e-9:
                    product_flow_summary[sku] = product_flow_summary.get(sku, 0.0) + q_val
                    if i == 0 and j != 0:
                        direct_qty += q_val
                if i != 0 and j != 0:
                    y_val = float(y[p_id, i, j, v, t].X if artifact.model.SolCount > 0 else 0.0)
                    if y_val > 1e-9:
                        lt_qty += y_val

        rows.append({
            "period": t,
            "vehicle": f"V{v}",
            "route": " -> ".join(route_labels),
            "arcs": str([("CW" if i == 0 else mapping.id_to_store[i], "CW" if j == 0 else mapping.id_to_store[j]) for i, j in arcs]),
            "total_direct_qty": round(direct_qty, 6),
            "total_lt_qty": round(lt_qty, 6),
            "product_flow_summary": str({k: round(vv, 6) for k, vv in product_flow_summary.items()}),
        })
    return pd.DataFrame(rows)


def compute_objective_breakdown(artifact: SolveArtifacts, mapping: MappingBundle) -> Dict[str, float]:
    inst = mapping.instance
    I = artifact.vars["I"]
    S = artifact.vars["S"]
    x = artifact.vars["x"]
    y = artifact.vars["y"]
    if artifact.model.SolCount <= 0:
        return {
            "holding_cost": math.nan,
            "routing_cost": math.nan,
            "transshipment_cost": math.nan,
            "shortage_cost": math.nan,
            "total": math.nan,
        }

    holding_cost = sum(
        inst.h[p, i] * float(I[p, i, t].X)
        for p in inst.P for i in inst.N0 for t in inst.H
    )
    routing_cost = sum(
        inst.alpha * inst.d[i, j] * float(x[i, j, v, t].X)
        for i in inst.N0 for j in inst.N0 if i != j for v in inst.V for t in inst.H
    )
    transshipment_cost = sum(
        inst.b[i, j] * float(y[p, i, j, v, t].X)
        for p in inst.P for i in inst.N for j in inst.N if i != j for v in inst.V for t in inst.H
    )
    shortage_cost = sum(
        inst.f[p, i] * float(S[p, i, t].X)
        for p in inst.P for i in inst.N for t in inst.H
    )
    return {
        "holding_cost": float(holding_cost),
        "routing_cost": float(routing_cost),
        "transshipment_cost": float(transshipment_cost),
        "shortage_cost": float(shortage_cost),
        "total": float(holding_cost + routing_cost + transshipment_cost + shortage_cost),
    }


def artifact_to_lt_detail_df(artifact: SolveArtifacts, mapping: MappingBundle) -> pd.DataFrame:
    y = artifact.vars["y"]
    rows = []
    for p_id in mapping.instance.P:
        sku = mapping.id_to_product[p_id]
        for i in mapping.instance.N:
            for j in mapping.instance.N:
                if i == j:
                    continue
                for v in mapping.instance.V:
                    for t in mapping.instance.H:
                        qty = float(y[p_id, i, j, v, t].X if artifact.model.SolCount > 0 else 0.0)
                        if qty > 1e-9:
                            rows.append({
                                "period": t,
                                "vehicle": f"V{v}",
                                "from_store": mapping.id_to_store[i],
                                "to_store": mapping.id_to_store[j],
                                "sku": sku,
                                "lt_qty": round(qty, 6),
                            })
    return pd.DataFrame(rows)


LT_PLAN_COLUMNS = [
    "source",
    "period",
    "vehicle",
    "from_store",
    "to_store",
    "sku",
    "lt_qty",
    "lt_unit_cost",
    "lt_fixed_cost",
    "lt_total_cost",
    "pattern_id",
    "lambda_value",
]


def artifact_to_lt_plan_df(artifact: SolveArtifacts, mapping: MappingBundle) -> pd.DataFrame:
    inst = mapping.instance
    y = artifact.vars["y"]
    rows = []
    for p_id in inst.P:
        sku = mapping.id_to_product[p_id]
        for i in inst.N:
            for j in inst.N:
                if i == j:
                    continue
                from_store = mapping.id_to_store[i]
                to_store = mapping.id_to_store[j]
                for v in inst.V:
                    for t in inst.H:
                        qty = float(y[p_id, i, j, v, t].X if artifact.model.SolCount > 0 else 0.0)
                        if qty <= 1e-9:
                            continue
                        unit_cost = float(inst.b[(i, j)])
                        rows.append({
                            "source": "Achamrah_Matheuristic",
                            "period": t,
                            "vehicle": f"V{v}",
                            "from_store": from_store,
                            "to_store": to_store,
                            "sku": sku,
                            "lt_qty": round(qty, 6),
                            "lt_unit_cost": round(unit_cost, 6),
                            "lt_fixed_cost": 0.0,
                            "lt_total_cost": round(unit_cost * qty, 6),
                            "pattern_id": "",
                            "lambda_value": 1.0,
                        })
    return pd.DataFrame(rows, columns=LT_PLAN_COLUMNS)


def print_lt_plan(lt_plan_df: pd.DataFrame, title: str = "Lateral Transshipment Plan") -> None:
    print(f"\n[{title}]")
    if lt_plan_df.empty:
        print("  No lateral transshipment moves selected.")
        return
    for _, row in lt_plan_df.sort_values(["period", "sku", "from_store", "to_store", "vehicle", "pattern_id"]).iterrows():
        vehicle = row["vehicle"] if str(row["vehicle"]) else "-"
        print(
            f"  source={row['source']} | period={row['period']} | vehicle={vehicle} "
            f"| {row['from_store']} -> {row['to_store']} | sku={row['sku']} "
            f"| qty={row['lt_qty']:.6f} | unit_cost={row['lt_unit_cost']:.6f} "
            f"| fixed_cost={row['lt_fixed_cost']:.6f} | total_cost={row['lt_total_cost']:.6f} "
            f"| pattern={row['pattern_id']} | lambda={row['lambda_value']:.6f}"
        )


def compute_validation_metrics(comp: pd.DataFrame) -> Dict[str, float]:
    df = comp.copy()
    if df.empty:
        return {"MAE": math.nan, "RMSE": math.nan, "Bias": math.nan, "MAPE": math.nan}
    df["abs_error"] = (df["predicted_end_qty"] - df["actual_end_qty"]).abs()
    df["sq_error"] = (df["predicted_end_qty"] - df["actual_end_qty"]) ** 2
    df["pct_error"] = df.apply(
        lambda r: abs(r["predicted_end_qty"] - r["actual_end_qty"]) / abs(r["actual_end_qty"])
        if r["actual_end_qty"] not in [0, 0.0] else math.nan,
        axis=1,
    )
    return {
        "MAE": float(df["abs_error"].mean()),
        "RMSE": float(math.sqrt(df["sq_error"].mean())),
        "Bias": float((df["predicted_end_qty"] - df["actual_end_qty"]).mean()),
        "MAPE": float(df["pct_error"].dropna().mean()) if df["pct_error"].notna().any() else math.nan,
    }


class AchamrahThesisFormatPipeline:
    def __init__(
        self,
        mapping: MappingBundle,
        params: Optional[HeuristicParams] = None,
        allow_lateral_transshipment: bool = True,
    ):
        self.mapping = mapping
        self.params = params or HeuristicParams()
        self.allow_lateral_transshipment = allow_lateral_transshipment
        self.solver = AchamrahIRPTSolver(mapping.instance, self.params)

    def run(self) -> Dict[str, Any]:
        if self.allow_lateral_transshipment:
            result = self.solver.solve_full_matheuristic()
            artifact = result.final_solution or result.constructive_solution
            if artifact is None:
                raise RuntimeError("Solver did not return a constructive or final solution.")
        else:
            result = None
            artifact = self.solver.solve_model(
                relaxed=True,
                time_limit=self.params.full_time_limit,
                mip_gap=None,
                allow_lateral_transshipment=False,
                model_name="Achamrah_No_LT_Baseline_IRPT",
            )
            if artifact.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) or not math.isfinite(artifact.objective):
                raise RuntimeError(f"Solver did not return a feasible no-LT baseline solution. Status={artifact.status}")

        predicted_df = artifact_to_predicted_inventory_df(artifact, self.mapping)
        route_df = artifact_to_route_df(artifact, self.mapping)
        lt_detail_df = artifact_to_lt_detail_df(artifact, self.mapping)
        lt_plan_df = artifact_to_lt_plan_df(artifact, self.mapping)
        objective_breakdown = compute_objective_breakdown(artifact, self.mapping)
        comparison_df = predicted_df.merge(
            self.mapping.validation_target,
            on=["store", "sku", "period"],
            how="inner",
        )
        comparison_df["error"] = comparison_df["predicted_end_qty"] - comparison_df["actual_end_qty"]
        metrics = compute_validation_metrics(comparison_df)

        summary = {
            "status": int(artifact.status),
            "objective": float(artifact.objective),
            "best_objective": float(result.best_objective) if result is not None else float(artifact.objective),
            "constructive_objective": float(result.constructive_solution.objective) if result is not None and result.constructive_solution else math.nan,
            "final_objective": float(result.final_solution.objective) if result is not None and result.final_solution else math.nan,
            "baseline_objective": float(artifact.objective) if result is None else math.nan,
            "allow_lateral_transshipment": bool(self.allow_lateral_transshipment),
            "matheuristic_used": bool(result is not None),
            "history_length": int(len(result.history)) if result is not None else 0,
            "objective_breakdown": objective_breakdown,
            "n_routes": int(len(route_df)),
            "n_lt_moves": int(len(lt_detail_df)),
            "validation_metrics": metrics,
        }

        return {
            "result": result,
            "artifact": artifact,
            "predicted_inventory": predicted_df,
            "routes": route_df,
            "lt_detail": lt_detail_df,
            "lt_plan": lt_plan_df,
            "validation_comparison": comparison_df,
            "validation_metrics": metrics,
            "summary": summary,
        }

    def export_outputs(self, outputs: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "validation_target": str(out_dir / "achamrah_validation_target.csv"),
            "predicted_inventory": str(out_dir / "achamrah_predicted_inventory.csv"),
            "routes": str(out_dir / "achamrah_routes.csv"),
            "baseline_routes": str(out_dir / "irp_baseline_routes.csv"),
            "irp_predicted_inventory": str(out_dir / "irp_predicted_inventory.csv"),
            "lt_detail": str(out_dir / "achamrah_lt_detail.csv"),
            "lt_plan": str(out_dir / "achamrah_lt_plan.csv"),
            "irp_lt_plan": str(out_dir / "irp_lt_plan.csv"),
            "validation_comparison": str(out_dir / "achamrah_validation_comparison.csv"),
            "irp_validation_comparison": str(out_dir / "irp_validation_comparison.csv"),
            "summary": str(out_dir / "achamrah_summary.txt"),
        }

        self.mapping.validation_target.to_csv(paths["validation_target"], index=False)
        outputs["predicted_inventory"].to_csv(paths["predicted_inventory"], index=False)
        outputs["predicted_inventory"].to_csv(paths["irp_predicted_inventory"], index=False)
        outputs["routes"].to_csv(paths["routes"], index=False)
        outputs["routes"].to_csv(paths["baseline_routes"], index=False)
        outputs["lt_detail"].to_csv(paths["lt_detail"], index=False)
        outputs["lt_plan"].to_csv(paths["lt_plan"], index=False)
        outputs["lt_plan"].to_csv(paths["irp_lt_plan"], index=False)
        outputs["validation_comparison"].to_csv(paths["validation_comparison"], index=False)
        outputs["validation_comparison"].to_csv(paths["irp_validation_comparison"], index=False)

        with open(paths["summary"], "w", encoding="utf-8") as f:
            f.write("Mapped dataset metadata:\n")
            f.write(pprint.pformat(self.mapping.metadata))
            f.write("\n\nAchamrah model summary:\n")
            f.write(pprint.pformat(outputs["summary"]))
            f.write("\n")

        return paths


def demo_main() -> None:
    EXCEL_PATH = "/Users/trannguyenhung/Downloads/1BISCR501V_90100140_20260313-185009126.xlsx"
    OUTPUT_DIR = "/Users/trannguyenhung/Documents/THESIS/Code/Output Code Key Ref Achamrah"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    mapper = DatasetToAchamrahMapper(
        excel_path=EXCEL_PATH,
        sheet_name="Sheet1",
        store_limit=3,
        sku_limit=2,
        start_date=None,
        end_date=None,
    )
    mapping = mapper.build_instance(
        wh_inventory_multiplier=2.5,
        store_capacity_multiplier=1.5,
        shortage_cost_rate=0.25,
        holding_cost_rate=0.01,
        vehicle_count=1,
        vehicle_capacity=120.0,
        alpha=1.0,
        cw_replenishment_factor=0.6,
        cw_capacity_factor=2.0,
    )

    print("Mapped dataset metadata:")
    pprint.pprint(mapping.metadata)

    params = HeuristicParams(
        initial_temperature=92,
        final_temperature=4.2,
        cooling_ratio=0.96,
        crossover_probability=0.84,
        mutation_probability=0.37,
        population_size=30,
        iterations_per_temp=30,
        constructive_time_limit=20,
        improvement_time_limit=30,
        full_time_limit=60,
        seed=0,
    )

    pipeline = AchamrahThesisFormatPipeline(mapping, params=params)
    outputs = pipeline.run()
    print_lt_plan(outputs["lt_plan"], title="Achamrah Lateral Transshipment Plan")
    export_paths = pipeline.export_outputs(outputs, OUTPUT_DIR)

    print("\nAchamrah model summary:")
    pprint.pprint(outputs["summary"])
    print("\nSaved files:")
    pprint.pprint(export_paths)


if __name__ == "__main__":
    demo_main()
