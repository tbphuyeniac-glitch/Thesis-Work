"""
achamrah_ga_sa_baseline.py
============================

Gurobi-free, pure-heuristic GA+SA matheuristic for the IRP baseline
(DC -> stores, no lateral transshipment).

Design:
- Constructive phase: nearest-neighbor + demand-weighted greedy (no RMILP)
- Improvement phase: same SA outer loop + GA operators as Achamrah 2022,
  but with a pure-Python forward-simulation evaluator instead of FMILP
- GA operators (crossover, mutate, two_opt, repair) are copied / adapted
  from achamrah_2022_irpt_matheuristic.py for fidelity to the paper

Apples-to-apples vs ALNS: both are pure heuristics, no Gurobi calls.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

Node = int
Product = int
Period = int
Vehicle = int
Arc = Tuple[int, int]
RouteKey = Tuple[int, int]
Chromosome = Dict[RouteKey, List[int]]


@dataclass
class BaselineInstance:
    """Lightweight instance for the Gurobi-free GA/SA baseline solver."""
    N: List[Node]                                        # POS nodes (1..n)
    P: List[Product]                                     # products
    H: List[Period]                                      # periods (sorted)
    V: List[Vehicle]                                     # vehicles
    alpha: float
    Q: float                                             # vehicle capacity
    d: Dict[Arc, float]                                  # distance over N0
    h: Dict[Tuple[int, int], float]                      # holding cost
    C: Dict[int, float]                                  # node capacity
    I0: Dict[Tuple[int, int], float]                     # initial inventory
    D: Dict[Tuple[int, int, int], float]                 # demand (p,i,t)
    g: Dict[Tuple[int, int], float]                      # CW replenishment
    f: Dict[Tuple[int, int], float]                      # lost-sales cost
    # Direct CW ship cost per unit per (store, product) — matches Gurobi objective
    ship_cost_cw: Dict[Tuple[int, int], float] = field(default_factory=dict)
    vehicle_fixed_cost: float = 0.0
    # CW dispatch cycle: deliveries only allowed every <cycle> periods (offset 0).
    # Matches BaselineALNSModel / AchamrahFullIRPTModel default of 5.
    cw_dispatch_cycle: Optional[int] = 5
    name: str = "baseline"

    def dispatch_periods(self) -> set:
        """Return the set of periods on which deliveries are permitted."""
        if self.cw_dispatch_cycle is None or int(self.cw_dispatch_cycle) <= 1:
            return set(self.H)
        cycle = int(self.cw_dispatch_cycle)
        first = min(self.H)
        return {t for t in self.H if (t - first) % cycle == 0}

    @property
    def N0(self) -> List[int]:
        return [0] + list(self.N)


@dataclass
class GASAResult:
    objective: float
    routes: Chromosome
    best_history: List[Tuple[float, float]] = field(default_factory=list)
    constructive_runtime_seconds: float = 0.0
    improvement_runtime_seconds: float = 0.0
    total_runtime_seconds: float = 0.0
    new_best_count: int = 0
    routing_cost: float = 0.0
    holding_cost: float = 0.0
    shortage_cost: float = 0.0
    shortage_qty: float = 0.0   # total shortage in units (for service-level computation)
    ship_cost: float = 0.0
    vehicle_fixed_cost: float = 0.0


@dataclass
class GASAParams:
    initial_temperature: float = 92.0
    final_temperature: float = 4.2
    cooling_ratio: float = 0.96
    crossover_probability: float = 0.84
    mutation_probability: float = 0.37
    population_size: int = 30
    iterations_per_temp: int = 30
    seed: int = 0
    max_iterations: int = 1000          # hard cap for benchmarking
    time_limit: Optional[float] = None  # None = no time limit


class BaselineGASASolver:
    """
    Pure-heuristic GA+SA solver:
      - solve()                     — public entry point, returns GASAResult
      - _constructive_greedy()      — nearest-neighbor + demand-weighted seeding
      - _evaluate_routes(routes)    — forward-simulation objective (no Gurobi)
      - _crossover, _mutate, _two_opt, _repair_chromosome — GA operators
    """

    def __init__(self, inst: BaselineInstance, params: Optional[GASAParams] = None):
        self.inst = inst
        self.params = params or GASAParams()
        self.rng = random.Random(self.params.seed)
        self._dispatch = inst.dispatch_periods()

    # =========================================================
    # Public API
    # =========================================================
    def solve(self, initial_routes: Optional[Chromosome] = None) -> GASAResult:
        """
        initial_routes: if provided, skip constructive phase and seed the
        population from these routes (e.g. extracted from a Gurobi solution).
        """
        start = time.time()

        t0 = time.time()
        if initial_routes is not None:
            seed_routes = initial_routes
        else:
            seed_routes = self._constructive_greedy()
        constructive_runtime = time.time() - t0
        seed_obj = self._evaluate_routes(seed_routes)["total"]

        # Population seeded by perturbing the initial solution
        population: List[Tuple[Chromosome, float]] = [(seed_routes, seed_obj)]
        for _ in range(max(1, self.params.population_size - 1)):
            perturbed = self._two_opt(seed_routes)
            population.append((perturbed, self._evaluate_routes(perturbed)["total"]))

        current_routes, current_obj = self._copy(seed_routes), seed_obj
        best_routes,    best_obj    = self._copy(seed_routes), seed_obj
        new_best_count = 0
        history: List[Tuple[float, float]] = []

        T = self.params.initial_temperature
        iter_count = 0
        improvement_start = time.time()

        def _time_exceeded() -> bool:
            return (self.params.time_limit is not None
                    and (time.time() - start) >= self.params.time_limit)

        # Outer loop: SA with reheat — when T cools to final_temperature,
        # reheat back to initial_temperature (restarting from best) and
        # continue until time_limit or max_iterations is reached.
        while not _time_exceeded() and iter_count <= self.params.max_iterations:
            for _ in range(self.params.iterations_per_temp):
                iter_count += 1
                if iter_count > self.params.max_iterations or _time_exceeded():
                    break

                # GA: select parents → crossover → mutate
                p1 = self._tournament(population)
                p2 = self._tournament(population)
                if self.rng.random() < self.params.crossover_probability:
                    child, _ = self._crossover(p1, p2)
                else:
                    child = self._copy(p1)
                if self.rng.random() < self.params.mutation_probability:
                    child = self._mutate(child)

                child_obj = self._evaluate_routes(child)["total"]

                # SA Metropolis acceptance
                delta = child_obj - current_obj
                if delta < 0 or self.rng.random() < math.exp(-delta / max(1e-9, T)):
                    current_routes = child
                    current_obj    = child_obj
                    population.append((child, child_obj))
                    population.sort(key=lambda x: x[1])
                    population = population[: self.params.population_size]

                if current_obj < best_obj - 1e-9:
                    best_routes = self._copy(current_routes)
                    best_obj    = current_obj
                    new_best_count += 1

            history.append((T, best_obj))
            if iter_count > self.params.max_iterations or _time_exceeded():
                break

            T *= self.params.cooling_ratio

            # Reheat: when cooled to final_temperature, restart from best
            if T <= self.params.final_temperature:
                T = self.params.initial_temperature
                current_routes = self._copy(best_routes)
                current_obj    = best_obj

        improvement_runtime = time.time() - improvement_start

        # Final breakdown
        bd = self._evaluate_routes(best_routes)
        return GASAResult(
            objective=best_obj,
            routes=best_routes,
            best_history=history,
            constructive_runtime_seconds=constructive_runtime,
            improvement_runtime_seconds=improvement_runtime,
            total_runtime_seconds=time.time() - start,
            new_best_count=new_best_count,
            routing_cost=bd["routing"],
            holding_cost=bd["holding"],
            shortage_cost=bd["shortage"],
            shortage_qty=bd["shortage_qty"],
            ship_cost=bd["ship"],
            vehicle_fixed_cost=bd["vehicle_fixed"],
        )

    # =========================================================
    # Pure-Python evaluator (no Gurobi)
    # =========================================================
    def _evaluate_routes(self, routes: Chromosome) -> Dict[str, float]:
        """Forward-simulate inventory; return cost breakdown.

        Cost components match AchamrahFullIRPTModel objective exactly:
          routing + holding + shortage + ship_cost_cw + vehicle_fixed_cost
        """
        inst = self.inst
        H = sorted(inst.H)

        # Pre-compute next-dispatch period for each t — deliveries on dispatch
        # day must cover demand until next dispatch day, since no shipment in between.
        next_dispatch_after: Dict[int, int] = {}
        for idx, t in enumerate(H):
            nxt = next(
                (H[k] for k in range(idx + 1, len(H)) if H[k] in self._dispatch),
                H[-1] + 1,
            )
            next_dispatch_after[t] = nxt

        # 1) Routing cost + vehicle fixed cost (deterministic from routes)
        routing = 0.0
        veh_fixed = 0.0
        for (t, v), seq in routes.items():
            if not seq:
                continue
            veh_fixed += inst.vehicle_fixed_cost
            prev = 0
            for node in seq:
                routing += inst.alpha * inst.d.get((prev, node), 0.0)
                prev = node
            routing += inst.alpha * inst.d.get((prev, 0), 0.0)

        # 2) State + inventory simulation
        inv_store: Dict[Tuple[int, int], float] = {
            (p, i): float(inst.I0.get((p, i), 0.0)) for p in inst.P for i in inst.N
        }
        inv_cw: Dict[int, float] = {p: float(inst.I0.get((p, 0), 0.0)) for p in inst.P}

        holding      = 0.0
        shortage     = 0.0
        shortage_qty = 0.0
        ship_cost    = 0.0

        for t in H:
            # 1. CW replenishment
            for p in inst.P:
                inv_cw[p] += float(inst.g.get((p, t), 0.0))

            # 2. Vehicle deliveries (greedy: per stop, fill highest-need first).
            # Skip entirely on non-dispatch periods — Gurobi/ALNS forbid Qdir there.
            if t not in self._dispatch:
                for p in inst.P:
                    holding += float(inst.h.get((p, 0), 0.0)) * inv_cw[p]
                # Demand still consumed; shortage accrues on non-dispatch periods too
                for store in inst.N:
                    for p in inst.P:
                        d_t = float(inst.D.get((p, store, t), 0.0))
                        if inv_store[(p, store)] >= d_t:
                            inv_store[(p, store)] -= d_t
                        else:
                            _sh = d_t - inv_store[(p, store)]
                            shortage     += float(inst.f.get((p, store), 0.25)) * _sh
                            shortage_qty += _sh
                            inv_store[(p, store)] = 0.0
                # End-of-period store holding cost
                for p in inst.P:
                    for store in inst.N:
                        holding += float(inst.h.get((p, store), 0.0)) * inv_store[(p, store)]
                continue

            # Cumulative demand from current dispatch period to next dispatch
            # (covers all non-dispatch periods in between since no shipments occur).
            t_end = next_dispatch_after[t]

            def _cycle_demand(p: int, store: int) -> float:
                return sum(
                    float(inst.D.get((p, store, tt), 0.0))
                    for tt in H if t <= tt < t_end
                )

            for v in inst.V:
                seq = routes.get((t, v), [])
                if not seq:
                    continue
                veh_remaining = float(inst.Q)
                for store in seq:
                    if veh_remaining <= 1e-9:
                        break
                    # Priority by cycle-need
                    prods_sorted = sorted(
                        inst.P,
                        key=lambda p: max(0.0, _cycle_demand(p, store) - inv_store.get((p, store), 0.0)),
                        reverse=True,
                    )
                    for p in prods_sorted:
                        if veh_remaining <= 1e-9 or inv_cw[p] <= 1e-9:
                            continue
                        target = _cycle_demand(p, store)
                        need = max(0.0, target - inv_store[(p, store)])
                        used_cap = sum(inv_store.get((pp, store), 0.0) for pp in inst.P)
                        cap_room = max(0.0, float(inst.C.get(store, float("inf"))) - used_cap)
                        deliver = min(need, veh_remaining, inv_cw[p], cap_room)
                        if deliver > 1e-9:
                            inv_store[(p, store)] += deliver
                            inv_cw[p]             -= deliver
                            veh_remaining         -= deliver
                            ship_cost             += float(inst.ship_cost_cw.get((p, store), 0.0)) * deliver

            # 3. Demand consumption + shortage
            for store in inst.N:
                for p in inst.P:
                    d_t = float(inst.D.get((p, store, t), 0.0))
                    if inv_store[(p, store)] >= d_t:
                        inv_store[(p, store)] -= d_t
                    else:
                        _sh = d_t - inv_store[(p, store)]
                        shortage     += float(inst.f.get((p, store), 0.25)) * _sh
                        shortage_qty += _sh
                        inv_store[(p, store)] = 0.0

            # 4. End-of-period holding cost
            for p in inst.P:
                holding += float(inst.h.get((p, 0), 0.0)) * inv_cw[p]
                for store in inst.N:
                    holding += float(inst.h.get((p, store), 0.0)) * inv_store[(p, store)]

        total = routing + holding + shortage + ship_cost + veh_fixed
        return {
            "routing":       routing,
            "holding":       holding,
            "shortage":      shortage,
            "shortage_qty":  shortage_qty,
            "ship":          ship_cost,
            "vehicle_fixed": veh_fixed,
            "total":         total,
        }

    # =========================================================
    # Constructive: nearest-neighbor with demand-weighted scoring
    # =========================================================
    def _constructive_greedy(self) -> Chromosome:
        inst = self.inst
        routes: Chromosome = {(t, v): [] for t in inst.H for v in inst.V}

        inv_store = {(p, i): float(inst.I0.get((p, i), 0.0)) for p in inst.P for i in inst.N}

        for t in sorted(inst.H):
            # On non-dispatch periods, no routes; just consume demand
            if t not in self._dispatch:
                for store in inst.N:
                    for p in inst.P:
                        d_t = float(inst.D.get((p, store, t), 0.0))
                        inv_store[(p, store)] = max(0.0, inv_store[(p, store)] - d_t)
                continue

            # Aggregate demand from current period until next dispatch
            sorted_h = sorted(inst.H)
            t_idx = sorted_h.index(t)
            next_disp = next(
                (sorted_h[k] for k in range(t_idx + 1, len(sorted_h)) if sorted_h[k] in self._dispatch),
                sorted_h[-1] + 1,
            )
            covered_periods = [tt for tt in sorted_h if t <= tt < next_disp]

            need_by_store: Dict[int, float] = {}
            for store in inst.N:
                total_need = 0.0
                for p in inst.P:
                    cum_demand = sum(float(inst.D.get((p, store, tt), 0.0)) for tt in covered_periods)
                    total_need += max(0.0, cum_demand - inv_store[(p, store)])
                need_by_store[store] = total_need

            unvisited = {s for s, n in need_by_store.items() if n > 1e-6}

            for v in inst.V:
                if not unvisited:
                    break
                seq: List[int] = []
                current = 0
                cap_remaining = float(inst.Q)
                while unvisited and cap_remaining > 1.0:
                    best = None
                    best_score = -1.0
                    for s in unvisited:
                        dist = max(1e-3, float(inst.d.get((current, s), 1.0)))
                        score = need_by_store[s] / dist
                        if score > best_score:
                            best, best_score = s, score
                    if best is None:
                        break
                    seq.append(best)
                    cap_remaining -= min(cap_remaining, need_by_store[best])
                    unvisited.discard(best)
                    current = best
                routes[(t, v)] = seq

            # Update inventory simulation for visited stores (delivers up to current need)
            for v in inst.V:
                for store in routes[(t, v)]:
                    for p in inst.P:
                        d_t = float(inst.D.get((p, store, t), 0.0))
                        inv_store[(p, store)] = max(inv_store[(p, store)], d_t)
            for store in inst.N:
                for p in inst.P:
                    d_t = float(inst.D.get((p, store, t), 0.0))
                    inv_store[(p, store)] = max(0.0, inv_store[(p, store)] - d_t)

        return routes

    # =========================================================
    # GA operators (adapted from Achamrah 2022)
    # =========================================================
    def _tournament(self, pop: List[Tuple[Chromosome, float]]) -> Chromosome:
        a, b = self.rng.sample(pop, k=2) if len(pop) >= 2 else (pop[0], pop[0])
        return self._copy(a[0] if a[1] <= b[1] else b[0])

    def _crossover(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        c1, c2 = self._copy(p1), self._copy(p2)
        for key in sorted(c1.keys()):
            seq1, seq2 = c1[key][:], c2[key][:]
            n = min(len(seq1), len(seq2))
            if n < 2:
                continue
            i, j = sorted(self.rng.sample(range(n), 2))
            seq1[i:j + 1], seq2[i:j + 1] = seq2[i:j + 1], seq1[i:j + 1]
            c1[key] = self._repair_seq(seq1)
            c2[key] = self._repair_seq(seq2)
        return self._repair_chromosome(c1), self._repair_chromosome(c2)

    def _mutate(self, chrom: Chromosome) -> Chromosome:
        out = self._copy(chrom)
        keys = [k for k, s in out.items() if len(s) >= 2]
        if not keys:
            return out
        key = self.rng.choice(keys)
        seq = out[key][:]
        i, j = sorted(self.rng.sample(range(len(seq)), 2))
        seq[i:j + 1] = list(reversed(seq[i:j + 1]))
        out[key] = seq
        return self._repair_chromosome(out)

    def _two_opt(self, chrom: Chromosome) -> Chromosome:
        out = self._copy(chrom)
        cand_keys = [k for k, s in out.items() if len(s) >= 2]
        if not cand_keys:
            return out
        key = self.rng.choice(cand_keys)
        seq = out[key][:]
        best_seq, best_cost = seq[:], self._route_distance(seq)
        for _ in range(min(8, len(seq) ** 2)):
            i, j = sorted(self.rng.sample(range(len(seq)), 2))
            new_seq = seq[:]
            new_seq[i:j + 1] = list(reversed(new_seq[i:j + 1]))
            c = self._route_distance(new_seq)
            if c < best_cost:
                best_seq, best_cost = new_seq, c
        out[key] = best_seq

        # Inter-route swap within same period (50% chance)
        same_period = [k for k in out.keys() if k[0] == key[0] and len(out[k]) >= 1]
        if len(same_period) >= 2 and self.rng.random() < 0.5:
            k1, k2 = self.rng.sample(same_period, 2)
            i1 = self.rng.randrange(len(out[k1]))
            i2 = self.rng.randrange(len(out[k2]))
            out[k1][i1], out[k2][i2] = out[k2][i2], out[k1][i1]

        return self._repair_chromosome(out)

    def _repair_seq(self, seq: List[int]) -> List[int]:
        seen = set()
        out: List[int] = []
        for node in seq:
            if node not in seen and node in self.inst.N:
                out.append(node)
                seen.add(node)
        return out

    def _repair_chromosome(self, chrom: Chromosome) -> Chromosome:
        # Per period: each POS visited at most once across all vehicles.
        # Non-dispatch periods are forced to empty routes.
        for t in self.inst.H:
            if t not in self._dispatch:
                for v in self.inst.V:
                    chrom[(t, v)] = []
                continue
            seen_in_period = set()
            for v in self.inst.V:
                key = (t, v)
                cleaned: List[int] = []
                for node in chrom.get(key, []):
                    if node in seen_in_period or node not in self.inst.N:
                        continue
                    cleaned.append(node)
                    seen_in_period.add(node)
                chrom[key] = cleaned
        return chrom

    def _route_distance(self, seq: List[int]) -> float:
        if not seq:
            return 0.0
        d = self.inst.d
        total = d.get((0, seq[0]), 0.0)
        for i in range(len(seq) - 1):
            total += d.get((seq[i], seq[i + 1]), 0.0)
        total += d.get((seq[-1], 0), 0.0)
        return total

    @staticmethod
    def _copy(chrom: Chromosome) -> Chromosome:
        return {k: list(v) for k, v in chrom.items()}
