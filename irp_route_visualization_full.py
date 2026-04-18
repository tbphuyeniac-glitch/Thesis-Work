
import matplotlib.pyplot as plt
import math
import random

# =========================
# ROUTE EXTRACTION
# =========================
def extract_routes_from_solution(solution):
    routes = []
    arcs_by_v_t = {}

    for (i, j, v, t), val in solution.x.items():
        if val > 0.5:
            arcs_by_v_t.setdefault((v, t), []).append((i, j))

    for (v, t), arcs in sorted(arcs_by_v_t.items(), key=lambda item: (item[0][1], item[0][0])):
        outgoing = {}
        incoming = {}
        for i, j in arcs:
            outgoing.setdefault(i, []).append(j)
            incoming.setdefault(j, []).append(i)

        degree_warnings = []
        for node in sorted(set(outgoing) | set(incoming)):
            in_deg = len(incoming.get(node, []))
            out_deg = len(outgoing.get(node, []))
            if in_deg > 1 or out_deg > 1:
                degree_warnings.append(f"{node}:in={in_deg},out={out_deg}")

        next_node = {i: js[0] for i, js in outgoing.items() if js}
        if "CW" not in next_node:
            routes.append({
                "period": t,
                "vehicle": v,
                "route": [f"UNRESOLVED_ARCS::{arcs}"],
                "arcs": arcs,
                "degree_warnings": degree_warnings,
                "unvisited_arcs": arcs,
                "load_departure": _get_solution_load(solution, "CW", v, t),
            })
            continue

        route = ["CW"]
        current = "CW"
        visited = set()

        while current in next_node and (current, next_node[current]) not in visited:
            nxt = next_node[current]
            visited.add((current, nxt))
            route.append(nxt)
            current = nxt
            if current == "CW":
                break
        unvisited_arcs = [arc for arc in arcs if arc not in visited]

        total_direct_qty = 0.0
        if hasattr(solution, "deliv"):
            total_direct_qty = sum(
                float(qty)
                for (s, p, vv, tt), qty in solution.deliv.items()
                if vv == v and tt == t and qty > 1e-9
            )

        routes.append({
            "period": t,
            "vehicle": v,
            "route": route,
            "arcs": arcs,
            "degree_warnings": degree_warnings,
            "unvisited_arcs": unvisited_arcs,
            "total_direct_qty": round(total_direct_qty, 6),
            "load_departure": _get_solution_load(solution, "CW", v, t),
        })

    return routes


def _get_solution_load(solution, node, vehicle, period):
    if not hasattr(solution, "load"):
        return 0.0
    return round(float(solution.load.get((node, vehicle, period), 0.0)), 6)


# =========================
# VISUALIZATION
# =========================
def visualize_routes_by_period(routes):
    if not routes:
        print("No routes to visualize")
        return

    nodes = set()
    for r in routes:
        nodes.update(r["route"])
    nodes = list(nodes)

    pos = {}
    if "CW" in nodes:
        pos["CW"] = (0, 0)
        nodes.remove("CW")

    n = len(nodes)
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / max(n, 1)
        pos[node] = (math.cos(angle), math.sin(angle))

    period_groups = {}
    for r in routes:
        period_groups.setdefault(r["period"], []).append(r)

    for t, route_list in period_groups.items():
        plt.figure(figsize=(6, 6))

        for node, (x, y) in pos.items():
            plt.scatter(x, y)
            plt.text(x + 0.02, y + 0.02, node, fontsize=9)

        vehicles = list(set(r["vehicle"] for r in route_list))
        color_map = {
            v: (random.random(), random.random(), random.random())
            for v in vehicles
        }

        for r in route_list:
            route = r["route"]
            v = r["vehicle"]
            color = color_map[v]

            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                x1, y1 = pos[a]
                x2, y2 = pos[b]

                plt.arrow(
                    x1, y1,
                    x2 - x1, y2 - y1,
                    length_includes_head=True,
                    head_width=0.05,
                    alpha=0.7,
                    color=color
                )

        plt.title(f"Period {t} (All Vehicles)")
        plt.axis("off")

        for v, c in color_map.items():
            plt.plot([], [], color=c, label=f"Vehicle {v}")
        plt.legend()

        plt.show()


# =========================
# MAIN PIPELINE (HOOK INTO YOUR MODEL)
# =========================
def run_pipeline(data, AchamrahFullIRPTModel):
    print("=" * 60)
    print("BASELINE FULL MODEL (NO LT, WITH 16–20)")
    print("=" * 60)

    baseline_sol = AchamrahFullIRPTModel(data).solve(
        msg=True,
        time_limit=60,
        enforce_integer_flows=False,
        add_valid_16_20=True,
        allow_lateral_transshipment=False,
    )

    print("Objective:", baseline_sol.objective)

    routes = extract_routes_from_solution(baseline_sol)

    print("\nExtracted Routes:")
    for r in routes:
        print(
            f"t={r['period']} | v={r['vehicle']} | route={r['route']} "
            f"| direct_qty={r.get('total_direct_qty', 0.0)} "
            f"| load_departure={r.get('load_departure', 0.0)}"
        )
        if r.get("degree_warnings") or r.get("unvisited_arcs"):
            print(
                f"  route_warning degree={r.get('degree_warnings', [])} "
                f"unvisited_arcs={r.get('unvisited_arcs', [])}"
            )

    visualize_routes_by_period(routes)

    return baseline_sol, routes


if __name__ == "__main__":
    print("Plug your data + model into run_pipeline(data, AchamrahFullIRPTModel)")
