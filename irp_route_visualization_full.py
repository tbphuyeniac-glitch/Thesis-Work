
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

    for (v, t), arcs in arcs_by_v_t.items():
        next_node = {i: j for (i, j) in arcs}

        route = ["CW"]
        current = "CW"
        visited = set()

        while current in next_node and current not in visited:
            visited.add(current)
            nxt = next_node[current]
            route.append(nxt)
            current = nxt
            if current == "CW":
                break

        routes.append({
            "period": t,
            "vehicle": v,
            "route": route
        })

    return routes


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
        print(f"t={r['period']} | v={r['vehicle']} | route={r['route']}")

    visualize_routes_by_period(routes)

    return baseline_sol, routes


if __name__ == "__main__":
    print("Plug your data + model into run_pipeline(data, AchamrahFullIRPTModel)")
