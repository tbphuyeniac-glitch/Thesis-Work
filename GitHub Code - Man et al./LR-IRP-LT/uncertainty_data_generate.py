import os
import numpy as np
from itertools import product
import random

# Set seed for reproducibility


# Parameters
# customer_counts = [5, 10, 15, 20, 25]
# product_counts = [1, 3, 5, 7, 9]
# vehicle_counts = [1, 3, 5]
# samples_per_combination = 5
customer_counts = [5]
product_counts = [9]
vehicle_counts = [5]
samples_per_combination = 5
rho = np.random.uniform(-1,1) # 扰动比例
# product_values=[0.1,1,5]
# product_volume=[0.1,1,5]
# precision_sigma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
# Output folder
output_folder = "rho_sensitivity"
os.makedirs(output_folder, exist_ok=True)


# Helper function to generate epsilon_im with constraints
def generate_epsilon(num_customers, num_products):
    epsilon = np.random.uniform(epsilon_lb, epsilon_ub, size=(num_customers, num_products))
    node_product_relationships = {}  # 记录节点-产品关系
    product_node_relationships = {}  # 记录产品-节点关系

    # 如果节点数大于3，构建冲突或协同关系
    if num_customers >= 3:
        # 随机选择2到num_customers个节点
        V_m = random.sample(range(num_customers), random.randint(2, num_customers))
        # 随机决定是冲突还是协同
        relation = random.choice(["conflict", "region"])
        if relation=="conflict":  # 冲突
            for m in range(num_products):
                epsilon[V_m, m] = np.random.dirichlet(np.ones(len(V_m))) * random.uniform(epsilon_lb, epsilon_ub)
                # 记录产品-节点关系
                product_node_relationships[f"product {m + 1}"] = (V_m, relation)
            if num_customers-len(V_m)>=2:
                # 随机选择2到num_customers个节点
                remaining_data = set(range(num_customers)) - set(V_m)
                remaining_data = list(remaining_data)
                V_m = random.sample(remaining_data, random.randint(2, len(remaining_data)))
                for m_ in range(num_products):
                    base = np.random.uniform(epsilon_lb, epsilon_ub)  # 基准扰动
                    for i in range(len(V_m) - 2):
                        epsilon[V_m[i], m_] = base + np.random.uniform(-0.1, 0.1)
                        epsilon[V_m[i + 1], m_] = epsilon[V_m[i], m_] + np.random.uniform(-0.1, 0.1)
                    # 记录产品-节点关系
                    product_node_relationships[f"product {m_ + 1}"] = (V_m, "region")
        else:  # 协同
            for m in range(num_products):
                base = np.random.uniform(epsilon_lb, epsilon_ub)  # 基准扰动
                for i in range(len(V_m) - 2):
                    epsilon[V_m[i], m] = base + np.random.uniform(-0.1, 0.1)
                    epsilon[V_m[i + 1], m] = epsilon[V_m[i], m] + np.random.uniform(-0.1, 0.1)
                # 记录产品-节点关系
                product_node_relationships[f"product {m + 1}"] = (V_m, relation)
                if num_customers - len(V_m) >= 2:
                    remaining_data = set(range(num_customers)) - set(V_m)
                    remaining_data = list(remaining_data)
                    V_m = random.sample(remaining_data, random.randint(2, len(remaining_data)))
                    for m_ in range(num_products):
                        epsilon[V_m, m_] = np.random.dirichlet(np.ones(len(V_m))) * random.uniform(epsilon_lb, epsilon_ub)
                        # 记录产品-节点关系
                        product_node_relationships[f"product {m_ + 1}"] = (V_m, "conflict")

    # 如果产品数大于3，构建冲突或协同关系
    if num_products >= 3:
        M_i = random.sample(range(num_products), random.randint(2, num_products))
        # 随机决定是冲突还是协同
        relation = random.choice(["comp", "sync"])
        if relation=="comp":  # 冲突
            for i in range(num_customers):
                epsilon[i, M_i] = np.random.dirichlet(np.ones(len(M_i))) * random.uniform(epsilon_lb, epsilon_ub)
                node_product_relationships[f"customer {i + 1}"] = (M_i, relation)
                if num_products - len(M_i) >= 2:
                    remaining_data = set(range(num_products)) - set(M_i)
                    remaining_data = list(remaining_data)
                    M_i = random.sample(remaining_data, random.randint(2, len(remaining_data)))
                    for i_ in range(num_customers):
                        base = np.random.uniform(epsilon_lb, epsilon_ub)  # 基准扰动
                        for m in range(len(M_i) - 2):
                            epsilon[i_, M_i[m]] = base + np.random.uniform(-0.1, 0.1)
                            epsilon[i_, M_i[m + 1]] = epsilon[i_, M_i[m]] + np.random.uniform(-0.1, 0.1)
                        # 记录产品-节点关系
                        node_product_relationships[f"customer {i_ + 1}"] = (M_i, "sync")
        else:  # 协同
            for i in range(num_customers):
                base = np.random.uniform(epsilon_lb, epsilon_ub)  # 基准扰动
                for m in range(len(M_i) - 2):
                    epsilon[i, M_i[m]] = base + np.random.uniform(-0.1, 0.1)
                    epsilon[i, M_i[m + 1]] = epsilon[i, M_i[m]] + np.random.uniform(-0.1, 0.1)
                node_product_relationships[f"customer {i + 1}"] = (M_i, relation)
                if num_products - len(M_i) >= 2:
                    remaining_data = set(range(num_products)) - set(M_i)
                    remaining_data = list(remaining_data)
                    M_i = random.sample(remaining_data, random.randint(2, len(remaining_data)))
                    for i_ in range(num_customers):
                        epsilon[i_, M_i] = np.random.dirichlet(np.ones(len(M_i))) * random.uniform(epsilon_lb, epsilon_ub)
                        # 记录产品-节点关系
                        node_product_relationships[f"customer {i_ + 1}"] = (M_i, "comp")
    return epsilon, node_product_relationships, product_node_relationships


# Helper function to generate actual demand
def generate_actual_demand(pre_demand, epsilon_im, rho):
    return np.round(pre_demand * (1 + epsilon_im * rho),2)



# Helper function to generate non-negative forecast demand
# def generate_actual_demand(pre_demand,sigma=1):
#     if pre_demand == 0:
#         g = np.random.uniform(0, 1)
#         theta = np.random.normal(1, sigma)  # Ensure non-negative
#         actual_demand = g * (1 + abs(theta))
#     else:
#         theta = np.random.normal(1, sigma)  # Ensure non-negative
#         actual_demand = pre_demand * (1 + abs(theta))
#     return round(actual_demand, 2)

epsilon_lb = 0
epsilon_ub = 1
# Generate data files
for num_customers, num_products, num_vehicles,sample in product(
        customer_counts, product_counts, vehicle_counts,range(1, samples_per_combination + 1)
):
    file_name = f"mirplr-{num_customers}-{num_products}-{num_vehicles}-{sample}.dat"
    file_path = os.path.join(output_folder, file_name)

    # Generate supplier and customer data
    supplier_coords = np.random.randint(0, 1001, size=2)
    supplier_inventory_cost = [0.01 for _ in range(num_products)]
    supplier_inventory = []
    supplier_capacity = 0
    customers = []

    price_values=[random.randint(10, 99) for _ in range(num_products)]
    volumn_values=[round(random.uniform(0.1, 1.5), 2) for _ in range(num_products)]

    # Generate epsilon_im with constraints and relationships
    epsilon, node_product_relationships, product_node_relationships = generate_epsilon(num_customers, num_products)

    # Generate customer data
    for customer_id in range(1, num_customers + 1):
        customer_coords = np.random.randint(0, 1001, size=2)
        pre_demand = np.random.randint(10, 101, size=num_products)
        pre_demand = np.round(pre_demand.astype(float), 2).tolist()
        actual_dim = [generate_actual_demand(pre_demand[m], epsilon[customer_id - 1, m], rho) for m in range(num_products)]
        initial_inventory = np.random.randint(30, 51, size=num_products).tolist()
        gi = np.random.choice([2, 3, 4])
        fi = np.random.randint(150, 201)
        customer_capacity = gi * fi*num_products
        supplier_capacity += customer_capacity
        inventory_cost = [round(np.random.uniform(0.02, 0.2), 2) for _ in range(num_products)]
        # Add selling price and inventory occupancy factor
        selling_price = [round(random.uniform(max(0.01,p - 9), p + 9),2) for p in price_values]
        volume_factors = [v for v in volumn_values]

        customers.append(
            {
                "id": customer_id,
                "coords": customer_coords,
                "actual_demand": actual_dim,
                "forecast_demand": pre_demand,
                "initial_inventory": initial_inventory,
                "capacity": customer_capacity,
                "inventory_cost": inventory_cost,
                "selling_price": selling_price,
                "volume_factors": volume_factors,
            }
        )
        supplier_inventory.append(sum(initial_inventory))

    # Set supplier data
    supplier_inventory_total = [sum(col) for col in zip(*[c["initial_inventory"] for c in customers])]
    vehicle_capacity = int(2 * sum(sum(c["actual_demand"]) for c in customers) / num_vehicles)
    vehicle_fixed_cost=[round(vehicle_capacity/10,2) for _ in range(num_vehicles)]

    # Write to file
    with open(file_path, "w") as f:
        # Header
        f.write(f"{num_customers} {num_products} {num_vehicles} {np.round(vehicle_capacity)}\n")
        # Supplier data
        f.write(f"{supplier_coords[0]} {supplier_coords[1]}\n")
        f.write(
            " ".join(map(str, supplier_inventory_total)) +
            f" {supplier_capacity} " +
            " ".join(map(str,supplier_inventory_cost))+" "+
            " ".join(map(str,vehicle_fixed_cost))+"\n"
        )
        # Customer data
        for customer in customers:
            f.write(
                f"{customer['id']} {customer['coords'][0]} {customer['coords'][1]} "
                f"{' '.join(map(str, customer['initial_inventory']))} "
                f"{' '.join(map(str, customer['forecast_demand']))} "
                f"{' '.join(map(str, customer['actual_demand']))} "
                f"{customer['capacity']} "
                f"{' '.join(map(str, customer['inventory_cost']))} "
                f"{' '.join(map(str, customer['selling_price']))} "
                f"{' '.join(map(str, customer['volume_factors']))}\n"
            )
        # 输出节点-产品关系
        f.write("\n# Node-Product Relationships\n")
        for node, (products, relation) in node_product_relationships.items():
            f.write(f"{node}: {[p + 1 for p in products]} {relation}\n")
        for product, (nodes, relation) in product_node_relationships.items():
            f.write(f"{product}: {[n + 1 for n in nodes]} {relation}\n")

print(
    f"Generated {customer_counts[0] * product_counts[0] * vehicle_counts[0] * samples_per_combination} files in {output_folder}.")
