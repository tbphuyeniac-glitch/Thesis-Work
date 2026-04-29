import os
import random
import numpy as np
import math

def parse_original_file(filepath):
    """
    Parse the original .dat file into structured data.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Extract basic parameters from the first line
    first_line = lines[0].strip().split()
    num_customers, num_products, num_vehicles, num_period, vehicle_capacity = map(int, first_line)

    # Extract supplier data
    supplier_coordinates = list(map(float, lines[1].strip().split()))
    supplier_inventory_data = list(map(float, lines[2].strip().split()))
    supplier_initial_inventory = supplier_inventory_data[:num_products]
    supplier_inventory_cost = supplier_inventory_data[-num_products:]
    # Remove per-cycle supply
    supplier_per_cycle_supply = supplier_inventory_data[num_products:-num_products]

    # Extract customer data
    customers = []
    for line in lines[3:]:
        customer_data = list(map(float, line.strip().split()))
        customer_info = {
            "customer_id": int(customer_data[0]),
            "coordinates": customer_data[1:3],
            "initial_inventory": customer_data[3:3+num_products],
            "demand": customer_data[3+num_products:3+num_products+num_products*num_period],
            "inventory_capacity": customer_data[3+num_products+num_products*num_period],
            "inventory_cost": customer_data[3+num_products+num_products*num_period+1:],
        }
        customers.append(customer_info)

    return {
        "num_customers": num_customers,
        "num_products": num_products,
        "num_vehicles": num_vehicles,
        "vehicle_capacity": vehicle_capacity,
        "supplier": {
            "coordinates": supplier_coordinates,
            "initial_inventory": supplier_initial_inventory,
            "inventory_cost": supplier_inventory_cost,
        },
        "customers": customers
    }

def modify_data(data, sigma_values, output_dir):
    """
    Modify the data as per the requirements and save to new files.
    """
    num_customers = data["num_customers"]
    num_products = data["num_products"]
    num_vehicles = data["num_vehicles"]

    price_values=[random.randint(10, 99) for _ in range(num_products)]
    volumn_values=[round(random.uniform(0.1, 1.5), 2) for _ in range(num_products)]

    error_names={sigma_values[0]:"lowError",sigma_values[1]:"mediumError",sigma_values[2]:"highError"}
    for sigma in sigma_values:
        for sample in range(1, 6):  # Five samples for each configuration
            # Prepare new data structure
            modified_data = data.copy()

            # Adjust customer demands to generate forecast demands
            for customer in modified_data["customers"]:
                actual_demand = customer["demand"][:num_products]  # Only first cycle
                customer["actual_demand"]=actual_demand
                customer["initial_inventory"]=[random.randint(30,50) for _ in range(num_products)]
                forecast_demand = []
                for demand in actual_demand:
                    if demand > 0:
                        theta = round(abs(np.random.normal(1, sigma)),2)
                        forecast_demand.append(round(demand * theta,2))
                    else:
                        g = round(random.uniform(0, 1),2)
                        theta = round(abs(np.random.normal(1, sigma)),2)
                        forecast_demand.append(round(random.choice([0, g * (1 + theta)]),2))
                customer["forecast_demand"] = forecast_demand

                # Add selling price and inventory occupancy factor
                customer["selling_price"] = [random.randint(p-9, p+9) for p in price_values]
                customer["occupancy_factor"] = [v for v in volumn_values]

            # Calculate supplier inventory capacity
            supplier_capacity = sum(customer["inventory_capacity"] for customer in modified_data["customers"])
            modified_data["supplier"]["capacity"] = supplier_capacity
            modified_data["supplier"]["initial_inventory"] = [sum(customer["initial_inventory"][i] for customer in modified_data["customers"]) for i in range(num_products)]


            # Generate new file name
            filename = f"mirplr-{num_customers}-{num_products}-{num_vehicles}-{error_names[sigma]}-{sample}.dat"
            filepath = os.path.join(output_dir, filename)
            modified_data["sigma"]=sigma

            modified_data["vehicle_fixed_cost"]=[data["vehicle_capacity"]/10 for _ in range(num_vehicles)]
            # Save the modified data
            save_modified_data(modified_data, filepath)

def save_modified_data(data, filepath):
    """
    Save the modified data to a new .dat file.
    """
    with open(filepath, 'w') as file:
        # Write first line
        file.write(f"{data['num_customers']} {data['num_products']} {data['num_vehicles']} {data['vehicle_capacity']} {data['sigma']}\n")

        # Write supplier data
        supplier = data["supplier"]
        file.write(" ".join(map(str, supplier["coordinates"])) + "\n")
        file.write(" ".join(map(str, supplier["initial_inventory"])) + f" {supplier['capacity']} " +" ".join(map(str, supplier["inventory_cost"])) +" "+" ".join(map(str, data["vehicle_fixed_cost"])) + "\n")

        # Write customer data
        for customer in data["customers"]:
            customer_data = [
                customer["customer_id"],
                *customer["coordinates"],
                *customer["initial_inventory"],
                *customer["actual_demand"],
                *customer["forecast_demand"],
                customer["inventory_capacity"],
                *customer["inventory_cost"],
                *customer["selling_price"],
                *customer["occupancy_factor"],
            ]
            file.write(" ".join(map(str, customer_data)) + "\n")

# Main function to process all files
def main(input_dir, output_dir):
    sigma_values = [0.1, 0.5, 1]  # Error degrees
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".dat"):
            filepath = os.path.join(input_dir, filename)
            data = parse_original_file(filepath)
            modify_data(data, sigma_values, output_dir)


# revise data
# input_dir = "data_set"
# output_dir = "new_data"
# main(input_dir, output_dir)

def parse_data_file(filepath):
    """
    Parse a modified .dat file into a structured Python dictionary.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Extract the first line (basic parameters)
    first_line = lines[0].strip().split()
    num_customers, num_products, num_vehicles= (
        int(first_line[0]),
        int(first_line[1]),
        int(first_line[2]),
    )
    Qk=[int(first_line[3]) for _ in range(num_vehicles)]
    x_co=[]
    y_co=[]
    Iim0=[]
    Ui=[]
    him=[]
    dim_actual=[]
    dim_predict=[]
    pim=[]
    vim=[]
    node_product_relationships = {}  # 存储节点-产品关系
    product_node_relationships = {}  # 存储产品-节点关系

    # Extract supplier data
    supplier_coordinates = list(map(float, lines[1].strip().split()))
    x_co.append(supplier_coordinates[0])
    y_co.append(supplier_coordinates[1])
    supplier_data = list(map(float, lines[2].strip().split()))
    supplier_initial_inventory = supplier_data[:num_products]
    Iim0.append(supplier_initial_inventory)
    supplier_capacity = supplier_data[num_products]
    Ui.append(supplier_capacity)
    supplier_inventory_cost = supplier_data[num_products + 1:num_products*2 + 1]
    him.append(supplier_inventory_cost)
    bk=supplier_data[num_products*2 + 1:]

    # Extract customer data
    customers = []
    for line in lines[3:3 + num_customers]:
        customer_data = list(map(float, line.strip().split()))
        customer_info = {
            "customer_id": int(customer_data[0]),
            "coordinates": customer_data[1:3],
            "initial_inventory": customer_data[3:3+num_products],
            "actual_demand": customer_data[3+num_products:3+2*num_products],
            "forecast_demand": customer_data[3+2*num_products:3+3*num_products],
            "inventory_capacity": customer_data[3+3*num_products],
            "inventory_cost": customer_data[3+3*num_products+1:3+4*num_products+1],
            "selling_price": customer_data[3+4*num_products+1:3+5*num_products+1],
            "occupancy_factor": customer_data[3+5*num_products+1:]
        }
        customers.append(customer_info)

    # Extract node-product and product-node relationships
    for line in lines[3 + num_customers:]:  # 读取关系部分
        if line.strip().startswith("#"):  # 跳过注释行
            continue
        if "节点" in line:  # 节点-产品关系
            parts = line.strip().split(":")
            node = parts[0].strip()
            products_and_relation = parts[1].strip().split()
            products = list(map(int, products_and_relation[0].strip("[]").split(",")))
            relation = products_and_relation[1]
            node_product_relationships[node] = (products, relation)
        elif "产品" in line:  # 产品-节点关系
            parts = line.strip().split(":")
            product = parts[0].strip()
            nodes_and_relation = parts[1].strip().split()
            nodes = list(map(int, nodes_and_relation[0].strip("[]").split(",")))
            relation = nodes_and_relation[1]
            product_node_relationships[product] = (nodes, relation)

    for cust in customers:
        x_co.append(cust["coordinates"][0])
        y_co.append(cust["coordinates"][1])
        Iim0.append(cust["initial_inventory"])
        Ui.append(cust["inventory_capacity"])
        him.append(cust["inventory_cost"])
        dim_actual.append(cust["actual_demand"])
        dim_predict.append(cust["forecast_demand"])
        pim.append(cust["selling_price"])
        vim.append(cust["occupancy_factor"])

    cij=euclidean_distance(x_co,y_co)

    return {
        "num_customers": num_customers,
        "num_products": num_products,
        "num_vehicles": num_vehicles,
        "Ui": Ui,
        "cij": cij,
        "pim": pim,
        "Qk": Qk,
        "bk": bk,
        "vim": vim,
        "him": him,
        "Iim0": Iim0,
        "dim_actual": dim_actual,
        "dim_predict": dim_predict,
        "node_product_relationships": node_product_relationships,
        "product_node_relationships": product_node_relationships,
    }

def euclidean_distance(x_coords, y_coords):
    # 计算两点之间的欧氏距离，并保存在二维列表中
    distances = []
    for i in range(len(x_coords)):
        row = []
        for j in range(len(x_coords)):
            distance = math.sqrt((x_coords[i] - x_coords[j]) ** 2 + (y_coords[i] - y_coords[j]) ** 2)
            if distance==0:
                row.append(9999)
            else:
                row.append(distance)
        distances.append(row)
    return distances

def revise_file_name(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Loop through all files
    for file_name in files:
        # Check if the file matches the naming pattern with "highError", "lowError", etc.
        if "." in file_name:
            # Split the file name into parts (excluding the file extension)
            name_parts = file_name.rsplit('.', 1)  # Separate the name and the extension
            base_name = name_parts[0]
            new_file_name = base_name
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_file_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {file_name} -> {new_file_name}")


    print("Batch renaming completed!")

def parse_relationships_and_define_gamma(node_product_relationships, product_node_relationships, b_comp=0.3, b_sync=0.3, b_conflict=0.3, b_region=0.3):
    """
    Parse node-product and product-node relationships to define Gamma values.

    Parameters:
        node_product_relationships: Dictionary of node-product relationships.
        product_node_relationships: Dictionary of product-node relationships.
        b_comp: Budget factor for complementary products (default: 0.5).
        b_sync: Budget factor for synchronous products (default: 0.5).
        b_conflict: Budget factor for conflict products (default: 0.5).
        b_region: Budget factor for regional products (default: 0.5).

    Returns:
        Gamma_comp: Dictionary of complementary product budgets for each node.
        Gamma_sync: Dictionary of synchronous product budgets for each node.
        Gamma_conflict: Dictionary of conflict budgets for each product.
        Gamma_region: Dictionary of regional budgets for each product.
    """
    # Initialize Gamma dictionaries
    Gamma_comp = {}
    Gamma_sync = {}
    Gamma_conflict = {}
    Gamma_region = {}

    # Parse node-product relationships
    for node, (products, relation) in node_product_relationships.items():
        if relation == "comp":
            Gamma_comp[node] = b_comp * len(products)  # Gamma_i^{comp} = b_comp * |M_i^{comp}|
        elif relation == "sync":
            Gamma_sync[node] = b_sync  # Gamma_i^{sync} = b_sync

    # Parse product-node relationships
    for product, (nodes, relation) in product_node_relationships.items():
        if relation == "conflict":
            Gamma_conflict[product] = b_conflict * len(nodes)  # Gamma_m^{conflict} = b_conflict * |V_m^{conflict}|
        elif relation == "region":
            Gamma_region[product] = b_region  # Gamma_m^{region} = b_region

    return Gamma_comp, Gamma_sync, Gamma_conflict, Gamma_region

