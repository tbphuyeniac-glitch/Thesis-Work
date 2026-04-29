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
vehicle_counts = [3]
samples_per_combination = 5
product_values=[0.1,1,10]
product_volume=[0.1,1,5]
# precision_sigma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
# Output folder
output_folder = "product_sensity"
os.makedirs(output_folder, exist_ok=True)


# Helper function to generate non-negative forecast demand
def generate_forecast_demand(actual_demand,sigma=1):
    if actual_demand == 0:
        g = np.random.uniform(0, 1)
        theta = np.random.normal(1, sigma)  # Ensure non-negative
        forecast_demand = g * (1 + abs(theta))
    else:
        theta = np.random.normal(1, sigma)  # Ensure non-negative
        forecast_demand = actual_demand * (1 + abs(theta))
    return round(forecast_demand, 2)


# Generate data files
for num_customers, num_products, num_vehicles,value,volume ,sample in product(
        customer_counts, product_counts, vehicle_counts, product_values,product_volume,range(1, samples_per_combination + 1)
):
    file_name = f"mirplr-{num_customers}-{num_products}-{num_vehicles}-{value}-{volume}-{sample}.dat"
    file_path = os.path.join(output_folder, file_name)

    # Generate supplier and customer data
    supplier_coords = np.random.randint(0, 1001, size=2)
    supplier_inventory_cost = [0.01 for _ in range(num_products)]
    supplier_inventory = []
    supplier_capacity = 0
    customers = []

    price_values=[value*random.randint(10, 99) for _ in range(num_products)]
    volumn_values=[volume*round(random.uniform(0.1, 1.5), 2) for _ in range(num_products)]

    # Generate customer data
    for customer_id in range(1, num_customers + 1):
        customer_coords = np.random.randint(0, 1001, size=2)
        actual_demand = np.random.randint(10, 101, size=num_products).tolist()
        forecast_demand = [generate_forecast_demand(d) for d in actual_demand]
        initial_inventory = np.random.randint(30, 51, size=num_products).tolist()
        gi = np.random.choice([2, 3, 4])
        fi = np.random.randint(150, 201)
        customer_capacity = gi * fi*num_products
        supplier_capacity += customer_capacity
        inventory_cost = [round(np.random.uniform(0.02, 0.2), 2) for _ in range(num_products)]
        # Add selling price and inventory occupancy factor
        selling_price = [random.uniform(max(0.01,p - 9), p + 9) for p in price_values]
        volume_factors = [v for v in volumn_values]

        customers.append(
            {
                "id": customer_id,
                "coords": customer_coords,
                "actual_demand": actual_demand,
                "forecast_demand": forecast_demand,
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
    vehicle_capacity = 2 * sum(sum(c["actual_demand"]) for c in customers) / num_vehicles
    vehicle_fixed_cost=[round(vehicle_capacity/10,2) for _ in range(num_vehicles)]

    # Write to file
    with open(file_path, "w") as f:
        # Header
        f.write(f"{num_customers} {num_products} {num_vehicles} {round(vehicle_capacity)}\n")
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
                f"{' '.join(map(str, customer['actual_demand']))} "
                f"{' '.join(map(str, customer['forecast_demand']))} "
                f"{customer['capacity']} "
                f"{' '.join(map(str, customer['inventory_cost']))} "
                f"{' '.join(map(str, customer['selling_price']))} "
                f"{' '.join(map(str, customer['volume_factors']))}\n"
            )

print(
    f"Generated {len(customer_counts) * len(product_counts) * len(vehicle_counts) * samples_per_combination} files in {output_folder}.")
