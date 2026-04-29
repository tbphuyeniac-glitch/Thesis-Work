import subprocess
import os
import re
import time

import pandas as pd


def extract_outputs(output_lines):
    """
    Extract the required outputs from the algorithm's output text.
    """
    results = {"stage1 LR": None, "stock out": None, "time": None,
               "total cost": None, "total LR": None, "final stock out": None, "GAP": None}
    for line in output_lines:
        if "stage1 LR" in line:
            results["stage1 LR"] = float(line.split(":")[1].strip())
        elif "stock out" in line and "final" not in line:
            results["stock out"] = float(line.split(":")[1].strip())
        elif "time" in line:
            results["time"] = float(line.split(":")[1].strip())
        elif "total cost" in line:
            results["total cost"] = float(line.split(":")[1].strip())
        elif "total LR" in line:
            results["total LR"] = float(line.split(":")[1].strip())
        elif "final stock out" in line:
            results["final stock out"] = float(line.split(":")[1].strip())
        elif "GAP" in line:
            results["GAP"] = float(line.split(":")[1].strip())
    return results


def process_files(data_folder, algorithms, output_file):
    """
    Batch process .dat files with the provided algorithm scripts.
    Save the results into an Excel file.
    """
    results = []
    data_files = [f for f in os.listdir(data_folder) if f.endswith(".dat")]

    for data_file in data_files:
        data_path = os.path.join(data_folder, data_file)
        print(f"Processing: {data_file}")

        for algo_name, algo_path in algorithms.items():
            # Run the algorithm model script with the current data file
            print(f"Running {algo_name}...")
            try:
                process = subprocess.Popen(
                    ["python", algo_path, data_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                stdout, stderr = process.communicate()

                # Check for errors
                if process.returncode != 0:
                    print(f"Error in {algo_name} for {data_file}:\n{stderr}")
                    continue

                # Extract the outputs
                output_lines = stdout.splitlines()
                results_dict = extract_outputs(output_lines)
                results_dict["Algorithm"] = algo_name
                results_dict["Data File"] = data_file

                results.append(results_dict)

            except Exception as e:
                print(f"Failed to execute {algo_name} for {data_file}: {e}")

    # Save results to an Excel file
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    start=time.time()
    # Specify the folder containing the .dat files
    data_folder = "product_sensity"  # Replace with your folder path containing .dat files

    # Specify the paths to the algorithm scripts
    algorithms = {
        "DB_FP": "solve_DB_FP.py",
        "RO_FP": "solve_RO_FP.py",
        "TSRO_FP": "solve_TSRO_FP.py"
    }

    # Specify the output Excel file
    output_file = "product_sensity/sensity_prod.xlsx"

    # Run the batch process
    process_files(data_folder, algorithms, output_file)
    end=time.time()
    print("total run time:",end-start)