import sys
import os
import pickle
import logging
import csv

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from the original main script
from showcases.Simulating_a_data_collection_scenario.main import run_simulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def batch_main():
    """
    Main function to run batch simulations for different algorithms,
    instance sizes, and number of stages.
    """
    # --- Batch Simulation Configuration ---
    algorithms = ["LKH", "PSO"]  # Algorithms to test
    instance_sizes = [50, 100, 150, 200]  # TSP problem sizes
    num_stages_list = [2, 4, 6, 8, 10]  # Number of instances to run in sequence
    repetitions = 1  # Number of times to repeat each experiment configuration

    results = []
    run_counter = 1

    # Pre-load all TSP instances to ensure consistency across algorithms
    # This dictionary will hold the loaded data for each size.
    all_instances_data = {}
    for size in instance_sizes:
        # Construct the path to the TSP data file
        tsp_file = f"data/tsp/tsp{size}_val_mg_seed2222_size10K.pkl"
        try:
            with open(tsp_file, 'rb') as f:
                all_instances_data[size] = pickle.load(f)
            logging.info(f"Successfully loaded {len(all_instances_data[size])} instances from {tsp_file}")
        except FileNotFoundError:
            logging.error(f"Data file not found: {tsp_file}. Skipping size {size}.")
            continue

    # --- Main Experiment Loop ---
    for size in instance_sizes:
        if size not in all_instances_data:
            continue # Skip if data wasn't loaded

        for num_stages in num_stages_list:
            # Check if the requested number of stages is valid for the loaded data
            if num_stages > len(all_instances_data[size]):
                logging.warning(f"Skipping scenario with {num_stages} stages for size {size} "
                                f"because only {len(all_instances_data[size])} instances are available.")
                continue

            for algorithm in algorithms:
                for i in range(repetitions):
                    # Define the scenario for the current run
                    scenario = {
                        "name": f"{num_stages} Instances of {size} ({algorithm})",
                        "num_instances": num_stages,
                        "instance_size": size
                    }
                    
                    # The tsp_file path is now constructed dynamically for each size
                    tsp_file = f"data/tsp/tsp{size}_val_mg_seed2222_size10K.pkl"

                    logging.info(f"--- Starting Run #{run_counter}: Algorithm={algorithm}, Size={size}, Stages={num_stages}, Repetition={i+1} ---")
                    
                    # Call the original run_simulation function
                    result = run_simulation(
                        run_id=run_counter,
                        algorithm=algorithm,
                        scenario=scenario,
                        repetition=i + 1,
                        tsp_file=tsp_file
                    )
                    results.append(result)
                    run_counter += 1

    # --- Save Results to CSV ---
    output_file = "showcases/Simulating_a_data_collection_scenario/simulation_results_batch.csv"
    if not results:
        print("No results to save.")
        return

    try:
        with open(output_file, 'w', newline='') as f:
            # The keys from the first result dictionary are used as headers
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logging.info(f"\nBatch simulation results saved to {output_file}")
    except (IOError, IndexError) as e:
        logging.error(f"Error saving results to {output_file}: {e}")


if __name__ == "__main__":
    batch_main()