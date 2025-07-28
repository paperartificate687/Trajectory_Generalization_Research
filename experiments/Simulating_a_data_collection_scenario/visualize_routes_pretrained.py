# Script to load a trained model and visualize generated routes

import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser # Use ArgumentParser for flexibility

# Add project root to path (assuming this script is in the root or a subfolder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Project imports
from utils import load_problem, torch_load_cpu
from src.mutils import plot_tsp # Import plot_tsp instead
from showcases.Simulating_a_data_collection_scenario.inference_utils import load_attention_model, solve_tsp_with_attention

def generate_and_plot(model, dataset_path, num_instances_to_plot, save_filename, device):
    """Loads data, generates tours with the model, and plots them."""
    print(f"Loading data for visualization from: {dataset_path}")
    # Load a few instances from the specified dataset
    problem = load_problem('tsp') # Assume TSP
    try:
        # make_dataset likely returns a torch.utils.data.Dataset object
        dataset = problem.make_dataset(filename=dataset_path, num_samples=num_instances_to_plot)
        # Validate dataset length
        if len(dataset) < num_instances_to_plot:
            print(f"Warning: Requested {num_instances_to_plot} instances, but dataset only contains {len(dataset)}. Plotting available instances.")
            num_instances_to_plot = len(dataset)
        if num_instances_to_plot == 0:
            print("Error: Dataset is empty. Cannot visualize.")
            return

    except FileNotFoundError:
        print(f"Error: Data file not found at {dataset_path}")
        return
    except Exception as e:
        print(f"Error loading data {dataset_path}: {e}")
        return

    print(f"Generating routes for {num_instances_to_plot} instances...")

    num_cols = 3 # Number of plots per row
    num_rows = (num_instances_to_plot + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), squeeze=False)
    axes = axes.flatten()

    # Iterate directly through the Dataset object
    for i in range(num_instances_to_plot):
        # Get the i-th instance from the dataset
        # The dataset likely yields tensors directly, or dicts containing tensors
        raw_instance_data = dataset[i]
        
        # --- Adjust data handling based on actual dataset[i] format ---
        # Option 1: If dataset[i] is already the coordinate tensor
        if isinstance(raw_instance_data, torch.Tensor):
            instance_data_tensor = raw_instance_data.float() # Ensure float
        # Option 2: If dataset[i] is a dict (e.g., {'loc': tensor, ...})
        elif isinstance(raw_instance_data, dict):
            # Adjust key based on how TSP dataset is structured
            instance_data_tensor = raw_instance_data.get('loc', raw_instance_data.get('coords', None))
            if instance_data_tensor is None:
                print(f"Error: Could not find coordinate data in dataset instance {i}. Keys: {raw_instance_data.keys()}")
                continue
            instance_data_tensor = instance_data_tensor.float()
        # Option 3: If dataset[i] is numpy array
        elif isinstance(raw_instance_data, np.ndarray):
             instance_data_tensor = torch.from_numpy(raw_instance_data).float()
        else:
            print(f"Error: Unexpected data type for instance {i}: {type(raw_instance_data)}")
            continue
        # --- End data handling ---

        # --- Add code to save raw instance data for inspection ---
        temp_data_path = f"temp_instance_{i}.npy"
        np.save(temp_data_path, instance_data_tensor.cpu().numpy())
        print(f"Saved raw instance {i} data to {temp_data_path}")
        # --- End add code ---
            
        # The data from the .pkl file is normalized. We need to scale it to the physical area.
        # The physical area is 500x500 as seen in main_pretrained.py.
        scaled_coords_np = instance_data_tensor.cpu().numpy() * 500.0
        locations_to_visit = [(float(x), float(y), 0.0) for x, y in scaled_coords_np]

        # Log the exact data being sent for inference
        log_filename = f"showcases/Simulating_a_data_collection_scenario/inference_input_visualizer_instance_{i}.json"
        with open(log_filename, 'w') as f:
            json.dump(locations_to_visit, f, indent=4)
        print(f"Saved inference input data for instance {i} to {log_filename}")

        # Use the centralized solver with fixed area bounds for normalization
        ordered_locations, cost_val, _ = solve_tsp_with_attention(
            model, locations_to_visit, device, area_bounds=(500.0, 500.0)
        )

        # Convert back for plotting using the scaled coordinates
        instance_np = scaled_coords_np
        
        # The tour needs to be represented by indices into the original instance_np
        # Create a mapping from location tuple back to its original index
        # The keys for the map should be the 2D physical coordinates
        location_to_index_map = {tuple(loc): idx for idx, loc in enumerate(instance_np)}
        
        # Convert the ordered 3D locations back to 2D for the map lookup
        ordered_locations_2d = [(x, y) for x, y, z in ordered_locations]

        # Reconstruct tour_np using the map
        tour_np = [location_to_index_map[tuple(loc)] for loc in ordered_locations_2d]

        ax = axes[i]
        plot_tsp(instance_np, tour_np, ax)
        ax.set_title(f"Instance {i+1} | Cost: {cost_val:.4f}")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    try:
        plt.savefig(save_filename)
        print(f"Visualization saved to {save_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Uncomment to display plot


if __name__ == '__main__':
    parser = ArgumentParser(description="Load a trained TSP model and visualize routes.")
    parser.add_argument("--model_path", default="showcases/new_main/tsp50_attention_custom_final.pt", help="Path to the trained model (.pt file)")
    parser.add_argument("--data_paths", nargs='+', default=["data/tsp/tsp50_val_mg_seed2222_size10K.pkl"], help="Path(s) to the dataset file(s) (.pkl or .tsp) for visualization")
    parser.add_argument("--num_instances", type=int, default=6, help="Number of instances to visualize")
    parser.add_argument("--save_path", default="visualization_output.png", help="Filename for the output plot")
    parser.add_argument("--no_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--graph_size", type=int, help="Graph size of the TSP problem (required if opts not in checkpoint)")

    args = parser.parse_args()

    if args.graph_size is None:
        print("Warning: --graph_size not provided. Attempting to load opts from checkpoint.")

    # Setup device
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Using CUDA device 0")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load the model using the centralized function
    model, _ = load_attention_model(args.model_path, device, graph_size=args.graph_size)

    # Generate and plot routes
    for data_path in args.data_paths:
        base_name = os.path.basename(data_path)
        name, ext = os.path.splitext(base_name)
        save_filename = f"{name}_visualization.png"
        
        print(f"\nProcessing {data_path}...")
        generate_and_plot(model, data_path, args.num_instances, save_filename, device)

    print("Visualization script finished.")