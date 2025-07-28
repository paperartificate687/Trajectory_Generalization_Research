import sys
import os
import json # Added for saving args
import math
import matplotlib.pyplot as plt # Added import
import glob # Added for finding latest checkpoint

# Add the project root directory (where this script is located) to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
# Go one level up if the script is in a subdirectory like new_main (adjust if needed)
# project_root = os.path.dirname(project_root)
# sys.path.insert(0, project_root) # Assuming script is in root for now. Adjust if you place it elsewhere.

# Script for TSP-200 Uniform Pretraining and Saving

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Imports from the project structure
from src.train import validate, clip_grad_norms, set_decode_type, get_inner_model
from src.options import get_options
from utils import torch_load_cpu, load_problem, move_to
from problems.tsp.problem_tsp import TSP
from src.mutils import init, ConcatDataset, set_random_seed_all # Removed get_hard_samples as not needed for uniform

# --- Configuration ---
print("Setting up configuration for TSP-200 Uniform Pretraining...")
opts = get_options('') # Load default options first

# --- Override options for this specific training run ---
opts.problem = 'tsp'
opts.graph_size = 200
opts.use_cuda = True
opts.no_progress_bar = False
opts.log_dir = 'logs/tsp200_uniform_pretrained' # Specific log directory for this run
opts.save_dir = 'new_main' # Directory to save model and args
opts.model_save_filename = 'tsp200_uniform_pretrained.pt'
opts.args_save_filename = 'args_tsp200_uniform_pretrained.json'
opts.plot_save_filename = 'tsp200_uniform_curve.png' # Added plot filename
opts.checkpoint_epochs = 10 # Save checkpoint every N epochs, 0 to disable
opts.checkpoint_save_dir = 'checkpoints/tsp200_uniform' # Directory to save checkpoints

# --- Find the latest checkpoint ---
latest_checkpoint_path = None
if os.path.isdir(opts.checkpoint_save_dir):
    checkpoint_files = glob.glob(os.path.join(opts.checkpoint_save_dir, f"tsp{opts.graph_size}_checkpoint_epoch_*.pt"))
    if checkpoint_files:
        # Extract epoch numbers and find the latest
        latest_checkpoint_path = max(checkpoint_files, key=lambda p: int(p.split('_epoch_')[-1].split('.pt')[0]))
        print(f"Found latest checkpoint: {latest_checkpoint_path}")
    else:
        print(f"No checkpoint files found in {opts.checkpoint_save_dir}")
else:
    print(f"Checkpoint directory not found: {opts.checkpoint_save_dir}")

opts.load_checkpoint_path = latest_checkpoint_path # Path to a checkpoint file to load and resume training

opts.val_dataset = 'data/tsp/tsp200_val_mg_seed2222_size10K.pkl'
# Ground truth path is not strictly needed for training, only validation metric calculation.
# Keep it for potential validation during training. Use LKH costs if available.
opts.val_gt_path = 'data/tsp/tsp200_val_mg_seed2222_size10K_lkh_costs.txt'

opts.load_path = None # Ensure no pre-trained model is loaded
opts.baseline = 'rollout' # Use rollout baseline
# Uniform training settings
opts.reweight = 0
use_hard_setting = 0 # Use get_hard_data2 (random uniform)
# Training duration and parameters
opts.n_epochs = 100 # Train for 100 epochs
opts.epoch_size = 1280000
opts.batch_size = 512
opts.max_grad_norm = 1.0
opts.lr_model = 1e-4
opts.lr_critic = 1e-4
opts.lr_decay = 0.98
opts.seed = 1234 # Use a specific seed

# Ensure log and save directories exist
os.makedirs(opts.log_dir, exist_ok=True)
os.makedirs(opts.save_dir, exist_ok=True)
if opts.checkpoint_epochs > 0:
    os.makedirs(opts.checkpoint_save_dir, exist_ok=True)

# --- Setup Device, Seed, Problem ---
if opts.use_cuda and torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("Using CPU")
opts.device = device

set_random_seed_all(opts.seed, deterministic=True)
problem = load_problem(opts.problem)

# --- Load Validation Data (for potential mid-training validation) ---
print(f"Loading validation data from: {opts.val_dataset}")
try:
    val_dataset = problem.make_dataset(filename=opts.val_dataset, num_samples=opts.val_size)
    print(f"Loaded validation dataset with size: {len(val_dataset)}")
    # Try loading GT, but don't exit if it fails, just skip gap calculation
    gt = None
    if os.path.exists(opts.val_gt_path):
        print(f"Loading ground truth costs from: {opts.val_gt_path}")
        with open(opts.val_gt_path) as file:
            lines = file.readlines()
            gt = np.array([float(line) for line in lines])
            print(f"Loaded ground truth costs with size: {len(gt)}")
    else:
        print(f"Warning: Ground truth cost file not found at {opts.val_gt_path}. Validation gap will not be calculated.")
except FileNotFoundError as e:
    print(f"Warning: Could not find validation data file: {e}")
    val_dataset = None # Set to None if loading fails
except Exception as e:
    print(f"Warning: Error loading validation data: {e}")
    val_dataset = None


# --- Helper Functions (Adapted/Copied) ---

# train_batch for standard REINFORCE (no reweighting)
def train_batch_uniform(
        model,
        optimizer,
        baseline,
        batch,
        opts # Pass opts directly
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    cost, log_likelihood = model(x)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    loss = (cost - bl_val) * log_likelihood
    reinforce_loss = loss.mean()
    total_loss = reinforce_loss + bl_loss

    optimizer.zero_grad()
    total_loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    return cost.mean().item(), total_loss.mean().item()

# train_epoch uses the uniform train_batch
def train_epoch_uniform(model, optimizer, baseline, opts, train_dataset, epoch):
    print(f"===== Starting Epoch {epoch+1}/{opts.n_epochs} =====")
    training_dataset = baseline.wrap_dataset(train_dataset)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0, shuffle=True, drop_last=True)

    model.train()
    set_decode_type(model, "sampling")

    batch_costs = []
    batch_losses = []

    pbar = tqdm(training_dataloader, desc=f"Epoch {epoch+1} Training", leave=False, disable=opts.no_progress_bar)
    for batch in pbar:
        cost, loss = train_batch_uniform(
            model,
            optimizer,
            baseline,
            batch,
            opts
        )
        batch_costs.append(cost)
        batch_losses.append(loss)
        pbar.set_postfix({'cost': cost, 'loss': loss})

    avg_cost = np.mean(batch_costs) if batch_costs else 0
    avg_loss = np.mean(batch_losses) if batch_losses else 0
    print(f"Epoch {epoch+1} completed. Avg Train Cost: {avg_cost:.4f}, Avg Train Loss: {avg_loss:.4f}")

    baseline.epoch_callback(model, epoch, dataset=train_dataset) # Use wrapped dataset? Check baseline code.

    return avg_cost, avg_loss

# Function to generate uniform random data (replaces get_hard_data)
def get_uniform_data(size, graph_size):
    print(f"Generating {size} uniform random samples...")
    return torch.FloatTensor(np.random.uniform(size=(size, graph_size, 2)))

# Simplified validation (optional, can be run periodically)
def run_validation(model, val_dataset, gt_costs, opts):
    if val_dataset is None:
        print("Skipping validation as validation data failed to load.")
        return np.nan, np.nan # Return NaN for cost and gap

    print("Running validation...")
    model.eval()
    set_decode_type(model, "greedy")

    all_costs = []
    val_dl = DataLoader(val_dataset, batch_size=opts.eval_batch_size, shuffle=False)
    for bat in tqdm(val_dl, desc="Validation", leave=False, disable=opts.no_progress_bar):
         with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
            all_costs.append(cost.data.cpu())

    if not all_costs:
        print("Error: No costs collected during validation.")
        return np.nan, np.nan

    costs_tensor = torch.cat(all_costs, 0).numpy()
    mean_cost = costs_tensor.mean()

    mean_gap = np.nan # Default gap to NaN
    if gt_costs is not None and len(costs_tensor) == len(gt_costs):
        ratio = (costs_tensor - gt_costs) / gt_costs
        mean_gap = ratio.mean()
    elif gt_costs is not None:
         print(f"Warning: Validation costs ({len(costs_tensor)}) and GT costs ({len(gt_costs)}) length mismatch. Cannot calculate accurate gap.")

    print(f"Validation results: Avg Cost={mean_cost:.4f}, Avg Gap={mean_gap:.4f}")
    return mean_cost, mean_gap


# --- Main Training Execution ---
if __name__ == '__main__':
    start_epoch = 0
    best_val_cost = float('inf')
    plot_data = [] # To store (epoch, val_cost) tuples

    if opts.load_checkpoint_path and os.path.exists(opts.load_checkpoint_path):
        print(f"Loading checkpoint from: {opts.load_checkpoint_path}")
        checkpoint = torch_load_cpu(opts.load_checkpoint_path)
        
        # Restore opts from checkpoint, but keep some current opts like checkpoint paths, device
        loaded_opts = checkpoint.get('opts', None)
        if loaded_opts:
            # Convert loaded_opts (Namespace or dict) to dict
            loaded_opts_dict = vars(loaded_opts) if not isinstance(loaded_opts, dict) else loaded_opts
            
            # Preserve critical current opts
            preserved_opts = {
                'use_cuda': opts.use_cuda, # Keep user's current device choice
                'device': opts.device,
                'log_dir': opts.log_dir, # Keep current log/save paths
                'save_dir': opts.save_dir,
                'model_save_filename': opts.model_save_filename,
                'args_save_filename': opts.args_save_filename,
                'plot_save_filename': opts.plot_save_filename,
                'checkpoint_epochs': opts.checkpoint_epochs,
                'checkpoint_save_dir': opts.checkpoint_save_dir,
                'load_checkpoint_path': opts.load_checkpoint_path, # Keep this to know we loaded
                'val_dataset': opts.val_dataset, # Allow overriding val set
                'val_gt_path': opts.val_gt_path,
                'no_progress_bar': opts.no_progress_bar,
                'n_epochs': opts.n_epochs # Allow overriding total epochs
            }
            # Update current opts with loaded opts, then override with preserved ones
            current_opts_dict = vars(opts)
            current_opts_dict.update(loaded_opts_dict)
            current_opts_dict.update(preserved_opts)
            # opts = argparse.Namespace(**current_opts_dict) # Recreate Namespace if needed, or just update
            for key, value in current_opts_dict.items():
                setattr(opts, key, value)
            print("Options restored from checkpoint and merged with current settings.")
        else:
            print("Warning: 'opts' not found in checkpoint. Using current opts.")

        # Initialize model and baseline first using (potentially updated) opts
        # We need to pass the correct opts to init
        model, base_train_dataset, _, baseline, optimizer = init(pretrain=False, device=opts.device, opts=opts)

        get_inner_model(model).load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer states to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = move_to(v, opts.device)
        if 'baseline_state_dict' in checkpoint and baseline is not None and hasattr(baseline, 'load_state_dict'):
            baseline.load_state_dict(checkpoint['baseline_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0)
        best_val_cost = checkpoint.get('best_val_cost', float('inf'))
        plot_data = checkpoint.get('plot_data', []) # Restore plot data
        
        print(f"Resuming training from epoch {start_epoch + 1}")
        print(f"Restored best_val_cost: {best_val_cost}")
        if plot_data:
            print(f"Restored plot data with {len(plot_data)} points.")

    else:
        if opts.load_checkpoint_path:
            print(f"Warning: Checkpoint file not found at {opts.load_checkpoint_path}. Starting from scratch.")
        print("Initializing model, baseline, optimizer for training from scratch...")
        # Force pretrain=False
        model, base_train_dataset, _, baseline, optimizer = init(pretrain=False, device=device, opts=opts)

        # --- Record initial performance for plot if not resuming ---
        print("Running initial validation for new training...")
        initial_val_cost, initial_val_gap = run_validation(model, val_dataset, gt, opts)
        plot_data.append((0, initial_val_cost)) # Record cost at epoch 0 (start_epoch is 0 here)
        if not np.isnan(initial_val_cost):
            best_val_cost = initial_val_cost
        # --- End ---


    if base_train_dataset is None: # This check should be after potential init
        print("Error: Base training dataset failed to load or init. Cannot proceed.")
        exit(1)

    print(f"Using baseline: {type(baseline).__name__}")
    print(f"Initial base training dataset size: {len(base_train_dataset)}")


    print(f"===== Starting Uniform Training (from epoch {start_epoch + 1} to {opts.n_epochs}) =====")
    start_time_total = time.time()

    for epoch in range(start_epoch, opts.n_epochs):
        epoch_start_time = time.time()

        # 1. Generate uniform random data for this epoch (standard training)
        # The base_train_dataset from init is often the primary source for uniform training.
        # Using dynamically generated uniform data might deviate from standard practice.
        # Let's stick to using the base_train_dataset loaded by init.
        # If epoch_size > len(base_train_dataset), DataLoader with shuffle will repeat samples.
        current_train_dataset = base_train_dataset

        # --- If dynamic uniform data is truly desired (uncomment if needed) ---
        # uniform_data_part = get_uniform_data(opts.epoch_size, opts.graph_size) # Generate samples matching epoch_size
        # current_train_dataset = uniform_data_part # Use only the generated data
        # --- End dynamic uniform data ---

        print(f"Epoch {epoch+1}: Using training dataset of size {len(current_train_dataset)}")

        # 2. Train one epoch
        avg_cost, avg_loss = train_epoch_uniform(
            model, optimizer, baseline, opts, current_train_dataset, epoch
        )

        # 3. Validate the model
        val_cost, val_gap = run_validation(model, val_dataset, gt, opts)
        plot_data.append((epoch + 1, val_cost)) # <<<--- Record data for plot

        # Optional: Save model if it's the best so far based on validation cost
        if not np.isnan(val_cost) and val_cost < best_val_cost: # Check for NaN
             print(f"New best validation cost: {val_cost:.4f} (previous: {best_val_cost:.4f}).") # Removed 'Saving model...' for now
             best_val_cost = val_cost
             # Optional: Add saving best model logic here if desired

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")

        # --- Save Checkpoint ---
        if opts.checkpoint_epochs > 0 and (epoch + 1) % opts.checkpoint_epochs == 0 and (epoch + 1) < opts.n_epochs:
            checkpoint_path = os.path.join(opts.checkpoint_save_dir, f"tsp{opts.graph_size}_checkpoint_epoch_{epoch + 1}.pt")
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'epoch': epoch + 1, # Save current epoch number (1-based)
                'model_state_dict': get_inner_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
                'best_val_cost': best_val_cost,
                'plot_data': plot_data, # Save current plot data
                'opts': opts, # Save opts
            }, checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch + 1}.")
        # --- End Save Checkpoint ---

    total_training_time = time.time() - start_time_total
    print(f"===== Training Finished ({opts.n_epochs} epochs) =====")
    print(f"Total training time: {total_training_time / 3600:.2f} hours")

    # --- Save Final Model and Arguments ---
    final_model_save_path = os.path.join(opts.save_dir, opts.model_save_filename)
    final_args_save_path = os.path.join(opts.save_dir, opts.args_save_filename)

    print(f"Saving final model to: {final_model_save_path}")
    # Save model state dictionary, optimizer state, epoch number, etc.
    torch.save({
        'epoch': opts.n_epochs, # Save final epoch number
        'model_state_dict': get_inner_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
        'best_val_cost': best_val_cost,
        'plot_data': plot_data, # Also save plot_data with the final model for completeness
        'opts': opts,
    }, final_model_save_path)

    print(f"Saving training arguments to: {final_args_save_path}")
    # Convert Namespace to dict, handling potential non-serializable items if necessary
    opts_dict = vars(opts)
    # Remove non-serializable items if any (e.g., device object, loaded datasets)
    opts_to_save = {k: v for k, v in opts_dict.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    try:
        with open(final_args_save_path, 'w') as f:
            json.dump(opts_to_save, f, indent=4)
        print("Arguments saved successfully.")
    except Exception as e:
        print(f"Error saving arguments to JSON: {e}")

    print("===== Script Finished =====")

    # --- Added Plotting Code ---
    print("Generating training curve plot...")
    if plot_data:
        plot_epochs = [p[0] for p in plot_data if not np.isnan(p[1])] # Filter out NaN costs for plotting
        plot_costs = [p[1] for p in plot_data if not np.isnan(p[1])]

        if plot_epochs: # Check if there is valid data to plot
            plt.figure(figsize=(10, 6))
            plt.plot(plot_epochs, plot_costs, marker='o', linestyle='-')
            plt.xlabel("Epoch")
            plt.ylabel("Average Validation Cost")
            plt.title(f"TSP-{opts.graph_size} Uniform Training Curve")
            plt.grid(True)
            plt.xticks(np.arange(min(plot_epochs), max(plot_epochs)+1, step=max(1, opts.n_epochs // 10))) # Adjust x-ticks
            plt.tight_layout()

            plot_save_path = os.path.join(opts.save_dir, opts.plot_save_filename) # Save plot in the save_dir
            try:
                plt.savefig(plot_save_path)
                print(f"Training curve plot saved to {plot_save_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")
            # plt.show() # Uncomment to display plot
        else:
             print("No valid cost data collected for plotting (all were NaN?).")
    else:
        print("No data collected for plotting.")
    # --- End Added Plotting Code ---