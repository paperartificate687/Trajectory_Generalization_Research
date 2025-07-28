import sys
import os
import json # Added for saving args
import math
import matplotlib.pyplot as plt # Added import

# Add the project root directory (where this script is located) to the Python path
# Assuming this script is in the project root.
# If it's in a subdirectory, you might need to adjust the path.
# project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, project_root)

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
from src.mutils import init, ConcatDataset, set_random_seed_all

# --- Configuration ---
print("Setting up configuration for TSP-200 Attention Custom Training...")
opts = get_options('') # Load default options first

# --- Override options for this specific training run ---
opts.problem = 'tsp'
opts.graph_size = 200
opts.use_cuda = True
opts.no_progress_bar = False # Enable progress bar for interactive training
opts.log_dir = 'logs/tsp200_attention_custom' # Specific log directory
opts.save_dir = 'new_main' # Directory to save model and args (can be customized)
opts.model_save_filename = 'tsp200_attention_custom_final.pt' # Custom model filename
opts.args_save_filename = 'args_tsp200_attention_custom.json' # Custom args filename
opts.plot_save_filename = 'tsp200_attention_custom_curve.png' # Custom plot filename

# Validation dataset (adjust if needed)
opts.val_dataset = 'data/tsp/tsp200_val_mg_seed2222_size10K.pkl'
opts.val_gt_path = 'data/tsp/tsp200_val_mg_seed2222_size10K_lkh_costs.txt' 

opts.load_path = None # Ensure no pre-trained model is loaded for a fresh training
opts.baseline = 'rollout' # Use rollout baseline
opts.reweight = 0 # Standard REINFORCE, no reweighting

# Training duration and parameters
opts.n_epochs = 100
opts.epoch_size = 1280000 # Number of samples per epoch
opts.batch_size = 512 # May need adjustment for larger graph sizes
opts.max_grad_norm = 1.0
opts.lr_model = 1e-4
opts.lr_critic = 1e-4 # Learning rate for the baseline critic if applicable
opts.lr_decay = 1.0 # No learning rate decay by default
opts.seed = 2024 # Using a new seed for this custom run, can be changed
opts.checkpoint_epochs = 0 # Set to 0 to save only at the end, or e.g. 10 to save every 10 epochs

# Ensure log and save directories exist
os.makedirs(opts.log_dir, exist_ok=True)
os.makedirs(opts.save_dir, exist_ok=True)

# --- Setup Device, Seed, Problem ---
if opts.use_cuda and torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("Using CPU")
opts.device = device

set_random_seed_all(opts.seed, deterministic=True) # Set seed for reproducibility
problem = load_problem(opts.problem)

# --- Load Validation Data ---
print(f"Loading validation data from: {opts.val_dataset}")
gt = None # Ground truth costs
try:
    val_dataset = problem.make_dataset(filename=opts.val_dataset, num_samples=opts.val_size)
    print(f"Loaded validation dataset with size: {len(val_dataset)}")
    if opts.val_gt_path and os.path.exists(opts.val_gt_path):
        print(f"Loading ground truth costs from: {opts.val_gt_path}")
        with open(opts.val_gt_path) as file:
            lines = file.readlines()
            gt = np.array([float(line) for line in lines])
            print(f"Loaded ground truth costs with size: {len(gt)}")
    else:
        print(f"Warning: Ground truth cost file not found at {opts.val_gt_path} or path not specified. Validation gap will not be calculated.")
except FileNotFoundError as e:
    print(f"Warning: Could not find validation data file: {e}. Validation will be skipped or limited.")
    val_dataset = None
except Exception as e:
    print(f"Warning: Error loading validation data: {e}. Validation will be skipped or limited.")
    val_dataset = None


# --- Helper Training Functions (adapted from train_and_save_tsp50.py) ---

def train_batch_custom(
        model,
        optimizer,
        baseline,
        batch,
        opts # Pass full opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x) # model is AttentionModel instance

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    
    # Calculate REINFORCE loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss # Add baseline loss if applicable

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradient norms
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    
    return cost.mean().item(), loss.item() # Return average cost and total loss for the batch

def train_epoch_custom(model, optimizer, baseline, opts, train_dataset, epoch_num):
    print(f"===== Starting Epoch {epoch_num + 1}/{opts.n_epochs} for TSP-{opts.graph_size} Custom Training =====")
    
    # Wrap training dataset with baseline information
    # The dataset passed here should be the raw TSP instances (coordinates)
    training_dataset_wrapped = baseline.wrap_dataset(train_dataset)
    training_dataloader = DataLoader(training_dataset_wrapped, batch_size=opts.batch_size, num_workers=0, shuffle=True, drop_last=True)
            
    model.train() # Set model to training mode
    set_decode_type(model, "sampling") # Use sampling for exploration during training
    
    epoch_batch_costs = []
    epoch_batch_losses = []
    
    pbar = tqdm(training_dataloader, desc=f"Epoch {epoch_num + 1} Training", leave=False, disable=opts.no_progress_bar)
    for batch_data_wrapped in pbar:
        # train_batch_custom expects the full opts object
        cost, loss = train_batch_custom(
            model,
            optimizer,
            baseline,
            batch_data_wrapped, # This is the wrapped batch from baseline
            opts
        )
        epoch_batch_costs.append(cost)
        epoch_batch_losses.append(loss)
        pbar.set_postfix({'cost': f'{cost:.4f}', 'loss': f'{loss:.4f}'})

    avg_epoch_cost = np.mean(epoch_batch_costs) if epoch_batch_costs else 0
    avg_epoch_loss = np.mean(epoch_batch_losses) if epoch_batch_losses else 0
    print(f"Epoch {epoch_num + 1} completed. Avg Train Cost: {avg_epoch_cost:.4f}, Avg Train Loss: {avg_epoch_loss:.4f}")
    
    # Baseline epoch callback (e.g., for rollout baseline to update its model)
    baseline.epoch_callback(model, epoch_num, dataset=train_dataset) # Pass original dataset
    
    return avg_epoch_cost, avg_epoch_loss

def run_validation_custom(model, validation_dataset, gt_costs, opts):
    if validation_dataset is None:
        print("Skipping validation as validation data is not available.")
        return np.nan, np.nan

    print("Running validation...")
    model.eval() 
    set_decode_type(model, "greedy") # Use greedy decoding for validation
    
    all_costs_list = []
    # Create DataLoader for validation dataset
    val_dl = DataLoader(validation_dataset, batch_size=opts.eval_batch_size, shuffle=False) # opts.eval_batch_size
    
    for batch_coords in tqdm(val_dl, desc="Validation", leave=False, disable=opts.no_progress_bar):
         with torch.no_grad():
            # Model expects input Fahrrad (batch_size, graph_size, node_dim)
            cost, _ = model(move_to(batch_coords, opts.device)) # model is AttentionModel
            all_costs_list.append(cost.data.cpu())
            
    if not all_costs_list:
        print("Error: No costs collected during validation.")
        return np.nan, np.nan
        
    costs_tensor = torch.cat(all_costs_list, 0).numpy()
    mean_val_cost = costs_tensor.mean()
    
    mean_val_gap = np.nan # Default gap to NaN
    if gt_costs is not None:
        if len(costs_tensor) == len(gt_costs):
            # Ensure gt_costs is aligned with the order of val_dataset if shuffle=False was used for DataLoader
            ratio = (costs_tensor - gt_costs[:len(costs_tensor)]) / gt_costs[:len(costs_tensor)]
            mean_val_gap = ratio.mean()
        else:
             print(f"Warning: Validation costs ({len(costs_tensor)}) and GT costs ({len(gt_costs)}) length mismatch. Cannot calculate accurate gap.")

    print(f"Validation results: Avg Cost={mean_val_cost:.4f}, Avg Gap={mean_val_gap:.4f}")
    return mean_val_cost, mean_val_gap


# --- Main Training Execution ---
if __name__ == '__main__':
    print("Initializing model, baseline, optimizer for training from scratch...")
    # init from mutils.py will load the base training dataset specified in its own logic
    # (e.g., data/tsp/tsp200_train_seed1111_size10K.pkl)
    # It returns: model, dataset, dataloader, baseline, optimizer, start_epoch
    model, base_train_dataset, _, baseline, optimizer, start_epoch = init(pretrain=False, opts=opts)
    
    if model is None or base_train_dataset is None:
        print("Error: Model or base training dataset failed to initialize. Exiting.")
        sys.exit(1) # Exit if essential components are not loaded
        
    print(f"Using baseline: {type(baseline).__name__}")
    print(f"Base training dataset size (from init): {len(base_train_dataset)}")
    print(f"Training will start from epoch: {start_epoch + 1}")


    # Variables to track best validation performance and plot data
    best_val_cost = float('inf')
    training_plot_data = [] 

    # --- Initial validation before training starts ---
    if start_epoch == 0: # Only run initial validation if not resuming
        print("Running initial validation before training...")
        initial_val_cost, initial_val_gap = run_validation_custom(model, val_dataset, gt, opts)
        training_plot_data.append((0, initial_val_cost)) 
        if not np.isnan(initial_val_cost):
            best_val_cost = initial_val_cost 
    else: # If resuming, try to load previous best_val_cost if saved in checkpoint opts
        if hasattr(opts, 'best_val_cost_resumed') and opts.best_val_cost_resumed is not None:
            best_val_cost = opts.best_val_cost_resumed
            print(f"Resumed with best_val_cost: {best_val_cost}")
        # Plot data would need to be loaded if resuming and wanting to continue the plot

    print(f"===== Starting TSP-{opts.graph_size} Custom Attention Model Training ({opts.n_epochs} epochs planned) =====")
    total_training_start_time = time.time()
    
    for epoch_idx in range(start_epoch, opts.n_epochs):
        current_epoch_num = epoch_idx # 0-indexed for loop, +1 for display
        epoch_timer_start = time.time()
        
        # For standard REINFORCE, the training data is typically fixed or generated per epoch.
        # Here, base_train_dataset is loaded once by init.
        # DataLoader with shuffle=True will present different batches.
        # If opts.epoch_size is larger than len(base_train_dataset), samples will be repeated.
        
        avg_train_cost, avg_train_loss = train_epoch_custom(
            model, optimizer, baseline, opts, base_train_dataset, current_epoch_num
        )

        # Validate the model
        current_val_cost, current_val_gap = run_validation_custom(model, val_dataset, gt, opts)
        training_plot_data.append((current_epoch_num + 1, current_val_cost))
        
        # Save model if it's the best so far based on validation cost
        if not np.isnan(current_val_cost) and current_val_cost < best_val_cost:
             print(f"New best validation cost: {current_val_cost:.4f} (previous: {best_val_cost:.4f}).")
             best_val_cost = current_val_cost
             # Save best model (optional, can be frequent)
             best_model_path = os.path.join(opts.save_dir, f"{opts.model_save_filename.replace('.pt', '_best.pt')}")
             torch.save({
                 'epoch': current_epoch_num,
                 'model_state_dict': get_inner_model(model).state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
                 'best_val_cost': best_val_cost,
                 'opts': opts,
             }, best_model_path)
             print(f"Saved new best model to: {best_model_path}")
        
        # Checkpoint saving logic (e.g., every N epochs)
        if opts.checkpoint_epochs > 0 and (current_epoch_num + 1) % opts.checkpoint_epochs == 0:
            checkpoint_path = os.path.join(opts.save_dir, f"checkpoint_epoch-{current_epoch_num + 1}.pt")
            torch.save({
                'epoch': current_epoch_num,
                'model_state_dict': get_inner_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
                'best_val_cost': best_val_cost, # Save current best_val_cost
                'opts': opts,
            }, checkpoint_path)
            print(f"Saved checkpoint to: {checkpoint_path}")

        epoch_duration_secs = time.time() - epoch_timer_start
        print(f"Epoch {current_epoch_num + 1} duration: {epoch_duration_secs:.2f} seconds")

    total_training_duration_secs = time.time() - total_training_start_time
    print(f"===== Training Finished ({opts.n_epochs} total epochs run) =====")
    print(f"Total training time: {total_training_duration_secs / 3600:.2f} hours")

    # --- Save Final Model and Arguments ---
    final_model_path = os.path.join(opts.save_dir, opts.model_save_filename)
    final_args_path = os.path.join(opts.save_dir, opts.args_save_filename)

    print(f"Saving final model to: {final_model_path}")
    torch.save({
        'epoch': opts.n_epochs -1, # Save last completed epoch index
        'model_state_dict': get_inner_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
        'best_val_cost': best_val_cost,
        'opts': opts,
    }, final_model_path)

    print(f"Saving training arguments to: {final_args_path}")
    opts_dict_to_save = {k: v for k, v in vars(opts).items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    try:
        with open(final_args_path, 'w') as f:
            json.dump(opts_dict_to_save, f, indent=4)
        print("Arguments saved successfully.")
    except Exception as e:
        print(f"Error saving arguments to JSON: {e}")

    # --- Plotting Training Curve ---
    print("Generating training curve plot...")
    if training_plot_data:
        plot_epochs = [p[0] for p in training_plot_data if not np.isnan(p[1])]
        plot_costs = [p[1] for p in training_plot_data if not np.isnan(p[1])]
        
        if plot_epochs:
            plt.figure(figsize=(12, 7))
            plt.plot(plot_epochs, plot_costs, marker='o', linestyle='-', color='b', label='Avg Validation Cost')
            plt.xlabel("Epoch")
            plt.ylabel("Average Validation Cost")
            plt.title(f"TSP-{opts.graph_size} Attention Model Custom Training Curve")
            plt.grid(True)
            # Ensure x-ticks are reasonable
            step = max(1, (max(plot_epochs) - min(plot_epochs)) // 10 if max(plot_epochs) > min(plot_epochs) else 1)
            if max(plot_epochs) == 0 and min(plot_epochs) == 0 and len(plot_epochs) == 1: # Single point at epoch 0
                 plt.xticks([0])
            elif max(plot_epochs) > min(plot_epochs):
                 plt.xticks(np.arange(min(plot_epochs), max(plot_epochs) + step, step=step))
            else: # Only one epoch plotted (e.g. epoch 1)
                 plt.xticks([plot_epochs[0]])

            plt.legend()
            plt.tight_layout()
            
            plot_path = os.path.join(opts.save_dir, opts.plot_save_filename)
            try:
                plt.savefig(plot_path)
                print(f"Training curve plot saved to {plot_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")
            # plt.show() 
        else:
             print("No valid cost data (all NaN) collected for plotting.")
    else:
        print("No data collected for plotting training curve.")

    print("===== Custom Training Script Finished =====")