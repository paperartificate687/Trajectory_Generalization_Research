#!/usr/bin/env python
# coding: utf-8

# Script for TSP-100 Hardness-Adaptive Fine-Tuning and Saving

import sys
import os
import json # Added for saving args
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd # Added import

# Add the project root directory (where this script is located) to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root) # Assuming script is in root

# Imports from the project structure
from src.train import validate, clip_grad_norms, set_decode_type, get_inner_model
from src.options import get_options
from utils import torch_load_cpu, load_problem, move_to
from utils.log_utils import log_values # Added import
from problems.tsp.problem_tsp import TSP
from src.mutils import init, get_hard_samples, ConcatDataset, set_random_seed_all # Import necessary functions/classes from mutils

# --- Configuration ---
print("Setting up configuration for TSP-100 Hardness-Adaptive Fine-Tuning...")
opts = get_options('') # Load default options first

# --- Override options for TSP-100 fine-tuning run ---
opts.problem = 'tsp'
opts.graph_size = 100 # <<<--- Set graph size to 100
opts.use_cuda = True
opts.no_progress_bar = False
opts.log_dir = 'logs/tsp100_finetuned_adaptive' # Specific log directory for fine-tuning
opts.save_plot = 'tsp100_finetuned_adaptive_curve.png' # Specific plot filename for fine-tuning
opts.save_dir = 'new_main' # Reuse the save directory
opts.finetuned_model_save_filename = 'tsp100_finetuned_adaptive.pt' # Specific model save name for fine-tuning
opts.finetuned_args_save_filename = 'args_tsp100_finetuned_adaptive.json' # Specific args save name for fine-tuning

# --- Pre-trained model path ---
# opts.load_path = 'new_main/tsp100_uniform_pretrained.pt' # <<<--- Load the pre-trained uniform model
# Uncomment and set the path below to resume from a checkpoint
opts.load_path = None # Set to None to start from scratch or specify checkpoint path
# opts.load_path = 'checkpoints/tsp100_finetuned_adaptive/tsp100_checkpoint_epoch_XX.pt' # Example: replace XX with epoch number
opts.baseline = 'rollout'

# Hardness-Adaptive specific settings (should be already set)
opts.reweight = 1
# use_hard_setting = 1 # Removed as not used in tsp100

# Training duration and parameters for fine-tuning
opts.n_epochs = 100 # <<<--- Reduced epochs for fine-tuning (matching tsp50)
opts.epoch_size = 1280000
opts.batch_size = 512 # <<<--- Matching tsp50 batch size
opts.max_grad_norm = 1.0
opts.lr_model = 1e-5 # Reduced LR for fine-tuning
opts.lr_critic = 1e-5 # Reduced LR for fine-tuning
opts.lr_decay = 0.95 # Added LR decay
opts.seed = 42 # Matching tsp50 seed

# Hardness sampling parameters
opts.hardness_eps = 5
opts.hardness_train_size = 10000

# --- ASSUMED DATA PATHS FOR TSP-100 (Ensure these files exist!) ---
# These should be the same as used in the uniform pretraining script
opts.val_dataset = 'data/tsp/tsp100_val_mg_seed2222_size10K.pkl' # <<<--- Updated validation data path based on list_files
opts.val_gt_path = 'data/tsp/tsp100_val_mg_seed2222_size10K_lkh_costs.txt' # <<<--- Updated LKH costs path based on list_files
# Base training data path used by init()
opts.train_dataset = 'data/tsp/tsp100_train_seed1111_size10K.pkl' # <<<--- Updated training data path based on list_files
# ------------------------------------------------------------------

# Ensure log directory exists
os.makedirs(opts.log_dir, exist_ok=True)

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

# --- Load Validation Data ---
print(f"Loading validation data from: {opts.val_dataset}")
gt = None # Initialize gt before the try block
try:
    val_dataset = problem.make_dataset(filename=opts.val_dataset, num_samples=opts.val_size)
    print(f"Loaded validation dataset with size: {len(val_dataset)}")
    if os.path.exists(opts.val_gt_path):
        print(f"Loading ground truth costs from: {opts.val_gt_path}")
        with open(opts.val_gt_path) as file:
            lines = file.readlines()
            gt_costs_read = []
            for line in lines:
                line_strip = line.strip()
                if line_strip and line_strip.lower() != 'error':
                    try:
                        gt_costs_read.append(float(line_strip))
                    except ValueError:
                        print(f"Warning: Skipping non-float value in GT file: {line_strip}")
            gt = np.array(gt_costs_read)
            print(f"Loaded ground truth costs with size: {len(gt)}")
            if len(gt) == 0:
                print("Warning: Ground truth file loaded but resulted in an empty cost array.")
                gt = None
    else:
        print(f"Warning: Ground truth cost file not found at {opts.val_gt_path}. Validation gap will not be calculated.")
except FileNotFoundError as e:
    print(f"Error: Could not find validation data file: {e}. Please ensure TSP-100 validation data exists.")
    val_dataset = None
except Exception as e:
    print(f"Warning: Error loading validation data: {e}")
    val_dataset = None


# Functions for generating hardness-adaptive data
def get_hard_data_adaptive(model, base_dataset, size, eps, baseline, device):
    print(f"Generating {size} hardness-adaptive samples (eps={eps})...")
    # Generate random samples first
    random_samples = torch.FloatTensor(np.random.uniform(size=(size, opts.graph_size, 2)))
    # Use model to select hard samples
    hard_samples = get_hard_samples(model, random_samples, eps, batch_size=opts.eval_batch_size, device=device, baseline=baseline) # Use opts.eval_batch_size
    print(f"Generated {len(hard_samples)} hard samples.")
    return hard_samples
# --- Helper Functions (Adapted/Copied) ---

def train_batch_adaptive(
        model,
        optimizer,
        baseline,
        batch,
        epoch, # Added epoch for reweighting schedule
        opts # Pass opts directly
):
    x, bl_val = baseline.unwrap_batch(batch) # Use standard unwrap
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    loss = (cost - bl_val) * log_likelihood

    # Hardness-adaptive reweighting logic (from tsp50)
    if opts.reweight == 1:
        w = ((cost/bl_val) * log_likelihood).detach()
        t = torch.FloatTensor([20-(epoch % 20)]).to(loss.device) # Annealing schedule for weight influence
        w = torch.tanh(w)
        w /= t
        w = torch.nn.functional.softmax(w, dim=0)
        reinforce_loss = (w*loss).sum()
    else: # Fallback to standard REINFORCE if reweight is not 1 (shouldn't happen here)
        reinforce_loss = loss.mean()

    total_loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    total_loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    return cost.mean().item(), total_loss.mean().item()

# train_epoch uses the adaptive train_batch
def train_epoch_adaptive(model, optimizer, baseline, opts, train_dataset, epoch): # Removed hard_dataset parameter
    print(f"===== Starting Epoch {epoch+1}/{opts.n_epochs} =====") # Changed print message
    training_dataset = baseline.wrap_dataset(train_dataset) # Use standard wrap
    # Use opts.epoch_size to determine number of batches, not len(training_dataset) directly?
    # Let's assume DataLoader handles it correctly with drop_last=True or epoch_size is just for logging.
    # num_batches = (opts.epoch_size + opts.batch_size - 1) // opts.batch_size

    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0, shuffle=True, drop_last=True) # Set num_workers=0 for simplicity on Windows?

    model.train()
    set_decode_type(model, "sampling")

    batch_costs = []
    batch_losses = []

    pbar = tqdm(training_dataloader, desc=f"Epoch {epoch+1} Training", leave=False, disable=opts.no_progress_bar) # Changed description
    for batch in pbar:
        cost, loss = train_batch_adaptive(
            model,
            optimizer,
            baseline,
            batch,
            epoch, # Pass current epoch
            opts
        )
        batch_costs.append(cost)
        batch_losses.append(loss)
        pbar.set_postfix({'cost': cost, 'loss': loss})

    avg_cost = np.mean(batch_costs) if batch_costs else 0 # Changed default to 0
    avg_loss = np.mean(batch_losses) if batch_losses else 0 # Changed default to 0
    print(f"Epoch {epoch+1} completed. Avg Train Cost: {avg_cost:.4f}, Avg Train Loss: {avg_loss:.4f}")

    # Baseline epoch callback
    baseline.epoch_callback(model, epoch, dataset=train_dataset) # Pass original unwrapped dataset? Check baseline code. Let's pass wrapped for now.

    return avg_cost, avg_loss

# Function to get validation ratio/costs (from tsp50)
def get_validation_metrics(model, val_dataset, gt_costs, opts):
    print("Running validation...")
    model.eval() # Ensure model is in eval mode
    set_decode_type(model, "greedy") # Use greedy decoding for validation

    all_costs = []
    val_dl = DataLoader(val_dataset, batch_size=opts.eval_batch_size, shuffle=False) # Use opts.eval_batch_size
    print(f"Iterating through validation DataLoader...")
    for i, bat in enumerate(tqdm(val_dl, desc="Validation", leave=False, disable=opts.no_progress_bar)):
        if i == 0:
             print(f"Processing first validation batch...")
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
            all_costs.append(cost.data.cpu())

    print(f"Validation loop finished. Number of cost batches collected: {len(all_costs)}")

    if not all_costs:
        print("Error: No costs were collected during validation loop!")
        return [np.nan, np.nan, np.nan, np.nan]

    costs_tensor = torch.cat(all_costs, 0).numpy()
    print(f"Concatenated costs_tensor shape: {costs_tensor.shape}")

    current_gt = gt_costs
    print(f"Initial current_gt shape: {current_gt.shape}")

    if len(costs_tensor) != len(current_gt):
         print(f"Warning: Mismatch! costs_tensor length ({len(costs_tensor)}) != gt_costs length ({len(current_gt)}). Adjusting...")
         min_len = min(len(costs_tensor), len(current_gt))
         costs_tensor = costs_tensor[:min_len]
         current_gt = current_gt[:min_len]
         print(f"Adjusted shapes - costs_tensor: {costs_tensor.shape}, current_gt: {current_gt.shape}")

    if costs_tensor.size == 0 or current_gt.size == 0:
        print("Error: costs_tensor or current_gt is empty before calculating ratio!")
        return [np.nan, np.nan, costs_tensor.mean() if costs_tensor.size > 0 else np.nan, costs_tensor.max() if costs_tensor.size > 0 else np.nan]

    ratio = (costs_tensor - current_gt) / current_gt
    print(f"Calculated ratio shape: {ratio.shape}")

    if ratio.size == 0:
        print("Error: ratio array is empty!")
        mean_cost = costs_tensor.mean()
        max_cost = costs_tensor.max()
        return [np.nan, np.nan, mean_cost, max_cost]

    mean_gap = ratio.mean()
    max_gap = ratio.max()
    mean_cost = costs_tensor.mean()
    max_cost = costs_tensor.max()
    print(f"Validation results: Avg Cost={mean_cost:.4f}, Avg Gap={mean_gap:.4f}")
    return [mean_gap, max_gap, mean_cost, max_cost]


# --- Main Fine-Tuning Execution ---
if __name__ == '__main__':
    print("Initializing model, baseline, optimizer for **Fine-tuning**...") # Changed print message
    # init function should load the model specified in opts.load_path
    # Pass pretrain=True explicitly, although load_path should trigger it.
    # Also, init might reload the base_train_dataset, which is fine.
    model, base_train_dataset, _, baseline, optimizer, start_epoch = init(pretrain=True, device=device, opts=opts)

    # Check if base_train_dataset was loaded, needed for ConcatDataset
    if base_train_dataset is None:
        print("Error: Base training dataset not loaded by init function. Cannot proceed.")
        exit(1)

    print(f"Using baseline: {type(baseline).__name__}")

    plot_data = [] # List to store (epoch, mean_cost)

    # Record initial performance
    initial_metrics = get_validation_metrics(model, val_dataset, gt, opts) # Use get_validation_metrics
    plot_data.append((0, initial_metrics[2])) # Record cost at epoch 0

    print(f"===== Starting Hardness-Adaptive Fine-tuning ({opts.n_epochs} epochs) =====") # Changed print message
    for epoch in range(start_epoch, opts.n_epochs):
        epoch_start_time = time.time()

        # 1. Generate hard samples for this epoch
        hard_dataset_part = get_hard_data_adaptive(
            model=get_inner_model(model), # Pass inner model if DataParallel
            base_dataset=base_train_dataset, # Pass base dataset if needed by get_hard_samples
            size=opts.hardness_train_size,
            eps=opts.hardness_eps,
            baseline=baseline,
            device=device
        )

        # 2. Create combined training dataset for this epoch
        # Use a subset of the base training data based on opts.val_size
        # Using full base_train_dataset might be too large? Let's use a fixed number like opts.val_size for now.
        if opts.val_size > len(base_train_dataset):
             print(f"Warning: opts.val_size ({opts.val_size}) > base_train_dataset size ({len(base_train_dataset)}). Using full base dataset.")
             base_part = base_train_dataset
        else:
             base_part = Subset(base_train_dataset, list(range(opts.val_size))) # Use first val_size samples from base

        combined_train_dataset = ConcatDataset([base_part, hard_dataset_part])
        print(f"Epoch {epoch+1}: Combined training dataset size = {len(combined_train_dataset)}")

        # 3. Train one epoch using the combined dataset
        # Pass the combined dataset to train_epoch_adaptive
        avg_cost, avg_loss = train_epoch_adaptive(
            model, optimizer, baseline, opts, combined_train_dataset, epoch # Removed hard_dataset_part
        )

        # 4. Validate the model
        current_metrics = get_validation_metrics(model, val_dataset, gt, opts) # Use get_validation_metrics
        plot_data.append((epoch + 1, current_metrics[2])) # Record cost after epoch

        # Optional: Save model checkpoint periodically
        checkpoint_epoch_frequency = 10 # Save every 10 epochs
        if (epoch + 1) % checkpoint_epoch_frequency == 0 and (epoch + 1) < opts.n_epochs:
            checkpoint_save_dir = os.path.join('checkpoints', 'tsp100_finetuned_adaptive') # Use a specific checkpoint directory
            os.makedirs(checkpoint_save_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_save_dir, f'tsp100_checkpoint_epoch_{epoch + 1}.pt')
            print(f"Saving checkpoint to: {checkpoint_path}")
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': get_inner_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
                }, checkpoint_path)
                print("Checkpoint saved successfully.")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

    print("===== Fine-tuning Finished =====") # Changed print message

    # --- Added: Save Final Fine-tuned Model and Arguments ---
    final_model_save_path = os.path.join(opts.save_dir, opts.finetuned_model_save_filename) # Use finetuned filename
    final_args_save_path = os.path.join(opts.save_dir, opts.finetuned_args_save_filename) # Use finetuned filename

    print(f"Saving final fine-tuned model to: {final_model_save_path}")
    # Save model state dictionary, optimizer state, epoch number, etc.
    try:
        torch.save({
            'epoch': opts.n_epochs, # Save last fine-tuning epoch number
            'model_state_dict': get_inner_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Add baseline state if needed
            'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
            # Add relevant metrics if available (e.g., final validation cost)
            # 'final_val_cost': plot_data[-1][1] if plot_data else None
        }, final_model_save_path)
        print("Fine-tuned model saved successfully.")
    except Exception as e:
        print(f"Error saving fine-tuned model: {e}")

    print(f"Saving fine-tuning arguments to: {final_args_save_path}")
    # Convert Namespace to dict, handling potential non-serializable items if necessary
    opts_dict = vars(opts)
    opts_to_save = {k: v for k, v in opts_dict.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    try:
        with open(final_args_save_path, 'w') as f:
            json.dump(opts_to_save, f, indent=4)
        print("Arguments saved successfully.")
    except Exception as e:
        print(f"Error saving arguments to JSON: {e}")
    # --- End Added Save Logic ---

    # --- Plotting ---
    if plot_data:
        plot_epochs = [p[0] for p in plot_data]
        plot_costs = [p[1] for p in plot_data]

        plt.figure(figsize=(10, 6))
        plt.plot(plot_epochs, plot_costs, marker='o', linestyle='-')
        plt.xlabel("Epoch") # Changed xlabel
        plt.ylabel("Average Validation Cost")
        plt.title(f"TSP-{opts.graph_size} Hardness-Adaptive Fine-tuning Curve") # Changed title
        plt.grid(True)
        plt.xticks(np.arange(min(plot_epochs), max(plot_epochs)+1, step=max(1, opts.n_epochs // 10))) # Adjust x-ticks
        plt.tight_layout()

        try:
            plt.savefig(opts.save_plot) # Use opts.save_plot
            print(f"Training curve plot saved to {opts.save_plot}") # Use opts.save_plot
        except Exception as e:
            print(f"Error saving plot: {e}")
        # plt.show() # Uncomment to display plot
    else:
        print("No data collected for plotting.")