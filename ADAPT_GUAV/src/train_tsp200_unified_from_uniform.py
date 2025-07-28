import sys
import os
import json # Added for saving args
import math
import matplotlib.pyplot as plt # Added import
import glob # Added for finding latest checkpoint
from itertools import cycle

# Add the project root directory (where this script is located) to the Python path
# Determine the project root directory, assuming the script is in the root of the project structure
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Script for TSP-200 Unified Pretraining (based on uniform logic) and Saving

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.cuda.amp import GradScaler # OPTIMIZATION: For Automatic Mixed Precision

# Imports from the project structure
from src.train import validate, clip_grad_norms, set_decode_type, get_inner_model
from src.options import get_options
from utils import torch_load_cpu, load_problem, move_to
from problems.tsp.problem_tsp import TSP
from src.mutils import init, ConcatDataset, set_random_seed_all



# --- Helper Functions (Copied from train_and_save_tsp200.py) ---

def train_batch_optimized(
        model,
        optimizer,
        baseline,
        batch,
        opts,
        scaler
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # OPTIMIZATION: Forward pass with autocasting for mixed precision
    with torch.autocast(device_type='cuda', enabled=opts.use_cuda):
        cost, log_likelihood = model(x)
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
        loss = (cost - bl_val) * log_likelihood
        reinforce_loss = loss.mean()
        total_loss = reinforce_loss + bl_loss

    optimizer.zero_grad()
    # OPTIMIZATION: Scales loss. Calls backward() on scaled loss to create scaled gradients.
    scaler.scale(total_loss).backward()
    # OPTIMIZATION: Unscales the gradients of optimizer's assigned params in-place
    scaler.unscale_(optimizer)
    # Since the gradients of optimizer's assigned params are unscaled, clip_grad_norm can be called.
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    # OPTIMIZATION: scaler.step() first unscales the gradients of the optimizer's assigned params.
    scaler.step(optimizer)
    # OPTIMIZATION: Updates the scale for next iteration.
    scaler.update()
    
    return cost.mean().item(), total_loss.mean().item()

def train_epoch_uniform(model, optimizer, baseline, opts, train_dataset, epoch, scaler):
    print(f"===== Starting Epoch {epoch+1}/{opts.n_epochs} =====")
    # The dataloader will loop over the dataset to fill the epoch_size
    training_dataset_wrapped = baseline.wrap_dataset(train_dataset)
    num_batches_to_process = opts.epoch_size // opts.batch_size
    
    training_dataloader = DataLoader(training_dataset_wrapped, batch_size=opts.batch_size, num_workers=0, shuffle=True, pin_memory=True)

    model.train()
    set_decode_type(model, "sampling")

    batch_costs = []
    batch_losses = []
    
    dataloader_iterator = cycle(training_dataloader)
    
    with tqdm(range(num_batches_to_process), desc=f"Epoch {epoch+1} Training", leave=True, disable=opts.no_progress_bar) as pbar:
        for _ in pbar:
            batch = next(dataloader_iterator)
            cost, loss = train_batch_optimized(
                model,
                optimizer,
                baseline,
                batch,
                opts,
                scaler
            )
            batch_costs.append(cost)
            batch_losses.append(loss)
            pbar.set_postfix({'cost': cost, 'loss': loss})

    avg_cost = np.mean(batch_costs) if batch_costs else 0
    avg_loss = np.mean(batch_losses) if batch_losses else 0
    print(f"Epoch {epoch+1} completed. Avg Train Cost: {avg_cost:.4f}, Avg Train Loss: {avg_loss:.4f}")

    baseline.epoch_callback(model, epoch, dataset=train_dataset)

    return avg_cost, avg_loss

def run_validation(model, val_dataset, gt_costs, opts):
    if val_dataset is None:
        print("Skipping validation as validation data failed to load.")
        return np.nan, np.nan

    print("Running validation...")
    model.eval()
    set_decode_type(model, "greedy")

    all_costs = []
    # OPTIMIZATION: Also optimize validation loader
    val_dl = DataLoader(
        val_dataset, batch_size=opts.eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    for bat in tqdm(val_dl, desc="Validation", leave=False, disable=opts.no_progress_bar):
         with torch.no_grad():
            # OPTIMIZATION: Use autocast for validation as well
            with torch.autocast(device_type='cuda', enabled=opts.use_cuda):
                cost, _ = model(move_to(bat, opts.device))
            all_costs.append(cost.data.cpu())

    if not all_costs:
        print("Error: No costs collected during validation.")
        return np.nan, np.nan

    costs_tensor = torch.cat(all_costs, 0).numpy()
    mean_cost = costs_tensor.mean()

    mean_gap = np.nan
    if gt_costs is not None and len(costs_tensor) == len(gt_costs):
        ratio = (costs_tensor - gt_costs) / gt_costs
        mean_gap = ratio.mean()
    elif gt_costs is not None:
         print(f"Warning: Validation costs ({len(costs_tensor)}) and GT costs ({len(gt_costs)}) length mismatch.")

    print(f"Validation results: Avg Cost={mean_cost:.4f}, Avg Gap={mean_gap:.4f}")
    return mean_cost, mean_gap


# --- Main Training Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    print("Setting up configuration for TSP-200 Unified (from Uniform Logic) Pretraining...")
    opts = get_options('') # Load default options first

    # --- Override options for this specific training run ---
    opts.problem = 'tsp'
    opts.graph_size = 200
    opts.use_cuda = True
    opts.no_progress_bar = False
    opts.log_dir = os.path.join(project_root, 'logs/tsp200_unified_from_uniform') # Specific log directory for this run
    opts.save_dir = os.path.join(project_root, 'new_main') # Directory to save model and args
    opts.model_save_filename = 'tsp200_unified_from_uniform.pt'
    opts.args_save_filename = 'args_tsp200_unified_from_uniform.json'
    opts.plot_save_filename = 'tsp200_unified_from_uniform_curve.png' # Added plot filename
    opts.checkpoint_epochs = 10 # Save checkpoint every N epochs, 0 to disable
    opts.checkpoint_save_dir = os.path.join(project_root, 'checkpoints/tsp200_unified_from_uniform') # Directory to save checkpoints

    # Define the path to the unified training dataset
    opts.train_dataset = os.path.join(project_root, 'data/tsp/tsp200_train_unified_seed5678_size10K.pkl')

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

    opts.val_dataset = os.path.join(project_root, 'data/tsp/tsp200_val_mg_seed2222_size10K.pkl')
    opts.val_gt_path = os.path.join(project_root, 'data/tsp/tsp200_val_mg_seed2222_size10K_lkh_costs.txt')

    opts.load_path = None # Ensure no pre-trained model is loaded
    opts.baseline = 'rollout' # Use rollout baseline
    opts.reweight = 0
    # Training duration and parameters from train_and_save_tsp200.py
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

    # --- Load Training Data (Unified) ---
    print(f"Loading **Unified** training data from: {opts.train_dataset}")
    try:
        # Load the specified dataset from file, do not rely on init() to generate data
        train_dataset = problem.make_dataset(filename=opts.train_dataset)
        print(f"Loaded Unified training dataset with size: {len(train_dataset)}")
    except FileNotFoundError as e:
        print(f"Error: Unified training data file not found at {opts.train_dataset}.")
        print(f"Please ensure the file exists.")
        exit(1)
    except Exception as e:
        print(f"Error loading unified training data: {e}")
        exit(1)

    # --- Load Validation Data ---
    print(f"Loading validation data from: {opts.val_dataset}")
    try:
        val_dataset = problem.make_dataset(filename=opts.val_dataset, num_samples=opts.val_size)
        print(f"Loaded validation dataset with size: {len(val_dataset)}")
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
        val_dataset = None
    except Exception as e:
        print(f"Warning: Error loading validation data: {e}")
        val_dataset = None
    
    start_epoch = 0
    best_val_cost = float('inf')
    plot_data = []

    # We pass pretrain=True to init so it does not load any default dataset
    # The train_dataset is loaded manually above
    model, _, _, baseline, optimizer, _ = init(pretrain=True, device=device, opts=opts)
    
    # OPTIMIZATION: Initialize GradScaler for Automatic Mixed Precision (AMP)
    scaler = GradScaler(enabled=opts.use_cuda)

    if opts.load_checkpoint_path and os.path.exists(opts.load_checkpoint_path):
        print(f"Loading checkpoint from: {opts.load_checkpoint_path}")
        checkpoint = torch_load_cpu(opts.load_checkpoint_path)
        
        get_inner_model(model).load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = move_to(v, opts.device)
        if 'baseline_state_dict' in checkpoint and baseline is not None and hasattr(baseline, 'load_state_dict'):
            baseline.load_state_dict(checkpoint['baseline_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0)
        best_val_cost = checkpoint.get('best_val_cost', float('inf'))
        plot_data = checkpoint.get('plot_data', [])
        
        print(f"Resuming training from epoch {start_epoch + 1}")

    else:
        if opts.load_checkpoint_path:
            print(f"Warning: Checkpoint file not found at {opts.load_checkpoint_path}. Starting from scratch.")
        print("Initializing model, baseline, optimizer for training from scratch...")
        
        print("Running initial validation for new training...")
        initial_val_cost, initial_val_gap = run_validation(model, val_dataset, gt, opts)
        plot_data.append((0, initial_val_cost))
        if not np.isnan(initial_val_cost):
            best_val_cost = initial_val_cost

    print(f"Using baseline: {type(baseline).__name__}")
    print(f"Using training dataset: {opts.train_dataset} with size {len(train_dataset)}")

    print(f"===== Starting Training (from epoch {start_epoch + 1} to {opts.n_epochs}) =====")
    start_time_total = time.time()

    for epoch in range(start_epoch, opts.n_epochs):
        epoch_start_time = time.time()

        avg_cost, avg_loss = train_epoch_uniform(
            model, optimizer, baseline, opts, train_dataset, epoch, scaler
        )

        val_cost, val_gap = run_validation(model, val_dataset, gt, opts)
        plot_data.append((epoch + 1, val_cost))

        if not np.isnan(val_cost) and val_cost < best_val_cost:
             print(f"New best validation cost: {val_cost:.4f} (previous: {best_val_cost:.4f}).")
             best_val_cost = val_cost

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")

        if opts.checkpoint_epochs > 0 and (epoch + 1) % opts.checkpoint_epochs == 0 and (epoch + 1) < opts.n_epochs:
            checkpoint_path = os.path.join(opts.checkpoint_save_dir, f"tsp{opts.graph_size}_checkpoint_epoch_{epoch + 1}.pt")
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': get_inner_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
                'best_val_cost': best_val_cost,
                'plot_data': plot_data,
                'opts': opts,
            }, checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch + 1}.")

    total_training_time = time.time() - start_time_total
    print(f"===== Training Finished ({opts.n_epochs} epochs) =====")
    print(f"Total training time: {total_training_time / 3600:.2f} hours")

    final_model_save_path = os.path.join(opts.save_dir, opts.model_save_filename)
    final_args_save_path = os.path.join(opts.save_dir, opts.args_save_filename)

    print(f"Saving final model to: {final_model_save_path}")
    torch.save({
        'epoch': opts.n_epochs,
        'model_state_dict': get_inner_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
        'best_val_cost': best_val_cost,
        'plot_data': plot_data,
        'opts': opts,
    }, final_model_save_path)

    print(f"Saving training arguments to: {final_args_save_path}")
    opts_dict = vars(opts)
    opts_to_save = {k: v for k, v in opts_dict.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    try:
        with open(final_args_save_path, 'w') as f:
            json.dump(opts_to_save, f, indent=4)
        print("Arguments saved successfully.")
    except Exception as e:
        print(f"Error saving arguments to JSON: {e}")

    print("===== Script Finished =====")

    print("Generating training curve plot...")
    if plot_data:
        plot_epochs = [p[0] for p in plot_data if not np.isnan(p[1])]
        plot_costs = [p[1] for p in plot_data if not np.isnan(p[1])]

        if plot_epochs:
            plt.figure(figsize=(10, 6))
            plt.plot(plot_epochs, plot_costs, marker='o', linestyle='-')
            plt.xlabel("Epoch")
            plt.ylabel("Average Validation Cost")
            plt.title(f"TSP-{opts.graph_size} Unified (from Uniform Logic) Training Curve")
            plt.grid(True)
            plt.xticks(np.arange(min(plot_epochs), max(plot_epochs)+1, step=max(1, opts.n_epochs // 10)))
            plt.tight_layout()
            
            plot_save_path = os.path.join(opts.save_dir, opts.plot_save_filename)
            try:
                plt.savefig(plot_save_path)
                print(f"Training curve plot saved to {plot_save_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")
        else:
             print("No valid cost data collected for plotting (all were NaN?).")
    else:
        print("No data collected for plotting.")