import sys
import os
import json
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from itertools import cycle

from src.train import clip_grad_norms, set_decode_type, get_inner_model
from src.options import get_options
from utils import torch_load_cpu, load_problem, move_to
from problems.tsp.problem_tsp import TSP
from src.mutils import init, set_random_seed_all

def train_batch_unified(
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

    with torch.autocast(device_type='cuda', enabled=opts.use_cuda):
        cost, log_likelihood = model(x)
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
        loss = (cost - bl_val) * log_likelihood
        reinforce_loss = loss.mean()
        total_loss = reinforce_loss + bl_loss

    optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    
    return cost.mean().item(), total_loss.mean().item()

def train_epoch_unified(model, optimizer, baseline, opts, train_dataset_to_use, epoch, scaler):
    print(f"===== Starting Epoch {epoch+1}/{opts.n_epochs} for Unified Dataset (Graph Size: {opts.graph_size}) =====")
    training_dataset_wrapped = baseline.wrap_dataset(train_dataset_to_use)
    num_batches_to_process = opts.epoch_size // opts.batch_size
    
    # OPTIMIZATION: Set num_workers to 0 to avoid multiprocessing issues on Windows
    training_dataloader = DataLoader(training_dataset_wrapped, batch_size=opts.batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=True if len(training_dataset_wrapped) >= opts.batch_size else False)
            
    model.train()
    set_decode_type(model, "sampling")
    
    batch_costs = []
    batch_losses = []
    
    # OPTIMIZATION: Use itertools.cycle for efficient dataloading
    dataloader_iterator = cycle(training_dataloader)
    
    # OPTIMIZATION: Set leave=True for better progress visibility
    with tqdm(range(num_batches_to_process), desc=f"Epoch {epoch+1} Training (Unified G{opts.graph_size})", leave=True, disable=opts.no_progress_bar) as pbar:
        for _ in pbar:
            batch = next(dataloader_iterator)
            cost, loss = train_batch_unified(
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
    print(f"Epoch {epoch+1} completed. Avg Train Cost (Unified G{opts.graph_size}): {avg_cost:.4f}, Avg Train Loss: {avg_loss:.4f}")
    
    baseline.epoch_callback(model, epoch, dataset=train_dataset_to_use)
    
    return avg_cost, avg_loss

def run_validation(model, current_val_dataset, gt_costs, opts):
    if current_val_dataset is None:
        print("Skipping validation as validation data failed to load.")
        return np.nan, np.nan

    print(f"Running validation for G{opts.graph_size}...")
    model.eval()
    set_decode_type(model, "greedy")
    
    all_costs = []
    val_dl = DataLoader(
        current_val_dataset, batch_size=opts.eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    for bat in tqdm(val_dl, desc=f"Validation G{opts.graph_size}", leave=False, disable=opts.no_progress_bar):
         with torch.no_grad():
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
         print(f"Warning: Validation costs ({len(costs_tensor)}) and GT costs ({len(gt_costs)}) length mismatch. Cannot calculate accurate gap.")

    print(f"Validation results (G{opts.graph_size}): Avg Cost={mean_cost:.4f}, Avg Gap={mean_gap:.4f}")
    return mean_cost, mean_gap

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print("Setting up configuration for TSP-200 Unified Dataset Training...")
    opts = get_options('')

    opts.problem = 'tsp'
    opts.graph_size = 200
    opts.use_cuda = True
    opts.no_progress_bar = False
    opts.log_dir = 'logs/tsp200_unified_trained'
    opts.save_dir = 'new_main'
    opts.model_save_filename = 'tsp200_unified_trained.pt'
    opts.args_save_filename = 'args_tsp200_unified_trained.json'
    opts.plot_save_filename = 'tsp200_unified_curve.png'
    opts.checkpoint_save_dir = 'checkpoints/tsp200_unified_trained'
    opts.checkpoint_epoch_frequency = 10
    opts.train_dataset = os.path.join(project_root, 'data/tsp/tsp200_train_unified_seed5678_size10K.pkl')
    opts.val_dataset = os.path.join(project_root, 'data/tsp/tsp200_val_mg_seed2222_size10K.pkl')
    opts.val_gt_path = os.path.join(project_root, 'data/tsp/tsp200_val_mg_seed2222_size10K_lkh_costs.txt')
    opts.load_path = None
    opts.baseline = 'rollout'
    opts.reweight = 0
    opts.n_epochs = 100
    opts.epoch_size = 1280000
    opts.batch_size = 512
    opts.max_grad_norm = 1.0
    opts.lr_model = 1e-4
    opts.lr_critic = 1e-4
    opts.lr_decay = 0.98
    opts.seed = 5678

    os.makedirs(opts.log_dir, exist_ok=True)
    os.makedirs(opts.save_dir, exist_ok=True)
    os.makedirs(opts.checkpoint_save_dir, exist_ok=True)

    if opts.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    opts.device = device

    set_random_seed_all(opts.seed, deterministic=True)
    problem = load_problem(opts.problem)

    print(f"Loading **Unified** training data from: {opts.train_dataset}")
    try:
        train_dataset_unified = problem.make_dataset(filename=opts.train_dataset, num_samples=opts.epoch_size)
        print(f"Loaded Unified training dataset with size: {len(train_dataset_unified)}")
    except FileNotFoundError as e:
        print(f"Error: Unified training data file not found at {opts.train_dataset}.")
        print(f"Please generate it first using: python src/generate_data.py --problem tsp --graph_sizes {opts.graph_size} --name train_unified --seed YOUR_SEED --dataset_size YOUR_SIZE --generate_type unified -f")
        exit(1)
    except Exception as e:
        print(f"Error loading unified training data: {e}")
        exit(1)

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

    print(f"Initializing model, baseline, optimizer for training with Unified dataset (Graph Size: {opts.graph_size})...")
    model, _, _, baseline, optimizer, start_epoch = init(pretrain=bool(opts.load_path), device=device, opts=opts)

    if model is None:
        print("Error: Model initialization failed. Check 'load_path' or dataset paths.")
        exit(1)
        
    if torch.__version__ >= "2.0.0":
        print("PyTorch 2.0+ detected. Compiling model for performance...")
        print("Model compilation with torch.compile is temporarily disabled to avoid Triton error.")
        
    print(f"Using baseline: {type(baseline).__name__}")
    
    scaler = GradScaler(enabled=opts.use_cuda)
    
    best_val_cost = float('inf')
    plot_data = []

    print(f"Running initial validation before training (G{opts.graph_size})...")
    initial_val_cost, initial_val_gap = run_validation(model, val_dataset, gt, opts)
    plot_data.append((start_epoch, initial_val_cost))
    if not np.isnan(initial_val_cost):
        best_val_cost = initial_val_cost

    print(f"===== Starting Unified Dataset Training from epoch {start_epoch + 1} to {opts.n_epochs} (G{opts.graph_size}) =====")
    start_time_total = time.time()
    
    for epoch in range(start_epoch, opts.n_epochs):
        epoch_start_time = time.time()
        
        print(f"Epoch {epoch+1} (G{opts.graph_size}): Using **Unified** training dataset of size {len(train_dataset_unified)}")

        avg_cost, avg_loss = train_epoch_unified(
            model, optimizer, baseline, opts, train_dataset_unified, epoch, scaler
        )

        val_cost, val_gap = run_validation(model, val_dataset, gt, opts)
        plot_data.append((epoch + 1, val_cost)) 
        
        if not np.isnan(val_cost) and val_cost < best_val_cost: 
             print(f"New best validation cost (G{opts.graph_size}): {val_cost:.4f} (previous: {best_val_cost:.4f}).")
             best_val_cost = val_cost
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} (G{opts.graph_size}) duration: {epoch_duration:.2f} seconds")

        if (epoch + 1) % opts.checkpoint_epoch_frequency == 0 and (epoch + 1) < opts.n_epochs:
            checkpoint_path = os.path.join(opts.checkpoint_save_dir, f'tsp{opts.graph_size}_unified_checkpoint_epoch_{epoch + 1}.pt')
            print(f"Saving checkpoint to: {checkpoint_path}")
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': get_inner_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
                    'best_val_cost': best_val_cost,
                    'plot_data': plot_data
                }, checkpoint_path)
                print("Checkpoint saved successfully.")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

    total_training_time = time.time() - start_time_total
    print(f"===== Unified Training Finished (Reached epoch {opts.n_epochs}, G{opts.graph_size}) =====")
    print(f"Total training time: {total_training_time / 3600:.2f} hours")

    final_model_save_path = os.path.join(opts.save_dir, opts.model_save_filename)
    final_args_save_path = os.path.join(opts.save_dir, opts.args_save_filename)

    print(f"Saving final model (trained on Unified G{opts.graph_size}) to: {final_model_save_path}")
    torch.save({
        'epoch': opts.n_epochs, 
        'model_state_dict': get_inner_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(), 
        'best_val_cost': best_val_cost, 
        'opts': opts, 
    }, final_model_save_path)

    print(f"Saving training arguments (Unified G{opts.graph_size}) to: {final_args_save_path}")
    opts_dict = vars(opts) 
    opts_to_save = {k: v for k, v in opts_dict.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    try:
        with open(final_args_save_path, 'w') as f:
            json.dump(opts_to_save, f, indent=4)
        print("Arguments saved successfully.")
    except Exception as e:
        print(f"Error saving arguments to JSON: {e}")

    print(f"===== Unified Training Script Finished (G{opts.graph_size}) =====") 
    
    print(f"Generating training curve plot (Unified G{opts.graph_size})...")
    if plot_data:
        plot_epochs = [p[0] for p in plot_data if not np.isnan(p[1])] 
        plot_costs = [p[1] for p in plot_data if not np.isnan(p[1])]
        
        if plot_epochs: 
            plt.figure(figsize=(10, 6))
            plt.plot(plot_epochs, plot_costs, marker='o', linestyle='-')
            plt.xlabel("Epoch")
            plt.ylabel("Average Validation Cost")
            plt.title(f"TSP-{opts.graph_size} Unified Dataset Training Curve")
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

if __name__ == '__main__':
    main()