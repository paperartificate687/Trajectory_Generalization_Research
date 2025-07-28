import sys
import os
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train import validate, clip_grad_norms, set_decode_type, get_inner_model
from src.options import get_options
from utils import torch_load_cpu, load_problem, move_to
from problems.mtsp.problem_mtsp import MTSP
from src.mutils import init, ConcatDataset, set_random_seed_all

# --- Configuration ---
print("Setting up configuration for MTSP-50 (k=2) Training...")
opts = get_options('')

# --- Override options for this specific training run ---
opts.problem = 'mtsp'
opts.graph_size = 50
opts.num_salesmen = 2 # Number of salesmen for MTSP
opts.use_cuda = True
opts.no_progress_bar = False
opts.log_dir = 'logs/mtsp50_k2_trained'
opts.save_dir = 'new_main'
opts.model_save_filename = f'mtsp{opts.graph_size}_k{opts.num_salesmen}_trained.pt'
opts.args_save_filename = f'args_mtsp{opts.graph_size}_k{opts.num_salesmen}_trained.json'
opts.plot_save_filename = f'mtsp{opts.graph_size}_k{opts.num_salesmen}_curve.png'

opts.checkpoint_save_dir = f'checkpoints/mtsp{opts.graph_size}_k{opts.num_salesmen}_trained'
opts.checkpoint_epoch_frequency = 10

# Use a TSP dataset, it will be adapted by MTSPDataset
opts.train_dataset = 'data/tsp/tsp50_train_seed1111_size10K.pkl'
opts.val_dataset = 'data/tsp/tsp50_val_mg_seed2222_size10K.pkl'
opts.val_gt_path = 'data/mtsp/mtsp50_val_k2_lkh_costs.txt'

opts.load_path = None
opts.baseline = 'rollout'
opts.n_epochs = 100
opts.epoch_size = 128000
opts.batch_size = 256 # Reduced batch size for potentially larger model/state
opts.max_grad_norm = 1.0
opts.lr_model = 1e-4
opts.lr_critic = 1e-4
opts.lr_decay = 1.0
opts.seed = 1234

os.makedirs(opts.log_dir, exist_ok=True)
os.makedirs(opts.save_dir, exist_ok=True)
os.makedirs(opts.checkpoint_save_dir, exist_ok=True)

# --- Setup Device, Seed, Problem ---
if opts.use_cuda and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
opts.device = device

set_random_seed_all(opts.seed, deterministic=True)
problem = load_problem(opts.problem)

# --- Load Data ---
train_dataset = problem.make_dataset(filename=opts.train_dataset, num_samples=opts.epoch_size)
val_dataset = problem.make_dataset(filename=opts.val_dataset, num_samples=opts.val_size)
gt = None
if opts.val_gt_path is not None and os.path.exists(opts.val_gt_path):
    print(f"Loading ground truth costs from: {opts.val_gt_path}")
    with open(opts.val_gt_path) as file:
        lines = file.readlines()
        gt = np.array([float(line) for line in lines])
else:
    print("Warning: Ground truth cost file not found. Validation gap will not be calculated.")

# --- Helper Functions (mostly unchanged from TSP) ---
def train_batch_mtsp(model, optimizer, baseline, batch, opts):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Pass num_salesmen to the model forward pass if needed
    # The current AttentionModel may need slight modification to accept this
    cost, log_likelihood = model(x) # This might need adjustment
    
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    loss = (cost - bl_val) * log_likelihood
    reinforce_loss = loss.mean()
    total_loss = reinforce_loss + bl_loss

    optimizer.zero_grad()
    total_loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    
    return cost.mean().item(), total_loss.mean().item()

def train_epoch_mtsp(model, optimizer, baseline, opts, train_dataset, epoch):
    print(f"===== Starting Epoch {epoch+1}/{opts.n_epochs} for MTSP =====")
    training_dataloader = DataLoader(baseline.wrap_dataset(train_dataset), batch_size=opts.batch_size, num_workers=0, shuffle=True)
            
    model.train()
    set_decode_type(model, "sampling")
    
    batch_costs, batch_losses = [], []
    pbar = tqdm(training_dataloader, desc=f"Epoch {epoch+1} Training (MTSP)", leave=False)
    
    for batch in pbar:
        cost, loss = train_batch_mtsp(model, optimizer, baseline, batch, opts)
        batch_costs.append(cost)
        batch_losses.append(loss)
        pbar.set_postfix({'cost': cost, 'loss': loss})
    
    avg_cost = np.mean(batch_costs)
    avg_loss = np.mean(batch_losses)
    print(f"Epoch {epoch+1} completed. Avg Train Cost: {avg_cost:.4f}, Avg Train Loss: {avg_loss:.4f}")
    
    baseline.epoch_callback(model, epoch, dataset=train_dataset)
    return avg_cost, avg_loss

def run_validation_mtsp(model, val_dataset, gt_costs, opts):
    print("Running validation...")
    model.eval()
    set_decode_type(model, "greedy")
    
    all_costs = []
    val_dl = DataLoader(val_dataset, batch_size=opts.eval_batch_size, shuffle=False)
    for bat in tqdm(val_dl, desc="Validation", leave=False):
         with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
            all_costs.append(cost.data.cpu())
            
    costs_tensor = torch.cat(all_costs, 0).numpy()
    mean_cost = costs_tensor.mean()
    
    mean_gap = np.nan
    if gt_costs is not None:
        mean_gap = ((costs_tensor - gt_costs) / gt_costs).mean()
        print(f"Validation results: Avg Cost={mean_cost:.4f}, Avg Gap={mean_gap:.4f}")
    else:
        print(f"Validation results: Avg Cost={mean_cost:.4f}")
        
    return mean_cost, mean_gap

# --- Main Training Execution ---
if __name__ == '__main__':
    # The init function needs to be aware of MTSP and its parameters
    # We might need to adjust it or pass parameters differently.
    model, _, _, baseline, optimizer, start_epoch = init(pretrain=bool(opts.load_path), device=device, opts=opts)

    if model is None:
        print("Error: Model initialization failed.")
        exit(1)
        
    best_val_cost = float('inf')
    plot_data = []

    initial_val_cost, initial_val_gap = run_validation_mtsp(model, val_dataset, gt, opts)
    plot_data.append((start_epoch, initial_val_cost))
    if not np.isnan(initial_val_cost):
        best_val_cost = initial_val_cost

    for epoch in range(start_epoch, opts.n_epochs):
        avg_cost, avg_loss = train_epoch_mtsp(model, optimizer, baseline, opts, train_dataset, epoch)
        val_cost, val_gap = run_validation_mtsp(model, val_dataset, gt, opts)
        plot_data.append((epoch + 1, val_cost))
        
        if not np.isnan(val_cost) and val_cost < best_val_cost:
             print(f"New best validation cost: {val_cost:.4f}")
             best_val_cost = val_cost
        
        if (epoch + 1) % opts.checkpoint_epoch_frequency == 0:
            checkpoint_path = os.path.join(opts.checkpoint_save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': get_inner_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'baseline_state_dict': getattr(baseline, 'state_dict', lambda: None)(),
                'best_val_cost': best_val_cost,
                'plot_data': plot_data
            }, checkpoint_path)

    # --- Save Final Model ---
    final_model_save_path = os.path.join(opts.save_dir, opts.model_save_filename)
    torch.save({
        'model_state_dict': get_inner_model(model).state_dict(),
        'opts': opts,
    }, final_model_save_path)

    # --- Plotting ---
    plot_epochs = [p[0] for p in plot_data]
    plot_costs = [p[1] for p in plot_data]
    plt.figure()
    plt.plot(plot_epochs, plot_costs, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Avg Validation Cost")
    plt.title(f"MTSP-{opts.graph_size} (k={opts.num_salesmen}) Training Curve")
    plt.grid(True)
    plot_save_path = os.path.join(opts.save_dir, opts.plot_save_filename)
    plt.savefig(plot_save_path)
    print(f"Training curve saved to {plot_save_path}")

    print("===== MTSP Training Script Finished =====")