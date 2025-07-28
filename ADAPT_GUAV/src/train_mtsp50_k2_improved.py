import sys
import os
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Project Imports ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.train import clip_grad_norms, set_decode_type, get_inner_model
from src.options import get_options
from utils import torch_load_cpu, load_problem, move_to
from src.mutils import init, get_hard_samples, ConcatDataset, set_random_seed_all
# --- 2-Opt Heuristic Implementation ---

def improve_tour_with_2opt_np(tour_indices, dist_matrix):
    """Improves a single closed tour using 2-opt, operating on numpy arrays."""
    best_tour = np.copy(tour_indices)
    num_nodes = len(best_tour)
    if num_nodes < 4:
        return best_tour

    improved = True
    while improved:
        improved = False
        # We don't want to change the depot, so we iterate from 1 to num_nodes-2
        for i in range(1, num_nodes - 2):
            for j in range(i + 1, num_nodes - 1):
                # Current tour: ...-A-B-...-C-D-... where A=i-1, B=i, C=j, D=j+1
                # We check if swapping B and C is better: ...-A-C-...-B-D-...
                # This is done by reversing the segment from B to C (i to j)
                current_dist = dist_matrix[best_tour[i - 1], best_tour[i]] + dist_matrix[best_tour[j], best_tour[j + 1]]
                new_dist = dist_matrix[best_tour[i - 1], best_tour[j]] + dist_matrix[best_tour[i], best_tour[j + 1]]

                if new_dist < current_dist:
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    improved = True
    return best_tour

def calculate_tour_cost_np(tour_indices, dist_matrix):
    """Calculates cost of a tour given by indices and a distance matrix."""
    cost = 0
    for i in range(len(tour_indices) - 1):
        cost += dist_matrix[tour_indices[i], tour_indices[i+1]]
    return cost

def apply_2_opt_and_recalculate_cost(pi, batch, num_salesmen):
    """
    Applies 2-opt to each sub-tour in the batch and returns the new total costs.
    """
    batch_size, seq_len = pi.size()
    loc_with_depot = torch.cat((batch['depot'], batch['loc']), 1)
    new_costs = torch.zeros(batch_size, device=pi.device)

    for i in range(batch_size):
        dist_matrix = torch.cdist(loc_with_depot[i], loc_with_depot[i], p=2).cpu().numpy()
        instance_pi = pi[i].cpu().numpy()
        
        depot_visits = np.where(instance_pi == 0)[0]
        
        sub_tours_indices = []
        start_idx = 0
        for depot_idx in depot_visits:
            segment = instance_pi[start_idx:depot_idx]
            full_tour = np.concatenate(([0], segment, [0]))
            sub_tours_indices.append(full_tour)
            start_idx = depot_idx + 1
        
        if start_idx < len(instance_pi):
             segment = instance_pi[start_idx:]
             full_tour = np.concatenate(([0], segment, [0]))
             sub_tours_indices.append(full_tour)

        total_instance_cost = 0
        for tour in sub_tours_indices:
            if len(tour) < 3:
                continue
            
            improved_tour = improve_tour_with_2opt_np(tour, dist_matrix)
            total_instance_cost += calculate_tour_cost_np(improved_tour, dist_matrix)

        new_costs[i] = float(total_instance_cost)
        
    return new_costs

# --- Configuration for Improved MTSP-50 (k=2) Fine-tuning ---
print("Setting up configuration for Improved MTSP-50 (k=2) Hardness-Adaptive Fine-tuning...")
opts = get_options('')

# --- Override options ---
opts.problem = 'mtsp'
opts.graph_size = 50
opts.num_salesmen = 2
opts.use_cuda = True
opts.no_progress_bar = False
opts.log_dir = 'logs/mtsp50_k2_improved'
opts.save_dir = 'new_main'
opts.finetuned_model_save_filename = 'mtsp50_k2_improved.pt'
opts.finetuned_args_save_filename = 'args_mtsp50_k2_improved.json'
opts.plot_save_filename = 'mtsp50_k2_improved_curve.png'

# --- CRITICAL: Load the pre-trained TSP model as a starting point ---
# This is the key for better performance. We fine-tune from a model that already understands TSP.
opts.load_path = 'new_main/tsp50_uniform_pretrained.pt'

opts.val_dataset = 'data/tsp/tsp50_val_mg_seed2222_size10K.pkl'
opts.val_gt_path = 'data/mtsp/mtsp50_val_k2_lkh_costs.txt' # Use MTSP ground truth for validation
opts.baseline = 'rollout'

# --- Hardness-Adaptive Settings ---
opts.reweight = 1
opts.hardness_eps = 5
opts.hardness_train_size = 10000

# --- Fine-tuning Duration & Parameters ---
opts.n_epochs = 100 # A shorter run to demonstrate improvement
opts.epoch_size = 1280000
opts.batch_size = 512 # Adjusted for MTSP
opts.max_grad_norm = 1.0
opts.lr_model = 1e-5 # Use a smaller learning rate for fine-tuning
opts.lr_critic = 1e-5
opts.lr_decay = 0.98
opts.seed = 2024

# --- Setup Directories, Device, Seed, Problem ---
os.makedirs(opts.log_dir, exist_ok=True)
os.makedirs(opts.save_dir, exist_ok=True)

if opts.use_cuda and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
opts.device = device

set_random_seed_all(opts.seed, deterministic=True)
problem = load_problem(opts.problem)

# --- Load Validation Data ---
print(f"Loading validation data from: {opts.val_dataset}")
val_dataset = problem.make_dataset(filename=opts.val_dataset, num_samples=opts.val_size)
gt = None
if opts.val_gt_path and os.path.exists(opts.val_gt_path):
    print(f"Loading ground truth costs from: {opts.val_gt_path}")
    with open(opts.val_gt_path) as f:
        gt = np.array([float(line) for line in f.readlines()])
else:
    print("Warning: MTSP ground truth not found. Validation gap will not be calculated.")

# --- Helper Functions (Adapted for MTSP from adaptive script) ---

def train_batch_adaptive_mtsp(model, optimizer, baseline, batch, epoch, opts):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    cost, log_likelihood = model(x)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    loss = (cost - bl_val) * log_likelihood

    if opts.reweight == 1:
        w = ((cost / bl_val) * log_likelihood).detach()
        t = torch.FloatTensor([20 - (epoch % 20)]).to(loss.device)
        w = torch.tanh(w / t)
        w = torch.nn.functional.softmax(w, dim=0)
        reinforce_loss = (w * loss).sum()
    else:
        reinforce_loss = loss.mean()

    total_loss = reinforce_loss + bl_loss
    optimizer.zero_grad()
    total_loss.backward()
    clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    return cost.mean().item(), total_loss.mean().item()

def train_epoch_adaptive_mtsp(model, optimizer, baseline, opts, train_dataset, epoch):
    print(f"===== Starting Fine-tuning Epoch {epoch+1}/{opts.n_epochs} for MTSP =====")
    training_dataloader = DataLoader(baseline.wrap_dataset(train_dataset), batch_size=opts.batch_size, num_workers=0, shuffle=True)
    model.train()
    set_decode_type(model, "sampling")
    
    batch_costs, batch_losses = [], []
    pbar = tqdm(training_dataloader, desc=f"Epoch {epoch+1} Fine-tuning (MTSP)", leave=False)
    for batch in pbar:
        cost, loss = train_batch_adaptive_mtsp(model, optimizer, baseline, batch, epoch, opts)
        batch_costs.append(cost)
        batch_losses.append(loss)
        pbar.set_postfix({'cost': f'{cost:.4f}', 'loss': f'{loss:.4f}'})
    
    avg_cost = np.mean(batch_costs)
    avg_loss = np.mean(batch_losses)
    print(f"Epoch {epoch+1} completed. Avg Train Cost: {avg_cost:.4f}, Avg Train Loss: {avg_loss:.4f}")
    baseline.epoch_callback(model, epoch, dataset=train_dataset)
    return avg_cost, avg_loss

def run_validation_mtsp(model, val_dataset, gt_costs, opts):
    print("Running validation on MTSP dataset with 2-opt post-processing...")
    model.eval()
    set_decode_type(model, "greedy")
    
    all_costs_optimized = []
    val_dl = DataLoader(val_dataset, batch_size=opts.eval_batch_size, shuffle=False)
    for bat in tqdm(val_dl, desc="Validation (MTSP with 2-Opt)", leave=False):
        with torch.no_grad():
            # Model inference returns the tour (pi)
            _, _, pi = model(move_to(bat, opts.device), return_pi=True)
            
            # Apply 2-opt to the generated tours and get new costs
            optimized_costs = apply_2_opt_and_recalculate_cost(pi, bat, opts.num_salesmen)
            all_costs_optimized.append(optimized_costs.cpu())
            
    costs_tensor = torch.cat(all_costs_optimized, 0).numpy()
    mean_cost = costs_tensor.mean()
    
    mean_gap = np.nan
    if gt_costs is not None:
        mean_gap = ((costs_tensor - gt_costs) / gt_costs).mean()
        print(f"Validation results (2-Opt): Avg Cost={mean_cost:.4f}, Avg Gap={mean_gap:.4f}")
    else:
        print(f"Validation results (2-Opt): Avg Cost={mean_cost:.4f}")
    return mean_cost, mean_gap

# --- Main Training Execution ---
if __name__ == '__main__':
    print("Initializing model for fine-tuning. Loading pre-trained TSP model...")
    # The init function will load the model from `opts.load_path`
    model, base_train_dataset, _, baseline, optimizer, _ = init(pretrain=True, device=device, opts=opts)
    
    if base_train_dataset is None:
        print("Error: Base training dataset not loaded. Cannot proceed.")
        exit(1)

    best_val_cost = float('inf')
    plot_data = []

    initial_val_cost, _ = run_validation_mtsp(model, val_dataset, gt, opts)
    plot_data.append((0, initial_val_cost))
    if not np.isnan(initial_val_cost):
        best_val_cost = initial_val_cost

    print(f"===== Starting MTSP Hardness-Adaptive Fine-tuning ({opts.n_epochs} epochs) =====")
    for epoch in range(opts.n_epochs):
        # Generate hard samples by creating random data and then hardening it
        print(f"Generating {opts.hardness_train_size} random samples for hardening...")
        random_data_for_hardening = torch.FloatTensor(np.random.uniform(size=(opts.hardness_train_size, opts.graph_size, 2)))
        
        hard_dataset_part = get_hard_samples(
            model=get_inner_model(model),
            data=random_data_for_hardening,
            eps=opts.hardness_eps,
            batch_size=opts.eval_batch_size,
            device=device,
            baseline=baseline
        )
        
        # Combine base data with hard samples for this epoch's training
        base_part = Subset(base_train_dataset, list(range(opts.val_size)))
        combined_train_dataset = ConcatDataset([base_part, hard_dataset_part])
        print(f"Epoch {epoch+1}: Combined training dataset size = {len(combined_train_dataset)}")

        # Train one epoch
        train_epoch_adaptive_mtsp(model, optimizer, baseline, opts, combined_train_dataset, epoch)

        # Validate
        val_cost, val_gap = run_validation_mtsp(model, val_dataset, gt, opts)
        plot_data.append((epoch + 1, val_cost))
        
        if not np.isnan(val_cost) and val_cost < best_val_cost:
             print(f"New best validation cost: {val_cost:.4f}")
             best_val_cost = val_cost

    print("===== Fine-tuning Finished =====")

    # --- Save Final Model ---
    final_model_save_path = os.path.join(opts.save_dir, opts.finetuned_model_save_filename)
    print(f"Saving final fine-tuned model to: {final_model_save_path}")
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
    plt.title(f"Improved MTSP-{opts.graph_size} (k={opts.num_salesmen}) Fine-tuning Curve")
    plt.grid(True)
    plot_save_path = os.path.join(opts.save_dir, opts.plot_save_filename)
    plt.savefig(plot_save_path)
    print(f"Training curve saved to {plot_save_path}")

    print("===== Improved MTSP Training Script Finished =====")