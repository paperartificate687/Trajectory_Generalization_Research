import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
import torch
from problems.tsp.tsp_baseline import run_insertion,nearest_neighbour
from functools import partial
from tqdm import tqdm
from utils import load_model
from problems.tsp.problem_tsp import TSP
import os
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from problems.tsp.problem_tsp import TSP
from torch.utils.data import DataLoader
import pandas as pd
from utils import load_model
from tqdm import tqdm
from utils.data_utils import save_dataset
from src.options import get_options
from torch import optim
from utils import torch_load_cpu, load_problem
from src.reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
import random
class Solver:
    @staticmethod
    def model(model,data):
        if len(data.shape)>2:
            return make_tour_batch(model,data)
        else:
            return make_tour(model,data)
    @staticmethod
    def gurobi(data):
        if len(data.shape)>2:
            return [solve_euclidian_tsp(x) for x in data]
        else:
            return solve_euclidian_tsp(data)

def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]

def make_tour_batch(model, batch):
    '''
    :params batch: Tensor 
    :return: list of [cost ,tour]
    '''
    model.eval()
    model.set_decode_type("greedy" )
    results=[]
    with torch.no_grad():
        batch_rep = 1
        iter_rep = 1
        sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
        batch_size = len(costs)
        ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
    if sequences is None:
        sequences = [None] * batch_size
        costs = [math.inf] * batch_size
    else:
        sequences, costs = get_best(
            sequences.cpu().numpy(), costs.cpu().numpy(),
            ids.cpu().numpy() if ids is not None else None,
            batch_size
        )
    for seq, cost in zip(sequences, costs):
        seq = seq.tolist()
        results.append([cost, seq])
    return results

def make_oracle(model, xy, temperature=1.0):
    model.eval()
    model.set_decode_type("greedy" )
    num_nodes = len(xy)
    
    xyt = torch.tensor(xy).float()[None]  # Add batch dimension
    
    with torch.no_grad():  # Inference only
        embeddings, _ = model.embedder(model._init_embed(xyt))

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = model._precompute(embeddings)
    
    def oracle(tour):
        with torch.no_grad():  # Inference only
            # Input tour with 0 based indices
            # Output vector with probabilities for locations not in tour
            tour = torch.tensor(tour).long()
            if len(tour) == 0:
                step_context = model.W_placeholder
            else:
                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

            # Compute query = context node embedding, add batch and step dimensions (both 1)
            query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

            # Create the mask and convert to bool depending on PyTorch version
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension

            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            p = torch.softmax(log_p / temperature, -1)[0, 0]
            assert (p[tour] == 0).all()
            assert (p.sum() - 1).abs() < 1e-5
            #assert np.allclose(p.sum().item(), 1)
        return p.numpy()
    
    return oracle
        

def make_tour(model,xy):
    '''
    xy : numpy 
    return : tour
    '''
    oracle = make_oracle(model, xy)
    sample = False
    tour = []
    tour_p = []
    while(len(tour) < len(xy)):
        p = oracle(tour)

        if sample:
            # Advertising the Gumbel-Max trick
            g = -np.log(-np.log(np.random.rand(*p.shape)))
            i = np.argmax(np.log(p) + g)
            # i = np.random.multinomial(1, p)
        else:
            # Greedy
            i = np.argmax(p)
        tour.append(i)
        tour_p.append(p)

    return tour

from gurobipy import *
def solve_euclidian_tsp(points, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: cost, tour
    """

    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
        vars[j,i] = vars[i,j] # edge in opposite direction

    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))


    # Add degree-2 constraint

    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)


    # Optimize model

    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected)
    assert len(tour) == n

    return m.objVal, tour

def get_costs_batch(model,dataloader,device=torch.device('cuda:1')):
    '''
    get costs of dataloader 
    '''
    preds=[]
    model=model.to(device)
    with tqdm(dataloader) as bar:
        for data in bar:
            data=data.to(device)
            t=Solver.model(model,data)
            preds.extend([x[0] for x in t])
        preds=np.array(preds)
    return preds
    
def minmax(xy_):
    '''
    min max batch of graphs [b,n,2]
    '''
    xy_=(xy_-xy_.min(dim=1,keepdims=True)[0])/(xy_.max(dim=1,keepdims=True)[0]-xy_.min(dim=1,keepdims=True)[0])
    return xy_

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def mg(cdist=1000, graph_size=50):
    '''
    GMM create one instance of TSP, using cdist
    '''
    nc=np.random.randint(3,7)
    nums=np.random.multinomial(graph_size,np.ones(nc)/nc)
    xy=[]
    for num in nums:
        center=np.random.uniform(0,cdist,size=(1,2))
        nxy=np.random.multivariate_normal(mean=center.squeeze(),cov=np.eye(2,2),size=(num,))
        xy.extend(nxy)
    
    xy=np.array(xy)
    xy=MinMaxScaler().fit_transform(xy)
    return xy

def mg_batch(cdist,size):
    '''
    GMM create a batch size instance of TSP-50, using cdist
    '''
    xy=[]
    for i in range(size):
        xy.append(mg(cdist))
    return np.array(xy)

def generate_tsp_data_mg(dataset_size, graph_size):
    '''
    generate a TSP instance with MG
    dataset_size is the num of instances
    graph_size is the num of cities
    '''
    datas = []
    for _ in tqdm(range(dataset_size)):
        # Call mg directly with graph_size
        loc = mg(1000, graph_size)
        datas.append(loc)
    return datas

# --- Unified Distribution Model Data Generation ---

import math
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler # Ensure MinMaxScaler is imported if not already

def generate_sparse_random(num_instances, graph_size, scale=10.0):
    """Generates sparse random TSP instances by sampling in a larger area and scaling."""
    data = []
    for _ in tqdm(range(num_instances), desc="Generating Sparse Random"):
        # Sample uniformly in a larger square
        xy = np.random.uniform(0, scale, size=(graph_size, 2))
        # Scale to [0, 1]
        xy = MinMaxScaler().fit_transform(xy)
        data.append(xy)
    return data

def generate_dense_clustered(num_instances, graph_size, num_clusters_range=(3, 7), cluster_std_range=(0.05, 0.15), uniform_ratio=0.2):
    """Generates dense clustered TSP instances (mixture of uniform and Gaussians)."""
    data = []
    for _ in tqdm(range(num_instances), desc="Generating Dense Clustered"):
        num_uniform = int(graph_size * uniform_ratio)
        num_clustered = graph_size - num_uniform
        
        # Generate uniform points
        uniform_points = np.random.uniform(0, 1, size=(num_uniform, 2)) if num_uniform > 0 else np.array([]).reshape(0,2)

        # Generate clustered points
        clustered_points_list = []
        if num_clustered > 0:
            num_clusters = np.random.randint(num_clusters_range[0], num_clusters_range[1] + 1)
            points_per_cluster = np.random.multinomial(num_clustered, np.ones(num_clusters) / num_clusters)
            cluster_centers = np.random.uniform(0.1, 0.9, size=(num_clusters, 2)) # Keep centers away from borders

            for i in range(num_clusters):
                num_points_in_cluster = points_per_cluster[i]
                if num_points_in_cluster > 0:
                    center = cluster_centers[i]
                    cluster_std = np.random.uniform(cluster_std_range[0], cluster_std_range[1])
                    cov = np.eye(2) * cluster_std**2
                    
                    # Generate points and clip to [0,1] to avoid points outside the boundary
                    points = np.random.multivariate_normal(mean=center, cov=cov, size=num_points_in_cluster)
                    points = np.clip(points, 0, 1)
                    clustered_points_list.append(points)
            
        if clustered_points_list:
            clustered_points = np.vstack(clustered_points_list)
            if uniform_points.size > 0:
                 xy = np.vstack([uniform_points, clustered_points])
            else:
                 xy = clustered_points
        elif uniform_points.size > 0 :
            xy = uniform_points
        else: # Should not happen if graph_size > 0
            xy = np.random.uniform(0,1, size=(graph_size,2))


        # Ensure correct number of points if rounding caused issues
        if xy.shape[0] != graph_size:
            # Fallback or adjustment logic if point count is off
            # For simplicity, if too few, add random; if too many, truncate.
            if xy.shape[0] < graph_size:
                needed = graph_size - xy.shape[0]
                extra_points = np.random.uniform(0,1, size=(needed,2))
                xy = np.vstack([xy, extra_points]) if xy.size > 0 else extra_points
            else:
                xy = xy[:graph_size, :]
                
        xy = MinMaxScaler().fit_transform(xy) # Final scaling
        data.append(xy)
    return data

def generate_grid_based(num_instances, graph_size, jitter_strength=0.05):
    """Generates grid-based TSP instances with optional jitter."""
    data = []
    
    for _ in tqdm(range(num_instances), desc="Generating Grid-Based"):
        # Determine grid dimensions (approximate square root)
        # Ensure at least graph_size points can be placed on the grid
        grid_dim = int(np.ceil(np.sqrt(graph_size)))
        
        # Create base grid points
        x_coords = np.linspace(0.1, 0.9, grid_dim) # Keep grid away from borders
        y_coords = np.linspace(0.1, 0.9, grid_dim)
        xv, yv = np.meshgrid(x_coords, y_coords)
        base_grid_points = np.vstack([xv.ravel(), yv.ravel()]).T
        
        # Select graph_size points from the grid
        if base_grid_points.shape[0] >= graph_size:
            indices = np.random.choice(base_grid_points.shape[0], graph_size, replace=False)
            selected_points = base_grid_points[indices]
        else:
            # If grid is smaller, use all grid points and add more to reach graph_size
            selected_points = base_grid_points
            needed = graph_size - selected_points.shape[0]
            # Add points by duplicating existing grid points with jitter, or add random points
            # For simplicity, let's add random points if needed (could be improved)
            if needed > 0:
                additional_points = np.random.uniform(0, 1, size=(needed, 2))
                selected_points = np.vstack([selected_points, additional_points])

        # Apply jitter
        jitter = np.random.normal(0, jitter_strength, selected_points.shape)
        xy = selected_points + jitter
        xy = np.clip(xy, 0, 1) # Clip to [0,1] after jitter

        xy = MinMaxScaler().fit_transform(xy) # Final scaling
        data.append(xy)
    return data

def generate_structured_linear(num_instances, graph_size, num_lines_range=(1,3),
                               points_per_cluster_avg=8,
                               line_std_major_range=(0.15, 0.3),
                               line_std_minor_range=(0.01, 0.05)):
    """Generates structured linear TSP instances."""
    data = []
    for _ in tqdm(range(num_instances), desc="Generating Structured Linear"):
        xy_instance = []
        points_remaining = graph_size
        
        num_lines = np.random.randint(num_lines_range[0], num_lines_range[1] + 1)
        
        for line_idx in range(num_lines):
            if points_remaining <= 0:
                break

            # Define line start and end points (randomly within [0.1, 0.9] to avoid edges)
            start_point = np.random.uniform(0.1, 0.9, size=2)
            end_point = np.random.uniform(0.1, 0.9, size=2)
            
            # Determine number of clusters along this line
            # Distribute remaining points among remaining lines
            avg_points_this_line = points_remaining // (num_lines - line_idx) if (num_lines - line_idx) > 0 else points_remaining
            num_line_clusters = max(1, avg_points_this_line // points_per_cluster_avg if points_per_cluster_avg > 0 else 1)
            
            # Generate cluster centers along the line
            line_coords_t = np.sort(np.random.uniform(0, 1, num_line_clusters)) # Random t values for centers
            cluster_centers = np.array([start_point + t * (end_point - start_point) for t in line_coords_t])
            
            # Distribute points for this line among its clusters
            points_for_this_line = min(points_remaining, avg_points_this_line if line_idx < num_lines -1 else points_remaining)
            
            if points_for_this_line <=0: continue

            points_per_cluster_dist = np.random.multinomial(points_for_this_line, np.ones(num_line_clusters) / num_line_clusters)

            for i in range(num_line_clusters):
                num_points = points_per_cluster_dist[i]
                if num_points > 0:
                    center = cluster_centers[i]
                    line_std_major = np.random.uniform(line_std_major_range[0], line_std_major_range[1])
                    line_std_minor = np.random.uniform(line_std_minor_range[0], line_std_minor_range[1])
                    
                    line_vector = end_point - start_point
                    if np.linalg.norm(line_vector) > 1e-6:
                        line_direction = line_vector / np.linalg.norm(line_vector)
                        perp_direction = np.array([-line_direction[1], line_direction[0]])
                        cov_matrix = (line_std_major**2) * np.outer(line_direction, line_direction) + \
                                     (line_std_minor**2) * np.outer(perp_direction, perp_direction)
                    else:
                        cov_matrix = np.eye(2) * ((line_std_major + line_std_minor)/2)**2
                        
                    cluster_points = np.random.multivariate_normal(mean=center, cov=cov_matrix, size=num_points)
                    cluster_points = np.clip(cluster_points, 0, 1) # Clip points
                    xy_instance.append(cluster_points)
            points_remaining -= points_for_this_line

        if not xy_instance: # If no points generated (e.g. graph_size was 0)
             xy = np.random.uniform(0,1,size=(graph_size,2)) if graph_size > 0 else np.array([]).reshape(0,2)
        else:
             xy = np.vstack(xy_instance)

        # Ensure correct number of points
        if graph_size > 0 and xy.shape[0] != graph_size :
            if xy.shape[0] < graph_size:
                needed = graph_size - xy.shape[0]
                extra_points = np.random.uniform(0,1, size=(needed,2))
                xy = np.vstack([xy, extra_points]) if xy.size > 0 else extra_points
            else:
                xy = xy[:graph_size, :]
        elif graph_size == 0:
            xy = np.array([]).reshape(0,2)

        if graph_size > 0:
            xy = MinMaxScaler().fit_transform(xy) # Final scaling
        data.append(xy)
    return data


def generate_tsp_data_unified(dataset_size, graph_size):
    '''
    generate a TSP instance with a unified distribution model
    dataset_size is the num of instances
    graph_size is the num of cities
    The dataset will be a mix of four scenarios, each occupying 1/4.
    '''
    print(f"Generating {dataset_size} TSP instances with unified distribution (graph size {graph_size})...")
    
    num_scenarios = 4
    base_size_per_scenario = dataset_size // num_scenarios
    remainder = dataset_size % num_scenarios

    sizes = [base_size_per_scenario] * num_scenarios
    for i in range(remainder):
        sizes[i] += 1

    all_data = []
    
    # Generate data for each scenario
    if sizes[0] > 0:
        print(f"Generating {sizes[0]} Sparse Random instances...")
        all_data.extend(generate_sparse_random(sizes[0], graph_size))
    
    if sizes[1] > 0:
        print(f"Generating {sizes[1]} Dense Clustered instances...")
        all_data.extend(generate_dense_clustered(sizes[1], graph_size))
    
    if sizes[2] > 0:
        print(f"Generating {sizes[2]} Grid-Based instances...")
        all_data.extend(generate_grid_based(sizes[2], graph_size))

    if sizes[3] > 0:
        print(f"Generating {sizes[3]} Structured Linear instances...")
        all_data.extend(generate_structured_linear(sizes[3], graph_size))
    
    # Shuffle the combined data
    np.random.shuffle(all_data)
    
    print(f"Unified dataset generation complete. Total instances: {len(all_data)}")
    return all_data

# --- End of Unified Distribution Model Data Generation ---

from torch.utils.data import Dataset
class ListDictDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ConcatDataset(Dataset):
    '''
    concat a list of datasets
    '''
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length
    
from torch.utils.data import DataLoader
def get_hard_samples(model, data, eps=5, batch_size=1024, device='cpu', baseline=None):
    model.eval()
    set_decode_type(model, "greedy")

    def get_hard(model, data_batch, eps):
        data_batch = data_batch.to(device)
        data_batch.requires_grad_()

        if getattr(model, 'is_mtsp', False):
            formatted_data = {'depot': data_batch[:, 0:1, :], 'loc': data_batch[:, 1:, :]}
        else:
            formatted_data = data_batch

        cost, ll, pi = model(formatted_data, return_pi=True)

        if baseline is not None:
            with torch.no_grad():
                cost_b, _ = baseline.model(formatted_data)
            loss = (cost / cost_b) * ll
            delta = torch.autograd.grad(eps * loss.mean(), data_batch)[0]
        else:
            loss = cost * ll
            delta = torch.autograd.grad(eps * loss.mean(), data_batch)[0]
            
        ndata = data_batch + delta
        ndata = minmax(ndata)
        return ndata.detach().cpu()

    dataloader = DataLoader(data, batch_size=batch_size)
    hard_tensors = torch.cat([get_hard(model, data_batch, eps) for data_batch in dataloader], dim=0)
    
    # Convert the tensor of hard samples back to a list of dictionaries
    # to match the MTSPDataset format.
    hard_dataset_list = [
        {
            'depot': instance[0].unsqueeze(0),
            'loc': instance[1:]
        }
        for instance in hard_tensors
    ]
    return ListDictDataset(hard_dataset_list)

def get_gap(model,data,device):
    '''
    get gap for model on a batch of data
    '''
    data=data.cpu().numpy()
    hard_gt=[]
    for x in tqdm(data):
        cost=Solver.gurobi(x)[0]
        hard_gt.append(cost)
    hard_gt=np.array(hard_gt)
    costs=Solver.model(model,torch.FloatTensor(data).to(device))
    costs=np.array([c[0] for c in costs])
    ratio=(costs-hard_gt)/hard_gt
    info=[ratio.mean(),costs.mean(),hard_gt.mean()]
    return ratio



from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
def init(pretrain=True, device=torch.device('cpu'), opts=None):
    '''
    init a TSP-50 model. using uniform 10k data for baseline
    Modified to support loading checkpoint for resuming training.
    '''
    problem = load_problem(opts.problem)

    # Initialize model first
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        num_salesmen=getattr(opts, 'num_salesmen', None)  # Pass num_salesmen if available
    ).to(opts.device)

    # dataset
    # Determine project root dynamically, assuming this script is in 'src'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, f'data/tsp/tsp{opts.graph_size}_train_seed1111_size10K.pkl')
    print(f"Loading training dataset from: {dataset_path}")
    try:
        dataset = problem.make_dataset(filename=dataset_path)
        print(f"Loaded training dataset with size: {len(dataset)}")
    except FileNotFoundError as e:
        print(f"Error: Training data file not found at {dataset_path}. Please generate it first.")
        print(f"Example command: python src/generate_data.py --problem tsp --graph_sizes {opts.graph_size} --name train --seed 1111 --dataset_size 10000 --generate_type uniform")
        return None, None, None, None, None, None # Return None for all outputs if dataset loading fails
    except Exception as e:
        print(f"Error loading training data from {dataset_path}: {e}")
        return None, None, None, None, None, None # Return None for all outputs if dataset loading fails

    dataloader = DataLoader(dataset, batch_size=1024)

    # Initialize baseline
    baseline = RolloutBaseline(model, problem, opts, dataset=dataset)

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    start_epoch = 0  # Initialize start epoch

    # Load checkpoint if load_path is specified
    if opts.load_path is not None:
        print(f"Attempting to load checkpoint from {opts.load_path}")
        try:
            # Temporarily allowlist AttentionModel for loading
            # Temporarily allowlist AttentionModel and Linear for loading
            # Temporarily allowlist AttentionModel and Linear for loading
            torch.serialization.add_safe_globals([AttentionModel, torch.nn.modules.linear.Linear])
            checkpoint = torch.load(opts.load_path, map_location=device, weights_only=False)
            # Remove from allowlist after loading (optional, but good practice)
            # Note: add_safe_globals modifies the global list, so removing might be tricky.
            # Using safe_globals context manager is better, but requires restructuring.
            # For simplicity here, we'll just add it.

            # Print keys in the loaded checkpoint for debugging
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {checkpoint.keys()}")
            else:
                print("Loaded checkpoint is not a dictionary.")

            # Print keys in the loaded checkpoint for debugging
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {checkpoint.keys()}")
            else:
                print("Loaded checkpoint is not a dictionary.")

            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded.")

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded.")
            else:
                print("Warning: Checkpoint does not contain optimizer state. Optimizer initialized from scratch.")

            # Load baseline state if available and baseline supports it
            if 'baseline_state_dict' in checkpoint and hasattr(baseline, 'load_state_dict'):
                 try:
                     baseline.load_state_dict(checkpoint['baseline_state_dict'])
                     print("Baseline state loaded.")
                 except RuntimeError as e:
                     print(f"Warning: Could not load baseline state dictionary: {e}")
            elif 'baseline_state_dict' in checkpoint and not hasattr(baseline, 'load_state_dict'):
                 print("Warning: Checkpoint contains baseline state, but baseline does not have load_state_dict method.")
            else:
                 print("No baseline state found in checkpoint or baseline does not support loading state.")


            # Load epoch number
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"Resuming training from epoch {start_epoch + 1}") # Print next epoch number
            else:
                print("Warning: Checkpoint does not contain epoch number. Starting training from epoch 0.")

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {opts.load_path}. Starting training from scratch.")
            # If checkpoint not found, proceed with newly initialized model, optimizer, baseline
        except KeyError as e:
            print(f"Error loading checkpoint: Missing key {e}. Starting training from scratch.")
            # If key error, proceed with newly initialized model, optimizer, baseline
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            # Catch other potential errors during loading

    # Set decode type and eval mode after loading state
    model.set_decode_type("greedy")
    model.eval()  # Put in evaluation mode to not track gradients

    return model, dataset, dataloader, baseline, optimizer, start_epoch # Add start_epoch to return

from copy import deepcopy
from nets.attention_model import set_decode_type
from utils import move_to
from src.train import *


def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def greedy_batch_seq(model, batch):
    model.eval()
    model.set_decode_type("greedy" )
    sequences, costs = model.sample_many(batch, batch_rep=1, iter_rep=1)
    return sequences

def plot_one(ax,model,data):
    '''
    plot one solution of data(shape [1,1]) on ax
    '''
    xy=data[0].detach().cpu().numpy()
    tour=greedy_batch_seq(model,data).cpu().numpy()[0]
    gtc,tour2=solve_euclidian_tsp(xy)
    plot_tsp(xy, tour, ax,tour2,gtc)

def plot_tsp(xy, tour, ax1,tour2=None,cost2=None):
    """
    Plot the TSP tour on matplotlib axis ax1.
    """
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    xs, ys = xy[tour].transpose()
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()
#     print(d)
    # Scatter nodes
    ax1.scatter(xs, ys, s=40, color='blue')
    # Starting node
    ax1.scatter([xs[0]], [ys[0]], s=40, color='blue')
    
    
    # Arcs
    qv = ax1.quiver(
        xs, ys, dx, dy,
        scale_units='xy',
        angles='xy',
        scale=1,
        alpha=0.5
    )
    
    if tour2 is not None:
        xs2,ys2,dx2,dy2,c=getdelta(xy,tour2)
        qv = ax1.quiver(
            xs2, ys2, dx2, dy2,
            scale_units='xy',
            angles='xy',
            scale=1,
            color='green',
            alpha=0.5
        )

        ax1.set_title('{} nodes, optimal cost {:.1f}, gap {:.1f}%'.format(len(tour), cost2,100*(lengths[-1]-cost2)/cost2))
    else:
        ax1.set_title('{} nodes, total length {:.2f}'.format(len(tour), lengths[-1]))

def getdelta(xy,tour):
    xs, ys = xy[tour].transpose()
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = sum(d)
    return xs,ys,dx,dy,lengths