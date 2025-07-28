import os
import torch
import numpy as np
import logging
import time

from src.options import get_options
from utils import load_problem, torch_load_cpu
from nets.attention_model import AttentionModel
from typing import List, Tuple

# Type alias for position
Position3D = Tuple[float, float, float]

# It's better to pass a logger than to use a global one
logger = logging.getLogger(__name__)

def load_attention_model(checkpoint_path, device, graph_size=None):
    """
    Loads a pre-trained AttentionModel for inference from a checkpoint file.
    This function is designed to be robust and handle various checkpoint formats.
    """
    logger.info(f"Loading model from: {checkpoint_path}")

    # This logic is to handle relative paths if the script is run from different locations.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)
        logger.info(f"Resolved relative model path to: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        logger.error(f"Model checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    checkpoint = torch_load_cpu(checkpoint_path)

    # Load opts from checkpoint or create default
    opts = checkpoint.get('opts')
    if opts is None:
        logger.warning("'opts' not found in checkpoint. Using default options.")
        opts = get_options([])  # Pass empty list for defaults
        opts.problem = 'tsp'
        if graph_size is not None:
            opts.graph_size = graph_size
        else:
            logger.warning("`graph_size` not provided and 'opts' are missing. Model might not initialize correctly if it depends on graph_size.")
            # Fallback for simple_protocol which might not know the graph size beforehand
            opts.graph_size = 50 # Defaulting to 50 as a fallback

    # Initialize model structure
    problem = load_problem(opts.problem)
    model = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=getattr(opts, 'checkpoint_encoder', False),
        shrink_size=getattr(opts, 'shrink_size', None)
    ).to(device)

    # Extract model state dict with robust checking from visualize_routes_pretrained.py
    model_state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    if 'model_state_dict' in model_state_dict: # Handles nested dicts
        model_state_dict = model_state_dict['model_state_dict']

    # Load the state dict, handling DataParallel wrapper
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError as e:
        logger.warning(f"Loading state_dict failed: {e}. Trying to unwrap DataParallel keys.")
        new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(new_state_dict)

    model.eval()
    logger.info("AttentionModel loaded successfully and set to evaluation mode.")
    return model, opts


def solve_tsp_with_attention(
    model: AttentionModel,
    locations_to_visit: List[Position3D],
    device: torch.device,
    area_bounds: Tuple[float, float] = None
) -> Tuple[List[Position3D], float, float]:
    """
    Solves the TSP for the given locations using the pre-trained Attention Model.
    
    Args:
        model: The pre-trained AttentionModel.
        locations_to_visit: A list of (x, y, z) tuples.
        device: The torch device to use.
        area_bounds: A tuple (max_x, max_y) for fixed normalization.
                     If None, normalization is based on the min/max of the input coordinates.

    Returns:
        A tuple containing:
        - The ordered list of locations forming the tour.
        - The tour cost.
        - The computation time in milliseconds.
    """
    if not locations_to_visit or len(locations_to_visit) <= 1:
        return [], 0.0, 0.0

    logger.info(f"Solving TSP with Attention Model for {len(locations_to_visit)} nodes.")
    
    coords_np = np.array([loc[:2] for loc in locations_to_visit])

    # --- Normalization ---
    if area_bounds:
        min_coords = np.array([0.0, 0.0])
        max_coords = np.array(area_bounds)
    else:
        min_coords = coords_np.min(axis=0)
        max_coords = coords_np.max(axis=0)
    
    scale = max_coords - min_coords
    scale[scale == 0] = 1
    
    normalized_coords_np = (coords_np - min_coords) / scale
    # --- End Normalization ---

    input_tensor = torch.tensor(normalized_coords_np, dtype=torch.float, device=device).unsqueeze(0)

    start_time = time.time()
    with torch.no_grad():
        model.set_decode_type("greedy")
        _, _, tour_indices = model(input_tensor, return_pi=True)
    end_time = time.time()
    
    computation_time_ms = (end_time - start_time) * 1000
    
    tour_indices_np = tour_indices.squeeze(0).cpu().numpy()
    ordered_locations = [locations_to_visit[i] for i in tour_indices_np]

    # --- Recalculate cost with original coordinates ---
    ordered_coords_np = np.array([loc[:2] for loc in ordered_locations])
    rolled_coords = np.roll(ordered_coords_np, -1, axis=0)
    distances = np.linalg.norm(ordered_coords_np - rolled_coords, axis=1)
    cost_val = np.sum(distances)
    # --- End Recalculate cost ---

    logger.info(f"Attention Model inference took: {computation_time_ms:.2f} ms. Recalculated Cost: {cost_val:.4f}")
    
    return ordered_locations, cost_val, computation_time_ms