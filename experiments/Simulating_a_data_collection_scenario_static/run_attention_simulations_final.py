import glob
import os
import sys
import csv
import re
from typing import Dict, List, Tuple, Type
import subprocess
import tempfile
import time

import torch
import numpy as np

# Gradysim imports
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from gradysim.protocol.interface import IProtocol, IProvider

# Project-specific imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..', '..'))

from showcases.Simulating_a_data_collection_scenario_static.simple_protocol import SimpleSensorProtocol, SimpleGroundStationProtocol, SimpleUAVProtocol
from nets.attention_model import AttentionModel
from problems.tsp.problem_tsp import TSP
from utils import load_problem, torch_load_cpu
from src.options import get_options

# Type alias for clarity
Position3D = Tuple[float, float, float]


# --- Utility functions ---

def build_protocol_with_args(protocol_class: Type[IProtocol], **kwargs) -> Type[IProtocol]:
    """Dynamically creates a new protocol class with patched __init__."""
    class PatchedProtocol(protocol_class):
        @classmethod
        def instantiate(cls, provider: IProvider) -> 'IProtocol':
            protocol = protocol_class(**kwargs)
            protocol.provider = provider
            return protocol
    return PatchedProtocol

def calculate_path_length(path: List[Position3D]) -> float:
    """Calculates the total length of a path."""
    total_length = 0.0
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5
        total_length += distance
    return total_length

def load_sensor_locations_from_tsp(tsp_file_path: str) -> Tuple[Dict[int, Position3D], int]:
    """Parses a TSP file to extract node coordinates and dimension."""
    coordinates_map: Dict[int, Position3D] = {}
    dimension = 0
    in_coord_section = False
    with open(tsp_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("DIMENSION"):
                try:
                    dimension = int(line.split(":")[1])
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse DIMENSION from line: {line}")
            elif line == "NODE_COORD_SECTION":
                in_coord_section = True
            elif in_coord_section and line != "EOF":
                try:
                    parts = line.split()
                    node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                    coordinates_map[node_id] = (x, y, 0)
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse coordinate line: {line}")
    return coordinates_map, dimension

def solve_tsp_with_attention_model(tsp_file_path: str, model_path: str, graph_size: int) -> Tuple[List[int], float, List[Position3D]]:
    """Solves a TSP instance using a pre-trained Attention Model."""
    opts = get_options(['--problem', 'tsp', '--graph_size', str(graph_size)])
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    problem = load_problem(opts.problem)

    # Load model
    model = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    model_data = torch_load_cpu(model_path)
    model.load_state_dict({**model.state_dict(), **model_data.get('model', {})})
    model.set_decode_type("greedy")

    # Load data and solve
    dataset = problem.make_dataset(filename=tsp_file_path, num_samples=1, offset=0)
    data_tensor = dataset[0].unsqueeze(0).to(opts.device)
    results = model(data_tensor, return_pi=True)
    tour = results[1][0].tolist()
    cost = results[0].item()

    # Map tour indices to coordinates
    coords, _ = load_sensor_locations_from_tsp(tsp_file_path)
    ordered_route = [coords[node_id] for node_id in tour]
    
    return tour, cost, ordered_route


# --- Main execution block ---

if __name__ == "__main__":
    # Configuration
    scenarios = ["agriculture", "custom_hybrid", "factory", "forest", "urban"]
    graph_sizes = [50, 100, 150, 200]
    num_instances = 10
    output_csv = "attention_final_results.csv"

    # Prepare CSV output
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Scenario", "GraphSize", "Instance", "Model", "PathLength", "MissionDuration", "EnergyConsumed", "DataCollectionRate"])

    # Main loop
    for scenario in scenarios:
        for graph_size in graph_sizes:
            for instance_num in range(1, num_instances + 1):
                # File paths
                tsp_file = f"showcases/Simulating_a_data_collection_scenario_static/data/sensor_locations_{scenario}_{graph_size}_{instance_num}.tsp"
                model_file = f"showcases/new_main/tsp{graph_size}_attention_custom_final.pt"

                if not os.path.exists(tsp_file) or not os.path.exists(model_file):
                    print(f"Skipping {tsp_file} or {model_file} as it does not exist.")
                    continue

                print(f"Processing: {scenario}, Size: {graph_size}, Instance: {instance_num}")

                # Solve TSP to get the initial route
                _, _, initial_route = solve_tsp_with_attention_model(tsp_file, model_file, graph_size)
                
                # Simulation setup
                builder = SimulationBuilder(SimulationConfiguration(duration=3600, debug=False))
                
                # Handlers
                builder.add_handler(CommunicationHandler(CommunicationMedium(transmission_range=50)))
                builder.add_handler(TimerHandler())
                builder.add_handler(MobilityHandler())

                # Nodes
                sensor_locations, _ = load_sensor_locations_from_tsp(tsp_file)
                for sensor_id, pos in sensor_locations.items():
                    builder.add_node(build_protocol_with_args(SimpleSensorProtocol, sensor_id=sensor_id), (pos[0], pos[1], 0))

                builder.add_node(SimpleGroundStationProtocol, (0, 0, 0))
                builder.add_node(build_protocol_with_args(SimpleUAVProtocol,
                                                          uav_id=len(sensor_locations) + 2,
                                                          route=initial_route,
                                                          speed=10),
                                 (initial_route[0][0], initial_route[0][1], 10))

                # Run simulation
                simulation = builder.build()
                simulation.run()

                # Collect results
                results = simulation.get_results()
                uav_protocol = next(p for p in results['protocols'] if isinstance(p, SimpleUAVProtocol))
                gs_protocol = next(p for p in results['protocols'] if isinstance(p, SimpleGroundStationProtocol))

                path_length = calculate_path_length(uav_protocol.trajectory)
                mission_duration = uav_protocol.mission_end_time - uav_protocol.mission_start_time if uav_protocol.mission_end_time > 0 else 0
                energy_consumed = uav_protocol.energy_consumed
                data_collection_rate = len(gs_protocol.all_collected_packets) / len(sensor_locations) if len(sensor_locations) > 0 else 0

                # Write to CSV
                with open(output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([scenario, graph_size, instance_num, "Attention", path_length, mission_duration, energy_consumed, data_collection_rate])

    print(f"Batch simulation finished. Results saved to {output_csv}")