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

# Project-specific imports (assuming structure from visualize_routes.py)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..', '..')) # Adjust path to project root

from simple_protocol import SimpleSensorProtocol, SimpleGroundStationProtocol, SimpleUAVProtocol
from nets.attention_model import AttentionModel
from problems.tsp.problem_tsp import TSP
from utils import load_problem, torch_load_cpu
from src.options import get_options

# Type alias for clarity
Position3D = Tuple[float, float, float]


# --- Utility functions from main.py and visualize_routes.py ---

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
                    dimension = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse DIMENSION: {line}")
            elif line == "NODE_COORD_SECTION":
                in_coord_section = True
                continue
            elif line == "EOF":
                break
            
            if in_coord_section and line:
                parts = re.split(r'\s+', line)
                if len(parts) >= 3:
                    try:
                        node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                        coordinates_map[node_id] = (x, y, 0.0)
                    except ValueError:
                        print(f"Warning: Could not parse coordinate line: {line}")
    return coordinates_map, dimension

def load_model_for_inference(checkpoint_path, device, graph_size=None):
    """Loads a trained model from a checkpoint file for inference."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch_load_cpu(checkpoint_path)

    opts = None
    if isinstance(checkpoint, dict) and 'opts' in checkpoint:
        opts = checkpoint['opts']
    else:
        opts = get_options('')
        opts.problem = 'tsp'
        if graph_size is not None:
            opts.graph_size = graph_size
        else:
            raise ValueError("graph_size must be provided if 'opts' not in checkpoint.")

    # Correctly extract the model's state dictionary
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    else:
        model_state_dict = checkpoint
    
    model = AttentionModel(
        opts.embedding_dim, opts.hidden_dim, load_problem(opts.problem),
        n_encode_layers=opts.n_encode_layers, mask_inner=True, mask_logits=True,
        normalization=opts.normalization, tanh_clipping=opts.tanh_clipping
    ).to(device)

    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError:
        new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(new_state_dict)
    
    model.eval()
    return model

def generate_tour_from_model(model: AttentionModel, sensor_coords: Dict[int, Position3D], device: torch.device) -> List[int]:
    """Generates a TSP tour for the given sensor coordinates using the loaded model."""
    print("Generating tour from model...")
    
    # The model expects a tensor of coordinates. We need to maintain the mapping from tensor index to node ID.
    node_ids = list(sensor_coords.keys())
    coords_list = [sensor_coords[node_id][:2] for node_id in node_ids] # Use (x, y)

    # Normalize coordinates to [0, 1] range
    if coords_list:
        coords_np = np.array(coords_list)
        min_x, min_y = coords_np[:, 0].min(), coords_np[:, 1].min()
        max_x, max_y = coords_np[:, 0].max(), coords_np[:, 1].max()

        range_x = max_x - min_x
        range_y = max_y - min_y

        # Avoid division by zero if all points have the same x or y coordinate
        normalized_coords_list = []
        for x, y in coords_list:
            norm_x = (x - min_x) / range_x if range_x > 0 else 0.5
            norm_y = (y - min_y) / range_y if range_y > 0 else 0.5
            normalized_coords_list.append((norm_x, norm_y))
        coords_list_for_tensor = normalized_coords_list
    else:
        coords_list_for_tensor = coords_list
    
    # Convert to tensor, add batch dimension, and move to device
    coords_tensor = torch.tensor(coords_list_for_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    model.set_decode_type("greedy")
    with torch.no_grad():
        _, _, tour_indices = model(coords_tensor, return_pi=True)
    
    # Squeeze batch dimension and convert to numpy array
    tour_indices_np = tour_indices.squeeze(0).cpu().numpy()
    
    # Map tour indices back to original node IDs
    tour_node_ids = [node_ids[i] for i in tour_indices_np]
    
    print(f"Generated tour with {len(tour_node_ids)} nodes.")
    return tour_node_ids


# --- LKH Solver Integration ---

def solve_tsp_with_lkh(
    ground_station_pos: Position3D,
    sensor_coords: Dict[int, Position3D]
) -> Tuple[List[int], float]:
    """
    Solves the TSP using the LKH-3 solver.

    Args:
        ground_station_pos: The (x, y, z) coordinates of the ground station (depot).
        sensor_coords: A dictionary mapping sensor node IDs to their (x, y, z) coordinates.

    Returns:
        A tuple containing:
        - A list of node IDs representing the optimized tour (excluding the depot).
        - The execution time of the LKH solver.
    """
    # The depot is node 1 in the TSP problem. We map our internal node IDs to TSP node IDs.
    depot_node_id = 1
    
    # Create a list of all points to visit, with the depot first.
    # Also create a mapping from TSP node ID (1-based index) back to original sensor ID.
    tsp_node_map = {depot_node_id: 0} # Map TSP node 1 to a placeholder ID 0 for the depot
    points_for_tsp = [ground_station_pos]
    
    current_tsp_node_id = 2
    for original_node_id, pos in sensor_coords.items():
        points_for_tsp.append(pos)
        tsp_node_map[current_tsp_node_id] = original_node_id
        current_tsp_node_id += 1

    if len(points_for_tsp) <= 1:
        return [], 0.0

    # --- LKH Execution ---
    lkh_executable_path = os.path.abspath("showcases/Simulating_a_data_collection_scenario_static/LKH-3.exe")
    if not os.path.exists(lkh_executable_path):
        print(f"Error: LKH executable not found at {lkh_executable_path}")
        return [], 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        problem_file = os.path.join(tmpdir, "problem.tsp")
        par_file = os.path.join(tmpdir, "problem.par")
        tour_file = os.path.join(tmpdir, "solution.tour")

        # --- Write TSP file ---
        with open(problem_file, 'w') as f:
            f.write(f"NAME: LKH_TSP_Instance\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {len(points_for_tsp)}\n")
            f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
            for i, (x, y, _) in enumerate(points_for_tsp):
                f.write(f"{i + 1} {x} {y}\n")
            f.write(f"DEPOT_SECTION\n{depot_node_id}\n-1\n")
            f.write("EOF\n")

        # --- Write PAR file ---
        par_content = f"""PROBLEM_FILE = {problem_file}
TOUR_FILE = {tour_file}
MOVE_TYPE = 5
PATCHING_C = 3
PATCHING_A = 2
RUNS = 1
"""
        with open(par_file, 'w') as f:
            f.write(par_content)

        # --- Run LKH ---
        try:
            start_time = time.time()
            subprocess.run(
                [lkh_executable_path, par_file],
                check=True,
                cwd=tmpdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            end_time = time.time()
            
            # --- Parse Tour File ---
            with open(tour_file, 'r') as f:
                in_tour_section = False
                tour_tsp_indices = []
                for line in f:
                    line = line.strip()
                    if line == "TOUR_SECTION":
                        in_tour_section = True
                        continue
                    if line == "-1" or line == "EOF":
                        break
                    if in_tour_section:
                        tsp_node_id = int(line)
                        # The tour starts and ends at the depot. We only need the sensor nodes.
                        if tsp_node_id != depot_node_id:
                            tour_tsp_indices.append(tsp_node_id)
            
            # Map TSP tour indices back to original sensor node IDs
            tour_original_node_ids = [tsp_node_map[i] for i in tour_tsp_indices]
            
            print(f"LKH generated tour with {len(tour_original_node_ids)} nodes.")
            return tour_original_node_ids, end_time - start_time

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error running LKH solver: {e}")
            if isinstance(e, subprocess.CalledProcessError):
                print(f"LKH stderr: {e.stderr.decode()}")
            return [], 0.0

def generate_tour_from_lkh(
    ground_station_pos: Position3D,
    sensor_coords: Dict[int, Position3D]
) -> List[int]:
    """
    Generates a TSP tour using the LKH solver.
    This function acts as a wrapper around solve_tsp_with_lkh to match the
    signature expected by the main simulation logic.
    """
    print("Generating tour from LKH...")
    tour_node_ids, exec_time = solve_tsp_with_lkh(ground_station_pos, sensor_coords)
    print(f"LKH solver finished in {exec_time:.4f} seconds.")
    return tour_node_ids


# --- Main simulation logic ---

def run_simulation(run_id: int, model_name: str, algorithm_alias: str, data_path: str, repetition: int):
    """
    Main function to configure and run a single simulation instance.
    """
    # --- Configuration ---
    file_basename = os.path.basename(data_path)
    # Expected format: sensor_locations_{SCENARIO}_{NUM_DEVICES}_{INSTANCE_ID}.tsp
    # Remove prefix and suffix
    core_name = file_basename.replace("sensor_locations_", "").replace(".tsp", "")
    core_parts = core_name.split('_')

    instance_id_str = "N/A"
    # num_devices_str_from_filename = "N/A" # This will be used later for the main loop's data_size
    scenario_name_parts = []

    if len(core_parts) >= 3: # Minimum: SCENARIO_DEVICES_INSTANCE
        instance_id_str = core_parts[-1]
        # num_devices_str_from_filename = core_parts[-2] # For run_simulation, num_devices is determined by loaded data
        scenario_name_parts = core_parts[:-2]
        scenario_name = "_".join(scenario_name_parts)
    elif len(core_parts) == 2: # SCENARIO_INSTANCE (assuming num_devices might be implicit or part of scenario)
        instance_id_str = core_parts[-1]
        scenario_name = core_parts[0]
    elif len(core_parts) == 1: # Only scenario name, or malformed
        scenario_name = core_parts[0]
    else: # Should not happen with expected filenames
        scenario_name = "unknown_scenario"
    
    uav_flight_altitude = 20.0
    # --- Load Data ---
    raw_sensor_coords_map, num_devices_from_tsp = load_sensor_locations_from_tsp(data_path)
    if not raw_sensor_coords_map:
        print(f"Error: No sensor positions loaded from {data_path}. Skipping.")
        return

    # Determine Ground Station position from the first node in TSP (e.g., node 1)
    ground_station_node_id = None
    if 1 in raw_sensor_coords_map: # Prefer node ID 1 if it exists
        ground_station_node_id = 1
    elif raw_sensor_coords_map: # Otherwise, take the first node from the dict
        ground_station_node_id = next(iter(raw_sensor_coords_map))
    
    if ground_station_node_id is not None:
        ground_station_actual_pos = raw_sensor_coords_map[ground_station_node_id]
        print(f"Ground Station position set to node {ground_station_node_id}: {ground_station_actual_pos}")
    else:
        print("Error: Could not determine ground station from TSP data. Defaulting to (0,0,0).")
        ground_station_actual_pos: Position3D = (0.0, 0.0, 0.0)

    # Filter out the ground station from the sensor coordinates map for path planning and sensor deployment
    sensor_coords_map_for_planning = {
        node_id: pos for node_id, pos in raw_sensor_coords_map.items() if node_id != ground_station_node_id
    }
    
    num_devices = len(sensor_coords_map_for_planning) # Number of actual sensors to visit

    # --- Generate UAV Tour ---
    tour_node_ids = []
    if algorithm_alias == "LKH":
        if num_devices == 0:
            print("Warning: No sensor nodes to visit. LKH will not be run.")
            tour_node_ids = []
        else:
            # For LKH, we pass the ground station position and the sensor map
            tour_node_ids = generate_tour_from_lkh(ground_station_actual_pos, sensor_coords_map_for_planning)
    else:
        # For models, we load the model and generate the tour
        device = torch.device('cpu')
        model_path = os.path.join("showcases", "new_main", f"{model_name}.pt")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}. Skipping.")
            return
            
        model = load_model_for_inference(model_path, device, graph_size=num_devices if num_devices > 0 else 1)
        
        if num_devices == 0:
            print(f"Warning: No sensor nodes to visit for {data_path}. Tour will be empty.")
            tour_node_ids = []
        else:
            tour_node_ids = generate_tour_from_model(model, sensor_coords_map_for_planning, device)

    if not tour_node_ids and num_devices > 0:
        print(f"Error: Failed to generate tour for {data_path} with algorithm {algorithm_alias}. Skipping.")
        return

    # --- Generate UAV Mission Path ---
    mission_points = []
    gs_above = (ground_station_actual_pos[0], ground_station_actual_pos[1], uav_flight_altitude)
    mission_points.append(gs_above)
    for node_id in tour_node_ids: # tour_node_ids are from sensor_coords_map_for_planning
        if node_id in sensor_coords_map_for_planning: # Should always be true if tour is generated correctly
            sensor_pos = sensor_coords_map_for_planning[node_id]
            mission_points.append((sensor_pos[0], sensor_pos[1], uav_flight_altitude))
    mission_points.append(gs_above) # Return to ground station
    # The final landing is handled by the protocol's logic, not as part of the initial mission plan.
    # This prevents a race condition where the simulation ends before the data dump message is processed.

    # Calculate path length for the mission, and add the final descent for a more accurate time estimate.
    path_length = calculate_path_length(mission_points)
    path_length += uav_flight_altitude # Add final descent distance to path length for time calculation

    # --- Dynamic Simulation Duration Calculation ---
    uav_speed = 25.0  # Must match the speed in SimpleUAVProtocol's MissionMobilityConfiguration
    estimated_time = path_length / uav_speed if uav_speed > 0 else 0
    # Add a generous buffer (e.g., 50% of travel time + 50s fixed) to account for non-travel time like takeoff/landing.
    simulation_duration = estimated_time * 1.5 + 50
    print(f"Calculated Path Length: {path_length:.2f}, Estimated Time: {estimated_time:.2f}s, Set Sim Duration: {simulation_duration:.2f}s")

    # --- Simulation Setup ---
    builder = SimulationBuilder(SimulationConfiguration(duration=simulation_duration, real_time=False))

    # Add Ground Station first to get its ID
    GroundStationProtocol = build_protocol_with_args(SimpleGroundStationProtocol)
    gs_id = builder.add_node(GroundStationProtocol, ground_station_actual_pos) # Use actual GS position
    print(f"Ground Station created with ID: {gs_id} at {ground_station_actual_pos}")

    SensorProtocol = build_protocol_with_args(SimpleSensorProtocol)
    sensor_node_ids = []
    # Deploy only the sensor nodes, not the ground station node again
    for node_id, pos in sensor_coords_map_for_planning.items():
        sensor_sim_id = builder.add_node(SensorProtocol, pos)
        sensor_node_ids.append(sensor_sim_id)
    print(f"Deployed {len(sensor_node_ids)} sensor nodes.")

    # Add UAV and pass the Ground Station's ID to it
    UAVProtocol = build_protocol_with_args(SimpleUAVProtocol,
                                           mission_path=mission_points,
                                           output_dir="showcases/Simulating_a_data_collection_scenario_static",
                                           ground_station_id=gs_id,
                                           ground_station_pos=ground_station_actual_pos)
    # UAV starts at the ground station's actual position (on the ground)
    uav_id = builder.add_node(UAVProtocol, ground_station_actual_pos)
    print(f"UAV created with ID: {uav_id} at {ground_station_actual_pos}")

    builder.add_handler(TimerHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(transmission_range=30)))
    builder.add_handler(MobilityHandler())
    # Visualization can be disabled for batch runs to speed up execution
    # builder.add_handler(VisualizationHandler(VisualizationConfiguration(x_range=(-20, 120), y_range=(-20, 120), z_range=(0, 50))))

    simulation = builder.build()
    
    # Inject simulation object into the UAV protocol
    uav_protocol_instance = simulation.get_node(uav_id).protocol_encapsulator.protocol
    uav_protocol_instance.set_simulation(simulation)

    simulation.start_simulation()

    # --- Data Collection and Output ---
    gs_protocol: SimpleGroundStationProtocol = simulation.get_node(gs_id).protocol_encapsulator.protocol
    uav_protocol: SimpleUAVProtocol = simulation.get_node(uav_id).protocol_encapsulator.protocol

    packets_received = gs_protocol.get_total_packets_received()
    avg_latency = gs_protocol.get_average_latency()
    energy_consumed = uav_protocol.get_energy_consumed()
    mission_duration = uav_protocol.get_mission_duration()
    avg_throughput = packets_received / mission_duration if mission_duration > 0 else 0

    # After simulation, collect the total number of packets generated by all sensors
    total_packets_generated = sum(
        simulation.get_node(node_id).protocol_encapsulator.protocol.get_total_packets_generated()
        for node_id in sensor_node_ids
    )

    output_data = {
        "Run ID": run_id,
        "Algorithm": algorithm_alias,
        "Scenario": scenario_name,
        "Num Devices": num_devices,
        "Instance ID": instance_id_str, # Added Instance ID
        "Repetition": repetition,
        "Path Length (L)": path_length,
        "Packets Expected": total_packets_generated,
        "Packets Received": packets_received,
        "Avg Latency (s)": avg_latency,
        "Avg Throughput (pkt/s)": avg_throughput,
        "Energy Consumed (J)": energy_consumed,
        "Mission Duration (s)": mission_duration,
    }

    output_filename = "showcases/Simulating_a_data_collection_scenario_static/raw_experimental_data_batch.csv"
    
    # Define fieldnames explicitly to ensure order and inclusion of new fields
    fieldnames = [
        "Run ID", "Algorithm", "Scenario", "Num Devices", "Instance ID",
        "Repetition", "Path Length (L)", "Packets Expected", "Packets Received",
        "Avg Latency (s)", "Avg Throughput (pkt/s)", "Energy Consumed (J)",
        "Mission Duration (s)"
    ]
    
    file_exists = os.path.isfile(output_filename)
    with open(output_filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(output_data)
        
    print(f"Results for Run ID {run_id} ({model_name} on {os.path.basename(data_path)}) written to {output_filename}")


# --- Batch execution logic ---

def get_algorithm_alias(model_name: str) -> str:
    """Maps a model filename to a plotting-friendly algorithm name."""
    if "attention_custom_final" in model_name:
        return "CRL-AM"
    elif "finetuned_adaptive" in model_name:
        return "CRL-HA"
    elif "unified_trained" in model_name:
        return "CRL-FD"
    elif "uniform_pretrained" in model_name:
        return "CRL-U"
    else:
        return model_name

if __name__ == "__main__":
    # --- Batch Run Configuration ---
    model_dir = os.path.join("showcases", "new_main")
    
    all_model_paths = glob.glob(os.path.join(model_dir, '*.pt'))
    model_names = [os.path.splitext(os.path.basename(p))[0] for p in all_model_paths]

    if not model_names:
        print(f"Warning: No .pt model files found in {model_dir}. Will only run LKH.")
    else:
        print(f"Found {len(model_names)} models to test: {model_names}")

    # Add LKH as a method to test
    # algorithms_to_test = model_names + ["LKH"]
    algorithms_to_test = [#"tsp50_attention_custom_final","tsp100_attention_custom_final",
                          #"tsp150_attention_custom_final","tsp200_attention_custom_final",
                          "tsp200_unified_trained"]
    
    data_directory = "showcases/Simulating_a_data_collection_scenario_static/data"
    data_files = glob.glob(os.path.join(data_directory, '*.tsp'))
    
    if not data_files:
        print(f"Error: No .tsp files found in {data_directory}. Exiting.")
        sys.exit(1)
    
    # --- End of Configuration ---

    run_counter = 1
    # Calculate total runs, considering model-data size matching
    total_runs = 0
    for algorithm in algorithms_to_test:
        for data_path in data_files:
            if algorithm == "LKH":
                total_runs += 1
                continue
            try:
                model_size = int(re.search(r'tsp(\d+)', algorithm).group(1))
                data_size = int(re.search(r'_(\d+)_', os.path.basename(data_path)).group(1))
                if model_size == data_size:
                    total_runs += 1
            except (AttributeError, ValueError):
                total_runs += 1 # Run if size cannot be determined

    print(f"--- Starting Batch Run: {total_runs} total simulations planned ---")

    for algorithm_name in algorithms_to_test:
        is_lkh = (algorithm_name == "LKH")
        
        if is_lkh:
            algorithm_alias = "LKH"
        else:
            algorithm_alias = get_algorithm_alias(algorithm_name)

        for data_path in data_files:
            if not os.path.exists(data_path):
                print(f"Warning: Data file not found at {data_path}. Skipping.")
                continue

            # If it's a model, match its size to the data size from the filename
            if not is_lkh:
                try:
                    model_size = int(re.search(r'tsp(\d+)', algorithm_name).group(1))
                    data_size = int(re.search(r'_(\d+)_', os.path.basename(data_path)).group(1))
                    if model_size != data_size:
                        continue # Skip if sizes do not match
                except (AttributeError, ValueError):
                    print(f"Warning: Could not determine size for model '{algorithm_name}' or data '{os.path.basename(data_path)}'. Running anyway.")

            # if run_counter <= 950:
            #     run_counter += 1
            #     continue
            print(f"\n--- Running Simulation {run_counter}/{total_runs} ---")
            print(f"Algorithm: {algorithm_alias}")
            print(f"Data: {os.path.basename(data_path)}")
            
            try:
                run_simulation(
                    run_id=run_counter,
                    model_name=algorithm_name, # For LKH, this will be "LKH"
                    algorithm_alias=algorithm_alias,
                    data_path=data_path,
                    repetition=1
                )
            except Exception as e:
                print(f"Error during simulation for algorithm {algorithm_alias} and data {data_path}: {e}")
            
            run_counter += 1

    print(f"\n--- Batch run finished. Total simulations executed: {run_counter - 1} ---")