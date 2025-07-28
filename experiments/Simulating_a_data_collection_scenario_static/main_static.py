from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration
from simple_protocol import SimpleSensorProtocol, SimpleGroundStationProtocol, SimpleUAVProtocol
import re # For parsing TSP file
from typing import Dict, List, Tuple, Type
from gradysim.protocol.interface import IProtocol, IProvider

# Type alias for position for clarity
Position3D = Tuple[float, float, float]


def build_protocol_with_args(protocol_class: Type[IProtocol], **kwargs) -> Type[IProtocol]:
    """
    Dynamically creates a new protocol class that inherits from protocol_class
    and has its __init__ method patched to accept the given kwargs.
    """
    class PatchedProtocol(protocol_class):
        @classmethod
        def instantiate(cls, provider: IProvider) -> 'IProtocol':
            # We instantiate the original class with the provided arguments
            protocol = protocol_class(**kwargs)
            protocol.provider = provider
            return protocol
    return PatchedProtocol

def calculate_path_length(path: List[Position3D]) -> float:
    """Calculates the total length of a path defined by a list of 3D points."""
    total_length = 0.0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5
        total_length += distance
    return total_length

def load_sensor_locations_from_tsp(tsp_file_path: str) -> Tuple[Dict[int, Position3D], int]:
    """
    Parses a TSP file to extract node coordinates and the dimension.
    Assumes EUC_2D format where coordinates are (id, x, y).
    Returns a tuple containing:
    - A dictionary mapping node ID (int) to (x, y, z) tuples, with z defaulting to 0.0.
    - The dimension (number of nodes) specified in the file.
    """
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
                    print(f"Warning: Could not parse DIMENSION line: {line}")
            elif line == "NODE_COORD_SECTION":
                in_coord_section = True
                continue
            elif line == "EOF":
                break
            
            if in_coord_section and line:
                parts = re.split(r'\s+', line)
                if len(parts) >= 3:
                    try:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        z = 0.0  # Default Z for sensors on the ground
                        if len(parts) >= 4: # Check for optional Z
                            try:
                                z_val = float(parts[3])
                                if -1000 < z_val < 10000:
                                     z = z_val
                            except ValueError:
                                pass # z remains 0.0
                        coordinates_map[node_id] = (x, y, z)
                    except ValueError:
                        print(f"Warning: Could not parse coordinate line: {line}")
                else:
                    print(f"Warning: Skipping malformed coordinate line: {line}")
    return coordinates_map, dimension

def load_tsp_tour_sequence(tour_file_path: str) -> List[int]:
    """
    Parses a TSP tour file to extract the sequence of node IDs.
    """
    tour_sequence: List[int] = []
    in_tour_section = False
    with open(tour_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_tour_section = True
                continue
            if line == "-1" or line == "EOF": # Tour termination markers
                break
            if in_tour_section and line:
                try:
                    tour_sequence.append(int(line))
                except ValueError:
                    print(f"Warning: Could not parse tour node ID: {line}")
    return tour_sequence

def generate_uav_mission_path_str(
    sensor_coords_map: Dict[int, Position3D],
    tour_sequence: List[int],
    flight_altitude: float = 20.0,
    ground_station_pos: Position3D = (0.0, 0.0, 0.0)
) -> str:
    """
    Generates the UAV mission path string based on TSP tour sequence and sensor coordinates.
    The path starts above the ground station, visits each sensor in order at flight_altitude,
    returns above the ground station, and then lands at the ground station.
    """
    path_points: List[Position3D] = []

    gs_above = (ground_station_pos[0], ground_station_pos[1], flight_altitude)

    # Start above ground station
    path_points.append(gs_above)

    # Visit sensors in tour order
    for node_id in tour_sequence:
        if node_id in sensor_coords_map:
            sensor_pos = sensor_coords_map[node_id]
            path_points.append((sensor_pos[0], sensor_pos[1], flight_altitude))
        else:
            print(f"Warning: Node ID {node_id} from tour not found in sensor coordinates map. Skipping.")
    
    # Return above ground station
    path_points.append(gs_above)
    # Land at ground station
    path_points.append(ground_station_pos)

    # Format as a Python list string
    path_str = "[\n"
    for point in path_points:
        path_str += f"    {point},\n"
    path_str += "]"
    return path_str

def main(run_id=1, algorithm="ADAPT-GUAV", scenario="Forest", num_sensors=100, repetition=1):
    # --- Configuration ---
    sensor_locations_tsp_file = f"showcases/Simulating_a_data_collection_scenario_static/sensor_locations_{scenario}_{num_sensors}.tsp"
    tsp_tour_file = f"showcases/Simulating_a_data_collection_scenario_static/tour_{algorithm}_{scenario}_{num_sensors}.tsp"
    uav_flight_altitude = 20.0 # Altitude for visiting sensors
    ground_station_actual_pos: Position3D = (0.0, 0.0, 0.0) # Physical location of GS
    simulation_duration = 450 # May need adjustment based on path length and speed

    # --- Load Data ---
    sensor_coords_map, num_devices = load_sensor_locations_from_tsp(sensor_locations_tsp_file)
    if not sensor_coords_map:
        print(f"Error: No sensor positions loaded from {sensor_locations_tsp_file}. Exiting.")
        return
    if num_devices == 0:
        print(f"Warning: Dimension not found or is zero in {sensor_locations_tsp_file}. Using number of loaded coordinates.")
        num_devices = len(sensor_coords_map)

    tour_node_ids = load_tsp_tour_sequence(tsp_tour_file)
    if not tour_node_ids:
        print(f"Error: No tour sequence loaded from {tsp_tour_file}. Exiting.")
        return
        
    # --- Generate UAV Mission Path String (for simple_protocol.py) ---
    # This string will be manually copied or programmatically inserted into simple_protocol.py
    # For this exercise, we'll print it. In a more advanced setup, we might pass it directly.
    # Generate the list of mission points
    mission_points = []
    gs_above = (ground_station_actual_pos[0], ground_station_actual_pos[1], uav_flight_altitude)
    mission_points.append(gs_above)
    for node_id in tour_node_ids:
        if node_id in sensor_coords_map:
            sensor_pos = sensor_coords_map[node_id]
            mission_points.append((sensor_pos[0], sensor_pos[1], uav_flight_altitude))
    mission_points.append(gs_above)
    mission_points.append(ground_station_actual_pos)

    # Calculate path length
    path_length = calculate_path_length(mission_points)

    # --- Simulation Setup ---
    # Pass mission parameters to the simulation
    config = SimulationConfiguration(
        duration=simulation_duration,
        real_time=False,
    )
    builder = SimulationBuilder(config)

    # Instantiating sensors based on TSP file data (using values from the map)
    SensorProtocol = build_protocol_with_args(SimpleSensorProtocol)
    for node_id, pos in sensor_coords_map.items():
        builder.add_node(SensorProtocol, pos)
    print(f"Added {len(sensor_coords_map)} sensors from {sensor_locations_tsp_file}")

    # Instantiating 1 UAV. Its mission is defined in simple_protocol.py
    # It will start at the first point of its mission path.
    UAVProtocol = build_protocol_with_args(SimpleUAVProtocol,
                                           mission_path=mission_points,
                                           output_dir="showcases/Simulating_a_data_collection_scenario_static")
    uav_id = builder.add_node(UAVProtocol, (0,0,0)) # Initial physical spawn, mission plugin handles actual start pos.
    # builder.add_node(SimpleUAVProtocol, (0, 0, 0))
    # builder.add_node(SimpleUAVProtocol, (0, 0, 0))
    # builder.add_node(SimpleUAVProtocol, (0, 0, 0))

    # Instantiating ground station at (0,0,0)
    GroundStationProtocol = build_protocol_with_args(SimpleGroundStationProtocol)
    gs_id = builder.add_node(GroundStationProtocol, (0, 0, 0))

    # Adding required handlers
    builder.add_handler(TimerHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=30
    )))
    builder.add_handler(MobilityHandler())
    builder.add_handler(VisualizationHandler(VisualizationConfiguration(
        x_range=(-20, 120),  # Adjusted for sensor grid 0-90
        y_range=(-20, 120),  # Adjusted for sensor grid 0-90
        z_range=(0, 50)      # Adjusted for UAV flight altitude
    )))

    # Building & starting
    simulation = builder.build()

    # Inject simulation object into the UAV protocol
    uav_protocol = simulation.get_node(uav_id).protocol_encapsulator.protocol
    uav_protocol.set_simulation(simulation)

    simulation.start_simulation()

    # --- Data Collection and Output ---
    # After simulation, collect data from protocols
    gs_protocol: SimpleGroundStationProtocol = simulation.get_node(gs_id).protocol_encapsulator.protocol
    uav_protocol: SimpleUAVProtocol = simulation.get_node(uav_id).protocol_encapsulator.protocol

    packets_received = gs_protocol.get_total_packets_received()
    avg_latency = gs_protocol.get_average_latency()
    
    energy_consumed = uav_protocol.get_energy_consumed()
    mission_duration = uav_protocol.get_mission_duration()

    avg_throughput = packets_received / mission_duration if mission_duration > 0 else 0

    # Prepare data for CSV
    output_data = {
        "Run ID": run_id,
        "Algorithm": algorithm,
        "Scenario": scenario,
        "Num Devices": num_devices,
        "Repetition": repetition,
        "Path Length (L)": path_length,
        "Packets Expected": num_devices,
        "Packets Received": packets_received,
        "Avg Latency (s)": avg_latency,
        "Avg Throughput (pkt/s)": avg_throughput,
        "Energy Consumed (J)": energy_consumed,
        "Mission Duration (s)": mission_duration,
    }

    # Write to CSV
    import csv
    import os
    
    output_filename = "showcases/Simulating_a_data_collection_scenario_static/raw_experimental_data.csv"
    file_exists = os.path.isfile(output_filename)
    
    with open(output_filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(output_data)
        
    print(f"Results for Run ID {run_id} written to {output_filename}")


if __name__ == "__main__":
    # This part can be wrapped in a loop for multiple runs
    main(run_id=1, algorithm="ADAPT-GUAV", scenario="Forest", num_sensors=100, repetition=1)