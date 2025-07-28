import sys
import os

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import logging
import json
import random
import csv
import os
import statistics
import pickle
from typing import Dict, List, Tuple
import glob # Added for finding dynamically deactivated sensors file

from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration

from simple_protocol import (
    SimpleSensorProtocol,
    SimpleGroundStationProtocol,
    SimpleUAVProtocol,
    PacketBatch,
    SensorStatus  # Import SensorStatus enum
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

Position3D = Tuple[float, float, float]

def load_tsp_instances(file_path: str, num_instances: int) -> List[List[Tuple[float, float]]]:
    """Loads a specified number of TSP instances from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    # Assuming data is a list of instances, and each instance has 'depot' and 'loc'
    # We are interested in the 'loc' part, which are the sensor coordinates.
    # Let's take the first 'num_instances' from the dataset.
    instances = data[:num_instances]
    return instances

def run_simulation(run_id, algorithm, scenario, repetition, tsp_file):
    # --- Clean up and Initialize sensor_statuses.json ---
    status_file = "showcases/Simulating_a_data_collection_scenario/sensor_statuses.json"
    if os.path.exists(status_file):
        os.remove(status_file)
        logging.info(f"Removed existing {status_file}")

    # --- Simulation Configuration ---
    UAV_FLIGHT_ALTITUDE = 20.0
    
    # --- Setup Sensor Locations from TSP Instances ---
    print(f"Loading {scenario['num_instances']} TSP instances from {tsp_file}...")
    tsp_instances_coords = load_tsp_instances(tsp_file, scenario['num_instances'])

    sensor_locations = {} # This will hold all sensor locations, tsp_id -> pos
    instance_tsp_ids = [] # List of lists, where each inner list holds the tsp_ids for an instance
    tsp_id_counter = 1  # Start TSP IDs from 1

    for instance_coords in tsp_instances_coords:
        current_instance_ids = []
        for x, y in instance_coords:
            # Assuming coordinates are normalized, scaling them to a 500x500 area
            pos = (x * 500, y * 500, 0.0)
            sensor_locations[tsp_id_counter] = pos
            current_instance_ids.append(tsp_id_counter)
            tsp_id_counter += 1
        instance_tsp_ids.append(current_instance_ids)

    # Ground station position
    ground_station_pos = (250, 250, 0)
    
    # Initialize sensor_statuses.json with all sensors as RAW
    initial_statuses = {str(tsp_id): SensorStatus.RAW.value for tsp_id in sensor_locations.keys()}
    try:
        with open(status_file, 'w') as f:
            json.dump(initial_statuses, f, indent=4)
        logging.info(f"Initialized {status_file} with all sensors in RAW state.")
    except IOError as e:
        logging.error(f"Error initializing {status_file}: {e}")

    all_available_sensor_tsp_ids = list(sensor_locations.keys())
    TOTAL_SENSORS_FROM_FILE = len(all_available_sensor_tsp_ids)
    print(f"Total sensors loaded from {scenario['num_instances']} instances: {TOTAL_SENSORS_FROM_FILE}")

    # --- Simulation Setup ---
    config = SimulationConfiguration(real_time=False)
    builder = SimulationBuilder(config)

    # Add handlers
    builder.add_handler(TimerHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(transmission_range=1000)))
    mobility_config = MobilityConfiguration(default_speed=20.0, update_rate=0.1)
    builder.add_handler(MobilityHandler(mobility_config))

    # --- Node Initialization ---
    # Ground Station
    gs_sim_id = builder.add_node(SimpleGroundStationProtocol, ground_station_pos)

    # Sensors
    sensor_sim_ids = {} # tsp_id -> sim_id
    for tsp_id, pos in sensor_locations.items():
        if tsp_id == 0: continue # Skip ground station
        sim_id = builder.add_node(SimpleSensorProtocol, pos)
        sensor_sim_ids[tsp_id] = sim_id

    # UAV
    uav_start_pos = (ground_station_pos[0], ground_station_pos[1], UAV_FLIGHT_ALTITUDE)
    uav_sim_id = builder.add_node(SimpleUAVProtocol, uav_start_pos)

    # --- Build Simulation ---
    simulation = builder.build()

    # --- Configure UAV Protocol ---
    uav_protocol_instance: SimpleUAVProtocol = simulation.get_node(uav_sim_id).protocol_encapsulator.protocol
    # --- Select Initial Active Sensors Randomly ---
    uav_protocol_instance.configure(
        tsp_instances=instance_tsp_ids,  # Pass the list of TSP ID lists
        sensor_coords_map_tsp_id_key=sensor_locations,
        flight_altitude=UAV_FLIGHT_ALTITUDE,
        ground_station_sim_id=gs_sim_id,
        ground_station_physical_pos=ground_station_pos,
        area_bounds=(500.0, 500.0),
        algorithm=algorithm,
        # The following parameters might need to be re-evaluated or are no longer applicable
        # in the same way for the new multi-stage logic.
        replan_threshold=0, # Deactivate replanning based on count
        num_sensors_to_activate=0,
        num_sensors_to_standby=0
    )
    uav_protocol_instance.set_simulation(simulation)

    # Set TSP IDs for sensors and then set their initial state (ACTIVE or STANDBY)
    # This ensures TSP ID is available when activate/set_standby updates the status file.
    for tsp_id, sim_id in sensor_sim_ids.items():
        protocol: SimpleSensorProtocol = simulation.get_node(sim_id).protocol_encapsulator.protocol
        protocol.set_tsp_id(tsp_id) # Set TSP ID first

    # --- Set Initial Sensor States (ACTIVE or STANDBY) ---
    for tsp_id in all_available_sensor_tsp_ids: # Iterate through ALL sensors from file
        sim_id = sensor_sim_ids[tsp_id]
        protocol: SimpleSensorProtocol = simulation.get_node(sim_id).protocol_encapsulator.protocol
        # In the new logic, all sensors start in a non-active state (RAW, then STANDBY).
        # The UAV will activate them as it processes each TSP instance.
        protocol.set_standby()
        logging.debug(f"Sensor TSP ID {tsp_id} (Sim ID {sim_id}) initialized to standby.")
            
    # --- Start Simulation ---
    print(f"Starting simulation for Run ID: {run_id}...")
    simulation.start_simulation()
    print("Simulation ended.")

    # --- Generate plot_categories.json ---
    plot_categories = {}
    status_file_path = "showcases/Simulating_a_data_collection_scenario/sensor_statuses.json"
    final_sensor_statuses = {}
    try:
        with open(status_file_path, 'r') as f:
            final_sensor_statuses = json.load(f)
    except Exception as e:
        logging.error(f"Could not read final sensor statuses from {status_file_path}: {e}")

    dynamically_deactivated_tsp_ids = set()
    # Assuming UAV ID might be part of the filename, e.g., uav_1000_dynamically_deactivated_sensors.json
    # For simplicity, we'll try to find one such file. If multiple UAVs, this might need adjustment.
    script_dir = "showcases/Simulating_a_data_collection_scenario" # Assuming main.py is run from project root
    deactivated_files = glob.glob(os.path.join(script_dir, "uav_*_dynamically_deactivated_sensors.json"))
    if deactivated_files:
        try:
            with open(deactivated_files[0], 'r') as f:
                deactivated_data = json.load(f)
                dynamically_deactivated_tsp_ids = {s['tsp_id'] for s in deactivated_data}
        except Exception as e:
            logging.error(f"Could not read dynamically deactivated sensors from {deactivated_files[0]}: {e}")
    
    # all_available_sensor_tsp_ids is already defined and contains TSP IDs of sensors (excluding GS)
    # initial_sensor_tour_tsp_ids is also available in this scope
    initial_sensor_tour_tsp_ids = all_available_sensor_tsp_ids
    logging.debug(f"DEBUG main.py: initial_sensor_tour_tsp_ids (size {len(initial_sensor_tour_tsp_ids)}): {list(initial_sensor_tour_tsp_ids)[:20]}") # Print first 20

    for tsp_id_str in final_sensor_statuses.keys(): # Keys in sensor_statuses are strings
        tsp_id = int(tsp_id_str)
        final_status = final_sensor_statuses[tsp_id_str] # This is the integer value
        category = "UNKNOWN" # Default category

        # SensorStatus enum values: RAW = -1, STANDBY = 0, ACTIVE = 1, SERVICED = 2
        is_active_or_serviced = (final_status == SensorStatus.ACTIVE.value or final_status == SensorStatus.SERVICED.value)
        is_standby_or_raw = (final_status == SensorStatus.STANDBY.value or final_status == SensorStatus.RAW.value)
        
        is_initial_target = tsp_id in initial_sensor_tour_tsp_ids

        if tsp_id in dynamically_deactivated_tsp_ids:
            category = "DYNAMIC_DEACTIVATED_EXPLICIT"
        elif is_initial_target:
            if is_active_or_serviced:
                category = "INITIAL_ACTIVE"
            elif is_standby_or_raw:
                category = "DYNAMIC_DEACTIVATED_IMPLICIT"
            # else: it remains UNKNOWN if status is weird
        elif is_active_or_serviced: # Not initial and active/serviced
             category = "DYNAMIC_NEWLY_ACTIVE"
        elif final_status == SensorStatus.STANDBY.value:
             category = "OTHER_STANDBY"
        elif final_status == SensorStatus.RAW.value:
             category = "OTHER_RAW" # Will likely not be plotted by plot_script
        
        logging.debug(f"DEBUG main.py: TSP ID: {tsp_id}, Final Status: {final_status}, Is Initial: {is_initial_target}, Assigned Category: {category}")
        plot_categories[str(tsp_id)] = category

    plot_categories_file_path = os.path.join(script_dir, "plot_categories.json")
    try:
        with open(plot_categories_file_path, 'w') as f:
            json.dump(plot_categories, f, indent=4)
        logging.info(f"Sensor plot categories saved to {plot_categories_file_path}")
    except IOError as e:
        logging.error(f"Error writing plot categories to {plot_categories_file_path}: {e}")
    # --- End Generate plot_categories.json ---

    # --- Results ---
    gs_proto: SimpleGroundStationProtocol = simulation.get_node(gs_sim_id).protocol_encapsulator.protocol
    all_batches = gs_proto.get_all_received_batches()
    
    latencies = [(b['reception_time_gs'] - b['gen_time']) for b in all_batches if b['reception_time_gs']]
    
    avg_latency = statistics.mean(latencies) if latencies else 0
    
    packets_received = gs_proto.get_total_packets_received_count()
    
    total_packets_generated = 0
    # Iterate over sensor_sim_ids which contains only sensors that were part of the simulation
    for tsp_id_of_active_sensor in sensor_sim_ids.keys(): # sensor_sim_ids maps tsp_id to sim_id
        sim_id = sensor_sim_ids[tsp_id_of_active_sensor]
        sensor_proto: SimpleSensorProtocol = simulation.get_node(sim_id).protocol_encapsulator.protocol
        total_packets_generated += sensor_proto.get_total_generated_packets()
 
    mission_duration = gs_proto.get_current_time()
    avg_throughput = packets_received / mission_duration if mission_duration > 0 else 0
    
    path_length = uav_protocol_instance.get_path_length()
    energy_consumed = uav_protocol_instance.get_energy_consumed()
    total_computation_time = uav_protocol_instance.get_total_computation_time()

    return {
        "Run ID": run_id,
        "Algorithm (mathcalM)": algorithm,
        "Scenario (mathcalS)": scenario["name"],
        "Data Source": tsp_file,
        "Instances": scenario['num_instances'],
        "Instance Size": scenario['instance_size'],
        "Repetition": repetition,
        "Path Length (L)": path_length,
        "Packets Expected": total_packets_generated,
        "Packets Received": packets_received,
        "Avg Latency (mathcalL)": avg_latency,
        "Avg Throughput (mathcalT)": avg_throughput,
        "Energy Consumed": energy_consumed,
        "Mission Duration": mission_duration,
        "Total Computation Time (ms)": total_computation_time,
    }

def main():
    # --- Simulation Execution ---
    # 0 <= devices <= TOTAL_SENSORS
    # 1 <= replan_threshold <= devices
    # 0 <= num_sensors_to_activate <= (TOTAL_SENSORS - devices)
    # 0 <= num_sensors_to_standby <= (devices - replan_threshold)
    scenarios = [
        {"name": "4 Instances of 50", "num_instances": 4, "instance_size": 50},
    ]
    repetitions = 1
    results = []
    run_counter = 1

    for i in range(repetitions):
        for scenario in scenarios:
            result = run_simulation(
                run_id=run_counter,
                algorithm="LKH",
                scenario=scenario,
                repetition=i + 1,
                tsp_file="data/tsp/tsp50_train_seed1111_size10K.pkl"
            )
            results.append(result)
            run_counter += 1

    # --- Save Results to CSV ---
    output_file = "showcases/Simulating_a_data_collection_scenario/simulation_results.csv"
    if not results:
        print("No results to save.")
        return

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSimulation results saved to {output_file}")

if __name__ == "__main__":
    main()
