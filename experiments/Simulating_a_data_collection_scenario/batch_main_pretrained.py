import logging
import json
import random
import csv
import os
import statistics
import pickle
from typing import Dict, List, Tuple
import glob
import sys

# Add project root to sys.path to ensure modules like 'src' and 'nets' can be found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration

from showcases.Simulating_a_data_collection_scenario.simple_protocol import (
    SimpleSensorProtocol,
    SimpleGroundStationProtocol,
    SimpleUAVProtocol,
    PacketBatch,
    SensorStatus
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

Position3D = Tuple[float, float, float]

def load_tsp_instances(file_path: str, num_instances: int) -> List[List[Tuple[float, float]]]:
    """Loads a specified number of TSP instances from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    instances = data[:num_instances]
    return instances

def run_simulation_from_batch(run_id, algorithm, scenario, repetition, model_path, tsp_file):
    status_file = "showcases/Simulating_a_data_collection_scenario/sensor_statuses.json"
    if os.path.exists(status_file):
        os.remove(status_file)
        logging.info(f"Removed existing {status_file}")

    UAV_FLIGHT_ALTITUDE = 20.0
    
    print(f"Loading {scenario['num_instances']} TSP instances from {tsp_file}...")
    tsp_instances_coords = load_tsp_instances(tsp_file, scenario['num_instances'])

    sensor_locations = {}
    instance_tsp_ids = []
    tsp_id_counter = 1

    for instance_coords in tsp_instances_coords:
        current_instance_ids = []
        for x, y in instance_coords:
            pos = (x * 500, y * 500, 0.0)
            sensor_locations[tsp_id_counter] = pos
            current_instance_ids.append(tsp_id_counter)
            tsp_id_counter += 1
        instance_tsp_ids.append(current_instance_ids)

    ground_station_pos = (250, 250, 0)
    
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

    config = SimulationConfiguration(real_time=False)
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(transmission_range=1000)))
    mobility_config = MobilityConfiguration(default_speed=20.0, update_rate=0.1)
    builder.add_handler(MobilityHandler(mobility_config))

    gs_sim_id = builder.add_node(SimpleGroundStationProtocol, ground_station_pos)

    sensor_sim_ids = {}
    for tsp_id, pos in sensor_locations.items():
        if tsp_id == 0: continue
        sim_id = builder.add_node(SimpleSensorProtocol, pos)
        sensor_sim_ids[tsp_id] = sim_id

    uav_start_pos = (ground_station_pos[0], ground_station_pos[1], UAV_FLIGHT_ALTITUDE)
    uav_sim_id = builder.add_node(SimpleUAVProtocol, uav_start_pos)

    simulation = builder.build()

    uav_protocol_instance: SimpleUAVProtocol = simulation.get_node(uav_sim_id).protocol_encapsulator.protocol
    uav_protocol_instance.configure(
        tsp_instances=instance_tsp_ids,
        sensor_coords_map_tsp_id_key=sensor_locations,
        flight_altitude=UAV_FLIGHT_ALTITUDE,
        ground_station_sim_id=gs_sim_id,
        ground_station_physical_pos=ground_station_pos,
        area_bounds=(500.0, 500.0),
        replan_threshold=0,
        num_sensors_to_activate=0,
        num_sensors_to_standby=0,
        algorithm=algorithm,
        model_path=model_path
    )
    uav_protocol_instance.set_simulation(simulation)

    for tsp_id, sim_id in sensor_sim_ids.items():
        protocol: SimpleSensorProtocol = simulation.get_node(sim_id).protocol_encapsulator.protocol
        protocol.set_tsp_id(tsp_id)

    for tsp_id in all_available_sensor_tsp_ids:
        sim_id = sensor_sim_ids[tsp_id]
        protocol: SimpleSensorProtocol = simulation.get_node(sim_id).protocol_encapsulator.protocol
        protocol.set_standby()
        logging.debug(f"Sensor TSP ID {tsp_id} (Sim ID {sim_id}) initialized to standby.")
            
    print(f"Starting simulation for Run ID: {run_id}...")
    simulation.start_simulation()
    print("Simulation ended.")

    gs_proto: SimpleGroundStationProtocol = simulation.get_node(gs_sim_id).protocol_encapsulator.protocol
    all_batches = gs_proto.get_all_received_batches()
    
    latencies = [(b['reception_time_gs'] - b['gen_time']) for b in all_batches if b['reception_time_gs']]
    
    avg_latency = statistics.mean(latencies) if latencies else 0
    
    packets_received = gs_proto.get_total_packets_received_count()
    
    total_packets_generated = 0
    for tsp_id_of_active_sensor in sensor_sim_ids.keys():
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

def batch_main_pretrained():
    models_config = {
        'CRL-AM': 'tsp50_attention_custom_final.pt',
        'CRL-HA': 'tsp50_finetuned_adaptive.pt',
        'CRL-FD': 'tsp50_unified_trained.pt',
        'CRL-U': 'tsp50_uniform_pretrained.pt'
    }
    
    instance_sizes = [50, 100, 150, 200]
    num_stages_list = [2, 4, 6, 8, 10]
    repetitions = 1
    
    results = []
    run_counter = 1

    all_instances_data = {}
    for size in instance_sizes:
        tsp_file = f"data/tsp/tsp{size}_val_mg_seed2222_size10K.pkl"
        try:
            with open(tsp_file, 'rb') as f:
                all_instances_data[size] = pickle.load(f)
            logging.info(f"Successfully loaded {len(all_instances_data[size])} instances from {tsp_file}")
        except FileNotFoundError:
            logging.error(f"Data file not found: {tsp_file}. Skipping size {size}.")
            continue

    for size in instance_sizes:
        if size not in all_instances_data:
            continue

        for num_stages in num_stages_list:
            if num_stages > len(all_instances_data[size]):
                logging.warning(f"Skipping scenario with {num_stages} stages for size {size} because only {len(all_instances_data[size])} instances are available.")
                continue

            for model_alias, model_file in models_config.items():
                # Adapt model file for different sizes, assuming a pattern like tsp{size}_...
                # This is a bit of a guess, adjust if the pattern is different.
                # e.g., tsp50_attention_custom_final.pt -> tsp100_attention_custom_final.pt
                
                # Correctly replace the size in the filename
                base_model_name = model_file.replace('tsp50_', '')
                current_model_file = f"tsp{size}_{base_model_name}"
                model_path = f"showcases/new_main/{current_model_file}"

                if not os.path.exists(model_path):
                    logging.warning(f"Model file not found: {model_path}. Skipping this configuration.")
                    continue

                for i in range(repetitions):
                    scenario = {
                        "name": f"{num_stages} Instances of {size} ({model_alias})",
                        "num_instances": num_stages,
                        "instance_size": size
                    }
                    
                    tsp_file = f"data/tsp/tsp{size}_val_mg_seed2222_size10K.pkl"

                    logging.info(f"--- Starting Run #{run_counter}: Model={model_alias}, Size={size}, Stages={num_stages}, Repetition={i+1} ---")
                    
                    result = run_simulation_from_batch(
                        run_id=run_counter,
                        algorithm=model_alias, # Use the alias as the algorithm name
                        scenario=scenario,
                        repetition=i + 1,
                        model_path=model_path,
                        tsp_file=tsp_file
                    )
                    results.append(result)
                    run_counter += 1

    output_file = "showcases/Simulating_a_data_collection_scenario/simulation_results_pretrained_batch.csv"
    if not results:
        print("No results to save.")
        return

    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logging.info(f"\nBatch simulation results for pretrained models saved to {output_file}")
    except (IOError, IndexError) as e:
        logging.error(f"Error saving results to {output_file}: {e}")

if __name__ == "__main__":
    batch_main_pretrained()