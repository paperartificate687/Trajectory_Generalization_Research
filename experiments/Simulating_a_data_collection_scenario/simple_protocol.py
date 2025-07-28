import enum
import json
import random
import logging
import csv
import sys
import os
import torch
from src.options import get_options
from utils import load_problem, torch_load_cpu
from nets.attention_model import AttentionModel

from typing import TypedDict, Dict, List, Tuple as TypingTuple, Optional, Union, TYPE_CHECKING
import re
import subprocess
import tempfile
import numpy as np
import time

# Add project root to sys.path to allow for project-level imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new centralized inference functions
from showcases.Simulating_a_data_collection_scenario.inference_utils import load_attention_model, solve_tsp_with_attention

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand
from gradysim.protocol.position import Position

if TYPE_CHECKING:
    from gradysim.simulator.simulation import Simulation

# Type alias for position
Position3D = TypingTuple[float, float, float]

STATUS_FILE_PATH = "showcases/Simulating_a_data_collection_scenario/sensor_statuses.json"

# Helper function to update the sensor_statuses.json file
# This function should be called by the protocol whenever its state changes
def _update_sensor_status_file(tsp_id: Optional[int], status_value: int, current_time: float, node_id_for_log: int):
    logger = logging.getLogger(f"{__name__}._update_sensor_status_file")
    if tsp_id is None:
        logger.warning(f"Node {node_id_for_log}: Attempted to update status for sensor with no TSP ID.")
        return

    statuses = {}
    try:
        with open(STATUS_FILE_PATH, 'r') as f:
            statuses = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # File might not exist yet or is empty/corrupted.
        # main.py should pre-populate it. If called before that, it's an issue.
        logger.warning(f"Node {node_id_for_log} (TSP ID: {tsp_id}): {STATUS_FILE_PATH} not found or unreadable. Creating/overwriting.")
        pass # Allow creation/overwrite

    statuses[str(tsp_id)] = status_value # Store only the status value

    try:
        with open(STATUS_FILE_PATH, 'w') as f:
            json.dump(statuses, f, indent=4)
        logger.debug(f"Node {node_id_for_log} (TSP ID: {tsp_id}): Updated status to {status_value} at time {current_time} in {STATUS_FILE_PATH}")
    except IOError as e:
        logger.error(f"Node {node_id_for_log} (TSP ID: {tsp_id}): Error writing to {STATUS_FILE_PATH}: {e}")


def write_tsp_file(filename: str, points: List[Position3D], name="TSP", depot_node=1):
    with open(filename, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {len(points)}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y, z) in enumerate(points):
            f.write(f"{i + 1} {x} {y}\n")
        f.write(f"DEPOT_SECTION\n{depot_node}\n-1\n")
        f.write("EOF\n")

class SimpleSender(enum.Enum):
    SENSOR = 0
    UAV = 1
    GROUND_STATION = 2

class SensorStatus(enum.Enum):
    RAW = -1  # Or any other distinct value, like 3 if you prefer positive
    STANDBY = 0
    ACTIVE = 1
    SERVICED = 2

class PacketBatch(TypedDict):
    sensor_id: int
    gen_time: float
    count: int
    reception_time_gs: Optional[float]

class SimpleMessage(TypedDict):
    sender_type: int
    sender: int
    uav_position: Optional[Position3D]
    packet_batches: Optional[List[PacketBatch]]
    packet_count: Optional[int]
    is_ack: Optional[bool]
    is_data_dump: Optional[bool]
    simulation_should_end: Optional[bool]
    is_service_request: Optional[bool]
    sensor_position: Optional[Position3D]
    sensor_tsp_id: Optional[int]
    sensors_to_activate: Optional[List[int]]

class SimpleSensorProtocol(IProtocol):
    provider_id: Optional[int] # Can be None initially
    tsp_id: Optional[int] = None
    _log: Optional[logging.Logger] = None # Initialize as None

    packet_count: int
    current_batch_gen_time: float
    my_position: Position3D
    _position_initialized: bool = False
    _has_data_to_send: bool = False
    _service_request_sent: bool = False
    deactivated: bool = False
    active_on_start: bool = False # This will be removed from direct use in initialize
    status: SensorStatus = SensorStatus.RAW # Initial internal state

    COMMUNICATION_RANGE = 25.0

    def set_tsp_id(self, tsp_id: int):
        self.tsp_id = tsp_id

    def set_active_on_start(self, active: bool):
        # This method is kept for compatibility if main.py still uses it,
        # but activate() or set_standby() should be called directly by main.py
        # after tsp_id is set.
        self.active_on_start = active

    def __init__(self):
        super().__init__()
        self._log = None
        self.provider_id = None # Will be set in _ensure_logger_initialized or initialize
        self.packet_count = 0
        self.current_batch_gen_time = 0.0
        self.my_position: Position3D = (0.0, 0.0, 0.0) # Default position
        self.total_packets_generated = 0

    def _ensure_logger_initialized(self):
        if self._log is None:
            provider_id_for_logger = "UNKNOWN_YET"
            current_provider_id = None

            if hasattr(self, 'provider') and self.provider is not None:
                try:
                    current_provider_id = self.provider.get_id()
                    provider_id_for_logger = str(current_provider_id)
                except Exception as e_get_id:
                    # Can't use self._log here as it's what we are trying to set up
                    fallback_logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.ENSURE_LOG_FAIL")
                    fallback_logger.error(f"Error getting provider_id in _ensure_logger_initialized: {e_get_id}")
            
            self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{provider_id_for_logger}")
            
            if self.provider_id is None and current_provider_id is not None:
                self.provider_id = current_provider_id
            elif self.provider_id is None and current_provider_id is None:
                 self.provider_id = "NOT_SET" # Fallback if provider_id was never set

    def initialize(self) -> None:
        self._ensure_logger_initialized() # Ensure logger is ready

        # Set self.provider_id if not already set by _ensure_logger_initialized
        if self.provider_id is None or self.provider_id == "NOT_SET" or self.provider_id == "UNKNOWN_YET":
            try:
                self.provider_id = self.provider.get_id()
                # If logger was named with a temporary ID, update it
                if self._log.name.endswith("UNKNOWN_YET") or self._log.name.endswith("NOT_SET"):
                     self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{self.provider_id}")
            except Exception as e:
                self._log.error(f"Failed to get provider_id in initialize after _ensure_logger: {e}")
                self.provider_id = "INIT_FAIL_ID" # Fallback

        self.packet_count = 0
        self.total_packets_generated = 0
        self.current_batch_gen_time = 0.0
        self.deactivated = False
        # self.status is intentionally NOT set here.
        # Its initial state is set by the default class-level variable and then
        # configured by main.py before the simulation starts.
        self._log.info(f"Sensor Sim ID {self.provider_id} initialized. TSP ID (at init): {self.tsp_id}. Current status in memory: {self.status.name}")

    def deactivate(self): # This effectively sets it to STANDBY
        self._ensure_logger_initialized()
        self._log.info(f"Sensor (TSP ID: {self.tsp_id}, Sim ID: {self.provider_id}) deactivated (set to STANDBY).")
        self.deactivated = True # This flag prevents packet generation
        self._has_data_to_send = False
        self.packet_count = 0
        
        if self.status != SensorStatus.STANDBY:
            self.status = SensorStatus.STANDBY
            if self.provider_id is not None: # Ensure provider_id is available for the helper
                _update_sensor_status_file(self.tsp_id, self.status.value, self.provider.current_time(), self.provider_id)
            else:
                self._log.error("Cannot update status file in deactivate: provider_id is None.")

    def _generate_packet(self) -> None:
        if self.deactivated or not self._has_data_to_send:
            return
        if self.packet_count == 0:
            self.current_batch_gen_time = self.provider.current_time()
        self.packet_count += 1
        self.total_packets_generated += 1
        self.provider.schedule_timer("generate", self.provider.current_time() + 0.5)

    def activate(self):
        self._ensure_logger_initialized()
        if self.deactivated:
            self._log.warning(f"Sensor (TSP ID: {self.tsp_id}, Sim ID: {self.provider_id}) activate called but it's marked as deactivated.")
            return

        self._log.info(f"Sensor (TSP ID: {self.tsp_id}, Sim ID: {self.provider_id}) activating.")
        self._has_data_to_send = True
        self.deactivated = False # Ensure it's not deactivated if being activated

        if self.status != SensorStatus.ACTIVE:
            self.status = SensorStatus.ACTIVE
            _update_sensor_status_file(self.tsp_id, self.status.value, self.provider.current_time(), self.provider_id)
        
        self._generate_packet()
        if self._position_initialized:
            self._request_service()

    def set_standby(self):
        self._ensure_logger_initialized()
        self._log.info(f"Sensor (TSP ID: {self.tsp_id}, Sim ID: {self.provider_id}) being set to STANDBY.")
        self.deactivated = True # Standby sensors don't generate packets unless activated
        self._has_data_to_send = False # No data to send when standby
        self.packet_count = 0

        if self.status != SensorStatus.STANDBY:
            self.status = SensorStatus.STANDBY
            _update_sensor_status_file(self.tsp_id, self.status.value, self.provider.current_time(), self.provider_id)

    def handle_timer(self, timer: str) -> None:
        if self.deactivated: return
        if timer == "generate":
            self._generate_packet()
        elif timer == "service_request":
            self._request_service()

    def _request_service(self):
        if self.deactivated or not self._position_initialized or not self._has_data_to_send or self._service_request_sent:
            return
        
        request_msg: SimpleMessage = {
            'sender_type': SimpleSender.SENSOR.value, 'sender': self.provider_id,
            'sensor_tsp_id': self.tsp_id, 'is_service_request': True,
            'sensor_position': self.my_position, 'uav_position': None,
            'packet_batches': None, 'packet_count': None, 'is_ack': None,
            'is_data_dump': None, 'simulation_should_end': None
        }
        self.provider.send_communication_command(BroadcastMessageCommand(json.dumps(request_msg)))
        self._service_request_sent = True
        self.provider.schedule_timer("service_request", self.provider.current_time() + 1)

    def handle_packet(self, message_json: str) -> None:
        simple_message: SimpleMessage = json.loads(message_json)

        if simple_message['sender_type'] == SimpleSender.UAV.value:
            # Activation logic: Check if this sensor is in the activation list.
            sensors_to_activate = simple_message.get('sensors_to_activate')
            if sensors_to_activate and self.tsp_id in sensors_to_activate:
                if self.status != SensorStatus.ACTIVE:
                    self._log.info(f"Received activation command.")
                    self.activate()
            
            # Data upload logic is for when the sensor is already active and has data to send.
            if self.deactivated or not self._has_data_to_send or self.packet_count == 0:
                return

            uav_pos = simple_message.get('uav_position')
            if not uav_pos or not self._position_initialized:
                return

            distance = np.linalg.norm(np.array(uav_pos) - np.array(self.my_position))

            if distance <= self.COMMUNICATION_RANGE:
                batch_to_send: PacketBatch = {
                    'sensor_id': self.tsp_id,
                    'gen_time': self.current_batch_gen_time,
                    'count': self.packet_count,
                    'reception_time_gs': None
                }
                response: SimpleMessage = {
                    'sender_type': SimpleSender.SENSOR.value, 'sender': self.provider_id,
                    'sensor_tsp_id': self.tsp_id, 'packet_batches': [batch_to_send],
                    'packet_count': self.packet_count, 'uav_position': None, 'is_ack': None,
                    'is_data_dump': None, 'is_service_request': None, 'sensor_position': None,
                    'simulation_should_end': None
                }
                self.provider.send_communication_command(SendMessageCommand(json.dumps(response), simple_message['sender']))
                self.packet_count = 0
                self._has_data_to_send = False # Data has been offloaded
                self._service_request_sent = False # Reset service request flag
                
                if self.status != SensorStatus.SERVICED:
                    self.status = SensorStatus.SERVICED
                    _update_sensor_status_file(self.tsp_id, self.status.value, self.provider.current_time(), self.provider_id)

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        if self.deactivated: return
        self.my_position = telemetry.current_position
        if not self._position_initialized:
            self._position_initialized = True
            if self._has_data_to_send:
                self._request_service()

    def finish(self) -> None:
        # Ensure the final status is recorded.
        # This is a safeguard, as status changes should be recorded live.
        _update_sensor_status_file(self.tsp_id, self.status.value, self.provider.current_time(), self.provider_id)
        self._log.info(f"Sensor (TSP ID: {self.tsp_id}, Sim ID: {self.provider_id}) finished with status {self.status.name}.")

    def get_total_generated_packets(self) -> int:
        return self.total_packets_generated

class SimpleUAVProtocol(IProtocol):
    _log: logging.Logger
    provider_id: int
    
    mission_queue: List[Position3D]
    serviced_sensor_tsp_ids: set
    pending_service_requests: Dict[int, Position3D]
    current_destination: Optional[Position3D] = None

    def __init__(self):
        super().__init__()
        self._configured = False

    def configure(self,
                 tsp_instances: List[List[int]],
                 sensor_coords_map_tsp_id_key: Dict[int, Position3D],
                 flight_altitude: float,
                 ground_station_sim_id: int,
                 ground_station_physical_pos: Position3D,
                 area_bounds: TypingTuple[float, float],
                 initial_energy: float = 50000.0,
                 energy_consumption_rate: float = 1.0,
                 # Deprecated parameters, kept for compatibility if old main.py calls it
                 replan_threshold: int = 0,
                 num_sensors_to_activate: int = 0,
                 num_sensors_to_standby: int = 0,
                 algorithm: str = "LKH",
                 model_path: Optional[str] = None):
        self.tsp_instances = tsp_instances
        self.sensor_coords_map_tsp_id_key = sensor_coords_map_tsp_id_key
        self.flight_altitude = flight_altitude
        self.ground_station_sim_id = ground_station_sim_id
        self.ground_station_physical_pos = ground_station_physical_pos
        self.area_bounds = area_bounds
        self.initial_energy = initial_energy
        self.energy_consumption_rate = energy_consumption_rate
        self.algorithm = algorithm
        self.model_path = model_path
        self.model = None # To be loaded in initialize if needed
        self._configured = True

    def set_simulation(self, simulation: 'Simulation'): # Used TYPE_CHECKING for Simulation
        self._simulation = simulation

    def initialize(self) -> None:
        if not self._configured:
            raise RuntimeError("SimpleUAVProtocol not configured before initialization.")

        self.provider_id = self.provider.get_id()
        self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{self.provider_id}")

        self.buffered_packet_batches: List[PacketBatch] = []
        self.current_position: Position3D = (*self.ground_station_physical_pos[:2], self.flight_altitude)
        self.trajectory: List[TypingTuple[float, Position3D]] = [(self.provider.current_time(), self.current_position)]
        self._position_initialized = True
        
        self.serviced_sensor_tsp_ids = set()
        self.pending_service_requests = {}
        self.current_destination = None
        
        self.is_idle = True
        self.data_dump_initiated = False
        
        self.current_energy = self.initial_energy
        self.energy_log: List[TypingTuple[float, float]] = [(self.provider.current_time(), self.current_energy)]
        
        self.path_length = 0.0
        self.last_pos_for_dist_calc = self.current_position
        
        self.replan_time = None
        self.dynamically_deactivated_sensors_log = []

        # New state for multi-stage missions
        self.mission_queue: List[Position3D] = []
        self.current_instance_index: int = 0
        self.current_stage_entry_node_pos: Optional[Position3D] = None
        self.is_mission_complete = False
        
        self.computation_times = []

        if self.algorithm == "ATTENTION":
            if self.model_path is None:
                raise ValueError("Model path must be provided for ATTENTION algorithm.")
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._log.info(f"Using device: {self.device}")
            # Use the new centralized model loading function
            self.model, _ = load_attention_model(self.model_path, self.device)

        self._start_mission()
        
        self.provider.schedule_timer("heartbeat", self.provider.current_time() + 1)

    def _start_mission(self):
        self._log.info("Starting multi-stage mission.")
        self._plan_and_execute_stage()

    def _send_heartbeat(self) -> None:
        if self.current_energy > 0:
            self.current_energy -= self.energy_consumption_rate
            if self.current_energy < 0:
                self.current_energy = 0
                self._log.warning("UAV OUT OF ENERGY!")
        
        self.energy_log.append((self.provider.current_time(), self.current_energy))
        
        msg: SimpleMessage = {'sender_type': SimpleSender.UAV.value, 'sender': self.provider_id, 'uav_position': self.current_position}
        self.provider.send_communication_command(BroadcastMessageCommand(json.dumps(msg)))
        
        self.provider.schedule_timer("heartbeat", self.provider.current_time() + 1)

    def handle_timer(self, timer: str) -> None:
        if timer == "heartbeat":
            self._send_heartbeat()
        elif timer == "process_queue":
            self._process_mission_queue()
        elif timer == "plan_next_stage":
            self._plan_and_execute_stage()

    def handle_packet(self, message_json: str) -> None:
        msg: SimpleMessage = json.loads(message_json)

        if msg['sender_type'] == SimpleSender.SENSOR.value and msg.get('packet_batches'):
            batches = msg['packet_batches']
            self.buffered_packet_batches.extend(batches)
            sensor_tsp_id = msg.get('sensor_tsp_id')
            if sensor_tsp_id is not None:
                self.serviced_sensor_tsp_ids.add(sensor_tsp_id)

        elif msg['sender_type'] == SimpleSender.GROUND_STATION.value and msg.get('is_ack'):
            self._log.info("ACK received from ground station. Mission complete.")
            if msg.get('simulation_should_end') and self._simulation:
                self._log.info("UAV: Event loop clear requested by ACK message.")
                self._simulation._event_loop.clear()
                self._log.info("UAV: Event loop clear command executed.")

    def _solve_tsp_with_lkh(self, locations_to_visit: List[Position3D]) -> List[Position3D]:
        if len(locations_to_visit) <= 1:
            return []

        lkh_path = os.path.abspath("showcases/Simulating_a_data_collection_scenario/LKH-3.exe")
        if not os.path.exists(lkh_path):
            self._log.error(f"LKH executable not found at {lkh_path}")
            return locations_to_visit[1:]

        with tempfile.TemporaryDirectory() as tmpdir:
            problem_file = os.path.join(tmpdir, "problem.tsp")
            par_file = os.path.join(tmpdir, "problem.par")
            tour_file = os.path.join(tmpdir, "solution.tour")

            write_tsp_file(problem_file, locations_to_visit, name="UAV_TSP", depot_node=1)

            par_content = f"""PROBLEM_FILE = {problem_file}
TOUR_FILE = {tour_file}
MOVE_TYPE = 5
PATCHING_C = 3
PATCHING_A = 2
RUNS = 1
"""
            with open(par_file, 'w') as f:
                f.write(par_content)

            try:
                start_time = time.time()
                subprocess.run([lkh_path, par_file], check=True, cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                end_time = time.time()
                self.computation_times.append((end_time - start_time) * 1000)
                
                with open(tour_file, 'r') as f:
                    in_tour_section = False
                    tour_indices = []
                    for line in f:
                        line = line.strip()
                        if line == "TOUR_SECTION":
                            in_tour_section = True
                            continue
                        if line == "-1" or line == "EOF":
                            break
                        if in_tour_section:
                            node_id = int(line)
                            if node_id != 1:
                                tour_indices.append(node_id)
                
                return [locations_to_visit[i - 1] for i in tour_indices]

            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                self._log.error(f"LKH solver failed: {e}")
                return locations_to_visit[1:]

    def _find_closest_node(self, target_pos: Position3D, node_tsp_ids: List[int]) -> Optional[int]:
        """Finds the TSP ID of the closest node in a list to a target position."""
        closest_node_id = None
        min_dist = float('inf')

        for tsp_id in node_tsp_ids:
            node_pos = self.sensor_coords_map_tsp_id_key.get(tsp_id)
            if node_pos:
                dist = np.linalg.norm(np.array(target_pos[:2]) - np.array(node_pos[:2]))
                if dist < min_dist:
                    min_dist = dist
                    closest_node_id = tsp_id
        return closest_node_id

    def _activate_sensors_for_instance(self, tsp_ids_to_activate: List[int]):
        """Broadcasts a message to activate a specific list of sensors."""
        self._log.info(f"Broadcasting activation command for {len(tsp_ids_to_activate)} sensors for instance {self.current_instance_index + 1}.")
        activation_msg: SimpleMessage = {
            'sender_type': SimpleSender.UAV.value,
            'sender': self.provider_id,
            'sensors_to_activate': tsp_ids_to_activate,
            'uav_position': self.current_position,
            'packet_batches': None,
            'packet_count': None,
            'is_ack': None,
            'is_data_dump': None,
            'simulation_should_end': None,
            'is_service_request': None,
            'sensor_position': None,
            'sensor_tsp_id': None
        }
        self.provider.send_communication_command(BroadcastMessageCommand(json.dumps(activation_msg)))

    def _plan_and_execute_stage(self):
        if self.current_instance_index >= len(self.tsp_instances):
            self._log.info("All TSP instances processed. Returning to ground station.")
            self.mission_queue.append(self.ground_station_physical_pos)
            self.is_mission_complete = True
            self.provider.schedule_timer("process_queue", self.provider.current_time() + 0.1)
            return

        self._log.info(f"--- Starting Stage {self.current_instance_index + 1} ---")
        
        current_instance_tsp_ids = self.tsp_instances[self.current_instance_index]
        
        self._activate_sensors_for_instance(current_instance_tsp_ids)

        if self.current_stage_entry_node_pos is None:
            entry_point_for_tsp = self.current_position
        else:
            entry_point_for_tsp = self.current_stage_entry_node_pos

        start_node_tsp_id = self._find_closest_node(entry_point_for_tsp, current_instance_tsp_ids)
        if start_node_tsp_id is None:
            self._log.error(f"Could not find a starting node for instance {self.current_instance_index + 1}. Skipping.")
            self.current_instance_index += 1
            self.provider.schedule_timer("plan_next_stage", self.provider.current_time() + 0.1)
            return

        start_node_pos = self.sensor_coords_map_tsp_id_key[start_node_tsp_id]
        
        locations_for_tsp = [start_node_pos] + [self.sensor_coords_map_tsp_id_key[sid] for sid in current_instance_tsp_ids if sid != start_node_tsp_id]

        self._log.info(f"Solving TSP for stage {self.current_instance_index + 1} with {len(locations_for_tsp)} nodes using algorithm: {self.algorithm}...")
        if self.algorithm == "ATTENTION":
            # Log the exact data being sent for inference
            log_filename = f"showcases/Simulating_a_data_collection_scenario/inference_input_protocol_stage_{self.current_instance_index + 1}.json"
            with open(log_filename, 'w') as f:
                json.dump(locations_for_tsp, f, indent=4)
            self._log.info(f"Saved inference input data for stage {self.current_instance_index + 1} to {log_filename}")

            # Use the new centralized TSP solving function
            tour, cost, computation_time = solve_tsp_with_attention(self.model, locations_for_tsp, self.device, self.area_bounds)
            self.computation_times.append(computation_time)
        else: # Default to LKH
            tour = self._solve_tsp_with_lkh(locations_for_tsp)
        
        tour_file = f"showcases/Simulating_a_data_collection_scenario/tour_stage_{self.current_instance_index + 1}.csv"
        with open(tour_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z'])
            for (x, y, z) in tour:
                writer.writerow([x, y, z])
        
        self.mission_queue = tour
        
        if tour:
            self.current_stage_entry_node_pos = tour[-1]
        else:
            self.current_stage_entry_node_pos = start_node_pos
        
        self.current_instance_index += 1
        self.provider.schedule_timer("process_queue", self.provider.current_time() + 0.1)

    def _process_mission_queue(self):
        if self.mission_queue:
            self.current_destination = self.mission_queue.pop(0)
            self.provider.send_mobility_command(GotoCoordsMobilityCommand(*self.current_destination))
            self.is_idle = False
        else:
            self.is_idle = True
            if not self.is_mission_complete:
                self._plan_and_execute_stage()
            else:
                self._log.info("Final mission queue empty and mission marked complete. Idling at GS.")

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.current_position = telemetry.current_position
        self.trajectory.append((self.provider.current_time(), self.current_position))
        
        distance_traveled = np.linalg.norm(np.array(self.current_position) - np.array(self.last_pos_for_dist_calc))
        self.path_length += distance_traveled
        self.last_pos_for_dist_calc = self.current_position

        if self.current_destination and np.linalg.norm(np.array(self.current_position) - np.array(self.current_destination)) < 1.0:
            self.current_destination = None
            if self.mission_queue:
                self._process_mission_queue()
            elif self.is_mission_complete and not self.data_dump_initiated:
                self._log.info("Arrived at ground station. Dumping data.")
                self.data_dump_initiated = True
                dump_msg: SimpleMessage = {
                    'sender_type': SimpleSender.UAV.value, 'sender': self.provider_id,
                    'is_data_dump': True, 'packet_batches': self.buffered_packet_batches,
                    'uav_position': self.current_position, 'packet_count': None, 'is_ack': None,
                    'is_service_request': None, 'sensor_position': None, 'sensor_tsp_id': None,
                    'simulation_should_end': None
                }
                self.provider.send_communication_command(SendMessageCommand(json.dumps(dump_msg), self.ground_station_sim_id))
                self.buffered_packet_batches = []
            elif not self.is_mission_complete:
                self._plan_and_execute_stage()

    def finish(self) -> None:
        self._log.info(f"UAV {self.provider_id}: Starting finish() method.")
        
        replan_trajectory = None

        with open(f"showcases/Simulating_a_data_collection_scenario/uav_{self.provider_id}_initial_trajectory.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'z'])
            trajectory_to_write = replan_trajectory if replan_trajectory else self.trajectory
            for t, (x, y, z) in trajectory_to_write:
                writer.writerow([t, x, y, z])

        if replan_trajectory:
            with open(f"showcases/Simulating_a_data_collection_scenario/uav_{self.provider_id}_replan_trajectory.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'x', 'y', 'z'])
                for t, (x, y, z) in replan_trajectory:
                    writer.writerow([t, x, y, z])

        with open(f"showcases/Simulating_a_data_collection_scenario/uav_{self.provider_id}_energy.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'energy'])
            writer.writerows(self.energy_log)

        if self.dynamically_deactivated_sensors_log:
            with open(f"showcases/Simulating_a_data_collection_scenario/uav_{self.provider_id}_dynamically_deactivated_sensors.json", 'w') as f:
                json.dump(self.dynamically_deactivated_sensors_log, f, indent=4)

        self._log.info(f"UAV {self.provider_id}: Finish method completed. Trajectory and energy logs saved.")

    def get_path_length(self) -> float:
        return self.path_length

    def get_energy_consumed(self) -> float:
        return self.initial_energy - self.current_energy
        
    def get_total_computation_time(self) -> float:
        return sum(self.computation_times)

class SimpleGroundStationProtocol(IProtocol):
    _log: logging.Logger
    provider_id: int
    
    def __init__(self):
        super().__init__()
        self.received_packet_batches: List[PacketBatch] = []
        self.total_packets_received = 0

    def initialize(self) -> None:
        self.provider_id = self.provider.get_id()
        self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{self.provider_id}")
        self.received_packet_batches = []
        self.total_packets_received = 0
        self._log.info("Ground station initialized.")

    def handle_packet(self, message_json: str) -> None:
        msg: SimpleMessage = json.loads(message_json)
        if msg.get('is_data_dump'):
            batches = msg.get('packet_batches', [])
            for batch in batches:
                batch['reception_time_gs'] = self.provider.current_time()
                self.total_packets_received += batch['count']
            self.received_packet_batches.extend(batches)
            self._log.info(f"Received data dump with {len(batches)} batches. Total packets so far: {self.total_packets_received}.")
            
            ack_msg: SimpleMessage = {
                'sender_type': SimpleSender.GROUND_STATION.value, 'sender': self.provider_id,
                'is_ack': True, 'simulation_should_end': True,
                'uav_position': None, 'packet_batches': None, 'packet_count': None,
                'is_data_dump': None, 'is_service_request': None, 'sensor_position': None,
                'sensor_tsp_id': None
            }
            self.provider.send_communication_command(SendMessageCommand(json.dumps(ack_msg), msg['sender']))

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def handle_timer(self, timer: str) -> None:
        pass

    def finish(self) -> None:
        self._log.info("Ground station finishing.")
        output_file = f"showcases/Simulating_a_data_collection_scenario/gs_{self.provider_id}_received_batches.json"
        with open(output_file, 'w') as f:
            json.dump(self.received_packet_batches, f, indent=4)
        self._log.info(f"Saved {len(self.received_packet_batches)} batches to {output_file}")

    def get_all_received_batches(self) -> List[PacketBatch]:
        return self.received_packet_batches

    def get_total_packets_received_count(self) -> int:
        return self.total_packets_received

    def get_current_time(self) -> float:
        return self.provider.current_time()
