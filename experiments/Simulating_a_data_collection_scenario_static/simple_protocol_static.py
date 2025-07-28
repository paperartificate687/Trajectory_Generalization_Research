import enum
import json
import logging
import csv  # Import the csv module
import os
from typing import TypedDict, Dict, List, Tuple as TypingTuple, Optional, Union # Renamed Tuple to avoid conflict
import re # For parsing TSP files
import statistics # For calculating mean

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand # Import GotoCoordsMobilityCommand
from gradysim.protocol.plugin.mission_mobility import MissionMobilityPlugin, MissionMobilityConfiguration, LoopMission
from gradysim.protocol.position import Position # Import Position


# Type alias for position for clarity
Position3D = TypingTuple[float, float, float]

def load_sensor_coordinates_from_tsp(tsp_file_path: str) -> Dict[int, Position3D]:
    """
    Parses a TSP file to extract node coordinates.
    Assumes EUC_2D format where coordinates are (id, x, y).
    Returns a dictionary mapping node ID (int) to (x, y, z) tuples, with z defaulting to 0.0.
    """
    coordinates_map: Dict[int, Position3D] = {}
    in_coord_section = False
    try:
        with open(tsp_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == "NODE_COORD_SECTION":
                    in_coord_section = True
                    continue
                if line == "EOF":
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
                            logging.warning(f"Could not parse coordinate line in {tsp_file_path}: {line}")
                    else:
                        logging.warning(f"Skipping malformed coordinate line in {tsp_file_path}: {line}")
    except FileNotFoundError:
        logging.error(f"Sensor locations TSP file not found: {tsp_file_path}")
    return coordinates_map

def load_tsp_tour_sequence(tour_file_path: str) -> List[int]:
    """
    Parses a TSP tour file to extract the sequence of node IDs.
    """
    tour_sequence: List[int] = []
    in_tour_section = False
    try:
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
                        logging.warning(f"Could not parse tour node ID in {tour_file_path}: {line}")
    except FileNotFoundError:
        logging.error(f"TSP tour file not found: {tour_file_path}")
    return tour_sequence


class SimpleSender(enum.Enum):
    SENSOR = 0
    UAV = 1
    GROUND_STATION = 2

# Define a structure for a batch of packets for latency calculation
class PacketBatch(TypedDict):
    sensor_id: int
    gen_time: float
    count: int

class SimpleMessage(TypedDict):
    sender_type: int
    sender: int
    uav_position: Optional[Position3D] 
    
    packet_batches: Optional[List[PacketBatch]] 
    packet_count: Optional[int] 

    is_ack: Optional[bool]
    is_data_dump: Optional[bool]
    simulation_should_end: Optional[bool] # True if the simulation should be gracefully ended



def report_message(message: SimpleMessage) -> str:
    base = f"sender_type={SimpleSender(message['sender_type']).name}, sender={message['sender']}"
    if message.get('uav_position'):
        base += f", uav_pos={message['uav_position']}"
    if message.get('packet_count') is not None:
        base += f", pkt_count={message['packet_count']}"
    if message.get('packet_batches'):
        base += f", batches_len={len(message['packet_batches'])}"
    if message.get('is_ack'):
        base += ", is_ack=True"
    if message.get('is_data_dump'):
        base += ", is_data_dump=True"
    if message.get('simulation_should_end'):
        base += ", simulation_should_end=True"
    return f"Msg RX: {base}"


class SimpleSensorProtocol(IProtocol):
    _log: logging.Logger
    packet_count: int
    total_packets_generated: int
    current_batch_gen_time: float 
    my_position: Position = (0.0, 0.0, 0.0)
    _position_initialized: bool = False

    XY_THRESHOLD_ABOVE = 15.0
    Z_MIN_DIFFERENCE_ABOVE = 1.0

    def __init__(self):
        pass

    def initialize(self) -> None:
        self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{self.provider.get_id()}")
        self.packet_count = 0
        self.total_packets_generated = 0
        self.current_batch_gen_time = 0.0
        self._is_active = True
        self._generate_packet()

    def _generate_packet(self) -> None:
        if not self._is_active:
            return

        if self.packet_count == 0:
            self.current_batch_gen_time = self.provider.current_time()
        
        self.packet_count += 1
        self.total_packets_generated += 1
        self.provider.schedule_timer("generate", self.provider.current_time() + 1)

    def handle_timer(self, timer: str) -> None:
        if timer == "generate":
            self._generate_packet()

    def handle_packet(self, message_json: str) -> None:
        simple_message: SimpleMessage = json.loads(message_json)
        self._log.debug(f"Handling packet: {report_message(simple_message)}")

        if simple_message.get('simulation_should_end'):
            self._log.info("Received simulation end signal. Stopping packet generation.")
            self._is_active = False
            try:
                self.provider.cancel_timer("generate")
            except ValueError:
                # This is fine, means the timer was not active, so no more packets would be generated anyway.
                pass
            return

        if simple_message['sender_type'] == SimpleSender.UAV.value:
            if 'uav_position' not in simple_message or simple_message['uav_position'] is None:
                self._log.error("Received UAV message without uav_position field.")
                return

            uav_pos = simple_message['uav_position']
            is_directly_above = (
                self._position_initialized and
                abs(uav_pos[0] - self.my_position[0]) < self.XY_THRESHOLD_ABOVE and
                abs(uav_pos[1] - self.my_position[1]) < self.XY_THRESHOLD_ABOVE and
                uav_pos[2] > self.my_position[2] and
                (uav_pos[2] - self.my_position[2]) > self.Z_MIN_DIFFERENCE_ABOVE
            )

            if is_directly_above and self.packet_count > 0:
                self._log.info(f"UAV {simple_message['sender']} is above. Sending {self.packet_count} packets.")
                
                batch_to_send: PacketBatch = {
                    'sensor_id': self.provider.get_id(),
                    'gen_time': self.current_batch_gen_time,
                    'count': self.packet_count
                }
                
                response: SimpleMessage = {
                    'sender_type': SimpleSender.SENSOR.value,
                    'sender': self.provider.get_id(),
                    'packet_batches': [batch_to_send], 
                    'packet_count': self.packet_count, 
                    'uav_position': None, 
                    'is_ack': None,
                    'is_data_dump': None,
                    'simulation_should_end': None
                }
                command = SendMessageCommand(json.dumps(response), simple_message['sender'])
                self.provider.send_communication_command(command)

                self._log.info(f"Sent batch of {self.packet_count} packets (gen_time {self.current_batch_gen_time:.2f}) to UAV {simple_message['sender']}")
                self.packet_count = 0
                self.current_batch_gen_time = 0.0 

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        if not self._position_initialized:
            self.my_position = telemetry.current_position
            self._position_initialized = True
            self._log.info(f"Sensor {self.provider.get_id()} position initialized to: {self.my_position}")

    def finish(self) -> None:
        self._log.info(f"Final unsent packet count: {self.packet_count}")

    def get_total_packets_generated(self) -> int:
        return self.total_packets_generated


class SimpleUAVProtocol(IProtocol):
    _log: logging.Logger

    packet_count: int # Total individual packets collected by UAV
    buffered_packet_batches: List[PacketBatch] # Stores batches from sensors

    current_position: Position = (0.0, 0.0, 0.0)
    trajectory: list
    mission_start_time: float = 0.0
    mission_end_time: float = 0.0
    main_mission_started: bool = False
    landing_initiated: bool = False
    landing_complete: bool = False
    data_dump_initiated: bool = False

    initial_energy: float = 20000.0  
    current_energy: float
    energy_consumption_rate: float = 1.0  
    energy_log: list 

    _mission: MissionMobilityPlugin
    _simulation: Optional['IProtocol'] = None

    GROUND_STATION_POS: Position3D = (0.0, 0.0, 0.0)
    ground_station_id: int

    def __init__(self, mission_path: List[Position3D], ground_station_id: int, ground_station_pos: Position3D, output_dir: str = "."):
        self.mission_path = mission_path
        self.ground_station_id = ground_station_id
        self.GROUND_STATION_POS = ground_station_pos
        self.output_dir = output_dir

    def set_simulation(self, simulation: 'IProtocol'):
        self._simulation = simulation

    def initialize(self) -> None:
        self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{self.provider.get_id()}")
        self.packet_count = 0
        self.buffered_packet_batches = []
        self.trajectory = []
        # Manually add the starting position to the trajectory at initialization
        if self.mission_path:
            self.trajectory.append(self.mission_path[0])
            
        self.mission_start_time = 0.0
        self.mission_end_time = 0.0
        self.main_mission_started = False
        self.landing_initiated = False
        self.landing_complete = False
        self.data_dump_initiated = False

        self.current_energy = self.initial_energy
        self.energy_log = []
        current_sim_time = self.provider.current_time() if self.provider else 0.0
        self.energy_log.append((current_sim_time, self.current_energy))
        self._log.info(f"UAV {self.provider.get_id()} initialized with {self.current_energy} energy.")

        mission_config = MissionMobilityConfiguration(speed=25.0, loop_mission=LoopMission.NO)
        self._mission = MissionMobilityPlugin(self, mission_config)

        if self.mission_path and len(self.mission_path) > 0:
            self._mission.start_mission(self.mission_path)
            self.main_mission_started = True
            self.mission_start_time = self.provider.current_time()
            self.current_position = self.mission_path[0]
            self._log.info(f"UAV {self.provider.get_id()} starting mission at {self.mission_start_time:.2f}s. First waypoint: {self.current_position}")
        else:
            self._log.error(f"UAV {self.provider.get_id()}: Mission path empty. Stationary.")
            self.current_position = (0.0,0.0,0.0)

        self._send_heartbeat()

    def _send_heartbeat(self) -> None:
        if self.current_energy > 0:
            self.current_energy -= self.energy_consumption_rate
            if self.current_energy < 0: self.current_energy = 0
            self.energy_log.append((self.provider.current_time(), self.current_energy))
        else:
            self._log.warning(f"UAV {self.provider.get_id()} out of energy!")

        self._log.debug(f"Heartbeat: pos={self.current_position}, pkts={self.packet_count}, energy={self.current_energy:.2f}")
        heartbeat_msg: SimpleMessage = {
            'sender_type': SimpleSender.UAV.value,
            'sender': self.provider.get_id(),
            'uav_position': self.current_position,
            'packet_count': self.packet_count, 
            'packet_batches': None, 
            'is_ack': None,
            'is_data_dump': None,
            'simulation_should_end': None
        }
        command = BroadcastMessageCommand(json.dumps(heartbeat_msg))
        self.provider.send_communication_command(command)
        self.provider.schedule_timer("heartbeat", self.provider.current_time() + 1)

    def handle_timer(self, timer: str) -> None:
        if timer == "heartbeat":
            self._send_heartbeat()

    def handle_packet(self, message_json: str) -> None:
        simple_message: SimpleMessage = json.loads(message_json)
        self._log.debug(f"UAV {self.provider.get_id()} received packet: {report_message(simple_message)}")

        if simple_message['sender_type'] == SimpleSender.SENSOR.value:
            if simple_message.get('packet_batches'):
                for batch in simple_message['packet_batches']: 
                    self.buffered_packet_batches.append(batch)
                    self.packet_count += batch['count'] 
                self._log.info(f"UAV {self.provider.get_id()} collected batch(es) from sensor {simple_message['sender']}. Total packets: {self.packet_count}. Batches buffered: {len(self.buffered_packet_batches)}")
            elif simple_message.get('packet_count') is not None: 
                 current_sensor_pkt_count = simple_message['packet_count']
                 self.packet_count += current_sensor_pkt_count
                 dummy_batch: PacketBatch = {
                     'sensor_id': simple_message['sender'],
                     'gen_time': self.provider.current_time(), 
                     'count': current_sensor_pkt_count
                 }
                 self.buffered_packet_batches.append(dummy_batch)
                 self._log.warning(f"UAV {self.provider.get_id()} received {current_sensor_pkt_count} packets (NO BATCH INFO) from SENSOR {simple_message['sender']}. Latency for these will be inaccurate. Total packets: {self.packet_count}")
        
        elif simple_message['sender_type'] == SimpleSender.GROUND_STATION.value:
            if simple_message.get('is_ack'):
                self._log.info(f"UAV {self.provider.get_id()}: ACK received from GS {simple_message['sender']} for data dump. Clearing UAV buffers.")
                self.buffered_packet_batches = [] 
                self.packet_count = 0 
                self.data_dump_initiated = False 
            else:
                self._log.info(f"UAV {self.provider.get_id()}: Msg from GS {simple_message['sender']} (not an ACK). No action taken.")

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.current_position = telemetry.current_position 
        self.trajectory.append(self.current_position) 
        
        if self.main_mission_started and self._mission.is_idle and not self.landing_initiated and not self.data_dump_initiated:
            self.data_dump_initiated = True 
            
            self._log.info(f"UAV {self.provider.get_id()} [Tel-Dump]: Main mission finished. At position {self.current_position}. Packets collected: {self.packet_count}. Buffered batches: {len(self.buffered_packet_batches)}")
            
            if self.buffered_packet_batches:
                self._log.info(f"UAV {self.provider.get_id()} [Tel-Dump]: Preparing to dump {len(self.buffered_packet_batches)} batches ({self.packet_count} packets) to GS {self.ground_station_id} from {self.current_position}")
                data_dump_msg: SimpleMessage = {
                    'sender_type': SimpleSender.UAV.value,
                    'sender': self.provider.get_id(),
                    'packet_batches': self.buffered_packet_batches,
                    'packet_count': self.packet_count,
                    'uav_position': self.current_position,
                    'is_data_dump': True,
                    'is_ack': None,
                    'simulation_should_end': None
                }
                dump_command = SendMessageCommand(json.dumps(data_dump_msg), self.ground_station_id)
                self.provider.send_communication_command(dump_command)
                self._log.info(f"UAV {self.provider.get_id()} [Tel-Dump]: Data dump message SENT to Ground Station {self.ground_station_id}.")
            else:
                self._log.info(f"UAV {self.provider.get_id()} [Tel-Dump]: No buffered packet batches to dump to Ground Station.")

            self._log.info(f"UAV {self.provider.get_id()} [Tel-Dump]: Initiating landing to {self.GROUND_STATION_POS}.")
            self.landing_initiated = True
            land_command = GotoCoordsMobilityCommand(*self.GROUND_STATION_POS)
            self.provider.send_mobility_command(land_command)
        elif self.landing_initiated and not self.landing_complete:  # Corrected indentation for this block
            # Check if UAV has landed
            current_pos = telemetry.current_position
            gs_pos = self.GROUND_STATION_POS
            # Using a small threshold for landing confirmation
            if (abs(current_pos[0] - gs_pos[0]) < 0.1 and
                abs(current_pos[1] - gs_pos[1]) < 0.1 and
                abs(current_pos[2] - gs_pos[2]) < 0.1):
                self.landing_complete = True
                self.mission_end_time = self.provider.current_time()
                self._log.info(f"UAV {self.provider.get_id()} has landed at {current_pos} at {self.mission_end_time:.2f}s. Broadcasting end signal and stopping heartbeat.")
                
                # Broadcast a message to all nodes that the simulation is ending
                end_msg: SimpleMessage = {
                    'sender_type': SimpleSender.UAV.value,
                    'sender': self.provider.get_id(),
                    'uav_position': self.current_position,
                    'packet_batches': None,
                    'packet_count': None,
                    'is_ack': None,
                    'is_data_dump': None,
                    'simulation_should_end': True
                }
                command = BroadcastMessageCommand(json.dumps(end_msg))
                self.provider.send_communication_command(command)

                self.provider.cancel_timer("heartbeat")

    def get_energy_consumed(self) -> float:
        if not self.energy_log:
            return 0.0
        return self.initial_energy - self.current_energy

    def get_mission_duration(self) -> float:
        if self.mission_start_time > 0 and self.mission_end_time > self.mission_start_time:
            return self.mission_end_time - self.mission_start_time
        elif self.main_mission_started:
            return self.provider.current_time() - self.mission_start_time
        return 0.0

    def finish(self) -> None:
        self._log.info(f"UAV {self.provider.get_id()} final packet count (unsent/buffered): {self.packet_count}")
        
        uav_id = self.provider.get_id()

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        trajectory_filename = os.path.join(self.output_dir, f"uav_{uav_id}_trajectory.csv")
        try:
            with open(trajectory_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['x', 'y', 'z'])
                for position in self.trajectory:
                    writer.writerow(position)
            self._log.info(f"UAV {uav_id} trajectory saved to {trajectory_filename}")
        except Exception as e:
            self._log.error(f"Error saving trajectory for UAV {uav_id} to CSV: {e}")

        energy_filename = os.path.join(self.output_dir, f"uav_{uav_id}_energy.csv")
        try:
            with open(energy_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'energy_level'])
                for timestamp, energy_level in self.energy_log:
                    writer.writerow([timestamp, energy_level])
            self._log.info(f"UAV {uav_id} energy log saved to {energy_filename}")
        except Exception as e:
            self._log.error(f"Error saving energy log for UAV {uav_id} to CSV: {e}")


class SimpleGroundStationProtocol(IProtocol):
    _log: logging.Logger
    total_packets_globally_received: int 
    latencies_collected: List[float] 
    my_position: Position = (0.0, 0.0, 0.0)
    _position_initialized: bool = False

    XYZ_THRESHOLD_AT_GS = 15.0 

    def __init__(self):
        pass

    def initialize(self) -> None:
        self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{self.provider.get_id()}")
        self.total_packets_globally_received = 0
        self.latencies_collected = []

    def handle_timer(self, timer: str) -> None:
        pass 

    def handle_packet(self, message_json: str) -> None:
        simple_message: SimpleMessage = json.loads(message_json)
        self._log.debug(f"Handling packet: {report_message(simple_message)}")

        if simple_message['sender_type'] == SimpleSender.UAV.value and simple_message.get('is_data_dump'):
            self._log.info(f"Received data dump from UAV {simple_message['sender']}.")
            
            reception_time = self.provider.current_time()
            batches = simple_message.get('packet_batches', [])
            
            num_packets_in_dump = 0
            if batches:
                for batch in batches:
                    latency = reception_time - batch['gen_time']
                    for _ in range(batch['count']):
                        self.latencies_collected.append(latency)
                    num_packets_in_dump += batch['count']
                self._log.info(f"Processed {len(batches)} batches, {num_packets_in_dump} packets from UAV {simple_message['sender']}. Latencies recorded.")
            
            if simple_message.get('packet_count') is not None:
                 self.total_packets_globally_received += simple_message['packet_count']
            else: 
                self.total_packets_globally_received += num_packets_in_dump


            ack_msg: SimpleMessage = {
                'sender_type': SimpleSender.GROUND_STATION.value,
                'sender': self.provider.get_id(),
                'is_ack': True,
                'packet_batches': None, 
                'packet_count': None,   
                'uav_position': None,
                'is_data_dump': None,
                'simulation_should_end': None
            }
            command = SendMessageCommand(json.dumps(ack_msg), simple_message['sender'])
            self.provider.send_communication_command(command)
            self._log.info(f"Sent ACK for data dump to UAV {simple_message['sender']}.")
        
        elif simple_message['sender_type'] == SimpleSender.UAV.value:
             self._log.debug(f"Received non-data-dump message (e.g. heartbeat) from UAV {simple_message['sender']}. No action taken for latency.")


    def handle_telemetry(self, telemetry: Telemetry) -> None:
        if not self._position_initialized:
            self.my_position = telemetry.current_position
            self._position_initialized = True
            self._log.info(f"Ground Station {self.provider.get_id()} position initialized to: {self.my_position}")

    def get_total_packets_received(self) -> int:
        return self.total_packets_globally_received

    def get_average_latency(self) -> float:
        if not self.latencies_collected:
            return 0.0
        return statistics.mean(self.latencies_collected)

    def finish(self) -> None:
        self._log.info(f"Ground Station {self.provider.get_id()} finished.")
        self._log.info(f"Total packets declared as received by GS (from UAV dumps): {self.total_packets_globally_received}")
        if self.latencies_collected:
            self._log.info(f"Average latency: {self.get_average_latency():.2f}s")
        
        if self.latencies_collected:
            average_latency = statistics.mean(self.latencies_collected)
            min_latency = min(self.latencies_collected)
            max_latency = max(self.latencies_collected)
            median_latency = statistics.median(self.latencies_collected)