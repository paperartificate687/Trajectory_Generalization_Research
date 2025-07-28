import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import glob
import os
import re # For parsing TSP file
from typing import Dict, Tuple as TypingTuple

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
                            print(f"Warning: Could not parse coordinate line in {tsp_file_path}: {line}")
                    else:
                        print(f"Warning: Skipping malformed coordinate line in {tsp_file_path}: {line}")
    except FileNotFoundError:
        print(f"Error: Sensor locations TSP file not found: {tsp_file_path}")
    return coordinates_map


def plot_uav_trajectories(scenario="Forest", num_sensors=100):
    fig = plt.figure(figsize=(12, 10)) # Slightly larger figure
    ax = fig.add_subplot(111, projection='3d')

    # Define paths and fixed positions
    script_dir = os.path.dirname(__file__)
    # sensor_locations_tsp_file = os.path.join(script_dir, f"sensor_locations_{scenario}_{num_sensors}.tsp")
    # Use a fixed path as requested
    sensor_locations_tsp_file = os.path.join(script_dir, "data", "sensor_locations_agriculture_150_1.tsp")
    
    # Load sensor positions from TSP file
    sensor_positions_map_all = load_sensor_coordinates_from_tsp(sensor_locations_tsp_file)
    
    ground_station_position: Position3D = (0.0, 0.0, 0.0) # Default
    ground_station_node_id = None

    if sensor_positions_map_all:
        # Attempt to use node 1 as the ground station
        if 1 in sensor_positions_map_all:
            ground_station_position = sensor_positions_map_all[1]
            ground_station_node_id = 1
            print(f"Using node 1 as Ground Station: {ground_station_position}")
        # If node 1 is not present, use the first node in the map as the ground station
        elif sensor_positions_map_all:
            first_node_id = next(iter(sensor_positions_map_all))
            ground_station_position = sensor_positions_map_all[first_node_id]
            ground_station_node_id = first_node_id
            print(f"Node 1 not found. Using the first available node ({first_node_id}) as Ground Station: {ground_station_position}")
        else:
            print("Warning: Sensor positions map is empty after loading. Cannot determine ground station from TSP.")
    else:
        print("Error: No sensor positions loaded from TSP. Ground station defaults to (0,0,0).")

    # Create a new map for other sensors, excluding the ground station
    sensor_positions_map_display = {
        node_id: pos for node_id, pos in sensor_positions_map_all.items() if node_id != ground_station_node_id
    }

    # Plot sensor positions (excluding the ground station)
    if sensor_positions_map_display:
        first_sensor = True
        for node_id, pos in sensor_positions_map_display.items():
            if first_sensor:
                ax.scatter(pos[0], pos[1], pos[2], color='blue', marker='s', s=50, label='Sensor Nodes')
                first_sensor = False
            else:
                ax.scatter(pos[0], pos[1], pos[2], color='blue', marker='s', s=50)
    else:
        # This case might occur if all nodes were considered ground stations or map was empty
        print("No other sensor positions to plot (excluding ground station).")


    # Plot ground station position
    ax.scatter(ground_station_position[0], ground_station_position[1], ground_station_position[2],
               color='black', marker='^', s=150, label=f'Ground Station (Node {ground_station_node_id if ground_station_node_id else "N/A"})')

    # Find all UAV trajectory CSV files
    csv_files = glob.glob(os.path.join(script_dir, "uav_*_trajectory.csv"))

    if not csv_files:
        print("No UAV trajectory CSV files found in the current directory.")
        print(f"Searched in: {script_dir}")
        return

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] # Colors for different UAVs

    for i, filename in enumerate(csv_files):
        x_coords = []
        y_coords = []
        z_coords = []
        
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader) # Skip header row
                if header != ['x', 'y', 'z']:
                    print(f"Warning: CSV file {filename} has unexpected header: {header}")
                    # Attempt to read assuming order is x, y, z
                
                for row in reader:
                    try:
                        x_coords.append(float(row[0]))
                        y_coords.append(float(row[1]))
                        z_coords.append(float(row[2]))
                    except (ValueError, IndexError) as e:
                        print(f"Skipping malformed row in {filename}: {row} - Error: {e}")
                        continue
            
            if x_coords and y_coords and z_coords: # Check if any data was actually read
                uav_id = filename.split('_')[1] # Extract UAV ID from filename
                ax.plot(x_coords, y_coords, z_coords, label=f'UAV {uav_id}', color=colors[i % len(colors)])
            else:
                print(f"No valid data points found in {filename}")

        except FileNotFoundError:
            print(f"Error: File not found {filename}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")


    # Determine plot limits based on sensor and trajectory data
    all_x = [ground_station_position[0]]
    all_y = [ground_station_position[1]]
    all_z = [ground_station_position[2]]

    if sensor_positions_map_all:
        for pos in sensor_positions_map_all.values():
            all_x.append(pos[0])
            all_y.append(pos[1])
            all_z.append(pos[2])
    
    # (Trajectory points will be added to all_x, all_y, all_z inside the loop below if needed for auto-scaling)
    # For now, let's set some reasonable defaults that should cover the 0-90 grid and UAV altitude
    min_x, max_x = -10, 100
    min_y, max_y = -10, 100
    min_z, max_z = -5, 50 # UAV altitude is 20, GS is 0

    if all_x: # if we had trajectory data to extend this
        min_x_data = min(all_x) -10
        max_x_data = max(all_x) +10
        min_y_data = min(all_y) -10
        max_y_data = max(all_y) +10
        min_z_data = min(all_z) -5
        max_z_data = max(all_z) +10
        # Update limits if data suggests wider range, but keep them reasonable
        min_x = min(min_x, min_x_data)
        max_x = max(max_x, max_x_data)
        min_y = min(min_y, min_y_data)
        max_y = max(max_y, max_y_data)
        min_z = min(min_z, min_z_data)
        max_z = max(max_z, max_z_data)


    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('UAV Trajectories and Sensor Network')
    
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])
    
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot UAV trajectories and sensor locations.")
    parser.add_argument("--scenario", type=str, default="Forest", help="Name of the scenario (e.g., Forest, Urban).")
    parser.add_argument("--num_sensors", type=int, default=100, help="Number of sensors in the scenario.")
    args = parser.parse_args()

    plot_uav_trajectories(scenario=args.scenario, num_sensors=args.num_sensors)