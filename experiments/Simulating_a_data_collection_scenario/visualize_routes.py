import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def visualize_routes():
    """
    Visualizes the routes from the first 4 tour_stage_*.csv files in the current directory,
    displaying them in a 2x2 grid of subplots.
    """
    tour_files = glob.glob("showcases/Simulating_a_data_collection_scenario/tour_stage_*.csv")
    if not tour_files:
        print("No tour files found.")
        return

    # Sort files by stage number
    tour_files.sort(key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))

    # Limit to at most 4 files
    files_to_plot = tour_files[:4]
    
    if not files_to_plot:
        print("No tour files to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, tour_file in enumerate(files_to_plot):
        stage_num = os.path.basename(tour_file).split('_')[-1].split('.')[0]
        df = pd.read_csv(tour_file)
        
        # Close the loop by appending the first point to the end
        # Use pd.concat instead of deprecated _append
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        
        ax = axes[i]
        ax.plot(df['x'], df['y'], marker='o', linestyle='-')
        ax.set_title(f"Stage {stage_num}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True)

    # Hide any unused subplots
    for j in range(len(files_to_plot), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("UAV Trajectories per Stage (First 4 Stages)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("showcases/Simulating_a_data_collection_scenario/routes_visualization_subplots.png")
    plt.show()

if __name__ == "__main__":
    visualize_routes()