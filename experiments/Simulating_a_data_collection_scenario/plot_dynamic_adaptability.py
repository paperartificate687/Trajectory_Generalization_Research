import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import numpy as np

def plot_dynamic_adaptability(csv_path, output_dir):
    """
    Reads simulation results, processes the data, and generates publication-ready plots
    to show adaptability in dynamic environments, following INFOCOM/IEEE TMC style.

    Generates two plots:
    1. Total Computation Time vs. Number of Dynamic Instances.
    2. System Utility Score vs. Number of Dynamic Instances.
    """
    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Try to read the csv file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # --- 1. Data Preprocessing and Style Configuration ---
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Filter for Instance Size == 200 and rename algorithm column for clarity
    df_filtered = df[df['Instance Size'] == 200].copy()
    df_filtered['Algorithm'] = df_filtered['Algorithm (mathcalM)'].str.strip()
    
    # Get unique algorithms
    algorithms = sorted(df_filtered['Algorithm'].unique())
    
    # --- Style Definitions (consistent with IEEE TMC) ---
    color_palette = {
        'ADAPT-GUAV': '#1f77b4', 'LKH': '#ff7f0e', 'PSO': '#2ca02c',
        'CRL-AM': '#8c564b', 'CRL-FD': '#9467bd', 'CRL-HA': '#d62728',
        'CRL-U': '#e377c2', 'Nearest Neighbor': '#7f7f7f', 'Random': '#bcbd22'
    }
    color_palette = {k: v for k, v in color_palette.items() if k in algorithms}

    marker_map = {
        'ADAPT-GUAV': 'o', 'LKH': 's', 'PSO': '^', 'CRL-AM': 'P',
        'CRL-FD': 'v', 'CRL-HA': 'D', 'CRL-U': 'X', 'Nearest Neighbor': '*', 'Random': '+'
    }
    marker_map = {k: v for k, v in marker_map.items() if k in algorithms}

    linestyle_map = {
        'ADAPT-GUAV': '-', 'LKH': '--', 'PSO': ':', 'CRL-AM': '--',
        'CRL-FD': '-', 'CRL-HA': '-.', 'CRL-U': ':', 'Nearest Neighbor': '-.', 'Random': '-'
    }
    linestyle_map = {k: v for k, v in linestyle_map.items() if k in algorithms}

    sns.set_theme(style="ticks")
    try:
        plt.rcParams.update({
            'font.family': 'Times New Roman', 'font.size': 10, 'axes.labelsize': 10,
            'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
            'lines.linewidth': 1.0, 'lines.markersize': 6,
            'axes.spines.right': True, 'axes.spines.top': True,
            'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'xtick.major.size': 4, 'ytick.major.size': 4,
            'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
            'xtick.minor.visible': True, 'ytick.minor.size': 2, 'ytick.minor.width': 0.6,
            'ytick.minor.visible': True, 'ytick.minor.size': 2, 'ytick.minor.width': 0.6,
            'axes.grid': True, 'grid.color': 'lightgray',
            'grid.linestyle': '--', 'grid.linewidth': 0.5,
        })
    except RuntimeError:
        print("Warning: Times New Roman font not found. Using default font.")

    # --- 2. Data Aggregation ---
    # Group by algorithm and number of instances to get mean values
    agg_df = df_filtered.groupby(['Algorithm', 'Instances']).mean(numeric_only=True).reset_index()

    # --- Plot 1: Total Computation Time vs. Instances ---
    plt.figure(figsize=(5.5, 3.5))
    ax1 = plt.gca()

    for algo in algorithms:
        algo_df = agg_df[agg_df['Algorithm'] == algo]
        if not algo_df.empty:
            sns.lineplot(
                data=algo_df, x='Instances', y='Total Computation Time (ms)',
                ax=ax1, label=algo, color=color_palette.get(algo),
                marker=marker_map.get(algo), linestyle=linestyle_map.get(algo)
            )
    
    ax1.set_xlabel('Number of Dynamic Instances')
    ax1.set_ylabel('Total Computation Time (ms)')
    ax1.set_yscale('log')
    ax1.set_xticks([2, 4, 6, 8, 10])
    ax1.xaxis.set_minor_locator(mticker.NullLocator())
    ax1.legend(frameon=True, edgecolor='black')
    
    plt.tight_layout()
    output_path1 = os.path.join(output_dir, 'dynamic_computation_time.png')
    plt.savefig(output_path1, dpi=300)
    plt.close()
    print(f"Saved Plot 1: Dynamic Computation Time to {output_path1}")

    # --- 3. System Utility Calculation and Plotting ---
    # Normalize Computation Time and Path Length for the weighted sum
    min_time, max_time = df_filtered['Total Computation Time (ms)'].min(), df_filtered['Total Computation Time (ms)'].max()
    min_path, max_path = df_filtered['Path Length (L)'].min(), df_filtered['Path Length (L)'].max()

    if (max_time - min_time) > 0:
        df_filtered['norm_time'] = (df_filtered['Total Computation Time (ms)'] - min_time) / (max_time - min_time)
    else:
        df_filtered['norm_time'] = 0.5

    if (max_path - min_path) > 0:
        df_filtered['norm_path'] = (df_filtered['Path Length (L)'] - min_path) / (max_path - min_path)
    else:
        df_filtered['norm_path'] = 0.5
    
    # Calculate the System Utility score (lower is better)
    w1, w2 = 0.5, 0.5
    df_filtered['weighted_utility'] = w1 * df_filtered['norm_time'] + w2 * df_filtered['norm_path']
    
    # Aggregate System Utility
    utility_df = df_filtered.groupby(['Algorithm', 'Instances'])['weighted_utility'].mean().reset_index()

    # --- Plot 2: System Utility vs. Instances ---
    plt.figure(figsize=(5.5, 3.5))
    ax2 = plt.gca()

    for algo in algorithms:
        algo_df = utility_df[utility_df['Algorithm'] == algo]
        if not algo_df.empty:
            sns.lineplot(
                data=algo_df, x='Instances', y='weighted_utility',
                ax=ax2, label=algo, color=color_palette.get(algo),
                marker=marker_map.get(algo), linestyle=linestyle_map.get(algo)
            )
    
    ax2.set_xlabel('Number of Dynamic Instances')
    ax2.set_ylabel('System Utility Score')
    ax2.set_xticks([2, 4, 6, 8, 10])
    ax2.xaxis.set_minor_locator(mticker.NullLocator())
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax2.legend(frameon=True, edgecolor='black')

    plt.tight_layout()
    output_path2 = os.path.join(output_dir, 'dynamic_weighted_utility.png')
    plt.savefig(output_path2, dpi=300)
    plt.close()
    print(f"Saved Plot 2: Dynamic System Utility to {output_path2}")

if __name__ == '__main__':
    CSV_FILE = 'INFOCOM2025_ADAPT_GUAV/showcases/Simulating_a_data_collection_scenario/simulation_results_pretrained_batch1.csv'
    OUTPUT_DIR = 'INFOCOM2025_ADAPT_GUAV/showcases/Simulating_a_data_collection_scenario/plot'
    plot_dynamic_adaptability(CSV_FILE, OUTPUT_DIR)
