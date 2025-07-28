import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics():
    # Load the data
    csv_path = 'INFOCOM2025_ADAPT_GUAV/showcases/Simulating_a_data_collection_scenario_static/raw_experimental_data_batch_static.csv'
    df = pd.read_csv(csv_path)

    # Clean up column names by stripping leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Filter for Num Devices = 199
    df_199 = df[df['Num Devices'] == 199].copy()

    # Calculate Packet Delivery Rate (PDR)
    df_199['PDR'] = df_199['Packets Received'] / df_199['Packets Expected']

    # Group by Scenario and Algorithm and calculate mean of metrics
    grouped_data = df_199.groupby(['Scenario', 'Algorithm']).agg({
        'Avg Throughput (pkt/s)': 'mean',
        'Energy Consumed (J)': 'mean',
        'PDR': 'mean',
        'Mission Duration (s)': 'mean'
    }).reset_index()

    scenarios = ['agriculture', 'custom_hybrid', 'factory', 'forest', 'urban']
    
    # Reorder scenarios to a more logical sequence if needed
    grouped_data['Scenario'] = pd.Categorical(grouped_data['Scenario'], categories=scenarios, ordered=True)
    grouped_data = grouped_data.sort_values('Scenario')

    # Plotting
    plot_bar_chart(grouped_data, 'Avg Throughput (pkt/s)', 'Average Throughput (pkt/s)', 'Throughput Comparison (199 Devices)', 'throughput_comparison_199.png')
    plot_bar_chart(grouped_data, 'Energy Consumed (J)', 'Average Energy Consumed (J)', 'Energy Consumption Comparison (199 Devices)', 'energy_consumption_199.png')
    plot_bar_chart(grouped_data, 'PDR', 'Packet Delivery Rate', 'PDR Comparison (199 Devices)', 'pdr_comparison_199.png')
    plot_bar_chart(grouped_data, 'Mission Duration (s)', 'Average Mission Duration (s)', 'Mission Duration Comparison (199 Devices)', 'mission_duration_199.png')

def plot_bar_chart(data, metric_col, ylabel, title, filename):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7))

    scenarios = data['Scenario'].unique()
    algorithms = sorted(data['Algorithm'].unique())
    n_algos = len(algorithms)
    n_scenarios = len(scenarios)
    
    bar_width = 0.12
    index = np.arange(n_scenarios)

    for i, algo in enumerate(algorithms):
        algo_data = data[data['Algorithm'] == algo]
        # Ensure we have data for all scenarios for this algorithm
        metric_values = []
        for scenario in scenarios:
            value = algo_data[algo_data['Scenario'] == scenario][metric_col]
            if not value.empty:
                metric_values.append(value.iloc[0])
            else:
                metric_values.append(0) # Or np.nan if you want to skip
        
        bar_positions = index + (i - n_algos / 2) * bar_width + bar_width/2
        ax.bar(bar_positions, metric_values, bar_width, label=algo)

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel('Scenario', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = f"INFOCOM2025_ADAPT_GUAV/showcases/[WPF] INFOCOM2026/fig/{filename}"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_metrics()