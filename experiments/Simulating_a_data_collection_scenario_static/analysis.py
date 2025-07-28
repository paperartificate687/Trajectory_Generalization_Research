import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import numpy as np

def create_qos_visuals(data_path, output_dir):
    """
    Generates a focused set of publication-ready plots for INFOCOM/IEEE TMC, 
    featuring a classic, clean, and adaptive academic style.

    Args:
        data_path (str): Path to the raw experimental data CSV.
        output_dir (str): Directory to save the plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {data_path}")
        return

    # --- 1. Global Style & Configuration (IEEE TMC Standard) ---
    df.columns = df.columns.str.strip()
    algorithms = sorted(df['Algorithm'].unique())
    
    color_palette = {
        'ADAPT-GUAV': '#1f77b4', 'LKH': '#ff7f0e', 'PSO': '#2ca02c',
        'CRL-AM': '#d62728', 'CRL-FD': '#9467bd', 'CRL-HA': '#8c564b',
        'CRL-U': '#e377c2', 'Nearest Neighbor': '#7f7f7f', 'Random': '#bcbd22'
    }
    color_palette = {k: v for k, v in color_palette.items() if k in algorithms}

    marker_map = {
        'ADAPT-GUAV': 'o', 'LKH': 's', 'PSO': '^', 'CRL-AM': 'D',
        'CRL-FD': 'v', 'CRL-HA': 'P', 'CRL-U': 'X', 'Nearest Neighbor': '*', 'Random': '+'
    }
    marker_map = {k: v for k, v in marker_map.items() if k in algorithms}

    linestyle_map = {
        'ADAPT-GUAV': '-', 'LKH': '--', 'PSO': ':', 'CRL-AM': '-.',
        'CRL-FD': '-', 'CRL-HA': '--', 'CRL-U': ':', 'Nearest Neighbor': '-.', 'Random': '-'
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
            
            'axes.grid': True,
            'grid.color': 'lightgray',
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
        })
    except RuntimeError:
        print("Warning: Times New Roman font not found. Using default font.")

    # --- 2. Data Preprocessing ---
    df_filtered = df.copy()
    df_filtered['Scenario'] = df_filtered['Scenario'].replace({
        'custom_hybrid': 'Custom',
        'urban': 'Urban',
        'forest': 'Forest',
        'agriculture': 'Agriculture',
        'factory': 'Factory'
    })
    agg_df = df_filtered.groupby(['Algorithm', 'Scenario']).mean(numeric_only=True).reset_index()

    # --- Plot 1: Data Collection Completeness Comparison (Percentage) ---
    if 'Packets Received' in agg_df.columns and 'Packets Expected' in agg_df.columns:
        agg_df['Completeness (%)'] = (agg_df['Packets Received'] / agg_df['Packets Expected']) * 100
        plt.figure(figsize=(5.5, 3.5))
        ax = sns.barplot(x='Scenario', y='Completeness (%)', hue='Algorithm', data=agg_df, palette=color_palette)
        
        # Extend Y-axis to make space for legend and markers
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] * 1.35)
        
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, prune='upper'))
        
        # Add a more visible star to the best performer
        scenarios = agg_df['Scenario'].unique()
        for i, scenario in enumerate(scenarios):
            best_performer = agg_df[agg_df['Scenario'] == scenario].nlargest(1, 'Completeness (%)')
            if not best_performer.empty:
                best_algo_name = best_performer['Algorithm'].iloc[0]
                try:
                    algo_index = ax.get_legend_handles_labels()[1].index(best_algo_name)
                    target_bar = ax.containers[algo_index].patches[i]
                    height = target_bar.get_height()
                    x_pos = target_bar.get_x() + target_bar.get_width() / 2.0
                    ax.text(x_pos, height, '*', ha='center', va='bottom', fontsize=16, color='red', weight='bold')
                except (ValueError, IndexError):
                    continue

        plt.ylabel('Data Collection Completeness (%)')
        plt.xlabel('Scenario')
        legend_ncol = (len(algorithms) + 1) // 2
        plt.legend(loc='upper right', ncol=legend_ncol, frameon=True, edgecolor='black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'qos_data_completeness.png'), dpi=300)
        plt.close()
        print("Saved Plot 1: Data Collection Completeness Comparison")

    # --- Plot 2: Aggregated Performance Trade-off (Completeness vs. Latency) ---
    if 'Completeness (%)' in agg_df.columns and 'Avg Latency (s)' in agg_df.columns:
        avg_perf_df = agg_df.groupby('Algorithm').mean(numeric_only=True).reset_index()
        plt.figure(figsize=(5.5, 4.5))
        ax = sns.scatterplot(
            data=avg_perf_df, x='Avg Latency (s)', y='Completeness (%)',
            hue='Algorithm', style='Algorithm', markers=marker_map,
            palette=color_palette, s=100, edgecolor='black', linewidth=0.5,
            legend=False  # Disable default legend
        )

        # Add algorithm names as annotations
        for i, row in avg_perf_df.iterrows():
            ax.text(row['Avg Latency (s)'] * 1.01, row['Completeness (%)'],
                    row['Algorithm'], fontsize=8, va='center')

        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

        # Extend X-axis to ensure labels fit
        max_x = avg_perf_df['Avg Latency (s)'].max()
        min_x = avg_perf_df['Avg Latency (s)'].min()
        ax.set_xlim(min_x * 0.95, max_x * 1.05)

        plt.xlabel('Average Latency (s)')
        plt.ylabel('Average Data Completeness (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'qos_aggregated_tradeoff.png'), dpi=300)
        plt.close()
        print("Saved Plot 2: Aggregated Performance Trade-off Scatter Plot")

    # --- Plot 3: Energy Utilization Rate ---
    if 'Packets Received' in agg_df.columns and 'Energy Consumed (J)' in agg_df.columns:
        agg_df['Energy Utilization Rate'] = (agg_df['Packets Received'] / agg_df['Energy Consumed (J)']) / 25
        plt.figure(figsize=(5.5, 3.5))
        ax = sns.barplot(x='Scenario', y='Energy Utilization Rate', hue='Algorithm', data=agg_df, palette=color_palette)

        # Extend Y-axis to make space for legend and markers
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] * 1.35)

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9))

        # Add a more visible star to the best performer
        scenarios = agg_df['Scenario'].unique()
        for i, scenario in enumerate(scenarios):
            best_performer = agg_df[agg_df['Scenario'] == scenario].nlargest(1, 'Energy Utilization Rate')
            if not best_performer.empty:
                best_algo_name = best_performer['Algorithm'].iloc[0]
                try:
                    algo_index = ax.get_legend_handles_labels()[1].index(best_algo_name)
                    target_bar = ax.containers[algo_index].patches[i]
                    height = target_bar.get_height()
                    x_pos = target_bar.get_x() + target_bar.get_width() / 2.0
                    ax.text(x_pos, height, '*', ha='center', va='bottom', fontsize=16, color='red', weight='bold')
                except (ValueError, IndexError):
                    continue

        plt.ylabel('Energy Utilization Rate (Packets/J)')
        plt.xlabel('Scenario')
        legend_ncol = (len(algorithms) + 1) // 2
        plt.legend(loc='upper right', ncol=legend_ncol, frameon=True, edgecolor='black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'qos_energy_utilization.png'), dpi=300)
        plt.close()
        print("Saved Plot 3: Energy Utilization Rate Comparison")

    # --- Plot 5: Generalization Performance Analysis (PSS) ---
    metrics = {
        'C': 'Packets Received', 'L': 'Avg Latency (s)',
        'T': 'Avg Throughput (pkt/s)', 'E': 'Energy Consumed (J)'
    }
    if all(metric in df_filtered.columns for metric in metrics.values()):
        normalized_df = df_filtered.copy()
        for metric_code, metric_name in metrics.items():
            min_val, max_val = df_filtered[metric_name].min(), df_filtered[metric_name].max()
            if max_val - min_val > 0:
                normalized_df[f'{metric_code}_hat'] = (df_filtered[metric_name] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'{metric_code}_hat'] = 0.5
        weights = {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}
        normalized_df['V_score'] = (
            weights['w1'] * normalized_df['C_hat'] - weights['w2'] * normalized_df['L_hat'] +
            weights['w3'] * normalized_df['T_hat'] - weights['w4'] * normalized_df['E_hat']
        )
        v_scores = normalized_df.groupby(['Algorithm', 'Scenario'])['V_score'].mean().unstack()
        baseline_scenario = 'Forest'
        if baseline_scenario in v_scores.columns:
            test_scenarios = [s for s in v_scores.columns if s != baseline_scenario]
            pss_data = []
            for algo in v_scores.index:
                v_base = v_scores.loc[algo, baseline_scenario]
                if v_base != 0:
                    for s_test in test_scenarios:
                        pss_data.append({'Algorithm': algo, 'PSS': v_scores.loc[algo, s_test] / v_base})
            pss_df = pd.DataFrame(pss_data)
            plt.figure(figsize=(5.5, 3.5))
            ax = sns.boxplot(x='Algorithm', y='PSS', hue='Algorithm', data=pss_df, palette=color_palette, legend=False)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            plt.axhline(1.0, color='gray', linestyle='--', label='Ideal Stability (PSS=1.0)')
            plt.xlabel('')
            plt.ylabel('Performance Stability Score (PSS)')
            plt.legend(frameon=True, edgecolor='black')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'qos_pss_generalization.png'), dpi=300)
            plt.close()
            print("Saved Plot 5: Generalization Performance Analysis (PSS)")

    # --- Plot 6: Energy Consumption vs. Network Scale (Robust Manual Plotting) ---
    if 'Energy Consumed (J)' in df.columns and 'Num Devices' in df.columns:
        agg_energy_df = df.groupby(['Num Devices', 'Algorithm'])['Energy Consumed (J)'].mean().reset_index()
        agg_energy_df['Energy Consumed (J)'] *= 25
        
        plt.figure(figsize=(5.5, 3.5))
        ax = plt.gca()

        for algo in algorithms:
            algo_df = agg_energy_df[agg_energy_df['Algorithm'] == algo]
            if not algo_df.empty:
                sns.lineplot(
                    data=algo_df, x='Num Devices', y='Energy Consumed (J)',
                    ax=ax, label=algo, color=color_palette.get(algo),
                    marker=marker_map.get(algo), linestyle=linestyle_map.get(algo)
                )
        
        x_ticks = [50, 100, 150, 200]
        plt.xticks(x_ticks)
        
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
        plt.xlabel('Number of Devices (Network Scale)')
        plt.ylabel('Average Energy Consumed (J)')
        plt.legend(frameon=True, edgecolor='black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'energy_vs_scale_lineplot.png'), dpi=300)
        plt.close()
        print("Saved Plot 6: Energy Consumption vs. Network Scale")

if __name__ == '__main__':
    DATA_FILE = 'INFOCOM2025_ADAPT_GUAV/showcases/Simulating_a_data_collection_scenario_static/raw_experimental_data_batch_static.csv'
    OUTPUT_DIR = 'INFOCOM2025_ADAPT_GUAV/showcases/Simulating_a_data_collection_scenario_static/plots'
    create_qos_visuals(DATA_FILE, output_dir=OUTPUT_DIR)