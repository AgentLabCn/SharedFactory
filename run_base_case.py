import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.factory_model import FactoryModel

def run_base_case(n_runs=30):  # Number of runs parameter, default 30
    """Run median case multiple times and take average"""
    # Basic parameter settings
    params = {
        'num_production_lines': 10,        # Production line capacity
        'simulation_duration': 120,       # Simulation duration
        'order_generation_interval': 3,   # Order generation interval
        'knowledge_transfer_rate': 0.04,  # Knowledge transfer rate
        'inconvenience_level': 0.02,     # Inconvenience cost coefficient
        'order_duration': 30,            # Order duration
        'base_order_value': 300.0,       # Base order value
        # Weight parameters
        'alpha1': 0.5,  # Cost-profit ratio weight
        'alpha2': 0.3,  # Equipment utilization weight
        'alpha3': 0.2   # Delay rate weight
    }
    
    # Store multiple run results
    all_runs = []
    
    for i in range(n_runs):
        if (i + 1) % 5 == 0:
            print(f"Completed {i + 1}/{n_runs} runs")
            
        # Create and run model
        model = FactoryModel(params)
        model.run()
        
        # Performance metrics
        df = pd.DataFrame(model.data)
        df['performance'] = (
            (df['profit_cost_ratio'].clip(lower=0.1) ** 0.5) * 
            (df['equipment_efficiency'].clip(lower=0.1) ** 0.3) * 
            (1 / (1 + df['delay_ratio'])) ** 0.2
        )
        all_runs.append(df)
    
    # Calculate mean and standard deviation
    mean_df = pd.concat(all_runs).groupby(level=0).mean()
    std_df = pd.concat(all_runs).groupby(level=0).std()
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Average Base Case Results in Shared Factory\n' + 
                f'(Capacity={params["num_production_lines"]}, ' +
                f'Knowledge Rate={params["knowledge_transfer_rate"]}, ' +
                f'Inconvenience Level={params["inconvenience_level"]}, ' +
                f'Order Duration={params["order_duration"]}, ' +
                f'N={n_runs} runs)',
                fontsize=14)
    
    # Define metrics and their properties
    metrics = {
        'performance': {
            'title': 'Performance Index',
            'color': 'blue',
            'position': (0, 0),
            'ylim': (0, 1.7)
        },
        'profit_cost_ratio': {
            'title': 'Profit-Cost Ratio',
            'color': 'red',
            'position': (0, 1),
            'ylim': (0, 3.5)
        },
        'equipment_efficiency': {
            'title': 'Equipment Efficiency',
            'color': 'green',
            'position': (1, 0),
            'ylim': (0, 1.2)
        },
        'delay_ratio': {
            'title': 'Delay Ratio',
            'color': 'orange',
            'position': (1, 1),
            'ylim': (0, 0.6)  
        }
    }
    
    # Plot each metric subplot
    for metric, properties in metrics.items():
        row, col = properties['position']
        ax = axes[row, col]
        
        # Plot mean line
        ax.plot(mean_df.index, mean_df[metric], 
                color=properties['color'], 
                label=properties['title'],
                linewidth=2)
        
        # Add standard deviation area
        ax.fill_between(mean_df.index,
                       mean_df[metric] - std_df[metric],
                       mean_df[metric] + std_df[metric],
                       color=properties['color'],
                       alpha=0.2)
        
        # Set title and labels
        ax.set_title(properties['title'], fontsize=12, pad=10)
        ax.set_xlabel('Time Steps', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=10)
        
        # Set y-axis range
        ax.set_ylim(properties['ylim'])
        
        # Beautify axes
        ax.tick_params(axis='both', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('base_case_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final results
    print("\nFinal Results (Average of {} runs):".format(n_runs))
    print("-" * 50)
    final_results = {
        'Performance Index': mean_df['performance'].iloc[-1],
        'Profit-Cost Ratio': mean_df['profit_cost_ratio'].iloc[-1],
        'Equipment Efficiency': mean_df['equipment_efficiency'].iloc[-1],
        'Delay Ratio': mean_df['delay_ratio'].iloc[-1]
    }
    
    for metric_name, value in final_results.items():
        # Get standard deviation
        df_column = metric_name.lower().replace('-', '_').replace(' ', '_')
        if df_column == 'performance_index':
            df_column = 'performance' 
        std_value = std_df[df_column].iloc[-1]
        print(f"{metric_name}: {value:.4f} ± {std_value:.4f}")
    
    # Save data to Excel
    with pd.ExcelWriter('base_case_results.xlsx') as writer:
        mean_df.to_excel(writer, sheet_name='mean_values')
        std_df.to_excel(writer, sheet_name='std_values')
        
        # Save configuration info
        pd.DataFrame([{
            'number_of_runs': n_runs,
            'capacity': params['num_production_lines'],
            'knowledge_rate': params['knowledge_transfer_rate'],
            'inconvenience_level': params['inconvenience_level'],
            'order_duration': params['order_duration']
        }]).to_excel(writer, sheet_name='configuration')
    
    return mean_df, std_df, final_results

if __name__ == "__main__":
    mean_df, std_df, results = run_base_case() 