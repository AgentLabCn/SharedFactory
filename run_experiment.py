# run_experiment.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.factory_model import FactoryModel
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from datetime import datetime
import os
import sys
import traceback
import agentpy as ap

# Metric abbreviation definitions
metric_abbr = {
    'profit_ratio': 'pft',
    'equipment_efficiency': 'eqp',
    'delay_ratio': 'dly',
    'performance': 'prf'
}

def plot_metrics_trends(model_data, title):
    """Plot metrics trend graphs"""
    metrics = ['delay_ratio', 'knowledge_effect', 'equipment_efficiency', 
               'profit_cost_ratio', 'average_delay_time', 'performance']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Metrics Trends - {title}', fontsize=16)
    
    df = pd.DataFrame(model_data)
    
    # Performance index calculation
    df['performance'] = (
        (df['profit_cost_ratio'].clip(lower=0.1) ** 0.5) * 
        (df['equipment_efficiency'].clip(lower=0.1) ** 0.3) * 
        (1 / (1 + df['delay_ratio'])) ** 0.2
    )
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        ax.plot(df[metric])
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('metrics_trends.png')
    plt.close()

def run_experiment(params: dict, n_samples: int = 10):
    """Run experiment"""
    results = []
    experiment_trends = []
    
    try:
        for i in range(n_samples):
            if i % 5 == 0:
                print(f"Running experiment {i+1}/{n_samples}")
            
            try:
                model = FactoryModel(params)
                model.run()
                
                if model.data:
                    experiment_trends.append(model.data)
                    final_state = model.data[-1]
                    
                    # Add data validation
                    result = {
                        'delay_ratio': max(0, min(1, final_state.get('delay_ratio', 0))),
                        'knowledge_effect': max(0.1, final_state.get('knowledge_effect', 1)),
                        'equipment_efficiency': max(0.1, min(1, final_state.get('equipment_efficiency', 0))),
                        'profit_cost_ratio': max(0.1, final_state.get('profit_cost_ratio', 0)),
                        'average_delay_time': max(0, final_state.get('average_delay_time', 0))
                    }
                    results.append(result)
            except Exception as e:
                print(f"Error in experiment {i+1}: {e}")
                continue
                
    except Exception as e:
        print(f"Fatal error in experiments: {e}")
    
    # Ensure at least one valid result
    if not results:
        print("Error: No valid results collected")
        return pd.DataFrame(), {'mean': pd.Series(), 'std': pd.Series()}, []
    
    try:
        df_results = pd.DataFrame(results)
        summary = {
            'mean': df_results.mean(),
            'std': df_results.std()
        }
        return df_results, summary, experiment_trends
    except Exception as e:
        print(f"Error creating summary: {e}")
        return pd.DataFrame(), {'mean': pd.Series(), 'std': pd.Series()}, []

def run_capacity_experiments():
    """Run experiments with different capacity configurations"""
    # Base parameters
    base_params = {
        'simulation_duration': 120,
        'order_generation_interval': 3,
        'knowledge_transfer_rate': 0.02,
        'inconvenience_level': 0.01,
        'alpha1': 0.5,
        'alpha2': 0.3,
        'alpha3': 0.2
    }
    
    # Capacity configurations
    capacity_settings = [5, 10, 15]
    capacity_results = {}
    
    for np in capacity_settings:
        print(f"\nRunning experiments with {np} production lines:")
        params = base_params.copy()
        params['num_production_lines'] = np
        
        results_df, summary, trends = run_experiment(params)
        
        if trends:
            plot_metrics_trends(trends, f"Capacity Analysis (NP={np})")
            capacity_results[np] = {
                'mean': summary['mean'],
                'std': summary['std']
            }
    
    # Save results for different capacity configurations
    capacity_comparison = pd.DataFrame({
        f'NP={np}': {
            'mean_profit_ratio': results['mean']['profit_cost_ratio'],
            'mean_efficiency': results['mean']['equipment_efficiency'],
            'mean_delay': results['mean']['delay_ratio'],
            'std_profit_ratio': results['std']['profit_cost_ratio'],
            'std_efficiency': results['std']['equipment_efficiency'],
            'std_delay': results['std']['delay_ratio']
        }
        for np, results in capacity_results.items()
    })
    
    capacity_comparison.to_excel('capacity_comparison.xlsx')
    
    return capacity_comparison

def analyze_capacity_impact(capacity_results: pd.DataFrame):
    """Analyze the impact of capacity configuration on performance"""
    plt.figure(figsize=(15, 5))
    
    # Plot comparison of three key metrics
    metrics = ['profit_ratio', 'efficiency', 'delay']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        means = [results[f'mean_{metric}'] for np, results in capacity_results.items()]
        stds = [results[f'std_{metric}'] for np, results in capacity_results.items()]
        
        plt.errorbar([5, 10, 15], means, yerr=stds, fmt='o-')
        plt.title(f'{metric.replace("_", " ").title()} vs Capacity')
        plt.xlabel('Number of Production Lines')
        plt.ylabel('Value')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('capacity_analysis.png')
    plt.close()

def run_factorial_experiment():
    """Run complete factorial experiment"""
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Experiment parameter levels
    factor_levels = {
        'knowledge_rate': [0.02, 0.04, 0.06],      # Knowledge transfer rate
        'inconvenience_level': [0.01, 0.02, 0.03], # Inconvenience cost coefficient
        'capacity': [5, 10, 15],                    # Capacity configuration
        'duration': [12, 24, 36]                     # Order duration
    }
    
    # Number of replications per group
    n_replications = 30  
    
    # Calculate total number of experiments
    total_experiments = (len(factor_levels['knowledge_rate']) * 
                        len(factor_levels['inconvenience_level']) * 
                        len(factor_levels['capacity']) * 
                        len(factor_levels['duration']) * 
                        n_replications)
    # total_experiments = 3 * 3 * 3 * 3 * 10 = 810
    
    # Base parameters (consistent with SharedFactory)
    base_params = {
        'simulation_duration': 120,
        'order_generation_interval': 3,
        'alpha1': 0.5,  # Cost-profit ratio weight
        'alpha2': 0.3,  # Equipment usage rate weight
        'alpha3': 0.2,  # Delay rate weight
        'base_order_value': 300.0  # Add base order value parameter
    }
    
    # Store experiment results
    results_data = []
    
    current_experiment = 0
    
    try:
        # Run all parameter combinations
        for b in factor_levels['knowledge_rate']:
            for h in factor_levels['inconvenience_level']:
                for np in factor_levels['capacity']:
                    for d in factor_levels['duration']:
                        params = base_params.copy()
                        params.update({
                            'knowledge_transfer_rate': b,
                            'inconvenience_level': h,
                            'num_production_lines': np,
                            'order_duration': d
                        })
                        
                        print(f"\nRunning experiments with parameters:")
                        print(f"Knowledge rate: {b}")
                        print(f"Inconvenience level: {h}")
                        print(f"Capacity: {np}")
                        print(f"Duration: {d}")
                        
                        # Run multiple repetitions for each combination
                        for rep in range(n_replications):
                            current_experiment += 1
                            progress = (current_experiment / total_experiments) * 100
                            print(f"\rProgress: {progress:.1f}% ", end="")
                            
                            try:
                                # Use FactoryModel directly
                                model = FactoryModel(params)
                                model.run()
                                
                                if model.data:
                                    final_state = model.data[-1]
                                    # Calculate performance index
                                    performance = (
                                        (final_state.get('profit_cost_ratio', 0.1) ** 0.5) * 
                                        (final_state.get('equipment_efficiency', 0.1) ** 0.3) * 
                                        (1 / (1 + final_state.get('delay_ratio', 0))) ** 0.2
                                    )
                                    
                                    results_data.append({
                                        'knowledge_rate': b,
                                        'inconvenience_level': h,
                                        'capacity': np,
                                        'duration': d,
                                        'repetition': rep,
                                        'profit_ratio': final_state.get('profit_cost_ratio', 0),
                                        'equipment_efficiency': final_state.get('equipment_efficiency', 0),
                                        'delay_ratio': final_state.get('delay_ratio', 0),
                                        'knowledge_effect': final_state.get('knowledge_effect', 1),
                                        'performance': performance
                                    })
                            except Exception as e:
                                print(f"\nError in experiment {current_experiment}: {e}")
                                continue
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # Save existing results
        results_df = pd.DataFrame(results_data)
        results_df.to_excel(f'raw_results_{timestamp}.xlsx')
    
    return pd.DataFrame(results_data)

def perform_anova_analysis(results_df):
    """Perform ANOVA analysis"""
    # Modify metric names to match column names in the DataFrame
    metrics_mapping = {
        'profit_ratio': 'profit_ratio',
        'equipment_efficiency': 'equipment_efficiency',  # Use full column name
        'delay_ratio': 'delay_ratio',
        'performance': 'performance'
    }
    
    anova_results = {}
    
    for metric_name, column_name in metrics_mapping.items():
        try:
            # Build ANOVA model
            formula = (f"{column_name} ~ C(knowledge_rate) + C(inconvenience_level) + "
                      f"C(capacity) + C(duration) + C(knowledge_rate):C(inconvenience_level) + "
                      f"C(knowledge_rate):C(capacity) + "
                      f"C(inconvenience_level):C(capacity) + "
                      f"C(knowledge_rate):C(duration) + "
                      f"C(inconvenience_level):C(duration) + "
                      f"C(capacity):C(duration)")
            
            model = ols(formula, data=results_df).fit()
            
            # Perform ANOVA
            anova_table = sm.stats.anova_lm(model, typ=2)
            anova_results[metric_name] = anova_table
            
            # Calculate effect sizes (Eta-squared)
            ss_total = anova_table['sum_sq'].sum()
            effect_sizes = anova_table['sum_sq'] / ss_total
            anova_results[f'{metric_name}_effect_sizes'] = effect_sizes
            
        except Exception as e:
            print(f"Error in ANOVA analysis for {metric_name}: {e}")
            continue
    
    return anova_results

def calculate_performance_index(metrics):
    """Calculate comprehensive performance index"""
    try:
        profit_ratio = max(0.1, metrics.get('profit_ratio', 0.1))
        efficiency = max(0.1, metrics.get('equipment_efficiency', 0.1))
        delay_ratio = max(0, min(1, metrics.get('delay_ratio', 0)))
        
        performance = (
            (profit_ratio ** 0.5) *  # alpha1 = 0.5
            (efficiency ** 0.3) *    # alpha2 = 0.3
            (1 / (1 + delay_ratio)) ** 0.2  # alpha3 = 0.2
        )
        return performance
    except Exception as e:
        print(f"Error calculating performance index: {e}")
        return 0.1  # Return a valid default value

def plot_main_effects(results_df):
    """Plot main effects"""
    factors = ['knowledge_rate', 'inconvenience_level', 'capacity', 'duration']
    metrics = ['profit_ratio', 'equipment_efficiency', 'delay_ratio', 'performance']
    
    fig, axes = plt.subplots(len(metrics), len(factors), figsize=(15, 20))
    
    for i, metric in enumerate(metrics):
        for j, factor in enumerate(factors):
            means = results_df.groupby(factor)[metric].mean()
            sems = results_df.groupby(factor)[metric].sem()
            
            axes[i, j].errorbar(means.index, means.values, yerr=sems.values, 
                              marker='o', capsize=5)
            axes[i, j].set_title(f'{factor} vs {metric}')
            axes[i, j].grid(True)
    
    plt.tight_layout()
    plt.savefig('main_effects.png')
    plt.close()

def plot_interaction_effects(results_df):
    """Plot interaction effects"""
    metrics = ['profit_ratio', 'equipment_efficiency', 'delay_ratio', 'performance']
    factors = ['knowledge_rate', 'inconvenience_level', 'capacity', 'duration']
    
    # Create a plot for each metric
    for metric in metrics:
        # Create a 2x3 subplot layout (6 interaction effects)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Interaction Effects for {metric}', fontsize=16)
        
        # Generate all factor pairs
        factor_pairs = [
            ('knowledge_rate', 'inconvenience_level'),
            ('knowledge_rate', 'capacity'),
            ('knowledge_rate', 'duration'),
            ('inconvenience_level', 'capacity'),
            ('inconvenience_level', 'duration'),
            ('capacity', 'duration')
        ]
        
        for idx, (factor1, factor2) in enumerate(factor_pairs):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Calculate interaction effects
            interaction = results_df.groupby([factor1, factor2])[metric].mean().unstack()
            
            # Plot interaction effects
            interaction.plot(marker='o', ax=ax)
            ax.set_title(f'{factor1} × {factor2}')
            ax.set_xlabel(factor1)
            ax.set_ylabel(metric)
            ax.grid(True)
            ax.legend(title=factor2)
        
        plt.tight_layout()
        plt.savefig(f'interaction_effects_{metric}.png')
        plt.close()

def validate_experiment_params(params: dict) -> bool:
    """Validate experiment parameters"""
    required_params = {
        'simulation_duration': (int, lambda x: x > 0),
        'order_generation_interval': (int, lambda x: x > 0),
        'knowledge_transfer_rate': (float, lambda x: 0 < x <= 0.1),
        'inconvenience_level': (float, lambda x: 0 < x <= 0.1),
        'num_production_lines': (int, lambda x: x in [5, 10, 15]),
        'alpha1': (float, lambda x: 0 <= x <= 1),
        'alpha2': (float, lambda x: 0 <= x <= 1),
        'alpha3': (float, lambda x: 0 <= x <= 1)
    }
    
    try:
        for param, (param_type, validator) in required_params.items():
            if param not in params:
                print(f"Missing parameter: {param}")
                return False
            
            if not isinstance(params[param], param_type):
                print(f"Invalid type for {param}: expected {param_type}")
                return False
                
            if not validator(params[param]):
                print(f"Invalid value for {param}: {params[param]}")
                return False
                
        # Validate weight sum is 1.0
        weight_sum = params['alpha1'] + params['alpha2'] + params['alpha3']
        if not np.isclose(weight_sum, 1.0):
            print(f"Weight sum must be 1.0, got {weight_sum}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error validating parameters: {e}")
        return False

def validate_results(results_df: pd.DataFrame) -> bool:
    """Validate experiment results"""
    if results_df.empty:
        print("Results DataFrame is empty")
        return False
    
    print("\nAvailable columns:", results_df.columns.tolist())
    print("\nFirst few rows of data:")
    print(results_df.head())
    
    # Column name matching
    validations = {
        'delay_ratio': (0, 1),
        'equipment_efficiency': (0, 1),
        'profit_ratio': (0, None), 
        'knowledge_effect': (0.1, None)
    }
    
    for column, (min_val, max_val) in validations.items():
        if column not in results_df.columns:
            print(f"\nMissing column: {column}")
            print(f"Available columns: {results_df.columns.tolist()}")
            return False
            
        if min_val is not None and (results_df[column] < min_val).any():
            print(f"Invalid values in {column}: below {min_val}")
            return False
            
        if max_val is not None and (results_df[column] > max_val).any():
            print(f"Invalid values in {column}: above {max_val}")
            return False
    
    return True

def validate_metrics(metrics: dict) -> bool:
    """Validate metric data"""
    try:
        validations = {
            'profit_ratio': (0, None),
            'equipment_efficiency': (0, 1),
            'delay_ratio': (0, 1),
            'knowledge_effect': (0.1, None)
        }
        
        for metric, (min_val, max_val) in validations.items():
            value = metrics.get(metric)
            if value is None:
                print(f"Missing metric: {metric}")
                return False
            
            if min_val is not None and value < min_val:
                print(f"Invalid {metric}: {value} < {min_val}")
                return False
            
            if max_val is not None and value > max_val:
                print(f"Invalid {metric}: {value} > {max_val}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error validating metrics: {e}")
        return False

def analyze_results(anova_results):
    """Analyze ANOVA results"""
    significant_effects = []
    
    for metric, table in anova_results.items():
        if not isinstance(table, pd.DataFrame):
            continue
            
        # Check for effects with p-value less than 0.05
        significant = table[table['PR(>F)'] < 0.05]
        
        for idx, row in significant.iterrows():
            significant_effects.append({
                'metric': metric,
                'factor': idx,
                'F_value': row['F'],
                'p_value': row['PR(>F)']
            })
    pd.DataFrame(significant_effects).to_excel('significant_effects.xlsx')
    
    return significant_effects

def analyze_duration_effects(results_df):
    """Analyze duration effects"""
    # Define metrics and their abbreviations
    metrics = ['profit_ratio', 'equipment_efficiency', 'delay_ratio', 'performance']
    metric_abbr = {
        'profit_ratio': 'pft',
        'equipment_efficiency': 'eqp',
        'delay_ratio': 'dly',
        'performance': 'prf'
    }
    
    # Add factor abbreviations
    factor_abbr = {
        'knowledge_rate': 'kr',
        'inconvenience_level': 'il',
        'capacity': 'cap',
        'duration': 'dur'
    }

    # 1. Duration main effect analysis
    duration_effects = {}
    for metric in metrics:
        duration_means = results_df.groupby('duration')[metric].mean()
        duration_effects[metric] = duration_means

    # 2. Interaction effect analysis
    interaction_effects = {}
    
    # Generate all possible combinations of interaction effects for each indicator
    for metric in metrics:
        # Generate all factor pairs
        factor_pairs = [
            ('knowledge_rate', 'inconvenience_level'),
            ('knowledge_rate', 'capacity'),
            ('knowledge_rate', 'duration'),
            ('inconvenience_level', 'capacity'),
            ('inconvenience_level', 'duration'),
            ('capacity', 'duration')
        ]
        
        for factor1, factor2 in factor_pairs:
            # Create interaction effect table
            interaction = results_df.groupby([factor1, factor2])[metric].mean().unstack()
            sheet_name = f"{factor_abbr[factor1]}_{factor_abbr[factor2]}_{metric_abbr[metric]}"
            interaction_effects[sheet_name] = interaction

    return duration_effects, interaction_effects

def plot_duration_effects(results_df):
    """Plot duration effects"""
    metrics = ['profit_ratio', 'equipment_efficiency', 'delay_ratio', 'performance']
    
    # 1. Main effect plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Duration Main Effects', fontsize=16)
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        
        means = results_df.groupby('duration')[metric].mean()
        sems = results_df.groupby('duration')[metric].sem()
        
        axes[row, col].errorbar(means.index, means.values, yerr=sems.values, 
                              marker='o', capsize=5)
        axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
        axes[row, col].set_xlabel('Order Duration')
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig('duration_main_effects.png')
    plt.close()
    
    # 2. Interaction effect plot
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Duration Interaction Effects - {metric}', fontsize=16)
        
        factors = ['knowledge_rate', 'inconvenience_level', 'capacity']
        for idx, factor in enumerate(factors):
            means = results_df.groupby([factor, 'duration'])[metric].mean().unstack()
            means.plot(marker='o', ax=axes[idx])
            axes[idx].set_title(f'{factor} × duration')
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'duration_interaction_{metric}.png')
        plt.close()

def create_three_line_tables(anova_results, result_dir):
    """Create three-line tables"""
    if not anova_results:
        print("No ANOVA results to create tables")
        return
        
    metrics = ['profit_ratio', 'equipment_efficiency', 'delay_ratio', 'performance']
    
    # Create data for two tables
    main_effects_data = []
    interaction_effects_data = []
    
    for metric in metrics:
        if metric not in anova_results:
            continue
            
        anova_table = anova_results[metric]
        
        # Extract main effects
        main_effects = anova_table[anova_table.index.str.contains('^C\([^:]+\)$')]
        
        # Extract interaction effects
        interaction_effects = anova_table[anova_table.index.str.contains(':')]
        
        # Format main effects data
        for idx, row in main_effects.iterrows():
            factor = idx.strip('C()').replace('knowledge_rate', 'KR')\
                       .replace('inconvenience_level', 'IL')\
                       .replace('capacity', 'CAP')\
                       .replace('duration', 'DUR')
            main_effects_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Factor': factor,
                'F-value': f"{row['F']:.3f}",
                'p-value': f"{row['PR(>F)']:.3f}",
                'Significance': '***' if row['PR(>F)'] < 0.001 else 
                              '**' if row['PR(>F)'] < 0.01 else 
                              '*' if row['PR(>F)'] < 0.05 else 'ns'
            })
        
        # Format interaction effects data
        for idx, row in interaction_effects.iterrows():
            factors = idx.split(':')
            factor_names = [f.strip('C()').replace('knowledge_rate', 'KR')\
                           .replace('inconvenience_level', 'IL')\
                           .replace('capacity', 'CAP')\
                           .replace('duration', 'DUR') for f in factors]
            interaction_name = ' × '.join(factor_names)
            
            interaction_effects_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Interaction': interaction_name,
                'F-value': f"{row['F']:.3f}",
                'p-value': f"{row['PR(>F)']:.3f}",
                'Significance': '***' if row['PR(>F)'] < 0.001 else 
                              '**' if row['PR(>F)'] < 0.01 else 
                              '*' if row['PR(>F)'] < 0.05 else 'ns'
            })
    
    # Create DataFrames
    main_effects_df = pd.DataFrame(main_effects_data)
    interaction_effects_df = pd.DataFrame(interaction_effects_data)
    
    # Save to Excel, using three-line table format
    excel_path = os.path.join(result_dir, 'anova_three_line_tables.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Main effects table
        main_effects_df.to_excel(writer, sheet_name='Main Effects', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Main Effects']
        
        # Set three-line table format
        header_format = workbook.add_format({
            'bold': True,
            'top': 2,
            'bottom': 1
        })
        bottom_format = workbook.add_format({
            'bottom': 2
        })
        
        # Apply format
        for col_num, value in enumerate(main_effects_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Add bottom double line
        last_row = len(main_effects_df) + 1
        for col_num in range(len(main_effects_df.columns)):
            worksheet.write(last_row, col_num, '', bottom_format)
            
        # Interaction effects table
        interaction_effects_df.to_excel(writer, sheet_name='Interaction Effects', index=False)
        worksheet = writer.sheets['Interaction Effects']
        
        # Apply same format
        for col_num, value in enumerate(interaction_effects_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        last_row = len(interaction_effects_df) + 1
        for col_num in range(len(interaction_effects_df.columns)):
            worksheet.write(last_row, col_num, '', bottom_format)
            
        # Adjust column widths
        for worksheet in [writer.sheets['Main Effects'], writer.sheets['Interaction Effects']]:
            worksheet.set_column(0, 0, 15)  # Metric column
            worksheet.set_column(1, 1, 20)  # Factor/Interaction column
            worksheet.set_column(2, 2, 12)  # F-value column
            worksheet.set_column(3, 3, 12)  # p-value column
            worksheet.set_column(4, 4, 12)  # Significance column
    
    print(f"Three-line tables saved to {excel_path}")
    
    # Create chart version
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    create_three_line_plot(main_effects_df, 'Main Effects Analysis')
    
    plt.subplot(2, 1, 2)
    create_three_line_plot(interaction_effects_df, 'Interaction Effects Analysis')
    
    plt.tight_layout()
    plot_path = os.path.join(result_dir, 'anova_three_line_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Three-line plots saved to {plot_path}")

def create_three_line_plot(df, title):
    """Create three-line chart"""
    # Create table
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    
    # Set table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Add top double line and bottom single line
    for key, cell in table._cells.items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_linewidth(2)
        if key[0] == len(df):  # Last row
            cell.set_linewidth(2)
    
    # Hide axes
    plt.axis('off')
    plt.title(title)

if __name__ == "__main__":
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    try:
        # Run factorial experiment
        results_df = run_factorial_experiment()
        
        if results_df.empty:
            print("No valid results collected")
            sys.exit(1)
            
        # Validate results
        if not validate_results(results_df):
            print("Invalid experimental results")
            sys.exit(1)
            
        print("\nColumns in results_df:", results_df.columns.tolist())  # 添加调试信息
        
        # Perform ANOVA analysis
        try:
            anova_results = perform_anova_analysis(results_df)
            if not anova_results:
                print("Warning: ANOVA analysis produced no results")
        except Exception as e:
            print(f"Error in ANOVA analysis: {e}")
            anova_results = {}
        
        # Plot main effects
        try:
            plot_main_effects(results_df)
        except Exception as e:
            print(f"Error plotting main effects: {e}")
        
        # Plot interaction effects
        try:
            plot_interaction_effects(results_df)
        except Exception as e:
            print(f"Error plotting interaction effects: {e}")
        
        # Save analysis results
        try:
            with pd.ExcelWriter(f'{result_dir}/analysis_results.xlsx') as writer:
                # Save raw data
                results_df.to_excel(writer, sheet_name='raw_data')
                
                # Save ANOVA results
                if anova_results:
                    for metric, table in anova_results.items():
                        if isinstance(table, pd.DataFrame):
                            table.to_excel(writer, sheet_name=f'anova_{metric[:30]}')
                
                # Save main effects
                for factor in ['knowledge_rate', 'inconvenience_level', 'capacity', 'duration']:
                    main_effect = results_df.groupby(factor).mean()
                    main_effect.to_excel(writer, sheet_name=f'{factor[:30]}_effects')
                
                # Save experiment configuration
                pd.DataFrame([{
                    'timestamp': timestamp,
                    'total_experiments': len(results_df),
                    'simulation_duration': 120,
                    'order_generation_interval': 3
                }]).to_excel(writer, sheet_name='experiment_config')
                
        except Exception as e:
            print(f"Error saving results: {e}")
        
        # Print ANOVA results
        if anova_results:
            print("\nANOVA Results:")
            print("==============")
            for metric, anova_table in anova_results.items():
                if isinstance(anova_table, pd.DataFrame):
                    print(f"\n{metric}:")
                    print(anova_table)
                    
        # Add duration analysis
        print("\nAnalyzing Duration Effects:")
        duration_effects, interaction_effects = analyze_duration_effects(results_df)
        
        # Plot duration effects
        plot_duration_effects(results_df)
        
        # Save duration analysis results
        with pd.ExcelWriter(f'{result_dir}/duration_analysis.xlsx') as writer:
            # Save main effects - use short sheet names
            for metric, effects in duration_effects.items():
                sheet_name = f"dur_{metric_abbr[metric]}"
                effects.to_excel(writer, sheet_name=sheet_name)
            
            # Save interaction effects - use predefined short sheet names
            for name, effects in interaction_effects.items():
                effects.to_excel(writer, sheet_name=name)
        
        # Print duration analysis results
        print("\nDuration Main Effects:")
        for metric, effects in duration_effects.items():
            print(f"\n{metric}:")
            print(effects)
        
        # Create three-line tables after ANOVA analysis
        try:
            create_three_line_tables(anova_results, result_dir)
        except Exception as e:
            print(f"Error creating three-line tables: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()