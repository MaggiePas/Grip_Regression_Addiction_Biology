import pandas as pd
import numpy as np
from scipy import stats
import os

def load_model_results(file_path):
    """Load model results from CSV file."""
    return pd.read_csv(file_path)

def calculate_mae(actual, predicted):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))

def compare_models(model_a_path, model_b_path):
    """Compare two models using paired t-test on their MAEs."""
    
    # Load results
    model_a_results = load_model_results(model_a_path)
    model_b_results = load_model_results(model_b_path)
    
    # Calculate MAE for each subject
    model_a_mae = abs(model_a_results['Actual'] - model_a_results['Predicted'])
    model_b_mae = abs(model_b_results['Actual'] - model_b_results['Predicted'])
    
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(model_a_mae, model_b_mae)
    
    # Calculate overall MAE for each model
    mae_a = calculate_mae(model_a_results['Actual'], model_a_results['Predicted'])
    mae_b = calculate_mae(model_b_results['Actual'], model_b_results['Predicted'])
    
    return {
        'MAE_A': mae_a,
        'MAE_B': mae_b,
        't_statistic': t_statistic,
        'p_value': p_value
    }

# Compare MLP with other models
base_path = 'revision_results'

for cohort in ['none', 'control', 'diseased']:
    mlp_path = f'{base_path}_mlp/predictions_mlp_{cohort}.csv'
    
    print(f"\nComparing models for cohort: {cohort}")
    for model in ['svr', 'ridge', 'rf_test']:
        model_path = f'{base_path}_{model}/predictions_{model}_{cohort}.csv'
        
        print(f"\nComparing MLP with {model.upper()}:")
        results = compare_models(mlp_path, model_path)
        
        print(f"MLP MAE: {results['MAE_A']:.4f}")
        print(f"{model.upper()} MAE: {results['MAE_B']:.4f}")
        print(f"T-statistic: {results['t_statistic']:.4f}")
        print(f"P-value: {results['p_value']:.4f}")
        
        if results['p_value'] < 0.05:
            print(f"The difference between MLP and {model.upper()} is statistically significant.")
        else:
            print(f"There is no statistically significant difference between MLP and {model.upper()}.")