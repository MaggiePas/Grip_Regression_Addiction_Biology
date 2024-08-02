import pandas as pd
import numpy as np
from scipy import stats
from utils import set_seed

def permutation_test(df, n_permutations=1000):
    set_seed(1964)
    # Binarize the diagnosis
    df['Binary_Diagnosis'] = (df['Diagnosis'] > 0).astype(int)
    
    # Calculate actual correlations
    control_corr = stats.pearsonr(df[df['Binary_Diagnosis'] == 0]['Actual'], df[df['Binary_Diagnosis'] == 0]['Predicted'])[0]
    diseased_corr = stats.pearsonr(df[df['Binary_Diagnosis'] == 1]['Actual'], df[df['Binary_Diagnosis'] == 1]['Predicted'])[0]
    
    # Initialize counters
    control_count = 0
    diseased_count = 0
    
    for _ in range(n_permutations):
        # Shuffle all actual grip strength values
        df['Shuffled_Actual'] = np.random.permutation(df['Actual'])
        
        # Calculate correlations for shuffled actual values
        shuffled_control_corr = stats.pearsonr(df[df['Binary_Diagnosis'] == 0]['Shuffled_Actual'], 
                                               df[df['Binary_Diagnosis'] == 0]['Predicted'])[0]
        shuffled_diseased_corr = stats.pearsonr(df[df['Binary_Diagnosis'] == 1]['Shuffled_Actual'], 
                                                df[df['Binary_Diagnosis'] == 1]['Predicted'])[0]
        
        # Count times shuffled correlation is higher than or equal to actual
        if abs(shuffled_control_corr) >= abs(control_corr):
            control_count += 1
        if abs(shuffled_diseased_corr) >= abs(diseased_corr):
            diseased_count += 1
    
    # Calculate p-values
    control_p = control_count / n_permutations
    diseased_p = diseased_count / n_permutations
    
    return control_corr, control_p, diseased_corr, diseased_p

# Read the CSV file

cohort = 'diseased' #control

df = pd.read_csv(f'revision_results_mlp_original/predictions_mlp_original_{cohort}.csv')

# Perform permutation test
control_corr, control_p, diseased_corr, diseased_p = permutation_test(df)

print(f"Control Group Actual Correlation: {control_corr:.4f}")
print(f"Control Group Permutation P-value: {control_p:.4f}")
print(f"Diseased Group Actual Correlation: {diseased_corr:.4f}")
print(f"Diseased Group Permutation P-value: {diseased_p:.4f}")