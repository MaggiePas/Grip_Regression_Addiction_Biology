import pandas as pd
import numpy as np
from scipy import stats

def fisher_r_to_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_correlations(r1, r2, n1, n2):
    z1 = fisher_r_to_z(r1)
    z2 = fisher_r_to_z(r2)
    se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
    z = (z1 - z2) / se_diff
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

# Load the data
cohort = 'diseased' #control
df = pd.read_csv(f'revision_results_mlp_original/predictions_mlp_original_{cohort}.csv')

# Separate the data into two groups
control_group = df[df['Diagnosis'] == 0]
aud_group = df[df['Diagnosis'] > 0]

# Calculate correlations for each group
r_control, p_control = stats.pearsonr(control_group['Actual'], control_group['Predicted'])
r_aud, p_aud = stats.pearsonr(aud_group['Actual'], aud_group['Predicted'])

# Compare correlations
z, p = compare_correlations(r_control, r_aud, len(control_group), len(aud_group))

print(f"Correlation coefficient for Control Group (Diagnosis = 0): {r_control:.3f}, p={p_control:.3f}")
print(f"Correlation coefficient for AUD Group (Diagnosis > 0): {r_aud:.3f}, p={p_aud:.3f}")
print(f"Z-statistic: {z:.3f}")
print(f"p-value: {p:.3f}")