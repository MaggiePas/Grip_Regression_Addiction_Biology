import pandas as pd
import numpy as np
from scipy import stats

# Load the data
df = pd.read_csv('revision_results_mlp_original/predictions_mlp_original_diseased.csv')

# Separate the data into two groups
group1 = df[df['Diagnosis'] == 1]
group3 = df[df['Diagnosis'] == 3]

# Calculate prediction errors for each group
errors1 = group1['Actual'] - group1['Predicted']
errors3 = group3['Actual'] - group3['Predicted']

# Perform two-sample t-test
t_statistic, p_value = stats.ttest_ind(errors1, errors3)

print(f"Number of samples in Group 1 (AUD without HIV): {len(group1)}")
print(f"Number of samples in Group 3 (AUD with HIV): {len(group3)}")
print(f"t-statistic: {t_statistic:.3f}")
print(f"p-value: {p_value:.3f}")