import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from utils import set_seed
import scipy.stats as stats

def perform_f_test(data, formula):
    md3 = smf.rlm(formula=formula, data=data, M=sm.robust.norms.HuberT()).fit()
    A = np.identity(len(md3.params))
    A = A[1:,:]
    fres = md3.f_test(A)
    return fres.fvalue, fres.pvalue


def run_f_test_double_dissociation(input_data, control_formula, diseased_formula):

    patient_cohort = input_data[input_data["demo_diag"] > 0]
    controls = input_data[input_data["demo_diag"] == 0]
    
    # Features of controls on control group
    f_value, p_value = perform_f_test(controls, control_formula)
    print('Features of controls on control group')
    print(f'F-value: {f_value:.2f}, p-value: {p_value:.3f}')
    
    # Features of controls on diseased group
    f_value, p_value = perform_f_test(patient_cohort, control_formula)
    print('Features of controls on diseased group')
    print(f'F-value: {f_value:.2f}, p-value: {p_value:.3f}')
    
    # Features of diseased on diseased group
    f_value, p_value = perform_f_test(patient_cohort, diseased_formula)
    print('Features of diseased on diseased group')
    print(f'F-value: {f_value:.2f}, p-value: {p_value:.3f}')
    
    # Features of diseased on control group
    f_value, p_value = perform_f_test(controls, diseased_formula)
    print('Features of diseased on control group')
    print(f'F-value: {f_value:.2f}, p-value: {p_value:.3f}')


def run_f_test_10_times(input_data, formula, num_iterations=10):
    # Setting the seed
    # 706, 2318, 3140
    
    set_seed(706)
    patient_cohort = input_data[input_data["demo_diag"] > 0]
    controls = input_data[input_data["demo_diag"] == 0]
    
    count_tiny = 0
    count_sign = 0
    count_non_sign = 0
    max_p_value = -1
    
    for _ in range(num_iterations):
        sampled_cohort = patient_cohort.sample(n=len(controls))
        f_value, p_value = perform_f_test(sampled_cohort, formula)
        
        if p_value >= max_p_value:
            max_p_value = p_value
            
        if p_value < 0.001:
            count_tiny += 1
        elif p_value <= 0.05:
            count_sign += 1
        else:
            count_non_sign += 1
    
    return count_tiny, count_sign, count_non_sign, max_p_value

def print_results(count_tiny, count_sign, count_non_sign):
    print(f'Under 0.001: {count_tiny}')
    print(f'Under 0.05: {count_sign}')
    print(f'Non-significant: {count_non_sign}')
    

def process_features(control_features, diseased_features):
    """
    Process the feature lists to find common and unique features,
    and create formulas for each group.
    
    Args:
    control_features (list): List of features for the control group
    diseased_features (list): List of features for the diseased group
    
    Returns:
    tuple: Formulas for control and diseased groups, common features, unique features for each group
    """
    common_features = list(set(control_features) & set(diseased_features))
    unique_control = list(set(control_features) - set(diseased_features))
    unique_diseased = list(set(diseased_features) - set(control_features))
    
    control_formula = "mean_grip_prime ~ " + " + ".join(unique_control)
    diseased_formula = "mean_grip_prime ~ " + " + ".join(unique_diseased)
    
    return control_formula, diseased_formula, common_features, unique_control, unique_diseased


def correlation_analysis(input_data, control_features, diseased_features, p_threshold=0.05):
    """
    Perform correlation analysis between features and grip strength for each cohort.
    
    Args:
    input_data (pd.DataFrame): The full dataset
    control_features (list): List of features significant for the control group
    diseased_features (list): List of features significant for the diseased group
    p_threshold (float): Significance threshold for correlations
    
    Returns:
    dict: Features to plot (significantly correlated with one group but not the other)
    """
    control_data = input_data[input_data["demo_diag"] == 0]
    diseased_data = input_data[input_data["demo_diag"] > 0]
    
    features_to_plot = {}
    
    def compute_correlation(data, feature):
        r, p = stats.pearsonr(data[feature], data['mean_grip_prime'])
        return r, p
    
    for feature in set(control_features + diseased_features):
        control_r, control_p = compute_correlation(control_data, feature)
        diseased_r, diseased_p = compute_correlation(diseased_data, feature)
        
        if feature in control_features and control_p < p_threshold and diseased_p >= p_threshold:
            features_to_plot[feature] = {
                'control_r': control_r, 'control_p': control_p,
                'diseased_r': diseased_r, 'diseased_p': diseased_p,
                'group': 'control'
            }
        elif feature in diseased_features and diseased_p < p_threshold and control_p >= p_threshold:
            features_to_plot[feature] = {
                'control_r': control_r, 'control_p': control_p,
                'diseased_r': diseased_r, 'diseased_p': diseased_p,
                'group': 'diseased'
            }
    
    return features_to_plot