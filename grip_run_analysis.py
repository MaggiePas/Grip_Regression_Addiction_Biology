# main.py - Run the whole analysis for the paper

import numpy as np
import pandas as pd
import data_loading
from data_loading import *
from train import train_and_evalute
from utils import set_seed
from plotting import correlation_plot, plot_feature_correlation
from f_test import *
import matplotlib.pyplot as plt
from plot_config import PLOT_INFO

def main():
    # Set random seed for reproducibility
    set_seed(1964)
    
    # Load and preprocess data
    X_dataframe, X, y, y_strat = data_loading.load_and_preprocess_data_for_training('/Users/magdalinipaschali/Documents/stanford/lab_data_code/grip_dataset_processed_apr_18_2023_onlyhead.csv')

    # Select cohort for model finetuning. Options are: none, control, diseased
    # finetune_cohort none simply means that the model is trained on everyone and has not been finetuned on any specific cohort
    
    print('-----------------------------Step 1--------------------------------')  
    # Step 1: Finetune model on controls
    finetune_cohort = 'control' 
    
    # Train and evaluate model. Generate SHAP plots and values for the trained/finetuned model
    all_predictions, top_6_features_list_controls = train_and_evalute(X_dataframe, X, y, y_strat, finetune_on=finetune_cohort)
    
    # Generate correlation plots and statistics between actual and predicted grip strength
    correlation_plot(all_predictions=all_predictions, cohort=finetune_cohort, save_path=f'revision_results/model_correlation/correlation_plot_{finetune_cohort}.png')
    
    print('-----------------------------Step 2--------------------------------')  
    # Step 2: Finetune model on diseased
    finetune_cohort = 'diseased' 
    
    plt.close()
    # Train and evaluate model. Generate SHAP plots and values for the trained/finetuned model
    all_predictions, top_6_features_list_diseased = train_and_evalute(X_dataframe, X, y, y_strat, finetune_on=finetune_cohort)
    
    # Generate correlation plots and statistics between actual and predicted grip strength
    correlation_plot(all_predictions=all_predictions, cohort=finetune_cohort, save_path=f'revision_results/model_correlation/correlation_plot_{finetune_cohort}.png')

    print('-----------------------------Step 3--------------------------------')  
    # Step 3: Find common and unique features between control and diseased groups based on SHAP values and create formulas for F-tests
    control_formula, diseased_formula, common_features, unique_control, unique_diseased = process_features(top_6_features_list_controls, top_6_features_list_diseased)
    print(f'Common features: {common_features}')
    print("Unique control features:", unique_control)
    print("Unique diseased features:", unique_diseased)
    
    print('-----------------------------Step 4--------------------------------')  
    # Step 4: Perfomr F-tests on the unique features for each group and print the results hoping to get double dissociation
    f_data = load_and_preprocess_data_for_f_tests('/Users/magdalinipaschali/Documents/stanford/lab_data_code/grip_dataset_processed_apr_18_2023_onlyhead.csv')
    
    run_f_test_double_dissociation(f_data, control_formula, diseased_formula)
    
    print('-----------------------------Step 5--------------------------------')    
    # Step 5: Repeat F-tests 10 times undersampling the AUD to match the control group and print the results
    count_tiny, count_sign, count_non_sign, max_p_value = run_f_test_10_times(f_data, diseased_formula)
    
    print(f"Max p-value after 10 iterations: {max_p_value:.2f}")
    print_results(count_tiny, count_sign, count_non_sign)

    print('-----------------------------Step 6--------------------------------') 
    # Step 6: Find the features of controls that are significantly correlated with the grip of the controls and not with the grip of the diseased and vice versa
    
    features_to_plot = correlation_analysis(f_data, unique_control, unique_diseased)
    print("\nFeatures to plot:")
    for feature, info in features_to_plot.items():
        print(f"{feature}:")
        print(f"  Control: r = {info['control_r']:.2f}, p = {info['control_p']:.3f}")
        print(f"  Diseased: r = {info['diseased_r']:.2f}, p = {info['diseased_p']:.3f}")
        print(f"  Significant for: {info['group']}")
    
    # Plot correlations for selected features
    for feature, info in features_to_plot.items():
        if feature in PLOT_INFO:
            plot_feature_correlation(f_data, feature, PLOT_INFO[feature], f"revision_results/feature_correlation/{feature}_correlation.png", top_6_features_list_controls, top_6_features_list_diseased)
        else:
            print(f"Warning: Plot information not found for feature {feature}. Please add them in plot_config.py")
    
if __name__ == "__main__":
    main()