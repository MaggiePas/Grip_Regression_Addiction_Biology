# main.py

import torch
import numpy as np
import pandas as pd
import data_loading
from data_loading import *
from train import train_and_evalute
from utils import set_seed
from plotting import correlation_plot
from f_test import *

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
    correlation_plot(all_predictions=all_predictions, cohort=finetune_cohort, save_path=f'revision_results/correlation_plot_{finetune_cohort}.png')
    
    print('-----------------------------Step 2--------------------------------')  
    # Step 2: Finetune model on diseased
    finetune_cohort = 'diseased' 
    
    # Train and evaluate model. Generate SHAP plots and values for the trained/finetuned model
    all_predictions, top_6_features_list_diseased = train_and_evalute(X_dataframe, X, y, y_strat, finetune_on=finetune_cohort)
    
    # Generate correlation plots and statistics between actual and predicted grip strength
    correlation_plot(all_predictions=all_predictions, cohort=finetune_cohort, save_path=f'revision_results/correlation_plot_{finetune_cohort}.png')

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

if __name__ == "__main__":
    main()