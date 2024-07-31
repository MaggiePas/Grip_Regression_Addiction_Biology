# main.py - Run the whole analysis for the paper

import data_loading
from data_loading import *
from utils import *
from plotting import correlation_plot, plot_feature_correlation, plot_feature_boxplot, plot_feature_correlation_hiv
from f_test import *
import matplotlib.pyplot as plt
from plot_config import PLOT_INFO
from train import train_and_evalute, train_and_evaluate_traditional_model

def main():
    # Set random seed for reproducibility
    set_seed(1964)
    
    # Load and preprocess data
    data_path = 'data_files/grip_dataset_processed_7_29_no_imputation_no_residualization.csv'

    X_dataframe, y_strat = data_loading.load_and_preprocess_data_for_training(data_path, preparing=False)

    model_type = 'svr' # Options are 'mlp', 'svr', 'ridge', 'rf'
    
    # Create all directories in the path if they don't exist for this model type
    check_create_paths(model_type)
    
    steps = [
        ('Step 0: Train on everyone', 'none'),
        ('Step 1: Finetune model on controls', 'control'),
        ('Step 2: Finetune model on diseased', 'diseased')
    ]

    for step_name, finetune_cohort in steps:
        print(f'-----------------------------{step_name}--------------------------------')
        top_features = run_model(model_type, finetune_cohort, X_dataframe, y_strat)
        if finetune_cohort == 'control':
            top_6_features_list_controls = top_features
        elif finetune_cohort == 'diseased':
            top_6_features_list_diseased = top_features

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
            plot_feature_correlation(f_data, feature, PLOT_INFO[feature], f"revision_results_{model_type}/feature_correlation/{feature}_correlation.png", top_6_features_list_controls, top_6_features_list_diseased)
        else:
            print(f"Warning: Plot information not found for feature {feature}. Please add them in plot_config.py")
            
    print('-----------------------------Step 7--------------------------------') 
    # Step 6: Generate Figure 3 for the paper, boxplots comparing selected features between subjects with and without HIV        
    # For cerebellum
    plot_feature_boxplot(
        input_data=f_data,
        feature='cerebellum_cc',
        output_path="revision_results_mlp/figure_3/hiv_grip_cerebellum.png",
        y_label="Cerebellum (cc)",
        y_ticks=[6, 7, 8, 9, 10]
    )

    # For platelet count
    plot_feature_boxplot(
        input_data=f_data,
        feature='lr_cbc_plt_prime',
        output_path="revision_results_mlp/figure_3/hiv_grip_platelet_count.png",
        y_label="Platelet Count (thou/ul)",
        y_ticks=[100, 200, 300, 400]
    )
    
    print('-----------------------------Step 8--------------------------------') 
    # Step 6: Generate Figure for the supplementary, correlation plots between selected features and grip for subjects with and without HIV        
    print('Generated Supplementary Figures')
    # For cerebellum
    plot_feature_correlation_hiv(
        input_data=f_data,
        feature='cerebellum_cc',
        output_path="revision_results_mlp/figure_supplementary/hiv_grip_cerebellum_correlation.png",
        y_label="Cerebellum (cc)",
        y_ticks=[6, 7, 8, 9, 10]
    )

    # For platelet count
    plot_feature_correlation_hiv(
        input_data=f_data,
        feature='lr_cbc_plt_prime',
        output_path="revision_results_mlp/figure_supplementary/hiv_grip_platelet_count_correlation.png",
        y_label="Platelet Count (thou/ul)",
        y_ticks=[100, 200, 300, 400]
    )
    
    
def run_model(model_type, finetune_cohort, X_dataframe, y_strat):
    if model_type == 'mlp':
        all_predictions, top_features = train_and_evalute(X_dataframe, y_strat, finetune_on=finetune_cohort)
    else:
        all_predictions, top_features = train_and_evaluate_traditional_model(X_dataframe, y_strat, train_on=finetune_cohort, model_type=model_type)
    
    all_predictions.to_csv(f'revision_results_{model_type}/predictions_{model_type}_{finetune_cohort}.csv')
    correlation_plot(all_predictions=all_predictions, cohort=finetune_cohort, 
                     save_path=f'revision_results_{model_type}/model_correlation/correlation_plot_{finetune_cohort}.png')
    
    return top_features


if __name__ == "__main__":
    main()