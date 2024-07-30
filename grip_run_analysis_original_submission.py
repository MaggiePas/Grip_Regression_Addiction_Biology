# main.py - Run the whole analysis for the paper

import numpy as np
import pandas as pd
from data_loading import *
from utils import set_seed
from plotting import correlation_plot, plot_feature_correlation, plot_feature_boxplot, plot_feature_correlation_hiv
from f_test import *
import matplotlib.pyplot as plt
from plot_config import PLOT_INFO

def main():
    # Set random seed for reproducibility
    set_seed(1964)
    model_type = 'original'
    check_create_paths(model_type)
        
    print('-----------------------------Step 1--------------------------------')  
    # Step 1: Finetune model on controls
    finetune_cohort = 'control'  # Options are: none (train on everyone), control, diseased
    
    all_predictions_controls = pd.read_csv('data_files/original_predictions/predictions_finetuned_controls_12_4.csv')

    top_6_features_list_controls = ['sri24_parc116_cblmhemiwht_wm_prime', 'lr_cbc_mchc_prime', 'np_wmsr_dig_back_span_prime', 'sri24_parc116_precentral_gm_prime', 'phys_bp_diastolic_prime', 'lr_cbc_mpv_prime', ]
    
    # Generate correlation plots and statistics between actual and predicted grip strength
    correlation_plot(all_predictions=all_predictions_controls, cohort=finetune_cohort, save_path=f'revision_results_original/model_correlation/correlation_plot_{finetune_cohort}.png')
    
    print('-----------------------------Step 2--------------------------------')  
    # Step 2: Finetune model on diseased
    finetune_cohort = 'diseased' 
    
    all_predictions_diseased = pd.read_csv('data_files/original_predictions/predictions_finetuned_diseased_12_4.csv')

    top_6_features_list_diseased = ['sri24_parc116_cblmhemiwht_wm_prime', 'lr_cbc_mchc_prime', 'sri24_parc6_insula_gm_prime' , 'lr_hiv', 'qx_hf_socfxn_prime', 'lr_cbc_plt_prime', ]
     
    # Generate correlation plots and statistics between actual and predicted grip strength
    correlation_plot(all_predictions=all_predictions_diseased, cohort=finetune_cohort, save_path=f'revision_results_original/model_correlation/correlation_plot_{finetune_cohort}.png')

    print('-----------------------------Step 3--------------------------------')  
    # Step 3: Find common and unique features between control and diseased groups based on SHAP values and create formulas for F-tests
    control_formula, diseased_formula, common_features, unique_control, unique_diseased = process_features(top_6_features_list_controls, top_6_features_list_diseased)
    print(f'Common features: {common_features}')
    print("Unique control features:", unique_control)
    print("Unique diseased features:", unique_diseased)
    
    print('-----------------------------Step 4--------------------------------')  
    # Step 4: Perfomr F-tests on the unique features for each group and print the results hoping to get double dissociation
    f_data = load_and_preprocess_data_for_f_tests('data_files/grip_dataset_processed_apr_18_2023_onlyhead.csv')
    
    run_f_test_double_dissociation(f_data, control_formula, diseased_formula)
    
    print('-----------------------------Step 5--------------------------------')    
    # Step 5: Repeat F-tests 10 times undersampling the AUD to match the control group and print the results
    count_tiny, count_sign, count_non_sign, max_p_value = run_f_test_10_times(f_data, diseased_formula)
    
    print(f"Max p-value after 10 iterations: {max_p_value:.2f}")
    print_results(count_tiny, count_sign, count_non_sign)

    print('-----------------------------Step 6--------------------------------') 
    # Step 6: Find the features of controls that are significantly correlated with the grip of the controls and not with the grip of the diseased and vice versa
    
    features_to_plot = correlation_analysis(f_data, top_6_features_list_controls, top_6_features_list_diseased )
    print("\nFeatures to plot:")
    for feature, info in features_to_plot.items():
        print(f"{feature}:")
        print(f"  Control: r = {info['control_r']:.2f}, p = {info['control_p']:.3f}")
        print(f"  Diseased: r = {info['diseased_r']:.2f}, p = {info['diseased_p']:.3f}")
        print(f"  Significant for: {info['group']}")
    
    # Plot correlations for selected features
    for feature, info in features_to_plot.items():
        if feature in PLOT_INFO:
            plot_feature_correlation(f_data, feature, PLOT_INFO[feature], f"revision_results_original/feature_correlation/{feature}_correlation.png", top_6_features_list_controls, top_6_features_list_diseased)
        else:
            print(f"Warning: Plot information not found for feature {feature}. Please add them in plot_config.py")
            
    print('-----------------------------Step 7--------------------------------') 
    # Step 6: Generate Figure 3 for the paper, boxplots comparing selected features between subjects with and without HIV        
    # For cerebellum
    plot_feature_boxplot(
        input_data=f_data,
        feature='cerebellum_cc',
        output_path="revision_results_original/figure_3/hiv_grip_cerebellum.png",
        y_label="Cerebellum (cc)",
        y_ticks=[6, 7, 8, 9, 10]
    )

    # For platelet count
    plot_feature_boxplot(
        input_data=f_data,
        feature='lr_cbc_plt_prime',
        output_path="revision_results_original/figure_3/hiv_grip_platelet_count.png",
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
        output_path="revision_results_original/figure_supplementary/hiv_grip_cerebellum_correlation.png",
        y_label="Cerebellum (cc)",
        y_ticks=[6, 7, 8, 9, 10]
    )

    # For platelet count
    plot_feature_correlation_hiv(
        input_data=f_data,
        feature='lr_cbc_plt_prime',
        output_path="revision_results_original/figure_supplementary/hiv_grip_platelet_count_correlation.png",
        y_label="Platelet Count (thou/ul)",
        y_ticks=[100, 200, 300, 400]
    )
        
if __name__ == "__main__":
    main()