# main.py

import torch
import numpy as np
import pandas as pd
import data_loading
from data_loading import *
from train import train_and_evalute
from utils import set_seed
from plotting import correlation_plot

def main():
    # Set random seed for reproducibility
    set_seed(1964)
    
    # Load and preprocess data
    X_dataframe, X, y, y_strat = data_loading.load_and_preprocess_data_for_training('/Users/magdalinipaschali/Documents/stanford/lab_data_code/grip_dataset_processed_apr_18_2023_onlyhead.csv')

    # Select cohort for model finetuning. Options are: none, control, diseased
    finetune_cohort = 'none' # finetune_cohort none simply means that the model is trained on everyone and has not been finetuned on any specific cohort
    
    # Train and evaluate model. Generate SHAP plots and values for the trained/finetuned model
    all_predictions, new_shaps_arr_deep, XX = train_and_evalute(X_dataframe, X, y, y_strat, finetune_on=finetune_cohort)
    print(all_predictions)
    
    # Generate correlation plots and statistics between actual and predicted grip strength
    correlation_plot(all_predictions=all_predictions, cohort=finetune_cohort, save_path=f'revision_results/correlation_plot_{finetune_cohort}.png')

if __name__ == "__main__":
    main()