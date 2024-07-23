# main.py

import torch
import numpy as np
import pandas as pd
import data_loading
from data_loading import *
from train import old_training_notebook_v2
from utils import set_seed, compare_csv_files

def main():
    # Set random seed for reproducibility
    set_seed(1964)
    
    # Load and preprocess data
    X_dataframe, X, y, y_strat = data_loading.load_and_preprocess_data_for_training('/Users/magdalinipaschali/Documents/stanford/lab_data_code/grip_dataset_processed_apr_18_2023_onlyhead.csv')

    all_predictions, new_shaps_arr_deep, XX = old_training_notebook_v2(X_dataframe, X, y, y_strat)
    print(all_predictions)

if __name__ == "__main__":
    main()