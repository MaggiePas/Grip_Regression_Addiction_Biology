# main.py

import torch
import numpy as np
import pandas as pd
import data_loading
from data_loading import *
from train import train_and_evalute
from utils import set_seed

def main():
    # Set random seed for reproducibility
    set_seed(1964)
    
    # Load and preprocess data
    X_dataframe, X, y, y_strat = data_loading.load_and_preprocess_data_for_training('/Users/magdalinipaschali/Documents/stanford/lab_data_code/grip_dataset_processed_apr_18_2023_onlyhead.csv')

    all_predictions, new_shaps_arr_deep, XX = train_and_evalute(X_dataframe, X, y, y_strat, finetune_on='control')
    print(all_predictions)

if __name__ == "__main__":
    main()