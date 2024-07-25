# grip_dataset_preprocessing.py

import pandas as pd
import numpy as np
import data_loading
from data_imputation import *
from residualization import *
from utils import *

import sys
print(sys.path)

def main():
    # Load and preprocess data
    input_path = 'data_files/grip_dataset_no_processing_feb_6_2023.csv'
    output_path = 'data_files/'
    
    df = data_loading.load_and_preprocess_data(input_path)
    
    # Impute missing values
    # df = impute_missing_values(df)
    
    # Residualize grip strength
    df = residualize_grip_strength(df)
    
    # Residualize non-imaging measurements
    df = residualize_measurements(df)
    
    # Residualize imaging measurements
    df = residualize_imaging(df)
    
    # Sum left and right hemisphere values
    df = sum_hemispheres(df)
    
    # Move grip strength to the last column
    df = move_column_to_end(df, 'mean_grip_prime')
    
    # Save the processed dataset
    df.to_csv(f'{output_path}/grip_dataset_processed_7_25_no_imputation.csv', index=False)

if __name__ == "__main__":
    main()