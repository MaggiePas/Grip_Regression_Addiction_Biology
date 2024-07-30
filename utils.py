# utils.py

import pandas as pd
import random
import numpy as np
import torch
import os

def text_to_codes(df, categorical):
    """
    Convert text categories to numeric codes.
    
    Args:
    df (pd.DataFrame): Input dataframe.
    categorical (list): List of categorical columns.
    
    Returns:
    pd.DataFrame: Dataframe with converted categories.
    """
    # Variables in categorical format that need to be converted
    # categorical = ['site', 'sex', 'cahalan_score', 'scanner', 'scanner_model', 'hispanic']
    # categorical = ['site', 'sex', 'cahalan_score', 'hispanic']

    for category in categorical:
        df[category] = df[category].astype('category')

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    
    return df


def sum_hemispheres(df):
    """
    Sum left and right hemisphere values.
    
    Args:
    df (pd.DataFrame): Input dataframe.
    
    Returns:
    pd.DataFrame: Dataframe with summed hemisphere values.
    """
    remove = []
    for name in df.columns:
        if "_l_wm" in name:
            r_name = name.replace("_l_wm", "_r_wm")
            remove.extend([name, r_name])
            new_name = name.replace("_l_wm", "_wm")
            df[new_name] = (df[name] + df[r_name]) / 2
    df = df.drop(remove, axis=1)
    
    remove = []
    for name in df.columns:
        if "_l_gm" in name:
            r_name = name.replace("_l_gm", "_r_gm")
            remove.extend([name, r_name])
            new_name = name.replace("_l_gm", "_gm")
            df[new_name] = (df[name] + df[r_name]) / 2
    df = df.drop(remove, axis=1)
    
    return df

def move_column_to_end(df, column_name):
    """
    Move a column to the end of the dataframe.
    
    Args:
    df (pd.DataFrame): Input dataframe.
    column_name (str): Name of the column to move.
    
    Returns:
    pd.DataFrame: Dataframe with the column moved to the end.
    """
    column = df.pop(column_name)
    df[column_name] = column
    return df


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    
    
def keep_first_visit(df):
    """Keep only the first available visit per subject."""
    return df.loc[df.groupby('subject').demo_age.idxmin()].reset_index(drop=True)


def create_aud_binary(df):
    """Create binary variable for AUD diagnosis."""
    df['lr_aud'] = df['demo_diag'].map({0: 0, 1: 1, 2: 0, 3: 1})
    return df


def reorder_columns(df, column_name, position):
    """Move a column to a specific position in the dataframe."""
    col = df.pop(column_name)
    df.insert(position, column_name, col)
    return df


def exclude_subjects(df):
    """Exclude specific subjects from the dataframe."""
    subjects_to_exclude = [917, 313, 391, 232, 95, 412, 401, 234, 223, 71, 376, 269, 386, 176, 388, 355, 1352]
    return df[~df['subject'].isin(subjects_to_exclude)]


def remove_columns(df, columns_to_remove):
    """Remove specified columns from the dataframe."""
    return df.drop(columns_to_remove, axis=1)


def prepare_data_for_training(df):
    """Prepare data for training by creating X, y, and y_strat."""
    X = np.array(df.iloc[:,:-1])
    X_dataframe = df.drop(["mean_grip_prime", "demo_diag", "demo_sex", "demo_ses", "subject"], axis=1)
    y = np.array(df['mean_grip_prime'])
    y_strat = np.array(df['demo_diag'])
    return X_dataframe, X, y, y_strat
    

def compare_csv_files(file1_path, file2_path):
       # Load the CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Get common columns
    common_columns = list(set(df1.columns) & set(df2.columns))

    print(f"Common columns: {common_columns}")

    # Dictionary to store differing rows for each column
    different_rows = {}
    
    # Dictionary to store average values
    average_values = {'file1': {}, 'file2': {}}

    # Compare common columns
    for column in common_columns:
        if df1[column].dtype != df2[column].dtype:
            print(f"Column {column} has different data types: {df1[column].dtype} vs {df2[column].dtype}")
        elif not np.array_equal(df1[column], df2[column]):
            print(f"Column {column} has different values")
            # Get the rows where values differ
            mask = df1[column] != df2[column]
            different_rows[column] = {
                'file1': df1.loc[mask, common_columns],
                'file2': df2.loc[mask, common_columns]
            }
        else:
            print(f"Column {column} is identical in both files")
        
        # Calculate average for numeric columns
        if pd.api.types.is_numeric_dtype(df1[column]) and pd.api.types.is_numeric_dtype(df2[column]):
            average_values['file1'][column] = df1[column].mean()
            average_values['file2'][column] = df2[column].mean()

    # Check for columns unique to each file
    only_in_file1 = set(df1.columns) - set(df2.columns)
    only_in_file2 = set(df2.columns) - set(df1.columns)

    if only_in_file1:
        print(f"Columns only in file1: {only_in_file1}")
    if only_in_file2:
        print(f"Columns only in file2: {only_in_file2}")

    return different_rows, average_values


def check_create_paths(model_type):
    # Create all directories in the path if they don't exist
    full_path = f'revision_results_{model_type}/'
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    full_path = f'revision_results_{model_type}/feature_correlation/'
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    full_path = f'revision_results_{model_type}/model_correlation/'
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    full_path = f'revision_results_{model_type}/shap_barplots/'
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if model_type == 'mlp' or model_type == 'original':
        full_path = f'revision_results_{model_type}/figure_3/'
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        full_path = f'revision_results_{model_type}/figure_supplementary/'
        os.makedirs(os.path.dirname(full_path), exist_ok=True)