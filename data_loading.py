# data_loading.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_imputation import *
from residualization import *
from utils import *

def load_and_preprocess_data(input_path):
    """
    Load and preprocess the grip dataset.
    
    Args:
    input_path (str): Path to the input CSV file.
    
    Returns:
    pd.DataFrame: Preprocessed dataframe.
    """
    # Read file
    df = pd.read_csv(input_path, sep=',', na_values=["?"])
    df = df.replace({'?': np.NaN})
    
    # Add year column
    df['year'] = pd.DatetimeIndex(df['query__date']).year
    
    # Keep visits before 2021
    df = df[df['year'] < 2021]
    
    # Remove subjects to be excluded
    df = exclude_subjects(df)
    
    # Delete specified features
    tabular_to_delete = ['demo_dob', 'query__date', 'demo_crovitz', 'demo_race', 'demo_handedness', 'demo_hispanic', 'demo_yrs_ed', 'lr_fasting', 'lr_hepatic_agr', 'lr_hepatic_albumin', 'lr_hepatic_alp', 'lr_hepatic_alt', 'lr_hepatic_ast', 'lr_hepatic_bili_total', 'lr_hepatic_ggt', 'lr_hepatic_globulin', 'lr_hepatic_protein_tot', 'lr_wbc_baso', 'lr_wbc_baso_abs', 'lr_wbc_eos', 'lr_wbc_eos_abs', 'lr_wbc_mono', 'lr_wbc_mono_abs', 'lr_wbc_neut', 'lr_wbc_neut_abs', 'med_hx_allergy_e', 'med_hx_asthma_e', 'med_hx_bp_high_e', 'med_hx_cancer_e', 'med_hx_cancer_rad_e', 'med_hx_eye_lenses_e', 'med_hx_eye_prob_e', 'med_hx_hav_e', 'med_hx_hbv_e', 'med_hx_hcv_e', 'med_hx_hcv_rx_e', 'med_hx_heart_disease_e', 'med_hx_hiv_e', 'med_hx_kidney_disease_e', 'med_hx_ldl_high_e', 'med_hx_leg_prob_e', 'med_hx_liver_prob_e', 'med_hx_loc_lt_30m_e', 'med_hx_migraine_e', 'med_hx_resp_dis_e', 'med_hx_std_e', 'med_hx_sum', 'med_hx_tb_e', 'med_hx_tens_headache_e', 'med_hx_tox_occ_exp_e', 'meds_sum', 'np_ff_animal_tot', 'np_ff_bird_color_tot', 'np_ff_object_tot', 'np_reyo_imm_raw', 'np_reyo_recog_raw', 'np_wmsr_logic_imm_tot']
    df = df.drop(tabular_to_delete, axis=1)
    
    # Keep only specified imaging features
    imaging_to_keep = ['sri24_suptent_supratentorium_volume', 'sri24_parc116_cblmhemiwht_l_wm', 'sri24_parc116_cblmhemiwht_r_wm', 'sri24_parc116_pons_wm', 'sri24_parc116_corpus_callosum_wm', 'sri24_parc116_precentral_l_gm', 'sri24_parc116_precentral_r_gm', 'sri24_parc116_insula_l_gm', 'sri24_parc116_insula_r_gm', 'sri24_parc116_caudate_l_gm', 'sri24_parc116_caudate_r_gm', 'sri24_parc116_putamen_l_gm', 'sri24_parc116_putamen_r_gm', 'sri24_parc116_pallidum_l_gm', 'sri24_parc116_pallidum_r_gm', 'sri24_parc116_thalamus_l_gm', 'sri24_parc116_thalamus_r_gm', 'sri24_parc116_vermis_1_gm', 'sri24_parc116_vermis_2_gm', 'sri24_parc116_vermis_3_gm', 'sri24_parc6_cingulate_gm', 'sri24_parc6_cingulate_l_gm', 'sri24_parc6_cingulate_r_gm', 'sri24_parc6_frontal_gm', 'sri24_parc6_frontal_l_gm', 'sri24_parc6_frontal_r_gm', 'sri24_parc6_insula_gm', 'sri24_parc6_insula_l_gm', 'sri24_parc6_insula_r_gm', 'sri24_parc6_occipital_gm', 'sri24_parc6_occipital_l_gm', 'sri24_parc6_occipital_r_gm', 'sri24_parc6_parietal_gm', 'sri24_parc6_parietal_l_gm', 'sri24_parc6_parietal_r_gm', 'sri24_parc6_temporal_gm', 'sri24_parc6_temporal_l_gm', 'sri24_parc6_temporal_r_gm']
    all_imaging_features = [i for i in df.columns if i.startswith('sri24')]
    imaging_to_delete = list(set(all_imaging_features) - set(imaging_to_keep))
    df = df.drop(imaging_to_delete, axis=1)
    
    # Convert sex to binary
    df = text_to_codes(df, categorical=['demo_sex', 'demo_diag'])
    
    # Exclude the HIV patients that we don't use in the study
    df = df[df[("demo_diag")]!=2]
    
    # Keep only first visit
    df = df.loc[df.groupby('subject').demo_age.idxmin()].reset_index(drop=True)
    
    return df


def load_and_preprocess_data_for_training(file_path, preparing=True):
    """Main function to load and preprocess data for training."""
    # Load data
    input_data = pd.read_csv(file_path, sep=',')
    
    # Preprocess data
    input_data = keep_first_visit(input_data)
    input_data = create_aud_binary(input_data)
    input_data = reorder_columns(input_data, 'lr_aud', 4)
    input_data = reorder_columns(input_data, 'lr_hiv', 5)
    
    # Exclude subjects
    input_data = exclude_subjects(input_data)
    
    # Exclude subjects with only HIV diagnosis
    input_data = input_data[input_data.demo_diag != 2]
    
    if preparing:
         # Remove specific columns
        med_to_remove = [
        "med_hx_arthritis_e", "med_hx_back_pain_e", "med_hx_diabetes_e",
        "med_hx_head_injury_e", "med_hx_neuropathy_e", "med_hx_sleep_disorder_e",
        "med_hx_thy_hyper_e", "med_hx_thy_hypo_e", "lr_hbv", "lr_hcv",
        "sri24_parc116_insula_gm_prime"
        ]
        input_data = remove_columns(input_data, med_to_remove)
    
        # Prepare data for training
        X_dataframe, X, y, y_strat = prepare_data_for_training(input_data)
    
        # Print information
        print(f'Features used: {X_dataframe.columns}')
        print(f'Number of features used: {len(X_dataframe.columns)}')
    
        return X_dataframe, X, y, y_strat
    else:
        # Remove specific columns
        med_to_remove = [
        "med_hx_arthritis_e", "med_hx_back_pain_e", "med_hx_diabetes_e",
        "med_hx_head_injury_e", "med_hx_neuropathy_e", "med_hx_sleep_disorder_e",
        "med_hx_thy_hyper_e", "med_hx_thy_hypo_e", "lr_hbv", "lr_hcv"
        ]
        input_data = remove_columns(input_data, med_to_remove)
        y_strat = np.array(input_data['demo_diag'])
        return input_data, y_strat


def load_and_preprocess_data_for_f_tests(input_path):
    input_data = pd.read_csv(input_path, sep=',')
    input_data = input_data[input_data.demo_diag != 2]
    
    input_data = exclude_subjects(input_data)
    
    input_data.loc[input_data['demo_diag'].isin([1, 3]), 'lr_aud'] = 1
    input_data.loc[input_data['demo_diag'].isin([0, 2]), 'lr_aud'] = 0
    
    needed_column = input_data.pop('lr_aud')
    input_data.insert(4, 'lr_aud', needed_column)
    
    return input_data


def preprocess_data_no_leakage(train_data, test_data):
    """Preprocess the training and test data."""
    train_data, test_data = impute_missing_values_no_leakage(train_data, test_data)
    train_data, test_data = residualize_data_no_leakage(train_data, test_data)
    train_data = post_process_data_no_leakage(train_data)
    test_data = post_process_data_no_leakage(test_data)
    return train_data, test_data


def post_process_data_no_leakage(data):
    """Perform post-processing steps on the data."""
    data = sum_hemispheres(data)
    # We have insula twice, one from parc6 and one from parc116
    data = remove_columns(data, ['sri24_parc116_insula_gm_prime'])
    data = move_column_to_end(data, 'mean_grip_prime')
    data = reorder_columns(data, 'lr_aud', 5)
    data = reorder_columns(data, 'lr_hiv', 5)
    return data