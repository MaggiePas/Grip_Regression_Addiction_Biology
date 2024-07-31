# data_imputation.py

import pandas as pd
import numpy as np
from statistics import mode

def impute_missing_values(df):
    """
    Impute missing values in the dataframe.
    
    Args:
    df (pd.DataFrame): Input dataframe.
    
    Returns:
    pd.DataFrame: Dataframe with imputed values.
    """
    mode_columns = [
        'lr_fasting', 'lr_hiv', 'lr_hcv', 'lr_hbv', 'demo_handedness',
        'med_hx_bp_high_e', 'med_hx_heart_disease_e', 'med_hx_ldl_high_e',
        'med_hx_kidney_disease_e', 'med_hx_hav_e', 'med_hx_hbv_e', 'med_hx_hcv_e',
        'med_hx_hcv_rx_e', 'med_hx_liver_prob_e', 'med_hx_hiv_e', 'med_hx_std_e',
        'med_hx_tb_e', 'med_hx_asthma_e', 'med_hx_resp_dis_e', 'med_hx_tox_occ_exp_e',
        'med_hx_allergy_e', 'med_hx_migraine_e', 'med_hx_tens_headache_e',
        'med_hx_thy_hyper_e', 'med_hx_thy_hypo_e', 'med_hx_diabetes_e',
        'med_hx_head_injury_e', 'med_hx_loc_lt_30m_e', 'med_hx_sleep_disorder_e',
        'med_hx_cancer_e', 'med_hx_cancer_rad_e', 'med_hx_neuropathy_e',
        'med_hx_arthritis_e', 'med_hx_back_pain_e', 'med_hx_leg_prob_e',
        'med_hx_eye_prob_e', 'med_hx_eye_lenses_e', 'qx_audit_total', 'qx_bdi_total'
    ]

    for column_name in df.columns:
        if column_name in mode_columns:
            df[column_name] = df.groupby(['demo_sex', 'demo_diag'])[column_name].transform(
                lambda x: x.fillna(x.median())
            )
        else:
            df[column_name] = df.groupby(['demo_sex', 'demo_diag'])[column_name].transform(
                lambda x: x.fillna(x.mean())
            )

    return df, mode_columns


def impute_missing_values_no_leakage(X_train, X_test):
    """
    Impute missing values in X_train and X_test dataframes.
    
    Args:
    X_train (pd.DataFrame): Training dataframe.
    X_test (pd.DataFrame): Test dataframe.
    
    Returns:
    Imputed X_train and X_test dataframes
    """
    mode_columns = [
        'lr_fasting', 'lr_hiv', 'lr_hcv', 'lr_hbv', 'demo_handedness',
        'med_hx_bp_high_e', 'med_hx_heart_disease_e', 'med_hx_ldl_high_e',
        'med_hx_kidney_disease_e', 'med_hx_hav_e', 'med_hx_hbv_e', 'med_hx_hcv_e',
        'med_hx_hcv_rx_e', 'med_hx_liver_prob_e', 'med_hx_hiv_e', 'med_hx_std_e',
        'med_hx_tb_e', 'med_hx_asthma_e', 'med_hx_resp_dis_e', 'med_hx_tox_occ_exp_e',
        'med_hx_allergy_e', 'med_hx_migraine_e', 'med_hx_tens_headache_e',
        'med_hx_thy_hyper_e', 'med_hx_thy_hypo_e', 'med_hx_diabetes_e',
        'med_hx_head_injury_e', 'med_hx_loc_lt_30m_e', 'med_hx_sleep_disorder_e',
        'med_hx_cancer_e', 'med_hx_cancer_rad_e', 'med_hx_neuropathy_e',
        'med_hx_arthritis_e', 'med_hx_back_pain_e', 'med_hx_leg_prob_e',
        'med_hx_eye_prob_e', 'med_hx_eye_lenses_e', 'qx_audit_total', 'qx_bdi_total'
    ]
    
    for column_name in X_train.columns:
        if column_name in mode_columns:
            # For mode columns, use median
            imputation_values = X_train.groupby(['demo_sex', 'demo_diag'])[column_name].median()
            
            # Impute X_train
            X_train[column_name] = X_train.groupby(['demo_sex', 'demo_diag'])[column_name].transform(
                lambda x: x.fillna(x.median())
            )
            
            # Impute X_test
            for (sex, diag), value in imputation_values.items():
                mask = (X_test['demo_sex'] == sex) & (X_test['demo_diag'] == diag)
                X_test.loc[mask, column_name] = X_test.loc[mask, column_name].fillna(value)
        
        else:
            # For other columns, use mean
            imputation_values = X_train.groupby(['demo_sex', 'demo_diag'])[column_name].mean()
            
            # Impute X_train
            X_train[column_name] = X_train.groupby(['demo_sex', 'demo_diag'])[column_name].transform(
                lambda x: x.fillna(x.mean())
            )
            
            # Impute X_test
            for (sex, diag), value in imputation_values.items():
                mask = (X_test['demo_sex'] == sex) & (X_test['demo_diag'] == diag)
                X_test.loc[mask, column_name] = X_test.loc[mask, column_name].fillna(value)
    
    return X_train, X_test