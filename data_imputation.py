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
    
    # Exclude the HIV patients that we don't use in the study
    df = df[df[("demo_diag")]!=2]

    for column_name in df.columns:
        if column_name in mode_columns:
            df[column_name] = df.groupby(['demo_sex', 'demo_diag'])[column_name].transform(
                lambda x: x.fillna(x.median())
            )
        else:
            df[column_name] = df.groupby(['demo_sex', 'demo_diag'])[column_name].transform(
                lambda x: x.fillna(x.mean())
            )

    # Keep only the first available visit per subject
    df = df.loc[df.groupby('subject').demo_age.idxmin()].reset_index(drop=True)

    return df