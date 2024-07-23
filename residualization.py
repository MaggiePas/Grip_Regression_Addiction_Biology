# residualization.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def residualize_grip_strength(df):
    """
    Residualize grip strength.
    
    Args:
    df (pd.DataFrame): Input dataframe.
    
    Returns:
    pd.DataFrame: Dataframe with residualized grip strength.
    """
    df['mean_grip'] = df[['np_grip_xl_calc', 'np_grip_xr_calc']].mean(axis=1)
    
    md = smf.mixedlm("mean_grip ~ demo_sex + demo_ses", df, groups=df["subject"])
    md = md.fit(reml=True)
    
    max_sex = 0
    mean_ses = np.mean(df['demo_ses'])
    
    df['mean_grip_prime'] = df['mean_grip'] - (
        md.params[1] * (df["demo_sex"] - max_sex) + 
        md.params[2] * (df["demo_ses"] - mean_ses)
    )
    
    return df

def residualize_measurements(df):
    """
    Residualize non-imaging measurements.
    
    Args:
    df (pd.DataFrame): Input dataframe.
    
    Returns:
    pd.DataFrame: Dataframe with residualized measurements.
    """
    non_imaging_measurements = [
        col for col in df.columns 
        if not col.startswith('sri24') and not col.endswith('wm')
        and not col.endswith('prime') and col not in [
            'subject', 'demo_age', 'demo_ses', 'demo_sex', 'demo_diag',
            'np_grip_xl_calc', 'np_grip_xr_calc', 'lr_hbv', 'lr_hcv', 'lr_hiv',
            'med_hx_arthritis_e', 'med_hx_back_pain_e', 'med_hx_diabetes_e',
            'med_hx_head_injury_e', 'med_hx_neuropathy_e', 'med_hx_thy_hyper_e',
            'med_hx_thy_hypo_e', 'med_hx_sleep_disorder_e', 'year', 'mean_grip'
        ]
    ]
    
    max_sex = 0
    mean_ses = np.mean(df['demo_ses'])
    
    for feature in non_imaging_measurements:
        endog = df[feature]
        df["Intercept"] = 1
        exog = df[["Intercept", "demo_sex", "demo_ses"]]
        md = sm.GLM(endog, exog, family=sm.families.Gaussian()).fit()
        
        feature_name = feature + '_prime'
        df[feature_name] = df[feature] - (
            md.params[1] * (df["demo_sex"] - max_sex) +
            md.params[2] * (df["demo_ses"] - mean_ses)
        )
    
    # Delete the separate grip values for each hand that are not needed for training
    df = df.drop(['np_grip_xl_calc', 'np_grip_xr_calc','mean_grip','year', 'Intercept'], axis=1)

    df = df.drop(non_imaging_measurements, axis=1)

    return df

def residualize_imaging(df):
    """
    Residualize imaging measurements.
    
    Args:
    df (pd.DataFrame): Input dataframe.
    
    Returns:
    pd.DataFrame: Dataframe with residualized imaging measurements.
    """
    mean_head_size = np.mean(df['sri24_suptent_supratentorium_volume'])
    
    imaging_measurements = [col for col in df.columns if col.startswith('sri24') and (col.endswith('wm') or col.endswith('gm'))]
    
    for feature in imaging_measurements:
        endog = df[feature]
        df["Intercept"] = 1
        exog = df[["Intercept", "sri24_suptent_supratentorium_volume"]]
        md = sm.GLM(endog, exog, family=sm.families.Gaussian()).fit()
        
        feature_name = feature + '_prime'
        df[feature_name] = df[feature] - md.params[1] * (df["sri24_suptent_supratentorium_volume"] - mean_head_size)
    
    # Remove the columns of features before residualization
    droplist = [i for i in df.columns if i.endswith('gm') or i.endswith('wm')]

    df = df.drop(droplist, axis=1)
    df = df.drop('sri24_suptent_supratentorium_volume', axis=1)
    df = df.drop('Intercept', axis=1)

    return df