# plot_config.py

PLOT_INFO = {
    'sri24_parc116_cblmhemiwht_wm_prime': {
        'y_label': 'Cerebellum (cc)',
        'y_ticks': [6, 7, 8, 9, 10]
    },
    'lr_cbc_plt_prime': {
        'y_label': 'Platelet Count (thou/ul)',
        'y_ticks': [100, 200, 300, 400]
    },
    'lr_cbc_mchc_prime': {
        'y_label': 'MCHC (g/dl)',
        'y_ticks': [31, 32, 33, 34, 35]
    },
    'lr_cbc_mpv_prime': {
        'y_label': 'Mean Platelet Volume (fl)',
        'y_ticks': [7, 8, 9, 10, 11, 12]
    },
    'sri24_parc116_vermis_1_gm_prime': {
        'y_label': 'Vermis 1 Gray Matter (cc)',
        'y_ticks': [1, 2, 3, 4]
    },
    'sri24_parc6_occipital_gm_prime': {
        'y_label': 'Occipital Gray Matter (cc)',
        'y_ticks': [20, 25, 30, 35, 40]
    },
    'sri24_parc116_vermis_2_gm_prime': {
        'y_label': 'Vermis 2 Gray Matter (cc)',
        'y_ticks': [1, 1.5, 2, 2.5]
    },
    # Add more features as needed
}

COLUMN_RENAME_DICT = {
    "sri24_parc116_cblmhemiwht_wm_prime": "Cerebellum",
    "np_wmsr_dig_back_span_prime": "Digits Backwards Span",
    "sri24_parc116_precentral_gm_prime": "Precentral Gyrus",
    "lr_cbc_mchc_prime": "MCHC",
    "phys_bp_diastolic_prime": "Diastolic Blood Pressure",
    "lr_cbc_mpv_prime": "Mean Platelet Volume",
    "lr_hiv": "HIV Status",
    "qx_hf_socfxn_prime": "Social Functioning",
    "lr_cbc_plt_prime": "Platelet Count",
    "sri24_parc6_insula_gm_prime": "Insula",
    "lr_cbc_hct_prime": "Hematocrit",
    "lr_cbc_hgb_prime": "Hemoglobin",
    "lr_cbc_mch_prime": "Mean Corpuscular Hemoglobin",
    "lr_cbc_mcv_prime": "Mean Corpuscular Volume",
    "lr_cbc_rbc_prime": "Red Blood Count",
    "lr_cbc_rdw_prime": "Red Cell Distribution Width",
    "lr_cbc_wbc_prime": "White Blood Count",
    "lr_nutrition_b12_prime": "Vitamin B12",
    "lr_nutrition_folate_prime": "Folate",
    "lr_nutrition_prealbumin_prime": "Prealbumin",
    "phys_bmi_prime": "BMI",
    "phys_bp_systolic_prime": "Blood Pressure Systolic",
    "phys_heart_rate_prime": "Heart Rate",
    "np_fas_total_prime": "Total FAS Words",
    "np_gold_stroop_cw_raw_prime": "Golden Stroop Test",
    "np_reyo_copy_raw_prime": "Rey-o Copy Score",
    "np_reyo_delay_raw_prime": "Rey-o Delayed Score",
    "np_ruff_des_unq_tot_prime": "Ruff Designs",
    "np_trails_a_time_prime": "Trails A",
    "np_trails_b_time_prime": "Trails B",
    "np_wmsr_logic_del_tot_prime": "Logical Memory Score",
    "np_wmsr_vis_back_span_prime": "Visual Backward Span",
    "demo_gaf_prime": "GAF",
    "qx_audit_total_prime": "AUDIT",
    "qx_bdi_total_prime": "BDI",
    "qx_hf_cogfxn_prime": "Cognitive Function",
    "qx_hf_emobeing_prime": "Emotional Wellbeing",
    "qx_hf_energy_prime": "Energy/Fatigue",
    "qx_hf_healthpcp_prime": "Health Perception",
    "qx_hf_pain_prime": "Pain",
    "qx_hf_physfxn_prime": "Physical Functioning",
    "qx_hf_rolefxn_prime": "Role Functioning",
    "sri24_parc116_pons_wm_prime": "Pons",
    "sri24_parc116_vermis_1_gm_prime": "Vermis 1",
    "sri24_parc116_vermis_2_gm_prime": "Vermis 2",
    "sri24_parc116_vermis_3_gm_prime": "Vermis 3",
    "sri24_parc6_frontal_gm_prime": "Frontal",
    "sri24_parc6_occipital_gm_prime": "Occipital",
    "sri24_parc6_temporal_gm_prime": "Temporal",
    "sri24_parc116_caudate_gm_prime": "Caudate",
    "sri24_parc116_pallidum_gm_prime": "Pallidum",
    "sri24_parc116_corpus_callosum_wm_prime": "Corpus Callosum",
    "sri24_parc6_cingulate_gm_prime": "Cingulate",
    "sri24_parc6_parietal_gm_prime": "Parietal",
    "sri24_parc116_putamen_gm_prime": "Putamen",
    "sri24_parc116_thalamus_gm_prime": "Thalamus",
    "lr_aud": "AUD Diagnosis",
    "demo_age": "Age"
}