from f_test import *
from data_loading import *

unique_control = ['lr_cbc_mpv_prime', 'lr_cbc_mchc_prime', 'sri24_parc116_vermis_1_gm_prime', 'sri24_parc116_vermis_2_gm_prime', 'phys_bp_diastolic_prime']

unique_diseased = ['lr_cbc_plt_prime', 'qx_hf_socfxn_prime', 'sri24_parc6_insula_gm_prime']

control_formula = "mean_grip_prime ~ " + " + ".join(unique_control)

diseased_formula = "mean_grip_prime ~ " + " + ".join(unique_diseased)

f_data = load_and_preprocess_data_for_f_tests('/Users/magdalinipaschali/Documents/stanford/lab_data_code/grip_dataset_processed_apr_18_2023_onlyhead.csv')

run_f_test_double_dissociation(f_data, control_formula, diseased_formula)

count_tiny, count_sign, count_non_sign, max_p_value = run_f_test_10_times(f_data, diseased_formula)

print(f"Max p-value after 10 iterations: {max_p_value:.2f}")
print_results(count_tiny, count_sign, count_non_sign)
