# Grip Strength Analysis

## Reproduce original experiments and plots
To reproduce the original results and all plots run grip_run_analysis_original_submission.py. This will load the csv files with the original predictions and generate all figures for the main paper and the supplementary.
   
## Run revised code
To run the revised version, where the imputation is performed inside the CV loop to avoid data leakage run: grip_run_analysis.py.
There, we can select different model types for the analysis. The current version contains MLP (proposed). Support Vector Regressor (svr), Ridge Regression (ridge) and Random Forests (rf).

The analysis includes the following steps:

0. Train the model on all subjects - compute correlation between real and predicted grip values and SHAP values
2. Finetune the model on controls - compute correlation between real and predicted grip values and SHAP values
3. Finetune the model on diseased - compute correlation between real and predicted grip values and SHAP values
4. Perform F-test to check for double dissociation using the unique features identified for each cohort
5. Repeart the F-test 10 times by sub-sampling the AUD/HIV cohort to 53 subjects (number of controls we have) and count how many times the results are significant and what is the max p-value
6. Chech whether the feautures that were identified by the SHAP values are significantly correlated with one cohort and not the other and create the correlation plots
7. Generate Figure 3 of the paper - Boxplots comparing significant features between subjects with and without HIV
8. Generate Supplementary Figure - Correlation plots for significant features with measured grip strength, separated for subjects with and without HIV

## Data Pre-processing
The script grip_dataset_preprocessing.py performs the residualization and other data pre-processing, like summation of the imaging values between the two hemishperes. You don't need to run it every time, unless some pre-procesing step needs to change.

## Environment
Create conda environment with Python 3.9.2 and install the packages found in requirements.txt
