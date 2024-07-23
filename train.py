# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
from sklearn.model_selection import StratifiedKFold
from model import MLP
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import torch.utils.data as utils_data


def old_training_notebook(X_dataframe, X, y, y_strat):
    
    # Feature normalizaton to 0-1
    scaler = MinMaxScaler()
    # scaler = StandardScaler()

    # 5 Folds used with stratified cross-validation to maintain the ratio of each disease cohort at each fold
    num_folds = 5

    gss = StratifiedKFold(n_splits=num_folds, shuffle=True)

    # Initialize variables for logging SHAP and errors during training
    SHAP_values_per_fold = []
    SHAP_values_per_fold_deep = []
    ix_training, ix_test = [], []
    overall_mae_train = 0
    overall_mse_train = 0
    overall_rmse_train = 0
    overall_r2_train = 0
    overall_mae_test = 0
    overall_mse_test = 0
    overall_rmse_test = 0
    overall_r2_test = 0
    overall_r2_sk_test = 0

    # This dataframe will hold all the predictions of the model
    all_predictions = pd.DataFrame({'Actual': [1.0], 'Predicted': [1.0], 
                                'Sex': [1.0], 'Diagnosis': [1.0], 'Subject': [1.0], 'Age':[1.0]})

    # Iterate over the 5 folds
    for i, (train_idx, test_idx) in enumerate(gss.split(X, y_strat)): 
        print('New fold')
        # Split the train and test sets for this fold and save the indices (useful for the SHAP values across all subjects)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_hiv = y_strat[train_idx]
        
        ix_training.append(train_idx), ix_test.append(test_idx)
        
        # Exclude subject ID (column 0) since it should not be used for training/testing
        # Fit the normalizer on the training data and transford the test data
        scalar = scaler.fit(X_train[:,1:])
        X_scaled = scaler.transform(X_train[:,1:])
        X_scaled_test = scaler.transform(X_test[:,1:])
        
        # Delete the demo_diag column - we only need the binary variables lr_aud and lr_hiv + also delete sex and ses from input features
        X_scaled = np.delete(X_scaled, [1,2,6], 1)
        X_scaled_test = np.delete(X_scaled_test, [1,2,6], 1)
        
        # Create datasets and dataloaders for training and testing
        training_samples = utils_data.TensorDataset(torch.Tensor(X_scaled), torch.Tensor(y_train))
        data_loader_trn = utils_data.DataLoader(training_samples, batch_size=64, drop_last=False, shuffle=True, num_workers=0)
        
        testing_samples = utils_data.TensorDataset(torch.Tensor(X_scaled_test), torch.Tensor(y_test))
        data_loader_test = utils_data.DataLoader(testing_samples, batch_size=64, drop_last=False, shuffle=False, num_workers=0)

        # Save the subject ID, age, sex and diagnosis for logging of the results
        subjects = X_test[:,0]
        age = X_test[:,1]
        sex = X_test[:,2]
        diagnosis = X_test[:,3]
        
        # Define model hyperparameters
        input_size = torch.Tensor(X_scaled).size()[1]
        
        ############################# QZ ###############################
    # Training on everyone
    #     hidden_size = 32
    #     output_size = 1
    #     num_epoch = 300
    #     learning_rate = 1e-2

    # Finetuning
        hidden_size = 32 
        output_size = 1
        num_epoch = 200 
        learning_rate = 1e-2
        
        # Define model, dropout probability set to 0.2
        model = MLP(input_size = input_size, hidden_size = hidden_size,
                    output_size = output_size, p=0.0)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1)
        ############################# QZ ###############################
        
        # Learning rate scheduler to drop the learning rate at the defined epochs
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,300,400], gamma=0.3)
        
        # MSE Loss
        loss_fct = nn.MSELoss()
        
        # Loss logging lists
        train_loss_all = []
        val_loss_all = []
        
        # Training for this fold
        for epoch in range(num_epoch):

            cum_loss = 0
            num_tr = 0
            
            model.train()
            # Training over batches
            for batch_idx, (data, target) in enumerate(data_loader_trn):
                tr_x, tr_y = data.float(), target.float()
                
                num_tr = num_tr + len(tr_x)
                
                pred = model(tr_x)
                loss = loss_fct(pred, tr_y.unsqueeze(1))
                        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
            
            # Log training loss
            train_loss_all.append(cum_loss/len(X_train))
                
            if epoch % 100 == 0:
                print ('Epoch [%d/%d], Train Loss: %.4f' 
                        %(epoch+1, num_epoch, cum_loss/num_tr))
                
            val_loss = 0
            num_test = 0 
            
            model.eval()
            # Testing over batches
            for batch_idx, (data, target) in enumerate(data_loader_test):
                tr_x_val, tr_y_val = data.float(), target.float()
                
                num_test = num_test + len(tr_x_val)
                    
                pred_val = model(tr_x_val)
                loss_val = loss_fct(pred_val, tr_y_val.unsqueeze(1))
                    
                val_loss += loss_val.item()
            
            # Log validation loss
            val_loss_all.append(val_loss/len(X_test))
            
            if epoch % 100 == 0:
                print ('Epoch [%d/%d], Validation Loss: %.4f' 
                        %(epoch+1, num_epoch, val_loss/num_test))
            
            lr_scheduler.step()
        
        ############################# QZ ###############################
    #     Fine-tune on AUD
    #     training_samples_finetune = utils_data.TensorDataset(torch.Tensor(X_scaled[y_train_hiv>0]), torch.Tensor(y_train[y_train_hiv>0]))
    #     training_samples_finetune = utils_data.TensorDataset(torch.Tensor(X_scaled[y_train_hiv==1]), torch.Tensor(y_train[y_train_hiv==1]))
    #     training_samples_finetune = utils_data.TensorDataset(torch.Tensor(X_scaled[y_train_hiv==3]), torch.Tensor(y_train[y_train_hiv==3]))

        # Fine-tune on CTRL
        training_samples_finetune = utils_data.TensorDataset(torch.Tensor(X_scaled[y_train_hiv==0]), torch.Tensor(y_train[y_train_hiv==0]))
        
        data_loader_trn_finetune = utils_data.DataLoader(training_samples_finetune, batch_size=64, drop_last=False, shuffle=True)
        optimizer_finetune = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=5e-1)
        
        # Set this to 0 to omit fune-tuning 40
        for epoch in range(40):

            cum_loss = 0
            num_tr = 0
            
            model.train()
            # Training over batches
            for batch_idx, (data, target) in enumerate(data_loader_trn_finetune):
                tr_x, tr_y = data.float(), target.float()
                
                num_tr = num_tr + len(tr_x)
                
                pred = model(tr_x)
                loss = loss_fct(pred, tr_y.unsqueeze(1))
                        
                optimizer_finetune.zero_grad()
                loss.backward()
                optimizer_finetune.step()
                cum_loss += loss.item()
            
            # Log training loss
            train_loss_all.append(cum_loss/len(X_train))
                
            if epoch % 100 == 0:
                print ('Epoch [%d/%d], Train Loss: %.4f' 
                        %(epoch+1, num_epoch, cum_loss/num_tr))
                
            val_loss = 0
            num_test = 0 
            
            model.eval()
            # Testing over batches
            for batch_idx, (data, target) in enumerate(data_loader_test):
                tr_x_val, tr_y_val = data.float(), target.float()
                
                num_test = num_test + len(tr_x_val)
                    
                pred_val = model(tr_x_val)
                loss_val = loss_fct(pred_val, tr_y_val.unsqueeze(1))
                    
                val_loss += loss_val.item()
            
            # Log validation loss
            val_loss_all.append(val_loss/len(X_test))
            
            if epoch % 100 == 0:
                print ('Epoch [%d/%d], Validation Loss: %.4f' 
                        %(epoch+1, num_epoch, val_loss/num_test))
            
            lr_scheduler.step()   
        ############################# QZ ###############################
        
        # Show loss plots
        # plt.plot(train_loss_all, 'b', linewidth=3)
        # plt.plot(val_loss_all, 'r', linewidth=3)
        # plt.show()
        
        # Save model to disc
    #     PATH = f'grip_finetuned_on_AUD_HIVAUD_fold_{i}.ckpt'
    #     PATH = f'grip_finetuned_on_CTRL_fold_{i}.ckpt'
    #     PATH = f'grip_trained_on_ALL_fold_{i}.ckpt'

    #     torch.save(model.state_dict(), PATH)
        
        # Collect all model predictions for this fold to y_pred
        model.eval()
        y_pred = model(torch.Tensor(X_scaled_test))
        y_pred = y_pred.detach().numpy()
        
        # SHAP Values computation for this fold
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Deep Explainer - define it with the training data of the fold X_scaled
            deep_explainer = shap.DeepExplainer(model, data=torch.Tensor(X_scaled))
            
            # Compute the SHAP values over the test data of the fold X_scaled_test
            shap_values_deep = deep_explainer.shap_values(torch.Tensor(X_scaled_test))
            
            # Plot top 15 features of this fold
            #shap.summary_plot(shap_values_deep, torch.Tensor(X_scaled_test), plot_type="bar", feature_names=feature_names,
            #                          plot_size=(16,7), max_display=15)
            
            # Keep SHAP values of the fold for overall plots at the end of all folds
            for SHAPs in shap_values_deep:
                SHAP_values_per_fold_deep.append(SHAPs)
        
        # Save all predictions for this fold along with sex, diagnosis and subject ID
        df_preds = pd.DataFrame({'Actual': y_test.squeeze().astype(float), 'Predicted': y_pred.squeeze().astype(float), 
                                'Sex': sex.squeeze().astype(float), 'Diagnosis': diagnosis.squeeze().astype(float),
                                'Subject': subjects.squeeze().astype(float), 'Age': age.squeeze().astype(float)})

        all_predictions = pd.concat([all_predictions, df_preds])

        # Compute model prediction for the train data of the fold
        y_pred_train = model(torch.Tensor(X_scaled))
        y_pred_train = y_pred_train.detach().numpy()
        
        # Calucalte errors for the train set of this fold
        mae = mean_absolute_error(y_train, y_pred_train)
        mse = mean_squared_error(y_train, y_pred_train)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_train, y_pred_train)
        
        overall_mae_train += mae
        overall_mse_train += mse
        overall_rmse_train += rmse
        overall_r2_train += r2

        # Calucalte errors for the test set of this fold
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        overall_mae_test += mae
        overall_mse_test += mse
        overall_rmse_test += rmse
        overall_r2_test += r2


    # Aggregate SHAP values for all folds and all subjects based on the fold they were in the validation set
    new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
    # reindex subjects based on the order they were used during training
    X_dataframe_ri = X_dataframe.reset_index(drop=True)
    # Aggregate SHAP of all folds
    new_shaps_arr_deep = np.vstack(SHAP_values_per_fold_deep)

    X_dataframe_ri = X_dataframe_ri.rename(columns={"sri24_parc116_cblmhemiwht_wm_prime": "Cerebellum", "np_wmsr_dig_back_span_prime": "Digits Backwards Span",
                                "sri24_parc116_precentral_gm_prime": "Precentral Gyrus", "lr_cbc_mchc_prime": "MCHC", 
                                "phys_bp_diastolic_prime": "Diastolic Blood Pressure", "lr_cbc_mpv_prime": "Mean Platelet Volume",
                                                "lr_hiv":"HIV Status", "qx_hf_socfxn_prime":"Social Functioning",
                                                "lr_cbc_plt_prime":"Platelet Count", "sri24_parc6_insula_gm_prime":"Insula",
                                                "lr_cbc_hct_prime": "Hematocrit", "lr_cbc_hgb_prime": "Hemoglobin",
                                "lr_cbc_mch_prime": "Mean Corpuscular Hemoglobin", "lr_cbc_mcv_prime": "Mean Corpuscular Volume", 
                                "lr_cbc_rbc_prime": "Red Blood Count", "lr_cbc_rdw_prime": "Red Cell Distribution Width",
                                                "lr_cbc_wbc_prime":"White Blood Count", "lr_nutrition_b12_prime":"Vitamin B12",
                                                "lr_nutrition_folate_prime":"Folate", "lr_nutrition_prealbumin_prime":"Prealbumin",
                                                    "phys_bmi_prime": "BMI", "phys_bp_systolic_prime": "Blood Pressure Systolic",
                                "phys_heart_rate_prime": "Heart Rate", "np_fas_total_prime": "Total FAS Words", 
                                "np_gold_stroop_cw_raw_prime": "Golden Stroop Test", "np_reyo_copy_raw_prime": "Rey-o Copy Score",
                                                "np_reyo_delay_raw_prime":"Rey-o Delayed Score", "np_ruff_des_unq_tot_prime":"Ruff Designs",
                                                "np_trails_a_time_prime":"Trails A", "np_trails_b_time_prime":"Trails B", 
                                                    "np_wmsr_logic_del_tot_prime": "Logical Memory Score", "np_wmsr_vis_back_span_prime": "Visual Backward Span",
                                "demo_gaf_prime": "GAF", "qx_audit_total_prime": "AUDIT", 
                                "qx_bdi_total_prime": "BDI", "qx_hf_cogfxn_prime": "Cognitive Function",
                                                "qx_hf_emobeing_prime":"Emotional Wellbeing", "qx_hf_energy_prime":"Energy/Fatigue",
                                                "qx_hf_healthpcp_prime":"Health Perception", "qx_hf_pain_prime":"Pain",
                                                    "qx_hf_physfxn_prime": "Physical Functioning", "qx_hf_rolefxn_prime": "Role Functioning",
                                "sri24_parc116_pons_wm_prime": "Pons", "sri24_parc116_vermis_1_gm_prime": "Vermis 1", 
                                "sri24_parc116_vermis_2_gm_prime": "Vermis 2", "sri24_parc116_vermis_3_gm_prime": "Vermis 3",
                                                "sri24_parc6_frontal_gm_prime":"Frontal", "sri24_parc6_occipital_gm_prime":"Occipital",
                                                "sri24_parc6_temporal_gm_prime":"Temporal", "sri24_parc116_caudate_gm_prime":"Caudate",
                                                    "sri24_parc116_pallidum_gm_prime": "Pallidum", "sri24_parc116_corpus_callosum_wm_prime": "Corpus Callosum",
                                "sri24_parc6_cingulate_gm_prime": "Cingulate", "sri24_parc6_parietal_gm_prime": "Parietal", 
                                "sri24_parc116_putamen_gm_prime": "Putamen", "sri24_parc116_thalamus_gm_prime": "Thalamus",
                                                "lr_aud":"AUD Diagnosis", "demo_age":"Age"
                                                })



    print(X_dataframe_ri.columns)
    # Show overall SHAP Values for all subjects/folds

    shap.summary_plot(np.array(new_shaps_arr_deep), X_dataframe_ri.reindex(new_index),max_display=57, 
                    color_bar_label='', color='#4472C4', plot_type="bar", show=False)

    #orange: #ED7D31
    # blue: #4472C4
    import matplotlib.pyplot as pl;
    f = pl.gcf()

    plt.xlabel('Feature Strength', fontsize = 16)
    plt.ylabel('')
    # f.savefig("save_me_shap_finetuned_disease_12_4.png", dpi=500)

    
    # displaying the title
    # plt.title("ReLU Function",
    #           fontsize = 40)

    # Add this code
    # print(f'Original size: {plt.gcf().get_size_inches()}')
    # w, _ = plt.gcf().get_size_inches()
    # plt.gcf().set_size_inches(w, w*3/4)
    plt.tight_layout()
    # print(f'New size: {plt.gcf().get_size_inches()}')

    # plt.savefig('shap_AUD.png', bbox_inches='tight',dpi=300)

    # shap.summary_plot(np.array(new_shaps_arr_deep), X_dataframe_ri.reindex(new_index), plot_type="bar",
    #                   max_display=6)

    # Print overall results during training
    print('OVERALL TRAIN')
    print(f'Mean absolute error: {(overall_mae_train/num_folds):.2f}')
    print(f'Mean squared error: {(overall_mse_train/num_folds):.2f}')
    print(f'Root mean squared error: {(overall_rmse_train/num_folds):.2f}')
    print(f'R2: {(overall_r2_train/num_folds):.2f}')

    # Print overall results for all subjects where they were in the validation set
    print('OVERALL TEST')
    mm = mean_absolute_error(all_predictions['Actual'], all_predictions['Predicted'])
    print(f'Mean absolute error: {mm:.2f}')
    ms = mean_squared_error(all_predictions['Actual'], all_predictions['Predicted'])
    print(f'Mean squared error: {ms:.2f}')
    rr = r2_score(all_predictions['Actual'], all_predictions['Predicted'])
    print(f'R2 overall: {rr:.2f}')

    # The first row has dummy variables we used above to define the dataframe for the concatenation so we don't need it
    all_predictions = all_predictions.iloc[1: , :]
    print(all_predictions)
    return all_predictions


def create_model(input_size, hidden_size, output_size, p=0.0):
    return MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, p=p)

def create_optimizer(model, learning_rate, weight_decay=1):
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def create_scheduler(optimizer, milestones, gamma):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

def preprocess_data(X, scaler):
    X_scaled = scaler.fit_transform(X[:, 1:])
    X_scaled = np.delete(X_scaled, [1, 2, 6], 1)
    return X_scaled

def create_dataloaders(X_scaled, y, batch_size=64, shuffle=True):
    dataset = utils_data.TensorDataset(torch.Tensor(X_scaled), torch.Tensor(y))
    return utils_data.DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)

def train_epoch(model, dataloader, optimizer, loss_fct):
    model.train()
    cum_loss = 0
    num_tr = 0
    for data, target in dataloader:
        tr_x, tr_y = data.float(), target.float()
        num_tr += len(tr_x)
        pred = model(tr_x)
        loss = loss_fct(pred, tr_y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
    return cum_loss / num_tr

def validate(model, dataloader, loss_fct):
    model.eval()
    val_loss = 0
    num_test = 0
    for data, target in dataloader:
        tr_x_val, tr_y_val = data.float(), target.float()
        num_test += len(tr_x_val)
        pred_val = model(tr_x_val)
        loss_val = loss_fct(pred_val, tr_y_val.unsqueeze(1))
        val_loss += loss_val.item()
    return val_loss / num_test

def compute_shap_values(model, X_background, X_test):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        deep_explainer = shap.DeepExplainer(model, data=torch.Tensor(X_background))
        return deep_explainer.shap_values(torch.Tensor(X_test))

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def old_training_notebook_v2(X_dataframe, X, y, y_strat):
    scaler = MinMaxScaler()
    num_folds = 5
    gss = StratifiedKFold(n_splits=num_folds, shuffle=True)

    SHAP_values_per_fold_deep = []
    ix_training, ix_test = [], []
    overall_metrics = {'train': {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0},
                       'test': {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0}}

    all_predictions = pd.DataFrame({'Actual': [], 'Predicted': [], 
                                    'Sex': [], 'Diagnosis': [], 'Subject': [], 'Age':[]})

    for i, (train_idx, test_idx) in enumerate(gss.split(X, y_strat)):
        print(f'Fold {i+1}')
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_hiv = y_strat[train_idx]
        
        ix_training.append(train_idx), ix_test.append(test_idx)
        
        X_scaled = preprocess_data(X_train, scaler)
        X_scaled_test = preprocess_data(X_test, scaler)
        
        data_loader_trn = create_dataloaders(X_scaled, y_train, shuffle=True)
        data_loader_test = create_dataloaders(X_scaled_test, y_test, shuffle=False)

        subjects = X_test[:,0]
        age, sex, diagnosis = X_test[:,1], X_test[:,2], X_test[:,3]
        
        input_size = torch.Tensor(X_scaled).size()[1]
        hidden_size, output_size = 32, 1
        num_epoch, learning_rate = 200, 1e-2
        
        model = create_model(input_size, hidden_size, output_size)
        optimizer = create_optimizer(model, learning_rate)
        lr_scheduler = create_scheduler(optimizer, milestones=[200,300,400], gamma=0.3)
        loss_fct = nn.MSELoss()
        
        train_loss_all, val_loss_all = [], []
        
        for epoch in range(num_epoch):
            train_loss = train_epoch(model, data_loader_trn, optimizer, loss_fct)
            val_loss = validate(model, data_loader_test, loss_fct)
            
            train_loss_all.append(train_loss)
            val_loss_all.append(val_loss)
            
            if epoch % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epoch}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            
            lr_scheduler.step()
        
        # Fine-tuning on CTRL
        X_scaled_finetune = X_scaled[y_train_hiv==0]
        y_train_finetune = y_train[y_train_hiv==0]
        data_loader_trn_finetune = create_dataloaders(X_scaled_finetune, y_train_finetune, shuffle=True)
        optimizer_finetune = create_optimizer(model, learning_rate, weight_decay=5e-1)
        
        for epoch in range(40):
            train_loss = train_epoch(model, data_loader_trn_finetune, optimizer_finetune, loss_fct)
            val_loss = validate(model, data_loader_test, loss_fct)
            
            if epoch % 100 == 0:
                print(f'Fine-tuning Epoch [{epoch+1}/40], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        model.eval()
        y_pred = model(torch.Tensor(X_scaled_test)).detach().numpy()
        
        shap_values_deep = compute_shap_values(model, X_scaled, X_scaled_test)
        SHAP_values_per_fold_deep.extend(shap_values_deep)
        
        df_preds = pd.DataFrame({'Actual': y_test.squeeze().astype(float), 'Predicted': y_pred.squeeze().astype(float), 
                                'Sex': sex.squeeze().astype(float), 'Diagnosis': diagnosis.squeeze().astype(float),
                                'Subject': subjects.squeeze().astype(float), 'Age': age.squeeze().astype(float)})
        all_predictions = pd.concat([all_predictions, df_preds])

        y_pred_train = model(torch.Tensor(X_scaled)).detach().numpy()
        
        for set_type, y_true, y_pred in [('train', y_train, y_pred_train), ('test', y_test, y_pred)]:
            mae, mse, rmse, r2 = calculate_metrics(y_true, y_pred)
            for metric, value in zip(['mae', 'mse', 'rmse', 'r2'], [mae, mse, rmse, r2]):
                overall_metrics[set_type][metric] += value

    # Aggregate SHAP values and print results
    new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
    X_dataframe_ri = X_dataframe.reset_index(drop=True)
    new_shaps_arr_deep = np.vstack(SHAP_values_per_fold_deep)

    # Print results
    for set_type in ['train', 'test']:
        print(f'OVERALL {set_type.upper()}')
        for metric, value in overall_metrics[set_type].items():
            print(f'{metric.upper()}: {value/num_folds:.2f}')

    print(all_predictions)
    return all_predictions, new_shaps_arr_deep, X_dataframe_ri.reindex(new_index)