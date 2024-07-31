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
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import torch.utils.data as utils_data
from plot_config import COLUMN_RENAME_DICT
from model import create_traditional_model
from utils import *
from data_loading import *

warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_model(input_size, hidden_size, output_size, p=0.0):
    return MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, p=p)

def create_optimizer(model, learning_rate, weight_decay=1):
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def create_scheduler(optimizer, milestones, gamma):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

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


def train_and_evalute(X_dataframe, y_strat, finetune_on='control'):
    scaler = MinMaxScaler()
    
    num_folds = 5
    n_samples = len(y_strat)
    gss = StratifiedKFold(n_splits=num_folds, shuffle=True)

    SHAP_values_per_fold_deep = []
    ix_training, ix_test = [], []
    overall_metrics = {'train': {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0},
                       'test': {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0}}

    all_predictions = pd.DataFrame({'Actual': [], 'Predicted': [], 
                                    'Sex': [], 'Diagnosis': [], 'Subject': [], 'Age':[]})
    X_dataframe_fold = X_dataframe.copy()
    X_dataframe_fold['original_index'] = range(len(X_dataframe_fold))
    for i, (train_idx, test_idx) in enumerate(gss.split(np.zeros(n_samples), y_strat)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            warnings.futurewarnings = False
            
            # Split data into train and test
            X_train_data = X_dataframe_fold.iloc[train_idx]
            X_test_data = X_dataframe_fold.iloc[test_idx]
            
            # Save original indices
            train_original_index = X_train_data['original_index']
            test_original_index = X_test_data['original_index']
            
            # Remove original_index before preprocessing
            X_train_data = X_train_data.drop('original_index', axis=1)
            X_test_data = X_test_data.drop('original_index', axis=1)
            
            # Preprocess data with imputation and residualization without data leakage
            X_train_processed, X_test_processed = preprocess_data_no_leakage(X_train_data, X_test_data)
            
            # Prepare data for model
            X_train = np.array(X_train_processed.iloc[:,:-1])
            X_test = np.array(X_test_processed.iloc[:,:-1])
            y_train = np.array(X_train_processed['mean_grip_prime'])
            y_test = np.array(X_test_processed['mean_grip_prime'])
            
            # Restore original indices and combine processed data - this is for SHAP values and correct feature names
            X_train_processed['original_index'] = train_original_index
            X_test_processed['original_index'] = test_original_index
            X_processed_all = pd.concat([X_train_processed, X_test_processed])
            X_processed_all = X_processed_all.sort_values('original_index').reset_index(drop=True)
            X_processed_all = X_processed_all.drop('original_index', axis=1)
            
            # Update y_strat and X_dataframe
            y_strat = np.array(X_processed_all['demo_diag'])
            y_train_hiv = y_strat[train_idx]
            X_dataframe = X_processed_all.drop(["mean_grip_prime", "demo_diag", "demo_sex", "demo_ses", "subject"], axis=1)
                
        print(f'Fold {i+1}')
        
        ix_training.append(train_idx), ix_test.append(test_idx)
                
        X_scaled = scaler.fit_transform(X_train[:, 1:])
        # Remove the variables for demo_diag, demo_sex and demo_ses that will not be used during training
        X_scaled = np.delete(X_scaled, [1, 2, 5], 1)
        
        X_scaled_test = scaler.transform(X_test[:, 1:])
        # Remove the variables for demo_diag, demo_sex and demo_ses that will not be used during training
        X_scaled_test = np.delete(X_scaled_test, [1, 2, 5], 1)
        
        data_loader_trn = create_dataloaders(X_scaled, y_train, shuffle=True)
        data_loader_test = create_dataloaders(X_scaled_test, y_test, shuffle=False)

        subjects = X_test[:,0]
        age, sex, diagnosis = X_test[:,1], X_test[:,2], X_test[:,3]
        
        input_size = torch.Tensor(X_scaled).size()[1]
        hidden_size, output_size = 32, 1
        learning_rate = 1e-2
        
        # Number of epochs defers based on if we are only training the model on everyone or if we are finetuning
        if finetune_on != 'none':
            num_epoch = 70 #100 #80
        elif finetune_on == 'none':
            num_epoch = 70 #80
        
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
        
        # Fine-tuning
        if finetune_on != 'none':
            if finetune_on == 'control':
                finetune_epochs = 213
                X_scaled_finetune = X_scaled[y_train_hiv==0]
                y_train_finetune = y_train[y_train_hiv==0]
                shap_plot_color = '#4472C4'
            elif finetune_on == 'diseased':
                finetune_epochs = 90
                X_scaled_finetune = X_scaled[y_train_hiv>0]
                y_train_finetune = y_train[y_train_hiv>0]
                shap_plot_color = '#ED7D31'
            else:
                raise ValueError("finetune_on must be 'none', 'control', or 'diseased'")
            data_loader_trn_finetune = create_dataloaders(X_scaled_finetune, y_train_finetune, shuffle=True)
            optimizer_finetune = create_optimizer(model, learning_rate, weight_decay=5e-1)
        else:
            shap_plot_color = '#FFC0CB'
            finetune_epochs = 0
            
        for epoch in range(finetune_epochs):
            train_loss = train_epoch(model, data_loader_trn_finetune, optimizer_finetune, loss_fct)
            val_loss = validate(model, data_loader_test, loss_fct)
            
            if epoch % 100 == 0:
                print(f'Fine-tuning Epoch [{epoch+1}/{finetune_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
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
    
    # Get the 6 most important features with their original names
    vals = np.abs(new_shaps_arr_deep).mean(0)
    feature_names = X_dataframe_ri.columns

    feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                    columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)
    top_6_features = feature_importance['col_name'].values[:6]
    
    X_dataframe_ri = X_dataframe_ri.rename(columns=COLUMN_RENAME_DICT)
    
    shap.summary_plot(np.array(new_shaps_arr_deep), X_dataframe_ri.reindex(new_index),max_display=58, 
                  color_bar_label='', color=shap_plot_color, plot_type="bar", show=False)

    f = plt.gcf()

    plt.xlabel('Feature Strength', fontsize = 16)
    plt.ylabel('')
    f.savefig(f"revision_results_mlp/shap_barplots/save_me_shap_finetuned_{finetune_on}.png", dpi=500)

    # Print results
    for set_type in ['train', 'test']:
        print(f'OVERALL {set_type.upper()}')
        for metric, value in overall_metrics[set_type].items():
            print(f'{metric.upper()}: {value/num_folds:.2f}')

    return all_predictions, list(top_6_features)


def train_and_evaluate_traditional_model(X_dataframe, y_strat, model_type='mlp', train_on='control', **model_params):
    scaler = MinMaxScaler()
    num_folds = 5
    n_samples = len(y_strat)
    gss = StratifiedKFold(n_splits=num_folds, shuffle=True)

    SHAP_values_per_fold = []
    ix_training, ix_test = [], []
    overall_metrics = {'train': {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0},
                       'test': {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0}}

    all_predictions = pd.DataFrame({'Actual': [], 'Predicted': [], 
                                    'Sex': [], 'Diagnosis': [], 'Subject': [], 'Age':[]})
    
    X_dataframe_fold = X_dataframe.copy()
    X_dataframe_fold['original_index'] = range(len(X_dataframe_fold))
    for i, (train_idx, test_idx) in enumerate(gss.split(np.zeros(n_samples), y_strat)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            warnings.futurewarnings = False
            
            # Split data into train and test
            X_train_data = X_dataframe_fold.iloc[train_idx]
            X_test_data = X_dataframe_fold.iloc[test_idx]
            
            # Save original indices
            train_original_index = X_train_data['original_index']
            test_original_index = X_test_data['original_index']
            
            # Remove original_index before preprocessing
            X_train_data = X_train_data.drop('original_index', axis=1)
            X_test_data = X_test_data.drop('original_index', axis=1)
            
            # Preprocess data with imputation and residualization without data leakage
            X_train_processed, X_test_processed = preprocess_data_no_leakage(X_train_data, X_test_data)
            
            # Prepare data for model
            X_train = np.array(X_train_processed.iloc[:,:-1])
            X_test = np.array(X_test_processed.iloc[:,:-1])
            y_train = np.array(X_train_processed['mean_grip_prime'])
            y_test = np.array(X_test_processed['mean_grip_prime'])
            
            # Restore original indices and combine processed data - this is for SHAP values and correct feature names
            X_train_processed['original_index'] = train_original_index
            X_test_processed['original_index'] = test_original_index
            X_processed_all = pd.concat([X_train_processed, X_test_processed])
            X_processed_all = X_processed_all.sort_values('original_index').reset_index(drop=True)
            X_processed_all = X_processed_all.drop('original_index', axis=1)
            
            # Update y_strat and X_dataframe
            y_strat = np.array(X_processed_all['demo_diag'])
            y_train_hiv = y_strat[train_idx]
            X_dataframe = X_processed_all.drop(["mean_grip_prime", "demo_diag", "demo_sex", "demo_ses", "subject"], axis=1)
        
        ix_training.append(train_idx), ix_test.append(test_idx)
        
        X_scaled = scaler.fit_transform(X_train[:, 1:])
        X_scaled = np.delete(X_scaled, [1, 2, 5], 1)
        
        X_scaled_test = scaler.transform(X_test[:, 1:])
        # 1 - sex, 2 - diagnosis, 5 - ses
        X_scaled_test = np.delete(X_scaled_test, [1, 2, 5], 1)
        
        subjects = X_test[:,0]
        age, sex, diagnosis = X_test[:,1], X_test[:,2], X_test[:,3]
        
        # Create and train the model
        model = create_traditional_model(model_type, **model_params)
        
        if train_on == 'control':
            X_train_finetune = X_scaled[y_train_hiv==0]
            y_train_finetune = y_train[y_train_hiv==0]
            shap_plot_color = '#4472C4'
        elif train_on == 'diseased':
            X_train_finetune = X_scaled[y_train_hiv>0]
            y_train_finetune = y_train[y_train_hiv>0]
            shap_plot_color = '#ED7D31'
        else:  # 'none' or any other value
            X_train_finetune = X_scaled
            y_train_finetune = y_train
            shap_plot_color = '#FFC0CB'
        
        model.fit(X_train_finetune, y_train_finetune)
        
        # Predictions
        y_pred = model.predict(X_scaled_test)
        y_pred_train = model.predict(X_scaled)
        
        # SHAP values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            warnings.futurewarnings = False
            explainer = shap.KernelExplainer(model.predict, X_scaled[0:50,:])
            shap_values = explainer.shap_values(X_scaled_test)
        SHAP_values_per_fold.extend(shap_values)
        
        # Store predictions
        df_preds = pd.DataFrame({'Actual': y_test.squeeze().astype(float), 'Predicted': y_pred.squeeze().astype(float), 
                                'Sex': sex.squeeze().astype(float), 'Diagnosis': diagnosis.squeeze().astype(float),
                                'Subject': subjects.squeeze().astype(float), 'Age': age.squeeze().astype(float)})
        all_predictions = pd.concat([all_predictions, df_preds])

        # Calculate metrics
        for set_type, y_true, y_pred in [('train', y_train, y_pred_train), ('test', y_test, y_pred)]:
            mae, mse, rmse, r2 = calculate_metrics(y_true, y_pred)
            for metric, value in zip(['mae', 'mse', 'rmse', 'r2'], [mae, mse, rmse, r2]):
                overall_metrics[set_type][metric] += value

    # Aggregate SHAP values and print results
    new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
    X_dataframe_ri = X_dataframe.reset_index(drop=True)
    new_shaps_arr = np.vstack(SHAP_values_per_fold)
    
    # Get the 6 most important features with their original names
    vals = np.abs(new_shaps_arr).mean(0)
    feature_names = X_dataframe_ri.columns

    feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                    columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)
    top_6_features = feature_importance['col_name'].values[:6]
    
    X_dataframe_ri = X_dataframe_ri.rename(columns=COLUMN_RENAME_DICT)
    
    # SHAP summary plot
    shap.summary_plot(np.array(new_shaps_arr), X_dataframe_ri.reindex(new_index), max_display=58, 
                      color_bar_label='', plot_type="bar", show=False, color=shap_plot_color)
    plt.xlabel('Feature Strength', fontsize = 16)
    plt.ylabel('')
    plt.savefig(f"revision_results_{model_type}/shap_barplots/shap_{model_type}_{train_on}.png", dpi=500)
    plt.close()

    # Print results
    for set_type in ['train', 'test']:
        print(f'OVERALL {set_type.upper()}')
        for metric, value in overall_metrics[set_type].items():
            print(f'{metric.upper()}: {value/num_folds:.2f}')

    return all_predictions, list(top_6_features)