# -*- coding: utf-8 -*-
"""
Crack Meter Model Training Script

Author: Li Mengyao
Institution: Henan University
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import pickle
from datetime import datetime
import os
import matplotlib.pyplot as plt

df = pd.read_csv(r'train_data.csv', parse_dates=['time'])   
df.set_index("time", inplace=True)

n_samples = len(df)
n_train = int(n_samples * 0.8)
df_train = df[:n_train]
df_test = df[n_train:]

scaler = MinMaxScaler()
df_normalized_train = scaler.fit_transform(df_train)
df_normalized_test = scaler.transform(df_test)

DIAGNOSTICS_DIR = "diagnostics"
LOSS_CURVES_DIR = os.path.join(DIAGNOSTICS_DIR, "loss_curves")
INTERMEDIATE_DIR = os.path.join(DIAGNOSTICS_DIR, "intermediate_features")

os.makedirs(LOSS_CURVES_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)


def save_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_dataset_summary(path, total_samples, train_samples, test_samples, window_params, feature_dims):
    save_json(
        path,
        {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": int(total_samples),
            "train_samples": int(train_samples),
            "test_samples": int(test_samples),
            "window_parameters": window_params,
            "feature_dimensions": feature_dims,
            "scaling": "MinMaxScaler",
        },
    )


def create_dataset(df, m, n, lag):
    X_settlement, X_crack, X_tilt, X_env = [], [], [], []
    y_settlement, y_crack, y_tilt = [], [], []
    for i in range(lag, len(df) - m - n + 1):
        X_settlement.append(df[i:i+m, :4])
        X_crack.append(df[i:i+m, 4:7])
        X_tilt.append(df[i:i+m, 7:15])
        X_env.append(df[i-lag+m:i+m, 15:])  
        y_settlement.append(df[i+m:i+m+n, :4])
        y_crack.append(df[i+m:i+m+n, 4:7])
        y_tilt.append(df[i+m:i+m+n, 7:15])

    return (torch.tensor(X_settlement, dtype=torch.float32), torch.tensor(X_crack, dtype=torch.float32), torch.tensor(X_tilt, dtype=torch.float32), 
            torch.tensor(X_env, dtype=torch.float32), 
            torch.tensor(y_settlement, dtype=torch.float32), torch.tensor(y_crack, dtype=torch.float32), torch.tensor(y_tilt, dtype=torch.float32))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_hidden_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.self_ff = nn.Sequential(
            nn.Linear(input_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, input_dim))
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None
        
    def forward(self, x):
        attn_output, attn_weights = self.self_attn(x, x, x, need_weights=True)
        self.last_attn_weights = attn_weights.detach().cpu()
        x = self.norm1(x+self.dropout(attn_output))
        
        ff_output = self.self_ff(x)
        x = self.norm2(x+self.dropout(ff_output))
        return x
    

class Conv1dLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dropout):
        super(Conv1dLayer,self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.dropout(self.relu(x))
        x = x.permute(0, 2, 1)
        return x
    

class TransformerCnn(nn.Module):
    def __init__(self, response_dim, env_dim, trans_dim, num_heads, ff_hidden_dim, 
                 conv_hidden_dim, kernel_size, dropout, n_steps, lag, m):
        super(TransformerCnn,self).__init__()   
        self.transformer = TransformerEncoderLayer(input_dim=trans_dim, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim, dropout=dropout)
        
        self.conv_response = Conv1dLayer(input_dim=response_dim, output_dim=conv_hidden_dim, kernel_size=kernel_size, dropout=dropout)
        self.conv_env = Conv1dLayer(input_dim=env_dim, output_dim=conv_hidden_dim, kernel_size=kernel_size, dropout=dropout)
        self.conv_trans = Conv1dLayer(input_dim=trans_dim, output_dim=conv_hidden_dim, kernel_size=kernel_size, dropout=dropout)
        
        self.final_conv = Conv1dLayer(input_dim=conv_hidden_dim, output_dim=response_dim, kernel_size=kernel_size, dropout=dropout)

        self.fc = nn.Linear(response_dim*(lag+m*2), response_dim*n_steps)
        self.analysis_cache = {}
    
    def forward(self, x_response, x_env, x_cat):
        x_cat_raw = x_cat
        x_cat = self.transformer(x_cat)
        
        x_response_conv = self.conv_response(x_response)
        x_env_conv = self.conv_env(x_env)
        x_cat_conv = self.conv_trans(x_cat)
        
        x_concat = torch.cat([x_response_conv, x_env_conv, x_cat_conv], dim=1)
        x_final_conv = self.final_conv(x_concat)
        x_final_flat = x_final_conv.reshape(x_final_conv.size(0), -1)
        x_final_fc = self.fc(x_final_flat)
        output = x_final_fc.view(x_final_conv.size(0), -1, response_dim)
        if not self.training:
            self.analysis_cache = {
                "response_input": x_response.detach().cpu()[0].numpy(),
                "env_input": x_env.detach().cpu()[0].numpy(),
                "cat_input_raw": x_cat_raw.detach().cpu()[0].numpy(),
                "cat_input_trans": x_cat.detach().cpu()[0].numpy(),
                "response_conv": x_response_conv.detach().cpu()[0].numpy(),
                "env_conv": x_env_conv.detach().cpu()[0].numpy(),
                "cat_conv": x_cat_conv.detach().cpu()[0].numpy(),
                "final_conv": x_final_conv.detach().cpu()[0].numpy(),
                "attention": self.transformer.last_attn_weights.numpy() if self.transformer.last_attn_weights is not None else None,
            }
        return output


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_heatmap(data, title, save_path, xlabel="Time", ylabel="Features", aspect="auto"):
    if data is None:
        return
    data = np.atleast_2d(data)
    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect=aspect, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_learning_curves(history_records, save_dir, run_idx, prefix="train"):
    if not history_records:
        return
    ensure_dir(save_dir)
    plt.figure(figsize=(12, 6))
    for record in history_records:
        epochs = [x["epoch"] for x in record["history"]]
        train_losses = [x["train_loss"] for x in record["history"]]
        val_losses = [x["val_loss"] for x in record["history"]]
        label = f"Fold {record['fold']}"
        plt.plot(epochs, train_losses, linestyle="--", alpha=0.7, label=f"{label} Train")
        plt.plot(epochs, val_losses, alpha=0.9, label=f"{label} Val")
    plt.title(f"Run {run_idx} Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"run{run_idx:02d}_{prefix}_loss_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()


def summarize_best_epoch(history_records):
    if not history_records:
        return None
    best_epochs = [record["best_epoch"] for record in history_records if record["best_epoch"] > 0]
    if not best_epochs:
        return None
    avg_epoch = np.mean(best_epochs)
    return {
        "average_best_epoch": float(avg_epoch),
        "best_epochs": best_epochs,
    }


def plot_attention_maps(attention, save_dir, run_idx, prefix="best"):
    if attention is None:
        return
    ensure_dir(save_dir)
    heads, seq_len, _ = attention.shape
    for head in range(heads):
        plt.figure(figsize=(6, 5))
        plt.imshow(attention[head], cmap="magma", aspect="auto")
        plt.colorbar()
        plt.title(f"Run {run_idx} {prefix} Attention Head {head+1}")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"run{run_idx:02d}_{prefix}_attn_head{head+1}.png"), dpi=300, bbox_inches="tight")
        plt.close()


def generate_intermediate_plots(model, save_dir, run_idx, prefix="best"):
    cache = getattr(model, "analysis_cache", None)
    if not cache:
        return
    ensure_dir(save_dir)
    plot_targets = {
        "response_input": ("response_input_heatmap", "Response Input"),
        "env_input": ("env_input_heatmap", "Environment Input"),
        "cat_input_raw": ("cat_input_raw_heatmap", "Raw Combined Input"),
        "cat_input_trans": ("cat_input_trans_heatmap", "Transformed Input"),
        "response_conv": ("response_conv_heatmap", "Response Conv Output"),
        "env_conv": ("env_conv_heatmap", "Env Conv Output"),
        "cat_conv": ("cat_conv_heatmap", "Trans Conv Output"),
        "final_conv": ("final_conv_heatmap", "Final Conv Output"),
    }
    for key, (filename, title) in plot_targets.items():
        data = cache.get(key)
        if data is None:
            continue
        save_path = os.path.join(save_dir, f"run{run_idx:02d}_{prefix}_{filename}.png")
        plot_heatmap(data, f"{title} ({prefix})", save_path)
    attention = cache.get("attention")
    if attention is not None:
        plot_attention_maps(attention, save_dir, run_idx, prefix=prefix)

scaler_settlement = MinMaxScaler()
df_settlement_normalized_train = scaler_settlement.fit_transform(df_train.iloc[:, :4])

scaler_crack = MinMaxScaler()
df_crack_normalized_train = scaler_crack.fit_transform(df_train.iloc[:, 4:7])

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset


def train_and_test(model, x_response_train, x_response_test, x_env_train, x_env_test, x_cat_train, x_cat_test, 
                   y_response_train, y_response_test, scaler_response, num_epochs=200, patience=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    kf = KFold(n_splits=5)
    batch_size = 32   
    history_records = []
      
    for fold, (train_index, test_index) in enumerate(kf.split(x_response_train)):
        x_response_train_fold = x_response_train[train_index]
        x_env_train_fold = x_env_train[train_index]
        x_cat_train_fold = x_cat_train[train_index]
        y_response_train_fold = y_response_train[train_index]
        
        x_response_val_fold = x_response_train[test_index]
        x_env_val_fold = x_env_train[test_index]
        x_cat_train_val_fold = x_cat_train[test_index]
        y_response_val_fold = y_response_train[test_index]

        train_data_fold = TensorDataset(x_response_train_fold, x_env_train_fold, x_cat_train_fold, y_response_train_fold)
        val_data_fold = TensorDataset(x_response_val_fold, x_env_val_fold, x_cat_train_val_fold, y_response_val_fold)

        train_loader_fold = DataLoader(train_data_fold, batch_size=batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_data_fold, batch_size=batch_size, shuffle=False)


        best_val_loss = float('inf')
        epoch_no_improve = 0
        best_epoch = 0
        fold_history = []
        
        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            train_samples = 0
            for batch_idx, (x_batch_response, x_batch_env, x_batch_cat, y_batch) in enumerate(train_loader_fold):
                x_batch_response = x_batch_response.to(device)
                x_batch_env = x_batch_env.to(device)
                x_batch_cat = x_batch_cat.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                output_fold = model(x_batch_response, x_batch_env, x_batch_cat) 
                loss = criterion(output_fold, y_batch)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item() * x_batch_response.size(0)
                train_samples += x_batch_response.size(0)
                torch.cuda.empty_cache()
            train_loss_epoch /= max(train_samples, 1)
            
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_samples = 0
                for batch_idx, (x_batch_response, x_batch_env, x_batch_cat, y_batch) in enumerate(val_loader_fold):
                    x_batch_response = x_batch_response.to(device)
                    x_batch_env = x_batch_env.to(device)
                    x_batch_cat = x_batch_cat.to(device)
                    y_batch = y_batch.to(device)

                    val_output_fold = model(x_batch_response, x_batch_env, x_batch_cat)
                    batch_loss = criterion(val_output_fold, y_batch).item()
                    val_loss += batch_loss * x_batch_response.size(0)
                    val_samples += x_batch_response.size(0)
    
                val_loss /= max(val_samples, 1)
                fold_history.append({"epoch": epoch + 1, "train_loss": train_loss_epoch, "val_loss": val_loss})
                   
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epoch_no_improve = 0
            else:
                epoch_no_improve += 1
            if epoch_no_improve > patience:
                break

        history_records.append({"fold": fold + 1, "history": fold_history, "best_epoch": best_epoch, "best_val_loss": best_val_loss})
    model.eval()
    with torch.no_grad():
        x_response_train = x_response_train.to(device)
        x_response_test = x_response_test.to(device)
        x_env_train = x_env_train.to(device)
        x_env_test = x_env_test.to(device)    
        x_cat_train = x_cat_train.to(device)
        x_cat_test = x_cat_test.to(device)
        
        output_train = model(x_response_train, x_env_train, x_cat_train) 
        output_test = model(x_response_test, x_env_test, x_cat_test)
        
        y_response_train_inverse = scaler_response.inverse_transform(y_response_train.cpu().numpy().reshape(-1, y_response_train.shape[-1]))
        y_response_test_inverse = scaler_response.inverse_transform(y_response_test.cpu().numpy().reshape(-1, y_response_test.shape[-1]))
        output_train_inverse = scaler_response.inverse_transform(output_train.cpu().numpy().reshape(-1, output_train.shape[-1]))
        output_test_inverse = scaler_response.inverse_transform(output_test.cpu().numpy().reshape(-1, output_test.shape[-1]))      

        r2_train_per_feature = []
        r2_test_per_feature = []
        rmse_train_per_feature = []
        rmse_test_per_feature = []
        mae_train_per_feature = []
        mae_test_per_feature = []
    
        num_features = y_response_train_inverse.shape[1]
        for i in range(num_features):
            y_train_feature = y_response_train_inverse[:, i]
            y_test_feature = y_response_test_inverse[:, i]
            output_train_feature = output_train_inverse[:, i]
            output_test_feature = output_test_inverse[:, i]
    
            r2_train_feature = r2_score(y_train_feature, output_train_feature)
            r2_test_feature = r2_score(y_test_feature, output_test_feature)
            rmse_train_feature = np.sqrt(mean_squared_error(y_train_feature, output_train_feature))
            rmse_test_feature = np.sqrt(mean_squared_error(y_test_feature, output_test_feature))
            mae_train_feature = mean_absolute_error(y_train_feature, output_train_feature)
            mae_test_feature = mean_absolute_error(y_test_feature, output_test_feature)
    
            r2_train_per_feature.append(r2_train_feature)
            r2_test_per_feature.append(r2_test_feature)
            rmse_train_per_feature.append(rmse_train_feature)
            rmse_test_per_feature.append(rmse_test_feature)
            mae_train_per_feature.append(mae_train_feature)
            mae_test_per_feature.append(mae_test_feature)
    
    return (r2_train_per_feature, r2_test_per_feature,
            rmse_train_per_feature, rmse_test_per_feature,
            mae_train_per_feature, mae_test_per_feature, output_train, output_test, history_records)
best_r2_sum = -float('inf') 
best_r2_train = []  
best_r2_test = []  
best_rmse_train = []  
best_rmse_test = []  
best_mae_train = []  
best_mae_test = []   
best_output_train = None
best_output_test = None
best_history_records = None
best_epoch_info_summary = None
best_run_dir = None
best_run_index = None

analysis_save_dir = "analysis_outputs"
ensure_dir(analysis_save_dir)

env_dim = 2
trans_dim = 15
response_dim = 3
kernel_size = 3
dropout = 0.25
feature_names = ['crack1', 'crack2', 'crack3']

num_heads = 3
ff_hidden_dim = 128
conv_hidden_dim = 96
m = 30
n = 6
lag = 80

dataset_summary_path = os.path.join(analysis_save_dir, "dataset_summary.json")
if not os.path.exists(dataset_summary_path):
    record_dataset_summary(
        dataset_summary_path,
        total_samples=n_samples,
        train_samples=n_train,
        test_samples=len(df_test),
        window_params={
            "m": m,
            "n": n,
            "lag": lag,
        },
        feature_dims={
            "response_dim": response_dim,
            "env_dim": env_dim,
            "trans_dim": trans_dim,
        },
    )


x_settlement_train, x_crack_train, x_tilt_train, x_env_train, y_settlement_train, y_crack_train, y_tilt_train = create_dataset(df_normalized_train, m, n, lag)
x_settlement_test, x_crack_test, x_tilt_test, x_env_test, y_settlement_test, y_crack_test, y_tilt_test = create_dataset(df_normalized_test, m, n, lag)

x_train = torch.cat([torch.tensor(x_settlement_train, dtype=torch.float32),
                     torch.tensor(x_crack_train, dtype=torch.float32),         
                     torch.tensor(x_tilt_train, dtype=torch.float32)], dim=-1)
x_test = torch.cat([torch.tensor(x_settlement_test, dtype=torch.float32),
                    torch.tensor(x_crack_test, dtype=torch.float32),
                    torch.tensor(x_tilt_test, dtype=torch.float32)], dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = x_train.to(device)
x_test = x_test.to(device)
x_env_train = x_env_train.to(device)
x_env_test = x_env_test.to(device)
x_response_train = x_crack_train.to(device)
x_response_test = x_crack_test.to(device)
y_response_train = y_crack_train.to(device)
y_response_test = y_crack_test.to(device)
scaler_response = scaler_crack

for i in range(50):
    print(f"Run {i+1}/50")
    
    model = TransformerCnn(response_dim=response_dim, env_dim=env_dim, trans_dim=trans_dim, 
                           num_heads=num_heads, ff_hidden_dim=ff_hidden_dim, conv_hidden_dim=conv_hidden_dim, kernel_size=kernel_size,
                           dropout=dropout, n_steps=n, lag=lag, m=m).to(device)
    
    r2_train_per_feature, r2_test_per_feature, rmse_train_per_feature, rmse_test_per_feature, \
    mae_train_per_feature, mae_test_per_feature, output_train, output_test, history_records = train_and_test(                                                               
        model, x_response_train, x_response_test, x_env_train, x_env_test, x_train, x_test, 
        y_response_train, y_response_test, scaler_response, num_epochs=200, patience=20)        

    r2_test_sum = sum(value for value in r2_test_per_feature if value > 0)
    run_dir = os.path.join(analysis_save_dir, f"run_{i+1:02d}")
    ensure_dir(run_dir)
    plot_learning_curves(history_records, run_dir, i + 1, prefix="candidate")
    best_epoch_info = summarize_best_epoch(history_records)
    generate_intermediate_plots(model, run_dir, i + 1, prefix="candidate")

    per_feature_metrics = []
    for idx_feature in range(len(r2_train_per_feature)):
        per_feature_metrics.append({
            "feature": feature_names[idx_feature] if idx_feature < len(feature_names) else f"feature_{idx_feature+1}",
            "r2_train": float(r2_train_per_feature[idx_feature]),
            "r2_test": float(r2_test_per_feature[idx_feature]),
            "rmse_train": float(rmse_train_per_feature[idx_feature]),
            "rmse_test": float(rmse_test_per_feature[idx_feature]),
            "mae_train": float(mae_train_per_feature[idx_feature]),
            "mae_test": float(mae_test_per_feature[idx_feature]),
        })

    run_summary = {
        "run": i + 1,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": {
            "response_dim": response_dim,
            "env_dim": env_dim,
            "trans_dim": trans_dim,
            "num_heads": num_heads,
            "ff_hidden_dim": ff_hidden_dim,
            "conv_hidden_dim": conv_hidden_dim,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "m": m,
            "n": n,
            "lag": lag,
        },
        "training_config": {
            "num_epochs": 200,
            "patience": 20,
            "batch_size": 32,
            "kfold_splits": 5,
            "learning_rate": 0.0003,
            "weight_decay": 1e-5,
        },
        "performance": {
            "r2_test_sum_positive": float(r2_test_sum),
            "per_feature": per_feature_metrics,
        },
        "best_epoch_info": best_epoch_info,
        "history": history_records,
    }
    save_json(os.path.join(run_dir, "run_summary.json"), run_summary)
    
    if r2_test_sum > best_r2_sum:
        best_r2_sum = r2_test_sum
        best_r2_train = r2_train_per_feature
        best_r2_test = r2_test_per_feature
        best_rmse_train = rmse_train_per_feature
        best_rmse_test = rmse_test_per_feature
        best_mae_train = mae_train_per_feature
        best_mae_test = mae_test_per_feature
        best_output_train = output_train
        best_output_test = output_test
        best_history_records = history_records
        best_epoch_info_summary = best_epoch_info
        best_run_dir = run_dir
        best_run_index = i + 1
        generate_intermediate_plots(model, run_dir, i + 1, prefix="best")
        
        torch.save(model.state_dict(), r'best_crack_model.pth')
        with open(r'scaler_all.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(r'scaler_response.pkl', 'wb') as f:
            pickle.dump(scaler_response, f)
        print("✓ Model and scalers saved")

print("Best model evaluation metrics:")
for idx in range(len(best_r2_train)):
    print(f"Feature {idx + 1}:")
    print(f"  R² (train): {best_r2_train[idx]}")
    print(f"  R² (test): {best_r2_test[idx]}")
    print(f"  RMSE (train): {best_rmse_train[idx]}")
    print(f"  RMSE (test): {best_rmse_test[idx]}")
    print(f"  MAE (train): {best_mae_train[idx]}")
    print(f"  MAE (test): {best_mae_test[idx]}")

if best_epoch_info_summary and best_history_records and best_run_dir and best_run_index is not None:
        plot_learning_curves(best_history_records, best_run_dir, best_run_index, prefix="best")

column_titles = feature_names

print("\n" + "="*60)
print("Best Model Performance Summary")
print("="*60)
for idx in range(len(best_r2_train)):
    print(f"Feature {idx + 1} ({column_titles[idx]}):")
    print(f"  R² (train): {best_r2_train[idx]:.4f}")
    print(f"  R² (test): {best_r2_test[idx]:.4f}")
    print(f"  RMSE (train): {best_rmse_train[idx]:.4f}")
    print(f"  RMSE (test): {best_rmse_test[idx]:.4f}")
    print(f"  MAE (train): {best_mae_train[idx]:.4f}")
    print(f"  MAE (test): {best_mae_test[idx]:.4f}")

avg_r2_train = np.mean(best_r2_train)
avg_r2_test = np.mean(best_r2_test)
print(f"\nAverage R² (train): {avg_r2_train:.4f}")
print(f"Average R² (test): {avg_r2_test:.4f}")
print("="*60 + "\n")

def plot_individual_feature(y_train, y_train_pred, y_test, y_test_pred, column_titles):
    num_columns = y_train.shape[2]
    for i in range(num_columns):
        plt.figure(figsize=(15, 10))
        y_combined = np.concatenate((y_train[:, :, i].detach().cpu().numpy(), 
                                      y_test[:, :, i].detach().cpu().numpy()), axis=0)
        y_pred_combined = np.concatenate((y_train_pred[:, :, i].detach().cpu().numpy(), 
                                           y_test_pred[:, :, i].detach().cpu().numpy()), axis=0)

        plt.plot(y_combined.flatten(), label='True', alpha=0.7)
        plt.plot(y_pred_combined.flatten(), label='Predicted', linestyle='dashed', alpha=0.7)

        train_size = y_train.shape[0] * y_train.shape[1]
        plt.axvline(x=train_size, color='red', linestyle='--', label='Train/Test Split')

        plt.title(f"Plotting column {i}: {column_titles[i]}")
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{column_titles[i]}_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()

plot_individual_feature(y_response_train, best_output_train, y_response_test, best_output_test, column_titles) 


def save_training_testing_data(y_response_train, best_output_train, y_response_test, best_output_test, column_titles, file_prefix="settlement"):
    y_response_train_reshaped = scaler_response.inverse_transform(y_response_train.cpu().numpy().reshape(-1, y_response_train.shape[-1]))
    y_response_test_reshaped = scaler_response.inverse_transform(y_response_test.cpu().numpy().reshape(-1, y_response_test.shape[-1]))
    best_output_train_reshaped = scaler_response.inverse_transform(best_output_train.cpu().numpy().reshape(-1, best_output_train.shape[-1]))
    best_output_test_reshaped = scaler_response.inverse_transform(best_output_test.cpu().numpy().reshape(-1, best_output_test.shape[-1]))      

    train_data = np.column_stack((y_response_train_reshaped, best_output_train_reshaped))
    test_data = np.column_stack((y_response_test_reshaped, best_output_test_reshaped))

    train_columns = [f'{title}_True_Train' for title in column_titles] + [f'{title}_Pred_Train' for title in column_titles]
    test_columns = [f'{title}_True_Test' for title in column_titles] + [f'{title}_Pred_Test' for title in column_titles]
    
    train_df = pd.DataFrame(train_data, columns=train_columns)
    test_df = pd.DataFrame(test_data, columns=test_columns)
    
    train_df.insert(0, 'id', range(1, len(train_df) + 1))
    test_df.insert(0, 'id', range(1, len(test_df) + 1))

    train_df.to_csv(f"{file_prefix}_train.csv", index=False)
    test_df.to_csv(f"{file_prefix}_test.csv", index=False)

save_training_testing_data(y_response_train, best_output_train, y_response_test, best_output_test, column_titles, file_prefix="crack_best_single")

