"""
Evaluate Chickenpox ensemble (XGB + LGB blend) on train/val/test.
Creates 2 plots: Model Loss (MAE) and Model Accuracy (R²).
Saves to plots/ folder.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
model_data_dir = "/home/chanuka002/Research/model_data"
artifacts_dir = os.path.join(model_data_dir, "artifacts", "chickenpox")
plots_dir = "/home/chanuka002/Research/plots"
os.makedirs(plots_dir, exist_ok=True)

def build_features_lepto(full_df, train_df_for_stats):
    df = full_df.sort_values(['district', 'week_id']).copy()
    grp = df.groupby('district')['target']
    for lag in (1, 2, 3, 4, 5, 6, 8, 12):
        df[f'target_lag_{lag}'] = grp.shift(lag)
    for w in (2, 4, 8, 12):
        s = grp.shift(1)
        df[f'target_roll_mean_{w}'] = s.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'target_roll_std_{w}'] = s.transform(lambda x: x.rolling(w, min_periods=1).std())
        df[f'target_roll_max_{w}'] = s.transform(lambda x: x.rolling(w, min_periods=1).max())
        df[f'target_roll_min_{w}'] = s.transform(lambda x: x.rolling(w, min_periods=1).min())
    df['target_trend_4'] = df.groupby('district')['target_roll_mean_4'].diff()
    df['target_trend_8'] = df.groupby('district')['target_roll_mean_8'].diff()
    df['target_accel'] = df.groupby('district')['target_trend_4'].diff()
    if 'T2M_avg' in df.columns and 'RH2M_avg' in df.columns:
        df['heat_humidity'] = df['T2M_avg'] * df['RH2M_avg']
    if 'PRECTOTCORR_avg' in df.columns and 'RH2M_avg' in df.columns:
        df['rain_humidity'] = df['PRECTOTCORR_avg'] * df['RH2M_avg']
    if 'PRECTOTCORR_avg' in df.columns and 'T2M_avg' in df.columns:
        df['rain_temp'] = df['PRECTOTCORR_avg'] * df['T2M_avg']
    if 'T2M_max' in df.columns and 'T2M_min' in df.columns:
        df['temp_range'] = df['T2M_max'] - df['T2M_min']
    if 'PRECTOTCORR_avg_lag_1' in df.columns and 'RH2M_avg_lag_1' in df.columns:
        df['rain_hum_lag1'] = df['PRECTOTCORR_avg_lag_1'] * df['RH2M_avg_lag_1']
    if 'PRECTOTCORR_avg_lag_2' in df.columns and 'RH2M_avg_lag_2' in df.columns:
        df['rain_hum_lag2'] = df['PRECTOTCORR_avg_lag_2'] * df['RH2M_avg_lag_2']
    rain_cols_4 = [f'PRECTOTCORR_avg_lag_{i}' for i in range(1, 5)]
    rain_cols_8 = [f'PRECTOTCORR_avg_lag_{i}' for i in range(1, 9)]
    df['cum_rain_4w'] = df[[c for c in rain_cols_4 if c in df.columns]].sum(axis=1)
    df['cum_rain_8w'] = df[[c for c in rain_cols_8 if c in df.columns]].sum(axis=1)
    dw_mean = train_df_for_stats.groupby(['district', 'week_number'])['target'].mean().rename('district_week_mean').reset_index()
    dist_mean = train_df_for_stats.groupby('district')['target'].mean().rename('dist_mean')
    late_train = train_df_for_stats[train_df_for_stats['week_id'] >= train_df_for_stats['week_id'].max() - 52]
    late_dist_mean = late_train.groupby('district')['target'].mean().rename('late_dist_mean')
    df = df.merge(dw_mean, on=['district', 'week_number'], how='left')
    df = df.merge(dist_mean, on='district', how='left')
    df = df.merge(late_dist_mean, on='district', how='left')
    df['district_week_mean'] = df['district_week_mean'].fillna(df['dist_mean'])
    df['late_dist_mean'] = df['late_dist_mean'].fillna(df['dist_mean'])
    df['excess_over_seasonal'] = df['target_roll_mean_4'] - df['district_week_mean']
    df['ratio_to_seasonal'] = df['target_roll_mean_4'] / (df['district_week_mean'] + 0.5)
    le = LabelEncoder()
    le.fit(df['district'])
    df['district_enc'] = le.transform(df['district'])
    return df

EXCLUDE_COLS = {'district', 'week_id', 'start_date', 'end_date', 'Duration', 'target', 'dist_mean'}

# Load saved feature names (exact order used by trained model)
with open(os.path.join(artifacts_dir, "feature_names.json")) as f:
    feature_cols = json.load(f)

# Load data
train_df = pd.read_csv(os.path.join(model_data_dir, "chickenpox_train.csv"))
val_df = pd.read_csv(os.path.join(model_data_dir, "chickenpox_val.csv"))
test_df = pd.read_csv(os.path.join(model_data_dir, "chickenpox_test.csv"))

# Build features (same as step_03)
full_raw = pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values('week_id')
full_fe = build_features_lepto(full_raw, train_df)
max_tr_wk = train_df['week_id'].max()
min_va_wk = val_df['week_id'].min()
max_va_wk = val_df['week_id'].max()
min_te_wk = test_df['week_id'].min()
train_df = full_fe[full_fe['week_id'] <= max_tr_wk].dropna(subset=['target']).copy()
val_df = full_fe[(full_fe['week_id'] >= min_va_wk) & (full_fe['week_id'] <= max_va_wk)].dropna(subset=['target']).copy()
test_df = full_fe[full_fe['week_id'] >= min_te_wk].dropna(subset=['target']).copy()

# Ensure all feature columns exist, fill missing with 0
for df in (train_df, val_df, test_df):
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
        elif df[c].dtype == bool:
            df[c] = df[c].astype(np.float64)
    df[feature_cols] = df[feature_cols].fillna(0)

X_train = np.ascontiguousarray(train_df[feature_cols].values.astype(np.float64))
y_train = train_df['target'].values.astype(np.float64)
X_val = np.ascontiguousarray(val_df[feature_cols].values.astype(np.float64))
y_val = val_df['target'].values.astype(np.float64)
X_test = np.ascontiguousarray(test_df[feature_cols].values.astype(np.float64))
y_test = test_df['target'].values.astype(np.float64)

# Load artifacts
scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
with open(os.path.join(artifacts_dir, "blending_weights.json")) as f:
    weights = json.load(f)
w_xgb = weights['xgb_weight']
w_lgb = weights['lgb_weight']

xgb_q50 = joblib.load(os.path.join(artifacts_dir, "xgb_q50.pkl"))
lgb_q50 = joblib.load(os.path.join(artifacts_dir, "lgb_q50.pkl"))

# Scale and predict
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

def ensemble_predict(X):
    return np.maximum(w_xgb * xgb_q50.predict(X) + w_lgb * lgb_q50.predict(X), 0)

pred_train = ensemble_predict(X_train_s)
pred_val_raw = ensemble_predict(X_val_s)
pred_test_raw = ensemble_predict(X_test_s)

# District bias correction (from step_03): computed from val, applied only to test
pred_val = pred_val_raw.copy()
pred_test = pred_test_raw.copy()
if 'district' in test_df.columns:
    val_corr = val_df[['district']].copy()
    val_corr['actual'] = y_val
    val_corr['pred'] = pred_val_raw
    val_corr['ratio'] = np.where(val_corr['pred'] > 0.5, val_corr['actual'] / val_corr['pred'], 1.0)
    district_correction = val_corr.groupby('district')['ratio'].median().clip(0.3, 4.0)
    test_corr = test_df[['district']].copy()
    test_corr['factor'] = test_corr['district'].map(district_correction).fillna(1.0)
    pred_test = np.maximum(pred_test_raw * test_corr['factor'].values, 0)

# Metrics
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

mae_tr, mse_tr, r2_tr = metrics(y_train, pred_train)
mae_va, mse_va, r2_va = metrics(y_val, pred_val)
mae_te, mse_te, r2_te = metrics(y_test, pred_test)

# Model Accuracy = R²
acc_tr, acc_va, acc_te = r2_tr, r2_va, r2_te

# Plots (2 only)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
splits = ['Train', 'Valid', 'Test']
colors = ['#e377c2', '#ff7f0e', '#2ca02c']

# Plot 1: Model Loss (MAE)
ax1 = axes[0]
mae_vals = [mae_tr, mae_va, mae_te]
bars = ax1.bar(splits, mae_vals, color=colors)
ax1.set_ylabel('MAE (Model Loss)')
ax1.set_title('Chickenpox Ensemble - Model Loss (MAE)')
ax1.set_ylim(0, max(mae_vals) * 1.15)
for bar, v in zip(bars, mae_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Model Accuracy (R²)
ax2 = axes[1]
acc_vals = [acc_tr, acc_va, acc_te]
bars2 = ax2.bar(splits, acc_vals, color=colors)
ax2.set_ylabel('R² (Model Accuracy)')
ax2.set_title('Chickenpox Ensemble - Model Accuracy (R²)')
ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ymin = min(acc_vals)
ymax = max(acc_vals)
ax2.set_ylim(min(0, ymin) - 0.05, max(0.1, ymax) + 0.05)
for bar, v in zip(bars2, acc_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if v >= 0 else -0.08, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(plots_dir, "chickenpox_ensemble_model_loss_accuracy.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
print(f"  Model Loss (MAE): Train={mae_tr:.3f}, Valid={mae_va:.3f}, Test={mae_te:.3f}")
print(f"  Model Accuracy (R²): Train={r2_tr:.3f}, Valid={r2_va:.3f}, Test={r2_te:.3f}")
