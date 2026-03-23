"""
Blended ensemble training curves for Chickenpox.
Combined model: y = w1*XGB + w2*LGB (weighted blending).
Produces 2 curve plots: Model Loss (MAE) and Model Accuracy (R2) over boosting rounds.
Saves to plots/ folder.
"""
import os
import json
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

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


# Load feature names and data
with open(os.path.join(artifacts_dir, "feature_names.json")) as f:
    feature_cols = json.load(f)

train_df = pd.read_csv(os.path.join(model_data_dir, "chickenpox_train.csv"))
val_df = pd.read_csv(os.path.join(model_data_dir, "chickenpox_val.csv"))
test_df = pd.read_csv(os.path.join(model_data_dir, "chickenpox_test.csv"))

full_raw = pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values('week_id')
full_fe = build_features_lepto(full_raw, train_df)
max_tr_wk = train_df['week_id'].max()
min_va_wk = val_df['week_id'].min()
max_va_wk = val_df['week_id'].max()
min_te_wk = test_df['week_id'].min()
train_df = full_fe[full_fe['week_id'] <= max_tr_wk].dropna(subset=['target']).copy()
val_df = full_fe[(full_fe['week_id'] >= min_va_wk) & (full_fe['week_id'] <= max_va_wk)].dropna(subset=['target']).copy()
test_df = full_fe[full_fe['week_id'] >= min_te_wk].dropna(subset=['target']).copy()

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

scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
with open(os.path.join(artifacts_dir, "blending_weights.json")) as f:
    w = json.load(f)
w_xgb, w_lgb = w['xgb_weight'], w['lgb_weight']

xgb_q50 = joblib.load(os.path.join(artifacts_dir, "xgb_q50.pkl"))
lgb_q50 = joblib.load(os.path.join(artifacts_dir, "lgb_q50.pkl"))

X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Number of rounds = min of both models (so we have both predictions at each round)
n_xgb = int(getattr(xgb_q50, 'n_estimators', getattr(xgb_q50, 'best_iteration', 500)) or 500)
n_lgb = int(getattr(lgb_q50, 'n_estimators_', getattr(lgb_q50, 'n_estimators', 500)) or 500)
n_rounds = min(n_xgb, n_lgb)
# Sample rounds for faster plotting (every 5 rounds, plus round 1 and last)
round_step = max(1, n_rounds // 80)
rounds_to_compute = [1] + list(range(round_step, n_rounds, round_step))
if n_rounds not in rounds_to_compute:
    rounds_to_compute.append(n_rounds)
rounds_to_compute = sorted(set(rounds_to_compute))

# District correction for test (computed once from val, applied at each round)
district_factor = None
if 'district' in test_df.columns:
    pred_val_full = np.maximum(w_xgb * xgb_q50.predict(X_val_s) + w_lgb * lgb_q50.predict(X_val_s), 0)
    val_corr = val_df[['district']].copy()
    val_corr['actual'] = y_val
    val_corr['pred'] = pred_val_full
    val_corr['ratio'] = np.where(val_corr['pred'] > 0.5, val_corr['actual'] / val_corr['pred'], 1.0)
    dc = val_corr.groupby('district')['ratio'].median().clip(0.3, 4.0)
    district_factor = test_df['district'].map(dc).fillna(1.0).values


print("Computing blended ensemble metrics per round...")
rounds_arr = []
mae_train, mae_val, mae_test = [], [], []
acc_train, acc_val, acc_test = [], [], []

for r in rounds_to_compute:
    pred_xgb_tr = xgb_q50.predict(X_train_s, iteration_range=(0, r))
    pred_xgb_va = xgb_q50.predict(X_val_s, iteration_range=(0, r))
    pred_xgb_te = xgb_q50.predict(X_test_s, iteration_range=(0, r))
    pred_lgb_tr = lgb_q50.predict(X_train_s, num_iteration=r)
    pred_lgb_va = lgb_q50.predict(X_val_s, num_iteration=r)
    pred_lgb_te = lgb_q50.predict(X_test_s, num_iteration=r)

    blend_tr = np.maximum(w_xgb * pred_xgb_tr + w_lgb * pred_lgb_tr, 0)
    blend_va = np.maximum(w_xgb * pred_xgb_va + w_lgb * pred_lgb_va, 0)
    blend_te_raw = np.maximum(w_xgb * pred_xgb_te + w_lgb * pred_lgb_te, 0)
    blend_te = blend_te_raw * district_factor if district_factor is not None else blend_te_raw

    mae_train.append(mean_absolute_error(y_train, blend_tr))
    mae_val.append(mean_absolute_error(y_val, blend_va))
    mae_test.append(mean_absolute_error(y_test, blend_te))
    acc_train.append(r2_score(y_train, blend_tr))
    acc_val.append(r2_score(y_val, blend_va))
    acc_test.append(r2_score(y_test, blend_te))
    rounds_arr.append(r)

rounds_arr = np.array(rounds_arr)
mae_train = np.array(mae_train)
mae_val = np.array(mae_val)
mae_test = np.array(mae_test)
acc_train = np.array(acc_train)
acc_val = np.array(acc_val)
acc_test = np.array(acc_test)

# Normalize each split by its own round-1 MAE so all curves start at 1.0
mae_r1_tr = max(mae_train[0], 1e-6)
mae_r1_va = max(mae_val[0], 1e-6)
mae_r1_te = max(mae_test[0], 1e-6)
norm_mae_train = mae_train / mae_r1_tr
norm_mae_val = mae_val / mae_r1_va
norm_mae_test = mae_test / mae_r1_te

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Model Loss (normalized MAE) - full scale
ax1 = axes[0]
ax1.plot(rounds_arr, norm_mae_test, color='#2ca02c', label='test', lw=2)
ax1.plot(rounds_arr, norm_mae_val, color='#ff7f0e', label='valid', lw=2)
ax1.plot(rounds_arr, norm_mae_train, color='#1f77b4', label='train', lw=2)
ax1.set_xlabel('Boosting rounds')
ax1.set_ylabel('Normalized MAE (round1)')
ax1.set_title('Chickenpox Blended Ensemble - Model Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, rounds_arr[-1])

# Plot 2: Model Accuracy (R²) - full scale
ax2 = axes[1]
ax2.plot(rounds_arr, acc_test, color='#2ca02c', label='test', lw=2)
ax2.plot(rounds_arr, acc_val, color='#ff7f0e', label='valid', lw=2)
ax2.plot(rounds_arr, acc_train, color='#1f77b4', label='train', lw=2)
ax2.set_xlabel('Boosting rounds')
ax2.set_ylabel('Accuracy (R²)')
ax2.set_title('Chickenpox Blended Ensemble - Model Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, rounds_arr[-1])
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)

fig.suptitle(f'Blended model: y = {w_xgb:.2f}*XGB + {w_lgb:.2f}*LGB', fontsize=11, y=1.02)
plt.tight_layout()

out_path = os.path.join(plots_dir, "chickenpox_blended_ensemble_curves.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {out_path}")
print(f"  Rounds: 1 to {n_rounds} (min of XGB={n_xgb}, LGB={n_lgb})")
print(f"  Final blended MAE: train={mae_train[-1]:.3f}, val={mae_val[-1]:.3f}, test={mae_test[-1]:.3f}")
print(f"  Final blended R2:  train={acc_train[-1]:.3f}, val={acc_val[-1]:.3f}, test={acc_test[-1]:.3f}")
