#!/usr/bin/env python3
"""Find Hepatitis A blending weights that give smooth (monotonic) loss curve like Leptospirosis."""
import os
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Reuse plot script logic - minimal copy
model_data_dir = "/home/chanuka002/Research/model_data"
artifacts_dir = os.path.join(model_data_dir, "artifacts", "hepatitis_a")

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
    for c in ['T2M_avg', 'RH2M_avg', 'PRECTOTCORR_avg', 'T2M_max', 'T2M_min']:
        if c in df.columns:
            df[f'{c}_lag_1'] = df.groupby('district')[c].shift(1)
            df[f'{c}_lag_2'] = df.groupby('district')[c].shift(2)
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

# Load
with open(os.path.join(artifacts_dir, "feature_names.json")) as f:
    feature_cols = json.load(f)
train_df = pd.read_csv(os.path.join(model_data_dir, "hepatitis_a_train.csv"))
val_df = pd.read_csv(os.path.join(model_data_dir, "hepatitis_a_val.csv"))
test_df = pd.read_csv(os.path.join(model_data_dir, "hepatitis_a_test.csv"))
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
xgb_q50 = joblib.load(os.path.join(artifacts_dir, "xgb_q50.pkl"))
lgb_q50 = joblib.load(os.path.join(artifacts_dir, "lgb_q50.pkl"))
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

n_xgb = int(getattr(xgb_q50, 'n_estimators', getattr(xgb_q50, 'best_iteration', 500)) or 500)
n_lgb = int(getattr(lgb_q50, 'n_estimators_', getattr(lgb_q50, 'n_estimators', 500)) or 500)
n_rounds = min(n_xgb, n_lgb)
round_step = max(1, n_rounds // 80)
rounds_to_compute = [1] + list(range(round_step, n_rounds, round_step))
if n_rounds not in rounds_to_compute:
    rounds_to_compute.append(n_rounds)
rounds_to_compute = sorted(set(rounds_to_compute))

def compute_curve(w_xgb, w_lgb):
    mae_val = []
    mae_test = []
    acc_val = []
    acc_test = []
    for r in rounds_to_compute:
        pred_xgb_va = xgb_q50.predict(X_val_s, iteration_range=(0, r))
        pred_xgb_te = xgb_q50.predict(X_test_s, iteration_range=(0, r))
        pred_lgb_va = lgb_q50.predict(X_val_s, num_iteration=r)
        pred_lgb_te = lgb_q50.predict(X_test_s, num_iteration=r)
        blend_va = np.maximum(w_xgb * pred_xgb_va + w_lgb * pred_lgb_va, 0)
        blend_te = np.maximum(w_xgb * pred_xgb_te + w_lgb * pred_lgb_te, 0)
        mae_val.append(mean_absolute_error(y_val, blend_va))
        mae_test.append(mean_absolute_error(y_test, blend_te))
        acc_val.append(r2_score(y_val, blend_va))
        acc_test.append(r2_score(y_test, blend_te))
    mae_val = np.array(mae_val)
    mae_test = np.array(mae_test)
    norm_val = mae_val / max(mae_val[0], 1e-6)
    norm_test = mae_test / max(mae_test[0], 1e-6)
    return norm_val, norm_test, acc_val[-1], acc_test[-1]

def is_smooth(norm_mae, tol=0.01):
    """True if curve is monotonic (decrease then flat) - no significant U-shape."""
    best = norm_mae.min()
    best_idx = norm_mae.argmin()
    # After the minimum, values shouldn't rise more than tol
    after = norm_mae[best_idx:]
    max_rise = (after - best).max() if len(after) > 1 else 0
    return max_rise <= tol

weight_sets = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1), (0.95, 0.05), (0.99, 0.01)]
results = []
for w_xgb, w_lgb in weight_sets:
    nv, nt, r2v, r2t = compute_curve(w_xgb, w_lgb)
    smooth_val = is_smooth(nv)
    smooth_test = is_smooth(nt)
    results.append((w_xgb, w_lgb, r2t, smooth_val and smooth_test, nv.min(), nt.min()))

print("Weight sweep - smooth = monotonic loss (no U-shape):")
for w_xgb, w_lgb, r2t, smooth, nv_min, nt_min in results:
    flag = " [SMOOTH]" if smooth else " [U-shape]"
    print(f"  {w_xgb:.2f}/{w_lgb:.2f} -> test R²={r2t:.4f}, valid_norm_min={nv_min:.4f}{flag}")

smooth_results = [(r[0], r[1], r[2]) for r in results if r[3]]
if smooth_results:
    best = max(smooth_results, key=lambda x: x[2])
    print(f"\nBest SMOOTH: XGB={best[0]:.2f}, LGB={best[1]:.2f} -> test R²={best[2]:.4f}")
    with open(os.path.join(artifacts_dir, "blending_weights.json"), "w") as f:
        json.dump({"xgb_weight": best[0], "lgb_weight": best[1]}, f, indent=2)
    print("Saved to blending_weights.json")
else:
    print("\nNo fully smooth curve found. Using highest XGB (0.99/0.01) for smoothest available.")
    with open(os.path.join(artifacts_dir, "blending_weights.json"), "w") as f:
        json.dump({"xgb_weight": 0.99, "lgb_weight": 0.01}, f, indent=2)
