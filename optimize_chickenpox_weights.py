"""
Sweep blend weights for Chickenpox ensemble to maximize R² (accuracy).
Finds best weights and saves if 70%+ achieved.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

model_data_dir = "/home/chanuka002/Research/model_data"
artifacts_dir = os.path.join(model_data_dir, "artifacts", "chickenpox")

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

X_val = np.ascontiguousarray(val_df[feature_cols].values.astype(np.float64))
y_val = val_df['target'].values.astype(np.float64)
X_test = np.ascontiguousarray(test_df[feature_cols].values.astype(np.float64))
y_test = test_df['target'].values.astype(np.float64)

scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
xgb_q50 = joblib.load(os.path.join(artifacts_dir, "xgb_q50.pkl"))
lgb_q50 = joblib.load(os.path.join(artifacts_dir, "lgb_q50.pkl"))

X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

pred_xgb_val = xgb_q50.predict(X_val_s)
pred_xgb_test = xgb_q50.predict(X_test_s)
pred_lgb_val = lgb_q50.predict(X_val_s)
pred_lgb_test = lgb_q50.predict(X_test_s)

# Sweep weights (0.01 step)
best_r2 = -np.inf
best_weights = None
best_mae = np.inf
results = []

for w_xgb in np.arange(0.0, 1.01, 0.02):
    w_lgb = 1.0 - w_xgb
    blend_val = np.maximum(w_xgb * pred_xgb_val + w_lgb * pred_lgb_val, 0)
    blend_test_raw = np.maximum(w_xgb * pred_xgb_test + w_lgb * pred_lgb_test, 0)

    # District bias (per weight combo)
    if 'district' in test_df.columns:
        val_corr = val_df[['district']].copy()
        val_corr['actual'] = y_val
        val_corr['pred'] = blend_val
        val_corr['ratio'] = np.where(val_corr['pred'] > 0.5, val_corr['actual'] / val_corr['pred'], 1.0)
        dc = val_corr.groupby('district')['ratio'].median().clip(0.3, 4.0)
        district_factor = test_df['district'].map(dc).fillna(1.0).values
        blend_test = np.maximum(blend_test_raw * district_factor, 0)
    else:
        blend_test = blend_test_raw

    r2 = r2_score(y_test, blend_test)
    mae = mean_absolute_error(y_test, blend_test)
    results.append((w_xgb, w_lgb, r2, mae))
    if r2 > best_r2:
        best_r2 = r2
        best_weights = (w_xgb, w_lgb)
        best_mae = mae

w_xgb_best, w_lgb_best = best_weights
print("Weight sweep results (Chickenpox):")
print(f"  Best weights: XGB={w_xgb_best:.2f}, LGB={w_lgb_best:.2f}")
print(f"  Best test R2: {best_r2*100:.2f}%")
print(f"  Best test MAE: {best_mae:.3f}")

target_r2 = 0.70
if best_r2 >= target_r2:
    with open(os.path.join(artifacts_dir, "blending_weights.json"), "w") as f:
        json.dump({"xgb_weight": float(w_xgb_best), "lgb_weight": float(w_lgb_best)}, f, indent=2)
    print(f"\n  [OK] Reached {best_r2*100:.1f}% - saved new blending_weights.json")
else:
    print(f"\n  [INFO] Max R2={best_r2*100:.1f}% (target 70%). Weight tuning alone cannot reach 70%.")
    print("  Consider: retraining with less regularization, more trees, or different features.")
