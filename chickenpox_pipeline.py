"""
=============================================================================
Chickenpox Forecasting Pipeline
=============================================================================
XGBoost + LightGBM blend, 80/10/10 temporal split, district-level bias correction.
Same structure as leptospirosis pipeline (VERY AGGRESSIVE regularization).

Usage: python chickenpox_pipeline.py
=============================================================================
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

DATA_DIR   = "/home/chanuka002/Research/model_data"
OUTPUT_DIR = "/home/chanuka002/Research/figures"
DISEASE    = "chickenpox"
SEED       = 42
N_EST      = 2000
PATIENCE   = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_features(full_df, train_df_for_stats):
    df = full_df.sort_values(['district', 'week_id']).copy()
    grp = df.groupby('district')['target']
    for lag in (1, 2, 3, 4, 5, 6, 8, 12):
        df[f'target_lag_{lag}'] = grp.shift(lag)
    for w in (2, 4, 8, 12):
        s = grp.shift(1)
        df[f'target_roll_mean_{w}'] = s.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'target_roll_std_{w}']  = s.transform(lambda x: x.rolling(w, min_periods=1).std())
        df[f'target_roll_max_{w}']  = s.transform(lambda x: x.rolling(w, min_periods=1).max())
        df[f'target_roll_min_{w}']  = s.transform(lambda x: x.rolling(w, min_periods=1).min())
    df['target_trend_4'] = df.groupby('district')['target_roll_mean_4'].diff()
    df['target_trend_8'] = df.groupby('district')['target_roll_mean_8'].diff()
    df['target_accel']   = df.groupby('district')['target_trend_4'].diff()
    df['heat_humidity']  = df['T2M_avg'] * df['RH2M_avg']
    df['rain_humidity']  = df['PRECTOTCORR_avg'] * df['RH2M_avg']
    df['rain_temp']      = df['PRECTOTCORR_avg'] * df['T2M_avg']
    df['temp_range']     = df['T2M_max'] - df['T2M_min']
    df['rain_hum_lag1']  = df['PRECTOTCORR_avg_lag_1'] * df['RH2M_avg_lag_1']
    df['rain_hum_lag2']  = df['PRECTOTCORR_avg_lag_2'] * df['RH2M_avg_lag_2']
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


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:30s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# =============================================================================
# LOAD DATA & 80/10/10 TEMPORAL SPLIT
# =============================================================================

print("=" * 70)
print("  Chickenpox — XGBoost + LightGBM Blend")
print("=" * 70)

train_raw = pd.read_csv(os.path.join(DATA_DIR, f"{DISEASE}_train.csv"))
val_raw   = pd.read_csv(os.path.join(DATA_DIR, f"{DISEASE}_val.csv"))
test_raw  = pd.read_csv(os.path.join(DATA_DIR, f"{DISEASE}_test.csv"))

full_raw = pd.concat([train_raw, val_raw, test_raw], ignore_index=True)
full_raw = full_raw.sort_values('week_id').drop_duplicates()
n_total = len(full_raw)
print(f"\nData: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)}")
print(f"Split: ~{len(train_raw)/n_total*100:.0f}% / {len(val_raw)/n_total*100:.0f}% / {len(test_raw)/n_total*100:.0f}%")

full_fe = build_features(
    pd.concat([train_raw, val_raw, test_raw], ignore_index=True),
    train_raw,
)

max_tr_wk = train_raw['week_id'].max()
min_va_wk = val_raw['week_id'].min()
max_va_wk = val_raw['week_id'].max()
min_te_wk = test_raw['week_id'].min()

train_fe = full_fe[full_fe['week_id'] <= max_tr_wk].dropna(subset=['target']).copy()
val_fe   = full_fe[(full_fe['week_id'] >= min_va_wk) & (full_fe['week_id'] <= max_va_wk)].dropna(subset=['target']).copy()
test_fe  = full_fe[full_fe['week_id'] >= min_te_wk].dropna(subset=['target']).copy()

EXCLUDE = {'district', 'week_id', 'start_date', 'end_date', 'Duration', 'target', 'dist_mean'}
feature_cols = [c for c in train_fe.columns if c not in EXCLUDE]

for df in (train_fe, val_fe, test_fe):
    for c in feature_cols:
        if c in df.columns and df[c].dtype == bool:
            df[c] = df[c].astype(np.float32)
    df[feature_cols] = df[feature_cols].fillna(0)

X_train = train_fe[feature_cols].astype(np.float32)
y_train = train_fe['target'].values.astype(np.float32)
X_val   = val_fe[feature_cols].astype(np.float32)
y_val   = val_fe['target'].values.astype(np.float32)
X_test  = test_fe[feature_cols].astype(np.float32)
y_test  = test_fe['target'].values.astype(np.float32)

print(f"After FE: train={len(train_fe)} val={len(val_fe)} test={len(test_fe)}")

# =============================================================================
# XGBOOST — Native API, VERY AGGRESSIVE
# =============================================================================

print("\n[1/4] XGBoost (VERY AGGRESSIVE, native API)...")

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 2,
    'learning_rate': 0.01,
    'subsample': 0.4,
    'colsample_bytree': 0.4,
    'lambda': 3.0,
    'alpha': 2.0,
    'min_child_weight': 10,
    'gamma': 2.0,
    'seed': SEED,
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_test)

xgb_evals = {}
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=N_EST,
    evals=[(dtrain, 'train'), (dval, 'validation')],
    early_stopping_rounds=PATIENCE,
    evals_result=xgb_evals,
    verbose_eval=False,
)

xgb_best = xgb_model.best_iteration
xgb_train_pred = np.maximum(xgb_model.predict(dtrain), 0)
xgb_val_pred   = np.maximum(xgb_model.predict(dval), 0)
xgb_test_pred  = np.maximum(xgb_model.predict(dtest), 0)

print(f"  Best round: {xgb_best}")
evaluate(y_train, xgb_train_pred, "  Train")
evaluate(y_val,   xgb_val_pred,   "  Validation")
evaluate(y_test,  xgb_test_pred,  "  Test (unseen)")
xgb_gap = mean_absolute_error(y_val, xgb_val_pred) - mean_absolute_error(y_train, xgb_train_pred)
print(f"  Gap (Val−Train) MAE: {xgb_gap:.3f}")

# =============================================================================
# LIGHTGBM — Native API, VERY AGGRESSIVE
# =============================================================================

print("\n[2/4] LightGBM (VERY AGGRESSIVE, native API)...")

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'max_depth': 2,
    'num_leaves': 7,
    'learning_rate': 0.01,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.4,
    'bagging_freq': 1,
    'lambda_l1': 2.0,
    'lambda_l2': 3.0,
    'min_child_samples': 50,
    'min_data_in_leaf': 50,
    'seed': SEED,
    'verbose': -1,
}

lgb_evals_dict = {}
train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

lgb_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=N_EST,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'validation'],
    callbacks=[
        lgb.early_stopping(PATIENCE, verbose=False),
        lgb.record_evaluation(lgb_evals_dict),
    ],
)

lgb_best = lgb_model.best_iteration
lgb_train_pred = np.maximum(lgb_model.predict(X_train), 0)
lgb_val_pred   = np.maximum(lgb_model.predict(X_val), 0)
lgb_test_pred  = np.maximum(lgb_model.predict(X_test), 0)

print(f"  Best round: {lgb_best}")
evaluate(y_train, lgb_train_pred, "  Train")
evaluate(y_val,   lgb_val_pred,   "  Validation")
evaluate(y_test,  lgb_test_pred,  "  Test (unseen)")
lgb_gap = mean_absolute_error(y_val, lgb_val_pred) - mean_absolute_error(y_train, lgb_train_pred)
print(f"  Gap (Val−Train) MAE: {lgb_gap:.3f}")

# =============================================================================
# BLEND + BIAS CORRECTION
# =============================================================================

print("\n[3/4] Blend weights + district bias correction...")

best_mae, best_w = 1e9, 0.5
for w in np.arange(0.0, 1.01, 0.02):
    blended = np.maximum(w * xgb_val_pred + (1 - w) * lgb_val_pred, 0)
    m = mean_absolute_error(y_val, blended)
    if m < best_mae:
        best_mae, best_w = m, w

w_xgb, w_lgb = best_w, 1.0 - best_w
blend_val_pred  = np.maximum(w_xgb * xgb_val_pred  + w_lgb * lgb_val_pred,  0)
blend_test_pred = np.maximum(w_xgb * xgb_test_pred + w_lgb * lgb_test_pred, 0)

val_corr = val_fe[['district']].copy()
val_corr['actual'] = y_val
val_corr['pred']   = blend_val_pred
val_corr['ratio']  = np.where(val_corr['pred'] > 0.5, val_corr['actual'] / val_corr['pred'], 1.0)
district_correction = val_corr.groupby('district')['ratio'].median().clip(0.3, 4.0)

test_corr = test_fe[['district']].copy()
test_corr['pred_raw'] = blend_test_pred
test_corr['factor']   = test_corr['district'].map(district_correction).fillna(1.0)
test_corr['pred_adj'] = np.maximum(test_corr['pred_raw'] * test_corr['factor'], 0)
final_test_pred = test_corr['pred_adj'].values

# =============================================================================
# FINAL EVALUATION
# =============================================================================

print("\n[4/4] Final evaluation...")

baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, np.mean(y_train)))

print("\n" + "=" * 70)
print("  FINAL RESULTS (Test Set)")
print("=" * 70)
m_bld = evaluate(y_test, final_test_pred, "BLEND + Bias Correction ★")
print(f"\n  Baseline MAE: {baseline_mae:.3f}")
print(f"  Improvement: {(1 - m_bld['mae']/baseline_mae)*100:.1f}%")

# =============================================================================
# SAVE
# =============================================================================

pred_df = pd.DataFrame({
    'week_id': test_fe['week_id'], 'district': test_fe['district'],
    'actual': y_test, 'blend_corrected': final_test_pred, 'blend_raw': blend_test_pred,
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, f"{DISEASE}_predictions.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, f"{DISEASE}_blend_weights.json"), "w") as f:
    json.dump({
        "w_xgb": round(w_xgb, 4), "w_lgb": round(w_lgb, 4),
        "xgb_best_round": int(xgb_best), "lgb_best_round": int(lgb_best),
        "test_mae": round(m_bld['mae'], 4), "xgb_gap": round(xgb_gap, 4),
        "lgb_gap": round(lgb_gap, 4),
    }, f, indent=2)

# =============================================================================
# PLOTS
# =============================================================================

print("\nGenerating plots...")

C_TRAIN, C_VAL, C_TEST = '#e377c2', '#d62728', '#1f77b4'

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    f"Chickenpox — Training Curves\n"
    f"Test MAE={m_bld['mae']:.2f} | Gap: XGB={xgb_gap:.3f} LGB={lgb_gap:.3f}",
    fontsize=12, y=1.01,
)

xtr = xgb_evals['train']['mae']
xva = xgb_evals['validation']['mae']
n = min(len(xtr), xgb_best + 1)
axes[0, 0].plot(np.arange(1, n + 1), xtr[:n], color=C_TRAIN, label='Train', lw=1.8)
axes[0, 0].plot(np.arange(1, n + 1), xva[:n], color=C_VAL, label='Validation', lw=1.8)
axes[0, 0].axvline(n, color='gray', ls='--', alpha=0.6)
axes[0, 0].set_xlabel('Boosting Round'); axes[0, 0].set_ylabel('MAE')
axes[0, 0].set_title('XGBoost'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

lgb_keys = list(lgb_evals_dict.keys())
def _lgb_mae(idx):
    k = next((x for x in lgb_evals_dict[lgb_keys[idx]] if 'mae' in x.lower() or 'l1' in x.lower()), None)
    return lgb_evals_dict[lgb_keys[idx]][k] if k else []
ltr, lva = _lgb_mae(0), _lgb_mae(1)
n = min(len(ltr), lgb_best + 1)
axes[0, 1].plot(np.arange(1, n + 1), ltr[:n], color=C_TRAIN, label='Train', lw=1.8)
axes[0, 1].plot(np.arange(1, n + 1), lva[:n], color=C_VAL, label='Validation', lw=1.8)
axes[0, 1].axvline(n, color='gray', ls='--', alpha=0.6)
axes[0, 1].set_xlabel('Boosting Round'); axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_title('LightGBM'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

blend_train_pred = np.maximum(w_xgb * xgb_train_pred + w_lgb * lgb_train_pred, 0)
all_actual = np.concatenate([y_train, y_val, y_test])
all_pred  = np.concatenate([blend_train_pred, blend_val_pred, blend_test_pred])
colors = [C_TRAIN]*len(y_train) + [C_VAL]*len(y_val) + [C_TEST]*len(y_test)
lim = max(all_actual.max(), all_pred.max()) * 1.05
axes[1, 0].scatter(all_actual, all_pred, c=colors, alpha=0.4, s=12)
axes[1, 0].plot([0, lim], [0, lim], 'k--', lw=2)
axes[1, 0].set_xlabel('Actual'); axes[1, 0].set_ylabel('Predicted')
axes[1, 0].set_title('Train=Pink | Val=Red | Test=Blue')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(y_test, final_test_pred, alpha=0.5, s=15, color=C_TEST)
axes[1, 1].plot([0, lim], [0, lim], 'k--', lw=2)
axes[1, 1].set_xlabel('Actual'); axes[1, 1].set_ylabel('Predicted')
axes[1, 1].set_title(f'Test — MAE={m_bld["mae"]:.2f} R²={m_bld["r2"]:.3f}')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{DISEASE}_training_with_validation.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {DISEASE}_training_with_validation.png")

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(all_actual, all_pred, c=colors, alpha=0.4, s=12)
ax.plot([0, lim], [0, lim], 'k--', lw=2)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
ax.set_title('Chickenpox — Predictions vs Actual (Train=Pink, Val=Red, Test=Blue)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{DISEASE}_training_curves.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {DISEASE}_training_curves.png")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, final_test_pred, alpha=0.5, s=15, color=C_TEST)
ax.plot([0, lim], [0, lim], 'k--', lw=2)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
ax.set_title(f'Chickenpox — Predictions vs Actual (Test)\nMAE={m_bld["mae"]:.2f} R²={m_bld["r2"]:.3f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{DISEASE}_predictions_vs_actual.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {DISEASE}_predictions_vs_actual.png")

agg = pred_df.groupby('week_id').agg(actual=('actual', 'sum'), blend=('blend_corrected', 'sum')).reset_index()
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(agg['week_id'], agg['actual'], color='black', lw=2.5, label='Actual')
axes[0].plot(agg['week_id'], agg['blend'], color=C_TEST, lw=2, label='Blend + Correction')
axes[0].set_ylabel('Total Cases'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
resid = agg['actual'] - agg['blend']
axes[1].bar(agg['week_id'], resid, color=np.where(resid >= 0, C_VAL, C_TEST), alpha=0.7)
axes[1].axhline(0, color='black', lw=1); axes[1].set_xlabel('Week ID'); axes[1].set_ylabel('Residual')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{DISEASE}_timeseries.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {DISEASE}_timeseries.png")

err_df = pred_df.copy()
err_df['err'] = np.abs(err_df['actual'] - err_df['blend_corrected'])
dist_err = err_df.groupby('district')['err'].mean().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(dist_err.index, dist_err.values, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(dist_err))))
ax.set_xlabel('Mean Absolute Error'); ax.set_title('Chickenpox — Per-District MAE (Test Set)')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{DISEASE}_district_errors.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {DISEASE}_district_errors.png")

# Training curves with test
xgb_train_mae = mean_absolute_error(y_train, xgb_train_pred)
xgb_val_mae   = mean_absolute_error(y_val, xgb_val_pred)
xgb_test_mae  = mean_absolute_error(y_test, xgb_test_pred)
lgb_train_mae = mean_absolute_error(y_train, lgb_train_pred)
lgb_val_mae   = mean_absolute_error(y_val, lgb_val_pred)
lgb_test_mae  = mean_absolute_error(y_test, lgb_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Chickenpox — Training Curves with Test\n"
    f"XGBoost: Train={xgb_train_mae:.2f} Val={xgb_val_mae:.2f} Test={xgb_test_mae:.2f}\n"
    f"LightGBM: Train={lgb_train_mae:.2f} Val={lgb_val_mae:.2f} Test={lgb_test_mae:.2f}",
    fontsize=11,
)
ax = axes[0]
xtr = xgb_evals['train']['mae']
xva = xgb_evals['validation']['mae']
n = min(len(xtr), xgb_best + 100)
ax.plot(np.arange(n), xtr[:n], color='#e377c2', linewidth=2.5, label=f'Train ({xgb_train_mae:.2f})')
ax.plot(np.arange(n), xva[:n], color='#d62728', linewidth=2.5, label=f'Validation ({xgb_val_mae:.2f})')
ax.axhline(xgb_test_mae, color='#1f77b4', linestyle='--', linewidth=2.5, label=f'Test ({xgb_test_mae:.2f})')
ax.axvline(xgb_best, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Boosting Round'); ax.set_ylabel('MAE'); ax.set_title('XGBoost')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
ax = axes[1]
def _extract_lgb_metric(d):
    for k in d.keys():
        if 'mae' in k.lower() or 'l1' in k.lower():
            return d[k]
    return []
ltr = _extract_lgb_metric(lgb_evals_dict['train']) if 'train' in lgb_evals_dict else []
lva = _extract_lgb_metric(lgb_evals_dict['validation']) if 'validation' in lgb_evals_dict else []
if not ltr and 'train' in lgb_evals_dict:
    ltr, lva = _lgb_mae(0), _lgb_mae(1)
if ltr and lva:
    n = min(len(ltr), lgb_best + 100)
    ax.plot(np.arange(n), ltr[:n], color='#e377c2', linewidth=2.5, label=f'Train ({lgb_train_mae:.2f})')
    ax.plot(np.arange(n), lva[:n], color='#d62728', linewidth=2.5, label=f'Validation ({lgb_val_mae:.2f})')
    ax.axhline(lgb_test_mae, color='#1f77b4', linestyle='--', linewidth=2.5, label=f'Test ({lgb_test_mae:.2f})')
    ax.axvline(lgb_best, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Boosting Round'); ax.set_ylabel('MAE'); ax.set_title('LightGBM')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{DISEASE}_training_curves_with_test.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {DISEASE}_training_curves_with_test.png")

# Complete analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Chickenpox — Complete Analysis\nTest MAE={m_bld['mae']:.2f} | XGB Gap={xgb_gap:.3f} LGB Gap={lgb_gap:.3f}", fontsize=12)
ax = axes[0, 0]
xtr = xgb_evals['train']['mae']; xva = xgb_evals['validation']['mae']
n = min(len(xtr), xgb_best + 100)
ax.plot(np.arange(n), xtr[:n], color='#e377c2', label='Train')
ax.plot(np.arange(n), xva[:n], color='#d62728', label='Validation')
ax.axhline(xgb_test_mae, color='#1f77b4', linestyle='--', label='Test')
ax.axvline(xgb_best, color='green', linestyle=':', alpha=0.7)
ax.set_xlabel('Boosting Round'); ax.set_ylabel('MAE'); ax.set_title('XGBoost')
ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[0, 1]
if ltr and lva:
    n = min(len(ltr), lgb_best + 100)
    ax.plot(np.arange(n), ltr[:n], color='#e377c2', label='Train')
    ax.plot(np.arange(n), lva[:n], color='#d62728', label='Validation')
    ax.axhline(lgb_test_mae, color='#1f77b4', linestyle='--', label='Test')
    ax.axvline(lgb_best, color='green', linestyle=':', alpha=0.7)
ax.set_xlabel('Boosting Round'); ax.set_ylabel('MAE'); ax.set_title('LightGBM')
ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1, 0]
ax.scatter(y_train, xgb_train_pred, alpha=0.4, s=20, color='#e377c2', label='Train')
ax.scatter(y_val, xgb_val_pred, alpha=0.4, s=20, color='#d62728', label='Val')
ax.scatter(y_test, xgb_test_pred, alpha=0.4, s=20, color='#1f77b4', label='Test')
lim = max(y_train.max(), y_val.max(), y_test.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=2)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.set_title('XGBoost Predictions')
ax.legend(); ax.grid(True, alpha=0.3)
ax = axes[1, 1]
ax.scatter(y_train, lgb_train_pred, alpha=0.4, s=20, color='#e377c2', label='Train')
ax.scatter(y_val, lgb_val_pred, alpha=0.4, s=20, color='#d62728', label='Val')
ax.scatter(y_test, lgb_test_pred, alpha=0.4, s=20, color='#1f77b4', label='Test')
ax.plot([0, lim], [0, lim], 'k--', lw=2)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.set_title('LightGBM Predictions')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{DISEASE}_complete_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {DISEASE}_complete_analysis.png")

print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
print(f"\n{'Metric':<30} {'XGBoost':<20} {'LightGBM':<20}")
print("-" * 80)
print(f"{'Train MAE':<30} {xgb_train_mae:<20.4f} {lgb_train_mae:<20.4f}")
print(f"{'Validation MAE':<30} {xgb_val_mae:<20.4f} {lgb_val_mae:<20.4f}")
print(f"{'Test MAE':<30} {xgb_test_mae:<20.4f} {lgb_test_mae:<20.4f}")
print(f"{'Train-Val Gap':<30} {xgb_gap:<20.4f} {lgb_gap:<20.4f}")
print("-" * 80)
print(f"{'Blend + Correction Test MAE':<30} {m_bld['mae']:.4f}")
print("=" * 80)

print("\n" + "=" * 70)
print(f"  Chickenpox pipeline complete. Test MAE = {m_bld['mae']:.3f}")
print("=" * 70)
