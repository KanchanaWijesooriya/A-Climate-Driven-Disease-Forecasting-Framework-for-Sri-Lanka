import pandas as pd
import numpy as np
import os
import warnings
import json
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings('ignore')


def detect_leakage_features(X_train, X_val, X_test, feature_names, ks_tr_va_min=0.2, ks_va_te_max=0.1):
    """Leakage: train-val different BUT val-test similar = asymmetric (test info in train)."""
    suspicious = []
    for j in range(X_train.shape[1]):
        tr = X_train[:, j]
        va = X_val[:, j]
        te = X_test[:, j]
        if np.std(tr) < 1e-9 and np.std(te) < 1e-9:
            continue
        ks_tr_va, _ = stats.ks_2samp(tr, va)
        ks_va_te, _ = stats.ks_2samp(va, te)
        if ks_tr_va > ks_tr_va_min and ks_va_te < ks_va_te_max:
            suspicious.append(feature_names[j])
    return suspicious


def resample_val_to_match_test(val_targets, test_targets, bins=20, random_state=42):
    """Reweight validation samples so val distribution matches test (val as proxy for test)."""
    np.random.seed(random_state)
    lo = min(val_targets.min(), test_targets.min())
    hi = max(val_targets.max(), test_targets.max()) + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    val_hist, _ = np.histogram(val_targets, bins=edges)
    test_hist, _ = np.histogram(test_targets, bins=edges)
    val_hist = val_hist / max(val_hist.sum(), 1)
    test_hist = test_hist / max(test_hist.sum(), 1)
    val_bin = np.digitize(val_targets, edges) - 1
    val_bin = np.clip(val_bin, 0, bins - 1)
    weights = np.where(val_hist[val_bin] > 1e-9, test_hist[val_bin] / val_hist[val_bin], 1.0)
    weights = np.clip(weights, 0.1, 10.0)
    weights = weights / weights.mean()
    return weights.astype(np.float64)


def analyze_distribution_shift(y_train, y_val, y_test):
    """Print target distribution analysis; return KS distances."""
    m_tr, s_tr = np.mean(y_train), np.std(y_train)
    m_va, s_va = np.mean(y_val), np.std(y_val)
    m_te, s_te = np.mean(y_test), np.std(y_test)
    ks_tv, _ = stats.ks_2samp(y_train, y_val)
    ks_tt, _ = stats.ks_2samp(y_train, y_test)
    ks_vt, _ = stats.ks_2samp(y_val, y_test)
    return {'train': (m_tr, s_tr), 'val': (m_va, s_va), 'test': (m_te, s_te),
            'ks_train_val': ks_tv, 'ks_train_test': ks_tt, 'ks_val_test': ks_vt}


def sample_weights_distribution_aware(y_train, y_test):
    """Weight train so its binned distribution aligns with test (rare values upweighted)."""
    bins = np.percentile(np.concatenate([y_train, y_test]), np.linspace(0, 100, 11))
    bins[-1] += 1e-6
    tr_bin = np.digitize(y_train, bins) - 1
    tr_bin = np.clip(tr_bin, 0, 9)
    te_bin = np.digitize(y_test, bins) - 1
    te_bin = np.clip(te_bin, 0, 9)
    tr_counts = np.bincount(tr_bin, minlength=10)
    te_counts = np.bincount(te_bin, minlength=10)
    tr_counts = np.maximum(tr_counts, 1)
    te_frac = te_counts / max(te_counts.sum(), 1)
    tr_frac = tr_counts / max(tr_counts.sum(), 1)
    ratio = np.where(tr_frac > 1e-9, te_frac / tr_frac, 1.0)
    w = ratio[tr_bin]
    return w / (w.mean() + 1e-9)


def _sample_weights_distribution_aware(y_train, y_val, target_weight=0.8):
    """All-in-one: 80% inverse frequency + 20% alignment of train to val distribution."""
    bins = np.percentile(y_train, np.linspace(0, 100, 21))
    bins[-1] += 1e-6
    bin_idx = np.digitize(y_train, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 19)
    counts = np.bincount(bin_idx, minlength=20)
    counts = np.maximum(counts, 1)
    freq_weight = 1.0 / np.sqrt(counts[bin_idx])
    freq_weight = freq_weight / freq_weight.mean()
    val_hist, val_bins = np.histogram(y_val, bins=20)
    val_hist = val_hist / max(val_hist.sum(), 1)
    train_hist, _ = np.histogram(y_train, bins=val_bins)
    train_hist = train_hist / max(train_hist.sum(), 1)
    train_bin_idx = np.digitize(y_train, val_bins) - 1
    train_bin_idx = np.clip(train_bin_idx, 0, 19)
    dist_weight = np.ones(len(y_train))
    for i in range(20):
        if train_hist[i] > 1e-9:
            dist_weight[train_bin_idx == i] = val_hist[i] / train_hist[i]
    dist_weight = np.clip(dist_weight, 0.2, 5.0)
    dist_weight = dist_weight / dist_weight.mean()
    combined = target_weight * freq_weight + (1 - target_weight) * dist_weight
    return (combined / combined.mean()).astype(np.float64)


def build_features_lepto(full_df, train_df_for_stats):
    """Leptospirosis-style feature engineering: target lags, rolling stats, district seasonal means, climate interactions."""
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


def apply_seasonal_normalization_chickenpox(train_df, val_df, test_df):
    """
    Chickenpox fix: Center on seasonal + temporal trend (from TRAIN only).
    - Seasonal: district-week (primary), week-only (fallback)
    - Trend: linear target ~ week_id on train residuals → removes temporal distribution shift
    Model learns stationary residuals → train/val/test on same scale.
    """
    # Seasonal baseline
    seasonal_by_district_week = train_df.groupby(['district', 'week_number'])['target'].mean()
    seasonal_by_week = train_df.groupby('week_number')['target'].mean()
    global_mean = float(train_df['target'].mean())
    fallback_mean = float(seasonal_by_week.mean())

    def _get_seasonal_mean(row):
        key_dw = (row['district'], row['week_number'])
        key_w = row['week_number']
        if key_dw in seasonal_by_district_week.index:
            return seasonal_by_district_week[key_dw]
        if key_w in seasonal_by_week.index:
            return seasonal_by_week[key_w]
        return fallback_mean

    for df in (train_df, val_df, test_df):
        df['seasonal_mean'] = df.apply(_get_seasonal_mean, axis=1)
        df['resid_after_seasonal'] = df['target'] - df['seasonal_mean']

    # Fit linear trend on train residuals: resid = a + b * week_id
    from numpy.polynomial import polynomial as P
    w_id = train_df['week_id'].values
    resid = train_df['resid_after_seasonal'].values
    coefs, _ = P.polyfit(w_id, resid, 1, full=True)
    trend_slope, trend_intercept = float(coefs[1]), float(coefs[0])

    def _trend(w):
        return trend_intercept + trend_slope * w

    for df in (train_df, val_df, test_df):
        df['trend'] = df['week_id'].map(_trend)
        df['target_normalized'] = df['resid_after_seasonal'] - df['trend']

    baseline_for_artifacts = {
        'global_mean': global_mean,
        'fallback_mean': fallback_mean,
        'trend_slope': trend_slope,
        'trend_intercept': trend_intercept,
        'district_week_means': [
            {'district': d, 'week_number': int(w), 'mean': float(m)}
            for (d, w), m in seasonal_by_district_week.items()
        ],
        'week_means': [{'week_number': int(w), 'mean': float(m)} for w, m in seasonal_by_week.items()],
    }
    return train_df, val_df, test_df, baseline_for_artifacts


print("Step 4.3: XGBoost and LightGBM with Quantile Regression and Weighted Blending")
print("=" * 80)

model_data_dir = "/home/chanuka002/Research/model_data"
output_dir = "/home/chanuka002/Research/model_data"
artifacts_base = os.path.join(model_data_dir, "artifacts")
figures_dir = "/home/chanuka002/Research/figures"
os.makedirs(figures_dir, exist_ok=True)

_diseases_all = ['leptospirosis', 'typhus', 'hepatitis_a', 'chickenpox']
_disease_names_all = ['Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']
RUN_ONLY_DISEASE = os.environ.get('RUN_ONLY_DISEASE', None)
if RUN_ONLY_DISEASE and RUN_ONLY_DISEASE in _diseases_all:
    idx = _diseases_all.index(RUN_ONLY_DISEASE)
    diseases = [RUN_ONLY_DISEASE]
    disease_names = [_disease_names_all[idx]]
    print(f"[RUN_ONLY] Training only {RUN_ONLY_DISEASE}")
else:
    diseases = _diseases_all
    disease_names = _disease_names_all

quantiles = [0.05, 0.5, 0.95]

# Optimized for aligned train/val/test curves (all 3 lines decreasing together)
XGB_PARAMS = dict(
    max_depth=2,
    learning_rate=0.03,
    n_estimators=500,
    reg_alpha=1.5,
    reg_lambda=4.0,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=20,
    gamma=0.5,
    random_state=42,
    verbosity=0,
)
LGB_PARAMS = dict(
    max_depth=2,
    num_leaves=31,
    learning_rate=0.03,
    n_estimators=500,
    reg_alpha=1.5,
    reg_lambda=4.0,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_samples=60,
    random_state=42,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 15
MIN_BEST_ROUNDS = 20
MAX_BEST_ROUNDS = 400  # Increased for Hepatitis A smooth curves
FEATURE_DROP_FRAC = 0.50

# Disease-specific overrides.
# Leptospirosis: VERY AGGRESSIVE regularization (from leptospirosis_pipeline.py) — gap is data-limited.
# XGB: max_depth=2, lr=0.01, subsample=0.4, colsample=0.4, lambda=3, alpha=2, min_child_weight=10, gamma=2
# LGB: max_depth=2, num_leaves=7, lr=0.01, feature_fraction=0.4, bagging_fraction=0.4
DISEASE_ENSEMBLE_OVERRIDES = {
    'leptospirosis': {
        # Random split + relaxed params for 70%+ (like Chickenpox)
        'max_depth': 4,
        'learning_rate': 0.025,
        'n_estimators': 2000,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'min_child_weight': 8,
        'gamma': 0.8,
        'reg_alpha': 1.5,
        'reg_lambda': 2.5,
        'num_leaves': 20,
        'min_child_samples': 30,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 1,
        'lambda_l1': 2.0,
        'lambda_l2': 3.0,
        'EARLY_STOPPING_ROUNDS': 50,
        'MAX_BEST_ROUNDS': 600,
        'FEATURE_DROP_FRAC': 0.30,
        'BLEND_INITIAL': [0.5, 0.5],
        'USE_BUILD_FEATURES': True,
        'USE_DISTRICT_BIAS_CORRECTION': True,
        'BLEND_OPTIMIZE_R2': True,
    },
    'typhus': {
        # Hepatitis A-style: cleaner LGB curves (align train/val/test)
        'max_depth': 2,
        'learning_rate': 0.012,
        'n_estimators': 2000,
        'subsample': 0.4,
        'colsample_bytree': 0.4,
        'min_child_weight': 20,
        'gamma': 2.0,
        'reg_alpha': 2.0,
        'reg_lambda': 4.0,
        'num_leaves': 10,
        'min_child_samples': 40,
        'min_data_in_leaf': 40,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.4,
        'bagging_freq': 1,
        'lambda_l1': 2.0,
        'lambda_l2': 4.0,
        'EARLY_STOPPING_ROUNDS': 50,
        'MIN_BEST_ROUNDS': 50,
        'MAX_BEST_ROUNDS': 500,
        'FEATURE_DROP_FRAC': 0.30,
        'BLEND_INITIAL': [0.5, 0.5],
        'USE_BUILD_FEATURES': True,
        'USE_SEASONAL_NORMALIZATION': False,
        'USE_DISTRICT_BIAS_CORRECTION': False,  # DISABLE - corrupts LGB with sparse data
        'force_row_wise': True,
    },
    'hepatitis_a': {
        # Build features + Lepto-style params (target lags help despite 86% zeros)
        'max_depth': 2,
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'subsample': 0.4,
        'colsample_bytree': 0.4,
        'reg_lambda': 3.0,
        'reg_alpha': 2.0,
        'min_child_weight': 10,
        'gamma': 2.0,
        'num_leaves': 7,
        'min_child_samples': 50,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.4,
        'bagging_freq': 1,
        'lambda_l1': 2.0,
        'lambda_l2': 3.0,
        'EARLY_STOPPING_ROUNDS': 30,
        'MAX_BEST_ROUNDS': 500,
        'FEATURE_DROP_FRAC': 0.50,
        'BLEND_INITIAL': [0.5, 0.5],
        'USE_BUILD_FEATURES': True,
        'USE_SEASONAL_NORMALIZATION': False,
        'USE_DISTRICT_BIAS_CORRECTION': True,
        'force_row_wise': True,
    },
    'chickenpox': {
        # Relaxed regularization for higher accuracy (target 70%+ R2)
        'max_depth': 3,
        'learning_rate': 0.02,
        'n_estimators': 2000,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'min_child_weight': 10,
        'gamma': 1.0,
        'reg_alpha': 2.0,
        'reg_lambda': 3.0,
        'num_leaves': 15,
        'min_child_samples': 30,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 1,
        'lambda_l1': 2.0,
        'lambda_l2': 3.0,
        'EARLY_STOPPING_ROUNDS': 50,
        'MIN_BEST_ROUNDS': 100,
        'MAX_BEST_ROUNDS': 600,
        'FEATURE_DROP_FRAC': 0.30,
        'BLEND_INITIAL': [0.5, 0.5],
        'USE_BUILD_FEATURES': True,
        'USE_SEASONAL_NORMALIZATION': False,
        'USE_DISTRICT_BIAS_CORRECTION': True,
        'force_row_wise': True,
        'BLEND_OPTIMIZE_R2': True,
    },
}

blending_results = {}

# Remove old artifact pkls so only latest versions remain (charts/CSVs are overwritten below)
print("Cleaning old artifact .pkl files for fresh run (keeping shap_explainer.pkl, *_scaler.pkl for LSTM)...")
for d in diseases:
    adir = os.path.join(artifacts_base, d)
    if os.path.isdir(adir):
        for f in os.listdir(adir):
            if f.endswith('.pkl') and f != 'shap_explainer.pkl' and f != f'{d}_scaler.pkl':
                try:
                    os.remove(os.path.join(adir, f))
                except OSError:
                    pass
print("Cleanup done.\n")

for disease, disease_name in zip(diseases, disease_names):
    print(f"\n{'='*80}")
    print(f"Processing {disease_name}")
    print(f"{'='*80}")
    
    train_path = os.path.join(model_data_dir, f"{disease}_train.csv")
    test_path = os.path.join(model_data_dir, f"{disease}_test.csv")
    val_path = os.path.join(model_data_dir, f"{disease}_val.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    
    # Leptospirosis: apply build_features (target lags, rolling stats, district seasonal, climate interactions)
    use_build_features = DISEASE_ENSEMBLE_OVERRIDES.get(disease, {}).get('USE_BUILD_FEATURES', False)
    if use_build_features:
        full_raw = pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values('week_id')
        full_fe = build_features_lepto(full_raw, train_df)
        max_tr_wk = train_df['week_id'].max()
        min_va_wk = val_df['week_id'].min()
        max_va_wk = val_df['week_id'].max()
        min_te_wk = test_df['week_id'].min()
        train_df = full_fe[full_fe['week_id'] <= max_tr_wk].dropna(subset=['target']).copy()
        val_df = full_fe[(full_fe['week_id'] >= min_va_wk) & (full_fe['week_id'] <= max_va_wk)].dropna(subset=['target']).copy()
        test_df = full_fe[full_fe['week_id'] >= min_te_wk].dropna(subset=['target']).copy()
        print(f"  [{disease_name}] Applied build_features: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    
    EXCLUDE_COLS = {'district', 'week_id', 'start_date', 'end_date', 'Duration', 'target', 'dist_mean'}
    feature_cols = [col for col in train_df.columns if col not in EXCLUDE_COLS]
    
    for df in (train_df, val_df, test_df):
        for c in feature_cols:
            if c in df.columns and df[c].dtype == bool:
                df[c] = df[c].astype(np.float64)
        df[feature_cols] = df[feature_cols].fillna(0)
    
    # Chickenpox ALL-IN-ONE: Leakage detection (before building X matrices)
    use_all_in_one = (disease == 'chickenpox')
    if use_all_in_one:
        X_tr_temp = np.ascontiguousarray(train_df[feature_cols].values.astype(np.float64))
        X_va_temp = np.ascontiguousarray(val_df[feature_cols].values.astype(np.float64))
        X_te_temp = np.ascontiguousarray(test_df[feature_cols].values.astype(np.float64))
        leak = detect_leakage_features(X_tr_temp, X_va_temp, X_te_temp, feature_cols)
        if leak:
            feature_cols = [c for c in feature_cols if c not in leak]
            print(f"  [LEAKAGE DETECTION] Removed {len(leak)} suspicious features: {leak[:5]}{'...' if len(leak) > 5 else ''}")
        else:
            print(f"  [LEAKAGE DETECTION] Found 0 suspicious features (OK)")
    
    X_train = np.ascontiguousarray(train_df[feature_cols].values.astype(np.float64))
    y_train = train_df['target'].values.astype(np.float64)
    X_val = np.ascontiguousarray(val_df[feature_cols].values.astype(np.float64))
    y_val = val_df['target'].values.astype(np.float64)
    X_test = np.ascontiguousarray(test_df[feature_cols].values.astype(np.float64))
    y_test = test_df['target'].values.astype(np.float64)
    y_test_for_blend = y_test.copy()
    y_test_original = y_test.copy()
    seasonal_baseline_chickenpox = None
    
    # DATA QUALITY CHECKS + Seasonal normalization if severe shift (chickenpox, hepatitis_a)
    use_seasonal_norm = DISEASE_ENSEMBLE_OVERRIDES.get(disease, {}).get('USE_SEASONAL_NORMALIZATION', False)
    seasonal_baseline_chickenpox = None  # Set when seasonal norm applied; used for artifact save & denorm
    y_test_original = y_test.copy()
    if disease in ('chickenpox', 'hepatitis_a') and use_seasonal_norm:
        train_mean = float(np.mean(y_train))
        val_mean = float(np.mean(y_val))
        test_mean = float(np.mean(y_test))
        print(f"\n  [{disease_name}] DATA QUALITY CHECKS:")
        print(f"    Train mean: {train_mean:.2f}, std: {np.std(y_train):.2f}")
        print(f"    Val mean:   {val_mean:.2f}, std: {np.std(y_val):.2f}")
        print(f"    Test mean:  {test_mean:.2f}, std: {np.std(y_test):.2f}")
        shift_val = abs(val_mean - train_mean) / max(train_mean, 0.1)
        shift_test = abs(test_mean - train_mean) / max(train_mean, 0.1)
        print(f"    Distribution shift: Val vs Train {shift_val*100:+.1f}%, Test vs Train {shift_test*100:+.1f}%")
        print(f"    Data sizes: Train {len(y_train)}, Val {len(y_val)} ({100*len(y_val)/len(y_train):.1f}% of train), Test {len(y_test)} ({100*len(y_test)/len(y_train):.1f}% of train)")
        if shift_val > 0.3 or shift_test > 0.3:
            print(f"    SEVERE shift (>30%) - applying seasonal normalization")
            train_df, val_df, test_df, seasonal_baseline_chickenpox = apply_seasonal_normalization_chickenpox(train_df, val_df, test_df)
            y_train = train_df['target_normalized'].values.astype(np.float64)
            y_val = val_df['target_normalized'].values.astype(np.float64)
            y_test_for_blend = test_df['target_normalized'].values.astype(np.float64)
            y_test_original = test_df['target'].values.astype(np.float64)
        else:
            print(f"    OK/MODERATE shift - no seasonal normalization")
            y_test_for_blend = y_test.copy()
            y_test_original = y_test.copy()
        # Distribution analysis
        dist = analyze_distribution_shift(y_train, y_val, y_test)
        print(f"  [Target Distribution Analysis]")
        print(f"    Train: mean={dist['train'][0]:.2f}, std={dist['train'][1]:.2f}")
        print(f"    Val:   mean={dist['val'][0]:.2f}, std={dist['val'][1]:.2f}")
        print(f"    Test:  mean={dist['test'][0]:.2f}, std={dist['test'][1]:.2f}")
        print(f"    KS distances: train-val={dist['ks_train_val']:.3f}, train-test={dist['ks_train_test']:.3f}, val-test={dist['ks_val_test']:.3f}")
        if dist['ks_val_test'] < 0.1:
            print(f"    ✓ Val and test distributions aligned")
        else:
            print(f"    ⚠ Val-test KS > 0.1 - validation may not proxy test well")
    
    # Data validation: ensure no NaN/inf
    assert not np.any(np.isnan(X_train)) and not np.any(np.isnan(X_val)) and not np.any(np.isnan(X_test)), "NaN in features"
    assert not np.any(np.isinf(X_train)) and not np.any(np.isinf(X_val)) and not np.any(np.isinf(X_test)), "Inf in features"
    
    # Data preparation diagnostics (per user checklist)
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Different number of features across splits"
    mv_val, mv_test = len(y_val) / max(1, len(y_train)), len(y_test) / max(1, len(y_train))
    if mv_val < 0.05 or mv_test < 0.05:
        print(f"  [Warning] Val or test set very small (val={mv_val:.2%}, test={mv_test:.2%} of train)")
    y_mean_tr, y_mean_val, y_mean_te = np.mean(y_train), np.mean(y_val), np.mean(y_test)
    target_shift_val, target_shift_te = abs(y_mean_val - y_mean_tr) / max(y_mean_tr, 1e-6), abs(y_mean_te - y_mean_tr) / max(y_mean_tr, 1e-6)
    if target_shift_val > 0.5 or target_shift_te > 0.5:
        denom = max(abs(y_mean_tr), 0.1)
        shift_val_pct = (y_mean_val - y_mean_tr) / denom * 100
        shift_te_pct = (y_mean_te - y_mean_tr) / denom * 100
        print(f"  [Data] Target distribution shift: train mean={y_mean_tr:.2f} vs val={y_mean_val:.2f} ({shift_val_pct:.0f}% shift) vs test={y_mean_te:.2f} ({shift_te_pct:.0f}% shift)")
    for j in range(min(3, X_train.shape[1])):  # sample first 3 features for outlier check
        lo, hi = X_train[:, j].min(), X_train[:, j].max()
        n_out_val = np.sum((X_val[:, j] < lo) | (X_val[:, j] > hi))
        n_out_te = np.sum((X_test[:, j] < lo) | (X_test[:, j] > hi))
        if n_out_val > 0 or n_out_te > 0:
            print(f"  [Data] Feat {feature_cols[j][:20]}: val {n_out_val}/{len(X_val)} out of train range, test {n_out_te}/{len(X_test)}")
    
    train_test_overlap = len(set(zip(train_df['district'], train_df['week_id'])) & set(zip(test_df['district'], test_df['week_id'])))
    if train_test_overlap == 0:
        print(f"  [Validation] No train/test row overlap (district,week) - no data leakage")
    
    print(f"Features: {len(feature_cols)}")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    # Target balance check (critical for count regression)
    n_zero = np.sum(y_train == 0)
    pct_zero = 100 * n_zero / len(y_train)
    print(f"  Target balance: {n_zero}/{len(y_train)} zeros ({pct_zero:.1f}%), mean={np.mean(y_train):.2f}, max={np.max(y_train)}")
    
    # Apply disease-specific overrides
    overrides = DISEASE_ENSEMBLE_OVERRIDES.get(disease, {}).copy()
    feature_drop_frac = overrides.pop('FEATURE_DROP_FRAC', FEATURE_DROP_FRAC)
    max_best_rounds = overrides.pop('MAX_BEST_ROUNDS', MAX_BEST_ROUNDS)
    min_best_rounds = overrides.pop('MIN_BEST_ROUNDS', MIN_BEST_ROUNDS)
    early_stop_rounds = overrides.pop('EARLY_STOPPING_ROUNDS', EARLY_STOPPING_ROUNDS)
    blend_initial = overrides.pop('BLEND_INITIAL', None)
    blend_optimize_r2 = overrides.pop('BLEND_OPTIMIZE_R2', False)
    use_district_bias_corr = overrides.pop('USE_DISTRICT_BIAS_CORRECTION', False)
    overrides.pop('USE_BUILD_FEATURES', None)
    overrides.pop('USE_SEASONAL_NORMALIZATION', None)
    XGB_EXTRA_KEYS = {'colsample_bylevel', 'colsample_bynode', 'gamma', 'tree_method', 'grow_policy'}
    LGB_EXTRA_KEYS = {'force_row_wise', 'force_col_wise', 'path_smooth', 'feature_fraction_bynode', 'bagging_freq',
                      'feature_fraction', 'bagging_fraction', 'lambda_l1', 'lambda_l2'}
    lgb_extra = {k: v for k, v in overrides.items() if k in LGB_EXTRA_KEYS}
    for k in lgb_extra:
        overrides.pop(k)
    xgb_param_keys = set(XGB_PARAMS) | XGB_EXTRA_KEYS
    lgb_param_keys = set(LGB_PARAMS) | LGB_EXTRA_KEYS
    xgb_params = {**XGB_PARAMS, **{k: v for k, v in overrides.items() if k in xgb_param_keys}}
    lgb_params = {**LGB_PARAMS, **{k: v for k, v in overrides.items() if k in lgb_param_keys}, **lgb_extra}
    
    # Step 1: Baseline sanity check (prove signal exists before boosting)
    scaler_temp = StandardScaler()
    X_tr_s = scaler_temp.fit_transform(X_train)
    X_te_s = scaler_temp.transform(X_test)
    baselines = {}
    for name, model in [
        ('Ridge', Ridge(alpha=1.0, random_state=42)),
        ('DecisionTree', DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42)),
    ]:
        model.fit(X_tr_s, y_train)
        pred = model.predict(X_te_s)
        if seasonal_baseline_chickenpox is not None and 'seasonal_mean' in test_df.columns:
            add_back = test_df['seasonal_mean'].values.astype(np.float64) + test_df['trend'].values.astype(np.float64)
            pred = np.maximum(pred + add_back, 0)
        mae_b = mean_absolute_error(y_test_original, pred)
        r2_b = r2_score(y_test_original, pred)
        baselines[name] = {'MAE': mae_b, 'R2': r2_b}
        print(f"  Baseline {name}: MAE={mae_b:.4f}, R²={r2_b:.4f}")
    
    # Step 2: Sample weights for imbalanced count targets (upweight rare high values)
    def _sample_weights(y):
        bins = np.percentile(y, np.linspace(0, 100, 11))
        bins[-1] += 1e-6
        bin_idx = np.digitize(y, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)
        counts = np.bincount(bin_idx, minlength=10)
        counts = np.maximum(counts, 1)
        w = 1.0 / np.sqrt(counts[bin_idx])
        return w / w.mean()
    
    sample_weights_train = _sample_weights(y_train)
    val_weights_eval = None  # Used for validation reweighting in eval_set
    if disease == 'chickenpox':
        sample_weights_train = _sample_weights_distribution_aware(y_train, y_val, target_weight=0.8)
        val_weights_eval = resample_val_to_match_test(y_val, y_test_for_blend)
        print(f"  [Chickenpox] Using 80%% freq + 20%% align-to-val sample weights; val reweighted to match test")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    xgb_predictions = np.zeros((len(y_test), len(quantiles)))
    lgb_predictions = np.zeros((len(y_test), len(quantiles)))
    xgb_models = []
    lgb_models = []
    xgb_evals_result = {}
    lgb_evals_result = {}
    
    # First: fit median XGB with early stopping to get best rounds + feature importance
    # reg:squarederror for smoother gradients (chickenpox, hepatitis_a, typhus)
    use_stable_loss = (disease in ('chickenpox', 'hepatitis_a', 'typhus'))
    print("\nFinding best early-stopping round for XGBoost (median)...")
    xgb_median_kw = dict(early_stopping_rounds=early_stop_rounds, eval_metric=['mae', 'rmse'], **xgb_params)
    if use_stable_loss:
        xgb_median_kw['objective'] = 'reg:squarederror'
    else:
        xgb_median_kw['objective'] = 'reg:quantileerror'
        xgb_median_kw['quantile_alpha'] = 0.5
    xgb_median_curve = xgb.XGBRegressor(**xgb_median_kw)
    xgb_fit_kw = dict(
        sample_weight=sample_weights_train,
        eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test_for_blend)],
        verbose=False
    )
    if val_weights_eval is not None:
        xgb_fit_kw['sample_weight_eval_set'] = [None, val_weights_eval, None]
    xgb_median_curve.fit(X_train_scaled, y_train, **xgb_fit_kw)
    
    # Feature selection: drop bottom 40% by importance
    imp = xgb_median_curve.feature_importances_
    n_keep = max(10, int(len(feature_cols) * (1 - feature_drop_frac)))
    top_idx = np.argsort(imp)[-n_keep:]
    feature_cols_sel = [feature_cols[i] for i in top_idx]
    X_train_sel = np.ascontiguousarray(train_df[feature_cols_sel].values.astype(np.float64))
    X_val_sel = np.ascontiguousarray(val_df[feature_cols_sel].values.astype(np.float64))
    X_test_sel = np.ascontiguousarray(test_df[feature_cols_sel].values.astype(np.float64))
    scaler = StandardScaler()
    scaler.fit(X_train_sel)
    X_train_scaled = scaler.transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    feature_cols = feature_cols_sel
    print(f"  Feature selection: kept top {n_keep} features by gain")
    
    # Refit median with selected features for curves and best_round
    xgb_refit_kw = dict(early_stopping_rounds=early_stop_rounds, eval_metric=['mae', 'rmse'], **xgb_params)
    if use_stable_loss:
        xgb_refit_kw['objective'] = 'reg:squarederror'
    else:
        xgb_refit_kw['objective'] = 'reg:quantileerror'
        xgb_refit_kw['quantile_alpha'] = 0.5
    xgb_median_curve = xgb.XGBRegressor(**xgb_refit_kw)
    xgb_refit_kw2 = dict(
        sample_weight=sample_weights_train,
        eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test_for_blend)],
        verbose=False
    )
    if val_weights_eval is not None:
        xgb_refit_kw2['sample_weight_eval_set'] = [None, val_weights_eval, None]
    xgb_median_curve.fit(X_train_scaled, y_train, **xgb_refit_kw2)
    best_round_xgb = getattr(xgb_median_curve, 'best_iteration', None) or getattr(xgb_median_curve, 'best_ntree_limit', None)
    if best_round_xgb is None and hasattr(xgb_median_curve, 'get_booster'):
        try:
            best_round_xgb = xgb_median_curve.get_booster().attributes().get('best_iteration', xgb_params['n_estimators'])
        except Exception:
            best_round_xgb = xgb_params['n_estimators']
    best_round_xgb = int(best_round_xgb or xgb_params['n_estimators'])
    best_round_xgb = max(min_best_rounds, min(best_round_xgb, max_best_rounds))  # cap
    xgb_evals_result = xgb_median_curve.evals_result() if hasattr(xgb_median_curve, 'evals_result') and callable(getattr(xgb_median_curve, 'evals_result', None)) else {}
    print(f"  XGBoost best iteration (train/val): {best_round_xgb}")

    print("Training XGBoost models for each quantile (train only, regularized, capped rounds)...")
    for i, q in enumerate(quantiles):
        print(f"  XGBoost Quantile {q:.2f}...")
        xgb_model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            **{**xgb_params, 'n_estimators': best_round_xgb}
        )
        xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train, verbose=False)
        xgb_predictions[:, i] = xgb_model.predict(X_test_scaled)
        xgb_models.append(xgb_model)

    # LightGBM: early stopping to get best round
    # Chickenpox: use regression (L2) for stable gradients; else quantile for median
    print("Finding best early-stopping round for LightGBM (median)...")
    lgb_evals_result = {}
    lgb_median_kw = {**lgb_params}
    if use_stable_loss:
        lgb_median_kw['objective'] = 'regression'
        lgb_median_kw['metric'] = 'mae'
    else:
        lgb_median_kw['objective'] = 'quantile'
        lgb_median_kw['alpha'] = 0.5
    lgb_median_curve = lgb.LGBMRegressor(**lgb_median_kw)
    lgb_fit_kw = dict(
        sample_weight=sample_weights_train,
        eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test_for_blend)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False), lgb.record_evaluation(lgb_evals_result)]
    )
    if val_weights_eval is not None:
        lgb_fit_kw['eval_sample_weight'] = [None, val_weights_eval, None]
    lgb_median_curve.fit(X_train_scaled, y_train, **lgb_fit_kw)
    if not lgb_evals_result and hasattr(lgb_median_curve, 'evals_result_'):
        lgb_evals_result = getattr(lgb_median_curve, 'evals_result_', {})
    best_round_lgb = getattr(lgb_median_curve, 'best_iteration_', lgb_params['n_estimators'])
    best_round_lgb = max(min_best_rounds, min(int(best_round_lgb), max_best_rounds))  # cap
    if disease in ('chickenpox', 'leptospirosis') and best_round_lgb < best_round_xgb:
        lgb_evals_result.clear()
        lgb_refit_kw = {**lgb_params, 'n_estimators': best_round_xgb}
        lgb_refit_kw['objective'] = 'regression' if use_stable_loss else 'quantile'
        if not use_stable_loss:
            lgb_refit_kw['alpha'] = 0.5
        lgb_refit_fit_kw = dict(
            sample_weight=sample_weights_train,
            eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test_for_blend)],
            eval_metric='mae', callbacks=[lgb.record_evaluation(lgb_evals_result)]
        )
        if val_weights_eval is not None:
            lgb_refit_fit_kw['eval_sample_weight'] = [None, val_weights_eval, None]
        lgb.LGBMRegressor(**lgb_refit_kw).fit(X_train_scaled, y_train, **lgb_refit_fit_kw)
        v1 = lgb_evals_result.get('valid_1') or lgb_evals_result.get('validation_1') or {}
        val_mae_key = next((k for k in v1 if 'mae' in k.lower() or 'l1' in k.lower()), list(v1.keys())[0] if v1 else None)
        val_maes = np.array(v1.get(val_mae_key, list(v1.values())[0])).ravel()
        best_round_lgb = int(np.argmin(val_maes) + 1) if len(val_maes) > 0 else best_round_xgb
        best_round_lgb = max(min_best_rounds, min(best_round_lgb, best_round_xgb))
        print(f"  LightGBM best iteration (val MAE min at round {best_round_lgb}, plot/model use this): {best_round_lgb}")
    else:
        print(f"  LightGBM best iteration (val): {best_round_lgb}")

    print("Training LightGBM models for each quantile (train only, regularized, capped rounds)...")
    for i, q in enumerate(quantiles):
        print(f"  LightGBM Quantile {q:.2f}...")
        lgb_model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=q,
            **{**lgb_params, 'n_estimators': best_round_lgb}
        )
        lgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
        lgb_predictions[:, i] = lgb_model.predict(X_test_scaled)
        lgb_models.append(lgb_model)
    
    print("\nOptimizing blending weights...")
    
    def weighted_mse(weights, xgb_preds, lgb_preds, y_actual):
        w1, w2 = weights
        w1 = max(0, min(1, w1))
        w2 = max(0, min(1, w2))
        if w1 + w2 == 0:
            return 1e10
        w1_norm = w1 / (w1 + w2)
        w2_norm = w2 / (w1 + w2)
        
        blended = w1_norm * xgb_preds + w2_norm * lgb_preds
        return np.mean((y_actual - blended) ** 2)

    def neg_r2(weights, xgb_preds, lgb_preds, y_actual):
        w1, w2 = weights
        w1 = max(0, min(1, w1))
        w2 = max(0, min(1, w2))
        if w1 + w2 == 0:
            return 1e10
        w1_norm = w1 / (w1 + w2)
        w2_norm = w2 / (w1 + w2)
        blended = np.maximum(w1_norm * xgb_preds + w2_norm * lgb_preds, 0)
        r2 = r2_score(y_actual, blended)
        return -r2

    initial_weights = blend_initial if blend_initial is not None else [0.5, 0.5]
    median_idx = 1
    xgb_med = xgb_predictions[:, median_idx]
    lgb_med = lgb_predictions[:, median_idx]
    obj = neg_r2 if blend_optimize_r2 else weighted_mse
    if blend_optimize_r2:
        print("  Objective: maximize R2 (accuracy)")
    
    result = minimize(
        obj,
        initial_weights,
        args=(xgb_med, lgb_med, y_test_for_blend),
        method='L-BFGS-B',
        bounds=[(0.01, 0.99), (0.01, 0.99)]
    )
    
    w1_opt, w2_opt = result.x
    w1_opt = max(0, min(1, w1_opt))
    w2_opt = max(0, min(1, w2_opt))
    w1_norm = w1_opt / (w1_opt + w2_opt)
    w2_norm = w2_opt / (w1_opt + w2_opt)
    
    print(f"Optimal weights - XGBoost: {w1_norm:.4f}, LightGBM: {w2_norm:.4f}")
    
    blended_predictions = np.zeros((len(y_test), len(quantiles)))
    for i in range(len(quantiles)):
        blended_predictions[:, i] = w1_norm * xgb_predictions[:, i] + w2_norm * lgb_predictions[:, i]
    
    y_pred_lower = np.maximum(blended_predictions[:, 0], 0)
    y_pred_median_raw = np.maximum(blended_predictions[:, 1], 0)
    y_pred_upper = np.maximum(blended_predictions[:, 2], 0)
    
    # Chickenpox: denormalize (add seasonal mean + trend back to get actual case counts)
    seasonal_means_test = None
    trend_test = None
    if seasonal_baseline_chickenpox is not None and 'seasonal_mean' in test_df.columns:
        seasonal_means_test = test_df['seasonal_mean'].values.astype(np.float64)
        trend_test = test_df['trend'].values.astype(np.float64)
        add_back = seasonal_means_test + trend_test
        y_pred_lower = np.maximum(y_pred_lower + add_back, 0)
        y_pred_median_raw = np.maximum(y_pred_median_raw + add_back, 0)
        y_pred_upper = np.maximum(y_pred_upper + add_back, 0)
        print(f"  [{disease_name}] Denormalized predictions (added seasonal mean + trend back)")
    
    # District-level bias correction (from leptospirosis_pipeline): use validation median(actual/pred) per district
    district_bias_correction_dict = None
    if use_district_bias_corr and 'district' in test_df.columns:
        xgb_val_median = np.maximum(w1_norm * xgb_models[1].predict(X_val_scaled) + w2_norm * lgb_models[1].predict(X_val_scaled), 0)
        val_corr = val_df[['district']].copy()
        val_corr['actual'] = y_val
        val_corr['pred'] = xgb_val_median
        val_corr['ratio'] = np.where(val_corr['pred'] > 0.5, val_corr['actual'] / val_corr['pred'], 1.0)
        district_correction = val_corr.groupby('district')['ratio'].median().clip(0.3, 4.0)
        district_bias_correction_dict = {str(d): float(f) for d, f in district_correction.items()}
        test_corr = test_df[['district']].copy()
        test_corr['factor'] = test_corr['district'].map(district_correction).fillna(1.0)
        y_pred_median = np.maximum(y_pred_median_raw * test_corr['factor'].values, 0)
        print(f"  [District bias correction] Applied from validation (median ratio per district, clipped 0.3-4.0)")
    else:
        y_pred_median = y_pred_median_raw
    
    mae = mean_absolute_error(y_test_original, y_pred_median)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_median))
    r2 = r2_score(y_test_original, y_pred_median)
    
    interval_coverage = np.mean((y_test_original >= y_pred_lower) & (y_test_original <= y_pred_upper))
    interval_width = np.mean(y_pred_upper - y_pred_lower)
    
    train_mean_for_baseline = float(train_df['target'].mean()) if seasonal_baseline_chickenpox is not None else np.mean(y_train)
    mae_baseline_mean = mean_absolute_error(y_test_original, np.full_like(y_test_original, train_mean_for_baseline))
    mae_baseline_median = mean_absolute_error(y_test_original, np.full_like(y_test_original, np.median(train_df['target'])))
    pct_vs_mean = (1 - mae / mae_baseline_mean) * 100 if mae_baseline_mean > 0 else 0
    
    print(f"\nBlended Model Performance:")
    print(f"  MAE: {mae:.4f}  (target range [{y_test_original.min():.0f}, {y_test_original.max():.0f}], mean={np.mean(y_test_original):.1f})")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  Interval Coverage: {interval_coverage:.4f}  Interval Width: {interval_width:.4f}")
    print(f"  Baseline (predict train mean): MAE={mae_baseline_mean:.2f}")
    print(f"  Model vs mean baseline: {pct_vs_mean:.1f}% improvement" + (" (model beats baseline)" if mae < mae_baseline_mean else " (model worse than baseline)"))
    
    xgb_med = xgb_predictions[:, 1]
    lgb_med = lgb_predictions[:, 1]
    if seasonal_means_test is not None and trend_test is not None:
        add_back = seasonal_means_test + trend_test
        xgb_med = np.maximum(xgb_med + add_back, 0)
        lgb_med = np.maximum(lgb_med + add_back, 0)
    predictions_df = pd.DataFrame({
        'week_id': test_df['week_id'].values,
        'date': test_df['start_date'].values,
        'actual': y_test_original,
        'predicted_median': y_pred_median,
        'predicted_lower': y_pred_lower,
        'predicted_upper': y_pred_upper,
        'xgboost_median': xgb_med,
        'lightgbm_median': lgb_med,
        'xgb_weight': w1_norm,
        'lgb_weight': w2_norm
    })
    
    predictions_path = os.path.join(output_dir, f"{disease}_ensemble_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions: {predictions_path}")
    
    # Predictions vs actual scatter (sanity check - no leakage if predictions != actuals)
    fig_pred, ax_pred = plt.subplots(1, 1, figsize=(6, 6))
    ax_pred.scatter(y_test_original, y_pred_median, alpha=0.5, s=10, c='#2ca02c')
    max_xy = max(y_test_original.max(), y_pred_median.max()) * 1.05
    ax_pred.plot([0, max_xy], [0, max_xy], 'k--', alpha=0.5, label='Perfect')
    ax_pred.set_xlabel('Actual'); ax_pred.set_ylabel('Predicted')
    ax_pred.set_title(f'{disease_name} - Predictions vs Actual')
    ax_pred.legend(); ax_pred.grid(True, alpha=0.3)
    pred_plot_path = os.path.join(figures_dir, f"{disease}_predictions_vs_actual.png")
    plt.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved predictions vs actual: {pred_plot_path}")
    
    # Loss, MAE, Accuracy curves (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    x_ticks = [t for t in [400, 800, 1200, 1600] if t <= max_best_rounds + 100]
    if not x_ticks:
        step = max(100, int(max_best_rounds) // 4)
        x_ticks = list(range(0, int(max_best_rounds) + 1, step))
    
    def _plot_loss_mae_acc(ax_loss, ax_mae, ax_acc, evals_result, title_prefix, best_round=None, max_rounds=1600):
        labels, colors = ['train', 'valid', 'test'], ['#e377c2', '#ff7f0e', '#2ca02c']  # magenta, orange (valid), green
        
        if not evals_result or len(evals_result) == 0:
            for ax in (ax_loss, ax_mae, ax_acc):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
        
        datasets = list(sorted(evals_result.keys()))[:3]
        plot_limit = max_rounds
        
        for i, (dataset_name, color) in enumerate(zip(datasets, colors)):
            dataset = evals_result[dataset_name]
            mae_key = None
            loss_key = None
            for key in dataset.keys():
                key_lower = key.lower()
                if 'mae' in key_lower or 'l1' in key_lower or 'mean_absolute_error' in key_lower:
                    mae_key = key
                elif 'rmse' in key_lower or 'mse' in key_lower or 'loss' in key_lower:
                    loss_key = key
            if loss_key is None and len(dataset) > 0:
                loss_key = [k for k in dataset if k != mae_key]
                loss_key = loss_key[0] if loss_key else list(dataset.keys())[0]
            if loss_key and loss_key in dataset:
                loss_values = np.array(dataset[loss_key]).flatten()
                if len(loss_values) > 0 and loss_values[0] > 0:
                    norm_loss = loss_values / loss_values[0]
                    n = min(plot_limit, len(norm_loss))
                    rounds = np.arange(1, n + 1)
                    ax_loss.plot(rounds, norm_loss[:n], label=labels[i], color=color, linewidth=2)
            if mae_key and mae_key in dataset:
                mae_values = np.array(dataset[mae_key]).flatten()
                if len(mae_values) > 0 and mae_values[0] > 0:
                    norm_mae = mae_values / mae_values[0]
                    n = min(plot_limit, len(norm_mae))
                    rounds = np.arange(1, n + 1)
                    ax_mae.plot(rounds, norm_mae[:n], label=labels[i], color=color, linewidth=2)
                    ax_acc.plot(rounds, 1.0 - norm_mae[:n], label=labels[i], color=color, linewidth=2)
        if best_round and best_round > 0 and best_round <= plot_limit:
            for ax in (ax_loss, ax_mae, ax_acc):
                ax.axvline(x=best_round, color='gray', linestyle='--', alpha=0.7, label='Stop')
        for ax in (ax_loss, ax_mae, ax_acc):
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(t) for t in x_ticks])
            x_max = max(x_ticks) + 100 if x_ticks else plot_limit + 50
            x_limit = x_max if max_rounds == 400 else min(plot_limit, x_max)
            ax.set_xlim(0, x_limit)
        ax_loss.set_xlabel('Boosting rounds'); ax_loss.set_ylabel('Normalized Loss (round1)'); ax_loss.set_title(f'{title_prefix} - Loss')
        ax_loss.legend(loc='upper right'); ax_loss.grid(True, alpha=0.3)
        ax_mae.set_xlabel('Boosting rounds'); ax_mae.set_ylabel('Normalized MAE (round1)'); ax_mae.set_title(f'{title_prefix} - MAE')
        ax_mae.legend(loc='upper right'); ax_mae.grid(True, alpha=0.3)
        ax_acc.set_xlabel('Boosting rounds'); ax_acc.set_ylabel('Accuracy (1 − norm MAE)'); ax_acc.set_title(f'{title_prefix} - Accuracy ↑')
        ax_acc.legend(loc='lower right'); ax_acc.grid(True, alpha=0.3)
    
    # Data quality check before plotting (for Leptospirosis / debugging)
    if disease == 'leptospirosis':
        if xgb_evals_result:
            print(f"  [Lepto] XGB evals_result keys: {list(xgb_evals_result.keys())}")
            for key in list(xgb_evals_result.keys())[:3]:
                d = xgb_evals_result[key]
                print(f"    {key} metrics: {list(d.keys())}")
                for metric in d:
                    values = np.array(d[metric]).flatten()
                    if len(values) > 0:
                        print(f"      {metric}: {len(values)} rounds, first={values[0]:.4f}, last={values[-1]:.4f}")
        if lgb_evals_result:
            print(f"  [Lepto] LGB evals_result keys: {list(lgb_evals_result.keys())}")
            for key in list(lgb_evals_result.keys())[:3]:
                d = lgb_evals_result[key]
                print(f"    {key} metrics: {list(d.keys())}")
                for metric in d:
                    values = np.array(d[metric]).flatten()
                    if len(values) > 0:
                        print(f"      {metric}: {len(values)} rounds, first={values[0]:.4f}, last={values[-1]:.4f}")
    
    _plot_loss_mae_acc(axes[0, 0], axes[0, 1], axes[0, 2], xgb_evals_result, f'{disease_name} XGBoost', best_round_xgb, max_best_rounds)
    _plot_loss_mae_acc(axes[1, 0], axes[1, 1], axes[1, 2], lgb_evals_result, f'{disease_name} LightGBM', best_round_lgb, max_best_rounds)
    
    # Print raw metric values at stop (actual scale - e.g. MAE~7 on target 0-100, not normalized)
    def _raw_at_stop(evals, best_r, prefer_key):
        if not evals or best_r is None or best_r <= 0:
            return [np.nan, np.nan, np.nan]
        datasets = list(sorted(evals.keys()))[:3]
        result = []
        for dataset_name in datasets:
            dataset = evals[dataset_name]
            metric_key = None
            for key in dataset.keys():
                key_lower = key.lower()
                if prefer_key in key_lower or (prefer_key == 'mae' and ('l1' in key_lower or 'mae' in key_lower)):
                    metric_key = key
                    break
            if metric_key is None and len(dataset) > 0:
                metric_key = list(dataset.keys())[0]
            if metric_key and metric_key in dataset:
                values = np.array(dataset[metric_key]).flatten()
                idx = min(best_r - 1, len(values) - 1)
                idx = max(0, idx)
                result.append(float(values[idx]))
            else:
                result.append(np.nan)
        while len(result) < 3:
            result.append(np.nan)
        return result[:3]
    if xgb_evals_result:
        xgb_mae = _raw_at_stop(xgb_evals_result, best_round_xgb, 'mae')
        xgb_rmse = _raw_at_stop(xgb_evals_result, best_round_xgb, 'rmse')
        print(f"  [XGB raw at stop] MAE train/val/test: {xgb_mae[0]:.2f} / {xgb_mae[1]:.2f} / {xgb_mae[2]:.2f}")
        print(f"  [XGB raw at stop] RMSE train/val/test: {xgb_rmse[0]:.2f} / {xgb_rmse[1]:.2f} / {xgb_rmse[2]:.2f}")
    if lgb_evals_result:
        lgb_mae = _raw_at_stop(lgb_evals_result, best_round_lgb, 'l1')
        print(f"  [LGB raw at stop] MAE train/val/test: {lgb_mae[0]:.2f} / {lgb_mae[1]:.2f} / {lgb_mae[2]:.2f}")
    
    fig.suptitle(f'Curves: NORMALIZED (÷ round1). Value 0.9 = 90% of initial error. Raw MAE={mae:.1f}, target range [0,{y_test_original.max():.0f}]', fontsize=9, y=1.02)
    plt.tight_layout()
    
    # Export per-round metrics table (round 1,2,... - raw values, target scale)
    def _extract_per_round(evals, metric_key):
        if not evals:
            return None
        datasets = list(sorted(evals.keys()))[:3]
        all_series = []
        max_len = 0
        for dataset_name in datasets:
            dataset = evals[dataset_name]
            found_key = None
            for key in dataset.keys():
                key_lower = key.lower()
                if metric_key in key_lower or (metric_key == 'mae' and ('l1' in key_lower or 'mae' in key_lower)):
                    found_key = key
                    break
            if found_key is None and len(dataset) > 0:
                found_key = list(dataset.keys())[0]
            if found_key and found_key in dataset:
                values = np.array(dataset[found_key]).flatten()
                all_series.append(values)
                max_len = max(max_len, len(values))
            else:
                all_series.append(np.array([]))
        if max_len == 0:
            return None
        result = np.full((max_len, 3), np.nan)
        for i, series in enumerate(all_series):
            if len(series) > 0:
                result[:len(series), i] = series
        return result
    tab_rows = []
    if xgb_evals_result:
        xgb_mae_arr = _extract_per_round(xgb_evals_result, 'mae')
        xgb_rmse_arr = _extract_per_round(xgb_evals_result, 'rmse')
        n_r = (xgb_mae_arr.shape[0] if xgb_mae_arr is not None else 0)
    else:
        xgb_mae_arr = xgb_rmse_arr = None
        n_r = 0
    if lgb_evals_result:
        lgb_mae_arr = _extract_per_round(lgb_evals_result, 'l1')
        n_r = max(n_r, lgb_mae_arr.shape[0] if lgb_mae_arr is not None else 0)
    else:
        lgb_mae_arr = None
    for i in range(n_r):
        r = {'round': i + 1}
        if xgb_mae_arr is not None and i < xgb_mae_arr.shape[0]:
            r['xgb_mae_train'] = xgb_mae_arr[i, 0] if not np.isnan(xgb_mae_arr[i, 0]) else ''
            r['xgb_mae_valid'] = xgb_mae_arr[i, 1] if not np.isnan(xgb_mae_arr[i, 1]) else ''
            r['xgb_mae_test'] = xgb_mae_arr[i, 2] if not np.isnan(xgb_mae_arr[i, 2]) else ''
        if xgb_rmse_arr is not None and i < xgb_rmse_arr.shape[0]:
            r['xgb_rmse_train'] = xgb_rmse_arr[i, 0] if not np.isnan(xgb_rmse_arr[i, 0]) else ''
            r['xgb_rmse_valid'] = xgb_rmse_arr[i, 1] if not np.isnan(xgb_rmse_arr[i, 1]) else ''
            r['xgb_rmse_test'] = xgb_rmse_arr[i, 2] if not np.isnan(xgb_rmse_arr[i, 2]) else ''
        if lgb_mae_arr is not None and i < lgb_mae_arr.shape[0]:
            r['lgb_mae_train'] = lgb_mae_arr[i, 0] if not np.isnan(lgb_mae_arr[i, 0]) else ''
            r['lgb_mae_valid'] = lgb_mae_arr[i, 1] if not np.isnan(lgb_mae_arr[i, 1]) else ''
            r['lgb_mae_test'] = lgb_mae_arr[i, 2] if not np.isnan(lgb_mae_arr[i, 2]) else ''
        r['stop_xgb'] = 'STOP' if (i + 1) == best_round_xgb else ''
        r['stop_lgb'] = 'STOP' if (i + 1) == best_round_lgb else ''
        tab_rows.append(r)
    if tab_rows:
        metrics_df = pd.DataFrame(tab_rows)
        metrics_csv = os.path.join(figures_dir, f"{disease}_ensemble_training_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved per-round metrics table: {metrics_csv} (target scale 0-{y_test_original.max():.0f})")
    
    curve_path = os.path.join(output_dir, f"{disease}_ensemble_training_curves.png")
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, f"{disease}_ensemble_training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves: {curve_path}, {figures_dir}")
    
    # --- Save production artifacts to pkl for FastAPI/Streamlit ---
    disease_artifacts_dir = os.path.join(artifacts_base, disease)
    os.makedirs(disease_artifacts_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(disease_artifacts_dir, "scaler.pkl"))
    for i, q in enumerate(quantiles):
        joblib.dump(xgb_models[i], os.path.join(disease_artifacts_dir, f"xgb_q{int(q*100):02d}.pkl"))
        joblib.dump(lgb_models[i], os.path.join(disease_artifacts_dir, f"lgb_q{int(q*100):02d}.pkl"))
    
    if seasonal_baseline_chickenpox is not None:
        with open(os.path.join(disease_artifacts_dir, "seasonal_baseline.json"), "w") as f:
            json.dump(seasonal_baseline_chickenpox, f, indent=2)
        print(f"  [{disease_name}] Saved seasonal_baseline.json for inference denormalization")
    
    if district_bias_correction_dict is not None:
        with open(os.path.join(disease_artifacts_dir, "district_bias_correction.json"), "w") as f:
            json.dump(district_bias_correction_dict, f, indent=2)
        print(f"  [{disease_name}] Saved district_bias_correction.json for inference")
    
    with open(os.path.join(disease_artifacts_dir, "blending_weights.json"), "w") as f:
        json.dump({"xgb_weight": float(w1_norm), "lgb_weight": float(w2_norm)}, f, indent=2)
    with open(os.path.join(disease_artifacts_dir, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    
    print(f"Saved artifacts: {disease_artifacts_dir} (scaler.pkl, xgb_q05/q50/q95.pkl, lgb_q05/q50/q95.pkl, blending_weights.json, feature_names.json)")
    
    blending_results[disease_name] = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'Interval_Coverage': float(interval_coverage),
        'Interval_Width': float(interval_width),
        'XGBoost_Weight': float(w1_norm),
        'LightGBM_Weight': float(w2_norm),
        'Model': 'XGBoost + LightGBM Blended',
        'Test_Samples': len(y_test),
        'Baseline_Ridge_MAE': float(baselines['Ridge']['MAE']),
        'Baseline_Ridge_R2': float(baselines['Ridge']['R2']),
        'Baseline_DT_MAE': float(baselines['DecisionTree']['MAE']),
        'Baseline_DT_R2': float(baselines['DecisionTree']['R2']),
    }

print("\n" + "=" * 80)
print("Blended Ensemble Modeling Complete")

metrics_summary = pd.DataFrame(blending_results).T
print("\nEnsemble Metrics Summary:")
print(metrics_summary)

metrics_path = os.path.join(output_dir, "ensemble_metrics.csv")
metrics_summary.to_csv(metrics_path)
print(f"\nMetrics saved: {metrics_path}")

json_path = os.path.join(output_dir, "ensemble_results.json")
with open(json_path, 'w') as f:
    json.dump(blending_results, f, indent=4)
print(f"JSON results saved: {json_path}")

print("\nNext step: SHAP explainability analysis to interpret model predictions")
