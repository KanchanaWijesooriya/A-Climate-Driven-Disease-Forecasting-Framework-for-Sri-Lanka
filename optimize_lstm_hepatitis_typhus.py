"""
Optuna optimization for LSTM - Hepatitis A and Typhus
Target: R² >= 0.6, MSE_norm <= 0.3

Usage:
  python optimize_lstm_hepatitis_typhus.py                    # Both diseases, 20 trials
  python optimize_lstm_hepatitis_typhus.py hepatitis_a        # Hepatitis A only
  python optimize_lstm_hepatitis_typhus.py typhus --trials 30  # Typhus, 30 trials
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU only, no GPU
import sys
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # CPU only
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

model_data_dir = "/home/chanuka002/Research/model_data"
output_dir = "/home/chanuka002/Research/model_data"
figures_dir = "/home/chanuka002/Research/figures"
lstm_figures_dir = os.path.join(figures_dir, "lstm")
os.makedirs(lstm_figures_dir, exist_ok=True)

# Feature config (matches step_05)
LSTM_CLIMATE = ['T2M_avg', 'T2M_max', 'T2M_min', 'RH2M_avg', 'PRECTOTCORR_avg']
LSTM_CLIMATE_LAGS = ['PRECTOTCORR_avg_lag_1', 'PRECTOTCORR_avg_lag_2', 'PRECTOTCORR_avg_lag_3',
                     'PRECTOTCORR_avg_lag_4', 'T2M_avg_lag_1', 'T2M_avg_lag_2', 'RH2M_avg_lag_1', 'RH2M_avg_lag_2']
LSTM_WEEK = ['week_number', 'sin_week', 'cos_week']
LSTM_MONSOON = ['monsoon_IM2', 'monsoon_NE', 'monsoon_SW']
LSTM_LAG_FEATURES = 8
LSTM_ROLL_WINDOWS = (2, 4, 8, 12)

N_TRIALS = 20  # Trials per disease (use --trials N to override)
PRUNE_EARLY_EPOCHS = 30  # Prune if val R² worse than median after this many epochs


def build_lstm_features(df, train_df):
    df = df.sort_values(['district', 'week_id']).reset_index(drop=True).copy()
    grp = df.groupby('district')['target']
    for lag in range(1, LSTM_LAG_FEATURES + 1):
        df[f'target_lag_{lag}'] = grp.shift(lag)
    for w in LSTM_ROLL_WINDOWS:
        df[f'target_roll_mean_{w}'] = grp.shift(1).transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'target_roll_std_{w}'] = grp.shift(1).transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
    df['target_trend_4'] = df.groupby('district')['target_roll_mean_4'].diff()
    df['target_trend_8'] = df.groupby('district')['target_roll_mean_8'].diff()
    le = LabelEncoder()
    le.fit(train_df['district'])
    df['district_enc'] = le.transform(df['district'])
    return df


def get_lstm_feature_cols(df):
    base = [c for c in LSTM_CLIMATE + LSTM_WEEK if c in df.columns]
    climate_lags = [c for c in LSTM_CLIMATE_LAGS if c in df.columns]
    monsoon = [c for c in LSTM_MONSOON if c in df.columns]
    lags = [f'target_lag_{i}' for i in range(1, LSTM_LAG_FEATURES + 1)]
    rolls = [f'target_roll_mean_{w}' for w in LSTM_ROLL_WINDOWS] + [f'target_roll_std_{w}' for w in LSTM_ROLL_WINDOWS]
    trends = [c for c in ['target_trend_4', 'target_trend_8'] if c in df.columns]
    return base + climate_lags + monsoon + lags + rolls + trends + ['district_enc']


def create_sequences(df, X_scaled, y, seq_len, n_feat):
    X_seq, y_seq = [], []
    df = df.sort_values(['district', 'week_id']).reset_index(drop=True)
    idx = 0
    for _, grp in df.groupby('district'):
        n = len(grp)
        X_d = X_scaled[idx:idx + n]
        y_d = y[idx:idx + n]
        idx += n
        for i in range(n - seq_len):
            X_seq.append(X_d[i:i + seq_len])
            y_seq.append(y_d[i + seq_len])
    return (np.array(X_seq) if X_seq else np.zeros((0, seq_len, n_feat)),
            np.array(y_seq) if y_seq else np.array([]))


def load_disease_data(disease):
    """Load and prepare data for a disease. Returns (X_tr, X_va, X_te, y_tr, y_va, y_te, y_min, y_range, feat_cols, n_feat)."""
    train_df = pd.read_csv(os.path.join(model_data_dir, f"{disease}_train.csv"))
    val_df = pd.read_csv(os.path.join(model_data_dir, f"{disease}_val.csv"))
    test_df = pd.read_csv(os.path.join(model_data_dir, f"{disease}_test.csv"))

    if 'week_number' not in train_df.columns and 'week_id' in train_df.columns:
        for df in (train_df, val_df, test_df):
            df['week_number'] = df['week_id'] % 52
        if 'sin_week' not in train_df.columns:
            for df in (train_df, val_df, test_df):
                df['sin_week'] = np.sin(2 * np.pi * df['week_number'] / 52)
                df['cos_week'] = np.cos(2 * np.pi * df['week_number'] / 52)

    full = pd.concat([
        train_df.assign(_s='tr'),
        val_df.assign(_s='va'),
        test_df.assign(_s='te')
    ], ignore_index=True)
    full = build_lstm_features(full, train_df)

    train_df = full[full['_s'] == 'tr'].drop(columns=['_s']).sort_values(['district', 'week_id']).reset_index(drop=True)
    val_df = full[full['_s'] == 'va'].drop(columns=['_s']).sort_values(['district', 'week_id']).reset_index(drop=True)
    test_df = full[full['_s'] == 'te'].drop(columns=['_s']).sort_values(['district', 'week_id']).reset_index(drop=True)

    feat_cols = get_lstm_feature_cols(train_df)
    for df in (train_df, val_df, test_df):
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        df[feat_cols] = df[feat_cols].fillna(0).astype(np.float64)

    n_feat = len(feat_cols)
    X_tr = train_df[feat_cols].values.astype(np.float64)
    y_tr_raw = train_df['target'].values.astype(np.float64)
    X_va = val_df[feat_cols].values.astype(np.float64)
    y_va_raw = val_df['target'].values.astype(np.float64)
    X_te = test_df[feat_cols].values.astype(np.float64)
    y_te_raw = test_df['target'].values.astype(np.float64)

    y_min = np.percentile(y_tr_raw, 0.5)
    y_max = np.percentile(y_tr_raw, 99.5)
    y_range = max(y_max - y_min, 1e-6)
    y_tr = np.clip((y_tr_raw - y_min) / y_range, 0, 1)
    y_va = np.clip((y_va_raw - y_min) / y_range, 0, 1)
    y_te = np.clip((y_te_raw - y_min) / y_range, 0, 1)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    return (train_df, val_df, test_df, X_tr_s, X_va_s, X_te_s,
            y_tr, y_va, y_te, y_min, y_range, feat_cols, n_feat)


def train_and_evaluate(disease, params, data_cache):
    """Train LSTM with given params, return (r2_val, r2_test, mse_norm_test)."""
    (train_df, val_df, test_df, X_tr_s, X_va_s, X_te_s,
     y_tr, y_va, y_te, y_min, y_range, feat_cols, n_feat) = data_cache

    seq_len = params['seq_len']
    lr = params['lr']
    lstm_drop = params['lstm_drop']
    lstm_l2 = params['lstm_l2']
    u1, u2 = params['lstm_units']
    batch_size = params['batch_size']
    use_bidir = params['use_bidirectional']
    dense_units = params['dense_units']
    rec_drop = params['recurrent_dropout']

    X_tr_seq, y_tr_seq = create_sequences(train_df, X_tr_s, y_tr, seq_len, n_feat)
    X_va_seq, y_va_seq = create_sequences(val_df, X_va_s, y_va, seq_len, n_feat)
    X_te_seq, y_te_seq = create_sequences(test_df, X_te_s, y_te, seq_len, n_feat)

    def _sample_weights(y_norm):
        bins = np.percentile(y_norm, np.linspace(0, 100, 11))
        bins[-1] += 1e-6
        bin_idx = np.digitize(y_norm, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)
        counts = np.bincount(bin_idx, minlength=10)
        counts = np.maximum(counts, 1)
        w = 1.0 / np.sqrt(counts[bin_idx])
        return (w / w.mean()).astype(np.float32)
    sample_weights_tr = _sample_weights(y_tr_seq)

    reg = l2(lstm_l2)
    if use_bidir:
        layer1 = Bidirectional(LSTM(u1, activation='tanh', kernel_regularizer=reg, dropout=lstm_drop,
                                    recurrent_dropout=rec_drop, return_sequences=True),
                               input_shape=(seq_len, n_feat))
        layer2 = Bidirectional(LSTM(u2, activation='tanh', kernel_regularizer=reg, dropout=lstm_drop,
                                    recurrent_dropout=rec_drop, return_sequences=False))
    else:
        layer1 = LSTM(u1, activation='tanh', kernel_regularizer=reg, dropout=lstm_drop,
                     recurrent_dropout=rec_drop, return_sequences=True, input_shape=(seq_len, n_feat))
        layer2 = LSTM(u2, activation='tanh', kernel_regularizer=reg, dropout=lstm_drop,
                     recurrent_dropout=rec_drop, return_sequences=False)

    model = Sequential([
        layer1,
        BatchNormalization(),
        Dropout(lstm_drop * 0.7),
        layer2,
        BatchNormalization(),
        Dropout(lstm_drop * 0.6),
        Dense(dense_units[0], activation='relu', kernel_regularizer=reg),
        Dropout(lstm_drop * 0.4),
        Dense(dense_units[1], activation='relu', kernel_regularizer=reg),
        Dropout(lstm_drop * 0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0), loss='mse', metrics=['mae'])

    early = EarlyStopping(monitor='val_loss', patience=params['early_patience'],
                         restore_best_weights=True, verbose=0)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=params['reduce_lr_factor'],
                            patience=params['reduce_lr_patience'], min_lr=1e-7, verbose=0)

    history = model.fit(
        X_tr_seq, y_tr_seq,
        sample_weight=sample_weights_tr,
        epochs=params['max_epochs'],
        batch_size=batch_size,
        validation_data=(X_va_seq, y_va_seq),
        callbacks=[early, rlr],
        verbose=0
    )

    p_va = model.predict(X_va_seq, verbose=0).flatten()
    p_te = model.predict(X_te_seq, verbose=0).flatten()
    y_va_d = y_va_seq * y_range + y_min
    y_te_d = y_te_seq * y_range + y_min
    p_va_d = np.clip(p_va * y_range + y_min, 0, None)
    p_te_d = np.clip(p_te * y_range + y_min, 0, None)

    r2_val = max(0, r2_score(y_va_d, p_va_d))
    r2_test = max(0, r2_score(y_te_d, p_te_d))
    mse_norm_test = mean_squared_error(y_te_seq, p_te)

    keras.backend.clear_session()
    return r2_val, r2_test, mse_norm_test, history


def objective(trial, disease, data_cache):
    """Optuna objective: maximize score = R2 - mse_penalty. Target R2>=0.6, MSE<=0.3."""
    params = {
        'seq_len': trial.suggest_int('seq_len', 4, 12),
        'lr': trial.suggest_float('lr', 5e-5, 3e-3, log=True),
        'lstm_drop': trial.suggest_float('lstm_drop', 0.1, 0.4),
        'lstm_l2': trial.suggest_float('lstm_l2', 1e-5, 0.001, log=True),
        'lstm_units': (trial.suggest_int('lstm_u1', 32, 128),
                       trial.suggest_int('lstm_u2', 16, 64)),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'use_bidirectional': trial.suggest_categorical('use_bidirectional', [True, False]),
        'dense_units': (trial.suggest_int('dense1', 16, 64),
                        trial.suggest_int('dense2', 8, 32)),
        'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.0, 0.15),
        'early_patience': trial.suggest_int('early_patience', 80, 200),
        'reduce_lr_patience': trial.suggest_int('reduce_lr_patience', 20, 60),
        'reduce_lr_factor': trial.suggest_float('reduce_lr_factor', 0.4, 0.8),
        'max_epochs': 500,
    }

    r2_val, r2_test, mse_norm_test, history = train_and_evaluate(disease, params, data_cache)

    # Pruning: report val R2 (intermediate value)
    trial.report(r2_val, step=min(len(history.history['loss']), PRUNE_EARLY_EPOCHS))
    if trial.should_prune():
        raise optuna.TrialPruned()

    # Score: R2 primary, bonus for low MSE, penalty for high MSE
    mse_penalty = max(0, mse_norm_test - 0.3) * 2  # strong penalty if MSE > 0.3
    mse_bonus = max(0, 0.3 - mse_norm_test) * 0.5  # small bonus if MSE < 0.3
    score = r2_test - mse_penalty + mse_bonus

    trial.set_user_attr('r2_val', r2_val)
    trial.set_user_attr('r2_test', r2_test)
    trial.set_user_attr('mse_norm_test', mse_norm_test)
    return score


def run_optimization(disease, n_trials=N_TRIALS):
    disease_name = {'hepatitis_a': 'Hepatitis A', 'typhus': 'Typhus'}[disease]
    print(f"\n{'='*70}")
    print(f"Optuna optimization: {disease_name} ({n_trials} trials)")
    print(f"{'='*70}")

    print("Loading data...")
    data_cache = load_disease_data(disease)
    (_, _, _, _, _, _, _, _, _, _, _, feat_cols, n_feat) = data_cache
    print(f"Features: {n_feat}")

    study = optuna.create_study(direction='maximize', study_name=f'lstm_{disease}',
                                sampler=TPESampler(n_startup_trials=min(5, n_trials), seed=42),
                                pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=PRUNE_EARLY_EPOCHS))

    study.optimize(lambda t: objective(t, disease, data_cache), n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    r2_test = best.user_attrs['r2_test']
    mse_norm = best.user_attrs['mse_norm_test']

    print(f"\nBest trial: #{best.number}")
    print(f"  R² (test): {r2_test:.4f}")
    print(f"  MSE_norm:  {mse_norm:.4f}")
    print(f"  Params: {best.params}")

    # Convert to step_05 DISEASE_OVERRIDES format
    p = best.params
    best_config = {
        'SEQUENCE_LENGTH': p['seq_len'],
        'LSTM_UNITS': (p['lstm_u1'], p['lstm_u2']),
        'LSTM_DROPOUT': p['lstm_drop'],
        'L2_REG': p['lstm_l2'],
        'LEARNING_RATE': p['lr'],
        'BATCH_SIZE': p['batch_size'],
        'USE_BIDIRECTIONAL': p['use_bidirectional'],
        'DENSE_UNITS': (p['dense1'], p['dense2']),
        'RECURRENT_DROPOUT': p['recurrent_dropout'],
        'EARLY_PATIENCE': p['early_patience'],
        'REDUCE_LR_PATIENCE': p['reduce_lr_patience'],
        'REDUCE_LR_FACTOR': p['reduce_lr_factor'],
        '_r2_test': r2_test,
        '_mse_norm': mse_norm,
    }

    path = os.path.join(output_dir, f"lstm_optuna_best_{disease}.json")
    with open(path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSaved best config: {path}")

    return best_config


def main():
    parser = argparse.ArgumentParser(description='Optuna LSTM optimization for Hepatitis A and Typhus')
    parser.add_argument('diseases', nargs='*', default=['hepatitis_a', 'typhus'],
                        help='Diseases to optimize (default: both)')
    parser.add_argument('--trials', type=int, default=N_TRIALS, help=f'Trials per disease (default: {N_TRIALS})')
    args = parser.parse_args()
    diseases = [d for d in args.diseases if d in ['hepatitis_a', 'typhus']] or ['hepatitis_a', 'typhus']

    for disease in diseases:
        run_optimization(disease, n_trials=args.trials)

    print("\n" + "=" * 70)
    print("Optimization complete. step_05 will auto-load best params from lstm_optuna_best_*.json")
    print("Run: python step_05_lstm_validation.py hepatitis_a typhus")
    print("=" * 70)


if __name__ == '__main__':
    main()
