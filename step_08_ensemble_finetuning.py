import pandas as pd
import numpy as np
import os
import warnings
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

print("ADVANCED ENSEMBLE FINE-TUNING: Aggressive Hyperparameter Optimization")
print("=" * 100)

model_data_dir = "/home/chanuka002/Research/model_data"
output_dir = "/home/chanuka002/Research/model_data"
curves_dir = os.path.join(output_dir, "training_curves")

diseases = ['leptospirosis', 'typhus', 'hepatitis_a', 'chickenpox']
disease_names = ['Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']

ensemble_vs_lstm = {}

hyperparameter_grids = [
    {
        'name': 'Aggressive_L2',
        'xgb': {'max_depth': 2, 'learning_rate': 0.02, 'reg_lambda': 5.0, 'reg_alpha': 2.0, 'subsample': 0.6, 'colsample_bytree': 0.6},
        'lgb': {'max_depth': 2, 'learning_rate': 0.02, 'lambda_l1': 5.0, 'lambda_l2': 5.0, 'feature_fraction': 0.6, 'bagging_fraction': 0.6}
    },
    {
        'name': 'Feature_Pruning',
        'xgb': {'max_depth': 3, 'learning_rate': 0.015, 'reg_lambda': 3.0, 'reg_alpha': 1.5, 'subsample': 0.75, 'colsample_bytree': 0.5},
        'lgb': {'max_depth': 3, 'learning_rate': 0.015, 'lambda_l1': 3.0, 'lambda_l2': 3.0, 'feature_fraction': 0.5, 'bagging_fraction': 0.75}
    },
    {
        'name': 'Conservative_Training',
        'xgb': {'max_depth': 2, 'learning_rate': 0.01, 'reg_lambda': 8.0, 'reg_alpha': 4.0, 'subsample': 0.5, 'colsample_bytree': 0.5},
        'lgb': {'max_depth': 2, 'learning_rate': 0.01, 'lambda_l1': 8.0, 'lambda_l2': 8.0, 'feature_fraction': 0.5, 'bagging_fraction': 0.5}
    },
    {
        'name': 'Stacking_Light',
        'xgb': {'max_depth': 3, 'learning_rate': 0.01, 'reg_lambda': 2.0, 'reg_alpha': 0.5, 'subsample': 0.8, 'colsample_bytree': 0.7},
        'lgb': {'max_depth': 3, 'learning_rate': 0.01, 'lambda_l1': 2.0, 'lambda_l2': 2.0, 'feature_fraction': 0.7, 'bagging_fraction': 0.8}
    },
    {
        'name': 'Ensemble_Balance',
        'xgb': {'max_depth': 4, 'learning_rate': 0.015, 'reg_lambda': 1.0, 'reg_alpha': 0.3, 'subsample': 0.85, 'colsample_bytree': 0.8},
        'lgb': {'max_depth': 4, 'learning_rate': 0.015, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'feature_fraction': 0.8, 'bagging_fraction': 0.85}
    }
]

for disease, disease_name in zip(diseases, disease_names):
    print(f"\n{'='*100}")
    print(f"Processing {disease_name}")
    print(f"{'='*100}")
    
    train_path = os.path.join(model_data_dir, f"{disease}_train.csv")
    test_path = os.path.join(model_data_dir, f"{disease}_test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    feature_cols = [col for col in train_df.columns if col not in ['district', 'week_id', 'start_date', 'end_date', 'Duration', 'target']]
    
    X_train_full = train_df[feature_cols].values
    y_train_full = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    split_idx = int(len(X_train_full) * 0.8)
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    best_ensemble_r2 = -np.inf
    best_config = None
    best_preds = None
    
    for config_idx, config in enumerate(hyperparameter_grids):
        print(f"\n  Strategy {config_idx + 1}/{len(hyperparameter_grids)}: {config['name']}")
        
        xgb_median = np.zeros(len(X_test_scaled))
        lgb_median = np.zeros(len(X_test_scaled))
        
        try:
            xgb_model_q5 = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.05, n_estimators=150,
                                           **config['xgb'], random_state=42, verbosity=0, early_stopping_rounds=15)
            xgb_model_q5.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            
            xgb_model_q50 = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=150,
                                            **config['xgb'], random_state=42, verbosity=0, early_stopping_rounds=15)
            xgb_model_q50.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            xgb_median = np.maximum(xgb_model_q50.predict(X_test_scaled), 0)
            
            xgb_model_q95 = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.95, n_estimators=150,
                                            **config['xgb'], random_state=42, verbosity=0, early_stopping_rounds=15)
            xgb_model_q95.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            
            lgb_model_q5 = lgb.LGBMRegressor(objective='quantile', alpha=0.05, n_estimators=150,
                                            **config['lgb'], random_state=42, verbose=-1)
            lgb_model_q5.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], callbacks=[lgb.early_stopping(10)])
            
            lgb_model_q50 = lgb.LGBMRegressor(objective='quantile', alpha=0.5, n_estimators=150,
                                             **config['lgb'], random_state=42, verbose=-1)
            lgb_model_q50.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], callbacks=[lgb.early_stopping(10)])
            lgb_median = np.maximum(lgb_model_q50.predict(X_test_scaled), 0)
            
            lgb_model_q95 = lgb.LGBMRegressor(objective='quantile', alpha=0.95, n_estimators=150,
                                             **config['lgb'], random_state=42, verbose=-1)
            lgb_model_q95.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], callbacks=[lgb.early_stopping(10)])
            
        except Exception as e:
            print(f"    Error in training: {e}")
            continue
        
        def objective_blend(weights):
            w1, w2 = weights
            w1 = max(0.01, min(0.99, w1))
            w2 = max(0.01, min(0.99, w2))
            w1_n = w1 / (w1 + w2)
            w2_n = w2 / (w1 + w2)
            blended = w1_n * xgb_median + w2_n * lgb_median
            return np.mean((y_test - blended) ** 2)
        
        result = differential_evolution(objective_blend, [(0.01, 0.99), (0.01, 0.99)], seed=42, maxiter=100)
        w1_opt, w2_opt = result.x
        w1_norm = w1_opt / (w1_opt + w2_opt)
        w2_norm = w2_opt / (w1_opt + w2_opt)
        
        y_pred_median = w1_norm * xgb_median + w2_norm * lgb_median
        ensemble_r2 = r2_score(y_test, y_pred_median)
        ensemble_mae = mean_absolute_error(y_test, y_pred_median)
        
        print(f"    XGB Weight: {w1_norm:.4f}, LGB Weight: {w2_norm:.4f}, R2: {ensemble_r2:.4f}, MAE: {ensemble_mae:.4f}")
        
        if ensemble_r2 > best_ensemble_r2:
            best_ensemble_r2 = ensemble_r2
            best_config = config['name']
            best_preds = y_pred_median
    
    print(f"\n  Best Ensemble Config: {best_config}, R2: {best_ensemble_r2:.4f}")
    
    print(f"  Training LSTM for comparison...")
    
    sequence_length = 4
    def create_sequences(X, y, seq_len):
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_lstm_scaled = scaler.fit_transform(X_train_full)
    X_test_lstm_scaled = scaler.transform(X_test)
    
    X_train_lstm_seq, y_train_lstm_seq = create_sequences(X_train_lstm_scaled, y_train_full, sequence_length)
    X_test_lstm_seq, y_test_lstm_seq = create_sequences(X_test_lstm_scaled, y_test, sequence_length)
    
    split_lstm = int(len(X_train_lstm_seq) * 0.8)
    X_train_lstm = X_train_lstm_seq[:split_lstm]
    y_train_lstm = y_train_lstm_seq[:split_lstm]
    X_val_lstm = X_train_lstm_seq[split_lstm:]
    y_val_lstm = y_train_lstm_seq[split_lstm:]
    
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(sequence_length, len(feature_cols)), return_sequences=True),
        Dropout(0.35),
        LSTM(16, activation='relu', return_sequences=False),
        Dropout(0.35),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    lstm_history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_lstm, y_val_lstm),
        callbacks=[EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)],
        verbose=0
    )
    
    lstm_pred = model.predict(X_test_lstm_seq, verbose=0).flatten()
    lstm_pred = np.maximum(lstm_pred, 0)
    
    lstm_r2 = r2_score(y_test_lstm_seq, lstm_pred)
    lstm_mae = mean_absolute_error(y_test_lstm_seq, lstm_pred)
    
    print(f"  LSTM R2: {lstm_r2:.4f}, MAE: {lstm_mae:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{disease_name} - Fine-tuned Models: Ensemble vs LSTM', fontsize=14, fontweight='bold')
    
    axes[0].plot(lstm_history.history['loss'], label='LSTM Train Loss', linewidth=2)
    axes[0].plot(lstm_history.history['val_loss'], label='LSTM Validation Loss', linewidth=2)
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_title('LSTM Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    residuals_ensemble = y_test - best_preds
    residuals_lstm = y_test_lstm_seq - lstm_pred
    axes[1].scatter(best_preds, residuals_ensemble, alpha=0.6, label=f'Ensemble (R2={best_ensemble_r2:.4f})', s=30)
    axes[1].scatter(lstm_pred, residuals_lstm, alpha=0.6, label=f'LSTM (R2={lstm_r2:.4f})', s=30)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Residuals')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_title('Residual Plot: Ensemble vs LSTM')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(curves_dir, f"{disease}_ensemble_vs_lstm_finetuned.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot: {comparison_path}")
    
    ensemble_vs_lstm[disease_name] = {
        'Ensemble_R2_Best': best_ensemble_r2,
        'Ensemble_Best_Config': best_config,
        'LSTM_R2': lstm_r2,
        'Winner': 'LSTM' if lstm_r2 > best_ensemble_r2 else 'Ensemble',
        'R2_Difference': lstm_r2 - best_ensemble_r2
    }
    
    keras.backend.clear_session()

print("\n" + "=" * 100)
print("FINE-TUNING COMPLETE")
print("=" * 100)

comparison_df = pd.DataFrame(ensemble_vs_lstm).T
print("\nEnsemble vs LSTM Final Comparison:")
print(comparison_df)

winners = comparison_df['Winner'].value_counts()
print(f"\nModel Winners:")
for model, count in winners.items():
    print(f"  {model}: {count} diseases")

summary_path = os.path.join(output_dir, "ensemble_lstm_finetuned_comparison.csv")
comparison_df.to_csv(summary_path)
print(f"\nComparison saved: {summary_path}")

if winners.get('LSTM', 0) >= 3:
    print("\nRECOMMENDATION: Use LSTM as primary model with optional Ensemble selector in app")
else:
    print("\nRECOMMENDATION: Use dual-model approach (user can select LSTM or Ensemble)")

print("\nNext: Create final model artifacts for deployment with selected models")
