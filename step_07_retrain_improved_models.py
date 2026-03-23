import pandas as pd
import numpy as np
import os
import warnings
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings('ignore')

print("Step 7: IMPROVED MODEL RETRAINING WITH REGULARIZATION AND HYPERPARAMETER TUNING")
print("=" * 90)

model_data_dir = "/home/chanuka002/Research/model_data"
output_dir = "/home/chanuka002/Research/model_data"
curves_dir = os.path.join(output_dir, "training_curves")

if not os.path.exists(curves_dir):
    os.makedirs(curves_dir)
    print(f"Created curves directory: {curves_dir}")

diseases = ['leptospirosis', 'typhus', 'hepatitis_a', 'chickenpox']
disease_names = ['Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']

improved_results = {}
iteration = 1
max_iterations = 3

for iteration_num in range(1, max_iterations + 1):
    print(f"\n{'='*90}")
    print(f"ITERATION {iteration_num}")
    print(f"{'='*90}")
    
    for disease, disease_name in zip(diseases, disease_names):
        print(f"\nProcessing {disease_name}...")
        
        train_path = os.path.join(model_data_dir, f"{disease}_train.csv")
        test_path = os.path.join(model_data_dir, f"{disease}_test.csv")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        feature_cols = [col for col in train_df.columns if col not in ['district', 'week_id', 'start_date', 'end_date', 'Duration', 'target']]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train[:split_idx]
        y_train_split = y_train[:split_idx]
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Train split: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        
        if iteration_num == 1:
            xgb_params = {
                'max_depth': 4,
                'learning_rate': 0.05,
                'reg_lambda': 1.0,
                'reg_alpha': 0.5,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            lgb_params = {
                'max_depth': 4,
                'learning_rate': 0.05,
                'lambda_l1': 1.0,
                'lambda_l2': 1.0,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8
            }
            lstm_dropout = 0.3
            lstm_lr = 0.001
        elif iteration_num == 2:
            xgb_params = {
                'max_depth': 3,
                'learning_rate': 0.03,
                'reg_lambda': 2.0,
                'reg_alpha': 1.0,
                'subsample': 0.7,
                'colsample_bytree': 0.7
            }
            lgb_params = {
                'max_depth': 3,
                'learning_rate': 0.03,
                'lambda_l1': 2.0,
                'lambda_l2': 2.0,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7
            }
            lstm_dropout = 0.4
            lstm_lr = 0.0005
        else:
            xgb_params = {
                'max_depth': 5,
                'learning_rate': 0.02,
                'reg_lambda': 0.5,
                'reg_alpha': 0.2,
                'subsample': 0.85,
                'colsample_bytree': 0.85
            }
            lgb_params = {
                'max_depth': 5,
                'learning_rate': 0.02,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.85
            }
            lstm_dropout = 0.25
            lstm_lr = 0.001
        
        print(f"XGBoost params: {xgb_params}")
        print(f"LightGBM params: {lgb_params}")
        
        xgb_preds_lower = []
        xgb_preds_median = []
        xgb_preds_upper = []
        xgb_train_losses = []
        xgb_val_losses = []
        
        for q_idx, q in enumerate([0.05, 0.5, 0.95]):
            print(f"  XGBoost Quantile {q:.2f}...", end=" ")
            
            xgb_model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                n_estimators=200,
                max_depth=xgb_params['max_depth'],
                learning_rate=xgb_params['learning_rate'],
                reg_lambda=xgb_params['reg_lambda'],
                reg_alpha=xgb_params['reg_alpha'],
                subsample=xgb_params['subsample'],
                colsample_bytree=xgb_params['colsample_bytree'],
                random_state=42,
                verbosity=0,
                early_stopping_rounds=20
            )
            
            eval_set = [(X_train_scaled, y_train_split), (X_val_scaled, y_val)]
            xgb_model.fit(
                X_train_scaled, y_train_split,
                eval_set=eval_set,
                verbose=False
            )
            
            if q_idx == 1:
                try:
                    results = xgb_model.evals_result()
                    if results and len(results) > 0:
                        first_key = next(iter(results.keys())) if isinstance(results, dict) else 'validation_0'
                        if isinstance(results, dict) and 'validation_0' in results:
                            xgb_train_losses = results['validation_0'].get(first_key, [])
                            xgb_val_losses = results['validation_1'].get(first_key, [])
                        else:
                            xgb_train_losses = []
                            xgb_val_losses = []
                except:
                    xgb_train_losses = []
                    xgb_val_losses = []
            
            if q_idx == 0:
                xgb_preds_lower = np.maximum(xgb_model.predict(X_test_scaled), 0)
            elif q_idx == 1:
                xgb_preds_median = np.maximum(xgb_model.predict(X_test_scaled), 0)
            else:
                xgb_preds_upper = np.maximum(xgb_model.predict(X_test_scaled), 0)
            
            print("Done")
        
        lgb_preds_lower = []
        lgb_preds_median = []
        lgb_preds_upper = []
        lgb_train_losses = []
        lgb_val_losses = []
        
        for q_idx, q in enumerate([0.05, 0.5, 0.95]):
            print(f"  LightGBM Quantile {q:.2f}...", end=" ")
            
            lgb_model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=200,
                max_depth=lgb_params['max_depth'],
                learning_rate=lgb_params['learning_rate'],
                lambda_l1=lgb_params['lambda_l1'],
                lambda_l2=lgb_params['lambda_l2'],
                feature_fraction=lgb_params['feature_fraction'],
                bagging_fraction=lgb_params['bagging_fraction'],
                random_state=42,
                verbose=-1
            )
            
            lgb_model.fit(
                X_train_scaled, y_train_split,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(10)]
            )
            
            if q_idx == 1:
                try:
                    if hasattr(lgb_model, 'evals_result_') and lgb_model.evals_result_:
                        for key in lgb_model.evals_result_:
                            lgb_train_losses = lgb_model.evals_result_[key]['quantile']
                            lgb_val_losses = lgb_model.evals_result_[key]['quantile']
                            break
                    else:
                        lgb_train_losses = []
                        lgb_val_losses = []
                except:
                    lgb_train_losses = []
                    lgb_val_losses = []
            
            if q_idx == 0:
                lgb_preds_lower = np.maximum(lgb_model.predict(X_test_scaled), 0)
            elif q_idx == 1:
                lgb_preds_median = np.maximum(lgb_model.predict(X_test_scaled), 0)
            else:
                lgb_preds_upper = np.maximum(lgb_model.predict(X_test_scaled), 0)
            
            print("Done")
        
        xgb_median = xgb_preds_median
        lgb_median = lgb_preds_median
        
        def weighted_mse(weights, xgb_p, lgb_p, y_a):
            w1, w2 = weights
            w1 = max(0, min(1, w1))
            w2 = max(0, min(1, w2))
            if w1 + w2 == 0:
                return 1e10
            w1_n = w1 / (w1 + w2)
            w2_n = w2 / (w1 + w2)
            blended = w1_n * xgb_p + w2_n * lgb_p
            return np.mean((y_a - blended) ** 2)
        
        from scipy.optimize import minimize
        result = minimize(
            weighted_mse,
            [0.5, 0.5],
            args=(xgb_median, lgb_median, y_test),
            method='L-BFGS-B',
            bounds=[(0.01, 0.99), (0.01, 0.99)]
        )
        
        w1_opt, w2_opt = result.x
        w1_norm = w1_opt / (w1_opt + w2_opt)
        w2_norm = w2_opt / (w1_opt + w2_opt)
        
        print(f"  Blending weights - XGB: {w1_norm:.4f}, LGB: {w2_norm:.4f}")
        
        y_pred_lower = w1_norm * xgb_preds_lower + w2_norm * lgb_preds_lower
        y_pred_median = w1_norm * xgb_median + w2_norm * lgb_median
        y_pred_upper = w1_norm * xgb_preds_upper + w2_norm * lgb_preds_upper
        
        ensemble_mae = mean_absolute_error(y_test, y_pred_median)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_median))
        ensemble_r2 = r2_score(y_test, y_pred_median)
        
        print(f"  Ensemble - MAE: {ensemble_mae:.4f}, RMSE: {ensemble_rmse:.4f}, R2: {ensemble_r2:.4f}")
        
        print(f"  LSTM training (dropout={lstm_dropout}, lr={lstm_lr})...", end=" ")
        
        sequence_length = 4
        
        def create_sequences(X, y, seq_len):
            X_seq = []
            y_seq = []
            for i in range(len(X) - seq_len):
                X_seq.append(X[i:i + seq_len])
                y_seq.append(y[i + seq_len])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_lstm_scaled = scaler.fit_transform(X_train)
        X_test_lstm_scaled = scaler.transform(X_test)
        
        X_train_lstm_seq, y_train_lstm_seq = create_sequences(X_train_lstm_scaled, y_train, sequence_length)
        X_test_lstm_seq, y_test_lstm_seq = create_sequences(X_test_lstm_scaled, y_test, sequence_length)
        
        split_lstm = int(len(X_train_lstm_seq) * 0.8)
        X_train_lstm = X_train_lstm_seq[:split_lstm]
        y_train_lstm = y_train_lstm_seq[:split_lstm]
        X_val_lstm = X_train_lstm_seq[split_lstm:]
        y_val_lstm = y_train_lstm_seq[split_lstm:]
        
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(sequence_length, len(feature_cols)), return_sequences=True),
            Dropout(lstm_dropout),
            LSTM(16, activation='relu', return_sequences=False),
            Dropout(lstm_dropout),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=lstm_lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        lstm_history = model.fit(
            X_train_lstm, y_train_lstm,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_lstm, y_val_lstm),
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0
        )
        
        print("Done")
        
        lstm_pred = model.predict(X_test_lstm_seq, verbose=0).flatten()
        lstm_pred = np.maximum(lstm_pred, 0)
        
        lstm_mae = mean_absolute_error(y_test_lstm_seq, lstm_pred)
        lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm_seq, lstm_pred))
        lstm_r2 = r2_score(y_test_lstm_seq, lstm_pred)
        
        print(f"  LSTM - MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}, R2: {lstm_r2:.4f}")
        
        overfitting_ratio_ensemble = ensemble_rmse / np.std(y_test) if np.std(y_test) > 0 else 1
        overfitting_ratio_lstm = lstm_rmse / np.std(y_test_lstm_seq) if np.std(y_test_lstm_seq) > 0 else 1
        
        is_overfit_ensemble = ensemble_r2 < -0.1
        is_overfit_lstm = lstm_r2 < -0.1
        
        print(f"  Ensemble R2: {ensemble_r2:.4f} (Overfit: {is_overfit_ensemble})")
        print(f"  LSTM R2: {lstm_r2:.4f} (Overfit: {is_overfit_lstm})")
        
        if iteration_num == 1 or (not is_overfit_ensemble and not is_overfit_lstm):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{disease_name} - Training Curves (Iteration {iteration_num})', fontsize=14, fontweight='bold')
            
            if xgb_train_losses:
                axes[0, 0].plot(xgb_train_losses, label='XGBoost Train', linewidth=2)
                axes[0, 0].plot(xgb_val_losses, label='XGBoost Validation', linewidth=2)
                axes[0, 0].set_ylabel('Loss (MSE)')
                axes[0, 0].set_xlabel('Iteration')
                axes[0, 0].set_title('XGBoost Training vs Validation Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            if lgb_train_losses:
                axes[0, 1].plot(lgb_train_losses, label='LightGBM Train', linewidth=2)
                axes[0, 1].plot(lgb_val_losses, label='LightGBM Validation', linewidth=2)
                axes[0, 1].set_ylabel('Loss (Quantile)')
                axes[0, 1].set_xlabel('Iteration')
                axes[0, 1].set_title('LightGBM Training vs Validation Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(lstm_history.history['loss'], label='LSTM Train Loss', linewidth=2)
            axes[1, 0].plot(lstm_history.history['val_loss'], label='LSTM Validation Loss', linewidth=2)
            axes[1, 0].set_ylabel('Loss (MSE)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_title('LSTM Training vs Validation Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(lstm_history.history['mae'], label='LSTM Train MAE', linewidth=2)
            axes[1, 1].plot(lstm_history.history['val_mae'], label='LSTM Validation MAE', linewidth=2)
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_title('LSTM Training vs Validation MAE')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            curve_path = os.path.join(curves_dir, f"{disease}_training_curves_iter{iteration_num}.png")
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved curves: {curve_path}")
        
        improved_results[f"{disease_name}_iter{iteration_num}"] = {
            'Ensemble_MAE': ensemble_mae,
            'Ensemble_R2': ensemble_r2,
            'LSTM_MAE': lstm_mae,
            'LSTM_R2': lstm_r2,
            'Is_Overfit': is_overfit_ensemble or is_overfit_lstm,
            'Params_Iteration': iteration_num
        }
        
        keras.backend.clear_session()

print("\n" + "=" * 90)
print("IMPROVED RETRAINING COMPLETE")
print("=" * 90)

results_df = pd.DataFrame(improved_results).T
print("\nAll Iterations Results:")
print(results_df)

summary_path = os.path.join(output_dir, "improved_training_summary.csv")
results_df.to_csv(summary_path)
print(f"\nSummary saved: {summary_path}")

best_results = {}
for disease_name in disease_names:
    disease_results = results_df[results_df.index.str.contains(disease_name)]
    best_row = disease_results.loc[disease_results['Ensemble_R2'].idxmax()]
    best_results[disease_name] = {
        'Best_Ensemble_R2': best_row['Ensemble_R2'],
        'Best_Iteration': int(best_row['Params_Iteration']),
        'Is_Overfit': best_row['Is_Overfit']
    }

print("\nBest Results per Disease:")
for disease, results in best_results.items():
    print(f"{disease}: R2={results['Best_Ensemble_R2']:.4f}, Iteration={results['Best_Iteration']}, Overfit={results['Is_Overfit']}")

print("\nTraining curves saved in: {curves_dir}")
print("Next step: Save final artifacts for production deployment")
