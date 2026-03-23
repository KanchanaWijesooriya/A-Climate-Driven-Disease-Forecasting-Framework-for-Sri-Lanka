import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

warnings.filterwarnings('ignore')

print("Step 4.2: ARIMA Baseline Modeling - Univariate Time Series Forecasting")
print("=" * 70)

model_data_dir = "/home/chanuka002/Research/model_data"
output_dir = "/home/chanuka002/Research/model_data"

diseases = ['leptospirosis', 'typhus', 'hepatitis_a', 'chickenpox']
disease_names = ['Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']

arima_results = {}

for disease, disease_name in zip(diseases, disease_names):
    print(f"\n--- Processing {disease_name} ---")
    
    train_path = os.path.join(model_data_dir, f"{disease}_train.csv")
    test_path = os.path.join(model_data_dir, f"{disease}_test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    y_train = train_df['target'].values
    y_test = test_df['target'].values
    
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    print(f"Train target range: {y_train.min()} - {y_train.max()}")
    print(f"Test target range: {y_test.min()} - {y_test.max()}")
    
    adf_result = adfuller(y_train, autolag='AIC')
    print(f"ADF Test p-value: {adf_result[1]:.6f} (stationary: {adf_result[1] < 0.05})")
    
    fitted_model = None
    try:
        model = ARIMA(y_train, order=(1, 0, 1))
        fitted_model = model.fit()
        
        print(f"ARIMA(1,0,1) fitted successfully")
        print(f"AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")
        
        forecast = fitted_model.get_forecast(steps=len(y_test))
        pred_mean = forecast.predicted_mean
        y_pred = np.asarray(pred_mean).ravel()
        
        forecast_ci = forecast.conf_int(alpha=0.05)
        if hasattr(forecast_ci, 'iloc'):
            y_pred_lower = forecast_ci.iloc[:, 0].values
            y_pred_upper = forecast_ci.iloc[:, 1].values
        else:
            fc = np.asarray(forecast_ci)
            y_pred_lower = fc[:, 0]
            y_pred_upper = fc[:, 1]
        
    except Exception as e:
        print(f"Error fitting ARIMA: {e}")
        print(f"Falling back to simple exponential smoothing")
        y_pred = np.full(len(y_test), np.mean(y_train))
        y_pred_lower = y_pred - np.std(y_train)
        y_pred_upper = y_pred + np.std(y_train)
    
    y_pred = np.maximum(y_pred, 0)
    y_pred_lower = np.maximum(y_pred_lower, 0)
    y_pred_upper = np.maximum(y_pred_upper, 0)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    interval_coverage = np.mean((y_test >= y_pred_lower) & (y_test <= y_pred_upper))
    interval_width = np.mean(y_pred_upper - y_pred_lower)
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Interval Coverage: {interval_coverage:.4f}")
    print(f"Interval Width: {interval_width:.4f}")
    
    results_df = pd.DataFrame({
        'week_id': test_df['week_id'].values,
        'date': test_df['start_date'].values,
        'actual': y_test,
        'predicted': y_pred,
        'lower_bound': y_pred_lower,
        'upper_bound': y_pred_upper
    })
    
    results_path = os.path.join(output_dir, f"{disease}_arima_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved predictions: {results_path}")
    
    # --- ARIMA plots: fitted vs actual (train), predicted vs actual (test), uncertainty interval ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1) Train: fitted vs actual (only if we have fitted_model)
    if fitted_model is not None:
        try:
            fv = fitted_model.fittedvalues
            y_fitted = np.asarray(fv).ravel() if hasattr(fv, '__len__') else np.array([fv])
            n_fitted = len(y_fitted)
            y_train_aligned = y_train[-n_fitted:] if n_fitted < len(y_train) else y_train[:n_fitted]
            y_fitted_aligned = y_fitted[:len(y_train_aligned)] if len(y_fitted) > len(y_train_aligned) else y_fitted
            if len(y_fitted_aligned) < len(y_train_aligned):
                y_train_aligned = y_train_aligned[-len(y_fitted_aligned):]
            axes[0, 0].plot(y_train_aligned, label='Actual (train)', alpha=0.8)
            axes[0, 0].plot(y_fitted_aligned, label='Fitted (train)', alpha=0.8)
            axes[0, 0].set_title(f'{disease_name} ARIMA - Train: Actual vs Fitted')
        except Exception:
            axes[0, 0].plot(y_train, label='Actual (train)', alpha=0.8)
            axes[0, 0].set_title(f'{disease_name} ARIMA - Train Actual')
    else:
        axes[0, 0].plot(y_train, label='Actual (train)', alpha=0.8)
        axes[0, 0].set_title(f'{disease_name} ARIMA - Train Actual')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2) Test: predicted vs actual over time
    x_test = np.arange(len(y_test))
    axes[0, 1].plot(x_test, y_test, label='Actual', color='C0', alpha=0.8)
    axes[0, 1].plot(x_test, y_pred, label='Predicted', color='C1', alpha=0.8)
    axes[0, 1].fill_between(x_test, y_pred_lower, y_pred_upper, alpha=0.2, color='C1')
    axes[0, 1].set_title(f'{disease_name} ARIMA - Test: Actual vs Predicted (95% interval)')
    axes[0, 1].set_xlabel('Test week index')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3) Scatter: actual vs predicted (test)
    axes[1, 0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
    max_val = max(y_test.max(), y_pred.max()) * 1.05
    axes[1, 0].plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    axes[1, 0].set_xlabel('Actual')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].set_title(f'{disease_name} ARIMA - Test: Actual vs Predicted (scatter)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4) Residuals on test
    residuals = y_test - y_pred
    axes[1, 1].bar(np.arange(len(residuals)), residuals, color='steelblue', alpha=0.7, edgecolor='navy')
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Test week index')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].set_title(f'{disease_name} ARIMA - Test Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    arima_plot_path = os.path.join(output_dir, f"{disease}_arima_plots.png")
    plt.savefig(arima_plot_path, dpi=150, bbox_inches='tight')
    figures_dir = "/home/chanuka002/Research/figures"
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f"{disease}_arima_plots.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ARIMA plots: {arima_plot_path}")
    
    # Save fitted ARIMA model to pkl (for reproducibility / comparison)
    if fitted_model is not None:
        arima_artifacts_dir = os.path.join(output_dir, "arima_artifacts")
        os.makedirs(arima_artifacts_dir, exist_ok=True)
        arima_model_path = os.path.join(arima_artifacts_dir, f"{disease}_arima.pkl")
        joblib.dump(fitted_model, arima_model_path)
        print(f"Saved ARIMA model: {arima_model_path}")
    
    arima_results[disease_name] = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'Interval_Coverage': float(interval_coverage),
        'Interval_Width': float(interval_width),
        'Model': 'ARIMA(1,0,1)',
        'Train_Samples': len(y_train),
        'Test_Samples': len(y_test)
    }

print("\n" + "=" * 70)
print("ARIMA Baseline Modeling Complete")

metrics_summary = pd.DataFrame(arima_results).T
print("\nBaseline Metrics Summary:")
print(metrics_summary)

metrics_path = os.path.join(output_dir, "arima_baseline_metrics.csv")
metrics_summary.to_csv(metrics_path)
print(f"\nMetrics saved: {metrics_path}")

json_path = os.path.join(output_dir, "arima_results.json")
with open(json_path, 'w') as f:
    json.dump(arima_results, f, indent=4)
print(f"JSON results saved: {json_path}")

print("\nNext step: Train XGBoost and LightGBM models with quantile regression")
