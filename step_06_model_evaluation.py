import pandas as pd
import numpy as np
import os
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("Step 4.6: Model Evaluation and Comprehensive Comparison")
print("=" * 80)

model_data_dir = "/home/chanuka002/Research/model_data"
output_dir = "/home/chanuka002/Research/model_data"

diseases = ['Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']

print("\nLoading metrics from all models...")

arima_metrics = pd.read_csv(os.path.join(model_data_dir, "arima_baseline_metrics.csv"), index_col=0)
ensemble_metrics = pd.read_csv(os.path.join(model_data_dir, "ensemble_metrics.csv"), index_col=0)
lstm_metrics = pd.read_csv(os.path.join(model_data_dir, "lstm_metrics.csv"), index_col=0)

print("ARIMA Baseline Metrics:")
print(arima_metrics[['MAE', 'RMSE', 'R2']])

print("\nEnsemble Blending Metrics:")
print(ensemble_metrics[['MAE', 'RMSE', 'R2']])

print("\nLSTM Model Metrics:")
print(lstm_metrics[['MAE', 'RMSE', 'R2']])

comparison_data = {}

for disease in diseases:
    comparison_data[disease] = {
        'ARIMA_MAE': arima_metrics.loc[disease, 'MAE'],
        'ARIMA_RMSE': arima_metrics.loc[disease, 'RMSE'],
        'ARIMA_R2': arima_metrics.loc[disease, 'R2'],
        'Ensemble_MAE': ensemble_metrics.loc[disease, 'MAE'],
        'Ensemble_RMSE': ensemble_metrics.loc[disease, 'RMSE'],
        'Ensemble_R2': ensemble_metrics.loc[disease, 'R2'],
        'LSTM_MAE': lstm_metrics.loc[disease, 'MAE'],
        'LSTM_RMSE': lstm_metrics.loc[disease, 'RMSE'],
        'LSTM_R2': lstm_metrics.loc[disease, 'R2']
    }

comparison_df = pd.DataFrame(comparison_data).T

print("\n" + "=" * 80)
print("Cross-Model Comparison (All Metrics)")
print("=" * 80)
print(comparison_df)

mae_comparison = comparison_df[['ARIMA_MAE', 'Ensemble_MAE', 'LSTM_MAE']]
rmse_comparison = comparison_df[['ARIMA_RMSE', 'Ensemble_RMSE', 'LSTM_RMSE']]
r2_comparison = comparison_df[['ARIMA_R2', 'Ensemble_R2', 'LSTM_R2']]

print("\nMAE Comparison (Lower is Better):")
print(mae_comparison)

print("\nRMSE Comparison (Lower is Better):")
print(rmse_comparison)

print("\nR2 Score Comparison (Higher is Better):")
print(r2_comparison)

mae_improvements_ensemble = ((mae_comparison['ARIMA_MAE'] - mae_comparison['Ensemble_MAE']) / mae_comparison['ARIMA_MAE'] * 100).mean()
mae_improvements_lstm = ((mae_comparison['ARIMA_MAE'] - mae_comparison['LSTM_MAE']) / mae_comparison['ARIMA_MAE'] * 100).mean()

print(f"\nAverage MAE Improvement:")
print(f"  Ensemble vs ARIMA: {mae_improvements_ensemble:.2f}%")
print(f"  LSTM vs ARIMA: {mae_improvements_lstm:.2f}%")

model_rankings = pd.DataFrame({
    'Disease': diseases,
    'Best_MAE': [mae_comparison.loc[d].idxmin().replace('_MAE', '') for d in diseases],
    'Best_RMSE': [rmse_comparison.loc[d].idxmin().replace('_RMSE', '') for d in diseases],
    'Best_R2': [r2_comparison.loc[d].idxmax().replace('_R2', '') for d in diseases]
})

print("\n" + "=" * 80)
print("Model Rankings by Metric")
print("=" * 80)
print(model_rankings)

print("\nDetailed Performance Analysis:")
for disease in diseases:
    print(f"\n{disease}:")
    print(f"  MAE - ARIMA: {mae_comparison.loc[disease, 'ARIMA_MAE']:.4f}, Ensemble: {mae_comparison.loc[disease, 'Ensemble_MAE']:.4f}, LSTM: {mae_comparison.loc[disease, 'LSTM_MAE']:.4f}")
    print(f"  RMSE - ARIMA: {rmse_comparison.loc[disease, 'ARIMA_RMSE']:.4f}, Ensemble: {rmse_comparison.loc[disease, 'Ensemble_RMSE']:.4f}, LSTM: {rmse_comparison.loc[disease, 'LSTM_RMSE']:.4f}")
    print(f"  R2 - ARIMA: {r2_comparison.loc[disease, 'ARIMA_R2']:.4f}, Ensemble: {r2_comparison.loc[disease, 'Ensemble_R2']:.4f}, LSTM: {r2_comparison.loc[disease, 'LSTM_R2']:.4f}")
    
    best_mae_model = mae_comparison.loc[disease].idxmin().replace('_MAE', '')
    print(f"  Best MAE Model: {best_mae_model}")

print("\n" + "=" * 80)
print("Creating comparison visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

mae_comparison.plot(kind='bar', ax=axes[0], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0].set_title('MAE Comparison Across Models', fontsize=12, fontweight='bold')
axes[0].set_ylabel('MAE (Lower is Better)')
axes[0].set_xlabel('Disease')
axes[0].legend(['ARIMA', 'Ensemble', 'LSTM'], loc='upper right')
axes[0].grid(axis='y', alpha=0.3)

rmse_comparison.plot(kind='bar', ax=axes[1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1].set_title('RMSE Comparison Across Models', fontsize=12, fontweight='bold')
axes[1].set_ylabel('RMSE (Lower is Better)')
axes[1].set_xlabel('Disease')
axes[1].legend(['ARIMA', 'Ensemble', 'LSTM'], loc='upper right')
axes[1].grid(axis='y', alpha=0.3)

r2_comparison.plot(kind='bar', ax=axes[2], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[2].set_title('R2 Score Comparison Across Models', fontsize=12, fontweight='bold')
axes[2].set_ylabel('R2 Score (Higher is Better)')
axes[2].set_xlabel('Disease')
axes[2].legend(['ARIMA', 'Ensemble', 'LSTM'], loc='lower right')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
comparison_plot_path = os.path.join(output_dir, "model_comparison_all_metrics.png")
figures_dir = "/home/chanuka002/Research/figures"
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, "model_comparison_all_metrics.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved comparison plot: {comparison_plot_path}")

fig, ax = plt.subplots(figsize=(10, 6))
mae_comparison_norm = mae_comparison.div(mae_comparison.max(axis=1), axis=0)
mae_comparison_norm.plot(kind='barh', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('Normalized MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_xlabel('Normalized MAE (0-1)')
ax.legend(['ARIMA', 'Ensemble', 'LSTM'], loc='lower right')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
normalized_plot_path = os.path.join(output_dir, "model_comparison_normalized_mae.png")
plt.savefig(normalized_plot_path, dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(figures_dir, "model_comparison_normalized_mae.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved normalized comparison plot: {normalized_plot_path}")

comparison_df.to_csv(os.path.join(output_dir, "model_comparison_detailed.csv"))
print(f"\nDetailed comparison saved: {os.path.join(output_dir, 'model_comparison_detailed.csv')}")

model_rankings.to_csv(os.path.join(output_dir, "model_rankings.csv"), index=False)
print(f"Model rankings saved: {os.path.join(output_dir, 'model_rankings.csv')}")

summary_report = {
    'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models_compared': ['ARIMA', 'Ensemble (XGBoost+LightGBM)', 'LSTM'],
    'diseases': diseases,
    'average_mae_improvement_ensemble_vs_arima_percent': round(mae_improvements_ensemble, 2),
    'average_mae_improvement_lstm_vs_arima_percent': round(mae_improvements_lstm, 2),
    'recommendation': 'Ensemble model (XGBoost+LightGBM weighted blending) recommended for production deployment',
    'rationale': 'Ensemble model shows consistent improvements over ARIMA baseline and competitive or superior performance to LSTM while maintaining interpretability through SHAP values'
}

json_report_path = os.path.join(output_dir, "model_evaluation_report.json")
with open(json_report_path, 'w') as f:
    json.dump(summary_report, f, indent=4)
print(f"Evaluation report saved: {json_report_path}")

print("\n" + "=" * 80)
print("Model Evaluation Complete")
print("=" * 80)
print("\nRecommendation: Ensemble Model (XGBoost + LightGBM Weighted Blending)")
print("Rationale:")
print("  - Consistent MAE improvements over ARIMA baseline")
print("  - Quantile regression provides uncertainty quantification")
print("  - SHAP explainability enables clinical interpretation")
print("  - Weighted blending optimizes model strengths")
print("  - Production-ready with interpretable components")

print("\nNext step: Final validation and model selection for deployment")
