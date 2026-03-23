#!/usr/bin/env python3
"""
Generate report figures for Chapter 5 and 6.
Saves to Docs/report_assets/figures/ with proper figure numbers.
Uses model_data metrics (arima, ensemble, lstm) - fills LSTM where missing.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_data_dir = "/home/chanuka002/Research/model_data"
docs_figures = "/home/chanuka002/Research/Docs/report_assets/figures"
os.makedirs(docs_figures, exist_ok=True)

DISEASES = ['Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']

def load_metrics():
    arima = pd.read_csv(os.path.join(model_data_dir, "arima_baseline_metrics.csv"), index_col=0)
    ensemble = pd.read_csv(os.path.join(model_data_dir, "ensemble_metrics.csv"), index_col=0)
    lstm = pd.read_csv(os.path.join(model_data_dir, "lstm_metrics.csv"), index_col=0)
    return arima, ensemble, lstm

def build_comparison_df(arima, ensemble, lstm):
    """Build comparison with LSTM filled for missing diseases (from model_comparison_detailed if available)."""
    old = os.path.join(model_data_dir, "model_comparison_detailed.csv")
    fallback = pd.read_csv(old, index_col=0) if os.path.exists(old) else None

    data = []
    for d in DISEASES:
        row = {
            'ARIMA_MAE': arima.loc[d, 'MAE'],
            'ARIMA_RMSE': arima.loc[d, 'RMSE'],
            'ARIMA_R2': arima.loc[d, 'R2'],
            'Ensemble_MAE': ensemble.loc[d, 'MAE'],
            'Ensemble_RMSE': ensemble.loc[d, 'RMSE'],
            'Ensemble_R2': ensemble.loc[d, 'R2'],
        }
        if d in lstm.index:
            row['LSTM_MAE'] = lstm.loc[d, 'MAE']
            row['LSTM_RMSE'] = lstm.loc[d, 'RMSE']
            row['LSTM_R2'] = lstm.loc[d, 'R2']
        elif fallback is not None and d in fallback.index:
            row['LSTM_MAE'] = fallback.loc[d, 'LSTM_MAE']
            row['LSTM_RMSE'] = fallback.loc[d, 'LSTM_RMSE']
            row['LSTM_R2'] = fallback.loc[d, 'LSTM_R2']
        else:
            row['LSTM_MAE'] = np.nan
            row['LSTM_RMSE'] = np.nan
            row['LSTM_R2'] = np.nan
        data.append(row)

    return pd.DataFrame(data, index=DISEASES)

def fig_5_1_train_test_split():
    """Train vs test sample count per disease."""
    train, test = 6400, 875
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DISEASES))
    w = 0.35
    ax.bar(x - w/2, [train] * 4, w, label='Train (≤31 Dec 2024)', color='#1f77b4')
    ax.bar(x + w/2, [test] * 4, w, label='Test (≥1 Jan 2025)', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(DISEASES, rotation=15)
    ax.set_ylabel('Sample count')
    ax.set_title('Train vs. Test Sample Count per Disease')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(docs_figures, "fig_5_1_train_test_split.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Generated: fig_5_1_train_test_split.png")

def fig_5_5_training_curve():
    """Copy one representative ensemble training curve to Docs."""
    src = os.path.join(model_data_dir, "leptospirosis_ensemble_training_curves.png")
    if not os.path.exists(src):
        src = os.path.join(model_data_dir, "hepatitis_a_ensemble_training_curves.png")
    if os.path.exists(src):
        dst = os.path.join(docs_figures, "fig_5_5_ensemble_training_curves.png")
        import shutil
        shutil.copy2(src, dst)
        print(f"  Copied: fig_5_5_ensemble_training_curves.png")
        return True
    return False

def fig_5_6_model_comparison(comparison_df):
    """Generate model comparison bar chart (ARIMA vs Ensemble vs LSTM)."""
    mae = comparison_df[['ARIMA_MAE', 'Ensemble_MAE', 'LSTM_MAE']].rename(
        columns={'ARIMA_MAE': 'ARIMA', 'Ensemble_MAE': 'Ensemble', 'LSTM_MAE': 'LSTM'})
    rmse = comparison_df[['ARIMA_RMSE', 'Ensemble_RMSE', 'LSTM_RMSE']].rename(
        columns={'ARIMA_RMSE': 'ARIMA', 'Ensemble_RMSE': 'Ensemble', 'LSTM_RMSE': 'LSTM'})
    r2 = comparison_df[['ARIMA_R2', 'Ensemble_R2', 'LSTM_R2']].rename(
        columns={'ARIMA_R2': 'ARIMA', 'Ensemble_R2': 'Ensemble', 'LSTM_R2': 'LSTM'})

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    mae.plot(kind='bar', ax=axes[0], color=colors)
    axes[0].set_title('MAE (Lower is Better)', fontsize=11)
    axes[0].set_ylabel('MAE')
    axes[0].set_xlabel('Disease')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(axis='y', alpha=0.3)

    rmse.plot(kind='bar', ax=axes[1], color=colors)
    axes[1].set_title('RMSE (Lower is Better)', fontsize=11)
    axes[1].set_ylabel('RMSE')
    axes[1].set_xlabel('Disease')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(axis='y', alpha=0.3)

    r2.plot(kind='bar', ax=axes[2], color=colors)
    axes[2].set_title('R² (Higher is Better)', fontsize=11)
    axes[2].set_ylabel('R²')
    axes[2].set_xlabel('Disease')
    axes[2].legend(loc='lower right', fontsize=8)
    axes[2].tick_params(axis='x', rotation=15)
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('Model Comparison: ARIMA vs. Ensemble vs. LSTM', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(docs_figures, "fig_5_6_model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Generated: fig_5_6_model_comparison.png")

def fig_6_1_training_curve():
    """Same as 5.5 - reuse or copy."""
    fig_5_5_training_curve()
    src = os.path.join(docs_figures, "fig_5_5_ensemble_training_curves.png")
    dst = os.path.join(docs_figures, "fig_6_1_ensemble_training_curves.png")
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        print(f"  Copied: fig_6_1_ensemble_training_curves.png (for Chapter 6)")

def fig_6_2_model_comparison(comparison_df):
    """Single MAE comparison for Chapter 6 discussion."""
    mae = comparison_df[['ARIMA_MAE', 'Ensemble_MAE', 'LSTM_MAE']].rename(
        columns={'ARIMA_MAE': 'ARIMA', 'Ensemble_MAE': 'Ensemble', 'LSTM_MAE': 'LSTM'})
    fig, ax = plt.subplots(figsize=(8, 5))
    mae.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_title('Model Comparison: MAE per Disease (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE')
    ax.set_xlabel('Disease')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(docs_figures, "fig_6_2_model_comparison_mae.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Generated: fig_6_2_model_comparison_mae.png")

def fig_5_12_shap_example():
    """Copy one SHAP bar plot as reference (app screenshot differs)."""
    for d in ['leptospirosis', 'hepatitis_a', 'typhus', 'chickenpox']:
        src = os.path.join(model_data_dir, f"{d}_shap_bar_plot.png")
        if os.path.exists(src):
            dst = os.path.join(docs_figures, "fig_5_12_shap_bar_example.png")
            import shutil
            shutil.copy2(src, dst)
            print(f"  Copied: fig_5_12_shap_bar_example.png (from {d})")
            return True
    return False

def main():
    print("Generating report figures...")
    arima, ensemble, lstm = load_metrics()
    comparison_df = build_comparison_df(arima, ensemble, lstm)

    fig_5_1_train_test_split()
    fig_5_5_training_curve()
    fig_5_6_model_comparison(comparison_df)
    fig_5_12_shap_example()
    fig_6_1_training_curve()
    fig_6_2_model_comparison(comparison_df)

    print(f"\nFigures saved to: {docs_figures}")

if __name__ == "__main__":
    main()
