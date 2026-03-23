import pandas as pd
import numpy as np
import os
import sys
import warnings
import json
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Add project root for step_03 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step 4.4: SHAP Explainability Analysis - Feature Contribution Interpretation")
print("=" * 80)

model_data_dir = "/home/chanuka002/Research/model_data"
output_dir = "/home/chanuka002/Research/model_data"
figures_dir = "/home/chanuka002/Research/figures"
os.makedirs(figures_dir, exist_ok=True)
artifacts_base = os.path.join(model_data_dir, "artifacts")


def build_features_lepto(full_df, train_df_for_stats):
    """Matches step_03_ensemble_blending.build_features_lepto."""
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


diseases = ['leptospirosis', 'typhus', 'hepatitis_a', 'chickenpox']
disease_names = ['Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']

shap_results = {}

for disease, disease_name in zip(diseases, disease_names):
    print(f"\n{'='*80}")
    print(f"Processing {disease_name} - SHAP Analysis")
    print(f"{'='*80}")
    
    train_path = os.path.join(model_data_dir, f"{disease}_train.csv")
    test_path = os.path.join(model_data_dir, f"{disease}_test.csv")
    val_path = os.path.join(model_data_dir, f"{disease}_val.csv")
    feature_names_path = os.path.join(artifacts_base, disease, "feature_names.json")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path) if os.path.isfile(val_path) else None
    
    # Use feature_names.json from artifacts (matches ensemble model)
    if os.path.isfile(feature_names_path):
        with open(feature_names_path) as f:
            feature_cols = json.load(f)
        print(f"Using feature_names.json from artifacts ({len(feature_cols)} features)")
        # Apply build_features_lepto to match step_03
        full_raw = pd.concat([train_df, test_df] + ([val_df] if val_df is not None else []), ignore_index=True).sort_values('week_id')
        full_fe = build_features_lepto(full_raw, train_df)
        max_tr_wk = train_df['week_id'].max()
        min_te_wk = test_df['week_id'].min()
        train_df = full_fe[full_fe['week_id'] <= max_tr_wk].dropna(subset=['target']).copy()
        test_df = full_fe[full_fe['week_id'] >= min_te_wk].dropna(subset=['target']).copy()
        for df in (train_df, test_df):
            for c in feature_cols:
                if c not in df.columns:
                    df[c] = 0
                elif df[c].dtype == bool:
                    df[c] = df[c].astype(np.float64)
            df[feature_cols] = df[feature_cols].fillna(0)
    else:
        feature_cols = [col for col in train_df.columns if col not in ['district', 'week_id', 'start_date', 'end_date', 'Duration', 'target']]
        print(f"Using raw CSV columns ({len(feature_cols)} features)")
    
    X_train = np.ascontiguousarray(train_df[feature_cols].values.astype(np.float64))
    y_train = train_df['target'].values.astype(np.float64)
    X_test = np.ascontiguousarray(test_df[feature_cols].values.astype(np.float64))
    y_test = test_df['target'].values.astype(np.float64)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    disease_artifacts_dir = os.path.join(artifacts_base, disease)
    xgb_median_path = os.path.join(disease_artifacts_dir, "xgb_q50.pkl")
    scaler_path = os.path.join(disease_artifacts_dir, "scaler.pkl")
    
    if os.path.isfile(xgb_median_path) and os.path.isfile(scaler_path):
        print("Loading median XGBoost and scaler from artifacts (from step 4.3)...")
        scaler = joblib.load(scaler_path)
        xgb_model = joblib.load(xgb_median_path)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Loaded XGBoost model and scaler from pkl")
    else:
        print("Artifacts not found; fitting scaler and XGBoost for SHAP...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        xgb_model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.5,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train_scaled, y_train, verbose=False)
        print("XGBoost model fitted")
    
    xgb_predictions = xgb_model.predict(X_test_scaled)
    
    print("Creating SHAP explainer for XGBoost...")
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb.shap_values(X_test_scaled)
    
    print("Extracting SHAP values...")
    if isinstance(shap_values_xgb, list):
        shap_values_xgb = shap_values_xgb[0]
    
    mean_abs_shap = np.abs(shap_values_xgb).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap_value': mean_abs_shap
    }).sort_values('mean_abs_shap_value', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    features_path = os.path.join(output_dir, f"{disease}_feature_importance.csv")
    feature_importance.to_csv(features_path, index=False)
    print(f"Saved feature importance: {features_path}")
    
    print("Creating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_xgb, X_test_scaled, feature_names=feature_cols, show=False)
    summary_plot_path = os.path.join(output_dir, f"{disease}_shap_summary_plot.png")
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, f"{disease}_shap_summary_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot: {summary_plot_path}")
    
    print("Creating SHAP bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_xgb, X_test_scaled, feature_names=feature_cols, plot_type='bar', show=False)
    bar_plot_path = os.path.join(output_dir, f"{disease}_shap_bar_plot.png")
    plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, f"{disease}_shap_bar_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar plot: {bar_plot_path}")
    
    top_features = feature_importance.head(min(5, len(feature_importance)))['feature'].tolist()
    
    print("Creating SHAP dependence plots for top features...")
    for i, feature in enumerate(top_features):
        feature_idx = feature_cols.index(feature)
        
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            feature_idx,
            shap_values_xgb,
            X_test_scaled,
            feature_names=feature_cols,
            show=False
        )
        dependence_path = os.path.join(output_dir, f"{disease}_shap_dependence_{i+1}_{feature}.png")
        plt.savefig(dependence_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {dependence_path}")
    
    sample_instance = X_test_scaled[0:1]
    sample_shap = shap_values_xgb[0:1]
    
    plt.figure(figsize=(10, 4))
    shap.force_plot(
        explainer_xgb.expected_value,
        sample_shap,
        sample_instance,
        feature_names=feature_cols,
        matplotlib=True,
        show=False
    )
    force_plot_path = os.path.join(output_dir, f"{disease}_shap_force_plot.png")
    plt.savefig(force_plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, f"{disease}_shap_force_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved force plot (instance 0): {force_plot_path}")
    
    # Save SHAP explainer to pkl for use in FastAPI/Streamlit app
    os.makedirs(disease_artifacts_dir, exist_ok=True)
    explainer_path = os.path.join(disease_artifacts_dir, "shap_explainer.pkl")
    joblib.dump(explainer_xgb, explainer_path)
    print(f"Saved SHAP explainer: {explainer_path}")
    
    top_vals = {f'Top_Feature_{j+1}': feature_importance.iloc[j]['feature'] for j in range(min(5, len(feature_importance)))}
    top_imp = {f'Top_Feature_{j+1}_Importance': float(feature_importance.iloc[j]['mean_abs_shap_value']) for j in range(min(5, len(feature_importance)))}
    shap_results[disease_name] = {
        **top_vals,
        **top_imp,
        'Total_Features_Analyzed': len(feature_cols),
        'Test_Samples': len(y_test)
    }

print("\n" + "=" * 80)
print("SHAP Explainability Analysis Complete")

shap_summary = pd.DataFrame(shap_results).T
print("\nSHAP Analysis Summary:")
print(shap_summary)

summary_path = os.path.join(output_dir, "shap_analysis_summary.csv")
shap_summary.to_csv(summary_path)
print(f"\nSummary saved: {summary_path}")

json_path = os.path.join(output_dir, "shap_analysis.json")
with open(json_path, 'w') as f:
    json.dump(shap_results, f, indent=4)
print(f"JSON results saved: {json_path}")

print("\nGenerated outputs:")
print("- Feature importance rankings (CSV)")
print("- SHAP summary plots (dot plot)")
print("- SHAP bar plots (aggregated importance)")
print("- SHAP dependence plots (top 5 features)")
print("- SHAP force plots (instance-level explanations)")

print("\nNext step: LSTM validation model for comparison")
