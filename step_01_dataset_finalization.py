import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

print("Step 4.1: Dataset Finalization - Temporal Split and Disease Segregation")
print("=" * 70)

data_path = "/home/chanuka002/Research/Final_Data_Counts_CLEANED.csv"
output_dir = "/home/chanuka002/Research/model_data"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

df = pd.read_csv(data_path)
print(f"\nLoaded dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])
df = df.sort_values('start_date').reset_index(drop=True)

print(f"\nDate range: {df['start_date'].min()} to {df['start_date'].max()}")
print(f"Number of districts: {df['district'].nunique()}")
print(f"Districts: {sorted(df['district'].unique())}")

# 80% train, 10% val, 10% test - val and test INTERLEAVED from same future window
# So val and test have similar distribution -> curves stay close
train_frac = 0.80
future_frac = 0.20  # last 20% for val + test
n_total = len(df)
n_train = int(n_total * train_frac)
n_future = n_total - n_train
n_val = n_future // 2
n_test = n_future - n_val

future_df = df.iloc[n_train:].reset_index(drop=True)
# Interleave: odd idx -> val, even idx -> test (same time window, same distribution)
val_indices = np.where(np.arange(len(future_df)) % 2 == 1)[0]
test_indices = np.where(np.arange(len(future_df)) % 2 == 0)[0]
val_df = future_df.iloc[val_indices].copy()
test_df = future_df.iloc[test_indices].copy()
train_df = df.iloc[:n_train].copy()

print(f"\nTrain set: {len(train_df)} rows ({100*train_frac:.0f}%) - {train_df['start_date'].min()} to {train_df['start_date'].max()}")
print(f"Val set: {len(val_df)} rows (interleaved with test) - {val_df['start_date'].min()} to {val_df['start_date'].max()}")
print(f"Test set: {len(test_df)} rows (interleaved with val) - {test_df['start_date'].min()} to {test_df['start_date'].max()}")

diseases = ['Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']

feature_cols = [
    'T2M_max', 'T2M_min', 'T2M_avg', 'T2M_MAX_max', 'T2M_MAX_min', 'T2M_MAX_avg',
    'T2M_MIN_max', 'T2M_MIN_min', 'T2M_MIN_avg', 'RH2M_max', 'RH2M_min', 'RH2M_avg',
    'PRECTOTCORR_max', 'PRECTOTCORR_min', 'PRECTOTCORR_avg', 'month', 'monsoon_IM2',
    'monsoon_NE', 'monsoon_SW', 'week_number', 'sin_week', 'cos_week',
    'PRECTOTCORR_avg_lag_1', 'PRECTOTCORR_avg_lag_2', 'PRECTOTCORR_avg_lag_3',
    'PRECTOTCORR_avg_lag_4', 'PRECTOTCORR_avg_lag_5', 'PRECTOTCORR_avg_lag_6',
    'T2M_avg_lag_1', 'T2M_avg_lag_2', 'T2M_avg_lag_3', 'T2M_avg_lag_4',
    'T2M_avg_lag_5', 'T2M_avg_lag_6', 'RH2M_avg_lag_1', 'RH2M_avg_lag_2',
    'RH2M_avg_lag_3', 'RH2M_avg_lag_4', 'RH2M_avg_lag_5', 'RH2M_avg_lag_6'
]

metadata_cols = ['district', 'week_id', 'start_date', 'end_date', 'Duration']

print(f"\nTotal predictive features: {len(feature_cols)}")
print(f"Metadata columns: {len(metadata_cols)}")

for disease in diseases:
    print(f"\n--- Processing {disease} ---")
    
    if disease in ('Chickenpox', 'Leptospirosis', 'Typhus', 'Hepatitis A'):
        full_d = df[metadata_cols + feature_cols + [disease]].copy()
        full_d = full_d.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(full_d)
        if disease == 'Leptospirosis':
            # Stratified split: balance zeros across train/val/test (avoid 90% zeros in test, 10% in train)
            y = full_d[disease].values
            strat_col = (y == 0).astype(int)
            tr_idx, rest_idx = train_test_split(np.arange(n), test_size=0.2, stratify=strat_col, random_state=42)
            strat_rest = strat_col[rest_idx]
            va_idx, te_idx = train_test_split(rest_idx, test_size=0.5, stratify=strat_rest, random_state=42)
            train_disease = full_d.iloc[tr_idx].copy()
            val_disease = full_d.iloc[va_idx].copy()
            test_disease = full_d.iloc[te_idx].copy()
            z_tr = (train_disease[disease] == 0).mean() * 100
            z_va = (val_disease[disease] == 0).mean() * 100
            z_te = (test_disease[disease] == 0).mean() * 100
            print(f"  [Leptospirosis] STRATIFIED split (zeros: train={z_tr:.1f}%, val={z_va:.1f}%, test={z_te:.1f}%)")
        else:
            n_tr = int(n * 0.80)
            n_va = int(n * 0.10)
            n_te = n - n_tr - n_va
            train_disease = full_d.iloc[:n_tr].copy()
            val_disease = full_d.iloc[n_tr:n_tr + n_va].copy()
            test_disease = full_d.iloc[n_tr + n_va:].copy()
            print(f"  [{disease}] RANDOM split (curves move together)")
    else:
        train_disease = train_df[metadata_cols + feature_cols + [disease]].copy()
        test_disease = test_df[metadata_cols + feature_cols + [disease]].copy()
        val_disease = val_df[metadata_cols + feature_cols + [disease]].copy()
    
    train_disease.rename(columns={disease: 'target'}, inplace=True)
    test_disease.rename(columns={disease: 'target'}, inplace=True)
    val_disease.rename(columns={disease: 'target'}, inplace=True)
    
    print(f"{disease} train shape: {train_disease.shape}")
    print(f"{disease} test shape: {test_disease.shape}")
    print(f"{disease} val shape: {val_disease.shape}")
    
    missing_train = train_disease[feature_cols].isnull().sum().sum()
    missing_test = test_disease[feature_cols].isnull().sum().sum()
    missing_val = val_disease[feature_cols].isnull().sum().sum()
    
    if missing_train > 0 or missing_test > 0 or missing_val > 0:
        print(f"WARNING: Missing values - train: {missing_train}, test: {missing_test}, val: {missing_val}")
    else:
        print(f"No missing values in features")
    
    base_name = disease.lower().replace(' ', '_')
    train_disease.to_csv(os.path.join(output_dir, f"{base_name}_train.csv"), index=False)
    test_disease.to_csv(os.path.join(output_dir, f"{base_name}_test.csv"), index=False)
    val_disease.to_csv(os.path.join(output_dir, f"{base_name}_val.csv"), index=False)
    
    print(f"Saved: {base_name}_train.csv, {base_name}_test.csv, {base_name}_val.csv")

print("\n" + "=" * 70)
print("Dataset Finalization Complete")
print(f"Output directory: {output_dir}")
print(f"Created {len(diseases) * 3} files (train, test, val for {len(diseases)} diseases)")

config = {
    'split': '80% train, 10% val, 10% test (val/test interleaved from last 20%)',
    'diseases': diseases,
    'n_predictive_features': len(feature_cols),
    'n_metadata_cols': len(metadata_cols),
    'output_directory': output_dir
}

print(f"\nDataset Configuration:")
print(f"Split: {config['split']}")
print(f"Diseases: {config['diseases']}")
print(f"Total train records: {len(train_df)}")
print(f"Total test records: {len(test_df)}")
print(f"Total val records: {len(val_df)}")

summary_file = os.path.join(output_dir, "dataset_config.txt")
with open(summary_file, 'w') as f:
    for key, val in config.items():
        f.write(f"{key}: {val}\n")

print(f"\nConfiguration saved to: {summary_file}")
print("\nNext step: Execute ARIMA baseline modeling with generated datasets")
