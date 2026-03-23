import pandas as pd
import numpy as np
import os
import warnings
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

print("FINAL MODEL ARTIFACT PREPARATION FOR DEPLOYMENT")
print("=" * 90)

model_data_dir = "/home/chanuka002/Research/model_data"
output_dir = "/home/chanuka002/Research/model_data"
artifacts_dir = os.path.join(output_dir, "artifacts")

if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)
    print(f"Created artifacts directory: {artifacts_dir}")

diseases = ['leptospirosis', 'typhus', 'hepatitis_a', 'chickenpox']
disease_names = ['Leptospirosis', 'Typhus', 'Hepatitis A', 'Chickenpox']

sequence_length = 4
production_models = {}

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
    
    print(f"  Training data shape: {X_train.shape}")
    print(f"  Test data shape: {X_test.shape}")
    print(f"  Features: {len(feature_cols)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    def create_sequences(X, y, seq_len):
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)
    
    split_idx = int(len(X_train_seq) * 0.8)
    X_train_lstm = X_train_seq[:split_idx]
    y_train_lstm = y_train_seq[:split_idx]
    X_val_lstm = X_train_seq[split_idx:]
    y_val_lstm = y_train_seq[split_idx:]
    
    print(f"  LSTM train shape: {X_train_lstm.shape}, val shape: {X_val_lstm.shape}, test shape: {X_test_seq.shape}")
    
    print(f"  Building and training final LSTM model...")
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
    
    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_lstm, y_val_lstm),
        callbacks=[EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)],
        verbose=0
    )
    
    print(f"  Training completed in {len(history.history['loss'])} epochs")
    
    predictions = model.predict(X_test_seq, verbose=0).flatten()
    predictions = np.maximum(predictions, 0)
    
    mae = mean_absolute_error(y_test_seq, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_seq, predictions))
    r2 = r2_score(y_test_seq, predictions)
    
    print(f"  Test Performance - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    disease_dir = os.path.join(artifacts_dir, disease_name.lower().replace(' ', '_'))
    if not os.path.exists(disease_dir):
        os.makedirs(disease_dir)
    
    model_path = os.path.join(disease_dir, f"{disease}_lstm_model.h5")
    model.save(model_path)
    print(f"  Saved model: {model_path}")
    
    scaler_path = os.path.join(disease_dir, f"{disease}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler: {scaler_path}")
    
    config = {
        'disease': disease_name,
        'sequence_length': sequence_length,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'model_type': 'LSTM',
        'performance': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'test_samples': len(y_test_seq)
        },
        'training': {
            'epochs_trained': len(history.history['loss']),
            'batch_size': 32,
            'train_samples': len(y_train_lstm),
            'validation_samples': len(y_val_lstm)
        },
        'architecture': {
            'lstm_layers': [
                {'units': 32, 'activation': 'relu', 'dropout': 0.35},
                {'units': 16, 'activation': 'relu', 'dropout': 0.35}
            ],
            'dense_layers': [
                {'units': 8, 'activation': 'relu'},
                {'units': 1, 'activation': 'linear'}
            ]
        },
        'preprocessing': {
            'scaler_type': 'StandardScaler',
            'scaling_applied': True
        }
    }
    
    config_path = os.path.join(disease_dir, f"{disease}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"  Saved config: {config_path}")
    
    production_models[disease_name] = {
        'model_path': model_path,
        'scaler_path': scaler_path,
        'config_path': config_path,
        'mae': mae,
        'r2': r2,
        'directory': disease_dir
    }
    
    keras.backend.clear_session()

print("\n" + "=" * 90)
print("PRODUCTION ARTIFACTS CREATED")
print("=" * 90)

summary_config = {
    'production_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_selection': 'LSTM',
    'rationale': 'LSTM selected as primary model after fine-tuning comparison with ensemble methods',
    'diseases': disease_names,
    'model_artifacts': production_models,
    'sequence_length': sequence_length,
    'prediction_type': 'Point predictions with real-world output',
    'deployment_ready': True,
    'usage': {
        'step_1': 'Load LSTM model from .h5 file',
        'step_2': 'Load scaler from .pkl file',
        'step_3': 'Preprocess input features using scaler',
        'step_4': 'Create sequences of length 4 from time-series features',
        'step_5': 'Generate predictions using model.predict()',
        'step_6': 'Apply post-processing (ensure non-negative values)'
    }
}

summary_path = os.path.join(output_dir, "production_deployment_config.json")
with open(summary_path, 'w') as f:
    json.dump(summary_config, f, indent=4)
print(f"\nProduction config saved: {summary_path}")

artifact_summary = pd.DataFrame(production_models).T[['mae', 'r2', 'directory']]
artifact_summary_path = os.path.join(output_dir, "production_artifacts_summary.csv")
artifact_summary.to_csv(artifact_summary_path)
print(f"Artifacts summary saved: {artifact_summary_path}")

print("\nFinal Model Directory Structure:")
for disease, paths in production_models.items():
    print(f"\n{disease}:")
    print(f"  - Model: {paths['model_path']}")
    print(f"  - Scaler: {paths['scaler_path']}")
    print(f"  - Config: {paths['config_path']}")
    print(f"  - Performance R2: {paths['r2']:.4f}")

print("\n" + "=" * 90)
print("READY FOR SOFTWARE APPLICATION DEVELOPMENT")
print("=" * 90)
print("\nArtifact Location: /home/chanuka002/Research/model_data/artifacts/")
print("All models are trained and ready for integration into Flask/Streamlit application")
print("\nNext: Begin software application development phase")
