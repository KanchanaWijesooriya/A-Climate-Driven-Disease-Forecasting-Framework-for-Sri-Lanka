# Ensemble model artifacts (FastAPI + Streamlit ready)

These artifacts are produced by **step_03_ensemble_blending.py** and **step_04_shap_explainability.py**. Use them for inference in your FastAPI + Streamlit application.

## Directory layout (per disease)

For each disease (`leptospirosis`, `typhus`, `hepatitis_a`, `chickenpox`), the folder contains:

| File | Description |
|------|-------------|
| `scaler.pkl` | `joblib`-saved `StandardScaler` fitted on full training data. Use to normalize features before prediction. |
| `xgb_q05.pkl` | XGBoost model for 0.05 quantile (lower bound). |
| `xgb_q50.pkl` | XGBoost model for 0.5 quantile (median). |
| `xgb_q95.pkl` | XGBoost model for 0.95 quantile (upper bound). |
| `lgb_q05.pkl` | LightGBM model for 0.05 quantile. |
| `lgb_q50.pkl` | LightGBM model for 0.5 quantile. |
| `lgb_q95.pkl` | LightGBM model for 0.95 quantile. |
| `blending_weights.json` | `{"xgb_weight": float, "lgb_weight": float}`. Weights sum to 1. |
| `feature_names.json` | List of feature names in the same order as the training data. |
| `shap_explainer.pkl` | SHAP `TreeExplainer` for the median XGBoost model (from step_04). Use for feature importance / explanations. |

## Inference steps (pseudocode)

1. Load `scaler.pkl`, `xgb_q05.pkl`, `xgb_q50.pkl`, `xgb_q95.pkl`, `lgb_q05.pkl`, `lgb_q50.pkl`, `lgb_q95.pkl`, `blending_weights.json`, `feature_names.json` for the selected disease.
2. Build the feature vector for the requested district/week (same 40 features as in training, in the order of `feature_names.json`).
3. `X_scaled = scaler.transform(X)` (single row or batch).
4. `pred_lower = xgb_q05.predict(X_scaled) * w_xgb + lgb_q05.predict(X_scaled) * w_lgb` (and similarly for q50 and q95).
5. Clip predictions to `>= 0`.
6. Return `{ "lower": pred_lower, "median": pred_median, "upper": pred_upper }`.

Optional: load `shap_explainer.pkl` and call `explainer.shap_values(X_scaled)` for explanations.

## Loading with Python

```python
import joblib
import json

disease = "leptospirosis"
base = "model_data/artifacts/" + disease

scaler = joblib.load(f"{base}/scaler.pkl")
xgb_q05 = joblib.load(f"{base}/xgb_q05.pkl")
xgb_q50 = joblib.load(f"{base}/xgb_q50.pkl")
xgb_q95 = joblib.load(f"{base}/xgb_q95.pkl")
lgb_q05 = joblib.load(f"{base}/lgb_q05.pkl")
lgb_q50 = joblib.load(f"{base}/lgb_q50.pkl")
lgb_q95 = joblib.load(f"{base}/lgb_q95.pkl")
with open(f"{base}/blending_weights.json") as f:
    weights = json.load(f)
with open(f"{base}/feature_names.json") as f:
    feature_names = json.load(f)
# Optional:
# explainer = joblib.load(f"{base}/shap_explainer.pkl")
```

All `.pkl` files were saved with `joblib` (use `joblib.load`, not `pickle.load`).
