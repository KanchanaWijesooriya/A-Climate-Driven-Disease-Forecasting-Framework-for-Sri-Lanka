"""Load LSTM models and run prediction."""
import os
from typing import Optional

import numpy as np

from .config import ARTIFACTS_DIR, FEATURE_COLS

# SHAP explainer cache (disease_id -> explainer)
_shap_explainers = {}


def _get_artifact_path(disease_id: str, filename: str) -> str:
    return os.path.join(ARTIFACTS_DIR, disease_id, filename)


_models = {}  # disease_id -> {model, scaler, config, sequence_length}
_ensemble_models = {}  # disease_id -> {scaler, xgb_q05, xgb_q50, xgb_q95, lgb_q05, lgb_q50, lgb_q95, weights}


def load_model(disease_id: str) -> bool:
    """Load LSTM model, scaler, and config for a disease. Returns True if loaded."""
    if disease_id in _models:
        return True
    model_path = _get_artifact_path(disease_id, f"{disease_id}_lstm_model.h5")
    scaler_path = _get_artifact_path(disease_id, f"{disease_id}_scaler.pkl")
    config_path = _get_artifact_path(disease_id, f"{disease_id}_config.json")
    if not all(os.path.exists(p) for p in (model_path, scaler_path, config_path)):
        return False
    import pickle
    import json
    from tensorflow.keras.models import load_model as tf_load_model

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(config_path) as f:
        config = json.load(f)
    # Must use tensorflow.keras to load .h5 saved from step_09 (TF backend)
    # compile=False avoids deserialization issues with metrics (inference only)
    model = tf_load_model(model_path, compile=False)
    seq_len = config.get("sequence_length", 4)
    _models[disease_id] = {
        "model": model,
        "scaler": scaler,
        "config": config,
        "sequence_length": seq_len,
    }
    return True


def predict(
    disease_id: str,
    feature_matrix: np.ndarray,
) -> Optional[tuple[np.ndarray, list[str]]]:
    """
    Predict case counts for each district.
    feature_matrix: (n_districts, sequence_length, n_features) float array.
    Returns (predictions array of shape (n_districts,), list of district names)
    or None if model not loaded.
    """
    if not load_model(disease_id):
        return None
    store = _models[disease_id]
    model = store["model"]
    scaler = store["scaler"]
    seq_len = store["sequence_length"]
    # feature_matrix: (n_districts, 4, 40)
    n_districts = feature_matrix.shape[0]
    X_scaled = np.zeros_like(feature_matrix)
    for i in range(n_districts):
        X_scaled[i] = scaler.transform(feature_matrix[i])
    pred = model.predict(X_scaled, verbose=0)
    pred = pred.flatten()
    pred = np.maximum(pred, 0.0)
    return pred, []  # districts list filled by caller


def predict_with_districts(
    disease_id: str,
    feature_matrix: np.ndarray,
    districts: list[str],
) -> Optional[list[dict]]:
    """
    feature_matrix: (n_districts, 4, 40), districts: list of length n_districts.
    Returns list of {district, predicted_cases}.
    """
    out = predict(disease_id, feature_matrix)
    if out is None:
        return None
    pred, _ = out
    return [
        {"district": d, "predicted_cases": float(round(pred[i], 2))}
        for i, d in enumerate(districts)
    ]


# ---------- Ensemble (XGB + LGB) inference: lower, median, upper ----------

def load_ensemble(disease_id: str) -> bool:
    """Load ensemble artifacts (joblib scaler + xgb/lgb q05/q50/q95, blending_weights). Returns True if loaded."""
    if disease_id in _ensemble_models:
        return True
    import json
    try:
        import joblib
    except ImportError:
        import pickle as joblib
    base = _get_artifact_path(disease_id, "")
    scaler_path = os.path.join(base, "scaler.pkl")
    paths = {
        "scaler": scaler_path,
        "xgb_q05": os.path.join(base, "xgb_q05.pkl"),
        "xgb_q50": os.path.join(base, "xgb_q50.pkl"),
        "xgb_q95": os.path.join(base, "xgb_q95.pkl"),
        "lgb_q05": os.path.join(base, "lgb_q05.pkl"),
        "lgb_q50": os.path.join(base, "lgb_q50.pkl"),
        "lgb_q95": os.path.join(base, "lgb_q95.pkl"),
        "weights": os.path.join(base, "blending_weights.json"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return False
    scaler = joblib.load(paths["scaler"])
    xgb_q05 = joblib.load(paths["xgb_q05"])
    xgb_q50 = joblib.load(paths["xgb_q50"])
    xgb_q95 = joblib.load(paths["xgb_q95"])
    lgb_q05 = joblib.load(paths["lgb_q05"])
    lgb_q50 = joblib.load(paths["lgb_q50"])
    lgb_q95 = joblib.load(paths["lgb_q95"])
    with open(paths["weights"]) as f:
        weights = json.load(f)
    w_xgb = weights.get("xgb_weight", weights.get("xgb", 0.5))
    w_lgb = weights.get("lgb_weight", weights.get("lgb", 0.5))

    seasonal_baseline = None
    seasonal_path = os.path.join(base, "seasonal_baseline.json")
    if os.path.isfile(seasonal_path):
        with open(seasonal_path) as f:
            seasonal_baseline = json.load(f)

    district_bias = None
    bias_path = os.path.join(base, "district_bias_correction.json")
    if os.path.isfile(bias_path):
        with open(bias_path) as f:
            district_bias = json.load(f)

    _ensemble_models[disease_id] = {
        "scaler": scaler,
        "xgb_q05": xgb_q05, "xgb_q50": xgb_q50, "xgb_q95": xgb_q95,
        "lgb_q05": lgb_q05, "lgb_q50": lgb_q50, "lgb_q95": lgb_q95,
        "w_xgb": w_xgb, "w_lgb": w_lgb,
        "seasonal_baseline": seasonal_baseline,
        "district_bias": district_bias,
    }
    return True


def predict_ensemble(
    disease_id: str,
    feature_matrix: np.ndarray,
    districts: list[str],
    week_numbers: Optional[list[int]] = None,
    week_ids: Optional[list[int]] = None,
) -> Optional[list[dict]]:
    """
    feature_matrix: (n_districts, n_features). Returns list of {district, lower, median, upper}.
    week_numbers, week_ids: optional, for seasonal denormalization.
    """
    if not load_ensemble(disease_id):
        return None
    store = _ensemble_models[disease_id]
    scaler = store["scaler"]
    X = scaler.transform(feature_matrix)
    w_xgb, w_lgb = store["w_xgb"], store["w_lgb"]
    pred_lower = np.maximum(
        w_xgb * store["xgb_q05"].predict(X) + w_lgb * store["lgb_q05"].predict(X), 0
    )
    pred_median = np.maximum(
        w_xgb * store["xgb_q50"].predict(X) + w_lgb * store["lgb_q50"].predict(X), 0
    )
    pred_upper = np.maximum(
        w_xgb * store["xgb_q95"].predict(X) + w_lgb * store["lgb_q95"].predict(X), 0
    )

    # Seasonal denormalization (when seasonal_baseline.json exists)
    seasonal = store.get("seasonal_baseline")
    if seasonal is not None and week_numbers is not None and week_ids is not None:
        dw_means = {(r["district"], r["week_number"]): r["mean"] for r in seasonal.get("district_week_means", [])}
        week_means = {r["week_number"]: r["mean"] for r in seasonal.get("week_means", [])}
        fallback = seasonal.get("fallback_mean", seasonal.get("global_mean", 0.0))
        slope = seasonal.get("trend_slope", 0.0)
        intercept = seasonal.get("trend_intercept", 0.0)
        add_back = np.zeros(len(districts), dtype=np.float64)
        for i, (d, wn, wid) in enumerate(zip(districts, week_numbers, week_ids)):
            s_mean = dw_means.get((d, int(wn)), week_means.get(int(wn), fallback))
            trend = slope * wid + intercept
            add_back[i] = float(s_mean) + trend
        pred_lower = np.maximum(pred_lower + add_back, 0)
        pred_median = np.maximum(pred_median + add_back, 0)
        pred_upper = np.maximum(pred_upper + add_back, 0)

    # District bias correction
    district_bias = store.get("district_bias")
    if district_bias is not None and districts:
        factors = np.array([float(district_bias.get(d, 1.0)) for d in districts])
        pred_median = np.maximum(pred_median * factors, 0)

    return [
        {
            "district": d,
            "lower": round(float(pred_lower[i]), 2),
            "median": round(float(pred_median[i]), 2),
            "upper": round(float(pred_upper[i]), 2),
        }
        for i, d in enumerate(districts)
    ]


def load_shap_explainer(disease_id: str) -> bool:
    """Load SHAP TreeExplainer from artifacts (shap_explainer.pkl). Returns True if loaded."""
    if disease_id in _shap_explainers:
        return True
    try:
        import joblib
    except ImportError:
        import pickle as joblib
    path = _get_artifact_path(disease_id, "shap_explainer.pkl")
    if not os.path.isfile(path):
        return False
    try:
        explainer = joblib.load(path)
        _shap_explainers[disease_id] = explainer
        return True
    except Exception:
        return False


def explain_ensemble_row(
    disease_id: str,
    feature_row: np.ndarray,
    feature_names: list[str],
) -> Optional[dict]:
    """
    Get SHAP values for one row (1D array of length 40). Row should be unscaled; we scale with ensemble scaler.
    Returns { "expected_value": float, "shap_values": [ {"feature": str, "value": float}, ... ] } or None.
    """
    if not load_ensemble(disease_id) or not load_shap_explainer(disease_id):
        return None
    store = _ensemble_models[disease_id]
    scaler = store["scaler"]
    row = feature_row.reshape(1, -1)
    X_scaled = scaler.transform(row)
    explainer = _shap_explainers[disease_id]
    try:
        shap_vals = explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.array(shap_vals).flatten()
    except Exception:
        return None
    expected = float(getattr(explainer, "expected_value", 0.0))
    if hasattr(expected, "__len__"):
        expected = float(expected[0]) if len(expected) > 0 else 0.0
    out = [
        {"feature": feature_names[i] if i < len(feature_names) else f"f{i}", "value": round(float(shap_vals[i]), 6)}
        for i in range(len(shap_vals))
    ]
    return {"expected_value": expected, "shap_values": out}
