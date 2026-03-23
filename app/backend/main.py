"""FastAPI backend for disease case count and prediction."""
import csv
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import DISEASES, MODEL_DATA_DIR
from .data_service import (
    get_districts,
    get_past_cases,
    get_last_n_weeks_features_per_district,
    get_last_week_features_per_district,
    get_ensemble_payload_with_build_features,
    get_ensemble_payload_from_last_week,
)
from .inference import load_model, predict_with_districts, predict_ensemble, explain_ensemble_row
from .weather_service import (
    load_or_fetch_weather_for_week,
    get_epi_week_start_end,
    get_lstm_4weeks_payload,
)

app = FastAPI(
    title="Disease Case Forecast API",
    description="Climate-driven disease case counts and predictions for Sri Lanka",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/diseases")
def list_diseases():
    return {"diseases": DISEASES}


@app.get("/districts")
def list_districts(disease_id: str = Query(..., description="Disease id e.g. leptospirosis")):
    districts = get_districts(disease_id)
    return {"disease_id": disease_id, "districts": districts}


@app.get("/past_cases")
def past_cases(
    disease_id: str = Query(...),
    districts: Optional[str] = Query(None, description="Comma-separated district names"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    district_list = districts.split(",") if districts else None
    if district_list:
        district_list = [d.strip() for d in district_list if d.strip()]
    data = get_past_cases(disease_id, district_list, start_date, end_date)
    return {"disease_id": disease_id, "count": len(data), "data": data}


@app.get("/feature_importance")
def feature_importance(
    disease_id: str = Query(..., description="Disease id e.g. leptospirosis"),
):
    """Return precomputed SHAP feature importance (mean |SHAP|) for the selected disease."""
    path = os.path.join(MODEL_DATA_DIR, f"{disease_id}_feature_importance.csv")
    if not os.path.isfile(path):
        return {"disease_id": disease_id, "error": "Feature importance file not found", "features": []}
    features = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("feature", "").strip()
            try:
                val = float(row.get("mean_abs_shap_value", 0))
            except (ValueError, TypeError):
                val = 0.0
            features.append({"feature": name, "mean_abs_shap_value": round(val, 6)})
    return {"disease_id": disease_id, "features": features}


@app.get("/explain")
def explain_prediction(
    disease_id: str = Query(...),
    prediction_date: str = Query(..., description="Prediction week date (YYYY-MM-DD)"),
    district: Optional[str] = Query(None, description="District to explain; omit = first available"),
):
    """Return live SHAP values for one district's prediction (why this prediction). Uses ensemble build_features."""
    try:
        target_dt = datetime.strptime(prediction_date.strip(), "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid prediction_date; use YYYY-MM-DD", "disease_id": disease_id}
    df_weather, err = load_or_fetch_weather_for_week(target_dt, disease_id)
    if err or df_weather is None:
        payload = get_ensemble_payload_from_last_week(disease_id)
    else:
        payload = get_ensemble_payload_with_build_features(disease_id, df_weather)
        if payload is None:
            payload = get_ensemble_payload_from_last_week(disease_id)
    if payload is None:
        return {"error": err or "Could not build features for this week", "disease_id": disease_id}
    districts_list = payload["districts"]
    feature_matrix = payload["feature_matrix"]
    feature_names = payload.get("feature_names", [])
    idx = 0
    if district and district.strip() in districts_list:
        idx = districts_list.index(district.strip())
    row = feature_matrix[idx]
    result = explain_ensemble_row(disease_id, row, feature_names)
    if result is None:
        return {"error": "SHAP explainer not available (run step_04 and ensure shap_explainer.pkl exists)", "disease_id": disease_id}
    result["disease_id"] = disease_id
    result["prediction_date"] = prediction_date
    result["district"] = districts_list[idx]
    return result


@app.get("/predict")
def predict_next_week(
    disease_id: str = Query(...),
    districts: Optional[str] = Query(None, description="Comma-separated; omit = all districts"),
    model: str = Query("lstm", description="Model: lstm or ensemble"),
    prediction_date: Optional[str] = Query(None, description="Optional date (YYYY-MM-DD) for prediction week; uses weather API and cache"),
):
    """Predict case counts. If prediction_date given: load/fetch weather for that week and predict. Else: use last week in data."""
    if model.strip().lower() == "ensemble":
        weather_fallback = False
        if prediction_date:
            try:
                target_dt = datetime.strptime(prediction_date.strip(), "%Y-%m-%d")
            except ValueError:
                return {"error": "Invalid prediction_date; use YYYY-MM-DD", "disease_id": disease_id, "model": "ensemble"}
            df_weather, err = load_or_fetch_weather_for_week(target_dt, disease_id)
            if err or df_weather is None:
                payload = get_ensemble_payload_from_last_week(disease_id)
                if payload is None:
                    return {"error": err or "No historical data", "disease_id": disease_id, "model": "ensemble"}
                weather_fallback = True
            else:
                payload = get_ensemble_payload_with_build_features(disease_id, df_weather)
                if payload is None:
                    payload = get_ensemble_payload_from_last_week(disease_id)
                    weather_fallback = payload is not None
                if payload is None:
                    return {"error": "Could not build ensemble features", "disease_id": disease_id, "model": "ensemble"}
        else:
            payload = get_ensemble_payload_from_last_week(disease_id)
            if payload is None:
                return {"error": "Insufficient data for prediction", "disease_id": disease_id, "model": "ensemble"}
        all_districts = payload["districts"]
        feature_matrix = payload["feature_matrix"]
        prediction_week_start = payload["last_week_start"]
        last_dt = datetime.strptime(prediction_week_start, "%Y-%m-%d")
        prediction_week_end = (last_dt + timedelta(days=6)).strftime("%Y-%m-%d")

        predictions = predict_ensemble(
            disease_id,
            feature_matrix,
            all_districts,
            week_numbers=payload.get("week_numbers"),
            week_ids=payload.get("week_ids"),
        )
        if predictions is None:
            return {"error": "Ensemble not loaded (missing artifacts?)", "disease_id": disease_id, "model": "ensemble"}

        if districts:
            filter_set = {d.strip() for d in districts.split(",") if d.strip()}
            predictions = [p for p in predictions if p["district"] in filter_set]

        total_lower = sum(p["lower"] for p in predictions)
        total_median = sum(p["median"] for p in predictions)
        total_upper = sum(p["upper"] for p in predictions)
        out = {
            "disease_id": disease_id,
            "model": "ensemble",
            "prediction_week_start": prediction_week_start,
            "prediction_week_end": prediction_week_end,
            "districts": predictions,
            "country_total_lower": round(total_lower, 2),
            "country_total_median": round(total_median, 2),
            "country_total_upper": round(total_upper, 2),
            "n_districts": len(predictions),
        }
        if weather_fallback:
            out["weather_fallback"] = True
        return out

    # LSTM
    weather_fallback_lstm = False
    if prediction_date:
        try:
            target_dt = datetime.strptime(prediction_date.strip(), "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid prediction_date; use YYYY-MM-DD", "disease_id": disease_id, "model": "lstm"}
        week_start, _ = get_epi_week_start_end(target_dt)
        week_start_str = week_start.strftime("%Y-%m-%d")
        _, err = load_or_fetch_weather_for_week(target_dt, disease_id)
        if err:
            payload = get_last_n_weeks_features_per_district(disease_id, n_weeks=4)
            if payload is None:
                return {"error": err, "disease_id": disease_id, "model": "lstm"}
            weather_fallback_lstm = True
        else:
            payload = get_lstm_4weeks_payload(disease_id, week_start_str)
            if payload is None:
                payload = get_last_n_weeks_features_per_district(disease_id, n_weeks=4)
                weather_fallback_lstm = payload is not None
            if payload is None:
                return {"error": "Insufficient data for LSTM (need 4 weeks in cache/model_data for selected week)", "disease_id": disease_id, "model": "lstm"}
        all_districts = payload["districts"]
        feature_matrix = payload["feature_matrix"]
        last_week_start = payload["last_week_start"]
        if weather_fallback_lstm:
            last_dt = datetime.strptime(last_week_start, "%Y-%m-%d")
            prediction_week_start = (last_dt + timedelta(days=7)).strftime("%Y-%m-%d")
            prediction_week_end = (last_dt + timedelta(days=13)).strftime("%Y-%m-%d")
        else:
            prediction_week_start = (datetime.strptime(last_week_start, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
            prediction_week_end = (datetime.strptime(prediction_week_start, "%Y-%m-%d") + timedelta(days=6)).strftime("%Y-%m-%d")
    else:
        payload = get_last_n_weeks_features_per_district(disease_id, n_weeks=4)
        if payload is None:
            return {"error": "Insufficient data for prediction", "disease_id": disease_id, "model": "lstm"}
        all_districts = payload["districts"]
        feature_matrix = payload["feature_matrix"]
        last_week_start = payload["last_week_start"]
        last_dt = datetime.strptime(last_week_start, "%Y-%m-%d")
        prediction_week_start = (last_dt + timedelta(days=7)).strftime("%Y-%m-%d")
        prediction_week_end = (last_dt + timedelta(days=13)).strftime("%Y-%m-%d")

    predictions = predict_with_districts(disease_id, feature_matrix, all_districts)
    if predictions is None:
        return {"error": "Model not loaded", "disease_id": disease_id, "model": "lstm"}

    if districts:
        filter_set = {d.strip() for d in districts.split(",") if d.strip()}
        predictions = [p for p in predictions if p["district"] in filter_set]

    total = sum(p["predicted_cases"] for p in predictions)
    out = {
        "disease_id": disease_id,
        "model": "lstm",
        "prediction_week_start": prediction_week_start,
        "prediction_week_end": prediction_week_end,
        "districts": predictions,
        "country_total": round(total, 2),
        "n_districts": len(predictions),
    }
    if weather_fallback_lstm:
        out["weather_fallback"] = True
    return out
