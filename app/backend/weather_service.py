"""Load or fetch weather for a given week; cache in weather_processed for ensemble/LSTM."""
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Allow importing open_weather from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from .config import (
    MODEL_DATA_DIR,
    WEATHER_PROCESSED_DIR,
    FEATURE_COLS,
)
from .data_service import load_disease_data, get_districts


def _ensure_weather_processed_dir() -> None:
    os.makedirs(WEATHER_PROCESSED_DIR, exist_ok=True)


def get_epi_week_start_end(d: datetime) -> tuple[datetime, datetime]:
    """ISO week (Monday=1). Return (week_start, week_end) as datetime at midnight."""
    if isinstance(d, str):
        d = pd.Timestamp(d).to_pydatetime()
    # Monday = 0 in Python weekday()
    weekday = d.weekday()
    week_start = d - timedelta(days=weekday)
    week_start = datetime(week_start.year, week_start.month, week_start.day)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end


def get_cached_weather_path(week_start: datetime) -> Optional[str]:
    """Path to cached CSV for this week (week_start as YYYY-MM-DD)."""
    _ensure_weather_processed_dir()
    key = week_start.strftime("%Y-%m-%d")
    path = os.path.join(WEATHER_PROCESSED_DIR, f"week_{key}.csv")
    return path if os.path.isfile(path) else None


def _monsoon_flags(month: int) -> tuple[int, int, int]:
    """Sri Lanka: IM2 Oct-Nov, NE Dec-Mar, SW May-Sep. Return (monsoon_IM2, monsoon_NE, monsoon_SW)."""
    if month in (10, 11):
        return (1, 0, 0)
    if month in (12, 1, 2, 3):
        return (0, 1, 0)
    if month in (5, 6, 7, 8, 9):
        return (0, 0, 1)
    return (0, 0, 0)


def _week_number(week_start: datetime) -> int:
    """ISO week number (1-53)."""
    return week_start.isocalendar()[1]


def _sin_cos_week(week_number: int) -> tuple[float, float]:
    """Seasonality encoding."""
    return (np.sin(2 * np.pi * week_number / 52), np.cos(2 * np.pi * week_number / 52))


def _aggregate_to_weekly(point_df: pd.DataFrame) -> dict:
    """Aggregate point (3h or daily) T2M, RH2M, PRECTOTCORR to weekly max/min/avg."""
    if point_df.empty:
        return {}
    agg = {
        "T2M_max": point_df["T2M"].max(),
        "T2M_min": point_df["T2M"].min(),
        "T2M_avg": point_df["T2M"].mean(),
        "T2M_MAX_max": point_df["T2M_MAX"].max(),
        "T2M_MAX_min": point_df["T2M_MAX"].min(),
        "T2M_MAX_avg": point_df["T2M_MAX"].mean(),
        "T2M_MIN_max": point_df["T2M_MIN"].max(),
        "T2M_MIN_min": point_df["T2M_MIN"].min(),
        "T2M_MIN_avg": point_df["T2M_MIN"].mean(),
        "RH2M_max": point_df["RH2M"].max(),
        "RH2M_min": point_df["RH2M"].min(),
        "RH2M_avg": point_df["RH2M"].mean(),
        "PRECTOTCORR_max": point_df["PRECTOTCORR"].max(),
        "PRECTOTCORR_min": point_df["PRECTOTCORR"].min(),
        "PRECTOTCORR_avg": point_df["PRECTOTCORR"].mean(),
    }
    return agg


def _fetch_week_weather_from_api(week_start: datetime, week_end: datetime) -> Optional[pd.DataFrame]:
    """Fetch forecast/current for week_start--week_end for all districts; return one row per district (weekly aggregates)."""
    api_key = os.environ.get("OPENWEATHER_API_KEY") or os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return None
    try:
        from open_weather import OpenWeatherDataFetcher
    except ImportError:
        return None
    fetcher = OpenWeatherDataFetcher(api_key)
    start_ts = int(datetime(week_start.year, week_start.month, week_start.day).timestamp())
    end_ts = int(datetime(week_end.year, week_end.month, week_end.day, 23, 59, 59).timestamp())
    all_records = []
    for district, (lat, lon) in fetcher.districts.items():
        # Forecast returns 5-day 3-hourly; we take points in [start_ts, end_ts]
        forecast = fetcher.get_forecast_weather(lat, lon, district)
        if not forecast:
            continue
        df = pd.DataFrame(forecast)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"].dt.date >= week_start.date()) & (df["date"].dt.date <= week_end.date())]
        if df.empty:
            # Use current weather as fallback for one day
            cur = fetcher.get_current_weather(lat, lon, district)
            if cur:
                df = pd.DataFrame([{**cur, "date": week_start.strftime("%Y-%m-%d")}])
                df["date"] = pd.to_datetime(df["date"])
        if df.empty:
            time.sleep(0.3)
            continue
        agg = _aggregate_to_weekly(df)
        if not agg:
            time.sleep(0.3)
            continue
        agg["district"] = district
        all_records.append(agg)
        time.sleep(0.3)
    if not all_records:
        return None
    return pd.DataFrame(all_records)


def _get_lags_for_week(disease_id: str, week_start: datetime, districts: list[str]) -> Optional[pd.DataFrame]:
    """Get lag_1..6 (PRECTOTCORR_avg, T2M_avg, RH2M_avg) for the 6 weeks before week_start from model_data."""
    df = load_disease_data(disease_id)
    if df is None or df.empty:
        return None
    df["start_date"] = pd.to_datetime(df["start_date"])
    # 6 weeks before: week_start-7, week_start-14, ..., week_start-42
    week_starts_needed = [week_start - timedelta(days=7 * i) for i in range(1, 7)]
    start_strs = [d.strftime("%Y-%m-%d") for d in week_starts_needed]
    sub = df[df["start_date"].astype(str).str[:10].isin(start_strs)]
    if sub.empty:
        return None
    out = []
    for district in districts:
        dsub = sub[sub["district"] == district].sort_values("start_date", ascending=False)
        if len(dsub) < 6:
            continue
        dsub = dsub.head(6).reset_index(drop=True)  # lag_1 = most recent week before, lag_6 = 6 weeks before
        row = {"district": district}
        for i in range(6):
            r = dsub.iloc[i]
            row[f"PRECTOTCORR_avg_lag_{i+1}"] = r["PRECTOTCORR_avg"]
            row[f"T2M_avg_lag_{i+1}"] = r["T2M_avg"]
            row[f"RH2M_avg_lag_{i+1}"] = r["RH2M_avg"]
        out.append(row)
    if not out:
        return None
    return pd.DataFrame(out)


def build_week_row(week_agg: pd.DataFrame, week_start: datetime, lags_df: Optional[pd.DataFrame], districts_order: list[str]) -> Optional[pd.DataFrame]:
    """Build full feature rows (FEATURE_COLS + metadata) for one week. week_agg has district + T2M_*, RH2M_*, PRECTOTCORR_*."""
    week_end = week_start + timedelta(days=6)
    week_num = _week_number(week_start)
    sin_w, cos_w = _sin_cos_week(week_num)
    month = week_start.month
    m_im2, m_ne, m_sw = _monsoon_flags(month)
    duration_str = f" {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}"
    rows = []
    for district in districts_order:
        agg_row = week_agg[week_agg["district"] == district]
        if agg_row.empty:
            continue
        agg_row = agg_row.iloc[0]
        lag_row = {}
        for i in range(1, 7):
            lag_row[f"PRECTOTCORR_avg_lag_{i}"] = 0.0
            lag_row[f"T2M_avg_lag_{i}"] = 0.0
            lag_row[f"RH2M_avg_lag_{i}"] = 0.0
        if lags_df is not None:
            lr = lags_df[lags_df["district"] == district]
            if not lr.empty:
                lag_row = lr.iloc[0].to_dict()
        row = {
            "district": district,
            "week_id": week_num,
            "start_date": week_start.strftime("%Y-%m-%d"),
            "end_date": week_end.strftime("%Y-%m-%d"),
            "Duration": duration_str,
            "T2M_max": agg_row["T2M_max"],
            "T2M_min": agg_row["T2M_min"],
            "T2M_avg": agg_row["T2M_avg"],
            "T2M_MAX_max": agg_row["T2M_MAX_max"],
            "T2M_MAX_min": agg_row["T2M_MAX_min"],
            "T2M_MAX_avg": agg_row["T2M_MAX_avg"],
            "T2M_MIN_max": agg_row["T2M_MIN_max"],
            "T2M_MIN_min": agg_row["T2M_MIN_min"],
            "T2M_MIN_avg": agg_row["T2M_MIN_avg"],
            "RH2M_max": agg_row["RH2M_max"],
            "RH2M_min": agg_row["RH2M_min"],
            "RH2M_avg": agg_row["RH2M_avg"],
            "PRECTOTCORR_max": agg_row["PRECTOTCORR_max"],
            "PRECTOTCORR_min": agg_row["PRECTOTCORR_min"],
            "PRECTOTCORR_avg": agg_row["PRECTOTCORR_avg"],
            "month": month,
            "monsoon_IM2": m_im2,
            "monsoon_NE": m_ne,
            "monsoon_SW": m_sw,
            "week_number": week_num,
            "sin_week": sin_w,
            "cos_week": cos_w,
            "target": 0,
        }
        for k, v in lag_row.items():
            if k != "district":
                row[k] = v
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for col in ["monsoon_IM2", "monsoon_NE", "monsoon_SW"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def load_or_fetch_weather_for_week(
    target_date: datetime,
    disease_id: str,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load cached week CSV or fetch from API, build full rows with lags, save to cache.
    Returns (df with FEATURE_COLS + metadata, None) or (None, error_message).
    """
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date).to_pydatetime()
    week_start, week_end = get_epi_week_start_end(target_date)
    try:
        from datetime import timezone
        today = datetime.now(timezone.utc).date()
    except Exception:
        today = datetime.now().date()
    if hasattr(week_start, 'date'):
        ws_date = week_start.date()
    else:
        ws_date = week_start
    days_ahead = (ws_date - today).days if ws_date >= today else 0

    # More than 1 week ahead: API cannot provide forecast
    if days_ahead > 7:
        return None, "Cannot predict more than 1 week ahead. Weather forecast is not available for that date."

    cached = get_cached_weather_path(week_start)
    if cached:
        try:
            df = pd.read_csv(cached)
            df["start_date"] = pd.to_datetime(df["start_date"])
            return df, None
        except Exception:
            pass

    districts_order = get_districts(disease_id)
    if not districts_order:
        return None, "No districts found for this disease."

    # Fetch weekly weather from API
    week_agg = _fetch_week_weather_from_api(week_start, week_end)
    if week_agg is None or week_agg.empty:
        if days_ahead > 0:
            return None, "Cannot predict more than 1 week ahead. Weather forecast is not available for that date."
        return None, "Failed to fetch weather data from API. Check OPENWEATHER_API_KEY and network."

    # Align districts: only those in both week_agg and districts_order
    districts_order = [d for d in districts_order if d in week_agg["district"].tolist()]
    if not districts_order:
        return None, "No matching districts between API and training data."

    # Lags from model_data (6 weeks before)
    lags_df = _get_lags_for_week(disease_id, week_start, districts_order)
    if lags_df is None or len(lags_df) < len(districts_order):
        # Try without lags (zeros) for future weeks when model_data has no recent data
        lags_df = None

    full_df = build_week_row(week_agg, week_start, lags_df, districts_order)
    if full_df is None or full_df.empty:
        return None, "Failed to build feature rows for prediction."

    _ensure_weather_processed_dir()
    out_path = os.path.join(WEATHER_PROCESSED_DIR, f"week_{week_start.strftime('%Y-%m-%d')}.csv")
    full_df.to_csv(out_path, index=False)
    return full_df, None


def get_ensemble_payload_from_weather_df(df: pd.DataFrame) -> Optional[dict]:
    """From a weather week DataFrame (with FEATURE_COLS), build payload {districts, feature_matrix, last_week_start} for ensemble."""
    from .config import FEATURE_COLS
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) != len(FEATURE_COLS):
        return None
    districts = df["district"].tolist()
    X = df[FEATURE_COLS].values.astype(np.float64)
    week_start = df["start_date"].iloc[0]
    if hasattr(week_start, 'strftime'):
        week_start_str = week_start.strftime("%Y-%m-%d")
    else:
        week_start_str = pd.Timestamp(week_start).strftime("%Y-%m-%d")
    return {
        "districts": districts,
        "feature_matrix": X,
        "last_week_start": week_start_str,
    }


def get_lstm_4weeks_payload(disease_id: str, target_week_start: str) -> Optional[dict]:
    """Get 4 weeks of features ending at target_week_start: from model_data or weather_processed cache."""
    from .config import FEATURE_COLS
    target_dt = pd.Timestamp(target_week_start)
    # 4 weeks: week-3, week-2, week-1, target week (each as Monday)
    week_dates = [target_dt - pd.Timedelta(days=7 * i) for i in range(3, -1, -1)]
    week_starts = []
    for d in week_dates:
        ws, _ = get_epi_week_start_end(d.to_pydatetime())
        week_starts.append(ws.strftime("%Y-%m-%d"))
    all_weeks = []
    for ws_str in week_starts:
        cached = get_cached_weather_path(datetime.strptime(ws_str, "%Y-%m-%d"))
        if cached:
            df = pd.read_csv(cached)
            if "start_date" in df.columns and not df.empty:
                all_weeks.append(df)
                continue
        df = load_disease_data(disease_id)
        if df is None:
            return None
        df["start_date"] = pd.to_datetime(df["start_date"])
        sub = df[df["start_date"].astype(str).str[:10] == ws_str]
        if sub.empty:
            return None
        all_weeks.append(sub)
    if len(all_weeks) != 4:
        return None
    districts = sorted(all_weeks[0]["district"].unique().tolist())
    for w in all_weeks:
        districts = [d for d in districts if d in w["district"].tolist()]
    if not districts:
        return None
    mats = []
    for w in all_weeks:
        w = w[w["district"].isin(districts)].set_index("district").loc[districts]
        avail = [c for c in FEATURE_COLS if c in w.columns]
        if len(avail) != len(FEATURE_COLS):
            return None
        mats.append(w[FEATURE_COLS].values.astype(np.float64))
    feature_matrix = np.stack(mats, axis=1)
    return {
        "districts": districts,
        "feature_matrix": feature_matrix,
        "last_week_start": target_week_start,
        "n_weeks": 4,
    }
