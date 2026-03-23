"""Load and serve case-count and feature data from model_data CSVs."""
import os
from typing import Optional

import numpy as np
import pandas as pd

from .config import MODEL_DATA_DIR, FEATURE_COLS, DISEASES


def get_districts(disease_id: str) -> list[str]:
    """Return sorted list of district names for a disease."""
    train_path = os.path.join(MODEL_DATA_DIR, f"{disease_id}_train.csv")
    if not os.path.exists(train_path):
        return []
    df = pd.read_csv(train_path, nrows=10000)
    return sorted(df["district"].dropna().unique().tolist())


def load_disease_data(disease_id: str) -> Optional[pd.DataFrame]:
    """Load combined train + test data for a disease (all weeks)."""
    train_path = os.path.join(MODEL_DATA_DIR, f"{disease_id}_train.csv")
    test_path = os.path.join(MODEL_DATA_DIR, f"{disease_id}_test.csv")
    if not os.path.exists(train_path):
        return None
    train = pd.read_csv(train_path)
    train["start_date"] = pd.to_datetime(train["start_date"])
    train["end_date"] = pd.to_datetime(train["end_date"])
    if os.path.exists(test_path):
        test = pd.read_csv(test_path)
        test["start_date"] = pd.to_datetime(test["start_date"])
        test["end_date"] = pd.to_datetime(test["end_date"])
        df = pd.concat([train, test], ignore_index=True)
    else:
        df = train
    df = df.sort_values(["district", "start_date"]).reset_index(drop=True)
    # Ensure boolean monsoon columns are numeric for model
    for col in ["monsoon_IM2", "monsoon_NE", "monsoon_SW"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def load_full_disease_data(disease_id: str) -> Optional[pd.DataFrame]:
    """Load train + val + test for ensemble build_features (full temporal history)."""
    parts = []
    for split in ("train", "val", "test"):
        path = os.path.join(MODEL_DATA_DIR, f"{disease_id}_{split}.csv")
        if not os.path.exists(path):
            continue
        part = pd.read_csv(path)
        part["start_date"] = pd.to_datetime(part["start_date"])
        part["end_date"] = pd.to_datetime(part["end_date"])
        parts.append(part)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True).drop_duplicates().sort_values(["district", "start_date"]).reset_index(drop=True)
    for col in ["monsoon_IM2", "monsoon_NE", "monsoon_SW"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def get_past_cases(
    disease_id: str,
    districts: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    """Return past case counts: list of {district, week_start, week_end, cases}."""
    df = load_disease_data(disease_id)
    if df is None or df.empty:
        return []
    if districts is not None:
        df = df[df["district"].isin(districts)]
    if start_date:
        df = df[df["start_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["start_date"] <= pd.to_datetime(end_date)]
    out = []
    for _, row in df.iterrows():
        out.append({
            "district": row["district"],
            "week_start": row["start_date"].strftime("%Y-%m-%d"),
            "week_end": row["end_date"].strftime("%Y-%m-%d"),
            "cases": int(row["target"]),
        })
    return out


def get_past_cases_dataframe(
    disease_id: str,
    districts: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Same as get_past_cases but returns a DataFrame for charts/tables."""
    data = get_past_cases(disease_id, districts, start_date, end_date)
    if not data:
        return pd.DataFrame(columns=["district", "week_start", "week_end", "cases"])
    return pd.DataFrame(data)


def get_last_n_weeks_features_per_district(
    disease_id: str, n_weeks: int = 4
) -> Optional[dict]:
    """
    For each district, return the last n_weeks rows of features (for LSTM sequence).
    """
    df = load_disease_data(disease_id)
    if df is None or len(df) < n_weeks:
        return None
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) != len(FEATURE_COLS):
        return None
    districts_order = sorted(df["district"].unique().tolist())
    by_district = []
    last_week_start = None
    for district in districts_order:
        sub = df[df["district"] == district].tail(n_weeks)
        if len(sub) < n_weeks:
            continue
        sub = sub.sort_values("start_date")
        X = sub[FEATURE_COLS].values.astype(float)
        by_district.append((district, X, sub["start_date"].iloc[-1]))
        if last_week_start is None or sub["start_date"].iloc[-1] > last_week_start:
            last_week_start = sub["start_date"].iloc[-1]
    if not by_district:
        return None
    districts = [x[0] for x in by_district]
    feature_matrix = np.array([x[1] for x in by_district])
    return {
        "districts": districts,
        "feature_matrix": feature_matrix,
        "last_week_start": pd.Timestamp(last_week_start).strftime("%Y-%m-%d"),
        "n_weeks": n_weeks,
    }


def get_last_week_features_per_district(disease_id: str) -> Optional[dict]:
    """
    For each district, return the last week's single row of features (for ensemble).
    feature_matrix shape: (n_districts, 40).
    """
    payload = get_last_n_weeks_features_per_district(disease_id, n_weeks=1)
    if payload is None:
        return None
    # feature_matrix is (n_districts, 1, 40) -> squeeze to (n_districts, 40)
    payload["feature_matrix"] = payload["feature_matrix"][:, 0, :]
    return payload


def get_ensemble_payload_with_build_features(
    disease_id: str,
    prediction_week_df: pd.DataFrame,
) -> Optional[dict]:
    """
    Build ensemble payload using build_features_lepto (matches plots/step_03).
    prediction_week_df: weather rows for prediction week (from load_or_fetch_weather).
    Returns {districts, feature_matrix, last_week_start, feature_names} or None.
    """
    from .feature_builder import (
        build_features_lepto,
        build_ensemble_feature_matrix,
        load_ensemble_feature_names,
    )

    feature_names = load_ensemble_feature_names(disease_id)
    if not feature_names:
        return None

    historical = load_full_disease_data(disease_id)
    if historical is None or historical.empty:
        return None

    pred_df = prediction_week_df.copy()
    if "target" not in pred_df.columns:
        pred_df["target"] = 0.0
    if "week_id" not in pred_df.columns and "week_number" in pred_df.columns:
        pred_df["week_id"] = pred_df["week_number"]

    train_path = os.path.join(MODEL_DATA_DIR, f"{disease_id}_train.csv")
    train_df = pd.read_csv(train_path)
    train_df["start_date"] = pd.to_datetime(train_df["start_date"])
    if "week_id" not in train_df.columns and "week_number" in train_df.columns:
        train_df["week_id"] = train_df["week_number"]

    if "week_id" not in historical.columns and "week_number" in historical.columns:
        historical = historical.copy()
        historical["week_id"] = historical["week_number"]

    # Union columns, fill missing with 0
    all_cols = list(historical.columns) + [c for c in pred_df.columns if c not in historical.columns]
    for c in all_cols:
        if c not in historical.columns:
            historical[c] = 0.0
        if c not in pred_df.columns:
            pred_df[c] = 0.0 if c != "target" else 0.0
    pred_df = pred_df[historical.columns.tolist()]
    full = pd.concat([historical, pred_df], ignore_index=True)
    full = full.drop_duplicates(subset=["district", "start_date"], keep="last")
    full = full.sort_values(["district", "week_id" if "week_id" in full.columns else "start_date"])

    full_fe = build_features_lepto(full, train_df)

    week_start = prediction_week_df["start_date"].iloc[0]
    week_str = pd.Timestamp(week_start).strftime("%Y-%m-%d")
    full_fe["start_date"] = pd.to_datetime(full_fe["start_date"])
    pred_mask = full_fe["start_date"].astype(str).str[:10] == week_str[:10]
    pred_rows = full_fe[pred_mask]
    if pred_rows.empty:
        return None

    districts = pred_rows["district"].tolist()
    X = build_ensemble_feature_matrix(pred_rows, feature_names)
    week_numbers = pred_rows["week_number"].tolist() if "week_number" in pred_rows.columns else None
    week_ids = pred_rows["week_id"].tolist() if "week_id" in pred_rows.columns else None

    return {
        "districts": districts,
        "feature_matrix": X,
        "last_week_start": week_str,
        "feature_names": feature_names,
        "week_numbers": week_numbers,
        "week_ids": week_ids,
    }


def get_ensemble_payload_from_last_week(disease_id: str) -> Optional[dict]:
    """Build ensemble payload for the last week in historical data (fallback when no weather fetch)."""
    from .feature_builder import (
        build_features_lepto,
        build_ensemble_feature_matrix,
        load_ensemble_feature_names,
    )

    feature_names = load_ensemble_feature_names(disease_id)
    if not feature_names:
        return None

    historical = load_full_disease_data(disease_id)
    if historical is None or historical.empty:
        return None

    train_path = os.path.join(MODEL_DATA_DIR, f"{disease_id}_train.csv")
    train_df = pd.read_csv(train_path)
    if "week_id" not in train_df.columns and "week_number" in train_df.columns:
        train_df["week_id"] = train_df["week_number"]

    if "week_id" not in historical.columns and "week_number" in historical.columns:
        historical = historical.copy()
        historical["week_id"] = historical["week_number"]

    full_fe = build_features_lepto(historical, train_df)
    last_week_start = full_fe["start_date"].max()
    week_str = pd.Timestamp(last_week_start).strftime("%Y-%m-%d")
    pred_mask = full_fe["start_date"] == last_week_start
    pred_rows = full_fe[pred_mask]
    if pred_rows.empty:
        return None

    districts = pred_rows["district"].tolist()
    X = build_ensemble_feature_matrix(pred_rows, feature_names)
    week_numbers = pred_rows["week_number"].tolist() if "week_number" in pred_rows.columns else None
    week_ids = pred_rows["week_id"].tolist() if "week_id" in pred_rows.columns else None
    return {
        "districts": districts,
        "feature_matrix": X,
        "last_week_start": week_str,
        "feature_names": feature_names,
        "week_numbers": week_numbers,
        "week_ids": week_ids,
    }
