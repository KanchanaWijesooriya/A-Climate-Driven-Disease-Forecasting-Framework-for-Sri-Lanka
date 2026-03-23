"""Ensemble feature building: replicates step_03 build_features_lepto for production inference."""
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import ARTIFACTS_DIR


def build_features_lepto(full_df: pd.DataFrame, train_df_for_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Leptospirosis-style feature engineering: target lags, rolling stats, district seasonal means,
    climate interactions. Matches step_03_ensemble_blending.build_features_lepto.
    """
    df = full_df.sort_values(["district", "week_id"]).copy()
    grp = df.groupby("district")["target"]
    for lag in (1, 2, 3, 4, 5, 6, 8, 12):
        df[f"target_lag_{lag}"] = grp.shift(lag)
    for w in (2, 4, 8, 12):
        s = grp.shift(1)
        df[f"target_roll_mean_{w}"] = s.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"target_roll_std_{w}"] = s.transform(lambda x: x.rolling(w, min_periods=1).std())
        df[f"target_roll_max_{w}"] = s.transform(lambda x: x.rolling(w, min_periods=1).max())
        df[f"target_roll_min_{w}"] = s.transform(lambda x: x.rolling(w, min_periods=1).min())
    df["target_trend_4"] = df.groupby("district")["target_roll_mean_4"].diff()
    df["target_trend_8"] = df.groupby("district")["target_roll_mean_8"].diff()
    df["target_accel"] = df.groupby("district")["target_trend_4"].diff()
    if "T2M_avg" in df.columns and "RH2M_avg" in df.columns:
        df["heat_humidity"] = df["T2M_avg"] * df["RH2M_avg"]
    if "PRECTOTCORR_avg" in df.columns and "RH2M_avg" in df.columns:
        df["rain_humidity"] = df["PRECTOTCORR_avg"] * df["RH2M_avg"]
    if "PRECTOTCORR_avg" in df.columns and "T2M_avg" in df.columns:
        df["rain_temp"] = df["PRECTOTCORR_avg"] * df["T2M_avg"]
    if "T2M_max" in df.columns and "T2M_min" in df.columns:
        df["temp_range"] = df["T2M_max"] - df["T2M_min"]
    if "PRECTOTCORR_avg_lag_1" in df.columns and "RH2M_avg_lag_1" in df.columns:
        df["rain_hum_lag1"] = df["PRECTOTCORR_avg_lag_1"] * df["RH2M_avg_lag_1"]
    if "PRECTOTCORR_avg_lag_2" in df.columns and "RH2M_avg_lag_2" in df.columns:
        df["rain_hum_lag2"] = df["PRECTOTCORR_avg_lag_2"] * df["RH2M_avg_lag_2"]
    rain_cols_4 = [f"PRECTOTCORR_avg_lag_{i}" for i in range(1, 5)]
    rain_cols_8 = [f"PRECTOTCORR_avg_lag_{i}" for i in range(1, 9)]
    df["cum_rain_4w"] = df[[c for c in rain_cols_4 if c in df.columns]].sum(axis=1)
    df["cum_rain_8w"] = df[[c for c in rain_cols_8 if c in df.columns]].sum(axis=1)
    dw_mean = (
        train_df_for_stats.groupby(["district", "week_number"])["target"]
        .mean()
        .rename("district_week_mean")
        .reset_index()
    )
    dist_mean = train_df_for_stats.groupby("district")["target"].mean().rename("dist_mean")
    late_train = train_df_for_stats[
        train_df_for_stats["week_id"] >= train_df_for_stats["week_id"].max() - 52
    ]
    late_dist_mean = late_train.groupby("district")["target"].mean().rename("late_dist_mean")
    df = df.merge(dw_mean, on=["district", "week_number"], how="left")
    df = df.merge(dist_mean, on="district", how="left")
    df = df.merge(late_dist_mean, on="district", how="left")
    df["district_week_mean"] = df["district_week_mean"].fillna(df["dist_mean"])
    df["late_dist_mean"] = df["late_dist_mean"].fillna(df["dist_mean"])
    df["excess_over_seasonal"] = df["target_roll_mean_4"] - df["district_week_mean"]
    df["ratio_to_seasonal"] = df["target_roll_mean_4"] / (df["district_week_mean"] + 0.5)
    le = LabelEncoder()
    le.fit(df["district"])
    df["district_enc"] = le.transform(df["district"])
    return df


def load_ensemble_feature_names(disease_id: str) -> Optional[list[str]]:
    """Load feature_names.json from artifacts for a disease."""
    path = os.path.join(ARTIFACTS_DIR, disease_id, "feature_names.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_ensemble_feature_matrix(
    df_with_features: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """
    Extract feature matrix from DataFrame, ensuring columns match feature_names in order.
    Fills missing cols with 0; fills NaN with 0.
    """
    rows = []
    for _, row in df_with_features.iterrows():
        vec = []
        for col in feature_names:
            if col in row.index:
                val = row[col]
                if pd.isna(val) or (isinstance(val, bool) and not val):
                    vec.append(0.0)
                else:
                    vec.append(float(val))
            else:
                vec.append(0.0)
        rows.append(vec)
    return np.ascontiguousarray(rows, dtype=np.float64)
