"""Application configuration and paths."""
import os

# Project root (Research folder); app lives in Research/app/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Load .env from project root so OPENWEATHER_API_KEY etc. are available to the backend
try:
    from dotenv import load_dotenv
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()  # find .env in cwd or parents
except ImportError:
    pass

MODEL_DATA_DIR = os.path.join(PROJECT_ROOT, "model_data")
ARTIFACTS_DIR = os.path.join(MODEL_DATA_DIR, "artifacts")
# Processed weather for ensemble/LSTM (one CSV per epidemiological week)
WEATHER_PROCESSED_DIR = os.path.join(MODEL_DATA_DIR, "weather_processed")

DISEASES = [
    {"id": "leptospirosis", "name": "Leptospirosis"},
    {"id": "typhus", "name": "Typhus"},
    {"id": "hepatitis_a", "name": "Hepatitis A"},
    {"id": "chickenpox", "name": "Chickenpox"},
]

# Feature columns (must match training)
FEATURE_COLS = [
    "T2M_max", "T2M_min", "T2M_avg", "T2M_MAX_max", "T2M_MAX_min", "T2M_MAX_avg",
    "T2M_MIN_max", "T2M_MIN_min", "T2M_MIN_avg", "RH2M_max", "RH2M_min", "RH2M_avg",
    "PRECTOTCORR_max", "PRECTOTCORR_min", "PRECTOTCORR_avg", "month", "monsoon_IM2",
    "monsoon_NE", "monsoon_SW", "week_number", "sin_week", "cos_week",
    "PRECTOTCORR_avg_lag_1", "PRECTOTCORR_avg_lag_2", "PRECTOTCORR_avg_lag_3",
    "PRECTOTCORR_avg_lag_4", "PRECTOTCORR_avg_lag_5", "PRECTOTCORR_avg_lag_6",
    "T2M_avg_lag_1", "T2M_avg_lag_2", "T2M_avg_lag_3", "T2M_avg_lag_4",
    "T2M_avg_lag_5", "T2M_avg_lag_6", "RH2M_avg_lag_1", "RH2M_avg_lag_2",
    "RH2M_avg_lag_3", "RH2M_avg_lag_4", "RH2M_avg_lag_5", "RH2M_avg_lag_6",
]

METADATA_COLS = ["district", "week_id", "start_date", "end_date", "Duration", "target"]
