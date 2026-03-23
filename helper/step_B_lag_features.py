import pandas as pd

# ===============================
# STEP B: LOAD DATA
# ===============================
df = pd.read_csv("weather_weekly_with_seasonality.csv")

# Date parsing
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

# Sort to maintain correct temporal order
df = df.sort_values(['district', 'start_date']).reset_index(drop=True)

# ===============================
# STEP B1: DEFINE CLIMATE VARIABLES
# ===============================
climate_vars = [
    'PRECTOTCORR_avg',  # rainfall
    'T2M_avg',          # temperature
    'RH2M_avg'          # humidity
]

# ===============================
# STEP B2: CREATE LAG FEATURES (1–6 weeks)
# ===============================
for var in climate_vars:
    for lag in range(1, 7):
        df[f'{var}_lag_{lag}'] = df.groupby('district')[var].shift(lag)

# ===============================
# STEP B3: REMOVE INITIAL NaNs FROM LAGS
# ===============================
df_lagged = df.dropna().reset_index(drop=True)

# ===============================
# STEP B4: SAVE OUTPUT
# ===============================
output_path = "weather_weekly_with_seasonality_lags.csv"
df_lagged.to_csv(output_path, index=False)

print("✅ Step B completed. Saved to:", output_path)
