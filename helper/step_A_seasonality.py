import pandas as pd
import numpy as np

# ===============================
# STEP A: LOAD DATA
# ===============================
df = pd.read_csv("weather_weekly.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Date parsing
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

# Sort to keep temporal order
df = df.sort_values(['district', 'start_date']).reset_index(drop=True)

# ===============================
# STEP A1: MONSOON INDICATORS (Sri Lanka specific)
# ===============================
df['month'] = df['start_date'].dt.month

def monsoon_type(month):
    if month in [12, 1, 2]:
        return 'NE'     # Northeast Monsoon
    elif month in [3, 4]:
        return 'IM1'    # First Inter-monsoon
    elif month in [5, 6, 7, 8, 9]:
        return 'SW'     # Southwest Monsoon
    else:
        return 'IM2'    # Second Inter-monsoon

df['monsoon'] = df['month'].apply(monsoon_type)

# Convert to dummy variables
df = pd.get_dummies(df, columns=['monsoon'], drop_first=True)

# ===============================
# STEP A2: CYCLICAL WEEK FEATURES
# ===============================
df['week_number'] = df['start_date'].dt.isocalendar().week.astype(int)

df['sin_week'] = np.sin(2 * np.pi * df['week_number'] / 52)
df['cos_week'] = np.cos(2 * np.pi * df['week_number'] / 52)

# ===============================
# STEP A3: SAVE OUTPUT
# ===============================
output_path = "weather_weekly_with_seasonality.csv"
df.to_csv(output_path, index=False)

print("✅ Step A completed. Saved to:", output_path)
