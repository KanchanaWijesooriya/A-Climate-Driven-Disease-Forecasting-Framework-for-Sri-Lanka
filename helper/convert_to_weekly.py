import pandas as pd

# ---------------------------
# 1. Load daily dataset
# ---------------------------
df = pd.read_csv("nasa_daily_weather_2020_2025.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Sort values
df = df.sort_values(["district", "date"])

# --------------------------------------------------
# 2. Create 7-day block index per district
# --------------------------------------------------
def create_week_index(x):
    x = x.sort_values("date")
    x["week_id"] = ((x["date"] - x["date"].min()).dt.days // 7)
    return x

df = df.groupby("district").apply(create_week_index).reset_index(drop=True)

# --------------------------------------------------
# 3. Aggregate into weekly features
# --------------------------------------------------
weekly = df.groupby(["district", "week_id"]).agg(
    start_date=("date", "min"),
    end_date=("date", "max"),

    T2M_max=("T2M", "max"),
    T2M_min=("T2M", "min"),
    T2M_avg=("T2M", "mean"),

    T2M_MAX_max=("T2M_MAX", "max"),
    T2M_MAX_min=("T2M_MAX", "min"),
    T2M_MAX_avg=("T2M_MAX", "mean"),

    T2M_MIN_max=("T2M_MIN", "max"),
    T2M_MIN_min=("T2M_MIN", "min"),
    T2M_MIN_avg=("T2M_MIN", "mean"),

    RH2M_max=("RH2M", "max"),
    RH2M_min=("RH2M", "min"),
    RH2M_avg=("RH2M", "mean"),

    PRECTOTCORR_max=("PRECTOTCORR", "max"),
    PRECTOTCORR_min=("PRECTOTCORR", "min"),
    PRECTOTCORR_avg=("PRECTOTCORR", "mean")
).reset_index()

# --------------------------------------------------
# 4. Create readable duration column
# --------------------------------------------------
weekly["Duration"] = weekly["start_date"].dt.strftime("%Y-%m-%d") + " to " + weekly["end_date"].dt.strftime("%Y-%m-%d")

# --------------------------------------------------
# 5. Save the final weekly dataset
# --------------------------------------------------
weekly.to_csv("weather_weekly.csv", index=False)

print("✅ Weekly dataset created successfully!")
print(weekly.head())
