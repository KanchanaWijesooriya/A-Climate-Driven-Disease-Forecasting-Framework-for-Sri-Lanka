import requests
import pandas as pd
from tqdm import tqdm

# -------------------------------------
# 1. DISTRICT COORDINATES (Centroids)
# -------------------------------------
district_coords = {
    "Colombo": (6.9271, 79.8612),
    "Gampaha": (7.0850, 79.9994),
    "Kalutara": (6.5854, 80.0853),
    "Kandy": (7.2906, 80.6337),
    "Matale": (7.4675, 80.6234),
    "Nuwara Eliya": (6.9497, 80.7891),
    "Galle": (6.0535, 80.2210),
    "Matara": (5.9485, 80.5428),
    "Hambantota": (6.1246, 81.1185),
    "Jaffna": (9.6615, 80.0255),
    "Kilinochchi": (9.3890, 80.3990),
    "Mannar": (8.9800, 79.9095),
    "Vavuniya": (8.7500, 80.5000),
    "Mullaitivu": (9.2671, 80.8140),
    "Batticaloa": (7.7102, 81.6924),
    "Ampara": (7.2975, 81.6820),
    "Trincomalee": (8.5710, 81.2335),
    "Kurunegala": (7.4863, 80.3647),
    "Puttalam": (8.0400, 79.8283),
    "Anuradhapura": (8.3114, 80.4037),
    "Polonnaruwa": (7.9403, 81.0188),
    "Badulla": (6.9934, 81.0550),
    "Monaragala": (6.8900, 81.3500),
    "Ratnapura": (6.6850, 80.4037),
    "Kegalle": (7.2513, 80.3464)
}

# -------------------------
# 2. FUNCTION: GET DATA
# -------------------------
def fetch_nasa_data(lat, lon):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        "parameters=T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR&"
        "community=AG&"
        f"longitude={lon}&latitude={lat}&"
        "start=20260206&end=20260220&format=JSON"
    )

    response = requests.get(url)
    data = response.json()

    # Extract the parameters object
    params = data["properties"]["parameter"]

    # Build DataFrame with available parameters
    df_dict = {"date": list(params["T2M"].keys())}
    
    # Add each parameter if it exists
    for param_name in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR"]:
        if param_name in params:
            df_dict[param_name] = list(params[param_name].values())
    
    df = pd.DataFrame(df_dict)

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    return df


# -------------------------
# 3. LOOP THROUGH DISTRICTS
# -------------------------
all_districts = []

for district, (lat, lon) in tqdm(district_coords.items()):
    try:
        df = fetch_nasa_data(lat, lon)
        df["district"] = district
        all_districts.append(df)
    except Exception as e:
        print(f"Failed: {district} - Error: {str(e)}")

# -------------------------
# 4. MERGE & SAVE
# -------------------------
if all_districts:
    final_df = pd.concat(all_districts, ignore_index=True)
    final_df.to_csv("nasa_daily_weather_2020_2025.csv", index=False)
    print(f"✅ NASA daily weather saved! ({len(all_districts)} districts)")
else:
    print("❌ No data collected. All districts failed.")
