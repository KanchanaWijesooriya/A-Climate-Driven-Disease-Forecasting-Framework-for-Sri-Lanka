import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class OpenWeatherDataFetcher:
    def __init__(self, api_key):
        """
        Initialize the fetcher with OpenWeatherMap API key
        
        Args:
            api_key (str): Your OpenWeatherMap API key
        """
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.history_url = "https://history.openweathermap.org/data/2.5/history/city"

        # Districts with their coordinates (latitude, longitude)
        self.districts = {
            "Colombo": (6.9271, 79.8612),
            "Gampaha": (7.0892, 80.7744),
            "Kalutara": (6.5896, 80.2712),
            "Kandy": (7.2906, 80.6337),
            "Matale": (7.7674, 80.6270),
            "Nuwara Eliya": (6.9497, 80.7891),
            "Galle": (6.0535, 80.2179),
            "Matara": (5.7489, 80.5353),
            "Hambantota": (6.1256, 81.1242),
            "Jaffna": (9.6615, 80.7740),
            "Kilinochchi": (9.3886, 80.4318),
            "Mannar": (8.9833, 79.9167),
            "Vavuniya": (8.7606, 80.4945),
            "Mullaitivu": (8.3068, 81.8109),
            "Batticaloa": (7.7102, 81.7050),
            "Ampara": (7.2833, 81.6667),
            "Trincomalee": (8.5874, 81.2348),
            "Kurunegala": (7.4821, 80.6354),
            "Puttalam": (8.0306, 79.8336),
            "Anuradhapura": (8.3163, 80.3858),
            "Polonnaruwa": (7.9408, 81.0081),
            "Badulla": (6.9934, 81.2671),
            "Monaragala": (6.8297, 81.3530),
            "Ratnapura": (6.7148, 80.3987),
            "Kegalle": (7.2515, 80.8393),
        }
    
    def get_current_weather(self, lat, lon, district_name):
        """
        Fetch current weather data from OpenWeatherMap
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            district_name (str): Name of the district
            
        Returns:
            dict: Weather data with mapped field names
        """
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Map OpenWeatherMap data to our required fields
            weather_record = {
                "date": datetime.fromtimestamp(data["dt"]).strftime("%Y-%m-%d"),
                "T2M": data["main"]["temp"],  # Current temperature
                "T2M_MAX": data["main"]["temp_max"],  # Max temperature
                "T2M_MIN": data["main"]["temp_min"],  # Min temperature
                "RH2M": data["main"]["humidity"],  # Relative humidity
                "PRECTOTCORR": data.get("rain", {}).get("1h", 0.0),  # Precipitation (1 hour)
                "district": district_name,
                "description": data["weather"][0]["description"]
            }
            
            return weather_record
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {district_name}: {e}")
            return None
    
    def get_forecast_weather(self, lat, lon, district_name):
        """
        Fetch 5-day forecast data from OpenWeatherMap
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            district_name (str): Name of the district
            
        Returns:
            list: List of forecast records
        """
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            forecast_records = []
            
            for item in data["list"]:
                # Group by day (use daily max/min from 3-hourly forecast)
                record = {
                    "date": datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d"),
                    "T2M": item["main"]["temp"],
                    "T2M_MAX": item["main"]["temp_max"],
                    "T2M_MIN": item["main"]["temp_min"],
                    "RH2M": item["main"]["humidity"],
                    "PRECTOTCORR": item.get("rain", {}).get("3h", 0.0),
                    "district": district_name
                }
                forecast_records.append(record)
            
            return forecast_records
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast for {district_name}: {e}")
            return []

    def get_historical_weather(self, lat, lon, district_name, start_ts, end_ts):
        """
        Fetch hourly historical weather from OpenWeatherMap History API.

        Requires a subscription that includes History API (see openweathermap.org/price).
        Free tier may not include historical data.

        Args:
            lat (float): Latitude
            lon (float): Longitude
            district_name (str): Name of the district
            start_ts (int): Start Unix timestamp (UTC)
            end_ts (int): End Unix timestamp (UTC)

        Returns:
            list: Hourly weather records with same field names as current/forecast
        """
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "type": "hour",
                "start": start_ts,
                "end": end_ts,
                "appid": self.api_key,
                "units": "metric",
            }
            response = requests.get(self.history_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            records = []
            for item in data.get("list", []):
                record = {
                    "date": datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d"),
                    "T2M": item["main"]["temp"],
                    "T2M_MAX": item["main"].get("temp_max", item["main"]["temp"]),
                    "T2M_MIN": item["main"].get("temp_min", item["main"]["temp"]),
                    "RH2M": item["main"]["humidity"],
                    "PRECTOTCORR": item.get("rain", {}).get("1h") or item.get("rain", {}).get("3h", 0.0) or 0.0,
                    "district": district_name,
                }
                records.append(record)
            return records
        except requests.exceptions.RequestException as e:
            print(f"Error fetching history for {district_name}: {e}")
            return []

    def fetch_all_historical_weather(self, start_date, end_date):
        """
        Fetch historical weather for all districts between start_date and end_date.

        Args:
            start_date: datetime.date or str "YYYY-MM-DD"
            end_date: datetime.date or str "YYYY-MM-DD"

        Returns:
            pd.DataFrame: Hourly historical data for all districts
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_ts = int(datetime.combine(end_date, datetime.max.time().replace(microsecond=0)).timestamp())

        print(f"Fetching historical weather from {start_date} to {end_date}...")
        all_data = []
        for district, (lat, lon) in self.districts.items():
            print(f"Fetching history for {district}...")
            records = self.get_historical_weather(lat, lon, district, start_ts, end_ts)
            all_data.extend(records)
            time.sleep(0.5)
        return pd.DataFrame(all_data)

    def fetch_all_current_weather(self):
        """
        Fetch current weather for all districts
        
        Returns:
            pd.DataFrame: Weather data for all districts
        """
        print("Fetching current weather data for all districts...")
        all_data = []
        
        for district, (lat, lon) in self.districts.items():
            print(f"Fetching data for {district}...")
            weather = self.get_current_weather(lat, lon, district)
            
            if weather:
                all_data.append(weather)
            
            # Rate limiting: OpenWeatherMap free tier has limits
            time.sleep(0.5)
        
        df = pd.DataFrame(all_data)
        return df
    
    def fetch_all_forecast_weather(self):
        """
        Fetch 5-day forecast for all districts
        
        Returns:
            pd.DataFrame: Forecast data for all districts
        """
        print("Fetching forecast weather data for all districts...")
        all_data = []
        
        for district, (lat, lon) in self.districts.items():
            print(f"Fetching forecast for {district}...")
            forecast = self.get_forecast_weather(lat, lon, district)
            all_data.extend(forecast)
            
            # Rate limiting
            time.sleep(0.5)
        
        df = pd.DataFrame(all_data)
        return df
    
    def save_to_csv(self, df, filename):
        """
        Save weather data to CSV file
        
        Args:
            df (pd.DataFrame): Weather dataframe
            filename (str): Output filename
        """
        columns = ["date", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "district"]
        df = df[[c for c in columns if c in df.columns]]
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


# Main execution
if __name__ == "__main__":
    # API key from environment (e.g. OPENWEATHER_API_KEY or OPENWEATHERMAP_API_KEY)
    API_KEY = os.environ.get("OPENWEATHER_API_KEY") or os.environ.get("OPENWEATHERMAP_API_KEY")
    if not API_KEY:
        raise SystemExit(
            "Missing API key. Set OPENWEATHER_API_KEY or OPENWEATHERMAP_API_KEY in your environment."
        )

    # Initialize fetcher
    fetcher = OpenWeatherDataFetcher(API_KEY)
    
    # Fetch current weather
    print("\n=== CURRENT WEATHER ===")
    current_df = fetcher.fetch_all_current_weather()
    print(current_df)
    fetcher.save_to_csv(current_df, "openweather_current.csv")
    
    # Fetch forecast
    print("\n=== FORECAST WEATHER ===")
    forecast_df = fetcher.fetch_all_forecast_weather()
    print(forecast_df.head(20))
    fetcher.save_to_csv(forecast_df, "openweather_forecast.csv")

    # Optional: fetch past days (History API – may require paid plan on OpenWeatherMap)
    # from datetime import date
    # end = date.today()
    # start = end - timedelta(days=5)
    # hist_df = fetcher.fetch_all_historical_weather(start, end)
    # if not hist_df.empty:
    #     fetcher.save_to_csv(hist_df, "openweather_historical.csv")
    #     print(hist_df.head(20))

    print("\n✓ Data collection complete!")