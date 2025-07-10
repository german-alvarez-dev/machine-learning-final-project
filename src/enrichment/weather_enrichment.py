import pandas as pd
import requests
import time
from datetime import datetime


def load_park_locations(path="data/external/theme_park_locations.csv"):
    return pd.read_csv(path)


def fetch_weather_for_location(lat, lon, date):
    """
    Query Open-Meteo API for historical weather data at a specific lat/lon and date.
    Returns a dict with temperature and precipitation values.
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "America/New_York"
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature_max": data["daily"]["temperature_2m_max"][0],
            "temperature_min": data["daily"]["temperature_2m_min"][0],
            "precipitation_sum": data["daily"]["precipitation_sum"][0]
        }
    except Exception as e:
        print(f"Failed to fetch weather for {lat}, {lon}, {date}: {e}")
        return {"temperature_max": None, "temperature_min": None, "precipitation_sum": None}


def enrich_weather(df, park_location_path="data/external/theme_park_locations.csv", cache_path="data/external/weather_cache.parquet"):
    """
    Enrich the main dataframe with historical weather by (theme_park, incident_date_parsed).
    Saves a cache of retrieved weather to avoid repeated API calls.
    """
    df = df.copy()

    parks = load_park_locations(park_location_path)
    parks = parks.set_index("theme_park")

    df["weather_key"] = df["theme_park"] + "_" + df["incident_date_parsed"].dt.strftime("%Y-%m-%d")

    if cache_path:
        try:
            weather_cache = pd.read_parquet(cache_path)
        except FileNotFoundError:
            weather_cache = pd.DataFrame(columns=["weather_key", "temperature_max", "temperature_min", "precipitation_sum"])
    else:
        weather_cache = pd.DataFrame()

    missing_keys = set(df["weather_key"]) - set(weather_cache["weather_key"])
    new_weather_data = []

    for key in missing_keys:
        park, date_str = key.rsplit("_", 1)
        if park not in parks.index:
            continue
        lat = parks.loc[park, "latitude"]
        lon = parks.loc[park, "longitude"]
        weather = fetch_weather_for_location(lat, lon, date_str)
        weather["weather_key"] = key
        new_weather_data.append(weather)
        time.sleep(1.1)  # avoid rate limits

    if new_weather_data:
        new_df = pd.DataFrame(new_weather_data)
        weather_cache = pd.concat([weather_cache, new_df], ignore_index=True)
        if cache_path:
            weather_cache.to_parquet(cache_path, index=False)

    df = df.merge(weather_cache, on="weather_key", how="left")
    df.drop(columns=["weather_key"], inplace=True)

    return df
