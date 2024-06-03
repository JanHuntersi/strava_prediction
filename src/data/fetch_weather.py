from src.data.weather import get_weather_data
from definitions import ROOT_DIR
import os
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

path_to_weather = os.path.join(ROOT_DIR, "data", "raw", "weather.csv") 

def fetch_weather():
    
    data = get_weather_data()

    if not os.path.exists(path_to_weather):
        print(f"Path doesnt exist {path_to_weather}")

    data.to_csv(path_to_weather, mode='a', header=False, index=False)

    print("Weather data fetched and saved to data/raw")
    
if __name__ == "__main__":
    fetch_weather()



def get_forecast_data():
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)


    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "precipitation", "rain", "snowfall", "snow_depth", "is_day"],
        "forecast_hours": 6,
    }


    responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    response = responses[0]

    # Extract hourly data
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
    hourly_precipitation_probability = hourly.Variables(4).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(5).ValuesAsNumpy()
    hourly_rain = hourly.Variables(6).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(7).ValuesAsNumpy()
    hourly_snow_depth = hourly.Variables(8).ValuesAsNumpy()
    hourly_is_day = hourly.Variables(9).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["dew_point_2m"] = hourly_dew_point_2m
    hourly_data["apparent_temperature"] = hourly_apparent_temperature
    hourly_data["precipitation_probability"] = hourly_precipitation_probability
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["rain"] = hourly_rain
    hourly_data["snowfall"] = hourly_snowfall
    hourly_data["snow_depth"] = hourly_snow_depth
    hourly_data["is_day"] = hourly_is_day

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    return hourly_dataframe

