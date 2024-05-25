from src.data.weather import get_weather_data
from definitions import ROOT_DIR
import os
import pandas as pd


path_to_weather = os.path.join(ROOT_DIR, "data", "raw", "weather.csv") 

def fetch_weather():
    
    data = get_weather_data()

    if not os.path.exists(path_to_weather):
        print(f"Path doesnt exist {path_to_weather}")

    data.to_csv(path_to_weather, mode='a', header=False, index=False)

    print("Weather data fetched and saved to data/raw")
    
if __name__ == "__main__":
    fetch_weather()

