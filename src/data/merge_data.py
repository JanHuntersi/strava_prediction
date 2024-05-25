from definitions import PATH_TO_PREPROCESS_STRAVA, PATH_TO_PREPROCESS_WEATHER, PATH_TO_PROCESSED_IS_ACTIVE
import math
import pandas as pd
from datetime import timedelta



def preprocess_strava(df):

    # round start_date_local and for end_time down 
    df['start_date_local'] = pd.to_datetime(df['start_date_local'])

    df['start_date_local'] = df['start_date_local'].dt.floor('h')

    df['end_time'] = pd.to_datetime(df['end_time'])
    df['end_time'] = df['end_time'].dt.floor('h')

    # calculate duration in hours by elapsed time and round it upwards
    df['duration'] = df['elapsed_time'] / 3600 
    df['duration'] = df['duration'].apply(lambda x: math.ceil(x))
    
    return df


#Funkcija za označevanje aktivnih ur na podlagi trajanja
def mark_active_hours(row, weather_df):
    start_time = row['start_date_local']
    duration = row['duration']
    
    # Poiščemo indeks začetnega časa v weather_df
    try:
        start_index = weather_df[weather_df['date'] == start_time].index[0]
    except IndexError:
        raise ValueError(f"Start time {start_time} not found in weather data.")
    
    # Končni indeks, ki ni vključen
    end_index = start_index + duration - 1
    
    # Posodobimo is_active stolpec
    weather_df.loc[start_index:end_index, 'is_active'] = True
def merge_data():

    strava_data = pd.read_csv(PATH_TO_PREPROCESS_STRAVA)
    weather_data = pd.read_csv(PATH_TO_PREPROCESS_WEATHER)


    weather_data['date'] = pd.to_datetime(weather_data['date'])

    strava_data = preprocess_strava(strava_data)

    # merge the two dataframes
    weather_data['is_active'] = False

    strava_data.apply(lambda row: mark_active_hours(row, weather_data), axis=1)
        
    # SAVE TO PROCESSED DATA

    weather_data.to_csv(PATH_TO_PROCESSED_IS_ACTIVE, index=False)


if __name__ == "__main__":
    merge_data()
