from definitions import PATH_TO_PREPROCESS_STRAVA, PATH_TO_PREPROCESS_WEATHER, PATH_TO_PROCESSED_IS_ACTIVE,PATH_TO_KUDOS_DATASET, PATH_TO_KUDOS_FULL_DATASET
import math
import pandas as pd
from datetime import datetime, timedelta, timezone

def compare_dates(date1, date2):
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)

    date1 = datetime.strptime(date1, '%Y-%m-%d')
    date2 = datetime.strptime(date2, '%Y-%m-%d')

    difference = abs((date1 - date2).days)

    if difference >= 2:
        return True
    else:
        return False

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

    if strava_data.empty or weather_data.empty:
        print("No data to merge, exiting")
        return


    weather_data['date'] = pd.to_datetime(weather_data['date'])

    # merge the two dataframes
    weather_data['is_active'] = False

    strava_data.apply(lambda row: mark_active_hours(row, weather_data), axis=1)
        
    # SAVE TO PROCESSED DATA

    print("Merged completed. Saving data to processed folder")
    weather_data.to_csv(PATH_TO_PROCESSED_IS_ACTIVE, index=False)

    print("Data saved")

    last_five_activities = strava_data.tail(10)

    now = datetime.now(timezone.utc)
    day = timedelta(days=1)
    yesterday = now - day
    print("Yesterday: ", yesterday)

    # Processing kudos dataset
    kudos_dataset = pd.read_csv(PATH_TO_KUDOS_DATASET)
    for index, row in last_five_activities.iterrows():
        row_id = row['id']
        if row_id in kudos_dataset['id'].values:
            kudos_dataset.loc[kudos_dataset['id'] == row_id, :] = row.values
        else:
            row['start_date_local'] = pd.to_datetime(row['start_date_local'])
            if row['start_date_local'] <= yesterday:
                print("Activity is older than 1 day and hasn't been added to dataset... \nAdding activity to kudos dataset!")
                kudos_dataset = pd.concat([kudos_dataset, row.to_frame().T], ignore_index=True)

    kudos_dataset.to_csv(PATH_TO_KUDOS_DATASET, header=True, index=False)
    print("Updated kudos_dataset.csv")

    # Processing full kudos dataset
    kudos_full_dataset = pd.read_csv(PATH_TO_KUDOS_FULL_DATASET)
    for index, row in last_five_activities.iterrows():
        row_id = row['id']
        if row_id in kudos_full_dataset['id'].values:
            kudos_full_dataset.loc[kudos_full_dataset['id'] == row_id, :] = row.values
        else:
            print("Adding activity to kudos_full_dataset")
            kudos_full_dataset = pd.concat([kudos_full_dataset, row.to_frame().T], ignore_index=True)

    kudos_full_dataset.to_csv(PATH_TO_KUDOS_FULL_DATASET, header=True, index=False)
    print("Updated kudos_full_dataset.csv")



if __name__ == "__main__":
    merge_data()
