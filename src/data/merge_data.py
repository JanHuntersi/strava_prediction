from definitions import PATH_TO_PREPROCESS_STRAVA, PATH_TO_PREPROCESS_WEATHER, PATH_TO_PROCESSED_IS_ACTIVE,PATH_TO_KUDOS_DATASET, PATH_TO_KUDOS_FULL_DATASET
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

    #Process data for kudos_dataset and full_dataset
    last_five_activities = strava_data.tail(10)


    #KUDOS DATASET only contains activities that are older than 2 days
    kudos_dataset = pd.read_csv(PATH_TO_KUDOS_DATASET)
    for index, row in last_five_activities.iterrows():
        if row['id'] not in kudos_dataset['id'].values:
            # check if activity is older than 2 days
            if row['start_date_local'] < pd.Timestamp.now() - timedelta(days=2):
                print("Activity is older than 2 days and hasnt been added to dataset... \n Adding activity to kudos dataset!")
                kudos_dataset = kudos_dataset._append(row)
    kudos_dataset.to_csv(PATH_TO_KUDOS_DATASET, index=False)
    print("Updated kudos_dataset.csv")


    # PROCESSING FULL KUDOS DATASET
    kudos_full_dataset = pd.read_csv(PATH_TO_KUDOS_FULL_DATASET)
    #if activiities not in kudos dataset then add them
    for index, row in last_five_activities.iterrows():
        if row['id'] not in kudos_full_dataset['id'].values:
            print("Adding activity to kudos_full_dataset")
            kudos_full_dataset = kudos_full_dataset._append(row)
    kudos_full_dataset.to_csv(PATH_TO_KUDOS_DATASET, index=False)
    print("Updated kudus_full_dataset.csv")




if __name__ == "__main__":
    merge_data()
