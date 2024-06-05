import os
import json
import pandas as pd
import math
from datetime import datetime, timedelta
from definitions import PATH_TO_RAW_STRAVA, PATH_TO_PREPROCESS_STRAVA

def update_date(df):
    # Round start_date_local and end_time down 
    df['start_date_local'] = pd.to_datetime(df['start_date_local'])
    df['start_date_local'] = df['start_date_local'].dt.floor('h')

    df['end_time'] = pd.to_datetime(df['end_time'])
    df['end_time'] = df['end_time'].dt.floor('h')

    # Calculate duration in hours by elapsed time and round it upwards
    df['duration'] = df['elapsed_time'] / 3600 
    df['duration'] = df['duration'].apply(lambda x: math.ceil(x))

    return df

def add_end_date_time(row):
    start_time = row[6]  # Assuming 'start_date_local' is at index 6
    elapsed_time = row[4]  # Assuming 'elapsed_time' is at index 4

    start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
    elapsed_time = timedelta(seconds=elapsed_time)

    end_time = start_time + elapsed_time

    end_date_local = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    row.append(end_date_local)  # Append end_date_local to the row

    return row

def preprocess_strava():
    print("Preprocessing Strava data")
    with open(PATH_TO_RAW_STRAVA, "r") as f:
        activities = f.read()

    data = json.loads(activities)

    print(f"Number of activities: {len(data)}")

    if len(data) == 0:
        print("No activities found, exiting")
        return
    else:
        print(f"Activities found, preprocessing {len(data)} activities")
    
    columns = ['id', 'type', 'name', 'distance', 'elapsed_time', 'moving_time', 'start_date_local', 'achievement_count', 'kudos_count', 'comment_count', 'athlete_count', 'photo_count']

    csv_data = []

    for activity in data:
        row = []
        for column in columns:
            row.append(activity[column])
        row = add_end_date_time(row)  # Add end_date_local to the row

        csv_data.append(row)

    # Add 'end_time' to the column list
    columns.append('end_time')

    # Convert csv_data to DataFrame
    new_data_df = pd.DataFrame(csv_data, columns=columns)
    new_data_df = update_date(new_data_df)

    # Sort DataFrame by 'start_date_local'
    new_data_df = new_data_df.sort_values(by='start_date_local')

    # Read existing preprocessed Strava data if it exists
    if os.path.exists(PATH_TO_PREPROCESS_STRAVA):
        existing_data_df = pd.read_csv(PATH_TO_PREPROCESS_STRAVA)
        existing_data_df = update_date(existing_data_df)

        # Merge new data with existing data, updating kudos_count
        merged_df = pd.concat([existing_data_df, new_data_df]).drop_duplicates(subset=['id'], keep='last')
    else:
        print("strava_data.csv doesn't exist, creating new")
        merged_df = new_data_df

    # Save the merged DataFrame back to CSV


    merged_df.to_csv(PATH_TO_PREPROCESS_STRAVA, index=False)



if __name__ == "__main__":
    preprocess_strava()
