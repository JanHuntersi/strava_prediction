from definitions import STRAVA_ACTIVITY_URL,PATH_TO_RAW_STRAVA
from src.common.refresh_token import get_and_refresh_token
import requests
from datetime import datetime
import json

def get_activities_from_strava():
    """
    Fetches  todays activities from Strava API
    """

    print("Fetching Strava data")

    # Get access token
    access_token = get_and_refresh_token()

    if access_token is None:
        print("Error, received None as access token")
        return None
    

    headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
    }

    # Pridobimo trenutni čas
    now = datetime.now()

    # Nastavimo uro na 3 zjutraj
    start_of_day = datetime(now.year, now.month, now.day, 3, 0, 0)

    # Pretvorimo v epohalni časovni žig
    epoch_timestamp = int(start_of_day.timestamp())

    print(f"Fetching activities after {start_of_day} with epoch timestamp {epoch_timestamp}")

    
    req = requests.get(f"{STRAVA_ACTIVITY_URL}after={epoch_timestamp}", headers=headers)
    
    # Check if request was successful
    if req.status_code != 200:
        print(f"Error when fetching activities, received status code {req.status_code}")
        return None
    
    return req.text
def fetch_strava():
    activities = get_activities_from_strava()

    print(f"Activities: {activities}")
    activities_json=0
    
    try:
        activities_json = json.loads(activities)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None
    
    if activities is None:
        print("Error when fetching activities")
        return None
    

    if len(activities_json) == 0:
        print("No activities found, saving [] to file raw strava")
    else:
        print(f"Activities found, saving {len(activities_json)} activities to file raw strava")

    with open(PATH_TO_RAW_STRAVA, "w") as f:
        f.write(activities)
    print("Activities were saved.")

    return len(activities_json)


if __name__ == "__main__":
    fetch_strava()
