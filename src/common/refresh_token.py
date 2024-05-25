import src.settings as settings
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from src.database.connector import get_client, update_token
import src.settings as settings
from datetime import datetime
import requests


#MONGO access_token collection looks like
"""
{"$oid":"id"},
"access_token":"xxxxx",
"expires_at":"1716584049",
"expires_in":"5027",
"refresh_token":"xxxxx",
"name":"access_token"}
"""


def get_refresh_token():
    """
    This function is used to get the refresh token from MongoDB.
    """
    client = get_client()
    try:
        collection = client.get_database(settings.strava_db_name).get_collection(settings.strava_collection_name)
        token_obj = collection.find_one({"name": "access_token"})
        return token_obj
    except Exception as e:
        print(f"Error getting refresh token: {e}")
        return None
    
def check_token_expiry(token_obj):
    """
    This function is used to check if the token has expired.
    """

    current_time = datetime.now()
    current_time_unix = int(current_time.timestamp())
    expiry_time = int(token_obj["expires_at"])

    # Convert Unix timestamps to readable datetime format
    current_time_readable = datetime.fromtimestamp(current_time_unix)
    expiry_time_readable = datetime.fromtimestamp(expiry_time)


    print(f"Current time: {current_time_unix} => {current_time_readable}")
    print(f"Expiry time: {expiry_time} => {expiry_time_readable}")

    if current_time_unix > expiry_time:
        return True
    else:
        return False
    
def refresh_token(token_obj):
    """
    This function is used to refresh the token.
    """

    print("Refreshing token...")
    
    # Call the refresh token endpoint
    req = requests.post("https://www.strava.com/oauth/token", data={
        "client_id": settings.strava_client_id,
        "client_secret": settings.strava_client_secret,
        "refresh_token": token_obj["refresh_token"],
        "grant_type": "refresh_token"
    })

    if req.status_code == 200:
        print("Token refreshed")
        return req.json()
    else:
        print(f"Error refreshing token: {req.text}")
        return None
    




def get_and_refresh_token():
    """
    This function is used to get the refresh token from MongoDB and refresh it.
    """
    token_obj = get_refresh_token()
    
    if token_obj is None:
        print("Token object is None")
        return None
    
    if check_token_expiry(token_obj):
        print("Token has expired, will fetch a new one...")
        # Refresh the token
        
        new_token = refresh_token(token_obj)

        if new_token is None:
            print("Error refreshing token")
            return None
        
        else:
            print("Token has been refreshed... updating the database...")

            # add name to new_token
            new_token["name"] = "access_token"

            #update the database
            update_token(new_token)

            # return the access token
            return new_token["access_token"]

    else:
        print("Token has not expired")

        return token_obj["access_token"]
    


def main():
    token = get_and_refresh_token()
    print(f"TOKEN IS {token}")


if __name__ == "__main__":
    main()


