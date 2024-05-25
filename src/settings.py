import os
from dotenv import load_dotenv

load_dotenv()


mongodb_connection = os.getenv("MONGODB_CONNECTION")
strava_db_name = os.getenv("STRAVA_DB_NAME")
strava_collection_name = os.getenv("STRAVA_COLLECTION_NAME")
strava_client_id = os.getenv("STRAVA_CLIENT_ID")
strava_client_secret = os.getenv("STRAVA_CLIENT_SECRET")


print(f"mongodb_connection: {mongodb_connection}")
print(f"strava_db_name: {strava_db_name}")
print(f"strava_collection_name: {strava_collection_name}")
print(f"strava_client_id: {strava_client_id}")
print(f"strava_client_secret: {strava_client_secret}")
