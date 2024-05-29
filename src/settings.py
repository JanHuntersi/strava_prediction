import os
from dotenv import load_dotenv

load_dotenv()


mongodb_connection = os.getenv("MONGODB_CONNECTION")
strava_db_name = os.getenv("STRAVA_DB_NAME")
strava_collection_name = os.getenv("STRAVA_COLLECTION_NAME")
strava_client_id = os.getenv("STRAVA_CLIENT_ID")
strava_client_secret = os.getenv("STRAVA_CLIENT_SECRET")
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
dagshub_repo_name = os.getenv("DAGSHUB_REPO_NAME")


print(f"mongodb_connection: {mongodb_connection}")
print(f"strava_db_name: {strava_db_name}")
print(f"strava_collection_name: {strava_collection_name}")
print(f"strava_client_id: {strava_client_id}")
print(f"strava_client_secret: {strava_client_secret}")
print(f"mlflow_tracking_uri: {mlflow_tracking_uri}")
print(f"mlflow_tracking_username: {mlflow_tracking_username}")
print(f"mlflow_tracking_password: {mlflow_tracking_password}")
print(f"dagshub_repo_name: {dagshub_repo_name}")
