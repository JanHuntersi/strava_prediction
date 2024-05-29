import os
import src.settings as settings
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime


uri = settings.mongodb_connection

def get_client():
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None


def update_token(new_collection):
    client = get_client()
    try:
        db = client.get_database(settings.strava_db_name)
        collection = db.get_collection(settings.strava_collection_name)
        
        collection.delete_one({"name": "access_token"})
        collection.insert_one(new_collection)
    except Exception as e:
        print(f"Error updating token: {e}")

def save_production_metrics(collection_name, metrics):
    client = get_client()
    try:
        print(f"Trying to save {collection_name}  to mongodb")
        db = client.get_database(settings.strava_db_name)
        collection = db.get_collection(collection_name)
        collection.insert_one(metrics)
        print(f"{collection_name} saved to mongodb successfully!")
    except Exception as e:
        print(f"Error saving production metrics: {e}")
