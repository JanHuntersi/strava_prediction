import os
import src.settings as settings
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta


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

def save_kudos_predictions(collection_name, predictions):
    client = get_client()
    try:
        print(f"Trying to save {collection_name}  to mongodb")
        db = client.get_database(settings.strava_db_name)
        collection = db.get_collection(collection_name)

        #check if existing document already exists

        existing_document = collection.find_one({"kudos_id": predictions["kudos_id"]})
        if existing_document:
            print(f"Document with kudos_id {predictions['kudos_id']} already exists. Updating...")

        else:
            # add updated_at timestamp
            predictions["updated_at"] = datetime.now()
            collection.insert_one(predictions)
            print(f"{collection_name} saved to mongodb successfully!")
    except Exception as e:
        print(f"Error saving predictions: {e}")

def save_activities_predictions(collection_name, predictions):
    client = get_client()
    try:
        print(f"Trying to save {collection_name}  to mongodb")
        db = client.get_database(settings.strava_db_name)
        collection = db.get_collection(collection_name)
        collection.insert_one(predictions)
        print(f"{collection_name} saved to mongodb successfully!")
    except Exception as e:
        print(f"Error saving predictions: {e}")

def get_all_predictions(collection_name):
    client = get_client()
    try:
        print(f"Trying to get {collection_name} from MongoDB")
        db = client.get_database(settings.strava_db_name)
        collection = db.get_collection(collection_name)
        predictions = collection.find()
        return predictions
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return None


def get_yesterdays_predictions(collection_name):
    client = get_client()
    try:
        print(f"Trying to get {collection_name} from MongoDB")
        db = client.get_database(settings.strava_db_name)
        collection = db.get_collection(collection_name)
        
        # Get yesterday's date and start and end of day
        yesterday = datetime.now() - timedelta(days=1)
        start_of_day = datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0, 0)
        end_of_day = start_of_day + timedelta(days=1)
        
        # Query for documents updated on the previous day
        predictions = collection.find({
            "updated_at": {"$gte": start_of_day, "$lt": end_of_day}
        })
        return predictions
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return None
