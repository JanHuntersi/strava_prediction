import os

ROOT_DIR =os.path.dirname(os.path.abspath(__file__))


STRAVA_ACTIVITY_URL = "https://www.strava.com/api/v3/athlete/activities?"

PATH_TO_RAW_STRAVA = os.path.join(ROOT_DIR, "data", "raw", "strava_data.json")

PATH_TO_PREPROCESS_STRAVA = os.path.join(ROOT_DIR, "data", "preprocess", "strava_data.csv")

PATH_TO_PREPROCESS_WEATHER = os.path.join(ROOT_DIR, "data", "preprocess", "weather.csv")

PATH_TO_RAW_WEATHER = os.path.join(ROOT_DIR, "data", "raw", "weather.csv")

PATH_TO_PROCESSED_IS_ACTIVE = os.path.join(ROOT_DIR, "data", "processed", "is_active.csv")

PATH_TO_KUDOS_DATASET = os.path.join(ROOT_DIR, "data", "processed", "kudos_dataset.csv")

PATH_TO_TEST_TRAIN = os.path.join(ROOT_DIR, "data", "test_train")

PATH_TO_CURRENT_REFERENCE = os.path.join(ROOT_DIR, "data", "current_reference")

PATH_TO_REPORTS_EVIDENTLY = os.path.join(ROOT_DIR, "reports", "evidently")

PATH_TO_REPORTS = os.path.join(ROOT_DIR, "reports")
