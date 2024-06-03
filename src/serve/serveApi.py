from flask import Flask, jsonify
from flask_cors import CORS
from src.models.mlflow_helper import MlflowHelper
from src.models.predict import activity_prediction, predict_activities, predict_last_kudos
from src.database.connector import save_kudos_predictions,save_activities_predictions
import os


models_dict={}

def download_models():
    mlflow_helper = MlflowHelper()
    print("Trying to download models and pipelines...")

    # Load the production model and pipeline from MLflow
    is_active_model, is_active_pipeline = mlflow_helper.get_model_pipeline("is_active_model", "is_active_pipeline", "production")

    # Load the production model and pipeline from MLflow
    kudos_model, kudos_pipeline = mlflow_helper.get_model_pipeline("kudos_model", "kudos_pipeline", "production")

    # add to models_dict
    models_dict["is_active_model"] = is_active_model
    models_dict["is_active_pipeline"] = is_active_pipeline
    models_dict["kudos_model"] = kudos_model
    models_dict["kudos_pipeline"] = kudos_pipeline
    print("Models downloaded successfully")


app = Flask(__name__)
CORS(app)

download_models()

# Function to ensure Git repository is updated
def pull_git_repo():
    try:
        # Run the Git pull command
        os.system('git pull')
        print("Git repository pulled successfully.")
    except Exception as e:
        print(f"Error pulling Git repository: {e}")
        return False
    return True

# Function to ensure DVC data is pulled
def pull_dvc_data():
    try:
        # Run the DVC pull command
        os.system('dvc pull')
        print("DVC data pulled successfully.")
    except Exception as e:
        print(f"Error pulling DVC data: {e}")
        return False
    return True

@app.route('/pull-data')
def pull_data():
    print("Pulling Git repository...")
    pull_git_repo()
    print("Pulling DVC data...")
    pull_dvc_data()
    return jsonify("DVC data pulled successfully.")

@app.route('/')
def root():
    return jsonify({'message': 'Hello World!'})

@app.route('/kudos')
def make_kudos_prediction():
    print("Trying to predict last kudos")
    kudos_pred, prediction_object = predict_last_kudos(models_dict)
    print("Kudos prediction: ", kudos_pred)
    print("Prediction object: ", prediction_object)

    # add prediction to MongoDB
    save_kudos_predictions("kudos_predictions", prediction_object)

    return jsonify("prediction: ", kudos_pred)

@app.route('/predicted/kudos')
def predicted_kudos():
    print("Return kudos that were predicted...")

@app.route('/activities')
def activities():
    print("Predicting activities...")
    predictions, prediction_object = predict_activities(models_dict)

    print("Prediction object is:", prediction_object)

    # add predictions to MongoDB
    save_activities_predictions("activities_predictions", prediction_object)

    print("my Predictions:", predictions)

    return_pred = {"predictions": predictions}
    return jsonify(return_pred) 


def main():
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
