from flask import Flask, jsonify
from flask_cors import CORS
from src.models.predict import activity_prediction, predict_activities, predict_last_kudos
from src.database.connector import save_predictions
import os
app = Flask(__name__)
CORS(app)

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
    kudos_pred, prediction_object = predict_last_kudos()
    print("Kudos prediction: ", kudos_pred)
    print("Prediction object: ", prediction_object)

    # add prediction to MongoDB
    save_predictions("kudos_predictions", prediction_object)

    return jsonify("prediction: ", kudos_pred)

@app.route('/predicted/kudos')
def predicted_kudos():
    print("Return kudos that were predicted...")

@app.route('/activities')
def activities():
    print("Predicting activities...")
    predictions = predict_activities()


    # add predictions to MongoDB

    print("my Predictions:", predictions)

    return_pred = {"predictions": predictions}
    return jsonify(return_pred) 

def main():
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
