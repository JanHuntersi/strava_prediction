
from src.database.connector import get_yesterdays_predictions,get_all_predictions
from definitions import PATH_TO_KUDOS_FULL_DATASET,PATH_TO_PROCESSED_IS_ACTIVE
import pandas as pd
from datetime import datetime
from src.models.mlflow_helper import MlflowHelper
import mlflow
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, explained_variance_score

def kudos_find_true_predict(predictions, kudos_full_dataset):

    # Get the kudos_id from the prediction
    kudos_id = predictions["kudos_id"]

    # Get the true value from the kudos_full_dataset
    true_kudos = kudos_full_dataset[kudos_full_dataset["id"] == kudos_id]["kudos_count"].values[0]

    # Get the predicted value from the prediction
    predicted_kudos = predictions["kudos_prediction"]

    print(f"True kudos: {true_kudos}, Predicted kudos: {predicted_kudos}")

    return true_kudos, predicted_kudos
    
def evaluate_kudos():
    print("Trying to evaluate kudos predictions...")

    mlflow_helper = MlflowHelper()
    experiment_name="kudos_evaluation"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="kudos_eval"):


        kudos_df = pd.read_csv(PATH_TO_KUDOS_FULL_DATASET)

        # get predictions from MongoDB
        predictions = get_yesterdays_predictions("kudos_predictions")

        if predictions is None:
            print("No predictions yesterday!")
            return

        # Convert cursor to list to process the predictions
        predictions = list(predictions)

        if not predictions:
            print("No predictions yesterday!")
            return

        print(f"Found {len(predictions)} predictions")

        true_kudos_list = []
        predicted_kudos_list = []

        # Process each prediction
        for prediction in predictions:
            # Perform your evaluation logic here
            print(f"Evaluating prediction: {prediction}")
            true, predicted = kudos_find_true_predict(prediction, kudos_df)
            true_kudos_list.append(true)
            predicted_kudos_list.append(predicted)

        # Calculate the evaluation metric
        # For example, Mean Absolute Error (MAE)
        mae = mean_absolute_error(true_kudos_list, predicted_kudos_list)
        mse = mean_squared_error(true_kudos_list, predicted_kudos_list)
        evs = explained_variance_score(true_kudos_list, predicted_kudos_list)
        print(f"Mean Absolute Error: {mae}") 
        print(f"Mean Squared Error: {mse}")
        print(f"Explained Variance Score: {evs}")
        print("Kudos evaluation done!")

        #save to mlflow
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("evs", evs)
    #end run
    mlflow.end_run()

def get_active_values(all_predict_list, df):
    predictions = []
    actual_values = []

    for prediction in all_predict_list:
        date = prediction[1][1]  # Assuming the date is the second element of the tuple
        is_active_prediction = prediction[0][1]

        # Find the corresponding row in the DataFrame based on the date
        df_row = df[df['date'] == date]

        if not df_row.empty:
            actual_value = df_row.iloc[0]['is_active']
            predictions.append(is_active_prediction)
            actual_values.append(actual_value)

    return predictions, actual_values

def evaluate_is_active():
    print("Trying to evaluate is_active predictions...")

    mlflow_helper = MlflowHelper()
    experiment_name = "is_active_evaluation"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="is_active_eval"):

        df = pd.read_csv(PATH_TO_PROCESSED_IS_ACTIVE)
        df = df.tail(48)

        # Convert DataFrame date column to datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Remove duplicates based on the 'date' column
        df.drop_duplicates(subset=['date'], inplace=True)

        # get predictions from MongoDB
        predictions = get_yesterdays_predictions("activities_predictions")

        if predictions is None:
            print("No predictions yesterday!")
            return

        # Create an empty list to store all predictions
        all_predictions = []

        # Iterate through each prediction in predictions
        for prediction in predictions:
            pred_list = prediction["predictions"]
            # Extend the all_predictions list with the pred_list
            all_predictions.extend(pred_list)

        # Create a DataFrame from all_predictions
        df_predictions = pd.DataFrame(all_predictions)

        # Drop duplicate rows based on the 'date' column
        df_predictions.drop_duplicates(subset=['date'], inplace=True)

        if df_predictions.empty:
            print("No predictions yesterday!")
            return

        print(f"Found {len(df_predictions)} predictions")

        predicted_values = []
        actual_values = []

        print(df['date'])

        # Remove +00:00 from date
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        for prediction in df_predictions.to_dict('records'):
            date = prediction["date"]

            # Ensure that date is a string
            if isinstance(date, datetime):
                date_str = date.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                date_str = date

            # Parse the date string to match the format in DataFrame
            date_parsed = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

            is_active_prediction = prediction["prediction"]

            # Find the corresponding row in the DataFrame based on the date
            df_row = df[df['date'] == date_parsed]

            print(date_parsed)

            if not df_row.empty:
                actual_value = df_row.iloc[0]['is_active']
                predicted_values.append(is_active_prediction)
                actual_values.append(actual_value)
                print(f"Date: {date}, Predicted: {is_active_prediction}, Actual: {actual_value}")

        print("Predicted Values:", predicted_values)
        print("Actual Values:", actual_values)

        # Calculate evaluation metrics
        accuracy = accuracy_score(actual_values, predicted_values)
        balanced_accuracy = balanced_accuracy_score(actual_values, predicted_values)
        mcc = matthews_corrcoef(actual_values, predicted_values)
        cohen_kappa = cohen_kappa_score(actual_values, predicted_values)

        print(f"Accuracy: {accuracy}")
        print(f"Balanced Accuracy: {balanced_accuracy}")
        print(f"MCC: {mcc}")
        print(f"Cohen's Kappa: {cohen_kappa}")

        # Log metrics to mlflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("balanced_accuracy", balanced_accuracy)
        mlflow.log_metric("mcc", mcc)
        mlflow.log_metric("cohen_kappa", cohen_kappa)

        # End the mlflow run
        mlflow.end_run()

    print("Is_active evaluation done!")


def evaluate_predictions():

    evaluate_kudos()

    evaluate_is_active()


if __name__ == "__main__":
    evaluate_predictions()
