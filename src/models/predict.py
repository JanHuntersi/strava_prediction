
from datetime import timedelta
import pandas as pd
import numpy as np
import onnxruntime
from src.models.model_helper import ModelHelper
from src.models.mlflow_helper import MlflowHelper
from src.data.fetch_weather import get_forecast_data
from definitions import PATH_TO_KUDOS_DATASET, PATH_TO_PROCESSED_IS_ACTIVE
from datetime import datetime


WINDOW_SIZE = 8

def kudos_prediction(data, models_dict):

    mlflow_helper = MlflowHelper()
    mhelper = ModelHelper(mlflow_helper)

    print("getting model and pipeline...")

    # Load the production model and pipeline from MLflow
    model, pipeline = models_dict["kudos_model"], models_dict["kudos_pipeline"]

    #prepare data
    X_predict, Y_predict, pipeline = mhelper.prepare_data_kudos(pipeline,data)

    X_predict = X_predict.astype(np.float32)

    model = onnxruntime.InferenceSession(model.SerializeToString())

    # make prediction
    prediction = model.run(["variable"], {"float_input": X_predict})[0]

    print("Prediction shape:", prediction.shape)
    print("Prediction:", prediction)

    return int(prediction[0][0])



def activity_prediction(data, models_dict):

    mlflow_helper = MlflowHelper()
    mhelper = ModelHelper(mlflow_helper)

    print("getting model and pipeline...")

    # Load the production model and pipeline from MLflow
    model, pipeline = models_dict["is_active_model"], models_dict["is_active_pipeline"]

    print("Preparing data")

    # Prepare test data
    X_predict, Y_predict = mhelper.prepare_data_processed(pipeline,data)

    # Evaluate the staging and production models
    model = onnxruntime.InferenceSession(model.SerializeToString())


    Y_predict = Y_predict.values.reshape(-1, 1)  # Convert Series to numpy array
    
    multi_array_scaled = np.column_stack([Y_predict, X_predict])

    # get the last 8 hours
    multi_array_scaled = multi_array_scaled[-WINDOW_SIZE:]



    multi_array_scaled = multi_array_scaled.reshape(1, multi_array_scaled.shape[0], multi_array_scaled.shape[1])

    print("trying to predict...")

    # make prediction
    prediction = model.run(["output"], {"input": multi_array_scaled})[0]

    print("Prediction shape:", prediction.shape)
    print("PREDICTION VALUE", prediction)

    # turn to binary
    prediction = np.where(prediction > 0.4, 1, 0)

    print("Prediction:", prediction)

    return prediction

    

def predict_activities(models_dict):
    # Load the processed data
    data = pd.read_csv(PATH_TO_PROCESSED_IS_ACTIVE)
    print("testset")

    # Convert the 'date' column to datetime if it's not already converted
    if not pd.api.types.is_datetime64tz_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    # Get the current time in the same timezone as the data (assuming it's UTC)
    current_time = pd.Timestamp.now(tz='UTC').floor('h')

    # Filter the DataFrame to include only rows until the current time
    data_until_now = data[data["date"] <= current_time]

    data_after_now = data[data["date"] > current_time]

    print("Data after now:")
    print(data_after_now.head(5))

    if data_after_now.empty:
        print("No data after now")
        return

    num_predictions = 6

    if data_after_now.shape[0] < num_predictions:
        print(f"Less than {num_predictions} predictions available. Only {data_after_now.shape[0]} ")
        num_predictions = data_after_now.shape[0]

    predictions = []
    prediction_object = {"predictions": []}

    for i in range(num_predictions):
        try:
            print(f"Predicting {i}")
            prediction = activity_prediction(data_until_now.copy(),models_dict)
            predictions.append(int(prediction[0][0]))

            # append i row from date_after_now to data_until_now
            print(f"Prediction {i}: {prediction[0][0]}")

            bool_val = False
            if prediction[0][0] > 0.5:
                bool_val = True


            # set prediction to is_active
            data_after_now.iloc[i]['is_active'] = bool_val

    # append prediction, and time to prediction object
            prediction_object["predictions"].append({
                "prediction": bool_val,
                "date": data_after_now.iloc[i]['date']
            })

            # Concatenate the current row with the historical data
            data_until_now = pd.concat([data_until_now, data_after_now.iloc[[i]]], ignore_index=True)

            # Check if there are any NaN values in the data
            if data_until_now.isna().any().any():
                print("NaN values found in data_until_now:")
                print(data_until_now.isna().sum())
                return

        except Exception as e:
            print(f"Error in prediction {i}: {e}")
            return
    
    return predictions, prediction_object


def predict_last_kudos(models_dict):
    #read processed kudos
    data = pd.read_csv(PATH_TO_KUDOS_DATASET)

    #only last row with header
    data = data.tail(1)

    pred = kudos_prediction(data, models_dict)

    # Convert start_date_local to datetime or string
    start_date_local = data['start_date_local'].iloc[0]
    if isinstance(start_date_local, pd.Timestamp):
        start_date_local = start_date_local.to_pydatetime()
    elif not isinstance(start_date_local, (datetime, str)):
        start_date_local = str(start_date_local)

    prediction_object = {
        "kudos_prediction": int(pred),
        "date": start_date_local,
        "kudos_id": int(data['id'].iloc[0]),
        "is_evaluated": False,
        "actual_value": int(data['kudos_count'].iloc[0])
    }

    # Prediction object
    print("Prediction object:", prediction_object)

    return pred, prediction_object
    

if __name__ == "__main__":

    #predict_activities()

    #read processed kudos
    data = pd.read_csv(PATH_TO_KUDOS_DATASET)

    #only last row with header
    data = data.tail(1)
    print("Kudos data", data.head())
    pred = kudos_prediction(data)
