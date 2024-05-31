
import pandas as pd
from definitions import PATH_TO_PROCESSED_IS_ACTIVE
from src.models.predict import activity_prediction

def predict_activities():
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

    num_predictions = 3

    if data_after_now.shape[0] < num_predictions:
        print(f"Less than {num_predictions} predictions available. Only {data_after_now.shape[0]} ")
        num_predictions = data_after_now.shape[0]

    predictions = []

    for i in range(num_predictions):
        try:
            print(f"Predicting {i}")
            prediction = activity_prediction(data_until_now.copy())
            predictions.append(int(prediction[0][0]))

            # append i row from date_after_now to data_until_now
            print(f"Prediction {i}: {prediction[0][0]}")

            # set prediction to is_active
            data_after_now.iloc[i, data_after_now.columns.get_loc("is_active")] = prediction[0][0]

            # Concatenate the current row with the historical data
            data_until_now = pd.concat([data_until_now, data_after_now.iloc[[i]]], ignore_index=True)

            # Print the last 4 rows for debugging
            print("LAST 4 ROWS")
            print(data_until_now.tail(4))

            # Keep only the last 8 rows for the next prediction
            if data_until_now.shape[0] > 8:
                data_until_now = data_until_now.iloc[-8:]
            
            # Check if there are any NaN values in the data
            if data_until_now.isna().any().any():
                print("NaN values found in data_until_now:")
                print(data_until_now.isna().sum())
                return

        except Exception as e:
            print(f"Error in prediction {i}: {e}")
            return
    print("Predictions:", predictions)
    print("testing ended")

if __name__ == "__main__":
    predict_activities()


  


    

if __name__ == "__main__":
    predict_activities()
