import onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.models.model_helper import ModelHelper
from src.models.mlflow_helper import MlflowHelper
import mlflow
import os
import numpy as np

def train_is_active():

    mlflow_helper = MlflowHelper()
    mhelper = ModelHelper(mlflow_helper)
    print("Train is_active method called!")

    # Start MLflow run
    experiment_name = "is_active_prediction"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="is_active_train"):

        #autolog
        mlflow.tensorflow.autolog()

        X_train, y_train, X_test, y_test, pipeline = mhelper.prepare_train_test_active()

        y_train = y_train.values.reshape(-1, 1)  # Convert Series to numpy array
        y_test = y_test.values.reshape(-1, 1)  # Convert Series to numpy array

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)

        data_train = np.hstack((y_train,X_train))
        data_test = np.hstack((y_test,X_test))

        # Reshape only X_train
        X_train, y_train = mhelper.reshape_for_model(data_train)
        X_test, y_test = mhelper.reshape_for_model(data_test)

        print("Unique values in y_train before :", np.unique(y_train))
        print("Unique values in y_test before :", np.unique(y_test))

        # Check shapes
        print("X_train shape after reshape:", X_train.shape)
        print("y_train shape after reshape:", y_train.shape)
        print("X_test shape after reshape:", X_test.shape)
        print("y_test shape after reshape:", y_test.shape)

        
        input_shape = (X_train.shape[1], X_train.shape[2])
        print("Input shape for GRU model:", input_shape)
        

        # Build and compile the GRU model
        model = mhelper.create_and_compile_quantisized_model(input_shape, 2)

        # Train the model
        model.fit(X_train, y_train, epochs=5, batch_size=5, verbose=2, validation_split=0.2)

        # Save the model to MLflow
        mlflow_helper.save_model_onnx(model=model, model_name="is_active_model",X_test=X_test, stage="staging")

        #Save the pipeline to MLflow
        mlflow_helper.save_pipeline(pipeline, "is_active_pipeline", "staging")

    print("Model trained successfully!")

def train_drugega():
    pass

if __name__ == '__main__':
    train_is_active()  

    #train_drugega()
