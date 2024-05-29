import onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.models.model_helper import ModelHelper
from src.models.mlflow_helper import MlflowHelper
import mlflow
import os
import numpy as np

def create_dataset_with_steps(time_series, look_back=1, step=1):
    X, y = [], []
    for i in range(0, len(time_series) - look_back, step):
        X.append(time_series[i:(i + look_back), :])
        y.append(time_series[i + look_back, 0])
    return np.array(X), np.array(y)

def reshape_for_model(train_df):
    look_back = 8
    step = 1
    X_train, y_train = create_dataset_with_steps(train_df, look_back, step)
    
    # Pravilno oblikovanje X_train (samples, look_back, features)
    X_train = np.reshape(X_train, (X_train.shape[0], look_back, train_df.shape[1]))
    
    return X_train, y_train

def train_is_active():
    mlflow_helper = MlflowHelper()
    mhelper = ModelHelper(mlflow_helper)
    print("Train is_active method called!")

    X_train, y_train, X_test, y_test, pipeline = mhelper.prepare_train_test_active()

    y_train = y_train.values.reshape(-1, 1)  # Convert Series to numpy array

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    data_train = np.hstack((y_train,X_train))

    # Reshape only X_train
    X_train, y_train = reshape_for_model(data_train)

    print("Unique values in y_train before :", np.unique(y_train))

    # Check shapes
    print("X_train shape after reshape:", X_train.shape)
    print("y_train shape after reshape:", y_train.shape)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    print("Input shape for GRU model:", input_shape)
    
    # Build and compile the GRU model
    model = mhelper.create_and_compile_quantisized_model(input_shape, 2)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=5, verbose=2, validation_split=0.2)
    print("Model trained successfully!")

def train_drugega():
    pass

if __name__ == '__main__':
    train_is_active()  

    #train_drugega()
