import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import  explained_variance_score, mean_absolute_error, mean_squared_error
import os
from sklearn.tree import DecisionTreeRegressor
from definitions import PATH_TO_TEST_TRAIN,PATH_TO_PROCESSED_IS_ACTIVE
import onnx
from sklearn.base import  clone
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import tensorflow_model_optimization as tmo
from tensorflow.keras import Input
from tensorflow import keras
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer, quantize_apply
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

selected_features_kudos = ['elapsed_time','achievement_count', 'comment_count', 'athlete_count', 'duration', 'hour','day']
selected_features_is_active = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m','apparent_temperature', 'precipitation_probability', 'precipitation',
       'rain', 'snowfall', 'snow_depth', 'is_day', 'day', 'hour']

columns_to_normalize_is_active = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'precipitation_probability', 'precipitation',
        'rain', 'snowfall', 'snow_depth'
        ]


class ModelHelper:
    def __init__(self,ml_flow):
        self.ml_flow = ml_flow
        pass

    def fill_missing_values(self, data):
        # Get numerical and categorical columns
        num_cols = data.select_dtypes(include=[np.number]).columns
        cat_cols = data.select_dtypes(include=['object']).columns

        # Fill missing values for numerical columns with mean
        data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

        # Fill missing values for categorical columns with 'unknown'
        data[cat_cols] = data[cat_cols].fillna("unknown")

        return data
    
    def get_day_and_hour(self, data, date_column='start_date_local'):
        data[date_column] = pd.to_datetime(data[date_column])
        data['day'] = data[date_column].dt.day
        data['hour'] = data[date_column].dt.hour
        return data
    
    def separate_features_target(self, data, target):
        X = data.drop(columns=[target])
        y = data[target]
        return X, y
    
    def create_is_active_pipeline(self):

       

        norm_transformer = Pipeline([
            ('normalize', StandardScaler()),
        ])

        preprocessor = ColumnTransformer([
            ('num', norm_transformer, columns_to_normalize_is_active),
        ])
        
        return Pipeline([
            ('preprocessor', preprocessor),
        ])
    
    def create_kudos_pipeline(self):
        norm_transformer = Pipeline([
            ('normalize', StandardScaler()),
        ])

        preprocessor = ColumnTransformer([
            ('num', norm_transformer, selected_features_kudos),
        ])
        
        return Pipeline([
            ('preprocessor', preprocessor),
        ])
            
    def info_gain(self, X, y):
        info_gains = mutual_info_regression(X, y)
        return info_gains
    
    def prepare_data_kudos(self, pipeline, data):

        # Fill missing values
        data = self.fill_missing_values(data)

        # Get day and hour
        data = self.get_day_and_hour(data)

        # Select features
        data = data[selected_features_kudos + ['kudos_count']]

        # Separate features and target
        X, y = self.separate_features_target(data, 'kudos_count')

        if pipeline is None:
            pipeline = self.create_kudos_pipeline()

        print("X_train shape before transformation: ", X.shape)

        X_transformed = pipeline.transform(X)

        print(f"X_train shape: {X_transformed.shape}")


        return X_transformed, y, pipeline

    def prepare_train_test_kudos(self, pipeline=None):
        print("Preparing data for training or testing")

        test_data = pd.read_csv(os.path.join(PATH_TO_TEST_TRAIN, "kudos_test.csv"))
        train_data = pd.read_csv(os.path.join(PATH_TO_TEST_TRAIN, "kudos_train.csv"))

        # Fill null data
        print(f"Test data missing values: {test_data.isnull().sum().sum()}")
        print(f"Train data missing values: {train_data.isnull().sum().sum()}")

        test_data = self.fill_missing_values(test_data)
        train_data = self.fill_missing_values(train_data)

        # Get day and hour
        test_data = self.get_day_and_hour(test_data)
        train_data = self.get_day_and_hour(train_data)

        # Select features

        test_data = test_data[selected_features_kudos + ['kudos_count']]
        train_data = train_data[selected_features_kudos + ['kudos_count']]


        # Separate features and target
        X_train, y_train = self.separate_features_target(train_data, 'kudos_count')
        X_test, y_test = self.separate_features_target(test_data, 'kudos_count')


        if pipeline is None:
            pipeline = self.create_kudos_pipeline()

        print("X_train shape before transformation: ", X_train.shape)

        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        print(f"X_train shape: {X_train_transformed.shape}")

        # calculate info gain
        info_gains = self.info_gain(X_train_transformed, y_train)
        info_gains = pd.Series(info_gains, index=selected_features_kudos)
        info_gains.sort_values(ascending=False, inplace=True)

        print(info_gains)

        return X_train_transformed, y_train, X_test_transformed, y_test, pipeline
    
    def get_model_metrics(self, y_predictions, y_true):
        accuracy = accuracy_score(y_true, y_predictions)
        precision = precision_score(y_true, y_predictions, average='weighted')
        recall = recall_score(y_true, y_predictions, average='weighted')
        f1 = f1_score(y_true, y_predictions, average='weighted')
        return accuracy, precision, recall, f1

    def calculate_metrics(self,y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        return mae, mse, evs

    def train_random_forest_model(self,model, X_train, y_train, X_val=None, y_val=None):
        # Fit the model on the training data
        model.fit(X_train, y_train)
        
        # Predict on training data
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        print("Training MSE:", train_mse)
        
        # Optionally, evaluate on validation data if provided
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            print("Validation MSE:", val_mse)
        return model
    
    def quantize(self, weights, num_levels=256):
        # Preprosta kvantizacija uteži na `num_levels` nivojev
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        quantized_weights = np.round((weights - min_weight) / (max_weight - min_weight) * (num_levels - 1))
        quantized_weights = quantized_weights / (num_levels - 1) * (max_weight - min_weight) + min_weight
        return quantized_weights

    def quantize_model(self, model):
        quantized_model = RandomForestRegressor(n_estimators=model.n_estimators, random_state=model.random_state)
        quantized_model.fit(np.zeros((1, model.n_features_in_)), np.zeros(1))  # fit z dummy podatki

        for i, tree in enumerate(model.estimators_):
            # Extract tree structure and values
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold

            # Quantize the values
            original_values = tree.tree_.value.flatten()
            quantized_values = self.quantize(original_values)
            quantized_values = quantized_values.reshape(tree.tree_.value.shape)

            # Create a new DecisionTreeRegressor and set its properties
            new_tree = DecisionTreeRegressor()
            new_tree.fit(np.zeros((1, model.n_features_in_)), np.zeros(1))  # fit z dummy podatki
            new_tree.tree_ = clone(tree.tree_)
            new_tree.tree_.value[:] = quantized_values

            quantized_model.estimators_[i] = new_tree

        return quantized_model
    
    def evaluate_kudos_model(self, model, X_test, y_test,stage):
        print(f"Evaluating kudos {stage} model")
        # Predict on test data
        y_pred = model.run(["output"], {"input": X_test})[0]
        
        # Calculate metrics
        mae, mse, evs = self.calculate_metrics(y_test, y_pred)
        
        if stage == "production":
            print(f"Production model metrics: MAE={mae}, MSE={mse}, EVS={evs}")
            self.ml_flow.log_metric("production_mae", mae)
            self.ml_flow.log_metric("production_mse", mse)
            self.ml_flow.log_metric("production_evs", evs)
        else:
            print(f"Staging model metrics: MAE={mae}, MSE={mse}, EVS={evs}")
            self.ml_flow.log_metric("staging_mae", mae)
            self.ml_flow.log_metric("staging_mse", mse)
            self.ml_flow.log_metric("staging_evs", evs)

        return mae, mse, evs
    
    def calculate_info_gains(self, X_train, y_train):

        info_gains = mutual_info_regression(X_train, y_train)
        info_gains = pd.Series(info_gains, index=selected_features_kudos)
        info_gains.sort_values(ascending=False, inplace=True)

        return info_gains
    
    def convert_to_sin_cos(self, df):
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)  # Predpostavka: 'day' ima vrednosti 0-6 za dneve v tednu
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)  # 'hour' ima vrednosti 0-23
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Odstranimo originalne 'day' in 'hour' stolpce
        df.drop(['day', 'hour'], axis=1, inplace=True)
        return df
    
    def prepare_data_processed(self,pipeline,processed_data):

        #convert to date
        processed_data['date'] = pd.to_datetime(processed_data['date'])

        #print the columns with missing values
        print(processed_data.isnull().sum())

        #print categorical columns
        print(processed_data.select_dtypes(include=['object']).columns)

        #fill missing values
        processed_data = self.fill_missing_values(processed_data)

        # Get day and hour
        processed_data = self.get_day_and_hour(processed_data,'date')

        # Ustvarite novo značilko, ki označuje četrtino dneva, v kateri pade dana ura
        processed_data['quarter_of_day'] = pd.cut(processed_data['hour'], bins=[0, 6, 12, 18, 24], labels=[1, 2, 3, 4], include_lowest=True)

        # ustvarite znacilko za del dneva med 0,8,16,24
        processed_data['time_of_day'] = pd.cut(processed_data['hour'], bins=[0, 8, 16, 24], labels=[1, 2, 3], include_lowest=True)

        #convert to sin and cos
        processed_data = self.convert_to_sin_cos(processed_data)

        #Convert is_active to binary
        processed_data['is_active'] = processed_data['is_active'].astype(int)

        print("Processed data 0 and 1: ", processed_data['is_active'].value_counts())

        if pipeline is None:
            pipeline = self.create_is_active_pipeline()
        
        print("X_train shape before transformation: ", processed_data.shape)

        norm_processed_data = pipeline.fit_transform(processed_data)

        # Convert NumPy arrays back to DataFrames
        processed_df = pd.DataFrame(norm_processed_data, columns=columns_to_normalize_is_active)

        #Merge non normalized columns with normalized columns
        selected_non_normalized = ['day_sin','day_cos', 'hour_sin','hour_cos','is_day','quarter_of_day','time_of_day']

        processed_df[selected_non_normalized] = processed_data[selected_non_normalized]

        # final features list with y as the last column
        final_features_list = ['quarter_of_day', 'day_sin', 'precipitation_probability', 'time_of_day', 'hour_sin', 'day_cos', 'relative_humidity_2m', 'hour_cos', 'rain', 'precipitation', 'apparent_temperature', 'temperature_2m']

        data = processed_df[final_features_list]

        # Separate features and target
        y = processed_data['is_active']

        # return X and y
        return data, y
        

    def prepare_train_test_active(self, pipeline=None):
        
        # Load train test data
        train_data = pd.read_csv(os.path.join(PATH_TO_TEST_TRAIN, "is_active_train.csv"))
        test_data = pd.read_csv(os.path.join(PATH_TO_TEST_TRAIN, "is_active_test.csv"))

        # Convert to date
        train_data['date'] = pd.to_datetime(train_data['date'])
        test_data['date'] = pd.to_datetime(test_data['date'])

        # fill missing values
        test_data = self.fill_missing_values(test_data)
        train_data = self.fill_missing_values(train_data)

        # Get day and hour
        test_data = self.get_day_and_hour(test_data,'date')
        train_data = self.get_day_and_hour(train_data,'date')

        # Ustvarite novo značilko, ki označuje četrtino dneva, v kateri pade dana ura
        test_data['quarter_of_day'] = pd.cut(test_data['hour'], bins=[0, 6, 12, 18, 24], labels=[1, 2, 3, 4], include_lowest=True)
        train_data['quarter_of_day'] = pd.cut(train_data['hour'], bins=[0, 6, 12, 18, 24], labels=[1, 2, 3, 4], include_lowest=True)

        # ustvarite znacilko za del dneva med 0,8,16,24
        train_data['time_of_day'] = pd.cut(train_data['hour'], bins=[0, 8, 16, 24], labels=[1, 2, 3], include_lowest=True)
        test_data['time_of_day'] = pd.cut(test_data['hour'], bins=[0, 8, 16, 24], labels=[1, 2, 3], include_lowest=True)

        #convert to sin and cos
        test_data = self.convert_to_sin_cos(test_data)
        train_data = self.convert_to_sin_cos(train_data)

        #Convert is_active to binary
        train_data['is_active'] = train_data['is_active'].astype(int)
        test_data['is_active'] = test_data['is_active'].astype(int)

        print("Train data 0 and 1: ", train_data['is_active'].value_counts())
        print("Test data 0 and 1: ", test_data['is_active'].value_counts())


        """
        Data features:
        ['date', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
       'apparent_temperature', 'precipitation_probability', 'precipitation',
       'rain', 'snowfall', 'snow_depth', 'is_day', 'is_active', 'day', 'hour']

        y: is_active

        """

        if pipeline is None:
            pipeline = self.create_is_active_pipeline()

        print("X_train shape before transformation: ", train_data.shape)

        norm_train_data = pipeline.fit_transform(train_data)
        norm_test_data = pipeline.transform(test_data)


        # Convert NumPy arrays back to DataFrames
        train_df = pd.DataFrame(norm_train_data, columns=columns_to_normalize_is_active)
        test_df = pd.DataFrame(norm_test_data, columns=columns_to_normalize_is_active)

        #Merge non normalized columns with normalized columns
        selected_non_normalized = ['day_sin','day_cos', 'hour_sin','hour_cos','is_day','quarter_of_day','time_of_day']
        train_df[selected_non_normalized] = train_data[selected_non_normalized] 
        test_df[selected_non_normalized] = test_data[selected_non_normalized]


        # Separate features and target
        y_train, y_test = train_data['is_active'], test_data['is_active']

        

        final_features_list = ['quarter_of_day', 'day_sin', 'precipitation_probability', 'time_of_day', 'hour_sin', 'day_cos', 'relative_humidity_2m', 'hour_cos', 'rain', 'precipitation', 'apparent_temperature', 'temperature_2m']
        X_train, X_test = train_df[final_features_list], test_df[final_features_list]


        print(y_train)
        # count number of 1 and 0
        print(y_train.value_counts())

        #calculate info gain
        info_gains = mutual_info_regression(X_train, y_train)

        info_gains = pd.Series(info_gains, index=final_features_list)
        info_gains.sort_values(ascending=False, inplace=True)

       
        print(info_gains)

        return X_train, y_train, X_test, y_test, pipeline
    
    def create_and_compile_quantized_model(self, input_shape, num_classes):
        model = keras.models.Sequential([
            keras.layers.Input(shape=input_shape),
            tmo.quantization.keras.quantize_annotate_layer(keras.layers.Dense(32, activation='relu')),
            keras.layers.Dropout(0.1),
            tmo.quantization.keras.quantize_annotate_layer(keras.layers.Dense(16, activation='relu')),
            keras.layers.Dropout(0.1),
            tmo.quantization.keras.quantize_annotate_layer(keras.layers.Dense(num_classes, activation='sigmoid')),
        ])

        # Create a quantization scope for supported layers
        quantize_scope = tmo.quantization.keras.quantize_scope({
            keras.layers.Dense,
            keras.layers.Dropout,
        })

        # Apply quantization with the specified quantization scope
        tmo.quantization.keras.quantize_apply(model, quantize_scope)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def create_and_compile_quantisized_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(GRU(units=32, return_sequences=True, input_shape=input_shape))
        model.add(GRU(units=32))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        # Kompilacija modela
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def create_dataset_with_steps(self,time_series, look_back=1, step=1):
        X, y = [], []
        for i in range(0, len(time_series) - look_back, step):
            X.append(time_series[i:(i + look_back), :])
            y.append(time_series[i + look_back, 0])
        return np.array(X), np.array(y)

    def reshape_for_model(self,train_df):
        look_back = 8
        step = 1
        X_train, y_train = self.create_dataset_with_steps(train_df, look_back, step)
        
        # Pravilno oblikovanje X_train (samples, look_back, features)
        X_train = np.reshape(X_train, (X_train.shape[0], look_back, train_df.shape[1]))
        
        return X_train, y_train
