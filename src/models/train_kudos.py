import onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.models.model_helper import ModelHelper
from src.models.mlflow_helper import MlflowHelper
import mlflow

def train_kudos():
    max_depth=10 
    min_samples_split=2 
    min_samples_leaf=2
    mlflow_helper = MlflowHelper()
    mhelper = ModelHelper(mlflow_helper)


    print("Train kudos method called!")

    # starting mlflow run
    experiment_name = "kudos_prediction"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="kudos_train"):
        # autolog
        mlflow.sklearn.autolog()
        
        # Log parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)

        # Prepare train data
        X_train, y_train, X_test, y_test, pipeline = mhelper.prepare_train_test_kudos()


        # Train model
            # Splitting data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=42)

        num_features = X_train.shape[1]


        print("Number of features: ", num_features)

        # Create and train the RandomForestRegressor model
        random_forest_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf
        )

        trained_random_forest_model = mhelper.train_random_forest_model(random_forest_model, X_train, y_train, X_val, y_val)


        onnx_model = mlflow_helper.convert_kudos_onnx(trained_random_forest_model, num_features)

        # Save model to MLflow
        mlflow_helper.save_model(onnx_model, "kudos_model", "staging")

        #Save pipeline to Mlflow
        mlflow_helper.save_pipeline(pipeline, "kudos_pipeline", "staging")

    print("Model trained successfully!")


if __name__ == '__main__':
    train_kudos()
