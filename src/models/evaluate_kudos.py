import onnx
from src.models.model_helper import ModelHelper
from src.models.mlflow_helper import MlflowHelper
import mlflow
from src.database.connector import save_production_metrics
from datetime import datetime
from definitions import PATH_TO_REPORTS
import onnxruntime
import numpy as np
import os


def evaluate_kudos():
    mlflow_helper = MlflowHelper()
    mhelper = ModelHelper(mlflow_helper)
    print("Evaluate kudos method called!")

    # Start MLflow run
     # starting mlflow run
    experiment_name = "kudos_prediction"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="kudos_test"):
        # autolog
        mlflow.sklearn.autolog()
        
        # Load the staging model and pipeline from MLflow
        model_staging, pipeline_staging = mlflow_helper.get_model_pipeline("kudos_model", "kudos_pipeline", "staging")
        
        # Load the production model and pipeline from MLflow
        model_production, pipeline_production = mlflow_helper.get_model_pipeline("kudos_model", "kudos_pipeline", "production")

        if model_staging is None or model_production is None:
            print("Problem loading models from MLflow... exiting!")
            mlflow.end_run()
            return

        # Prepare test data
        X_train, y_train, X_test, y_test, pipeline = mhelper.prepare_train_test_kudos(pipeline_staging)

        # Evaluate the staging and production models
        model_staging = onnxruntime.InferenceSession(model_staging.SerializeToString())
        model_production = onnxruntime.InferenceSession(model_production.SerializeToString())

            # Retrieve input and output names
        input_name = model_staging.get_inputs()[0].name
        output_name = model_staging.get_outputs()[0].name
        
        print("Input name:", input_name)
        print("Output name:", output_name)

        input_name = model_production.get_inputs()[0].name
        output_name = model_production.get_outputs()[0].name
        
        print("Input name:", input_name)
        print("Output name:", output_name)
        
        # Before passing X_test to the model, ensure it is of type float32
        X_test = X_test.astype(np.float32)

        # get model information
        staging_model_predictions = model_staging.run(["variable"], {"float_input": X_test})[0]
        production_model_predictions = model_production.run(["variable"], {"float_input": X_test})[0]

        mae_staging, mse_staging, evs_staging = mhelper.calculate_metrics(y_test, staging_model_predictions)
        mae_production, mse_production, evs_production = mhelper.calculate_metrics(y_test, production_model_predictions)
        
        print(f"Staging model metrics: MAE={mae_staging}, MSE={mse_staging}, EVS={evs_staging}")
        print(f"Production model metrics: MAE={mae_production}, MSE={mse_production}, EVS={evs_production}")

        # compare the metrics
        if mae_staging < mae_production:
            print("Staging model is better than production model... replacing production model with staging model")
            mlflow_helper.update_to_production("kudos_model", "kudos_pipeline")

            #Save to mongodb
            metrics = {
                'datetime': datetime.now(),
                "mae_production": mae_production,
                "mse_production": mse_production,
                "evs_production": evs_production
            }
            save_production_metrics("kudos_production_metrics", metrics)

            #save metrics to mlflow
            mlflow.log_metric("mae", mae_production)
            mlflow.log_metric("mse", mse_production)
            mlflow.log_metric("evs", evs_production)

        else:
            # save staging metrics to mlflow
            mlflow.log_metric("mae", mae_staging)
            mlflow.log_metric("mse", mse_staging)
            mlflow.log_metric("evs", evs_staging)


        print("Evaluation completed!")
        print("Saving metrics to to report..")

    metrics = {
        'datetime': datetime.now(),
        "mae_staging": mae_staging,
        "mse_staging": mse_staging,
        "evs_staging": evs_staging
    }

    #print save staging metrics
    save_production_metrics("kudos_staging_metrics", metrics)

    # Save the metrics to the report
    path_to_file = os.path.join(PATH_TO_REPORTS, "kudos_report.txt")
    #create the file if it does not exist
    if not os.path.exists(path_to_file):
        with open(path_to_file, "x") as f:
            f.write("")

    with open(path_to_file, "w") as f:
        f.write(f"Staging model: MAE={mae_staging}, MSE={mse_staging}, EVS={evs_staging}\n")
        f.write(f"Production model: MAE={mae_production}, MSE={mse_production}, EVS={evs_production}\n")
    mlflow.end_run()


if __name__ == "__main__":
    evaluate_kudos()
