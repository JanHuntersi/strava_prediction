from src.models.model_helper import ModelHelper
from src.models.mlflow_helper import MlflowHelper
import mlflow
from src.database.connector import save_production_metrics
from datetime import datetime
from definitions import PATH_TO_REPORTS
import onnxruntime
import numpy as np
import os




def evaluate_is_active():
    mlflow_helper = MlflowHelper()
    mhelper = ModelHelper(mlflow_helper)
    print("Evaluating is_active finished")

    experiment_name = "is_active_prediction"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="is_active_test"):

        # Load the staging model and pipeline from MLflow
        model_staging, pipeline_staging = mlflow_helper.get_model_pipeline("is_active_model", "is_active_pipeline", "staging")

        # Load the production model and pipeline from MLflow
        model_production, pipeline_production = mlflow_helper.get_model_pipeline("is_active_model", "is_active_pipeline", "production")

        if model_staging is None or model_production is None or pipeline_staging is None or pipeline_production is None:
            print("Problem loading models and pipelines from MLflow... exiting!")
            mlflow.end_run()
            return
    
        # Prepare test data
        X_train, y_train, X_test, y_test, pipeline = mhelper.prepare_train_test_active(pipeline_staging)

        # Evaluate the staging and production models
        model_staging = onnxruntime.InferenceSession(model_staging.SerializeToString())
        model_production = onnxruntime.InferenceSession(model_production.SerializeToString())

        # Retrieve input and output names
        input_name = model_staging.get_inputs()[0].name
        output_name = model_staging.get_outputs()[0].name

        print("Staging Input name:", input_name)
        print("Staging Output name:", output_name)

        input_name = model_production.get_inputs()[0].name
        output_name = model_production.get_outputs()[0].name

        print("Production Input name:", input_name)
        print("Production Output name:", output_name)

        y_test = y_test.values.reshape(-1, 1)  # Convert Series to numpy array
        print("y_test shape:", y_test.shape)
        print("number of values in y_test:", y_test.shape[0])

        data_test = np.hstack((y_test,X_test))

        # Reshape only X_train
        X_test, y_test = mhelper.reshape_for_model(data_test)

        staging_model_predictions = model_staging.run(["output"], {"input": X_test})[0]
        production_model_predictions = model_production.run(["output"], {"input": X_test})[0]

        print("Actual vs Staging Predictions vs Production Predictions:")
        for actual, staging_pred, production_pred in zip(y_test, staging_model_predictions, production_model_predictions):
            print(f"Actual: {actual}, Staging Prediction: {staging_pred}, Production Prediction: {production_pred}")
        

        print("Staging model predictions shape:", staging_model_predictions.shape)
        print("Production model predictions shape:", production_model_predictions.shape)

        print("Staging model predictions:", staging_model_predictions)
        print("Production model predictions:", production_model_predictions)

        #convert to binary
        staging_model_predictions = np.where(staging_model_predictions >= 0.5, 1, 0)
        production_model_predictions = np.where(production_model_predictions >= 0.5, 1, 0)

        acc_staging,prec_staging, rec_staging, f1_staging = mhelper.get_model_metrics(y_test, staging_model_predictions)
        acc_production, prec_production, rec_production, f1_production = mhelper.get_model_metrics(y_test, production_model_predictions)

        print(f"Staging model metrics: Accuracy={acc_staging}, Precision={prec_staging}, Recall={rec_staging}, F1={f1_staging}")
        print(f"Production model metrics: Accuracy={acc_production}, Precision={prec_production}, Recall={rec_production}, F1={f1_production}")
        
        # compare the metrics
        if acc_staging > acc_production:
            print("Staging model is better than production model... replacing production model with staging model")
            mlflow_helper.update_to_production("is_active_model", "is_active_pipeline")

            #Save to mongodb
            metrics = {
                'datetime': datetime.now(),
                "accuracy_production": acc_production,
                "precision_production": prec_production,
                "recall_production": rec_production,
                "f1_production": f1_production
            }
            save_production_metrics("is_active_production_metrics", metrics)

        print("Evaluation completed!")
        print("Saving metrics to to report..")

    # Save the metrics to the report

    path_to_file = os.path.join(PATH_TO_REPORTS, "is_active_report.txt")

    #create the file if it does not exist
    if not os.path.exists(path_to_file):
        with open(path_to_file, "x") as f:
            f.write("")

    with open(path_to_file, "w") as f:
        f.write(f"Staging model: Accuracy={acc_staging}, Precision={prec_staging}, Recall={rec_staging}, F1={f1_staging}\n")
        f.write(f"Production model: Accuracy={acc_production}, Precision={prec_production}, Recall={rec_production}, F1={f1_production}\n")

    mlflow.end_run()



        


if __name__ == "__main__":

    evaluate_is_active()
