from mlflow.tracking import MlflowClient
import dagshub
import mlflow
import src.settings as settings
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import tensorflow as tf
import tf2onnx
from mlflow.models import infer_signature

class MlflowHelper:
    def __init__(self):
        
        dagshub.auth.add_app_token(settings.mlflow_tracking_password)
        dagshub.init(repo_owner=settings.mlflow_tracking_username, repo_name=settings.dagshub_repo_name,mlflow=True)
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient()

    def convert_kudos_onnx(self,model, num_features): 
        initial_type = [('float_input', FloatTensorType([None, num_features]))]  # Replace num_features with the number of features in your data
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        return onnx_model
    
    def convert_is_active_onnx(self,model, input_shape): 
        initial_type = [('float_input', FloatTensorType([None, input_shape[0], input_shape[1]]))]

        onnx_model = convert_sklearn(model, initial_types=initial_type)
        return onnx_model
    
    def save_model(self, model, model_name, stage):

        model = mlflow.onnx.log_model(
            onnx_model=model,
            artifact_path=f"models/{model_name}/model",
            registered_model_name=f"model={model_name}",
        )

        model_version = self.client.create_model_version(
            name=f"model={model_name}",
            source=model.model_uri,
            run_id=model.run_id
        )

        self.client.transition_model_version_stage(
            name=f"model={model_name}",
            version=model_version.version,
            stage=stage,
        )
    
    def save_pipeline(self, pipeline, pipeline_name, stage):
        pipeline = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=f"pipelines/{pipeline_name}/pipeline",
            registered_model_name=f"pipeline={pipeline_name}",
        )

        pipeline_version = self.client.create_model_version(
            name=f"pipeline={pipeline_name}",
            source=pipeline.model_uri,
            run_id=pipeline.run_id
        )

        self.client.transition_model_version_stage(
            name=f"pipeline={pipeline_name}",
            version=pipeline_version.version,
            stage=stage,
        )

    def download_pipeline(self, pipeline_name, stage):
        try:
            pipeline_name=f"pipeline={pipeline_name}"
            print("Downloading pipeline: ", pipeline_name)
            # Get pipeline
            latest_pipeline_source = self.client.get_latest_versions(name=pipeline_name, stages=[stage])[0].source

            # Load the pipeline in source
            return mlflow.sklearn.load_model(latest_pipeline_source)
        
        except IndexError:
            print(f"There was an error downloading {pipeline_name} in {stage} from mlfow.")
            return None
    
    def download_model(self, model_name, stage):
        try:
            model_name=f"model={model_name}"
            print("Downloading model: ", model_name)

            # Get model
            latest_model_source = self.client.get_latest_versions(name=model_name, stages=[stage])[0].source

            # Load the model in source
            return mlflow.onnx.load_model(latest_model_source)
        
        except IndexError:
            print(f"There was an error downloading {model_name} in {stage} from mlfow.")
            return None

    def get_model_pipeline(self,model_name,pipeline_name,stage="staging"):


        print(f"{stage} Getting model and pipeline from MLflow")
        
        model = self.download_model(model_name, stage)
        pipeline = self.download_pipeline(pipeline_name,stage)
        
        return model, pipeline
    
    def update_to_production(self, model_name):
        model_name=f"model={model_name}"
        pipeline_name=f"pipeline={model_name}"
        try:
            # Get model and scaler latest staging version
            model_version = self.client.get_latest_versions(name=model_name, stages=["staging"])[0].version
            pipeline_version = self.client.get_latest_versions(name=pipeline_name, stages=["staging"])[0].version

            # Update production model and scaler
            self.client.transition_model_version_stage(model_name, model_version, "production")
            self.client.transition_model_version_stage(pipeline_name, pipeline_version, "production")
        except IndexError:
            print(f"There was an error replacing production model {model_name}")
            return None

    def save_model_onnx(self,model,model_name,X_test,stage):
 # SAVE MODEL
        model.output_names = ['output']

        input_signature = [tf.TensorSpec(shape=(None, 8, 13), dtype=tf.double, name="input")]

        #convert model to onnx
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

        # Log the model
        onnx_model = mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path=f"models/{model_name}/model", 
            signature=infer_signature(X_test, model.predict(X_test)),
            registered_model_name=f"model={model_name}"
        )
        # Create model version

        model_version = self.client.create_model_version(
            name=f"model={model_name}",
            source=onnx_model.model_uri,
            run_id=onnx_model.run_id
        )

        # Transition model version to staging
        self.client.transition_model_version_stage(
            name=f"model={model_name}",
            version=model_version.version,
            stage=stage,
        )

        print(f"Saved model for {model_name}")
    
    


