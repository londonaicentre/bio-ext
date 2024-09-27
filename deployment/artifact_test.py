# for now, run this in a python venv. 
# requires pip install mlflow[extras] (this includes boto3 to talk to s3)
# test creation of artifacts using MLFlow and Minio 

import mlflow
import os
from mlflow.tracking import MlflowClient

def test_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run() as run:
        mlflow.log_param("ultimate_question", "life, universe, everything")
        mlflow.log_metric("ultimate_answer", 42)
        
        with open("test_artifact.txt", "w") as f:
            f.write("This is a test artifact...or is it?")
        
        mlflow.log_artifact("test_artifact.txt")

    # verify creation
    client = MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)

    artifact_found = any(artifact.path == "test_artifact.txt" for artifact in artifacts)
    if not artifact_found:
        raise Exception("Artifact was not created")

    print("MLflow test successful.")

if __name__ == "__main__":
    try:
        test_mlflow()
    except Exception as e:
        print(f"Failure: {e}")
        exit(1)