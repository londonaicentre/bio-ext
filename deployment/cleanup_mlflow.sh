#!/bin/bash

docker exec -it bioextmlflow \
    bash -c 'export MLFLOW_TRACKING_URI=http://localhost:5000 && \
    mlflow gc --backend-store-uri sqlite:///mlflow/data/mlflow.db --artifacts-destination s3://mlflow-artifacts'