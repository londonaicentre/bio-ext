#!/bin/bash
export AWS_ACCESS_KEY_ID=admin12345678
export AWS_SECRET_ACCESS_KEY=adminpw12345678
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export MLFLOW_S3_IGNORE_TLS=true
export AWS_DEFAULT_REGION=minio

mlflow models serve -m "s3://mlflow-artifacts/1/f460f9114c314b1d812d0ec48fea9728/artifacts/bert_model" -p 8080