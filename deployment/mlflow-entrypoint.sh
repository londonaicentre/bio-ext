#!/bin/sh

# install mlflow[extras] which includes boto3 for s3 interaction
pip install --no-cache-dir mlflow[extras]
exec "$@"