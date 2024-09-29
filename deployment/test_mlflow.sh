#!/bin/bash

# not in use, as can't connect to pip inside running container
# test MLFlow service from within docker container
echo "Testing MLflow artifact creation..."
docker run --rm \
    --network host \
    -v "$(pwd):/app" \
    -w /app \
    python:3.11.2 \
    bash -c "
        pip install mlflow[extras] &&
        python artifact_test.py
    "

echo "Test complete - see MLFlow UI"