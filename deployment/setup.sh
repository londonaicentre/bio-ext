#!/bin/bash

## this script sets up the bio-ext MLOps environment in a specified directory 
## usage: sudo ./setup.sh
DEPLOY_DIR="/srv/bioext"

# create directories for each service as mount (if they don't already exist)
mkdir -p "$DEPLOY_DIR/data/doccano"
mkdir -p "$DEPLOY_DIR/data/mlflow"
mkdir -p "$DEPLOY_DIR/data/minio"

# copy files to deployment directory
cp docker-compose.yml "$DEPLOY_DIR/"
echo "docker-compose.yml copied to $DEPLOY_DIR"

cp artifact_test.py "$DEPLOY_DIR/"
chmod +x "$DEPLOY_DIR/artifact_test.py"
echo "artifact_test.py copied to $DEPLOY_DIR"

# check if .env file exists, create if it doesn't, set new credentials
if [ ! -f "$DEPLOY_DIR/.env" ]; then
    MINIO_ACCESS_KEY=$(openssl rand -hex 8)
    MINIO_SECRET_KEY=$(openssl rand -hex 16)
    DOCCANO_ADMIN_USERNAME=admin
    DOCCANO_ADMIN_PASSWORD=adminpw
    DOCCANO_ADMIN_EMAIL=admin@admin.com
    sudo cat > "$DEPLOY_DIR/.env" << EOL
MINIO_ACCESS_KEY=$MINIO_ACCESS_KEY
MINIO_SECRET_KEY=$MINIO_SECRET_KEY
DOCCANO_ADMIN_USERNAME=$DOCCANO_ADMIN_USERNAME
DOCCANO_ADMIN_PASSWORD=$DOCCANO_ADMIN_PASSWORD
DOCCANO_ADMIN_EMAIL=$DOCCANO_ADMIN_EMAIL
EOL
    echo "New .env file created with credential generation"
else
    echo "Existing .env file found. Using existing credentials."
fi

#source .env into shell
set -a
source "$DEPLOY_DIR/.env"
set +a

# change to the deployment directory
cd "$DEPLOY_DIR" || { echo "Failed to change to $DEPLOY_DIR"; exit 1; }

# shut down any existing containers
echo "Shutting down existing containers..."
docker-compose --env-file "$DEPLOY_DIR/.env" down

# pull latest
echo "Pulling latest..."
docker-compose --env-file "$DEPLOY_DIR/.env" pull

# start up containers
echo "Starting up containers..."
docker-compose --env-file "$DEPLOY_DIR/.env" up -d

# check if MLFlow service running
# echo "Waiting for MLflow to be ready..."
# timeout 60s bash -c 'until curl -s http://localhost:5000 > /dev/null; do sleep 1; done' || { echo "MLflow failed to start"; exit 1; }

# test MLFlow service
# echo "Testing MLflow artifact creation..."
# docker run --rm --network host -v "$DEPLOY_DIR:/app" -w /app python:3.11.2 bash -c "
#    pip install mlflow[extras] &&
#    python artifact_test.py
#"
#rm "$DEPLOY_DIR/artifact_test.py"

echo "Bio-ext environment started. You can access:"
echo "Doccano at http://localhost:8000"
echo "MLflow at http://localhost:5000"
echo "MinIO at http://localhost:9000 (API) and http://localhost:9001 (Console)"
echo "MinIO Access Key: $MINIO_ACCESS_KEY"
echo "MinIO Secret Key: $MINIO_SECRET_KEY"
