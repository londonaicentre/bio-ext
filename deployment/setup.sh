#!/bin/bash

## this script allows docker-compose to be run from a local repository 
## usage: sudo ./setup.sh

DEPLOY_DIR="/srv/bioext"

# create directories for each service as mount (if they don't already exist)
mkdir -p "$DEPLOY_DIR/data/labelstudio"
mkdir -p "$DEPLOY_DIR/data/mlflow"
mkdir -p "$DEPLOY_DIR/data/minio"

# copy latest docker-compose.yml
cp docker-compose.yml "$DEPLOY_DIR/"
echo "docker-compose.yml copied to $DEPLOY_DIR"

# check if .env file exists, create if it doesn't with set credentials
if [ ! -f "$DEPLOY_DIR/.env" ]; then
    MINIO_ACCESS_KEY=$(openssl rand -hex 8)
    MINIO_SECRET_KEY=$(openssl rand -hex 16)
    sudo cat > "$DEPLOY_DIR/.env" << EOL
MINIO_ACCESS_KEY=$MINIO_ACCESS_KEY
MINIO_SECRET_KEY=$MINIO_SECRET_KEY
EOL
    echo "New .env file created with randomly generated credentials"
else
    echo "Existing .env file found. Using existing credentials."
fi

#source .env into shell
set -a
source "$DEPLOY_DIR/.env"
set +a

# shut down any existing containers
echo "Shutting down existing containers..."
docker-compose --env-file "$DEPLOY_DIR/.env" down

# pull latest
echo "Pulling latest..."
docker-compose --env-file "$DEPLOY_DIR/.env" pull

# start up containers
echo "Starting up containers..."
docker-compose --env-file "$DEPLOY_DIR/.env" up -d

echo "Bio-ext environment started. You can access:"
echo "LabelStudio at http://localhost:8080"
echo "MLflow at http://localhost:5000"
echo "MinIO at http://localhost:9000 (API) and http://localhost:9001 (Console)"
echo "MinIO Access Key: $MINIO_ACCESS_KEY"
echo "MinIO Secret Key: $MINIO_SECRET_KEY"
