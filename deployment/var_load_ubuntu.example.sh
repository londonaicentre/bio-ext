#!/bin/bash

# Ubuntu script to set environment variables
# Before using, replace variables and remove .example from filename

echo 'export DOCCANO_ADMIN_USERNAME="admin"' >> ~/.bashrc
echo 'export DOCCANO_ADMIN_PASSWORD="adminpw"' >> ~/.bashrc
echo 'export DOCCANO_ADMIN_EMAIL="admin@admin.com"' >> ~/.bashrc

echo 'export AWS_ACCESS_KEY_ID="access_key"' >> ~/.bashrc
echo 'export AWS_SECRET_ACCESS_KEY="secret_key"' >> ~/.bashrc

echo 'export ELASTIC_API_ID="elasticid"' >> ~/.bashrc
echo 'export ELASTIC_API_KEY="elastickey"' >> ~/.bashrc
echo 'export ELASTIC_ENCODED="elasticencodedy"' >> ~/.bashrc
echo 'export ELASTIC_SERVER="esserver"' >> ~/.bashrc

source ~/.bashrc

echo "Environment variables have been added to ~/.bashrc and applied to this session"