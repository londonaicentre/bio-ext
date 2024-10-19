#!/bin/bash

# macOS script to set environment variables
# Before using, replace variables and remove .example from filename

echo 'export DOCCANO_ADMIN_USERNAME="admin"' >> ~/.bash_profile
echo 'export DOCCANO_ADMIN_PASSWORD="yourpassword"' >> ~/.bash_profile
echo 'export DOCCANO_ADMIN_EMAIL="admin@example.com"' >> ~/.bash_profile
echo 'export AWS_ACCESS_KEY_ID="your_access_key"' >> ~/.bash_profile
echo 'export AWS_SECRET_ACCESS_KEY="your_secret_key"' >> ~/.bash_profile

echo 'export ELASTIC_API_ID="elasticid"' >> ~/.bash_profile
echo 'export ELASTIC_API_KEY="elastickey"' >> ~/.bash_profile
echo 'export ELASTIC_ENCODED="elasticencodedy"' >> ~/.bash_profile
echo 'export ELASTIC_SERVER="esserver"' >> ~/.bash_profile

source ~/.bash_profile

echo "Environment variables have been added to ~/.bash_profile and applied to this session."