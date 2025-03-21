# Running MedCAT Service

TODO Build Requirements:

1) Search an elasticsearch instance and retrieve records. (Skip this as it is general)
2) Load and run medcat to annotate documents
Features:
* FastAPI endpoints to serve MedCAT predictions.
* Dockerized application for easy deployment.


__Notes of requirements: __

# Three things are required:

### models dir
To hold models
1) base-

2) Fine-tuning-

### Docker file
To run and serve MedCAT model at an end point
Add anoncat endpoint

### A script to call the end point
 process -> endpoint 

 and documentation to test the service

 # Requirements

  - A MedCAT model pack store in the `./models/` directory
  - Python 3.11+


# Run the App

```bash
# Build the Docker image
docker-compose build

# Start the service
docker-compose up -d
```

### Test API endpoints
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Process text
curl -X 'POST' \
  'http://0.0.0.0:8000/api/v1/process' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "This patient has epilepsy",
  "filters": ["84757009"]
}'
```


# Instructions to Test locally
1) Download a Medcat model and add it to `./models/medcat_model_pack.zip`

2) Create a venv and install all packages from requirements.txt

3) Set the environment variable for the model path

```bash
export MEDCAT_MODEL_PATH=models/medcat_model_pack.zip  # Adjust path as needed
```

4) Run the FastAPI server
__Note__: make sure that you are in the medcat dir

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```