# Running MedCAT Service

This service provides a REST API to process text using MedCAT models.

 # Requirements

  - A MedCAT modelpack store in the `./models/` directory
  - Docker
  - Python 3.11+


# How to Run the Service

## Using Docker

1. **Build the Docker Image:**  
    ```bash
    # Build the Docker image
    docker-compose build

    # Start the service
    docker-compose up -d
    ```

2. **Test API Endpoints:**
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


## Running Locally (Without Docker)
1. **Download a MedCAT Model Pack:**

    Place the model file inside the ./models/ directory:

    ```bash
    ./models/medcat_model_pack.zip
    ```
2. **Set Up Virtual Environment:**

    Create a virtual environment and install dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Set Environment Variable for Model Path:**
    ```bash
    export MEDCAT_MODEL_PATH=models/medcat_model_pack.zip  # Adjust the path if needed
    ```

4. **Run the FastAPI Server:**
    Make sure you're in the medcat directory and then run:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

## Stopping and Cleaning Up
    
  Stop the Docker Service:

    docker-compose down


  Remove Images, Volumes, and Networks (Optional):

    docker-compose down --rmi all --volumes


## API Documentation
After starting the server, you can explore the API documentation at:
http://localhost:8000/docs


