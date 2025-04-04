# bio-ext
Biomarker and deep phenotype extraction platform, as an extension to CogStack, or other unstructured data stores in hospital EHR systems.

## Project Structure

`bio-ext` is set-up as a monorepo.

- `/deployment` deploying MLOps environment in production
- `/deployment_local` deploying MLOps and ElasticSearch in local dev environment
- `/src` bioext package that is imported into projects, containing utils for interacting with environment
- `/projects` individual projects that contain labelling/model development scripts, not part of bioext package

```
|deployment/
|--docker-compose.yml
|--.env
|
|deployment_local/
|--docker-compose.yml
|--.env
|
|src/
|--bioext/
|----__init__.py
|----doccano_utils.py
|----elastic_utils.py
|
|projects/
|--test_ml_runs/
|----requirements.txt
|----.env
|--project_a/
|----requirements.txt
|----.env
|--project_b/
|----requirements.txt
|----.env
|
|pyproject.toml
|README.md
```

## Getting Started

### Set up Python environment

Requirements should be configured per project, using preferred virtual environment manager. e.g.:
```
pip install -r requirements.txt
```
As well as other packages, this must contain `-e ../../.[ml]` to install bioext as an editable package, with all the extra required machine learning libraries.

### First time set-up (admin)

(1) In `/deployment/`, create a copy of `.env.example` as `.env` and configure variables prior to docker-compose.

(2) Pull in the mlflow-boto-docker Dockerfile by running `git submodule update --init`.

(3) From `/deployment/`, run:
```
docker-compose up -d
```

(4) Check that services are running on:
```
Doccano: http://localhost:8000
MLFlow: http://localhost:5000
Minio: http://localhost:9001
```

### Environment variables for projects

Necessary environmental variables should be separately configured as .env files for each project in `/projects/`, and loaded using load_dotenv()

### Test ML run

(1) For logistic regression on synthetic weight and height data, from `/projects/test_ml_runs/`, run:
```
python logreg_whg_test.py
```

(2) For wine quality elasticnet test, from `/projects/test_ml_runs/`, run:
```
python wine_test.py
```

(3) Log onto MLFlow frontend to confirm experiment logging, and check that model artifacts are stored and registered in S3. Model artifacts can also be directly viewed by logging into Minio.
