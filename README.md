# bio-ext
Biomarker and deep phenotype extraction platform, as an extension to CogStack, or other unstructured data stores in hospital EHR systems.

## Project Structure

`bio-ext` is set-up as a monorepo.

- `/deployment` deploying MLOps environment
- `/src` bioext package that is imported into projects, containing utils for interacting with environment
- `/projects` individual projects that contain labelling/model development scripts, not part of bioext package

```
|deployment/
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
|--.env
|--test_ml_runs/
|----requirements.txt
|--project_a/
|----requirements.txt
|--project_b/
|----requirements.txt
|
|pyproject.toml
|README.md
``` 

## Getting Started

### Set up Python environment

For the active project, and using preferred virtual environment manager, run:
```
pip install -r requirements.txt
```
As well as other packages, this must contain `-e ../../` to install bioext as an editable package.  

### Set up environmental variables

(1) (First time set-up) In `/deployment/`, create a copy of `.env.example` as `.env` and configure variables prior to docker-compose.

(2) From `/projects/`, add variables found in `.env.example` as provided by administrators to shell configuration:
```
MacOS: nano ~/.profile
Ubuntu: nano ~/.bashrc
```

(3) To apply changes, `source` the relevant configuration file

### Bio-ext deployment

(1) From `/deployment/`, run:
```
docker-compose up -d
```

(2) Check that services are running on:
```
Doccano: http://localhost:8000
MLFlow: http://localhost:5000
Minio: http://localhost:9001
```

### Test ML run

(1) For logistic regression on synthetic weight and height data, from `/projects/test_ml_runs/`, run:
```
python logreg_whg_test.py
```

(2) For wine quality elasticnet test, from `/projects/test_ml_runs/`, run:
```
python wine_test.py
```

(3) Log onto MLFlow frontend to confirm experiment logging, and check that model artifacts are stored and registered. Model artifacts can also be directly viewed by logging into Minio.
