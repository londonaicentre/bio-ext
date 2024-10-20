# bio-ext
Biomarker and deep phenotype extraction platform, as an extension to CogStack, or other unstructured data stores in hospital EHR systems.

## Project Structure

`bio-ext` is set-up as a monorepo.

- `/deployment` deploying MLOps environment
- `/src` utility packages for imports
- `/projects` individual projects that contain labelling/model development scripts

```
|deployment
|--docker-compose.yml
|
|src
|--doccano_utils.py
|--elastic_utils.py
|
|projects
|--test_ml_runs
|--project_a
|--project_b
|
|README.md
|requirements.txt
``` 

## Getting Started

### Set up Python environment

Using preferred virtual environment manager, run:
```
pip install -r requirements.txt
```

### Set up environmental variables

(1) From `/deployment/`, create a copy of `.env.example` as `.env` and insert variables as provided by administrators

(2) Add environmental variables to shell configuration.
```
MacOS: nano ~/.bash_profile
Ubuntu: nano ~/.bashrc
```

(3) To apply changes, either `source` file or restart the terminal

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

### Absolute imports

To enable imports from packages in `/src` into `__main__` scripts found in `/projects`, make `root_directory` the first import within the script, and use absolute imports. For example:
```
import root_directory
from src.elastic_connect import ElasticsearchSession
```