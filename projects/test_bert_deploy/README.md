# Example BERT model training on BRCA synth data

This folder contains example scripts to train a binary or multi-label classification BERT model.

# Set up environment
- In the project root folder, execute `git submodule init` and `git submodule update` commands to obtain the mlflow boto Dockerfile from a related repo.
- Ensure that environment variables are populated in the `deployment/.env` file (see example file and contact developers for expected values)
- Within the `deployment` folder, execute `docker compose up -d` and ensure that all containers are running, and that Kibana and MLflow interfaces are available (from Docker Desktop or directly using their expected ports).
- For local development, follow instructions in `local_synth_brca` to create and annotate samples in a Doccano project. Make note of the Doccano project ID.
- Create a new virtual environment (for Python 3.11.2<) and install the requirements from within `test_bert_deploy/requirements.txt`.

# Train example model for binary/multi-label classification
Execute `python test_bert_deploy/train.py`, using the required type of classification script.
