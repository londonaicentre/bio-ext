# Local synthetic BRCA project

This folder contains example data and scripts to load and retrieve from the local ElasticSearch instance.

# Set up environment
1. Within the `deployment_local` folder, create a `.env` file based on the example provided, ensuring that all variables have the appropriate values.
2. Copy the same values for ElasticSearch into a `.env` file into `local_synth_brca` folder, based on the example provided.
3. Within the `deployment_local` folder, execute `docker compose up -d` and ensure that all containers are running, and that Kibana and MLflow interfaces are available (from Docker Desktop).
4. Create a new virtual environment (for Python 3.11<) and install the requirements from the `local_synth_brca` folder.

```
cd development_local
cp .env.example .env
# populate env variables

docker compose up -d
# ensure that all containers are running

# create a new virtual environment
python -m venv ~/virtual_envs/bioext_Py3.11
source ~/virtual_envs/bioext_Py3.11/bin/activate

cd local_synth_brca
pip install -r requirements.txt
```

# Populate local database and labelling tool
1. First execute the `python main.py -c config.json ES_load -d data/brca_reports.json` to load the example data into the local ElasticSearch,
2. Then execute the `python main.py -c config.json ES2Doc 100` script as an example for retrieving data from the ElasticSearch, dumping output in its `data/breast_brca_status/` folder (This may be configured within the script).

```
# Test that the development environment is functional
# load synthetic test data into local ElasticSearch database
python main.py -c config.json ES_load -d data/brca_reports.json
# query samples matching query from ES and load into new Doccano project for labelling
python main.py -c config.json ES2Doc 100
```