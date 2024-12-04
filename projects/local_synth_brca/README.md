# Local synthetic BRCA project

This folder contains example data and scripts to load and retrieve from the local ElasticSearch instance.

# Set up environment
1. Within the `deployment_local` folder, create a `.env` file based on the example provided, ensuring that all variables have the appropriate values.
2. Copy the same values for ElasticSearch into a `.env` file into `local_synth_brca` folder.
3. Within the `deployment_local` folder, execute the `start.sh` and ensure that all containers are running, and that Kibana and MLflow interfaces are available (from Docker Desktop).
4. Create a new virtual environment and install the requirements from the `local_synth_brca` folder.
5. First execute the `synthetic_brca_load.py` script to load the example data into the local ElasticSearch,
6. Then execute the `synthetic_brca_retrieve.py` script as an example for retrieving data from the ElasticSearch, dumping output in its `data/breast_brca_status/` folder (This may be configured within the script).