# Multi class single label classifier on BRCA synthetic data

** Note: an adaptation from `test_bert_deploy`**

```
test_multiclass_classifier/
    |--data/
        |--br
    |--multiclassifier.py
    |--requirements.txt
```

# FIRST
you need to create a data folder and then put the brca labelled json file there within this 

This folder contains example scripts to train a multi-class single label classification BERT model.

# Set up environment
- In the project root folder, execute `git submodule init` and `git submodule update` commands to obtain the mlflow boto Dockerfile from a related repo.
- Ensure that environment variables are populated in the `deployment/.env` file (see example file and contact developers for expected values)
- Within the `deployment` folder, execute `docker compose up -d` and ensure that all containers are running, and that Kibana and MLflow interfaces are available (from Docker Desktop or directly using their expected ports).
- For local development, follow instructions in `local_synth_brca` to create and annotate samples in a Doccano project. Make note of the Doccano project ID.
- Create a new virtual environment (for Python 3.11.2<) and install the requirements from within `test_multiclass_classifier/requirements.txt`. It is effectively a `bioext_Py3.11` venv's requirements. 

# Train example model for binary/multi-label classification
Execute `python test_multiclass_classifier/multiclassifier.py`, using the required type of classification script.

# NOTES AND TODO's

# Refactor from pandas to hugginface datasets from scratch to completely remove pandas dependency
# TODO: Consider class weighted splitting. This probably need to be explored especailly if doing multiclass classification 
# TODO: Infer signature from MLFlow is fixed. However, currently confusion matrix elemenets are "flattened" this is because only scalars are accepted. We need to somehow reconstruct it or save this in a different way. 
# TODO: tidy up MPS / CUDA etc. Currently, line 226 is executed to make it work. There must be a more elegant solution.
# NOTE TO SELF: if using trainer.evaluate() no need for seperate evalaution metrics