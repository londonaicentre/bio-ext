# Cogstack Dash 

## What it does
It does 4 things

>Streamlit dashboard
- query and EDA cogstack database 
- help you simplify the workflow
- use by running Hello.py
(see below for details)

> EDA script
- generate eda of basic fields/i.e., columns of all indexes from cogtack
- use `gen_eda.py` and can be use from command line

> Query Checking
- Jupyter notebook to help download queries and explore.
- use `querychecker.ipynb`

> Experimental: Create embeddings from text, dimension reduce and visualise them
- this is currently experimental at this stage. 
- use `eda_experimental.py` 

## Requirements
- will require custom `.env` with own secrets and if you dont have, need to ask from admin. 
- will require coder environment to work within gstt local 
- currently pyproject.toml as not fully up-to-date but will need ["streamlit","pandas","bio-ext"]. Note: Bio-ext is local. 

## Dir structure 
```bash

├── Hello.py
├── README.md
├── csdash
│   ├── __init__.py
│   ├── dbaccess.py
│   ├── nlp.py
│   └── transforms.py
├── eda_experimental.py
├── gen_eda.py
├── pages
│   ├── 1_Indexes.py
│   ├── 2_Queries.py
│   └── 3_Searcher.py
├── pyproject.toml
├── querychecker.ipynb
├── test.py
├── utils
│   ├── config_dash.yaml
│   └── gstt_elastic_directory.md
└── uv.lock
```

## Feature 1: Streamlit app

* 3 pages at the moment, structured in `pages/` 
* `csdash` is a module that has the required functions
* `utils` hold yaml and markdown to make streamlit app less texty. 
* `congif_dash.yaml` has basic configs saved without any secrets
* `Hello.py` is an entry into the project.

### Run via
at project root execute `streamlit run Hello.py`

## Feature 2: EDA script
* standalone run `python3 query_gen.py` 
* can use -help function to find cli arguments 

## Feature 3: Query Checker.ipynb
* Standalone usage as jupyter notebook
* need activating the venv so dependencies are loaded properly.

## Feature 4: Experimental EDA
* Ideally we want to infer schema and that is general.
* this script `eda_experimental.py` convert text data to embeddings, dimension reduce using UMAP and then help visualise using altair.

## Known Bugs
- Streamlit converts to pyarrow dataframes and one of the data frames has Na/zero which the conversion has difficulty. 
- On Page Indexes, 

## Future
- I will probably make a bash script to help install easier.
- I will debug the above known bug. 
- best to transition to TOML from yaml and fix the dependencies properly 


