# Cogstack Dash 

## What it does

>Streamlit dashboard
- query and EDA cogstack database 
- help you simplify the workflow
- run Hello.py

> EDA script
- yaml generator

## Requirements
- will require custom `.env` with own secrets
- will require coder environment to work within gstt local 
- [need work on sorting pyproject.toml dependecies properly] for now pandas, streamlit, bio-ext elastic utils

## Multi-page app

## Dir structure
* 3 pages at the moment, structured in `pages/` 
* `congif_dash.yaml` has basic configs saved without any secrets
* `Hello.py` is an entry into the project.

## Run via
at project root execute `streamlit run Hello.py`

## Known Bugs
- Streamlit converts to pyarrow dataframes and one of the data frames has Na/zero which the conversion has difficulty. 

## Future
I will probably refactor to make it more maintainable with proper folder structure reflecting Flask apps
- I will probably make a bash script to help install easier.
- best to transition to TOML from yaml 
- ideally gstt_elastic_directory.md will sync with copy from `sde_aic_internal_docs`
- could do with better caching [now caching is only works partially after refactoring]
- ideally cache global vars at hello.py
- need to wrap Try to the functions so it doesnt break 

## Dir structure 
```bash
.
├── Hello.py
├── README.md
├── csdash
│   ├── __init__.py
│   ├── dbaccess.py
│   ├── nlp.py
│   └── transforms.py
├── pages
│   ├── 1_Indexes.py
│   ├── 2_Queries.py
│   └── 3_Searcher.py
├── pyproject.toml
└── utils
    ├── config_dash.yaml
    └── gstt_elastic_directory.md
```
