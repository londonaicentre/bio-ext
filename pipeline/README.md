This folder contains production code for deploying NLP pipelines and serving models, and processing batches of documents.

Each pipeline is contained in a subfolder.

A basic pipeline may include:
1. Query to extract selection of documents from source db (can be incremental)
2. Data pre-processing
3. Pipeline constructor (or model serving)
4. Passing data to pipeline or model endpoint
5. Staging NLP outputs + metadata in bio-ext elastic
6. Processing outputs into structured tabular content, including further NLP steps +/- entity linkage