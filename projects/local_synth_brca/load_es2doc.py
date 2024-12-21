from bioext.elastic_utils import ElasticsearchSession
from bioext.doccano_utils import DoccanoSession
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
import json

"""
Create a new Doccano project and populate it with random documents from Elasticsearch
based on specified query. This is the first step to create annotation projects.

Usage:
    python load_es2doc.py -c CONFIG_PATH -n NUMBER_OF_SAMPLES

Args:
    -c, --config    Path to config file (.json) containing Elasticsearch query and Doccano project settings
    -n, --number    Number of random documents to sample (default: 100)

Example:
    python load_es2doc.py -c config.json -n 50
"""

def es2doc(config, sample_size=100):
    """
    1. Create a new Doccano project
    2. Query Elasticsearch for matching documents
    3. Load random sample into Doccano
    """

    # connect to doccano and elastic
    es_session = ElasticsearchSession()
    print(f"Connected to Elastic as user: {es_session.es_user}")

    doc_session = DoccanoSession()
    print(f"Connected to Doccano as user: {doc_session.username}")

    # retrieve configurations
    doc_project_config = config["Doccano"]["project_setup__synthetic_brca"]
    es_query_config = config["ElasticSearch"]["query__retrieve_breast_brca"]

    # create project
    project = doc_session.create_project(**doc_project_config["PROJECT_DETAILS"])
    print(f"Created new project: {project.name}, {project.id}")

    # set up labels
    doc_session.setup_labels(doc_project_config["LABELS"])
    print(f"Created {len(doc_project_config['LABELS'])} labels")
    
    # get random document IDs from elastic using query
    print("Getting random document IDs using query...")
    random_ids = es_session.get_random_doc_ids(
        index_name=es_query_config["index_name"],
        size=sample_size,
        query=es_query_config["query"]
    )

    # Load documents into Doccano
    print(f"Loading {len(random_ids)} documents into Doccano...")
    
    successful_loads = 0
    failed_loads = 0

    content_field = es_query_config["content_field"]
    
    for doc_id in tqdm(random_ids):
        try:
            doc = es_session.get_document_by_id(
                index_name=es_query_config["index_name"],
                doc_id=doc_id
            )
            if doc and content_field in doc["_source"]:
                text = doc["_source"][content_field]
                doc_session.load_document(text)
                successful_loads += 1
            else:
                failed_loads += 1
                print(f"Document {doc_id} failed to load...")
        except Exception as e:
            failed_loads += 1
            print(f"Document {doc_id} failed to be retrieved: {e}")
    
    print(f"Success: {successful_loads}")
    print(f"Failed: {failed_loads}")
    print(f"Doccano Project ID: {project.id}")
    return project.id

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="path to config file")
    parser.add_argument("-n", "--number", type=int, default=100, help="number of random documents to sample")
    args = parser.parse_args()
    
    # load configs
    with open(args.config) as f:
        config = json.load(f)
    
    project_id = es2doc(config, args.number)