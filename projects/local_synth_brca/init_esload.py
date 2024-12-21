import json
from tqdm import tqdm
from bioext.elastic_utils import ElasticsearchSession
import argparse
from dotenv import load_dotenv

"""
Use this script to perform initial set up when working in a local dev environment
This loads synthetic data (as a .json) into a local Elastic database 

Usage:
    python init_esload.py -c CONFIG_PATH -d DATA_PATH

Args:
    -c, --config    Path to config file (.json)
    -d, --data      Path to data file (.json)

Example:
    python init_esload.py -c config.json -d /data/brca_reports.json
"""

def load_synthetic_data_into_elastic(config, data):
    """Load synthetic documents into Elasticsearch"""
    
    print("Connecting to ElasticSearch")
    es_session = ElasticsearchSession()

    print("Creating index...")
    es_session.create_index(
        index_name=config["index_name"],
        mappings=config["mappings"],
        overwrite=True
    )

    # load and parse json
    try:
        with open(data, "r") as file:
            documents = json.load(file)
    except Exception as e:
        print(f"Failed to load samples: {e}")
        return

    progress = tqdm(unit="docs", total=len(documents))
    
    # load into index
    print("Indexing documents...")
    successes = es_session.bulk_load_documents(
        index_name=config["index_name"],
        documents=documents,
        progress_callback=progress.update
    )
    
    print(f"Indexed {successes}/{len(documents)} documents")
    return successes

if __name__ == "__main__":
    load_dotenv()
    
    # parsing arguments per script
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="path to config file")
    parser.add_argument("-d", "--data", required=True, help="path to data file")
    args = parser.parse_args()
    
    # load config
    with open(args.config) as f:
        config = json.load(f)["ElasticSearch"]["create_index__synthetic_brca"]
    
    load_synthetic_data_into_elastic(config, args.data)