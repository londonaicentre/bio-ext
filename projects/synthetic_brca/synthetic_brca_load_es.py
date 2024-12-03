import os
import tqdm
import json
from bioext.elastic_utils import ElasticsearchSession
from elasticsearch import helpers

from dotenv import load_dotenv
load_dotenv()

DATA_FILE = 'data/brca_reports.json'

def create_index(client):
    """Creates an index in Elasticsearch if one isn't already there."""
    client.indices.create(
        index="brca_synth",
        body={
            "settings": {"number_of_shards": 1},
            "mappings": {
                "properties": {
                    "seed": {"type": "integer"},
                    "text": {"type": "text"},
                }
            },
        },
        ignore=400,
    )


def generate_actions():
    """Reads the file through csv.DictReader() and for each row
    yields a single document. This function is passed into the bulk()
    helper to create many documents in sequence.
    """
    # load json from data file
    try:
        with open(DATA_FILE, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Failed to load samples: {str(e)}")

    for doc in data:
        print(doc)
        print("-----")
        yield doc

def main():
    # connect and log on to ElasticSearch
    print("Connecting to ElasticSearch")
    es_session = ElasticsearchSession()
    
    print("Creating an index...")
    create_index(es_session.es)
    
    # load json to ElasticSearch
    print("Indexing documents...")
    progress = tqdm.tqdm(unit="docs", total=100)
    successes = 0
    for ok, action in helpers.streaming_bulk(
        client=es_session.es,
        index="brca_synth",
        actions=generate_actions(),
    ):
        progress.update(1)
        successes += ok
    print("Indexed %d/%d documents" % (successes, 100))


if __name__ == "__main__":
    main()
    print("\Ingestion complete")