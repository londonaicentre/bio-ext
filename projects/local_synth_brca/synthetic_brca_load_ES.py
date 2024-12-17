import tqdm
import json
from elasticsearch import helpers


def create_index(es_client, config):
    """Creates an index in Elasticsearch if one isn't already there."""
    es_client.indices.create(
        index=config["index_name"],
        body={
            "settings": {"number_of_shards": 1},
            "mappings": config["mappings"],
        },
        ignore=400,
    )


def yield_doc(data_file_path):
    """Reads the file through csv.DictReader() and for each row
    yields a single document. This function is passed into the bulk()
    helper to create many documents in sequence.
    """
    # load json from data file
    try:
        with open(data_file_path, "r") as file:
            data = json.load(file)
        for doc in data:
            print(doc)
            print("-----")
            yield doc
    except Exception as e:
        print(f"Failed to load samples: {str(e)}")


def load_docs_from_file(es_client, data_file_path, config):
    progress = tqdm.tqdm(unit="docs", total=100)
    successes = 0
    for ok, action in helpers.streaming_bulk(
        client=es_client,
        index=config["index_name"],
        actions=yield_doc(data_file_path),
    ):
        progress.update(1)
        successes += ok
    return successes


if __name__ == "__main__":
    print("Ingestion complete")
