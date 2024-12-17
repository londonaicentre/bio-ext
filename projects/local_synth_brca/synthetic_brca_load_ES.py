import os
import tqdm
import argparse
import json
from bioext.elastic_utils import ElasticsearchSession
from elasticsearch import helpers

from dotenv import load_dotenv


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
    except Exception as e:
        print(f"Failed to load samples: {str(e)}")

    for doc in data:
        print(doc)
        print("-----")
        yield doc


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


def parse_CLI_args():  # -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        args : Namespace
            Namespace of passed command line argument inputs
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="/config",
        help="Path to directory of config files",
    )

    parser.add_argument(
        "-l",
        "--log_level",
        choices=["info", "debug", "warning", "error", "critical"],
        default="info",
        help="Level of logs to be displayed",
    )

    parser.add_argument(
        "-d",
        "--data",
        default="data/brca_reports.json",
        help="Path to file to load documents from, expected to be JSON",
    )

    args = parser.parse_args()
    return args


def main():
    # Read the arguments from the CLI
    args = parse_CLI_args()

    # load config from file path provided (or default)
    with open(args.config) as json_data:
        app_config = json.load(json_data)
        assert "ElasticSearch" in app_config.keys()
        assert "load" in app_config["ElasticSearch"].keys()
        es_load_cfg = app_config["ElasticSearch"]["load"]

    # Load credentials from env file
    load_dotenv()
    # connect and log on to ElasticSearch
    print("Connecting to ElasticSearch")
    es_session = ElasticsearchSession()

    print(es_load_cfg)

    print("Creating an index...")
    create_index(es_session.es, es_load_cfg)

    # load json to ElasticSearch
    print("Indexing documents...")
    successes = load_docs_from_file(es_session.es, args.data, es_load_cfg)
    print("Indexed %d/%d documents" % (successes, 100))


if __name__ == "__main__":
    main()
    print("\Ingestion complete")
