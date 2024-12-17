import argparse
import json
import sys
from bioext.elastic_utils import ElasticsearchSession
from synthetic_brca_load_ES import create_index, yield_doc, load_docs_from_file
from dotenv import load_dotenv


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

    subparsers = parser.add_subparsers()
    # Parsing command line args for ES_load subcommand
    parser_ESl = subparsers.add_parser("ES_load", help="help")
    parser_ESl.add_argument(
        "-d",
        "--data",
        default="data/brca_reports.json",
        help="Path to file to load documents from, expected to be JSON",
    )

    # parser_ESl.set_defaults(which='ES_load')

    args = parser.parse_args()
    return args


def load_es(es_load_cfg, data_file_path):
    # connect and log on to ElasticSearch
    print("Connecting to ElasticSearch")
    es_session = ElasticsearchSession()

    # print(es_load_cfg)

    print("Creating an index...")
    create_index(es_session.es, es_load_cfg)

    # load json to ElasticSearch
    print("Indexing documents...")
    successes = load_docs_from_file(es_session.es, data_file_path, es_load_cfg)
    print("Indexed %d/%d documents" % (successes, 100))


if __name__ == "__main__":
    # Load credentials from env file
    load_dotenv()

    # Read the arguments from CLI
    args = parse_CLI_args()

    # load config from file path provided (or default)
    with open(args.config) as json_data:
        app_config = json.load(json_data)
        assert "ElasticSearch" in app_config.keys()
        assert "load" in app_config["ElasticSearch"].keys()

    subcommand = sys.argv[-1]
    if subcommand == "ES_load":
        es_load_cfg = app_config["ElasticSearch"]["load"]
        load_es(es_load_cfg, args.data)
        print(r"\Ingestion complete")
