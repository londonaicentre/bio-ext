import argparse
import json
import os
from synthetic_brca_load_Docc import file2Doc
from synthetic_brca_stream_Docc import stream_labelled_docs
from bioext.elastic_utils import ElasticsearchSession
from bioext.doccano_utils import DoccanoSession
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
    parser_ESl.set_defaults(subcommand="ES_load")

    # Parsing command line args for ES_query subcommand
    parser_ESq = subparsers.add_parser("ES_query", help="help")
    parser_ESq.add_argument(
        "output_dir",
        default="data/breast_brca_status",
        help="Path to folder to save results into",
    )
    parser_ESq.set_defaults(subcommand="ES_query")

    # Parsing command line args for Doc_load subcommand
    parser_Dl = subparsers.add_parser("Doc_load", help="help")
    parser_Dl.add_argument(
        "-d",
        "--data",
        default="data/brca_reports.json",
        help="Path to file to load documents from, expected to be JSON",
    )
    parser_Dl.set_defaults(subcommand="Doc_load")

    # Parsing command line args for Doc_stream subcommand
    parser_Ds = subparsers.add_parser("Doc_stream", help="help")
    parser_Ds.set_defaults(subcommand="Doc_stream")

    args = parser.parse_args()
    return args


def load_es(es_session, es_load_cfg, data_file_path):
    print("Creating an index...")
    es_session.create_index(es_load_cfg)

    # load json to ElasticSearch
    print("Indexing documents...")
    successes = es_session.load_docs_from_file(data_file_path, es_load_cfg)
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

    if args.subcommand.startswith("ES"):
        # Initialise a connection to ES server with env credentials
        # connect and log on to ElasticSearch
        print("Connecting to ElasticSearch")
        es_session = ElasticsearchSession()

        if args.subcommand == "ES_load":
            es_load_cfg = app_config["ElasticSearch"]["load"]
            load_es(es_session, es_load_cfg, args.data)
            print("Ingestion complete")

        elif args.subcommand == "ES_query":
            es_query_cfg = app_config["ElasticSearch"]["retrieve"]["breast_brca_query"]
            os.makedirs(args.output_dir, exist_ok=True)

            # retrieve and save to file the queried documents
            es_session.retrieve_docs(args.output_dir, es_query_cfg)

    elif args.subcommand.startswith("Doc"):
        # Initialise connection to Doccano

        if args.subcommand == "Doc_load":
            doc_load_cfg = app_config["Doccano"]["load"]
            doc_session = DoccanoSession()
            file2Doc(doc_session, args.data, doc_load_cfg)

            print("Doccano project setup complete")

        elif args.subcommand == "Doc_stream":
            doc_stream_cfg = app_config["Doccano"]["retrieve"]
            doc_session = DoccanoSession()
            stream_labelled_docs(doc_session, doc_stream_cfg)

            print("Labelled data streaming complete")
