import argparse
from tqdm import tqdm
import json
import os
from bioext.elastic_utils import ElasticsearchSession
from bioext.doccano_utils import (
    DoccanoSession,
    load_from_file,
    stream_labelled_docs,
    save_labelled_docs,
)
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

    # Parsing command line args for ES2Doc subcommand
    parser_ESDoc = subparsers.add_parser("ES2Doc", help="help")
    parser_ESDoc.add_argument(
        "sample_size",
        default=1000,
        help="Number of samples to load",
    )
    parser_ESDoc.set_defaults(subcommand="ES2Doc")

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

    # Parsing command line args for Doc_save subcommand
    parser_Dsa = subparsers.add_parser("Doc_save", help="help")
    parser_Dsa.add_argument(
        "--output_file",
        default="data/breast_brca_labelled.json",
        help="Path to folder to save results into",
        required=False,
    )
    parser_Dsa.set_defaults(subcommand="Doc_save")

    args = parser.parse_args()
    return args


def load_es_from_file(es_session, es_load_cfg, data_file_path):
    """Load synthetic documents into Elasticsearch"""

    print("Creating index...")
    es_session.create_index(
        index_name=es_load_cfg["index_name"],
        mappings=es_load_cfg["mappings"],
        overwrite=True,
    )

    # load and parse json
    try:
        with open(data_file_path, "r") as file:
            documents = json.load(file)
    except Exception as e:
        print(f"Failed to load samples: {e}")
        return

    progress = tqdm(unit="docs", total=len(documents))

    # load into index
    print("Indexing documents...")
    successes = es_session.bulk_load_documents(
        index_name=es_load_cfg["index_name"],
        documents=documents,
        progress_callback=progress.update,
    )

    print(f"Indexed {successes}/{len(documents)} documents")
    return successes


def es2doc(config, sample_size=100):
    """
    1. Create a new Doccano project
    2. Query ElasticSearch for matching documents
    3. Load random sample into Doccano
    4. outputs project variables as a dict after generating timestamp
    """

    # connect to Elastic and Doccano
    es_session = ElasticsearchSession()
    print(f"Connected to Elastic as user: {es_session.es_user}")

    doc_session = DoccanoSession()
    print(f"Connected to Doccano as user: {doc_session.username}")

    # retrieve configurations
    es_query_config = config["ElasticSearch"]["retrieve"]["breast_brca_query"]
    doc_load_cfg = config["Doccano"]["load"]

    # get random document IDs from Elastic using query
    print(f"Getting {sample_size} random document IDs using query...")
    random_ids = es_session.get_random_doc_ids(
        index_name=es_query_config["index_name"],
        size=int(sample_size),
        query=es_query_config["query"],
    )

    # Load documents into Doccano
    # Create project and create labels
    project = doc_session.create_or_update_project(**doc_load_cfg)
    print(f"Using project: {project.name}, with ID {project.id}")

    # Loading documents
    print(f"Loading {len(random_ids)} documents into Doccano...")
    successful_loads = 0
    failed_loads = 0

    content_field = es_query_config["content_field"]

    for doc_id in random_ids:
        try:
            doc = es_session.get_document_by_id(
                index_name=es_query_config["index_name"], doc_id=doc_id
            )
            if doc and content_field in doc["_source"]:
                text = doc["_source"][content_field]
                doc_session.load_document(text, metadata={"source_id": doc_id})
                successful_loads += 1
            else:
                failed_loads += 1
                print(f"Document {doc_id} failed to load...")
        except Exception as e:
            failed_loads += 1
            print(f"Document {doc_id} failed to be retrieved: {e}")

    print(f"Success: {successful_loads}")
    print(f"Failed: {failed_loads}")


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
        assert "Doccano" in app_config.keys()
        assert "load" in app_config["Doccano"].keys()

    if args.subcommand.startswith("ES"):
        # Initialise a connection to ES server with env credentials
        # connect and log on to ElasticSearch
        print("Connecting to ElasticSearch")
        es_session = ElasticsearchSession()

        if args.subcommand == "ES_load":
            es_load_cfg = app_config["ElasticSearch"]["load"]
            load_es_from_file(es_session, es_load_cfg, args.data)
            print("Ingestion complete")

        elif args.subcommand == "ES_query":
            es_query_cfg = app_config["ElasticSearch"]["retrieve"]["breast_brca_query"]

            # retrieve documents based on query
            es_session.bulk_retrieve_documents(
                es_query_cfg["index_name"],
                es_query_cfg["query"],
                save_to_file=args.output_dir,
            )

        elif args.subcommand == "ES2Doc":
            es2doc(app_config, args.sample_size)

    elif args.subcommand.startswith("Doc"):
        # Initialise connection to Doccano
        doc_session = DoccanoSession()

        if args.subcommand == "Doc_load":
            doc_load_cfg = app_config["Doccano"]["load"]
            load_from_file(doc_session, args.data, doc_load_cfg)

        elif args.subcommand == "Doc_stream":
            doc_stream_cfg = app_config["Doccano"]["retrieve"]
            stream_labelled_docs(doc_session, doc_stream_cfg)

            print("Labelled data streaming complete")

        elif args.subcommand == "Doc_save":
            print(args.output_file)
            with open("Doccano_project.json") as f:
                doc_metadata = json.load(f)
            project_id = doc_metadata["Project ID"]
            save_labelled_docs(doc_session, project_id, args.output_file)

            print("Labelled data saved to file.")
