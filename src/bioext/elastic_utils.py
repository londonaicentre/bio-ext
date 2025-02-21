import os
import json
from elasticsearch import Elasticsearch, helpers
from elastic_transport import RequestsHttpNode
import requests
import random


# thanks @LAdams for implementing required http proxy
class GsttProxyNode(RequestsHttpNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proxy_endpoint = os.getenv("http_proxy")
        self.session.proxies = {
            "http": self.proxy_endpoint,
            "https": self.proxy_endpoint,
        }


class ElasticsearchSession:
    def __init__(self, proxy=None, conn_mode: str = "HTTP"):
        requests.packages.urllib3.disable_warnings(
            requests.packages.urllib3.exceptions.InsecureRequestWarning
        )

        # set to GSTT server by default
        self.es_server = os.getenv("ELASTIC_SERVER", "https://sv-pr-elastic01:9200")

        if proxy:
            self.proxy_node = GsttProxyNode
        else:
            self.proxy_node = None

        if conn_mode == "API":
            self.api_id = os.getenv("ELASTIC_API_ID")
            self.api_key = os.getenv("ELASTIC_API_KEY")

            if not all([self.api_id, self.api_key]):
                raise ValueError(
                    "Check that ELASTIC_API_ID and ELASTIC_API_KEY are in env variables"
                )

            self.es = Elasticsearch(
                hosts=self.es_server,
                api_key=(self.api_id, self.api_key),
                node_class=self.proxy_node,
                verify_certs=False,
                ssl_show_warn=False,
            )

        elif conn_mode == "HTTP":
            self.es_user = os.getenv("ELASTIC_USER")
            self.es_pwd = os.getenv("ELASTIC_PWD")

            if not all([self.es_user, self.es_pwd]):
                raise ValueError("Check ELASTIC_USER and ELASTIC_PWD are in env variables")

            self.es = Elasticsearch(
                hosts=self.es_server,
                # http_auth has been deprecated
                basic_auth=(
                    self.es_user,
                    self.es_pwd,
                ),
                verify_certs=False,
                ssl_show_warn=False,
            )

        else:
            raise ValueError("Argument conn_mode must be 'HTTP' or 'API'")

        # Ensure that connection can actually be made with the creds provided
        # assert self.es.ping() is True, print(
        #     "Could not connect with credentials provided"
        # )

    def create_index(self, index_name, mappings, settings=None, overwrite=False):
        """
        Creates an index in Elasticsearch with option to overwrite existing one.
        Requires a config input (mappings) that describes fields, e.g.:
            "mappings": {
                "properties": {
                    "seed": {"type": "integer"},
                    "text": {"type": "text"}
                }
            }
        """
        if settings is None:
            settings = {"number_of_shards": 1}

        if overwrite:
            # delete index if it exists
            self.es.indices.delete(index=index_name, ignore=[400, 404])

        self.es.indices.create(
            index=index_name,
            body={
                "settings": settings,
                "mappings": mappings,
            },
            ignore=400,
        )

    def list_indices(self):
        return self.es.indices.get_alias(index="*")

    def _yield_doc(self, data_file_path):
        """Reads the file through csv.DictReader() and for each row
        yields a single document. This function is passed into the bulk()
        helper to create many documents in sequence.
        """
        # load json from data file
        try:
            with open(data_file_path, "r") as file:
                data = json.load(file)
                yield from data
        except Exception as e:
            print(f"Failed to load samples: {str(e)}")

    def bulk_load_documents(self, index_name, documents, progress_callback=None):
        """
        Bulk load documents into Elasticsearch.
        """

        def doc_generator():
            for doc in documents:
                yield doc

        successes = 0
        for ok, action in helpers.streaming_bulk(
            client=self.es,
            index=index_name,
            actions=doc_generator(),
        ):
            successes += ok
            if progress_callback:
                progress_callback(1)

        return successes

    def bulk_retrieve_documents(self, index_name, query, scroll="2m", save_to_file=None):
        """
        Retrieve documents from Elasticsearch using scroll API
        """
        docs = helpers.scan(
            client=self.es,
            query={"query": query},
            scroll=scroll,
            index=index_name,
        )

        # save queried documents to file
        if save_to_file is not None:
            os.makedirs(save_to_file, exist_ok=True)
            processed_count = 0
            for hit in docs:
                doc_id = hit["_id"]
                file_path = os.path.join(save_to_file, f"{doc_id}.json")
                with open(file_path, "w") as f:
                    json.dump(hit, f, indent=2)
                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"Up to {processed_count} docs...")
            print(f"{processed_count} docs were downloaded")

        return docs

    def get_random_doc_ids(self, index_name, size, query=None):
        """
        Get a random subset of document IDs from a given document index
        """
        # If no query provided, match all documents
        if query is None:
            query = {"match_all": {}}

        # return all document IDs first!
        all_ids = [
            doc["_id"]
            for doc in self.bulk_retrieve_documents(
                index_name=index_name,
                query=query,
            )
        ]

        # random sample
        return random.sample(all_ids, min(size, len(all_ids)))

    def get_document_by_id(self, index_name, doc_id):
        """
        Retrieve single document based on its ID
        """
        try:
            return self.es.get(index=index_name, id=doc_id)
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {e}")
            return None
