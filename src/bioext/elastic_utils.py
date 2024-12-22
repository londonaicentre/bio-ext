import os
import tqdm
import json
from elasticsearch import Elasticsearch, helpers
from elastic_transport import RequestsHttpNode
import requests


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
                raise ValueError(
                    "Check ELASTIC_USER and ELASTIC_PWD are in env variables"
                )

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

    def create_index(self, config):
        """Creates an index in ElasticSearch if one isn't already there."""

        self.es.indices.create(
            index=config["index_name"],
            body={
                "settings": {"number_of_shards": 1},
                "mappings": config["mappings"],
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

    def load_docs_from_file(self, data_file_path, config):
        progress = tqdm.tqdm(unit=" docs", total=100)
        successes = 0
        for ok, action in helpers.streaming_bulk(
            client=self.es,
            index=config["index_name"],
            actions=self._yield_doc(data_file_path),
        ):
            progress.update(1)
            successes += ok
        return successes

    def retrieve_docs(self, project_dir, query_cfg):
        index_name = query_cfg["index_name"]
        query = {"query": query_cfg["query"]}
        try:
            results = helpers.scan(
                client=self.es, query=query, scroll="2m", index=index_name
            )

            processed_count = 0
            for hit in results:
                doc_id = hit["_id"]
                file_path = os.path.join(project_dir, f"{doc_id}.json")

                with open(file_path, "w") as f:
                    json.dump(hit, f, indent=2)

                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"Up to {processed_count} docs...")

            print(f"\nTotal: {processed_count} docs")

        except Exception as e:
            print(f"Error: {str(e)}")
