import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional

import requests
from elastic_transport import RequestsHttpNode
from elasticsearch import Elasticsearch, helpers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GsttProxyNode(RequestsHttpNode):
    """Custom RequestsHttpNode to handle proxy settings for Elasticsearch at GSTT.

    Requires the `http_proxy` environment variable to be set."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proxy_endpoint = os.getenv("http_proxy")
        self.session.proxies = {
            "http": self.proxy_endpoint,
            "https": self.proxy_endpoint,
        }


class BaseElasticsearchSession:
    """Base class for Elasticsearch sessions with common functionality"""

    def __init__(
        self,
        elasticsearch_server: str = os.getenv(
            "ELASTICSEARCH_SERVER", "https://sv-pr-elastic01.gstt.local:9200"
        ),
        proxy_node: Optional[RequestsHttpNode] = None,
        elasticsearch_client: Optional[Elasticsearch] = None,
    ) -> None:
        self.es_server = elasticsearch_server
        self.proxy_node = proxy_node
        requests.packages.urllib3.disable_warnings(
            requests.packages.urllib3.exceptions.InsecureRequestWarning
        )
        if elasticsearch_client:
            self.es = elasticsearch_client
        else:
            self._configure_client()

    def _configure_client(self):
        """Abstract method to be implemented by child classes"""
        raise NotImplementedError

    def create_index(
        self,
        index_name: str,
        mappings: dict[Any],
        settings: dict[Any] = None,
        overwrite: bool = False,
    ):
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

    def bulk_load_documents(
        self,
        index_name: str,
        documents: Iterable,
        progress_callback: None | Callable = None,
    ) -> int:
        """
        Bulk load documents into Elasticsearch index using streaming_bulk.
        This function returns the number of successful document loads, whilst providing a
        callback to track progress.

        Args:
            index_name (str): The name of the Elasticsearch index.
            documents (Iterable): An iterable of documents to be indexed. (e.g. list of dicts)
            progress_callback (callable, optional): A callback function to track progress.
                It should accept a single argument indicating the number of documents processed.
                Can be useful for displaying progress in a GUI or logging. (e.g. tqdm)
        Returns:
            int: The number of successfully indexed documents.
        Example:
            >>> es.bulk_load_documents(
                    index_name="my_index",
                    documents=[{"doc1": "data"}, {"doc2": "data"}],
                    progress_callback=lambda x: print(f"Processed {x} documents")
                )
        """

        def doc_generator():
            for doc in documents:
                yield doc

        successes = 0
        for ok, _ in helpers.streaming_bulk(
            client=self.es,
            index=index_name,
            actions=doc_generator(),
        ):
            successes += ok
            if progress_callback:
                progress_callback(1)

        return successes

    def bulk_retrieve_documents(
        self,
        index_name: str,
        query: dict,
        scroll: str = "2m",
        save_to_file: None | str | Path = None,
    ) -> Iterable[dict[str, Any]]:
        """
        Retrieve documents from Elasticsearch using scroll API, and optionally save them to files.

        Args:
            index_name (str): The name of the Elasticsearch index.
            query (dict): The query to filter the documents.
            scroll (str): The scroll time for the search context. Default is "2m".
            save_to_file (str or Path, optional): Directory to save the retrieved documents.
                If None, documents are not saved to files.

        Returns:
            Iterable[dict]: An iterable of documents retrieved from Elasticsearch.

        Example:
            >>> es.bulk_retrieve_documents(
                    index_name="my_index",
                    query={"match_all": {}},
                    scroll="2m",
                    save_to_file="/path/to/save"
                )
        """
        docs = helpers.scan(
            client=self.es,
            query=query,
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
                    logger.debug(f"Up to {processed_count} docs...")
            logger.info(f"{processed_count} docs were downloaded")

        return docs

    def get_random_doc_ids(
        self, index_name: str, size: int = 100, query: None | dict[Any] = None
    ) -> list[str]:
        """
        Retrieve random document IDs from the specified index.

        Args:
            index_name (str): The name of the Elasticsearch index.
            size (int): The number of random document IDs to retrieve. Defaults to 100.
            query (dict[Any], optional): An optional query to filter the documents.
                Note: The query should be a valid Elasticsearch query context/predicate only - not a complete query.
                See here for more: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-filter-context.html
        Returns:
            list[str]: A list of random document IDs.

        Example:
            >>> es.get_random_doc_ids("my_index", size=10, query={"match": {"document_Content": "BRCA"}})
        """
        # Query to retrieve random documents using random_score
        if size > 10_000:
            raise ValueError(
                "This method is not designed to get large numbers of random values. Reduce size <= 10,000"
            )

        query = {
            "_source": False,
            "size": size,
            "query": {
                "function_score": {
                    "query": query if query else {"match_all": {}},
                    "functions": [{"random_score": {}}],
                }
            },
        }

        res = self.es.search(
            index=index_name,
            body=query,
        )

        random_ids = [doc["_id"] for doc in res["hits"]["hits"]]

        return random_ids


# == Override classes for different authentication methods ==
class ElasticsearchManualAuthSession(BaseElasticsearchSession):
    """Elasticsearch session with manually configured authentication"""

    def __init__(self, elasticsearch_client: Elasticsearch) -> None:
        self.es = elasticsearch_client


class ElasticsearchApiAuthSession(BaseElasticsearchSession):
    """Elasticsearch session using API key authentication"""

    def _configure_client(self):
        print("SETUP")
        api_id = os.getenv("ELASTIC_API_ID")
        api_key = os.getenv("ELASTIC_API_KEY")

        if not all([api_id, api_key]):
            raise ValueError(
                "Check ELASTIC_API_ID and ELASTIC_API_KEY are in env variables"
            )

        self.es = Elasticsearch(
            hosts=self.es_server,
            api_key=(api_id, api_key),
            node_class=self.proxy_node,
            verify_certs=False,
            ssl_show_warn=False,
        )


class ElasticsearchUserAuthSession(BaseElasticsearchSession):
    """Elasticsearch session using username/password authentication"""

    def _configure_client(self):
        es_user = os.getenv("ELASTIC_USER")
        es_pwd = os.getenv("ELASTIC_PWD")

        if not all([es_user, es_pwd]):
            raise ValueError("Check ELASTIC_USER and ELASTIC_PWD are in env variables")

        self.es = Elasticsearch(
            hosts=self.es_server,
            basic_auth=(es_user, es_pwd),
            verify_certs=False,
            ssl_show_warn=False,
        )


class ElasticsearchSession(BaseElasticsearchSession):
    """Deprecated: Use ElasticsearchApiAuthSession or ElasticsearchUserAuthSession instead"""

    def __init__(
        self,
        proxy_node: RequestsHttpNode | None = None,
        conn_mode: Literal["HTTP"] | Literal["API"] = "HTTP",
    ) -> None:
        self.proxy_node = proxy_node

        logger.warn(
            "ElasticsearchSession is deprecated. Use ElasticsearchApiAuthSession or ElasticsearchUserAuthSession instead",
        )

        requests.packages.urllib3.disable_warnings(
            requests.packages.urllib3.exceptions.InsecureRequestWarning
        )

        # set to GSTT server by default
        self.es_server = os.getenv(
            "ELASTIC_SERVER", "https://sv-pr-elastic01.gstt.local:9200"
        )

        # Use optional proxy node (useful if running in Proxied Environment)
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
