import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Union

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
        elasticsearch_server: str = None,
        proxy_node: Optional[RequestsHttpNode] = None,
        elasticsearch_client: Optional[Elasticsearch] = None,
    ) -> None:
        self.es_server = elasticsearch_server or os.getenv(
            "ELASTICSEARCH_SERVER", "https://sv-pr-elastic01.gstt.local:9200"
        )
        self.proxy_node = proxy_node

        # Disable SSL warnings
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
        mappings: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ):
        """
        Creates an index in Elasticsearch with option to overwrite existing one.

        Args:
            index_name: Name of the index to create
            mappings: Field mappings configuration
            settings: Index settings (defaults to single shard)
            overwrite: Whether to overwrite existing index
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
        """List all indices in the Elasticsearch cluster"""
        return self.es.indices.get_alias(index="*")

    def bulk_load_documents(
        self,
        index_name: str,
        documents: Iterable[Dict[str, Any]],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> int:
        """
        Bulk load documents into Elasticsearch index using streaming_bulk.

        Args:
            index_name: The name of the Elasticsearch index
            documents: An iterable of documents to be indexed
            progress_callback: A callback function to track progress

        Returns:
            Number of successfully indexed documents

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
        query: Dict[str, Any],
        scroll: str = "2m",
        save_to_file: Optional[Union[str, Path]] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Retrieve documents from Elasticsearch using scroll API, and optionally save them to files.

        Args:
            index_name: The name of the Elasticsearch index
            query: The query to filter the documents
            scroll: The scroll time for the search context. Default is "2m"
            save_to_file: Directory to save the retrieved documents

        Returns:
            An iterable of documents retrieved from Elasticsearch
        """
        docs = helpers.scan(
            client=self.es,
            query=query,
            scroll=scroll,
            index=index_name,
        )

        # Return immediately if not saving to file
        if save_to_file is None:
            return docs

        # If saving to file, we need to process and return a new generator
        return self._save_documents_to_file(docs, save_to_file)

    def _save_documents_to_file(self, docs, save_path):
        """Helper method to save documents to files and yield them"""
        os.makedirs(save_path, exist_ok=True)
        processed_count = 0

        for hit in docs:
            doc_id = hit["_id"]
            file_path = os.path.join(save_path, f"{doc_id}.json")

            with open(file_path, "w") as f:
                json.dump(hit, f, indent=2)

            processed_count += 1
            if processed_count % 1000 == 0:
                logger.debug(f"Processed {processed_count} docs...")

            yield hit

        logger.info(f"{processed_count} docs were downloaded")

    def get_random_doc_ids(
        self, index_name: str, size: int = 100, query: Optional[Dict[str, Any]] = None
    ) -> list[str]:
        """
        Retrieve random document IDs from the specified index.

        Args:
            index_name: The name of the Elasticsearch index
            size: The number of random document IDs to retrieve (max 10,000)
            query: An optional query to filter the documents

        Returns:
            A list of random document IDs
        """
        if size > 10_000:
            raise ValueError(
                "This method is not designed to get large numbers of random values. Reduce size <= 10,000"
            )

        search_query = {
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
            body=search_query,
        )

        return [doc["_id"] for doc in res["hits"]["hits"]]


class ElasticsearchApiAuthSession(BaseElasticsearchSession):
    """Elasticsearch session using API key authentication"""

    def _configure_client(self):
        api_id = os.getenv("ELASTIC_API_ID")
        api_key = os.getenv("ELASTIC_API_KEY")

        if not api_id:
            raise ValueError("Environment variable ELASTIC_API_ID is not set")
        if not api_key:
            raise ValueError("Environment variable ELASTIC_API_KEY is not set")

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

        if not es_user:
            raise ValueError("Environment variable ELASTIC_USER is not set")
        if not es_pwd:
            raise ValueError("Environment variable ELASTIC_PWD is not set")

        self.es = Elasticsearch(
            hosts=self.es_server,
            basic_auth=(es_user, es_pwd),
            verify_certs=False,
            ssl_show_warn=False,
        )


class ElasticsearchManualAuthSession(BaseElasticsearchSession):
    """Elasticsearch session with manually configured authentication"""

    def __init__(self, elasticsearch_client: Elasticsearch) -> None:
        super().__init__(elasticsearch_client=elasticsearch_client)

    def _configure_client(self):
        # Client is already configured in __init__
        pass


def create_elasticsearch_session(
    auth_type: Literal["api", "user", "manual"] = "api",
    elasticsearch_server: Optional[str] = None,
    elasticsearch_client: Optional[Elasticsearch] = None,
    use_proxy: bool = False,
) -> BaseElasticsearchSession:
    """
    Factory function to create an appropriate Elasticsearch session based on authentication type.

    Args:
        auth_type: Type of authentication to use ('api', 'user', or 'manual')
            'api' - API key authentication: relevant environment variables are ELASTIC_API_ID and ELASTIC_API_KEY
            'user' - Username/password authentication: relevant environment variables are ELASTIC_USER and ELASTIC_PWD
            'manual' - Manually configured Elasticsearch client
        elasticsearch_server: Elasticsearch server URL
        elasticsearch_client: Pre-configured Elasticsearch client (for 'manual' auth only)
        use_proxy: Whether to use the GSTT proxy

    Returns:
        Configured Elasticsearch session object

    Raises:
        ValueError: If auth_type is not one of 'api', 'user', or 'manual'
        ValueError: If auth_type is 'manual' and elasticsearch_client is not provided

    Example:
        >>> es_session = create_elasticsearch_session(
                auth_type="api", use_proxy=True
            )
    """
    proxy_node = GsttProxyNode if use_proxy else None

    if auth_type == "manual":
        if not elasticsearch_client:
            raise ValueError(
                "elasticsearch_client must be provided for manual client configuration"
            )
        return ElasticsearchManualAuthSession(elasticsearch_client)

    if auth_type == "api":
        return ElasticsearchApiAuthSession(
            elasticsearch_server=elasticsearch_server, proxy_node=proxy_node
        )

    if auth_type == "user":
        return ElasticsearchUserAuthSession(
            elasticsearch_server=elasticsearch_server, proxy_node=proxy_node
        )

    raise ValueError("auth_type must be one of: 'api', 'user', 'manual'")


# For backward compatibility
class ElasticsearchSession(BaseElasticsearchSession):
    """Deprecated: Use create_elasticsearch_session() factory function instead"""

    def __init__(
        self,
        proxy_node: Optional[RequestsHttpNode] = None,
        conn_mode: Literal["HTTP", "API"] = "HTTP",
    ) -> None:
        logger.warning(
            "ElasticsearchSession is deprecated. Use create_elasticsearch_session() instead."
        )

        self.proxy_node = proxy_node
        self.es_server = os.getenv(
            "ELASTIC_SERVER", "https://sv-pr-elastic01.gstt.local:9200"
        )

        # Disable SSL warnings
        requests.packages.urllib3.disable_warnings(
            requests.packages.urllib3.exceptions.InsecureRequestWarning
        )

        self._configure_client(conn_mode)

    def _configure_client(self, conn_mode):
        if conn_mode == "API":
            api_id = os.getenv("ELASTIC_API_ID")
            api_key = os.getenv("ELASTIC_API_KEY")

            if not api_id:
                raise ValueError("Environment variable ELASTIC_API_ID is not set")
            if not api_key:
                raise ValueError("Environment variable ELASTIC_API_KEY is not set")

            self.es = Elasticsearch(
                hosts=self.es_server,
                api_key=(api_id, api_key),
                node_class=self.proxy_node,
                verify_certs=False,
                ssl_show_warn=False,
            )

        elif conn_mode == "HTTP":
            es_user = os.getenv("ELASTIC_USER")
            es_pwd = os.getenv("ELASTIC_PWD")

            if not es_user:
                raise ValueError("Environment variable ELASTIC_USER is not set")
            if not es_pwd:
                raise ValueError("Environment variable ELASTIC_PWD is not set")

            self.es = Elasticsearch(
                hosts=self.es_server,
                basic_auth=(es_user, es_pwd),
                verify_certs=False,
                ssl_show_warn=False,
            )

        else:
            raise ValueError("Argument conn_mode must be 'HTTP' or 'API'")
