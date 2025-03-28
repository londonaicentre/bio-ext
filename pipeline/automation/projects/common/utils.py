from typing import Generator

from elasticsearch import Elasticsearch


def elasticsearch_scroll_generator(
    source_es: Elasticsearch,
    index: str,
    query: dict,
    page_size: int = 1000,
    scroll_timeout: str | None = None,
) -> Generator[dict, None, None]:
    """
    Implements the Elasticsearch scroll API to efficiently retrieve large result sets by scrolling
    through batches of documents. Handles scroll cleanup automatically.

    Args:
        source_es (Elasticsearch): Elasticsearch client instance to query from
        index (str): Name of the Elasticsearch index to search
        query (dict): Elasticsearch query DSL as a dictionary
        page_size (int, optional): Number of documents to retrieve per scroll. Defaults to 1000
        scroll_timeout (str, optional): How long to keep scroll context alive. Defaults to "5m"

    Yields:
        dict: Individual document hits from the search results

    Raises:
        ElasticsearchException: If there are issues connecting to Elasticsearch or with the query

    Example:
        >>> es = Elasticsearch()
        >>> query = {"query": {"match_all": {}}}
        >>> for doc in elasticsearch_scroll_generator(es, "my-index", query):
        ...     print(doc["_source"])
    """
    # Initialize the scroll
    result = source_es.search(
        index=index, body=query, size=page_size, scroll=scroll_timeout
    )

    scroll_id = result["_scroll_id"]
    hits = result["hits"]["hits"]

    try:
        # Continue scrolling while there are hits
        while hits:
            for hit in hits:
                yield hit

            # Get the next batch of hits using the scroll API
            result = source_es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
            scroll_id = result["_scroll_id"]
            hits = result["hits"]["hits"]
    finally:
        # Ensure scroll is cleared even if an exception occurs
        source_es.clear_scroll(scroll_id=scroll_id)


def save_result_to_index(
    es: Elasticsearch,
    index: str,
    doc_id: str,
    document: dict,
) -> None:
    """Save a document to an Elasticsearch index.

    Args:
        es (Elasticsearch): Elasticsearch client instance
        index (str): Name of the index to save the document to
        doc_id (str): Unique identifier for the document
        document (dict): Document data to be indexed

    Returns:
        None

    Raises:
        elasticsearch.exceptions.RequestError: If the document cannot be indexed
    """
    es.index(index=index, id=doc_id, body=document)


def get_count_of_documents(
    es: Elasticsearch,
    index: str,
    query: dict,
) -> int:
    """Get the count of documents matching a query in an Elasticsearch index.

    Args:
        es (Elasticsearch): Elasticsearch client instance
        index (str): Name of the index to search
        query (dict): Elasticsearch query DSL as a dictionary

    Returns:
        int: Count of documents matching the query

    Raises:
        elasticsearch.exceptions.RequestError: If the query is malformed
    """
    response = es.count(index=index, body=query)
    return response["count"]


def create_index_if_not_exists(
    es: Elasticsearch,
    index: str,
    settings: dict | None = None,
) -> None:
    """Create an Elasticsearch index if it does not already exist.

    Args:
        es (Elasticsearch): Elasticsearch client instance
        index (str): Name of the index to create
        settings (dict): Optional index settings and mappings

    Returns:
        None

    Raises:
        elasticsearch.exceptions.RequestError: If the index cannot be created
    """
    if not es.indices.exists(index=index):
        es.indices.create(index=index, body=settings)
