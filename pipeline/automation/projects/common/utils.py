from typing import Generator

from elasticsearch import Elasticsearch


def elasticsearch_scroll_generator(
    source_es: Elasticsearch,
    index: str,
    query: dict,
    page_size: int = 1000,
    scroll_timeout: str = "5m",
) -> Generator[dict, None, None]:
    """
    Generator function to scroll through Elasticsearch documents.

    :param source_es: Elasticsearch client
    :param index: Index to search
    :param query: Search query
    :param page_size: Number of documents to retrieve per scroll
    :param scroll_timeout: Timeout for scroll context
    :yield: Individual document hits
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
                # Yield the hit
                yield hit

            # Get the next batch of hits using the scroll API
            result = source_es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
            scroll_id = result["_scroll_id"]
            hits = result["hits"]["hits"]
    finally:
        # Ensure scroll is cleared even if an exception occurs
        source_es.clear_scroll(scroll_id=scroll_id)
