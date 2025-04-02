import pytest
from mock import Mock
from elasticsearch import Elasticsearch

from ..common.utils import elasticsearch_scroll_generator


@pytest.fixture
def mock_es_client():
    return Mock(spec=Elasticsearch)


def test_elasticsearch_scroll_generator_empty_results(mock_es_client):
    # Setup mock response for empty results
    mock_es_client.search.return_value = {
        "_scroll_id": "dummy_scroll_id",
        "hits": {"hits": []},
    }

    query = {"query": {"match_all": {}}}
    results = list(elasticsearch_scroll_generator(mock_es_client, "test_index", query))

    assert len(results) == 0
    mock_es_client.search.assert_called_once()
    mock_es_client.clear_scroll.assert_called_once_with(scroll_id="dummy_scroll_id")


def test_elasticsearch_scroll_generator_single_page(mock_es_client):
    # Setup mock response for single page of results
    mock_docs = [
        {"_id": "1", "_source": {"field": "value1"}},
        {"_id": "2", "_source": {"field": "value2"}},
    ]

    mock_es_client.search.return_value = {
        "_scroll_id": "dummy_scroll_id",
        "hits": {"hits": mock_docs},
    }
    mock_es_client.scroll.return_value = {
        "_scroll_id": "dummy_scroll_id",
        "hits": {"hits": []},
    }

    query = {"query": {"match_all": {}}}
    results = list(elasticsearch_scroll_generator(mock_es_client, "test_index", query))

    assert len(results) == 2
    assert results == mock_docs
    mock_es_client.search.assert_called_once()
    mock_es_client.clear_scroll.assert_called_once_with(scroll_id="dummy_scroll_id")


def test_elasticsearch_scroll_generator_with_exception(mock_es_client):
    # Setup mock to raise an exception
    mock_es_client.search.return_value = {
        "_scroll_id": "dummy_scroll_id",
        "hits": {"hits": [{"_id": "1"}]},
    }
    mock_es_client.scroll.side_effect = Exception("Scroll failed")

    query = {"query": {"match_all": {}}}

    with pytest.raises(Exception, match="Scroll failed"):
        list(elasticsearch_scroll_generator(mock_es_client, "test_index", query))

    # Verify clear_scroll was called even after exception
    mock_es_client.clear_scroll.assert_called_once_with(scroll_id="dummy_scroll_id")
