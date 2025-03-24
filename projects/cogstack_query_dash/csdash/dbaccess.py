import yaml
import pandas as pd
from elasticsearch import helpers
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode


def load_config(path):
    """
    simple function to load config.yaml returning dict.
    """
    with open(path, "r") as file:
        yamlvalues = yaml.safe_load(file)
    return yamlvalues


def connect_cogstack():
    """simple function to establish elastic search session using bio-ext"""
    session = ElasticsearchSession(conn_mode="API", proxy=GsttProxyNode)
    return session.es


def fetch_sample_data(es, query, index_name):
    """this function  need es object from elastic search
    once initiated, use this function to send queries to server.

    Args:
        query (json): search string as json object
        index_name (str): index names from the elastic seach

    Returns:
        dataframe: pandas which has flattened the json
    """
    response = es.search(index=index_name, body=query)
    _results = response["hits"]["hits"]
    _df = pd.json_normalize(_results)
    return _df


def list_and_fetch_data(_es, query):
    """iterate over each cogstack index and extract data.

    Args:
        es (elastic search session object): need to be initiated.

    Returns:
        cogstack_indexes: index name as list
        results: dict object which holds dataframes
    """
    cogstack_indexes = list(_es.indices.get_mapping().keys())
    results = {}
    for index in cogstack_indexes:
        _data = fetch_sample_data(es=_es,query=query, index_name=index)
        results[index] = _data
        print(f"{index} is appended to dict.")
    return cogstack_indexes, results
