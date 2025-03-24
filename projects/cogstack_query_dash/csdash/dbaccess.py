import yaml
import pandas as pd
from elasticsearch import helpers
from bioext.elastic_utils import ElasticsearchSession, GsttProxyNode
from datetime import datetime

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

def get_mapping_types(_es,indexname):
    """ will extract type of fields and returns a dict. 
    """
    mapping = _es.indices.get_mapping(index=indexname)
    properties = mapping[indexname]["mappings"]["properties"]
    mappings_by_type = {}
    
    for field,details in properties.items():
        field_type = details.get("type","object")
        
        if  field_type not in mappings_by_type:
            mappings_by_type[field_type]=[]
        mappings_by_type[field_type].append(field)
    return mappings_by_type

def get_top_10kw(_es,indexname,fieldlist):
    """queries unique counts and top 10 key words for each field that is keyword 

    Args:
        _es (es): elastic search conn object
        indexname (str): name of index 
        fieldlist (list): list of fields to query that are kw
        query (json): obtained from config_dash.yaml json string kw
    """
    result = {}
    try: 
        for field in fieldlist:
            query = {
                "size":0,
                "aggs":{
                    "top_keywords":{
                    "terms":{
                        "field": field,
                        "size":10
                    }
                    },
                    "unique_counts":{
                    "cardinality":{
                        "field": field
                    }
                    }
                }
            }
        
            response = _es.search(index=indexname, body = query)
            top_keywords = response["aggregations"]["top_keywords"]["buckets"]
            unique_counts = response["aggregations"]["unique_counts"]["value"]
        
            result[field]= {
                "top_10": [(item["key"], item["doc_count"]) for item in top_keywords],
                "unique_count": unique_counts
            }
    except KeyError: 
        print(f"Warning. KeyError in {field} in this index {indexname}")
        result[field] = {
            "top_10": None,
            "unique_count": None
        }
    except Exception as e:
        print(f"Warning {e} error for {field} in index {indexname}")
        result[field] = {
            "top_10": None,
            "unique_count": None
        }
        
    return result


def get_date_ranges(_es,indexname,fieldlist):
    """queries date ranges for each field that is date 

    Args:
        _es (es): elastic search conn object
        indexname (str): name of index 
        fieldlist (list): list of fields to query that are kw
        query (json): obtained from config_dash.yaml json string kw
    """
    result = {}
    try: 
        for field in fieldlist:
            query = {
                "size":0,
                "aggs":{
                    "min_date":{
                        "min":{
                            "field": field,
                        }
                    },
                    "max_date":{
                        "max":{
                            "field": field
                        }
                    }
                }
            }
        
            response = _es.search(index=indexname, body = query)
            min_date = response["aggregations"]["min_date"]["value"]
            max_date = response["aggregations"]["max_date"]["value"]
        
            min_utc = datetime.utcfromtimestamp(min_date / 1000).strftime("%Y-%m-%d %H:%M:%S")
            max_utc = datetime.utcfromtimestamp(max_date / 1000).strftime("%Y-%m-%d %H:%M:%S")
        
            result[field]= {
                "min_date": min_utc,
                "max_date": max_utc
            }
    except KeyError:
        print(f"Warning: datefield{field} not found in index {indexname} or is wrong dtype")
        result[field] ={
            "min_date": None,
            "max_date": None
        }
    except Exception as e:
        print(f"error processing {field} in index {indexname} as {e}")
        result[field] ={
            "min_date": None,
            "max_date": None
        }
        
    return result