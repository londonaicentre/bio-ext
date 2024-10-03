from scripts.elastic_connect import GsttProxyNode, ElasticsearchSession
from dotenv import load_dotenv

load_dotenv()
es_session = ElasticsearchSession()

# get info
info = es_session.es.info()
print("Elasticsearch Info:")
print(info)

# list indices
indices = es_session.list_indices()
print("\nElasticsearch Indices:")
for index in indices:
    print(index)

# test search
query = {
    "size": 500,
    "query": {
        "bool": {
            "must": [
                {"term": {"document_Name.keyword": "Clinical-MDM"}},
                {"wildcard": {"document_Content": "*breast*"}}
            ]
        }
    }
}

results = es_session.es.search(index="gstt_clinical_cancer_documents", body=query)

print("\nSearch Results:")
print(f"Total: {results['hits']['total']['value']}")
print(f"Retrieved: {len(results['hits']['hits'])}")
for hit in results['hits']['hits']:
    print(f"Document ID: {hit['_id']}")
    print(f"Score: {hit['_score']}")
    print(f"Document Name: {hit['_source'].get('document_Name', 'Missing')}")
    print(f"Content Preview: {hit['_source'].get('document_Content', 'Missing')[:42]}...")
print("\nRetrieval complete")

# Note that ES results are returned in the following format:
# https://www.elastic.co/guide/en/elasticsearch/reference/current/search-your-data.html
# 
# {
#   "took": <time in milliseconds>,
#   "timed_out": <boolean>,
#   "_shards": { ... },
#   "hits": { ... }
# }
#
# Where hits is further structured as:
# "hits": {
#   "total": { "value": <total_count>, "relation": "eq" },
#   "max_score": <highest_score>,
#   "hits": [ ... ]
# }
#
# And nested hits as:
# {
#   "_index": "<index_name>",
#   "_type": "_doc",
#   "_id": "<document_id>",
#   "_score": <relevance_score>,
#   "_source": { ... }
# }
#
# Where "_source" contains the document content.
#
