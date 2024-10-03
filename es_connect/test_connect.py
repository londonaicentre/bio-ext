from elastic_connect import GsttProxyNode, ElasticsearchSession
from dotenv import load_dotenv

load_dotenv()
es_session = ElasticsearchSession()

# get info
info = es_session.get_info()
print("Elasticsearch Info:")
print(info)

# list indices
indices = es_session.list_indices()
print("\nElasticsearch Indices:")
for index in indices:
    print(index)