from elasticsearch_session import ElasticsearchSession

es_session = ElasticsearchSession()

# get info
info = es_session.get_info()
print("Elasticsearch Info:")
print(info)

# list indexes
indexes = es_session.list_indexes()
print("\nElasticsearch Indexes:")
for index in indexes:
    print(index)