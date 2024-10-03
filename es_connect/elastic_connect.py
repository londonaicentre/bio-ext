import os
from elasticsearch import Elasticsearch
from elastic_transport import RequestsHttpNode
import requests

# thanks @LAdams for implementing required http proxy
class GsttProxyNode(RequestsHttpNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proxy_endpoint = os.getenv("ELASTIC_PROXY_ENDPOINT", "10.36.184.40:80")
        self.session.proxies = {"http": self.proxy_endpoint, "https":self.proxy_endpoint}

class ElasticsearchSession:
    def __init__(self, server=None):
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

        self.api_id = os.getenv("ELASTIC_API_ID")
        self.api_key = os.getenv("ELASTIC_API_KEY")
        self.es_server = server or os.getenv("ELASTIC_SERVER", "https://sv-pr-elastic01:9200")

        self.proxy_node = GsttProxyNode

        self.es = self.create_session()

    def create_session(self):
        return Elasticsearch(
            hosts=self.es_server,
            api_key=(self.api_id, self.api_key),
            node_class=self.proxy_node,
            verify_certs=False,
            ssl_show_warn=False
        )

    def get_info(self):
        return self.es.info()
    
    def list_indices(self):
        return self.es.indices.get_alias(index="*")    

    def search(self, index, body):
        return self.es.search(index=index, body=body)
