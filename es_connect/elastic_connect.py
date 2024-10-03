import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elastic_transport import RequestsHttpNode
import requests

# thanks @LAdams for implementing required http proxy
class GsttProxyNode(RequestsHttpNode):
    def __init__(self, *args, proxy_endpoint, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.session.proxies = {"http": proxy_endpoint, "https": proxy_endpoint}


class ElasticsearchSession:
    def __init__(self, server=None):
        requests.packages.urllib3.disable_warnings()

        load_dotenv()

        self.api_id = os.getenv("ELASTIC_API_ID")
        self.api_key = os.getenv("ELASTIC_API_KEY")
        self.es_server = server or os.getenv("ELASTIC_SERVER", "https://sv-pr-elastic01:9200")
        self.proxy_endpoint = os.getenv("ELASTIC_PROXY_ENDPOINT", "10.36.184.40:80")

        self.proxy_node = GsttProxyNode(proxy_endpoint = self.proxy_endpoint) 

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
    
    def list_indexes(self):
        return self.es.indices.get_alias(index="*")    

    def search(self, index, body):
        return self.es.search(index=index, body=body)
