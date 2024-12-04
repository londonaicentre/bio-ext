import os
from elasticsearch import Elasticsearch
from elastic_transport import RequestsHttpNode
import requests

# thanks @LAdams for implementing required http proxy
class GsttProxyNode(RequestsHttpNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proxy_endpoint = os.getenv("http_proxy")
        self.session.proxies = {"http": self.proxy_endpoint, "https":self.proxy_endpoint}

class ElasticsearchSession:
    def __init__(self, proxy=None, conn_mode:str="HTTP"):
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

        self.es_server = os.getenv("ELASTIC_SERVER", "https://sv-pr-elastic01:9200") # set to GSTT server by default

        if proxy:
            self.proxy_node = GsttProxyNode
        else:
            self.proxy_node = None

        try:
            if conn_mode == "API":
                self.api_id = os.getenv("ELASTIC_API_ID")
                self.api_key = os.getenv("ELASTIC_API_KEY")
                
                if not all([self.api_id, self.api_key]):
                    raise ValueError("Check that ELASTIC_API_ID and ELASTIC_API_KEY are in env variables")

            self.es = Elasticsearch(
                hosts=self.es_server,
                api_key=(self.api_id, self.api_key),
                node_class=self.proxy_node,
                verify_certs=False,
                ssl_show_warn=False
            )

        if conn_mode == "HTTP":
            self.es_user = os.getenv("ELASTIC_USER")
            self.es_pwd = os.getenv("ELASTIC_PWD")

            self.es = Elasticsearch(
                hosts=self.es_server,
                basic_auth=(self.es_user, self.es_pwd),
                verify_certs=False,
                ssl_show_warn=False
            )

    def list_indices(self):
        return self.es.indices.get_alias(index="*")
