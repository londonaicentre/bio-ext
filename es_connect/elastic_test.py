import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elastic_transport import RequestsHttpNode
import requests

## credit @LAdams for enabling http proxy in ES connection

requests.packages.urllib3,disable_warnings()

load_dotenv()

API_ID = os.getenv("ELASTIC_API_ID")
API_KEY = os.getenv("ELASTIC_API_KEY")

proxy_endpoint = "10.36.184.40:80"

class GsttProxyNode(RequestsHttpNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.session.proxies = {"http": proxy_endpoint, "https":proxy_endpoint}

es = Elasticsearch(
    hosts="https://sv-pr-elastic01:9200",
    api_key=(API_ID, API_KEY),
    node_class=GsttProxyNode,
    verify_certs=False,
    ssl_show_warn=False
)

info_response = es.info()

print(info_response)