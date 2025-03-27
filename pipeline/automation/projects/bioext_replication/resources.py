import os

from dagster import resource
from elastic_transport import RequestsHttpNode
from elasticsearch import Elasticsearch


class GsttProxyNode(RequestsHttpNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proxy_endpoint = os.getenv("http_proxy")
        self.session.proxies = {
            "http": self.proxy_endpoint,
            "https": self.proxy_endpoint,
        }


@resource
def source_elasticsearch_resource(init_context):
    """Resource for the source Elasticsearch instance."""

    return Elasticsearch(
        hosts=os.getenv("SOURCE_ES_HOST"),
        api_key=(
            os.getenv("SOURCE_ES_API"),
            os.getenv("SOURCE_ES_KEY"),
        ),
        verify_certs=False,
        ssl_show_warn=False,
        node_class=GsttProxyNode,
    )


@resource
def destination_elasticsearch_resource(init_context):
    """Resource for the destination Elasticsearch instance."""
    return Elasticsearch(
        hosts=os.getenv("DEST_ES_HOST"),
        api_key=(
            os.getenv("DEST_ES_API"),
            os.getenv("DEST_ES_KEY"),
        ),
        verify_certs=False,
        ssl_show_warn=False,
    )


# Exported resources
elasticsearch_resources = {
    "source_es": source_elasticsearch_resource,
    "dest_es": destination_elasticsearch_resource,
}
