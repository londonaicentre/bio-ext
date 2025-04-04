from dagster import InitResourceContext, resource, EnvVar
from dagster_slack import SlackResource
from elastic_transport import RequestsHttpNode
from elasticsearch import Elasticsearch


class GsttProxyNode(RequestsHttpNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proxy_endpoint = EnvVar("http_proxy")
        self.session.proxies = {
            "http": self.proxy_endpoint,
            "https": self.proxy_endpoint,
        }


@resource
def source_elasticsearch_resource(init_context: InitResourceContext):
    """Resource for the source Elasticsearch instance."""

    return Elasticsearch(
        hosts=EnvVar("SOURCE_ES_HOST"),
        api_key=(
            EnvVar("SOURCE_ES_API"),
            EnvVar("SOURCE_ES_KEY"),
        ),
        verify_certs=False,
        ssl_show_warn=False,
        node_class=GsttProxyNode,
    )


@resource
def destination_elasticsearch_resource(init_context: InitResourceContext):
    """Resource for the destination Elasticsearch instance."""
    return Elasticsearch(
        hosts=EnvVar("DEST_ES_HOST"),
        api_key=(
            EnvVar("DEST_ES_API"),
            EnvVar("DEST_ES_KEY"),
        ),
        verify_certs=False,
        ssl_show_warn=False,
    )


# Exported resources
global_resouces = {
    "cogstack_elastic": source_elasticsearch_resource,
    "bioext_elastic": destination_elasticsearch_resource,
    "slack": SlackResource(token=EnvVar("SLACK_API_TOKEN")),
}
