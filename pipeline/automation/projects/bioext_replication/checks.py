from dagster import AssetCheckResult, asset_check
from elasticsearch import Elasticsearch

from .assets import elasticsearch_replication_asset
from .config import ElasticsearchReplicationConfig

from datetime import date, timedelta


@asset_check(
    asset=elasticsearch_replication_asset,
    required_resource_keys={"source_es", "dest_es"},
)
def assert_equal_number_documents(context, config: ElasticsearchReplicationConfig):
    # Setup resources
    source_es: Elasticsearch = context.resources.source_es
    dest_es: Elasticsearch = context.resources.dest_es

    start_date = "2023-10-01"
    end_date = str(date.today() - timedelta(days=1))

    context.log.info(f"Check from {start_date} to {end_date}")

    validation_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "document_UpdatedWhen": {
                                "gte": start_date,
                                "lt": end_date,
                            }
                        }
                    }
                ],
                "filter": [
                    {"match_phrase": {"activity_Type": "Clinic/Practice Visit"}}
                ],
                "should": [],
                "must_not": [
                    {"match_phrase": {"document_Name.keyword": "Appointment Note"}},
                    {"match_phrase": {"document_Name.keyword": "Nursing Note"}},
                ],
            }
        }
    }

    # Count the number of documents in the source and destination indices
    source_document_count = source_es.count(
        index=config.source_index,
        body=validation_query,
    )["count"]

    dest_document_count = dest_es.count(
        index=config.dest_index,
        body=validation_query,
    )["count"]

    # Return the result of the check
    return AssetCheckResult(
        passed=(source_document_count == dest_document_count),
        metadata={
            "destination_document_count": dest_document_count,
            "source_document_count": source_document_count,
        },
        description="A check to validate the number of documents in the replicated index is equal to the source index.",
    )
