import dagster as dg
from dagster import (
    AssetExecutionContext,
    asset,
)

from datetime import datetime, timedelta

from ..bioext_replication.assets import (
    elasticsearch_replication_asset,
    epic_daily_partitions,
)
from ..common.utils import elasticsearch_scroll_generator
from .main import process_document


@asset(
    automation_condition=dg.AutomationCondition.eager(),
    deps=[elasticsearch_replication_asset],
    partitions_def=epic_daily_partitions,
    required_resource_keys={"dest_es"},
    code_version="1.0.0",  # This is equivalent to the oncollama version
)
def oncollama_epic_asset(context: AssetExecutionContext):
    """Asset for OncoLLAMA EPIC."""
    context.log.info("Starting OncoLLAMA EPIC asset execution.")

    partition_date_str = context.partition_key

    partition_date = datetime.strptime(partition_date_str, "%Y-%m-%d")

    # To accomodate lag from Replication the asset materialisation is actually the day before.
    start_date = partition_date - timedelta(days=1)
    end_date = partition_date

    context.log.info(
        f"Processing partition {partition_date_str}: from {start_date.date()} to {end_date.date()}"
    )

    # Get Elasticsearch client from resources
    dest_es = context.resources.dest_es
    # Query to fetch data for this time partition with explicit source selection
    query = {
        "_source": [
            "id",
            "document_EpicId",
            "activity_Date",
            "document_CreatedWhen",
            "document_Content",
            "document_UpdatedWhen",
            "patient_NHSNumber",
            "patient_DurableKey",
            "activity_EncounterEpicCsn",
            "activity_PatientAdministrativeCategory",
            "activity_PatientClass",
            "activity_Type",
            "activity_VisitClass",
            "activity_VisitType",
            "activity_DepartmentSpecialty",
        ],
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "document_UpdatedWhen": {
                                "gte": str(start_date.date()),
                                "lt": str(end_date.date()),
                            }
                        }
                    },
                ],
            }
        },
    }

    document_lengths = []
    durations = []
    number_of_documents = 0

    # Use the scroll generator to fetch documents
    for doc in elasticsearch_scroll_generator(
        dest_es, "gstt_epic_notes_replica", query
    ):
        document_text = doc["_source"]["document_Content"]
        document_lengths.append(len(document_text))
        start_time = datetime.now()

        try:
            res = process_document(document_text)
        except ValueError as e:
            context.log.warn(
                f"Failed to process document {doc['_source']['id']} due to {e}"
            )
        else:
            context.log.info(res)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        durations.append(duration)

        context.log.info(
            f"Processed document with ID {doc['_source']['id']} in {duration:.2f} seconds."
        )
        number_of_documents += 1

    # Log the average document length and processing time
    avg_length = (
        sum(document_lengths) / len(document_lengths) if document_lengths else 0
    )
    avg_duration = sum(durations) / len(durations) if durations else 0

    return dg.MaterializeResult(
        metadata={
            "average_document_length": avg_length,
            "average_processing_time": avg_duration,
            "number_of_documents": number_of_documents,
        }
    )
