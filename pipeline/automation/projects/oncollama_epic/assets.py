import dagster as dg
from dagster import (
    AssetExecutionContext,
    asset,
)

from datetime import datetime, timedelta

from projects.bioext_replication.assets import (
    elasticsearch_replication_asset,
    epic_daily_partitions,
)
from projects.common.utils import (
    create_index_if_not_exists,
    elasticsearch_scroll_generator,
    save_result_to_index,
    get_count_of_documents,
)
from .main import process_document

MODEL_VERSION = "1"
OUTPUT_INDEX = f"oncollama_epic_{MODEL_VERSION}"


@asset(
    automation_condition=dg.AutomationCondition.eager(),
    deps=[elasticsearch_replication_asset],
    partitions_def=epic_daily_partitions,
    required_resource_keys={"dest_es"},
    code_version="1.0.0",  # This is equivalent to the oncollama version
)
def oncollama_epic_asset(context: AssetExecutionContext):
    """Asset for OncoLlama output."""
    context.log.info("Starting OncoLLAMA Epic asset execution.")

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

    create_index_if_not_exists(dest_es, OUTPUT_INDEX)

    count_query = {"query": query["query"]}
    potential_documents = get_count_of_documents(
        dest_es, "gstt_epic_notes_replica", count_query
    )
    context.log.info(f"Potential documents to process: {potential_documents}")

    document_lengths = []
    durations = []
    failed_document_ids = []
    number_of_documents = 0

    # Use the scroll generator to fetch documents
    for doc in elasticsearch_scroll_generator(
        dest_es, "gstt_epic_notes_replica", query
    ):
        # Extract the document text
        document_text = doc["_source"]["document_Content"]
        document_id = doc["_source"]["id"]

        document_lengths.append(len(document_text))

        start_time = datetime.now()

        # == Process the document via vLLM ==
        # If the document is malformed or contains data that cannot be processed then we want to keep a track of it as being
        # non-processable but not fail the entire pipeline.
        try:
            res = process_document(document_text, api_url="http://vllm_oncollamav2.bioext_network:8000/v1/chat/completions")
        except ValueError as e:
            context.log.warning(
                f"Failed to process document {doc['_source']['id']} due to: {e}"
            )
            failed_document_ids.append(document_id)
            continue

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        durations.append(duration)

        # This may fail if the document is malformed or connectivity issues to Elastic arise
        # We want this to arrest the pipeline so we catch the exception
        save_result_to_index(
            dest_es,
            OUTPUT_INDEX,
            document_id,
            res,
        )

        context.log.info(
            f"Processed document with ID {document_id} in {duration:.2f} seconds."
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
            "average_vllm_time": avg_duration,
            "number_documents_processes": number_of_documents,
            "failed_document_ids": failed_document_ids,
            "number_total_documents": potential_documents,
        }
    )
