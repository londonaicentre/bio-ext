from datetime import datetime, timedelta

import dagster as dg
from dagster import (
    AssetExecutionContext,
    DailyPartitionsDefinition,
    asset,
)
from elasticsearch.helpers import bulk

from ..common.utils import elasticsearch_scroll_generator
from .config import EPIC_START_DATE, ElasticsearchReplicationConfig

epic_daily_partitions = DailyPartitionsDefinition(
    start_date=EPIC_START_DATE,
)


@asset(
    partitions_def=epic_daily_partitions,
    description="Replicates data from source Elasticsearch index to BioExt Elastic replica",
    required_resource_keys={"source_es", "dest_es"},
    backfill_policy=dg.BackfillPolicy.multi_run(),
)
def elasticsearch_replication_asset(
    context: AssetExecutionContext, config: ElasticsearchReplicationConfig
):
    partition_date_str = context.partition_key

    partition_date = datetime.strptime(partition_date_str, "%Y-%m-%d")

    # To accomodate lag from Replication the asset materialisation is actually the day before.
    start_date = partition_date - timedelta(days=1)
    end_date = partition_date

    context.log.info(
        f"Processing partition {partition_date_str}: from {start_date.date()} to {end_date.date()}"
    )

    # Get Elasticsearch clients from resources
    source_es = context.resources.source_es
    dest_es = context.resources.dest_es

    # Query to fetch data for this time partition with explicit source selection
    query = {
        "_source": [
            "id",
            "document_EpicId",
            "activity_Date",
            "document_CreatedWhen",
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
            "activity_ChiefComplaint",
            "document_Content",
            "document_Name",
            "document_AuthorType",
            "document_Service",
            "document_Status",
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
                "filter": [
                    {"match_phrase": {"activity_Type": "Clinic/Practice Visit"}}
                ],
                "should": [],
                "must_not": [
                    {"match_phrase": {"document_Name.keyword": "Appointment Note"}},
                    {"match_phrase": {"document_Name.keyword": "Nursing Note"}},
                ],
            }
        },
        "sort": ["_doc"],  # For efficient scrolling
    }

    total_docs = 0
    batch_size = 1000
    batch = []

    # Use the generator to iterate through documents
    for hit in elasticsearch_scroll_generator(source_es, config.source_index, query):
        # Prepare document for indexing
        doc = hit["_source"]
        batch.append(
            {
                "_index": config.dest_index,
                "_id": hit["_id"],
                "_source": doc,
            }
        )

        # If batch reaches desired size, index it
        if len(batch) >= batch_size:
            res = bulk(dest_es, batch)
            context.log.info(f"Indexed batch of {len(batch)} documents")
            total_docs += len(batch)
            batch = []

    # Index any remaining documents in the batch
    if batch:
        res = bulk(dest_es, batch)
        context.log.info(f"Indexed batch of {len(batch)} documents")
        context.log.debug(res)
        total_docs += len(batch)

    # Refresh the destination index to make the changes visible
    dest_es.indices.refresh(index=config.dest_index)

    context.log.info(
        f"Successfully replicated {total_docs} documents for partition {partition_date_str}"
    )

    return dg.MaterializeResult(
        metadata={
            "partition": partition_date_str,
            "document_count": total_docs,
            "time_range": f"{start_date} to {end_date}",
        }
    )


# Schedule the asset materialisation job
materialisation_job = dg.define_asset_job(
    "daily_elastic_job", selection=[elasticsearch_replication_asset]
)

# Schedule the job to run at midnight every day
materialisation_schedule = dg.build_schedule_from_partitioned_job(
    materialisation_job, hour_of_day=0
)
