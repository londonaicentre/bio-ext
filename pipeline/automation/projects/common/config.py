from dagster import Config, DailyPartitionsDefinition

EPIC_START_DATE = "2023-09-01"


class ElasticsearchReplicationConfig(Config):
    source_index: str = "gstt_clinical_epic_notes"


epic_daily_partitions = DailyPartitionsDefinition(
    start_date=EPIC_START_DATE,
)
