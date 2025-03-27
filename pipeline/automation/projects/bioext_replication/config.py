from dagster import Config

ELASTICSEARCH_DEFAULTS = {
    "SOURCE_INDEX": "gstt_clinical_epic_notes",
    "DEST_INDEX": "gstt_epic_notes_replica",
}

EPIC_START_DATE = "2023-09-01"


class ElasticsearchReplicationConfig(Config):
    source_index: str = ELASTICSEARCH_DEFAULTS["SOURCE_INDEX"]
    dest_index: str = ELASTICSEARCH_DEFAULTS["DEST_INDEX"]
