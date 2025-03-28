from dagster import Definitions

from .bioext_replication.assets import (
    elasticsearch_replication_asset,
    materialisation_schedule,
)
from .bioext_replication.checks import assert_equal_number_documents
from .bioext_replication.resources import elasticsearch_resources

from .oncollama_epic.assets import oncollama_epic_asset

from .common.resources import common_resources

defs = Definitions(
    assets=[elasticsearch_replication_asset, oncollama_epic_asset],
    resources=[elasticsearch_resources, common_resources],
    schedules=[materialisation_schedule],
    asset_checks=[assert_equal_number_documents],
)
