from dagster import Definitions

from .assets import epic_oncology_letters

defs = Definitions(
    assets=[epic_oncology_letters],
)
