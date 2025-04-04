from dagster import Definitions

from .assets import epic_oncology_mdt

defs = Definitions(assets=[epic_oncology_mdt])
