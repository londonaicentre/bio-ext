from dagster import Definitions

from .assets import oncollama_epic_mdt

defs = Definitions(
    assets=[oncollama_epic_mdt],
)
