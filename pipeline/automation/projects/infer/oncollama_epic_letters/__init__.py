from dagster import Definitions

from .assets import oncollama_epic_letter

defs = Definitions(
    assets=[oncollama_epic_letter],
)
