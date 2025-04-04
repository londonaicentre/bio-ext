from dagster import Definitions

import projects.ingest.epic_oncology_letters
import projects.ingest.epic_oncology_mdt
import projects.infer.oncollama_epic_letters
import projects.infer.oncollama_epic_mdt

from projects.common.resources import global_resouces

defs = Definitions.merge(
    projects.ingest.epic_oncology_letters.defs,
    projects.ingest.epic_oncology_mdt.defs,
    projects.infer.oncollama_epic_letters.defs,
    projects.infer.oncollama_epic_mdt.defs,
    Definitions(resources=global_resouces),
)
