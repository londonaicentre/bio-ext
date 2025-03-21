from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ProcessTextRequest(BaseModel):
    text: str = Field(..., description="The text for medcat to process")
    filters: List[str] = Field(default_factory=list, description="Optional cui filter list to apply to the results") # TODO: filters to apply to the model


class Entity(BaseModel):
    id: str = Field(..., description="Entity identifier")
    pretty_name: str = Field(..., description="Full name of entity")
    cui: str = Field(..., description="Concept Unique Identifier")
    type_ids: List[str] = Field(..., description="List of Id of Entity type")
    types: List[str] = Field(..., description="Name of Entity type")
    source_value: str = Field(..., description="Original value from the source Concept datamodel")
    detected_value: str = Field(..., description="Value as detected in the text")
    acc: float = Field(..., description="Detection accuracy score")
    context_similarity: str = Field(..., description="Context similarity score calculation")
    start_index: int = Field(..., description="Start character index in the text")
    end_index: int = Field(..., description="End character index in the text")
    icd10: List[str] = Field(..., description="List of ICD10 codes")
    ontologies: List[str] = Field(..., description="List of Ontology that the Medcat model has been constructed from")
    snomed: List[str] = Field(..., description="SNOMED CT Metadata")
    meta_anns: Dict[str, Any] = Field(..., description="All MetaAnnotations and associated Metadata")


class ProcessTextResponse(BaseModel):
    text: str = Field(..., description="The input text that was processed") # TODO: Check if I need to return the input text? Context calculations etc...
    entities: List[Entity] = Field(..., description="List of extracted medical entities")
