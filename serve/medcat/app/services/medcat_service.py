from typing import Dict, List, Optional, Any
import os
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.meta_cat import MetaCAT


class MedCatService:
    _instance = None

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None):
        """Get or create a singleton instance of MedCatService.
        
        Args:
            model_path: Path to the MedCAT model pack. If None, will use the environment variable MEDCAT_MODEL_PATH.
            
        Returns:
            MedCatService instance
        """
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the MedCAT service with model loading.
        
        Args:
            model_path: Path to the MedCAT model pack. If None, will use the environment variable MEDCAT_MODEL_PATH.
        """
        self.model_path = model_path or os.environ.get("MEDCAT_MODEL_PATH")
        if not self.model_path:
            raise ValueError("Model path not provided. Set MEDCAT_MODEL_PATH environment variable or pass model_path.")
        
        self.cat = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the MedCAT model."""
        try:
            self.cat = CAT.load_model_pack(self.model_path)
            print(f"MedCAT model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading MedCAT model: {e}")
            raise
    
    def process_text(self, text: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text with MedCAT and return the extracted entities.
        
        Args:
            text: Text to process
            filters: Optional filters to apply to the results
            
        Returns:
            Dictionary with extracted entities and metadata
        """
        if not self.cat:
            raise RuntimeError("MedCAT model not loaded")
        
        # Apply filters if provided
        if len(filters) > 0:
            self.cat.config.linking['filters'] = {'cuis':set(filters)}
            print("filter added")

        # Process the text
        result = self.cat.get_entities(text)

        return self._format_output(text, result)

    
    # Edit the output format if needed
    def _format_output(self, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format the MedCAT output to a more API-friendly structure.
            Args:
                text: Original text
                result: Raw MedCAT result
                
            Returns:
                Formatted results
        """
        entities = []
        for entity_id, entity_data in result['entities'].items():
            entities.append({
                "id": entity_id,
                "pretty_name": entity_data['pretty_name'],
                "cui": entity_data['cui'],
                "type_ids": entity_data['type_ids'],
                "types": entity_data['types'],
                "source_value": entity_data['source_value'],
                "detected_value": entity_data['detected_name'],
                "acc": entity_data['acc'],
                "context_similarity": entity_data['context_similarity'],
                "start_index": entity_data['start'],
                "end_index": entity_data['end'],
                "icd10": entity_data['icd10'],
                "ontologies": entity_data['ontologies'],
                "snomed": entity_data['snomed'],
                "meta_anns" : entity_data['meta_anns'], # TODO: Seek to flattern. But meta_anns can be custom so should not be hard coded
            })

        return {
            "text": text,
            "entities": entities,
            }

