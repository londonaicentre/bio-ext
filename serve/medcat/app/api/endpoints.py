from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any

from app.models import ProcessTextRequest, ProcessTextResponse
from app.services.medcat_service import MedCatService

router = APIRouter()

# Dependency for getting the MedCat service
def get_medcat_service():
    try:
        service = MedCatService.get_instance()
        yield service
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing MedCAT service: {str(e)}")


@router.post("/process", response_model=ProcessTextResponse, summary="Process text with MedCAT")
async def process_text(
    request: ProcessTextRequest = Body(...),
    medcat_service: MedCatService = Depends(get_medcat_service)
):
    """
    Process text with MedCAT and extract medical concepts.
    
    - **text**: The text to process
    - **filters**: Optional filters to apply to the results
    """
    try:
        print(f"processing text: {request.text}")
        print(f"processing filters: {request.filters}")
        result = medcat_service.process_text(text=request.text, filters=request.filters)
        
        '''# Ensure the result follows the correct structure by instantiating it as ProcessTextResponse
        validated_response = ProcessTextResponse(
            text=result.get("text", ""),  # Set a default empty string if `text` is missing
            entities=result.get("entities", []) if isinstance(result.get("entities"), list) else []
        )
        return validated_response'''
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@router.get("/health", summary="Health check endpoint")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}
