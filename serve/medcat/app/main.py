import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.services.medcat_service import MedCatService

# Initialize FastAPI app
app = FastAPI(
    title="MedCAT FastAPI Service",
    description="A simple API for MedCAT Models",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    model_path = os.environ.get("MEDCAT_MODEL_PATH")
    if not model_path or model_path == '/app/models/medcat_model_pack.zip':
        print("WARNING: MEDCAT_MODEL_PATH environment variable not set") #TODO: consider making this a fatal error
    elif not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"MedCAT model not found at {model_path}")
    
    try:
        MedCatService.get_instance(model_path)
        print("MedCAT service initialized successfully")
    except Exception as e:
        print(f"Error initializing MedCAT service: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with basic information about the API"""
    return {
        "service": "MedCAT FastAPI Service",
        "status": "running",
        "documentation": "/docs",
    }
