from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.serving.prediction_service import PredictionService

app = FastAPI(
    title="Gurgaon Real Estate Price Prediction API",
    description="Predicts property price (Crore INR) based on property attributes.",
    version="1.0.0",
)

service: Optional[PredictionService] = None


@app.on_event("startup")
def load_model():
    global service
    try:
        service = PredictionService()
    except FileNotFoundError as e:
        print(f"WARNING: {e}. /predict will fail until model is trained.")


class PropertyInput(BaseModel):
    property_type: str = "flat"
    sector: str = "sector 36"
    bedRoom: float = 3.0
    bathroom: float = 2.0
    balcony: str = "2"
    agePossession: str = "New Property"
    built_up_area: float = 1200.0
    servant_room: float = 0.0
    store_room: float = 0.0
    furnishing_type: str = "unfurnished"
    luxury_score: str = "Low"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PropertyInput):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    input_dict = {
        "property_type": data.property_type,
        "sector": data.sector,
        "bedRoom": data.bedRoom,
        "bathroom": data.bathroom,
        "balcony": data.balcony,
        "agePossession": data.agePossession,
        "built_up_area": data.built_up_area,
        "servant room": data.servant_room,
        "store room": data.store_room,
        "furnishing_type": data.furnishing_type,
        "luxury_score": data.luxury_score,
    }
    try:
        result = service.predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
