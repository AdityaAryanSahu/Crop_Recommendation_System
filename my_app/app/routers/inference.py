from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from datetime import datetime
from typing import Optional
from ..models import Input, Output, InferenceLog
from .data_fetch import get_coordinates, get_weather_details
from ..services.pred_services import MultiModal
from ..dependencies import get_multimodal_service
from sqlmodel import Session
from ..database import get_session
from zoneinfo import ZoneInfo
from tenacity import RetryError


router = APIRouter()

@router.post("/manual", response_model=Output)
async def predict_manual(device_id: Optional[str] = Form("MANUAL_001"), city: Optional[str] = Form(None, description="City name for context/validation."),N: float = Form(...), P: float = Form(...), K: float = Form(...), temperature:float= Form(...),
            Humidity: float = Form(...),ph: float = Form(...),  rainfall: float = Form(...), 
            image_file: UploadFile = File(..., description="The soil image file."), 
            service: MultiModal = Depends(get_multimodal_service),
            db: Session = Depends(get_session)
            ):
    image_content = await image_file.read()
    soil_prediction=service.predict_soil_type(image_content)
    
    crop_prediction=service.crop_pred( N, P, K, 
                                                 temperature, Humidity, ph, rainfall)
    
    final_crop, compatibility, alternatives, mssg= service.perform_fusion(crop_prediction, soil_prediction)
    
    date = datetime.now(ZoneInfo('Asia/Kolkata'))
    
    log_entry = InferenceLog(
        device_id=device_id,
        date_time=date,
        crop_pred=final_crop,
        soil_pred=soil_prediction,
        compatibility=compatibility,
        raw_tabular_prediction=crop_prediction,
        alternatives=",".join(alternatives), # Convert List[str] to single string
        mssg=mssg
    )
    
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
               
    return Output( 
    device_id=device_id,
    crop_pred=final_crop,
    soil_pred=soil_prediction,
    compatibility=compatibility,
    raw_tabular_prediction=crop_prediction,
    alternatives=alternatives,
    date_time=date,
    mssg=mssg)

@router.post("/auto", response_model=Output)
async def predict_auto( device_id: Optional[str] = Form("MANUAL_001"), N: float = Form(...), P: float = Form(...), K: float = Form(...), 
    ph: float = Form(...), 
    city: str = Form(..., description="City name for weather API lookup."),
    image_file: UploadFile = File(..., description="The soil image file."), service: MultiModal = Depends(get_multimodal_service),
    db: Session = Depends(get_session)
    ):
    image_content = await image_file.read()
    soil_prediction=service.predict_soil_type(image_content)
    try:
        lat, lon= get_coordinates(city)
    except Exception as e:
        raise HTTPException(status_code=404, detail="coordinates not found")
    try:
        temp, humidity, rain = get_weather_details(lat, lon) 
    except RetryError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"External Weather API failed after 3 retries. Please check the city name and try again."
        )
    
    crop_prediction= service.crop_pred(N, P, K,temp, humidity, ph, rain)
    
    final_crop, compatibility, alternatives, mssg= service.perform_fusion(crop_prediction, soil_prediction)
    
    date = datetime.now(ZoneInfo('Asia/Kolkata'))
    
    log_entry = InferenceLog(
        device_id=device_id,
        date_time=date,
        crop_pred=final_crop,
        soil_pred=soil_prediction,
        compatibility=compatibility,
        raw_tabular_prediction=crop_prediction,
        alternatives=",".join(alternatives), # Convert List[str] to single string
        mssg=mssg
    )
    
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    
    return Output( 
    device_id=device_id,
    crop_pred=final_crop,
    soil_pred=soil_prediction,
    compatibility=compatibility,
    raw_tabular_prediction=crop_prediction,
    alternatives=alternatives,
    date_time=date,
    mssg=mssg)
