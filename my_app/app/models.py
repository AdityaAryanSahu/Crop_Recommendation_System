from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional
from enum import Enum
from datetime import datetime

class SoilType(str, Enum):
    ALLUVIAL = "alluvial"
    CLAY = "clay"
    LOAMY = "loamy"
    BLACK = "black"
    RED = "red"
    OTHER = "other"

class Input(BaseModel):
    N:float = Field(ge=0, le=100)
    P:float = Field(ge=0, le=100)
    K:float = Field(ge=0, le=100)
    temperature:float = Field(ge=0, le=100)
    Humidity:float = Field(ge=0, le=100)
    rainfall:float = Field(ge=0)
    ph:float = Field(ge=0, le=10)
    city:Optional[str]=None
    
class Output(BaseModel):
    device_id:Optional[str]=None
    crop_pred:str
    soil_pred:str
    compatibility:bool
    raw_tabular_prediction:str
    alternatives:List[str]
    date_time:datetime
    mssg:str
    
class InferenceLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    device_id: Optional[str] = Field(default=None, index=True)
    date_time: datetime = Field(default_factory=datetime.now)
    crop_pred: str
    soil_pred: str
    compatibility: bool
    raw_tabular_prediction: str
    alternatives: str
    mssg: str
    
    
    
    