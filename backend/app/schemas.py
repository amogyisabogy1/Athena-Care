from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PredictRequest(BaseModel):
    model: str = Field(default="xgboost-denial-risk")
    best_iteration: Optional[int] = None
    provider_key: str
    features: Dict[str, float]

class TopFactor(BaseModel):
    feature: str
    impact: float

class PredictResponse(BaseModel):
    provider_key: str
    denial_probability: float
    top_factors: Optional[List[TopFactor]] = None
