import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./models/xgb_model.json")
print("MODEL_PATH =", os.path.abspath(MODEL_PATH))  # ✅ AFTER definition

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .schemas import PredictRequest, PredictResponse, TopFactor
from .models import init_model, predict_denial_probability

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./models/xgb_model.json")

app = FastAPI(title="HealthScore AI Model API", version="0.1.0")

# If you call this directly from the frontend in dev, enable CORS.
# If you proxy through Next.js (/api/predict), you can tighten/disable.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set to your frontend origin(s) in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    # If you want strict feature ordering, pass the exact list here.
    init_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        proba, top = predict_denial_probability(req.features, topk=5)
        return PredictResponse(
            provider_key=req.provider_key,
            denial_probability=proba,
            top_factors=[TopFactor(feature=f, impact=float(v)) for f, v in top],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
