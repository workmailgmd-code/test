from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import numpy as np

# 1. Dictionary to hold our loaded model in memory
ml_model = {}

# 2. Define the lifespan function FIRST
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE when the server starts up
    ml_model["predictor"] = joblib.load("model.pkl")
    print("Model loaded successfully")
    yield
    # Runs ONCE when the server shuts down (cleanup)
    ml_model.clear()

# 3. Define the app variable AFTER lifespan is declared
app = FastAPI(title="Student Pass/Fail Predictor", lifespan=lifespan)

# Pydantic models for request/response validation
class StudentFeatures(BaseModel):
    study_hours: float
    prev_score: float

class PredictionResult(BaseModel):
    prediction: int
    result_label: str
    confidence: float

@app.get("/")
def health():
    return {"status": "running", "model_loaded": "predictor" in ml_model}

@app.post("/predict", response_model=PredictionResult)
def predict(features: StudentFeatures):
    model = ml_model.get("predictor")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_data = np.array([[features.study_hours, features.prev_score]])
    
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    confidence = round(float(probabilities[prediction]), 3)
    
    return PredictionResult(
        prediction=prediction,
        result_label="Pass" if prediction == 1 else "Fail",
        confidence=confidence
    )