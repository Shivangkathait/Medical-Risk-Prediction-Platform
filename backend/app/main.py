from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

app = FastAPI(
    title="NovaGen Health Risk API",
    description="Binary health risk classifier — NovaGen Research Labs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("/app/models/random_forest_model.pkl")
SCALER_PATH = Path("/app/models/scaler.pkl")
HISTORY_PATH = Path("/app/data/prediction_history.json")

model = None
scaler = None


def load_artifacts():
    global model, scaler
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
    else:
        print("WARNING: Model files not found. Using rule-based fallback.")


@app.on_event("startup")
def startup_event():
    load_artifacts()
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text(json.dumps([]))


class PatientInput(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    blood_pressure: float = Field(..., ge=60, le=250, description="Systolic blood pressure (mmHg)")
    cholesterol: float = Field(..., ge=100, le=400, description="Cholesterol level (mg/dL)")
    glucose_level: float = Field(..., ge=50, le=300, description="Blood glucose level (mg/dL)")
    heart_rate: float = Field(..., ge=40, le=200, description="Resting heart rate (bpm)")
    sleep_hours: float = Field(..., ge=1, le=12, description="Average sleep hours per day")
    exercise_hours: float = Field(..., ge=0, le=5, description="Average exercise hours per day")
    water_intake: float = Field(..., ge=0, le=6, description="Daily water intake (litres)")
    stress_level: int = Field(..., ge=1, le=10, description="Stress level (1–10)")
    smoking: int = Field(..., ge=0, le=1, description="Smoking: 1=Yes, 0=No")
    alcohol: int = Field(..., ge=0, le=1, description="Alcohol: 1=Yes, 0=No")
    mental_health: int = Field(..., ge=1, le=10, description="Mental health score (1–10)")
    physical_activity: int = Field(..., ge=1, le=10, description="Physical activity level (1–10)")
    medical_history: int = Field(..., ge=0, le=1, description="Prior medical conditions: 1=Yes")
    allergies: int = Field(..., ge=0, le=1, description="Known allergies: 1=Yes")
    diet_type_vegan: int = Field(0, ge=0, le=1, description="Vegan diet: 1=Yes")
    diet_type_vegetarian: int = Field(0, ge=0, le=1, description="Vegetarian diet: 1=Yes")
    blood_group_ab: int = Field(0, ge=0, le=1, description="Blood group AB")
    blood_group_b: int = Field(0, ge=0, le=1, description="Blood group B")
    blood_group_o: int = Field(0, ge=0, le=1, description="Blood group O")
    diet: int = Field(1, ge=0, le=3, description="Diet category (encoded)")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    risk_score: float
    recommendation: str
    feature_contributions: dict
    timestamp: str


def rule_based_predict(data: PatientInput):
    """Fallback rule-based predictor when model file is not available."""
    risk = 0.0
    contributions = {}

    # BMI
    if data.bmi > 30:
        delta = (data.bmi - 30) * 0.04
        risk += delta
        contributions["BMI"] = round(delta, 3)
    elif data.bmi > 25:
        delta = (data.bmi - 25) * 0.02
        risk += delta
        contributions["BMI"] = round(delta, 3)
    else:
        contributions["BMI"] = 0.0

    # Blood pressure
    if data.blood_pressure > 140:
        delta = (data.blood_pressure - 140) * 0.012
        risk += delta
        contributions["Blood Pressure"] = round(delta, 3)
    elif data.blood_pressure > 120:
        delta = (data.blood_pressure - 120) * 0.006
        risk += delta
        contributions["Blood Pressure"] = round(delta, 3)
    else:
        contributions["Blood Pressure"] = 0.0

    # Glucose
    if data.glucose_level > 126:
        delta = (data.glucose_level - 126) * 0.009
        risk += delta
        contributions["Glucose"] = round(delta, 3)
    elif data.glucose_level > 100:
        delta = (data.glucose_level - 100) * 0.004
        risk += delta
        contributions["Glucose"] = round(delta, 3)
    else:
        contributions["Glucose"] = 0.0

    # Age
    if data.age > 60:
        delta = (data.age - 60) * 0.010
        risk += delta
        contributions["Age"] = round(delta, 3)
    elif data.age > 45:
        delta = (data.age - 45) * 0.005
        risk += delta
        contributions["Age"] = round(delta, 3)
    else:
        contributions["Age"] = 0.0

    # Cholesterol
    if data.cholesterol > 240:
        delta = (data.cholesterol - 240) * 0.006
        risk += delta
        contributions["Cholesterol"] = round(delta, 3)
    else:
        contributions["Cholesterol"] = 0.0

    # Lifestyle
    smoke_delta = 0.22 if data.smoking else 0.0
    risk += smoke_delta
    contributions["Smoking"] = round(smoke_delta, 3)

    alc_delta = 0.08 if data.alcohol else 0.0
    risk += alc_delta
    contributions["Alcohol"] = round(alc_delta, 3)

    stress_delta = max(0, (data.stress_level - 6) * 0.04)
    risk += stress_delta
    contributions["Stress"] = round(stress_delta, 3)

    sleep_delta = max(0, (6 - data.sleep_hours) * 0.05)
    risk += sleep_delta
    contributions["Sleep"] = round(sleep_delta, 3)

    exercise_benefit = min(0.3, max(0, (data.exercise_hours - 0.3) * 0.12))
    risk -= exercise_benefit
    contributions["Exercise"] = round(-exercise_benefit, 3)

    mh_delta = max(0, (data.mental_health - 7) * 0.03)
    risk += mh_delta
    contributions["Mental Health"] = round(mh_delta, 3)

    hist_delta = 0.15 if data.medical_history else 0.0
    risk += hist_delta
    contributions["Medical History"] = round(hist_delta, 3)

    risk = float(np.clip(risk, 0, 1))
    return risk, contributions


def get_recommendation(prediction: str, risk_score: float) -> str:
    if prediction == "Healthy":
        if risk_score < 0.2:
            return "Excellent health profile. Maintain current lifestyle habits."
        return "Good health profile. Continue regular exercise and balanced diet."
    else:
        if risk_score > 0.7:
            return "High risk detected. Immediate medical consultation recommended."
        return "Moderate risk detected. Consult a healthcare provider for a full assessment."


@app.get("/")
def root():
    return {"message": "NovaGen Health Risk API is running", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    timestamp = datetime.utcnow().isoformat()

    feature_order = [
        "age", "bmi", "blood_pressure", "cholesterol", "glucose_level",
        "heart_rate", "sleep_hours", "exercise_hours", "water_intake",
        "stress_level", "smoking", "alcohol", "diet", "mental_health",
        "physical_activity", "medical_history", "allergies",
        "diet_type_vegan", "diet_type_vegetarian",
        "blood_group_ab", "blood_group_b", "blood_group_o"
    ]

    if model is not None and scaler is not None:
        features = np.array([[getattr(patient, f) for f in feature_order]])
        features_scaled = scaler.transform(features)
        pred = int(model.predict(features_scaled)[0])
        proba = model.predict_proba(features_scaled)[0]
        risk_score = float(proba[1])
        confidence = float(max(proba))
        prediction = "Unhealthy" if pred == 1 else "Healthy"
        contributions = {f: 0.0 for f in feature_order}  # placeholder
    else:
        risk_score, contributions = rule_based_predict(patient)
        prediction = "Unhealthy" if risk_score >= 0.5 else "Healthy"
        confidence = abs(risk_score - 0.5) * 2
        confidence = float(np.clip(0.5 + confidence * 0.4, 0.5, 0.99))

    recommendation = get_recommendation(prediction, risk_score)

    record = {
        "timestamp": timestamp,
        "age": patient.age,
        "bmi": patient.bmi,
        "blood_pressure": patient.blood_pressure,
        "glucose_level": patient.glucose_level,
        "prediction": prediction,
        "risk_score": round(risk_score, 3),
        "confidence": round(confidence, 3),
    }
    try:
        history = json.loads(HISTORY_PATH.read_text())
        history.append(record)
        history = history[-500:]  # keep last 500
        HISTORY_PATH.write_text(json.dumps(history))
    except Exception:
        pass

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 3),
        risk_score=round(risk_score, 3),
        recommendation=recommendation,
        feature_contributions={k: round(v, 4) for k, v in contributions.items()},
        timestamp=timestamp,
    )


@app.get("/history")
def get_history(limit: int = 100):
    try:
        history = json.loads(HISTORY_PATH.read_text())
        return {"history": history[-limit:], "total": len(history)}
    except Exception:
        return {"history": [], "total": 0}


@app.delete("/history")
def clear_history():
    HISTORY_PATH.write_text(json.dumps([]))
    return {"message": "History cleared"}
