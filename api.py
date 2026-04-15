import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Depression Prediction API",
    description="Prédit si une personne est susceptible de souffrir de dépression.",
    version="1.0.0",
)

# ── Chargement du modèle au démarrage ─────────────────────────────────────────
MODEL_PATH = "model/model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    target_names = artifact["target_names"]
    sleep_map = artifact["sleep_map"]
    diet_map = artifact["diet_map"]
    print(f"Modèle chargé depuis {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(
        f"Modèle introuvable : {MODEL_PATH}. Lance d'abord train.py."
    )

# ── Schémas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    gender: str = Field(..., description="'Male' ou 'Female'")
    age: int = Field(..., description="Age de la personne")
    working_professional_or_student: str = Field(..., description="'Working Professional' ou 'Student'")
    cgpa: float = Field(0.0, description="CGPA si étudiant, sinon 0")
    sleep_duration: str = Field(..., description="'Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'")
    dietary_habits: str = Field(..., description="'Healthy', 'Moderate' ou 'Unhealthy'")
    suicidal_thoughts: str = Field(..., description="'Yes' ou 'No'")
    work_study_hours: float = Field(..., description="Heures de travail/étude par jour")
    financial_stress: int = Field(..., description="Niveau de stress financier (1-5)")
    family_history: str = Field(..., description="'Yes' ou 'No'")
    pressure: float = Field(..., description="Niveau de pression académique ou au travail (1-5)")
    satisfaction: float = Field(..., description="Niveau de satisfaction (1-5)")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "age": 28,
                "working_professional_or_student": "Working Professional",
                "cgpa": 0.0,
                "sleep_duration": "5-6 hours",
                "dietary_habits": "Unhealthy",
                "suicidal_thoughts": "Yes",
                "work_study_hours": 9,
                "financial_stress": 4,
                "family_history": "Yes",
                "pressure": 4.0,
                "satisfaction": 2.0,
            }
        }

class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability_no_depression: float
    probability_depression: float

# ── Helpers ───────────────────────────────────────────────────────────────────
def encode_input(req: PredictRequest) -> np.ndarray:
    gender_enc = 1 if req.gender == "Male" else 0
    status_enc = 1 if req.working_professional_or_student == "Working Professional" else 0
    suicidal_enc = 1 if req.suicidal_thoughts == "Yes" else 0
    family_enc = 1 if req.family_history == "Yes" else 0
    sleep_enc = sleep_map.get(req.sleep_duration, 1)
    diet_enc = diet_map.get(req.dietary_habits, 1)

    # Ordre identique à feature_names dans train.py
    features = [
        gender_enc,
        req.age,
        status_enc,
        req.cgpa,
        sleep_enc,
        diet_enc,
        suicidal_enc,
        req.work_study_hours,
        req.financial_stress,
        family_enc,
        req.pressure,
        req.satisfaction,
    ]
    return np.array(features).reshape(1, -1)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", summary="Vérification de l'état du service")
def health():
    return {
        "status": "ok",
        "model": "RandomForestClassifier",
        "features_expected": len(feature_names),
    }


@app.post("/predict", response_model=PredictResponse, summary="Prédiction de dépression")
def predict(request: PredictRequest):
    if request.sleep_duration not in sleep_map:
        raise HTTPException(
            status_code=422,
            detail=f"sleep_duration invalide. Valeurs acceptées : {list(sleep_map.keys())}",
        )
    if request.dietary_habits not in diet_map:
        raise HTTPException(
            status_code=422,
            detail=f"dietary_habits invalide. Valeurs acceptées : {list(diet_map.keys())}",
        )

    X = encode_input(request)
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]

    return PredictResponse(
        prediction=prediction,
        label=target_names[prediction],
        probability_no_depression=round(float(probabilities[0]), 4),
        probability_depression=round(float(probabilities[1]), 4),
    )
