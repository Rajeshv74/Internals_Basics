from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import json, time, os
app = FastAPI()

# ---------------------------
# PATH FIX (IMPORTANT)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rf_path = os.path.join(BASE_DIR, "models", "RandomForest.pkl")
ridge_path = os.path.join(BASE_DIR, "models", "Ridge.pkl")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")


model = None

if os.path.exists(rf_path):
    model = joblib.load(rf_path)
elif os.path.exists(ridge_path):
    model = joblib.load(ridge_path)

# ---------------------------
# INPUT MODEL
# ---------------------------
class InputData(BaseModel):
    screen_capacity: int
    movie_rating: float
    ticket_price: float
    days_since_release: int
# ---------------------------
# ENDPOINTS
# ---------------------------
@app.get("/heartbeat")
def heartbeat():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: InputData):

    features = [[
        data.screen_capacity,
        data.movie_rating,
        data.ticket_price,
        data.days_since_release
    ]]

    prediction = model.predict(features)[0]

    # 🔥 LOG ENTRY
    log_entry = {
        "timestamp": time.time(),
        "input": data.dict(),
        "prediction": float(prediction)
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": float(prediction)}