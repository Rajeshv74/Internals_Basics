import pandas as pd
import json
import os

# ---------------------------
# PATH SETUP
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
LOG_PATH = os.path.join(BASE_DIR, "logs", "predictions.jsonl")
RESULT_PATH = os.path.join(BASE_DIR, "results", "step3_s5.json")

# ---------------------------
# LOAD DATA
# ---------------------------
train_df = pd.read_csv(DATA_PATH)

# Load logs
logs = []
with open(LOG_PATH, "r") as f:
    for line in f:
        logs.append(json.loads(line))

# Convert logs → DataFrame
live_df = pd.DataFrame([entry["input"] for entry in logs])

# ---------------------------
# DRIFT CHECK FUNCTION
# ---------------------------
def check_drift(feature, threshold):
    train_mean = train_df[feature].mean()
    live_mean = live_df[feature].mean()
    shift = abs(live_mean - train_mean)

    return {
        "feature": feature,
        "train_mean": round(train_mean, 2),
        "live_mean": round(live_mean, 2),
        "shift": round(shift, 2),
        "threshold": threshold,
        "status": "ALERT" if shift > threshold else "OK"
    }

# ---------------------------
# CHECK FEATURES
# ---------------------------
alerts = [
    check_drift("movie_rating", 2.01),
    check_drift("days_since_release", 13.46)
]

# ---------------------------
# FINAL OUTPUT
# ---------------------------
mean_prediction = round(
    pd.DataFrame(logs)["prediction"].mean(), 2
)

result = {
    "total_predictions": len(logs),
    "mean_prediction": mean_prediction,
    "drift_detected": any(a["status"] == "ALERT" for a in alerts),
    "alerts": alerts
}

# Ensure results folder exists
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

# Save JSON
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=4)

# ---------------------------
# DONE
# ---------------------------
print("✅ Task 3 completed successfully!")
print("📄 Output saved at:", RESULT_PATH)