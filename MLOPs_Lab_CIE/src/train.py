import pandas as pd
import mlflow
import os
import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# PATH SETUP (works from src/)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create folders if not exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("seats_filled_pct", axis=1)
y = df["seats_filled_pct"]

# Train/test split (as per exam rule)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# MLFLOW SETUP
# ---------------------------
mlflow.set_experiment("ticketflick-seats-filled-pct")

results = []

models = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

# ---------------------------
# TRAIN & LOG MODELS
# ---------------------------
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)

        # Compatible RMSE (works for all sklearn versions)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # MLflow logging
        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("domain", "movie_booking")

        # Store results
        results.append({
            "name": name,
            "mae": float(mae),
            "rmse": float(rmse)
        })

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)

# ---------------------------
# SELECT BEST MODEL (by RMSE)
# ---------------------------
best = min(results, key=lambda x: x["rmse"])

# ---------------------------
# SAVE JSON OUTPUT
# ---------------------------
output = {
    "experiment_name": "ticketflick-seats-filled-pct",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

result_path = os.path.join(RESULTS_DIR, "step1_s1.json")

with open(result_path, "w") as f:
    json.dump(output, f, indent=4)

# ---------------------------
# DONE
# ---------------------------
print("✅ Task 1 completed successfully!")
print("📁 Models saved in:", MODEL_DIR)
print("📄 Result saved in:", result_path)