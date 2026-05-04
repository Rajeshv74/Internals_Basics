import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# PATHS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_path = os.path.join(BASE_DIR, "data", "training_data.csv")
new_path = os.path.join(BASE_DIR, "data", "new_data.csv")
model_dir = os.path.join(BASE_DIR, "models")
result_path = os.path.join(BASE_DIR, "results", "step4_s8.json")

# ---------------------------
# LOAD DATA
# ---------------------------
train_df = pd.read_csv(train_path)
new_df = pd.read_csv(new_path)

combined_df = pd.concat([train_df, new_df], ignore_index=True)

# ---------------------------
# FEATURES / TARGET
# ---------------------------
X = combined_df.drop("seats_filled_pct", axis=1)
y = combined_df["seats_filled_pct"]

# IMPORTANT: SAME SPLIT AS TASK 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# DETERMINE BEST MODEL TYPE
# ---------------------------
rf_path = os.path.join(model_dir, "RandomForest.pkl")
ridge_path = os.path.join(model_dir, "Ridge.pkl")

if os.path.exists(rf_path):
    model_type = "RandomForest"
    new_model = RandomForestRegressor(random_state=42)
elif os.path.exists(ridge_path):
    model_type = "Ridge"
    new_model = Ridge()
else:
    raise Exception("No trained model found from Task 1")

# ---------------------------
# RETRAIN MODEL
# ---------------------------
new_model.fit(X_train, y_train)
new_preds = new_model.predict(X_test)
retrained_mae = mean_absolute_error(y_test, new_preds)

# ---------------------------
# LOAD CHAMPION MODEL
# ---------------------------
champion_model = joblib.load(os.path.join(model_dir, f"{model_type}.pkl"))
champ_preds = champion_model.predict(X_test)
champion_mae = mean_absolute_error(y_test, champ_preds)

# ---------------------------
# COMPARE
# ---------------------------
improvement = champion_mae - retrained_mae
threshold = 0.3

if improvement >= threshold:
    action = "promoted"
    # replace old model
    joblib.dump(new_model, os.path.join(model_dir, f"{model_type}.pkl"))
else:
    action = "kept_champion"

# ---------------------------
# SAVE JSON
# ---------------------------
result = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_mae": round(champion_mae, 2),
    "retrained_mae": round(retrained_mae, 2),
    "improvement": round(improvement, 2),
    "min_improvement_threshold": 0.3,
    "action": action,
    "comparison_metric": "mae"
}

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

with open(result_path, "w") as f:
    json.dump(result, f, indent=4)

print("✅ Task 4 completed successfully!")
print("📄 Saved at:", result_path)
print("Champion MAE:", champion_mae)
print("Retrained MAE:", retrained_mae)
print("Improvement:", improvement)
print("Action:", action)
print("Saved to:", result_path)