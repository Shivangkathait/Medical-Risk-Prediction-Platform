"""
train_model.py — Run this once to generate the model artifacts.
Usage:  python train_model.py
Output: models/random_forest_model.pkl  +  models/scaler.pkl
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report
import joblib
from pathlib import Path

SEED = 42
np.random.seed(SEED)
N = 9800

print("Generating synthetic health dataset...")

age           = np.random.randint(18, 80, N)
bmi           = np.random.uniform(16, 45, N)
blood_pressure= np.random.uniform(80, 200, N)
cholesterol   = np.random.uniform(120, 380, N)
glucose_level = np.random.uniform(60, 260, N)
heart_rate    = np.random.uniform(50, 110, N)
sleep_hours   = np.random.uniform(3, 10, N)
exercise_hours= np.random.uniform(0, 3, N)
water_intake  = np.random.uniform(0.5, 5, N)
stress_level  = np.random.randint(1, 11, N)
smoking       = np.random.binomial(1, 0.3, N)
alcohol       = np.random.binomial(1, 0.4, N)
diet          = np.random.randint(0, 4, N)
mental_health = np.random.randint(1, 11, N)
physical_activity = np.random.randint(1, 11, N)
medical_history   = np.random.binomial(1, 0.25, N)
allergies         = np.random.binomial(1, 0.2, N)
diet_vegan        = np.random.binomial(1, 0.1, N)
diet_vegetarian   = np.random.binomial(1, 0.2, N)
bg_ab = np.random.binomial(1, 0.04, N)
bg_b  = np.random.binomial(1, 0.09, N)
bg_o  = np.random.binomial(1, 0.44, N)

# Construct risk score
risk = (
    (bmi > 30).astype(float) * 0.25 +
    (blood_pressure > 140).astype(float) * 0.2 +
    (glucose_level > 126).astype(float) * 0.25 +
    (age > 55).astype(float) * 0.15 +
    (cholesterol > 240).astype(float) * 0.15 +
    smoking * 0.2 +
    (stress_level > 7).astype(float) * 0.1 +
    (sleep_hours < 6).astype(float) * 0.1 +
    medical_history * 0.15 +
    alcohol * 0.05 -
    (exercise_hours > 0.5).astype(float) * 0.15 -
    (physical_activity > 7).astype(float) * 0.1 +
    np.random.normal(0, 0.1, N)
)
target = (risk > 0.5).astype(int)

df = pd.DataFrame({
    "age": age, "bmi": bmi, "blood_pressure": blood_pressure,
    "cholesterol": cholesterol, "glucose_level": glucose_level,
    "heart_rate": heart_rate, "sleep_hours": sleep_hours,
    "exercise_hours": exercise_hours, "water_intake": water_intake,
    "stress_level": stress_level, "smoking": smoking, "alcohol": alcohol,
    "diet": diet, "mental_health": mental_health,
    "physical_activity": physical_activity, "medical_history": medical_history,
    "allergies": allergies, "diet_type_vegan": diet_vegan,
    "diet_type_vegetarian": diet_vegetarian, "blood_group_ab": bg_ab,
    "blood_group_b": bg_b, "blood_group_o": bg_o, "target": target
})

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("Training Random Forest (200 trees)...")
rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1)
rf.fit(X_train_sc, y_train)

y_pred = rf.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f"\nTest Accuracy : {acc:.4f}")
print(f"Test Recall   : {rec:.4f}")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Unhealthy"]))

out = Path("models")
out.mkdir(exist_ok=True)
joblib.dump(rf, out / "random_forest_model.pkl")
joblib.dump(scaler, out / "scaler.pkl")
print("\nSaved: models/random_forest_model.pkl")
print("Saved: models/scaler.pkl")
