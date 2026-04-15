import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── 1. Chargement du dataset ──────────────────────────────────────────────────
print("Chargement du dataset Depression...")
df = pd.read_csv("data/final_depression_dataset.csv")

print(f"  Nombre d'exemples : {df.shape[0]}")
print(f"  Nombre de colonnes : {df.shape[1]}")

# ── 2. Prétraitement ──────────────────────────────────────────────────────────

# Suppression des colonnes inutiles ou trop creuses
df = df.drop(columns=["Name", "City", "Profession", "Degree"])

# Fusion des colonnes redondantes (étudiant vs professionnel)
df["Pressure"] = df["Academic Pressure"].fillna(0) + df["Work Pressure"].fillna(0)
df["Satisfaction"] = df["Study Satisfaction"].fillna(0) + df["Job Satisfaction"].fillna(0)
df["CGPA"] = df["CGPA"].fillna(0)
df = df.drop(columns=["Academic Pressure", "Work Pressure", "Study Satisfaction", "Job Satisfaction"])

# Encodage des colonnes catégorielles binaires
binary_cols = [
    "Gender",
    "Working Professional or Student",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness",
]
for col in binary_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Encodage Sleep Duration et Dietary Habits
sleep_map = {
    "Less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "More than 8 hours": 3,
}
diet_map = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
df["Sleep Duration"] = df["Sleep Duration"].map(sleep_map)
df["Dietary Habits"] = df["Dietary Habits"].map(diet_map)

# Cible
df["Depression"] = (df["Depression"] == "Yes").astype(int)

# Features et cible
feature_names = [c for c in df.columns if c != "Depression"]
X = df[feature_names].values
y = df["Depression"].values

print(f"  Features retenues : {feature_names}")
print(f"  Distribution cible : {dict(zip(*np.unique(y, return_counts=True)))}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 3. Entraînement ───────────────────────────────────────────────────────────
print("\nEntraînement du modèle (RandomForestClassifier)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ── 4. Évaluation ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n── Métriques ────────────────────────────────")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  F1-score  : {f1:.4f}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=["No Depression", "Depression"]))

# ── 5. Sauvegarde ─────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

artifact = {
    "model": model,
    "scaler": scaler,
    "feature_names": feature_names,
    "target_names": ["No Depression", "Depression"],
    "sleep_map": sleep_map,
    "diet_map": diet_map,
}

with open("model/model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nModèle sauvegardé dans model/model.pkl")
