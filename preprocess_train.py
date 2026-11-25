import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

RAW = "data/raw/usgs_earthquakes_2016_2025.csv"
PROCESSED = "data/processed/earthquakes_processed.csv"
METRICS = "data/processed/model_metrics.json"
MODEL = "models/magnitude_classifier.joblib"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv(RAW)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

required = {"time", "latitude", "longitude", "depth", "mag"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError("USGS columns missing: " + ", ".join(sorted(missing)))

df = df[["time", "latitude", "longitude", "depth", "mag"]].copy()
df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
for col in ["latitude", "longitude", "depth", "mag"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["time", "latitude", "longitude", "depth", "mag"])
df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180)) & (df["depth"] >= 0)]
df = df.drop_duplicates()

df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month

# Three-category historical classification. Magnitude is used only to create the target.
df["magnitude_class"] = pd.cut(
    df["mag"],
    bins=[-np.inf, 5.499999, 6.499999, np.inf],
    labels=["Low", "Moderate", "High"]
)

df.to_csv(PROCESSED, index=False)

features = ["year", "month", "latitude", "longitude", "depth"]
X = df[features]
y = df["magnitude_class"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Class distribution in full dataset:")
print(y.value_counts().sort_index().to_string())
print("Class distribution in training split:")
print(y_train.value_counts().sort_index().to_string())

models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
}

metrics = {}
trained = {}
class_names = ["Low", "Moderate", "High"]
for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    trained[name] = model
    report = classification_report(y_test, prediction, labels=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, prediction, labels=class_names)
    metrics[name] = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, prediction, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_test, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"])
            } for label in class_names
        },
        "confusion_matrix": {
            "labels": class_names,
            "matrix": matrix.tolist()
        }
    }

best_name = max(metrics, key=lambda name: metrics[name]["macro_f1"])
joblib.dump({"model": trained[best_name], "features": features, "selected_model": best_name}, MODEL)

with open(METRICS, "w", encoding="utf-8") as handle:
    json.dump({
        "models": metrics,
        "best_model": best_name,
        "test_size": len(y_test),
        "training_size": len(y_train),
        "class_distribution_full": y.value_counts().sort_index().to_dict(),
        "class_distribution_train": y_train.value_counts().sort_index().to_dict(),
        "class_distribution_test": y_test.value_counts().sort_index().to_dict(),
        "stratified_split": True,
        "class_weight": "balanced",
        "smote_applied": False,
        "smote_reason": "Not applied: synthetic seismic coordinates and depth could distort the physical meaning of only 104 High-class records."
    }, handle, indent=2)

print(json.dumps({"rows": len(df), "best_model": best_name, "metrics": metrics}, indent=2))
