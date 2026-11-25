import os
import joblib
import pandas as np_pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import Orange

DATA = "data/processed/earthquakes_processed.csv"
OUTPUT = "models/orange_magnitude_model.joblib"

os.makedirs("models", exist_ok=True)
df = np_pd.read_csv(DATA)

features = ["year", "month", "latitude", "longitude", "depth"]
X = df[features].to_numpy(dtype=float)
y = df["magnitude_class"].astype(str).to_numpy()

class_variable = Orange.data.DiscreteVariable("magnitude_class", values=["Low", "Moderate", "High"])
feature_variables = [Orange.data.ContinuousVariable(name) for name in features]
domain = Orange.data.Domain(feature_variables, class_variable)

mapped = []
for row, label in zip(X, y):
    target = class_variable.to_val(label)
    mapped.append(Orange.data.Instance(domain, list(row) + [target]))
table = Orange.data.Table(domain, mapped)

learners = {
    "Logistic Regression": Orange.classification.LogisticRegressionLearner(class_weight="balanced", max_iter=1000, random_state=42),
    "Decision Tree": Orange.classification.TreeLearner(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": Orange.classification.RandomForestLearner(n_estimators=150, max_depth=12, class_weight="balanced", random_state=42)
}

indices = np_pd.Series(range(len(df)))
train_index, test_index = train_test_split(indices, test_size=0.20, random_state=42, stratify=y)
train_index = np_pd.Series(train_index, dtype=int).to_numpy()
test_index = np_pd.Series(test_index, dtype=int).to_numpy()
train_table = Orange.data.Table.from_table(domain, table, train_index)
test_table = Orange.data.Table.from_table(domain, table, test_index)

trained = {}
orange_metrics = {}
for name, learner in learners.items():
    trained[name] = learner(train_table)
    predicted = trained[name](test_table)
    y_true = [class_variable.values[int(row.get_class())] for row in test_table]
    y_pred = [class_variable.values[int(value)] for value in predicted]
    report = classification_report(y_true, y_pred, labels=["Low", "Moderate", "High"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=["Low", "Moderate", "High"])
    orange_metrics[name] = {
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"])
            } for label in ["Low", "Moderate", "High"]
        },
        "confusion_matrix": {"labels": ["Low", "Moderate", "High"], "matrix": matrix.tolist()}
    }

results = Orange.evaluation.CrossValidation(table, learners=list(learners.values()), k=5)
print("Stratified Orange holdout and 5-fold cross-validation completed")
for name, metrics in orange_metrics.items():
    print(name, metrics)

# Verified Orange model object export and reload test.
selected_name = "Random Forest"
selected_model = trained[selected_name]
joblib.dump({"orange_model": selected_model, "domain": domain, "features": features, "model_name": selected_name}, OUTPUT)
reloaded = joblib.load(OUTPUT)
row = Orange.data.Table.from_list(domain, [[2025.0, 6.0, 28.61, 77.20, 10.0]])[0]
prediction = reloaded["orange_model"](row)
print("Orange model export verified:", reloaded["model_name"])
print("Test prediction:", class_variable.values[int(prediction)])

y_pred = reloaded["orange_model"](test_table)
print("Confusion Matrix:\n", confusion_matrix(test_table.Y, y_pred))
print("Classification Report:\n", classification_report(test_table.Y, y_pred, target_names=class_variable.values))
