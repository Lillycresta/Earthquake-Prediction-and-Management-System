import json
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, request, jsonify
import Orange

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
DATA_PATH = "data/processed/earthquakes_processed.csv"
MODEL_PATH = "models/magnitude_classifier.joblib"
METRICS_PATH = "data/processed/model_metrics.json"


def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    return df


def plot_json(fig):
    return fig.to_json()


def latest_model_data():
    df = load_data()
    if df.empty:
        return {"total": 0, "max_magnitude": 0, "average_depth": 0, "start_year": None, "end_year": None}
    return {
        "total": len(df),
        "max_magnitude": round(float(df["mag"].max()), 1),
        "average_depth": round(float(df["depth"].mean()), 1),
        "start_year": int(df["year"].min()),
        "end_year": int(df["year"].max()),
    }


@app.route("/")
def home():
    return render_template("home.html", stats=latest_model_data())


@app.route("/dashboard")
def dashboard():
    df = load_data()
    if df.empty:
        charts = {name: "{}" for name in ["map", "magnitude", "trend", "depth", "performance"]}
        return render_template("dashboard.html", charts=charts, stats=latest_model_data())

    sample = df if len(df) <= 12000 else df.sample(12000, random_state=42)

    map_fig = px.scatter_geo(
        sample,
        lat="latitude", lon="longitude", color="mag", size="mag",
        hover_name="place" if "place" in sample else None,
        hover_data={"mag": True, "depth": True, "latitude": False, "longitude": False},
        color_continuous_scale="YlOrRd", title="Geographic Earthquake Map"
    )
    map_fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10), coloraxis_colorbar_title="Magnitude")

    magnitude_fig = px.histogram(df, x="mag", nbins=40, title="Earthquake Magnitude Distribution", color_discrete_sequence=["#d97706"])
    magnitude_fig.update_layout(height=400, margin=dict(l=40, r=20, t=55, b=40), bargap=0.05)
    magnitude_fig.update_xaxes(title="Magnitude")
    magnitude_fig.update_yaxes(title="Number of earthquakes")

    yearly = df.groupby("year").size().reset_index(name="count")
    trend_fig = px.line(yearly, x="year", y="count", markers=True, title="Yearly Earthquake Trend")
    trend_fig.update_traces(line=dict(color="#147D64", width=3), marker=dict(size=8))
    trend_fig.update_layout(height=400, margin=dict(l=40, r=20, t=55, b=40))
    trend_fig.update_xaxes(title="Year", dtick=1)
    trend_fig.update_yaxes(title="Earthquake count")

    depth_fig = px.scatter(
        sample, x="mag", y="depth", color="mag", title="Earthquake Depth vs Magnitude",
        color_continuous_scale="Teal", opacity=0.7,
        hover_data={"mag": True, "depth": True, "latitude": False, "longitude": False}
    )
    depth_fig.update_layout(height=450, margin=dict(l=40, r=20, t=55, b=40))
    depth_fig.update_yaxes(autorange="reversed", title="Depth (km)")
    depth_fig.update_xaxes(title="Magnitude")

    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        metrics_doc = json.load(handle)
    metric_rows = []
    for model_name, values in metrics_doc["models"].items():
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            metric_rows.append({"Model": model_name, "Metric": metric_name.title(), "Score": round(values[metric_name] * 100, 2)})
    performance_df = pd.DataFrame(metric_rows)
    performance_fig = px.bar(
        performance_df, x="Model", y="Score", color="Metric", barmode="group",
        title="Machine Learning Model Performance Comparison", text="Score",
        color_discrete_sequence=["#147D64", "#D97706", "#2563EB", "#64748B"]
    )
    performance_fig.update_layout(height=450, margin=dict(l=40, r=20, t=55, b=40), yaxis_range=[0, 100])
    performance_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    charts = {
        "map": plot_json(map_fig),
        "magnitude": plot_json(magnitude_fig),
        "trend": plot_json(trend_fig),
        "depth": plot_json(depth_fig),
        "performance": plot_json(performance_fig),
    }
    return render_template("dashboard.html", charts=charts, stats=latest_model_data())


@app.route("/analysis")
def analysis():
    df = load_data()
    if df.empty:
        return render_template("analysis.html", stats={}, class_counts={}, missing={}, cleaned=0)

    class_counts = df["magnitude_class"].value_counts().to_dict() if "magnitude_class" in df else {}
    missing = df[["time", "latitude", "longitude", "depth", "mag"]].isna().sum().to_dict()

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)

    return render_template(
        "analysis.html",
        stats=latest_model_data(),
        class_counts=class_counts,
        missing=missing,
        cleaned=len(df),
        metrics=metrics,
        columns=["time", "latitude", "longitude", "depth", "mag", "magnitude_class"]
    )


@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form
    try:
        year = int(payload["year"])
        month = int(payload["month"])
        latitude = float(payload["latitude"])
        longitude = float(payload["longitude"])
        depth = float(payload["depth"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Please provide valid year, month, coordinates and depth."}), 400

    if not (2016 <= year <= 2025):
        return jsonify({"error": "Year must be between 2016 and 2025."}), 400
    if not (1 <= month <= 12):
        return jsonify({"error": "Month must be between 1 and 12."}), 400
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return jsonify({"error": "Latitude or longitude is outside the valid range."}), 400
    if depth < 0:
        return jsonify({"error": "Depth cannot be negative."}), 400

    orange_path = "models/orange_magnitude_model.joblib"
    if os.path.exists(orange_path):
        try:
            orange_bundle = joblib.load(orange_path)
            orange_domain = orange_bundle["domain"]
            row = Orange.data.Table.from_list(orange_domain, [[float(year), float(month), latitude, longitude, depth]])[0]
            prediction_value = orange_bundle["orange_model"](row)
            prediction = str(orange_domain.class_var.values[int(prediction_value)])
            orange_probabilities = orange_bundle["orange_model"].predict_proba(row)
            score_values = orange_probabilities[0] if len(orange_probabilities.shape) == 2 else orange_probabilities
            orange_labels = [str(label) for label in orange_domain.class_var.values]
            probabilities = {}
            for label in ("Low", "Moderate", "High"):
                try:
                    index = orange_labels.index(label)
                    probabilities[label] = round(float(score_values[index]) * 100, 1)
                except (ValueError, IndexError):
                    probabilities[label] = 0.0
            prediction = max(probabilities, key=probabilities.get)
            model_name = "Orange " + orange_bundle["model_name"]
        except Exception:
            prediction = None
            probabilities = None

    if "orange_path" not in locals() or not os.path.exists(orange_path):
        prediction = None
        probabilities = None

    if prediction is None:
        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "The trained model is not available. Run preprocess_train.py first."}), 503

        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        features = pd.DataFrame([[year, month, latitude, longitude, depth]], columns=bundle["features"])
        prediction = str(model.predict(features)[0])
        model_name = bundle["selected_model"]

        probabilities = None
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(features)[0]
            probabilities = {}
            for label in ("Low", "Moderate", "High"):
                try:
                    index = list(model.classes_).index(label)
                    probabilities[label] = round(float(scores[index]) * 100, 1)
                except (ValueError, IndexError):
                    probabilities[label] = 0.0
            prediction = max(probabilities, key=probabilities.get)

    confidence = round(float(probabilities[prediction]), 1)

    advice = {
        "Low": "Historical pattern category: low magnitude range (below 5.5).",
        "Moderate": "Historical pattern category: moderate magnitude range (5.5–6.49).",
        "High": "Historical pattern category: high magnitude range (6.5 or above)."
    }

    return jsonify({
        "category": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "advice": advice[prediction],
        "model": model_name
    })


@app.route("/management")
def management():
    return render_template("management.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=False)
