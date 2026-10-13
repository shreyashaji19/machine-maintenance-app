"""Flask web app.  Run:  python app.py   then open http://127.0.0.1:5000"""
from flask import Flask, jsonify, render_template, request

from src.predict import get_metrics, predict_one

app = Flask(__name__)

LIMITS = {"air_temp": (290, 310), "process_temp": (300, 320), "rpm": (1000, 3000),
          "torque": (0, 100), "tool_wear": (0, 300)}


@app.route("/")
def home():
    return render_template("index.html")


@app.get("/api/metrics")
def metrics():
    try:
        return jsonify(get_metrics())
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        for key, (lo, hi) in LIMITS.items():
            if not lo <= float(data[key]) <= hi:
                return jsonify(error=f"{key} must be between {lo} and {hi}"), 400
        if data.get("type") not in ("L", "M", "H"):
            return jsonify(error="type must be L, M or H"), 400
        return jsonify(predict_one(data))
    except (KeyError, ValueError, TypeError):
        return jsonify(error="Send all five sensor values and a type."), 400
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503


if __name__ == "__main__":
    app.run(debug=True)
