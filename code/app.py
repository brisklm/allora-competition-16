import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np

# Initialize app and env
app = Flask(__name__)
load_dotenv()

MCP_VERSION = "2025-07-23-competition16-topic62-app-v2-optimized"
FLASK_PORT = int(os.getenv("FLASK_PORT", 8001))

# MCP Tools
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning and returns results.",
        "parameters": {}
    },
    {
        "name": "write_code",
        "description": "Writes complete source code to a specified file, overwriting existing content after syntax validation.",
        "parameters": {
            "title": {"type": "string", "description": "Filename (e.g., model.py)", "required": True},
            "content": {"type": "string", "description": "Complete source code content", "required": True},
            "artifact_id": {"type": "string", "description": "Artifact UUID", "required": False},
            "artifact_version_id": {"type": "string", "description": "Version UUID", "required": False},
            "contentType": {"type": "string", "description": "Content type (e.g., text/python)", "required": False}
        }
    },
    {
        "name": "commit_to_github",
        "description": "Commits changes to GitHub repository.",
        "parameters": {
            "message": {"type": "string", "description": "Commit message", "required": True},
            "files": {"type": "array", "description": "List of files to commit", "items": {"type": "string"}}
        }
    }
]

# In-memory cache for inference
MODEL_CACHE = {"model": None, "selected_features": [], "scaler": None, "ensemble_models": []}

@app.route("/inference/<token>", methods=["GET"])
def inference(token: str):
    try:
        # Train lazily or on refresh
        refresh = request.args.get("refresh", "0") == "1"
        if MODEL_CACHE["model"] is None or refresh:
            from model import train_model
            model, scaler, metrics, ensemble_models = train_model()  # Assuming updated train_model returns ensemble for stabilizing
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["metrics"] = metrics
            MODEL_CACHE["ensemble_models"] = ensemble_models or []
            from config import SELECTED_FEATURES
            MODEL_CACHE["selected_features"] = SELECTED_FEATURES
        # Fetch latest features (assume data.py handles VADER, lags, ratios, NaN fixes, synthetic blend)
        from data import get_latest_features  # Assume this exists and incorporates optimizations
        features = get_latest_features(token)
        scaled_features = MODEL_CACHE["scaler"].transform([features])
        # Predict with ensemble or smoothing for stability
        if MODEL_CACHE["ensemble_models"]:
            predictions = [m.predict(scaled_features)[0] for m in MODEL_CACHE["ensemble_models"]]
            prediction = np.mean(predictions)  # Ensemble average
        else:
            prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        # Apply smoothing if configured
        from config import SMOOTHING_FACTOR, USE_ENSEMBLE
        if USE_ENSEMBLE:
            prediction = prediction * SMOOTHING_FACTOR + np.random.normal(0, 0.01) * (1 - SMOOTHING_FACTOR)  # Example smoothing
        return jsonify({"prediction": prediction, "metrics": MODEL_CACHE.get("metrics", {})})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=FLASK_PORT)