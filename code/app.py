import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np

# Initialize app and env
app = Flask(__name__)
load_dotenv()

MCP_VERSION = "2025-07-23-competition16-topic62-app-v3-optimized"
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
MODEL_CACHE = {"model": None, "selected_features": [], "scaler": None}


@app.route("/inference/<token>", methods=["GET"])
def inference(token: str):
    try:
        # Train lazily or on refresh
        refresh = request.args.get("refresh", "0") == "1"
        if MODEL_CACHE["model"] is None or refresh:
            from model import train_model
            model, scaler, metrics, selected_features = train_model()  # Assumed return values
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["selected_features"] = selected_features
        
        # Placeholder for getting current features (assume from data or request)
        # For optimization: add ensembling/smoothing
        from model import get_current_features  # Assume this exists
        features = get_current_features(token, MODEL_CACHE["selected_features"])
        scaled_features = MODEL_CACHE["scaler"].transform([features])
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        
        # Stabilize predictions via smoothing (e.g., exponential moving average simulation)
        if 'last_prediction' in MODEL_CACHE:
            prediction = 0.7 * prediction + 0.3 * MODEL_CACHE['last_prediction']
        MODEL_CACHE['last_prediction'] = prediction
        
        # Return prediction
        return jsonify({"prediction": prediction, "version": MCP_VERSION})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)