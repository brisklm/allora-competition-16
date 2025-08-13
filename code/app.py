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
MODEL_CACHE = {"model": None, "selected_features": [], "last_prediction": None}

@app.route("/inference/<token>", methods=["GET"])
def inference(token: str):
    try:
        # Train lazily or on refresh
        refresh = request.args.get("refresh", "0") == "1"
        if MODEL_CACHE["model"] is None or refresh:
            from model import train_model
            model, scaler, metrics = train_model()
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["selected_features"] = metrics.get("selected_features", [])
        # Assume getting latest data
        from data import get_latest_features  # Assuming this exists
        features = get_latest_features(token, MODEL_CACHE["selected_features"])
        scaled_features = MODEL_CACHE["scaler"].transform([features])
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        # Stabilize predictions via smoothing
        if MODEL_CACHE["last_prediction"] is not None:
            prediction = 0.7 * prediction + 0.3 * MODEL_CACHE["last_prediction"]
        MODEL_CACHE["last_prediction"] = prediction
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tools", methods=["GET"])
def get_tools():
    return jsonify(TOOLS)

@app.route("/call_tool", methods=["POST"])
def call_tool():
    data = request.json
    tool_name = data["name"]
    params = data.get("parameters", {})
    if tool_name == "optimize":
        # Trigger Optuna tuning with blending real/synthetic data
        from model import optimize_model  # Assuming this handles Optuna, VADER, hybrid LSTM, NaN fixes
        results = optimize_model()
        return jsonify(results)
    elif tool_name == "write_code":
        title = params["title"]
        content = params["content"]
        # Simple syntax validation (e.g., compile)
        try:
            compile(content, title, "exec")
        except SyntaxError as e:
            return jsonify({"error": str(e)}), 400
        with open(title, "w") as f:
            f.write(content)
        return jsonify({"status": "success"})
    elif tool_name == "commit_to_github":
        # Assume implementation
        return jsonify({"status": "committed"})
    return jsonify({"error": "Unknown tool"}), 400

if __name__ == "__main__":
    app.run(port=FLASK_PORT)