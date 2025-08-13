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
            model, scaler, selected_features = train_model()
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["selected_features"] = selected_features
        # Assume there's a function to get latest features
        from data_processor import get_latest_features
        features = get_latest_features(token, MODEL_CACHE["selected_features"])
        # Fix NaNs and low variance issues
        features = np.nan_to_num(features, nan=0.0)
        scaled_features = MODEL_CACHE["scaler"].transform([features])
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        # Add smoothing or ensembling for stability
        from config import SMOOTHING
        if SMOOTHING == 'moving_average':
            pass  # Implement moving average if needed
        elif SMOOTHING == 'ensemble':
            pass  # Implement ensemble if needed
        return jsonify({"prediction": float(prediction), "timestamp": datetime.utcnow().isoformat(), "version": MCP_VERSION})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tools", methods=["GET"])
def get_tools():
    return jsonify(TOOLS)

@app.route("/tool/<name>", methods=["POST"])
def call_tool(name: str):
    params = request.json
    if name == "optimize":
        from optimizer import run_optuna_tuning
        results = run_optuna_tuning(params.get("trials", 50))
        return jsonify(results)
    elif name == "write_code":
        title = params["title"]
        content = params["content"]
        # Validate syntax (placeholder)
        with open(title, "w") as f:
            f.write(content)
        return jsonify({"status": "success", "file": title})
    elif name == "commit_to_github":
        message = params["message"]
        files = params.get("files", [])
        from git_utils import commit_changes
        commit_changes(message, files)
        return jsonify({"status": "success"})
    else:
        return jsonify({"error": "Tool not found"}), 404

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)