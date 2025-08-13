import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import subprocess
from model import train_model, optuna_tune  # Assuming optuna_tune is in model.py

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
MODEL_CACHE = {"model": None, "selected_features": [], "scaler": None}


def optimize(parameters):
    # Trigger Optuna tuning
    results = optuna_tune()  # Assume this runs Optuna and returns best params and metrics
    return results

def write_code(parameters):
    title = parameters["title"]
    content = parameters["content"]
    # Validate syntax
    try:
        compile(content, title, 'exec')
    except SyntaxError as e:
        return {"error": str(e)}
    with open(title, 'w') as f:
        f.write(content)
    return {"success": True}

def commit_to_github(parameters):
    message = parameters["message"]
    files = parameters.get("files", [])
    try:
        for file in files:
            subprocess.run(["git", "add", file], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)
        return {"success": True}
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}

@app.route("/tool", methods=["POST"])
def call_tool():
    data = request.json
    tool_name = data.get("name")
    params = data.get("parameters", {})
    if tool_name == "optimize":
        result = optimize(params)
    elif tool_name == "write_code":
        result = write_code(params)
    elif tool_name == "commit_to_github":
        result = commit_to_github(params)
    else:
        return jsonify({"error": "Unknown tool"}), 400
    return jsonify(result)

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
        if token != "SOL":
            return jsonify({"error": "Only SOL supported"}), 400
        # Assume get_latest_features exists in data.py, includes sentiment, fixes NaNs, blends synthetic
        from data import get_latest_features
        features_df = get_latest_features(token, MODEL_CACHE["selected_features"])
        if features_df is None or len(features_df) == 0:
            return jsonify({"error": "No data available"}), 400
        scaled_features = MODEL_CACHE["scaler"].transform(np.array([features_df]))
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        # Stabilize with smoothing (simple moving average example, assume ensemble in model)
        # For demo, just return
        return jsonify({"log_return_prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)