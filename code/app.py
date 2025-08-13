import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import ast
import subprocess

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
MODEL_CACHE = {"model": None, "selected_features": []}


@app.route("/inference/<token>", methods=["GET"])
def inference(token: str):
    try:
        # Train lazily or on refresh
        refresh = request.args.get("refresh", "0") == "1"
        if MODEL_CACHE["model"] is None or refresh:
            from model import train_model
            model, scaler, metrics, selected_features = train_model()
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["metrics"] = metrics
            MODEL_CACHE["selected_features"] = selected_features
        
        # Get latest features
        from data import prepare_prediction_input
        input_data = prepare_prediction_input(token, MODEL_CACHE["selected_features"])
        scaled_input = MODEL_CACHE["scaler"].transform(np.array([input_data]))
        
        # Predict
        prediction = MODEL_CACHE["model"].predict(scaled_input)[0]
        
        # For stability, apply simple moving average smoothing if previous predictions exist (assuming cache has prev)
        if "prev_prediction" in MODEL_CACHE:
            prediction = (prediction + MODEL_CACHE["prev_prediction"]) / 2
        MODEL_CACHE["prev_prediction"] = prediction
        
        return jsonify({"prediction": float(prediction), "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tools/optimize", methods=["POST"])
def tool_optimize():
    from model import optimize_model
    results = optimize_model()
    return jsonify(results)

@app.route("/tools/write_code", methods=["POST"])
def tool_write_code():
    data = request.json
    title = data.get("title")
    content = data.get("content")
    contentType = data.get("contentType", "text/python")
    artifact_id = data.get("artifact_id")
    artifact_version_id = data.get("artifact_version_id")
    try:
        ast.parse(content)
    except SyntaxError as e:
        return jsonify({"error": "Syntax error: " + str(e)}), 400
    with open(title, "w") as f:
        f.write(content)
    return jsonify({"success": True, "artifact_id": artifact_id, "artifact_version_id": artifact_version_id})

@app.route("/tools/commit_to_github", methods=["POST"])
def tool_commit():
    data = request.json
    message = data.get("message")
    files = data.get("files", [])
    try:
        if files:
            subprocess.run(["git", "add"] + files, check=True)
        else:
            subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)