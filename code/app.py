import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
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
MODEL_CACHE = {"model": None, "scaler": None, "selected_features": []}

@app.route("/tools", methods=["GET"])
def get_tools():
    return jsonify(TOOLS)

@app.route("/tool/optimize", methods=["POST"])
def tool_optimize():
    try:
        from tuner import optimize_model  # Assume tuner.py with Optuna logic
        results = optimize_model(trials=int(os.getenv('OPTUNA_TRIALS', 50)))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tool/write_code", methods=["POST"])
def tool_write_code():
    data = request.json
    title = data.get("title")
    content = data.get("content")
    if not title or not content:
        return jsonify({"error": "Missing title or content"}), 400
    try:
        compile(content, title, 'exec')
    except SyntaxError as e:
        return jsonify({"error": f"Syntax error: {str(e)}"}), 400
    with open(title, "w") as f:
        f.write(content)
    return jsonify({"status": "success"})

@app.route("/tool/commit_to_github", methods=["POST"])
def tool_commit():
    data = request.json
    message = data.get("message")
    files = data.get("files", [])
    if not message:
        return jsonify({"error": "Missing commit message"}), 400
    try:
        subprocess.run(["git", "add"] + files, check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)
        return jsonify({"status": "committed"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

@app.route("/inference/<token>", methods=["GET"])
def inference(token: str):
    try:
        refresh = request.args.get("refresh", "0") == "1"
        if MODEL_CACHE["model"] is None or refresh:
            from model import train_model
            model, scaler, selected_features = train_model()
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["selected_features"] = selected_features
        from data_processor import get_latest_features  # Assume this fetches features including VADER sentiment, fixes NaNs, blends synthetic data
        latest_features = get_latest_features(token, MODEL_CACHE["selected_features"])
        features_array = np.array([latest_features[f] for f in MODEL_CACHE["selected_features"]]).reshape(1, -1)
        scaled_features = MODEL_CACHE["scaler"].transform(features_array)
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        # Add smoothing (e.g., simple moving average if ensemble, but assume in model)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=FLASK_PORT)