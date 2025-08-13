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
MODEL_CACHE = {"model": None, "scaler": None, "selected_features": []}

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
        # Fetch latest features (assume data.py has get_latest_features)
        from data import get_latest_features
        latest_features = get_latest_features(token, MODEL_CACHE["selected_features"])
        scaled_features = MODEL_CACHE["scaler"].transform([latest_features])
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        # Stabilize with simple moving average if ensemble (assume single for now)
        return jsonify({"log_return_prediction": float(prediction), "timestamp": datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/execute", methods=["POST"])
def execute():
    data = request.json
    tool_name = data.get("name")
    params = data.get("parameters", {})
    if tool_name == "optimize":
        from model import optimize_model
        results = optimize_model()  # Assumes Optuna tuning in model.py
        return jsonify(results)
    elif tool_name == "write_code":
        filename = params["title"]
        content = params["content"]
        # Simple syntax validation placeholder
        try:
            compile(content, filename, 'exec')
        except SyntaxError as e:
            return jsonify({"error": f"Syntax error: {str(e)}"}), 400
        with open(filename, "w") as f:
            f.write(content)
        return jsonify({"status": "success"})
    elif tool_name == "commit_to_github":
        # Placeholder for git commit
        return jsonify({"status": "committed"})
    else:
        return jsonify({"error": "Unknown tool"}), 400

if __name__ == "__main__":
    app.run(port=FLASK_PORT)