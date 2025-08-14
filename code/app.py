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
        refresh = request.args.get("refresh", "0") == "1"
        if MODEL_CACHE["model"] is None or refresh:
            from model import train_model
            model, scaler, metrics, selected_features = train_model()
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["selected_features"] = selected_features
        # Get latest features and predict
        from model import get_latest_features, predict
        features = get_latest_features(token)
        prediction = predict(MODEL_CACHE["model"], features, MODEL_CACHE["scaler"])
        return jsonify({"prediction": prediction, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tool", methods=["POST"])
def call_tool():
    data = request.json
    tool_name = data.get("name")
    params = data.get("parameters", {})
    if tool_name == "optimize":
        from model import optimize_model
        results = optimize_model()
        return jsonify({"results": results})
    elif tool_name == "write_code":
        title = params.get("title")
        content = params.get("content")
        if not title or not content:
            return jsonify({"error": "Missing title or content"}), 400
        try:
            compile(content, title, "exec")
        except SyntaxError as e:
            return jsonify({"error": f"Syntax error: {str(e)}"}), 400
        with open(title, "w") as f:
            f.write(content)
        return jsonify({"success": True, "message": f"File {title} written"})
    elif tool_name == "commit_to_github":
        message = params.get("message")
        files = params.get("files", [])
        if not message:
            return jsonify({"error": "Missing message"}), 400
        try:
            subprocess.run(["git", "add"] + files, check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unknown tool"}), 400

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)