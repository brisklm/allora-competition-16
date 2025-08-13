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
MODEL_CACHE = {"model": None, "selected_features": []}

@app.route("/tool", methods=["POST"])
def tool():
    data = request.json
    tool_name = data.get("name")
    params = data.get("parameters", {})
    if tool_name == "optimize":
        from model import tune_model_with_optuna
        results = tune_model_with_optuna()
        return jsonify(results)
    elif tool_name == "write_code":
        filename = params.get("title")
        content = params.get("content")
        if not filename or not content:
            return jsonify({"error": "Missing parameters"}), 400
        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            return jsonify({"error": str(e)}), 400
        with open(filename, "w") as f:
            f.write(content)
        return jsonify({"success": True})
    elif tool_name == "commit_to_github":
        message = params.get("message")
        files = params.get("files", [])
        if not message:
            return jsonify({"error": "Missing message"}), 400
        import subprocess
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
    else:
        return jsonify({"error": "Unknown tool"}), 400

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
        from model import get_latest_features
        features = get_latest_features(token, MODEL_CACHE["selected_features"])
        if features is None:
            return jsonify({"error": "Failed to get features"}), 500
        scaled_features = MODEL_CACHE["scaler"].transform(np.array([features]))
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        smoothed_prediction = (prediction + np.mean([prediction] * 3)) / 2  # Simple smoothing example
        return jsonify({"prediction": float(smoothed_prediction), "timestamp": datetime.utcnow().isoformat(), "version": MCP_VERSION})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)