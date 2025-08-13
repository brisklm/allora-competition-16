import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import ast
import git

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
MODEL_CACHE = {"model": None, "selected_features": [], "scaler": None, "metrics": None}

@app.route("/tools", methods=["GET"])
def get_tools():
    return jsonify(TOOLS)

@app.route("/tool/optimize", methods=["POST"])
def tool_optimize():
    try:
        from model import optimize_model
        results = optimize_model()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tool/write_code", methods=["POST"])
def tool_write_code():
    try:
        data = request.json
        title = data["title"]
        content = data["content"]
        try:
            ast.parse(content)
        except SyntaxError as e:
            return jsonify({"error": f"Syntax error: {str(e)}"}), 400
        with open(title, "w") as f:
            f.write(content)
        return jsonify({"success": True, "message": f"File {title} written successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tool/commit_to_github", methods=["POST"])
def tool_commit_to_github():
    try:
        data = request.json
        message = data["message"]
        files = data.get("files", [])
        repo = git.Repo(os.getcwd())
        if files:
            repo.git.add(files)
        else:
            repo.git.add(".")
        repo.git.commit("-m", message)
        repo.git.push()
        return jsonify({"success": True, "message": "Committed and pushed to GitHub"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/inference/<token>", methods=["GET"])
def inference(token: str):
    try:
        refresh = request.args.get("refresh", "0") == "1"
        if MODEL_CACHE["model"] is None or refresh:
            from model import train_model
            model, scaler, metrics, selected_features = train_model()
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["metrics"] = metrics
            MODEL_CACHE["selected_features"] = selected_features
        from model import get_current_features
        features = get_current_features(token)
        scaled_features = MODEL_CACHE["scaler"].transform([features])
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        smoothed_prediction = 0.8 * prediction + 0.2 * np.mean([prediction])  # Simple smoothing example
        response = {"prediction": smoothed_prediction, "timestamp": datetime.utcnow().isoformat(), "version": MCP_VERSION}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)