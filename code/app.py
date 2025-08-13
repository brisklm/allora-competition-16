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

@app.route("/inference/<token>", methods=["GET"])
def inference(token: str):
    try:
        # Train lazily or on refresh
        refresh = request.args.get("refresh", "0") == "1"
        if MODEL_CACHE["model"] is None or refresh:
            from model import train_model, get_selected_features
            model, scaler, metrics = train_model()
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["selected_features"] = get_selected_features()

        # Assume get_latest_features is defined in model.py for fetching/engineering latest features
        from model import get_latest_features
        features_df = get_latest_features(token)
        selected = features_df[MODEL_CACHE["selected_features"]].values.reshape(1, -1)
        scaled = MODEL_CACHE["scaler"].transform(selected)
        prediction = MODEL_CACHE["model"].predict(scaled)[0]
        return jsonify({"log_return_prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Tool implementations
@app.route("/tools/optimize", methods=["POST"])
def do_optimize():
    try:
        import optuna
        from config import OPTUNA_TRIALS
        from model import objective  # Assume objective function for Optuna in model.py
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        return jsonify({"best_params": study.best_params, "best_value": study.best_value})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tools/write_code", methods=["POST"])
def do_write_code():
    data = request.json
    title = data.get("title")
    content = data.get("content")
    if not title or not content:
        return jsonify({"error": "Missing title or content"}), 400
    try:
        ast.parse(content)
    except SyntaxError as e:
        return jsonify({"error": f"Syntax error: {str(e)}"}), 400
    with open(title, "w") as f:
        f.write(content)
    return jsonify({"success": True, "file": title})

@app.route("/tools/commit_to_github", methods=["POST"])
def do_commit_to_github():
    data = request.json
    message = data.get("message")
    files = data.get("files", [])
    if not message:
        return jsonify({"error": "Missing commit message"}), 400
    try:
        repo = git.Repo(os.getcwd())
        repo.git.add(files)
        repo.git.commit(m=message)
        repo.git.push()
        return jsonify({"success": True, "message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)