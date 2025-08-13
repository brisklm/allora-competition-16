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
            "title": {"type": "string", "description": "Filename (e.g., model.py)", "required": true},
            "content": {"type": "string", "description": "Complete source code content", "required": true},
            "artifact_id": {"type": "string", "description": "Artifact UUID", "required": false},
            "artifact_version_id": {"type": "string", "description": "Version UUID", "required": false},
            "contentType": {"type": "string", "description": "Content type (e.g., text/python)", "required": false}
        }
    },
    {
        "name": "commit_to_github",
        "description": "Commits changes to GitHub repository.",
        "parameters": {
            "message": {"type": "string", "description": "Commit message", "required": true},
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
            from model import train_model, get_latest_features
            model, scaler, selected_features = train_model()
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["selected_features"] = selected_features
        else:
            model = MODEL_CACHE["model"]
            scaler = MODEL_CACHE["scaler"]
            selected_features = MODEL_CACHE["selected_features"]

        features = get_latest_features(selected_features, token)
        scaled_features = scaler.transform(np.array([features]))
        prediction = model.predict(scaled_features)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tools", methods=["POST"])
def call_tool():
    data = request.json
    tool_name = data.get("name")
    params = data.get("parameters", {})

    if tool_name == "optimize":
        from optimize import run_optuna_tuning
        results = run_optuna_tuning()
        return jsonify(results)

    elif tool_name == "write_code":
        title = params.get("title")
        content = params.get("content")
        if not title or not content:
            return jsonify({"error": "Missing parameters"}), 400
        try:
            compile(content, title, 'exec')
            with open(title, "w") as f:
                f.write(content)
            return jsonify({"status": "success"})
        except SyntaxError as e:
            return jsonify({"error": f"Syntax error: {str(e)}"}), 400

    elif tool_name == "commit_to_github":
        message = params.get("message")
        files = params.get("files", [])
        import git
        repo = git.Repo(os.getcwd())
        repo.index.add(files)
        repo.index.commit(message)
        origin = repo.remote(name='origin')
        origin.push()
        return jsonify({"status": "success"})

    else:
        return jsonify({"error": "Unknown tool"}), 404

if __name__ == "__main__":
    app.run(port=FLASK_PORT)