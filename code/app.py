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

@app.route("/tools", methods=["GET"])
def get_tools():
    return jsonify(TOOLS)

@app.route("/", methods=["POST"])
def handle_tool():
    data = request.json
    tool_name = data.get("name")
    params = data.get("parameters", {})
    if tool_name == "optimize":
        from model import optuna_optimize
        results = optuna_optimize()
        return jsonify({"results": results})
    elif tool_name == "write_code":
        title = params["title"]
        content = params["content"]
        # Simple syntax validation (e.g., try to compile)
        try:
            compile(content, title, 'exec')
        except SyntaxError as e:
            return jsonify({"error": f"Syntax error: {str(e)}"}), 400
        with open(title, "w") as f:
            f.write(content)
        return jsonify({"status": "success"})
    elif tool_name == "commit_to_github":
        message = params["message"]
        files = params.get("files", [])
        if files:
            os.system("git add " + " ".join(files))
        os.system(f'git commit -m "{message}"')
        os.system("git push")
        return jsonify({"status": "success"})
    else:
        return jsonify({"error": "Unknown tool"}), 400

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
        # Assume data module for latest features
        from data import get_latest_features
        latest_data = get_latest_features(token)
        features = latest_data[MODEL_CACHE["selected_features"]].values.reshape(1, -1)
        scaled_features = MODEL_CACHE["scaler"].transform(features)
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        return jsonify({"prediction": float(prediction), "timestamp": datetime.utcnow().isoformat(), "version": MCP_VERSION})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)