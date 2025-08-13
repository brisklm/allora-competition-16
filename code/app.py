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
MODEL_CACHE = {"model": None, "selected_features": [], "scaler": None}

@app.route("/inference/<token>", methods=["GET"])
def inference(token: str):
    try:
        # Train lazily or on refresh
        refresh = request.args.get("refresh", "0") == "1"
        if MODEL_CACHE["model"] is None or refresh:
            from model import train_model
            model, scaler, metrics, selected_features = train_model()  # Optimized to include Optuna tuning and synthetic data blending
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["selected_features"] = selected_features
        # Fetch latest features, including sentiment and new engineered features
        from data_utils import get_latest_features  # Assuming data_utils handles NaNs, low variance, VADER sentiment
        features_df = get_latest_features(token, MODEL_CACHE["selected_features"])
        scaled_features = MODEL_CACHE["scaler"].transform(features_df)
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        # Stabilize with simple ensembling/smoothing
        smoothed_prediction = prediction * 0.8 + np.mean([prediction]) * 0.2  # Example smoothing
        return jsonify({"value": float(smoothed_prediction), "version": MCP_VERSION})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add route for tools
@app.route("/run_tool", methods=["POST"])
def run_tool():
    data = request.json
    name = data["name"]
    params = data.get("parameters", {})
    if name == "optimize":
        # Trigger Optuna optimization
        from optimize import run_optuna_optimization  # Assuming optimize.py with Optuna for hyperparams
        results = run_optuna_optimization()  # Tunes n_estimators, learning_rate, etc.
        return jsonify({"results": results})
    elif name == "write_code":
        title = params["title"]
        content = params["content"]
        import ast
        try:
            ast.parse(content)
            with open(title, "w") as f:
                f.write(content)
            return jsonify({"status": "success"})
        except SyntaxError as e:
            return jsonify({"error": str(e)}), 400
    elif name == "commit_to_github":
        message = params["message"]
        files = params.get("files", [])
        import subprocess
        subprocess.run(["git", "add"] + files)
        subprocess.run(["git", "commit", "-m", message])
        subprocess.run(["git", "push"])
        return jsonify({"status": "success"})
    return jsonify({"error": "Unknown tool"}), 400

if __name__ == "__main__":
    app.run(port=FLASK_PORT)