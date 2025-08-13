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
MODEL_CACHE = {"model": None, "selected_features": [], "scaler": None, "metrics": None}

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
        # Get current features for the token
        from features import get_current_features  # assuming this exists and handles sentiment, etc.
        features = get_current_features(token, MODEL_CACHE["selected_features"])
        # Fix NaNs and low variance
        features = np.nan_to_num(features, nan=0.0)
        # Assuming scaling handles low variance
        scaled_features = MODEL_CACHE["scaler"].transform([features])
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        # Stabilize with simple ensembling or smoothing (placeholder: average with previous if available)
        if "last_prediction" in MODEL_CACHE:
            prediction = 0.7 * prediction + 0.3 * MODEL_CACHE["last_prediction"]
        MODEL_CACHE["last_prediction"] = prediction
        return jsonify({"prediction": float(prediction), "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tools", methods=["GET"])
def get_tools():
    return jsonify(TOOLS)

@app.route("/call_tool", methods=["POST"])
def call_tool():
    data = request.json
    tool_name = data.get("name")
    params = data.get("parameters", {})
    if tool_name == "optimize":
        # Trigger Optuna optimization, blend data, tune for lower ZPTAE, higher R2, etc.
        from optimizer import run_optuna_optimization  # assuming implemented with suggestions
        results = run_optuna_optimization()  # Includes tuning n_estimators, learning_rate, adding lags, etc.
        return jsonify({"results": results})
    elif tool_name == "write_code":
        filename = params["title"]
        content = params["content"]
        content_type = params.get("contentType", "text/python")
        try:
            compile(content, filename, "exec")
        except SyntaxError as e:
            return jsonify({"error": "Syntax error: " + str(e)}), 400
        with open(filename, "w") as f:
            f.write(content)
        artifact_id = params.get("artifact_id", str(np.random.uuid4()))
        return jsonify({"success": True, "artifact_id": artifact_id})
    elif tool_name == "commit_to_github":
        message = params["message"]
        files = params["files"]
        try:
            os.system("git add " + " ".join(files))
            os.system(f'git commit -m "{message}"')
            os.system("git push")
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unknown tool"}), 400

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)