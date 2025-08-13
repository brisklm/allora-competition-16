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
MODEL_CACHE = {"model": None, "selected_features": [], "scaler": None, "last_prediction": None}

@app.route("/tools", methods=["GET"])
def get_tools():
    return jsonify(TOOLS)

@app.route("/tool", methods=["POST"])
def call_tool():
    data = request.json
    tool_name = data.get("name")
    params = data.get("parameters", {})
    if tool_name == "optimize":
        try:
            import optuna
            from config import MODEL_PARAMS, OPTUNA_TRIALS
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                    'hidden_size': trial.suggest_int('hidden_size', 32, 256),
                    'num_layers': trial.suggest_int('num_layers', 1, 4),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
                }
                from model import evaluate_params  # Assume this exists in model.py to evaluate
                score = evaluate_params(params)
                return score
            study = optuna.create_study(direction="minimize")  # Assuming minimize error
            study.optimize(objective, n_trials=OPTUNA_TRIALS)
            best_params = study.best_params
            # Update config or return
            return jsonify({"best_params": best_params, "best_value": study.best_value})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
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
        return jsonify({"success": True, "message": f"Wrote {title}"})
    elif tool_name == "commit_to_github":
        message = params.get("message")
        files = params.get("files", [])
        if not message:
            return jsonify({"error": "Missing commit message"}), 400
        try:
            if files:
                for file in files:
                    os.system(f"git add {file}")
            else:
                os.system("git add .")
            os.system(f'git commit -m "{message}"')
            os.system("git push")
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
            model, scaler, metrics, selected_features = train_model()  # Assume returns these
            MODEL_CACHE["model"] = model
            MODEL_CACHE["scaler"] = scaler
            MODEL_CACHE["selected_features"] = selected_features
        # Get latest features (assume function in data.py)
        from data import get_latest_features  # Assume this blends real/synthetic, fixes NaNs
        features = get_latest_features(token, MODEL_CACHE["selected_features"])
        scaled_features = MODEL_CACHE["scaler"].transform([features])
        prediction = MODEL_CACHE["model"].predict(scaled_features)[0]
        # Stabilize with smoothing
        if MODEL_CACHE["last_prediction"] is not None:
            prediction = 0.8 * prediction + 0.2 * MODEL_CACHE["last_prediction"]
        MODEL_CACHE["last_prediction"] = prediction
        return jsonify({"prediction": prediction, "timestamp": datetime.utcnow().isoformat(), "version": MCP_VERSION})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)