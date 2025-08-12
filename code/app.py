import os
import json
import ast
import shutil
from datetime import datetime
from flask import Flask, request, Response
from github import Github
from dotenv import load_dotenv
try:
    from model import train_model, get_inference
    from config import TIMEFRAME, training_price_data_path, TOKEN, REGION, DATA_PROVIDER, SELECTED_FEATURES, MODEL_PARAMS, OPTUNA_TRIALS, USE_SYNTHETIC_DATA
except ImportError as e:
    print(f"[{datetime.now()}] ImportError: {str(e)}. Ensure model.py and config.py are available.")
    raise
import numpy as np

app = Flask(__name__)

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME", "brisklm/allora-competition-16")
BACKUP_DIR = os.path.join(os.getcwd(), "backup")
MCP_VERSION = "2025-07-23-competition16-topic62-app-v2-optimized"
FLASK_PORT = int(os.getenv("FLASK_PORT", 8001))

# MCP Tools
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning and returns results. Incorporates VADER sentiment and LSTM hybrid model.",
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
        "description": "Commits specified files to the GitHub repository after optimization.",
        "parameters": {
            "file_paths": {"type": "array", "description": "List of file paths to commit", "required": True},
            "commit_message": {"type": "string", "description": "Commit message", "required": True}  # Completed from original
        }
    }
]

# Ensure compatibility with existing setup
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT)