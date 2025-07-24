import os
import json
import ast
import shutil
from datetime import datetime
from flask import Flask, request, Response
from github import Github
from dotenv import load_dotenv
try:
    from model import train_model
    from config import TIMEFRAME, training_price_data_path
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
MCP_VERSION = "2025-07-23-competition16-topic62-app-v2"
FLASK_PORT = int(os.getenv("FLASK_PORT", 8001))

# MCP Tools
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization and returns results.",
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
        "description": "Commits specified files to the GitHub repository.",
        "parameters": {
            "file_paths": {"type": "array", "description": "List of file paths to commit", "required": True},
            "commit_message": {"type": "string", "description": "Commit message", "default": f"Automated code commit {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
        }
    },
    {
        "name": "update_artifacts",
        "description": "Parses and writes multiple artifacts to the local directory and commits to GitHub, with full rollback on error.",
        "parameters": {
            "artifacts": {"type": "array", "description": "List of artifacts with title, content, artifact_id, artifact_version_id, contentType", "required": True}
        }
    },
    {
        "name": "evaluate_results",
        "description": "Evaluates model performance metrics and suggests code updates if necessary.",
        "parameters": {
            "metrics": {"type": "object", "description": "Model performance metrics (e.g., test_zptae, test_r2, directional_accuracy)", "required": True},
            "model_type": {"type": "string", "description": "Model type (e.g., LightGBM, XGBoost)", "required": True},
            "selected_features": {"type": "array", "description": "List of selected features", "required": True},
            "model_params": {"type": "object", "description": "Model hyperparameters", "required": True}
        }
    }
]

def validate_python_syntax(content):
    try:
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def correct_syntax_error(content, error_message):
    try:
        lines = content.split("\n")
        if "unexpected EOF" in error_message:
            for i, line in enumerate(lines):
                if line.strip().startswith("def ") and ":" not in line:
                    lines[i] = line + ":"
                elif line.strip().startswith("print(") and not line.strip().endswith(")"):
                    lines[i] = line + ")"
        elif "invalid syntax" in error_message:
            indent_level = 0
            corrected_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    corrected_lines.append(line)
                    continue
                if stripped.startswith("}") or stripped.startswith(")") or stripped.startswith("]"):
                    indent_level = max(0, indent_level - 1)
                corrected_lines.append("    " * indent_level + stripped)
                if stripped.endswith(":") or stripped.startswith("{") or stripped.startswith("(") or stripped.startswith("["):
                    indent_level += 1
            lines = corrected_lines
        return "\n".join(lines)
    except Exception as e:
        print(f"[{datetime.now()}] Error correcting syntax: {str(e)}")
        return content

def backup_files(files):
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        for file in files:
            if os.path.exists(file):
                shutil.copy(file, os.path.join(BACKUP_DIR, os.path.basename(file)))
                print(f"[{datetime.now()}] Backed up {file} to {BACKUP_DIR}")
    except Exception as e:
        print(f"[{datetime.now()}] Error backing up files: {str(e)}")

def rollback_files(files):
    try:
        for file in files:
            backup_path = os.path.join(BACKUP_DIR, os.path.basename(file))
            if os.path.exists(backup_path):
                shutil.copy(backup_path, file)
                print(f"[{datetime.now()}] Restored {file} from backup")
            else:
                g = Github(GITHUB_TOKEN)
                repo = g.get_repo(REPO_NAME)
                try:
                    contents = repo.get_contents(f"code/{file}", ref="main")
                    with open(file, "wb") as f:
                        f.write(contents.decoded_content)
                    print(f"[{datetime.now()}] Restored {file} from GitHub")
                except Exception as e:
                    print(f"[{datetime.now()}] Error restoring {file} from GitHub: {str(e)}")
    except Exception as e:
        print(f"[{datetime.now()}] Error rolling back files: {str(e)}")

def write_code_tool(params):
    try:
        title = params.get("title")
        content = params.get("content")
        artifact_id = params.get("artifact_id", "unknown")
        artifact_version_id = params.get("artifact_version_id", "unknown")
        content_type = params.get("contentType", "text/plain")
        
        if not title or not content:
            return {"status": "error", "message": "Missing title or content"}
        
        if content_type == "text/python" and not title.endswith(".py"):
            return {"status": "error", "message": f"Invalid file extension for {content_type}: {title}"}
        
        updated_content = content
        if content_type == "text/python":
            is_valid, error = validate_python_syntax(content)
            if not is_valid:
                print(f"[{datetime.now()}] SyntaxError in {title}: {error}")
                updated_content = correct_syntax_error(content, error)
                is_valid, error = validate_python_syntax(updated_content)
                if not is_valid:
                    return {"status": "error", "message": f"Failed to correct SyntaxError in {title}: {error}"}
        
        file_path = os.path.join(os.getcwd(), title)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(updated_content)
        print(f"[{datetime.now()}] Wrote complete source code to {file_path} (artifact_id: {artifact_id}, version: {artifact_version_id}, type: {content_type})")
        
        return {"status": "success", "message": f"Wrote code to {file_path}", "file_path": file_path}
    except Exception as e:
        print(f"[{datetime.now()}] Error in write_code_tool: {str(e)}")
        return {"status": "error", "message": str(e)}

def commit_to_github_tool(params):
    try:
        file_paths = params.get("file_paths", [])
        commit_message = params.get("commit_message", f"Automated code commit {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not file_paths:
            return {"status": "error", "message": "No file paths provided"}
        
        g = Github(GITHUB_TOKEN)
        try:
            repo = g.get_repo(REPO_NAME)
            print(f"[{datetime.now()}] Successfully accessed repository {REPO_NAME}")
        except Exception as e:
            print(f"[{datetime.now()}] Failed to access repository {REPO_NAME}: {str(e)}")
            return {"status": "error", "message": f"Failed to access repository {REPO_NAME}: {str(e)}"}
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"[{datetime.now()}] File not found: {file_path}")
                continue
            with open(file_path, "rb") as f:
                content = f.read()
            repo_path = f"code/{os.path.basename(file_path)}"
            try:
                contents = repo.get_contents(repo_path, ref="main")
                repo.update_file(repo_path, commit_message, content, contents.sha, branch="main")
                print(f"[{datetime.now()}] Updated {file_path} at {repo_path}")
            except:
                try:
                    repo.create_file(repo_path, commit_message, content, branch="main")
                    print(f"[{datetime.now()}] Created {file_path} at {repo_path}")
                except Exception as e:
                    print(f"[{datetime.now()}] Failed to commit {file_path}: {str(e)}")
                    return {"status": "error", "message": f"Failed to commit {file_path}: {str(e)}"}
        
        return {"status": "success", "message": f"Committed {len(file_paths)} files to {REPO_NAME}"}
    except Exception as e:
        print(f"[{datetime.now()}] Error in commit_to_github_tool: {str(e)}")
        return {"status": "error", "message": str(e)}

def update_artifacts_tool(params):
    try:
        artifacts = params.get("artifacts", [])
        if not artifacts:
            return {"status": "error", "message": "No artifacts provided"}
        
        files_to_update = ["model.py", "update_app.py", "mcp_client.py", "app.py", "config.py", "updater.py"]
        backup_files(files_to_update)
        
        for artifact in artifacts:
            title = artifact.get("title")
            content = artifact.get("content")
            artifact_id = artifact.get("artifact_id", "unknown")
            artifact_version_id = artifact.get("artifact_version_id", "unknown")
            content_type = artifact.get("contentType", "text/plain")
            if not title or not content:
                print(f"[{datetime.now()}] Skipping artifact with missing title or content: {artifact_id}")
                rollback_files(files_to_update)
                return {"status": "error", "message": f"Missing title or content for {artifact_id}"}
            if content_type == "text/python":
                is_valid, error = validate_python_syntax(content)
                if not is_valid:
                    print(f"[{datetime.now()}] SyntaxError in {title}: {error}")
                    corrected_content = correct_syntax_error(content, error)
                    is_valid, error = validate_python_syntax(corrected_content)
                    if not is_valid:
                        rollback_files(files_to_update)
                        return {"status": "error", "message": f"Failed to correct SyntaxError in {title}: {error}"}
                    artifact["content"] = corrected_content
        
        file_paths = []
        for artifact in artifacts:
            write_result = write_code_tool(artifact)
            if write_result["status"] == "success":
                file_paths.append(write_result["file_path"])
            else:
                print(f"[{datetime.now()}] Failed to write {artifact.get('title')}: {write_result['message']}")
                rollback_files(files_to_update)
                return {"status": "error", "message": f"Failed to write {artifact.get('title')}: {write_result['message']}"}
        
        if file_paths:
            commit_result = commit_to_github_tool({
                "file_paths": file_paths,
                "commit_message": f"Automated code update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            })
            if commit_result["status"] != "success":
                rollback_files(files_to_update)
                return {"status": "error", "message": f"Failed to commit to GitHub: {commit_result['message']}"}
        
        return {"status": "success", "message": f"Updated {len(file_paths)} artifacts and committed to GitHub"}
    except Exception as e:
        print(f"[{datetime.now()}] Error in update_artifacts_tool: {str(e)}")
        rollback_files(["model.py", "update_app.py", "mcp_client.py", "app.py", "config.py", "updater.py"])
        return {"status": "error", "message": str(e)}

def evaluate_results_tool(params):
    try:
        metrics = params.get("metrics", {})
        model_type = params.get("model_type")
        selected_features = params.get("selected_features", [])
        model_params = params.get("model_params", {})

        if not metrics or not model_type:
            return {"status": "error", "message": "Missing metrics or model_type"}

        zptae_threshold = 0.5
        r2_threshold = 0.0
        directional_accuracy_threshold = 0.5
        correlation_threshold = 0.0

        test_zptae = metrics.get("test_zptae", float('inf'))
        test_r2 = metrics.get("test_r2", float('-inf'))
        directional_accuracy = metrics.get("directional_accuracy", 0.0)
        correlation = metrics.get("correlation", float('nan'))
        binom_p_value = metrics.get("binom_p_value", 1.0)

        issues = []
        if np.isnan(test_zptae) or np.isinf(test_zptae):
            issues.append("Invalid ZPTAE (nan or inf)")
        if np.isnan(test_r2) or np.isinf(test_r2):
            issues.append("Invalid R² (nan or inf)")
        if np.isnan(directional_accuracy) or np.isinf(directional_accuracy):
            issues.append("Invalid directional accuracy (nan or inf)")
        if np.isnan(correlation) or np.isinf(correlation):
            issues.append("Invalid correlation (nan or inf)")
        if issues:
            return {
                "status": "needs_update",
                "message": f"Runtime errors detected: {', '.join(issues)}",
                "error": issues[0],
                "suggestions": ["Check for nan or inf in input data", "Add variance checks in model training"]
            }

        suggestions = []
        if test_zptae > zptae_threshold:
            suggestions.append(f"Reduce ZPTAE (current: {test_zptae:.6f}, threshold: {zptae_threshold})")
            suggestions.append("Add features: close_SOLUSDT_lag30, close_BTCUSDT_lag30, close_ETHUSDT_lag30")
            suggestions.append("Increase n_estimators or adjust learning_rate")
        if test_r2 < r2_threshold:
            suggestions.append(f"Improve test R² (current: {test_r2:.6f}, threshold: {r2_threshold})")
            suggestions.append("Add features: macd_SOLUSDT, bb_upper_SOLUSDT")
            suggestions.append("Tune max_depth or num_leaves")
        if directional_accuracy < directional_accuracy_threshold:
            suggestions.append(f"Improve directional accuracy (current: {directional_accuracy:.4f}, threshold: {directional_accuracy_threshold})")
            suggestions.append("Add cross-asset correlation features")
        if abs(correlation) < correlation_threshold or np.isnan(correlation):
            suggestions.append(f"Improve correlation (current: {correlation:.4f}, threshold: {correlation_threshold})")
            suggestions.append("Check for low variance in predictions")
        if binom_p_value > 0.05:
            suggestions.append(f"Improve binomial test p-value (current: {binom_p_value:.4f}, threshold: 0.05)")
            suggestions.append("Enhance feature engineering for directional prediction")

        if suggestions:
            return {
                "status": "needs_update",
                "message": f"Poor performance detected: {', '.join(suggestions)}",
                "suggestions": suggestions
            }
        else:
            return {
                "status": "success",
                "message": f"Performance satisfactory: ZPTAE={test_zptae:.6f}, R²={test_r2:.6f}, Directional Accuracy={directional_accuracy:.4f}, Correlation={correlation:.4f}"
            }
    except Exception as e:
        print(f"[{datetime.now()}] Error in evaluate_results_tool: {str(e)}")
        return {"status": "error", "message": str(e), "error": str(e)}

@app.route("/mcp/tools", methods=["GET"])
def get_tools():
    return Response(json.dumps(TOOLS), status=200, mimetype='application/json')

@app.route("/mcp/optimize", methods=["POST"])
def optimize():
    try:
        model_types = ["LightGBM", "XGBoost", "kNN", "RF"]
        best_model = None
        best_metrics = None
        best_features = []
        best_model_type = None
        best_zptae = float('inf')

        for model_type in model_types:
            model, scaler, metrics, selected_features, _ = train_model(TIMEFRAME, model_type=model_type)
            if model is not None and metrics.get("test_zptae", float('inf')) < best_zptae:
                best_model = model
                best_metrics = metrics
                best_features = selected_features
                best_model_type = model_type
                best_zptae = metrics.get("test_zptae", float('inf'))

        if best_model is None:
            return Response(json.dumps({"status": "error", "message": "No model trained successfully"}), status=500, mimetype='application/json')

        result = {
            "status": "success",
            "result": {
                "metrics": best_metrics,
                "model_type": best_model_type,
                "selected_features": best_features,
                "model_params": {}
            }
        }
        print(f"[{datetime.now()}] Optimization completed: {json.dumps(result, indent=2)}")
        return Response(json.dumps(result), status=200, mimetype='application/json')
    except Exception as e:
        print(f"[{datetime.now()}] Error in optimize: {str(e)}")
        return Response(json.dumps({"status": "error", "message": str(e)}), status=500, mimetype='application/json')

@app.route("/mcp/write_code", methods=["POST"])
def write_code():
    try:
        params = request.get_json() or {}
        result = write_code_tool(params)
        return Response(json.dumps(result), status=200 if result["status"] == "success" else 500, mimetype='application/json')
    except Exception as e:
        print(f"[{datetime.now()}] Error executing write_code: {str(e)}")
        return Response(json.dumps({"status": "error", "message": str(e)}), status=500, mimetype='application/json')

@app.route("/mcp/commit_to_github", methods=["POST"])
def commit_to_github():
    try:
        params = request.get_json() or {}
        result = commit_to_github_tool(params)
        return Response(json.dumps(result), status=200 if result["status"] == "success" else 500, mimetype='application/json')
    except Exception as e:
        print(f"[{datetime.now()}] Error executing commit_to_github: {str(e)}")
        return Response(json.dumps({"status": "error", "message": str(e)}), status=500, mimetype='application/json')

@app.route("/mcp/update_artifacts", methods=["POST"])
def update_artifacts():
    try:
        params = request.get_json() or {}
        result = update_artifacts_tool(params)
        return Response(json.dumps(result), status=200 if result["status"] == "success" else 500, mimetype='application/json')
    except Exception as e:
        print(f"[{datetime.now()}] Error executing update_artifacts: {str(e)}")
        return Response(json.dumps({"status": "error", "message": str(e)}), status=500, mimetype='application/json')

@app.route("/mcp/evaluate_results", methods=["POST"])
def evaluate_results():
    try:
        params = request.get_json() or {}
        result = evaluate_results_tool(params)
        return Response(json.dumps(result), status=200 if result["status"] in ["success", "needs_update"] else 500, mimetype='application/json')
    except Exception as e:
        print(f"[{datetime.now()}] Error executing evaluate_results: {str(e)}")
        return Response(json.dumps({"status": "error", "message": str(e)}), status=500, mimetype='application/json')

if __name__ == "__main__":
    print(f"[{datetime.now()}] Starting MCP app server on port {FLASK_PORT}...")
    app.run(host="0.0.0.0", port=FLASK_PORT)
