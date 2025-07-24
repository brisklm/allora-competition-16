import requests
import json
from datetime import datetime
import time
from dotenv import load_dotenv
import os
import uuid
from github import Github
import shutil
import ast
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()
MCP_SERVER_URL = "http://localhost:8001"  # Points to app.py
MCP_EVALUATOR_URL = "http://localhost:8004"
MCP_CODE_WRITER_URL = "http://localhost:8003"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME", "brisklm/allora-competition-16")
BACKUP_DIR = os.path.join(os.getcwd(), "backup")
UPDATE_INTERVAL = os.getenv("UPDATE_INTERVAL", "30m")
UPDATE_INTERVAL_MINUTES = int(UPDATE_INTERVAL.replace("m", ""))

def correct_syntax_error(content, error_message):
    try:
        lines = content.split("\n")
        if "unexpected EOF" in error_message:
            for i, line in enumerate(lines):
                if line.strip().startswith("def ") and ":" not in line:
                    lines[i] = line + ":"
                elif line.strip().startswith("print(") and not line.strip().endswith(")"):
                    lines[i] = line + ")"
        elif "invalid syntax" in error_message or "comprehension target" in error_message:
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
                # Fix comprehension errors
                if "for" in stripped and " in " in stripped and not stripped.endswith(")"):
                    if stripped.startswith("[") or stripped.startswith("("):
                        corrected_lines[-1] = corrected_lines[-1] + ")"
            lines = corrected_lines
        return "\n".join(lines)
    except Exception as e:
        print(f"[{datetime.now()}] Error correcting syntax: {str(e)}")
        return content

def revert_to_previous_version():
    try:
        print(f"[{datetime.now()}] Reverting to previous working version...")
        files = ["model.py", "update_app.py", "mcp_client.py", "app.py", "config.py", "updater.py"]
        os.makedirs(BACKUP_DIR, exist_ok=True)
        for file in files:
            backup_path = os.path.join(BACKUP_DIR, file)
            if os.path.exists(backup_path):
                shutil.copy(backup_path, file)
                print(f"[{datetime.now()}] Restored {file} from backup")
            else:
                try:
                    g = Github(GITHUB_TOKEN)
                    repo = g.get_repo(REPO_NAME)
                    contents = repo.get_contents(f"code/{file}", ref="main")
                    with open(file, "wb") as f:
                        f.write(contents.decoded_content)
                    print(f"[{datetime.now()}] Restored {file} from GitHub")
                except Exception as e:
                    print(f"[{datetime.now()}] No local backup or GitHub copy for {file}: {str(e)}")
    except Exception as e:
        print(f"[{datetime.now()}] Error reverting to previous version: {str(e)}")

def generate_updated_code(suggestions, error=None):
    try:
        source_files = {
            "model.py": "596cdbbf-fa6e-4024-a137-1c03b77e504a",
            "update_app.py": "4258058b-4714-4feb-bbfa-1128f3ba2037",
            "mcp_client.py": "10a92ca5-d97f-4c45-bf97-c166b45ea7c7",
            "app.py": "f6e7b2a4-3b1e-4f1c-9a6e-7b4e9f2b3d1c",
            "config.py": "d067c435-aa2b-4bd1-a08f-48442cf22419",
            "updater.py": "489495e3-a721-4e57-8ac0-fb727b101e84"
        }
        artifacts = []
        
        for file_name, artifact_id in source_files.items():
            try:
                with open(file_name, "r") as f:
                    content = f.read()
                updated_content = content
                if error and ("SyntaxError" in error or "NameError" in error):
                    updated_content = correct_syntax_error(content, error)
                if file_name == "model.py" and ("Add features" in str(suggestions) or "Enhance feature engineering" in str(suggestions)):
                    lines = content.split("\n")
                    updated_lines = []
                    in_feature_section = False
                    features_added = False
                    new_features = ["close_SOLUSDT_lag30", "close_BTCUSDT_lag30", "close_ETHUSDT_lag30", "sol_eth_vol_ratio", "sol_eth_momentum_ratio"]
                    indent_level = None
                    for line in lines:
                        if line.strip().startswith("all_features = ["):
                            in_feature_section = True
                            indent_level = len(line) - len(line.lstrip())
                            updated_lines.append(line)
                            for feature in new_features:
                                if f'"{feature}"' not in content:
                                    updated_lines.append(f'{" " * (indent_level + 4)}"{feature}",')
                                    features_added = True
                        elif in_feature_section and line.strip() == "]":
                            in_feature_section = False
                            updated_lines.append(line)
                        else:
                            updated_lines.append(line)
                    if features_added:
                        updated_content = "\n".join(updated_lines)
                        # Add sol_eth_vol_ratio and sol_eth_momentum_ratio calculations in format_data
                        format_data_index = updated_content.find("def format_data(")
                        if format_data_index != -1:
                            format_data_end = updated_content.find("def ", format_data_index + 1)
                            if format_data_end == -1:
                                format_data_end = len(updated_content)
                            format_data_content = updated_content[format_data_index:format_data_end]
                            insert_point = format_data_content.rfind('price_df["sol_btc_volume_ratio"]')
                            if insert_point != -1:
                                insert_point += format_data_content[insert_point:].find("\n") + 1
                                indent = " " * 8  # 8 spaces for proper indentation
                                format_data_content = (
                                    format_data_content[:insert_point] +
                                    f'{indent}price_df["sol_eth_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_ETHUSDT"] + 1e-10) if "volatility_ETHUSDT" in price_df.columns else pd.Series(1, index=price_df.index)\n' +
                                    f'{indent}price_df["sol_eth_momentum_ratio"] = price_df["momentum_SOLUSDT"] / (price_df["momentum_ETHUSDT"] + 1e-10) if "momentum_ETHUSDT" in price_df.columns else pd.Series(1, index=price_df.index)\n' +
                                    format_data_content[insert_point:]
                                )
                                updated_content = updated_content[:format_data_index] + format_data_content + updated_content[format_data_end:]
                    # Validate syntax
                    try:
                        ast.parse(updated_content)
                    except SyntaxError as e:
                        print(f"[{datetime.now()}] Syntax error in updated model.py: {str(e)}")
                        updated_content = correct_syntax_error(updated_content, str(e))
                        try:
                            ast.parse(updated_content)
                        except SyntaxError as e:
                            print(f"[{datetime.now()}] Failed to correct syntax in model.py: {str(e)}")
                            return []
                artifacts.append({
                    "artifact_id": artifact_id,
                    "title": file_name,
                    "contentType": "text/python",
                    "content": updated_content
                })
            except Exception as e:
                print(f"[{datetime.now()}] Error reading {file_name}: {str(e)}")
                return []
        return artifacts
    except Exception as e:
        print(f"[{datetime.now()}] Error generating updated code: {str(e)}")
        return []

def connect_to_mcp_server(max_retries=3, retry_delay=5):
    print(f"[{datetime.now()}] Starting cyclical optimization process...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{MCP_SERVER_URL}/mcp/tools", timeout=10)
            response.raise_for_status()
            tools = response.json()
            print(f"[{datetime.now()}] MCP server tools: {json.dumps(tools, indent=2)}")
            break
        except Exception as e:
            print(f"[{datetime.now()}] Error discovering MCP server tools (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                print(f"[{datetime.now()}] Failed to discover MCP server tools")
                return
    
    for attempt in range(max_retries):
        try:
            print(f"[{datetime.now()}] Triggering model optimization...")
            response = requests.post(f"{MCP_SERVER_URL}/mcp/optimize", json={}, timeout=300)
            response.raise_for_status()
            result = response.json()
            print(f"[{datetime.now()}] Optimization result: {json.dumps(result, indent=2)}")
            
            if result["status"] != "success":
                print(f"[{datetime.now()}] Optimization failed: {result.get('message', 'Unknown error')}")
                revert_to_previous_version()
                return
            
            evaluation_result = evaluate_results(result["result"])
            if evaluation_result["status"] == "needs_update":
                print(f"[{datetime.now()}] Performance issues detected: {evaluation_result['message']}")
                update_source_codes(evaluation_result["suggestions"], evaluation_result.get("error"))
            else:
                print(f"[{datetime.now()}] Performance satisfactory: {evaluation_result['message']}")
            break
        except Exception as e:
            print(f"[{datetime.now()}] Error in optimization (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                print(f"[{datetime.now()}] Failed to optimize after {max_retries} attempts")
                revert_to_previous_version()
                return
    
    try:
        with open(".env", "r") as f:
            env = dict(line.strip().split("=", 1) for line in f if "=" in line)
        print(f"[{datetime.now()}] Best model: {env.get('MODEL', 'N/A')}")
        print(f"[{datetime.now()}] Selected features: {env.get('SELECTED_FEATURES', '[]')}")
        print(f"[{datetime.now()}] Model parameters: {env.get('MODEL_PARAMS', '{}')}")
        print(f"[{datetime.now()}] GitHub commit: {result.get('commit', {}).get('message', 'N/A')}")
    except Exception as e:
        print(f"[{datetime.now()}] Error reading .env: {str(e)}")

def evaluate_results(result):
    try:
        response = requests.post(f"{MCP_EVALUATOR_URL}/mcp/execute/evaluate_results", json=result, timeout=10)
        response.raise_for_status()
        evaluation_result = response.json()
        print(f"[{datetime.now()}] Evaluation result: {json.dumps(evaluation_result, indent=2)}")
        return evaluation_result
    except Exception as e:
        print(f"[{datetime.now()}] Error evaluating results: {str(e)}")
        return {"status": "error", "message": str(e), "error": str(e)}

def update_source_codes(suggestions, error=None):
    try:
        artifacts = generate_updated_code(suggestions, error)
        if not artifacts:
            print(f"[{datetime.now()}] Failed to generate updated code")
            revert_to_previous_version()
            return
        
        response = requests.post(f"{MCP_CODE_WRITER_URL}/mcp/update_artifacts", json={"artifacts": artifacts}, timeout=30)
        response.raise_for_status()
        update_result = response.json()
        print(f"[{datetime.now()}] Source code update result: {json.dumps(update_result, indent=2)}")
        if update_result["status"] != "success":
            revert_to_previous_version()
    except Exception as e:
        print(f"[{datetime.now()}] Error updating source codes: {str(e)}")
        revert_to_previous_version()

if __name__ == "__main__":
    print(f"[{datetime.now()}] Running immediate optimization...")
    connect_to_mcp_server()
    scheduler = BackgroundScheduler()
    scheduler.add_job(connect_to_mcp_server, 'interval', minutes=UPDATE_INTERVAL_MINUTES)
    scheduler.start()
    print(f"[{datetime.now()}] Started scheduler with interval {UPDATE_INTERVAL_MINUTES} minutes")
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print(f"[{datetime.now()}] Scheduler stopped")
