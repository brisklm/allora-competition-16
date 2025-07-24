# Configuration for Allora Competition 16, Topic 62
import os
from datetime import datetime

# File paths
data_base_path = os.path.join(os.getcwd(), "data")
model_file_path = os.path.join(data_base_path, "model.pkl")
scaler_file_path = os.path.join(data_base_path, "scaler.pkl")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")
sol_source_path = os.path.join(data_base_path, os.getenv("SOL_SOURCE", "raw_sol.csv"))
eth_source_path = os.path.join(data_base_path, os.getenv("ETH_SOURCE", "raw_eth.csv"))
features_sol_path = os.path.join(data_base_path, os.getenv("FEATURES_PATH", "features_sol.csv"))
features_eth_path = os.path.join(data_base_path, os.getenv("FEATURES_PATH_ETH", "features_eth.csv"))

# Model and data settings
TOKEN = os.getenv("TOKEN", "SOL")
TIMEFRAME = os.getenv("TIMEFRAME", "1d")
TRAINING_DAYS = int(os.getenv("TRAINING_DAYS", 90))
MINIMUM_DAYS = 180
REGION = os.getenv("REGION", "com")
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "binance")
MODEL = os.getenv("MODEL", "LightGBM")
CG_API_KEY = os.getenv("CG_API_KEY", "CG-xA5NyokGEVbc4bwrvJPcpZvT")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "70ed65ce-4750-4fd5-83bd-5aee9aa79ead")

print(f"[{datetime.now()}] Loaded config.py at {os.path.abspath(__file__)}")
