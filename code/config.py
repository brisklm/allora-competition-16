import os
from datetime import datetime

data_base_path = os.path.join(os.getcwd(), "data")
model_file_path = os.path.join(data_base_path, "model.pkl")
scaler_file_path = os.path.join(data_base_path, "scaler.pkl")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")
sol_source_path = os.path.join(data_base_path, os.getenv("SOL_SOURCE", "raw_sol.csv"))
eth_source_path = os.path.join(data_base_path, os.getenv("ETH_SOURCE", "raw_eth.csv"))
features_sol_path = os.path.join(data_base_path, os.getenv("FEATURES_PATH", "features_sol.csv"))
features_eth_path = os.path.join(data_base_path, os.getenv("FEATURES_PATH_ETH", "features_eth.csv"))

TOKEN = os.getenv("TOKEN", "SOL")
TIMEFRAME = os.getenv("TIMEFRAME", "1d")
TRAINING_DAYS = int(os.getenv("TRAINING_DAYS", 180))
MINIMUM_DAYS = 180
REGION = os.getenv("REGION", "com")
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "binance")
MODEL = os.getenv("MODEL", "LightGBM")
CG_API_KEY = os.getenv("CG_API_KEY", "CG-xA5NyokGEVbc4bwrvJPcpZvT")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "70ed65ce-4750-4fd5-83bd-5aee9aa79ead")
BITQUERY_API_KEY = os.getenv("BITQUERY_API_KEY", "ory_at_LmFLzUutMY8EVb-P_PQVP9ntfwUVTV05LMal7xUqb2I.vxFLfMEoLGcu4XoVi47j-E2bspraTSrmYzCtN1A4y2k")

SELECTED_FEATURES = [
    'rsi_SOLUSDT', 'volatility_SOLUSDT', 'macd_SOLUSDT', 'sol_btc_corr', 'sol_eth_corr',
    'close_SOLUSDT_lag1', 'close_BTCUSDT_lag1', 'close_ETHUSDT_lag1', 'bb_upper_SOLUSDT',
    'bb_lower_SOLUSDT', 'volume_change_SOLUSDT', 'rsi_BTCUSDT', 'volatility_BTCUSDT',
    'macd_BTCUSDT', 'volume_change_BTCUSDT', 'close_SOLUSDT_lag30', 'close_BTCUSDT_lag30',
    'close_ETHUSDT_lag30', 'garch_vol_SOLUSDT'
]

MODEL_PARAMS = {
    "num_leaves": 50,
    "learning_rate": 0.005,
    "n_estimators": 2000,
    "max_depth": 6,
    "min_child_samples": 15,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "random_state": 42
}

VADER_ENABLED = True

print(f"[{datetime.now()}] Loaded config.py at {os.path.abspath(__file__)}")
