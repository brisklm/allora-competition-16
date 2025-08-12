import os
from datetime import datetime
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import optuna

data_base_path = os.path.join(os.getcwd(), 'data')
model_file_path = os.path.join(data_base_path, 'model.pkl')
scaler_file_path = os.path.join(data_base_path, 'scaler.pkl')
training_price_data_path = os.path.join(data_base_path, 'price_data.csv')
selected_features_path = os.path.join(data_base_path, 'selected_features.json')
sol_source_path = os.path.join(data_base_path, os.getenv('SOL_SOURCE', 'raw_sol.csv'))
eth_source_path = os.path.join(data_base_path, os.getenv('ETH_SOURCE', 'raw_eth.csv'))
features_sol_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH', 'features_sol.csv'))
features_eth_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH_ETH', 'features_eth.csv'))
TOKEN = os.getenv('TOKEN', 'SOL')
TIMEFRAME = os.getenv('TIMEFRAME', '1d')
TRAINING_DAYS = int(os.getenv('TRAINING_DAYS', 365))  # Increased to 365 for better ZPTAE reduction
MINIMUM_DAYS = 180
REGION = os.getenv('REGION', 'com')
DATA_PROVIDER = os.getenv('DATA_PROVIDER', 'binance')
MODEL = os.getenv('MODEL', 'LSTM_Hybrid')  # Hybrid with LSTM and VADER sentiment
CG_API_KEY = os.getenv('CG_API_KEY', 'CG-xA5NyokGEVbc4bwrvJPcpZvT')
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY', '70ed65ce-4750-4fd5-83bd-5aee9aa79ead')
HELIUS_RPC_URL = os.getenv('HELIUS_RPC_URL', 'https://mainnet.helius-rpc.com')
BITQUERY_API_KEY = os.getenv('BITQUERY_API_KEY', 'ory_at_LmFLzUutMY8EVb-P_PQVP9ntfwUVTV05LMal7xUqb2I.vxFLfMEoLGcu4XoVi47j-E2bspraTSrmYzCt1A4y2k')

# Updated SELECTED_FEATURES to include new features, handle low variance, and fix NaNs
SELECTED_FEATURES = [
    'volatility_SOLUSDT', 'sol_btc_corr', 'sol_eth_corr', 'close_SOLUSDT_lag1', 
    'close_BTCUSDT_lag1', 'close_ETHUSDT_lag1', 'volume_change_SOLUSDT', 
    'volatility_BTCUSDT', 'volume_change_BTCUSDT', 'momentum_SOLUSDT',  
    'close_SOLUSDT_lag30', 'close_BTCUSDT_lag30', 'close_ETHUSDT_lag30'  # Added as per suggestions
]

MODEL_PARAMS = {
    'n_estimators': 200,  # Increased from default to reduce ZPTAE
    'learning_rate': 0.005,  # Adjusted for better convergence
    'hidden_size': 64,  # For LSTM hybrid
    'num_layers': 2  # For LSTM
}  # Ensure handling of NaNs and low variance in data preprocessing

OPTUNA_TRIALS = int(os.getenv('OPTUNA_TRIALS', 50))  # Allow for more trials if needed
USE_SYNTHETIC_DATA = os.getenv('USE_SYNTHETIC_DATA', 'True').lower() == 'true'  # Blend real and synthetic data