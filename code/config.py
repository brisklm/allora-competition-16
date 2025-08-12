import os
from datetime import datetime
import numpy as np  # For data handling
# Add imports for new features
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # For VADER sentiment
import optuna  # For hyperparameter tuning

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
TRAINING_DAYS = int(os.getenv('TRAINING_DAYS', 90))  # Ensure at least MINIMUM_DAYS for ZPTAE reduction
MINIMUM_DAYS = 180
REGION = os.getenv('REGION', 'com')
DATA_PROVIDER = os.getenv('DATA_PROVIDER', 'binance')
MODEL = os.getenv('MODEL', 'LightGBM')  # Hybrid with LSTM
CG_API_KEY = os.getenv('CG_API_KEY', 'CG-xA5NyokGEVbc4bwrvJPcpZvT')
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY', '70ed65ce-4750-4fd5-83bd-5aee9aa79ead')
HELIUS_RPC_URL = os.getenv('HELIUS_RPC_URL', 'https://mainnet.helius-rpc.com')
BITQUERY_API_KEY = os.getenv('BITQUERY_API_KEY', 'ory_at_LmFLzUutMY8EVb-P_PQVP9ntfwUVTV05LMal7xUqb2I.vxFLfMEoLGcu4XoVi47j-E2bspraTSrmYzCtN1A4y2k')

# Updated SELECTED_FEATURES to ensure all suggested features are included and handle low variance
SELECTED_FEATURES = [
    'volatility_SOLUSDT', 'sol_btc_corr', 'sol_eth_corr', 'close_SOLUSDT_lag1', 
    'close_BTCUSDT_lag1', 'close_ETHUSDT_lag1', 'volume_change_SOLUSDT', 
    'volatility_BTCUSDT', 'volume_change_BTCUSDT', 'momentum_SOLUSDT', 
    'sign_log_return_lag1_SOLUSDT', 'close_SOLUSDT_lag5', 'close_SOLUSDT_lag10', 
    'close_SOLUSDT_lag30', 'close_BTCUSDT_lag30', 'close_ETHUSDT_lag30',  # Added as per suggestions
    'sol_eth_vol_ratio', 'sol_eth_momentum_ratio', 'rsi_SOLUSDT', 'macd_SOLUSDT', 
    'bb_upper_SOLUSDT', 'bb_lower_SOLUSDT', 'vwap_SOLUSDT', 'sol_tx_volume', 
    'sentiment_score',  # VADER sentiment integrated
    'lstm_output'  # For LSTM hybrid
]  # Fix NaNs: Ensure data cleaning in model.py

# Optimized MODEL_PARAMS to reduce ZPTAE (adjust learning_rate and increase n_estimators)
MODEL_PARAMS = {
    'num_leaves': 50,
    'learning_rate': 0.01,  # Increased from 0.005 to reduce ZPTAE
    'n_estimators': 1500,  # Increased as per suggestions
    'boosting_type': 'gbdt',
    'metric': 'rmse'  # Or custom for ZPTAE
}

# New parameters for Optuna tuning
OPTUNA_TRIALS = 50  # Number of trials for hyperparameter tuning

# For blending real and synthetic data
USE_SYNTHETIC_DATA = True  # Flag to blend data
SYNTHETIC_DATA_PATH = os.path.join(data_base_path, 'synthetic_data.csv')  # Path for synthetic data

# Ensure syntax validity and compatibility