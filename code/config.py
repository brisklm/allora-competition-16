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
TRAINING_DAYS = int(os.getenv('TRAINING_DAYS', 365))
MINIMUM_DAYS = 180
REGION = os.getenv('REGION', 'com')
DATA_PROVIDER = os.getenv('DATA_PROVIDER', 'binance')
MODEL = os.getenv('MODEL', 'LSTM_Hybrid')
CG_API_KEY = os.getenv('CG_API_KEY', 'CG-xA5NyokGEVbc4bwrvJPcpZvT')
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY', '70ed65ce-4750-4fd5-83bd-5aee9aa79ead')
HELIUS_RPC_URL = os.getenv('HELIUS_RPC_URL', 'https://mainnet.helius-rpc.com')
BITQUERY_API_KEY = os.getenv('BITQUERY_API_KEY', 'ory_at_LmFLzUutMY8EVb-P_PQVP9ntfwUVTV05LMal7xUqb2I.vxFLfMEoLGcu4XoVi47j-E2bspraTSrmYzCt1A4y2k')
SELECTED_FEATURES = ['volatility_SOLUSDT', 'sol_btc_corr', 'sol_eth_corr', 'close_SOLUSDT_lag1', 'close_BTCUSDT_lag1', 'close_ETHUSDT_lag1', 'volume_change_SOLUSDT', 'volatility_BTCUSDT', 'volume_change_BTCUSDT', 'momentum_SOLUSDT', 'close_SOLUSDT_lag30', 'close_BTCUSDT_lag30', 'close_ETHUSDT_lag30', 'sol_btc_ratio', 'sol_eth_ratio', 'vader_sentiment', 'log_return_lag1_SOL', 'sign_lag1_SOL', 'momentum_filter']
MODEL_PARAMS = {'n_estimators': 1000, 'learning_rate': 0.01, 'hidden_size': 128, 'num_layers': 3, 'max_depth': 8, 'num_leaves': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'dropout': 0.2}
OPTUNA_TRIALS = int(os.getenv('OPTUNA_TRIALS', 100))
USE_SYNTHETIC_DATA = 'blend'
SYNTHETIC_RATIO = 0.5
IMPUTE_METHOD = 'mean'
FEATURE_VARIANCE_THRESHOLD = 0.01