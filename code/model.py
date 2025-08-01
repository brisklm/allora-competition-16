import json
import os
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, binomtest
from datetime import datetime, timedelta
import time
import glob
import logging
try:
    import optuna
except ImportError:
    optuna = None
try:
    import torch
    from torch import nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None
try:
    from arch import arch_model
except ImportError:
    arch_model = None
try:
    from updater import download_binance_daily_data, download_binance_current_day_data
except ImportError as e:
    logging.error(f"ImportError: {str(e)}. Ensure updater.py is available.")
    raise
try:
    from config import data_base_path, model_file_path, scaler_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, MINIMUM_DAYS, REGION, DATA_PROVIDER, MODEL, sol_source_path, eth_source_path, features_sol_path, features_eth_path, HELIUS_API_KEY, BITQUERY_API_KEY, SELECTED_FEATURES, MODEL_PARAMS
except ImportError as e:
    logging.error(f"ImportError: {str(e)}. Using fallback configuration.")
    data_base_path = os.path.join(os.getcwd(), "data")
    model_file_path = os.path.join(data_base_path, "model.pkl")
    scaler_file_path = os.path.join(data_base_path, "scaler.pkl")
    TOKEN = "SOL"
    TIMEFRAME = "1d"
    TRAINING_DAYS = 180
    MINIMUM_DAYS = 180
    REGION = "com"
    DATA_PROVIDER = "binance"
    MODEL = "LightGBM"
    sol_source_path = os.path.join(data_base_path, "raw_sol.csv")
    eth_source_path = os.path.join(data_base_path, "raw_eth.csv")
    features_sol_path = os.path.join(data_base_path, "features_sol.csv")
    features_eth_path = os.path.join(data_base_path, "features_eth.csv")
    HELIUS_API_KEY = ""
    BITQUERY_API_KEY = ""
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

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")
MODEL_VERSION = "2025-07-31-competition16-topic62-multimodel-v80"
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
logging.info(f"Loaded model.py version {MODEL_VERSION}")

def calculate_rsi(data, periods=14):
    try:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logging.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_ma(data, periods=20):
    try:
        return data.rolling(window=periods).mean()
    except Exception as e:
        logging.error(f"Error calculating MA: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_macd(data, fast=12, slow=26, signal=9):
    try:
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    except Exception as e:
        logging.error(f"Error calculating MACD: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_bollinger_bands(data, periods=20, std_dev=2):
    try:
        sma = data.rolling(periods).mean()
        std = data.rolling(periods).std()
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        return upper.fillna(0), lower.fillna(0)
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands: {str(e)}")
        return pd.Series(0, index=data.index), pd.Series(0, index=data.index)

def calculate_volume_change(data):
    try:
        return data.pct_change().fillna(0)
    except Exception as e:
        logging.error(f"Error calculating volume change: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_vwap(data, price_col, volume_col):
    try:
        price = data[price_col]
        volume = data[volume_col].clip(lower=1e-10)
        return (price * volume).cumsum() / volume.cumsum()
    except Exception as e:
        logging.error(f"Error calculating VWAP: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_volatility(data, window=5):
    try:
        return data.pct_change().rolling(window=window).std().fillna(0)
    except Exception as e:
        logging.error(f"Error calculating volatility: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_garch_vol(data):
    try:
        if arch_model:
            model = arch_model(data.pct_change().dropna(), vol='Garch', p=1, q=1)
            res = model.fit(disp='off')
            return res.conditional_volatility.fillna(0)
        return pd.Series(0, index=data.index)
    except Exception as e:
        logging.error(f"Error calculating GARCH volatility: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_cross_asset_correlation(data, pair1, pair2, window=10):
    try:
        if pair1 not in data.columns or pair2 not in data.columns:
            logging.warning(f"Missing columns for correlation: {pair1}, {pair2}")
            return pd.Series(0, index=data.index)
        corr = data[pair1].pct_change().rolling(window=window).corr(data[pair2].pct_change())
        return corr.fillna(0)
    except Exception as e:
        logging.error(f"Error calculating cross-asset correlation: {str(e)}")
        return pd.Series(0, index=data.index)

def fetch_x_sentiment(keyword="solana price prediction", days=7):
    try:
        if SentimentIntensityAnalyzer:
            analyzer = SentimentIntensityAnalyzer()
            x_api_key = os.getenv("X_API_KEY")
            if x_api_key:
                response = requests.get(
                    f"https://api.x.com/2/tweets/search/recent?query={keyword}",
                    headers={"Authorization": f"Bearer {x_api_key}"},
                    timeout=10
                )
                response.raise_for_status()
                posts = [tweet['text'] for tweet in response.json()['data']]
                scores = [analyzer.polarity_scores(p)['compound'] for p in posts]
                sentiment_scores = [np.mean(scores) for _ in range(days)]
                logging.info(f"VADER sentiment score std: {np.std(sentiment_scores):.2f}")
                return {'sentiment_score': sentiment_scores}
            else:
                logging.warning("X_API_KEY missing, using mock sentiment")
        np.random.seed(42)
        sentiment_scores = [np.random.normal(0.5, 0.3) for _ in range(days)]
        logging.info(f"Mock sentiment score std: {np.std(sentiment_scores):.2f}")
        return {'sentiment_score': sentiment_scores}
    except Exception as e:
        logging.error(f"Error fetching X sentiment: {str(e)}")
        np.random.seed(42)
        sentiment_scores = [np.random.normal(0.5, 0.3) for _ in range(days)]
        return {'sentiment_score': sentiment_scores}

def fetch_solana_onchain_data(days=30):
    try:
        if HELIUS_API_KEY:
            url = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getRecentBlockhash",
                "params": []
            }
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            blockhash = data.get("result", {}).get("value", {}).get("blockhash", "")
            if not blockhash:
                raise ValueError("No blockhash returned")
            # Fetch recent transactions for volume estimation
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": ["11111111111111111111111111111111", {"limit": days}]
            }
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            signatures = data.get("result", [])
            tx_volumes = [len(signatures) for _ in range(min(days, len(signatures)))]
            logging.info(f"Fetched Solana tx_volumes std: {np.std(tx_volumes):.2f}")
            return {'tx_volume': tx_volumes}
        else:
            logging.warning("HELIUS_API_KEY missing, using mock data")
            np.random.seed(42)
            base_tx = 200000
            tx_volumes = [max(1000, int(base_tx + (i * 500) + np.random.normal(0, 20000))) for i in range(days)]
            logging.info(f"Mock Solana tx_volumes std: {np.std(tx_volumes):.2f}")
            return {'tx_volume': tx_volumes}
    except Exception as e:
        logging.error(f"Error fetching Solana on-chain data: {str(e)}")
        np.random.seed(42)
        base_tx = 200000
        tx_volumes = [max(1000, int(base_tx + (i * 500) + np.random.normal(0, 20000))) for i in range(days)]
        return {'tx_volume': tx_volumes}

def generate_synthetic_data(symbols, days=MINIMUM_DAYS):
    try:
        os.makedirs(coingecko_data_path, exist_ok=True)
        end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
        start_date = end_date - timedelta(days=days - 1)
        data = {symbol: [] for symbol in symbols}
        np.random.seed(42)
        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            for symbol in symbols:
                base_price = 185 if "SOL" in symbol else (95000 if "BTC" in symbol else 2800)
                price_noise = np.random.normal(0, 0.05)
                open_price = base_price * (1 + price_noise)
                high = open_price * (1 + np.random.uniform(0.01, 0.03))
                low = open_price * (1 - np.random.uniform(0.01, 0.03))
                close = open_price * (1 + np.random.normal(0, 0.02))
                volume = 1000 * (1 + np.random.uniform(0, 0.7))
                data[symbol].append({
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                })
        files = []
        for symbol in symbols:
            df = pd.DataFrame(data[symbol])
            file_path = os.path.join(coingecko_data_path, f"{symbol}_{end_date.strftime('%Y-%m-%d')}_synthetic.csv")
            df.to_csv(file_path, index=False)
            files.append(file_path)
        logging.info(f"Generated {len(files)} synthetic files for {symbols} with {days} rows")
        return files
    except Exception as e:
        logging.error(f"Error generating synthetic data: {str(e)}")
        return []

def download_data(token, training_days, region, data_provider=DATA_PROVIDER):
    try:
        logging.info(f"Checking cached data for {token} at {training_price_data_path}")
        if os.path.exists(training_price_data_path):
            cache_time = os.path.getmtime(training_price_data_path)
            if (time.time() - cache_time) > 3600:
                logging.info("Cache older than 1 hour, refreshing data")
                os.remove(training_price_data_path)
            else:
                try:
                    df = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
                    if not df.empty and 'target_SOLUSDT' in df.columns and not df['target_SOLUSDT'].isna().all():
                        if validate_data(df):
                            logging.info(f"Found valid cached data with {len(df)} rows")
                            return []
                        else:
                            logging.warning("Cached data failed validation")
                except Exception as e:
                    logging.error(f"Error loading cached data: {str(e)}")
        
        if data_provider == "binance":
            save_path = binance_data_path
            os.makedirs(save_path, exist_ok=True)
            if not os.access(save_path, os.W_OK):
                logging.error(f"No write permission for {save_path}")
                raise PermissionError(f"No write permission for {save_path}")
            if token == "SOL" and os.path.exists(sol_source_path):
                logging.info(f"Using SOL data from {sol_source_path}")
                return [sol_source_path]
            if token == "ETH" and os.path.exists(eth_source_path):
                logging.info(f"Using ETH data from {eth_source_path}")
                return [eth_source_path]
            end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
            start_date = end_date - timedelta(days=training_days)
            files = download_binance_daily_data(f"{token}USDT", training_days + 1, region, save_path)
            valid_files = [f for f in files if os.path.exists(f) and os.path.getsize(f) > 100]
            if not valid_files:
                logging.warning(f"No valid data files for {token}, generating synthetic data")
                return generate_synthetic_data([f"{token}USDT"], days=MINIMUM_DAYS)
            logging.info(f"Found {len(valid_files)} valid files for {token}")
            return valid_files
        else:
            save_path = coingecko_data_path
            os.makedirs(save_path, exist_ok=True)
            files = [f for f in glob.glob(os.path.join(save_path, f"{token}_*.csv")) if os.path.getsize(f) > 100]
            if files:
                total_rows = sum(len(pd.read_csv(f)) for f in files)
                if total_rows >= MINIMUM_DAYS:
                    logging.info(f"Found {len(files)} existing files for {token} with {total_rows} rows")
                    return files
                else:
                    logging.warning(f"Insufficient rows ({total_rows} < {MINIMUM_DAYS})")
            return generate_synthetic_data([f"{token}USDT"], days=MINIMUM_DAYS)
    except Exception as e:
        logging.error(f"Error downloading data for {token}: {str(e)}")
        return generate_synthetic_data([f"{token}USDT"], days=MINIMUM_DAYS)

def calculate_technical_indicators(data, pair):
    try:
        close = data[f"close_{pair}"]
        volume = data[f"volume_{pair}"]
        high = data[f"high_{pair}"]
        low = data[f"low_{pair}"]
        data[f"rsi_{pair}"] = calculate_rsi(close, periods=14).fillna(0)
        data[f"ma5_{pair}"] = calculate_ma(close, periods=5).fillna(0)
        data[f"ma20_{pair}"] = calculate_ma(close, periods=20).fillna(0)
        data[f"macd_{pair}"] = calculate_macd(close, fast=12, slow=26, signal=9).fillna(0)
        upper, lower = calculate_bollinger_bands(close, periods=20, std_dev=2)
        data[f"bb_upper_{pair}"] = upper
        data[f"bb_lower_{pair}"] = lower
        data[f"volume_change_{pair}"] = calculate_volume_change(volume)
        data[f"volatility_{pair}"] = calculate_volatility(close, window=5).fillna(0)
        data[f"momentum_{pair}"] = (close - close.shift(5)).fillna(0)
        data[f"sign_log_return_lag1_{pair}"] = np.sign(np.log(close / close.shift(1)).shift(1)).fillna(0)
        data[f"vwap_{pair}"] = calculate_vwap(data, f"close_{pair}", f"volume_{pair}")
        data[f"garch_vol_{pair}"] = calculate_garch_vol(close).fillna(0)
        for lag in [1, 2, 5, 10, 30]:
            data[f"close_{pair}_lag{lag}"] = close.shift(lag)
            data[f"log_return_{pair}_lag{lag}"] = data[f"log_return_{pair}"].shift(lag)
        return data
    except Exception as e:
        logging.error(f"Error calculating technical indicators for {pair}: {str(e)}")
        return data

def validate_data(df):
    try:
        required_columns = ['close_SOLUSDT', 'volume_SOLUSDT', 'close_BTCUSDT', 'close_ETHUSDT']
        feature_columns = SELECTED_FEATURES
        all_columns = required_columns + [col for col in feature_columns if col in df.columns]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns in data: {missing_cols}")
            return False
        for col in all_columns:
            if col in df.columns and df[col].isna().sum() > len(df) * 0.05:
                logging.error(f"Too many NaNs in {col}: {df[col].isna().sum()}/{len(df)}")
                return False
            if col in ['sol_tx_volume', 'sentiment_score'] and col in df.columns and df[col].std() < 1e-3:
                logging.error(f"Low variance in {col}: std={df[col].std()}")
                return False
        logging.info(f"Data validation passed: {len(df)} rows, columns: {list(df.columns)}")
        return True
    except Exception as e:
        logging.error(f"Error in data validation: {str(e)}")
        return False

def format_data(files_btc, files_sol, files_eth, data_provider):
    try:
        logging.info(f"Using TIMEFRAME={TIMEFRAME}, TRAINING_DAYS={TRAINING_DAYS}, Model Version={MODEL_VERSION}")
        logging.info(f"Input files: BTC={len(files_btc)}, SOL={len(files_sol)}, ETH={len(files_eth)}")
        
        if os.path.exists(training_price_data_path):
            logging.info(f"Checking cached data at {training_price_data_path}")
            try:
                df = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
                if not df.empty and 'target_SOLUSDT' in df.columns and not df['target_SOLUSDT'].isna().all():
                    if validate_data(df):
                        logging.info(f"Cached data meets requirements (Rows: {len(df)})")
                        return df
                    else:
                        logging.warning("Cached data failed validation")
            except Exception as e:
                logging.error(f"Error loading cached data: {str(e)}")

        files_btc = [f for f in files_btc if os.path.exists(f) and os.path.getsize(f) > 100]
        files_sol = [f for f in files_sol if os.path.exists(f) and os.path.getsize(f) > 100]
        files_eth = [f for f in files_eth if os.path.exists(f) and os.path.getsize(f) > 100]
        logging.info(f"After filtering empty files: BTC={len(files_btc)}, SOL={len(files_sol)}, ETH={len(files_eth)}")

        if not files_btc or not files_sol or not files_eth:
            logging.warning("No valid data files, generating synthetic data")
            synthetic_files = generate_synthetic_data(["BTCUSDT", "SOLUSDT", "ETHUSDT"], days=MINIMUM_DAYS)
            files_btc = [f for f in synthetic_files if "BTCUSDT" in f]
            files_sol = [f for f in synthetic_files if "SOLUSDT" in f]
            files_eth = [f for f in synthetic_files if "ETHUSDT" in f]
            logging.info(f"Generated synthetic files: BTC={len(files_btc)}, SOL={len(files_sol)}, ETH={len(files_eth)}")

        price_df_btc = pd.DataFrame()
        price_df_sol = pd.DataFrame()
        price_df_eth = pd.DataFrame()
        skipped_files = []

        for file in files_btc:
            try:
                df = pd.read_csv(file)
                if df.empty or 'date' not in df.columns:
                    logging.warning(f"Empty or invalid BTC file {file}")
                    skipped_files.append(file)
                    continue
                df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                if not df.empty:
                    df.set_index("date", inplace=True)
                    price_df_btc = pd.concat([price_df_btc, df], ignore_index=False)
                else:
                    logging.warning(f"Empty or invalid BTC file {file} after processing")
                    skipped_files.append(file)
            except Exception as e:
                logging.error(f"Error processing BTC file {file}: {str(e)}")
                skipped_files.append(file)
                continue

        for file in files_sol:
            try:
                df = pd.read_csv(file)
                if df.empty or 'date' not in df.columns:
                    logging.warning(f"Empty or invalid SOL file {file}")
                    skipped_files.append(file)
                    continue
                df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                if not df.empty:
                    df.set_index("date", inplace=True)
                    price_df_sol = pd.concat([price_df_sol, df], ignore_index=False)
                else:
                    logging.warning(f"Empty or invalid SOL file {file} after processing")
                    skipped_files.append(file)
            except Exception as e:
                logging.error(f"Error processing SOL file {file}: {str(e)}")
                skipped_files.append(file)
                continue

        for file in files_eth:
            try:
                df = pd.read_csv(file)
                if df.empty or 'date' not in df.columns:
                    logging.warning(f"Empty or invalid ETH file {file}")
                    skipped_files.append(file)
                    continue
                df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                if not df.empty:
                    df.set_index("date", inplace=True)
                    price_df_eth = pd.concat([price_df_eth, df], ignore_index=False)
                else:
                    logging.warning(f"Empty or invalid ETH file {file} after processing")
                    skipped_files.append(file)
            except Exception as e:
                logging.error(f"Error processing ETH file {file}: {str(e)}")
                skipped_files.append(file)
                continue

        if price_df_sol.empty:
            logging.error("SOL data is empty, generating synthetic SOL data")
            synthetic_files = generate_synthetic_data(["SOLUSDT"], days=MINIMUM_DAYS)
            if not synthetic_files:
                logging.error("Synthetic SOL data generation failed")
                return pd.DataFrame()
            for file in synthetic_files:
                try:
                    df = pd.read_csv(file)
                    if df.empty or 'date' not in df.columns:
                        continue
                    df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                    if not df.empty:
                        df.set_index("date", inplace=True)
                        price_df_sol = pd.concat([price_df_sol, df], ignore_index=False)
                except Exception as e:
                    logging.error(f"Error processing synthetic SOL file {file}: {str(e)}")
                    continue
            if price_df_sol.empty:
                logging.error("Synthetic SOL data is empty, cannot proceed")
                return pd.DataFrame()

        if price_df_eth.empty:
            logging.warning("No valid ETH data files, generating synthetic ETH data")
            synthetic_files = generate_synthetic_data(["ETHUSDT"], days=MINIMUM_DAYS)
            if synthetic_files:
                for file in synthetic_files:
                    try:
                        df = pd.read_csv(file)
                        if df.empty or 'date' not in df.columns:
                            continue
                        df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                        df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                        if not df.empty:
                            df.set_index("date", inplace=True)
                            price_df_eth = pd.concat([price_df_eth, df], ignore_index=False)
                    except Exception as e:
                        logging.error(f"Error processing synthetic ETH file {file}: {str(e)}")
                        continue

        if price_df_btc.empty:
            logging.warning("No valid BTC data files, generating synthetic BTC data")
            synthetic_files = generate_synthetic_data(["BTCUSDT"], days=MINIMUM_DAYS)
            if synthetic_files:
                for file in synthetic_files:
                    try:
                        df = pd.read_csv(file)
                        if df.empty or 'date' not in df.columns:
                            continue
                        df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                        df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                        if not df.empty:
                            df.set_index("date", inplace=True)
                            price_df_btc = pd.concat([price_df_btc, df], ignore_index=False)
                    except Exception as e:
                        logging.error(f"Error processing synthetic BTC file {file}: {str(e)}")
                        continue

        price_df_btc = price_df_btc.loc[~price_df_btc.index.duplicated(keep='last')]
        price_df_sol = price_df_sol.loc[~price_df_sol.index.duplicated(keep='last')]
        price_df_eth = price_df_eth.loc[~price_df_eth.index.duplicated(keep='last')]

        price_df_btc = price_df_btc.rename(columns=lambda x: f"{x}_BTCUSDT")
        price_df_sol = price_df_sol.rename(columns=lambda x: f"{x}_SOLUSDT")
        price_df_eth = price_df_eth.rename(columns=lambda x: f"{x}_ETHUSDT")

        all_dates = pd.Index(price_df_sol.index.unique(), name='date')
        price_df_btc = price_df_btc.reindex(all_dates, method='ffill')
        price_df_sol = price_df_sol.reindex(all_dates, method='ffill')
        price_df_eth = price_df_eth.reindex(all_dates, method='ffill')

        price_df = pd.concat([price_df_btc, price_df_sol, price_df_eth], axis=1)
        logging.info(f"Raw concatenated DataFrame rows: {len(price_df)}")
        logging.debug(f"Raw columns: {list(price_df.columns)}")

        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            for metric in ["open", "high", "low", "close", "volume"]:
                if f"{metric}_{pair}" in price_df.columns:
                    price_df[f"{metric}_{pair}"] = pd.to_numeric(price_df[f"{metric}_{pair}"], errors='coerce')
                    if metric == "volume":
                        price_df[f"{metric}_{pair}"] = price_df[f"{metric}_{pair}"].clip(lower=1e-10, upper=1e7)

        price_df = price_df.resample(TIMEFRAME, closed='right', label='right').agg({
            f"{metric}_{pair}": "last"
            for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
            for metric in ["open", "high", "low", "close", "volume"]
            if f"{metric}_{pair}" in price_df.columns
        })
        logging.info(f"After resampling rows: {len(price_df)}")

        price_df = price_df.infer_objects(copy=False).interpolate(method='linear').ffill().bfill()

        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            if f"close_{pair}" in price_df.columns:
                price_df = calculate_technical_indicators(price_df, pair)

        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            if f"close_{pair}" in price_df.columns:
                price_df[f"log_return_{pair}"] = price_df[f"close_{pair}"].pct_change().shift(-1)
                for lag in [1, 2, 5, 10, 30]:
                    price_df[f"close_{pair}_lag{lag}"] = price_df[f"close_{pair}"].shift(lag)
                    price_df[f"log_return_{pair}_lag{lag}"] = price_df[f"log_return_{pair}"].shift(lag)

        price_df["sol_btc_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_BTCUSDT", window=10) if "close_BTCUSDT" in price_df.columns else pd.Series(0, index=price_df.index)
        price_df["sol_eth_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_ETHUSDT", window=10) if "close_ETHUSDT" in price_df.columns else pd.Series(0, index=price_df.index)
        price_df["sol_btc_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_BTCUSDT"] + 1e-10) if "volatility_BTCUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_btc_volume_ratio"] = price_df["volume_change_SOLUSDT"] / (price_df["volume_change_BTCUSDT"] + 1e-10) if "volume_change_BTCUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_eth_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_ETHUSDT"] + 1e-10) if "volatility_ETHUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_eth_momentum_ratio"] = price_df["momentum_SOLUSDT"] / (price_df["momentum_ETHUSDT"] + 1e-10) if "momentum_ETHUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        tx_volumes = fetch_solana_onchain_data(days=len(price_df))['tx_volume']
        price_df["sol_tx_volume"] = pd.Series(tx_volumes, index=price_df.index[:len(tx_volumes)])
        sentiment_scores = fetch_x_sentiment(days=len(price_df))['sentiment_score']
        price_df["sentiment_score"] = pd.Series(sentiment_scores, index=price_df.index[:len(sentiment_scores)])
        price_df["target_SOLUSDT"] = price_df["log_return_SOLUSDT"]

        feature_columns = [col for col in price_df.columns if col != 'target_SOLUSDT']
        for col in feature_columns:
            price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

        price_df = price_df.ffill().bfill().infer_objects(copy=False)
        logging.info(f"After NaN handling rows: {len(price_df)}")
        logging.debug(f"Features generated: {list(price_df.columns)}")
        feature_stats = {col: {"mean": float(price_df[col].mean()), "std": float(price_df[col].std())} for col in feature_columns if col in price_df.columns}
        low_variance_features = [col for col, stats in feature_stats.items() if stats["std"] < 1e-3]
        if low_variance_features:
            logging.warning(f"Low variance features detected: {low_variance_features}")

        if len(price_df) < MINIMUM_DAYS:
            logging.error(f"Insufficient data ({len(price_df)} rows) after preprocessing, required: {MINIMUM_DAYS}")
            return pd.DataFrame()

        if not validate_data(price_df):
            logging.error("Final data validation failed")
            return pd.DataFrame()

        price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d')
        price_df.to_csv(features_sol_path, date_format='%Y-%m-%d')
        if "close_ETHUSDT" in price_df.columns:
            price_df.to_csv(features_eth_path, date_format='%Y-%m-%d')
        logging.info(f"Data saved to {training_price_data_path}, features saved to {features_sol_path}, {features_eth_path}, rows: {len(price_df)}")
        return price_df
    except Exception as e:
        logging.error(f"Error in format_data: {str(e)}")
        synthetic_files = generate_synthetic_data(["BTCUSDT", "SOLUSDT", "ETHUSDT"], days=MINIMUM_DAYS)
        if not synthetic_files:
            logging.error("Synthetic data generation failed")
            return pd.DataFrame()
        price_df = pd.DataFrame()
        for file in synthetic_files:
            try:
                df = pd.read_csv(file)
                if df.empty or 'date' not in df.columns:
                    continue
                df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                if not df.empty:
                    df.set_index("date", inplace=True)
                    symbol = "BTCUSDT" if "BTCUSDT" in file else "SOLUSDT" if "SOLUSDT" in file else "ETHUSDT"
                    df = df.rename(columns=lambda x: f"{x}_{symbol}")
                    price_df = pd.concat([price_df, df], axis=1)
            except Exception as e:
                logging.error(f"Error processing synthetic file {file}: {str(e)}")
                continue
        if price_df.empty:
            logging.error("Synthetic data processing failed")
            return pd.DataFrame()

        price_df = price_df.infer_objects(copy=False).interpolate(method='linear').ffill().bfill()
        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            if f"close_{pair}" in price_df.columns:
                price_df = calculate_technical_indicators(price_df, pair)
                price_df[f"log_return_{pair}"] = price_df[f"close_{pair}"].pct_change().shift(-1)
                for lag in [1, 2, 5, 10, 30]:
                    price_df[f"close_{pair}_lag{lag}"] = price_df[f"close_{pair}"].shift(lag)
                    price_df[f"log_return_{pair}_lag{lag}"] = price_df[f"log_return_{pair}"].shift(lag)
                price_df[f"vwap_{pair}"] = calculate_vwap(price_df, f"close_{pair}", f"volume_{pair}")
                price_df[f"volatility_{pair}"] = calculate_volatility(price_df[f"close_{pair}"])

        price_df["sol_btc_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_BTCUSDT", window=10) if "close_BTCUSDT" in price_df.columns else pd.Series(0, index=price_df.index)
        price_df["sol_eth_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_ETHUSDT", window=10) if "close_ETHUSDT" in price_df.columns else pd.Series(0, index=price_df.index)
        price_df["sol_btc_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_BTCUSDT"] + 1e-10) if "volatility_BTCUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_btc_volume_ratio"] = price_df["volume_change_SOLUSDT"] / (price_df["volume_change_BTCUSDT"] + 1e-10) if "volume_change_BTCUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_eth_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_ETHUSDT"] + 1e-10) if "volatility_ETHUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_eth_momentum_ratio"] = price_df["momentum_SOLUSDT"] / (price_df["momentum_ETHUSDT"] + 1e-10) if "momentum_ETHUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        tx_volumes = fetch_solana_onchain_data(days=len(price_df))['tx_volume']
        price_df["sol_tx_volume"] = pd.Series(tx_volumes, index=price_df.index[:len(tx_volumes)])
        sentiment_scores = fetch_x_sentiment(days=len(price_df))['sentiment_score']
        price_df["sentiment_score"] = pd.Series(sentiment_scores, index=price_df.index[:len(sentiment_scores)])
        price_df["target_SOLUSDT"] = price_df["log_return_SOLUSDT"]

        price_df = price_df.ffill().bfill().infer_objects(copy=False)
        logging.info(f"After synthetic data NaN handling rows: {len(price_df)}")
        price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d')
        price_df.to_csv(features_sol_path, date_format='%Y-%m-%d')
        if "close_ETHUSDT" in price_df.columns:
            price_df.to_csv(features_eth_path, date_format='%Y-%m-%d')
        logging.info(f"Data saved to {training_price_data_path}, features saved to {features_sol_path}, {features_eth_path}, rows: {len(price_df)}")
        return price_df

def select_features(X, y, k=15):
    try:
        priority_features = SELECTED_FEATURES
        available_features = [f for f in priority_features if f in X.columns]
        logging.info(f"Priority features available: {available_features}")
        logging.debug(f"All columns in X: {list(X.columns)}")

        feature_stats = {col: {"std": float(X[col].std())} for col in X.columns}
        low_variance_features = [col for col, stats in feature_stats.items() if stats["std"] < 1e-3]
        if low_variance_features:
            logging.error(f"Low variance features in X: {low_variance_features}")
            return [], []

        # Use all available priority features if they meet the minimum requirement
        if len(available_features) >= k:
            selected_features = available_features[:k]
            selected_indices = [X.columns.get_loc(f) for f in selected_features]
            logging.info(f"Using {len(selected_features)} priority features: {selected_features}")
        else:
            # Fall back to SelectKBest only if insufficient priority features
            logging.warning(f"Insufficient priority features ({len(available_features)}), selecting additional features")
            selected_features = available_features.copy()
            remaining_k = k - len(selected_features)
            non_priority_features = [col for col in X.columns if col not in selected_features and col != 'target_SOLUSDT']
            if non_priority_features and remaining_k > 0:
                X_non_priority = X[non_priority_features]
                selector = SelectKBest(score_func=mutual_info_regression, k=min(remaining_k, len(non_priority_features)))
                selector.fit(X_non_priority, y)
                scores = selector.scores_
                top_indices = selector.get_support(indices=True)
                top_features = [non_priority_features[i] for i in top_indices]
                logging.info(f"Top additional features by mutual information: {list(zip(top_features, scores[top_indices]))}")
                selected_features.extend(top_features)
            
            selected_features = selected_features[:k]
            selected_indices = [X.columns.get_loc(f) for f in selected_features if f in X.columns]
        
        logging.info(f"Final selected features: {selected_features}")
        return selected_indices, [1.0 if f in available_features else 0.5 for f in selected_features]
    except Exception as e:
        logging.error(f"Error in select_features: {str(e)}")
        return [], []

def load_frame(file_path, timeframe, feature_subset=None):
    try:
        if not os.path.exists(file_path):
            logging.error(f"Training data file {file_path} does not exist. Generating synthetic data.")
            synthetic_files = generate_synthetic_data(["BTCUSDT", "SOLUSDT", "ETHUSDT"], days=MINIMUM_DAYS)
            if not synthetic_files:
                logging.error("Synthetic data generation failed")
                return None, None, None, None, None, None
            df = format_data(synthetic_files, synthetic_files, synthetic_files, "coingecko")
            if df.empty:
                logging.error("Synthetic data formatting failed")
                return None, None, None, None, None, None
            file_path = training_price_data_path
        
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        df = df.infer_objects(copy=False).interpolate(method='linear').ffill().bfill()

        if 'target_SOLUSDT' not in df.columns:
            logging.error(f"target_SOLUSDT missing in {file_path}")
            return None, None, None, None, None, None
        if df['target_SOLUSDT'].isna().all():
            logging.error(f"target_SOLUSDT contains only NaN values in {file_path}")
            return None, None, None, None, None, None

        all_features = SELECTED_FEATURES
        features = all_features
        if feature_subset:
            if all(isinstance(i, str) for i in feature_subset):
                features = [f for f in feature_subset if f in all_features and f in df.columns]
            else:
                feature_subset = [int(i) for i in feature_subset if isinstance(i, (int, str)) and int(i) < len(all_features)]
                features = [all_features[i] for i in feature_subset if i < len(all_features)]

        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logging.warning(f"Missing features in load_frame: {missing_features}. Using available features.")
            features = [f for f in features if f in df.columns]

        X = df[features]
        y = df["target_SOLUSDT"]

        if len(X) < MINIMUM_DAYS:
            logging.error(f"Insufficient samples ({len(X)}) for scaling in load_frame, required: {MINIMUM_DAYS}")
            return None, None, None, None, None, None

        X_selected = X
        selected_features = features
        if not feature_subset:
            selected_indices, _ = select_features(X, y, k=15)
            if not selected_indices:
                logging.error("Feature selection failed, returning None")
                return None, None, None, None, None, None
            selected_features = [X.columns[i] for i in selected_indices]
            X_selected = X[selected_features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features, index=X.index)

        split_idx = int(len(X) * 0.8)
        if split_idx == 0:
            logging.error("Not enough data to split into training and test sets")
            return None, None, None, None, None, None
        X_train, X_test = X_scaled_df[:split_idx], X_scaled_df[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logging.info(f"Loaded frame: {len(X_train)} training samples, {len(X_test)} test samples, features: {selected_features}")
        return X_train, X_test, y_train, y_test, scaler, selected_features
    except Exception as e:
        logging.error(f"Error in load_frame: {str(e)}")
        return None, None, None, None, None, None

def zptae_loss(y_true, y_pred, weights=None):
    try:
        ref_std = np.std(y_true[-100:]) if len(y_true) >= 100 else np.std(y_true)
        if ref_std == 0:
            ref_std = 1e-10
        abs_error = np.abs(y_true - y_pred) / ref_std
        power_tanh = np.tanh(abs_error) ** 2
        return np.mean(power_tanh) if weights is None else np.average(power_tanh, weights=weights)
    except Exception as e:
        logging.error(f"Error calculating ZPTAE loss: {str(e)}")
        return float('inf')

def weighted_rmse(y_true, y_pred, weights):
    try:
        return np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights))
    except Exception as e:
        logging.error(f"Error calculating weighted RMSE: {str(e)}")
        return float('inf')

def weighted_mztae(y_true, y_pred, weights):
    try:
        ref_std = np.std(y_true[-100:]) if len(y_true) >= 100 else np.std(y_true)
        return np.average(np.abs((y_true - y_pred) / ref_std), weights=weights)
    except Exception as e:
        logging.error(f"Error calculating weighted MZTAE: {str(e)}")
        return float('inf')

def train_model(timeframe, file_path=training_price_data_path, model_type=MODEL, model_params=None, feature_subset=None):
    try:
        logging.info(f"Starting model training for {model_type} with timeframe {timeframe}")
        X_train, X_test, y_train, y_test, scaler, selected_features = load_frame(file_path, timeframe, feature_subset)
        if X_train is None:
            logging.error("Failed to load frame, cannot train model")
            return None, None, {}, [], []

        logging.info(f"Training {model_type} with features: {selected_features}")

        feature_stats = {col: {"std": float(X_train[col].std())} for col in X_train.columns}
        low_variance_features = [col for col, stats in feature_stats.items() if stats["std"] < 1e-3]
        if low_variance_features:
            logging.error(f"Low variance features in X_train: {low_variance_features}, switching to synthetic data")
            synthetic_files = generate_synthetic_data(["BTCUSDT", "SOLUSDT", "ETHUSDT"], days=MINIMUM_DAYS)
            if not synthetic_files:
                logging.error("Synthetic data generation failed")
                return None, None, {}, [], []
            df = format_data(synthetic_files, synthetic_files, synthetic_files, "coingecko")
            if df.empty:
                logging.error("Synthetic data formatting failed")
                return None, None, {}, [], []
            X_train, X_test, y_train, y_test, scaler, selected_features = load_frame(training_price_data_path, timeframe, feature_subset)
            if X_train is None:
                logging.error("Failed to load synthetic frame, cannot train model")
                return None, None, {}, [], []

        baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test)
        baseline_zptae = zptae_loss(y_test, baseline_pred, np.abs(y_test))
        baseline_mztae = weighted_mztae(y_test, baseline_pred, np.abs(y_test))
        logging.info(f"Baseline (Random Forest) ZPTAE: {baseline_zptae:.6f}, Weighted MZTAE: {baseline_mztae:.6f}")

        if optuna and model_type == "LightGBM":
            def objective(trial):
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "random_state": 42
                }
                model = LGBMRegressor(**params)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                return zptae_loss(y_test, pred)
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=50)
            model_params = study.best_params
            logging.info(f"Optuna best params: {model_params}")

        if model_type == "LightGBM":
            model_params = model_params or MODEL_PARAMS
            logging.info(f"Using LightGBM parameters: {model_params}")
            model = LGBMRegressor(**model_params)
        elif model_type == "LSTM" and torch:
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size=50, num_layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])
            model = LSTMModel(input_size=len(selected_features))
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            X_train_tensor = torch.tensor(X_train.values.reshape(-1, 1, X_train.shape[1]), dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            for epoch in range(50):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
        else:
            logging.warning(f"Unsupported model type: {model_type}. Falling back to LightGBM.")
            model_params = model_params or MODEL_PARAMS
            logging.info(f"Using LightGBM parameters: {model_params}")
            model = LGBMRegressor(**model_params)

        if model_type != "LSTM" or not torch:
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model_rf = LGBMRegressor(**model_params) if model_type == "LightGBM" else model
                model_rf.fit(X_cv_train, y_cv_train)
                cv_pred = model_rf.predict(X_cv_val)
                cv_score = r2_score(y_cv_val, cv_pred)
                cv_scores.append(cv_score)
            
            logging.info(f"Cross-validation R² scores: {cv_scores}, Mean: {np.mean(cv_scores):.6f}, Std: {np.std(cv_scores):.6f}")

            model.fit(X_train, y_train)

        pred_train = model.predict(X_train) if model_type != "LSTM" or not torch else model(torch.tensor(X_train.values.reshape(-1, 1, X_train.shape[1]), dtype=torch.float32)).detach().numpy().flatten()
        pred_test = model.predict(X_test) if model_type != "LSTM" or not torch else model(torch.tensor(X_test.values.reshape(-1, 1, X_test.shape[1]), dtype=torch.float32)).detach().numpy().flatten()

        pred_test_std = np.std(pred_test)
        y_test_std = np.std(y_test)
        if pred_test_std < 1e-3 or y_test_std < 1e-3:
            logging.error(f"Low variance (pred: {pred_test_std:.6e}, target: {y_test_std:.6e}), model may be underfitting")
            return None, None, {}, [], []

        weights = np.abs(y_test)
        train_mae = mean_absolute_error(y_train, pred_train)
        test_mae = mean_absolute_error(y_test, pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        train_r2 = r2_score(y_train, pred_train)
        test_r2 = r2_score(y_test, pred_test)
        train_zptae = zptae_loss(y_train, pred_train, np.abs(y_train))
        test_zptae = zptae_loss(y_test, pred_test, weights)
        train_mztae = weighted_mztae(y_train, pred_train, np.abs(y_train))
        test_mztae = weighted_mztae(y_test, pred_test, weights)
        directional_accuracy = np.mean(np.sign(pred_test) == np.sign(y_test))
        correlation, p_value = pearsonr(y_test, pred_test)

        n_successes = int(directional_accuracy * len(y_test))
        binom_p_value = binomtest(n_successes, len(y_test), p=0.5, alternative='greater').pvalue
        logging.info(f"Binomial Test p-value for Directional Accuracy: {binom_p_value:.4f}")
        logging.info(f"Training MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")
        logging.info(f"Training RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")
        logging.info(f"Training R²: {train_r2:.6f}, Test R²: {test_r2:.6f}")
        logging.info(f"Training ZPTAE: {train_zptae:.6f}, Test ZPTAE: {test_zptae:.6f}")
        logging.info(f"Weighted MZTAE: {test_mztae:.6f}")
        logging.info(f"Baseline ZPTAE: {baseline_zptae:.6f}, Improvement: {100 * (baseline_zptae - test_zptae) / baseline_zptae:.2f}%")
        logging.info(f"Baseline MZTAE: {baseline_mztae:.6f}, Improvement: {100 * (baseline_mztae - test_mztae) / baseline_mztae:.2f}%")
        logging.info(f"Directional Accuracy: {directional_accuracy:.4f}")
        logging.info(f"Correlation: {correlation:.4f}, p-value: {p_value:.4f}")

        feature_importances = getattr(model, 'feature_importances_', np.zeros(len(selected_features))) if model_type != "LSTM" or not torch else np.ones(len(selected_features)) / len(selected_features)
        if np.sum(feature_importances) == 0:
            logging.warning("Zero feature importances detected, using uniform weights")
            feature_importances = np.ones(len(selected_features)) / len(selected_features)
        else:
            feature_importances = feature_importances / np.sum(feature_importances)
        logging.info(f"Feature importances: {list(zip(selected_features, feature_importances))}")

        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_file_path, "wb") as f:
            pickle.dump(scaler, f)
        logging.info(f"Model saved to {model_file_path}, scaler saved to {scaler_file_path}")

        metrics = {
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_zptae': float(train_zptae),
            'test_zptae': float(test_zptae),
            'train_mztae': float(train_mztae),
            'test_mztae': float(test_mztae),
            'directional_accuracy': float(directional_accuracy),
            'correlation': float(correlation),
            'correlation_p_value': float(p_value),
            'binom_p_value': float(binom_p_value),
            'baseline_zptae': float(baseline_zptae),
            'baseline_mztae': float(baseline_mztae),
            'cv_r2_mean': float(np.mean(cv_scores)) if cv_scores else 0.0,
            'cv_r2_std': float(np.std(cv_scores)) if cv_scores else 0.0
        }

        return model, scaler, metrics, selected_features, feature_subset
    except Exception as e:
        logging.error(f"Error in train_model: {str(e)}")
        return None, None, {}, [], []

def get_inference(token, timeframe, region, data_provider, features, cached_data=None):
    try:
        if not os.path.exists(model_file_path):
            logging.error(f"Model file {model_file_path} not found")
            return 0.0
        with open(model_file_path, "rb") as f:
            model = pickle.load(f)

        df = cached_data
        if df is None:
            files_btc = download_data("BTC", 30, region, data_provider)
            files_sol = download_data("SOL", 30, region, data_provider)
            files_eth = download_data("ETH", 30, region, data_provider)
            df = format_data(files_btc, files_sol, files_eth, data_provider)
            if df.empty:
                logging.error("Failed to fetch or format inference data")
                return 0.0

        available_features = [f for f in features if f in df.columns]
        missing_cols = [f for f in features if f not in df.columns]
        if missing_cols:
            logging.warning(f"Missing feature columns: {missing_cols}. Using available features: {available_features}")
            if not available_features:
                logging.error("No valid features available for prediction")
                return 0.0
            features = available_features
        X = df[features]
        if len(X) < 1:
            logging.error("No valid data for prediction")
            return 0.0

        with open(scaler_file_path, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)

        last_row = X_scaled_df.iloc[-1:]
        pred = model.predict(last_row) if not isinstance(model, nn.Module) else model(torch.tensor(last_row.values.reshape(-1, 1, last_row.shape[1]), dtype=torch.float32)).detach().numpy().flatten()

        latest_price = 100.0
        for attempt in range(3):
            try:
                ticker_url = f'https://api.binance.{region}/api/v3/ticker/price?symbol=SOLUSDT'
                response = requests.get(ticker_url, timeout=15)
                response.raise_for_status()
                latest_price = float(response.json()['price'])
                logging.info(f"Binance API success: Latest SOL price {latest_price:.3f}")
                break
            except Exception as e:
                logging.warning(f"Binance API attempt {attempt+1} failed: {str(e)}")
                if attempt == 2:
                    logging.warning("Binance API failed, using CoinGecko fallback.")
                    try:
                        cg_url = 'https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd'
                        response = requests.get(cg_url, timeout=15)
                        response.raise_for_status()
                        latest_price = float(response.json()['solana']['usd'])
                        logging.info(f"CoinGecko API success: Latest SOL price {latest_price:.3f}")
                    except Exception as cg_e:
                        logging.error(f"CoinGecko fallback failed: {str(cg_e)}. Using fallback price: {latest_price:.3f}")

        log_return_pred = pred[0]
        predicted_price = latest_price * (1 + log_return_pred)
        logging.info(f"Predicted {timeframe} SOL/USD Log Return: {log_return_pred:.6f}")
        logging.info(f"Latest SOL Price: {latest_price:.3f}")
        logging.info(f"Predicted SOL Price in {timeframe}: {predicted_price:.3f}")
        return log_return_pred
    except Exception as e:
        logging.error(f"Error in get_inference: {str(e)}")
        return 0.0

if __name__ == "__main__":
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    files_sol = download_data("SOL", TRAINING_DAYS, REGION, DATA_PROVIDER)
    files_eth = download_data("ETH", TRAINING_DAYS, REGION, DATA_PROVIDER)
    df = format_data(files_btc, files_sol, files_eth, DATA_PROVIDER)
    if not df.empty:
        model, scaler, metrics, features, _ = train_model(TIMEFRAME, model_type=MODEL)
        if model is not None:
            log_return = get_inference(TOKEN, TIMEFRAME, REGION, DATA_PROVIDER, features, cached_data=df)
        else:
            logging.error(f"Training failed for {MODEL}, cannot perform inference")
    else:
        logging.error("Data formatting failed, cannot train or infer")
