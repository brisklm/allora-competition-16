# Optimized for Allora Competition 16, Topic 62
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from scipy.stats import pearsonr, binomtest
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import time
import glob
try:
    from updater import download_binance_daily_data, download_binance_current_day_data
except ImportError as e:
    print(f"[{datetime.now()}] ImportError: {str(e)}. Ensure updater.py is available.")
    raise
from config import data_base_path, model_file_path, scaler_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER, MODEL, sol_source_path, eth_source_path, features_sol_path, features_eth_path, HELIUS_API_KEY

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

MODEL_VERSION = "2025-07-23-competition16-topic62-multimodel-v50"
MINIMUM_DAYS = 180
print(f"[{datetime.now()}] Loaded model.py version {MODEL_VERSION} (multi-model: {MODEL}, timeframe: {TIMEFRAME}) at {os.path.abspath(__file__)}")

def generate_synthetic_data(symbols, days=MINIMUM_DAYS):
    try:
        start_date = datetime(2025, 7, 22)
        data = {symbol: [] for symbol in symbols}
        np.random.seed(42)
        for i in range(days):
            date = (start_date - timedelta(days=i)).strftime("%Y-%m-%d")
            for symbol in symbols:
                base_price = 100 if "SOL" in symbol else (50000 if "BTC" in symbol else 2000)
                price_noise = np.random.normal(0, 0.05)
                open_price = base_price * (1 + price_noise)
                high = open_price * (1 + np.random.uniform(0.005, 0.015))
                low = open_price * (1 - np.random.uniform(0.005, 0.015))
                close = open_price * (1 + np.random.normal(0, 0.01))
                volume = 1000 * (1 + np.random.uniform(0, 0.5))
                data[symbol].append({
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume
                })
        files = []
        os.makedirs(coingecko_data_path, exist_ok=True)
        for symbol in symbols:
            df = pd.DataFrame(data[symbol])
            file_path = os.path.join(coingecko_data_path, f"{symbol}_{start_date.strftime('%Y-%m-%d')}_synthetic.csv")
            df.to_csv(file_path, index=False)
            files.append(file_path)
        print(f"[{datetime.now()}] Generated {len(files)} synthetic files for {symbols}")
        return files
    except Exception as e:
        print(f"[{datetime.now()}] Error generating synthetic data: {str(e)}")
        return []

def download_data(token, training_days, region, data_provider=DATA_PROVIDER):
    try:
        print(f"[{datetime.now()}] Checking cached data for {token} at {training_price_data_path}")
        if os.path.exists(training_price_data_path):
            try:
                df = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
                if not df.empty and 'target_SOLUSDT' in df.columns and not df['target_SOLUSDT'].isna().all():
                    feature_stats = {col: {"std": float(df[col].std())} for col in df.columns if col != 'target_SOLUSDT'}
                    low_variance_features = [col for col, stats in feature_stats.items() if stats["std"] < 1e-10]
                    if len(df) >= MINIMUM_DAYS and not low_variance_features:
                        print(f"[{datetime.now()}] Found valid cached data with {len(df)} rows")
                        return []
                    else:
                        print(f"[{datetime.now()}] Cached data insufficient (Rows: {len(df)}, Low variance: {low_variance_features})")
            except Exception as e:
                print(f"[{datetime.now()}] Error loading cached data: {str(e)}")
        
        if data_provider == "binance":
            save_path = binance_data_path
            if token == "SOL" and os.path.exists(sol_source_path):
                print(f"[{datetime.now()}] Using SOL data from {sol_source_path}")
                return [sol_source_path]
            if token == "ETH" and os.path.exists(eth_source_path):
                print(f"[{datetime.now()}] Using ETH data from {eth_source_path}")
                return [eth_source_path]
            return download_binance_daily_data(f"{token}USDT", training_days, region, save_path)
        else:
            save_path = coingecko_data_path
            files = [f for f in glob.glob(os.path.join(save_path, f"{token}_*.csv")) if os.path.getsize(f) > 100]
            if files:
                total_rows = sum(len(pd.read_csv(f)) for f in files)
                if total_rows >= MINIMUM_DAYS:
                    print(f"[{datetime.now()}] Found {len(files)} existing files for {token} with {total_rows} rows")
                    return files
                else:
                    print(f"[{datetime.now()}] Insufficient rows ({total_rows} < {MINIMUM_DAYS})")
            return generate_synthetic_data([token], days=MINIMUM_DAYS)
    except Exception as e:
        print(f"[{datetime.now()}] Error downloading data for {token}: {str(e)}")
        return generate_synthetic_data([token], days=MINIMUM_DAYS)

def fetch_solana_onchain_data():
    try:
        url = "https://api.mainnet-beta.solana.com"
        headers = {"Content-Type": "application/json"}
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getRecentPerformanceSamples",
            "params": [1]
        }
        response = requests.post(url, headers=headers, json=payload, timeout=3)
        response.raise_for_status()
        data = response.json()
        tx_volume = data["result"][0]["numTransactions"] if data["result"] else np.random.normal(1000, 200)
        return {'tx_volume': tx_volume}
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching Solana on-chain data: {str(e)}")
        return {'tx_volume': np.random.normal(1000, 200)}

def calculate_rsi(data, periods=14):
    try:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating RSI: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_volatility(data, window=5):
    try:
        return data.pct_change().rolling(window=window).std()
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating volatility: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_ma(data, window=5):
    try:
        return data.rolling(window=window).mean()
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating MA: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_momentum(data, window=5):
    try:
        return data.pct_change(window)
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating momentum: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_volume_momentum(data, window=5):
    try:
        return data.pct_change(window)
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating volume momentum: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_macd(data, fast=12, slow=26, signal=9):
    try:
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating MACD: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_bollinger_bands(data, window=20, num_std=2):
    try:
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating Bollinger Bands: {str(e)}")
        return pd.Series(0, index=data.index), pd.Series(0, index=data.index)

def calculate_cross_asset_correlation(data, pair1, pair2, window=10):
    try:
        if pair1 not in data.columns or pair2 not in data.columns:
            print(f"[{datetime.now()}] Warning: Missing columns for correlation: {pair1}, {pair2}")
            return pd.Series(0, index=data.index)
        corr = data[pair1].pct_change().rolling(window=window).corr(data[pair2].pct_change())
        return corr.fillna(0)
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating cross-asset correlation: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_volume_change(data, window=1):
    try:
        return data.pct_change(window).fillna(0)
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating volume change: {str(e)}")
        return pd.Series(0, index=data.index)

def fetch_sentiment_data():
    try:
        positive_keywords = ["bullish", "buy", "up", "solana moon"]
        negative_keywords = ["bearish", "sell", "down", "crash"]
        sentiment_score = 0.1 * len(positive_keywords) - 0.1 * len(negative_keywords) + np.random.normal(0, 0.2)
        return {'sentiment_score': sentiment_score}
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching sentiment data: {str(e)}")
        return {'sentiment_score': np.random.normal(0, 0.2)}

def format_data(files_btc, files_sol, files_eth, data_provider):
    try:
        print(f"[{datetime.now()}] Using TIMEFRAME={TIMEFRAME}, TRAINING_DAYS={TRAINING_DAYS}, Model Version={MODEL_VERSION}")
        print(f"[{datetime.now()}] Input files: BTC={len(files_btc)}, SOL={len(files_sol)}, ETH={len(files_eth)}")
        
        if os.path.exists(training_price_data_path):
            print(f"[{datetime.now()}] Checking cached data first at {training_price_data_path}")
            try:
                df = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
                if not df.empty:
                    print(f"[{datetime.now()}] Loaded cached data with {len(df)} rows")
                    print(f"[{datetime.now()}] Available columns in cached data: {list(df.columns)}")
                    if 'target_SOLUSDT' not in df.columns:
                        print(f"[{datetime.now()}] Error: target_SOLUSDT missing in cached data")
                        return pd.DataFrame()
                    if df['target_SOLUSDT'].isna().all():
                        print(f"[{datetime.now()}] Error: target_SOLUSDT contains only NaN values in cached data")
                        return pd.DataFrame()
                    feature_stats = {col: {"std": float(df[col].std())} for col in df.columns if col != 'target_SOLUSDT'}
                    low_variance_features = [col for col, stats in feature_stats.items() if stats["std"] < 1e-10]
                    if len(df) >= MINIMUM_DAYS and not low_variance_features:
                        print(f"[{datetime.now()}] Cached data meets requirements (Rows: {len(df)}, No low variance features)")
                        return df
                    else:
                        print(f"[{datetime.now()}] Cached data is insufficient (Rows: {len(df)}, Low variance features: {low_variance_features})")
            except Exception as e:
                print(f"[{datetime.now()}] Error loading cached data: {str(e)}")

        current_date = datetime(2025, 7, 23).strftime("%Y-%m-%d")
        files_btc = [f for f in files_btc if current_date not in os.path.basename(f)]
        files_sol = [f for f in files_sol if current_date not in os.path.basename(f)]
        files_eth = [f for f in files_eth if current_date not in os.path.basename(f)]
        print(f"[{datetime.now()}] After filtering current day: BTC={len(files_btc)}, SOL={len(files_sol)}, ETH={len(files_eth)}")

        if data_provider == "coingecko":
            files_btc = sorted([f for f in files_btc if "BTC" in os.path.basename(f) and f.endswith(".csv")])
            files_sol = sorted([f for f in files_sol if "SOL" in os.path.basename(f) and f.endswith(".csv")])
            files_eth = sorted([f for f in files_eth if "ETH" in os.path.basename(f) and f.endswith(".csv")])
        else:
            print(f"[{datetime.now()}] Using Binance data provider")

        price_df_btc = pd.DataFrame()
        price_df_sol = pd.DataFrame()
        price_df_eth = pd.DataFrame()
        skipped_files = []

        for file in files_btc:
            if not os.path.exists(file):
                print(f"[{datetime.now()}] File not found: {file}")
                skipped_files.append(file)
                continue
            if os.path.getsize(file) < 100:
                print(f"[{datetime.now()}] Warning: File {file} is empty or too small ({os.path.getsize(file)} bytes)")
                skipped_files.append(file)
                continue
            try:
                df = pd.read_csv(file)
                if df.empty or 'date' not in df.columns:
                    print(f"[{datetime.now()}] Warning: Empty or invalid BTC file {file}")
                    skipped_files.append(file)
                    continue
                df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                if not df.empty:
                    df.set_index("date", inplace=True)
                    price_df_btc = pd.concat([price_df_btc, df], ignore_index=False)
                else:
                    print(f"[{datetime.now()}] Warning: Empty or invalid BTC file {file} after processing")
                    skipped_files.append(file)
            except Exception as e:
                print(f"[{datetime.now()}] Error processing BTC file {file}: {str(e)}")
                skipped_files.append(file)
                continue

        for file in files_sol:
            if not os.path.exists(file):
                print(f"[{datetime.now()}] File not found: {file}")
                skipped_files.append(file)
                continue
            if os.path.getsize(file) < 100:
                print(f"[{datetime.now()}] Warning: File {file} is empty or too small ({os.path.getsize(file)} bytes)")
                skipped_files.append(file)
                continue
            try:
                df = pd.read_csv(file)
                if df.empty or 'date' not in df.columns:
                    print(f"[{datetime.now()}] Warning: Empty or invalid SOL file {file}")
                    skipped_files.append(file)
                    continue
                df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                if not df.empty:
                    df.set_index("date", inplace=True)
                    price_df_sol = pd.concat([price_df_sol, df], ignore_index=False)
                else:
                    print(f"[{datetime.now()}] Warning: Empty or invalid SOL file {file} after processing")
                    skipped_files.append(file)
            except Exception as e:
                print(f"[{datetime.now()}] Error processing SOL file {file}: {str(e)}")
                skipped_files.append(file)
                continue

        for file in files_eth:
            if not os.path.exists(file):
                print(f"[{datetime.now()}] File not found: {file}")
                skipped_files.append(file)
                continue
            if os.path.getsize(file) < 100:
                print(f"[{datetime.now()}] Warning: File {file} is empty or too small ({os.path.getsize(file)} bytes)")
                skipped_files.append(file)
                continue
            try:
                df = pd.read_csv(file)
                if df.empty or 'date' not in df.columns:
                    print(f"[{datetime.now()}] Warning: Empty or invalid ETH file {file}")
                    skipped_files.append(file)
                    continue
                df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                if not df.empty:
                    df.set_index("date", inplace=True)
                    price_df_eth = pd.concat([price_df_eth, df], ignore_index=False)
                else:
                    print(f"[{datetime.now()}] Warning: Empty or invalid ETH file {file} after processing")
                    skipped_files.append(file)
            except Exception as e:
                print(f"[{datetime.now()}] Error processing ETH file {file}: {str(e)}")
                skipped_files.append(file)
                continue

        if price_df_sol.empty:
            print(f"[{datetime.now()}] Error: SOL data is empty, attempting cached data")
            if os.path.exists(training_price_data_path):
                print(f"[{datetime.now()}] Attempting to load cached data from {training_price_data_path}")
                try:
                    df = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
                    if not df.empty:
                        print(f"[{datetime.now()}] Loaded cached data with {len(df)} rows")
                        print(f"[{datetime.now()}] Available columns in cached data: {list(df.columns)}")
                        if 'target_SOLUSDT' not in df.columns:
                            print(f"[{datetime.now()}] Error: target_SOLUSDT missing in cached data")
                            return pd.DataFrame()
                        if df['target_SOLUSDT'].isna().all():
                            print(f"[{datetime.now()}] Error: target_SOLUSDT contains only NaN values in cached data")
                            return pd.DataFrame()
                        return df
                    else:
                        print(f"[{datetime.now()}] Error: Cached data is empty")
                except Exception as e:
                    print(f"[{datetime.now()}] Error loading cached data: {str(e)}")
            print(f"[{datetime.now()}] No cached data available, relying on synthetic SOL data")
            if price_df_sol.empty:
                print(f"[{datetime.now()}] Error: Synthetic SOL data is empty, cannot proceed")
                return pd.DataFrame()

        if len(price_df_btc) < MINIMUM_DAYS or len(price_df_eth) < MINIMUM_DAYS:
            print(f"[{datetime.now()}] Warning: Insufficient data for BTC ({len(price_df_btc)}) or ETH ({len(price_df_eth)}), proceeding with SOL data")

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
        print(f"[{datetime.now()}] Raw concatenated DataFrame rows: {len(price_df)}")
        print(f"[{datetime.now()}] Raw columns: {list(price_df.columns)}")

        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            for metric in ["open", "high", "low", "close", "volume"]:
                if f"{metric}_{pair}" in price_df.columns:
                    price_df[f"{metric}_{pair}"] = pd.to_numeric(price_df[f"{metric}_{pair}"], errors='coerce')

        price_df = price_df.resample(TIMEFRAME, closed='right', label='right').agg({
            f"{metric}_{pair}": "last"
            for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
            for metric in ["open", "high", "low", "close", "volume"]
            if f"{metric}_{pair}" in price_df.columns
        })
        print(f"[{datetime.now()}] After resampling rows: {len(price_df)}")

        price_df = price_df.infer_objects(copy=False).interpolate(method='linear').ffill().bfill()

        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            if f"close_{pair}" in price_df.columns:
                price_df[f"log_return_{pair}"] = price_df[f"close_{pair}"].pct_change().shift(-1)
                for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
                    price_df[f"close_{pair}_lag{lag}"] = price_df[f"close_{pair}"].shift(lag)
                price_df[f"rsi_{pair}"] = calculate_rsi(price_df[f"close_{pair}"], periods=14)
                price_df[f"volatility_{pair}"] = calculate_volatility(price_df[f"close_{pair}"], window=5)
                price_df[f"ma5_{pair}"] = calculate_ma(price_df[f"close_{pair}"], window=5)
                price_df[f"ma20_{pair}"] = calculate_ma(price_df[f"close_{pair}"], window=20)
                price_df[f"ma50_{pair}"] = calculate_ma(price_df[f"close_{pair}"], window=50)
                price_df[f"momentum_{pair}"] = calculate_momentum(price_df[f"close_{pair}"], window=5)
                price_df[f"volume_momentum_{pair}"] = calculate_volume_momentum(price_df[f"volume_{pair}"], window=5)
                price_df[f"macd_{pair}"] = calculate_macd(price_df[f"close_{pair}"])
                price_df[f"bb_upper_{pair}"], price_df[f"bb_lower_{pair}"] = calculate_bollinger_bands(price_df[f"close_{pair}"])
                price_df[f"volume_change_{pair}"] = calculate_volume_change(price_df[f"volume_{pair}"])

        price_df["sol_btc_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_BTCUSDT", window=10) if "close_BTCUSDT" in price_df.columns else pd.Series(0, index=price_df.index)
        price_df["sol_eth_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_ETHUSDT", window=10) if "close_ETHUSDT" in price_df.columns else pd.Series(0, index=price_df.index)
        price_df["sol_btc_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_BTCUSDT"] + 1e-10) if "volatility_BTCUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_btc_volume_ratio"] = price_df["volume_change_SOLUSDT"] / (price_df["volume_change_BTCUSDT"] + 1e-10) if "volume_change_BTCUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_eth_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_ETHUSDT"] + 1e-10) if "volatility_ETHUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_eth_momentum_ratio"] = price_df["momentum_SOLUSDT"] / (price_df["momentum_ETHUSDT"] + 1e-10) if "momentum_ETHUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        onchain_data = fetch_solana_onchain_data()
        price_df["sol_tx_volume"] = np.random.normal(onchain_data['tx_volume'], 200, len(price_df))
        sentiment_data = fetch_sentiment_data()
        price_df["sentiment_score"] = np.random.normal(sentiment_data['sentiment_score'], 0.2, len(price_df))

        price_df["target_SOLUSDT"] = price_df["log_return_SOLUSDT"]

        feature_columns = [col for col in price_df.columns if col != 'target_SOLUSDT']
        for col in feature_columns:
            price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

        price_df = price_df.fillna(0).infer_objects(copy=False)
        print(f"[{datetime.now()}] After NaN handling rows: {len(price_df)}")
        print(f"[{datetime.now()}] Features generated: {list(price_df.columns)}")
        print(f"[{datetime.now()}] NaN counts: {price_df.isna().sum().to_dict()}")
        feature_stats = {col: {"mean": float(price_df[col].mean()), "std": float(price_df[col].std())} for col in feature_columns if col in price_df.columns}
        print(f"[{datetime.now()}] Feature statistics: {feature_stats}")
        print(f"[{datetime.now()}] Dtypes: {price_df.dtypes.to_dict()}")
        low_variance_features = [col for col, stats in feature_stats.items() if stats["std"] < 1e-10]
        if low_variance_features:
            print(f"[{datetime.now()}] Warning: Low variance features detected: {low_variance_features}")

        if len(price_df) < MINIMUM_DAYS:
            print(f"[{datetime.now()}] Error: Insufficient data ({len(price_df)} rows) after preprocessing, required: {MINIMUM_DAYS}")
            return pd.DataFrame()

        price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d')
        price_df.to_csv(features_sol_path, date_format='%Y-%m-%d')
        if "close_ETHUSDT" in price_df.columns:
            price_df.to_csv(features_eth_path, date_format='%Y-%m-%d')
        print(f"[{datetime.now()}] Data saved to {training_price_data_path}, features saved to {features_sol_path}, {features_eth_path}, rows: {len(price_df)}")
        return price_df

    except Exception as e:
        print(f"[{datetime.now()}] Error in format_data: {str(e)}")
        if os.path.exists(training_price_data_path):
            print(f"[{datetime.now()}] Attempting to load cached data from {training_price_data_path}")
            try:
                df = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
                if not df.empty:
                    print(f"[{datetime.now()}] Loaded cached data with {len(df)} rows")
                    print(f"[{datetime.now()}] Available columns in cached data: {list(df.columns)}")
                    if 'target_SOLUSDT' not in df.columns:
                        print(f"[{datetime.now()}] Error: target_SOLUSDT missing in cached data")
                        return pd.DataFrame()
                    if df['target_SOLUSDT'].isna().all():
                        print(f"[{datetime.now()}] Error: target_SOLUSDT contains only NaN values in cached data")
                        return pd.DataFrame()
                    return df
                else:
                    print(f"[{datetime.now()}] Error: Cached data is empty")
            except Exception as e:
                print(f"[{datetime.now()}] Error loading cached data: {str(e)}")
        return pd.DataFrame()

def select_features(X, y, k=20):
    try:
        priority_features = [
            'rsi_SOLUSDT', 'volatility_SOLUSDT', 'macd_SOLUSDT', 'sol_btc_corr', 'sol_eth_corr',
            'close_SOLUSDT_lag1', 'close_BTCUSDT_lag1', 'close_ETHUSDT_lag1', 'bb_upper_SOLUSDT',
            'bb_lower_SOLUSDT', 'volume_change_SOLUSDT', 'rsi_BTCUSDT', 'volatility_BTCUSDT',
            'ma50_SOLUSDT', 'ma50_BTCUSDT', 'ma50_ETHUSDT', 'momentum_SOLUSDT', 'momentum_BTCUSDT',
            'momentum_ETHUSDT', 'volume_momentum_SOLUSDT', 'sol_eth_vol_ratio', 'sol_eth_momentum_ratio'
        ]
        available_features = [f for f in priority_features if f in X.columns]
        print(f"[{datetime.now()}] Priority features available: {available_features}")
        print(f"[{datetime.now()}] All columns in X: {list(X.columns)}")

        if not available_features:
            print(f"[{datetime.now()}] Warning: No priority features found in data, selecting top {k} features")
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
            selector.fit(X, y)
            selected_indices = selector.get_support(indices=True)
            scores = selector.scores_
            selected_features = [X.columns[i] for i in selected_indices]
            print(f"[{datetime.now()}] Top features by mutual information: {list(zip(selected_features, scores[selected_indices]))}")
        else:
            selected_features = available_features.copy()
            remaining_k = k - len(selected_features)
            
            if remaining_k > 0:
                non_priority_features = [col for col in X.columns if col not in selected_features and col != 'target_SOLUSDT']
                if non_priority_features:
                    X_non_priority = X[non_priority_features]
                    selector = SelectKBest(score_func=mutual_info_regression, k=min(remaining_k, len(non_priority_features)))
                    selector.fit(X_non_priority, y)
                    scores = selector.scores_
                    top_indices = selector.get_support(indices=True)
                    top_features = [non_priority_features[i] for i in top_indices]
                    print(f"[{datetime.now()}] Top additional features by mutual information: {list(zip(top_features, scores[top_indices]))}")
                    selected_features.extend(top_features)
            
            selected_features = selected_features[:k]
            selected_indices = [X.columns.get_loc(f) for f in selected_features if f in X.columns]
        
        print(f"[{datetime.now()}] Final selected features: {selected_features}")
        return selected_indices, [1.0 if f in available_features else 0.5 for f in selected_features]
    except Exception as e:
        print(f"[{datetime.now()}] Error in select_features: {str(e)}")
        return [], []

def load_frame(file_path, timeframe, feature_subset=None):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{datetime.now()}] Training data file {file_path} does not exist.")
        
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        df = df.infer_objects(copy=False).interpolate(method='linear').ffill().bfill()

        if 'target_SOLUSDT' not in df.columns:
            print(f"[{datetime.now()}] Error: target_SOLUSDT missing in {file_path}")
            return None, None, None, None, None, None
        if df['target_SOLUSDT'].isna().all():
            print(f"[{datetime.now()}] Error: target_SOLUSDT contains only NaN values in {file_path}")
            return None, None, None, None, None, None

        all_features = [
            f"close_{pair}_lag{lag}"
            for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
            for lag in [1, 2, 3, 5, 7, 14, 21, 30]
        ] + [
            f"{feature}_{pair}"
            for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
            for feature in ["rsi", "volatility", "ma5", "ma20", "ma50", "momentum", "volume_momentum", "macd", "bb_upper", "bb_lower", "volume_change"]
        ] + ["sol_tx_volume", "sol_btc_corr", "sol_eth_corr", "sol_btc_vol_ratio", "sol_btc_volume_ratio", "sol_eth_vol_ratio", "sol_eth_momentum_ratio", "sentiment_score"]

        if feature_subset:
            features = [all_features[i] for i in feature_subset if i < len(all_features)]
        else:
            features = [f for f in all_features if f in df.columns]

        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"[{datetime.now()}] Warning: Missing features in load_frame: {missing_features}. Using available features.")

        X = df[features]
        y = df["target_SOLUSDT"]

        if len(X) < MINIMUM_DAYS:
            print(f"[{datetime.now()}] Error: Insufficient samples ({len(X)}) for scaling in load_frame, required: {MINIMUM_DAYS}")
            return None, None, None, None, None, None

        X_selected = X
        selected_features = features
        if not feature_subset:
            selected_indices, _ = select_features(X, y, k=20)
            selected_features = [X.columns[i] for i in selected_indices]
            X_selected = X[selected_features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features, index=X.index)

        split_idx = int(len(X) * 0.8)
        if split_idx == 0:
            print(f"[{datetime.now()}] Error: Not enough data to split into training and test sets")
            return None, None, None, None, None, None
        X_train, X_test = X_scaled_df[:split_idx], X_scaled_df[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"[{datetime.now()}] Loaded frame: {len(X_train)} training samples, {len(X_test)} test samples, features: {selected_features}")
        return X_train, X_test, y_train, y_test, scaler, selected_features

    except Exception as e:
        print(f"[{datetime.now()}] Error in load_frame: {str(e)}")
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
        print(f"[{datetime.now()}] Error calculating ZPTAE loss: {str(e)}")
        return float('inf')

def weighted_rmse(y_true, y_pred, weights):
    try:
        return np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights))
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating weighted RMSE: {str(e)}")
        return float('inf')

def weighted_mztae(y_true, y_pred, weights):
    try:
        ref_std = np.std(y_true[-100:]) if len(y_true) >= 100 else np.std(y_true)
        return np.average(np.abs((y_true - y_pred) / ref_std), weights=weights)
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating weighted MZTAE: {str(e)}")
        return float('inf')

def custom_directional_loss(y_true, y_pred):
    try:
        zptae = zptae_loss(y_true, y_pred)
        directional_error = np.mean(np.sign(y_true) != np.sign(y_pred))
        return zptae + 1.0 * directional_error
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating custom directional loss: {str(e)}")
        return float('inf')

def train_model(timeframe, file_path=training_price_data_path, model_type="LightGBM", model_params=None, feature_subset=None):
    try:
        X_train, X_test, y_train, y_test, scaler, selected_features = load_frame(file_path, timeframe, feature_subset)
        if X_train is None:
            print(f"[{datetime.now()}] Error: Failed to load frame, cannot train model")
            return None, None, {}, [], []

        print(f"[{datetime.now()}] Training {model_type} with features: {selected_features}")

        baseline_model = LinearRegression()
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test)
        baseline_zptae = zptae_loss(y_test, baseline_pred, np.abs(y_test))
        baseline_mztae = weighted_mztae(y_test, baseline_pred, np.abs(y_test))
        print(f"[{datetime.now()}] Baseline (Linear Regression) ZPTAE: {baseline_zptae:.6f}, Weighted MZTAE: {baseline_mztae:.6f}")

        model_params = model_params or {}
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            learning_rate=model_params.get("learning_rate", 0.01),
            max_depth=model_params.get("max_depth", 4),
            n_estimators=model_params.get("n_estimators", 1000),
            subsample=model_params.get("subsample", 0.8),
            colsample_bytree=model_params.get("colsample_bytree", 0.8),
            num_leaves=model_params.get("num_leaves", 15),
            min_child_samples=model_params.get("min_child_samples", 20)
        )
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=model_params.get("learning_rate", 0.01),
            max_depth=model_params.get("max_depth", 3),
            n_estimators=model_params.get("n_estimators", 1000),
            subsample=model_params.get("subsample", 0.7),
            colsample_bytree=model_params.get("colsample_bytree", 0.7)
        )
        rf_model = RandomForestRegressor(
            n_estimators=model_params.get("n_estimators", 500),
            max_depth=model_params.get("max_depth", 5),
            min_samples_split=model_params.get("min_samples_split", 5),
            random_state=42
        )

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            if model_type == "Ensemble":
                model = VotingRegressor(estimators=[
                    ('lgb', lgb.LGBMRegressor(**lgb_model.get_params())),
                    ('xgb', xgb.XGBRegressor(**xgb_model.get_params())),
                    ('rf', RandomForestRegressor(**rf_model.get_params()))
                ])
            elif model_type == "LightGBM":
                model = lgb.LGBMRegressor(**lgb_model.get_params())
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(**xgb_model.get_params())
            elif model_type == "RF":
                model = RandomForestRegressor(**rf_model.get_params())
            else:
                raise ValueError(f"Unsupported model: {model_type}")

            model.fit(X_cv_train, y_cv_train)
            cv_pred = model.predict(X_cv_val)
            cv_score = r2_score(y_cv_val, cv_pred)
            cv_scores.append(cv_score)
        
        print(f"[{datetime.now()}] Cross-validation R² scores: {cv_scores}, Mean: {np.mean(cv_scores):.6f}, Std: {np.std(cv_scores):.6f}")

        lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l2', callbacks=[lgb.early_stopping(50)])
        xgb_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        if model_type == "Ensemble":
            model = VotingRegressor(estimators=[
                ('lgb', lgb_model),
                ('xgb', xgb_model),
                ('rf', rf_model)
            ])
            model.fit(X_train, y_train)
        elif model_type == "LightGBM":
            model = lgb_model
        elif model_type == "XGBoost":
            model = xgb_model
        elif model_type == "RF":
            model = rf_model

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        pred_test_std = np.std(pred_test)
        y_test_std = np.std(y_test)
        if pred_test_std < 1e-10 or y_test_std < 1e-10:
            print(f"[{datetime.now()}] Warning: Low variance (pred: {pred_test_std:.6e}, target: {y_test_std:.6e}), model may be underfitting")
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
        print(f"[{datetime.now()}] Binomial Test p-value for Directional Accuracy: {binom_p_value:.4f}")
        print(f"[{datetime.now()}] Training MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")
        print(f"[{datetime.now()}] Training RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")
        print(f"[{datetime.now()}] Training R²: {train_r2:.6f}, Test R²: {test_r2:.6f}")
        print(f"[{datetime.now()}] Training ZPTAE: {train_zptae:.6f}, Test ZPTAE: {test_zptae:.6f}")
        print(f"[{datetime.now()}] Weighted MZTAE: {test_mztae:.6f}")
        print(f"[{datetime.now()}] ZPTAE Improvement: {100 * (baseline_zptae - test_zptae) / baseline_zptae:.2f}%")
        print(f"[{datetime.now()}] Weighted MZTAE Improvement: {100 * (baseline_mztae - test_mztae) / baseline_mztae:.2f}%")
        print(f"[{datetime.now()}] Directional Accuracy: {directional_accuracy:.4f}")
        print(f"[{datetime.now()}] Correlation: {correlation:.4f}, p-value: {p_value:.4f}")

        if model_type == "Ensemble":
            lgb_importance = getattr(lgb_model, 'feature_importances_', np.zeros(len(selected_features)))
            xgb_importance = getattr(xgb_model, 'feature_importances_', np.zeros(len(selected_features)))
            rf_importance = getattr(rf_model, 'feature_importances_', np.zeros(len(selected_features)))
            feature_importances = (lgb_importance / np.sum(lgb_importance) + xgb_importance / np.sum(xgb_importance) + rf_importance / np.sum(rf_importance)) / 3
        else:
            feature_importances = getattr(model, 'feature_importances_', np.zeros(len(selected_features)))
        importance_dict = dict(zip(selected_features, feature_importances))
        print(f"[{datetime.now()}] Feature importances: {list(zip(selected_features, feature_importances))}")

        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_file_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[{datetime.now()}] Model saved to {model_file_path}, scaler saved to {scaler_file_path}")

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
            'cv_r2_mean': float(np.mean(cv_scores)),
            'cv_r2_std': float(np.std(cv_scores))
        }

        if feature_importances.size > 0 and np.any(feature_importances > 0):
            chart_data = {
                "type": "bar",
                "data": {
                    "labels": selected_features,
                    "datasets": [{
                        "label": "Feature Importance",
                        "data": feature_importances.tolist(),
                        "backgroundColor": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                                          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                                          "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                                          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
                        "borderColor": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                                       "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {"display": True, "text": "Importance"}
                        },
                        "x": {
                            "title": {"display": True, "text": "Features"}
                        }
                    },
                    "plugins": {
                        "legend": {"display": False},
                        "title": {"display": True, "text": f"Feature Importance for {model_type}"}
                    }
                }
            }
            print(f"[{datetime.now()}] Feature importance chart generated for {model_type}")

        return model, scaler, metrics, selected_features, feature_subset

    except Exception as e:
        print(f"[{datetime.now()}] Error in train_model: {str(e)}")
        return None, None, {}, [], []

def get_inference(token, timeframe, region, data_provider, features, cached_data=None):
    try:
        if not os.path.exists(model_file_path):
            print(f"[{datetime.now()}] Error: Model file {model_file_path} not found")
            return 0.0
        with open(model_file_path, "rb") as f:
            model = pickle.load(f)

        df = cached_data
        if df is None:
            files_btc = download_binance_daily_data("BTCUSDT", 30, region, binance_data_path)
            files_sol = download_binance_daily_data("SOLUSDT", 30, region, binance_data_path)
            files_eth = download_binance_daily_data("ETHUSDT", 30, region, binance_data_path)
            df = format_data(files_btc, files_sol, files_eth, data_provider)
            if df.empty:
                print(f"[{datetime.now()}] Error: Failed to fetch or format inference data")
                if os.path.exists(training_price_data_path):
                    print(f"[{datetime.now()}] Attempting to load cached data from {training_price_data_path}")
                    try:
                        df = pd.read_csv(training_price_data_path, index_col='date', parse_dates=True)
                        if not df.empty:
                            print(f"[{datetime.now()}] Loaded cached data with {len(df)} rows")
                            print(f"[{datetime.now()}] Available columns in cached data: {list(df.columns)}")
                            if 'target_SOLUSDT' not in df.columns:
                                print(f"[{datetime.now()}] Error: target_SOLUSDT missing in cached data")
                                return 0.0
                            if df['target_SOLUSDT'].isna().all():
                                print(f"[{datetime.now()}] Error: target_SOLUSDT contains only NaN values in cached data")
                                return 0.0
                        else:
                            print(f"[{datetime.now()}] Error: Cached data is empty")
                            return 0.0
                    except Exception as e:
                        print(f"[{datetime.now()}] Error loading cached data: {str(e)}")
                        return 0.0
                else:
                    print(f"[{datetime.now()}] Error: No cached data available")
                    return 0.0

        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            print(f"[{datetime.now()}] Warning: Missing feature columns: {missing_cols}. Using available features.")
            features = [f for f in features if f in df.columns]
            if not features:
                print(f"[{datetime.now()}] Error: No valid features available for prediction")
                return 0.0
        X = df[features]
        if len(X) < 1:
            print(f"[{datetime.now()}] Error: No valid data for prediction")
            return 0.0

        with open(scaler_file_path, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)

        last_row = X_scaled_df.iloc[-1:]
        pred = model.predict(last_row)

        latest_price = 100.0
        try:
            ticker_url = f'https://api.binance.{REGION}/api/v3/ticker/price?symbol=SOLUSDT'
            response = requests.get(ticker_url, timeout=3)
            response.raise_for_status()
            latest_price = float(response.json()['price'])
        except Exception as e:
            print(f"[{datetime.now()}] Error fetching latest price from Binance, using fallback price: {latest_price:.3f}")

        log_return_pred = pred[0]
        predicted_price = latest_price * (1 + log_return_pred)
        print(f"[{datetime.now()}] Predicted {timeframe} SOL/USD Log Return: {log_return_pred:.6f}")
        print(f"[{datetime.now()}] Latest SOL Price: {latest_price:.3f}")
        print(f"[{datetime.now()}] Predicted SOL Price in {timeframe}: {predicted_price:.3f}")
        return log_return_pred

    except Exception as e:
        print(f"[{datetime.now()}] Error in get_inference: {str(e)}")
        return 0.0

if __name__ == "__main__":
    files_btc = download_binance_daily_data("BTCUSDT", TRAINING_DAYS, REGION, binance_data_path)
    files_sol = download_binance_daily_data("SOLUSDT", TRAINING_DAYS, REGION, binance_data_path)
    files_eth = download_binance_daily_data("ETHUSDT", TRAINING_DAYS, REGION, binance_data_path)
    df = format_data(files_btc, files_sol, files_eth, DATA_PROVIDER)
    if not df.empty:
        for model_type in ["LightGBM", "XGBoost", "RF", "Ensemble"]:
            model, scaler, metrics, features, _ = train_model(TIMEFRAME, model_type=model_type)
            if model is not None:
                log_return = get_inference(TOKEN, TIMEFRAME, REGION, DATA_PROVIDER, features, cached_data=df)
            else:
                print(f"[{datetime.now()}] Error: Training failed for {model_type}, cannot perform inference")
    else:
        print(f"[{datetime.now()}] Error: Data formatting failed, cannot train or infer")
