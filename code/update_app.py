import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
try:
    from model import calculate_rsi, calculate_ma, calculate_macd, calculate_bollinger_bands, calculate_volatility, calculate_cross_asset_correlation, calculate_volume_change, fetch_solana_onchain_data, fetch_x_sentiment, calculate_garch_vol
    from config import data_base_path, training_price_data_path, TIMEFRAME, DATA_PROVIDER, TOKEN, TRAINING_DAYS
except ImportError as e:
    print(f"[{datetime.now()}] ImportError: {str(e)}. Ensure model.py and config.py are available.")
    raise

load_dotenv()
UPDATE_INTERVAL = os.getenv("UPDATE_INTERVAL", "30m")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
features_path = os.path.join(data_base_path, "features_sol.csv")

def parse_interval(interval):
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    elif unit == 'd':
        return value * 86400
    else:
        raise ValueError(f"Invalid interval unit: {unit}")

def generate_synthetic_data(symbols, days=180):
    os.makedirs(coingecko_data_path, exist_ok=True)
    start_date = datetime.utcnow()
    data = {symbol: [] for symbol in symbols}
    np.random.seed(42)
    for i in range(days):
        date = (start_date - timedelta(days=i)).strftime("%Y-%m-%d")
        for symbol in symbols:
            base_price = 185 if "SOL" in symbol else (95000 if "BTC" in symbol else 2800)
            price_noise = np.random.normal(0, 0.05) + i * 0.001
            open_price = base_price * (1 + price_noise)
            high = open_price * (1 + np.random.uniform(0.02, 0.06))
            low = open_price * (1 - np.random.uniform(0.02, 0.06))
            close = open_price * (1 + np.random.normal(0, 0.025))
            volume = 1000 * (1 + np.random.uniform(0, 1.5))
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
        file_path = os.path.join(coingecko_data_path, f"{symbol}_{start_date.strftime('%Y-%m-%d')}_synthetic.csv")
        df.to_csv(file_path, index=False)
        files.append(file_path)
    print(f"[{datetime.now()}] Generated {len(files)} synthetic files for {symbols}")
    return files

def calculate_technical_indicators(data, pair):
    try:
        close = data[f"close_{pair}"]
        volume = data[f"volume_{pair}"]
        high = data[f"high_{pair}"]
        low = data[f"low_{pair}"]
        print(f"[{datetime.now()}] Calculating indicators for {pair}")
        data[f"rsi_{pair}"] = calculate_rsi(close, periods=14).fillna(0)
        data[f"ma5_{pair}"] = calculate_ma(close, periods=5).fillna(0)
        data[f"ma20_{pair}"] = calculate_ma(close, periods=20).fillna(0)
        data[f"macd_{pair}"] = calculate_macd(close, fast=12, slow=26, signal=9).fillna(0)
        upper, lower = calculate_bollinger_bands(close, periods=20, std_dev=2)
        data[f"bb_upper_{pair}"] = upper.fillna(0)
        data[f"bb_lower_{pair}"] = lower.fillna(0)
        data[f"volume_change_{pair}"] = calculate_volume_change(volume).fillna(0)
        data[f"volatility_{pair}"] = calculate_volatility(close, window=5).fillna(0)
        data[f"momentum_{pair}"] = (close - close.shift(5)).fillna(0)
        data[f"sign_log_return_lag1_{pair}"] = np.sign(np.log(close / close.shift(1)).shift(1)).fillna(0)
        data[f"garch_vol_{pair}"] = calculate_garch_vol(close).fillna(0)
        for lag in [1, 2, 5, 10, 30]:
            data[f"close_{pair}_lag{lag}"] = close.shift(lag)
            data[f"log_return_{pair}_lag{lag}"] = data[f"log_return_{pair}"].shift(lag)
        print(f"[{datetime.now()}] Completed indicators for {pair}")
        return data
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating technical indicators for {pair}: {str(e)}")
        return data

def update_data():
    try:
        print(f"[{datetime.now()}] Updating data at {training_price_data_path}")
        symbols = ["BTCUSDT", "SOLUSDT", "ETHUSDT"]
        
        real_files = {symbol: [] for symbol in symbols}
        if DATA_PROVIDER == "binance":
            from updater import download_binance_daily_data
            for symbol in symbols:
                print(f"[{datetime.now()}] Fetching data for {symbol}")
                real_files[symbol] = download_binance_daily_data(symbol, TRAINING_DAYS, "com", os.path.join(data_base_path, "binance"))
                print(f"[{datetime.now()}] Fetched {len(real_files[symbol])} files for {symbol}")
        
        dfs = []
        real_data_available = False
        for symbol in symbols:
            files = real_files[symbol]
            if files and all(os.path.getsize(f) > 100 for f in files):
                real_data_available = True
                for file in files:
                    print(f"[{datetime.now()}] Processing file {file}")
                    df = pd.read_csv(file)
                    df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                    df.set_index("date", inplace=True)
                    df = df.rename(columns=lambda x: f"{x}_{symbol}")
                    dfs.append(df)
                    print(f"[{datetime.now()}] Processed {file}, rows: {len(df)}")
        
        if not real_data_available:
            print(f"[{datetime.now()}] Real data fetch failed, using synthetic data")
            files = generate_synthetic_data(symbols, days=TRAINING_DAYS)
            for symbol in symbols:
                symbol_files = [f for f in files if symbol in f]
                for file in symbol_files:
                    print(f"[{datetime.now()}] Processing synthetic file {file}")
                    df = pd.read_csv(file)
                    df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
                    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                    df.set_index("date", inplace=True)
                    df = df.rename(columns=lambda x: f"{x}_{symbol}")
                    dfs.append(df)
                    print(f"[{datetime.now()}] Processed synthetic {file}, rows: {len(df)}")
        
        print(f"[{datetime.now()}] Concatenating dataframes")
        price_df = pd.concat(dfs, axis=1)
        print(f"[{datetime.now()}] Resampling data")
        price_df = price_df.resample(TIMEFRAME, closed='right', label='right').agg({
            f"{metric}_{pair}": "last"
            for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
            for metric in ["open", "high", "low", "close", "volume"]
            if f"{metric}_{pair}" in price_df.columns
        })
        print(f"[{datetime.now()}] Interpolating and filling NaNs")
        price_df = price_df.infer_objects(copy=False).interpolate(method='linear').ffill().bfill()
        
        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            if f"close_{pair}" in price_df.columns:
                print(f"[{datetime.now()}] Calculating log returns for {pair}")
                price_df[f"log_return_{pair}"] = price_df[f"close_{pair}"].pct_change().shift(-1)
                price_df = calculate_technical_indicators(price_df, pair)
        
        print(f"[{datetime.now()}] Calculating cross-asset correlations and ratios")
        price_df["sol_btc_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_BTCUSDT", window=10) if "close_BTCUSDT" in price_df.columns else pd.Series(0, index=price_df.index)
        price_df["sol_eth_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_ETHUSDT", window=10) if "close_ETHUSDT" in price_df.columns else pd.Series(0, index=price_df.index)
        price_df["sol_btc_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_BTCUSDT"] + 1e-10) if "volatility_BTCUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_btc_volume_ratio"] = price_df["volume_change_SOLUSDT"] / (price_df["volume_change_BTCUSDT"] + 1e-10) if "volume_change_BTCUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_eth_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_ETHUSDT"] + 1e-10) if "volatility_ETHUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        price_df["sol_eth_momentum_ratio"] = price_df["momentum_SOLUSDT"] / (price_df["momentum_ETHUSDT"] + 1e-10) if "momentum_ETHUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
        
        print(f"[{datetime.now()}] Fetching on-chain and sentiment data")
        tx_volumes = fetch_solana_onchain_data(days=len(price_df))['tx_volume']
        price_df["sol_tx_volume"] = pd.Series(tx_volumes, index=price_df.index[:len(tx_volumes)])
        sentiment_scores = fetch_x_sentiment(days=len(price_df))['sentiment_score']
        price_df["sentiment_score"] = pd.Series(sentiment_scores, index=price_df.index[:len(sentiment_scores)])
        price_df["target_SOLUSDT"] = price_df["log_return_SOLUSDT"]
        
        print(f"[{datetime.now()}] Handling final NaNs")
        price_df = price_df.fillna(0).infer_objects(copy=False)
        feature_stats = {col: {"mean": float(price_df[col].mean()), "std": float(price_df[col].std())} for col in price_df.columns if col != 'target_SOLUSDT'}
        low_variance_features = [col for col, stats in feature_stats.items() if stats["std"] < 1e-5]
        if low_variance_features:
            print(f"[{datetime.now()}] Warning: Low variance features detected: {low_variance_features}")
        print(f"[{datetime.now()}] Feature statistics: {feature_stats}")
        os.makedirs(os.path.dirname(training_price_data_path), exist_ok=True)
        print(f"[{datetime.now()}] Saving data")
        price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d')
        price_df.to_csv(features_path, date_format='%Y-%m-%d')
        print(f"[{datetime.now()}] Data saved to {training_price_data_path}, features saved to {features_path}, rows: {len(price_df)}")
    except Exception as e:
        print(f"[{datetime.now()}] Error updating data: {str(e)}")

if __name__ == "__main__":
    interval_seconds = parse_interval(UPDATE_INTERVAL)
    print(f"[{datetime.now()}] Starting data updater with interval {UPDATE_INTERVAL} ({interval_seconds} seconds)")
    while True:
        update_data()
        time.sleep(interval_seconds)
