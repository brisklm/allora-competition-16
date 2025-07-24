import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import data_base_path, TOKEN, TRAINING_DAYS, DATA_PROVIDER
from model import calculate_rsi, calculate_volatility, calculate_ma, calculate_macd, calculate_bollinger_bands, calculate_cross_asset_correlation, calculate_volume_change

coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")
features_path = os.path.join(data_base_path, "features_sol.csv")

def generate_synthetic_data(symbols, days=180):
    os.makedirs(coingecko_data_path, exist_ok=True)
    start_date = datetime(2025, 7, 22)
    data = {symbol: [] for symbol in symbols}
    np.random.seed(42)
    for i in range(days):
        date = (start_date - timedelta(days=i)).strftime("%Y-%m-%d")
        for symbol in symbols:
            base_price = 100 if "SOL" in symbol else (50000 if "BTC" in symbol else 2000)
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
    return files

def main():
    symbols = ["BTC", "SOL", "ETH"]
    files = {symbol: generate_synthetic_data([symbol], days=180) for symbol in symbols}
    
    dfs = []
    for symbol in symbols:
        for file in files[symbol]:
            df = pd.read_csv(file)
            df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True).dt.floor('D')
            df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
            df.set_index("date", inplace=True)
            df = df.rename(columns=lambda x: f"{x}_{symbol}USDT")
            dfs.append(df)
    
    price_df = pd.concat(dfs, axis=1)
    price_df = price_df.resample('1d', closed='right', label='right').agg({
        f"{metric}_{pair}": "last"
        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
        for metric in ["open", "high", "low", "close", "volume"]
        if f"{metric}_{pair}" in price_df.columns
    })
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
            price_df[f"macd_{pair}"] = calculate_macd(price_df[f"close_{pair}"])
            price_df[f"bb_upper_{pair}"], price_df[f"bb_lower_{pair}"] = calculate_bollinger_bands(price_df[f"close_{pair}"])
            price_df[f"volume_change_{pair}"] = calculate_volume_change(price_df[f"volume_{pair}"])

    price_df["sol_btc_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_BTCUSDT", window=10) if "close_BTCUSDT" in price_df.columns else pd.Series(0, index=price_df.index)
    price_df["sol_eth_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_ETHUSDT", window=10) if "close_ETHUSDT" in price_df.columns else pd.Series(0, index=price_df.index)
    price_df["sol_btc_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_BTCUSDT"] + 1e-10) if "volatility_BTCUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
    price_df["sol_btc_volume_ratio"] = price_df["volume_change_SOLUSDT"] / (price_df["volume_change_BTCUSDT"] + 1e-10) if "volume_change_BTCUSDT" in price_df.columns else pd.Series(1, index=price_df.index)
    price_df["sol_tx_volume"] = np.random.normal(1000, 200, len(price_df))
    price_df["sentiment_score"] = np.random.normal(0, 0.2, len(price_df))
    price_df["target_SOLUSDT"] = price_df["log_return_SOLUSDT"]

    price_df = price_df.fillna(0).infer_objects(copy=False)
    print(f"[{datetime.now()}] Feature statistics: { {col: {'mean': float(price_df[col].mean()), 'std': float(price_df[col].std())} for col in price_df.columns if col != 'target_SOLUSDT'} }")
    price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d')
    print(f"[{datetime.now()}] Data saved to {training_price_data_path}, rows: {len(price_df)}")
    price_df[price_df.columns.difference(['target_SOLUSDT'])].to_csv(features_path, date_format='%Y-%m-%d')
    print(f"[{datetime.now()}] Features saved to {features_path}, rows: {len(price_df)}")

if __name__ == "__main__":
    main()
