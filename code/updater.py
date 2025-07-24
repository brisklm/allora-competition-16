import os
import pandas as pd
import requests
from datetime import datetime, timedelta

def download_binance_daily_data(symbol, days, region, save_path):
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=int(days))
        files = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            url = f"https://api.binance.{region}/api/v3/klines?symbol={symbol}&interval=1d&startTime={int(current_date.timestamp()*1000)}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data, columns=["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd", "ignore"])
            file_path = os.path.join(save_path, f"{symbol}_{date_str}.csv")
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(file_path, index=False)
            files.append(file_path)
            current_date += timedelta(days=1)
        return files
    except Exception as e:
        print(f"[{datetime.now()}] Error in download_binance_daily_data: {str(e)}")
        return []

def download_binance_current_day_data(symbol, region):
    try:
        url = f"https://api.binance.{region}/api/v3/klines?symbol={symbol}&interval=1d&limit=1"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd", "ignore"])
        df["date"] = pd.to_datetime(df["end_time"], unit="ms", utc=True)
        return df[["date", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"[{datetime.now()}] Error in download_binance_current_day_data: {str(e)}")
        return pd.DataFrame()

def download_coingecko_data(token, days, save_path, api_key):
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=int(days))
        files = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            url = f"https://api.coingecko.com/api/v3/coins/{token}/history?date={current_date.strftime('%d-%m-%Y')}&localization=false&x_cg_api_key={api_key}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame([{
                "date": date_str,
                "open": data.get("market_data", {}).get("current_price", {}).get("usd", 0),
                "high": data.get("market_data", {}).get("high_24h", {}).get("usd", 0),
                "low": data.get("market_data", {}).get("low_24h", {}).get("usd", 0),
                "close": data.get("market_data", {}).get("current_price", {}).get("usd", 0),
                "volume": data.get("market_data", {}).get("total_volume", {}).get("usd", 0)
            }])
            file_path = os.path.join(save_path, f"{token}_{date_str}.csv")
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(file_path, index=False)
            files.append(file_path)
            current_date += timedelta(days=1)
        return files
    except Exception as e:
        print(f"[{datetime.now()}] Error in download_coingecko_data: {str(e)}")
        return []

def download_coingecko_current_day_data(token, api_key):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{token}/market_chart?vs_currency=usd&days=1&interval=daily&x_cg_api_key={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 0  # Placeholder, as volume may not be available
        return df[["date", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"[{datetime.now()}] Error in download_coingecko_current_day_data: {str(e)}")
        return pd.DataFrame()
