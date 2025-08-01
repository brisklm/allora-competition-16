import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
try:
    from arch import arch_model
except ImportError:
    arch_model = None
try:
    from config import data_base_path
except ImportError:
    data_base_path = os.path.join(os.getcwd(), "data")

def calculate_garch_vol(data):
    try:
        if arch_model:
            data = pd.to_numeric(data, errors='coerce').dropna()
            if len(data) < 10:
                print(f"[{datetime.now()}] Insufficient data for GARCH: {len(data)} points")
                return pd.Series(0, index=data.index)
            pct_changes = data.pct_change().dropna()
            if len(pct_changes) < 10 or np.std(pct_changes) < 1e-6:
                print(f"[{datetime.now()}] Low variance or insufficient valid data for GARCH: {len(pct_changes)} points, std={np.std(pct_changes):.6e}")
                return pd.Series(0, index=data.index)
            try:
                model = arch_model(pct_changes, vol='Garch', p=1, q=1, rescale=False)
                res = model.fit(disp='off')
                return res.conditional_volatility.fillna(0)
            except Exception as e:
                print(f"[{datetime.now()}] GARCH(1,1) failed: {str(e)}, trying GARCH(1,0)")
                model = arch_model(pct_changes, vol='Garch', p=1, q=0, rescale=False)
                res = model.fit(disp='off')
                return res.conditional_volatility.fillna(0)
        return pd.Series(0, index=data.index)
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating GARCH volatility: {str(e)}")
        return pd.Series(0, index=data.index)

def download_binance_daily_data(symbol, days, region, save_path):
    try:
        os.makedirs(save_path, exist_ok=True)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=int(days))
        files = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            url = f"https://api.binance.{region}/api/v3/klines?symbol={symbol}&interval=1d&startTime={int(current_date.timestamp()*1000)}"
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    df = pd.DataFrame(data, columns=["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd", "ignore"])
                    df["date"] = pd.to_datetime(df["end_time"], unit="ms", utc=True)
                    df["close"] = pd.to_numeric(df["close"], errors='coerce')
                    df["garch_vol"] = calculate_garch_vol(df["close"])
                    file_path = os.path.join(save_path, f"{symbol}_{date_str}.csv")
                    df.to_csv(file_path, index=False)
                    files.append(file_path)
                    break
                except Exception as e:
                    print(f"[{datetime.now()}] Attempt {attempt+1}/3 failed for {symbol} on {date_str}: {str(e)}")
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        print(f"[{datetime.now()}] Failed to fetch data for {symbol} on {date_str}")
            current_date += timedelta(days=1)
        print(f"[{datetime.now()}] Successfully saved {len(files)} Binance files for {symbol} to {save_path}")
        return files
    except Exception as e:
        print(f"[{datetime.now()}] Error in download_binance_daily_data: {str(e)}")
        return []

def download_binance_current_day_data(symbol, region):
    try:
        url = f"https://api.binance.{region}/api/v3/klines?symbol={symbol}&interval=1d&limit=1"
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                df = pd.DataFrame(data, columns=["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd", "ignore"])
                df["date"] = pd.to_datetime(df["end_time"], unit="ms", utc=True)
                df["close"] = pd.to_numeric(df["close"], errors='coerce')
                df["garch_vol"] = calculate_garch_vol(df["close"])
                return df[["date", "open", "high", "low", "close", "volume", "garch_vol"]]
            except Exception as e:
                print(f"[{datetime.now()}] Attempt {attempt+1}/3 failed for {symbol}: {str(e)}")
                if attempt < 2:
                    time.sleep(5)
        print(f"[{datetime.now()}] Failed to fetch current day data for {symbol}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[{datetime.now()}] Error in download_binance_current_day_data: {str(e)}")
        return pd.DataFrame()

def download_coingecko_data(token, days, save_path, api_key):
    try:
        os.makedirs(save_path, exist_ok=True)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=int(days))
        files = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            url = f"https://api.coingecko.com/api/v3/coins/{token}/history?date={current_date.strftime('%d-%m-%Y')}&localization=false&x_cg_api_key={api_key}"
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    close_price = data.get("market_data", {}).get("current_price", {}).get("usd", 0)
                    df = pd.DataFrame([{
                        "date": date_str,
                        "open": close_price,
                        "high": data.get("market_data", {}).get("high_24h", {}).get("usd", 0),
                        "low": data.get("market_data", {}).get("low_24h", {}).get("usd", 0),
                        "close": close_price,
                        "volume": data.get("market_data", {}).get("total_volume", {}).get("usd", 0),
                        "garch_vol": calculate_garch_vol(pd.Series([close_price]))
                    }])
                    file_path = os.path.join(save_path, f"{token}_{date_str}.csv")
                    df.to_csv(file_path, index=False)
                    files.append(file_path)
                    break
                except Exception as e:
                    print(f"[{datetime.now()}] Attempt {attempt+1}/3 failed for {token} on {date_str}: {str(e)}")
                    if attempt < 2:
                        time.sleep(5)
            current_date += timedelta(days=1)
        print(f"[{datetime.now()}] Successfully saved {len(files)} CoinGecko files for {token} to {save_path}")
        return files
    except Exception as e:
        print(f"[{datetime.now()}] Error in download_coingecko_data: {str(e)}")
        return []

def download_coingecko_current_day_data(token, api_key):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{token}/market_chart?vs_currency=usd&days=1&interval=daily&x_cg_api_key={api_key}"
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df["open"] = df["close"].shift(1).fillna(df["close"])
                df["high"] = df["close"]
                df["low"] = df["close"]
                df["volume"] = 0
                df["close"] = pd.to_numeric(df["close"], errors='coerce')
                df["garch_vol"] = calculate_garch_vol(df["close"])
                return df[["date", "open", "high", "low", "close", "volume", "garch_vol"]]
            except Exception as e:
                print(f"[{datetime.now()}] Attempt {attempt+1}/3 failed for {token}: {str(e)}")
                if attempt < 2:
                    time.sleep(5)
        print(f"[{datetime.now()}] Failed to fetch current day data for {token}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[{datetime.now()}] Error in download_coingecko_current_day_data: {str(e)}")
        return pd.DataFrame()
