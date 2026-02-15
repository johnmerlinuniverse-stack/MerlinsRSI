"""
Data fetching from Binance and CoinGecko APIs.
Binance: OHLCV klines for technical analysis (fast, free, no auth)
CoinGecko: Market cap, rank, metadata (free tier)
"""

import time
import requests
import pandas as pd
import streamlit as st
from config import BINANCE_BASE_URL, COINGECKO_BASE_URL, KLINE_LIMIT, TOP_COINS


# ============================================================
# BINANCE API
# ============================================================

def get_binance_klines(symbol: str, interval: str, limit: int = KLINE_LIMIT) -> pd.DataFrame:
    """Fetch OHLCV klines from Binance for a given symbol and interval."""
    pair = f"{symbol}USDT"
    url = f"{BINANCE_BASE_URL}/klines"
    params = {"symbol": pair, "interval": interval, "limit": limit}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
    
    except Exception:
        return pd.DataFrame()


def get_binance_ticker_24h(symbol: str) -> dict:
    """Get 24h ticker data for a symbol."""
    pair = f"{symbol}USDT"
    url = f"{BINANCE_BASE_URL}/ticker/24hr"
    params = {"symbol": pair}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def get_all_binance_tickers() -> dict:
    """Fetch all 24h tickers at once (single API call for all coins)."""
    url = f"{BINANCE_BASE_URL}/ticker/24hr"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return {
                item["symbol"].replace("USDT", ""): item
                for item in data
                if item["symbol"].endswith("USDT")
            }
    except Exception:
        pass
    return {}


def get_binance_price(symbol: str) -> float:
    """Get current price for a symbol."""
    pair = f"{symbol}USDT"
    url = f"{BINANCE_BASE_URL}/ticker/price"
    params = {"symbol": pair}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return float(resp.json()["price"])
    except Exception:
        pass
    return 0.0


# ============================================================
# COINGECKO API
# ============================================================

def get_coingecko_market_data(page: int = 1, per_page: int = 250) -> list:
    """Fetch market data from CoinGecko (market cap, rank, etc)."""
    url = f"{COINGECKO_BASE_URL}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": False,
        "price_change_percentage": "1h,24h,7d,30d",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


# ============================================================
# BATCH DATA FETCHING
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_market_data() -> pd.DataFrame:
    """
    Fetch market overview data from CoinGecko.
    Returns DataFrame with market cap, rank, price changes.
    Cached for 5 minutes.
    """
    all_data = []
    for page in [1, 2]:
        data = get_coingecko_market_data(page=page)
        if data:
            all_data.extend(data)
        time.sleep(1)  # Rate limit respect
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        "symbol": "symbol_lower",
        "current_price": "price",
        "market_cap_rank": "rank",
        "price_change_percentage_1h_in_currency": "change_1h",
        "price_change_percentage_24h_in_currency": "change_24h",
        "price_change_percentage_7d_in_currency": "change_7d",
        "price_change_percentage_30d_in_currency": "change_30d",
    })
    df["symbol"] = df["symbol_lower"].str.upper()
    
    return df


@st.cache_data(ttl=120, show_spinner=False)
def fetch_klines_for_coin(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch and cache klines for a single coin/interval. Cached 2 min."""
    return get_binance_klines(symbol, interval)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_all_tickers() -> dict:
    """Fetch all Binance 24h tickers. Cached 1 min."""
    return get_all_binance_tickers()


def fetch_multi_timeframe_klines(symbol: str, timeframes: dict) -> dict:
    """Fetch klines for multiple timeframes for a single coin."""
    result = {}
    for tf_name, tf_interval in timeframes.items():
        df = fetch_klines_for_coin(symbol, tf_interval)
        if not df.empty:
            result[tf_name] = df
    return result
