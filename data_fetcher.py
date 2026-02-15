"""
Data fetching from Binance and CoinGecko APIs.
Binance: OHLCV klines for technical analysis (fast, free, no auth)
CoinGecko: Market cap, rank, metadata, OHLC fallback (free tier)

Uses Binance as primary kline source, falls back to CoinGecko if unavailable.
"""

import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from config import BINANCE_BASE_URL, COINGECKO_BASE_URL, KLINE_LIMIT, TOP_COINS, COINGECKO_ID_MAP


# ============================================================
# CONNECTIVITY CHECK
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def check_binance_available() -> bool:
    """Check if Binance API is reachable. Cached 10 min."""
    try:
        resp = requests.get(f"{BINANCE_BASE_URL}/ping", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


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
        if not data or isinstance(data, dict):
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


def get_all_binance_tickers() -> dict:
    """Fetch all 24h tickers at once (single API call for all coins)."""
    url = f"{BINANCE_BASE_URL}/ticker/24hr"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                return {
                    item["symbol"].replace("USDT", ""): item
                    for item in data
                    if item["symbol"].endswith("USDT")
                }
    except Exception:
        pass
    return {}


# ============================================================
# COINGECKO API
# ============================================================

def _get_coingecko_id(symbol: str) -> str:
    """Map a symbol to its CoinGecko ID."""
    return COINGECKO_ID_MAP.get(symbol, symbol.lower())


def get_coingecko_market_data(page: int = 1, per_page: int = 250) -> list:
    """Fetch market data from CoinGecko (market cap, rank, prices, changes)."""
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
        elif resp.status_code == 429:
            time.sleep(10)  # Rate limit hit, wait
            return []
    except Exception:
        pass
    return []


def get_coingecko_ohlc(cg_id: str, days: int = 7) -> pd.DataFrame:
    """
    Fetch OHLC data from CoinGecko (free, no auth).
    Returns DataFrame with open, high, low, close columns.
    Granularity: 4h candles for 1-14 days, daily for 15+ days.
    """
    url = f"{COINGECKO_BASE_URL}/coins/{cg_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        if not data or not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["open_time"] = pd.to_datetime(df["timestamp"], unit="ms")

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        # CoinGecko OHLC doesn't include volume — estimate from price range
        df["volume"] = (df["high"] - df["low"]) / df["close"] * 1e6  # proxy
        df["quote_volume"] = df["volume"] * df["close"]
        df["trades"] = 0

        return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]

    except Exception:
        return pd.DataFrame()


def get_coingecko_market_chart(cg_id: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch price history from CoinGecko /market_chart endpoint.
    Returns OHLCV-like DataFrame synthesized from price data.
    Good for 1D+ timeframes.
    """
    url = f"{COINGECKO_BASE_URL}/coins/{cg_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            return pd.DataFrame()

        df_price = pd.DataFrame(prices, columns=["timestamp", "close"])
        df_price["open_time"] = pd.to_datetime(df_price["timestamp"], unit="ms")

        if volumes:
            df_vol = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            df_price = df_price.merge(df_vol, on="timestamp", how="left")
        else:
            df_price["volume"] = 0

        # Synthesize OHLC from close prices
        df_price["open"] = df_price["close"].shift(1).fillna(df_price["close"])
        df_price["high"] = df_price[["open", "close"]].max(axis=1) * 1.001
        df_price["low"] = df_price[["open", "close"]].min(axis=1) * 0.999
        df_price["quote_volume"] = df_price["volume"]
        df_price["trades"] = 0

        return df_price[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]

    except Exception:
        return pd.DataFrame()


# ============================================================
# SMART FETCHING WITH FALLBACK
# ============================================================

def fetch_klines_smart(symbol: str, interval: str, use_binance: bool = True) -> pd.DataFrame:
    """
    Fetch klines with automatic fallback:
    1. Try Binance (if available)
    2. Fall back to CoinGecko OHLC
    3. Fall back to CoinGecko market_chart
    """
    # --- Try Binance first ---
    if use_binance:
        df = get_binance_klines(symbol, interval)
        if not df.empty:
            return df

    # --- CoinGecko fallback ---
    cg_id = _get_coingecko_id(symbol)

    # Map interval to CoinGecko days parameter
    interval_to_days = {
        "1h": 2,    # 2 days of hourly-ish data
        "4h": 7,    # 7 days = 4h candles
        "1d": 30,   # 30 days = daily candles
        "1w": 180,  # 180 days for weekly view
    }
    days = interval_to_days.get(interval, 7)

    # Try OHLC endpoint (better for 4h)
    df = get_coingecko_ohlc(cg_id, days=days)
    if not df.empty and len(df) >= 15:
        return df

    # Try market_chart endpoint (better for 1D/1W)
    df = get_coingecko_market_chart(cg_id, days=max(days, 30))
    if not df.empty:
        return df

    return pd.DataFrame()


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
        time.sleep(1.5)  # Rate limit respect

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        "symbol": "symbol_lower",
        "current_price": "price",
        "market_cap_rank": "rank",
        "market_cap": "market_cap",
        "total_volume": "volume_24h",
        "price_change_percentage_24h": "change_24h_pct",
        "price_change_percentage_1h_in_currency": "change_1h",
        "price_change_percentage_24h_in_currency": "change_24h",
        "price_change_percentage_7d_in_currency": "change_7d",
        "price_change_percentage_30d_in_currency": "change_30d",
    })
    df["symbol"] = df["symbol_lower"].str.upper()

    return df


@st.cache_data(ttl=120, show_spinner=False)
def fetch_klines_cached(symbol: str, interval: str, use_binance: bool = True) -> pd.DataFrame:
    """Fetch and cache klines for a single coin/interval. Cached 2 min."""
    return fetch_klines_smart(symbol, interval, use_binance)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_all_tickers() -> dict:
    """Fetch all Binance 24h tickers. Cached 1 min."""
    return get_all_binance_tickers()


def fetch_multi_timeframe_klines(symbol: str, timeframes: dict, use_binance: bool = True) -> dict:
    """Fetch klines for multiple timeframes for a single coin."""
    result = {}
    for tf_name, tf_interval in timeframes.items():
        df = fetch_klines_cached(symbol, tf_interval, use_binance)
        if not df.empty:
            result[tf_name] = df
    return result
