"""
Data fetching using CCXT (108+ exchanges) + CoinGecko.
Auto-fallback: Binance → Bybit → OKX → KuCoin → Gate → CoinGecko.
Plus: Fear & Greed Index, Funding Rates via futures exchanges.
No API keys needed for public market data.
"""

import time
import ccxt
import requests
import pandas as pd
import numpy as np
import streamlit as st
from config import COINGECKO_BASE_URL, KLINE_LIMIT, COINGECKO_ID_MAP


# ============================================================
# EXCHANGE PRIORITY — SPOT (tried in order, first working wins)
# ============================================================
EXCHANGE_PRIORITY = [
    ("binance", ccxt.binance),
    ("bybit", ccxt.bybit),
    ("okx", ccxt.okx),
    ("kucoin", ccxt.kucoin),
    ("gate", ccxt.gateio),
    ("kraken", ccxt.kraken),
]

# FUTURES exchanges for Funding Rate (following NR7 scanner pattern)
FUTURES_EXCHANGE_PRIORITY = [
    ("bitget", ccxt.bitget),
    ("bybit", ccxt.bybit),
    ("okx", ccxt.okx),
    ("binance", ccxt.binance),
]

# Timeframe mapping
TF_MAP = {
    "1h": "1h", "4h": "4h", "1d": "1d", "1D": "1d", "1w": "1w", "1W": "1w",
}


# ============================================================
# EXCHANGE CONNECTION — SPOT (cached, auto-fallback)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_working_exchange() -> tuple:
    for name, exchange_cls in EXCHANGE_PRIORITY:
        try:
            ex = exchange_cls({
                "enableRateLimit": True,
                "timeout": 10000,
                "options": {"defaultType": "spot"},
            })
            ex.fetch_ticker("BTC/USDT")
            return (name, ex)
        except Exception:
            continue
    return ("none", None)


def get_exchange_status() -> dict:
    name, ex = get_working_exchange()
    return {"active_exchange": name, "connected": ex is not None}


# ============================================================
# EXCHANGE CONNECTION — FUTURES (for Funding Rates)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_futures_exchange() -> tuple:
    """Find the first reachable futures exchange for funding rate data.
    Uses shorter timeout to avoid blocking main app."""
    for name, exchange_cls in FUTURES_EXCHANGE_PRIORITY:
        try:
            ex = exchange_cls({
                "enableRateLimit": True,
                "timeout": 5000,  # shorter timeout — don't block main app
                "options": {"defaultType": "swap"},
            })
            ex.load_markets()
            return (name, ex)
        except Exception:
            continue
    return ("none", None)


# ============================================================
# CCXT DATA FETCHING — SPOT
# ============================================================

def fetch_ohlcv_ccxt(symbol: str, timeframe: str, limit: int = KLINE_LIMIT) -> pd.DataFrame:
    name, ex = get_working_exchange()
    if ex is None:
        return pd.DataFrame()
    pair = f"{symbol}/USDT"
    tf = TF_MAP.get(timeframe, timeframe)
    try:
        ohlcv = ex.fetch_ohlcv(pair, tf, limit=limit)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["open_time"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["quote_volume"] = df["volume"] * df["close"]
        df["trades"] = 0
        return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
    except Exception:
        return pd.DataFrame()


def fetch_ticker_ccxt(symbol: str) -> dict:
    name, ex = get_working_exchange()
    if ex is None:
        return {}
    pair = f"{symbol}/USDT"
    try:
        ticker = ex.fetch_ticker(pair)
        return {
            "last": ticker.get("last", 0),
            "change_pct": ticker.get("percentage", 0),
            "volume": ticker.get("quoteVolume", 0) or ticker.get("baseVolume", 0) * (ticker.get("last", 1) or 1),
            "high": ticker.get("high", 0),
            "low": ticker.get("low", 0),
        }
    except Exception:
        return {}


def fetch_all_tickers_ccxt() -> dict:
    name, ex = get_working_exchange()
    if ex is None:
        return {}
    try:
        if ex.has.get("fetchTickers", False):
            all_tickers = ex.fetch_tickers()
            result = {}
            for pair, ticker in all_tickers.items():
                if pair.endswith("/USDT"):
                    sym = pair.replace("/USDT", "")
                    result[sym] = {
                        "last": ticker.get("last", 0),
                        "change_pct": ticker.get("percentage", 0),
                        "volume": ticker.get("quoteVolume", 0) or 0,
                        "high": ticker.get("high", 0),
                        "low": ticker.get("low", 0),
                    }
            return result
    except Exception:
        pass
    return {}


# ============================================================
# COINGECKO FALLBACK
# ============================================================

def _get_coingecko_id(symbol: str) -> str:
    return COINGECKO_ID_MAP.get(symbol, symbol.lower())


def get_coingecko_market_data(page: int = 1, per_page: int = 250) -> list:
    url = f"{COINGECKO_BASE_URL}/coins/markets"
    params = {
        "vs_currency": "usd", "order": "market_cap_desc",
        "per_page": per_page, "page": page, "sparkline": False,
        "price_change_percentage": "1h,24h,7d,30d",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            time.sleep(10)
    except Exception:
        pass
    return []


def get_coingecko_ohlc(symbol: str, days: int = 7) -> pd.DataFrame:
    cg_id = _get_coingecko_id(symbol)
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
        df["volume"] = (df["high"] - df["low"]) / df["close"] * 1e6
        df["quote_volume"] = df["volume"] * df["close"]
        df["trades"] = 0
        return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
    except Exception:
        return pd.DataFrame()


# ============================================================
# FEAR & GREED INDEX (alternative.me API)
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_fear_greed_index() -> dict:
    """
    Fetch current Fear & Greed Index from alternative.me.
    Returns: {"value": 0-100, "label": "Extreme Fear"|"Fear"|"Neutral"|"Greed"|"Extreme Greed"}
    """
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and "data" in data and len(data["data"]) > 0:
                entry = data["data"][0]
                return {
                    "value": int(entry.get("value", 50)),
                    "label": entry.get("value_classification", "Neutral"),
                }
    except Exception:
        pass
    return {"value": None, "label": "N/A"}


# ============================================================
# FUNDING RATES (via CCXT Futures exchanges)
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_funding_rates(symbols: list) -> dict:
    """
    Fetch current funding rates for given symbols from futures exchanges.
    Auto-fallback: Bitget → Bybit → OKX → Binance
    Returns: {"BTC": 0.0001, "ETH": -0.0002, ...}
    """
    fname, fex = get_futures_exchange()
    if fex is None:
        return {}

    result = {}
    for sym in symbols:
        try:
            # CCXT uses different pair formats per exchange for perps
            # Common formats: BTC/USDT:USDT, BTCUSDT
            pair_candidates = [
                f"{sym}/USDT:USDT",   # bybit, bitget, okx
                f"{sym}/USDT",         # some exchanges
            ]
            for pair in pair_candidates:
                try:
                    if hasattr(fex, 'fetch_funding_rate'):
                        fr = fex.fetch_funding_rate(pair)
                        rate = fr.get("fundingRate", None)
                        if rate is not None:
                            result[sym] = float(rate)
                            break
                except Exception:
                    continue
        except Exception:
            continue

    return result


@st.cache_data(ttl=300, show_spinner=False)
def fetch_funding_rates_batch() -> dict:
    """
    Fetch all funding rates at once (more efficient).
    Returns: {"BTC": 0.0001, ...}
    """
    fname, fex = get_futures_exchange()
    if fex is None:
        return {}

    try:
        if hasattr(fex, 'fetch_funding_rates'):
            all_rates = fex.fetch_funding_rates()
            result = {}
            for pair, data in all_rates.items():
                # Extract symbol from pair
                sym = pair.split("/")[0] if "/" in pair else pair.replace("USDT", "").replace(":USDT", "")
                rate = data.get("fundingRate", None)
                if rate is not None:
                    result[sym] = float(rate)
            return result
    except Exception:
        pass

    return {}


# ============================================================
# SMART FETCHING (CCXT → CoinGecko fallback)
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_klines_smart(symbol: str, interval: str) -> pd.DataFrame:
    df = fetch_ohlcv_ccxt(symbol, interval)
    if not df.empty and len(df) >= 15:
        return df
    interval_to_days = {"1h": 2, "4h": 7, "1d": 30, "1D": 30, "1w": 180, "1W": 180}
    days = interval_to_days.get(interval, 7)
    df = get_coingecko_ohlc(symbol, days=days)
    if not df.empty:
        return df
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_market_data() -> pd.DataFrame:
    all_data = []
    for page in [1, 2]:
        data = get_coingecko_market_data(page=page)
        if data:
            all_data.extend(data)
        time.sleep(1.5)
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        "symbol": "symbol_lower", "current_price": "price",
        "market_cap_rank": "rank", "market_cap": "market_cap",
        "total_volume": "volume_24h",
        "price_change_percentage_24h": "change_24h_pct",
        "price_change_percentage_1h_in_currency": "change_1h",
        "price_change_percentage_24h_in_currency": "change_24h",
        "price_change_percentage_7d_in_currency": "change_7d",
        "price_change_percentage_30d_in_currency": "change_30d",
    })
    df["symbol"] = df["symbol_lower"].str.upper()
    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_all_tickers() -> dict:
    return fetch_all_tickers_ccxt()
