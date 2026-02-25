"""
Data fetching using CCXT (108+ exchanges) + CoinGecko.
Auto-fallback: Binance → Bybit → OKX → KuCoin → Gate → CoinGecko.
No API keys needed for public market data.

Additional lazy endpoints (only called from Confluence tab):
  - Fear & Greed Index (alternative.me)
  - Funding Rates (Binance Futures → OKX → Bitget → Bybit)
"""

import time
import ccxt
import requests
import pandas as pd
import numpy as np
import streamlit as st
from config import COINGECKO_BASE_URL, KLINE_LIMIT, COINGECKO_ID_MAP


# ============================================================
# EXCHANGE PRIORITY (tried in order, first working wins)
# ============================================================
EXCHANGE_PRIORITY = [
    ("binance", ccxt.binance),
    ("bybit", ccxt.bybit),
    ("okx", ccxt.okx),
    ("kucoin", ccxt.kucoin),
    ("gate", ccxt.gateio),
    ("kraken", ccxt.kraken),
]

# Timeframe mapping (our labels → ccxt format)
TF_MAP = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1D": "1d",
    "1w": "1w",
    "1W": "1w",
}


# ============================================================
# EXCHANGE CONNECTION (cached, auto-fallback)
# ============================================================

@st.cache_resource(show_spinner=False)
def _init_exchanges() -> list:
    """
    Initialize ALL reachable exchanges upfront.
    Returns list of (name, exchange_object) tuples.
    Cached for the entire session.
    """
    available = []
    for name, exchange_cls in EXCHANGE_PRIORITY:
        try:
            ex = exchange_cls({
                "enableRateLimit": True,
                "timeout": 10000,
                "options": {"defaultType": "spot"},
            })
            # Quick connectivity test
            ex.fetch_ticker("BTC/USDT")
            available.append((name, ex))
        except Exception:
            continue
    return available


def get_working_exchange() -> tuple:
    """Get the PRIMARY exchange (first reachable). For backward compat."""
    exchanges = _init_exchanges()
    if exchanges:
        return exchanges[0]
    return ("none", None)


def get_exchange_status() -> dict:
    """Check which exchanges are reachable."""
    exchanges = _init_exchanges()
    if exchanges:
        names = [n for n, _ in exchanges]
        return {
            "active_exchange": names[0],
            "connected": True,
            "all_exchanges": names,
        }
    return {"active_exchange": "none", "connected": False, "all_exchanges": []}


# Per-coin exchange cache: tracks which exchange works for each symbol
# This avoids retrying dead exchanges on every scan
_coin_exchange_cache = {}  # symbol → exchange_name


# ============================================================
# CCXT DATA FETCHING (multi-exchange fallback per coin)
# ============================================================

def fetch_ohlcv_ccxt(symbol: str, timeframe: str, limit: int = KLINE_LIMIT) -> pd.DataFrame:
    """
    Fetch OHLCV data, trying multiple exchanges per coin.
    Order: cached exchange for this coin → primary → all others.
    """
    exchanges = _init_exchanges()
    if not exchanges:
        return pd.DataFrame()

    pair = f"{symbol}/USDT"
    tf = TF_MAP.get(timeframe, timeframe)

    # Build exchange order: cached winner first, then primary, then rest
    cached_ex_name = _coin_exchange_cache.get(symbol)
    ordered = []
    if cached_ex_name:
        for n, ex in exchanges:
            if n == cached_ex_name:
                ordered.append((n, ex))
                break
    for n, ex in exchanges:
        if n != cached_ex_name:
            ordered.append((n, ex))

    for name, ex in ordered:
        try:
            ohlcv = ex.fetch_ohlcv(pair, tf, limit=limit)
            if not ohlcv:
                continue

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["open_time"] = pd.to_datetime(df["timestamp"], unit="ms")

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df["quote_volume"] = df["volume"] * df["close"]
            df["trades"] = 0

            # Remember which exchange worked for this coin
            _coin_exchange_cache[symbol] = name
            return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]

        except Exception:
            continue

    return pd.DataFrame()


def fetch_ticker_ccxt(symbol: str) -> dict:
    """Fetch current ticker data for a symbol."""
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
    """Fetch all tickers from ALL available exchanges (not just primary).
    Result is merged: primary exchange wins, others fill gaps."""
    exchanges = _init_exchanges()
    if not exchanges:
        return {}

    result = {}
    for name, ex in exchanges:
        try:
            if ex.has.get("fetchTickers", False):
                all_tickers = ex.fetch_tickers()
                for pair, ticker in all_tickers.items():
                    if pair.endswith("/USDT"):
                        sym = pair.replace("/USDT", "")
                        if sym not in result:  # First exchange wins (primary has priority)
                            result[sym] = {
                                "last": ticker.get("last", 0),
                                "change_pct": ticker.get("percentage", 0),
                                "volume": ticker.get("quoteVolume", 0) or 0,
                                "high": ticker.get("high", 0),
                                "low": ticker.get("low", 0),
                            }
        except Exception:
            continue

    return result


# ============================================================
# COINGECKO FALLBACK
# ============================================================

def _get_coingecko_id(symbol: str) -> str:
    """Map a symbol to its CoinGecko ID."""
    return COINGECKO_ID_MAP.get(symbol, symbol.lower())


def get_coingecko_market_data(page: int = 1, per_page: int = 250) -> list:
    """Fetch market data from CoinGecko (prices, changes, volume, rank)."""
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
            time.sleep(10)
    except Exception:
        pass
    return []


def get_coingecko_ohlc(symbol: str, days: int = 7) -> pd.DataFrame:
    """Fetch OHLC from CoinGecko as last-resort fallback."""
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

        df["volume"] = 0.0  # CoinGecko OHLC has no real volume — set to 0 so volume indicators return NEUTRAL
        df["quote_volume"] = 0.0
        df["trades"] = 0

        return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]

    except Exception:
        return pd.DataFrame()


# ============================================================
# SMART FETCHING (CCXT → CoinGecko fallback)
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_klines_cached(symbol: str, interval: str) -> pd.DataFrame:
    """Internal cached fetch — only called when we have real data."""
    return _fetch_klines_uncached(symbol, interval)


def _fetch_klines_uncached(symbol: str, interval: str) -> pd.DataFrame:
    """Actual fetch logic without caching."""
    # Try CCXT first
    df = fetch_ohlcv_ccxt(symbol, interval)
    if not df.empty and len(df) >= 15:
        return df

    # CoinGecko fallback
    interval_to_days = {"1h": 2, "4h": 7, "1d": 30, "1D": 30, "1w": 180, "1W": 180}
    days = interval_to_days.get(interval, 7)
    df = get_coingecko_ohlc(symbol, days=days)
    if not df.empty:
        return df

    return pd.DataFrame()


def fetch_klines_smart(symbol: str, interval: str) -> pd.DataFrame:
    """
    Fetch klines with automatic fallback:
    1. CCXT (auto-selected best exchange)
    2. CoinGecko OHLC

    IMPORTANT: Empty results are NOT cached to prevent 5-min lockout
    when rate-limiting causes temporary failures.
    """
    # Try cached version first
    try:
        df = _fetch_klines_cached(symbol, interval)
        if not df.empty and len(df) >= 15:
            return df
    except Exception:
        pass

    # Cache miss or cached empty — try fresh fetch
    df = _fetch_klines_uncached(symbol, interval)
    return df


# GitHub cryptocurrency-icons base URL (no API needed, always available)
_GH_ICON_BASE = "https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color"

# Coin name mapping for when CoinGecko is unavailable
_COIN_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "XRP": "XRP", "BNB": "BNB",
    "SOL": "Solana", "TRX": "TRON", "DOGE": "Dogecoin", "ADA": "Cardano",
    "LINK": "Chainlink", "AVAX": "Avalanche", "XLM": "Stellar", "HBAR": "Hedera",
    "DOT": "Polkadot", "BCH": "Bitcoin Cash", "LTC": "Litecoin", "SHIB": "Shiba Inu",
    "UNI": "Uniswap", "TON": "Toncoin", "NEAR": "NEAR Protocol", "AAVE": "Aave",
    "ZEC": "Zcash", "SUI": "Sui", "PEPE": "Pepe", "POL": "Polygon",
    "WLD": "Worldcoin", "ATOM": "Cosmos", "ENA": "Ethena", "ONDO": "Ondo",
    "QNT": "Quant", "RENDER": "Render", "FIL": "Filecoin", "VET": "VeChain",
    "ETC": "Ethereum Classic", "TAO": "Bittensor", "INJ": "Injective",
    "STX": "Stacks", "ICP": "Internet Computer", "PAXG": "PAX Gold",
    "ARB": "Arbitrum", "OP": "Optimism", "FET": "Fetch.ai", "SEI": "Sei",
    "JUP": "Jupiter", "BONK": "Bonk", "FLOKI": "Floki", "IMX": "Immutable",
    "APT": "Aptos", "ALGO": "Algorand", "GRT": "The Graph", "THETA": "Theta",
    "SAND": "The Sandbox", "MANA": "Decentraland", "AXS": "Axie Infinity",
    "GALA": "Gala", "CRV": "Curve", "DASH": "Dash", "LDO": "Lido DAO",
    "JASMY": "JasmyCoin", "IOTA": "IOTA", "PENGU": "Pudgy Penguins",
    "VIRTUAL": "Virtuals Protocol", "PUMP": "PumpFun", "MORPHO": "Morpho",
    "NEXO": "Nexo", "TUSD": "TrueUSD", "PYTH": "Pyth Network", "SUN": "Sun",
    "TIA": "Celestia", "CFX": "Conflux", "ENS": "ENS", "WIF": "dogwifhat",
    "COMP": "Compound", "DEXE": "DeXe", "LUNC": "Terra Luna Classic",
    "STRK": "Starknet", "PENDLE": "Pendle", "ETHFI": "Ether.fi", "CHZ": "Chiliz",
    "XTZ": "Tezos", "BAT": "Basic Attention Token", "ZRO": "LayerZero",
    "CAKE": "PancakeSwap", "TRUMP": "TRUMP", "GNO": "Gnosis", "CVX": "Convex",
    "ZK": "zkSync", "GLM": "Golem", "KITE": "Kite AI", "AWE": "Awe",
    "SKY": "Sky", "RAY": "Raydium", "SYRUP": "Maple", "BARD": "Bard AI",
    "SENT": "Sentinel", "DCR": "Decred", "JST": "JUST", "ONT": "Ontology",
    "FF": "Forefront", "KAIA": "Kaia", "NEO": "Neo", "TWT": "Trust Wallet",
    "USDC": "USD Coin", "USDE": "USDe", "USD1": "USD1", "WLFI": "WLFI",
    "RLUSD": "RLUSD", "FDUSD": "FDUSD",
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_market_data() -> pd.DataFrame:
    """
    Fetch market overview from CoinGecko (metadata, rank, images, changes).
    Cached 5min. If CoinGecko is unavailable, builds fallback with GitHub icons.
    """
    all_data = []
    for page in [1, 2]:
        for attempt in range(2):
            data = get_coingecko_market_data(page=page)
            if data:
                all_data.extend(data)
                break
            time.sleep(2 * (attempt + 1))
        if page < 2:
            time.sleep(0.5)

    if all_data:
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

    # --- FALLBACK: CoinGecko unavailable — build minimal metadata ---
    # Uses GitHub icons (no API call) + coin name map
    from config import CRYPTOWAVES_COINS
    fallback_rows = []
    for i, sym in enumerate(CRYPTOWAVES_COINS):
        fallback_rows.append({
            "symbol": sym,
            "name": _COIN_NAMES.get(sym, sym),
            "image": f"{_GH_ICON_BASE}/{sym.lower()}.png",
            "rank": i + 1,  # Approximate rank from list position
            "price": 0, "volume_24h": 0, "market_cap": 0,
            "change_1h": 0, "change_24h": 0, "change_7d": 0, "change_30d": 0,
        })
    return pd.DataFrame(fallback_rows)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_all_tickers() -> dict:
    """Fetch all tickers via CCXT. Cached 1 min."""
    return fetch_all_tickers_ccxt()


# ============================================================
# FEAR & GREED INDEX  (alternative.me — lazy, only on demand)
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_fear_greed_index() -> dict:
    """
    Fetch current Fear & Greed Index. Only called from Confluence tab.
    Returns: {"value": 0-100, "label": "..."}
    """
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
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
# FUNDING RATES  (CCXT Futures — lazy, only on demand)
# Priority: Binance Futures → OKX → Bitget → Bybit
# NEVER loaded during main scan. Only from Confluence tab.
# ============================================================

# Module-level cache (no @st.cache_resource to avoid import-time side effects)
_futures_cache = {"name": None, "ex": None, "tried": False}


def _get_futures_exchange() -> tuple:
    """
    Lazy futures exchange connection. Created only on first call.
    Uses fetch_funding_rate as connectivity test (fast, no load_markets).
    Priority: Binance → OKX → Bitget → Bybit.
    """
    if _futures_cache["tried"]:
        return (_futures_cache["name"], _futures_cache["ex"])

    _futures_cache["tried"] = True

    futures_priority = [
        ("binance", ccxt.binance),
        ("okx", ccxt.okx),
        ("bitget", ccxt.bitget),
        ("bybit", ccxt.bybit),
    ]

    for name, exchange_cls in futures_priority:
        try:
            ex = exchange_cls({
                "enableRateLimit": True,
                "timeout": 5000,
                "options": {"defaultType": "swap"},
            })
            # Quick test with ONE funding rate — NO load_markets()!
            ex.fetch_funding_rate("BTC/USDT:USDT")
            _futures_cache["name"] = name
            _futures_cache["ex"] = ex
            return (name, ex)
        except Exception:
            continue

    return ("none", None)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_funding_rates_batch() -> dict:
    """
    Fetch all funding rates via futures exchange.
    Only called from Confluence tab when funding_rate filter is active.
    Returns: {"BTC": 0.0001, ...}
    """
    fname, fex = _get_futures_exchange()
    if fex is None:
        return {}

    try:
        if hasattr(fex, 'fetch_funding_rates'):
            all_rates = fex.fetch_funding_rates()
            result = {}
            for pair, data in all_rates.items():
                sym = pair.split("/")[0] if "/" in pair else pair.replace("USDT", "").replace(":USDT", "")
                rate = data.get("fundingRate", None)
                if rate is not None:
                    result[sym] = float(rate)
            return result
    except Exception:
        pass

    return {}
