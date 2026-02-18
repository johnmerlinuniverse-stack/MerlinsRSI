"""
Technical indicators: RSI, MACD, Volume Analysis, Smart Money Concepts.
Uses the 'ta' library for standard indicators + custom SMC logic.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.volatility import BollingerBands, AverageTrueRange

from config import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    RSI_STRONG_OVERBOUGHT, RSI_STRONG_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
)


# ============================================================
# RSI
# ============================================================

def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> float:
    """Calculate current RSI value from OHLCV DataFrame."""
    if df.empty or len(df) < period + 1:
        return 50.0  # neutral default
    
    rsi = RSIIndicator(close=df["close"], window=period)
    rsi_series = rsi.rsi()
    
    if rsi_series.empty or rsi_series.isna().all():
        return 50.0
    
    return round(rsi_series.iloc[-1], 2)


def calculate_rsi_series(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate full RSI series."""
    if df.empty or len(df) < period + 1:
        return pd.Series(dtype=float)
    
    rsi = RSIIndicator(close=df["close"], window=period)
    return rsi.rsi()


def calculate_stoch_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> dict:
    """Calculate Stochastic RSI."""
    if df.empty or len(df) < period * 2:
        return {"stoch_rsi_k": 50.0, "stoch_rsi_d": 50.0}
    
    stoch_rsi = StochRSIIndicator(close=df["close"], window=period, smooth1=3, smooth2=3)
    k = stoch_rsi.stochrsi_k()
    d = stoch_rsi.stochrsi_d()
    
    return {
        "stoch_rsi_k": round(k.iloc[-1] * 100, 2) if not k.empty and not np.isnan(k.iloc[-1]) else 50.0,
        "stoch_rsi_d": round(d.iloc[-1] * 100, 2) if not d.empty and not np.isnan(d.iloc[-1]) else 50.0,
    }


# ============================================================
# MACD
# ============================================================

def calculate_macd(df: pd.DataFrame) -> dict:
    """Calculate MACD values."""
    if df.empty or len(df) < MACD_SLOW + MACD_SIGNAL:
        return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
    
    macd = MACD(close=df["close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    
    macd_line = macd.macd().iloc[-1]
    signal_line = macd.macd_signal().iloc[-1]
    histogram = macd.macd_diff().iloc[-1]
    
    # Determine trend
    if np.isnan(macd_line) or np.isnan(signal_line):
        trend = "NEUTRAL"
    elif macd_line > signal_line and histogram > 0:
        trend = "BULLISH"
    elif macd_line < signal_line and histogram < 0:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"
    
    return {
        "macd": round(macd_line, 4) if not np.isnan(macd_line) else 0,
        "signal": round(signal_line, 4) if not np.isnan(signal_line) else 0,
        "histogram": round(histogram, 4) if not np.isnan(histogram) else 0,
        "trend": trend,
    }


# ============================================================
# VOLUME ANALYSIS
# ============================================================

def calculate_volume_analysis(df: pd.DataFrame) -> dict:
    """Analyze volume patterns."""
    if df.empty or len(df) < 21:
        return {"vol_trend": "NEUTRAL", "vol_ratio": 1.0, "obv_trend": "NEUTRAL"}
    
    # Volume ratio (current vs 20-period average)
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]
    vol_current = df["volume"].iloc[-1]
    vol_ratio = round(vol_current / vol_avg, 2) if vol_avg > 0 else 1.0
    
    # Volume trend
    if vol_ratio > 1.5:
        vol_trend = "HIGH"
    elif vol_ratio > 1.0:
        vol_trend = "ABOVE_AVG"
    elif vol_ratio > 0.5:
        vol_trend = "BELOW_AVG"
    else:
        vol_trend = "LOW"
    
    # OBV trend
    try:
        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
        obv_series = obv.on_balance_volume()
        obv_sma = obv_series.rolling(10).mean()
        
        if obv_series.iloc[-1] > obv_sma.iloc[-1]:
            obv_trend = "BULLISH"
        else:
            obv_trend = "BEARISH"
    except Exception:
        obv_trend = "NEUTRAL"
    
    return {
        "vol_trend": vol_trend,
        "vol_ratio": vol_ratio,
        "obv_trend": obv_trend,
    }


# ============================================================
# SMART MONEY CONCEPTS (simplified)
# ============================================================

def detect_order_blocks(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detect potential order blocks (simplified SMC).
    Bullish OB: Last bearish candle before a strong bullish move.
    Bearish OB: Last bullish candle before a strong bearish move.
    """
    if df.empty or len(df) < lookback:
        return {"bullish_ob": None, "bearish_ob": None, "ob_signal": "NONE"}
    
    recent = df.tail(lookback).copy()
    recent["body"] = recent["close"] - recent["open"]
    recent["range"] = recent["high"] - recent["low"]
    avg_range = recent["range"].mean()
    
    bullish_ob = None
    bearish_ob = None
    
    for i in range(len(recent) - 3, 0, -1):
        # Bullish Order Block: bearish candle followed by strong bullish move
        if (recent.iloc[i]["body"] < 0 and  # bearish candle
            recent.iloc[i + 1]["body"] > avg_range * 1.5):  # strong bullish
            bullish_ob = {
                "price_high": recent.iloc[i]["high"],
                "price_low": recent.iloc[i]["low"],
                "idx": i,
            }
            break
    
    for i in range(len(recent) - 3, 0, -1):
        # Bearish Order Block: bullish candle followed by strong bearish move
        if (recent.iloc[i]["body"] > 0 and  # bullish candle
            recent.iloc[i + 1]["body"] < -avg_range * 1.5):  # strong bearish
            bearish_ob = {
                "price_high": recent.iloc[i]["high"],
                "price_low": recent.iloc[i]["low"],
                "idx": i,
            }
            break
    
    current_price = recent.iloc[-1]["close"]
    ob_signal = "NONE"
    
    if bullish_ob and current_price <= bullish_ob["price_high"] * 1.01:
        ob_signal = "NEAR_BULLISH_OB"
    elif bearish_ob and current_price >= bearish_ob["price_low"] * 0.99:
        ob_signal = "NEAR_BEARISH_OB"
    
    return {
        "bullish_ob": bullish_ob,
        "bearish_ob": bearish_ob,
        "ob_signal": ob_signal,
    }


def detect_fair_value_gaps(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detect Fair Value Gaps (FVG) - imbalances in price action.
    Bullish FVG: Gap between candle[i-1] high and candle[i+1] low.
    """
    if df.empty or len(df) < lookback:
        return {"fvg_signal": "NONE", "fvg_count_bull": 0, "fvg_count_bear": 0}
    
    recent = df.tail(lookback).copy()
    fvg_bull = 0
    fvg_bear = 0
    
    for i in range(1, len(recent) - 1):
        # Bullish FVG
        if recent.iloc[i + 1]["low"] > recent.iloc[i - 1]["high"]:
            fvg_bull += 1
        # Bearish FVG
        if recent.iloc[i + 1]["high"] < recent.iloc[i - 1]["low"]:
            fvg_bear += 1
    
    if fvg_bull > fvg_bear + 1:
        signal = "BULLISH_IMBALANCE"
    elif fvg_bear > fvg_bull + 1:
        signal = "BEARISH_IMBALANCE"
    else:
        signal = "BALANCED"
    
    return {
        "fvg_signal": signal,
        "fvg_count_bull": fvg_bull,
        "fvg_count_bear": fvg_bear,
    }


def detect_market_structure(df: pd.DataFrame, lookback: int = 30) -> dict:
    """
    Detect market structure: Higher Highs/Higher Lows (uptrend) or 
    Lower Highs/Lower Lows (downtrend).
    """
    if df.empty or len(df) < lookback:
        return {"structure": "UNKNOWN", "break_of_structure": False}
    
    recent = df.tail(lookback)
    
    # Find swing highs and lows using simple pivot detection
    highs = []
    lows = []
    
    for i in range(2, len(recent) - 2):
        if (recent.iloc[i]["high"] > recent.iloc[i - 1]["high"] and
            recent.iloc[i]["high"] > recent.iloc[i - 2]["high"] and
            recent.iloc[i]["high"] > recent.iloc[i + 1]["high"] and
            recent.iloc[i]["high"] > recent.iloc[i + 2]["high"]):
            highs.append(recent.iloc[i]["high"])
        
        if (recent.iloc[i]["low"] < recent.iloc[i - 1]["low"] and
            recent.iloc[i]["low"] < recent.iloc[i - 2]["low"] and
            recent.iloc[i]["low"] < recent.iloc[i + 1]["low"] and
            recent.iloc[i]["low"] < recent.iloc[i + 2]["low"]):
            lows.append(recent.iloc[i]["low"])
    
    if len(highs) < 2 or len(lows) < 2:
        return {"structure": "UNKNOWN", "break_of_structure": False}
    
    # Check for Higher Highs and Higher Lows (Bullish)
    hh = highs[-1] > highs[-2]
    hl = lows[-1] > lows[-2]
    # Lower Highs and Lower Lows (Bearish)
    lh = highs[-1] < highs[-2]
    ll = lows[-1] < lows[-2]
    
    if hh and hl:
        structure = "BULLISH"
    elif lh and ll:
        structure = "BEARISH"
    elif hh and ll:
        structure = "RANGING"
    else:
        structure = "TRANSITIONING"
    
    # Break of Structure: last close breaks previous swing high/low
    current_close = recent.iloc[-1]["close"]
    bos = False
    if current_close > highs[-1] and structure != "BULLISH":
        bos = True
    elif current_close < lows[-1] and structure != "BEARISH":
        bos = True
    
    return {"structure": structure, "break_of_structure": bos}


# ============================================================
# CONFLUENCE SIGNAL GENERATOR
# ============================================================

def generate_confluence_signal(
    rsi_4h: float,
    rsi_1d: float,
    macd_data: dict,
    volume_data: dict,
    smc_data: dict = None,
) -> dict:
    score = 0
    reasons = []
    if rsi_4h <= RSI_STRONG_OVERSOLD: score += 30; reasons.append(f"RSI 4h extremely oversold ({rsi_4h})")
    elif rsi_4h <= RSI_OVERSOLD: score += 20; reasons.append(f"RSI 4h oversold ({rsi_4h})")
    elif rsi_4h <= 45: score += 10; reasons.append(f"RSI 4h approaching oversold ({rsi_4h})")
    elif rsi_4h >= RSI_STRONG_OVERBOUGHT: score -= 30; reasons.append(f"RSI 4h extremely overbought ({rsi_4h})")
    elif rsi_4h >= RSI_OVERBOUGHT: score -= 20; reasons.append(f"RSI 4h overbought ({rsi_4h})")
    elif rsi_4h >= 60: score -= 10; reasons.append(f"RSI 4h approaching overbought ({rsi_4h})")
    if rsi_1d <= RSI_STRONG_OVERSOLD: score += 20; reasons.append(f"RSI 1D extremely oversold ({rsi_1d})")
    elif rsi_1d <= RSI_OVERSOLD: score += 15; reasons.append(f"RSI 1D oversold ({rsi_1d})")
    elif rsi_1d >= RSI_STRONG_OVERBOUGHT: score -= 20; reasons.append(f"RSI 1D extremely overbought ({rsi_1d})")
    elif rsi_1d >= RSI_OVERBOUGHT: score -= 15; reasons.append(f"RSI 1D overbought ({rsi_1d})")
    if macd_data["trend"] == "BULLISH":
        score += 15 + (5 if macd_data["histogram"] > 0 else 0)
        reasons.append("MACD bullish" + (" with growing histogram" if macd_data["histogram"] > 0 else ""))
    elif macd_data["trend"] == "BEARISH":
        score -= 15 + (5 if macd_data["histogram"] < 0 else 0)
        reasons.append("MACD bearish" + (" with growing histogram" if macd_data["histogram"] < 0 else ""))
    if volume_data["vol_trend"] == "HIGH" and volume_data["obv_trend"] == "BULLISH": score += 15; reasons.append("High volume bullish OBV")
    elif volume_data["vol_trend"] == "HIGH" and volume_data["obv_trend"] == "BEARISH": score -= 15; reasons.append("High volume bearish OBV")
    elif volume_data["obv_trend"] == "BULLISH": score += 8; reasons.append("Bullish OBV")
    elif volume_data["obv_trend"] == "BEARISH": score -= 8; reasons.append("Bearish OBV")
    if smc_data:
        if smc_data.get("ob_signal") == "NEAR_BULLISH_OB": score += 10; reasons.append("Near bullish OB")
        elif smc_data.get("ob_signal") == "NEAR_BEARISH_OB": score -= 10; reasons.append("Near bearish OB")
        if smc_data.get("structure") == "BULLISH": score += 5; reasons.append("Bullish structure")
        elif smc_data.get("structure") == "BEARISH": score -= 5; reasons.append("Bearish structure")
    score = max(-100, min(100, score))
    if score >= 60: rec = "STRONG BUY"
    elif score >= 30: rec = "BUY"
    elif score >= 10: rec = "LEAN BUY"
    elif score <= -60: rec = "STRONG SELL"
    elif score <= -30: rec = "SELL"
    elif score <= -10: rec = "LEAN SELL"
    else: rec = "WAIT"
    return {"score": score, "recommendation": rec, "reasons": reasons}


# ============================================================
# DETAIL TAB: ADVANCED INDICATORS (computed on-demand per coin)
# ============================================================

def calculate_ema_crosses(df: pd.DataFrame) -> dict:
    """EMA 9/21 and 50/200 crossover analysis."""
    if df.empty or len(df) < 200:
        short_ok = not df.empty and len(df) >= 21
        if short_ok:
            ema9 = EMAIndicator(close=df["close"], window=9).ema_indicator()
            ema21 = EMAIndicator(close=df["close"], window=21).ema_indicator()
            e9, e21 = ema9.iloc[-1], ema21.iloc[-1]
            e9p, e21p = ema9.iloc[-2], ema21.iloc[-2]
            cross_9_21 = "GOLDEN" if e9 > e21 and e9p <= e21p else ("DEATH" if e9 < e21 and e9p >= e21p else ("BULLISH" if e9 > e21 else "BEARISH"))
            return {"ema9": round(e9, 6), "ema21": round(e21, 6), "cross_9_21": cross_9_21,
                    "ema50": None, "ema200": None, "cross_50_200": "N/A",
                    "price_vs_ema21": round((df["close"].iloc[-1] / e21 - 1) * 100, 2)}
        return {"ema9": None, "ema21": None, "cross_9_21": "N/A", "ema50": None, "ema200": None, "cross_50_200": "N/A", "price_vs_ema21": 0}

    ema9 = EMAIndicator(close=df["close"], window=9).ema_indicator()
    ema21 = EMAIndicator(close=df["close"], window=21).ema_indicator()
    ema50 = EMAIndicator(close=df["close"], window=50).ema_indicator()
    ema200 = EMAIndicator(close=df["close"], window=200).ema_indicator()

    e9, e21, e50, e200 = ema9.iloc[-1], ema21.iloc[-1], ema50.iloc[-1], ema200.iloc[-1]
    e9p, e21p = ema9.iloc[-2], ema21.iloc[-2]
    e50p, e200p = ema50.iloc[-2], ema200.iloc[-2]

    # 9/21 cross
    if e9 > e21 and e9p <= e21p: c1 = "GOLDEN"
    elif e9 < e21 and e9p >= e21p: c1 = "DEATH"
    elif e9 > e21: c1 = "BULLISH"
    else: c1 = "BEARISH"

    # 50/200 cross
    if e50 > e200 and e50p <= e200p: c2 = "GOLDEN"
    elif e50 < e200 and e50p >= e200p: c2 = "DEATH"
    elif e50 > e200: c2 = "BULLISH"
    else: c2 = "BEARISH"

    price = df["close"].iloc[-1]
    return {
        "ema9": round(e9, 6), "ema21": round(e21, 6), "ema50": round(e50, 6), "ema200": round(e200, 6),
        "cross_9_21": c1, "cross_50_200": c2,
        "price_vs_ema21": round((price / e21 - 1) * 100, 2),
        "price_vs_ema200": round((price / e200 - 1) * 100, 2),
    }


def calculate_bollinger(df: pd.DataFrame, window: int = 20) -> dict:
    """Bollinger Bands position analysis."""
    if df.empty or len(df) < window + 1:
        return {"bb_upper": 0, "bb_middle": 0, "bb_lower": 0, "bb_width": 0, "bb_position": "MIDDLE", "bb_pct": 50.0}
    bb = BollingerBands(close=df["close"], window=window, window_dev=2)
    upper = bb.bollinger_hband().iloc[-1]
    middle = bb.bollinger_mavg().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    width = round((upper - lower) / middle * 100, 2) if middle > 0 else 0
    price = df["close"].iloc[-1]
    pct = round((price - lower) / (upper - lower) * 100, 2) if upper != lower else 50.0

    if pct >= 95: pos = "ABOVE_UPPER"
    elif pct >= 75: pos = "UPPER_ZONE"
    elif pct >= 25: pos = "MIDDLE"
    elif pct >= 5: pos = "LOWER_ZONE"
    else: pos = "BELOW_LOWER"

    return {"bb_upper": round(upper, 6), "bb_middle": round(middle, 6), "bb_lower": round(lower, 6),
            "bb_width": width, "bb_position": pos, "bb_pct": pct}


def calculate_atr(df: pd.DataFrame, window: int = 14) -> dict:
    """Average True Range for volatility and SL/TP."""
    if df.empty or len(df) < window + 1:
        return {"atr": 0, "atr_pct": 0, "volatility": "LOW"}
    atr_ind = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=window)
    atr_val = atr_ind.average_true_range().iloc[-1]
    price = df["close"].iloc[-1]
    atr_pct = round(atr_val / price * 100, 2) if price > 0 else 0

    if atr_pct > 5: vol = "VERY_HIGH"
    elif atr_pct > 3: vol = "HIGH"
    elif atr_pct > 1.5: vol = "MEDIUM"
    else: vol = "LOW"

    return {"atr": round(atr_val, 6), "atr_pct": atr_pct, "volatility": vol}


def calculate_support_resistance(df: pd.DataFrame, lookback: int = 50) -> dict:
    """Find key S/R levels from swing highs/lows."""
    if df.empty or len(df) < lookback:
        return {"supports": [], "resistances": [], "nearest_support": 0, "nearest_resistance": 0}
    recent = df.tail(lookback)
    highs, lows = [], []
    for i in range(2, len(recent) - 2):
        h = recent.iloc[i]["high"]; lo = recent.iloc[i]["low"]
        if h > recent.iloc[i-1]["high"] and h > recent.iloc[i-2]["high"] and h > recent.iloc[i+1]["high"] and h > recent.iloc[i+2]["high"]:
            highs.append(round(h, 6))
        if lo < recent.iloc[i-1]["low"] and lo < recent.iloc[i-2]["low"] and lo < recent.iloc[i+1]["low"] and lo < recent.iloc[i+2]["low"]:
            lows.append(round(lo, 6))
    price = recent.iloc[-1]["close"]
    supports = sorted([l for l in lows if l < price], reverse=True)[:3]
    resistances = sorted([h for h in highs if h > price])[:3]
    return {
        "supports": supports, "resistances": resistances,
        "nearest_support": supports[0] if supports else 0,
        "nearest_resistance": resistances[0] if resistances else 0,
    }


def calculate_fibonacci(df: pd.DataFrame, lookback: int = 50) -> dict:
    """Fibonacci retracement levels based on recent swing high/low."""
    if df.empty or len(df) < lookback:
        return {"fib_levels": {}, "fib_zone": "N/A"}
    recent = df.tail(lookback)
    swing_high = recent["high"].max()
    swing_low = recent["low"].min()
    diff = swing_high - swing_low
    if diff <= 0:
        return {"fib_levels": {}, "fib_zone": "N/A"}
    levels = {
        "0.0 (High)": round(swing_high, 6),
        "0.236": round(swing_high - 0.236 * diff, 6),
        "0.382": round(swing_high - 0.382 * diff, 6),
        "0.5": round(swing_high - 0.5 * diff, 6),
        "0.618": round(swing_high - 0.618 * diff, 6),
        "0.786": round(swing_high - 0.786 * diff, 6),
        "1.0 (Low)": round(swing_low, 6),
    }
    price = recent.iloc[-1]["close"]
    pct = (swing_high - price) / diff
    if pct <= 0.236: zone = "0-23.6% (near high)"
    elif pct <= 0.382: zone = "23.6-38.2%"
    elif pct <= 0.5: zone = "38.2-50%"
    elif pct <= 0.618: zone = "50-61.8% (golden zone)"
    elif pct <= 0.786: zone = "61.8-78.6%"
    else: zone = "78.6-100% (near low)"
    return {"fib_levels": levels, "fib_zone": zone}


def calculate_btc_correlation(coin_df: pd.DataFrame, btc_df: pd.DataFrame, window: int = 20) -> dict:
    """Pearson correlation between coin returns and BTC returns."""
    if coin_df.empty or btc_df.empty or len(coin_df) < window or len(btc_df) < window:
        return {"correlation": 0, "corr_label": "N/A"}
    coin_ret = coin_df["close"].pct_change().dropna().tail(window)
    btc_ret = btc_df["close"].pct_change().dropna().tail(window)
    min_len = min(len(coin_ret), len(btc_ret))
    if min_len < 5:
        return {"correlation": 0, "corr_label": "N/A"}
    corr = round(coin_ret.tail(min_len).reset_index(drop=True).corr(btc_ret.tail(min_len).reset_index(drop=True)), 2)
    if np.isnan(corr): corr = 0
    if corr >= 0.7: label = "STRONG_POS"
    elif corr >= 0.3: label = "MODERATE_POS"
    elif corr >= -0.3: label = "WEAK/NONE"
    elif corr >= -0.7: label = "MODERATE_NEG"
    else: label = "STRONG_NEG"
    return {"correlation": corr, "corr_label": label}


def calculate_sl_tp(price: float, atr: float, signal: str, sr_data: dict) -> dict:
    """Calculate Stop-Loss and Take-Profit based on ATR + S/R levels."""
    if atr <= 0 or price <= 0:
        return {"sl": 0, "tp1": 0, "tp2": 0, "tp3": 0, "risk_reward": 0}
    if signal in ("BUY", "CTB"):
        sl = sr_data["nearest_support"] if sr_data["nearest_support"] > 0 else price - 2 * atr
        sl = max(sl, price - 3 * atr)  # cap at 3x ATR
        tp1 = price + 1.5 * atr
        tp2 = sr_data["nearest_resistance"] if sr_data["nearest_resistance"] > 0 else price + 3 * atr
        tp3 = price + 4 * atr
    elif signal in ("SELL", "CTS"):
        sl = sr_data["nearest_resistance"] if sr_data["nearest_resistance"] > 0 else price + 2 * atr
        sl = min(sl, price + 3 * atr)
        tp1 = price - 1.5 * atr
        tp2 = sr_data["nearest_support"] if sr_data["nearest_support"] > 0 else price - 3 * atr
        tp3 = price - 4 * atr
    else:
        return {"sl": round(price - 2 * atr, 6), "tp1": round(price + 2 * atr, 6), "tp2": 0, "tp3": 0, "risk_reward": 1.0}
    risk = abs(price - sl)
    reward = abs(tp2 - price) if tp2 > 0 else abs(tp1 - price)
    rr = round(reward / risk, 2) if risk > 0 else 0
    return {"sl": round(sl, 6), "tp1": round(tp1, 6), "tp2": round(tp2, 6), "tp3": round(tp3, 6), "risk_reward": rr}


def calculate_price_range(df: pd.DataFrame) -> dict:
    """7d/30d high-low range analysis."""
    result = {}
    price = df["close"].iloc[-1] if not df.empty else 0
    for days, label in [(7, "7d"), (30, "30d")]:
        if len(df) >= days:
            subset = df.tail(days)
            hi, lo = subset["high"].max(), subset["low"].min()
            rng = hi - lo
            pos = round((price - lo) / rng * 100, 1) if rng > 0 else 50
            result[f"{label}_high"] = round(hi, 6)
            result[f"{label}_low"] = round(lo, 6)
            result[f"{label}_range_pct"] = round(rng / price * 100, 2) if price > 0 else 0
            result[f"{label}_position"] = pos
        else:
            result[f"{label}_high"] = result[f"{label}_low"] = result[f"{label}_range_pct"] = 0
            result[f"{label}_position"] = 50
    return result


def multi_tf_rsi_summary(rsi_values: dict) -> dict:
    """Multi-timeframe RSI traffic light summary."""
    bullish = 0; bearish = 0; total = 0
    for tf, val in rsi_values.items():
        if val is None: continue
        total += 1
        if val <= 42: bullish += 1
        elif val >= 58: bearish += 1
    if total == 0: return {"confluence": "N/A", "bullish_count": 0, "bearish_count": 0, "total": 0}
    if bullish >= total * 0.75: conf = "STRONG_BUY"
    elif bullish > bearish: conf = "LEAN_BUY"
    elif bearish >= total * 0.75: conf = "STRONG_SELL"
    elif bearish > bullish: conf = "LEAN_SELL"
    else: conf = "NEUTRAL"
    return {"confluence": conf, "bullish_count": bullish, "bearish_count": bearish, "total": total}
