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
    """
    Generate a confluence-based signal combining multiple indicators.
    Returns signal strength (-100 to +100) and recommendation.
    
    Scoring:
    - RSI 4h: -30 to +30
    - RSI 1D: -20 to +20
    - MACD: -20 to +20
    - Volume: -15 to +15
    - SMC: -15 to +15
    """
    score = 0
    reasons = []
    
    # --- RSI 4H scoring (weight: 30) ---
    if rsi_4h <= RSI_STRONG_OVERSOLD:
        score += 30
        reasons.append(f"RSI 4h extremely oversold ({rsi_4h})")
    elif rsi_4h <= RSI_OVERSOLD:
        score += 20
        reasons.append(f"RSI 4h oversold ({rsi_4h})")
    elif rsi_4h <= 45:
        score += 10
        reasons.append(f"RSI 4h approaching oversold ({rsi_4h})")
    elif rsi_4h >= RSI_STRONG_OVERBOUGHT:
        score -= 30
        reasons.append(f"RSI 4h extremely overbought ({rsi_4h})")
    elif rsi_4h >= RSI_OVERBOUGHT:
        score -= 20
        reasons.append(f"RSI 4h overbought ({rsi_4h})")
    elif rsi_4h >= 60:
        score -= 10
        reasons.append(f"RSI 4h approaching overbought ({rsi_4h})")
    
    # --- RSI 1D scoring (weight: 20) ---
    if rsi_1d <= RSI_STRONG_OVERSOLD:
        score += 20
        reasons.append(f"RSI 1D extremely oversold ({rsi_1d})")
    elif rsi_1d <= RSI_OVERSOLD:
        score += 15
        reasons.append(f"RSI 1D oversold ({rsi_1d})")
    elif rsi_1d >= RSI_STRONG_OVERBOUGHT:
        score -= 20
        reasons.append(f"RSI 1D extremely overbought ({rsi_1d})")
    elif rsi_1d >= RSI_OVERBOUGHT:
        score -= 15
        reasons.append(f"RSI 1D overbought ({rsi_1d})")
    
    # --- MACD scoring (weight: 20) ---
    if macd_data["trend"] == "BULLISH":
        score += 15
        if macd_data["histogram"] > 0:
            score += 5
            reasons.append("MACD bullish with growing histogram")
        else:
            reasons.append("MACD bullish")
    elif macd_data["trend"] == "BEARISH":
        score -= 15
        if macd_data["histogram"] < 0:
            score -= 5
            reasons.append("MACD bearish with growing histogram")
        else:
            reasons.append("MACD bearish")
    
    # --- Volume scoring (weight: 15) ---
    if volume_data["vol_trend"] == "HIGH" and volume_data["obv_trend"] == "BULLISH":
        score += 15
        reasons.append("High volume with bullish OBV")
    elif volume_data["vol_trend"] == "HIGH" and volume_data["obv_trend"] == "BEARISH":
        score -= 15
        reasons.append("High volume with bearish OBV")
    elif volume_data["obv_trend"] == "BULLISH":
        score += 8
        reasons.append("Bullish OBV trend")
    elif volume_data["obv_trend"] == "BEARISH":
        score -= 8
        reasons.append("Bearish OBV trend")
    
    # --- SMC scoring (weight: 15) ---
    if smc_data:
        ob_signal = smc_data.get("ob_signal", "NONE")
        fvg_signal = smc_data.get("fvg_signal", "BALANCED")
        structure = smc_data.get("structure", "UNKNOWN")
        
        if ob_signal == "NEAR_BULLISH_OB":
            score += 10
            reasons.append("Near bullish order block")
        elif ob_signal == "NEAR_BEARISH_OB":
            score -= 10
            reasons.append("Near bearish order block")
        
        if structure == "BULLISH":
            score += 5
            reasons.append("Bullish market structure")
        elif structure == "BEARISH":
            score -= 5
            reasons.append("Bearish market structure")
    
    # --- Generate recommendation ---
    score = max(-100, min(100, score))
    
    if score >= 60:
        recommendation = "STRONG BUY"
    elif score >= 30:
        recommendation = "BUY"
    elif score >= 10:
        recommendation = "LEAN BUY"
    elif score <= -60:
        recommendation = "STRONG SELL"
    elif score <= -30:
        recommendation = "SELL"
    elif score <= -10:
        recommendation = "LEAN SELL"
    else:
        recommendation = "WAIT"
    
    return {
        "score": score,
        "recommendation": recommendation,
        "reasons": reasons,
    }
