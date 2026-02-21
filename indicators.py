"""
Technical indicators: RSI, MACD, Volume Analysis, Smart Money Concepts,
plus extended Confluence factors (EMA, Divergence, Bollinger, Stoch RSI).
Uses the 'ta' library for standard indicators + custom logic.
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
    if df.empty or len(df) < period + 1:
        return 50.0
    rsi = RSIIndicator(close=df["close"], window=period)
    rsi_series = rsi.rsi()
    if rsi_series.empty or rsi_series.isna().all():
        return 50.0
    return round(rsi_series.iloc[-1], 2)


def calculate_rsi_series(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
    if df.empty or len(df) < period + 1:
        return pd.Series(dtype=float)
    rsi = RSIIndicator(close=df["close"], window=period)
    return rsi.rsi()


def calculate_stoch_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> dict:
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
    if df.empty or len(df) < MACD_SLOW + MACD_SIGNAL:
        return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
    macd = MACD(close=df["close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    macd_line = macd.macd().iloc[-1]
    signal_line = macd.macd_signal().iloc[-1]
    histogram = macd.macd_diff().iloc[-1]
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
    if df.empty or len(df) < 21:
        return {"vol_trend": "NEUTRAL", "vol_ratio": 1.0, "obv_trend": "NEUTRAL"}
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]
    vol_current = df["volume"].iloc[-1]
    vol_ratio = round(vol_current / vol_avg, 2) if vol_avg > 0 else 1.0
    if vol_ratio > 1.5: vol_trend = "HIGH"
    elif vol_ratio > 1.0: vol_trend = "ABOVE_AVG"
    elif vol_ratio > 0.5: vol_trend = "BELOW_AVG"
    else: vol_trend = "LOW"
    try:
        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
        obv_series = obv.on_balance_volume()
        obv_sma = obv_series.rolling(10).mean()
        obv_trend = "BULLISH" if obv_series.iloc[-1] > obv_sma.iloc[-1] else "BEARISH"
    except Exception:
        obv_trend = "NEUTRAL"
    return {"vol_trend": vol_trend, "vol_ratio": vol_ratio, "obv_trend": obv_trend}


# ============================================================
# SMART MONEY CONCEPTS (simplified)
# ============================================================

def detect_order_blocks(df: pd.DataFrame, lookback: int = 20) -> dict:
    if df.empty or len(df) < lookback:
        return {"bullish_ob": None, "bearish_ob": None, "ob_signal": "NONE"}
    recent = df.tail(lookback).copy()
    recent["body"] = recent["close"] - recent["open"]
    recent["range"] = recent["high"] - recent["low"]
    avg_range = recent["range"].mean()
    bullish_ob = None
    bearish_ob = None
    for i in range(len(recent) - 3, 0, -1):
        if recent.iloc[i]["body"] < 0 and recent.iloc[i + 1]["body"] > avg_range * 1.5:
            bullish_ob = {"price_high": recent.iloc[i]["high"], "price_low": recent.iloc[i]["low"], "idx": i}
            break
    for i in range(len(recent) - 3, 0, -1):
        if recent.iloc[i]["body"] > 0 and recent.iloc[i + 1]["body"] < -avg_range * 1.5:
            bearish_ob = {"price_high": recent.iloc[i]["high"], "price_low": recent.iloc[i]["low"], "idx": i}
            break
    current_price = recent.iloc[-1]["close"]
    ob_signal = "NONE"
    if bullish_ob and current_price <= bullish_ob["price_high"] * 1.01:
        ob_signal = "NEAR_BULLISH_OB"
    elif bearish_ob and current_price >= bearish_ob["price_low"] * 0.99:
        ob_signal = "NEAR_BEARISH_OB"
    return {"bullish_ob": bullish_ob, "bearish_ob": bearish_ob, "ob_signal": ob_signal}


def detect_fair_value_gaps(df: pd.DataFrame, lookback: int = 20) -> dict:
    if df.empty or len(df) < lookback:
        return {"fvg_signal": "NONE", "fvg_count_bull": 0, "fvg_count_bear": 0}
    recent = df.tail(lookback).copy()
    fvg_bull = 0
    fvg_bear = 0
    for i in range(1, len(recent) - 1):
        if recent.iloc[i + 1]["low"] > recent.iloc[i - 1]["high"]:
            fvg_bull += 1
        if recent.iloc[i + 1]["high"] < recent.iloc[i - 1]["low"]:
            fvg_bear += 1
    if fvg_bull > fvg_bear + 1: signal = "BULLISH_IMBALANCE"
    elif fvg_bear > fvg_bull + 1: signal = "BEARISH_IMBALANCE"
    else: signal = "BALANCED"
    return {"fvg_signal": signal, "fvg_count_bull": fvg_bull, "fvg_count_bear": fvg_bear}


def detect_market_structure(df: pd.DataFrame, lookback: int = 30) -> dict:
    if df.empty or len(df) < lookback:
        return {"structure": "UNKNOWN", "break_of_structure": False}
    recent = df.tail(lookback)
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
    hh = highs[-1] > highs[-2]
    hl = lows[-1] > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1] < lows[-2]
    if hh and hl: structure = "BULLISH"
    elif lh and ll: structure = "BEARISH"
    elif hh and ll: structure = "RANGING"
    else: structure = "TRANSITIONING"
    current_close = recent.iloc[-1]["close"]
    bos = False
    if current_close > highs[-1] and structure != "BULLISH": bos = True
    elif current_close < lows[-1] and structure != "BEARISH": bos = True
    return {"structure": structure, "break_of_structure": bos}


# ============================================================
# NEW CONFLUENCE FACTORS
# ============================================================

def calculate_ema_alignment_fast(df: pd.DataFrame) -> dict:
    """
    Quick EMA alignment check for scan: Price vs EMA21/50.
    Returns direction and strength.
    """
    if df.empty or len(df) < 52:
        return {"ema_trend": "NEUTRAL", "ema_bull_count": 0, "ema_bear_count": 0}

    close = df["close"]
    price = close.iloc[-1]
    ema9 = EMAIndicator(close=close, window=9).ema_indicator().iloc[-1]
    ema21 = EMAIndicator(close=close, window=21).ema_indicator().iloc[-1]
    ema50 = EMAIndicator(close=close, window=50).ema_indicator().iloc[-1]

    bull = 0
    bear = 0
    # Price above EMA = bullish
    if price > ema9: bull += 1
    else: bear += 1
    if price > ema21: bull += 1
    else: bear += 1
    if price > ema50: bull += 1
    else: bear += 1
    # EMA order (9 > 21 > 50 = perfect bull)
    if ema9 > ema21 > ema50: bull += 1
    elif ema9 < ema21 < ema50: bear += 1

    if bull >= 3: trend = "BULLISH"
    elif bear >= 3: trend = "BEARISH"
    else: trend = "NEUTRAL"

    return {"ema_trend": trend, "ema_bull_count": bull, "ema_bear_count": bear}


def detect_rsi_divergence(df: pd.DataFrame, period: int = RSI_PERIOD, lookback: int = 30) -> dict:
    """
    Detect RSI divergence (bullish & bearish).
    Bullish: Price makes Lower Low, RSI makes Higher Low.
    Bearish: Price makes Higher High, RSI makes Lower High.
    """
    if df.empty or len(df) < lookback + period:
        return {"divergence": "NONE", "div_type": "NONE"}

    rsi_series = RSIIndicator(close=df["close"], window=period).rsi()
    if rsi_series.isna().sum() > len(rsi_series) * 0.5:
        return {"divergence": "NONE", "div_type": "NONE"}

    prices = df["close"].tail(lookback).values
    rsi_vals = rsi_series.tail(lookback).values

    # Find recent swing lows/highs in price (simplified)
    price_lows = []
    price_highs = []
    rsi_at_lows = []
    rsi_at_highs = []

    for i in range(2, len(prices) - 2):
        if prices[i] < prices[i-1] and prices[i] < prices[i-2] and prices[i] < prices[i+1] and prices[i] < prices[i+2]:
            price_lows.append(prices[i])
            rsi_at_lows.append(rsi_vals[i])
        if prices[i] > prices[i-1] and prices[i] > prices[i-2] and prices[i] > prices[i+1] and prices[i] > prices[i+2]:
            price_highs.append(prices[i])
            rsi_at_highs.append(rsi_vals[i])

    # Bullish divergence: Price LL + RSI HL
    if len(price_lows) >= 2 and len(rsi_at_lows) >= 2:
        if price_lows[-1] < price_lows[-2] and rsi_at_lows[-1] > rsi_at_lows[-2]:
            return {"divergence": "BULLISH", "div_type": "REGULAR"}

    # Bearish divergence: Price HH + RSI LH
    if len(price_highs) >= 2 and len(rsi_at_highs) >= 2:
        if price_highs[-1] > price_highs[-2] and rsi_at_highs[-1] < rsi_at_highs[-2]:
            return {"divergence": "BEARISH", "div_type": "REGULAR"}

    # Hidden bullish: Price HL + RSI LL
    if len(price_lows) >= 2 and len(rsi_at_lows) >= 2:
        if price_lows[-1] > price_lows[-2] and rsi_at_lows[-1] < rsi_at_lows[-2]:
            return {"divergence": "BULLISH", "div_type": "HIDDEN"}

    # Hidden bearish: Price LH + RSI HH
    if len(price_highs) >= 2 and len(rsi_at_highs) >= 2:
        if price_highs[-1] < price_highs[-2] and rsi_at_highs[-1] > rsi_at_highs[-2]:
            return {"divergence": "BEARISH", "div_type": "HIDDEN"}

    return {"divergence": "NONE", "div_type": "NONE"}


def calculate_bb_squeeze(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Bollinger Bands squeeze detection + position scoring.
    Squeeze = bands narrowing (width < 3%) → breakout imminent.
    """
    if df.empty or len(df) < window + 5:
        return {"bb_squeeze": False, "bb_pct": 50.0, "bb_width": 0, "bb_direction": "NEUTRAL"}

    bb = BollingerBands(close=df["close"], window=window, window_dev=2)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    middle = bb.bollinger_mavg().iloc[-1]
    price = df["close"].iloc[-1]

    width = round((upper - lower) / middle * 100, 2) if middle > 0 else 0
    pct = round((price - lower) / (upper - lower) * 100, 2) if upper != lower else 50.0

    # Check squeeze (compare current width to average width)
    bb_widths = ((bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100).dropna()
    avg_width = bb_widths.rolling(20).mean().iloc[-1] if len(bb_widths) >= 20 else width
    squeeze = width < avg_width * 0.6 and width < 3.0

    # Direction hint based on price position
    if pct < 20: direction = "BULLISH"
    elif pct > 80: direction = "BEARISH"
    elif squeeze: direction = "SQUEEZE"
    else: direction = "NEUTRAL"

    return {"bb_squeeze": squeeze, "bb_pct": pct, "bb_width": width, "bb_direction": direction}


# ============================================================
# MODULAR CONFLUENCE SCORING — individual scores per factor
# ============================================================

def compute_individual_scores(
    rsi_4h: float = 50.0,
    rsi_1d: float = 50.0,
    macd_data: dict = None,
    volume_data: dict = None,
    stoch_rsi_data: dict = None,
    smc_data: dict = None,
    ema_data: dict = None,
    divergence_data: dict = None,
    bb_data: dict = None,
    funding_rate: float = None,
    fear_greed: int = None,
) -> dict:
    """
    Compute individual score for EACH confluence factor.
    Returns dict with keys like 'rsi_4h_score', 'macd_score', etc.
    Each score is in the range defined by its max weight.
    Also returns 'max_weights' so caller can normalize.
    """
    scores = {}
    reasons = {}

    # --- RSI 4h (max ±30) ---
    s = 0
    r = ""
    if rsi_4h <= RSI_STRONG_OVERSOLD:
        s = 30; r = f"RSI 4h stark überverkauft ({rsi_4h:.1f})"
    elif rsi_4h <= RSI_OVERSOLD:
        s = 20; r = f"RSI 4h überverkauft ({rsi_4h:.1f})"
    elif rsi_4h <= 45:
        s = 10; r = f"RSI 4h Richtung überverkauft ({rsi_4h:.1f})"
    elif rsi_4h >= RSI_STRONG_OVERBOUGHT:
        s = -30; r = f"RSI 4h stark überkauft ({rsi_4h:.1f})"
    elif rsi_4h >= RSI_OVERBOUGHT:
        s = -20; r = f"RSI 4h überkauft ({rsi_4h:.1f})"
    elif rsi_4h >= 60:
        s = -10; r = f"RSI 4h Richtung überkauft ({rsi_4h:.1f})"
    scores["rsi_4h"] = s
    reasons["rsi_4h"] = r

    # --- RSI 1D (max ±20) ---
    s = 0; r = ""
    if rsi_1d <= RSI_STRONG_OVERSOLD:
        s = 20; r = f"RSI 1D stark überverkauft ({rsi_1d:.1f})"
    elif rsi_1d <= RSI_OVERSOLD:
        s = 15; r = f"RSI 1D überverkauft ({rsi_1d:.1f})"
    elif rsi_1d >= RSI_STRONG_OVERBOUGHT:
        s = -20; r = f"RSI 1D stark überkauft ({rsi_1d:.1f})"
    elif rsi_1d >= RSI_OVERBOUGHT:
        s = -15; r = f"RSI 1D überkauft ({rsi_1d:.1f})"
    scores["rsi_1d"] = s
    reasons["rsi_1d"] = r

    # --- MACD (max ±20) ---
    md = macd_data or {"trend": "NEUTRAL", "histogram": 0}
    s = 0; r = ""
    if md["trend"] == "BULLISH":
        s = 15 + (5 if md["histogram"] > 0 else 0)
        r = "MACD bullish" + (" + Hist. steigend" if md["histogram"] > 0 else "")
    elif md["trend"] == "BEARISH":
        s = -(15 + (5 if md["histogram"] < 0 else 0))
        r = "MACD bearish" + (" + Hist. fallend" if md["histogram"] < 0 else "")
    scores["macd"] = s
    reasons["macd"] = r

    # --- Volume & OBV (max ±15) ---
    vd = volume_data or {"vol_trend": "NEUTRAL", "obv_trend": "NEUTRAL"}
    s = 0; r = ""
    if vd["vol_trend"] == "HIGH" and vd["obv_trend"] == "BULLISH":
        s = 15; r = "Hohes Vol + bullish OBV"
    elif vd["vol_trend"] == "HIGH" and vd["obv_trend"] == "BEARISH":
        s = -15; r = "Hohes Vol + bearish OBV"
    elif vd["obv_trend"] == "BULLISH":
        s = 8; r = "Bullish OBV"
    elif vd["obv_trend"] == "BEARISH":
        s = -8; r = "Bearish OBV"
    scores["volume_obv"] = s
    reasons["volume_obv"] = r

    # --- Stoch RSI (max ±12) ---
    sk = stoch_rsi_data or {"stoch_rsi_k": 50.0, "stoch_rsi_d": 50.0}
    s = 0; r = ""
    k_val = sk["stoch_rsi_k"]
    d_val = sk["stoch_rsi_d"]
    if k_val < 15:
        s = 12; r = f"Stoch RSI extrem überverkauft ({k_val:.0f})"
    elif k_val < 25:
        s = 8; r = f"Stoch RSI überverkauft ({k_val:.0f})"
    elif k_val > 85:
        s = -12; r = f"Stoch RSI extrem überkauft ({k_val:.0f})"
    elif k_val > 75:
        s = -8; r = f"Stoch RSI überkauft ({k_val:.0f})"
    # K/D crossover bonus
    if k_val < 30 and k_val > d_val:
        s = min(s + 3, 12); r += " + K>D Cross"
    elif k_val > 70 and k_val < d_val:
        s = max(s - 3, -12); r += " + K<D Cross"
    scores["stoch_rsi"] = s
    reasons["stoch_rsi"] = r

    # --- Smart Money (max ±15) ---
    s = 0; r = ""
    if smc_data:
        if smc_data.get("ob_signal") == "NEAR_BULLISH_OB":
            s += 10; r = "Nahe bullish Order Block"
        elif smc_data.get("ob_signal") == "NEAR_BEARISH_OB":
            s -= 10; r = "Nahe bearish Order Block"
        if smc_data.get("structure") == "BULLISH":
            s += 5; r += " + Bullish Struktur"
        elif smc_data.get("structure") == "BEARISH":
            s -= 5; r += " + Bearish Struktur"
        s = max(-15, min(15, s))
    scores["smart_money"] = s
    reasons["smart_money"] = r

    # --- EMA Alignment (max ±12) ---
    s = 0; r = ""
    if ema_data:
        if ema_data.get("ema_trend") == "BULLISH":
            bc = ema_data.get("ema_bull_count", 0)
            s = min(bc * 3, 12)
            r = f"EMA Alignment bullish ({bc}/4)"
        elif ema_data.get("ema_trend") == "BEARISH":
            bc = ema_data.get("ema_bear_count", 0)
            s = -min(bc * 3, 12)
            r = f"EMA Alignment bearish ({bc}/4)"
    scores["ema_alignment"] = s
    reasons["ema_alignment"] = r

    # --- RSI Divergence (max ±15) ---
    s = 0; r = ""
    if divergence_data:
        div = divergence_data.get("divergence", "NONE")
        dtype = divergence_data.get("div_type", "NONE")
        if div == "BULLISH":
            s = 15 if dtype == "REGULAR" else 10
            r = f"Bullish Divergenz ({dtype.lower()})"
        elif div == "BEARISH":
            s = -15 if dtype == "REGULAR" else -10
            r = f"Bearish Divergenz ({dtype.lower()})"
    scores["rsi_divergence"] = s
    reasons["rsi_divergence"] = r

    # --- Bollinger Squeeze (max ±10) ---
    s = 0; r = ""
    if bb_data:
        pct = bb_data.get("bb_pct", 50)
        squeeze = bb_data.get("bb_squeeze", False)
        if squeeze:
            r = f"BB Squeeze aktiv (Width: {bb_data.get('bb_width', 0):.1f}%)"
            # Squeeze alone is neutral — use price position for direction
            if pct < 30: s = 8; r += " → bullish Tendenz"
            elif pct > 70: s = -8; r += " → bearish Tendenz"
            else: s = 0; r += " → Ausbruch steht bevor"
        elif pct < 15:
            s = 10; r = f"BB unteres Band ({pct:.0f}%)"
        elif pct < 25:
            s = 5; r = f"BB untere Zone ({pct:.0f}%)"
        elif pct > 85:
            s = -10; r = f"BB oberes Band ({pct:.0f}%)"
        elif pct > 75:
            s = -5; r = f"BB obere Zone ({pct:.0f}%)"
    scores["bollinger"] = s
    reasons["bollinger"] = r

    # --- Funding Rate (max ±10) ---
    s = 0; r = ""
    if funding_rate is not None:
        if funding_rate > 0.05:
            s = -10; r = f"Funding Rate hoch ({funding_rate:.4f}) → viele Longs"
        elif funding_rate > 0.02:
            s = -5; r = f"Funding Rate leicht hoch ({funding_rate:.4f})"
        elif funding_rate < -0.05:
            s = 10; r = f"Funding Rate negativ ({funding_rate:.4f}) → viele Shorts"
        elif funding_rate < -0.02:
            s = 5; r = f"Funding Rate leicht negativ ({funding_rate:.4f})"
    scores["funding_rate"] = s
    reasons["funding_rate"] = r

    # --- Fear & Greed (max ±8) ---
    s = 0; r = ""
    if fear_greed is not None:
        if fear_greed <= 15:
            s = 8; r = f"Extreme Fear ({fear_greed}) → konträres Buy-Signal"
        elif fear_greed <= 30:
            s = 4; r = f"Fear ({fear_greed}) → eher Buy"
        elif fear_greed >= 85:
            s = -8; r = f"Extreme Greed ({fear_greed}) → konträres Sell-Signal"
        elif fear_greed >= 70:
            s = -4; r = f"Greed ({fear_greed}) → eher Sell"
    scores["fear_greed"] = s
    reasons["fear_greed"] = r

    # Max possible score per factor (for normalization)
    max_weights = {
        "rsi_4h": 30, "rsi_1d": 20, "macd": 20, "volume_obv": 15,
        "stoch_rsi": 12, "smart_money": 15, "ema_alignment": 12,
        "rsi_divergence": 15, "bollinger": 10, "funding_rate": 10,
        "fear_greed": 8,
    }

    return {"scores": scores, "reasons": reasons, "max_weights": max_weights}


def compute_confluence_total(individual: dict, active_filters: dict) -> dict:
    """
    Sum up only active filter scores and normalize to -100...+100.
    active_filters: {"rsi_4h": True, "macd": False, ...}
    """
    scores = individual["scores"]
    reasons = individual["reasons"]
    max_weights = individual["max_weights"]

    raw_score = 0
    max_possible = 0
    active_reasons = []

    for key, is_active in active_filters.items():
        if is_active and key in scores:
            raw_score += scores[key]
            max_possible += max_weights.get(key, 10)
            if reasons.get(key):
                active_reasons.append(reasons[key])

    # Normalize to -100 ... +100
    if max_possible > 0:
        normalized = round(raw_score / max_possible * 100)
    else:
        normalized = 0
    normalized = max(-100, min(100, normalized))

    # Recommendation
    if normalized >= 60: rec = "STRONG BUY"
    elif normalized >= 30: rec = "BUY"
    elif normalized >= 10: rec = "LEAN BUY"
    elif normalized <= -60: rec = "STRONG SELL"
    elif normalized <= -30: rec = "SELL"
    elif normalized <= -10: rec = "LEAN SELL"
    else: rec = "WAIT"

    return {
        "score": normalized,
        "raw_score": raw_score,
        "max_possible": max_possible,
        "recommendation": rec,
        "reasons": active_reasons,
        "active_count": sum(1 for v in active_filters.values() if v),
    }


# Legacy wrapper for backward compatibility
def generate_confluence_signal(rsi_4h, rsi_1d, macd_data, volume_data, smc_data=None):
    individual = compute_individual_scores(
        rsi_4h=rsi_4h, rsi_1d=rsi_1d,
        macd_data=macd_data, volume_data=volume_data,
        smc_data=smc_data,
    )
    default_filters = {"rsi_4h": True, "rsi_1d": True, "macd": True, "volume_obv": True, "smart_money": smc_data is not None}
    return compute_confluence_total(individual, default_filters)


# ============================================================
# DETAIL TAB: ADVANCED INDICATORS (computed on-demand per coin)
# ============================================================

def calculate_ema_crosses(df: pd.DataFrame) -> dict:
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
    if e9 > e21 and e9p <= e21p: c1 = "GOLDEN"
    elif e9 < e21 and e9p >= e21p: c1 = "DEATH"
    elif e9 > e21: c1 = "BULLISH"
    else: c1 = "BEARISH"
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
    if df.empty or len(df) < lookback:
        return {"fib_levels": {}, "fib_zone": "N/A"}
    recent = df.tail(lookback)
    swing_high = recent["high"].max()
    swing_low = recent["low"].min()
    diff = swing_high - swing_low
    if diff <= 0:
        return {"fib_levels": {}, "fib_zone": "N/A"}
    levels = {
        "0.0 (High)": round(swing_high, 6), "0.236": round(swing_high - 0.236 * diff, 6),
        "0.382": round(swing_high - 0.382 * diff, 6), "0.5": round(swing_high - 0.5 * diff, 6),
        "0.618": round(swing_high - 0.618 * diff, 6), "0.786": round(swing_high - 0.786 * diff, 6),
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


def calculate_sl_tp(price, atr, signal, sr_data):
    if atr <= 0 or price <= 0:
        return {"sl": 0, "tp1": 0, "tp2": 0, "tp3": 0, "risk_reward": 0}
    if signal in ("BUY", "CTB"):
        sl = sr_data["nearest_support"] if sr_data["nearest_support"] > 0 else price - 2 * atr
        sl = max(sl, price - 3 * atr)
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

