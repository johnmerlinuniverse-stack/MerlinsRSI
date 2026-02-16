"""
🧙‍♂️ Merlin Crypto Scanner
RSI Heatmap + 24h Alerts Dashboard
Inspired by CryptoWaves.app — with CCXT multi-exchange data
"""

import time
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from config import (
    PAGE_TITLE, PAGE_ICON, CRYPTOWAVES_COINS, TOP_COINS_EXTENDED,
    TIMEFRAMES, COLORS,
    RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_STRONG_OVERBOUGHT, RSI_STRONG_OVERSOLD,
)
from data_fetcher import (
    fetch_all_market_data, fetch_klines_smart, fetch_all_tickers,
    get_exchange_status,
)
from indicators import (
    calculate_rsi, calculate_macd, calculate_volume_analysis,
    detect_order_blocks, detect_fair_value_gaps, detect_market_structure,
    generate_confluence_signal, calculate_stoch_rsi,
)
from alerts import (
    check_and_send_alerts, format_summary_alert, send_telegram_alert,
)


# ============================================================
# SPARKLINE SVG GENERATOR
# ============================================================

def make_sparkline_svg(closes: list, width: int = 120, height: int = 36) -> str:
    """Generate inline SVG sparkline from close prices."""
    if not closes or len(closes) < 3:
        return ""
    
    vals = [float(v) for v in closes if v and float(v) > 0]
    if len(vals) < 3:
        return ""
    
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1
    
    # Normalize to SVG coordinates
    padding = 2
    w = width - 2 * padding
    h = height - 2 * padding
    
    points = []
    for i, v in enumerate(vals):
        x = padding + (i / (len(vals) - 1)) * w
        y = padding + h - ((v - mn) / rng) * h
        points.append(f"{x:.1f},{y:.1f}")
    
    path = "M" + "L".join(points)
    
    # Color: green if trending up, red if down
    color = "#00FF7F" if vals[-1] >= vals[0] else "#FF6347"
    
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>'
        f'</svg>'
    )


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .block-container { padding: 1rem; max-width: 100%; }
    
    .header-bar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 12px;
        display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;
    }
    .header-title { font-size: 20px; font-weight: bold; color: #FFD700; }
    .header-stat { font-size: 13px; color: #ccc; }
    .header-stat b { color: white; }
    
    .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }
    .badge-neutral { background: #FFD70033; color: #FFD700; }
    .badge-bullish { background: #00FF7F33; color: #00FF7F; }
    .badge-bearish { background: #FF634733; color: #FF6347; }
    
    /* Alert card — CryptoWaves style */
    .cw-alert {
        background: #1a1a2e;
        border-left: 4px solid #FF6347;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin: 5px 0;
        display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    }
    .cw-alert-buy { border-color: #00FF7F; }
    .cw-alert .coin-block { min-width: 200px; flex: 1; }
    .cw-alert .coin-name { font-size: 16px; font-weight: bold; color: white; }
    .cw-alert .coin-full { font-size: 12px; color: #888; margin-left: 6px; }
    .cw-alert .coin-rank { font-size: 10px; background: #2a2a4a; padding: 1px 6px; border-radius: 6px; color: #888; margin-left: 4px; }
    .cw-alert .price-line { font-size: 12px; color: #aaa; margin-top: 2px; }
    .cw-alert .changes { font-size: 11px; color: #888; margin-top: 2px; }
    
    .cw-alert .chart-block { display: flex; gap: 12px; align-items: center; }
    .cw-alert .chart-label { font-size: 10px; color: #666; }
    
    .cw-alert .signal-block { text-align: right; min-width: 160px; }
    .cw-alert .alert-label { font-size: 14px; font-weight: bold; }
    .cw-alert .rsi-line { font-size: 12px; color: #888; margin-top: 2px; }
    .cw-alert .rsi-line b { font-size: 14px; }
    
    .change-pos { color: #00FF7F; }
    .change-neg { color: #FF6347; }
    
    /* Market row */
    .market-row {
        background: #1a1a2e; border-radius: 10px; padding: 12px 16px; margin: 4px 0;
        display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; font-weight: 600; }
    
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem; }
        .cw-alert .chart-block { display: none; }
        .cw-alert .coin-block { min-width: 140px; }
        .cw-alert .signal-block { min-width: 120px; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧙‍♂️ Settings")
    st.markdown("---")
    
    coin_list_mode = st.radio(
        "📋 Coin List",
        ["CryptoWaves (107)", "Top 100 Dynamic", "Extended (180+)"],
        index=0,
    )
    
    if coin_list_mode == "CryptoWaves (107)":
        coin_source = CRYPTOWAVES_COINS
    elif coin_list_mode == "Top 100 Dynamic":
        coin_source = CRYPTOWAVES_COINS[:50]
    else:
        coin_source = TOP_COINS_EXTENDED
    
    max_coins = st.slider("Max Coins", 20, 180, min(len(coin_source), 80), 10)
    
    st.markdown("---")
    selected_timeframes = st.multiselect("Timeframes", list(TIMEFRAMES.keys()), default=["4h", "1D"])
    show_smc = st.checkbox("Smart Money Concepts", value=False)
    
    st.markdown("---")
    st.markdown("### 📱 Telegram")
    telegram_token = st.text_input("Bot Token", type="password", key="tg_token")
    telegram_chat_id = st.text_input("Chat ID", key="tg_chat")
    alert_min_score = st.slider("Min. Score", 10, 80, 30, 5)
    
    st.markdown("---")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    if st.button("📤 Telegram Summary", use_container_width=True):
        if telegram_token and telegram_chat_id:
            st.session_state["send_summary"] = True
        else:
            st.warning("Configure Telegram first!")
    st.caption("Data: CCXT + CoinGecko")


# ============================================================
# SCAN FUNCTION (with momentum + sparkline data)
# ============================================================

@st.cache_data(ttl=180, show_spinner="🧙‍♂️ Scanning crypto market...")
def scan_all_coins(coins: tuple, timeframes_to_scan: tuple, include_smc: bool = False) -> pd.DataFrame:
    results = []

    ex_status = get_exchange_status()
    ex_connected = ex_status["connected"]
    market_df = fetch_all_market_data()
    tickers = fetch_all_tickers() if ex_connected else {}

    market_lookup = {}
    if not market_df.empty:
        for _, mrow in market_df.iterrows():
            sym = mrow.get("symbol", "").upper()
            market_lookup[sym] = mrow

    coins_list = list(coins)

    for idx, symbol in enumerate(coins_list):
        row = {"symbol": symbol}

        ticker = tickers.get(symbol, {})
        mkt = market_lookup.get(symbol, {})

        if ticker:
            row["price"] = float(ticker.get("last", 0) or 0)
            row["change_24h"] = float(ticker.get("change_pct", 0) or 0)
            row["volume_24h"] = float(ticker.get("volume", 0) or 0)
        elif isinstance(mkt, pd.Series) and not mkt.empty:
            row["price"] = float(mkt.get("price", 0)) if pd.notna(mkt.get("price")) else 0
            row["change_24h"] = float(mkt.get("change_24h", 0)) if pd.notna(mkt.get("change_24h")) else 0
            row["volume_24h"] = float(mkt.get("volume_24h", 0)) if pd.notna(mkt.get("volume_24h")) else 0
        else:
            row["price"] = 0
            row["change_24h"] = 0
            row["volume_24h"] = 0

        if isinstance(mkt, pd.Series) and not mkt.empty:
            row["rank"] = int(mkt.get("rank", 999)) if pd.notna(mkt.get("rank")) else 999
            row["coin_name"] = str(mkt.get("name", symbol)) if pd.notna(mkt.get("name")) else symbol
            row["change_1h"] = float(mkt.get("change_1h", 0)) if pd.notna(mkt.get("change_1h")) else 0
            row["change_7d"] = float(mkt.get("change_7d", 0)) if pd.notna(mkt.get("change_7d")) else 0
            row["change_30d"] = float(mkt.get("change_30d", 0)) if pd.notna(mkt.get("change_30d")) else 0
            row["market_cap"] = float(mkt.get("market_cap", 0)) if pd.notna(mkt.get("market_cap")) else 0
        else:
            row["rank"] = 999
            row["coin_name"] = symbol
            row["change_1h"] = 0
            row["change_7d"] = 0
            row["change_30d"] = 0
            row["market_cap"] = 0

        if row["price"] == 0:
            continue

        # RSI per timeframe + previous RSI for trend lines + sparkline closes
        klines_data = {}
        for tf in timeframes_to_scan:
            tf_interval = TIMEFRAMES.get(tf, tf)
            df_klines = fetch_klines_smart(symbol, tf_interval)
            if not df_klines.empty and len(df_klines) >= 15:
                klines_data[tf] = df_klines

                # Current RSI
                from ta.momentum import RSIIndicator
                rsi_ind = RSIIndicator(close=df_klines["close"], window=14)
                rsi_series = rsi_ind.rsi().dropna()
                if len(rsi_series) >= 2:
                    row[f"rsi_{tf}"] = round(float(rsi_series.iloc[-1]), 2)
                    # Previous RSI (4 candles back) for trend line
                    prev_idx = max(-5, -len(rsi_series))
                    row[f"rsi_prev_{tf}"] = round(float(rsi_series.iloc[prev_idx]), 2)
                else:
                    row[f"rsi_{tf}"] = 50.0
                    row[f"rsi_prev_{tf}"] = 50.0

                # Store last 20 close prices for sparklines (as JSON string)
                closes = df_klines["close"].tail(20).tolist()
                row[f"closes_{tf}"] = json.dumps([round(c, 6) for c in closes])
            else:
                row[f"rsi_{tf}"] = 50.0
                row[f"rsi_prev_{tf}"] = 50.0
                row[f"closes_{tf}"] = "[]"

        # MACD + Volume + SMC + Confluence
        primary_tf = "4h" if "4h" in klines_data else (list(timeframes_to_scan)[0] if timeframes_to_scan else None)
        if primary_tf and primary_tf in klines_data:
            macd_data = calculate_macd(klines_data[primary_tf])
            row["macd_trend"] = macd_data["trend"]
            row["macd_histogram"] = macd_data["histogram"]

            vol_data = calculate_volume_analysis(klines_data[primary_tf])
            row["vol_trend"] = vol_data["vol_trend"]
            row["vol_ratio"] = vol_data["vol_ratio"]
            row["obv_trend"] = vol_data["obv_trend"]

            stoch = calculate_stoch_rsi(klines_data[primary_tf])
            row["stoch_rsi_k"] = stoch["stoch_rsi_k"]
            row["stoch_rsi_d"] = stoch["stoch_rsi_d"]

            if include_smc:
                ob = detect_order_blocks(klines_data[primary_tf])
                fvg = detect_fair_value_gaps(klines_data[primary_tf])
                ms = detect_market_structure(klines_data[primary_tf])
                row["ob_signal"] = ob["ob_signal"]
                row["fvg_signal"] = fvg["fvg_signal"]
                row["market_structure"] = ms["structure"]
                row["bos"] = ms["break_of_structure"]

            rsi_4h = row.get("rsi_4h", row.get(f"rsi_{primary_tf}", 50))
            rsi_1d = row.get("rsi_1D", 50)
            smc_combined = {
                "ob_signal": row.get("ob_signal", "NONE"),
                "fvg_signal": row.get("fvg_signal", "BALANCED"),
                "structure": row.get("market_structure", "UNKNOWN"),
            } if include_smc else None

            confluence = generate_confluence_signal(
                rsi_4h=rsi_4h, rsi_1d=rsi_1d,
                macd_data=macd_data, volume_data=vol_data, smc_data=smc_combined,
            )
            row["score"] = confluence["score"]
            row["recommendation"] = confluence["recommendation"]
            row["reasons"] = " | ".join(confluence["reasons"][:3])
        else:
            row.update({
                "macd_trend": "NEUTRAL", "macd_histogram": 0,
                "vol_trend": "—", "vol_ratio": 1.0, "obv_trend": "—",
                "stoch_rsi_k": 50.0, "stoch_rsi_d": 50.0,
                "score": 0, "recommendation": "WAIT", "reasons": "",
            })

        results.append(row)

        if not ex_connected and idx % 5 == 0 and idx > 0:
            time.sleep(2)
        elif idx % 15 == 0 and idx > 0:
            time.sleep(0.3)

    return pd.DataFrame(results) if results else pd.DataFrame()


# ============================================================
# LOAD DATA
# ============================================================
coins_to_scan = coin_source[:max_coins]
tf_to_scan = selected_timeframes if selected_timeframes else ["4h", "1D"]

df = scan_all_coins(tuple(coins_to_scan), tuple(tf_to_scan), include_smc=show_smc)

if df.empty:
    st.warning("⚠️ No data. Reduce coins to 30–50 and click Refresh.")
    st.stop()


# ============================================================
# HEADER BAR
# ============================================================
avg_rsi_4h = df["rsi_4h"].mean() if "rsi_4h" in df.columns else 50
ob_count = len(df[df["rsi_4h"] >= RSI_OVERBOUGHT]) if "rsi_4h" in df.columns else 0
os_count = len(df[df["rsi_4h"] <= RSI_OVERSOLD]) if "rsi_4h" in df.columns else 0
neutral_count = len(df) - ob_count - os_count
alert_count = len(df[df["recommendation"] != "WAIT"])

if avg_rsi_4h >= 60:
    market_label, badge_cls = "BULLISH", "badge-bullish"
elif avg_rsi_4h >= 45:
    market_label, badge_cls = "NEUTRAL", "badge-neutral"
else:
    market_label, badge_cls = "BEARISH", "badge-bearish"

avg_ch1h = df["change_1h"].mean() if "change_1h" in df.columns else 0
avg_ch24h = df["change_24h"].mean() if "change_24h" in df.columns else 0
avg_ch7d = df["change_7d"].mean() if "change_7d" in df.columns else 0
avg_ch30d = df["change_30d"].mean() if "change_30d" in df.columns else 0
ex_status = get_exchange_status()
ex_name = ex_status["active_exchange"].upper() if ex_status["connected"] else "CoinGecko"

def ch_cls(v):
    return "change-pos" if v >= 0 else "change-neg"

st.markdown(f"""
<div class="header-bar">
    <div>
        <span class="header-title">🧙‍♂️ Merlin Crypto Scanner</span>
        <span class="badge {badge_cls}">Market: {market_label}</span>
    </div>
    <div class="header-stat">
        Avg RSI (4h): <b>{avg_rsi_4h:.1f}</b> |
        Ch%: <span class="{ch_cls(avg_ch1h)}">{avg_ch1h:+.2f}%</span>
        <span class="{ch_cls(avg_ch24h)}">{avg_ch24h:+.2f}%</span>
        <span class="{ch_cls(avg_ch7d)}">{avg_ch7d:+.2f}%</span>
        <span class="{ch_cls(avg_ch30d)}">{avg_ch30d:+.2f}%</span>
    </div>
    <div class="header-stat">
        <span style="color:#FF6347;">🔴 {ob_count}</span>
        <span style="color:#FFD700;">🟡 {neutral_count}</span>
        <span style="color:#00FF7F;">🟢 {os_count}</span>
        | 📡 {ex_name}
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab_alerts, tab_heatmap, tab_market, tab_confluence, tab_detail = st.tabs([
    f"🚨 24h Alerts ({alert_count})",
    "🔥 RSI Heatmap",
    "📊 By Market Cap",
    "🎯 Confluence",
    "🔍 Detail",
])


# ============================================================
# TAB 1: 24h ALERTS (CryptoWaves style with sparklines)
# ============================================================
with tab_alerts:
    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.markdown("### 🚨 24h Alerts")
    with col_s:
        alert_sort = st.selectbox("Sort:", ["Alert Strength", "RSI (4h)", "RSI (1D)", "Rank"], key="asort", label_visibility="collapsed")

    alert_df = df[df["recommendation"] != "WAIT"].copy()
    
    if alert_sort == "RSI (4h)" and "rsi_4h" in alert_df.columns:
        alert_df = alert_df.sort_values("rsi_4h", ascending=False)
    elif alert_sort == "RSI (1D)" and "rsi_1D" in alert_df.columns:
        alert_df = alert_df.sort_values("rsi_1D", ascending=False)
    elif alert_sort == "Rank":
        alert_df = alert_df.sort_values("rank")
    else:
        alert_df = alert_df.sort_values("score", key=abs, ascending=False)

    if alert_df.empty:
        st.info("No active alerts — market in WAIT mode.")
    else:
        st.caption(f"**{len(alert_df)}** active signals")

        for _, row in alert_df.head(50).iterrows():
            rec = row.get("recommendation", "WAIT")
            is_buy = "BUY" in rec
            alert_color = "#00FF7F" if is_buy else "#FF6347"
            alert_label = "BUY" if is_buy else "SELL"
            card_cls = "cw-alert-buy" if is_buy else ""

            sym = row["symbol"]
            name = row.get("coin_name", sym)
            rank = row.get("rank", 999)
            rank_str = f"#{int(rank)}" if rank < 999 else ""
            price = row["price"]
            
            # Format price
            if price >= 1000: ps = f"${price:,.2f}"
            elif price >= 1: ps = f"${price:,.4f}"
            elif price >= 0.001: ps = f"${price:,.6f}"
            else: ps = f"${price:.8f}"

            rsi4 = row.get("rsi_4h", 50)
            rsi1d = row.get("rsi_1D", 50)
            ch1h = row.get("change_1h", 0)
            ch24h = row.get("change_24h", 0)
            ch7d = row.get("change_7d", 0)
            ch30d = row.get("change_30d", 0)

            r4c = "#FF6347" if rsi4 > 70 else "#00FF7F" if rsi4 < 30 else "white"
            r1c = "#FF6347" if rsi1d > 70 else "#00FF7F" if rsi1d < 30 else "white"

            # Sparklines
            closes_4h_raw = row.get("closes_4h", "[]")
            closes_1d_raw = row.get("closes_1D", "[]")
            try:
                spark_4h = make_sparkline_svg(json.loads(closes_4h_raw), 100, 32)
            except Exception:
                spark_4h = ""
            try:
                spark_1d = make_sparkline_svg(json.loads(closes_1d_raw), 100, 32)
            except Exception:
                spark_1d = ""

            st.markdown(f"""
            <div class="cw-alert {card_cls}">
                <div class="coin-block">
                    <div>
                        <span class="coin-name">{sym}</span>
                        <span class="coin-full">{name}</span>
                        <span class="coin-rank">{rank_str}</span>
                    </div>
                    <div class="price-line">Price: <b style="color:white;">{ps}</b></div>
                    <div class="changes">
                        Ch%:
                        <span class="{ch_cls(ch1h)}">{ch1h:+.2f}%</span>
                        <span class="{ch_cls(ch24h)}">{ch24h:+.2f}%</span>
                        <span class="{ch_cls(ch7d)}" style="font-weight:bold;">{ch7d:+.2f}%</span>
                        <span class="{ch_cls(ch30d)}">{ch30d:+.2f}%</span>
                    </div>
                </div>
                <div class="chart-block">
                    <div>
                        <div class="chart-label">● 7d</div>
                        {spark_4h}
                    </div>
                    <div>
                        <div class="chart-label">● 30d</div>
                        {spark_1d}
                    </div>
                </div>
                <div class="signal-block">
                    <div>
                        <span class="alert-label" style="color: {alert_color};">Alert: {alert_label}</span>
                    </div>
                    <div class="rsi-line">
                        RSI (4h): <b style="color:{r4c};">{rsi4:.2f}</b>
                    </div>
                    <div class="rsi-line">
                        RSI (1D): <b style="color:{r1c};">{rsi1d:.2f}</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.get("send_summary") and telegram_token and telegram_chat_id:
        check_and_send_alerts(alert_df.to_dict("records"), telegram_token, telegram_chat_id, alert_min_score, send_summary=True)
        st.success("✅ Sent!")
        st.session_state["send_summary"] = False


# ============================================================
# TAB 2: RSI HEATMAP (colored dots + trend lines + rich hover)
# ============================================================
with tab_heatmap:
    heatmap_tf = st.selectbox("Timeframe", tf_to_scan, index=0, key="heatmap_tf")
    rsi_col = f"rsi_{heatmap_tf}"
    rsi_prev_col = f"rsi_prev_{heatmap_tf}"

    if rsi_col in df.columns:
        cols_needed = ["symbol", rsi_col, "price", "change_24h", "volume_24h",
                       "recommendation", "score", "rank", "coin_name",
                       "change_1h", "change_7d", "change_30d"]
        if rsi_prev_col in df.columns:
            cols_needed.append(rsi_prev_col)
        # Add other RSI if available
        other_rsi = "rsi_1D" if heatmap_tf == "4h" else "rsi_4h"
        if other_rsi in df.columns:
            cols_needed.append(other_rsi)

        avail_cols = [c for c in cols_needed if c in df.columns]
        plot_df = df[avail_cols].copy().dropna(subset=[rsi_col])

        np.random.seed(42)
        plot_df["x_pos"] = np.random.uniform(0, 100, len(plot_df))

        # Color by SIGNAL (red=sell, green=buy, gray=neutral)
        def dot_color(rec):
            if "BUY" in str(rec):
                return "#00FF7F"
            elif "SELL" in str(rec):
                return "#FF6347"
            else:
                return "#888888"

        plot_df["dot_color"] = plot_df["recommendation"].apply(dot_color)

        # Calculate trend line endpoints (RSI movement direction)
        if rsi_prev_col in plot_df.columns:
            plot_df["rsi_delta"] = plot_df[rsi_col] - plot_df[rsi_prev_col]
        else:
            plot_df["rsi_delta"] = 0

        fig = go.Figure()

        # RSI zones
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(255,0,0,0.12)", line_width=0,
                      annotation_text="OVERBOUGHT", annotation_position="top right",
                      annotation_font_color="#FF6347")
        fig.add_hrect(y0=70, y1=80, fillcolor="rgba(255,99,71,0.06)", line_width=0,
                      annotation_text="STRONG", annotation_position="top right",
                      annotation_font_color="rgba(255,99,71,0.5)")
        fig.add_hrect(y0=60, y1=70, fillcolor="rgba(255,215,0,0.03)", line_width=0)
        fig.add_hrect(y0=30, y1=40, fillcolor="rgba(50,205,50,0.06)", line_width=0,
                      annotation_text="WEAK", annotation_position="bottom right",
                      annotation_font_color="rgba(0,255,127,0.5)")
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,127,0.12)", line_width=0,
                      annotation_text="OVERSOLD", annotation_position="bottom right",
                      annotation_font_color="#00FF7F")

        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,99,71,0.5)", line_width=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,127,0.5)", line_width=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)

        # TREND LINES — vertical dashed lines showing RSI momentum
        for _, r in plot_df.iterrows():
            delta = r.get("rsi_delta", 0)
            if abs(delta) > 0.5:  # Only show if meaningful change
                line_color = "rgba(255,99,71,0.4)" if delta < 0 else "rgba(0,255,127,0.4)"
                y_start = r[rsi_col]
                y_end = r[rsi_col] - delta  # Line goes FROM previous TO current
                fig.add_trace(go.Scatter(
                    x=[r["x_pos"], r["x_pos"]],
                    y=[y_end, y_start],
                    mode="lines",
                    line=dict(color=line_color, width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                ))

        # Build hover data
        hover_texts = []
        custom_data = []
        for _, r in plot_df.iterrows():
            rank = r.get("rank", 999)
            rank_str = f"#{int(rank)}" if rank < 999 else ""
            name = r.get("coin_name", r["symbol"])
            other_rsi_val = r.get(other_rsi, 50) if other_rsi in plot_df.columns else 50
            other_tf = "1D" if heatmap_tf == "4h" else "4h"
            rec = r.get("recommendation", "WAIT")

            hover_texts.append(r["symbol"])
            custom_data.append([
                r["price"], name, rank_str,
                r[rsi_col], other_rsi_val, other_tf,
                r.get("change_1h", 0), r.get("change_24h", 0),
                r.get("change_7d", 0), r.get("change_30d", 0),
                rec,
            ])

        # DOTS — colored by signal
        fig.add_trace(go.Scatter(
            x=plot_df["x_pos"],
            y=plot_df[rsi_col],
            mode="markers+text",
            text=plot_df["symbol"],
            textposition="top center",
            textfont=dict(size=9, color="white"),
            marker=dict(
                size=10,
                color=plot_df["dot_color"],
                opacity=0.9,
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            customdata=custom_data,
            hovertemplate=(
                "<b>%{customdata[1]} - %{text} (%{customdata[2]})</b><br>"
                f"RSI ({heatmap_tf}): " + "%{customdata[3]:.2f} | RSI (%{customdata[5]}): %{customdata[4]:.2f}<br>"
                "Price: $%{customdata[0]:,.4f}<br>"
                "1h, 24h, 7d, 30d: %{customdata[6]:+.2f}%, %{customdata[7]:+.2f}%, %{customdata[8]:+.2f}%, %{customdata[9]:+.2f}%<br>"
                "Signal: %{customdata[10]}<br>"
                "<extra></extra>"
            ),
        ))

        avg_val = plot_df[rsi_col].mean()
        fig.add_hline(y=avg_val, line_dash="dashdot", line_color="rgba(255,215,0,0.6)",
                      line_width=1.5,
                      annotation_text=f"AVG RSI: {avg_val:.1f}",
                      annotation_font_color="#FFD700")

        fig.update_layout(
            title=dict(
                text=f"Crypto Market RSI({heatmap_tf}) Heatmap<br><sup>{datetime.now().strftime('%d/%m/%Y %H:%M')} UTC</sup>",
                font=dict(size=16, color="white"), x=0.5,
            ),
            template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=700,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(title=f"RSI ({heatmap_tf})", range=[15, 90], gridcolor="rgba(255,255,255,0.05)"),
            showlegend=False,
            margin=dict(l=50, r=20, t=60, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🔴 Most Overbought**")
            for _, r in plot_df.nlargest(5, rsi_col).iterrows():
                st.markdown(f"`{r['symbol']}` RSI: **{r[rsi_col]:.1f}**")
        with c2:
            st.markdown("**🟢 Most Oversold**")
            for _, r in plot_df.nsmallest(5, rsi_col).iterrows():
                st.markdown(f"`{r['symbol']}` RSI: **{r[rsi_col]:.1f}**")
        with c3:
            st.markdown("**📊 Distribution**")
            st.markdown(f"RSI > 70: **{len(plot_df[plot_df[rsi_col] > 70])}**")
            st.markdown(f"RSI 30–70: **{len(plot_df[(plot_df[rsi_col] >= 30) & (plot_df[rsi_col] <= 70)])}**")
            st.markdown(f"RSI < 30: **{len(plot_df[plot_df[rsi_col] < 30])}**")


# ============================================================
# TAB 3: BY MARKET CAP
# ============================================================
with tab_market:
    sort_market = st.selectbox("Sort:", ["Rank", "1h", "24h", "7d", "30d", "RSI (4h)", "RSI (1D)"], key="msort")

    display_df = df.copy()
    sm = {"Rank": ("rank", True), "1h": ("change_1h", False), "24h": ("change_24h", False),
          "7d": ("change_7d", False), "30d": ("change_30d", False),
          "RSI (4h)": ("rsi_4h", False), "RSI (1D)": ("rsi_1D", False)}
    sc, sa = sm.get(sort_market, ("rank", True))
    if sc in display_df.columns:
        display_df = display_df.sort_values(sc, ascending=sa)

    for _, row in display_df.iterrows():
        sym = row["symbol"]
        name = row.get("coin_name", sym)
        price = row["price"]
        rank = row.get("rank", 999)
        rank_str = f"#{int(rank)}" if rank < 999 else ""

        if price >= 1000: ps = f"${price:,.2f}"
        elif price >= 1: ps = f"${price:,.4f}"
        elif price >= 0.001: ps = f"${price:,.6f}"
        else: ps = f"${price:.8f}"

        ch1h = row.get("change_1h", 0)
        ch24h = row.get("change_24h", 0)
        ch7d = row.get("change_7d", 0)
        ch30d = row.get("change_30d", 0)
        rsi4 = row.get("rsi_4h", 50)
        rsi1d = row.get("rsi_1D", 50)
        rec = row.get("recommendation", "WAIT")

        sig_color = "#00FF7F" if "BUY" in rec else "#FF6347" if "SELL" in rec else "#FFD700"
        sig_label = "CTB" if "STRONG" in rec and "BUY" in rec else rec.replace("LEAN ", "").replace("STRONG ", "")
        r4c = "#FF6347" if rsi4 > 70 else "#00FF7F" if rsi4 < 30 else "white"
        r1c = "#FF6347" if rsi1d > 70 else "#00FF7F" if rsi1d < 30 else "white"

        st.markdown(f"""
        <div class="market-row">
            <div style="min-width:180px;">
                <span style="font-size:15px;font-weight:bold;color:white;">{sym}</span>
                <span style="font-size:12px;color:#888;">{name}</span>
                <span class="coin-rank">{rank_str}</span>
                <br>
                <span style="font-size:12px;color:#aaa;">Price: {ps}</span>
                <br>
                <span style="font-size:11px;color:#888;">
                    Ch%: <span class="{ch_cls(ch1h)}">{ch1h:+.2f}%</span>
                    <span class="{ch_cls(ch24h)}">{ch24h:+.2f}%</span>
                    <span class="{ch_cls(ch7d)}" style="font-weight:bold;">{ch7d:+.2f}%</span>
                    <span class="{ch_cls(ch30d)}">{ch30d:+.2f}%</span>
                </span>
            </div>
            <div style="text-align:right;min-width:160px;">
                <span style="font-size:12px;color:#888;">Now:</span>
                <span style="font-size:14px;font-weight:bold;color:{sig_color};">{sig_label}</span>
                <br>
                <span style="font-size:12px;color:#888;">RSI (4h): <b style="color:{r4c};">{rsi4:.2f}</b></span>
                <span style="font-size:12px;color:#888;">RSI (1D): <b style="color:{r1c};">{rsi1d:.2f}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 4: CONFLUENCE
# ============================================================
with tab_confluence:
    st.markdown("### 🎯 Confluence Scanner")

    st.markdown("#### 🟢 Top Buy")
    for _, row in df[df["score"] > 0].sort_values("score", ascending=False).head(15).iterrows():
        s = row["score"]
        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;margin:4px 0;">
            <div style="display:flex;justify-content:space-between;"><b style="color:white;">{row['symbol']}</b>
            <span style="color:#00FF7F;font-weight:bold;">{row['recommendation']} ({s})</span></div>
            <div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:6px;">
                <div style="background:linear-gradient(90deg,#00FF7F,#32CD32);height:6px;width:{min(abs(s),100)}%;border-radius:4px;"></div></div>
            <div style="font-size:11px;color:#888;margin-top:4px;">RSI 4h: {row.get('rsi_4h',50):.1f} | MACD: {row.get('macd_trend','—')} | Vol: {row.get('vol_trend','—')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔴 Top Sell")
    for _, row in df[df["score"] < 0].sort_values("score").head(15).iterrows():
        s = abs(row["score"])
        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;margin:4px 0;">
            <div style="display:flex;justify-content:space-between;"><b style="color:white;">{row['symbol']}</b>
            <span style="color:#FF6347;font-weight:bold;">{row['recommendation']} ({row['score']})</span></div>
            <div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:6px;">
                <div style="background:linear-gradient(90deg,#FF6347,#FF0000);height:6px;width:{min(s,100)}%;border-radius:4px;"></div></div>
            <div style="font-size:11px;color:#888;margin-top:4px;">RSI 4h: {row.get('rsi_4h',50):.1f} | MACD: {row.get('macd_trend','—')} | Vol: {row.get('vol_trend','—')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(x=df["score"], nbinsx=30, marker_color=["#00FF7F" if x > 20 else "#FF6347" if x < -20 else "#FFD700" for x in sorted(df["score"])], opacity=0.8))
    fig_d.add_vline(x=0, line_dash="dash", line_color="white", line_width=1)
    fig_d.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=280, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_d, use_container_width=True)


# ============================================================
# TAB 5: DETAIL
# ============================================================
with tab_detail:
    sel = st.selectbox("Coin", df["symbol"].tolist(), key="dcoin")
    if sel:
        c = df[df["symbol"] == sel].iloc[0]
        rec = c.get("recommendation", "WAIT")
        rc = {"STRONG BUY": "#00FF7F", "BUY": "#32CD32", "LEAN BUY": "#90EE90", "WAIT": "#FFD700",
              "LEAN SELL": "#FFA07A", "SELL": "#FF6347", "STRONG SELL": "#FF0000"}.get(rec, "#FFD700")

        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:10px;padding:16px;margin-bottom:16px;">
            <div style="display:flex;flex-wrap:wrap;align-items:center;gap:16px;">
                <h2 style="margin:0;color:white;">{sel}/USDT</h2>
                <span style="font-size:22px;color:white;font-weight:bold;">${c['price']:,.4f}</span>
                <span style="font-size:16px;color:{'#00FF7F' if c.get('change_24h',0)>=0 else '#FF6347'};">{c.get('change_24h',0):+.2f}%</span>
                <span style="background:{rc}22;color:{rc};padding:5px 14px;border-radius:16px;font-weight:bold;border:1px solid {rc}44;">{rec} ({c.get('score',0)})</span>
            </div>
        </div>""", unsafe_allow_html=True)

        rsi_cols = [col for col in df.columns if col.startswith("rsi_") and "prev" not in col and "closes" not in col]
        gcols = st.columns(len(rsi_cols))
        for i, rcol in enumerate(rsi_cols):
            v = c.get(rcol, 50)
            with gcols[i]:
                fg = go.Figure(go.Indicator(mode="gauge+number", value=v,
                    title={"text": rcol.replace("rsi_","RSI "), "font": {"size": 14, "color": "white"}},
                    number={"font": {"size": 22, "color": "white"}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#FFD700"}, "bgcolor": "#1a1a2e",
                           "steps": [{"range": [0, 30], "color": "rgba(0,255,127,0.2)"},
                                     {"range": [30, 70], "color": "rgba(255,215,0,0.1)"},
                                     {"range": [70, 100], "color": "rgba(255,99,71,0.2)"}]}))
                fg.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", height=180, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fg, use_container_width=True)

        st.markdown(f"""
        | Indicator | Value |
        |-----------|-------|
        | MACD | **{c.get('macd_trend','—')}** (hist: {c.get('macd_histogram',0):.4f}) |
        | Stoch RSI | K: **{c.get('stoch_rsi_k',50):.1f}** / D: **{c.get('stoch_rsi_d',50):.1f}** |
        | Volume | **{c.get('vol_trend','—')}** ({c.get('vol_ratio',1.0):.2f}x) |
        | OBV | **{c.get('obv_trend','—')}** |
        """)

        if show_smc:
            st.markdown(f"""
            | SMC | Value |
            |-----|-------|
            | Structure | **{c.get('market_structure','—')}** |
            | Order Block | **{c.get('ob_signal','—')}** |
            | FVG | **{c.get('fvg_signal','—')}** |
            | BOS | **{'✅' if c.get('bos', False) else '—'}** |
            """)

        if c.get("reasons"):
            st.markdown("**Signal Reasoning:**")
            for r in str(c["reasons"]).split(" | "):
                if r.strip():
                    st.markdown(f"- {r.strip()}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#555;font-size:11px;'>🧙‍♂️ Merlin Crypto Scanner | {len(coins_to_scan)} coins × {len(tf_to_scan)} TFs | {ex_name} + CoinGecko | DYOR!</div>", unsafe_allow_html=True)
