"""
🧙‍♂️ Merlin Crypto Scanner
RSI Heatmap + 24h Alerts Dashboard
CryptoWaves-style alert logic + CCXT multi-exchange data
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
# CryptoWaves-style ALERT LOGIC
# ============================================================
# Simple RSI-based: RSI 4h above threshold → SELL, below → BUY
CW_SELL_THRESHOLD = 58   # RSI 4h >= 58 → SELL alert
CW_BUY_THRESHOLD = 42    # RSI 4h <= 42 → BUY alert


def cw_alert_type(rsi_4h: float) -> str:
    """CryptoWaves-style alert: pure RSI 4h threshold."""
    if rsi_4h >= CW_SELL_THRESHOLD:
        return "SELL"
    elif rsi_4h <= CW_BUY_THRESHOLD:
        return "BUY"
    return "NONE"


def cw_border_intensity(rsi_4h: float) -> float:
    """
    Border color intensity 0.0–1.0 based on how extreme RSI is.
    RSI 58 → 0.3 (light), RSI 80 → 1.0 (full intensity) for SELL
    RSI 42 → 0.3 (light), RSI 20 → 1.0 (full intensity) for BUY
    """
    if rsi_4h >= CW_SELL_THRESHOLD:
        return min(1.0, 0.3 + (rsi_4h - CW_SELL_THRESHOLD) / (80 - CW_SELL_THRESHOLD) * 0.7)
    elif rsi_4h <= CW_BUY_THRESHOLD:
        return min(1.0, 0.3 + (CW_BUY_THRESHOLD - rsi_4h) / (CW_BUY_THRESHOLD - 20) * 0.7)
    return 0.0


# ============================================================
# SPARKLINE SVG
# ============================================================

def make_sparkline_svg(closes: list, width: int = 120, height: int = 36) -> str:
    if not closes or len(closes) < 3:
        return ""
    vals = [float(v) for v in closes if v and float(v) > 0]
    if len(vals) < 3:
        return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1
    pad = 2
    w, h = width - 2 * pad, height - 2 * pad
    pts = []
    for i, v in enumerate(vals):
        x = pad + (i / (len(vals) - 1)) * w
        y = pad + h - ((v - mn) / rng) * h
        pts.append(f"{x:.1f},{y:.1f}")
    path = "M" + "L".join(pts)
    color = "#00FF7F" if vals[-1] >= vals[0] else "#FF6347"
    return (f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/></svg>')


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .block-container { padding: 1rem; max-width: 100%; }

    .header-bar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px; padding: 12px 20px; margin-bottom: 12px;
        display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;
    }
    .header-title { font-size: 20px; font-weight: bold; color: #FFD700; }
    .header-stat { font-size: 13px; color: #ccc; }
    .header-stat b { color: white; }

    .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }
    .badge-neutral { background: #FFD70033; color: #FFD700; }
    .badge-bullish { background: #00FF7F33; color: #00FF7F; }
    .badge-bearish { background: #FF634733; color: #FF6347; }

    /* CW-style alert card with variable border intensity */
    .cw-alert {
        background: #1a1a2e; border-radius: 0 10px 10px 0;
        padding: 12px 16px; margin: 5px 0;
        display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
        border-left: 4px solid;
    }
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

    coin_list_mode = st.radio("📋 Coin List", ["CryptoWaves (107)", "Top 100 Dynamic", "Extended (180+)"], index=0)
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
            st.warning("Configure Telegram!")
    st.caption("Data: CCXT + CoinGecko")


# ============================================================
# SCAN FUNCTION
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
            market_lookup[mrow.get("symbol", "").upper()] = mrow

    for idx, symbol in enumerate(list(coins)):
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
            row["price"], row["change_24h"], row["volume_24h"] = 0, 0, 0

        if isinstance(mkt, pd.Series) and not mkt.empty:
            row["rank"] = int(mkt.get("rank", 999)) if pd.notna(mkt.get("rank")) else 999
            row["coin_name"] = str(mkt.get("name", symbol)) if pd.notna(mkt.get("name")) else symbol
            row["change_1h"] = float(mkt.get("change_1h", 0)) if pd.notna(mkt.get("change_1h")) else 0
            row["change_7d"] = float(mkt.get("change_7d", 0)) if pd.notna(mkt.get("change_7d")) else 0
            row["change_30d"] = float(mkt.get("change_30d", 0)) if pd.notna(mkt.get("change_30d")) else 0
            row["market_cap"] = float(mkt.get("market_cap", 0)) if pd.notna(mkt.get("market_cap")) else 0
        else:
            row["rank"], row["coin_name"] = 999, symbol
            row["change_1h"], row["change_7d"], row["change_30d"], row["market_cap"] = 0, 0, 0, 0

        if row["price"] == 0:
            continue

        # RSI + sparkline data per timeframe
        klines_data = {}
        for tf in timeframes_to_scan:
            tf_interval = TIMEFRAMES.get(tf, tf)
            df_kl = fetch_klines_smart(symbol, tf_interval)
            if not df_kl.empty and len(df_kl) >= 15:
                klines_data[tf] = df_kl
                from ta.momentum import RSIIndicator
                rsi_s = RSIIndicator(close=df_kl["close"], window=14).rsi().dropna()
                if len(rsi_s) >= 2:
                    row[f"rsi_{tf}"] = round(float(rsi_s.iloc[-1]), 2)
                    prev_idx = max(-5, -len(rsi_s))
                    row[f"rsi_prev_{tf}"] = round(float(rsi_s.iloc[prev_idx]), 2)
                else:
                    row[f"rsi_{tf}"], row[f"rsi_prev_{tf}"] = 50.0, 50.0
                row[f"closes_{tf}"] = json.dumps([round(c, 6) for c in df_kl["close"].tail(20).tolist()])
            else:
                row[f"rsi_{tf}"], row[f"rsi_prev_{tf}"] = 50.0, 50.0
                row[f"closes_{tf}"] = "[]"

        # --- CryptoWaves-style alert (simple RSI 4h threshold) ---
        rsi_4h_val = row.get("rsi_4h", 50.0)
        row["cw_alert"] = cw_alert_type(rsi_4h_val)
        row["cw_intensity"] = cw_border_intensity(rsi_4h_val)

        # --- Additional indicators (for Confluence tab) ---
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

            rsi_4h = row.get("rsi_4h", 50)
            rsi_1d = row.get("rsi_1D", 50)
            smc_c = {"ob_signal": row.get("ob_signal", "NONE"), "fvg_signal": row.get("fvg_signal", "BALANCED"),
                     "structure": row.get("market_structure", "UNKNOWN")} if include_smc else None
            conf = generate_confluence_signal(rsi_4h=rsi_4h, rsi_1d=rsi_1d, macd_data=macd_data, volume_data=vol_data, smc_data=smc_c)
            row["score"] = conf["score"]
            row["confluence_rec"] = conf["recommendation"]
            row["reasons"] = " | ".join(conf["reasons"][:3])
        else:
            row.update({"macd_trend": "NEUTRAL", "macd_histogram": 0, "vol_trend": "—", "vol_ratio": 1.0,
                        "obv_trend": "—", "stoch_rsi_k": 50.0, "stoch_rsi_d": 50.0,
                        "score": 0, "confluence_rec": "WAIT", "reasons": ""})

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
sell_alerts = len(df[df["cw_alert"] == "SELL"]) if "cw_alert" in df.columns else 0
buy_alerts = len(df[df["cw_alert"] == "BUY"]) if "cw_alert" in df.columns else 0
wait_count = len(df) - sell_alerts - buy_alerts

if avg_rsi_4h >= 60:
    market_label, badge_cls = "BULLISH", "badge-bullish"
elif avg_rsi_4h >= 45:
    market_label, badge_cls = "NEUTRAL", "badge-neutral"
else:
    market_label, badge_cls = "BEARISH", "badge-bearish"

avg_ch1h = df["change_1h"].mean() if "change_1h" in df.columns else 0
avg_ch24h = df["change_24h"].mean()
avg_ch7d = df["change_7d"].mean() if "change_7d" in df.columns else 0
avg_ch30d = df["change_30d"].mean() if "change_30d" in df.columns else 0
ex_status = get_exchange_status()
ex_name = ex_status["active_exchange"].upper() if ex_status["connected"] else "CoinGecko"

def ch_cls(v): return "change-pos" if v >= 0 else "change-neg"

st.markdown(f"""
<div class="header-bar">
    <div>
        <span class="header-title">🧙‍♂️ Merlin Crypto Scanner</span>
        <span class="badge {badge_cls}">Market: {market_label}</span>
    </div>
    <div class="header-stat">
        Avg RSI (4h): <b>{avg_rsi_4h:.2f}</b> |
        Ch%: <span class="{ch_cls(avg_ch1h)}">{avg_ch1h:+.2f}%</span>
        <span class="{ch_cls(avg_ch24h)}">{avg_ch24h:+.2f}%</span>
        <span class="{ch_cls(avg_ch7d)}">{avg_ch7d:+.2f}%</span>
        <span class="{ch_cls(avg_ch30d)}">{avg_ch30d:+.2f}%</span>
    </div>
    <div class="header-stat">
        <span style="color:#FF6347;">🔴 {sell_alerts}</span>
        <span style="color:#FFD700;">🟡 {wait_count}</span>
        <span style="color:#00FF7F;">🟢 {buy_alerts}</span>
        | 📡 {ex_name}
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
total_alerts = sell_alerts + buy_alerts
tab_alerts, tab_heatmap, tab_market, tab_confluence, tab_detail = st.tabs([
    f"🚨 24h Alerts {sell_alerts} 🔴 {buy_alerts} 🟢",
    "🔥 RSI Heatmap",
    "📊 By Market Cap",
    "🎯 Confluence",
    "🔍 Detail",
])


# ============================================================
# HELPER: Format price
# ============================================================
def fmt_price(p):
    if p >= 1000: return f"${p:,.2f}"
    elif p >= 1: return f"${p:,.4f}"
    elif p >= 0.001: return f"${p:,.6f}"
    else: return f"${p:.8f}"


# ============================================================
# HELPER: TradingView chart iframe
# ============================================================
def tradingview_chart(symbol: str, interval: str = "240"):
    """Generate TradingView advanced chart widget HTML."""
    pair = f"BINANCE:{symbol}USDT"
    return f"""
    <div style="height: 500px; background: #131722; border-radius: 8px; overflow: hidden;">
        <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tv_chart&symbol={pair}&interval={interval}&hidesidetoolbar=0&symboledit=1&saveimage=0&toolbarbg=131722&studies=%5B%22RSI%40tv-basicstudies%22%5D&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1&showpopupbutton=0&studies_overrides=%7B%7D&overrides=%7B%7D&enabled_features=%5B%5D&disabled_features=%5B%5D&locale=en&utm_source=merlin&utm_medium=widget&utm_campaign=chart" 
            style="width: 100%; height: 500px; border: none;">
        </iframe>
    </div>
    """


# ============================================================
# TAB 1: 24h ALERTS
# ============================================================
with tab_alerts:
    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.markdown("### 🚨 24h Alerts")
    with col_s:
        alert_sort = st.selectbox("Sort:", ["Alert Time", "RSI (4h)", "RSI (1D)", "Rank"], key="asort", label_visibility="collapsed")

    alert_df = df[df["cw_alert"] != "NONE"].copy()

    if alert_sort == "RSI (4h)" and "rsi_4h" in alert_df.columns:
        alert_df = alert_df.sort_values("rsi_4h", ascending=False)
    elif alert_sort == "RSI (1D)" and "rsi_1D" in alert_df.columns:
        alert_df = alert_df.sort_values("rsi_1D", ascending=False)
    elif alert_sort == "Rank":
        alert_df = alert_df.sort_values("rank")
    else:
        # Default: SELL first (highest RSI first), then BUY (lowest RSI first)
        sell_part = alert_df[alert_df["cw_alert"] == "SELL"].sort_values("rsi_4h", ascending=False)
        buy_part = alert_df[alert_df["cw_alert"] == "BUY"].sort_values("rsi_4h", ascending=True)
        alert_df = pd.concat([sell_part, buy_part])

    if alert_df.empty:
        st.info("No active alerts — all coins in neutral RSI zone (42–58).")
    else:
        st.caption(f"**{len(alert_df)}** active signals | 🔴 {sell_alerts} SELL  🟢 {buy_alerts} BUY")

        for _, row in alert_df.head(50).iterrows():
            cw = row["cw_alert"]
            is_buy = cw == "BUY"
            intensity = row.get("cw_intensity", 0.5)

            # Dynamic border color with intensity
            if is_buy:
                border_rgba = f"rgba(0, 255, 127, {intensity:.2f})"
                alert_label_color = "#00FF7F"
            else:
                border_rgba = f"rgba(255, 99, 71, {intensity:.2f})"
                alert_label_color = "#FF6347"

            sym = row["symbol"]
            name = row.get("coin_name", sym)
            rank = row.get("rank", 999)
            rank_str = f"#{int(rank)}" if rank < 999 else ""

            rsi4 = row.get("rsi_4h", 50)
            rsi1d = row.get("rsi_1D", 50)
            r4c = "#FF6347" if rsi4 > 70 else "#00FF7F" if rsi4 < 30 else "white"
            r1c = "#FF6347" if rsi1d > 70 else "#00FF7F" if rsi1d < 30 else "white"

            ch1h = row.get("change_1h", 0)
            ch24h = row.get("change_24h", 0)
            ch7d = row.get("change_7d", 0)
            ch30d = row.get("change_30d", 0)

            # Sparklines
            try: spark_7d = make_sparkline_svg(json.loads(row.get("closes_4h", "[]")), 100, 32)
            except Exception: spark_7d = ""
            try: spark_30d = make_sparkline_svg(json.loads(row.get("closes_1D", "[]")), 100, 32)
            except Exception: spark_30d = ""

            st.markdown(f"""
            <div class="cw-alert" style="border-left: 4px solid {border_rgba};">
                <div class="coin-block">
                    <div>
                        <span class="coin-name">{sym}</span>
                        <span class="coin-full">{name}</span>
                        <span class="coin-rank">{rank_str}</span>
                    </div>
                    <div class="price-line">Price: <b style="color:white;">{fmt_price(row['price'])}</b></div>
                    <div class="changes">
                        Ch%:
                        <span class="{ch_cls(ch1h)}">{ch1h:+.2f}%</span>
                        <span class="{ch_cls(ch24h)}">{ch24h:+.2f}%</span>
                        <span class="{ch_cls(ch7d)}" style="font-weight:bold;">{ch7d:+.2f}%</span>
                        <span class="{ch_cls(ch30d)}">{ch30d:+.2f}%</span>
                    </div>
                </div>
                <div class="chart-block">
                    <div><div class="chart-label">● 7d</div>{spark_7d}</div>
                    <div><div class="chart-label">● 30d</div>{spark_30d}</div>
                </div>
                <div class="signal-block">
                    <span class="alert-label" style="color: {alert_label_color};">Alert: {cw}</span>
                    <div class="rsi-line">RSI (4h): <b style="color:{r4c};">{rsi4:.2f}</b></div>
                    <div class="rsi-line">RSI (1D): <b style="color:{r1c};">{rsi1d:.2f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Clickable: show TradingView chart
            if st.button(f"📈 Chart {sym}", key=f"chart_{sym}", help=f"Open {sym} chart"):
                st.session_state["chart_coin"] = sym

    # Show TradingView chart if coin selected
    if st.session_state.get("chart_coin"):
        coin_sym = st.session_state["chart_coin"]
        coin_data = df[df["symbol"] == coin_sym]
        if not coin_data.empty:
            cr = coin_data.iloc[0]
            cw = cr.get("cw_alert", "NONE")
            sig_color = "#FF6347" if cw == "SELL" else "#00FF7F" if cw == "BUY" else "#FFD700"

            st.markdown(f"""
            <div style="background:#1a1a2e;border-radius:10px;padding:14px;margin:10px 0;">
                <div style="display:flex;flex-wrap:wrap;align-items:center;gap:12px;">
                    <b style="color:white;font-size:18px;">{coin_sym}</b>
                    <span style="color:#888;">{cr.get('coin_name', coin_sym)}</span>
                    <span class="coin-rank">#{int(cr.get('rank',999))}</span>
                    <span style="color:white;">Price: {fmt_price(cr['price'])}</span>
                    <span class="{ch_cls(cr.get('change_24h',0))}">{cr.get('change_24h',0):+.2f}%</span>
                    <span style="color:{sig_color};font-weight:bold;">Now: {cw}</span>
                    <span style="color:#888;">RSI (4h): <b style="color:{'#FF6347' if cr.get('rsi_4h',50)>70 else '#00FF7F' if cr.get('rsi_4h',50)<30 else 'white'};">{cr.get('rsi_4h',50):.2f}</b></span>
                    <span style="color:#888;">RSI (1D): <b>{cr.get('rsi_1D',50):.2f}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.components.v1.html(tradingview_chart(coin_sym), height=520)

        if st.button("✖ Close Chart"):
            del st.session_state["chart_coin"]
            st.rerun()

    # Telegram
    if st.session_state.get("send_summary") and telegram_token and telegram_chat_id:
        check_and_send_alerts(alert_df.to_dict("records"), telegram_token, telegram_chat_id, alert_min_score, send_summary=True)
        st.success("✅ Sent!")
        st.session_state["send_summary"] = False


# ============================================================
# TAB 2: RSI HEATMAP
# ============================================================
with tab_heatmap:
    hm_col1, hm_col2 = st.columns([3, 1])
    with hm_col1:
        heatmap_tf = st.selectbox("Timeframe", tf_to_scan, index=0, key="heatmap_tf")
    with hm_col2:
        hm_xaxis = st.selectbox("X-Axis", ["Random Spread", "Coin Rank"], index=0, key="hm_xaxis")

    rsi_col = f"rsi_{heatmap_tf}"
    rsi_prev_col = f"rsi_prev_{heatmap_tf}"

    if rsi_col in df.columns:
        avail = ["symbol", rsi_col, "price", "change_24h", "volume_24h", "cw_alert",
                 "rank", "coin_name", "change_1h", "change_7d", "change_30d"]
        if rsi_prev_col in df.columns: avail.append(rsi_prev_col)
        other_rsi = "rsi_1D" if heatmap_tf == "4h" else "rsi_4h"
        if other_rsi in df.columns: avail.append(other_rsi)
        avail = [c for c in avail if c in df.columns]
        plot_df = df[avail].copy().dropna(subset=[rsi_col])

        # X-axis: rank or random
        if hm_xaxis == "Coin Rank":
            plot_df["x_pos"] = plot_df["rank"].clip(upper=200)
        else:
            np.random.seed(42)
            plot_df["x_pos"] = np.random.uniform(0, 100, len(plot_df))

        # DOT COLOR: gray by default, colored only if has CW alert
        def dot_color(cw):
            if cw == "SELL": return "#FF6347"
            elif cw == "BUY": return "#00FF7F"
            return "#888888"
        plot_df["dot_color"] = plot_df["cw_alert"].apply(dot_color)

        # RSI delta for trend lines
        if rsi_prev_col in plot_df.columns:
            plot_df["rsi_delta"] = plot_df[rsi_col] - plot_df[rsi_prev_col]
        else:
            plot_df["rsi_delta"] = 0

        fig = go.Figure()

        # RSI zones
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(255,0,0,0.12)", line_width=0,
                      annotation_text="OVERBOUGHT", annotation_position="top right", annotation_font_color="#FF6347")
        fig.add_hrect(y0=70, y1=80, fillcolor="rgba(255,99,71,0.06)", line_width=0,
                      annotation_text="STRONG", annotation_position="top right", annotation_font_color="rgba(255,99,71,0.5)")
        fig.add_hrect(y0=60, y1=70, fillcolor="rgba(255,215,0,0.03)", line_width=0)
        fig.add_hrect(y0=30, y1=40, fillcolor="rgba(50,205,50,0.06)", line_width=0,
                      annotation_text="WEAK", annotation_position="bottom right", annotation_font_color="rgba(0,255,127,0.5)")
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,127,0.12)", line_width=0,
                      annotation_text="OVERSOLD", annotation_position="bottom right", annotation_font_color="#00FF7F")

        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,99,71,0.5)", line_width=1)
        fig.add_hline(y=60, line_dash="dash", line_color="rgba(255,99,71,0.25)", line_width=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,127,0.5)", line_width=1)
        fig.add_hline(y=40, line_dash="dash", line_color="rgba(0,255,127,0.25)", line_width=0.5)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)

        # Trend lines (RSI momentum indicator)
        for _, r in plot_df.iterrows():
            delta = r.get("rsi_delta", 0)
            if abs(delta) > 0.5:
                lc = "rgba(255,99,71,0.35)" if delta < 0 else "rgba(0,255,127,0.35)"
                fig.add_trace(go.Scatter(
                    x=[r["x_pos"], r["x_pos"]], y=[r[rsi_col] - delta, r[rsi_col]],
                    mode="lines", line=dict(color=lc, width=1, dash="dot"),
                    hoverinfo="skip", showlegend=False))

        # Build hover data
        custom = []
        for _, r in plot_df.iterrows():
            rk = f"#{int(r.get('rank',999))}" if r.get('rank',999) < 999 else ""
            o_rsi = r.get(other_rsi, 50) if other_rsi in plot_df.columns else 50
            o_tf = "1D" if heatmap_tf == "4h" else "4h"
            cw = r.get("cw_alert", "NONE")
            alert_str = f"{cw}" if cw != "NONE" else "none for 24 hours"
            custom.append([r["price"], r.get("coin_name", r["symbol"]), rk,
                           r[rsi_col], o_rsi, o_tf,
                           r.get("change_1h", 0), r.get("change_24h", 0),
                           r.get("change_7d", 0), r.get("change_30d", 0), alert_str])

        fig.add_trace(go.Scatter(
            x=plot_df["x_pos"], y=plot_df[rsi_col],
            mode="markers+text",
            text=plot_df["symbol"],
            textposition="top center",
            textfont=dict(size=9, color="white"),
            marker=dict(size=10, color=plot_df["dot_color"], opacity=0.9,
                        line=dict(width=1, color="rgba(255,255,255,0.3)")),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[1]} - %{text} (%{customdata[2]})</b><br>"
                f"RSI ({heatmap_tf}): " + "%{customdata[3]:.2f} | RSI (%{customdata[5]}): %{customdata[4]:.2f}<br>"
                "Price: $%{customdata[0]:,.4f}<br>"
                "1h, 24h, 7d, 30d: %{customdata[6]:+.2f}%, %{customdata[7]:+.2f}%, %{customdata[8]:+.2f}%, %{customdata[9]:+.2f}%<br>"
                "Latest Alert: %{customdata[10]}<br>"
                "<extra></extra>"),
        ))

        avg_val = plot_df[rsi_col].mean()
        fig.add_hline(y=avg_val, line_dash="dashdot", line_color="rgba(255,215,0,0.6)", line_width=1.5,
                      annotation_text=f"AVG RSI: {avg_val:.1f}", annotation_font_color="#FFD700")

        x_title = "Coin Rank" if hm_xaxis == "Coin Rank" else ""
        fig.update_layout(
            title=dict(text=f"Crypto Market RSI({heatmap_tf}) Heatmap<br><sup>{datetime.now().strftime('%d/%m/%Y %H:%M')} UTC</sup>",
                       font=dict(size=16, color="white"), x=0.5),
            template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=700,
            xaxis=dict(showticklabels=(hm_xaxis == "Coin Rank"), showgrid=False, zeroline=False, title=x_title),
            yaxis=dict(title=f"RSI ({heatmap_tf})", range=[15, 90], gridcolor="rgba(255,255,255,0.05)"),
            showlegend=False, margin=dict(l=50, r=20, t=60, b=30))

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
        sym, name = row["symbol"], row.get("coin_name", row["symbol"])
        rank = row.get("rank", 999)
        rk = f"#{int(rank)}" if rank < 999 else ""
        rsi4 = row.get("rsi_4h", 50)
        rsi1d = row.get("rsi_1D", 50)
        cw = row.get("cw_alert", "NONE")
        sig_color = "#FF6347" if cw == "SELL" else "#00FF7F" if cw == "BUY" else "#FFD700"
        sig_label = cw if cw != "NONE" else "WAIT"
        r4c = "#FF6347" if rsi4 > 70 else "#00FF7F" if rsi4 < 30 else "white"
        r1c = "#FF6347" if rsi1d > 70 else "#00FF7F" if rsi1d < 30 else "white"
        ch1h, ch24h = row.get("change_1h", 0), row.get("change_24h", 0)
        ch7d, ch30d = row.get("change_7d", 0), row.get("change_30d", 0)

        st.markdown(f"""
        <div class="market-row">
            <div style="min-width:180px;">
                <span style="font-size:15px;font-weight:bold;color:white;">{sym}</span>
                <span style="font-size:12px;color:#888;">{name}</span>
                <span class="coin-rank">{rk}</span><br>
                <span style="font-size:12px;color:#aaa;">Price: {fmt_price(row['price'])}</span><br>
                <span style="font-size:11px;color:#888;">Ch%:
                    <span class="{ch_cls(ch1h)}">{ch1h:+.2f}%</span>
                    <span class="{ch_cls(ch24h)}">{ch24h:+.2f}%</span>
                    <span class="{ch_cls(ch7d)}" style="font-weight:bold;">{ch7d:+.2f}%</span>
                    <span class="{ch_cls(ch30d)}">{ch30d:+.2f}%</span></span>
            </div>
            <div style="text-align:right;min-width:160px;">
                <span style="font-size:12px;color:#888;">Now:</span>
                <span style="font-size:14px;font-weight:bold;color:{sig_color};">{sig_label}</span><br>
                <span style="font-size:12px;color:#888;">RSI (4h): <b style="color:{r4c};">{rsi4:.2f}</b></span>
                <span style="font-size:12px;color:#888;">RSI (1D): <b style="color:{r1c};">{rsi1d:.2f}</b></span>
            </div>
        </div>""", unsafe_allow_html=True)


# ============================================================
# TAB 4: CONFLUENCE (advanced multi-indicator)
# ============================================================
with tab_confluence:
    st.markdown("### 🎯 Confluence Scanner")
    st.caption("Advanced multi-indicator analysis (RSI + MACD + Volume + SMC)")

    st.markdown("#### 🟢 Top Buy Signals")
    for _, row in df[df["score"] > 0].sort_values("score", ascending=False).head(15).iterrows():
        s = row["score"]
        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;margin:4px 0;">
            <div style="display:flex;justify-content:space-between;"><b style="color:white;">{row['symbol']}</b>
            <span style="color:#00FF7F;font-weight:bold;">{row.get('confluence_rec','WAIT')} ({s})</span></div>
            <div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:6px;">
                <div style="background:linear-gradient(90deg,#00FF7F,#32CD32);height:6px;width:{min(abs(s),100)}%;border-radius:4px;"></div></div>
            <div style="font-size:11px;color:#888;margin-top:4px;">RSI 4h: {row.get('rsi_4h',50):.1f} | MACD: {row.get('macd_trend','—')} | Vol: {row.get('vol_trend','—')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔴 Top Sell Signals")
    for _, row in df[df["score"] < 0].sort_values("score").head(15).iterrows():
        s = abs(row["score"])
        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;margin:4px 0;">
            <div style="display:flex;justify-content:space-between;"><b style="color:white;">{row['symbol']}</b>
            <span style="color:#FF6347;font-weight:bold;">{row.get('confluence_rec','WAIT')} ({row['score']})</span></div>
            <div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:6px;">
                <div style="background:linear-gradient(90deg,#FF6347,#FF0000);height:6px;width:{min(s,100)}%;border-radius:4px;"></div></div>
            <div style="font-size:11px;color:#888;margin-top:4px;">RSI 4h: {row.get('rsi_4h',50):.1f} | MACD: {row.get('macd_trend','—')} | Vol: {row.get('vol_trend','—')}</div>
        </div>""", unsafe_allow_html=True)


# ============================================================
# TAB 5: COIN DETAIL with TradingView Chart
# ============================================================
with tab_detail:
    sel = st.selectbox("Select Coin", df["symbol"].tolist(), key="dcoin")
    if sel:
        c = df[df["symbol"] == sel].iloc[0]
        cw = c.get("cw_alert", "NONE")
        sig_color = "#FF6347" if cw == "SELL" else "#00FF7F" if cw == "BUY" else "#FFD700"
        sig_label = cw if cw != "NONE" else "WAIT"

        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:10px;padding:16px;margin-bottom:12px;">
            <div style="display:flex;flex-wrap:wrap;align-items:center;gap:14px;">
                <h2 style="margin:0;color:white;">{sel}</h2>
                <span style="color:#888;">{c.get('coin_name',sel)}</span>
                <span class="coin-rank">#{int(c.get('rank',999))}</span>
                <span style="font-size:20px;color:white;font-weight:bold;">{fmt_price(c['price'])}</span>
                <span class="{ch_cls(c.get('change_24h',0))}" style="font-size:16px;">{c.get('change_24h',0):+.2f}%</span>
                <span style="color:{sig_color};font-weight:bold;font-size:16px;">Now: {sig_label}</span>
                <span style="color:#888;">RSI (4h): <b style="color:{'#FF6347' if c.get('rsi_4h',50)>70 else '#00FF7F' if c.get('rsi_4h',50)<30 else 'white'};">{c.get('rsi_4h',50):.2f}</b></span>
                <span style="color:#888;">RSI (1D): <b>{c.get('rsi_1D',50):.2f}</b></span>
            </div>
        </div>""", unsafe_allow_html=True)

        # TradingView chart
        st.components.v1.html(tradingview_chart(sel), height=520)

        # RSI gauges
        st.markdown("#### RSI Multi-Timeframe")
        rsi_cols = [col for col in df.columns if col.startswith("rsi_") and "prev" not in col and "closes" not in col]
        gcols = st.columns(len(rsi_cols))
        for i, rc in enumerate(rsi_cols):
            v = c.get(rc, 50)
            with gcols[i]:
                fg = go.Figure(go.Indicator(mode="gauge+number", value=v,
                    title={"text": rc.replace("rsi_", "RSI "), "font": {"size": 14, "color": "white"}},
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
            | BOS | **{'✅' if c.get('bos') else '—'}** |
            """)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#555;font-size:11px;'>🧙‍♂️ Merlin Crypto Scanner | {len(coins_to_scan)} coins × {len(tf_to_scan)} TFs | {ex_name} + CoinGecko | Alert: RSI 4h ≥{CW_SELL_THRESHOLD}→SELL ≤{CW_BUY_THRESHOLD}→BUY | DYOR!</div>", unsafe_allow_html=True)
