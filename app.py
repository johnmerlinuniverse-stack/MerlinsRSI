"""
🧙‍♂️ Merlin Crypto Scanner
RSI Heatmap + 24h Alerts Dashboard
Inspired by CryptoWaves.app — with CCXT multi-exchange data
"""

import time
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
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# MOBILE-FRIENDLY CSS
# ============================================================
st.markdown("""
<style>
    /* Dark theme */
    .stApp { background-color: #0E1117; }
    
    /* Reduce padding for mobile */
    .block-container { padding: 1rem 1rem 1rem 1rem; max-width: 100%; }
    
    /* Header bar like CryptoWaves */
    .header-bar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 12px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
    }
    .header-title {
        font-size: 20px;
        font-weight: bold;
        color: #FFD700;
    }
    .header-stat {
        font-size: 13px;
        color: #ccc;
    }
    .header-stat b { color: white; }
    
    /* Status badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-neutral { background: #FFD70033; color: #FFD700; }
    .badge-bullish { background: #00FF7F33; color: #00FF7F; }
    .badge-bearish { background: #FF634733; color: #FF6347; }
    
    /* Alert card */
    .alert-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 6px 0;
        border-left: 4px solid;
    }
    .alert-card-buy { border-color: #FF6347; }
    .alert-card-sell { border-color: #FF6347; }
    .alert-card .coin-name {
        font-size: 16px;
        font-weight: bold;
        color: white;
    }
    .alert-card .coin-rank {
        font-size: 11px;
        background: #2a2a4a;
        padding: 1px 6px;
        border-radius: 6px;
        color: #888;
        margin-left: 6px;
    }
    .alert-card .alert-type {
        font-size: 14px;
        font-weight: bold;
    }
    .alert-sell { color: #FF6347; }
    .alert-buy { color: #00FF7F; }
    .alert-card .rsi-values {
        font-size: 13px;
        color: #aaa;
        margin-top: 6px;
    }
    .alert-card .rsi-values b { color: white; }
    .rsi-warn { color: #FF6347 !important; }
    .rsi-ok { color: #00FF7F !important; }
    
    /* Market row card (for By Market Cap view) */
    .market-row {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 4px 0;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    }
    .market-row .coin-info {
        min-width: 140px;
    }
    .market-row .price-info {
        text-align: right;
        min-width: 120px;
    }
    .market-row .rsi-info {
        text-align: right;
        min-width: 160px;
    }
    .change-pos { color: #00FF7F; }
    .change-neg { color: #FF6347; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem; }
        .header-bar { padding: 10px 12px; }
        .header-title { font-size: 16px; }
        .market-row { padding: 10px 12px; }
        .alert-card { padding: 10px 12px; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR (collapsed by default on mobile)
# ============================================================
with st.sidebar:
    st.markdown("## 🧙‍♂️ Settings")
    st.markdown("---")
    
    # Coin list selection
    coin_list_mode = st.radio(
        "📋 Coin List",
        ["CryptoWaves (107)", "Top 100 Dynamic", "Extended (180+)"],
        index=0,
    )
    
    if coin_list_mode == "CryptoWaves (107)":
        coin_source = CRYPTOWAVES_COINS
    elif coin_list_mode == "Top 100 Dynamic":
        coin_source = CRYPTOWAVES_COINS[:50]  # Will be replaced by CoinGecko top 100
    else:
        coin_source = TOP_COINS_EXTENDED
    
    max_coins = st.slider("Max Coins to Scan", 20, 180, min(len(coin_source), 80), 10)
    
    st.markdown("---")
    
    # Timeframes
    selected_timeframes = st.multiselect(
        "Timeframes",
        options=list(TIMEFRAMES.keys()),
        default=["4h", "1D"],
    )
    
    show_smc = st.checkbox("Smart Money Concepts", value=False)
    
    st.markdown("---")
    
    # Telegram
    st.markdown("### 📱 Telegram Alerts")
    telegram_token = st.text_input("Bot Token", type="password", key="tg_token")
    telegram_chat_id = st.text_input("Chat ID", key="tg_chat")
    alert_min_score = st.slider("Min. Alert Score", 10, 80, 30, 5)
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("📤 Send Telegram Summary", use_container_width=True):
        if telegram_token and telegram_chat_id:
            st.session_state["send_summary"] = True
        else:
            st.warning("Configure Telegram first!")
    
    st.markdown("---")
    st.caption("Data: CCXT (auto-exchange) + CoinGecko")


# ============================================================
# SCAN FUNCTION
# ============================================================

@st.cache_data(ttl=180, show_spinner="🧙‍♂️ Scanning crypto market...")
def scan_all_coins(coins: tuple, timeframes_to_scan: tuple, include_smc: bool = False) -> pd.DataFrame:
    """Scan all coins with auto-fallback exchange data."""
    results = []

    ex_status = get_exchange_status()
    ex_connected = ex_status["connected"]

    market_df = fetch_all_market_data()
    tickers = fetch_all_tickers() if ex_connected else {}

    # Build market data lookup
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

        # Price data
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

        # Market cap rank from CoinGecko
        if isinstance(mkt, pd.Series) and not mkt.empty:
            row["rank"] = int(mkt.get("rank", 999)) if pd.notna(mkt.get("rank")) else 999
            row["change_1h"] = float(mkt.get("change_1h", 0)) if pd.notna(mkt.get("change_1h")) else 0
            row["change_7d"] = float(mkt.get("change_7d", 0)) if pd.notna(mkt.get("change_7d")) else 0
            row["change_30d"] = float(mkt.get("change_30d", 0)) if pd.notna(mkt.get("change_30d")) else 0
            row["market_cap"] = float(mkt.get("market_cap", 0)) if pd.notna(mkt.get("market_cap")) else 0
        else:
            row["rank"] = 999
            row["change_1h"] = 0
            row["change_7d"] = 0
            row["change_30d"] = 0
            row["market_cap"] = 0

        if row["price"] == 0:
            continue

        # RSI per timeframe
        klines_data = {}
        for tf in timeframes_to_scan:
            tf_interval = TIMEFRAMES.get(tf, tf)
            df_klines = fetch_klines_smart(symbol, tf_interval)
            if not df_klines.empty:
                klines_data[tf] = df_klines
                row[f"rsi_{tf}"] = calculate_rsi(df_klines)
            else:
                row[f"rsi_{tf}"] = 50.0

        # MACD + Volume + StochRSI + SMC
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
                macd_data=macd_data, volume_data=vol_data,
                smc_data=smc_combined,
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

        # Rate limiting
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

df = scan_all_coins(
    tuple(coins_to_scan),
    tuple(tf_to_scan),
    include_smc=show_smc,
)

if df.empty:
    st.warning("⚠️ No data loaded. Try reducing coins to 30-50 and click Refresh.")
    st.stop()


# ============================================================
# HEADER BAR (like CryptoWaves top bar)
# ============================================================
avg_rsi_4h = df["rsi_4h"].mean() if "rsi_4h" in df.columns else 50
avg_rsi_1d = df["rsi_1D"].mean() if "rsi_1D" in df.columns else 50
ob_count = len(df[df.get("rsi_4h", pd.Series(dtype=float)) >= RSI_OVERBOUGHT]) if "rsi_4h" in df.columns else 0
os_count = len(df[df.get("rsi_4h", pd.Series(dtype=float)) <= RSI_OVERSOLD]) if "rsi_4h" in df.columns else 0
neutral_count = len(df) - ob_count - os_count
buy_count = len(df[df["score"] > 0])
sell_count = len(df[df["score"] < 0])

if avg_rsi_4h >= 60:
    market_label, badge_cls = "BULLISH", "badge-bullish"
elif avg_rsi_4h >= 45:
    market_label, badge_cls = "NEUTRAL", "badge-neutral"
else:
    market_label, badge_cls = "BEARISH", "badge-bearish"

# Average price changes
avg_change_1h = df["change_1h"].mean() if "change_1h" in df.columns else 0
avg_change_24h = df["change_24h"].mean() if "change_24h" in df.columns else 0
avg_change_7d = df["change_7d"].mean() if "change_7d" in df.columns else 0
avg_change_30d = df["change_30d"].mean() if "change_30d" in df.columns else 0

ex_status = get_exchange_status()
ex_name = ex_status["active_exchange"].upper() if ex_status["connected"] else "CoinGecko"

st.markdown(f"""
<div class="header-bar">
    <div>
        <span class="header-title">🧙‍♂️ Merlin Crypto Scanner</span>
        <span class="badge {badge_cls}" style="margin-left: 10px;">Market: {market_label}</span>
    </div>
    <div class="header-stat">
        Avg RSI (4h): <b>{avg_rsi_4h:.1f}</b>
        &nbsp;|&nbsp;
        Ch%: <span class="{'change-pos' if avg_change_1h >= 0 else 'change-neg'}">{avg_change_1h:+.2f}%</span>
        <span class="{'change-pos' if avg_change_24h >= 0 else 'change-neg'}">{avg_change_24h:+.2f}%</span>
        <span class="{'change-pos' if avg_change_7d >= 0 else 'change-neg'}">{avg_change_7d:+.2f}%</span>
        <span class="{'change-pos' if avg_change_30d >= 0 else 'change-neg'}">{avg_change_30d:+.2f}%</span>
    </div>
    <div class="header-stat">
        <span style="color: #FF6347;">🔴 {ob_count}</span> &nbsp;
        <span style="color: #FFD700;">🟡 {neutral_count}</span> &nbsp;
        <span style="color: #00FF7F;">🟢 {os_count}</span> &nbsp;
        | 📡 {ex_name}
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS — 24h Alerts is FIRST (main focus)
# ============================================================
tab_alerts, tab_heatmap, tab_market, tab_confluence, tab_detail = st.tabs([
    f"🚨 24h Alerts ({buy_count + sell_count})",
    "🔥 RSI Heatmap",
    "📊 By Market Cap",
    "🎯 Confluence",
    "🔍 Detail",
])


# ============================================================
# TAB 1: 24h ALERTS (MAIN FOCUS)
# ============================================================
with tab_alerts:
    # RSI interval selector (like CryptoWaves)
    col_title, col_filter = st.columns([3, 1])
    with col_title:
        st.markdown("### 🚨 24h Alerts")
    with col_filter:
        alert_rsi_tf = st.selectbox(
            "Set RSI Interval",
            ["4h/1D", "1h/4h"],
            index=0,
            key="alert_rsi_tf",
            label_visibility="collapsed",
        )
    
    # Sort options (like CryptoWaves)
    sort_options = ["Alert Strength", "RSI (4h)", "RSI (1D)", "24h Change", "7d Change", "Rank"]
    sort_choice = st.selectbox("Sort by:", sort_options, index=0, key="alert_sort")
    
    # Filter signals
    alert_df = df[df["recommendation"] != "WAIT"].copy()
    
    # Sort
    if sort_choice == "Alert Strength":
        alert_df = alert_df.sort_values("score", key=abs, ascending=False)
    elif sort_choice == "RSI (4h)" and "rsi_4h" in alert_df.columns:
        alert_df = alert_df.sort_values("rsi_4h", ascending=False)
    elif sort_choice == "RSI (1D)" and "rsi_1D" in alert_df.columns:
        alert_df = alert_df.sort_values("rsi_1D", ascending=False)
    elif sort_choice == "24h Change":
        alert_df = alert_df.sort_values("change_24h", ascending=False)
    elif sort_choice == "7d Change":
        alert_df = alert_df.sort_values("change_7d", ascending=False)
    elif sort_choice == "Rank":
        alert_df = alert_df.sort_values("rank", ascending=True)
    
    if alert_df.empty:
        st.info("No active alerts. Market is in WAIT mode across all coins.")
    else:
        st.caption(f"**{len(alert_df)}** active signals | 🔴 SELL  🟢 BUY")
        
        for _, row in alert_df.head(50).iterrows():
            rec = row.get("recommendation", "WAIT")
            score = row.get("score", 0)
            is_buy = "BUY" in rec
            is_sell = "SELL" in rec
            
            alert_color = "#00FF7F" if is_buy else "#FF6347"
            alert_label = "BUY" if is_buy else "SELL"
            alert_cls = "alert-buy" if is_buy else "alert-sell"
            
            rsi_4h = row.get("rsi_4h", 50)
            rsi_1d = row.get("rsi_1D", 50)
            rank = row.get("rank", "—")
            rank_str = f"#{rank}" if isinstance(rank, (int, float)) and rank < 999 else ""
            
            price = row["price"]
            if price >= 1000:
                price_str = f"${price:,.2f}"
            elif price >= 1:
                price_str = f"${price:,.4f}"
            elif price >= 0.001:
                price_str = f"${price:,.6f}"
            else:
                price_str = f"${price:.8f}"
            
            ch1h = row.get("change_1h", 0)
            ch24h = row.get("change_24h", 0)
            ch7d = row.get("change_7d", 0)
            ch30d = row.get("change_30d", 0)
            
            # Color RSI values
            rsi_4h_color = "#FF6347" if rsi_4h > 70 else "#00FF7F" if rsi_4h < 30 else "white"
            rsi_1d_color = "#FF6347" if rsi_1d > 70 else "#00FF7F" if rsi_1d < 30 else "white"
            
            st.markdown(f"""
            <div class="alert-card" style="border-color: {alert_color};">
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px;">
                    <div>
                        <span class="coin-name">{row['symbol']}</span>
                        <span class="coin-rank">{rank_str}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <span style="color: white; font-weight: bold;">{price_str}</span>
                        <span class="alert-type {'alert-buy' if is_buy else 'alert-sell'}">Alert: {alert_label}</span>
                    </div>
                </div>
                <div style="margin-top: 6px; font-size: 12px; color: #888;">
                    Ch%:
                    <span class="{'change-pos' if ch1h >= 0 else 'change-neg'}">{ch1h:+.2f}%</span>
                    <span class="{'change-pos' if ch24h >= 0 else 'change-neg'}">{ch24h:+.2f}%</span>
                    <span class="{'change-pos' if ch7d >= 0 else 'change-neg'}" style="font-weight: bold;">{ch7d:+.2f}%</span>
                    <span class="{'change-pos' if ch30d >= 0 else 'change-neg'}">{ch30d:+.2f}%</span>
                </div>
                <div class="rsi-values" style="display: flex; justify-content: flex-end; gap: 16px;">
                    <span>RSI (4h): <b style="color: {rsi_4h_color};">{rsi_4h:.2f}</b></span>
                    <span>RSI (1D): <b style="color: {rsi_1d_color};">{rsi_1d:.2f}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Telegram summary
    if st.session_state.get("send_summary") and telegram_token and telegram_chat_id:
        all_data = alert_df.to_dict("records")
        check_and_send_alerts(all_data, telegram_token, telegram_chat_id, alert_min_score, send_summary=True)
        st.success("✅ Summary sent to Telegram!")
        st.session_state["send_summary"] = False


# ============================================================
# TAB 2: RSI HEATMAP (same-size dots)
# ============================================================
with tab_heatmap:
    heatmap_tf = st.selectbox("Timeframe", tf_to_scan, index=0, key="heatmap_tf")
    rsi_col = f"rsi_{heatmap_tf}"
    
    if rsi_col in df.columns:
        plot_df = df[["symbol", rsi_col, "price", "change_24h", "volume_24h"]].copy()
        plot_df = plot_df.dropna(subset=[rsi_col])
        
        # Spread coins horizontally
        np.random.seed(42)
        plot_df["x_pos"] = np.random.uniform(0, 100, len(plot_df))
        
        # Color by RSI zone
        def rsi_color(val):
            if val >= RSI_STRONG_OVERBOUGHT:
                return COLORS["strong_sell"]
            elif val >= RSI_OVERBOUGHT:
                return COLORS["sell"]
            elif val <= RSI_STRONG_OVERSOLD:
                return COLORS["strong_buy"]
            elif val <= RSI_OVERSOLD:
                return COLORS["buy"]
            else:
                return COLORS["neutral"]
        
        plot_df["color"] = plot_df[rsi_col].apply(rsi_color)
        
        fig = go.Figure()
        
        # RSI zones
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(255,0,0,0.12)", line_width=0,
                      annotation_text="OVERBOUGHT", annotation_position="top right",
                      annotation_font_color="#FF6347")
        fig.add_hrect(y0=70, y1=80, fillcolor="rgba(255,99,71,0.06)", line_width=0,
                      annotation_text="STRONG", annotation_position="top right",
                      annotation_font_color="#FF634777")
        fig.add_hrect(y0=60, y1=70, fillcolor="rgba(255,215,0,0.03)", line_width=0)
        fig.add_hrect(y0=30, y1=40, fillcolor="rgba(50,205,50,0.06)", line_width=0,
                      annotation_text="WEAK", annotation_position="bottom right",
                      annotation_font_color="#00FF7F77")
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,127,0.12)", line_width=0,
                      annotation_text="OVERSOLD", annotation_position="bottom right",
                      annotation_font_color="#00FF7F")
        
        # Reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,99,71,0.5)", line_width=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,127,0.5)", line_width=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)
        
        # ALL DOTS SAME SIZE (fix #2)
        fig.add_trace(go.Scatter(
            x=plot_df["x_pos"],
            y=plot_df[rsi_col],
            mode="markers+text",
            text=plot_df["symbol"],
            textposition="top center",
            textfont=dict(size=9, color="white"),
            marker=dict(
                size=10,  # FIXED SIZE — no volume scaling
                color=plot_df["color"],
                opacity=0.9,
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"RSI ({heatmap_tf}): " + "%{y:.1f}<br>"
                "Price: $%{customdata[0]:,.4f}<br>"
                "24h: %{customdata[1]:+.2f}%<br>"
                "<extra></extra>"
            ),
            customdata=np.column_stack([plot_df["price"], plot_df["change_24h"]]),
        ))
        
        # Average RSI line
        avg_val = plot_df[rsi_col].mean()
        fig.add_hline(y=avg_val, line_dash="dashdot", line_color="rgba(255,215,0,0.6)",
                      line_width=1.5,
                      annotation_text=f"AVG RSI: {avg_val:.1f}",
                      annotation_font_color="#FFD700")
        
        fig.update_layout(
            title=dict(
                text=f"Crypto Market RSI({heatmap_tf}) Heatmap<br><sup>{datetime.now().strftime('%d/%m/%Y %H:%M')} UTC by Merlin Scanner</sup>",
                font=dict(size=16, color="white"),
                x=0.5,
            ),
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=650,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
            yaxis=dict(
                title=f"RSI ({heatmap_tf})",
                range=[15, 90],
                gridcolor="rgba(255,255,255,0.05)",
            ),
            showlegend=False,
            margin=dict(l=50, r=20, t=60, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats below
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
            st.markdown(f"RSI > 70: **{len(plot_df[plot_df[rsi_col] > 70])}** coins")
            st.markdown(f"RSI 30–70: **{len(plot_df[(plot_df[rsi_col] >= 30) & (plot_df[rsi_col] <= 70)])}** coins")
            st.markdown(f"RSI < 30: **{len(plot_df[plot_df[rsi_col] < 30])}** coins")


# ============================================================
# TAB 3: BY MARKET CAP (CryptoWaves style list)
# ============================================================
with tab_market:
    sort_market = st.selectbox(
        "Sort by:",
        ["Rank", "1h", "24h", "7d", "30d", "RSI (4h)", "RSI (1D)"],
        index=0,
        key="market_sort",
    )
    
    display_df = df.copy()
    sort_map = {
        "Rank": ("rank", True),
        "1h": ("change_1h", False),
        "24h": ("change_24h", False),
        "7d": ("change_7d", False),
        "30d": ("change_30d", False),
        "RSI (4h)": ("rsi_4h", False),
        "RSI (1D)": ("rsi_1D", False),
    }
    sort_col, sort_asc = sort_map.get(sort_market, ("rank", True))
    if sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=sort_asc)
    
    for _, row in display_df.iterrows():
        symbol = row["symbol"]
        price = row["price"]
        rank = row.get("rank", "—")
        rank_str = f"#{int(rank)}" if isinstance(rank, (int, float)) and rank < 999 else ""
        
        if price >= 1000:
            price_str = f"${price:,.2f}"
        elif price >= 1:
            price_str = f"${price:,.4f}"
        elif price >= 0.001:
            price_str = f"${price:,.6f}"
        else:
            price_str = f"${price:.8f}"
        
        ch1h = row.get("change_1h", 0)
        ch24h = row.get("change_24h", 0)
        ch7d = row.get("change_7d", 0)
        ch30d = row.get("change_30d", 0)
        rsi_4h = row.get("rsi_4h", 50)
        rsi_1d = row.get("rsi_1D", 50)
        rec = row.get("recommendation", "WAIT")
        
        # Signal label
        if rec == "WAIT":
            sig_label = "WAIT"
            sig_color = "#FFD700"
        elif "BUY" in rec:
            sig_label = "CTB" if "STRONG" in rec else "BUY"
            sig_color = "#00FF7F"
        else:
            sig_label = "SELL"
            sig_color = "#FF6347"
        
        rsi_4h_color = "#FF6347" if rsi_4h > 70 else "#00FF7F" if rsi_4h < 30 else "white"
        rsi_1d_color = "#FF6347" if rsi_1d > 70 else "#00FF7F" if rsi_1d < 30 else "white"
        
        st.markdown(f"""
        <div class="market-row">
            <div class="coin-info">
                <span style="font-size: 15px; font-weight: bold; color: white;">{symbol}</span>
                <span class="coin-rank">{rank_str}</span>
                <br>
                <span style="font-size: 12px; color: #aaa;">Price: {price_str}</span>
                <br>
                <span style="font-size: 11px; color: #888;">
                    Ch%:
                    <span class="{'change-pos' if ch1h >= 0 else 'change-neg'}">{ch1h:+.2f}%</span>
                    <span class="{'change-pos' if ch24h >= 0 else 'change-neg'}">{ch24h:+.2f}%</span>
                    <span class="{'change-pos' if ch7d >= 0 else 'change-neg'}" style="font-weight:bold;">{ch7d:+.2f}%</span>
                    <span class="{'change-pos' if ch30d >= 0 else 'change-neg'}">{ch30d:+.2f}%</span>
                </span>
            </div>
            <div class="rsi-info">
                <span style="font-size: 12px; color: #888;">Now:</span>
                <span style="font-size: 14px; font-weight: bold; color: {sig_color};">{sig_label}</span>
                <br>
                <span style="font-size: 12px; color: #888;">RSI (4h): <b style="color: {rsi_4h_color};">{rsi_4h:.2f}</b></span>
                &nbsp;
                <span style="font-size: 12px; color: #888;">RSI (1D): <b style="color: {rsi_1d_color};">{rsi_1d:.2f}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 4: CONFLUENCE SCANNER
# ============================================================
with tab_confluence:
    st.markdown("### 🎯 Confluence Scanner")
    
    # Top Buy
    st.markdown("#### 🟢 Top Buy Signals")
    buy_df = df[df["score"] > 0].sort_values("score", ascending=False).head(15)
    for _, row in buy_df.iterrows():
        score = row["score"]
        bar_w = min(abs(score), 100)
        st.markdown(f"""
        <div style="background: #1a1a2e; border-radius: 8px; padding: 10px 14px; margin: 4px 0;">
            <div style="display: flex; justify-content: space-between;">
                <b style="color: white;">{row['symbol']}</b>
                <span style="color: #00FF7F; font-weight: bold;">{row['recommendation']} ({score})</span>
            </div>
            <div style="background: #2a2a4a; border-radius: 4px; height: 6px; margin-top: 6px;">
                <div style="background: linear-gradient(90deg, #00FF7F, #32CD32); height: 6px; width: {bar_w}%; border-radius: 4px;"></div>
            </div>
            <div style="font-size: 11px; color: #888; margin-top: 4px;">
                RSI 4h: {row.get('rsi_4h', 50):.1f} | MACD: {row.get('macd_trend', '—')} | Vol: {row.get('vol_trend', '—')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top Sell
    st.markdown("#### 🔴 Top Sell Signals")
    sell_df = df[df["score"] < 0].sort_values("score").head(15)
    for _, row in sell_df.iterrows():
        score = abs(row["score"])
        bar_w = min(score, 100)
        st.markdown(f"""
        <div style="background: #1a1a2e; border-radius: 8px; padding: 10px 14px; margin: 4px 0;">
            <div style="display: flex; justify-content: space-between;">
                <b style="color: white;">{row['symbol']}</b>
                <span style="color: #FF6347; font-weight: bold;">{row['recommendation']} ({row['score']})</span>
            </div>
            <div style="background: #2a2a4a; border-radius: 4px; height: 6px; margin-top: 6px;">
                <div style="background: linear-gradient(90deg, #FF6347, #FF0000); height: 6px; width: {bar_w}%; border-radius: 4px;"></div>
            </div>
            <div style="font-size: 11px; color: #888; margin-top: 4px;">
                RSI 4h: {row.get('rsi_4h', 50):.1f} | MACD: {row.get('macd_trend', '—')} | Vol: {row.get('vol_trend', '—')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Score distribution
    st.markdown("---")
    st.markdown("#### Score Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df["score"], nbinsx=30,
        marker_color=["#00FF7F" if x > 20 else "#FF6347" if x < -20 else "#FFD700" for x in sorted(df["score"])],
        opacity=0.8,
    ))
    fig_dist.add_vline(x=0, line_dash="dash", line_color="white", line_width=1)
    fig_dist.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=280, xaxis_title="Score", yaxis_title="Coins",
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ============================================================
# TAB 5: COIN DETAIL
# ============================================================
with tab_detail:
    selected_coin = st.selectbox("Select Coin", df["symbol"].tolist(), key="detail_coin")
    
    if selected_coin:
        coin = df[df["symbol"] == selected_coin].iloc[0]
        rec = coin.get("recommendation", "WAIT")
        rec_color = {
            "STRONG BUY": "#00FF7F", "BUY": "#32CD32", "LEAN BUY": "#90EE90",
            "WAIT": "#FFD700",
            "LEAN SELL": "#FFA07A", "SELL": "#FF6347", "STRONG SELL": "#FF0000",
        }.get(rec, "#FFD700")
        
        st.markdown(f"""
        <div style="background: #1a1a2e; border-radius: 10px; padding: 16px; margin-bottom: 16px;">
            <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 16px;">
                <h2 style="margin: 0; color: white;">{selected_coin}/USDT</h2>
                <span style="font-size: 22px; color: white; font-weight: bold;">
                    ${coin['price']:,.4f}
                </span>
                <span style="font-size: 16px; color: {'#00FF7F' if coin.get('change_24h', 0) >= 0 else '#FF6347'};">
                    {coin.get('change_24h', 0):+.2f}%
                </span>
                <span style="background: {rec_color}22; color: {rec_color}; padding: 5px 14px; border-radius: 16px; font-weight: bold; border: 1px solid {rec_color}44;">
                    {rec} ({coin.get('score', 0)})
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # RSI gauges
        st.markdown("#### RSI Multi-Timeframe")
        rsi_cols = [c for c in df.columns if c.startswith("rsi_")]
        gauge_cols = st.columns(len(rsi_cols))
        
        for i, rc in enumerate(rsi_cols):
            val = coin.get(rc, 50)
            tf_label = rc.replace("rsi_", "")
            with gauge_cols[i]:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=val,
                    title={"text": f"RSI {tf_label}", "font": {"size": 14, "color": "white"}},
                    number={"font": {"size": 22, "color": "white"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#FFD700"},
                        "bgcolor": "#1a1a2e",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(0,255,127,0.2)"},
                            {"range": [30, 70], "color": "rgba(255,215,0,0.1)"},
                            {"range": [70, 100], "color": "rgba(255,99,71,0.2)"},
                        ],
                    },
                ))
                fig_g.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", height=180, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig_g, use_container_width=True)
        
        # Indicators
        st.markdown("#### Indicators")
        st.markdown(f"""
        | Indicator | Value |
        |-----------|-------|
        | MACD Trend | **{coin.get('macd_trend', '—')}** |
        | MACD Histogram | **{coin.get('macd_histogram', 0):.4f}** |
        | Stoch RSI K/D | **{coin.get('stoch_rsi_k', 50):.1f}** / **{coin.get('stoch_rsi_d', 50):.1f}** |
        | Volume Trend | **{coin.get('vol_trend', '—')}** |
        | Volume Ratio | **{coin.get('vol_ratio', 1.0):.2f}x** |
        | OBV Trend | **{coin.get('obv_trend', '—')}** |
        """)
        
        if show_smc:
            st.markdown("#### Smart Money Concepts")
            st.markdown(f"""
            | SMC | Value |
            |-----|-------|
            | Market Structure | **{coin.get('market_structure', '—')}** |
            | Order Block | **{coin.get('ob_signal', '—')}** |
            | Fair Value Gap | **{coin.get('fvg_signal', '—')}** |
            | Break of Structure | **{'✅' if coin.get('bos', False) else '—'}** |
            """)
        
        if coin.get("reasons"):
            st.markdown("#### Signal Reasoning")
            for r in str(coin["reasons"]).split(" | "):
                if r.strip():
                    st.markdown(f"- {r.strip()}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #555; font-size: 11px;'>"
    f"🧙‍♂️ Merlin Crypto Scanner | {len(coins_to_scan)} coins × {len(tf_to_scan)} TFs | "
    f"Data: CCXT ({ex_name}) + CoinGecko | Not financial advice — DYOR!"
    f"</div>",
    unsafe_allow_html=True,
)
