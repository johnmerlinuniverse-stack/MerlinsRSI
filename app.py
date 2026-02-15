"""
🧙‍♂️ Merlin Crypto Scanner
A comprehensive crypto RSI scanner & alert dashboard.
Combines Binance + CoinGecko data with multi-indicator confluence analysis.
"""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

from config import (
    PAGE_TITLE, PAGE_ICON, TOP_COINS, TIMEFRAMES, COLORS,
    RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_STRONG_OVERBOUGHT, RSI_STRONG_OVERSOLD,
)
from data_fetcher import (
    fetch_all_market_data, fetch_klines_cached, fetch_all_tickers,
    check_binance_available,
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
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp { background-color: #0E1117; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px;
        margin: 4px 0;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; }
    .metric-label { font-size: 13px; color: #888; margin-top: 4px; }
    
    /* Signal badges */
    .signal-strong-buy { color: #00FF7F; font-weight: bold; }
    .signal-buy { color: #32CD32; font-weight: bold; }
    .signal-sell { color: #FF6347; font-weight: bold; }
    .signal-strong-sell { color: #FF0000; font-weight: bold; }
    .signal-wait { color: #FFD700; font-weight: bold; }
    
    /* Alert row styling */
    .alert-row {
        background: #1a1a2e;
        border-left: 4px solid;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
    }
    .alert-buy { border-color: #00FF7F; }
    .alert-sell { border-color: #FF6347; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    
    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Progress bar colors */
    .rsi-bar-container {
        background: #2a2a4a;
        border-radius: 10px;
        height: 8px;
        width: 100%;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧙‍♂️ Merlin Scanner")
    st.markdown("---")
    
    # Telegram settings
    st.markdown("### 📱 Telegram Alerts")
    telegram_token = st.text_input("Bot Token", type="password", key="tg_token")
    telegram_chat_id = st.text_input("Chat ID", key="tg_chat")
    alert_min_score = st.slider("Min. Alert Score", 10, 80, 30, 5)
    
    st.markdown("---")
    
    # Display settings
    st.markdown("### ⚙️ Settings")
    selected_timeframes = st.multiselect(
        "Timeframes",
        options=list(TIMEFRAMES.keys()),
        default=["4h", "1D"],
    )
    
    max_coins = st.slider("Number of Coins", 20, 180, 50, 10)
    
    show_smc = st.checkbox("Show Smart Money Concepts", value=True)
    show_confluence = st.checkbox("Show Confluence Scores", value=True)
    
    st.markdown("---")
    
    # Manual scan trigger
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Send summary alert
    if st.button("📤 Send Telegram Summary", use_container_width=True):
        if telegram_token and telegram_chat_id:
            st.session_state["send_summary"] = True
        else:
            st.warning("Configure Telegram first!")
    
    st.markdown("---")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    st.caption("Data: Binance + CoinGecko")


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=180, show_spinner=False)
def scan_all_coins(coins: tuple, timeframes_to_scan: tuple, include_smc: bool = False) -> pd.DataFrame:
    """
    Main scanning function. Fetches data and calculates all indicators
    for all coins. Uses Binance when available, CoinGecko as fallback.
    Returns a comprehensive DataFrame.
    """
    results = []

    # --- Step 1: Check which data source is available ---
    binance_ok = check_binance_available()

    if binance_ok:
        st.toast("✅ Binance API connected", icon="🟢")
    else:
        st.toast("⚠️ Binance unavailable — using CoinGecko data", icon="🟡")

    # --- Step 2: Get market data from CoinGecko (always works) ---
    market_df = fetch_all_market_data()

    # --- Step 3: Get Binance tickers if available ---
    tickers = {}
    if binance_ok:
        tickers = fetch_all_tickers()

    # --- Step 4: Build symbol->market data lookup ---
    market_lookup = {}
    if not market_df.empty:
        for _, row in market_df.iterrows():
            sym = row.get("symbol", "").upper()
            market_lookup[sym] = row

    # --- Step 5: Scan each coin ---
    coins_list = list(coins)
    progress_bar = st.progress(0, text="Scanning crypto market...")
    total = len(coins_list)
    scanned = 0

    for idx, symbol in enumerate(coins_list):
        progress_bar.progress((idx + 1) / total, text=f"Scanning {symbol}... ({idx+1}/{total})")

        row = {"symbol": symbol}

        # --- Get price/change from Binance tickers OR CoinGecko ---
        ticker = tickers.get(symbol, {})
        mkt = market_lookup.get(symbol, {})

        if ticker:
            row["price"] = float(ticker.get("lastPrice", 0))
            row["change_24h"] = float(ticker.get("priceChangePercent", 0))
            row["volume_24h"] = float(ticker.get("quoteVolume", 0))
        elif isinstance(mkt, pd.Series) and not mkt.empty:
            row["price"] = float(mkt.get("price", 0)) if pd.notna(mkt.get("price")) else 0
            row["change_24h"] = float(mkt.get("change_24h", 0)) if pd.notna(mkt.get("change_24h")) else 0
            row["volume_24h"] = float(mkt.get("volume_24h", 0)) if pd.notna(mkt.get("volume_24h")) else 0
        else:
            row["price"] = 0
            row["change_24h"] = 0
            row["volume_24h"] = 0

        if row["price"] == 0:
            continue  # Skip coins with no price data

        # --- RSI for each timeframe ---
        klines_data = {}
        for tf in timeframes_to_scan:
            tf_interval = TIMEFRAMES.get(tf, tf)
            df_klines = fetch_klines_cached(symbol, tf_interval, use_binance=binance_ok)

            if not df_klines.empty:
                klines_data[tf] = df_klines
                rsi_val = calculate_rsi(df_klines)
                row[f"rsi_{tf}"] = rsi_val
            else:
                row[f"rsi_{tf}"] = 50.0

        # --- MACD (using 4h or first available) ---
        primary_tf = "4h" if "4h" in klines_data else (list(timeframes_to_scan)[0] if timeframes_to_scan else None)
        if primary_tf and primary_tf in klines_data:
            macd_data = calculate_macd(klines_data[primary_tf])
            row["macd_trend"] = macd_data["trend"]
            row["macd_histogram"] = macd_data["histogram"]

            # Volume analysis
            vol_data = calculate_volume_analysis(klines_data[primary_tf])
            row["vol_trend"] = vol_data["vol_trend"]
            row["vol_ratio"] = vol_data["vol_ratio"]
            row["obv_trend"] = vol_data["obv_trend"]

            # Stochastic RSI
            stoch = calculate_stoch_rsi(klines_data[primary_tf])
            row["stoch_rsi_k"] = stoch["stoch_rsi_k"]
            row["stoch_rsi_d"] = stoch["stoch_rsi_d"]

            # Smart Money Concepts
            if include_smc:
                ob = detect_order_blocks(klines_data[primary_tf])
                fvg = detect_fair_value_gaps(klines_data[primary_tf])
                ms = detect_market_structure(klines_data[primary_tf])
                row["ob_signal"] = ob["ob_signal"]
                row["fvg_signal"] = fvg["fvg_signal"]
                row["market_structure"] = ms["structure"]
                row["bos"] = ms["break_of_structure"]

            # Confluence signal
            rsi_4h = row.get("rsi_4h", row.get(f"rsi_{primary_tf}", 50))
            rsi_1d = row.get("rsi_1D", 50)
            smc_combined = {
                "ob_signal": row.get("ob_signal", "NONE"),
                "fvg_signal": row.get("fvg_signal", "BALANCED"),
                "structure": row.get("market_structure", "UNKNOWN"),
            } if include_smc else None

            confluence = generate_confluence_signal(
                rsi_4h=rsi_4h,
                rsi_1d=rsi_1d,
                macd_data=macd_data,
                volume_data=vol_data,
                smc_data=smc_combined,
            )
            row["score"] = confluence["score"]
            row["recommendation"] = confluence["recommendation"]
            row["reasons"] = " | ".join(confluence["reasons"][:3])
        else:
            row["macd_trend"] = "NEUTRAL"
            row["macd_histogram"] = 0
            row["vol_trend"] = "—"
            row["vol_ratio"] = 1.0
            row["obv_trend"] = "—"
            row["stoch_rsi_k"] = 50.0
            row["stoch_rsi_d"] = 50.0
            row["score"] = 0
            row["recommendation"] = "WAIT"
            row["reasons"] = ""

        results.append(row)
        scanned += 1

        # Rate limit: CoinGecko free = 10-30 req/min
        if not binance_ok and idx % 5 == 0 and idx > 0:
            time.sleep(2)
        elif idx % 15 == 0 and idx > 0:
            time.sleep(0.3)

    progress_bar.empty()

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ============================================================
# HEADER
# ============================================================
col_h1, col_h2, col_h3 = st.columns([3, 2, 2])
with col_h1:
    st.markdown("# 🧙‍♂️ Merlin Crypto Scanner")
with col_h2:
    st.markdown("")
with col_h3:
    st.markdown(f"**{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC**")

# ============================================================
# LOAD DATA
# ============================================================
coins_to_scan = TOP_COINS[:max_coins]
tf_to_scan = selected_timeframes if selected_timeframes else ["4h", "1D"]

# Show data source status
binance_status = check_binance_available()
if binance_status:
    st.caption("🟢 Data: Binance (klines) + CoinGecko (market data)")
else:
    st.caption("🟡 Data: CoinGecko only (Binance unavailable in this region)")

df = scan_all_coins(
    tuple(coins_to_scan),
    tuple(tf_to_scan),
    include_smc=show_smc,
)

if df.empty:
    st.warning(
        "⚠️ No data loaded yet. This can happen if CoinGecko rate limits are hit. "
        "Try reducing 'Number of Coins' to 30-50 in the sidebar, then click 'Refresh Data'."
    )
    st.info(
        "💡 **Tip:** CoinGecko free API allows ~30 requests/minute. "
        "With 50 coins × 2 timeframes = ~100 requests, it may take a moment. "
        "The data is cached for 3 minutes after the first load."
    )
    st.stop()

# ============================================================
# MARKET OVERVIEW METRICS
# ============================================================
avg_rsi_4h = df["rsi_4h"].mean() if "rsi_4h" in df.columns else 50
avg_rsi_1d = df["rsi_1D"].mean() if "rsi_1D" in df.columns else 50
overbought_count = len(df[df.get("rsi_4h", pd.Series(dtype=float)) >= RSI_OVERBOUGHT]) if "rsi_4h" in df.columns else 0
oversold_count = len(df[df.get("rsi_4h", pd.Series(dtype=float)) <= RSI_OVERSOLD]) if "rsi_4h" in df.columns else 0
buy_signals = len(df[df["score"] > 0])
sell_signals = len(df[df["score"] < 0])

# Market sentiment
if avg_rsi_4h >= 65:
    market_status = "🟢 STRONG"
elif avg_rsi_4h >= 55:
    market_status = "🟡 NEUTRAL"
elif avg_rsi_4h >= 45:
    market_status = "🟠 WEAK"
else:
    market_status = "🔴 BEARISH"

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("Market", market_status)
with m2:
    st.metric("Avg RSI (4h)", f"{avg_rsi_4h:.1f}")
with m3:
    st.metric("Avg RSI (1D)", f"{avg_rsi_1d:.1f}")
with m4:
    st.metric("Overbought", f"{overbought_count}", delta=None)
with m5:
    st.metric("Oversold", f"{oversold_count}", delta=None)
with m6:
    st.metric("Buy / Sell", f"{buy_signals} / {sell_signals}")

st.markdown("---")


# ============================================================
# TABS
# ============================================================
tab_heatmap, tab_market, tab_alerts, tab_confluence, tab_detail = st.tabs([
    "🔥 RSI Heatmap",
    "📊 Market Overview",
    "🚨 24h Alerts",
    "🎯 Confluence Scanner",
    "🔍 Coin Detail",
])


# ============================================================
# TAB 1: RSI HEATMAP
# ============================================================
with tab_heatmap:
    st.markdown("### Crypto Market RSI Heatmap")
    
    heatmap_tf = st.selectbox(
        "Timeframe", tf_to_scan, index=0, key="heatmap_tf"
    )
    rsi_col = f"rsi_{heatmap_tf}"
    
    if rsi_col in df.columns:
        plot_df = df[["symbol", rsi_col, "price", "change_24h", "volume_24h"]].copy()
        plot_df = plot_df.dropna(subset=[rsi_col])
        
        # Assign x position spread for visualization
        np.random.seed(42)
        plot_df["x_pos"] = np.random.uniform(0, 100, len(plot_df))
        
        # Color based on RSI zones
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
        
        # Size based on volume
        vol_norm = plot_df["volume_24h"]
        vol_norm = (vol_norm - vol_norm.min()) / (vol_norm.max() - vol_norm.min() + 1e-10)
        plot_df["size"] = 8 + vol_norm * 25
        
        fig = go.Figure()
        
        # RSI zones
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(255,0,0,0.12)", line_width=0,
                      annotation_text="OVERBOUGHT", annotation_position="top right",
                      annotation_font_color="#FF6347")
        fig.add_hrect(y0=70, y1=80, fillcolor="rgba(255,99,71,0.08)", line_width=0)
        fig.add_hrect(y0=60, y1=70, fillcolor="rgba(255,215,0,0.05)", line_width=0)
        fig.add_hrect(y0=30, y1=40, fillcolor="rgba(50,205,50,0.08)", line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,127,0.12)", line_width=0,
                      annotation_text="OVERSOLD", annotation_position="bottom right",
                      annotation_font_color="#00FF7F")
        
        # Reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,99,71,0.5)", line_width=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,127,0.5)", line_width=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)
        
        # Scatter points
        fig.add_trace(go.Scatter(
            x=plot_df["x_pos"],
            y=plot_df[rsi_col],
            mode="markers+text",
            text=plot_df["symbol"],
            textposition="top center",
            textfont=dict(size=9, color="white"),
            marker=dict(
                size=plot_df["size"],
                color=plot_df["color"],
                opacity=0.85,
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
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=700,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
            yaxis=dict(
                title=f"RSI ({heatmap_tf})",
                range=[15, 90],
                gridcolor="rgba(255,255,255,0.05)",
            ),
            showlegend=False,
            margin=dict(l=60, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats below heatmap
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            top_overbought = plot_df.nlargest(5, rsi_col)[["symbol", rsi_col]]
            st.markdown("**🔴 Most Overbought**")
            for _, r in top_overbought.iterrows():
                st.markdown(f"**{r['symbol']}** — RSI: {r[rsi_col]:.1f}")
        with col_b:
            top_oversold = plot_df.nsmallest(5, rsi_col)[["symbol", rsi_col]]
            st.markdown("**🟢 Most Oversold**")
            for _, r in top_oversold.iterrows():
                st.markdown(f"**{r['symbol']}** — RSI: {r[rsi_col]:.1f}")
        with col_c:
            st.markdown("**📊 Distribution**")
            st.markdown(f"RSI > 70: **{len(plot_df[plot_df[rsi_col] > 70])}** coins")
            st.markdown(f"RSI 30-70: **{len(plot_df[(plot_df[rsi_col] >= 30) & (plot_df[rsi_col] <= 70)])}** coins")
            st.markdown(f"RSI < 30: **{len(plot_df[plot_df[rsi_col] < 30])}** coins")


# ============================================================
# TAB 2: MARKET OVERVIEW (like CryptoWaves By Market Cap)
# ============================================================
with tab_market:
    st.markdown("### Market Overview — By Market Cap")
    
    sort_col = st.selectbox(
        "Sort by",
        ["volume_24h", "rsi_4h", "rsi_1D", "change_24h", "score"],
        index=0,
        key="market_sort",
    )
    sort_asc = st.checkbox("Ascending", value=False, key="market_asc")
    
    display_df = df.copy()
    if sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=sort_asc)
    
    # Signal color function
    def signal_color(rec):
        colors = {
            "STRONG BUY": "🟢🟢", "BUY": "🟢", "LEAN BUY": "🟡↗️",
            "WAIT": "⏳", "LEAN SELL": "🟡↘️", "SELL": "🔴", "STRONG SELL": "🔴🔴",
        }
        return colors.get(rec, "⏳")
    
    def format_price(p):
        if p >= 1000:
            return f"${p:,.2f}"
        elif p >= 1:
            return f"${p:,.4f}"
        else:
            return f"${p:,.6f}"
    
    def format_volume(v):
        if v >= 1e9:
            return f"${v/1e9:.2f}B"
        elif v >= 1e6:
            return f"${v/1e6:.2f}M"
        else:
            return f"${v/1e3:.1f}K"
    
    # Build display table
    table_data = []
    rsi_cols = [c for c in display_df.columns if c.startswith("rsi_")]
    
    for _, row in display_df.iterrows():
        entry = {
            "Signal": signal_color(row.get("recommendation", "WAIT")),
            "Coin": row["symbol"],
            "Price": format_price(row["price"]),
            "24h %": f"{row.get('change_24h', 0):+.2f}%",
            "Volume": format_volume(row.get("volume_24h", 0)),
        }
        for rc in rsi_cols:
            tf_label = rc.replace("rsi_", "RSI ")
            val = row.get(rc, 50)
            entry[tf_label] = f"{val:.1f}"
        
        entry["MACD"] = row.get("macd_trend", "—")
        entry["Score"] = row.get("score", 0)
        entry["Action"] = row.get("recommendation", "WAIT")
        
        table_data.append(entry)
    
    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        height=600,
        hide_index=True,
    )


# ============================================================
# TAB 3: 24h ALERTS
# ============================================================
with tab_alerts:
    st.markdown("### 🚨 24h RSI Alerts")
    
    alert_filter = st.selectbox(
        "Filter",
        ["All Signals", "BUY Signals Only", "SELL Signals Only", "Strong Signals Only"],
        key="alert_filter",
    )
    
    alert_df = df.copy()
    
    if alert_filter == "BUY Signals Only":
        alert_df = alert_df[alert_df["score"] > 0]
    elif alert_filter == "SELL Signals Only":
        alert_df = alert_df[alert_df["score"] < 0]
    elif alert_filter == "Strong Signals Only":
        alert_df = alert_df[alert_df["score"].abs() >= 40]
    else:
        alert_df = alert_df[alert_df["recommendation"] != "WAIT"]
    
    alert_df = alert_df.sort_values("score", key=abs, ascending=False)
    
    if alert_df.empty:
        st.info("No active alerts at the moment. Market is in WAIT mode.")
    else:
        st.markdown(f"**{len(alert_df)}** active signals")
        st.markdown("")
        
        for _, row in alert_df.head(30).iterrows():
            rec = row.get("recommendation", "WAIT")
            score = row.get("score", 0)
            
            if "BUY" in rec:
                border_color = "#00FF7F"
                bg = "rgba(0, 255, 127, 0.05)"
            elif "SELL" in rec:
                border_color = "#FF6347"
                bg = "rgba(255, 99, 71, 0.05)"
            else:
                border_color = "#FFD700"
                bg = "rgba(255, 215, 0, 0.05)"
            
            rsi_4h = row.get("rsi_4h", "—")
            rsi_1d = row.get("rsi_1D", "—")
            rsi_4h_str = f"{rsi_4h:.1f}" if isinstance(rsi_4h, (int, float)) else rsi_4h
            rsi_1d_str = f"{rsi_1d:.1f}" if isinstance(rsi_1d, (int, float)) else rsi_1d
            
            reasons = row.get("reasons", "")
            macd_trend = row.get("macd_trend", "—")
            vol_trend = row.get("vol_trend", "—")
            
            st.markdown(f"""
            <div style="
                background: {bg};
                border-left: 4px solid {border_color};
                border-radius: 8px;
                padding: 14px 18px;
                margin: 6px 0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 18px; font-weight: bold; color: white;">
                            {row['symbol']}/USDT
                        </span>
                        <span style="font-size: 14px; color: {border_color}; margin-left: 12px; font-weight: bold;">
                            {rec}
                        </span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 16px; color: white; font-weight: bold;">
                            ${row['price']:,.4f}
                        </span>
                        <span style="font-size: 13px; color: {'#00FF7F' if row.get('change_24h', 0) >= 0 else '#FF6347'}; margin-left: 8px;">
                            {row.get('change_24h', 0):+.2f}%
                        </span>
                    </div>
                </div>
                <div style="margin-top: 8px; display: flex; gap: 24px; font-size: 13px; color: #ccc;">
                    <span>RSI 4h: <b style="color: {'#FF6347' if isinstance(rsi_4h, float) and rsi_4h > 70 else '#00FF7F' if isinstance(rsi_4h, float) and rsi_4h < 30 else '#FFD700'}">{rsi_4h_str}</b></span>
                    <span>RSI 1D: <b style="color: {'#FF6347' if isinstance(rsi_1d, float) and rsi_1d > 70 else '#00FF7F' if isinstance(rsi_1d, float) and rsi_1d < 30 else '#FFD700'}">{rsi_1d_str}</b></span>
                    <span>MACD: <b>{macd_trend}</b></span>
                    <span>Volume: <b>{vol_trend}</b></span>
                    <span>Score: <b style="color: {border_color};">{score}</b></span>
                </div>
                <div style="margin-top: 5px; font-size: 11px; color: #888;">
                    {reasons}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Trigger Telegram alerts
    if st.session_state.get("send_summary") and telegram_token and telegram_chat_id:
        all_data = alert_df.to_dict("records")
        check_and_send_alerts(all_data, telegram_token, telegram_chat_id, alert_min_score, send_summary=True)
        st.success("✅ Summary sent to Telegram!")
        st.session_state["send_summary"] = False


# ============================================================
# TAB 4: CONFLUENCE SCANNER
# ============================================================
with tab_confluence:
    st.markdown("### 🎯 Multi-Indicator Confluence Scanner")
    
    if not show_confluence:
        st.info("Enable 'Show Confluence Scores' in sidebar to use this tab.")
    else:
        col_buy, col_sell = st.columns(2)
        
        with col_buy:
            st.markdown("#### 🟢 Top Buy Confluence")
            buy_df = df[df["score"] > 0].sort_values("score", ascending=False).head(15)
            
            for _, row in buy_df.iterrows():
                score = row["score"]
                bar_width = min(score, 100)
                
                st.markdown(f"""
                <div style="background: #1a1a2e; border-radius: 8px; padding: 10px 14px; margin: 4px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <b style="color: white;">{row['symbol']}</b>
                        <span style="color: #00FF7F; font-weight: bold;">{row['recommendation']} ({score})</span>
                    </div>
                    <div style="background: #2a2a4a; border-radius: 4px; height: 6px; margin-top: 6px;">
                        <div style="background: linear-gradient(90deg, #00FF7F, #32CD32); height: 6px; width: {bar_width}%; border-radius: 4px;"></div>
                    </div>
                    <div style="font-size: 11px; color: #888; margin-top: 4px;">
                        RSI 4h: {row.get('rsi_4h', '—'):.1f} | MACD: {row.get('macd_trend', '—')} | Vol: {row.get('vol_trend', '—')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_sell:
            st.markdown("#### 🔴 Top Sell Confluence")
            sell_df = df[df["score"] < 0].sort_values("score").head(15)
            
            for _, row in sell_df.iterrows():
                score = abs(row["score"])
                bar_width = min(score, 100)
                
                st.markdown(f"""
                <div style="background: #1a1a2e; border-radius: 8px; padding: 10px 14px; margin: 4px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <b style="color: white;">{row['symbol']}</b>
                        <span style="color: #FF6347; font-weight: bold;">{row['recommendation']} ({row['score']})</span>
                    </div>
                    <div style="background: #2a2a4a; border-radius: 4px; height: 6px; margin-top: 6px;">
                        <div style="background: linear-gradient(90deg, #FF6347, #FF0000); height: 6px; width: {bar_width}%; border-radius: 4px;"></div>
                    </div>
                    <div style="font-size: 11px; color: #888; margin-top: 4px;">
                        RSI 4h: {row.get('rsi_4h', '—'):.1f} | MACD: {row.get('macd_trend', '—')} | Vol: {row.get('vol_trend', '—')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Confluence distribution chart
        st.markdown("---")
        st.markdown("#### Score Distribution")
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df["score"],
            nbinsx=30,
            marker_color=[
                "#00FF7F" if x > 20 else "#FF6347" if x < -20 else "#FFD700"
                for x in sorted(df["score"])
            ],
            opacity=0.8,
        ))
        fig_dist.add_vline(x=0, line_dash="dash", line_color="white", line_width=1)
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=300,
            xaxis_title="Confluence Score",
            yaxis_title="Number of Coins",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ============================================================
# TAB 5: COIN DETAIL
# ============================================================
with tab_detail:
    st.markdown("### 🔍 Coin Detail View")
    
    selected_coin = st.selectbox(
        "Select Coin",
        df["symbol"].tolist(),
        key="detail_coin",
    )
    
    if selected_coin:
        coin_row = df[df["symbol"] == selected_coin].iloc[0]
        
        # Header
        rec = coin_row.get("recommendation", "WAIT")
        rec_color = {
            "STRONG BUY": "#00FF7F", "BUY": "#32CD32", "LEAN BUY": "#90EE90",
            "WAIT": "#FFD700",
            "LEAN SELL": "#FFA07A", "SELL": "#FF6347", "STRONG SELL": "#FF0000",
        }.get(rec, "#FFD700")
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: white;">{selected_coin}/USDT</h2>
            <span style="font-size: 24px; color: white; font-weight: bold;">${coin_row['price']:,.4f}</span>
            <span style="font-size: 18px; color: {'#00FF7F' if coin_row.get('change_24h', 0) >= 0 else '#FF6347'};">
                {coin_row.get('change_24h', 0):+.2f}%
            </span>
            <span style="background: {rec_color}22; color: {rec_color}; padding: 6px 16px; border-radius: 20px; font-weight: bold; border: 1px solid {rec_color}44;">
                {rec} (Score: {coin_row.get('score', 0)})
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # RSI gauges
        st.markdown("#### RSI Multi-Timeframe")
        rsi_cols_detail = [c for c in df.columns if c.startswith("rsi_")]
        gauge_cols = st.columns(len(rsi_cols_detail))
        
        for i, rc in enumerate(rsi_cols_detail):
            val = coin_row.get(rc, 50)
            tf_label = rc.replace("rsi_", "")
            
            with gauge_cols[i]:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    title={"text": f"RSI {tf_label}", "font": {"size": 14, "color": "white"}},
                    number={"font": {"size": 24, "color": "white"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "white"},
                        "bar": {"color": "#FFD700"},
                        "bgcolor": "#1a1a2e",
                        "bordercolor": "#2a2a4a",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(0,255,127,0.2)"},
                            {"range": [30, 70], "color": "rgba(255,215,0,0.1)"},
                            {"range": [70, 100], "color": "rgba(255,99,71,0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": 0.8,
                            "value": val,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    height=200,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Additional indicators
        st.markdown("#### Indicator Details")
        
        det_c1, det_c2, det_c3 = st.columns(3)
        
        with det_c1:
            st.markdown("**MACD**")
            st.markdown(f"Trend: **{coin_row.get('macd_trend', '—')}**")
            st.markdown(f"Histogram: **{coin_row.get('macd_histogram', 0):.4f}**")
            st.markdown(f"Stoch RSI K: **{coin_row.get('stoch_rsi_k', 50):.1f}**")
            st.markdown(f"Stoch RSI D: **{coin_row.get('stoch_rsi_d', 50):.1f}**")
        
        with det_c2:
            st.markdown("**Volume**")
            st.markdown(f"Trend: **{coin_row.get('vol_trend', '—')}**")
            st.markdown(f"Ratio: **{coin_row.get('vol_ratio', 1.0):.2f}x**")
            st.markdown(f"OBV: **{coin_row.get('obv_trend', '—')}**")
        
        with det_c3:
            if show_smc:
                st.markdown("**Smart Money**")
                st.markdown(f"Structure: **{coin_row.get('market_structure', '—')}**")
                st.markdown(f"Order Block: **{coin_row.get('ob_signal', '—')}**")
                st.markdown(f"FVG: **{coin_row.get('fvg_signal', '—')}**")
                st.markdown(f"BOS: **{'✅' if coin_row.get('bos', False) else '—'}**")
            else:
                st.info("Enable SMC in sidebar")
        
        # Confluence reasons
        if coin_row.get("reasons"):
            st.markdown("#### Signal Reasoning")
            for reason in str(coin_row["reasons"]).split(" | "):
                if reason.strip():
                    st.markdown(f"• {reason.strip()}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #555; font-size: 12px;'>"
    "🧙‍♂️ Merlin Crypto Scanner | Data: Binance + CoinGecko | "
    "Not financial advice — DYOR! | "
    f"Scanning {len(coins_to_scan)} coins across {len(tf_to_scan)} timeframes"
    "</div>",
    unsafe_allow_html=True,
)
