"""
🧙‍♂️ Merlin Crypto Scanner
RSI Heatmap + 24h Alerts Dashboard
CryptoWaves-style 5-status signal engine + CCXT multi-exchange
"""

import time
import json
import base64
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
# 5-STATUS SIGNAL ENGINE
# ============================================================
def compute_signal(rsi_4h: float, rsi_4h_prev: float, rsi_1d: float) -> str:
    if rsi_4h >= 40 and rsi_4h_prev < 40 and rsi_1d >= 35:
        return "CTB"
    if rsi_4h <= 60 and rsi_4h_prev > 60 and rsi_1d <= 65:
        return "CTS"
    if rsi_4h >= 50 and rsi_1d >= 50 and rsi_4h > rsi_4h_prev:
        return "BUY"
    if rsi_4h <= 45 and rsi_1d <= 45 and rsi_4h < rsi_4h_prev:
        return "SELL"
    return "WAIT"

def signal_color(sig: str) -> str:
    return {"CTB": "#00FF7F", "BUY": "#00FF7F", "CTS": "#FF6347", "SELL": "#FF6347"}.get(sig, "#FFD700")

def border_intensity(rsi_4h: float, sig: str) -> float:
    if sig in ("CTS", "SELL"):
        if rsi_4h >= 70: return 1.0
        elif rsi_4h >= 60: return 0.7
        else: return 0.45
    elif sig in ("CTB", "BUY"):
        if rsi_4h <= 30: return 1.0
        elif rsi_4h <= 40: return 0.7
        else: return 0.45
    return 0.0


# ============================================================
# SPARKLINE AS BASE64 IMG (fixes Streamlit SVG stripping)
# ============================================================
def make_sparkline_img(closes: list, width: int = 110, height: int = 30) -> str:
    """Generate sparkline as base64-encoded SVG data URI for use in <img> tags."""
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
    path_d = "M" + "L".join(pts)
    color = "#00FF7F" if vals[-1] >= vals[0] else "#FF6347"
    svg = (f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
           f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/></svg>')
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{width}" height="{height}" style="vertical-align:middle;">'


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="collapsed")

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

    /* Coin row — shared between alerts and market cap */
    .crow {
        background: #1a1a2e; border-radius: 0 10px 10px 0;
        padding: 10px 14px; margin: 4px 0;
        display: flex; align-items: center; gap: 12px;
    }
    .crow .ic { width: 34px; height: 34px; border-radius: 50%; overflow: hidden; flex-shrink: 0; }
    .crow .ic img { width: 34px; height: 34px; border-radius: 50%; }
    .crow .inf { flex: 1; min-width: 160px; }
    .crow .cn { font-size: 15px; font-weight: bold; color: white; }
    .crow .cf { font-size: 11px; color: #888; margin-left: 4px; }
    .crow .cr { font-size: 10px; background: #2a2a4a; padding: 1px 6px; border-radius: 6px; color: #888; margin-left: 4px; }
    .crow .pl { font-size: 12px; color: #aaa; margin-top: 2px; }
    .crow .chs { font-size: 11px; color: #888; margin-top: 2px; }
    .crow .charts { display: flex; gap: 16px; align-items: center; flex-shrink: 0; }
    .crow .clbl { font-size: 9px; color: #555; text-align: center; }
    .crow .sig { text-align: right; min-width: 140px; flex-shrink: 0; }
    .crow .sl { font-size: 15px; font-weight: bold; }
    .crow .rl { font-size: 12px; color: #888; margin-top: 1px; }
    .crow .rl b { font-size: 14px; }

    .cp { color: #00FF7F; }
    .cm { color: #FF6347; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; font-weight: 600; }
    #MainMenu, footer, header { visibility: hidden; }

    @media (max-width: 768px) {
        .block-container { padding: 0.5rem; }
        .crow .charts { display: none; }
        .crow .inf { min-width: 120px; }
        .crow .sig { min-width: 100px; }
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
    coin_source = {"CryptoWaves (107)": CRYPTOWAVES_COINS, "Top 100 Dynamic": CRYPTOWAVES_COINS[:50],
                   "Extended (180+)": TOP_COINS_EXTENDED}[coin_list_mode]
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
# SCAN
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
            row["coin_image"] = str(mkt.get("image", "")) if pd.notna(mkt.get("image")) else ""
            row["change_1h"] = float(mkt.get("change_1h", 0)) if pd.notna(mkt.get("change_1h")) else 0
            row["change_7d"] = float(mkt.get("change_7d", 0)) if pd.notna(mkt.get("change_7d")) else 0
            row["change_30d"] = float(mkt.get("change_30d", 0)) if pd.notna(mkt.get("change_30d")) else 0
        else:
            row["rank"], row["coin_name"], row["coin_image"] = 999, symbol, ""
            row["change_1h"], row["change_7d"], row["change_30d"] = 0, 0, 0

        if row["price"] == 0:
            continue

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
                    row[f"rsi_prev_{tf}"] = round(float(rsi_s.iloc[-2]), 2)
                else:
                    row[f"rsi_{tf}"], row[f"rsi_prev_{tf}"] = 50.0, 50.0
                row[f"closes_{tf}"] = json.dumps([round(c, 6) for c in df_kl["close"].tail(20).tolist()])
            else:
                row[f"rsi_{tf}"], row[f"rsi_prev_{tf}"] = 50.0, 50.0
                row[f"closes_{tf}"] = "[]"

        rsi_4h = row.get("rsi_4h", 50.0)
        rsi_4h_prev = row.get("rsi_prev_4h", 50.0)
        rsi_1d = row.get("rsi_1D", 50.0)
        row["signal"] = compute_signal(rsi_4h, rsi_4h_prev, rsi_1d)
        row["border_intensity"] = border_intensity(rsi_4h, row["signal"])

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
    st.warning("⚠️ No data. Reduce coins and Refresh.")
    st.stop()


# ============================================================
# HELPERS
# ============================================================
def cc(v): return "cp" if v >= 0 else "cm"

def fp(p):
    if p >= 1000: return f"${p:,.2f}"
    elif p >= 1: return f"${p:,.4f}"
    elif p >= 0.001: return f"${p:,.6f}"
    else: return f"${p:.8f}"

def rc(val):
    if val > 70: return "#FF6347"
    elif val < 30: return "#00FF7F"
    return "white"

def icon_html(url):
    if url:
        return f'<img src="{url}" width="34" height="34" style="border-radius:50%;">'
    return '<div style="width:34px;height:34px;border-radius:50%;background:#2a2a4a;"></div>'

def coin_row_html(row, show_charts=True):
    """Build one coin row HTML with base64 sparklines."""
    sig = row.get("signal", "WAIT")
    sc = signal_color(sig)
    inten = row.get("border_intensity", 0)

    if sig in ("CTS", "SELL"):
        bdr = f"border-left:6px solid rgba(255,80,80,{max(inten,0.35):.2f});"
    elif sig in ("CTB", "BUY"):
        bdr = f"border-left:6px solid rgba(0,255,140,{max(inten,0.35):.2f});"
    else:
        bdr = "border-left:6px solid transparent;"

    sym = row["symbol"]
    name = row.get("coin_name", sym)
    rank = row.get("rank", 999)
    rk = f"#{int(rank)}" if rank < 999 else ""
    img = row.get("coin_image", "")
    rsi4 = row.get("rsi_4h", 50)
    rsi1d = row.get("rsi_1D", 50)
    ch1h, ch24h = row.get("change_1h", 0), row.get("change_24h", 0)
    ch7d, ch30d = row.get("change_7d", 0), row.get("change_30d", 0)

    # Build sparkline images (base64 encoded)
    charts_html = ""
    if show_charts:
        try: s7 = make_sparkline_img(json.loads(row.get("closes_4h", "[]")), 110, 30)
        except: s7 = ""
        try: s30 = make_sparkline_img(json.loads(row.get("closes_1D", "[]")), 110, 30)
        except: s30 = ""
        charts_html = f'<div class="charts"><div><div class="clbl">● 7d</div>{s7}</div><div><div class="clbl">● 30d</div>{s30}</div></div>'

    return f"""<div class="crow" style="{bdr}">
<div class="ic">{icon_html(img)}</div>
<div class="inf">
<div><span class="cn">{sym}</span><span class="cf">{name}</span><span class="cr">{rk}</span></div>
<div class="pl">Price: <b style="color:white;">{fp(row['price'])}</b></div>
<div class="chs">Ch%: <span class="{cc(ch1h)}">{ch1h:+.2f}%</span> <span class="{cc(ch24h)}">{ch24h:+.2f}%</span> <span class="{cc(ch7d)}" style="font-weight:bold;">{ch7d:+.2f}%</span> <span class="{cc(ch30d)}">{ch30d:+.2f}%</span></div>
</div>
{charts_html}
<div class="sig">
<span style="font-size:11px;color:#888;">Now:</span> <span class="sl" style="color:{sc};">{sig}</span>
<div class="rl">RSI (4h): <b style="color:{rc(rsi4)};">{rsi4:.2f}</b></div>
<div class="rl">RSI (1D): <b style="color:{rc(rsi1d)};">{rsi1d:.2f}</b></div>
</div></div>"""


# ============================================================
# HEADER
# ============================================================
avg_rsi_4h = df["rsi_4h"].mean() if "rsi_4h" in df.columns else 50
sell_ct = len(df[df["signal"].isin(["CTS", "SELL"])]) if "signal" in df.columns else 0
buy_ct = len(df[df["signal"].isin(["CTB", "BUY"])]) if "signal" in df.columns else 0
wait_ct = len(df) - sell_ct - buy_ct

if avg_rsi_4h >= 60: ml, bc = "BULLISH", "badge-bullish"
elif avg_rsi_4h >= 45: ml, bc = "NEUTRAL", "badge-neutral"
else: ml, bc = "BEARISH", "badge-bearish"

avg_ch1h = df["change_1h"].mean() if "change_1h" in df.columns else 0
avg_ch24h = df["change_24h"].mean()
avg_ch7d = df["change_7d"].mean() if "change_7d" in df.columns else 0
avg_ch30d = df["change_30d"].mean() if "change_30d" in df.columns else 0
ex_status = get_exchange_status()
ex_name = ex_status["active_exchange"].upper() if ex_status["connected"] else "CoinGecko"

st.markdown(f"""<div class="header-bar">
<div><span class="header-title">🧙‍♂️ Merlin Crypto Scanner</span> <span class="badge {bc}">Market: {ml}</span></div>
<div class="header-stat">Avg RSI (4h): <b>{avg_rsi_4h:.2f}</b> | Ch%: <span class="{cc(avg_ch1h)}">{avg_ch1h:+.2f}%</span> <span class="{cc(avg_ch24h)}">{avg_ch24h:+.2f}%</span> <span class="{cc(avg_ch7d)}">{avg_ch7d:+.2f}%</span> <span class="{cc(avg_ch30d)}">{avg_ch30d:+.2f}%</span></div>
<div class="header-stat"><span style="color:#FF6347;">🔴 {sell_ct}</span> <span style="color:#FFD700;">🟡 {wait_ct}</span> <span style="color:#00FF7F;">🟢 {buy_ct}</span> | 📡 {ex_name}</div>
</div>""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab_alerts, tab_heatmap, tab_market, tab_confluence, tab_detail = st.tabs([
    f"🚨 24h Alerts {sell_ct}🔴 {buy_ct}🟢",
    "🔥 RSI Heatmap", "📊 By Market Cap", "🎯 Confluence", "🔍 Detail"])


# ============================================================
# TAB 1: 24h ALERTS (with BUY/SELL vs CTB/CTS filter)
# ============================================================
with tab_alerts:
    col_t, col_f, col_s = st.columns([2, 1, 1])
    with col_t:
        st.markdown("### 🚨 24h Alerts")
    with col_f:
        alert_mode = st.selectbox("Show:", ["BUY & SELL only", "All signals (incl. CTB/CTS)"], key="amode", label_visibility="collapsed")
    with col_s:
        alert_sort = st.selectbox("Sort:", ["Signal Strength", "RSI (4h)", "RSI (1D)", "Rank"], key="asort", label_visibility="collapsed")

    if alert_mode == "BUY & SELL only":
        alert_df = df[df["signal"].isin(["BUY", "SELL"])].copy()
    else:
        alert_df = df[df["signal"] != "WAIT"].copy()

    if alert_sort == "RSI (4h)" and "rsi_4h" in alert_df.columns:
        alert_df = alert_df.sort_values("rsi_4h", ascending=False)
    elif alert_sort == "RSI (1D)" and "rsi_1D" in alert_df.columns:
        alert_df = alert_df.sort_values("rsi_1D", ascending=False)
    elif alert_sort == "Rank":
        alert_df = alert_df.sort_values("rank")
    else:
        sell_p = alert_df[alert_df["signal"].isin(["CTS", "SELL"])].sort_values("rsi_4h", ascending=False)
        buy_p = alert_df[alert_df["signal"].isin(["CTB", "BUY"])].sort_values("rsi_4h", ascending=True)
        alert_df = pd.concat([sell_p, buy_p])

    if alert_df.empty:
        st.info("No active alerts with current filter.")
    else:
        bs = len(alert_df[alert_df["signal"].isin(["BUY", "CTB"])])
        ss = len(alert_df[alert_df["signal"].isin(["SELL", "CTS"])])
        st.caption(f"**{len(alert_df)}** signals | 🔴 {ss} CTS/SELL  🟢 {bs} CTB/BUY")
        for _, row in alert_df.head(60).iterrows():
            st.markdown(coin_row_html(row, show_charts=True), unsafe_allow_html=True)

    if st.session_state.get("send_summary") and telegram_token and telegram_chat_id:
        check_and_send_alerts(alert_df.to_dict("records"), telegram_token, telegram_chat_id, alert_min_score, send_summary=True)
        st.success("✅ Sent!")
        st.session_state["send_summary"] = False


# ============================================================
# TAB 2: RSI HEATMAP (4 dot colors + fixed hover decimals)
# ============================================================
with tab_heatmap:
    hm1, hm2 = st.columns([3, 1])
    with hm1:
        heatmap_tf = st.selectbox("Timeframe", tf_to_scan, index=0, key="heatmap_tf")
    with hm2:
        hm_x = st.selectbox("X-Axis", ["Random Spread", "Coin Rank"], index=0, key="hm_x")

    rsi_col = f"rsi_{heatmap_tf}"
    rsi_prev_col = f"rsi_prev_{heatmap_tf}"

    if rsi_col in df.columns:
        avail = [c for c in ["symbol", rsi_col, "price", "change_24h", "signal",
                 "rank", "coin_name", "change_1h", "change_7d", "change_30d",
                 rsi_prev_col, "rsi_1D" if heatmap_tf == "4h" else "rsi_4h"] if c in df.columns]
        plot_df = df[avail].copy().dropna(subset=[rsi_col])

        if hm_x == "Coin Rank":
            plot_df["x_pos"] = plot_df["rank"].clip(upper=200)
        else:
            np.random.seed(42)
            plot_df["x_pos"] = np.random.uniform(0, 100, len(plot_df))

        # 4 DISTINCT DOT COLORS: dark red=SELL, light red=CTS, dark green=BUY, light green=CTB, gray=WAIT
        def dot_color_4(sig):
            if sig == "SELL": return "#FF3030"       # bright red
            elif sig == "CTS": return "#FF8888"      # light red
            elif sig == "BUY": return "#00DD66"      # bright green
            elif sig == "CTB": return "#88FFAA"      # light green
            return "#888888"                         # gray
        plot_df["dot_color"] = plot_df["signal"].apply(dot_color_4)

        if rsi_prev_col in plot_df.columns:
            plot_df["rsi_delta"] = plot_df[rsi_col] - plot_df[rsi_prev_col]
        else:
            plot_df["rsi_delta"] = 0

        fig = go.Figure()

        # Zones
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(255,0,0,0.12)", line_width=0,
                      annotation_text="OVERBOUGHT", annotation_position="top right", annotation_font_color="#FF6347")
        fig.add_hrect(y0=70, y1=80, fillcolor="rgba(255,99,71,0.06)", line_width=0,
                      annotation_text="STRONG", annotation_position="top right", annotation_font_color="rgba(255,99,71,0.5)")
        fig.add_hrect(y0=30, y1=40, fillcolor="rgba(50,205,50,0.06)", line_width=0,
                      annotation_text="WEAK", annotation_position="bottom right", annotation_font_color="rgba(0,255,127,0.5)")
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,127,0.12)", line_width=0,
                      annotation_text="OVERSOLD", annotation_position="bottom right", annotation_font_color="#00FF7F")

        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,99,71,0.5)", line_width=1)
        fig.add_hline(y=60, line_dash="dash", line_color="rgba(255,99,71,0.25)", line_width=0.5)
        fig.add_hline(y=40, line_dash="dash", line_color="rgba(0,255,127,0.25)", line_width=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,127,0.5)", line_width=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)

        # Trend lines
        for _, r in plot_df.iterrows():
            d = r.get("rsi_delta", 0)
            if abs(d) > 0.5:
                lc = "rgba(255,99,71,0.35)" if d < 0 else "rgba(0,255,127,0.35)"
                fig.add_trace(go.Scatter(x=[r["x_pos"], r["x_pos"]], y=[r[rsi_col] - d, r[rsi_col]],
                    mode="lines", line=dict(color=lc, width=1, dash="dot"), hoverinfo="skip", showlegend=False))

        # Hover data — FIXED: round percentages to 2 decimals
        other_rsi = "rsi_1D" if heatmap_tf == "4h" else "rsi_4h"
        other_tf = "1D" if heatmap_tf == "4h" else "4h"
        hover_texts = []
        for _, r in plot_df.iterrows():
            rk = f"#{int(r.get('rank',999))}" if r.get('rank',999) < 999 else ""
            o_rsi = r.get(other_rsi, 50) if other_rsi in plot_df.columns else 50
            sig = r.get("signal", "WAIT")
            alert_str = sig if sig != "WAIT" else "none for 24 hours"
            ch1 = r.get("change_1h", 0)
            ch24 = r.get("change_24h", 0)
            ch7 = r.get("change_7d", 0)
            ch30 = r.get("change_30d", 0)
            hover_texts.append(
                f"<b>{r.get('coin_name', r['symbol'])} - {r['symbol']} ({rk})</b><br>"
                f"RSI ({heatmap_tf}): {r[rsi_col]:.2f} | RSI ({other_tf}): {o_rsi:.2f}<br>"
                f"Price: ${r['price']:,.4f}<br>"
                f"1h, 24h, 7d, 30d: {ch1:+.2f}%, {ch24:+.2f}%, {ch7:+.2f}%, {ch30:+.2f}%<br>"
                f"Latest Alert: {alert_str}"
            )

        fig.add_trace(go.Scatter(
            x=plot_df["x_pos"], y=plot_df[rsi_col], mode="markers+text",
            text=plot_df["symbol"], textposition="top center", textfont=dict(size=9, color="white"),
            marker=dict(size=10, color=plot_df["dot_color"], opacity=0.9,
                        line=dict(width=1, color="rgba(255,255,255,0.3)")),
            hovertext=hover_texts, hoverinfo="text"))

        avg_val = plot_df[rsi_col].mean()
        fig.add_hline(y=avg_val, line_dash="dashdot", line_color="rgba(255,215,0,0.6)", line_width=1.5,
                      annotation_text=f"AVG RSI: {avg_val:.1f}", annotation_font_color="#FFD700")

        xt = "Coin Rank" if hm_x == "Coin Rank" else ""
        fig.update_layout(
            title=dict(text=f"Crypto Market RSI({heatmap_tf}) Heatmap<br><sup>{datetime.now().strftime('%d/%m/%Y %H:%M')} UTC</sup>",
                       font=dict(size=16, color="white"), x=0.5),
            template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=700,
            xaxis=dict(showticklabels=(hm_x == "Coin Rank"), showgrid=False, zeroline=False, title=xt),
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
# TAB 3: BY MARKET CAP (with sparklines + icons)
# ============================================================
with tab_market:
    sort_market = st.selectbox("Sort:", ["Rank", "1h", "24h", "7d", "30d", "RSI (4h)", "RSI (1D)"], key="msort")
    display_df = df.copy()
    sm = {"Rank": ("rank", True), "1h": ("change_1h", False), "24h": ("change_24h", False),
          "7d": ("change_7d", False), "30d": ("change_30d", False),
          "RSI (4h)": ("rsi_4h", False), "RSI (1D)": ("rsi_1D", False)}
    sc_col, sa = sm.get(sort_market, ("rank", True))
    if sc_col in display_df.columns:
        display_df = display_df.sort_values(sc_col, ascending=sa)
    for _, row in display_df.iterrows():
        st.markdown(coin_row_html(row, show_charts=True), unsafe_allow_html=True)


# ============================================================
# TAB 4: CONFLUENCE
# ============================================================
with tab_confluence:
    st.markdown("### 🎯 Confluence Scanner")
    st.caption("Multi-indicator analysis (RSI + MACD + Volume + SMC)")
    st.markdown("#### 🟢 Top Buy")
    for _, row in df[df["score"] > 0].sort_values("score", ascending=False).head(15).iterrows():
        s = row["score"]
        st.markdown(f"""<div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;margin:4px 0;">
<div style="display:flex;justify-content:space-between;"><b style="color:white;">{row['symbol']}</b>
<span style="color:#00FF7F;font-weight:bold;">{row.get('confluence_rec','WAIT')} ({s})</span></div>
<div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:6px;">
<div style="background:linear-gradient(90deg,#00FF7F,#32CD32);height:6px;width:{min(abs(s),100)}%;border-radius:4px;"></div></div>
<div style="font-size:11px;color:#888;margin-top:4px;">RSI 4h: {row.get('rsi_4h',50):.1f} | MACD: {row.get('macd_trend','—')} | Vol: {row.get('vol_trend','—')}</div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 🔴 Top Sell")
    for _, row in df[df["score"] < 0].sort_values("score").head(15).iterrows():
        s = abs(row["score"])
        st.markdown(f"""<div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;margin:4px 0;">
<div style="display:flex;justify-content:space-between;"><b style="color:white;">{row['symbol']}</b>
<span style="color:#FF6347;font-weight:bold;">{row.get('confluence_rec','WAIT')} ({row['score']})</span></div>
<div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:6px;">
<div style="background:linear-gradient(90deg,#FF6347,#FF0000);height:6px;width:{min(s,100)}%;border-radius:4px;"></div></div>
<div style="font-size:11px;color:#888;margin-top:4px;">RSI 4h: {row.get('rsi_4h',50):.1f} | MACD: {row.get('macd_trend','—')} | Vol: {row.get('vol_trend','—')}</div>
</div>""", unsafe_allow_html=True)


# ============================================================
# TAB 5: DETAIL + TradingView
# ============================================================
with tab_detail:
    sel = st.selectbox("Select Coin", df["symbol"].tolist(), key="dcoin")
    if sel:
        c = df[df["symbol"] == sel].iloc[0]
        sig = c.get("signal", "WAIT")
        sc = signal_color(sig)
        img = c.get("coin_image", "")

        st.markdown(f"""<div style="background:#1a1a2e;border-radius:10px;padding:16px;margin-bottom:12px;">
<div style="display:flex;flex-wrap:wrap;align-items:center;gap:14px;">
{icon_html(img)}
<h2 style="margin:0;color:white;">{sel}</h2>
<span style="color:#888;">{c.get('coin_name',sel)}</span>
<span class="cr">#{int(c.get('rank',999))}</span>
<span style="font-size:20px;color:white;font-weight:bold;">{fp(c['price'])}</span>
<span class="{cc(c.get('change_24h',0))}" style="font-size:16px;">{c.get('change_24h',0):+.2f}%</span>
<span style="color:{sc};font-weight:bold;font-size:16px;">Now: {sig}</span>
<span style="color:#888;">RSI (4h): <b style="color:{rc(c.get('rsi_4h',50))};">{c.get('rsi_4h',50):.2f}</b></span>
<span style="color:#888;">RSI (1D): <b>{c.get('rsi_1D',50):.2f}</b></span>
</div></div>""", unsafe_allow_html=True)

        pair = f"BINANCE:{sel}USDT"
        st.components.v1.html(f"""<div style="height:500px;background:#131722;border-radius:8px;overflow:hidden;">
<iframe src="https://s.tradingview.com/widgetembed/?symbol={pair}&interval=240&hidesidetoolbar=0&symboledit=1&saveimage=0&toolbarbg=131722&studies=%5B%22RSI%40tv-basicstudies%22%5D&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1" style="width:100%;height:500px;border:none;"></iframe></div>""", height=520)

        rsi_cols = [col for col in df.columns if col.startswith("rsi_") and "prev" not in col and "closes" not in col]
        gcols = st.columns(len(rsi_cols))
        for i, rcl in enumerate(rsi_cols):
            v = c.get(rcl, 50)
            with gcols[i]:
                fg = go.Figure(go.Indicator(mode="gauge+number", value=v,
                    title={"text": rcl.replace("rsi_", "RSI "), "font": {"size": 14, "color": "white"}},
                    number={"font": {"size": 22, "color": "white"}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#FFD700"}, "bgcolor": "#1a1a2e",
                           "steps": [{"range": [0, 30], "color": "rgba(0,255,127,0.2)"},
                                     {"range": [30, 70], "color": "rgba(255,215,0,0.1)"},
                                     {"range": [70, 100], "color": "rgba(255,99,71,0.2)"}]}))
                fg.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", height=180, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fg, use_container_width=True)

        st.markdown(f"""| Indicator | Value |
|-----------|-------|
| MACD | **{c.get('macd_trend','—')}** (hist: {c.get('macd_histogram',0):.4f}) |
| Stoch RSI | K: **{c.get('stoch_rsi_k',50):.1f}** / D: **{c.get('stoch_rsi_d',50):.1f}** |
| Volume | **{c.get('vol_trend','—')}** ({c.get('vol_ratio',1.0):.2f}x) |
| OBV | **{c.get('obv_trend','—')}** |""")

        if show_smc:
            st.markdown(f"""| SMC | Value |
|-----|-------|
| Structure | **{c.get('market_structure','—')}** |
| Order Block | **{c.get('ob_signal','—')}** |
| FVG | **{c.get('fvg_signal','—')}** |
| BOS | **{'✅' if c.get('bos') else '—'}** |""")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#555;font-size:11px;'>🧙‍♂️ Merlin Crypto Scanner | {len(coins_to_scan)} coins × {len(tf_to_scan)} TFs | {ex_name} + CoinGecko | DYOR!</div>", unsafe_allow_html=True)
