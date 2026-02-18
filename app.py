"""
🧙‍♂️ Merlin Crypto Scanner — CryptoWaves-style RSI Dashboard
"""
import time, json, base64
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
    fetch_all_market_data, fetch_klines_smart, fetch_all_tickers, get_exchange_status,
)
from indicators import (
    calculate_rsi, calculate_macd, calculate_volume_analysis,
    detect_order_blocks, detect_fair_value_gaps, detect_market_structure,
    generate_confluence_signal, calculate_stoch_rsi,
)
from alerts import check_and_send_alerts

# ============================================================
# SIGNAL ENGINE — matched to CryptoWaves.app behavior
# ============================================================
# CWApp uses persistent state. We approximate with thresholds:
#   RSI_4h >= 58 → SELL zone    RSI_4h <= 42 → BUY zone
#   Cross events for CTB/CTS    Otherwise WAIT

def compute_signal(rsi_4h, rsi_4h_prev, rsi_1d):
    # CTB: RSI_4H just crossed UP through 40
    if rsi_4h_prev < 40 and rsi_4h >= 40 and rsi_1d >= 35:
        return "CTB"
    # CTS: RSI_4H just crossed DOWN through 60
    if rsi_4h_prev > 60 and rsi_4h <= 60 and rsi_1d <= 65:
        return "CTS"
    # BUY zone: RSI_4h in oversold territory
    if rsi_4h <= 42:
        return "BUY"
    # SELL zone: RSI_4h in overbought territory
    if rsi_4h >= 58:
        return "SELL"
    return "WAIT"

def sig_color(s):
    return {"CTB": "#00FF7F", "BUY": "#00FF7F", "CTS": "#FF6347", "SELL": "#FF6347"}.get(s, "#FFD700")

def border_alpha(rsi, sig):
    if sig in ("CTS", "SELL"):
        return min(1.0, 0.35 + (rsi - 58) / 22 * 0.65) if rsi >= 58 else 0.35
    elif sig in ("CTB", "BUY"):
        return min(1.0, 0.35 + (42 - rsi) / 22 * 0.65) if rsi <= 42 else 0.35
    return 0.0

# ============================================================
# SPARKLINE (base64 img — Streamlit-safe)
# ============================================================
def sparkline_img(closes, w=110, h=30):
    if not closes or len(closes) < 3: return ""
    vals = [float(v) for v in closes if v and float(v) > 0]
    if len(vals) < 3: return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1
    pts = []
    for i, v in enumerate(vals):
        x = 2 + (i / (len(vals)-1)) * (w-4)
        y = 2 + (h-4) - ((v-mn)/rng) * (h-4)
        pts.append(f"{x:.1f},{y:.1f}")
    color = "#00FF7F" if vals[-1] >= vals[0] else "#FF6347"
    svg = f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg"><path d="M{"L".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'
    return f'<img src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}" width="{w}" height="{h}" style="vertical-align:middle;">'

# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.stApp{background:#0E1117}.block-container{padding:1rem;max-width:100%}
.hbar{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:10px;padding:12px 20px;margin-bottom:12px;display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px}
.htitle{font-size:20px;font-weight:bold;color:#FFD700}.hstat{font-size:13px;color:#ccc}.hstat b{color:white}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:bold}
.bn{background:#FFD70033;color:#FFD700}.bb{background:#00FF7F33;color:#00FF7F}.br{background:#FF634733;color:#FF6347}
.crow{background:#1a1a2e;border-radius:0 10px 10px 0;padding:10px 14px;margin:4px 0;display:flex;align-items:center;gap:12px}
.crow .ic img{width:34px;height:34px;border-radius:50%}
.crow .inf{flex:1;min-width:160px}.crow .cn{font-size:15px;font-weight:bold;color:white}
.crow .cf{font-size:11px;color:#888;margin-left:4px}.crow .cr{font-size:10px;background:#2a2a4a;padding:1px 6px;border-radius:6px;color:#888;margin-left:4px}
.crow .pl{font-size:12px;color:#aaa;margin-top:2px}.crow .chs{font-size:11px;color:#888;margin-top:2px}
.crow .charts{display:flex;gap:16px;align-items:center;flex-shrink:0}.crow .clbl{font-size:9px;color:#555;text-align:center}
.crow .sig{text-align:right;min-width:140px;flex-shrink:0}.crow .sl{font-size:15px;font-weight:bold}
.crow .rl{font-size:12px;color:#888;margin-top:1px}.crow .rl b{font-size:14px}
.cp{color:#00FF7F}.cm{color:#FF6347}
#MainMenu,footer,header{visibility:hidden}
/* Compact inline chart buttons */
.stButton>button{padding:2px 10px !important;font-size:11px !important;height:auto !important;min-height:0 !important;background:#12121f !important;color:#888 !important;border:1px solid #2a2a4a !important;border-radius:4px !important;margin-top:-6px !important}
.stButton>button:hover{color:#FFD700 !important;border-color:#FFD700 !important}
.stTabs [data-baseweb="tab-list"]{gap:4px}.stTabs [data-baseweb="tab"]{border-radius:8px;padding:8px 16px;font-weight:600}
@media(max-width:768px){.block-container{padding:.5rem}.crow .charts{display:none}.crow .inf{min-width:120px}.crow .sig{min-width:100px}}
</style>""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧙‍♂️ Settings")
    coin_list_mode = st.radio("📋 Coins", ["CryptoWaves (107+)", "Top 100 Dynamic", "Extended (180+)"], index=0)
    coin_source = {"CryptoWaves (107+)": CRYPTOWAVES_COINS, "Top 100 Dynamic": CRYPTOWAVES_COINS[:50],
                   "Extended (180+)": TOP_COINS_EXTENDED}[coin_list_mode]
    max_coins = st.slider("Max Coins", 20, 180, len(coin_source), 10)
    st.markdown("---")
    selected_timeframes = st.multiselect("Timeframes", list(TIMEFRAMES.keys()), default=["4h", "1D"])
    show_smc = st.checkbox("Smart Money Concepts", value=False)
    st.markdown("---")
    st.markdown("### 📱 Telegram")
    tg_token = st.text_input("Bot Token", type="password", key="tg_token")
    tg_chat = st.text_input("Chat ID", key="tg_chat")
    alert_min = st.slider("Min Score", 10, 80, 30, 5)
    st.markdown("---")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.caption("Data: CCXT + CoinGecko")

# ============================================================
# SCAN
# ============================================================
@st.cache_data(ttl=180, show_spinner="🧙‍♂️ Scanning crypto market...")
def scan_all(coins, tfs, smc=False):
    results = []
    ex = get_exchange_status(); connected = ex["connected"]
    mkt_df = fetch_all_market_data(); tickers = fetch_all_tickers() if connected else {}
    mkt_lk = {}
    if not mkt_df.empty:
        for _, m in mkt_df.iterrows(): mkt_lk[m.get("symbol","").upper()] = m
    for idx, sym in enumerate(list(coins)):
        r = {"symbol": sym}
        tk = tickers.get(sym, {}); mk = mkt_lk.get(sym, {})
        if tk:
            r["price"]=float(tk.get("last",0)or 0); r["change_24h"]=float(tk.get("change_pct",0)or 0); r["volume_24h"]=float(tk.get("volume",0)or 0)
        elif isinstance(mk,pd.Series) and not mk.empty:
            r["price"]=float(mk.get("price",0))if pd.notna(mk.get("price"))else 0; r["change_24h"]=float(mk.get("change_24h",0))if pd.notna(mk.get("change_24h"))else 0; r["volume_24h"]=float(mk.get("volume_24h",0))if pd.notna(mk.get("volume_24h"))else 0
        else: r["price"],r["change_24h"],r["volume_24h"]=0,0,0
        if isinstance(mk,pd.Series) and not mk.empty:
            r["rank"]=int(mk.get("rank",999))if pd.notna(mk.get("rank"))else 999
            r["coin_name"]=str(mk.get("name",sym))if pd.notna(mk.get("name"))else sym
            r["coin_image"]=str(mk.get("image",""))if pd.notna(mk.get("image"))else ""
            for f in ["change_1h","change_7d","change_30d"]:
                r[f]=float(mk.get(f,0))if pd.notna(mk.get(f))else 0
        else:
            r["rank"],r["coin_name"],r["coin_image"]=999,sym,""; r["change_1h"],r["change_7d"],r["change_30d"]=0,0,0
        if r["price"]==0: continue
        kld={}
        for tf in tfs:
            df_k = fetch_klines_smart(sym, TIMEFRAMES.get(tf,tf))
            if not df_k.empty and len(df_k)>=15:
                kld[tf]=df_k
                from ta.momentum import RSIIndicator
                rs = RSIIndicator(close=df_k["close"],window=14).rsi().dropna()
                if len(rs)>=2:
                    r[f"rsi_{tf}"]=round(float(rs.iloc[-1]),2); r[f"rsi_prev_{tf}"]=round(float(rs.iloc[-2]),2)
                else: r[f"rsi_{tf}"],r[f"rsi_prev_{tf}"]=50.0,50.0
                r[f"closes_{tf}"]=json.dumps([round(c,6) for c in df_k["close"].tail(20).tolist()])
            else: r[f"rsi_{tf}"],r[f"rsi_prev_{tf}"]=50.0,50.0; r[f"closes_{tf}"]="[]"
        r["signal"]=compute_signal(r.get("rsi_4h",50),r.get("rsi_prev_4h",50),r.get("rsi_1D",50))
        r["border_alpha"]=border_alpha(r.get("rsi_4h",50),r["signal"])
        ptf="4h" if "4h" in kld else (list(tfs)[0] if tfs else None)
        if ptf and ptf in kld:
            md=calculate_macd(kld[ptf]); r["macd_trend"]=md["trend"]; r["macd_histogram"]=md["histogram"]
            vd=calculate_volume_analysis(kld[ptf]); r["vol_trend"]=vd["vol_trend"]; r["vol_ratio"]=vd["vol_ratio"]; r["obv_trend"]=vd["obv_trend"]
            sk=calculate_stoch_rsi(kld[ptf]); r["stoch_rsi_k"]=sk["stoch_rsi_k"]; r["stoch_rsi_d"]=sk["stoch_rsi_d"]
            if smc:
                ob=detect_order_blocks(kld[ptf]); fv=detect_fair_value_gaps(kld[ptf]); ms=detect_market_structure(kld[ptf])
                r["ob_signal"]=ob["ob_signal"]; r["fvg_signal"]=fv["fvg_signal"]; r["market_structure"]=ms["structure"]; r["bos"]=ms["break_of_structure"]
            sc_d={"ob_signal":r.get("ob_signal","NONE"),"fvg_signal":r.get("fvg_signal","BALANCED"),"structure":r.get("market_structure","UNKNOWN")} if smc else None
            cf=generate_confluence_signal(rsi_4h=r.get("rsi_4h",50),rsi_1d=r.get("rsi_1D",50),macd_data=md,volume_data=vd,smc_data=sc_d)
            r["score"]=cf["score"]; r["confluence_rec"]=cf["recommendation"]; r["reasons"]=" | ".join(cf["reasons"][:3])
        else:
            r.update({"macd_trend":"NEUTRAL","macd_histogram":0,"vol_trend":"—","vol_ratio":1.0,"obv_trend":"—","stoch_rsi_k":50.0,"stoch_rsi_d":50.0,"score":0,"confluence_rec":"WAIT","reasons":""})
        results.append(r)
        if not connected and idx%5==0 and idx>0: time.sleep(2)
        elif idx%15==0 and idx>0: time.sleep(0.3)
    return pd.DataFrame(results) if results else pd.DataFrame()

# ============================================================
# LOAD
# ============================================================
coins_to_scan = coin_source[:max_coins]
tf_to_scan = selected_timeframes or ["4h","1D"]
df = scan_all(tuple(coins_to_scan), tuple(tf_to_scan), show_smc)
if df.empty: st.warning("⚠️ No data."); st.stop()

# ============================================================
# HELPERS
# ============================================================
def cc(v): return "cp" if v>=0 else "cm"
def fp(p):
    if p>=1000: return f"${p:,.2f}"
    elif p>=1: return f"${p:,.4f}"
    elif p>=0.001: return f"${p:,.6f}"
    else: return f"${p:.8f}"
def rsc(v):
    if v>70: return "#FF6347"
    elif v<30: return "#00FF7F"
    return "white"
def icon(url): return f'<img src="{url}" width="34" height="34" style="border-radius:50%;">' if url else '<div style="width:34px;height:34px;border-radius:50%;background:#2a2a4a;"></div>'

def crow_html(row, charts=True):
    sig=row.get("signal","WAIT"); ba=row.get("border_alpha",0)
    r4=row.get("rsi_4h",50); r1=row.get("rsi_1D",50)
    # Neon colors for strong signals
    is_strong_sell = sig in ("CTS","SELL") and r4 >= 70
    is_strong_buy = sig in ("CTB","BUY") and r4 <= 30
    if is_strong_sell:
        sc="#FF0040"; bdr=f"border-left:6px solid rgba(255,0,64,1.0);"  # neon red
    elif sig in ("CTS","SELL"):
        sc=sig_color(sig); bdr=f"border-left:6px solid rgba(255,80,80,{max(ba,.35):.2f});"
    elif is_strong_buy:
        sc="#00FF00"; bdr=f"border-left:6px solid rgba(0,255,0,1.0);"  # neon green
    elif sig in ("CTB","BUY"):
        sc=sig_color(sig); bdr=f"border-left:6px solid rgba(0,255,140,{max(ba,.35):.2f});"
    else:
        sc=sig_color(sig); bdr="border-left:6px solid transparent;"
    sym=row["symbol"]; nm=row.get("coin_name",sym); rk=row.get("rank",999)
    rks=f"#{int(rk)}" if rk<999 else ""; im=row.get("coin_image","")
    c1h,c24,c7,c30=row.get("change_1h",0),row.get("change_24h",0),row.get("change_7d",0),row.get("change_30d",0)
    ch=""
    if charts:
        try: s7=sparkline_img(json.loads(row.get("closes_4h","[]")),110,30)
        except: s7=""
        try: s30=sparkline_img(json.loads(row.get("closes_1D","[]")),110,30)
        except: s30=""
        ch=f'<div class="charts"><div><div class="clbl">● 7d</div>{s7}</div><div><div class="clbl">● 30d</div>{s30}</div></div>'
    # Strong glow effect on signal label
    glow = f'text-shadow:0 0 8px {sc},0 0 16px {sc};' if is_strong_sell or is_strong_buy else ""
    return f'''<div class="crow" style="{bdr}">
<div class="ic">{icon(im)}</div>
<div class="inf"><div><span class="cn">{sym}</span><span class="cf">{nm}</span><span class="cr">{rks}</span></div>
<div class="pl">Price: <b style="color:white;">{fp(row["price"])}</b></div>
<div class="chs">Ch%: <span class="{cc(c1h)}">{c1h:+.2f}%</span> <span class="{cc(c24)}">{c24:+.2f}%</span> <span class="{cc(c7)}" style="font-weight:bold;">{c7:+.2f}%</span> <span class="{cc(c30)}">{c30:+.2f}%</span></div></div>
{ch}
<div class="sig"><span style="font-size:11px;color:#888;">Now:</span> <span class="sl" style="color:{sc};{glow}">{sig}</span>
<div class="rl">RSI (4h): <b style="color:{rsc(r4)};">{r4:.2f}</b></div>
<div class="rl">RSI (1D): <b style="color:{rsc(r1)};">{r1:.2f}</b></div></div></div>'''

def render_rows_with_chart(dataframe, tab_key, max_rows=60):
    """Render coin rows with inline chart button under each row."""
    chart_state_key = f"chart_{tab_key}"
    for _,row in dataframe.head(max_rows).iterrows():
        sym=row["symbol"]
        st.markdown(crow_html(row), unsafe_allow_html=True)
        # Inline chart button
        if st.button(f"📈 {sym}", key=f"btn_{tab_key}_{sym}", use_container_width=False):
            # Toggle: click same coin = close, click different = switch
            if st.session_state.get(chart_state_key)==sym:
                st.session_state[chart_state_key]=None
            else:
                st.session_state[chart_state_key]=sym
            st.rerun()
        # Show chart if this coin is selected
        if st.session_state.get(chart_state_key)==sym:
            st.components.v1.html(tv_iframe(sym), height=440)

def tv_iframe(sym, h=420):
    pair=f"BINANCE:{sym}USDT"
    # Use TradingView Advanced Chart widget with RSI study enabled
    return (f'<div style="height:{h}px;background:#131722;border-radius:0 0 8px 8px;overflow:hidden;">'
            f'<iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tv_chart'
            f'&symbol={pair}&interval=240&hidesidetoolbar=0&symboledit=1&saveimage=0'
            f'&toolbarbg=131722&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1'
            f'&studies=%5B%7B%22id%22%3A%22RSI%40tv-basicstudies%22%2C%22inputs%22%3A%7B%22length%22%3A14%7D%7D%5D'
            f'" style="width:100%;height:{h}px;border:none;"></iframe></div>')

# ============================================================
# HEADER
# ============================================================
avg4=df["rsi_4h"].mean() if "rsi_4h" in df.columns else 50
sct=len(df[df["signal"].isin(["CTS","SELL"])]) if "signal" in df.columns else 0
bct=len(df[df["signal"].isin(["CTB","BUY"])]) if "signal" in df.columns else 0
wct=len(df)-sct-bct
if avg4>=60: ml,bc="BULLISH","bb"
elif avg4>=45: ml,bc="NEUTRAL","bn"
else: ml,bc="BEARISH","br"
a1=df["change_1h"].mean() if "change_1h" in df.columns else 0
a24=df["change_24h"].mean(); a7=df["change_7d"].mean() if "change_7d" in df.columns else 0; a30=df["change_30d"].mean() if "change_30d" in df.columns else 0
ex=get_exchange_status(); exn=ex["active_exchange"].upper() if ex["connected"] else "CoinGecko"
st.markdown(f'''<div class="hbar"><div><span class="htitle">🧙‍♂️ Merlin Crypto Scanner</span> <span class="badge {bc}">Market: {ml}</span></div>
<div class="hstat">Avg RSI (4h): <b>{avg4:.2f}</b> ({(avg4-50)/50*100:+.2f}%) | Ch%: <span class="{cc(a1)}">{a1:+.2f}%</span> <span class="{cc(a24)}">{a24:+.2f}%</span> <span class="{cc(a7)}">{a7:+.2f}%</span> <span class="{cc(a30)}">{a30:+.2f}%</span></div>
<div class="hstat"><span style="color:#FF6347;">🔴 {sct}</span> <span style="color:#FFD700;">🟡 {wct}</span> <span style="color:#00FF7F;">🟢 {bct}</span> | 📡 {exn}</div></div>''', unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab_alerts, tab_hm, tab_mc, tab_conf, tab_det = st.tabs([
    f"🚨 24h Alerts {sct}🔴 {bct}🟢", "🔥 RSI Heatmap", "📊 By Market Cap", "🎯 Confluence", "🔍 Detail"])

# ============================================================
# TAB 1: 24h ALERTS — single-click chart, Strong filter
# ============================================================
with tab_alerts:
    ct,cf,cs=st.columns([2,1,1])
    with ct: st.markdown("### 🚨 24h Alerts")
    with cf: amode=st.selectbox("Show:",["BUY & SELL only","Strong signals only","All (incl. CTB/CTS)"],key="amode",label_visibility="collapsed")
    with cs: asort=st.selectbox("Sort:",["Signal Strength","RSI (4h)","RSI (1D)","Rank"],key="asort",label_visibility="collapsed")

    if amode=="Strong signals only":
        # Strong BUY: RSI_4h <= 30, Strong SELL: RSI_4h >= 70
        adf=df[((df["signal"].isin(["BUY","CTB"])) & (df["rsi_4h"]<=30)) | ((df["signal"].isin(["SELL","CTS"])) & (df["rsi_4h"]>=70))].copy()
    elif amode=="BUY & SELL only":
        adf=df[df["signal"].isin(["BUY","SELL"])].copy()
    else:
        adf=df[df["signal"]!="WAIT"].copy()

    if asort=="RSI (4h)" and "rsi_4h" in adf.columns: adf=adf.sort_values("rsi_4h",ascending=False)
    elif asort=="RSI (1D)" and "rsi_1D" in adf.columns: adf=adf.sort_values("rsi_1D",ascending=False)
    elif asort=="Rank": adf=adf.sort_values("rank")
    else: adf=pd.concat([adf[adf["signal"].isin(["CTS","SELL"])].sort_values("rsi_4h",ascending=False),adf[adf["signal"].isin(["CTB","BUY"])].sort_values("rsi_4h")])

    if adf.empty: st.info("No active alerts with this filter.")
    else:
        st.caption(f"**{len(adf)}** signals | 🔴 {len(adf[adf['signal'].isin(['SELL','CTS'])])} SELL/CTS  🟢 {len(adf[adf['signal'].isin(['BUY','CTB'])])} BUY/CTB")
        render_rows_with_chart(adf, "alert", 40)

# ============================================================
# TAB 2: RSI HEATMAP — Coin Rank default, search, grid
# ============================================================
with tab_hm:
    h1,h2,h3=st.columns([2,1,1])
    with h1: htf=st.selectbox("Timeframe",tf_to_scan,index=0,key="htf")
    with h2: hx=st.selectbox("X-Axis",["Coin Rank","Random"],index=0,key="hx")
    with h3: search=st.text_input("🔍 Search Coin",key="hm_search",placeholder="e.g. BTC").strip().upper()

    rc_col=f"rsi_{htf}"; rp_col=f"rsi_prev_{htf}"
    if rc_col in df.columns:
        avail=[c for c in ["symbol",rc_col,"price","change_24h","signal","rank","coin_name","change_1h","change_7d","change_30d",rp_col,"rsi_1D" if htf=="4h" else "rsi_4h"] if c in df.columns]
        pdf=df[avail].copy().dropna(subset=[rc_col])
        if hx=="Coin Rank": pdf["x"]=pdf["rank"].clip(upper=200)
        else: np.random.seed(42); pdf["x"]=np.random.uniform(0,100,len(pdf))

        # 4-color dots + orange for search
        def dc(sig,sym):
            if search and search in sym: return "#FFA500"  # orange highlight
            if sig=="SELL": return "#FF3030"
            elif sig=="CTS": return "#FF8888"
            elif sig=="BUY": return "#00DD66"
            elif sig=="CTB": return "#88FFAA"
            return "#888888"
        pdf["dcol"]=pdf.apply(lambda r: dc(r["signal"],r["symbol"]),axis=1)
        # Searched coin gets larger dot
        pdf["dsz"]=pdf["symbol"].apply(lambda s: 16 if search and search in s else 10)

        if rp_col in pdf.columns: pdf["rd"]=pdf[rc_col]-pdf[rp_col]
        else: pdf["rd"]=0

        fig=go.Figure()
        # Zones
        fig.add_hrect(y0=80,y1=100,fillcolor="rgba(255,0,0,0.12)",line_width=0,annotation_text="OVERBOUGHT",annotation_position="top right",annotation_font_color="#FF6347")
        fig.add_hrect(y0=70,y1=80,fillcolor="rgba(255,99,71,0.06)",line_width=0,annotation_text="STRONG",annotation_position="top right",annotation_font_color="rgba(255,99,71,0.5)")
        fig.add_hrect(y0=30,y1=40,fillcolor="rgba(50,205,50,0.06)",line_width=0,annotation_text="WEAK",annotation_position="bottom right",annotation_font_color="rgba(0,255,127,0.5)")
        fig.add_hrect(y0=0,y1=30,fillcolor="rgba(0,255,127,0.12)",line_width=0,annotation_text="OVERSOLD",annotation_position="bottom right",annotation_font_color="#00FF7F")
        # Lines
        for y,c,w in [(70,"rgba(255,99,71,0.5)",1),(60,"rgba(255,99,71,0.25)",0.5),(40,"rgba(0,255,127,0.25)",0.5),(30,"rgba(0,255,127,0.5)",1),(50,"rgba(255,255,255,0.15)",1)]:
            fig.add_hline(y=y,line_dash="dash",line_color=c,line_width=w)
        # Trend lines
        for _,r in pdf.iterrows():
            d=r.get("rd",0)
            if abs(d)>0.5:
                lc="rgba(255,99,71,0.35)" if d<0 else "rgba(0,255,127,0.35)"
                fig.add_trace(go.Scatter(x=[r["x"],r["x"]],y=[r[rc_col]-d,r[rc_col]],mode="lines",line=dict(color=lc,width=1,dash="dot"),hoverinfo="skip",showlegend=False))
        # Hover
        otf="1D" if htf=="4h" else "4h"; ors="rsi_1D" if htf=="4h" else "rsi_4h"
        hvr=[]
        for _,r in pdf.iterrows():
            rk=f"#{int(r.get('rank',999))}" if r.get('rank',999)<999 else ""
            ov=r.get(ors,50) if ors in pdf.columns else 50
            sig=r.get("signal","WAIT"); astr=sig if sig!="WAIT" else "none for 24h"
            hvr.append(f"<b>{r.get('coin_name',r['symbol'])} - {r['symbol']} ({rk})</b><br>RSI ({htf}): {r[rc_col]:.2f} | RSI ({otf}): {ov:.2f}<br>Price: ${r['price']:,.4f}<br>1h, 24h, 7d, 30d: {r.get('change_1h',0):+.2f}%, {r.get('change_24h',0):+.2f}%, {r.get('change_7d',0):+.2f}%, {r.get('change_30d',0):+.2f}%<br>Latest Alert: {astr}")
        fig.add_trace(go.Scatter(x=pdf["x"],y=pdf[rc_col],mode="markers+text",text=pdf["symbol"],textposition="top center",textfont=dict(size=9,color="white"),
            marker=dict(size=pdf["dsz"],color=pdf["dcol"],opacity=0.9,line=dict(width=1,color="rgba(255,255,255,0.3)")),hovertext=hvr,hoverinfo="text"))
        # AVG line
        av=pdf[rc_col].mean()
        fig.add_hline(y=av,line_dash="dashdot",line_color="rgba(255,215,0,0.6)",line_width=1.5,annotation_text=f"AVG RSI: {av:.1f}",annotation_font_color="#FFD700")
        # Layout
        show_xgrid = hx=="Coin Rank"
        fig.update_layout(
            title=dict(text=f"Crypto Market RSI({htf}) Heatmap<br><sup>{datetime.now().strftime('%d/%m/%Y %H:%M')} UTC by Merlin Scanner</sup>",font=dict(size=16,color="white"),x=0.5),
            template="plotly_dark",paper_bgcolor="#0E1117",plot_bgcolor="#0E1117",height=700,
            xaxis=dict(showticklabels=show_xgrid,showgrid=show_xgrid,gridcolor="rgba(255,255,255,0.06)",zeroline=False,title="Coin Rank" if show_xgrid else "",dtick=20),
            yaxis=dict(title=f"RSI ({htf})",range=[15,90],gridcolor="rgba(255,255,255,0.05)"),
            showlegend=False,margin=dict(l=50,r=20,t=60,b=30))
        st.plotly_chart(fig,use_container_width=True)

        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown("**🔴 Most Overbought**")
            for _,r in pdf.nlargest(5,rc_col).iterrows(): st.markdown(f'<span style="color:#FF6347;">`{r["symbol"]}` RSI: **{r[rc_col]:.1f}**</span>', unsafe_allow_html=True)
        with c2:
            st.markdown("**🟢 Most Oversold**")
            for _,r in pdf.nsmallest(5,rc_col).iterrows(): st.markdown(f'<span style="color:#00FF7F;">`{r["symbol"]}` RSI: **{r[rc_col]:.1f}**</span>', unsafe_allow_html=True)
        with c3:
            st.markdown("**📊 Distribution**")
            st.markdown(f"RSI > 70: **{len(pdf[pdf[rc_col]>70])}** | 30–70: **{len(pdf[(pdf[rc_col]>=30)&(pdf[rc_col]<=70)])}** | < 30: **{len(pdf[pdf[rc_col]<30])}**")

# ============================================================
# TAB 3: BY MARKET CAP — single-click chart
# ============================================================
with tab_mc:
    sm_sort=st.selectbox("Sort:",["Rank","1h","24h","7d","30d","RSI (4h)","RSI (1D)"],key="ms")
    ddf=df.copy()
    sm={"Rank":("rank",True),"1h":("change_1h",False),"24h":("change_24h",False),"7d":("change_7d",False),"30d":("change_30d",False),"RSI (4h)":("rsi_4h",False),"RSI (1D)":("rsi_1D",False)}
    sc,sa=sm.get(sm_sort,("rank",True))
    if sc in ddf.columns: ddf=ddf.sort_values(sc,ascending=sa)
    render_rows_with_chart(ddf, "mc", 80)

# ============================================================
# TAB 4: CONFLUENCE
# ============================================================
with tab_conf:
    st.markdown("### 🎯 Confluence Scanner")
    st.markdown("#### 🟢 Top Buy")
    for _,r in df[df["score"]>0].sort_values("score",ascending=False).head(15).iterrows():
        s=r["score"]
        st.markdown(f'<div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;margin:4px 0;"><div style="display:flex;justify-content:space-between;"><b style="color:white;">{r["symbol"]}</b><span style="color:#00FF7F;font-weight:bold;">{r.get("confluence_rec","WAIT")} ({s})</span></div><div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:6px;"><div style="background:linear-gradient(90deg,#00FF7F,#32CD32);height:6px;width:{min(abs(s),100)}%;border-radius:4px;"></div></div><div style="font-size:11px;color:#888;margin-top:4px;">RSI 4h: {r.get("rsi_4h",50):.1f} | MACD: {r.get("macd_trend","—")} | Vol: {r.get("vol_trend","—")}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 🔴 Top Sell")
    for _,r in df[df["score"]<0].sort_values("score").head(15).iterrows():
        s=abs(r["score"])
        st.markdown(f'<div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;margin:4px 0;"><div style="display:flex;justify-content:space-between;"><b style="color:white;">{r["symbol"]}</b><span style="color:#FF6347;font-weight:bold;">{r.get("confluence_rec","WAIT")} ({r["score"]})</span></div><div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:6px;"><div style="background:linear-gradient(90deg,#FF6347,#FF0000);height:6px;width:{min(s,100)}%;border-radius:4px;"></div></div><div style="font-size:11px;color:#888;margin-top:4px;">RSI 4h: {r.get("rsi_4h",50):.1f} | MACD: {r.get("macd_trend","—")} | Vol: {r.get("vol_trend","—")}</div></div>', unsafe_allow_html=True)

# ============================================================
# TAB 5: DETAIL + TradingView
# ============================================================
with tab_det:
    sel=st.selectbox("Select Coin",df["symbol"].tolist(),key="dc")
    if sel:
        c=df[df["symbol"]==sel].iloc[0]; sig=c.get("signal","WAIT"); sc=sig_color(sig); im=c.get("coin_image","")
        st.markdown(f'<div style="background:#1a1a2e;border-radius:10px;padding:16px;margin-bottom:12px;"><div style="display:flex;flex-wrap:wrap;align-items:center;gap:14px;">{icon(im)}<h2 style="margin:0;color:white;">{sel}</h2><span style="color:#888;">{c.get("coin_name",sel)}</span><span class="cr">#{int(c.get("rank",999))}</span><span style="font-size:20px;color:white;font-weight:bold;">{fp(c["price"])}</span><span class="{cc(c.get("change_24h",0))}" style="font-size:16px;">{c.get("change_24h",0):+.2f}%</span><span style="color:{sc};font-weight:bold;font-size:16px;">Now: {sig}</span><span style="color:#888;">RSI (4h): <b style="color:{rsc(c.get("rsi_4h",50))};">{c.get("rsi_4h",50):.2f}</b></span><span style="color:#888;">RSI (1D): <b>{c.get("rsi_1D",50):.2f}</b></span></div></div>', unsafe_allow_html=True)
        st.components.v1.html(tv_iframe(sel,500),height=520)
        rcs=[col for col in df.columns if col.startswith("rsi_") and "prev" not in col and "closes" not in col]
        gc=st.columns(len(rcs))
        for i,rcl in enumerate(rcs):
            v=c.get(rcl,50)
            with gc[i]:
                fg=go.Figure(go.Indicator(mode="gauge+number",value=v,title={"text":rcl.replace("rsi_","RSI "),"font":{"size":14,"color":"white"}},number={"font":{"size":22,"color":"white"}},gauge={"axis":{"range":[0,100]},"bar":{"color":"#FFD700"},"bgcolor":"#1a1a2e","steps":[{"range":[0,30],"color":"rgba(0,255,127,0.2)"},{"range":[30,70],"color":"rgba(255,215,0,0.1)"},{"range":[70,100],"color":"rgba(255,99,71,0.2)"}]}))
                fg.update_layout(template="plotly_dark",paper_bgcolor="#0E1117",height=180,margin=dict(l=20,r=20,t=40,b=10))
                st.plotly_chart(fg,use_container_width=True)
        st.markdown(f"| Indicator | Value |\n|---|---|\n| MACD | **{c.get('macd_trend','—')}** (hist: {c.get('macd_histogram',0):.4f}) |\n| Stoch RSI | K: **{c.get('stoch_rsi_k',50):.1f}** / D: **{c.get('stoch_rsi_d',50):.1f}** |\n| Volume | **{c.get('vol_trend','—')}** ({c.get('vol_ratio',1.0):.2f}x) |\n| OBV | **{c.get('obv_trend','—')}** |")

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#555;font-size:11px;'>🧙‍♂️ Merlin | {len(coins_to_scan)}×{len(tf_to_scan)}TF | {exn} | BUY≤42 SELL≥58 | DYOR!</div>",unsafe_allow_html=True)
