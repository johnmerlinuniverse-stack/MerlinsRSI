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
    fetch_fear_greed_index, fetch_funding_rates_batch,
)
from indicators import (
    calculate_rsi, calculate_macd, calculate_volume_analysis,
    detect_order_blocks, detect_fair_value_gaps, detect_market_structure,
    generate_confluence_signal, calculate_stoch_rsi,
    calculate_ema_alignment_fast, detect_rsi_divergence, calculate_bb_squeeze,
    compute_individual_scores, compute_confluence_total,
)
from alerts import check_and_send_alerts

# ============================================================
# SIGNAL ENGINE — Multi-TF RSI Signal
# ============================================================
# Core logic from CryptoWaves.app (4h-based).
# Extended: user can toggle 1h, 1D, 1W to influence the signal.
# 4H is ALWAYS ON as the primary timeframe.

# TF weights for multi-TF scoring
SIG_TF_WEIGHTS = {"1h": 1.0, "4h": 3.0, "1D": 2.0, "1W": 1.5}

def compute_signal_base(rsi_4h, rsi_4h_prev, rsi_1d):
    """Original CryptoWaves-compatible 4h signal (always computed)."""
    if rsi_4h_prev < 40 and rsi_4h >= 40 and rsi_1d >= 35:
        return "CTB"
    if rsi_4h_prev > 60 and rsi_4h <= 60 and rsi_1d <= 65:
        return "CTS"
    if rsi_4h <= 42:
        return "BUY"
    if rsi_4h >= 58:
        return "SELL"
    return "WAIT"

def compute_signal_multi_tf(row, active_sig_tfs):
    """
    Compute signal using multiple timeframes.
    Each TF votes BUY/SELL with its weight.
    If only 4H is active → identical to CryptoWaves logic.
    """
    # If only 4h selected → original CryptoWaves logic
    if active_sig_tfs == {"4h"}:
        return compute_signal_base(
            row.get("rsi_4h", 50), row.get("rsi_prev_4h", 50), row.get("rsi_1D", 50))

    bull = 0.0; bear = 0.0
    for tf in active_sig_tfs:
        rsi = row.get(f"rsi_{tf}", 50)
        w = SIG_TF_WEIGHTS.get(tf, 1.0)
        if rsi == 50.0:  # Skip fallback values (no data)
            continue
        if rsi <= 25: bull += w * 2.0    # strongly oversold
        elif rsi <= 42: bull += w * 1.0  # oversold
        elif rsi >= 75: bear += w * 2.0  # strongly overbought
        elif rsi >= 58: bear += w * 1.0  # overbought

    # CTB/CTS cross detection (only from 4h)
    if "4h" in active_sig_tfs:
        rsi_4h = row.get("rsi_4h", 50)
        rsi_4h_prev = row.get("rsi_prev_4h", 50)
        rsi_1d = row.get("rsi_1D", 50)
        if rsi_4h_prev < 40 and rsi_4h >= 40 and rsi_1d >= 35 and bull >= bear:
            return "CTB"
        if rsi_4h_prev > 60 and rsi_4h <= 60 and rsi_1d <= 65 and bear >= bull:
            return "CTS"

    # Net score determines signal
    net = bull - bear
    if net >= 2.0: return "BUY"
    elif net <= -2.0: return "SELL"
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

# Font size from session state (default: 1 = normal)
fz = st.session_state.get("font_scale", 1.0)

st.markdown(f"""<style>
:root{{--fz:{fz}}}
.stApp{{background:#0E1117}}.block-container{{padding:1rem;max-width:100%}}
.hbar{{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:10px;padding:12px 20px;margin-bottom:12px;display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px}}
.htitle{{font-size:calc(20px * var(--fz));font-weight:bold;color:#FFD700}}.hstat{{font-size:calc(14px * var(--fz));color:#ccc}}.hstat b{{color:white}}
.badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:calc(13px * var(--fz));font-weight:bold}}
.bn{{background:#FFD70033;color:#FFD700}}.bb{{background:#00FF7F33;color:#00FF7F}}.br{{background:#FF634733;color:#FF6347}}
.crow{{background:#1a1a2e;border-radius:0 10px 10px 0;padding:10px 14px;margin:4px 0;display:flex;align-items:center;gap:12px}}
.crow .ic img{{width:34px;height:34px;border-radius:50%}}
.crow .inf{{flex:1;min-width:160px}}.crow .cn{{font-size:calc(16px * var(--fz));font-weight:bold;color:white}}
.crow .cf{{font-size:calc(12px * var(--fz));color:#888;margin-left:4px}}.crow .cr{{font-size:calc(11px * var(--fz));background:#2a2a4a;padding:1px 6px;border-radius:6px;color:#888;margin-left:4px}}
.crow .pl{{font-size:calc(14px * var(--fz));color:#aaa;margin-top:2px}}.crow .chs{{font-size:calc(12px * var(--fz));color:#888;margin-top:2px}}
.crow .charts{{display:flex;gap:16px;align-items:center;flex-shrink:0}}.crow .clbl{{font-size:calc(10px * var(--fz));color:#555;text-align:center}}
.crow .sig{{text-align:right;min-width:170px;flex-shrink:0}}.crow .sl{{font-size:calc(16px * var(--fz));font-weight:bold}}
.crow .rl{{font-size:calc(12px * var(--fz));color:#888;margin-top:1px}}.crow .rl b{{font-size:calc(14px * var(--fz))}}
.crow .rsi-row{{display:flex;gap:6px;margin-top:3px;flex-wrap:wrap}}.crow .rsi-pill{{font-size:calc(11px * var(--fz));padding:1px 5px;border-radius:4px;background:#1e1e3a}}
.cp{{color:#00FF7F}}.cm{{color:#FF6347}}
#MainMenu,footer,header{{visibility:hidden}}
.stButton>button{{padding:2px 10px !important;font-size:calc(12px * var(--fz)) !important;height:auto !important;min-height:0 !important;background:#12121f !important;color:#888 !important;border:1px solid #2a2a4a !important;border-radius:4px !important;margin-top:-6px !important}}
.stButton>button:hover{{color:#FFD700 !important;border-color:#FFD700 !important}}
.stTabs [data-baseweb="tab-list"]{{gap:4px}}.stTabs [data-baseweb="tab"]{{border-radius:8px;padding:8px 16px;font-weight:600;font-size:calc(14px * var(--fz))}}
@media(max-width:768px){{.block-container{{padding:.5rem}}.crow .charts{{display:none}}.crow .inf{{min-width:120px}}.crow .sig{{min-width:120px}}}}
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

    st.markdown("### 🔤 Schriftgröße")
    font_scale = st.select_slider("Text", options=[0.85, 0.9, 1.0, 1.1, 1.2, 1.3],
        value=st.session_state.get("font_scale", 1.0), format_func=lambda x: {0.85:"Klein",0.9:"Kompakt",1.0:"Normal",1.1:"Groß",1.2:"Sehr groß",1.3:"XXL"}[x], key="fs_slider")
    if font_scale != st.session_state.get("font_scale", 1.0):
        st.session_state["font_scale"] = font_scale; st.rerun()
    st.markdown("---")

    st.markdown("### ⏱️ Extra Timeframes")
    st.caption("RSI wird **immer** auf 1h, 4h, 1D, 1W berechnet. Hier kannst du zusätzliche Analysen aktivieren:")
    selected_timeframes = st.multiselect("Heatmap-Timeframes",
        list(TIMEFRAMES.keys()), default=["4h", "1D"],
        help="Welche Timeframes im RSI-Heatmap-Dropdown zur Auswahl stehen")
    st.markdown("---")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.caption("Data: CCXT + CoinGecko")

# ============================================================
# CONFLUENCE FILTERS (for Confluence tab — checkboxes)
# ============================================================
CONFLUENCE_FILTERS = {
    "rsi_4h":         {"label": "📊 RSI 4h",           "default": True,  "weight": 30, "desc": "Der wichtigste Faktor — Basis der App"},
    "rsi_1d":         {"label": "📈 RSI 1D",           "default": True,  "weight": 20, "desc": "Bestätigung vom Tages-Trend"},
    "macd":           {"label": "📉 MACD",             "default": True,  "weight": 20, "desc": "Momentum-Richtung + Histogramm"},
    "volume_obv":     {"label": "🔊 Volume & OBV",     "default": True,  "weight": 15, "desc": "Volumen + OBV-Richtung"},
    "rsi_divergence": {"label": "🔀 RSI Divergenz",    "default": True,  "weight": 15, "desc": "Preis-RSI Divergenzen (Regular + Hidden)"},
    "smart_money":    {"label": "🏦 Smart Money",      "default": False, "weight": 15, "desc": "Order Blocks + Market Structure (BOS/CHoCH)"},
    "ema_alignment":  {"label": "📐 EMA Alignment",    "default": True,  "weight": 12, "desc": "Preis vs EMA 9/21/50 Ausrichtung"},
    "stoch_rsi":      {"label": "🎛️ Stoch RSI",        "default": True,  "weight": 12, "desc": "Feintuning K/D Crossover"},
    "bollinger":      {"label": "📏 Bollinger Bands",   "default": False, "weight": 10, "desc": "Squeeze-Erkennung + Band-Position"},
    "funding_rate":   {"label": "💰 Funding Rate",     "default": False, "weight": 10, "desc": "Futures Funding Rate (konträr)"},
    "fear_greed":     {"label": "😱 Fear & Greed",     "default": False, "weight": 8,  "desc": "Markt-Sentiment (konträr)"},
}

# Session state for confluence filters (reset if defaults changed)
_default_conf = {k: v["default"] for k, v in CONFLUENCE_FILTERS.items()}
if "conf_filters" not in st.session_state or set(st.session_state["conf_filters"].keys()) != set(_default_conf.keys()):
    st.session_state["conf_filters"] = _default_conf

# Max possible score from active filters
FILTER_WEIGHTS = {k: v["weight"] for k, v in CONFLUENCE_FILTERS.items()}

def score_badge_html(row):
    """Generate inline score badge for use in crow_html."""
    sc = row.get("score", 0)
    rec = row.get("confluence_rec", "WAIT")
    if sc >= 30:
        return f'<span style="color:#00FF7F;font-size:11px;margin-left:6px;" title="{rec}">⚡{sc}</span>'
    elif sc >= 10:
        return f'<span style="color:#88FFAA;font-size:11px;margin-left:6px;" title="{rec}">↑{sc}</span>'
    elif sc <= -30:
        return f'<span style="color:#FF6347;font-size:11px;margin-left:6px;" title="{rec}">⚡{sc}</span>'
    elif sc <= -10:
        return f'<span style="color:#FF8888;font-size:11px;margin-left:6px;" title="{rec}">↓{sc}</span>'
    else:
        return f'<span style="color:#888;font-size:11px;margin-left:6px;" title="{rec}">–{sc}</span>'

# ============================================================
# SCAN (IDENTICAL to working original — DO NOT MODIFY)
# ============================================================
@st.cache_data(ttl=300, show_spinner="🧙‍♂️ Scanning crypto market...")
def scan_all(coins, tfs, smc=False):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from ta.momentum import RSIIndicator
    core_tfs = ["1h", "4h", "1D", "1W"]
    all_tfs = list(dict.fromkeys(core_tfs + list(tfs)))
    ex = get_exchange_status(); connected = ex["connected"]
    mkt_df = fetch_all_market_data(); tickers = fetch_all_tickers() if connected else {}
    mkt_lk = {}
    if not mkt_df.empty:
        for _, m in mkt_df.iterrows(): mkt_lk[m.get("symbol","").upper()] = m

    coin_list = list(coins)

    # --- PHASE 1: Parallel klines fetch (biggest bottleneck) ---
    klines_cache = {}  # (sym, tf) → DataFrame
    fetch_tasks = [(sym, tf) for sym in coin_list for tf in all_tfs]

    def _fetch_one(sym_tf):
        sym, tf = sym_tf
        try:
            return sym_tf, fetch_klines_smart(sym, TIMEFRAMES.get(tf, tf))
        except Exception:
            return sym_tf, pd.DataFrame()

    workers = 6 if connected else 3
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in fetch_tasks}
        for fut in as_completed(futures):
            try:
                key, df_k = fut.result()
                klines_cache[key] = df_k
            except Exception:
                pass

    # Track kline fetch success for diagnostics
    kl_total = len(fetch_tasks)
    kl_ok = sum(1 for v in klines_cache.values() if not v.empty and len(v) >= 15)
    kl_fail = kl_total - kl_ok

    # --- PHASE 2: Process results (CPU-only, fast) ---
    results = []
    for sym in coin_list:
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
        for tf in all_tfs:
            df_k = klines_cache.get((sym, tf), pd.DataFrame())
            if not df_k.empty and len(df_k)>=15:
                kld[tf]=df_k
                rs = RSIIndicator(close=df_k["close"],window=14).rsi().dropna()
                if len(rs)>=2:
                    r[f"rsi_{tf}"]=round(float(rs.iloc[-1]),2); r[f"rsi_prev_{tf}"]=round(float(rs.iloc[-2]),2)
                else: r[f"rsi_{tf}"],r[f"rsi_prev_{tf}"]=50.0,50.0
                r[f"closes_{tf}"]=json.dumps([round(c,6) for c in df_k["close"].tail(20).tolist()])
            else: r[f"rsi_{tf}"],r[f"rsi_prev_{tf}"]=50.0,50.0; r[f"closes_{tf}"]="[]"
        r["signal"]=compute_signal_base(r.get("rsi_4h",50),r.get("rsi_prev_4h",50),r.get("rsi_1D",50))
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
    return pd.DataFrame(results) if results else pd.DataFrame()

# ============================================================
# LOAD
# ============================================================
coins_to_scan = coin_source[:max_coins]
tf_to_scan = selected_timeframes or ["4h","1D"]
df = scan_all(tuple(coins_to_scan), tuple(tf_to_scan), False)
if df.empty: st.warning("⚠️ No data."); st.stop()

# ============================================================
# SIGNAL TF TOGGLES (inline, above header)
# ============================================================
if "sig_tfs" not in st.session_state:
    st.session_state["sig_tfs"] = {"4h": True, "1h": False, "1D": False, "1W": False}

stc1, stc2, stc3, stc4, stc5 = st.columns([1.5, 1, 1, 1, 1])
with stc1:
    st.markdown('<span style="color:#FFD700;font-weight:bold;font-size:13px;">📡 Signal-TFs:</span>', unsafe_allow_html=True)
with stc2:
    v = st.checkbox("4H ★", value=True, disabled=True, key="stf_4h",
                    help="4H ist immer aktiv (Kernstück der CryptoWaves-Logik)")
    st.session_state["sig_tfs"]["4h"] = True  # always on
with stc3:
    v = st.checkbox("1H", value=st.session_state["sig_tfs"].get("1h", False), key="stf_1h",
                    help="1H RSI zum Signal hinzufügen")
    st.session_state["sig_tfs"]["1h"] = v
with stc4:
    v = st.checkbox("1D", value=st.session_state["sig_tfs"].get("1D", False), key="stf_1d",
                    help="1D RSI zum Signal hinzufügen")
    st.session_state["sig_tfs"]["1D"] = v
with stc5:
    v = st.checkbox("1W", value=st.session_state["sig_tfs"].get("1W", False), key="stf_1w",
                    help="1W RSI zum Signal hinzufügen")
    st.session_state["sig_tfs"]["1W"] = v

active_sig_tfs = {tf for tf, on in st.session_state["sig_tfs"].items() if on}

# Re-compute signal with multi-TF (does NOT re-run scan_all)
df["signal"] = df.apply(lambda row: compute_signal_multi_tf(row, active_sig_tfs), axis=1)
df["border_alpha"] = df.apply(lambda row: border_alpha(row.get("rsi_4h", 50), row["signal"]), axis=1)

# ============================================================
# DIAGNOSTICS (detect kline failures + image issues)
# ============================================================
if "rsi_4h" in df.columns:
    rsi_ok = len(df[df["rsi_4h"] != 50.0])
    rsi_fail = len(df[df["rsi_4h"] == 50.0])
    total = len(df)
    if rsi_fail > total * 0.5:
        st.warning(f"⚠️ **Kline-Fetch Problem erkannt:** {rsi_fail}/{total} Coins haben RSI=50.0 (keine Daten). "
                   f"Das deutet auf Rate-Limiting der Exchange hin. Klicke **🔄 Refresh** in der Sidebar um den Cache zu leeren und erneut zu laden.")
    with st.expander(f"🔧 Diagnostik — {rsi_ok}/{total} Coins mit echten RSI-Daten", expanded=False):
        ex_info = get_exchange_status()
        st.markdown(f"**Exchange:** {ex_info['active_exchange'].upper()} ({'✅ verbunden' if ex_info['connected'] else '❌ nicht verbunden'})")
        st.markdown(f"**Klines OK:** {rsi_ok} / {total} ({rsi_ok/total*100:.0f}%)")
        st.markdown(f"**Klines fehlend:** {rsi_fail} / {total}")
        st.markdown(f"**Signal-TFs aktiv:** {', '.join(sorted(active_sig_tfs))}")
        if rsi_fail > 0:
            fail_coins = df[df["rsi_4h"] == 50.0]["symbol"].tolist()[:20]
            st.markdown(f"**Betroffene Coins (max 20):** {', '.join(fail_coins)}")
        # Image diagnostic
        if "coin_image" in df.columns:
            img_ok = len(df[df["coin_image"].astype(str).str.startswith("http")])
            img_fail = total - img_ok
            st.markdown(f"**Logos:** {img_ok}/{total} mit Bild-URL")
            if img_fail > total * 0.5:
                st.markdown("⚠️ Viele Coins ohne Logo — CoinGecko-Daten möglicherweise nicht geladen.")
            if img_ok > 0:
                sample = df[df["coin_image"].astype(str).str.startswith("http")].iloc[0]
                st.markdown(f"**Beispiel-URL:** `{sample['coin_image'][:80]}...`")
                st.markdown(f'Test: <img src="{sample["coin_image"]}" width="24" height="24" style="border-radius:50%;vertical-align:middle;"> ← Bild sichtbar?', unsafe_allow_html=True)
            if img_fail > 0:
                no_img = df[~df["coin_image"].astype(str).str.startswith("http")]["symbol"].tolist()[:10]
                st.markdown(f"**Coins ohne Logo:** {', '.join(no_img)}")

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
def icon(url, sym=""):
    """Coin icon with fallback sources."""
    if url and url.startswith("http"):
        return f'<img src="{url}" width="34" height="34" style="border-radius:50%;" onerror="this.onerror=null;this.src=\'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/{sym.lower()}.png\';this.onerror=function(){{this.style.display=\'none\';this.parentElement.innerHTML=\'<div style=&quot;width:34px;height:34px;border-radius:50%;background:#2a2a4a;display:flex;align-items:center;justify-content:center;color:#888;font-size:12px;font-weight:bold;&quot;>{sym[:2]}</div>\';}}">'
    elif sym:
        # No CoinGecko URL — try GitHub icons first
        gh_url = f"https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/{sym.lower()}.png"
        return f'<img src="{gh_url}" width="34" height="34" style="border-radius:50%;" onerror="this.onerror=null;this.style.display=\'none\';this.parentElement.innerHTML=\'<div style=&quot;width:34px;height:34px;border-radius:50%;background:#2a2a4a;display:flex;align-items:center;justify-content:center;color:#888;font-size:12px;font-weight:bold;&quot;>{sym[:2]}</div>\';">'
    return f'<div style="width:34px;height:34px;border-radius:50%;background:#2a2a4a;display:flex;align-items:center;justify-content:center;color:#888;font-size:12px;font-weight:bold;">{sym[:2]}</div>'

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
    r1h=row.get("rsi_1h",50); r4=row.get("rsi_4h",50); r1d=row.get("rsi_1D",50); r1w=row.get("rsi_1W",50)
    # Confluence score — separate line
    conf_sc = row.get("score", 0)
    conf_rec = row.get("confluence_rec", "WAIT")
    if conf_sc >= 30: conf_html = f'<span style="color:#00FF7F;font-size:10px;">⚡ {conf_rec} ({conf_sc})</span>'
    elif conf_sc >= 10: conf_html = f'<span style="color:#88FFAA;font-size:10px;">↑ {conf_rec} ({conf_sc})</span>'
    elif conf_sc <= -30: conf_html = f'<span style="color:#FF6347;font-size:10px;">⚡ {conf_rec} ({conf_sc})</span>'
    elif conf_sc <= -10: conf_html = f'<span style="color:#FF8888;font-size:10px;">↓ {conf_rec} ({conf_sc})</span>'
    else: conf_html = f'<span style="color:#666;font-size:10px;">{conf_rec} ({conf_sc})</span>'
    return f'''<div class="crow" style="{bdr}">
<div class="ic">{icon(im, sym)}</div>
<div class="inf"><div><span class="cn">{sym}</span><span class="cf">{nm}</span><span class="cr">{rks}</span></div>
<div class="pl">Price: <b style="color:white;">{fp(row["price"])}</b></div>
<div class="chs">Ch%: <span class="{cc(c1h)}">{c1h:+.2f}%</span> <span class="{cc(c24)}">{c24:+.2f}%</span> <span class="{cc(c7)}" style="font-weight:bold;">{c7:+.2f}%</span> <span class="{cc(c30)}">{c30:+.2f}%</span></div></div>
{ch}
<div class="sig"><span style="font-size:11px;color:#888;">RSI:</span> <span class="sl" style="color:{sc};{glow}">{sig}</span>
<div class="rsi-row"><span class="rsi-pill">1h: <b style="color:{rsc(r1h)};">{r1h:.1f}</b></span><span class="rsi-pill" style="background:#252550;"><b style="color:{rsc(r4)};">{r4:.1f}</b> 4h</span><span class="rsi-pill" style="background:#252550;"><b style="color:{rsc(r1d)};">{r1d:.1f}</b> 1D</span><span class="rsi-pill">1W: <b style="color:{rsc(r1w)};">{r1w:.1f}</b></span></div>
<div style="margin-top:2px;">Conf: {conf_html}</div>
</div></div>'''

def render_rows_with_chart(dataframe, tab_key, max_rows=60):
    """Render coin rows with inline chart button under each row."""
    chart_state_key = f"chart_{tab_key}"
    for _,row in dataframe.head(max_rows).iterrows():
        sym=row["symbol"]
        st.markdown(crow_html(row), unsafe_allow_html=True)
        # Inline buttons: Chart + Detail (compact, equal width)
        bc1, bc2, _ = st.columns([1, 1, 10])
        with bc1:
            if st.button(f"📈 Chart", key=f"btn_{tab_key}_{sym}", use_container_width=True):
                if st.session_state.get(chart_state_key)==sym:
                    st.session_state[chart_state_key]=None
                else:
                    st.session_state[chart_state_key]=sym
                st.rerun()
        with bc2:
            if st.button(f"🔍 Detail", key=f"det_{tab_key}_{sym}", use_container_width=True):
                st.session_state["dc"] = sym
                st.session_state["_go_detail"] = True
                st.rerun()
        # Show chart if this coin is selected
        if st.session_state.get(chart_state_key)==sym:
            st.components.v1.html(tv_chart_simple(sym), height=600)

def tv_chart_simple(sym, h=580):
    """Clean widgetembed chart with RSI only — fast, compact."""
    pair=f"BINANCE:{sym}USDT"
    return (f'<div style="height:{h}px;background:#131722;border-radius:0 0 8px 8px;overflow:hidden;">'
            f'<iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tv_{sym}'
            f'&symbol={pair}&interval=240&hidesidetoolbar=0&symboledit=1&saveimage=0'
            f'&toolbarbg=131722&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1'
            f'&allow_symbol_change=1'
            f'&studies=RSI%40tv-basicstudies'
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
sig_tfs_str = "+".join(sorted(active_sig_tfs, key=lambda t: ["1h","4h","1D","1W"].index(t) if t in ["1h","4h","1D","1W"] else 99))
st.markdown(f'''<div class="hbar"><div><span class="htitle">🧙‍♂️ Merlin Crypto Scanner</span> <span class="badge {bc}">Market: {ml}</span></div>
<div class="hstat">Avg RSI (4h): <b>{avg4:.2f}</b> ({(avg4-50)/50*100:+.2f}%) | Ch%: <span class="{cc(a1)}">{a1:+.2f}%</span> <span class="{cc(a24)}">{a24:+.2f}%</span> <span class="{cc(a7)}">{a7:+.2f}%</span> <span class="{cc(a30)}">{a30:+.2f}%</span></div>
<div class="hstat"><span style="color:#FF6347;">🔴 {sct}</span> <span style="color:#FFD700;">🟡 {wct}</span> <span style="color:#00FF7F;">🟢 {bct}</span> | 📡 {exn} | Sig: {sig_tfs_str}</div></div>''', unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab_alerts, tab_hm, tab_mc, tab_conf, tab_det = st.tabs([
    f"🚨 24h Alerts {sct}🔴 {bct}🟢", "🔥 RSI Heatmap", "📊 By Market Cap", "🎯 Confluence", "🔍 Detail"])

# Auto-switch to Detail tab when requested
if st.session_state.pop("_go_detail", False):
    st.components.v1.html("""
    <script>
    const tabs = window.parent.document.querySelectorAll('[role="tab"]');
    for (const tab of tabs) {
        if (tab.textContent.includes('Detail')) {
            tab.click();
            break;
        }
    }
    </script>
    """, height=0)

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

        if hx=="Coin Rank":
            # Sort by rank, then use sequential 1-based position
            # This prevents gaps (e.g., coins ranked #5, #12, #23 → x=1, 2, 3)
            pdf = pdf.sort_values("rank").copy()
            pdf["x"] = range(1, len(pdf) + 1)
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
        x_max_val = pdf["x"].max() if not pdf.empty else 100
        fig.update_layout(
            title=dict(text=f"Crypto Market RSI({htf}) Heatmap<br><sup>{datetime.now().strftime('%d/%m/%Y %H:%M')} UTC by Merlin Scanner</sup>",font=dict(size=16,color="white"),x=0.5),
            template="plotly_dark",paper_bgcolor="#0E1117",plot_bgcolor="#0E1117",height=700,
            xaxis=dict(showticklabels=show_xgrid,showgrid=show_xgrid,gridcolor="rgba(255,255,255,0.06)",zeroline=False,title="Position (sortiert nach Rank)" if show_xgrid else "",dtick=10,range=[0, x_max_val + 3]),
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
# TAB 4: CONFLUENCE — Enhanced with checkboxes + lazy indicator computation
# ============================================================
with tab_conf:
    st.markdown("### 🎯 Confluence Scanner")

    # --- FILTER CHECKBOXES ---
    with st.expander("⚙️ Confluence Filter konfigurieren", expanded=True):
        st.markdown('<span style="font-size:12px;color:#888;">Aktiviere/deaktiviere Faktoren für die Score-Berechnung. Der Score wird auf -100...+100 normalisiert.</span>', unsafe_allow_html=True)

        # Lazy-load Fear & Greed (only here, never blocks main scan)
        try:
            fg = fetch_fear_greed_index()
            fear_greed_val = fg.get("value", None)
        except Exception:
            fear_greed_val = None
        if fear_greed_val is not None:
            fg_col = "#00FF7F" if fear_greed_val <= 30 else ("#FF6347" if fear_greed_val >= 70 else "#FFD700")
            st.markdown(f'<span style="font-size:11px;color:#888;">📡 Fear & Greed Index: </span><span style="color:{fg_col};font-weight:bold;">{fear_greed_val}</span>', unsafe_allow_html=True)

        cols = st.columns(3)
        filter_keys = list(CONFLUENCE_FILTERS.keys())
        for i, key in enumerate(filter_keys):
            meta = CONFLUENCE_FILTERS[key]
            with cols[i % 3]:
                new_val = st.checkbox(
                    f"{meta['label']} (±{meta['weight']})",
                    value=st.session_state["conf_filters"].get(key, meta["default"]),
                    help=meta["desc"],
                    key=f"cf_{key}")
                st.session_state["conf_filters"][key] = new_val

        active_count = sum(1 for v in st.session_state["conf_filters"].values() if v)
        st.markdown(f'<span style="font-size:11px;color:#888;">✅ **{active_count}** von {len(CONFLUENCE_FILTERS)} Filtern aktiv</span>', unsafe_allow_html=True)

    with st.expander("ℹ️ Was ist der Confluence Scanner?"):
        st.markdown("""**Der Confluence Scanner** bewertet jeden Coin anhand **mehrerer unabhängiger Indikatoren** und berechnet einen normalisierten Score von -100 bis +100.

**Die 5 Standard-Filter (vorausgewählt):** RSI 4h (±30), RSI 1D (±20), MACD (±20), Volume & OBV (±15), Stoch RSI (±12)

**Zusätzliche Filter (optional):** Smart Money (±15), EMA Alignment (±12), RSI Divergenz (±15), Bollinger Bands (±10), Funding Rate (±10), Fear & Greed (±8)

**Score:** ≥60 STRONG BUY · 30-59 BUY · 10-29 LEAN BUY · -9 bis 9 WAIT · -29 bis -10 LEAN SELL · -59 bis -30 SELL · ≤-60 STRONG SELL

💡 **Score-Badge** (z.B. ⚡42) erscheint auch in *24h Alerts* und *By Market Cap*.""")

    # --- COMPUTE EXTENDED SCORES ON-DEMAND ---
    # This is the ONLY place where additional indicators are computed.
    # scan_all remains untouched — we fetch extra klines here if needed.
    active_filters = st.session_state.get("conf_filters", {k: v["default"] for k, v in CONFLUENCE_FILTERS.items()})
    needs_extra = any(active_filters.get(k, False) for k in ["smart_money", "ema_alignment", "rsi_divergence", "bollinger", "funding_rate", "fear_greed"])

    # Lazy-load funding rates only if filter is active
    funding_rates_map = {}
    if active_filters.get("funding_rate", False):
        try:
            funding_rates_map = fetch_funding_rates_batch()
        except Exception:
            funding_rates_map = {}

    # Compute scores for each coin
    scored_rows = []
    for _, row in df.iterrows():
        sym = row["symbol"]
        scores = {}
        reasons = {}

        # --- Basic scores (from data already in df) ---
        rsi4 = row.get("rsi_4h", 50)
        rsi1d = row.get("rsi_1D", 50)

        # RSI 4h
        if rsi4 <= 25: scores["rsi_4h"] = 30; reasons["rsi_4h"] = f"RSI 4h stark überverkauft ({rsi4:.0f})"
        elif rsi4 <= 35: scores["rsi_4h"] = 20; reasons["rsi_4h"] = f"RSI 4h überverkauft ({rsi4:.0f})"
        elif rsi4 <= 45: scores["rsi_4h"] = 10; reasons["rsi_4h"] = f"RSI 4h leicht bullish ({rsi4:.0f})"
        elif rsi4 >= 75: scores["rsi_4h"] = -30; reasons["rsi_4h"] = f"RSI 4h stark überkauft ({rsi4:.0f})"
        elif rsi4 >= 65: scores["rsi_4h"] = -20; reasons["rsi_4h"] = f"RSI 4h überkauft ({rsi4:.0f})"
        elif rsi4 >= 55: scores["rsi_4h"] = -10; reasons["rsi_4h"] = f"RSI 4h leicht bearish ({rsi4:.0f})"
        else: scores["rsi_4h"] = 0

        # RSI 1D
        if rsi1d <= 30: scores["rsi_1d"] = 20; reasons["rsi_1d"] = f"RSI 1D überverkauft ({rsi1d:.0f})"
        elif rsi1d <= 42: scores["rsi_1d"] = 10; reasons["rsi_1d"] = f"RSI 1D bullish ({rsi1d:.0f})"
        elif rsi1d >= 70: scores["rsi_1d"] = -20; reasons["rsi_1d"] = f"RSI 1D überkauft ({rsi1d:.0f})"
        elif rsi1d >= 58: scores["rsi_1d"] = -10; reasons["rsi_1d"] = f"RSI 1D bearish ({rsi1d:.0f})"
        else: scores["rsi_1d"] = 0

        # MACD (from scan_all)
        mt = row.get("macd_trend", "NEUTRAL")
        mh = row.get("macd_histogram", 0)
        if mt == "BULLISH": scores["macd"] = 15 + min(5, abs(mh) * 500); reasons["macd"] = f"MACD bullish"
        elif mt == "BEARISH": scores["macd"] = -(15 + min(5, abs(mh) * 500)); reasons["macd"] = f"MACD bearish"
        else: scores["macd"] = 0

        # Volume & OBV
        vr = row.get("vol_ratio", 1.0); ot = row.get("obv_trend", "NEUTRAL")
        vol_sc = 0
        if vr > 1.5 and ot == "BULLISH": vol_sc = 15; reasons["volume_obv"] = f"Vol hoch + OBV bullish"
        elif vr > 1.2 and ot == "BULLISH": vol_sc = 10; reasons["volume_obv"] = f"Vol ok + OBV bullish"
        elif ot == "BULLISH": vol_sc = 5; reasons["volume_obv"] = f"OBV bullish"
        elif vr > 1.5 and ot == "BEARISH": vol_sc = -15; reasons["volume_obv"] = f"Vol hoch + OBV bearish"
        elif vr > 1.2 and ot == "BEARISH": vol_sc = -10; reasons["volume_obv"] = f"Vol ok + OBV bearish"
        elif ot == "BEARISH": vol_sc = -5; reasons["volume_obv"] = f"OBV bearish"
        scores["volume_obv"] = vol_sc

        # Stoch RSI
        sk = row.get("stoch_rsi_k", 50); sd = row.get("stoch_rsi_d", 50)
        if sk < 20: scores["stoch_rsi"] = 12; reasons["stoch_rsi"] = f"StochRSI überverkauft ({sk:.0f})"
        elif sk < 30 and sk > sd: scores["stoch_rsi"] = 8; reasons["stoch_rsi"] = f"StochRSI bullish crossover"
        elif sk > 80: scores["stoch_rsi"] = -12; reasons["stoch_rsi"] = f"StochRSI überkauft ({sk:.0f})"
        elif sk > 70 and sk < sd: scores["stoch_rsi"] = -8; reasons["stoch_rsi"] = f"StochRSI bearish crossover"
        else: scores["stoch_rsi"] = 0

        # --- Extra indicators (computed on-demand, only if filter active) ---
        if needs_extra:
            kl_4h = fetch_klines_smart(sym, "4h")
            if not kl_4h.empty and len(kl_4h) >= 15:
                # Smart Money
                if active_filters.get("smart_money", False):
                    ob = detect_order_blocks(kl_4h); ms = detect_market_structure(kl_4h)
                    ob_sig = ob.get("ob_signal", "NONE"); struct = ms.get("structure", "UNKNOWN")
                    smc_sc = 0
                    if ob_sig == "BULLISH": smc_sc += 8
                    elif ob_sig == "BEARISH": smc_sc -= 8
                    if struct == "BULLISH": smc_sc += 7
                    elif struct == "BEARISH": smc_sc -= 7
                    scores["smart_money"] = max(-15, min(15, smc_sc))
                    if smc_sc != 0: reasons["smart_money"] = f"SMC: OB={ob_sig} Struct={struct}"

                # EMA Alignment
                if active_filters.get("ema_alignment", False):
                    ema_d = calculate_ema_alignment_fast(kl_4h)
                    et = ema_d.get("ema_trend", "NEUTRAL")
                    if et == "BULLISH": scores["ema_alignment"] = 12; reasons["ema_alignment"] = "EMA bullish aligned"
                    elif et == "BEARISH": scores["ema_alignment"] = -12; reasons["ema_alignment"] = "EMA bearish aligned"
                    else: scores["ema_alignment"] = 0

                # RSI Divergence
                if active_filters.get("rsi_divergence", False):
                    div_d = detect_rsi_divergence(kl_4h)
                    div = div_d.get("divergence", "NONE"); dt = div_d.get("div_type", "NONE")
                    if div == "BULLISH": scores["rsi_divergence"] = 15 if dt == "REGULAR" else 10; reasons["rsi_divergence"] = f"RSI Divergenz bullish ({dt})"
                    elif div == "BEARISH": scores["rsi_divergence"] = -(15 if dt == "REGULAR" else 10); reasons["rsi_divergence"] = f"RSI Divergenz bearish ({dt})"
                    else: scores["rsi_divergence"] = 0

                # Bollinger Squeeze
                if active_filters.get("bollinger", False):
                    bb_d = calculate_bb_squeeze(kl_4h)
                    sq = bb_d.get("bb_squeeze", False); bp = bb_d.get("bb_pct", 50)
                    if sq and bp < 30: scores["bollinger"] = 10; reasons["bollinger"] = f"BB Squeeze bullish ({bp:.0f}%)"
                    elif sq and bp > 70: scores["bollinger"] = -10; reasons["bollinger"] = f"BB Squeeze bearish ({bp:.0f}%)"
                    elif bp < 20: scores["bollinger"] = 7; reasons["bollinger"] = f"BB unteres Band ({bp:.0f}%)"
                    elif bp > 80: scores["bollinger"] = -7; reasons["bollinger"] = f"BB oberes Band ({bp:.0f}%)"
                    else: scores["bollinger"] = 0

        # Funding Rate (from lazy-loaded batch)
        if active_filters.get("funding_rate", False):
            fr_val = funding_rates_map.get(sym, None)
            if fr_val is not None:
                if fr_val > 0.05: scores["funding_rate"] = -10; reasons["funding_rate"] = f"Funding hoch ({fr_val:.4f})"
                elif fr_val > 0.02: scores["funding_rate"] = -5; reasons["funding_rate"] = f"Funding leicht hoch ({fr_val:.4f})"
                elif fr_val < -0.05: scores["funding_rate"] = 10; reasons["funding_rate"] = f"Funding negativ ({fr_val:.4f})"
                elif fr_val < -0.02: scores["funding_rate"] = 5; reasons["funding_rate"] = f"Funding leicht negativ ({fr_val:.4f})"
                else: scores["funding_rate"] = 0

        # Fear & Greed
        if active_filters.get("fear_greed", False) and fear_greed_val is not None:
            if fear_greed_val <= 15: scores["fear_greed"] = 8; reasons["fear_greed"] = f"Extreme Fear ({fear_greed_val})"
            elif fear_greed_val <= 30: scores["fear_greed"] = 4; reasons["fear_greed"] = f"Fear ({fear_greed_val})"
            elif fear_greed_val >= 85: scores["fear_greed"] = -8; reasons["fear_greed"] = f"Extreme Greed ({fear_greed_val})"
            elif fear_greed_val >= 70: scores["fear_greed"] = -4; reasons["fear_greed"] = f"Greed ({fear_greed_val})"

        # --- Compute total ---
        raw_score = sum(scores.get(k, 0) for k, v in active_filters.items() if v)
        max_possible = sum(FILTER_WEIGHTS.get(k, 0) for k, v in active_filters.items() if v)
        normalized = int(raw_score / max_possible * 100) if max_possible > 0 else 0
        normalized = max(-100, min(100, normalized))

        if normalized >= 60: rec = "STRONG BUY"
        elif normalized >= 30: rec = "BUY"
        elif normalized >= 10: rec = "LEAN BUY"
        elif normalized <= -60: rec = "STRONG SELL"
        elif normalized <= -30: rec = "SELL"
        elif normalized <= -10: rec = "LEAN SELL"
        else: rec = "WAIT"

        active_reasons = [reasons[k] for k in active_filters if active_filters.get(k) and k in reasons]
        scored_rows.append({**row.to_dict(), "dyn_score": normalized, "dyn_rec": rec,
                           "dyn_reasons": " | ".join(active_reasons[:4])})
    sdf = pd.DataFrame(scored_rows)

    # --- TOP BUY ---
    st.markdown("#### 🟢 Top Buy")
    buy_df = sdf[sdf["dyn_score"] > 0].sort_values("dyn_score", ascending=False).head(15)
    if buy_df.empty:
        st.info("Keine Buy-Signale mit den aktuellen Filtern.")
    else:
        for _, r in buy_df.iterrows():
            s = r["dyn_score"]; rec = r["dyn_rec"]
            glow = "text-shadow:0 0 6px #00FF7F;" if s >= 60 else ""
            st.markdown(f'''<div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;margin:4px 0;">
<div style="display:flex;justify-content:space-between;align-items:center;">
<b style="color:white;">{r["symbol"]}</b>
<span style="color:#00FF7F;font-weight:bold;{glow}">{rec} ({s})</span>
</div>
<div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:6px;">
<div style="background:linear-gradient(90deg,#00FF7F,#32CD32);height:6px;width:{min(abs(s),100)}%;border-radius:4px;"></div>
</div>
<div style="font-size:11px;color:#888;margin-top:4px;">{r.get("dyn_reasons","")}</div>
</div>''', unsafe_allow_html=True)

    st.markdown("---")

    # --- TOP SELL ---
    st.markdown("#### 🔴 Top Sell")
    sell_df = sdf[sdf["dyn_score"] < 0].sort_values("dyn_score").head(15)
    if sell_df.empty:
        st.info("Keine Sell-Signale mit den aktuellen Filtern.")
    else:
        for _, r in sell_df.iterrows():
            s = r["dyn_score"]; rec = r["dyn_rec"]
            glow = "text-shadow:0 0 6px #FF6347;" if s <= -60 else ""
            st.markdown(f'''<div style="background:#1a1a2e;border-radius:8px;padding:10px 14px;margin:4px 0;">
<div style="display:flex;justify-content:space-between;align-items:center;">
<b style="color:white;">{r["symbol"]}</b>
<span style="color:#FF6347;font-weight:bold;{glow}">{rec} ({s})</span>
</div>
<div style="background:#2a2a4a;border-radius:4px;height:6px;margin-top:6px;">
<div style="background:linear-gradient(90deg,#FF6347,#FF0000);height:6px;width:{min(abs(s),100)}%;border-radius:4px;"></div>
</div>
<div style="font-size:11px;color:#888;margin-top:4px;">{r.get("dyn_reasons","")}</div>
</div>''', unsafe_allow_html=True)

# ============================================================
# TAB 5: DETAIL — Complete Analysis Dashboard
# ============================================================
with tab_det:
    sel=st.selectbox("Select Coin",df["symbol"].tolist(),key="dc")
    if sel:
        c=df[df["symbol"]==sel].iloc[0]; sig=c.get("signal","WAIT"); sc=sig_color(sig); im=c.get("coin_image","")
        price=c["price"]; r4=c.get("rsi_4h",50); r1d=c.get("rsi_1D",50)

        # Header
        st.markdown(f'<div style="background:#1a1a2e;border-radius:10px;padding:16px;margin-bottom:12px;"><div style="display:flex;flex-wrap:wrap;align-items:center;gap:14px;">{icon(im, sel)}<h2 style="margin:0;color:white;">{sel}</h2><span style="color:#888;">{c.get("coin_name",sel)}</span><span class="cr">#{int(c.get("rank",999))}</span><span style="font-size:20px;color:white;font-weight:bold;">{fp(price)}</span><span class="{cc(c.get("change_24h",0))}" style="font-size:16px;">{c.get("change_24h",0):+.2f}%</span><span style="color:{sc};font-weight:bold;font-size:16px;">Now: {sig}</span></div></div>', unsafe_allow_html=True)

        # ---- COMPUTE ON-DEMAND INDICATORS ----
        from indicators import (calculate_ema_crosses, calculate_bollinger, calculate_atr,
            calculate_support_resistance, calculate_fibonacci, calculate_btc_correlation,
            calculate_sl_tp, calculate_price_range, multi_tf_rsi_summary)

        detail_df = fetch_klines_smart(sel, "4h")
        detail_1d = fetch_klines_smart(sel, "1d")
        btc_df = fetch_klines_smart("BTC", "4h")

        ema_data = calculate_ema_crosses(detail_df) if not detail_df.empty else {}
        bb_data = calculate_bollinger(detail_df) if not detail_df.empty else {}
        atr_data = calculate_atr(detail_df) if not detail_df.empty else {}
        sr_data = calculate_support_resistance(detail_df) if not detail_df.empty else {"supports":[],"resistances":[],"nearest_support":0,"nearest_resistance":0}
        fib_data = calculate_fibonacci(detail_df) if not detail_df.empty else {"fib_levels":{},"fib_zone":"N/A"}
        btc_corr = calculate_btc_correlation(detail_df, btc_df) if not detail_df.empty else {"correlation":0,"corr_label":"N/A"}
        pr_data = calculate_price_range(detail_df) if not detail_df.empty else {}
        sltp = calculate_sl_tp(price, atr_data.get("atr",0), sig, sr_data)

        # Multi-TF RSI — always in ascending order: 1h, 4h, 1D, 1W
        rsi_vals_raw = {}
        for tf in tf_to_scan:
            rsi_vals_raw[tf] = c.get(f"rsi_{tf}", None)
        for extra_tf, extra_int in [("1h","1h"),("1W","1w")]:
            if extra_tf not in rsi_vals_raw:
                edf = fetch_klines_smart(sel, extra_int)
                if not edf.empty and len(edf) >= 15:
                    from ta.momentum import RSIIndicator as _RSI
                    rs = _RSI(close=edf["close"],window=14).rsi().dropna()
                    rsi_vals_raw[extra_tf] = round(float(rs.iloc[-1]),2) if len(rs)>=1 else None
                else: rsi_vals_raw[extra_tf] = None
        # Force correct order
        tf_order = ["1h", "4h", "1D", "1W"]
        rsi_vals = {tf: rsi_vals_raw.get(tf) for tf in tf_order if tf in rsi_vals_raw}
        mtf = multi_tf_rsi_summary(rsi_vals)

        # =============================================
        # AI SUMMARY — Trading Recommendation
        # =============================================
        # Count bullish/bearish signals across all indicators
        bull_pts, bear_pts, reasons_bull, reasons_bear = 0, 0, [], []

        # RSI Multi-TF
        if mtf["confluence"] in ("STRONG_BUY",): bull_pts += 3; reasons_bull.append("Alle Timeframes im BUY-Bereich")
        elif mtf["confluence"] == "LEAN_BUY": bull_pts += 1; reasons_bull.append("Mehrheit der TFs bullish")
        elif mtf["confluence"] in ("STRONG_SELL",): bear_pts += 3; reasons_bear.append("Alle Timeframes im SELL-Bereich")
        elif mtf["confluence"] == "LEAN_SELL": bear_pts += 1; reasons_bear.append("Mehrheit der TFs bearish")
        # EMA
        for ek in ["cross_9_21","cross_50_200"]:
            ev = ema_data.get(ek,"N/A")
            if ev == "GOLDEN": bull_pts += 2; reasons_bull.append(f"{ek.replace('cross_','EMA ')}: Golden Cross")
            elif ev == "BULLISH": bull_pts += 1; reasons_bull.append(f"{ek.replace('cross_','EMA ')}: Bullish")
            elif ev == "DEATH": bear_pts += 2; reasons_bear.append(f"{ek.replace('cross_','EMA ')}: Death Cross")
            elif ev == "BEARISH": bear_pts += 1; reasons_bear.append(f"{ek.replace('cross_','EMA ')}: Bearish")
        # MACD
        mt = c.get("macd_trend","NEUTRAL")
        if mt == "BULLISH": bull_pts += 1; reasons_bull.append("MACD bullish")
        elif mt == "BEARISH": bear_pts += 1; reasons_bear.append("MACD bearish")
        # Bollinger
        bbp = bb_data.get("bb_pct", 50)
        if bbp < 15: bull_pts += 2; reasons_bull.append(f"Bollinger: Preis am unteren Band ({bbp:.0f}%)")
        elif bbp < 30: bull_pts += 1; reasons_bull.append(f"Bollinger: untere Zone ({bbp:.0f}%)")
        elif bbp > 85: bear_pts += 2; reasons_bear.append(f"Bollinger: Preis am oberen Band ({bbp:.0f}%)")
        elif bbp > 70: bear_pts += 1; reasons_bear.append(f"Bollinger: obere Zone ({bbp:.0f}%)")
        # OBV
        obv = c.get("obv_trend","NEUTRAL")
        if obv == "BULLISH": bull_pts += 1; reasons_bull.append("OBV-Trend steigend")
        elif obv == "BEARISH": bear_pts += 1; reasons_bear.append("OBV-Trend fallend")
        # Price range
        pos7 = pr_data.get("7d_position", 50)
        if pos7 < 20: bull_pts += 1; reasons_bull.append(f"Preis nahe 7d-Tief ({pos7:.0f}%)")
        elif pos7 > 80: bear_pts += 1; reasons_bear.append(f"Preis nahe 7d-Hoch ({pos7:.0f}%)")
        # Fib zone
        fz = fib_data.get("fib_zone","N/A")
        if "61.8" in fz or "near low" in fz: bull_pts += 1; reasons_bull.append(f"Fibonacci: Golden Zone ({fz})")
        elif "near high" in fz: bear_pts += 1; reasons_bear.append(f"Fibonacci: nahe Swing High ({fz})")

        total_pts = bull_pts + bear_pts
        if total_pts == 0: total_pts = 1
        bull_pct = bull_pts / total_pts * 100; bear_pct = bear_pts / total_pts * 100

        # Generate recommendation text
        if bull_pts >= bear_pts + 4:
            verdict = "STRONG BUY"; vcolor = "#00FF00"; vicon = "🟢"
            advice = "Starkes Kaufsignal: Die Mehrheit der Indikatoren ist bullish. Ein Einstieg kann in Betracht gezogen werden, idealerweise mit einem Stop-Loss unterhalb des nächsten Support-Levels."
        elif bull_pts >= bear_pts + 2:
            verdict = "BUY"; vcolor = "#00FF7F"; vicon = "🟢"
            advice = "Moderates Kaufsignal: Mehrere Indikatoren deuten auf steigende Kurse hin. Ein Einstieg ist möglich, aber auf Bestätigung durch Volumen achten."
        elif bear_pts >= bull_pts + 4:
            verdict = "STRONG SELL"; vcolor = "#FF0040"; vicon = "🔴"
            advice = "Starkes Verkaufsignal: Die Mehrheit der Indikatoren ist bearish. Bestehende Positionen absichern oder schließen. Neue Short-Positionen mit SL über der nächsten Resistance."
        elif bear_pts >= bull_pts + 2:
            verdict = "SELL"; vcolor = "#FF6347"; vicon = "🔴"
            advice = "Moderates Verkaufsignal: Mehrere Indikatoren deuten auf fallende Kurse hin. Vorsicht bei Long-Positionen, Stop-Loss eng setzen."
        else:
            verdict = "ABWARTEN"; vcolor = "#FFD700"; vicon = "🟡"
            advice = "Gemischte Signale: Bullische und bearische Indikatoren halten sich die Waage. Am besten auf eine klare Richtung warten, bevor ein Trade eingegangen wird."

        # Timing advice
        vol_note = ""
        atrvol = atr_data.get("volatility","LOW")
        if atrvol in ("HIGH","VERY_HIGH"):
            vol_note = " ⚠️ Hohe Volatilität — größere Stop-Losses nötig, kleinere Positionsgrößen empfohlen."
        elif atrvol == "LOW":
            vol_note = " 💤 Niedrige Volatilität — ein Ausbruch könnte bevorstehen. Auf Bollinger Squeeze achten."

        corr_note = ""
        btc_c = btc_corr.get("correlation",0)
        if abs(btc_c) > 0.7:
            corr_note = f" 🔗 Starke BTC-Korrelation ({btc_c:.2f}) — BTC-Bewegungen werden diesen Coin stark beeinflussen."

        st.markdown(f'''<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border:2px solid {vcolor}44;border-radius:12px;padding:18px;margin-bottom:16px;">
<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
<span style="font-size:28px;">{vicon}</span>
<div><span style="color:{vcolor};font-size:22px;font-weight:bold;">{verdict}</span>
<span style="color:#888;font-size:13px;margin-left:10px;">({bull_pts} bullish / {bear_pts} bearish Punkte)</span></div>
<div style="flex:1;"></div>
<div style="background:#2a2a4a;border-radius:20px;padding:2px;width:120px;height:14px;overflow:hidden;">
<div style="display:flex;height:100%;"><div style="width:{bull_pct}%;background:#00FF7F;"></div><div style="width:{bear_pct}%;background:#FF6347;"></div></div>
</div></div>
<div style="color:#ccc;font-size:13px;line-height:1.5;">{advice}{vol_note}{corr_note}</div>
<div style="margin-top:10px;font-size:11px;color:#666;">
{"✅ " + " · ".join(reasons_bull[:4]) if reasons_bull else ""}<br>
{"❌ " + " · ".join(reasons_bear[:4]) if reasons_bear else ""}
</div></div>''', unsafe_allow_html=True)

        # =============================================
        # TradingView Chart — nur bei Klick laden (verhindert Firewall-Blockade)
        # =============================================
        det_chart_key = "det_chart_open"
        dc1, dc2, _ = st.columns([1,1,4])
        with dc1:
            if st.button("📈 Chart öffnen (RSI)", key="dc_open", use_container_width=True,
                type="primary" if st.session_state.get(det_chart_key) else "secondary"):
                st.session_state[det_chart_key] = not st.session_state.get(det_chart_key, False)
                st.rerun()
        with dc2:
            if st.session_state.get(det_chart_key):
                if st.button("❌ Chart schließen", key="dc_close", use_container_width=True):
                    st.session_state[det_chart_key]=False; st.rerun()

        if st.session_state.get(det_chart_key):
            st.components.v1.html(tv_chart_simple(sel, 670), height=690)
        else:
            st.info("💡 Klicke **Chart öffnen** um den TradingView-Chart mit RSI zu laden.")

        # =============================================
        # Plotly Key Levels Chart (S/R + Fibonacci on candlesticks)
        # =============================================
        if not detail_df.empty and len(detail_df) >= 20:
            st.markdown("### 📊 Key Levels Chart")
            # Layer toggles
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1: show_sr = st.checkbox("Support/Resistance", value=True, key="lv_sr")
            with lc2: show_fib = st.checkbox("Fibonacci", value=True, key="lv_fib")
            with lc3: show_sltp = st.checkbox("SL / TP", value=True, key="lv_sltp")
            with lc4: show_bb = st.checkbox("Bollinger Bands", value=False, key="lv_bb")

            chart_df = detail_df.tail(60).copy()
            fig_levels = go.Figure()

            # --- Bollinger Bands (background layer, subtle) ---
            if show_bb and bb_data.get("bb_upper",0) > 0 and len(detail_df) >= 20:
                from ta.volatility import BollingerBands as _BB
                _bbc = _BB(close=detail_df["close"], window=20, window_dev=2)
                bb_u = _bbc.bollinger_hband().tail(60).values
                bb_m = _bbc.bollinger_mavg().tail(60).values
                bb_l = _bbc.bollinger_lband().tail(60).values
                x_range = list(range(len(chart_df)))
                fig_levels.add_trace(go.Scatter(x=x_range, y=bb_u, mode="lines",
                    line=dict(color="rgba(255,165,0,0.25)", width=1), name="BB Upper", showlegend=True,
                    legendgroup="bb", hoverinfo="skip"))
                fig_levels.add_trace(go.Scatter(x=x_range, y=bb_l, mode="lines",
                    line=dict(color="rgba(255,165,0,0.25)", width=1), name="BB Lower",
                    fill="tonexty", fillcolor="rgba(255,165,0,0.04)",
                    showlegend=False, legendgroup="bb", hoverinfo="skip"))
                fig_levels.add_trace(go.Scatter(x=x_range, y=bb_m, mode="lines",
                    line=dict(color="rgba(255,165,0,0.4)", width=1, dash="dot"),
                    name="BB Mid", showlegend=False, legendgroup="bb", hoverinfo="skip"))

            # --- Candlesticks ---
            fig_levels.add_trace(go.Candlestick(
                x=list(range(len(chart_df))), open=chart_df["open"], high=chart_df["high"],
                low=chart_df["low"], close=chart_df["close"], name="Price",
                increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
                increasing_fillcolor="#26A69A", decreasing_fillcolor="#EF5350"))

            # Price range for y-axis
            y_min = chart_df["low"].min()
            y_max = chart_df["high"].max()
            y_pad = (y_max - y_min) * 0.15
            x_max = len(chart_df) - 1

            # --- Support/Resistance zones ---
            if show_sr:
                supports = sr_data.get("supports",[])[:3]
                resistances = sr_data.get("resistances",[])[:3]
                zone_h = (y_max - y_min) * 0.008  # thin zone band

                for i, s_val in enumerate(supports):
                    opacity = 0.35 - i * 0.1
                    # Zone band
                    fig_levels.add_shape(type="rect", x0=-2, x1=x_max+2,
                        y0=s_val - zone_h, y1=s_val + zone_h,
                        fillcolor=f"rgba(0,221,102,{opacity})", line_width=0)
                    # Line
                    fig_levels.add_shape(type="line", x0=-2, x1=x_max+2,
                        y0=s_val, y1=s_val,
                        line=dict(color="#00DD66", width=1.5, dash="dash"))
                    # Right-side label
                    fig_levels.add_annotation(x=x_max+1, y=s_val, text=f"S{i+1} {fp(s_val)}",
                        showarrow=False, xanchor="left", font=dict(color="#00DD66", size=10),
                        bgcolor="rgba(0,221,102,0.12)", borderpad=3)

                for i, r_val in enumerate(resistances):
                    opacity = 0.35 - i * 0.1
                    fig_levels.add_shape(type="rect", x0=-2, x1=x_max+2,
                        y0=r_val - zone_h, y1=r_val + zone_h,
                        fillcolor=f"rgba(239,83,80,{opacity})", line_width=0)
                    fig_levels.add_shape(type="line", x0=-2, x1=x_max+2,
                        y0=r_val, y1=r_val,
                        line=dict(color="#EF5350", width=1.5, dash="dash"))
                    fig_levels.add_annotation(x=x_max+1, y=r_val, text=f"R{i+1} {fp(r_val)}",
                        showarrow=False, xanchor="left", font=dict(color="#EF5350", size=10),
                        bgcolor="rgba(239,83,80,0.12)", borderpad=3)

                # Current price marker
                fig_levels.add_annotation(x=x_max+1, y=price,
                    text=f"▸ {fp(price)}", showarrow=False, xanchor="left",
                    font=dict(color="white", size=11, family="monospace"),
                    bgcolor="rgba(255,255,255,0.1)", borderpad=3)

            # --- Fibonacci levels (left side labels, subtle lines) ---
            if show_fib:
                fib_styles = {
                    "0.236": ("#FFD700", "23.6%"),
                    "0.382": ("#FFA500", "38.2%"),
                    "0.5":   ("#FF69B4", "50%"),
                    "0.618": ("#9370DB", "61.8% ★"),
                    "0.786": ("#6495ED", "78.6%"),
                }
                for label, val in fib_data.get("fib_levels",{}).items():
                    for fk, (fc, short_label) in fib_styles.items():
                        if fk in label:
                            fig_levels.add_shape(type="line", x0=-2, x1=x_max+2,
                                y0=val, y1=val,
                                line=dict(color=fc, width=1, dash="dot"), opacity=0.4)
                            fig_levels.add_annotation(x=-1, y=val,
                                text=f"Fib {short_label}", showarrow=False, xanchor="right",
                                font=dict(color=fc, size=9),
                                bgcolor=f"rgba(0,0,0,0.5)", borderpad=2)

            # --- SL/TP levels (prominent, with zones) ---
            if show_sltp and sig in ("BUY","CTB","SELL","CTS") and sltp.get("sl",0) > 0:
                is_buy = sig in ("BUY","CTB")
                sl_val = sltp["sl"]; tp1_val = sltp["tp1"]; tp2_val = sltp.get("tp2",0)

                # SL zone
                sl_zone = (y_max - y_min) * 0.012
                fig_levels.add_shape(type="rect", x0=-2, x1=x_max+2,
                    y0=sl_val - sl_zone, y1=sl_val + sl_zone,
                    fillcolor="rgba(255,0,64,0.15)", line_width=0)
                fig_levels.add_shape(type="line", x0=-2, x1=x_max+2,
                    y0=sl_val, y1=sl_val,
                    line=dict(color="#FF0040", width=2, dash="dashdot"))
                fig_levels.add_annotation(x=x_max+1, y=sl_val,
                    text=f"🛑 SL {fp(sl_val)}", showarrow=False, xanchor="left",
                    font=dict(color="#FF0040", size=11, family="monospace"),
                    bgcolor="rgba(255,0,64,0.2)", borderpad=4)

                # TP1
                fig_levels.add_shape(type="line", x0=-2, x1=x_max+2,
                    y0=tp1_val, y1=tp1_val,
                    line=dict(color="#00DD66", width=1.5, dash="dashdot"))
                fig_levels.add_annotation(x=x_max+1, y=tp1_val,
                    text=f"🎯 TP1 {fp(tp1_val)}", showarrow=False, xanchor="left",
                    font=dict(color="#00DD66", size=10),
                    bgcolor="rgba(0,221,102,0.15)", borderpad=3)

                # TP2
                if tp2_val > 0:
                    fig_levels.add_shape(type="line", x0=-2, x1=x_max+2,
                        y0=tp2_val, y1=tp2_val,
                        line=dict(color="#00FF7F", width=1.5, dash="dashdot"))
                    fig_levels.add_annotation(x=x_max+1, y=tp2_val,
                        text=f"🎯 TP2 {fp(tp2_val)}", showarrow=False, xanchor="left",
                        font=dict(color="#00FF7F", size=10),
                        bgcolor="rgba(0,255,127,0.15)", borderpad=3)

                # Entry line
                fig_levels.add_shape(type="line", x0=-2, x1=x_max+2,
                    y0=price, y1=price,
                    line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dot"))

            # --- Layout ---
            fig_levels.update_layout(
                title=dict(text=f"{sel} — Key Levels (4h, letzte 60 Kerzen)",
                    font=dict(size=13, color="#aaa"), x=0.5),
                template="plotly_dark",
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                height=520,
                xaxis=dict(
                    showticklabels=False, showgrid=False,
                    rangeslider=dict(visible=False),
                    range=[-3, x_max + 8],  # extra space for right labels
                ),
                yaxis=dict(
                    title=None,
                    gridcolor="rgba(255,255,255,0.03)",
                    side="right",
                    tickfont=dict(size=10, color="#666"),
                    range=[y_min - y_pad, y_max + y_pad],
                ),
                showlegend=show_bb,  # only show legend when BB is on
                legend=dict(
                    orientation="h", yanchor="top", y=1.02, x=0.5, xanchor="center",
                    font=dict(size=10, color="#888"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=60, r=130, t=45, b=15),
            )
            st.plotly_chart(fig_levels, use_container_width=True)

        # =============================================
        # SECTION 1: Multi-TF RSI Traffic Light
        # =============================================
        st.markdown("### 🚦 Multi-Timeframe RSI")
        tf_cols = st.columns(len(rsi_vals))
        for i, (tf, val) in enumerate(rsi_vals.items()):
            with tf_cols[i]:
                if val is None:
                    st.markdown(f"<div style='text-align:center;background:#1a1a2e;border-radius:8px;padding:10px;'><div style='color:#555;font-size:12px;'>{tf}</div><div style='color:#555;font-size:18px;'>—</div></div>", unsafe_allow_html=True)
                else:
                    if val <= 30: bg, tc = "rgba(0,255,0,0.15)", "#00FF00"
                    elif val <= 42: bg, tc = "rgba(0,255,127,0.1)", "#00FF7F"
                    elif val >= 70: bg, tc = "rgba(255,0,64,0.15)", "#FF0040"
                    elif val >= 58: bg, tc = "rgba(255,99,71,0.1)", "#FF6347"
                    else: bg, tc = "rgba(255,215,0,0.08)", "#FFD700"
                    prev = c.get(f"rsi_prev_{tf}", val)
                    arrow = "↑" if val > prev else ("↓" if val < prev else "→")
                    st.markdown(f"<div style='text-align:center;background:{bg};border:1px solid {tc}33;border-radius:8px;padding:10px;'><div style='color:#888;font-size:12px;'>{tf}</div><div style='color:{tc};font-size:22px;font-weight:bold;'>{val:.1f} {arrow}</div></div>", unsafe_allow_html=True)
        conf_color = {"STRONG_BUY":"#00FF00","LEAN_BUY":"#00FF7F","STRONG_SELL":"#FF0040","LEAN_SELL":"#FF6347"}.get(mtf["confluence"],"#FFD700")
        st.markdown(f"<div style='text-align:center;padding:6px;'><span style='color:{conf_color};font-weight:bold;font-size:14px;'>TF Confluence: {mtf['confluence']}</span> <span style='color:#888;'>({mtf['bullish_count']} bullish / {mtf['bearish_count']} bearish von {mtf['total']} TFs)</span></div>", unsafe_allow_html=True)
        with st.expander("ℹ️ Was bedeutet Multi-TF RSI?"):
            st.markdown("""**RSI (Relative Strength Index)** misst die Stärke von Kursbewegungen auf einer Skala von 0-100.

**Zonen:** ≤30 = Überverkauft (potentieller Kauf) · 30-42 = Schwach · 42-58 = Neutral · 58-70 = Stark · ≥70 = Überkauft (potentieller Verkauf)

**Multi-TF Confluence:** Wenn mehrere Timeframes gleichzeitig im gleichen Bereich sind, ist das Signal stärker.
- **Alle TFs grün (≤42):** Starkes Kaufsignal — der Coin ist auf allen Ebenen überverkauft
- **Alle TFs rot (≥58):** Starkes Verkaufsignal — der Coin ist auf allen Ebenen überkauft
- **Gemischt:** Abwarten — kurzfristige und langfristige Trends widersprechen sich

**Pfeile ↑↓→** zeigen ob der RSI gerade steigt, fällt oder seitwärts läuft. Ein steigender RSI im überverkauften Bereich deutet auf eine Erholung hin.

**Empfehlung:** Idealerweise auf Übereinstimmung von mindestens 3 TFs warten, bevor ein Trade eingegangen wird.""")

        # =============================================
        # SECTION 2: Risk Management (moved up)
        # =============================================
        st.markdown("### 🛡️ Risk Management (ATR-based)")
        if sig in ("BUY","CTB","SELL","CTS"):
            is_buy = sig in ("BUY","CTB")
            sl_c = "#FF6347" if is_buy else "#00FF7F"
            tp_c = "#00FF7F" if is_buy else "#FF6347"
            sl_dist = (sltp["sl"]/price-1)*100
            tp1_dist = (sltp["tp1"]/price-1)*100
            tp2_dist = (sltp["tp2"]/price-1)*100 if sltp["tp2"] else 0
            st.markdown(f'<div style="background:#1a1a2e;border-radius:10px;padding:14px;display:flex;flex-wrap:wrap;gap:20px;justify-content:space-around;">'
                f'<div style="text-align:center;"><div style="color:#888;font-size:11px;">Stop-Loss</div><div style="color:{sl_c};font-size:18px;font-weight:bold;">{fp(sltp["sl"])}</div><div style="color:{sl_c};font-size:12px;">{sl_dist:+.2f}%</div></div>'
                f'<div style="text-align:center;"><div style="color:#888;font-size:11px;">Entry</div><div style="color:white;font-size:18px;font-weight:bold;">{fp(price)}</div><div style="color:#888;font-size:12px;">current</div></div>'
                f'<div style="text-align:center;"><div style="color:#888;font-size:11px;">TP 1 (1.5x ATR)</div><div style="color:{tp_c};font-size:18px;font-weight:bold;">{fp(sltp["tp1"])}</div><div style="color:{tp_c};font-size:12px;">{tp1_dist:+.2f}%</div></div>'
                f'<div style="text-align:center;"><div style="color:#888;font-size:11px;">TP 2 (S/R)</div><div style="color:{tp_c};font-size:18px;font-weight:bold;">{fp(sltp["tp2"])}</div><div style="color:{tp_c};font-size:12px;">{tp2_dist:+.2f}%</div></div>'
                f'<div style="text-align:center;"><div style="color:#888;font-size:11px;">Risk/Reward</div><div style="color:#FFD700;font-size:18px;font-weight:bold;">{sltp["risk_reward"]:.2f}</div></div>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.info("SL/TP nur verfügbar wenn Signal BUY, SELL, CTB oder CTS ist.")
        with st.expander("ℹ️ Was bedeutet Risk Management?"):
            st.markdown(f"""**ATR (Average True Range)** misst die durchschnittliche Kursbewegung pro Kerze. Aktuell: **{fp(atr_data.get('atr',0))}** ({atr_data.get('atr_pct',0):.2f}% vom Preis).

**Stop-Loss (SL):** Wird auf dem nächsten Support/Resistance-Level gesetzt, maximal 3x ATR vom Entry entfernt. Schützt vor großen Verlusten.

**Take-Profit Levels:**
- **TP1 (1.5x ATR):** Konservatives Ziel — hier kann ein Teil der Position geschlossen werden
- **TP2 (S/R Level):** Basiert auf dem nächsten Widerstands-/Unterstützungsniveau
- **TP3 (4x ATR):** Aggressives Ziel für Teilpositionen

**Risk/Reward Ratio:** Verhältnis von potentiellem Gewinn zu potentiellem Verlust.
- **≥ 2.0:** Gutes Setup — der potentielle Gewinn ist mindestens doppelt so hoch wie das Risiko
- **1.0-2.0:** Akzeptabel — nur mit starker Signalbestätigung traden
- **< 1.0:** Schlechtes Setup — besser auf eine bessere Gelegenheit warten

**Positionsgröße:** Bei {atr_data.get('volatility','LOW')} Volatilität empfohlen:
- {'⚠️ Kleine Position (0.5-1% des Kapitals) wegen hoher Volatilität' if atr_data.get('volatility') in ('HIGH','VERY_HIGH') else '✅ Normale Position (1-2% des Kapitals)'}""")

        # =============================================
        # SECTION 3: Trend & Volatility
        # =============================================
        st.markdown("### 📐 Trend & Volatilität")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown("**EMA Crossovers**")
            def ema_badge(cross):
                if cross in ("GOLDEN","BULLISH"): return f'<span style="color:#00FF7F;">🟢 {cross}</span>'
                elif cross in ("DEATH","BEARISH"): return f'<span style="color:#FF6347;">🔴 {cross}</span>'
                return f'<span style="color:#FFD700;">{cross}</span>'
            st.markdown(f"EMA 9/21: {ema_badge(ema_data.get('cross_9_21','N/A'))}", unsafe_allow_html=True)
            st.markdown(f"EMA 50/200: {ema_badge(ema_data.get('cross_50_200','N/A'))}", unsafe_allow_html=True)
            pve = ema_data.get('price_vs_ema21',0)
            st.markdown(f"Price vs EMA21: <span class='{cc(pve)}'>{pve:+.2f}%</span>", unsafe_allow_html=True)
            if ema_data.get('price_vs_ema200') is not None:
                pv200 = ema_data.get('price_vs_ema200',0)
                st.markdown(f"Price vs EMA200: <span class='{cc(pv200)}'>{pv200:+.2f}%</span>", unsafe_allow_html=True)
        with t2:
            st.markdown("**Bollinger Bands**")
            bbpos = bb_data.get("bb_position","MIDDLE")
            if bbp >= 80: bbc = "#FF6347"
            elif bbp <= 20: bbc = "#00FF7F"
            else: bbc = "#FFD700"
            st.markdown(f"Position: <span style='color:{bbc};font-weight:bold;'>{bbp:.0f}%</span> ({bbpos})", unsafe_allow_html=True)
            st.markdown(f"Width: **{bb_data.get('bb_width',0):.2f}%**")
            st.markdown(f"Upper: {fp(bb_data.get('bb_upper',0))}")
            st.markdown(f"Middle: {fp(bb_data.get('bb_middle',0))}")
            st.markdown(f"Lower: {fp(bb_data.get('bb_lower',0))}")
        with t3:
            st.markdown("**Volatilität (ATR)**")
            vol = atr_data.get("volatility","LOW")
            vc = {"VERY_HIGH":"#FF0040","HIGH":"#FF6347","MEDIUM":"#FFD700","LOW":"#00FF7F"}.get(vol,"#888")
            st.markdown(f"ATR: **{fp(atr_data.get('atr',0))}**")
            st.markdown(f"ATR %: <span style='color:{vc};font-weight:bold;'>{atr_data.get('atr_pct',0):.2f}%</span> ({vol})", unsafe_allow_html=True)
            st.markdown(f"BTC Korrelation: **{btc_corr.get('correlation',0):.2f}** ({btc_corr.get('corr_label','N/A')})")
        with st.expander("ℹ️ Was bedeuten Trend & Volatilität?"):
            st.markdown("""**EMA Crossovers (Exponential Moving Average):**
- **EMA 9/21:** Kurzfristiger Trend. Golden Cross (9 kreuzt 21 nach oben) = bullish Signal. Death Cross = bearish.
- **EMA 50/200:** Langfristiger Trend. Golden Cross hier ist ein sehr starkes Kaufsignal (historisch ~70% Trefferquote).
- **Preis vs EMA:** Wenn der Preis über der EMA liegt, ist der Trend bullish. Unter der EMA = bearish.
- **Kombination:** Kurzfristig bullish + langfristig bullish = stärkstes Signal für Long-Positionen.

**Bollinger Bands:**
- **Position 0-20%:** Preis am unteren Band → potentieller Kauf (überverkauft)
- **Position 80-100%:** Preis am oberen Band → potentieller Verkauf (überkauft)
- **Schmale Bänder (Width < 3%):** "Bollinger Squeeze" — ein starker Ausbruch steht bevor
- **Breite Bänder (Width > 8%):** Hohe Volatilität, Trend könnte sich erschöpfen

**ATR & Volatilität:**
- **Niedrig (< 1.5%):** Ruhiger Markt, enge SL möglich, Squeeze-Warnung
- **Mittel (1.5-3%):** Normales Trading-Umfeld
- **Hoch (> 3%):** Vorsicht — größere SL nötig, kleinere Positionsgrößen

**BTC Korrelation:**
- **> 0.7:** Coin folgt BTC eng → BTC-Chart beobachten
- **< -0.3:** Coin bewegt sich gegen BTC → potentieller Hedge
- **-0.3 bis 0.3:** Unabhängig — eigene Dynamik""")

        # =============================================
        # SECTION 4: Key Levels (S/R + Fibonacci)
        # =============================================
        st.markdown("### 🎯 Key Levels")
        l1, l2 = st.columns(2)
        with l1:
            st.markdown("**Support & Resistance**")
            for r_val in sr_data.get("resistances",[])[:3]:
                dist = (r_val/price-1)*100
                st.markdown(f'<span style="color:#FF6347;">R: {fp(r_val)}</span> <span style="color:#888;">({dist:+.2f}%)</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="color:white;font-weight:bold;">▸ Aktuell: {fp(price)}</span>', unsafe_allow_html=True)
            for s_val in sr_data.get("supports",[])[:3]:
                dist = (s_val/price-1)*100
                st.markdown(f'<span style="color:#00FF7F;">S: {fp(s_val)}</span> <span style="color:#888;">({dist:+.2f}%)</span>', unsafe_allow_html=True)
        with l2:
            st.markdown("**Fibonacci Retracement**")
            st.markdown(f"Zone: **{fib_data.get('fib_zone','N/A')}**")
            for label, val in fib_data.get("fib_levels",{}).items():
                dist = (val/price-1)*100
                active = "◀" if abs(dist) < 2 else ""
                st.markdown(f"`{label}` {fp(val)} ({dist:+.2f}%) {active}")
        with st.expander("ℹ️ Was bedeuten Key Levels?"):
            st.markdown("""**Support (Unterstützung):** Preislevels wo historisch Käufer eingestiegen sind. Der Kurs tendiert dazu, an diesen Levels zu "bouncen". Je öfter ein Level getestet wurde, desto stärker ist es.

**Resistance (Widerstand):** Preislevels wo historisch Verkäufer aktiv waren. Der Kurs prallt hier oft ab. Ein Durchbruch durch eine Resistance wird oft zu einer neuen Support.

**Fibonacci Retracement:** Basiert auf der Fibonacci-Zahlenfolge. Die wichtigsten Levels:
- **38.2%:** Erste Korrektur-Zone bei starken Trends
- **50%:** Psychologisches Level, oft respektiert
- **61.8% (Golden Zone):** Das wichtigste Fib-Level. Hier drehen die meisten Korrekturen um
- **78.6%:** Letzte Chance für einen Bounce, bevor der Trend bricht

**Trading mit Key Levels:**
- **Kaufen:** An Support-Levels oder in der Fib 61.8-78.6% Zone, mit SL knapp darunter
- **Verkaufen:** An Resistance-Levels oder wenn der Preis die Fib 0-23.6% Zone erreicht
- **Breakout:** Wenn ein Level klar durchbrochen wird (mit Volumen), in Richtung des Ausbruchs traden

**Im Chart oben** sind alle Levels als horizontale Linien eingezeichnet: Grün = Support, Rot = Resistance, Farbig = Fibonacci.""")

        # =============================================
        # SECTION 5: Price Range
        # =============================================
        st.markdown("### 📊 Price Range")
        pr1, pr2 = st.columns(2)
        for col, label in [(pr1,"7d"),(pr2,"30d")]:
            with col:
                hi = pr_data.get(f"{label}_high",0)
                lo = pr_data.get(f"{label}_low",0)
                pos = pr_data.get(f"{label}_position",50)
                rng = pr_data.get(f"{label}_range_pct",0)
                pc = "#00FF7F" if pos < 30 else ("#FF6347" if pos > 70 else "#FFD700")
                st.markdown(f"**{label} Range** ({rng:.1f}%)")
                st.markdown(f"High: {fp(hi)} → Low: {fp(lo)}")
                st.markdown(f'<div style="background:#2a2a4a;border-radius:4px;height:12px;position:relative;margin:4px 0;">'
                    f'<div style="position:absolute;left:{pos}%;top:-2px;width:4px;height:16px;background:{pc};border-radius:2px;"></div>'
                    f'<div style="background:linear-gradient(90deg,#00FF7F33,#FFD70033,#FF634733);height:12px;border-radius:4px;"></div></div>'
                    f'<span style="color:{pc};font-weight:bold;">{pos:.0f}%</span> <span style="color:#888;">Position in Range</span>', unsafe_allow_html=True)
        with st.expander("ℹ️ Was bedeutet Price Range?"):
            st.markdown("""**Price Range** zeigt wo sich der aktuelle Preis innerhalb der 7-Tage und 30-Tage Spanne befindet.

- **0-20% (nahe Low):** 🟢 Potentieller Kaufbereich — Preis ist nahe dem Tiefpunkt der letzten Tage
- **40-60% (Mitte):** 🟡 Neutraler Bereich
- **80-100% (nahe High):** 🔴 Potentieller Verkaufsbereich — Preis ist nahe dem Höchststand

**Range %** zeigt die Volatilität der Periode:
- Kleine Range (< 5%) = wenig Bewegung, möglicher Ausbruch bevorsteht
- Große Range (> 15%) = starke Schwankungen, höheres Risiko

**Kombination mit anderen Indikatoren:**
- Preis nahe 7d-Low + RSI überverkauft + Support-Level = starkes Kaufsignal
- Preis nahe 30d-High + RSI überkauft + Resistance = starkes Verkaufsignal""")

        # =============================================
        # SECTION 6: Indicator Summary Table
        # =============================================
        st.markdown("### 📋 Indikator-Übersicht")
        st.markdown(f"""| Indikator | Wert | Signal |
|---|---|---|
| MACD | {c.get('macd_trend','—')} (hist: {c.get('macd_histogram',0):.4f}) | {'🟢' if c.get('macd_trend')=='BULLISH' else ('🔴' if c.get('macd_trend')=='BEARISH' else '🟡')} |
| Stoch RSI | K: {c.get('stoch_rsi_k',50):.1f} / D: {c.get('stoch_rsi_d',50):.1f} | {'🟢' if c.get('stoch_rsi_k',50)<20 else ('🔴' if c.get('stoch_rsi_k',50)>80 else '🟡')} |
| Volume | {c.get('vol_trend','—')} ({c.get('vol_ratio',1.0):.2f}x avg) | {'🟢' if c.get('obv_trend')=='BULLISH' else ('🔴' if c.get('obv_trend')=='BEARISH' else '🟡')} |
| EMA 9/21 | {ema_data.get('cross_9_21','N/A')} | {'🟢' if ema_data.get('cross_9_21') in ('GOLDEN','BULLISH') else ('🔴' if ema_data.get('cross_9_21') in ('DEATH','BEARISH') else '🟡')} |
| EMA 50/200 | {ema_data.get('cross_50_200','N/A')} | {'🟢' if ema_data.get('cross_50_200') in ('GOLDEN','BULLISH') else ('🔴' if ema_data.get('cross_50_200') in ('DEATH','BEARISH') else '🟡')} |
| Bollinger | {bb_data.get('bb_position','—')} ({bb_data.get('bb_pct',50):.0f}%) | {'🟢' if bb_data.get('bb_pct',50)<20 else ('🔴' if bb_data.get('bb_pct',50)>80 else '🟡')} |
| ATR Volatilität | {atr_data.get('volatility','—')} ({atr_data.get('atr_pct',0):.2f}%) | {'⚠️' if atr_data.get('volatility') in ('HIGH','VERY_HIGH') else '✅'} |
| BTC Korrelation | {btc_corr.get('correlation',0):.2f} ({btc_corr.get('corr_label','N/A')}) | {'🔗' if abs(btc_corr.get('correlation',0))>0.5 else '🔓'} |""")
        with st.expander("ℹ️ Was bedeutet die Indikator-Übersicht?"):
            st.markdown("""Diese Tabelle fasst alle Indikatoren zusammen. **Je mehr 🟢 desto bullischer, je mehr 🔴 desto bearischer.**

**MACD:** Zeigt Momentum-Änderungen. Bullish = Kaufdruck nimmt zu. Das Histogramm zeigt die Stärke.

**Stoch RSI:** Kombination aus Stochastik und RSI. K < 20 = überverkauft (kaufen), K > 80 = überkauft (verkaufen). K kreuzt D nach oben = Kaufsignal.

**Volume:** Vergleicht aktuelles Volumen mit dem 20-Perioden-Durchschnitt. Hohes Volumen bestätigt Trends. OBV (On-Balance Volume) steigend = Akkumulation.

**Ideales Kauf-Setup:** RSI < 42 + EMA bullish + MACD bullish + Bollinger < 20% + Volumen steigend → starkes Kaufsignal mit hoher Erfolgswahrscheinlichkeit.

**Ideales Verkauf-Setup:** RSI > 58 + EMA bearish + MACD bearish + Bollinger > 80% + Volumen steigend → starkes Verkaufsignal.""")

        st.caption("⚠️ DYOR — Dies ist keine Finanzberatung. Alle Berechnungen basieren auf historischen Daten.")

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#555;font-size:11px;'>🧙‍♂️ Merlin | {len(coins_to_scan)}×{len(tf_to_scan)}TF | {exn} | BUY≤42 SELL≥58 | DYOR!</div>",unsafe_allow_html=True)
