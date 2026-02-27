"""
🧙‍♂️ Merlin Crypto Scanner — CryptoWaves-style RSI Dashboard
"""
import time, json, base64, os
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
    compute_individual_scores, compute_confluence_total, compute_alert_priority,
    detect_nr_pattern, nr_confluence_score, generate_nr_chart,
    backtest_signals,
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

# CTB/CTS cooldown: prevents same cross signal from firing repeatedly
# when RSI oscillates around the 40/60 boundary.
# Key: "symbol:signal" → timestamp of last fire.
CTB_CTS_COOLDOWN_SECONDS = 12 * 3600  # 12 hours

if "_ctb_cts_cooldown" not in st.session_state:
    st.session_state["_ctb_cts_cooldown"] = {}

def _check_cross_cooldown(symbol: str, signal: str) -> bool:
    """Returns True if this cross signal is allowed (not in cooldown)."""
    key = f"{symbol}:{signal}"
    cd = st.session_state["_ctb_cts_cooldown"]
    last_fire = cd.get(key, 0)
    now = time.time()
    if now - last_fire < CTB_CTS_COOLDOWN_SECONDS:
        return False  # Still in cooldown
    return True

def _record_cross_signal(symbol: str, signal: str):
    """Record that this cross signal just fired."""
    key = f"{symbol}:{signal}"
    st.session_state["_ctb_cts_cooldown"][key] = time.time()
    # Cleanup old entries (>24h)
    now = time.time()
    cd = st.session_state["_ctb_cts_cooldown"]
    st.session_state["_ctb_cts_cooldown"] = {
        k: v for k, v in cd.items() if now - v < 24 * 3600
    }

def compute_signal_base(rsi_4h, rsi_4h_prev, rsi_1d, symbol=""):
    """Original CryptoWaves-compatible 4h signal (always computed)."""
    if rsi_4h_prev < 40 and rsi_4h >= 40 and rsi_1d >= 35:
        if symbol and _check_cross_cooldown(symbol, "CTB"):
            _record_cross_signal(symbol, "CTB")
            return "CTB"
        return "BUY"  # Cooldown active → downgrade to BUY
    if rsi_4h_prev > 60 and rsi_4h <= 60 and rsi_1d <= 65:
        if symbol and _check_cross_cooldown(symbol, "CTS"):
            _record_cross_signal(symbol, "CTS")
            return "CTS"
        return "SELL"  # Cooldown active → downgrade to SELL
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
    sym = row.get("symbol", "")

    # If only 4h selected → original CryptoWaves logic
    if active_sig_tfs == {"4h"}:
        return compute_signal_base(
            row.get("rsi_4h", 50), row.get("rsi_prev_4h", 50), row.get("rsi_1D", 50), sym)

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

    # CTB/CTS cross detection (only from 4h) — with cooldown
    if "4h" in active_sig_tfs:
        rsi_4h = row.get("rsi_4h", 50)
        rsi_4h_prev = row.get("rsi_prev_4h", 50)
        rsi_1d = row.get("rsi_1D", 50)
        if rsi_4h_prev < 40 and rsi_4h >= 40 and rsi_1d >= 35 and bull >= bear:
            if _check_cross_cooldown(sym, "CTB"):
                _record_cross_signal(sym, "CTB")
                return "CTB"
            return "BUY"
        if rsi_4h_prev > 60 and rsi_4h <= 60 and rsi_1d <= 65 and bear >= bull:
            if _check_cross_cooldown(sym, "CTS"):
                _record_cross_signal(sym, "CTS")
                return "CTS"
            return "SELL"

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
# CUSTOM COINS — Persistent JSON storage
# ============================================================
CUSTOM_COINS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_coins.json")

def load_custom_coins() -> list:
    """Load custom coins from JSON file. Survives page reloads."""
    try:
        if os.path.exists(CUSTOM_COINS_FILE):
            with open(CUSTOM_COINS_FILE, "r") as f:
                data = json.load(f)
                return list(data) if isinstance(data, list) else []
    except Exception:
        pass
    return []

def save_custom_coins(coins: list):
    """Save custom coins to JSON file."""
    try:
        # Deduplicate, uppercase, strip
        clean = list(dict.fromkeys(c.strip().upper() for c in coins if c.strip()))
        with open(CUSTOM_COINS_FILE, "w") as f:
            json.dump(clean, f, indent=2)
    except Exception as e:
        st.warning(f"Fehler beim Speichern: {e}")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧙‍♂️ Settings")

    # Load custom coins for display
    _custom_coins = load_custom_coins()
    _custom_label = f"⭐ Custom ({len(_custom_coins)})" if _custom_coins else "⭐ Custom (leer)"

    coin_list_mode = st.radio("📋 Coins",
        ["CryptoWaves (107+)", "Top 100 Dynamic", "Extended (180+)", _custom_label], index=0)

    if coin_list_mode.startswith("⭐"):
        coin_source = _custom_coins if _custom_coins else CRYPTOWAVES_COINS
        if not _custom_coins:
            st.warning("⚠️ Custom-Liste ist leer. Füge unten Coins hinzu oder wechsle zu einer anderen Liste.")
    else:
        coin_source = {"CryptoWaves (107+)": CRYPTOWAVES_COINS, "Top 100 Dynamic": CRYPTOWAVES_COINS[:50],
                       "Extended (180+)": TOP_COINS_EXTENDED}[coin_list_mode]

    max_coins = st.slider("Max Coins", 20, 250, min(len(coin_source), 180), 10)
    st.markdown("---")

    # Custom Coins Management
    with st.expander("⭐ Custom Coins verwalten", expanded=coin_list_mode.startswith("⭐")):
        st.caption(f"**{len(_custom_coins)}** Coins gespeichert")

        # Add single coin
        new_coin = st.text_input("Coin hinzufügen:", placeholder="z.B. BTC, ETH, SOL", key="add_custom").strip().upper()
        if st.button("➕ Hinzufügen", key="btn_add_custom") and new_coin:
            coins_to_add = [c.strip() for c in new_coin.replace(" ", ",").split(",") if c.strip()]
            updated = _custom_coins + [c.upper() for c in coins_to_add if c.upper() not in _custom_coins]
            save_custom_coins(updated)
            st.success(f"✅ {', '.join(coins_to_add)} hinzugefügt")
            st.rerun()

        # Quick import from presets
        if st.button("📥 CryptoWaves-Liste importieren", key="import_cw"):
            merged = list(dict.fromkeys(_custom_coins + CRYPTOWAVES_COINS))
            save_custom_coins(merged)
            st.success(f"✅ {len(CRYPTOWAVES_COINS)} Coins importiert")
            st.rerun()

        # Show current list with remove buttons
        if _custom_coins:
            st.markdown("**Aktuelle Liste:**")
            # Show in rows of 4
            remove_coins = []
            cols_per_row = 4
            for i in range(0, len(_custom_coins), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(_custom_coins):
                        with col:
                            if st.button(f"❌ {_custom_coins[idx]}", key=f"rm_c_{idx}",
                                        use_container_width=True):
                                remove_coins.append(_custom_coins[idx])
            if remove_coins:
                updated = [c for c in _custom_coins if c not in remove_coins]
                save_custom_coins(updated)
                st.rerun()

            # Clear all
            if st.button("🗑️ Alle entfernen", key="clear_custom"):
                save_custom_coins([])
                st.rerun()

    st.markdown("---")
    st.markdown("### 🔤 Schriftgröße")
    font_scale = st.select_slider("Text", options=[0.85, 0.9, 1.0, 1.1, 1.2, 1.3],
        value=st.session_state.get("font_scale", 1.0), format_func=lambda x: {0.85:"Klein",0.9:"Kompakt",1.0:"Normal",1.1:"Groß",1.2:"Sehr groß",1.3:"XXL"}[x], key="fs_slider")
    if font_scale != st.session_state.get("font_scale", 1.0):
        st.session_state["font_scale"] = font_scale; st.rerun()
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
    "stoch_rsi":      {"label": "🎛️ Stoch RSI",        "default": True,  "weight": 15, "desc": "Feintuning K/D Crossover"},
    "bollinger":      {"label": "📏 Bollinger Bands",   "default": False, "weight": 10, "desc": "Squeeze-Erkennung + Band-Position"},
    "funding_rate":   {"label": "💰 Funding Rate",     "default": False, "weight": 10, "desc": "Futures Funding Rate (konträr)"},
    "fear_greed":     {"label": "😱 Fear & Greed",     "default": False, "weight": 8,  "desc": "Markt-Sentiment (konträr)"},
    "nr_breakout":    {"label": "🔮 NR Breakout",      "default": True,  "weight": 18, "desc": "NR4/NR7/NR10 Volatilitäts-Kontraktion → Ausbruch"},
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
    # P1: Only fetch essential TFs during scan. 1W is lazy (fetched on-demand per coin).
    scan_tfs = ["1h", "4h", "1D"]  # 1W excluded from bulk fetch
    all_tfs = list(dict.fromkeys(scan_tfs + [t for t in tfs if t != "1W"]))
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

    # --- PHASE 1.5: Quick RSI pre-check to find signal candidates ---
    # Identify coins likely to have signals, so we only fetch 1W for those
    signal_candidates = set()
    for sym in coin_list:
        df_4h = klines_cache.get((sym, "4h"), pd.DataFrame())
        if not df_4h.empty and len(df_4h) >= 15:
            rs = RSIIndicator(close=df_4h["close"], window=14).rsi().dropna()
            if len(rs) >= 1:
                rsi_val = float(rs.iloc[-1])
                if rsi_val <= 45 or rsi_val >= 55:  # Likely BUY/SELL/CTB/CTS
                    signal_candidates.add(sym)

    # P1: Fetch 1W klines ONLY for signal candidates (saves ~60-70% of 1W API calls)
    if signal_candidates:
        w_tasks = [(sym, "1W") for sym in signal_candidates]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            w_futures = {pool.submit(_fetch_one, t): t for t in w_tasks}
            for fut in as_completed(w_futures):
                try:
                    key, df_k = fut.result()
                    klines_cache[key] = df_k
                except Exception:
                    pass

    # --- PHASE 2: Process results (CPU-only, fast) ---
    results = []
    for sym in coin_list:
        r = {"symbol": sym}
        tk = tickers.get(sym, {}); mk = mkt_lk.get(sym, {})

        # Price: Ticker → CoinGecko Market → Klines Close (fallback chain)
        if tk:
            r["price"]=float(tk.get("last",0)or 0); r["change_24h"]=float(tk.get("change_pct",0)or 0); r["volume_24h"]=float(tk.get("volume",0)or 0)
        elif isinstance(mk,pd.Series) and not mk.empty:
            r["price"]=float(mk.get("price",0))if pd.notna(mk.get("price"))else 0; r["change_24h"]=float(mk.get("change_24h",0))if pd.notna(mk.get("change_24h"))else 0; r["volume_24h"]=float(mk.get("volume_24h",0))if pd.notna(mk.get("volume_24h"))else 0
        else: r["price"],r["change_24h"],r["volume_24h"]=0,0,0

        # Klines price fallback: if no ticker/market price, use last close from 4h klines
        if r["price"]==0:
            for fb_tf in ["4h","1h","1D","1W"]:
                fb_df = klines_cache.get((sym, fb_tf), pd.DataFrame())
                if not fb_df.empty and len(fb_df)>=2:
                    r["price"]=float(fb_df["close"].iloc[-1])
                    # Estimate 24h change from klines
                    if len(fb_df)>=6 and fb_tf=="4h":
                        r["change_24h"]=round((fb_df["close"].iloc[-1]/fb_df["close"].iloc[-7]-1)*100,2)
                    break

        # Metadata from CoinGecko
        if isinstance(mk,pd.Series) and not mk.empty:
            r["rank"]=int(mk.get("rank",999))if pd.notna(mk.get("rank"))else 999
            r["coin_name"]=str(mk.get("name",sym))if pd.notna(mk.get("name"))else sym
            r["coin_image"]=str(mk.get("image",""))if pd.notna(mk.get("image"))else ""
            for f in ["change_1h","change_7d","change_30d"]:
                r[f]=float(mk.get(f,0))if pd.notna(mk.get(f))else 0
        else:
            r["rank"],r["coin_name"],r["coin_image"]=999,sym,""; r["change_1h"],r["change_7d"],r["change_30d"]=0,0,0

        # Skip only if we truly have no price from ANY source
        if r["price"]==0: continue

        kld={}
        process_tfs = list(all_tfs) + (["1W"] if "1W" not in all_tfs else [])
        for tf in process_tfs:
            df_k = klines_cache.get((sym, tf), pd.DataFrame())
            if not df_k.empty and len(df_k)>=15:
                kld[tf]=df_k
                rs = RSIIndicator(close=df_k["close"],window=14).rsi().dropna()
                if len(rs)>=2:
                    r[f"rsi_{tf}"]=round(float(rs.iloc[-1]),2); r[f"rsi_prev_{tf}"]=round(float(rs.iloc[-2]),2)
                else: r[f"rsi_{tf}"],r[f"rsi_prev_{tf}"]=50.0,50.0
                r[f"closes_{tf}"]=json.dumps([round(c,6) for c in df_k["close"].tail(20).tolist()])
            else: r[f"rsi_{tf}"],r[f"rsi_prev_{tf}"]=50.0,50.0; r[f"closes_{tf}"]="[]"
        r["signal"]=compute_signal_base(r.get("rsi_4h",50),r.get("rsi_prev_4h",50),r.get("rsi_1D",50),sym)
        r["border_alpha"]=border_alpha(r.get("rsi_4h",50),r["signal"])
        ptf="4h" if "4h" in kld else (list(tfs)[0] if tfs else None)
        if ptf and ptf in kld:
            md=calculate_macd(kld[ptf]); r["macd_trend"]=md["trend"]; r["macd_histogram"]=md["histogram"]; r["macd_histogram_pct"]=md.get("histogram_pct",0); r["macd_hist_dir"]=md.get("hist_direction","FLAT")
            vd=calculate_volume_analysis(kld[ptf]); r["vol_trend"]=vd["vol_trend"]; r["vol_ratio"]=vd["vol_ratio"]; r["obv_trend"]=vd["obv_trend"]; r["candle_dir"]=vd.get("candle_dir","NEUTRAL")
            sk=calculate_stoch_rsi(kld[ptf]); r["stoch_rsi_k"]=sk["stoch_rsi_k"]; r["stoch_rsi_d"]=sk["stoch_rsi_d"]
            # NR4/NR7/NR10 detection (uses 4h klines, no extra API call)
            nr=detect_nr_pattern(kld[ptf]); r["nr_level"]=nr.get("nr_level",""); r["nr_breakout"]=nr.get("breakout",""); r["nr_range_ratio"]=nr.get("range_ratio",1.0); r["nr_box_high"]=nr.get("box_high",0); r["nr_box_low"]=nr.get("box_low",0)
            # EMA alignment + RSI divergence (cheap CPU, no API calls)
            ema_scan=calculate_ema_alignment_fast(kld[ptf])
            r["ema_trend"]=ema_scan.get("ema_trend","NEUTRAL"); r["ema_bull_count"]=ema_scan.get("ema_bull_count",0); r["ema_bear_count"]=ema_scan.get("ema_bear_count",0)
            div_scan=detect_rsi_divergence(kld[ptf])
            r["divergence"]=div_scan.get("divergence","NONE"); r["div_type"]=div_scan.get("div_type","NONE")
            if smc:
                ob=detect_order_blocks(kld[ptf]); fv=detect_fair_value_gaps(kld[ptf]); ms=detect_market_structure(kld[ptf])
                r["ob_signal"]=ob["ob_signal"]; r["fvg_signal"]=fv["fvg_signal"]; r["market_structure"]=ms["structure"]; r["bos"]=ms["break_of_structure"]
            sc_d={"ob_signal":r.get("ob_signal","NONE"),"fvg_signal":r.get("fvg_signal","BALANCED"),"structure":r.get("market_structure","UNKNOWN")} if smc else None
            # NR data dict for scoring
            nr_d={"nr_level":r["nr_level"],"breakout":r["nr_breakout"],"range_ratio":r["nr_range_ratio"]} if r["nr_level"] else None
            # FULL scoring — same function used everywhere (Bug 1 fix)
            individual=compute_individual_scores(
                rsi_4h=r.get("rsi_4h",50), rsi_1d=r.get("rsi_1D",50),
                macd_data=md, volume_data=vd, stoch_rsi_data=sk,
                smc_data=sc_d, ema_data=ema_scan, divergence_data=div_scan,
                nr_data=nr_d,
            )
            # Default scan filters: everything we compute at scan time
            scan_filters={"rsi_4h":True,"rsi_1d":True,"macd":True,"volume_obv":True,
                         "stoch_rsi":True,"ema_alignment":True,"rsi_divergence":True,
                         "nr_breakout":True,"smart_money":smc,
                         "bollinger":False,"funding_rate":False,"fear_greed":False}
            cf=compute_confluence_total(individual, scan_filters)
            r["score"]=cf["score"]; r["confluence_rec"]=cf["recommendation"]; r["reasons"]=" | ".join(cf["reasons"][:3])
            # Alert priority: how many indicators align in signal direction
            pri=compute_alert_priority(individual, r["signal"])
            r["priority"]=pri["priority"]; r["priority_label"]=pri["label"]; r["priority_aligned"]=pri["aligned"]; r["priority_special"]=" ".join(pri["special"])
        else:
            r.update({"macd_trend":"NEUTRAL","macd_histogram":0,"macd_histogram_pct":0,"macd_hist_dir":"FLAT","vol_trend":"—","vol_ratio":1.0,"obv_trend":"—","candle_dir":"NEUTRAL","stoch_rsi_k":50.0,"stoch_rsi_d":50.0,"nr_level":"","nr_breakout":"","nr_range_ratio":1.0,"nr_box_high":0,"nr_box_low":0,"ema_trend":"NEUTRAL","ema_bull_count":0,"ema_bear_count":0,"divergence":"NONE","div_type":"NONE","score":0,"confluence_rec":"WAIT","reasons":"","priority":0,"priority_label":"","priority_aligned":0,"priority_special":""})
        # P2: Precompute sparklines once (cached with scan_all results)
        try: r["spark_4h"] = sparkline_img(json.loads(r.get("closes_4h", "[]")), 110, 30)
        except: r["spark_4h"] = ""
        try: r["spark_1D"] = sparkline_img(json.loads(r.get("closes_1D", "[]")), 110, 30)
        except: r["spark_1D"] = ""
        results.append(r)
    return pd.DataFrame(results) if results else pd.DataFrame()

# ============================================================
# LOAD
# ============================================================
coins_to_scan = coin_source[:max_coins]
tf_to_scan = ["1h", "4h", "1D", "1W"]  # Always scan all timeframes
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
        st.markdown(f"**Primary Exchange:** {ex_info['active_exchange'].upper()} ({'✅ verbunden' if ex_info['connected'] else '❌ nicht verbunden'})")
        all_ex = ex_info.get("all_exchanges", [])
        if len(all_ex) > 1:
            st.markdown(f"**Verfügbare Exchanges:** {', '.join(e.upper() for e in all_ex)}")
        st.markdown(f"**Klines OK:** {rsi_ok} / {total} ({rsi_ok/total*100:.0f}%)")
        st.markdown(f"**Klines fehlend:** {rsi_fail} / {total}")
        st.markdown(f"**Signal-TFs aktiv:** {', '.join(sorted(active_sig_tfs))}")
        if rsi_fail > 0:
            fail_coins = df[df["rsi_4h"] == 50.0]["symbol"].tolist()[:20]
            st.markdown(f"**Betroffene Coins (max 20):** {', '.join(fail_coins)}")
            # Known untradeable tokens
            stablecoins = {"USD1", "WLFI", "RLUSD", "FDUSD", "TUSD", "USDC", "USDE"}
            untradeable = [c for c in fail_coins if c in stablecoins]
            tradeable_fail = [c for c in fail_coins if c not in stablecoins]
            if untradeable:
                st.caption(f"ℹ️ Stablecoins/Governance (kein RSI möglich): {', '.join(untradeable)}")
            if tradeable_fail:
                st.caption(f"⚠️ Echte Coins ohne Daten: {', '.join(tradeable_fail)} — auf keiner Exchange als /USDT gefunden")
        # Show per-coin exchange mapping
        from data_fetcher import _coin_exchange_cache
        if _coin_exchange_cache:
            ex_counts = {}
            for coin, exname in _coin_exchange_cache.items():
                ex_counts[exname] = ex_counts.get(exname, 0) + 1
            ex_summary = " | ".join(f"{n.upper()}: {c} Coins" for n, c in sorted(ex_counts.items(), key=lambda x: -x[1]))
            st.markdown(f"**Exchange-Verteilung:** {ex_summary}")
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
        # P2: Use precomputed sparklines from scan_all (no JSON parse/SVG gen per render)
        s7 = row.get("spark_4h", "")
        s30 = row.get("spark_1D", "")
        ch=f'<div class="charts"><div><div class="clbl">● 7d</div>{s7}</div><div><div class="clbl">● 30d</div>{s30}</div></div>'
    # Strong glow effect on signal label
    glow = f'text-shadow:0 0 8px {sc},0 0 16px {sc};' if is_strong_sell or is_strong_buy else ""
    r1h=row.get("rsi_1h",50); r4=row.get("rsi_4h",50); r1d=row.get("rsi_1D",50); r1w=row.get("rsi_1W",50)
    # Confluence score — separate line + Alignment badge
    conf_sc = row.get("score", 0)
    conf_rec = row.get("confluence_rec", "WAIT")
    if conf_sc >= 30: conf_html = f'<span style="color:#00FF7F;font-size:10px;">⚡ {conf_rec} ({conf_sc})</span>'
    elif conf_sc >= 10: conf_html = f'<span style="color:#88FFAA;font-size:10px;">↑ {conf_rec} ({conf_sc})</span>'
    elif conf_sc <= -30: conf_html = f'<span style="color:#FF6347;font-size:10px;">⚡ {conf_rec} ({conf_sc})</span>'
    elif conf_sc <= -10: conf_html = f'<span style="color:#FF8888;font-size:10px;">↓ {conf_rec} ({conf_sc})</span>'
    else: conf_html = f'<span style="color:#666;font-size:10px;">{conf_rec} ({conf_sc})</span>'

    # RSI ↔ Confluence Alignment Badge
    align_badge = ""
    if sig in ("BUY", "CTB", "SELL", "CTS"):
        is_bull = sig in ("BUY", "CTB")
        if (is_bull and conf_sc > 5) or (not is_bull and conf_sc < -5):
            align_badge = '<span style="background:#00FF7F22;color:#00FF7F;font-size:9px;padding:1px 5px;border-radius:4px;margin-left:4px;" title="RSI + Confluence zeigen in die gleiche Richtung">✅ Bestätigt</span>'
        elif (is_bull and conf_sc < -5) or (not is_bull and conf_sc > 5):
            align_badge = '<span style="background:#FF634722;color:#FF6347;font-size:9px;padding:1px 5px;border-radius:4px;margin-left:4px;" title="RSI + Confluence widersprechen sich — Vorsicht!">⚠️ Konflikt</span>'
    # NR badge
    nr_lv = row.get("nr_level", "")
    nr_bo = row.get("nr_breakout", "")
    if nr_lv:
        nr_colors = {"NR4": "#66bb6a", "NR7": "#ffee58", "NR10": "#ff9800"}
        nr_c = nr_colors.get(nr_lv, "#ffee58")
        nr_bo_txt = f" ▲UP" if nr_bo == "UP" else (f" ▼DN" if nr_bo == "DOWN" else " ⏳")
        nr_badge = f'<span style="background:{nr_c};color:#000;font-size:10px;font-weight:bold;padding:1px 5px;border-radius:8px;margin-left:4px;">{nr_lv}{nr_bo_txt}</span>'
    else:
        nr_badge = ""
    # Priority badge (🔥🔥🔥 = extreme, 🔥🔥 = strong, 🔥 = moderate)
    pri_label = row.get("priority_label", "")
    pri_aligned = row.get("priority_aligned", 0)
    pri_special = row.get("priority_special", "")
    if pri_label:
        pri_title = f"{pri_aligned} Indikatoren aligned"
        if pri_special:
            pri_title += f" + {pri_special}"
        pri_badge = f'<span title="{pri_title}" style="margin-left:4px;font-size:11px;">{pri_label}</span>'
    else:
        pri_badge = ""
    return f'''<div class="crow" style="{bdr}">
<div class="ic">{icon(im, sym)}</div>
<div class="inf"><div><span class="cn">{sym}</span><span class="cf">{nm}</span><span class="cr">{rks}</span></div>
<div class="pl">Price: <b style="color:white;">{fp(row["price"])}</b></div>
<div class="chs">Ch%: <span class="{cc(c1h)}">{c1h:+.2f}%</span> <span class="{cc(c24)}">{c24:+.2f}%</span> <span class="{cc(c7)}" style="font-weight:bold;">{c7:+.2f}%</span> <span class="{cc(c30)}">{c30:+.2f}%</span></div></div>
{ch}
<div class="sig"><span style="font-size:11px;color:#888;">RSI:</span> <span class="sl" style="color:{sc};{glow}">{sig}</span>{pri_badge}{nr_badge}
<div class="rsi-row"><span class="rsi-pill">1h: <b style="color:{rsc(r1h)};">{r1h:.1f}</b></span><span class="rsi-pill" style="background:#252550;"><b style="color:{rsc(r4)};">{r4:.1f}</b> 4h</span><span class="rsi-pill" style="background:#252550;"><b style="color:{rsc(r1d)};">{r1d:.1f}</b> 1D</span><span class="rsi-pill">1W: <b style="color:{rsc(r1w)};">{r1w:.1f}</b></span></div>
<div style="margin-top:2px;">Conf: {conf_html}{align_badge}</div>
</div></div>'''

def render_rows_with_chart(dataframe, tab_key, max_rows=60):
    """Render coin rows with inline chart button under each row."""
    chart_state_key = f"chart_{tab_key}"
    for _,row in dataframe.head(max_rows).iterrows():
        sym=row["symbol"]
        st.markdown(crow_html(row), unsafe_allow_html=True)
        # Inline buttons: Chart + Detail + Watchlist (compact)
        in_wl = sym in st.session_state.get("_watchlist", set())
        bc1, bc2, bc3, _ = st.columns([1, 1, 1, 8])
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
        with bc3:
            star_label = "⭐" if in_wl else "☆"
            if st.button(star_label, key=f"wl_{tab_key}_{sym}", use_container_width=True,
                        help="Aus Watchlist entfernen" if in_wl else "Zur Watchlist"):
                if in_wl:
                    st.session_state["_watchlist"].discard(sym)
                else:
                    st.session_state["_watchlist"].add(sym)
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
all_ex = ex.get("all_exchanges", [])
ex_display = "+".join(e.upper() for e in all_ex) if len(all_ex) > 1 else exn
sig_tfs_str = "+".join(sorted(active_sig_tfs, key=lambda t: ["1h","4h","1D","1W"].index(t) if t in ["1h","4h","1D","1W"] else 99))
st.markdown(f'''<div class="hbar"><div><span class="htitle">🧙‍♂️ Merlin Crypto Scanner</span> <span class="badge {bc}">Market: {ml}</span></div>
<div class="hstat">Avg RSI (4h): <b>{avg4:.2f}</b> ({(avg4-50)/50*100:+.2f}%) | Ch%: <span class="{cc(a1)}">{a1:+.2f}%</span> <span class="{cc(a24)}">{a24:+.2f}%</span> <span class="{cc(a7)}">{a7:+.2f}%</span> <span class="{cc(a30)}">{a30:+.2f}%</span></div>
<div class="hstat"><span style="color:#FF6347;">🔴 {sct}</span> <span style="color:#FFD700;">🟡 {wct}</span> <span style="color:#00FF7F;">🟢 {bct}</span> | 📡 {ex_display} | Sig: {sig_tfs_str}</div></div>''', unsafe_allow_html=True)

# F3: Watchlist initialization
if "_watchlist" not in st.session_state:
    st.session_state["_watchlist"] = set()
wl_count = len(st.session_state["_watchlist"])
wl_label = f"⭐ Watchlist ({wl_count})" if wl_count > 0 else "⭐ Watchlist"

# ============================================================
# TABS
# ============================================================
tab_alerts, tab_watch, tab_hm, tab_mc, tab_conf, tab_nr, tab_hist, tab_det = st.tabs([
    f"🚨 24h Alerts {sct}🔴 {bct}🟢", wl_label, "🔥 RSI Heatmap", "📊 By Market Cap", "🎯 Confluence", "🔮 NR Breakout", "📊 Signal History", "🔍 Detail"])

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
    with cs: asort=st.selectbox("Sort:",["Priority","Signal Strength","RSI (4h)","RSI (1D)","Rank"],key="asort",label_visibility="collapsed")

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
    elif asort=="Priority" and "priority" in adf.columns:
        # Sort by priority desc, then by abs(score) desc within same priority
        adf["_abs_score"] = adf["score"].abs()
        adf=adf.sort_values(["priority","_abs_score"], ascending=[False,False])
    else: adf=pd.concat([adf[adf["signal"].isin(["CTS","SELL"])].sort_values("rsi_4h",ascending=False),adf[adf["signal"].isin(["CTB","BUY"])].sort_values("rsi_4h")])

    if adf.empty: st.info("No active alerts with this filter.")
    else:
        st.caption(f"**{len(adf)}** signals | 🔴 {len(adf[adf['signal'].isin(['SELL','CTS'])])} SELL/CTS  🟢 {len(adf[adf['signal'].isin(['BUY','CTB'])])} BUY/CTB")
        render_rows_with_chart(adf, "alert", 40)

# ============================================================
# TAB 2: WATCHLIST — pinned favorite coins
# ============================================================
with tab_watch:
    st.markdown("### ⭐ Watchlist")
    wl = st.session_state.get("_watchlist", set())

    # Add coins
    wc1, wc2 = st.columns([3, 1])
    with wc1:
        all_syms = sorted(df["symbol"].tolist()) if not df.empty else []
        add_sym = st.selectbox("Coin hinzufügen:", [""] + [s for s in all_syms if s not in wl],
                               key="wl_add", label_visibility="collapsed",
                               placeholder="Coin zur Watchlist hinzufügen...")
    with wc2:
        if st.button("➕ Hinzufügen", key="wl_add_btn", disabled=not add_sym):
            if add_sym:
                st.session_state["_watchlist"].add(add_sym)
                st.rerun()

    if wl:
        # Show watchlist coins with full cards
        wl_df = df[df["symbol"].isin(wl)].copy()
        if not wl_df.empty:
            # Sort by priority then score
            if "priority" in wl_df.columns:
                wl_df["_abs_score"] = wl_df["score"].abs()
                wl_df = wl_df.sort_values(["priority", "_abs_score"], ascending=[False, False])
            st.caption(f"**{len(wl_df)}** Coins in Watchlist")
            for _, row in wl_df.iterrows():
                c1, c2 = st.columns([10, 1])
                with c1:
                    st.markdown(crow_html(row), unsafe_allow_html=True)
                with c2:
                    if st.button("❌", key=f"wl_rm_{row['symbol']}", help=f"{row['symbol']} entfernen"):
                        st.session_state["_watchlist"].discard(row["symbol"])
                        st.rerun()
        else:
            st.info("Watchlist-Coins nicht im aktuellen Scan gefunden.")
    else:
        st.info("Deine Watchlist ist leer. Füge Coins über das Dropdown oben hinzu oder nutze die ⭐-Buttons auf den Coin-Karten.")

# ============================================================
# TAB 3: RSI HEATMAP — Coin Rank default, search, grid
# ============================================================
with tab_hm:
    h1,h2,h3=st.columns([2,1,1])
    with h1: htf=st.selectbox("Timeframe",tf_to_scan,index=1,key="htf")
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

**Die 7 Standard-Filter (vorausgewählt):** RSI 4h (±30), RSI 1D (±20), MACD (±20), NR Breakout (±18), Volume & OBV (±15), Stoch RSI (±15), RSI Divergenz (±15), EMA Alignment (±12)

**Zusätzliche Filter (optional):** Smart Money (±15), Bollinger Bands (±10), Funding Rate (±10), Fear & Greed (±8)

**Score:** ≥60 STRONG BUY · 30-59 BUY · 10-29 LEAN BUY · -9 bis 9 WAIT · -29 bis -10 LEAN SELL · -59 bis -30 SELL · ≤-60 STRONG SELL

💡 **NR Badge** (z.B. NR7 ⏳) auf Coin-Karten zeigt aktive Kompression.""")

    # --- COMPUTE EXTENDED SCORES ON-DEMAND ---
    # Uses the SAME compute_individual_scores() as scan_all for consistency.
    # Extra indicators (EMA, Divergence, BB, SMC) are computed here on-demand.
    active_filters = st.session_state.get("conf_filters", {k: v["default"] for k, v in CONFLUENCE_FILTERS.items()})
    # P3: Fast path — if no extra filters changed from scan_all defaults, reuse cached scores
    scan_defaults = {"rsi_4h":True,"rsi_1d":True,"macd":True,"volume_obv":True,
                    "stoch_rsi":True,"ema_alignment":True,"rsi_divergence":True,
                    "nr_breakout":True,"smart_money":False,"bollinger":False,
                    "funding_rate":False,"fear_greed":False}
    filters_match_scan = all(active_filters.get(k, v) == v for k, v in scan_defaults.items())

    if filters_match_scan and "score" in df.columns:
        # Filters identical to scan_all → reuse cached scores (skip 107× recomputation)
        sdf = df.copy()
        sdf["dyn_score"] = sdf["score"]
        sdf["dyn_rec"] = sdf["confluence_rec"]
        sdf["dyn_reasons"] = sdf["reasons"]
    else:
        # Only need extra klines fetch for indicators NOT computed in scan_all
        needs_extra = any(active_filters.get(k, False) for k in ["smart_money", "bollinger"])

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

            # Prepare data dicts from scan_all results
            macd_data = {
                "trend": row.get("macd_trend", "NEUTRAL"),
                "histogram": row.get("macd_histogram", 0),
                "histogram_pct": row.get("macd_histogram_pct", 0),
                "hist_direction": row.get("macd_hist_dir", "FLAT"),
            }
            volume_data = {
                "vol_trend": row.get("vol_trend", "NEUTRAL"),
                "vol_ratio": row.get("vol_ratio", 1.0),
                "obv_trend": row.get("obv_trend", "NEUTRAL"),
                "candle_dir": row.get("candle_dir", "NEUTRAL"),
            }
            stoch_data = {
                "stoch_rsi_k": row.get("stoch_rsi_k", 50.0),
                "stoch_rsi_d": row.get("stoch_rsi_d", 50.0),
            }

            # On-demand extra indicators (only what scan_all doesn't compute)
            smc_data = None; bb_data = None
            ema_data = {"ema_trend": row.get("ema_trend", "NEUTRAL"),
                        "ema_bull_count": int(row.get("ema_bull_count", 0)),
                        "ema_bear_count": int(row.get("ema_bear_count", 0))}
            div_data = {"divergence": row.get("divergence", "NONE"), "div_type": row.get("div_type", "NONE")}
            if needs_extra:
                kl_4h = fetch_klines_smart(sym, "4h")
                if not kl_4h.empty and len(kl_4h) >= 15:
                    if active_filters.get("smart_money", False):
                        ob = detect_order_blocks(kl_4h); ms = detect_market_structure(kl_4h)
                        smc_data = {"ob_signal": ob.get("ob_signal", "NONE"),
                                    "fvg_signal": "BALANCED",
                                    "structure": ms.get("structure", "UNKNOWN")}
                    if active_filters.get("bollinger", False):
                        bb_data = calculate_bb_squeeze(kl_4h)

            fr_val = funding_rates_map.get(sym, None) if active_filters.get("funding_rate", False) else None
            fg_val = fear_greed_val if active_filters.get("fear_greed", False) else None

            nr_d = None
            if active_filters.get("nr_breakout", False) and row.get("nr_level", ""):
                nr_d = {
                    "nr_level": row.get("nr_level", ""),
                    "breakout": row.get("nr_breakout", ""),
                    "range_ratio": row.get("nr_range_ratio", 1.0),
                }

            individual = compute_individual_scores(
                rsi_4h=row.get("rsi_4h", 50),
                rsi_1d=row.get("rsi_1D", 50),
                macd_data=macd_data,
                volume_data=volume_data,
                stoch_rsi_data=stoch_data,
                smc_data=smc_data,
                ema_data=ema_data,
                divergence_data=div_data,
                bb_data=bb_data,
                funding_rate=fr_val,
                fear_greed=fg_val,
                nr_data=nr_d,
            )
            result = compute_confluence_total(individual, active_filters)

            scored_rows.append({**row.to_dict(),
                "dyn_score": result["score"],
                "dyn_rec": result["recommendation"],
                "dyn_reasons": " | ".join(result["reasons"][:4])})
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
# TAB 5: NR BREAKOUT SCANNER
# ============================================================
with tab_nr:
    st.markdown("### 🔮 NR Breakout Scanner — Narrow Range Detection")
    with st.expander("ℹ️ Was ist NR4/NR7/NR10?", expanded=False):
        st.markdown("""**Narrow Range (NR)** erkennt Coins in einer **Volatilitäts-Kompression** — die Ruhe vor dem Sturm.

**NR4:** Engste Kerze seit 4 Perioden → leichte Kompression
**NR7:** Engste Kerze seit 7 Perioden → starke Kompression, Ausbruch wahrscheinlich
**NR10:** Engste Kerze seit 10 Perioden → extreme Kompression, explosiver Ausbruch erwartet

**Kombination mit RSI:** NR + RSI überverkauft → Breakout UP wahrscheinlich · NR + RSI überkauft → Breakout DOWN wahrscheinlich · NR + RSI neutral → Richtung offen, Box beobachten

**Box:** High/Low der NR-Kerze definieren die Breakout-Levels. Preis über Box = UP Breakout, Preis unter Box = DOWN Breakout.""")

    # Filter NR coins
    nr_coins = df[df["nr_level"].astype(str).isin(["NR4","NR7","NR10"])].copy()
    nr_count = len(nr_coins)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"**{nr_count}** von {len(df)} Coins aktuell in NR-Kompression")
    with c2:
        nr_filter = st.selectbox("Filter:", ["Alle NR", "NR7 + NR10", "Nur NR10"], key="nr_filt")

    if nr_filter == "NR7 + NR10":
        nr_coins = nr_coins[nr_coins["nr_level"].isin(["NR7", "NR10"])]
    elif nr_filter == "Nur NR10":
        nr_coins = nr_coins[nr_coins["nr_level"] == "NR10"]

    if nr_coins.empty:
        st.info("🔮 Aktuell keine Coins mit NR-Pattern erkannt. Das ist normal — NR tritt nur bei 5-15% der Coins gleichzeitig auf.")
    else:
        # Sort: NR10 first, then NR7, then NR4, then by RSI extremity
        nr_order = {"NR10": 0, "NR7": 1, "NR4": 2}
        nr_coins["_nr_sort"] = nr_coins["nr_level"].map(nr_order)
        nr_coins["_rsi_ext"] = (nr_coins["rsi_4h"] - 50).abs()
        nr_coins = nr_coins.sort_values(["_nr_sort", "_rsi_ext"], ascending=[True, False])

        for idx, (_, row) in enumerate(nr_coins.iterrows()):
            sym = row["symbol"]
            nr_lv = row["nr_level"]
            rsi4 = row.get("rsi_4h", 50)
            bo = row.get("nr_breakout", "")
            box_h = row.get("nr_box_high", 0)
            box_l = row.get("nr_box_low", 0)
            price = row.get("price", 0)
            rr = row.get("nr_range_ratio", 1.0)

            # Colors
            nr_colors = {"NR4": "#66bb6a", "NR7": "#ffee58", "NR10": "#ff9800"}
            nr_c = nr_colors.get(nr_lv, "#ffee58")

            # Direction prediction based on RSI
            if rsi4 <= 30:
                direction = "🟢 Breakout UP sehr wahrscheinlich"
                dir_c = "#00FF7F"
            elif rsi4 <= 42:
                direction = "🟢 Breakout UP wahrscheinlich"
                dir_c = "#88FFAA"
            elif rsi4 >= 70:
                direction = "🔴 Breakout DOWN sehr wahrscheinlich"
                dir_c = "#FF4444"
            elif rsi4 >= 58:
                direction = "🔴 Breakout DOWN wahrscheinlich"
                dir_c = "#FF8888"
            else:
                direction = "⏳ Richtung offen — Box beobachten"
                dir_c = "#FFD700"

            # Breakout status
            if bo == "UP":
                bo_html = '<span style="color:#00FF7F;font-weight:bold;">✅ BREAKOUT UP</span>'
            elif bo == "DOWN":
                bo_html = '<span style="color:#FF4444;font-weight:bold;">✅ BREAKOUT DOWN</span>'
            else:
                bo_html = '<span style="color:#888;">⏳ Ausstehend</span>'

            im = row.get("coin_image", "")
            st.markdown(f'''<div style="background:#1a1a2e;border-radius:10px;padding:12px 16px;margin:6px 0;border-left:5px solid {nr_c};">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
<div style="display:flex;align-items:center;gap:8px;">
{icon(im, sym)}
<div>
<b style="color:white;font-size:16px;">{sym}</b>
<span style="background:{nr_c};color:#000;font-size:12px;font-weight:bold;padding:2px 8px;border-radius:10px;margin-left:6px;">{nr_lv}</span>
<div style="color:#888;font-size:11px;">Range-Ratio: {rr:.3f} · Box: {fp(box_l)} — {fp(box_h)}</div>
</div>
</div>
<div style="text-align:right;">
<div style="font-size:13px;">{bo_html}</div>
<div style="color:{dir_c};font-size:12px;margin-top:2px;">{direction}</div>
<div style="color:#888;font-size:11px;">RSI 4h: <b style="color:{rsc(rsi4)};">{rsi4:.1f}</b> · Preis: {fp(price)}</div>
</div>
</div>
</div>''', unsafe_allow_html=True)

            # On-demand chart button
            chart_key = f"nr_chart_{sym}"
            if st.button(f"📊 Chart anzeigen — {sym}", key=chart_key):
                with st.spinner(f"Lade {sym} 4h Klines..."):
                    chart_df = fetch_klines_smart(sym, "4h")
                    if not chart_df.empty and len(chart_df) >= 15:
                        nr_d = detect_nr_pattern(chart_df)
                        chart_bytes = generate_nr_chart(chart_df, sym, nr_d)
                        if chart_bytes:
                            st.image(chart_bytes, use_container_width=True)
                            # Box info below chart
                            st.caption(f"🔮 {nr_d.get('nr_level','—')} · Box: {fp(nr_d.get('box_low',0))} — {fp(nr_d.get('box_high',0))} · Mid: {fp(nr_d.get('box_mid',0))} · Breakout: {nr_d.get('breakout','⏳')}")
                        else:
                            st.warning("Chart konnte nicht generiert werden — zu wenige Daten.")
                    else:
                        st.warning(f"Keine ausreichenden Klines für {sym}.")

# ============================================================
# TAB 7: SIGNAL HISTORY — Backtest past signals
# ============================================================
with tab_hist:
    st.markdown("### 📊 Signal History")

    # Explanation box
    st.markdown("""<div style="background:#12121f;border-radius:10px;padding:12px 16px;margin-bottom:12px;border-left:4px solid #FFD700;">
<div style="color:#FFD700;font-size:13px;font-weight:bold;">So funktioniert der Backtest</div>
<div style="color:#aaa;font-size:12px;margin-top:4px;">
Der Scanner geht die <b style="color:white;">letzten 4h-Kerzen</b> durch und prüft: <em>Was hätte das RSI-Signal zu dem Zeitpunkt angezeigt?</em>
Dann wird gemessen, ob der Preis danach <b style="color:#00FF7F;">in die richtige Richtung</b> ging.<br>
<span style="color:#00FF7F;">● Bestätigt</span> = RSI + Confluence stimmen überein &nbsp;│&nbsp;
<span style="color:#FF6347;">● Konflikt</span> = RSI + Confluence widersprechen sich &nbsp;│&nbsp;
<span style="color:#00FF7F;">✅</span> = Signal war richtig &nbsp;│&nbsp; <span style="color:#FF6347;">❌</span> = Signal war falsch
</div></div>""", unsafe_allow_html=True)

    hc1, hc2, hc3 = st.columns([2, 1, 1])
    with hc1:
        signal_coins = df[df["signal"] != "WAIT"]["symbol"].tolist() if not df.empty else []
        all_coins = df["symbol"].tolist() if not df.empty else []
        hist_coin = st.selectbox("Coin:", all_coins,
                                 index=all_coins.index(signal_coins[0]) if signal_coins else 0,
                                 key="hist_coin")
    with hc2:
        hist_lookback = st.selectbox("Anzahl Signale:", [10, 15, 20, 30], index=2, key="hist_lb")
    with hc3:
        hist_forward = st.selectbox("Ergebnis nach:", ["12h (3 Kerzen)", "24h (6 Kerzen)", "2 Tage (12 Kerzen)"],
                                    index=0, key="hist_fw")
        fw_map = {"12h (3 Kerzen)": 3, "24h (6 Kerzen)": 6, "2 Tage (12 Kerzen)": 12}
        forward_candles = fw_map[hist_forward]

    if hist_coin:
        with st.spinner(f"⏳ Lade {hist_coin} Backtest..."):
            hist_df = fetch_klines_smart(hist_coin, "4h")

        if not hist_df.empty and len(hist_df) >= hist_lookback + forward_candles + 50:
            signals = backtest_signals(hist_df, lookback=hist_lookback, forward=forward_candles)

            if signals:
                total = len(signals)
                correct = sum(1 for s in signals if s["correct"])
                wrong = total - correct
                hit_rate = round(correct / total * 100, 1) if total > 0 else 0

                buy_signals = [s for s in signals if s["rsi_signal"] in ("BUY", "CTB")]
                sell_signals = [s for s in signals if s["rsi_signal"] in ("SELL", "CTS")]
                buy_correct = sum(1 for s in buy_signals if s["correct"])
                sell_correct = sum(1 for s in sell_signals if s["correct"])
                buy_rate = round(buy_correct / len(buy_signals) * 100, 1) if buy_signals else 0
                sell_rate = round(sell_correct / len(sell_signals) * 100, 1) if sell_signals else 0

                # Average PnL
                avg_pnl = round(sum(s["pnl_pct"] for s in signals) / total, 2) if total > 0 else 0
                buy_pnl = round(sum(s["pnl_pct"] for s in buy_signals) / len(buy_signals), 2) if buy_signals else 0
                sell_pnl = round(sum(-s["pnl_pct"] for s in sell_signals) / len(sell_signals), 2) if sell_signals else 0

                # Aligned vs Conflicting
                aligned = [s for s in signals if
                    (s["rsi_signal"] in ("BUY","CTB") and s["conf_score"] > 5) or
                    (s["rsi_signal"] in ("SELL","CTS") and s["conf_score"] < -5)]
                conflicting = [s for s in signals if
                    (s["rsi_signal"] in ("BUY","CTB") and s["conf_score"] < -5) or
                    (s["rsi_signal"] in ("SELL","CTS") and s["conf_score"] > 5)]
                neutral = [s for s in signals if s not in aligned and s not in conflicting]

                aligned_correct = sum(1 for s in aligned if s["correct"])
                conflict_correct = sum(1 for s in conflicting if s["correct"])
                aligned_rate = round(aligned_correct / len(aligned) * 100, 1) if aligned else 0
                conflict_rate = round(conflict_correct / len(conflicting) * 100, 1) if conflicting else 0

                hr_color = "#00FF7F" if hit_rate >= 60 else ("#FFD700" if hit_rate >= 45 else "#FF6347")
                pnl_color = "#00FF7F" if avg_pnl > 0 else ("#FF6347" if avg_pnl < 0 else "#888")

                # ── STATS DASHBOARD ──
                st.markdown(f"""<div style="background:#1a1a2e;border-radius:10px;padding:16px;margin:8px 0;">
<div style="display:flex;justify-content:space-around;flex-wrap:wrap;gap:12px;text-align:center;">
<div>
<div style="color:#888;font-size:11px;">TREFFERQUOTE</div>
<div style="color:{hr_color};font-size:28px;font-weight:bold;">{hit_rate}%</div>
<div style="color:#888;font-size:11px;">{correct} ✅ / {wrong} ❌ von {total}</div>
</div>
<div style="border-left:1px solid #333;padding-left:16px;">
<div style="color:#888;font-size:11px;">Ø RENDITE / Signal</div>
<div style="color:{pnl_color};font-size:28px;font-weight:bold;">{avg_pnl:+.2f}%</div>
<div style="color:#888;font-size:11px;">nach {forward_candles*4}h</div>
</div>
<div style="border-left:1px solid #333;padding-left:16px;">
<div style="color:#00FF7F;font-size:11px;">🟢 BUY/CTB</div>
<div style="color:#00FF7F;font-size:22px;font-weight:bold;">{buy_rate}%</div>
<div style="color:#888;font-size:11px;">{buy_correct}/{len(buy_signals)} · Ø {buy_pnl:+.2f}%</div>
</div>
<div style="border-left:1px solid #333;padding-left:16px;">
<div style="color:#FF6347;font-size:11px;">🔴 SELL/CTS</div>
<div style="color:#FF6347;font-size:22px;font-weight:bold;">{sell_rate}%</div>
<div style="color:#888;font-size:11px;">{sell_correct}/{len(sell_signals)} · Ø {sell_pnl:+.2f}%</div>
</div>
</div></div>""", unsafe_allow_html=True)

                # ── KEY INSIGHT: Aligned vs Conflicting ──
                if aligned or conflicting:
                    al_bar_w = aligned_rate
                    co_bar_w = conflict_rate
                    st.markdown(f"""<div style="background:#12121f;border-radius:10px;padding:14px;margin:8px 0;">
<div style="color:white;font-size:13px;font-weight:bold;margin-bottom:10px;">💡 Lohnt es sich auf Confluence zu achten?</div>
<div style="display:flex;gap:16px;flex-wrap:wrap;">
<div style="flex:1;min-width:200px;">
<div style="color:#00FF7F;font-size:12px;font-weight:bold;margin-bottom:4px;">✅ Bestätigt — RSI + Confluence gleiche Richtung</div>
<div style="background:#1a1a2e;border-radius:6px;height:28px;position:relative;overflow:hidden;">
<div style="background:#00FF7F33;height:100%;width:{al_bar_w}%;border-radius:6px;display:flex;align-items:center;padding-left:8px;">
<span style="color:#00FF7F;font-weight:bold;font-size:13px;">{aligned_rate}%</span>
<span style="color:#888;font-size:11px;margin-left:8px;">({aligned_correct}/{len(aligned)} Signale)</span>
</div></div>
</div>
<div style="flex:1;min-width:200px;">
<div style="color:#FF6347;font-size:12px;font-weight:bold;margin-bottom:4px;">⚠️ Konflikt — RSI + Confluence gegeneinander</div>
<div style="background:#1a1a2e;border-radius:6px;height:28px;position:relative;overflow:hidden;">
<div style="background:#FF634733;height:100%;width:{co_bar_w}%;border-radius:6px;display:flex;align-items:center;padding-left:8px;">
<span style="color:#FF6347;font-weight:bold;font-size:13px;">{conflict_rate}%</span>
<span style="color:#888;font-size:11px;margin-left:8px;">({conflict_correct}/{len(conflicting)} Signale)</span>
</div></div>
</div>
</div>
<div style="color:#888;font-size:11px;margin-top:8px;">
{'⚡ <b style=\"color:#00FF7F;\">Bestätigte Signale performen besser!</b> Warte auf Confluence-Bestätigung für bessere Trefferquote.' if aligned_rate > conflict_rate + 10 else ('⚠️ <b style=\"color:#FFD700;\">Kein klarer Vorteil</b> bei diesem Coin/Zeitraum — beide ähnlich gut.' if abs(aligned_rate - conflict_rate) <= 10 else '🤔 <b style=\"color:#FF6347;\">Konfligierende Signale performen hier besser</b> — RSI allein scheint zuverlässiger.')}
</div></div>""", unsafe_allow_html=True)

                # ── SIGNAL TIMELINE ──
                st.markdown(f"""<div style="color:white;font-size:13px;font-weight:bold;margin:16px 0 8px;">
📋 Letzte {total} Signale — neueste zuerst
</div>""", unsafe_allow_html=True)

                # Table header
                st.markdown("""<div style="display:flex;padding:4px 14px;color:#555;font-size:10px;font-weight:bold;text-transform:uppercase;letter-spacing:0.5px;">
<div style="flex:1.2;">Signal</div>
<div style="flex:1;">RSI · Confluence</div>
<div style="flex:1;">Einstieg → Ausstieg</div>
<div style="flex:0.6;text-align:right;">Ergebnis</div>
</div>""", unsafe_allow_html=True)

                for i, sig in enumerate(signals):
                    is_bull = sig["rsi_signal"] in ("BUY", "CTB")
                    sig_c = "#00FF7F" if is_bull else "#FF6347"
                    pnl = sig["pnl_pct"]
                    pnl_c = "#00FF7F" if pnl > 0 else ("#FF6347" if pnl < 0 else "#888")
                    result_icon = "✅" if sig["correct"] else "❌"

                    # Alignment status
                    if sig in aligned:
                        al_dot = '<span style="color:#00FF7F;">●</span>'
                        al_txt = "Bestätigt"
                    elif sig in conflicting:
                        al_dot = '<span style="color:#FF6347;">●</span>'
                        al_txt = "Konflikt"
                    else:
                        al_dot = '<span style="color:#888;">●</span>'
                        al_txt = "Neutral"

                    # Confluence score display
                    cs = sig["conf_score"]
                    cs_c = "#00FF7F" if cs > 10 else ("#FF6347" if cs < -10 else "#888")

                    # PnL bar (visual indicator)
                    bar_w = min(abs(pnl) * 10, 100)  # Scale: 1% = 10px width
                    bar_dir = "right" if pnl < 0 else "left"

                    st.markdown(f"""<div style="background:#1a1a2e;border-radius:8px;padding:8px 14px;margin:2px 0;display:flex;align-items:center;border-left:4px solid {sig_c};">
<div style="flex:1.2;">
<span style="color:{sig_c};font-weight:bold;font-size:14px;">{sig["rsi_signal"]}</span>
<span style="font-size:10px;margin-left:4px;">{al_dot} {al_txt}</span>
</div>
<div style="flex:1;">
<span style="color:white;font-size:12px;">RSI {sig["rsi_4h"]}</span>
<span style="color:{cs_c};font-size:11px;margin-left:4px;">Conf {cs:+d}</span>
</div>
<div style="flex:1;color:#888;font-size:11px;">
{fp(sig["entry_price"])} <span style="color:#555;">→</span> {fp(sig["exit_price"])}
</div>
<div style="flex:0.6;text-align:right;">
<span style="color:{pnl_c};font-size:14px;font-weight:bold;">{pnl:+.2f}%</span>
<span style="font-size:14px;margin-left:2px;">{result_icon}</span>
</div>
</div>""", unsafe_allow_html=True)

                st.caption(f"⏱️ Basiert auf {total} Signalen aus den letzten ~{total*4}–{(hist_lookback+forward_candles)*4}h · Ergebnis nach {forward_candles} Kerzen ({forward_candles*4}h)")
            else:
                st.info(f"🔍 Keine aktiven Signale (BUY/SELL/CTB/CTS) in den letzten Kerzen für {hist_coin}. Der RSI war vermutlich durchgehend im neutralen Bereich (42–58).")
        else:
            needed = hist_lookback + forward_candles + 50
            got = len(hist_df) if not hist_df.empty else 0
            st.warning(f"Nicht genug Kline-Daten für {hist_coin}. Benötigt: {needed} Kerzen, vorhanden: {got}.")

# ============================================================
# TAB 8: DETAIL — Complete Analysis Dashboard
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
        # NR Breakout
        det_nr_lv = c.get("nr_level", "")
        det_nr_bo = c.get("nr_breakout", "")
        if det_nr_lv:
            nr_w = {"NR4": 1, "NR7": 2, "NR10": 3}.get(det_nr_lv, 1)
            if det_nr_bo == "UP": bull_pts += nr_w; reasons_bull.append(f"{det_nr_lv}: Breakout UP bestätigt")
            elif det_nr_bo == "DOWN": bear_pts += nr_w; reasons_bear.append(f"{det_nr_lv}: Breakout DOWN bestätigt")
            elif r4 <= 42: bull_pts += max(1, nr_w - 1); reasons_bull.append(f"{det_nr_lv}: Kompression + RSI bullish")
            elif r4 >= 58: bear_pts += max(1, nr_w - 1); reasons_bear.append(f"{det_nr_lv}: Kompression + RSI bearish")
            else: reasons_bull.append(f"{det_nr_lv}: Kompression aktiv, Richtung offen")

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
| BTC Korrelation | {btc_corr.get('correlation',0):.2f} ({btc_corr.get('corr_label','N/A')}) | {'🔗' if abs(btc_corr.get('correlation',0))>0.5 else '🔓'} |
| NR Breakout | {c.get('nr_level','—') or '—'} {('↑ UP' if c.get('nr_breakout')=='UP' else ('↓ DOWN' if c.get('nr_breakout')=='DOWN' else '⏳')) if c.get('nr_level') else ''} (ratio: {c.get('nr_range_ratio',1.0):.3f}) | {'🔮' if c.get('nr_level') else '—'} |""")
        with st.expander("ℹ️ Was bedeutet die Indikator-Übersicht?"):
            st.markdown("""Diese Tabelle fasst alle Indikatoren zusammen. **Je mehr 🟢 desto bullischer, je mehr 🔴 desto bearischer.**

**MACD:** Zeigt Momentum-Änderungen. Bullish = Kaufdruck nimmt zu. Das Histogramm zeigt die Stärke.

**Stoch RSI:** Kombination aus Stochastik und RSI. K < 20 = überverkauft (kaufen), K > 80 = überkauft (verkaufen). K kreuzt D nach oben = Kaufsignal.

**Volume:** Vergleicht aktuelles Volumen mit dem 20-Perioden-Durchschnitt. Hohes Volumen bestätigt Trends. OBV (On-Balance Volume) steigend = Akkumulation.

**Ideales Kauf-Setup:** RSI < 42 + EMA bullish + MACD bullish + Bollinger < 20% + Volumen steigend → starkes Kaufsignal mit hoher Erfolgswahrscheinlichkeit.

**Ideales Verkauf-Setup:** RSI > 58 + EMA bearish + MACD bearish + Bollinger > 80% + Volumen steigend → starkes Verkaufsignal.""")

        st.caption("⚠️ DYOR — Dies ist keine Finanzberatung. Alle Berechnungen basieren auf historischen Daten.")

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#555;font-size:11px;'>🧙‍♂️ Merlin | {len(coins_to_scan)}×{len(tf_to_scan)}TF | {ex_display} | BUY≤42 SELL≥58 | DYOR!</div>",unsafe_allow_html=True)
