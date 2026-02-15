"""
Alert system for Telegram notifications.
Tracks alert history to avoid spam.
"""

import time
import requests
import streamlit as st
from datetime import datetime
from config import ALERT_COOLDOWN_MINUTES


# ============================================================
# ALERT HISTORY (in-memory, resets on app restart)
# ============================================================

def get_alert_history() -> dict:
    """Get alert history from session state."""
    if "alert_history" not in st.session_state:
        st.session_state.alert_history = {}
    return st.session_state.alert_history


def can_send_alert(symbol: str, signal_type: str) -> bool:
    """Check if an alert can be sent (cooldown check)."""
    history = get_alert_history()
    key = f"{symbol}_{signal_type}"
    
    if key not in history:
        return True
    
    elapsed = (time.time() - history[key]) / 60
    return elapsed >= ALERT_COOLDOWN_MINUTES


def record_alert(symbol: str, signal_type: str):
    """Record that an alert was sent."""
    history = get_alert_history()
    key = f"{symbol}_{signal_type}"
    history[key] = time.time()


# ============================================================
# TELEGRAM BOT
# ============================================================

def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> bool:
    """Send a Telegram message."""
    if not bot_token or not chat_id:
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def format_alert_message(coin_data: dict) -> str:
    """Format a coin alert as HTML for Telegram."""
    symbol = coin_data.get("symbol", "???")
    price = coin_data.get("price", 0)
    rsi_4h = coin_data.get("rsi_4h", 50)
    rsi_1d = coin_data.get("rsi_1d", 50)
    signal = coin_data.get("recommendation", "WAIT")
    score = coin_data.get("score", 0)
    change_24h = coin_data.get("change_24h", 0)
    
    # Signal emoji
    emoji_map = {
        "STRONG BUY": "🟢🟢",
        "BUY": "🟢",
        "LEAN BUY": "🟡↗️",
        "WAIT": "⏳",
        "LEAN SELL": "🟡↘️",
        "SELL": "🔴",
        "STRONG SELL": "🔴🔴",
    }
    emoji = emoji_map.get(signal, "⏳")
    
    msg = (
        f"{emoji} <b>{symbol}/USDT</b> — {signal}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"💰 Price: <b>${price:,.4f}</b>\n"
        f"📊 RSI 4h: <b>{rsi_4h}</b> | RSI 1D: <b>{rsi_1d}</b>\n"
        f"📈 24h: <b>{change_24h:+.2f}%</b>\n"
        f"🎯 Confluence Score: <b>{score}/100</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    return msg


def format_summary_alert(alerts: list) -> str:
    """Format a summary of all active alerts for Telegram."""
    if not alerts:
        return "🧙‍♂️ Merlin Scanner: No active signals at this time."
    
    buy_alerts = [a for a in alerts if "BUY" in a.get("recommendation", "")]
    sell_alerts = [a for a in alerts if "SELL" in a.get("recommendation", "")]
    
    msg = "🧙‍♂️ <b>Merlin Crypto Scanner — Alert Summary</b>\n"
    msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
    
    if buy_alerts:
        msg += "🟢 <b>BUY SIGNALS:</b>\n"
        for a in sorted(buy_alerts, key=lambda x: x.get("score", 0), reverse=True)[:10]:
            msg += f"  • {a['symbol']} — RSI 4h: {a['rsi_4h']} | 1D: {a['rsi_1d']} | Score: {a['score']}\n"
        msg += "\n"
    
    if sell_alerts:
        msg += "🔴 <b>SELL SIGNALS:</b>\n"
        for a in sorted(sell_alerts, key=lambda x: x.get("score", 0))[:10]:
            msg += f"  • {a['symbol']} — RSI 4h: {a['rsi_4h']} | 1D: {a['rsi_1d']} | Score: {a['score']}\n"
    
    msg += f"\n📊 Total: {len(buy_alerts)} buy | {len(sell_alerts)} sell signals"
    
    return msg


# ============================================================
# ALERT TRIGGER LOGIC
# ============================================================

def check_and_send_alerts(
    all_coin_data: list,
    bot_token: str,
    chat_id: str,
    min_score: int = 30,
    send_summary: bool = False,
) -> list:
    """
    Check all coin data for alert conditions and send Telegram notifications.
    Returns list of triggered alerts.
    """
    triggered = []
    
    for coin in all_coin_data:
        symbol = coin.get("symbol", "")
        score = coin.get("score", 0)
        recommendation = coin.get("recommendation", "WAIT")
        
        if abs(score) < min_score:
            continue
        
        if recommendation == "WAIT":
            continue
        
        signal_type = "BUY" if score > 0 else "SELL"
        
        if can_send_alert(symbol, signal_type):
            triggered.append(coin)
            
            if bot_token and chat_id:
                msg = format_alert_message(coin)
                success = send_telegram_alert(bot_token, chat_id, msg)
                if success:
                    record_alert(symbol, signal_type)
    
    # Send summary if requested
    if send_summary and triggered and bot_token and chat_id:
        summary = format_summary_alert(triggered)
        send_telegram_alert(bot_token, chat_id, summary)
    
    return triggered
