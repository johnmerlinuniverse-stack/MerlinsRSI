# 🧙‍♂️ Merlin Crypto Scanner

A comprehensive crypto market RSI scanner & alert dashboard — inspired by CryptoWaves.app but with advanced features.

## Features

### 🔥 RSI Heatmap
- Interactive scatter plot showing all coins positioned by their RSI value
- Color-coded zones (Overbought/Oversold/Neutral)
- Bubble size based on 24h volume
- Quick-view of most overbought/oversold coins

### 📊 Market Overview
- Sortable table with all coins
- Multi-timeframe RSI (1h, 4h, 1D, 1W)
- MACD trend, Volume analysis
- Confluence-based Buy/Sell signals

### 🚨 24h Alerts
- Real-time signal detection
- Filterable by Buy/Sell/Strong signals
- Detailed reasoning for each alert
- Telegram notification integration

### 🎯 Confluence Scanner
- Multi-indicator scoring system (-100 to +100)
- Combines: RSI, MACD, Volume/OBV, Smart Money Concepts
- Visual score distribution
- Side-by-side Buy vs Sell signals

### 🔍 Coin Detail
- RSI gauge charts for all timeframes
- MACD, Stochastic RSI details
- Volume ratio and OBV trend
- Smart Money: Order Blocks, Fair Value Gaps, Market Structure, Break of Structure

### 📱 Telegram Alerts
- Instant notifications for strong signals
- Summary reports on demand
- Configurable alert threshold

---

## Data Sources

| Source | Data | Cost |
|--------|------|------|
| **Binance API** | OHLCV klines, prices, volume, 24h tickers | Free, no auth |
| **CoinGecko API** | Market cap, rank, metadata | Free (30 req/min) |

---

## Setup

### 1. Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/merlin-crypto-scanner.git
cd merlin-crypto-scanner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

### 2. Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select the repo and `app.py` as the main file
5. Click "Deploy"

### 3. Telegram Bot Setup (Optional)

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow instructions
3. Copy the **Bot Token**
4. Get your **Chat ID**:
   - Send a message to your bot
   - Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Find your `chat.id` in the response
5. Enter both in the sidebar of the dashboard

---

## Project Structure

```
merlin-crypto-scanner/
├── app.py              # Main Streamlit dashboard
├── config.py           # Settings, coin list, thresholds
├── data_fetcher.py     # Binance + CoinGecko API integration
├── indicators.py       # RSI, MACD, Volume, Smart Money Concepts
├── alerts.py           # Telegram alert system
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit theme config
└── README.md           # This file
```

---

## Customization

### Add/Remove Coins
Edit `TOP_COINS` list in `config.py`.

### Change RSI Thresholds
Edit `RSI_OVERBOUGHT`, `RSI_OVERSOLD` etc. in `config.py`.

### Adjust Confluence Scoring
Modify weights in `generate_confluence_signal()` in `indicators.py`.

### Add New Indicators
1. Add calculation function in `indicators.py`
2. Call it in `scan_all_coins()` in `app.py`
3. Display in the appropriate tab

---

## Roadmap / Future Features

- [ ] Email alerts (SMTP integration)
- [ ] Historical RSI alert log with charts
- [ ] Portfolio tracking with P&L
- [ ] Backtesting module
- [ ] Whale alert integration (on-chain data)
- [ ] Liquidation heatmap
- [ ] Elliott Wave detection
- [ ] Multi-exchange support (Bybit, OKX)
- [ ] Mobile-optimized PWA
- [ ] Database persistence (SQLite/Supabase)

---

## Disclaimer

This tool is for **educational and informational purposes only**. 
It is **not financial advice**. Always do your own research (DYOR) 
before making any trading decisions.

---

Built with ❤️ using Python, Streamlit, Binance API & CoinGecko API
