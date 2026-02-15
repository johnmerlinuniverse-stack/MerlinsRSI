"""
Configuration for Merlin Crypto Scanner
"""

# ============================================================
# API SETTINGS
# ============================================================
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# ============================================================
# RSI SETTINGS
# ============================================================
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_STRONG_OVERBOUGHT = 80
RSI_STRONG_OVERSOLD = 20

# Timeframes for klines
TIMEFRAMES = {
    "1h": "1h",
    "4h": "4h",
    "1D": "1d",
    "1W": "1w",
}

# Number of candles to fetch
KLINE_LIMIT = 100

# ============================================================
# MACD SETTINGS
# ============================================================
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ============================================================
# SIGNAL THRESHOLDS
# ============================================================
SIGNAL_RULES = {
    "STRONG_BUY": {"rsi_4h_max": 30, "rsi_1d_max": 35},
    "BUY": {"rsi_4h_max": 40, "rsi_1d_max": 45},
    "SELL": {"rsi_4h_min": 70, "rsi_1d_min": 65},
    "STRONG_SELL": {"rsi_4h_min": 80, "rsi_1d_min": 75},
}

# ============================================================
# COIN LISTS
# ============================================================

# CryptoWaves-style curated list (107 coins)
# Pattern: Top ~50 Market Cap + popular high-volume altcoins
# Includes: Major L1s, DeFi, Meme coins, AI tokens, Gaming
CRYPTOWAVES_COINS = [
    # Top 10 by Market Cap
    "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK", "AVAX",
    # Top 11-30
    "XLM", "HBAR", "DOT", "BCH", "LTC", "SHIB", "UNI", "TON", "NEAR",
    "AAVE", "ZEC", "SUI",
    # Top 31-60
    "PEPE", "POL", "WLD", "ATOM", "ENA", "ONDO", "QNT", "RENDER",
    "FIL", "VET", "ETC", "TAO", "INJ", "STX", "ICP",
    # Top 61-100
    "ARB", "OP", "FET", "SEI", "JUP", "BONK", "FLOKI", "IMX",
    "APT", "ALGO", "GRT", "THETA", "SAND", "MANA", "AXS",
    "GALA", "CRV", "DASH", "LDO", "JASMY", "IOTA",
    "PENGU", "VIRTUAL", "PUMP", "MORPHO",
    # Popular altcoins beyond Top 100 (high volume / trending)
    "PYTH", "SUN", "TIA", "CFX", "ENS", "WIF", "COMP", "DEXE",
    "LUNC", "STRK", "PENDLE", "ETHFI", "CHZ", "XTZ", "BAT",
    "ZRO", "NEXO", "CAKE", "TRUMP", "PAXG", "GNO", "CVX",
    "ZK", "GLM", "KITE", "AWE", "SKY",
    # Stablecoins (for reference)
    "USDC", "USDE",
]

# Full list for "Top 150+" mode
TOP_COINS_EXTENDED = [
    "BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "TRX", "DOT", "MATIC",
    "LTC", "SHIB", "AVAX", "LINK", "UNI", "ATOM", "XLM", "ETC", "HBAR", "FIL",
    "ICP", "APT", "NEAR", "ARB", "VET", "OP", "GRT", "ALGO", "STX", "INJ",
    "IMX", "RNDR", "FTM", "SAND", "MANA", "THETA", "AXS", "AAVE", "EOS", "FLOW",
    "XTZ", "NEO", "CHZ", "GALA", "KAVA", "CRV", "COMP", "LRC", "DASH", "ZEC",
    "BAT", "ENJ", "1INCH", "SUSHI", "CELO", "SNX", "DYDX", "LDO", "FXS", "GMX",
    "PENDLE", "SSV", "RPL", "MASK", "STORJ", "ANKR", "SKL", "OCEAN", "ICX", "ZIL",
    "IOTA", "ONT", "SXP", "RSR", "CELR", "MTL", "REN", "BAND", "KNC", "BLZ",
    "COTI", "DENT", "HOT", "WRX", "ROSE", "AUDIO", "LINA", "TWT", "SFP", "LEVER",
    "PEOPLE", "JASMY", "MAGIC", "HOOK", "HIGH", "AMB", "PERP", "AGLD", "RDNT", "WOO",
    "JOE", "ACH", "LOOM", "KEY", "BAKE", "BURGER", "NULS", "VITE", "QNT", "MKR",
    "RUNE", "EGLD", "FET", "AGIX", "WLD", "SEI", "SUI", "PEPE", "FLOKI", "BONK",
    "WIF", "JUP", "PYTH", "TIA", "STRK", "MANTA", "DYM", "PIXEL", "PORTAL", "AEVO",
    "ENA", "W", "ONDO", "TAO", "RENDER", "AR", "KAS", "ORDI", "BOME", "NOT",
    "IO", "ZRO", "LISTA", "BB", "REZ", "ETHFI", "SAGA", "TNSR", "OMNI", "ALT",
    "POLYX", "BCH", "TRUMP", "DEXE", "VIRTUAL", "MORPHO", "PENGU", "MOVE",
    "TON", "SUN", "CFX", "ENS", "LUNC", "PAXG", "PUMP", "NEXO", "CAKE",
    "GNO", "CVX", "ZK", "GLM", "KITE", "AWE", "SKY",
]

# Default list (used when no option selected)
TOP_COINS = CRYPTOWAVES_COINS

# CoinGecko ID mapping
COINGECKO_ID_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin", "XRP": "ripple",
    "ADA": "cardano", "DOGE": "dogecoin", "SOL": "solana", "TRX": "tron",
    "DOT": "polkadot", "MATIC": "matic-network", "LTC": "litecoin",
    "SHIB": "shiba-inu", "AVAX": "avalanche-2", "LINK": "chainlink",
    "UNI": "uniswap", "ATOM": "cosmos", "XLM": "stellar", "ETC": "ethereum-classic",
    "HBAR": "hedera-hashgraph", "FIL": "filecoin", "ICP": "internet-computer",
    "APT": "aptos", "NEAR": "near", "ARB": "arbitrum", "VET": "vechain",
    "OP": "optimism", "GRT": "the-graph", "ALGO": "algorand",
    "RENDER": "render-token", "RNDR": "render-token", "FTM": "fantom",
    "SAND": "the-sandbox", "MANA": "decentraland", "THETA": "theta-token",
    "AXS": "axie-infinity", "AAVE": "aave", "EOS": "eos", "FLOW": "flow",
    "XTZ": "tezos", "NEO": "neo", "CHZ": "chiliz", "GALA": "gala",
    "KAVA": "kava", "CRV": "curve-dao-token", "COMP": "compound-governance-token",
    "LRC": "loopring", "DASH": "dash", "ZEC": "zcash", "BAT": "basic-attention-token",
    "ENJ": "enjincoin", "1INCH": "1inch", "SUSHI": "sushi", "CELO": "celo",
    "SNX": "havven", "DYDX": "dydx-chain", "LDO": "lido-dao", "FXS": "frax-share",
    "GMX": "gmx", "PENDLE": "pendle", "MASK": "mask-network", "STORJ": "storj",
    "ANKR": "ankr", "SKL": "skale", "OCEAN": "ocean-protocol", "ICX": "icon",
    "ZIL": "zilliqa", "IOTA": "iota", "ONT": "ontology", "RSR": "reserve-rights-token",
    "BAND": "band-protocol", "KNC": "kyber-network-crystal", "COTI": "coti",
    "DENT": "dent", "HOT": "holotoken", "ROSE": "oasis-network",
    "AUDIO": "audius", "JASMY": "jasmycoin", "MAGIC": "magic",
    "PERP": "perpetual-protocol", "WOO": "woo-network", "QNT": "quant-network",
    "MKR": "maker", "RUNE": "thorchain", "EGLD": "elrond-erd-2",
    "FET": "fetch-ai", "AGIX": "singularitynet", "WLD": "worldcoin-wld",
    "SEI": "sei-network", "SUI": "sui", "PEPE": "pepe", "FLOKI": "floki",
    "BONK": "bonk", "WIF": "dogwifcoin", "JUP": "jupiter-exchange-solana",
    "PYTH": "pyth-network", "TIA": "celestia", "STRK": "starknet",
    "ENA": "ethena", "ONDO": "ondo-finance", "TAO": "bittensor",
    "AR": "arweave", "ORDI": "ordinals", "NOT": "notcoin",
    "BCH": "bitcoin-cash", "TRUMP": "official-trump", "VIRTUAL": "virtual-protocol",
    "PENGU": "pudgy-penguins", "STX": "blockstack", "INJ": "injective-protocol",
    "IMX": "immutable-x", "TON": "the-open-network", "SUN": "sun-token",
    "CFX": "conflux-token", "ENS": "ethereum-name-service", "LUNC": "terra-luna",
    "PAXG": "pax-gold", "PUMP": "pump-fun", "NEXO": "nexo",
    "CAKE": "pancakeswap-token", "GNO": "gnosis", "CVX": "convex-finance",
    "MORPHO": "morpho", "POL": "polygon-ecosystem-token",
    "USDC": "usd-coin", "USDE": "ethena-usde",
}

# ============================================================
# ALERT SETTINGS
# ============================================================
ALERT_COOLDOWN_MINUTES = 60

# ============================================================
# UI SETTINGS
# ============================================================
PAGE_TITLE = "🧙‍♂️ Merlin Crypto Scanner"
PAGE_ICON = "🧙‍♂️"
REFRESH_INTERVAL = 300

# Color scheme
COLORS = {
    "strong_buy": "#00FF7F",
    "buy": "#32CD32",
    "neutral": "#FFD700",
    "sell": "#FF6347",
    "strong_sell": "#FF0000",
    "bg_dark": "#0E1117",
    "text": "#FAFAFA",
    "overbought_zone": "rgba(255, 0, 0, 0.15)",
    "oversold_zone": "rgba(0, 255, 0, 0.15)",
}
