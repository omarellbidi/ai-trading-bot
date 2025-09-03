"""
AI Trading Bot Configuration
"""

import os
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class APIConfig:
    """
    External API configuration for trading and data services.
    
    Contains credentials and endpoints for:
    - Alpaca Trading API (paper trading)
    - News API for sentiment analysis
    - Yahoo Finance for market data
    """
    
    # Alpaca Paper Trading API Configuration
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "your_alpaca_api_key_here")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "your_alpaca_secret_key_here")
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    
    # News API for sentiment analysis
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "your_news_api_key_here")
    
    # Yahoo Finance API for market data (free tier)
    YAHOO_FINANCE_BASE_URL: str = "https://query1.finance.yahoo.com"


@dataclass
class TradingConfig:
    """
    Trading strategy and risk management configuration.
    
    Defines position sizing, risk limits, and trading hours.
    All percentage values are in decimal format (0.05 = 5%).
    """
    
    # Risk Management Limits
    MAX_POSITION_SIZE: float = 0.05      # Maximum 5% of portfolio per position
    MAX_DAILY_LOSS: float = 0.02         # Maximum 2% daily loss limit
    MAX_DRAWDOWN: float = 0.10           # Maximum 10% portfolio drawdown
    STOP_LOSS_PCT: float = 0.02          # 2% stop-loss on each position
    TAKE_PROFIT_PCT: float = 0.04        # 4% take-profit target
    MAX_TOTAL_EXPOSURE: float = 0.20     # Maximum 20% total market exposure
    
    # Portfolio Configuration
    STARTING_CASH: float = 100000.0      # Starting portfolio value
    MIN_CASH_BALANCE: float = 10000.0    # Minimum cash reserve
    
    # Market Hours (EST)
    MARKET_OPEN: str = "09:30"           # Market open time
    MARKET_CLOSE: str = "16:00"          # Market close time
    TIMEZONE: str = "America/New_York"   # Trading timezone


@dataclass
class AssetConfig:
    """
    Tradeable assets configuration.
    
    Defines the universe of assets the bot can trade across different categories:
    - Cryptocurrency (24/7 trading)
    - Commodity ETFs
    - Technology stocks
    - Market index ETFs
    """
    
    # Cryptocurrency pairs (24/7 trading when available)
    CRYPTO_SYMBOLS: List[str] = None
    
    # Commodity and precious metal ETFs
    COMMODITY_ETFS: List[str] = None
    
    # High-growth technology stocks
    TECH_STOCKS: List[str] = None
    
    # Broad market index ETFs
    MARKET_ETFS: List[str] = None
    
    def __post_init__(self):
        """Initialize default asset lists if not provided."""
        if self.CRYPTO_SYMBOLS is None:
            self.CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD"]
        
        if self.COMMODITY_ETFS is None:
            self.COMMODITY_ETFS = ["GLD", "IAU", "USO", "XLE", "SLV", "UNG"]
        
        if self.TECH_STOCKS is None:
            self.TECH_STOCKS = ["NVDA", "TSLA", "GOOGL", "AAPL", "MSFT", "AMD"]
        
        if self.MARKET_ETFS is None:
            self.MARKET_ETFS = ["SPY", "QQQ", "IWM", "VTI"]


@dataclass
class MLConfig:
    """
    Machine Learning model configuration.
    
    Parameters for:
    - Model architecture (LSTM sequence length, prediction horizon)
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Training parameters (epochs, batch size, learning rate)
    - Feature engineering settings
    """
    
    # Model Architecture
    SEQUENCE_LENGTH: int = 60            # LSTM lookback period (60 time steps)
    PREDICTION_HORIZON: int = 5          # Predict 5 periods into the future
    TRAIN_TEST_SPLIT: float = 0.8        # 80% training, 20% testing
    VALIDATION_SPLIT: float = 0.2        # 20% validation during training
    
    # Technical Indicator Parameters
    RSI_PERIOD: int = 14                 # RSI calculation period
    MACD_FAST: int = 12                  # MACD fast EMA period
    MACD_SLOW: int = 26                  # MACD slow EMA period
    MACD_SIGNAL: int = 9                 # MACD signal line period
    BB_PERIOD: int = 20                  # Bollinger Bands period
    BB_STD: float = 2.0                  # Bollinger Bands standard deviation
    
    # Feature Engineering
    FEATURE_SCALING: str = "minmax"      # Feature scaling method
    TOP_FEATURES: int = 30               # Select top N features (from 65 total)
    
    # Model Training Parameters
    EPOCHS: int = 100                    # Maximum training epochs
    BATCH_SIZE: int = 32                 # Training batch size
    LEARNING_RATE: float = 0.001         # Initial learning rate
    EARLY_STOPPING_PATIENCE: int = 10    # Early stopping patience


@dataclass
class DataConfig:
    """
    Data collection and processing configuration.
    
    Defines update frequencies, data retention policies, and sentiment analysis settings.
    All intervals are in seconds.
    """
    
    # Data Update Frequencies
    PRICE_UPDATE_INTERVAL: int = 60      # Price data update (1 minute)
    NEWS_UPDATE_INTERVAL: int = 300      # News data update (5 minutes)  
    CRYPTO_UPDATE_INTERVAL: int = 30     # Crypto data update (30 seconds)
    RISK_CHECK_INTERVAL: int = 30        # Risk monitoring (30 seconds)
    
    # Data Retention Policies
    MAX_PRICE_HISTORY_DAYS: int = 730    # Keep 2 years of price history
    MAX_NEWS_HISTORY_DAYS: int = 30      # Keep 30 days of news data
    
    # Sentiment Analysis Configuration
    SENTIMENT_MODEL: str = "vader"       # Sentiment analysis model
    NEWS_RELEVANCE_THRESHOLD: float = 0.5 # Minimum relevance score


# ============================================================================
# Global Configuration Instances
# ============================================================================

# Initialize configuration objects
api_config = APIConfig()
trading_config = TradingConfig()
asset_config = AssetConfig()
ml_config = MLConfig()
data_config = DataConfig()


def get_all_symbols() -> List[str]:
    """
    Get complete list of all tradeable symbols.
    
    Returns:
        List[str]: Combined list of all symbols from all asset categories
    """
    return (
        asset_config.CRYPTO_SYMBOLS + 
        asset_config.COMMODITY_ETFS + 
        asset_config.TECH_STOCKS + 
        asset_config.MARKET_ETFS
    )


def get_symbol_category(symbol: str) -> str:
    """
    Determine the category of a given symbol.
    
    Args:
        symbol: The trading symbol to categorize
        
    Returns:
        str: The category ('crypto', 'commodity', 'tech', 'market', 'unknown')
    """
    if symbol in asset_config.CRYPTO_SYMBOLS:
        return 'crypto'
    elif symbol in asset_config.COMMODITY_ETFS:
        return 'commodity'
    elif symbol in asset_config.TECH_STOCKS:
        return 'tech'
    elif symbol in asset_config.MARKET_ETFS:
        return 'market'
    else:
        return 'unknown'


# ============================================================================
# Logging Configuration
# ============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': 'logs/trading_bot.log',
            'mode': 'a',
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}


# ============================================================================
# Environment Validation
# ============================================================================

def validate_configuration() -> bool:
    """
    Validate that all required configuration is present and valid.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        # Validate API keys are present
        assert api_config.ALPACA_API_KEY, "Alpaca API key is required"
        assert api_config.ALPACA_SECRET_KEY, "Alpaca secret key is required"
        assert api_config.NEWS_API_KEY, "News API key is required"
        
        # Validate risk parameters
        assert 0 < trading_config.MAX_POSITION_SIZE <= 1, "Invalid position size limit"
        assert 0 < trading_config.MAX_DAILY_LOSS <= 1, "Invalid daily loss limit"
        assert 0 < trading_config.STOP_LOSS_PCT <= 1, "Invalid stop loss percentage"
        
        # Validate asset lists are not empty
        assert len(get_all_symbols()) > 0, "No trading symbols configured"
        
        return True
        
    except AssertionError as e:
        print(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during configuration validation: {e}")
        return False