# AI Trading Bot

An automated trading system that combines machine learning, sentiment analysis, and technical indicators to make trading decisions. The bot uses multiple data sources and advanced risk management to execute trades safely.

## Features

- **Multi-Signal Trading Strategy**: Combines ML predictions, sentiment analysis, and technical indicators
- **Risk Management**: Comprehensive risk controls with position sizing and stop-loss mechanisms
- **Real-time Data**: Market data collection from Yahoo Finance and news from NewsAPI
- **Machine Learning**: LSTM, Random Forest, and other ML models for price prediction
- **Sentiment Analysis**: News sentiment analysis using VADER and TextBlob
- **Paper Trading**: Safe testing with Alpaca's paper trading API

## Architecture

```
├── main.py              # Entry point and CLI
├── config/              # Configuration settings
├── engines/             # Core trading engines
│   ├── trading_engine.py    # Main trading orchestrator
│   ├── risk_manager.py      # Risk management system
│   ├── sentiment_analyzer.py # News sentiment analysis
│   └── market_regime.py     # Market condition detection
├── models/              # Machine learning models
│   ├── price_predictor.py   # ML price prediction
│   ├── deep_models.py       # Deep learning models
│   └── training_methods.py  # Advanced training techniques
├── data/                # Data collection and storage
├── utils/               # Utilities and indicators
└── tests/               # Test files
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-trading-bot.git
   cd ai-trading-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

4. **Get API Keys**
   - **Alpaca API**: Sign up at [Alpaca Markets](https://alpaca.markets/) for paper trading
   - **News API**: Get a free key from [NewsAPI](https://newsapi.org/) for sentiment analysis

## API Integration

This bot is **designed for**:
- **Alpaca Markets API**: Paper trading and brokerage integration
- **NewsAPI**: Financial news collection for sentiment analysis  
- **Yahoo Finance**: Free market data (no API key required)

**Using Other Brokers/APIs**: The system can be adapted for other trading platforms (Interactive Brokers, TD Ameritrade, etc.) by modifying the `AlpacaTrader` class in `engines/trading_engine.py`. The modular design allows easy integration with different data sources and brokers with minimal code changes.

## Quick Start

1. **Test the system**
   ```bash
   python main.py test
   ```

2. **Collect initial data**
   ```bash
   python main.py collect
   ```

3. **Train ML models**
   ```bash
   python main.py train
   ```

4. **Update sentiment analysis**
   ```bash
   python main.py sentiment
   ```

5. **Start trading (paper trading)**
   ```bash
   python main.py trade
   ```

## Configuration

Edit `config/settings.py` to customize:
- Trading symbols and asset categories
- Risk management parameters
- ML model configuration
- Data collection settings

## Risk Management

The bot includes multiple safety mechanisms:
- Maximum position size limits (5% of portfolio)
- Daily loss limits (2% of portfolio)
- Stop-loss orders (2% per position)
- Portfolio drawdown monitoring (10% max)
- Real-time risk assessment

## Supported Assets

- **Technology Stocks**: AAPL, MSFT, GOOGL, NVDA, TSLA, AMD
- **Market ETFs**: SPY, QQQ, IWM, VTI
- **Commodity ETFs**: GLD, IAU, USO, XLE, SLV, UNG
- **Cryptocurrency**: BTCUSD, ETHUSD 

## Machine Learning Models

- **LSTM Neural Networks**: For sequential pattern recognition
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Advanced boosting techniques
- **Support Vector Machines**: Non-linear classification
- **Logistic Regression**: Baseline linear model

### Model Performance & Predictions

The system generates **binary classification predictions** (Buy/Hold vs Sell) using:

- **Feature Engineering**: 66 technical indicators and market features
- **Training Data**: 2+ years of historical price, volume, and sentiment data (~430 samples per stock)
- **Data Sources**: Real market data from Yahoo Finance API
- **Validation**: Train/test splits with time-series validation to prevent data leakage
- **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score

**Performance Results**:
- **Training Accuracy**: 85-95% on major stocks (AAPL, TSLA, NVDA, GOOGL)
- **Out-of-Sample Accuracy**: 65-75% (realistic for financial markets)
- **Feature Selection**: Automated reduction from 66 to 30 most important features
- **Ensemble Methods**: Combines Random Forest, Gradient Boosting, and Logistic Regression

**Testing Data**:
- **Real Market Data**: Uses actual stock prices from Yahoo Finance (2022-2024)
- **Assets Tested**: Technology stocks (AAPL, MSFT, GOOGL, NVDA, TSLA, AMD)
- **Market ETFs**: SPY, QQQ, IWM, VTI for broader market validation
- **Commodity ETFs**: GLD, SLV, USO for diversified asset testing

**Important Notes**:
- Training accuracy (85-95%) represents in-sample performance
- Out-of-sample accuracy (65-75%) is more realistic for live trading
- Performance varies significantly across different market regimes
- Models are designed for retraining on new data
- Past performance does not guarantee future results

## Technical Indicators

- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, A/D Line, Volume-Price Trend

## Command Line Interface

```bash
# Available commands
python main.py collect    # Collect market data and news
python main.py train      # Train machine learning models
python main.py sentiment  # Update sentiment analysis
python main.py trade      # Start live trading
python main.py test       # Run system tests

# Options
python main.py collect --symbols AAPL TSLA  # Specific symbols
python main.py train --verbose              # Verbose logging
```

## Development

### Running Tests
```bash
python main.py test
```

### Adding New Models
1. Create model class in `models/`
2. Implement `train()` and `predict()` methods
3. Add to `ModelManager` in `price_predictor.py`

### Adding New Indicators
1. Add indicator function to `utils/technical_indicators.py`
2. Include in feature engineering pipeline

## Disclaimer

**⚠️ Important**: This is educational software for learning about algorithmic trading. 

- **Paper Trading Only**: Always test with paper trading first
- **No Financial Advice**: This bot does not provide financial advice
- **Use at Your Own Risk**: Trading involves significant financial risk
- **Past Performance**: Past performance does not guarantee future results

## License

MIT License - see LICENSE file for details
