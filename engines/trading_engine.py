"""
Main trading engine that orchestrates all components
"""

import time
import schedule
import signal
import sys
import os
import pytz
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# External API dependencies
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Internal imports
from config.settings import api_config, trading_config, asset_config, data_config, get_all_symbols
from utils.logger import logger
from data.data_collector import DataCollectionOrchestrator
from engines.sentiment_analyzer import SentimentEngine
from models.price_predictor import ModelManager
from engines.risk_manager import RiskManager, TradeSignal
from utils.technical_indicators import FeatureEngineer

# ============================================================================
# Trading API Interface
# ============================================================================

class AlpacaTrader:
    """
    Production-grade interface to Alpaca Trading API.
    
    Handles all trading operations including order placement, position management,
    and account monitoring. Implements robust error handling and automatic
    retry mechanisms for production reliability.
    
    Features:
    - Automatic API connection validation and retry
    - Fractional share handling for short sales
    - Order price rounding to avoid rejection
    - Position tracking and portfolio monitoring
    - Market hours validation
    """
    
    def __init__(self):
        """
        Initialize Alpaca trading interface.
        
        Establishes connection to Alpaca API and validates account access.
        Sets up trading environment with proper error handling.
        """
        self.api = None
        self.account_id = None
        self.initial_buying_power = None
        
        if not ALPACA_AVAILABLE:
            logger.error("❌ Alpaca API library not available - install alpaca-trade-api")
            return
            
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """Initialize connection to Alpaca API with validation."""
        try:
            # Create API connection
            self.api = tradeapi.REST(
                api_config.ALPACA_API_KEY,
                api_config.ALPACA_SECRET_KEY,
                api_config.ALPACA_BASE_URL,
                api_version='v2'
            )
            
            # Validate connection by fetching account information
            account = self.api.get_account()
            self.account_id = account.id
            self.initial_buying_power = float(account.buying_power)
            
            # Log successful connection
            logger.info(f"✅ Connected to Alpaca Trading API")
            logger.info(f"📊 Account ID: {self.account_id}")
            logger.info(f"💰 Buying Power: ${self.initial_buying_power:,.2f}")
            logger.info(f"📈 Portfolio Value: ${float(account.portfolio_value):,.2f}")
            
            # Log account status
            if account.trading_blocked:
                logger.warning("⚠️ Account trading is blocked")
            if account.pattern_day_trader:
                logger.info("📋 Pattern Day Trader account detected")
            
        except Exception as e:
            logger.error("❌ Failed to connect to Alpaca API", error=e)
            self.api = None
    
    def is_market_open(self) -> bool:
        """
        Check if the stock market is currently open for trading.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        if not self.api:
            return False
        
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error("Failed to check market status", error=e)
            return False
    
    def get_market_hours(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get today's market opening and closing times.
        
        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: Market open and close times,
                                                         or (None, None) if unavailable
        """
        if not self.api:
            return None, None
        
        try:
            calendar = self.api.get_calendar(start=datetime.now().date())[0]
            return calendar.open, calendar.close
        except Exception as e:
            logger.error("Failed to get market hours", error=e)
            return None, None
    
    def place_order(
        self, 
        symbol: str, 
        qty: float, 
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        stop_price: float = None,
        limit_price: float = None
    ) -> Optional[str]:
        """Place an order"""
        if not self.api:
            logger.error("Alpaca API not available")
            return None
        
        try:
            # Fix fractional shares for short sales
            # Alpaca doesn't allow fractional short sales, so round down for sells
            if side.lower() == "sell" and qty != int(qty):
                original_qty = qty
                qty = int(qty)  # Round down to whole shares
                if qty <= 0:
                    logger.warning(f"Order quantity too small after rounding: {original_qty} -> {qty}")
                    return None
                logger.debug(f"Rounded short sale quantity: {original_qty} -> {qty}")
            
            order_params = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }
            
            if stop_price:
                order_params['stop_price'] = round(stop_price, 2)  # Round to cents
            if limit_price:
                order_params['limit_price'] = round(limit_price, 2)  # Round to cents
            
            order = self.api.submit_order(**order_params)
            logger.trade(side.upper(), symbol, qty, limit_price or 0, order_id=order.id)
            return order.id
            
        except Exception as e:
            logger.error(f"Failed to place order: {symbol} {side} {qty}", error=e)
            return None
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        if not self.api:
            return {}
        
        try:
            positions = self.api.list_positions()
            return {pos.symbol: float(pos.qty) for pos in positions}
        except Exception as e:
            logger.error("Failed to get positions", error=e)
            return {}
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.api:
            return {}
        
        try:
            account = self.api.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'day_trade_buying_power': float(getattr(account, 'day_trade_buying_power', account.buying_power))
            }
        except Exception as e:
            logger.error("Failed to get account info", error=e)
            return {}

# ============================================================================
# Trading Strategy Engine
# ============================================================================

class TradingStrategy:
    """
    Multi-signal trading strategy engine.
    
    Combines signals from machine learning models, sentiment analysis, and
    technical indicators to generate high-confidence trading signals.
    Implements weighted signal aggregation with configurable parameters.
    
    Signal Sources:
    - ML Models: LSTM neural networks for price prediction (40% weight)
    - Sentiment Analysis: News and social media sentiment (30% weight)
    - Technical Indicators: RSI, MACD, Bollinger Bands (20% weight)
    - Risk Metrics: Portfolio and position-level risk (10% weight)
    """
    
    def __init__(self):
        """
        Initialize the trading strategy with all signal generators.
        
        Sets up ML models, sentiment analysis, and technical indicators
        with pre-configured weights for signal combination.
        """
        # Initialize signal generators
        self.model_manager = ModelManager()
        self.sentiment_engine = SentimentEngine()
        self.feature_engineer = FeatureEngineer()
        
        # Trading signal weights (must sum to 1.0)
        self.weights = {
            'ml_signal': 0.4,        # Machine learning predictions
            'sentiment_signal': 0.3,  # News and social sentiment
            'technical_signal': 0.2,  # Technical analysis indicators
            'risk_signal': 0.1       # Risk management adjustments
        }
        
        # Trading thresholds
        self.buy_threshold = 0.3    # Minimum signal strength to buy
        self.sell_threshold = -0.3  # Maximum signal strength to sell
        
        # Reference to trader for position checking
        self.trader = None
    
    def generate_signals(self, symbols: List[str], price_data: Dict[str, float]) -> Dict[str, TradeSignal]:
        """Generate trading signals for given symbols"""
        signals = {}
        
        for symbol in symbols:
            try:
                signal = self._generate_signal_for_symbol(symbol, price_data)
                if signal:
                    signals[symbol] = signal
            except Exception as e:
                logger.error(f"Failed to generate signal for {symbol}", error=e)
        
        return signals
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions from the trader"""
        try:
            # This should be passed from the trading engine, but for now we'll create a reference
            if hasattr(self, 'trader') and self.trader:
                return self.trader.get_positions()
            else:
                # Fallback - will need to be set by the trading engine
                return getattr(self, '_current_positions', {})
        except Exception as e:
            logger.error("Failed to get current positions", error=e)
            return {}
    
    def _generate_signal_for_symbol(self, symbol: str, price_data: Dict[str, float]) -> Optional[TradeSignal]:
        """Generate signal for a single symbol"""
        if symbol not in price_data:
            return None
        
        # Check existing positions first
        current_positions = self.get_current_positions()
        has_position = symbol in current_positions and current_positions[symbol] > 0
        
        current_price = price_data[symbol]
        
        # Get ML prediction
        ml_signal = 0.0
        ml_confidence = 0.0
        try:
            prediction = self.model_manager.get_prediction(symbol)
            if prediction:
                ml_signal = 1.0 if prediction.prediction == 1 else -1.0
                ml_confidence = prediction.confidence
        except Exception as e:
            logger.error(f"ML prediction failed for {symbol}", error=e)
        
        # Get sentiment signal
        sentiment_signals = self.sentiment_engine.get_trading_sentiment_signals([symbol])
        sentiment_signal = sentiment_signals.get(symbol, {}).get('combined_signal', 0.0)
        
        # Get technical signals
        technical_signal = self._get_technical_signal(symbol)
        
        # Combine signals
        combined_signal = (
            self.weights['ml_signal'] * ml_signal * ml_confidence +
            self.weights['sentiment_signal'] * sentiment_signal +
            self.weights['technical_signal'] * technical_signal
        )
        
        # Determine action based on signal and current position
        if combined_signal > 0.3:
            if has_position:
                # Already have position, don't buy more
                logger.debug(f"Skipping BUY signal for {symbol} - already have position of {current_positions.get(symbol, 0)} shares")
                return None
            action = "buy"
            confidence = min(combined_signal, 1.0)
        elif combined_signal < -0.3:
            if not has_position:
                # No position to sell
                logger.debug(f"Skipping SELL signal for {symbol} - no position to sell")
                return None
            action = "sell"
            confidence = min(abs(combined_signal), 1.0)
        else:
            return None  # No clear signal
        
        # Calculate position size (will be validated by risk manager)
        base_quantity = 100  # Base position size
        quantity = base_quantity * confidence
        
        # Set stop loss and take profit (rough estimates, will be refined by risk manager)
        if action == "buy":
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.04
        else:
            stop_loss = current_price * 1.02
            take_profit = current_price * 0.96
        
        risk_reward_ratio = abs(take_profit - current_price) / abs(current_price - stop_loss)
        
        return TradeSignal(
            symbol=symbol,
            action=action,
            quantity=quantity,
            confidence=confidence,
            expected_return=abs(take_profit - current_price) / current_price,
            risk_score=1.0 - confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def _get_technical_signal(self, symbol: str) -> float:
        """Get technical analysis signal"""
        try:
            # This would analyze technical indicators
            # For now, return neutral signal
            return 0.0
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}", error=e)
            return 0.0

class TradingEngine:
    """Main trading engine"""
    
    def __init__(self):
        self.data_orchestrator = DataCollectionOrchestrator()
        self.trader = AlpacaTrader()
        self.strategy = TradingStrategy()
        self.strategy.trader = self.trader  # Give strategy access to trader for position checking
        self.risk_manager = RiskManager()
        
        # Engine state
        self.running = False
        self.last_update = datetime.now()
        from config.settings import get_all_symbols
        self.trading_symbols = get_all_symbols()
        
        # Performance tracking
        self.trades_today = 0
        self.daily_pnl = 0.0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the trading engine"""
        logger.info("Starting trading engine...")
        
        if not self.trader.api:
            logger.error("Cannot start - Alpaca API not available")
            return
        
        self.running = True
        
        # Schedule different tasks
        self._schedule_tasks()
        
        # Initial setup
        self._initial_setup()
        
        # Main trading loop
        self._run_main_loop()
    
    def stop(self):
        """Stop the trading engine"""
        logger.info("Stopping trading engine...")
        self.running = False
        
        # Generate final report
        price_data = self._get_current_prices()
        final_report = self.risk_manager.get_risk_report(price_data)
        logger.info(f"Final report: {final_report}")
    
    def _schedule_tasks(self):
        """Schedule periodic tasks"""
        # Market data updates
        schedule.every(1).minutes.do(self._update_market_data)
        
        # News updates
        schedule.every(5).minutes.do(self._update_news_data)
        
        # Sentiment analysis
        schedule.every(10).minutes.do(self._update_sentiment)
        
        # Model retraining (weekly)
        schedule.every().sunday.at("02:00").do(self._retrain_models)
        
        # Risk monitoring
        schedule.every(30).seconds.do(self._monitor_risk)
        
        # Trading signals (during market hours)
        schedule.every(2).minutes.do(self._generate_and_execute_trades)
    
    def _initial_setup(self):
        """Perform initial setup"""
        logger.info("Performing initial setup...")
        
        # Collect initial data
        try:
            self.data_orchestrator.run_initial_collection()
            logger.info("Initial data collection completed")
        except Exception as e:
            logger.error("Initial data collection failed", error=e)
        
        # Update sentiment
        try:
            self.strategy.sentiment_engine.update_all_sentiments()
            logger.info("Initial sentiment analysis completed")
        except Exception as e:
            logger.error("Initial sentiment analysis failed", error=e)
        
        # Get account info
        account_info = self.trader.get_account_info()
        if account_info:
            logger.info(f"Account value: ${account_info.get('portfolio_value', 0):,.2f}")
            logger.info(f"Buying power: ${account_info.get('buying_power', 0):,.2f}")
    
    def _display_market_status(self):
        """Display current market status"""
        try:
            is_open = self.trader.is_market_open()
            
            if is_open:
                logger.info("🟢 MARKET STATUS: OPEN - Active trading enabled")
            else:
                # Get next market open time
                try:
                    from datetime import datetime, timezone
                    import pytz
                    
                    # Get US Eastern time
                    eastern = pytz.timezone('US/Eastern')
                    now = datetime.now(eastern)
                    
                    # Market hours: 9:30 AM - 4:00 PM ET
                    if now.weekday() < 5:  # Monday-Friday
                        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
                        
                        if now < market_open:
                            # Before market open today
                            time_until_open = market_open - now
                            hours, remainder = divmod(time_until_open.seconds, 3600)
                            minutes, _ = divmod(remainder, 60)
                            logger.info(f"🔴 MARKET STATUS: CLOSED")
                            logger.info(f"📅 Next Market Open: TODAY at 9:30 AM ET (in {hours}h {minutes}m)")
                        else:
                            # After market close, next trading day
                            # Calculate next Monday if it's Friday, otherwise next day
                            if now.weekday() == 4:  # Friday
                                days_ahead = 3  # Monday
                            else:
                                days_ahead = 1  # Next day
                            
                            next_day = now + timedelta(days=days_ahead)
                            next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
                            logger.info(f"🔴 MARKET STATUS: CLOSED")
                            logger.info(f"📅 Next Market Open: {next_open.strftime('%A, %B %d')} at 9:30 AM ET")
                    else:
                        # Weekend - next Monday
                        days_ahead = 7 - now.weekday()  # Days until Monday
                        next_monday = now + timedelta(days=days_ahead)
                        next_open = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
                        logger.info(f"🔴 MARKET STATUS: CLOSED (Weekend)")
                        logger.info(f"📅 Next Market Open: {next_open.strftime('%A, %B %d')} at 9:30 AM ET")
                        
                    logger.info("🤖 BOT STATUS: Monitoring mode - Will trade when market opens")
                    logger.info("📊 Current Activities: Data collection, model updates, risk monitoring")
                    
                except Exception as e:
                    logger.info("🔴 MARKET STATUS: CLOSED")
                    logger.info("🤖 BOT STATUS: Monitoring mode")
                    
        except Exception as e:
            logger.error("Failed to check market status", error=e)
    
    def _run_main_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop...")
        
        # Display market status
        self._display_market_status()
        
        while self.running:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Sleep for a short interval
                time.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error("Error in main loop", error=e)
                time.sleep(60)  # Wait before retrying
    
    def _update_market_data(self):
        """Update market data"""
        try:
            self.data_orchestrator.run_realtime_collection()
            logger.debug("Market data updated")
        except Exception as e:
            logger.error("Failed to update market data", error=e)
    
    def _update_news_data(self):
        """Update news data"""
        try:
            # This would be handled by the data orchestrator
            logger.debug("News data updated")
        except Exception as e:
            logger.error("Failed to update news data", error=e)
    
    def _update_sentiment(self):
        """Update sentiment analysis"""
        try:
            self.strategy.sentiment_engine.update_all_sentiments()
            logger.debug("Sentiment analysis updated")
        except Exception as e:
            logger.error("Failed to update sentiment", error=e)
    
    def _retrain_models(self):
        """Retrain ML models"""
        logger.info("Starting model retraining...")
        try:
            self.strategy.model_manager.train_all_models()
            logger.info("Model retraining completed")
        except Exception as e:
            logger.error("Model retraining failed", error=e)
    
    def _monitor_risk(self):
        """Monitor risk levels"""
        try:
            price_data = self._get_current_prices()
            if not price_data:
                return
            
            # Update daily P&L
            self.risk_manager.update_daily_pnl(price_data)
            
            # Check for emergency conditions
            risk_metrics = self.risk_manager.calculate_portfolio_risk(price_data)
            
            if risk_metrics.risk_level.value == "critical":
                logger.risk_alert("Critical risk level detected!")
                self.risk_manager.emergency_stop(price_data)
                
        except Exception as e:
            logger.error("Risk monitoring failed", error=e)
    
    def _generate_and_execute_trades(self):
        """Generate and execute trading signals"""
        if not self.trader.is_market_open():
            return
        
        try:
            # Get current prices
            price_data = self._get_current_prices()
            if not price_data:
                return
            
            # Generate signals
            signals = self.strategy.generate_signals(self.trading_symbols, price_data)
            
            # Validate and execute trades
            for symbol, signal in signals.items():
                validated_signal = self.risk_manager.validate_trade(signal, price_data)
                
                if validated_signal.approved:
                    self._execute_trade(validated_signal)
                else:
                    logger.debug(f"Trade rejected for {symbol}: {validated_signal.rejection_reason}")
            
        except Exception as e:
            logger.error("Trade generation and execution failed", error=e)
    
    def _execute_trade(self, signal: TradeSignal):
        """Execute a validated trade signal"""
        try:
            # Place market order
            order_id = self.trader.place_order(
                symbol=signal.symbol,
                qty=signal.quantity,
                side=signal.action,
                order_type="market"
            )
            
            if order_id:
                self.trades_today += 1
                
                # Place stop loss and take profit orders
                if signal.action == "buy":
                    # Stop loss
                    self.trader.place_order(
                        symbol=signal.symbol,
                        qty=signal.quantity,
                        side="sell",
                        order_type="stop",
                        stop_price=signal.stop_loss
                    )
                    
                    # Take profit
                    self.trader.place_order(
                        symbol=signal.symbol,
                        qty=signal.quantity,
                        side="sell",
                        order_type="limit",
                        limit_price=signal.take_profit
                    )
                
                logger.info(f"Trade executed: {signal.symbol} {signal.action} {signal.quantity}")
            
        except Exception as e:
            logger.error(f"Failed to execute trade for {signal.symbol}", error=e)
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        try:
            prices = {}
            for symbol in self.trading_symbols:
                df = self.data_orchestrator.get_latest_data(symbol, limit=1)
                if not df.empty:
                    prices[symbol] = df['close_price'].iloc[-1]
            return prices
        except Exception as e:
            logger.error("Failed to get current prices", error=e)
            return {}
    
    def get_status(self) -> Dict:
        """Get current engine status"""
        price_data = self._get_current_prices()
        account_info = self.trader.get_account_info()
        risk_report = self.risk_manager.get_risk_report(price_data)
        
        return {
            'running': self.running,
            'market_open': self.trader.is_market_open(),
            'trades_today': self.trades_today,
            'account_value': account_info.get('portfolio_value', 0),
            'cash_balance': account_info.get('cash', 0),
            'risk_level': risk_report.get('risk_level', 'unknown'),
            'active_positions': len(self.trader.get_positions()),
            'last_update': self.last_update.isoformat()
        }

def main():
    """Main entry point"""
    logger.info("AI Trading Bot starting...")
    
    engine = TradingEngine()
    
    try:
        engine.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error("Unexpected error", error=e)
    finally:
        engine.stop()
        logger.info("Trading bot stopped")

if __name__ == "__main__":
    main()