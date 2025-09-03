"""
Trading Bot Logger
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Import after ensuring path is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import LOGGING_CONFIG


class TradingLogger:
    """
    Enhanced logger specifically designed for trading operations.
    
    Provides structured logging with specialized methods for different types of trading events:
    - General system operations (info, debug, warning, error)
    - Trading actions (buy/sell orders)
    - Profit & Loss tracking
    - ML model predictions
    - Risk management alerts
    """
    
    def __init__(self, name: str = "trading_bot"):
        """
        Initialize the trading logger.
        
        Args:
            name: Logger name for categorization
        """
        self.name = name
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure the logging system with file and console handlers."""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Configure logging using the settings
            logging.config.dictConfig(LOGGING_CONFIG)
            self.logger = logging.getLogger(self.name)
            
            # Log successful initialization
            self.logger.debug(f"Logger initialized: {self.name}")
            
        except Exception as e:
            # Fallback to basic logging if configuration fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            self.logger = logging.getLogger(self.name)
            self.logger.error(f"Failed to configure advanced logging: {e}")
    
    # ============================================================================
    # General Logging Methods
    # ============================================================================
    
    def info(self, message: str, **kwargs) -> None:
        """
        Log informational message.
        
        Args:
            message: The message to log
            **kwargs: Additional context to include in the log
        """
        self.logger.info(self._format_message(message, **kwargs))
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Log debug message (only appears in log file).
        
        Args:
            message: The message to log
            **kwargs: Additional context to include in the log
        """
        self.logger.debug(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log warning message.
        
        Args:
            message: The warning message to log
            **kwargs: Additional context to include in the log
        """
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """
        Log error message with optional exception details.
        
        Args:
            message: The error message to log
            error: Optional exception object to include details from
            **kwargs: Additional context to include in the log
        """
        if error:
            message = f"{message} | Error: {str(error)}"
            if hasattr(error, '__class__'):
                message = f"{message} | Type: {error.__class__.__name__}"
        
        self.logger.error(self._format_message(message, **kwargs))
    
    # ============================================================================
    # Trading-Specific Logging Methods
    # ============================================================================
    
    def trade(self, action: str, symbol: str, quantity: float, price: float, 
              order_id: Optional[str] = None, **kwargs) -> None:
        """
        Log trading actions (buy/sell orders).
        
        Args:
            action: Trading action ('BUY' or 'SELL')
            symbol: Trading symbol (e.g., 'AAPL', 'BTCUSD')
            quantity: Number of shares/units
            price: Execution price
            order_id: Optional order ID for tracking
            **kwargs: Additional trading context
        """
        message = f"TRADE | {action.upper()} | {symbol} | Qty: {quantity} | Price: ${price:.2f}"
        if order_id:
            kwargs['order_id'] = order_id
        
        self.logger.info(self._format_message(message, **kwargs))
    
    def profit_loss(self, symbol: str, pnl: float, percentage: Optional[float] = None,
                   position_size: Optional[float] = None, **kwargs) -> None:
        """
        Log profit and loss for completed trades.
        
        Args:
            symbol: Trading symbol
            pnl: Profit/loss in dollars
            percentage: Optional percentage gain/loss
            position_size: Optional position size for context
            **kwargs: Additional P&L context
        """
        status = "PROFIT" if pnl >= 0 else "LOSS"
        message = f"P&L | {symbol} | {status}: ${pnl:.2f}"
        
        if percentage is not None:
            message += f" ({percentage:.2f}%)"
        if position_size is not None:
            kwargs['position_size'] = position_size
        
        self.logger.info(self._format_message(message, **kwargs))
    
    def model_prediction(self, symbol: str, prediction: str, confidence: float,
                        probability: Optional[float] = None, model_name: Optional[str] = None,
                        **kwargs) -> None:
        """
        Log machine learning model predictions.
        
        Args:
            symbol: Trading symbol
            prediction: Prediction result ('BUY', 'SELL', 'HOLD')
            confidence: Model confidence level (0-1)
            probability: Optional probability score
            model_name: Optional model identifier
            **kwargs: Additional prediction context
        """
        message = f"PREDICTION | {symbol} | {prediction} | Confidence: {confidence:.3f}"
        
        if probability is not None:
            kwargs['probability'] = f"{probability:.3f}"
        if model_name is not None:
            kwargs['model'] = model_name
        
        self.logger.info(self._format_message(message, **kwargs))
    
    def risk_alert(self, message: str, severity: str = "HIGH", **kwargs) -> None:
        """
        Log risk management alerts and warnings.
        
        Args:
            message: Risk alert message
            severity: Alert severity level ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
            **kwargs: Additional risk context
        """
        kwargs['severity'] = severity
        formatted_message = f"RISK ALERT | {self._format_message(message, **kwargs)}"
        
        # Use appropriate log level based on severity
        if severity in ['CRITICAL', 'HIGH']:
            self.logger.error(formatted_message)
        else:
            self.logger.warning(formatted_message)
    
    def portfolio_update(self, total_value: float, cash_balance: float,
                        positions_count: int, daily_pnl: Optional[float] = None,
                        **kwargs) -> None:
        """
        Log portfolio status updates.
        
        Args:
            total_value: Total portfolio value
            cash_balance: Available cash
            positions_count: Number of active positions
            daily_pnl: Optional daily P&L
            **kwargs: Additional portfolio context
        """
        message = (f"PORTFOLIO | Value: ${total_value:,.2f} | "
                  f"Cash: ${cash_balance:,.2f} | Positions: {positions_count}")
        
        if daily_pnl is not None:
            kwargs['daily_pnl'] = f"${daily_pnl:.2f}"
        
        self.logger.info(self._format_message(message, **kwargs))
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def _format_message(self, message: str, **kwargs) -> str:
        """
        Format log message with additional context.
        
        Args:
            message: Base message
            **kwargs: Additional key-value pairs to append
            
        Returns:
            str: Formatted message with context
        """
        if not kwargs:
            return message
        
        # Format context as key=value pairs
        context_parts = []
        for key, value in kwargs.items():
            # Handle different value types appropriately
            if isinstance(value, float):
                if key in ['price', 'pnl', 'value', 'balance']:
                    context_parts.append(f"{key}=${value:.2f}")
                elif key in ['confidence', 'probability', 'percentage']:
                    context_parts.append(f"{key}={value:.3f}")
                else:
                    context_parts.append(f"{key}={value:.4f}")
            else:
                context_parts.append(f"{key}={value}")
        
        context = " | ".join(context_parts)
        return f"{message} | {context}"
    
    def flush(self) -> None:
        """Force flush all log handlers."""
        for handler in self.logger.handlers:
            handler.flush()
    
    def get_log_level(self) -> str:
        """Get current logging level."""
        return logging.getLevelName(self.logger.level)
    
    def set_log_level(self, level: str) -> None:
        """
        Set logging level dynamically.
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        self.info(f"Log level changed to: {level}")


# ============================================================================
# Global Logger Instance
# ============================================================================

# Create the global logger instance that will be used throughout the application
logger = TradingLogger()


def get_logger(name: str = None) -> TradingLogger:
    """
    Get a logger instance with optional custom name.
    
    Args:
        name: Optional custom logger name
        
    Returns:
        TradingLogger: Logger instance
    """
    if name:
        return TradingLogger(name)
    return logger


# ============================================================================
# Testing and Validation
# ============================================================================

