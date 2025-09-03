"""
Risk management system for trading bot
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import trading_config, asset_config
from utils.logger import logger

# ============================================================================
# Risk Assessment Data Structures
# ============================================================================

class RiskLevel(Enum):
    """
    Risk level classification for portfolio and market conditions.
    
    Levels:
    - LOW: Normal market conditions, minimal restrictions
    - MEDIUM: Elevated risk, reduced position sizing
    - HIGH: High risk environment, 50% position size reduction
    - CRITICAL: Extreme risk, trading suspension and position liquidation
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Position:
    """Trading position data"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    side: str  # "buy" or "sell"
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_current_price(self, price: float):
        """Update current price and unrealized P&L"""
        self.current_price = price
        if self.side == "buy":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # sell/short
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    portfolio_value: float
    total_exposure: float
    leverage_ratio: float
    var_95: float  # Value at Risk (95%)
    max_drawdown: float
    sharpe_ratio: float
    current_drawdown: float
    risk_level: RiskLevel
    
@dataclass
class TradeSignal:
    """Trade signal with risk assessment"""
    symbol: str
    action: str  # "buy", "sell", "hold"
    quantity: float
    confidence: float
    expected_return: float
    risk_score: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    approved: bool = False
    rejection_reason: str = ""

class PortfolioTracker:
    """Track portfolio positions and performance"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.cash_balance: float = trading_config.STARTING_CASH
        self.initial_cash: float = trading_config.STARTING_CASH
        self.trade_history: List[Dict] = []
        self.daily_returns: List[float] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
    def add_position(self, position: Position):
        """Add or update a position"""
        self.positions[position.symbol] = position
        logger.info(f"Position added: {position.symbol} {position.side} {position.quantity} @ {position.entry_price}")
        
    def close_position(self, symbol: str, exit_price: float, timestamp: datetime = None):
        """Close a position"""
        if symbol not in self.positions:
            logger.warning(f"Attempting to close non-existent position: {symbol}")
            return
        
        position = self.positions[symbol]
        
        # Calculate realized P&L
        if position.side == "buy":
            realized_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            realized_pnl = (position.entry_price - exit_price) * position.quantity
        
        # Update cash balance
        if position.side == "buy":
            self.cash_balance += exit_price * position.quantity
        else:
            self.cash_balance += position.entry_price * position.quantity - realized_pnl
            
        # Record trade
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': timestamp or datetime.now(),
            'realized_pnl': realized_pnl,
            'return_pct': realized_pnl / (position.entry_price * position.quantity) * 100
        }
        
        self.trade_history.append(trade_record)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Position closed: {symbol} P&L: ${realized_pnl:.2f}")
        
    def update_positions(self, price_data: Dict[str, float]):
        """Update all positions with current prices"""
        for symbol, position in self.positions.items():
            if symbol in price_data:
                position.update_current_price(price_data[symbol])
                
    def get_portfolio_value(self, price_data: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        self.update_positions(price_data)
        
        portfolio_value = self.cash_balance
        for position in self.positions.values():
            if position.side == "buy":
                portfolio_value += position.current_price * position.quantity
            else:  # short position
                portfolio_value += position.entry_price * position.quantity + position.unrealized_pnl
                
        return portfolio_value
    
    def get_total_exposure(self, price_data: Dict[str, float]) -> float:
        """Calculate total market exposure"""
        exposure = 0
        for position in self.positions.values():
            symbol = position.symbol
            if symbol in price_data:
                exposure += abs(position.quantity * price_data[symbol])
        return exposure
    
    def calculate_returns(self, price_data: Dict[str, float]) -> float:
        """Calculate portfolio return"""
        current_value = self.get_portfolio_value(price_data)
        return (current_value - self.initial_cash) / self.initial_cash * 100

class RiskCalculator:
    """Calculate various risk metrics"""
    
    @staticmethod
    def calculate_var(returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 20:  # Need sufficient data
            return 0.0
        
        returns_array = np.array(returns)
        return np.percentile(returns_array, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[Tuple[datetime, float]]) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0.0
        
        values = [value for _, value in equity_curve]
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown * 100
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 10:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_position_risk(
        position_size: float,
        entry_price: float,
        current_price: float,
        volatility: float
    ) -> float:
        """Calculate individual position risk score"""
        # Position size risk
        position_value = position_size * current_price
        size_risk = min(position_value / 10000, 1.0)  # Normalize to $10k
        
        # Price movement risk
        price_change = abs(current_price - entry_price) / entry_price
        movement_risk = min(price_change / 0.1, 1.0)  # Normalize to 10% move
        
        # Volatility risk
        vol_risk = min(volatility / 0.3, 1.0)  # Normalize to 30% volatility
        
        # Combined risk score (0-1)
        return (size_risk * 0.4 + movement_risk * 0.3 + vol_risk * 0.3)

class RiskManager:
    """Main risk management system"""
    
    def __init__(self):
        self.portfolio = PortfolioTracker()
        self.risk_calculator = RiskCalculator()
        self.max_positions = 10
        self.max_correlation = 0.7
        
        # Risk limits
        self.daily_loss_limit = trading_config.MAX_DAILY_LOSS
        self.max_drawdown_limit = trading_config.MAX_DRAWDOWN
        self.max_position_size = trading_config.MAX_POSITION_SIZE
        
        # Tracking variables
        self.daily_pnl = 0.0
        self.session_start_value = trading_config.STARTING_CASH
        self.risk_breaches = []
        
    def assess_market_conditions(self, price_data: Dict[str, float]) -> RiskLevel:
        """Assess current market risk conditions"""
        try:
            # Calculate market volatility (using SPY as proxy)
            spy_data = self._get_volatility_data("SPY")
            
            if spy_data is None or len(spy_data) < 20:
                return RiskLevel.MEDIUM
            
            current_vol = spy_data['volatility'].iloc[-1] if 'volatility' in spy_data else 0.2
            avg_vol = spy_data['volatility'].mean() if 'volatility' in spy_data else 0.2
            
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Determine risk level based on volatility
            if vol_ratio > 2.0:
                return RiskLevel.CRITICAL
            elif vol_ratio > 1.5:
                return RiskLevel.HIGH
            elif vol_ratio > 1.2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error("Failed to assess market conditions", error=e)
            return RiskLevel.MEDIUM
    
    def _get_volatility_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get volatility data for a symbol"""
        try:
            # This would connect to your data source
            # For now, return None to use default values
            return None
        except:
            return None
    
    def calculate_portfolio_risk(self, price_data: Dict[str, float]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        portfolio_value = self.portfolio.get_portfolio_value(price_data)
        total_exposure = self.portfolio.get_total_exposure(price_data)
        
        # Calculate leverage ratio
        leverage_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate other metrics
        returns = [trade['return_pct'] for trade in self.portfolio.trade_history]
        var_95 = self.risk_calculator.calculate_var(returns) if returns else 0
        max_drawdown = self.risk_calculator.calculate_max_drawdown(self.portfolio.equity_curve)
        sharpe_ratio = self.risk_calculator.calculate_sharpe_ratio(self.portfolio.daily_returns)
        
        # Current drawdown
        if self.portfolio.equity_curve:
            peak_value = max(value for _, value in self.portfolio.equity_curve)
            current_drawdown = (peak_value - portfolio_value) / peak_value * 100
        else:
            current_drawdown = 0
        
        # Determine overall risk level
        risk_level = self._determine_risk_level(
            leverage_ratio, max_drawdown, current_drawdown, var_95
        )
        
        return RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            leverage_ratio=leverage_ratio,
            var_95=var_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            current_drawdown=current_drawdown,
            risk_level=risk_level
        )
    
    def _determine_risk_level(
        self, leverage: float, max_dd: float, current_dd: float, var: float
    ) -> RiskLevel:
        """Determine overall portfolio risk level"""
        risk_score = 0
        
        # Leverage risk
        if leverage > 2.0:
            risk_score += 3
        elif leverage > 1.5:
            risk_score += 2
        elif leverage > 1.0:
            risk_score += 1
        
        # Drawdown risk
        if max_dd > 15:
            risk_score += 3
        elif max_dd > 10:
            risk_score += 2
        elif max_dd > 5:
            risk_score += 1
        
        # Current drawdown
        if current_dd > 10:
            risk_score += 2
        elif current_dd > 5:
            risk_score += 1
        
        # VaR risk
        if var < -5:  # Very negative VaR
            risk_score += 2
        elif var < -3:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def validate_trade(self, signal: TradeSignal, price_data: Dict[str, float]) -> TradeSignal:
        """Validate and potentially modify trade signal based on risk rules"""
        signal.approved = False
        signal.rejection_reason = ""
        
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.daily_loss_limit * self.session_start_value:
                signal.rejection_reason = "Daily loss limit exceeded"
                logger.risk_alert(f"Trade rejected for {signal.symbol}: Daily loss limit exceeded")
                return signal
            
            # Check portfolio risk
            risk_metrics = self.calculate_portfolio_risk(price_data)
            
            # Check max drawdown
            if risk_metrics.current_drawdown > self.max_drawdown_limit * 100:
                signal.rejection_reason = "Maximum drawdown exceeded"
                logger.risk_alert(f"Trade rejected for {signal.symbol}: Max drawdown exceeded")
                return signal
            
            # Check position size limits
            position_value = signal.quantity * price_data.get(signal.symbol, 0)
            max_position_value = risk_metrics.portfolio_value * self.max_position_size
            
            if position_value > max_position_value:
                # Adjust position size
                signal.quantity = max_position_value / price_data.get(signal.symbol, 1)
                logger.info(f"Position size adjusted for {signal.symbol}: {signal.quantity}")
            
            # Check maximum number of positions
            if len(self.portfolio.positions) >= self.max_positions and signal.action == "buy":
                signal.rejection_reason = "Maximum positions limit reached"
                logger.risk_alert(f"Trade rejected for {signal.symbol}: Max positions limit")
                return signal
            
            # Check risk level adjustments
            if risk_metrics.risk_level == RiskLevel.CRITICAL:
                signal.rejection_reason = "Critical risk level - trading suspended"
                logger.risk_alert("Trading suspended due to critical risk level")
                return signal
            elif risk_metrics.risk_level == RiskLevel.HIGH:
                # Reduce position size by 50%
                signal.quantity *= 0.5
                logger.warning(f"High risk detected - reducing position size for {signal.symbol}")
            elif risk_metrics.risk_level == RiskLevel.MEDIUM:
                # Reduce position size by 25%
                signal.quantity *= 0.75
            
            # Check risk-reward ratio
            if signal.risk_reward_ratio < 1.5:
                signal.rejection_reason = "Poor risk-reward ratio"
                logger.info(f"Trade rejected for {signal.symbol}: Poor risk-reward ratio")
                return signal
            
            # All checks passed
            signal.approved = True
            logger.info(f"Trade approved for {signal.symbol}: {signal.action} {signal.quantity}")
            
        except Exception as e:
            signal.rejection_reason = f"Risk validation error: {str(e)}"
            logger.error(f"Risk validation failed for {signal.symbol}", error=e)
        
        return signal
    
    def calculate_position_size(
        self, 
        signal_confidence: float, 
        volatility: float, 
        portfolio_value: float
    ) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        # Kelly Criterion: f = (bp - q) / b
        # where: f = fraction to bet, b = odds, p = win probability, q = loss probability
        
        # Estimate win probability from confidence
        win_prob = 0.5 + (signal_confidence * 0.3)  # 50% base + confidence adjustment
        
        # Estimate average win/loss ratio (simplified)
        avg_win = 0.02  # 2% average win
        avg_loss = 0.015  # 1.5% average loss
        
        # Kelly fraction
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        
        # Apply safety factor and volatility adjustment
        safety_factor = 0.25  # Use 25% of Kelly recommendation
        volatility_adjustment = max(0.5, 1 - volatility)  # Reduce size with higher volatility
        
        optimal_fraction = kelly_fraction * safety_factor * volatility_adjustment
        
        # Ensure within limits
        optimal_fraction = max(0.01, min(optimal_fraction, self.max_position_size))
        
        return portfolio_value * optimal_fraction
    
    def set_stop_loss_take_profit(
        self, 
        entry_price: float, 
        side: str, 
        volatility: float
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        # Base stop loss on volatility
        stop_distance = max(volatility * 2, trading_config.STOP_LOSS_PCT)
        profit_distance = stop_distance * 2  # 2:1 reward to risk
        
        if side == "buy":
            stop_loss = entry_price * (1 - stop_distance)
            take_profit = entry_price * (1 + profit_distance)
        else:  # sell/short
            stop_loss = entry_price * (1 + stop_distance)
            take_profit = entry_price * (1 - profit_distance)
        
        return stop_loss, take_profit
    
    def update_daily_pnl(self, price_data: Dict[str, float]):
        """Update daily P&L tracking"""
        current_value = self.portfolio.get_portfolio_value(price_data)
        self.daily_pnl = current_value - self.session_start_value
        
        # Reset daily tracking at market open (if needed)
        current_time = datetime.now()
        if current_time.hour == 9 and current_time.minute == 30:  # Market open
            self.session_start_value = current_value
            self.daily_pnl = 0.0
    
    def emergency_stop(self, price_data: Dict[str, float]):
        """Emergency stop all trading activities"""
        logger.risk_alert("EMERGENCY STOP ACTIVATED")
        
        # Close all positions
        for symbol in list(self.portfolio.positions.keys()):
            if symbol in price_data:
                self.portfolio.close_position(symbol, price_data[symbol])
        
        # Record emergency stop
        self.risk_breaches.append({
            'timestamp': datetime.now(),
            'type': 'emergency_stop',
            'portfolio_value': self.portfolio.get_portfolio_value(price_data),
            'reason': 'Risk limits exceeded'
        })
    
    def get_risk_report(self, price_data: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        risk_metrics = self.calculate_portfolio_risk(price_data)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': risk_metrics.portfolio_value,
            'cash_balance': self.portfolio.cash_balance,
            'total_exposure': risk_metrics.total_exposure,
            'leverage_ratio': risk_metrics.leverage_ratio,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': risk_metrics.max_drawdown,
            'current_drawdown': risk_metrics.current_drawdown,
            'var_95': risk_metrics.var_95,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'risk_level': risk_metrics.risk_level.value,
            'active_positions': len(self.portfolio.positions),
            'total_trades': len(self.portfolio.trade_history),
            'risk_breaches': len(self.risk_breaches)
        }

