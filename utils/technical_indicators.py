"""
Technical indicators for market analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ml_config
from utils.logger import logger

class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """Price Momentum"""
        return data.diff(period)
    
    @staticmethod
    def rate_of_change(data: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change"""
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0))
        return pd.Series(obv).cumsum()

class FeatureEngineer:
    """Engineer features for machine learning models"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features_df = df.copy()
        
        # Price features
        features_df['returns'] = features_df['close_price'].pct_change()
        features_df['log_returns'] = np.log(features_df['close_price'] / features_df['close_price'].shift(1))
        features_df['price_range'] = (features_df['high_price'] - features_df['low_price']) / features_df['close_price']
        features_df['gap'] = (features_df['open_price'] - features_df['close_price'].shift(1)) / features_df['close_price'].shift(1)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = self.indicators.sma(features_df['close_price'], period)
            features_df[f'ema_{period}'] = self.indicators.ema(features_df['close_price'], period)
            features_df[f'price_to_sma_{period}'] = features_df['close_price'] / features_df[f'sma_{period}']
        
        # RSI
        features_df['rsi'] = self.indicators.rsi(features_df['close_price'], ml_config.RSI_PERIOD)
        features_df['rsi_oversold'] = (features_df['rsi'] < 30).astype(int)
        features_df['rsi_overbought'] = (features_df['rsi'] > 70).astype(int)
        
        # MACD
        macd, signal, histogram = self.indicators.macd(
            features_df['close_price'], 
            ml_config.MACD_FAST, 
            ml_config.MACD_SLOW, 
            ml_config.MACD_SIGNAL
        )
        features_df['macd'] = macd
        features_df['macd_signal'] = signal
        features_df['macd_histogram'] = histogram
        features_df['macd_bullish'] = (features_df['macd'] > features_df['macd_signal']).astype(int)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(
            features_df['close_price'], 
            ml_config.BB_PERIOD, 
            ml_config.BB_STD
        )
        features_df['bb_upper'] = bb_upper
        features_df['bb_middle'] = bb_middle
        features_df['bb_lower'] = bb_lower
        features_df['bb_position'] = (features_df['close_price'] - bb_lower) / (bb_upper - bb_lower)
        features_df['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
        
        # Stochastic
        stoch_k, stoch_d = self.indicators.stochastic(
            features_df['high_price'], 
            features_df['low_price'], 
            features_df['close_price']
        )
        features_df['stoch_k'] = stoch_k
        features_df['stoch_d'] = stoch_d
        features_df['stoch_oversold'] = (features_df['stoch_k'] < 20).astype(int)
        features_df['stoch_overbought'] = (features_df['stoch_k'] > 80).astype(int)
        
        # ATR
        features_df['atr'] = self.indicators.atr(
            features_df['high_price'], 
            features_df['low_price'], 
            features_df['close_price']
        )
        features_df['atr_normalized'] = features_df['atr'] / features_df['close_price']
        
        # Williams %R
        features_df['williams_r'] = self.indicators.williams_r(
            features_df['high_price'], 
            features_df['low_price'], 
            features_df['close_price']
        )
        
        # Volume features
        if 'volume' in features_df.columns:
            features_df['volume_sma'] = self.indicators.sma(features_df['volume'], 20)
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            features_df['vwap'] = self.indicators.vwap(
                features_df['high_price'], 
                features_df['low_price'], 
                features_df['close_price'], 
                features_df['volume']
            )
            features_df['obv'] = self.indicators.obv(features_df['close_price'], features_df['volume'])
        
        # Momentum indicators
        features_df['momentum'] = self.indicators.momentum(features_df['close_price'])
        features_df['roc'] = self.indicators.rate_of_change(features_df['close_price'])
        
        # Volatility features
        features_df['volatility'] = features_df['returns'].rolling(window=20).std()
        features_df['volatility_ratio'] = features_df['volatility'] / features_df['volatility'].rolling(window=50).mean()
        
        # Candlestick patterns (simplified)
        features_df['doji'] = (abs(features_df['open_price'] - features_df['close_price']) < 
                              0.1 * (features_df['high_price'] - features_df['low_price'])).astype(int)
        features_df['hammer'] = ((features_df['close_price'] - features_df['low_price']) > 
                                2 * abs(features_df['open_price'] - features_df['close_price'])).astype(int)
        
        # Time-based features
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['month'] = features_df['timestamp'].dt.month
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features_df[f'close_lag_{lag}'] = features_df['close_price'].shift(lag)
            features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
            features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag) if 'volume' in features_df.columns else 0
        
        logger.info(f"Created {len(features_df.columns)} features")
        return features_df
    
    def create_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Create prediction target"""
        # Predict future price direction (1 for up, 0 for down)
        future_returns = df['close_price'].shift(-horizon) / df['close_price'] - 1
        target = (future_returns > 0).astype(int)
        
        return target
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for machine learning"""
        # Create features
        features_df = self.create_features(df)
        
        # Create target
        target = self.create_target(features_df, ml_config.PREDICTION_HORIZON)
        
        # Select numeric features only
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_columns]
        
        # Remove rows with NaN values
        valid_idx = ~(features_df.isnull().any(axis=1) | target.isnull())
        features_df = features_df[valid_idx]
        target = target[valid_idx]
        
        # Feature selection to reduce overfitting (keep top 30 features)
        if len(features_df.columns) > 30 and len(features_df) > 50:
            features_df, self.feature_selector = self._select_best_features_with_selector(features_df, target, top_k=30)
        else:
            self.feature_selector = None
        
        logger.info(f"Prepared ML data: {len(features_df)} samples, {len(features_df.columns)} features")
        return features_df, target
    
    def _select_best_features_with_selector(self, features_df: pd.DataFrame, target: pd.Series, top_k: int = 30) -> Tuple[pd.DataFrame, Any]:
        """Select top features to reduce overfitting and return the selector"""
        try:
            from sklearn.feature_selection import SelectKBest, f_classif
            
            # Use statistical test to select best features
            selector = SelectKBest(score_func=f_classif, k=min(top_k, len(features_df.columns)))
            features_selected = selector.fit_transform(features_df, target)
            
            # Get selected feature names
            selected_columns = features_df.columns[selector.get_support()]
            
            # Return DataFrame with selected features and the selector
            result_df = pd.DataFrame(features_selected, columns=selected_columns, index=features_df.index)
            
            logger.info(f"Feature selection: {len(features_df.columns)} -> {len(result_df.columns)} features")
            return result_df, selector
            
        except ImportError:
            logger.warning("Scikit-learn not available for feature selection, using all features")
            return features_df, None
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using all features")
            return features_df, None
    
    def _select_best_features(self, features_df: pd.DataFrame, target: pd.Series, top_k: int = 30) -> pd.DataFrame:
        """Select top features to reduce overfitting"""
        result_df, _ = self._select_best_features_with_selector(features_df, target, top_k)
        return result_df
    
    def apply_feature_selection(self, features_df: pd.DataFrame, feature_selector) -> pd.DataFrame:
        """Apply existing feature selector to new data"""
        if feature_selector is None:
            return features_df
        
        try:
            # Apply the existing selector
            features_selected = feature_selector.transform(features_df)
            
            # Get selected feature names
            selected_columns = features_df.columns[feature_selector.get_support()]
            
            # Return DataFrame with selected features
            result_df = pd.DataFrame(features_selected, columns=selected_columns, index=features_df.index)
            return result_df
            
        except Exception as e:
            logger.warning(f"Feature selection application failed: {e}, using all features")
            return features_df

if __name__ == "__main__":
    # Test feature engineering
    from data.data_collector import DataCollectionOrchestrator
    
    orchestrator = DataCollectionOrchestrator()
    df = orchestrator.get_latest_data("AAPL", limit=200)
    
    if not df.empty:
        engineer = FeatureEngineer()
        features, target = engineer.prepare_ml_data(df)
        print(f"Features shape: {features.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Feature columns: {list(features.columns)}")
    else:
        print("No data available. Run data collection first.")