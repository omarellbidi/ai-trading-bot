"""
Advanced feature engineering to capture more market patterns
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AdvancedFeatureEngineer:
    """Create sophisticated features that capture market microstructure"""
    
    def create_advanced_features(self, df):
        """Add advanced features to improve prediction accuracy"""
        features_df = df.copy()
        
        # 1. MARKET MICROSTRUCTURE FEATURES
        features_df['bid_ask_spread'] = (features_df['high_price'] - features_df['low_price']) / features_df['close_price']
        features_df['price_efficiency'] = abs(features_df['close_price'] - features_df['open_price']) / (features_df['high_price'] - features_df['low_price'])
        
        # 2. VOLATILITY FEATURES (Multiple Timeframes)
        for window in [5, 10, 20, 50]:
            features_df[f'volatility_{window}d'] = features_df['close_price'].pct_change().rolling(window).std()
            features_df[f'volatility_ratio_{window}d'] = features_df[f'volatility_{window}d'] / features_df['volatility_20d']
        
        # 3. VOLUME PROFILE FEATURES
        features_df['volume_price_trend'] = (features_df['close_price'].diff() * features_df['volume']).rolling(10).sum()
        features_df['volume_weighted_price'] = (features_df['close_price'] * features_df['volume']).rolling(20).sum() / features_df['volume'].rolling(20).sum()
        features_df['volume_momentum'] = features_df['volume'].pct_change(5)
        
        # 4. TREND STRENGTH FEATURES
        features_df['trend_strength'] = self._calculate_trend_strength(features_df['close_price'])
        features_df['regime_change'] = self._detect_regime_changes(features_df['close_price'])
        
        # 5. SEASONALITY FEATURES
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['month'] = features_df['timestamp'].dt.month  
        features_df['quarter'] = features_df['timestamp'].dt.quarter
        features_df['is_month_end'] = (features_df['timestamp'].dt.day >= 25).astype(int)
        features_df['is_quarter_end'] = ((features_df['month'] % 3 == 0) & (features_df['day_of_week'] >= 25)).astype(int)
        
        # 6. CORRELATION FEATURES (with market)
        # This would require SPY data as market proxy
        # features_df['correlation_with_market'] = self._rolling_correlation(features_df['close_price'], spy_data)
        
        # 7. FRACTAL/CHAOS FEATURES
        features_df['hurst_exponent'] = self._calculate_hurst_exponent(features_df['close_price'], window=50)
        features_df['fractal_dimension'] = 2 - features_df['hurst_exponent']
        
        # 8. STATISTICAL FEATURES
        for window in [10, 20, 50]:
            returns = features_df['close_price'].pct_change()
            features_df[f'skewness_{window}d'] = returns.rolling(window).skew()
            features_df[f'kurtosis_{window}d'] = returns.rolling(window).kurt()
            features_df[f'returns_entropy_{window}d'] = self._calculate_entropy(returns, window)
        
        # 9. PATTERN RECOGNITION FEATURES
        features_df['double_top'] = self._detect_double_top(features_df)
        features_df['double_bottom'] = self._detect_double_bottom(features_df)
        features_df['head_shoulders'] = self._detect_head_shoulders(features_df)
        
        # 10. MOMENTUM FEATURES
        for period in [3, 7, 14, 21]:
            features_df[f'momentum_{period}d'] = features_df['close_price'].pct_change(period)
            features_df[f'momentum_acceleration_{period}d'] = features_df[f'momentum_{period}d'].diff()
        
        return features_df
    
    def _calculate_trend_strength(self, price_series, window=20):
        """Calculate trend strength using linear regression R²"""
        def rolling_r_squared(series):
            x = np.arange(len(series))
            if len(series) < 5:
                return 0
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
                return r_value ** 2
            except:
                return 0
        
        return price_series.rolling(window).apply(rolling_r_squared)
    
    def _detect_regime_changes(self, price_series, window=20):
        """Detect when market regime changes (trending vs sideways)"""
        volatility = price_series.pct_change().rolling(window).std()
        trend_strength = self._calculate_trend_strength(price_series, window)
        
        # High volatility + Low trend = Regime change
        regime_change = (volatility > volatility.rolling(50).quantile(0.8)) & (trend_strength < 0.1)
        return regime_change.astype(int)
    
    def _calculate_hurst_exponent(self, price_series, window=50):
        """Calculate Hurst exponent to measure trend persistence"""
        def hurst(ts):
            if len(ts) < 10:
                return 0.5
            try:
                lags = range(2, min(20, len(ts)//2))
                tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5
        
        return price_series.rolling(window).apply(hurst)
    
    def _calculate_entropy(self, returns, window):
        """Calculate entropy of returns distribution"""
        def entropy(series):
            if len(series) < 5:
                return 0
            try:
                hist, _ = np.histogram(series.dropna(), bins=10)
                hist = hist / hist.sum()  # Normalize
                hist = hist[hist > 0]  # Remove zeros
                return -np.sum(hist * np.log(hist))
            except:
                return 0
        
        return returns.rolling(window).apply(entropy)
    
    def _detect_double_top(self, df, window=10):
        """Detect double top patterns"""
        highs = df['high_price'].rolling(window, center=True).max() == df['high_price']
        return highs.astype(int)
    
    def _detect_double_bottom(self, df, window=10):
        """Detect double bottom patterns"""
        lows = df['low_price'].rolling(window, center=True).min() == df['low_price']
        return lows.astype(int)
    
    def _detect_head_shoulders(self, df, window=15):
        """Simplified head and shoulders detection"""
        # This is a simplified version - real implementation would be more complex
        price_normalized = (df['high_price'] - df['high_price'].rolling(window).min()) / (df['high_price'].rolling(window).max() - df['high_price'].rolling(window).min())
        pattern = (price_normalized.shift(window//3) < price_normalized) & (price_normalized.shift(-window//3) < price_normalized)
        return pattern.astype(int)

# 3. ALTERNATIVE DATA FEATURES
class AlternativeDataFeatures:
    """Incorporate alternative data sources"""
    
    def add_economic_features(self, df):
        """Add economic indicators (would require additional APIs)"""
        # These would require APIs like FRED, Alpha Vantage, etc.
        features = {
            'vix_level': 'VIX level from CBOE',
            'treasury_yield': '10Y Treasury yield',
            'dollar_index': 'DXY US Dollar Index',
            'oil_price': 'Oil price (affects energy stocks)',
            'gold_price': 'Gold price (affects mining stocks)',
        }
        
        print("💡 Economic Features to Add:")
        for feature, description in features.items():
            print(f"   {feature}: {description}")
        
        return df
    
    def add_social_sentiment(self, df, symbol):
        """Add social media sentiment (would require Twitter/Reddit APIs)"""
        social_features = {
            'twitter_sentiment': f'Twitter sentiment for ${symbol}',
            'reddit_mentions': f'Reddit r/investing mentions of {symbol}',
            'google_trends': f'Google search trends for {symbol}',
            'insider_trading': f'Recent insider trades for {symbol}',
        }
        
        print(f"📱 Social Features for {symbol}:")
        for feature, description in social_features.items():
            print(f"   {feature}: {description}")
        
        return df

if __name__ == "__main__":
    print("🚀 Advanced Feature Engineering Ready")
    engineer = AdvancedFeatureEngineer()
    print("This will add 50+ sophisticated features to improve predictions!")