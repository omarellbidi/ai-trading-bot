"""
Market Regime Detection - Different strategies for different market conditions
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MarketRegimeDetector:
    """Detect different market regimes and adapt trading strategy"""
    
    def __init__(self):
        self.regime_models = {}
        self.current_regime = "unknown"
    
    def detect_regimes(self, df):
        """Identify market regimes: Bull, Bear, Sideways, High Volatility"""
        
        # Calculate regime indicators
        returns = df['close_price'].pct_change()
        volatility = returns.rolling(20).std()
        trend = df['close_price'].rolling(50).mean().pct_change(10)
        volume_trend = df['volume'].rolling(20).mean().pct_change(10)
        
        # Create regime features
        regime_features = pd.DataFrame({
            'returns_20d': returns.rolling(20).mean(),
            'volatility_20d': volatility,
            'trend_strength': trend,
            'volume_trend': volume_trend,
            'price_momentum': df['close_price'].pct_change(5),
            'volatility_regime': (volatility > volatility.rolling(100).quantile(0.7)).astype(int)
        }).dropna()
        
        if len(regime_features) < 50:
            return df  # Not enough data
        
        # Use Gaussian Mixture Model to identify regimes
        gmm = GaussianMixture(n_components=4, random_state=42)
        regimes = gmm.fit_predict(regime_features)
        
        # Assign regime names based on characteristics
        regime_names = self._assign_regime_names(regime_features, regimes)
        
        # Add regime to dataframe
        df_with_regimes = df.copy()
        df_with_regimes['regime'] = 'unknown'
        df_with_regimes.iloc[-len(regimes):, df_with_regimes.columns.get_loc('regime')] = regime_names
        
        # Add regime-specific features
        df_with_regimes = self._add_regime_features(df_with_regimes)
        
        return df_with_regimes
    
    def _assign_regime_names(self, features, regimes):
        """Assign meaningful names to detected regimes"""
        regime_names = []
        
        for regime in regimes:
            regime_data = features[regimes == regime]
            
            avg_returns = regime_data['returns_20d'].mean()
            avg_volatility = regime_data['volatility_20d'].mean()
            avg_trend = regime_data['trend_strength'].mean()
            
            if avg_returns > 0.02 and avg_trend > 0.01:
                name = "bull_market"
            elif avg_returns < -0.02 and avg_trend < -0.01:
                name = "bear_market"  
            elif avg_volatility > features['volatility_20d'].quantile(0.7):
                name = "high_volatility"
            else:
                name = "sideways_market"
            
            regime_names.append(name)
        
        return regime_names
    
    def _add_regime_features(self, df):
        """Add regime-specific features"""
        df['is_bull_market'] = (df['regime'] == 'bull_market').astype(int)
        df['is_bear_market'] = (df['regime'] == 'bear_market').astype(int) 
        df['is_high_volatility'] = (df['regime'] == 'high_volatility').astype(int)
        df['is_sideways'] = (df['regime'] == 'sideways_market').astype(int)
        
        return df
    
    def create_regime_specific_models(self, X, y, regimes):
        """Train separate models for each market regime"""
        from sklearn.ensemble import RandomForestClassifier
        
        regime_models = {}
        
        for regime_name in np.unique(regimes):
            if regime_name == 'unknown':
                continue
                
            # Get data for this regime
            regime_mask = regimes == regime_name
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            if len(X_regime) < 20:  # Not enough data for this regime
                continue
            
            # Train regime-specific model
            model = RandomForestClassifier(
                n_estimators=30,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            )
            
            model.fit(X_regime, y_regime)
            regime_models[regime_name] = model
            
            print(f"📊 Trained model for {regime_name}: {len(X_regime)} samples")
        
        return regime_models
    
    def predict_with_regime(self, X, current_regime):
        """Make predictions using regime-specific model"""
        if current_regime in self.regime_models:
            return self.regime_models[current_regime].predict(X)
        else:
            # Fall back to general model
            if 'general' in self.regime_models:
                return self.regime_models['general'].predict(X)
            else:
                return np.array([0])  # Default prediction

class AdaptiveStrategy:
    """Adapt trading parameters based on market regime"""
    
    def __init__(self):
        self.regime_params = {
            'bull_market': {
                'position_size_multiplier': 1.2,    # Larger positions
                'stop_loss_pct': 0.03,              # 3% stop loss
                'take_profit_pct': 0.08,            # 8% take profit
                'signal_threshold': 0.6,            # Lower threshold (more trades)
                'max_positions': 12                 # More positions
            },
            'bear_market': {
                'position_size_multiplier': 0.6,    # Smaller positions
                'stop_loss_pct': 0.02,              # 2% stop loss (tighter)
                'take_profit_pct': 0.04,            # 4% take profit (take profits faster)
                'signal_threshold': 0.8,            # Higher threshold (fewer trades)
                'max_positions': 6                  # Fewer positions
            },
            'high_volatility': {
                'position_size_multiplier': 0.4,    # Much smaller positions
                'stop_loss_pct': 0.04,              # 4% stop loss (wider for volatility)
                'take_profit_pct': 0.06,            # 6% take profit
                'signal_threshold': 0.85,           # Very high threshold
                'max_positions': 4                  # Very few positions
            },
            'sideways_market': {
                'position_size_multiplier': 0.8,    # Normal positions
                'stop_loss_pct': 0.025,             # 2.5% stop loss
                'take_profit_pct': 0.05,            # 5% take profit
                'signal_threshold': 0.7,            # Medium threshold
                'max_positions': 8                  # Medium positions
            }
        }
    
    def get_regime_parameters(self, regime):
        """Get trading parameters for current regime"""
        return self.regime_params.get(regime, self.regime_params['sideways_market'])
    
    def adjust_signal_strength(self, base_signal, regime, confidence):
        """Adjust signal strength based on market regime"""
        params = self.get_regime_parameters(regime)
        
        # In volatile markets, require higher confidence
        if regime == 'high_volatility':
            adjusted_confidence = confidence * 0.8  # Reduce confidence
        elif regime == 'bull_market':
            adjusted_confidence = confidence * 1.1  # Increase confidence for bull signals
        else:
            adjusted_confidence = confidence
        
        # Apply threshold
        if adjusted_confidence > params['signal_threshold']:
            return base_signal * params['position_size_multiplier']
        else:
            return 0  # No signal
    
    def get_dynamic_stop_loss(self, entry_price, regime, side='long'):
        """Calculate dynamic stop loss based on regime"""
        params = self.get_regime_parameters(regime)
        stop_loss_pct = params['stop_loss_pct']
        
        if side == 'long':
            return entry_price * (1 - stop_loss_pct)
        else:  # short
            return entry_price * (1 + stop_loss_pct)

if __name__ == "__main__":
    print("🎯 Market Regime Detection System Ready")
    detector = MarketRegimeDetector()
    strategy = AdaptiveStrategy()
    print("This will adapt your trading strategy to market conditions!")