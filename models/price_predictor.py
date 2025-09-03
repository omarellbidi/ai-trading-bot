"""
Machine Learning models for stock price prediction
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Define placeholder classes for type hints
class Sequential:
    pass
class Model:
    pass
class Adam:
    pass
class EarlyStopping:
    pass
class ReduceLROnPlateau:
    pass

try:
    # Scikit-learn models
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
    
    # Try to import deep learning libraries
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, Model, load_model
        from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
        
except ImportError:
    SKLEARN_AVAILABLE = False
    TF_AVAILABLE = False
    logger.warning("ML libraries not available. Install scikit-learn and tensorflow.")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ml_config, asset_config
from utils.logger import logger
from utils.technical_indicators import FeatureEngineer
from data.data_collector import DataCollectionOrchestrator

@dataclass
class ModelPrediction:
    """Model prediction result"""
    symbol: str
    timestamp: datetime
    prediction: int  # 1 for buy, 0 for sell/hold
    confidence: float
    probability: float
    model_name: str

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int

class BasePredictor:
    """Base class for all prediction models"""
    
    def __init__(self, symbol: str, model_name: str):
        self.symbol = symbol
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        engineer = FeatureEngineer()
        features_df, target = engineer.prepare_ml_data(df)
        
        # Store feature columns and selector for later use
        self.feature_columns = features_df.columns.tolist()
        self.feature_selector = getattr(engineer, 'feature_selector', None)
        
        # Ensure we return numpy arrays
        X = features_df.values if hasattr(features_df, 'values') else features_df
        y = target.values if hasattr(target, 'values') else target
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train the model - to be implemented by subclasses"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions - to be implemented by subclasses"""
        raise NotImplementedError
    
    def save_model(self, filepath: str):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_selector': getattr(self, 'feature_selector', None),
            'symbol': self.symbol,
            'model_name': self.model_name
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.feature_selector = model_data.get('feature_selector', None)
            self.symbol = model_data['symbol']
            self.model_name = model_data['model_name']
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}", error=e)

class SklearnPredictor(BasePredictor):
    """Scikit-learn based predictor"""
    
    def __init__(self, symbol: str, model_type: str = "random_forest"):
        super().__init__(symbol, f"sklearn_{model_type}")
        self.model_type = model_type
        
        if SKLEARN_AVAILABLE:
            if model_type == "random_forest":
                self.model = RandomForestClassifier(
                    n_estimators=50,         # Reduced from 100 (fewer trees)
                    max_depth=6,            # Reduced from 10 (shallower trees)
                    min_samples_split=10,   # Increased from 5 (more samples needed)
                    min_samples_leaf=5,     # Added (minimum samples per leaf)
                    max_features=0.3,       # Added (use only 30% of features per tree)
                    oob_score=True,         # Added (out-of-bag validation)
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "gradient_boosting":
                self.model = GradientBoostingClassifier(
                    n_estimators=50,        # Reduced from 100 (fewer trees)
                    learning_rate=0.05,     # Reduced from 0.1 (slower learning)
                    max_depth=4,           # Reduced from 6 (shallower trees)
                    min_samples_split=10,   # Added (more samples needed)
                    min_samples_leaf=5,     # Added (minimum samples per leaf)
                    subsample=0.8,          # Added (use 80% of data per tree)
                    max_features=0.3,       # Added (use fewer features)
                    random_state=42
                )
            elif model_type == "logistic_regression":
                self.model = LogisticRegression(
                    C=0.1,                  # Strong L2 regularization (was default C=1.0)
                    penalty='l2',           # L2 regularization
                    solver='lbfgs',         # Better solver for L2
                    max_iter=1000,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.scaler = StandardScaler()
        else:
            raise ImportError("Scikit-learn not available")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train the sklearn model"""
        logger.info(f"Training {self.model_name} for {self.symbol}")
        
        # Convert to numpy arrays if they're pandas objects
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with time series consideration
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Validate
            y_pred = self.model.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))
        
        # Final training on all data
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate final performance
        y_pred = self.model.predict(X_scaled)
        performance = ModelPerformance(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y, y_pred, average='weighted', zero_division=0),
            f1_score=f1_score(y, y_pred, average='weighted', zero_division=0),
            total_samples=len(y)
        )
        
        logger.info(f"Model training completed. Accuracy: {performance.accuracy:.3f}")
        return performance
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
        
        return predictions, probabilities

class LSTMPredictor(BasePredictor):
    """LSTM-based deep learning predictor"""
    
    def __init__(self, symbol: str):
        super().__init__(symbol, "lstm")
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        self.sequence_length = ml_config.SEQUENCE_LENGTH
        self.scaler = MinMaxScaler()
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM"""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=ml_config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Train the LSTM model"""
        logger.info(f"Training LSTM for {self.symbol}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_scaled, y)
        
        if len(X_seq) < 100:  # Not enough data for LSTM
            logger.warning(f"Not enough data for LSTM training ({len(X_seq)} samples)")
            raise ValueError("Insufficient data for LSTM training")
        
        # Split data
        train_size = int(len(X_seq) * ml_config.TRAIN_TEST_SPLIT)
        X_train, X_val = X_seq[:train_size], X_seq[train_size:]
        y_train, y_val = y_seq[:train_size], y_seq[train_size:]
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=ml_config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=0.0001)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=ml_config.BATCH_SIZE,
            epochs=ml_config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_trained = True
        
        # Calculate performance
        y_pred = (self.model.predict(X_val) > 0.5).astype(int).flatten()
        performance = ModelPerformance(
            accuracy=accuracy_score(y_val, y_pred),
            precision=precision_score(y_val, y_pred, zero_division=0),
            recall=recall_score(y_val, y_pred, zero_division=0),
            f1_score=f1_score(y_val, y_pred, zero_division=0),
            total_samples=len(y_val)
        )
        
        logger.info(f"LSTM training completed. Accuracy: {performance.accuracy:.3f}")
        return performance
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.prepare_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        if len(X_seq) == 0:
            return np.array([]), np.array([])
        
        probabilities = self.model.predict(X_seq).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities

class EnsemblePredictor:
    """Ensemble of multiple prediction models"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.models: Dict[str, BasePredictor] = {}
        self.weights: Dict[str, float] = {}
        self.is_trained = False
    
    def add_model(self, predictor: BasePredictor, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[predictor.model_name] = predictor
        self.weights[predictor.model_name] = weight
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, ModelPerformance]:
        """Train all models in the ensemble"""
        logger.info(f"Training ensemble for {self.symbol}")
        performances = {}
        
        for name, model in self.models.items():
            try:
                performance = model.train(X, y)
                performances[name] = performance
                
                # Adjust weight based on performance
                self.weights[name] *= performance.f1_score
                
            except Exception as e:
                logger.error(f"Failed to train {name}", error=e)
                # Remove failed model
                self.weights.pop(name, None)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            self.is_trained = True
        
        logger.info(f"Ensemble training completed with {len(self.weights)} models")
        return performances
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        weighted_predictions = []
        weighted_probabilities = []
        total_weight = 0
        
        for name, model in self.models.items():
            if name in self.weights:
                try:
                    predictions, probabilities = model.predict(X)
                    if len(predictions) > 0:
                        weight = self.weights[name]
                        weighted_predictions.append(predictions[-1] * weight)
                        weighted_probabilities.append(probabilities[-1] * weight)
                        total_weight += weight
                except Exception as e:
                    logger.error(f"Prediction failed for {name}", error=e)
        
        if total_weight == 0:
            return ModelPrediction(
                symbol=self.symbol,
                timestamp=datetime.now(),
                prediction=0,
                confidence=0.0,
                probability=0.5,
                model_name="ensemble"
            )
        
        # Aggregate predictions
        final_probability = sum(weighted_probabilities) / total_weight
        final_prediction = 1 if final_probability > 0.5 else 0
        confidence = abs(final_probability - 0.5) * 2  # Convert to 0-1 scale
        
        return ModelPrediction(
            symbol=self.symbol,
            timestamp=datetime.now(),
            prediction=final_prediction,
            confidence=confidence,
            probability=final_probability,
            model_name="ensemble"
        )

class ModelManager:
    """Manage all prediction models"""
    
    def __init__(self):
        self.models: Dict[str, EnsemblePredictor] = {}
        self.data_orchestrator = DataCollectionOrchestrator()
        os.makedirs("models/saved_models", exist_ok=True)
    
    def create_ensemble_for_symbol(self, symbol: str) -> EnsemblePredictor:
        """Create ensemble model for a symbol"""
        ensemble = EnsemblePredictor(symbol)
        
        # Add sklearn models
        if SKLEARN_AVAILABLE:
            try:
                rf_model = SklearnPredictor(symbol, "random_forest")
                gb_model = SklearnPredictor(symbol, "gradient_boosting")
                lr_model = SklearnPredictor(symbol, "logistic_regression")
                
                ensemble.add_model(rf_model, weight=1.0)
                ensemble.add_model(gb_model, weight=1.0)
                ensemble.add_model(lr_model, weight=0.8)
            except Exception as e:
                logger.error(f"Failed to add sklearn models for {symbol}", error=e)
        
        # Add LSTM model if enough data
        if TF_AVAILABLE:
            try:
                lstm_model = LSTMPredictor(symbol)
                ensemble.add_model(lstm_model, weight=1.2)
            except Exception as e:
                logger.error(f"Failed to add LSTM model for {symbol}", error=e)
        
        return ensemble
    
    def train_model_for_symbol(self, symbol: str, min_samples: int = 200):
        """Train prediction model for a specific symbol"""
        logger.info(f"Training models for {symbol}")
        
        # Get historical data
        df = self.data_orchestrator.get_latest_data(symbol, limit=1000)
        
        if len(df) < min_samples:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} samples")
            return None
        
        # Create ensemble
        ensemble = self.create_ensemble_for_symbol(symbol)
        
        # Prepare data
        try:
            engineer = FeatureEngineer()
            X, y = engineer.prepare_ml_data(df)
            
            # Transfer feature selector to all models in ensemble
            feature_selector = getattr(engineer, 'feature_selector', None)
            for model in ensemble.models.values():
                model.feature_selector = feature_selector
            
            # Train ensemble
            performances = ensemble.train(X, y)
            
            # Store model
            self.models[symbol] = ensemble
            
            # Save to disk
            model_path = f"models/saved_models/{symbol}_ensemble.pkl"
            joblib.dump(ensemble, model_path)
            
            logger.info(f"Model training completed for {symbol}")
            return performances
            
        except Exception as e:
            logger.error(f"Failed to train model for {symbol}", error=e)
            return None
    
    def get_prediction(self, symbol: str) -> Optional[ModelPrediction]:
        """Get prediction for a symbol"""
        if symbol not in self.models:
            # Try to load from disk
            model_path = f"models/saved_models/{symbol}_ensemble.pkl"
            try:
                self.models[symbol] = joblib.load(model_path)
            except:
                logger.warning(f"No trained model found for {symbol}")
                return None
        
        # Get latest data
        df = self.data_orchestrator.get_latest_data(symbol, limit=100)
        if df.empty:
            return None
        
        # Prepare features using consistent feature engineering pipeline
        engineer = FeatureEngineer()
        
        # Create full feature set
        features_df = engineer.create_features(df)
        
        # Select numeric features only
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_columns]
        
        # Remove rows with NaN values
        valid_idx = ~features_df.isnull().any(axis=1)
        features_df = features_df[valid_idx]
        
        if len(features_df) == 0:
            return None
        
        # Apply feature selection if the model has a feature selector
        ensemble = self.models[symbol]
        if hasattr(ensemble, 'models') and ensemble.models:
            # Get feature selector from first model
            first_model = next(iter(ensemble.models.values()))
            if hasattr(first_model, 'feature_selector') and first_model.feature_selector is not None:
                features_df = engineer.apply_feature_selection(features_df, first_model.feature_selector)
        
        # Convert to numpy array
        X = features_df.values
        
        # Make prediction
        try:
            prediction = self.models[symbol].predict(X)
            return prediction
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}", error=e)
            return None
    
    def train_all_models(self):
        """Train models for all tracked symbols"""
        logger.info("Starting model training for all symbols")
        
        from config.settings import get_all_symbols
        symbols = get_all_symbols()
        results = {}
        
        for symbol in symbols:
            try:
                performances = self.train_model_for_symbol(symbol)
                results[symbol] = performances
            except Exception as e:
                logger.error(f"Failed to train model for {symbol}", error=e)
                results[symbol] = None
        
        logger.info("Completed model training for all symbols")
        return results

if __name__ == "__main__":
    # Test the model manager
    manager = ModelManager()
    
    # Train a model for AAPL
    test_symbol = "AAPL"
    performances = manager.train_model_for_symbol(test_symbol)
    
    if performances:
        print(f"Model performances for {test_symbol}:")
        for model_name, perf in performances.items():
            print(f"  {model_name}: Accuracy={perf.accuracy:.3f}, F1={perf.f1_score:.3f}")
        
        # Get a prediction
        prediction = manager.get_prediction(test_symbol)
        if prediction:
            print(f"\nPrediction: {prediction.prediction} (confidence: {prediction.confidence:.3f})")
    else:
        print("Failed to train model")