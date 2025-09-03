"""
Advanced Deep Learning models for better prediction accuracy
"""
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Attention
    from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AdvancedDeepLearningModels:
    """State-of-the-art deep learning models for trading"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.models = {}
    
    def create_lstm_attention_model(self, input_shape):
        """LSTM with Attention mechanism - captures long-term dependencies"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM layers with attention
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm1_norm = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(64, return_sequences=True, dropout=0.2)(lstm1_norm)
        lstm2_norm = BatchNormalization()(lstm2)
        
        # Attention mechanism (simplified)
        attention = Dense(64, activation='tanh')(lstm2_norm)
        attention = Dense(1, activation='sigmoid')(attention)
        attention_applied = tf.keras.layers.multiply([lstm2_norm, attention])
        
        # Global pooling and dense layers
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_applied)
        
        dense1 = Dense(32, activation='relu')(pooled)
        dense1_dropout = Dropout(0.3)(dense1)
        dense1_norm = BatchNormalization()(dense1_dropout)
        
        # Output
        outputs = Dense(1, activation='sigmoid')(dense1_norm)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_lstm_hybrid(self, input_shape):
        """CNN + LSTM hybrid - captures local patterns and sequences"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        inputs = Input(shape=input_shape)
        
        # CNN layers for pattern recognition
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        conv1_norm = BatchNormalization()(conv1)
        
        conv2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(conv1_norm)
        conv2_norm = BatchNormalization()(conv2)
        
        # LSTM for sequence learning
        lstm = LSTM(50, return_sequences=False, dropout=0.2)(conv2_norm)
        lstm_norm = BatchNormalization()(lstm)
        
        # Dense layers
        dense1 = Dense(25, activation='relu')(lstm_norm)
        dense1_dropout = Dropout(0.2)(dense1)
        
        outputs = Dense(1, activation='sigmoid')(dense1_dropout)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        
        return model
    
    def create_transformer_model(self, input_shape):
        """Transformer model - state-of-the-art for sequences"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        class MultiHeadSelfAttention(tf.keras.layers.Layer):
            def __init__(self, embed_dim, num_heads=8):
                super(MultiHeadSelfAttention, self).__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                if embed_dim % num_heads != 0:
                    raise ValueError("embed_dim must be divisible by num_heads")
                
                self.projection_dim = embed_dim // num_heads
                self.query_dense = Dense(embed_dim)
                self.key_dense = Dense(embed_dim)
                self.value_dense = Dense(embed_dim)
                self.combine_heads = Dense(embed_dim)
            
            def attention(self, query, key, value):
                score = tf.matmul(query, key, transpose_b=True)
                dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
                scaled_score = score / tf.math.sqrt(dim_key)
                weights = tf.nn.softmax(scaled_score, axis=-1)
                output = tf.matmul(weights, value)
                return output, weights
            
            def separate_heads(self, x, batch_size):
                x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
                return tf.transpose(x, perm=[0, 2, 1, 3])
            
            def call(self, inputs):
                batch_size = tf.shape(inputs)[0]
                query = self.query_dense(inputs)
                key = self.key_dense(inputs)
                value = self.value_dense(inputs)
                
                query = self.separate_heads(query, batch_size)
                key = self.separate_heads(key, batch_size)
                value = self.separate_heads(value, batch_size)
                
                attention, weights = self.attention(query, key, value)
                attention = tf.transpose(attention, perm=[0, 2, 1, 3])
                concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
                output = self.combine_heads(concat_attention)
                
                return output
        
        # Build transformer model
        inputs = Input(shape=input_shape)
        
        # Embedding/projection
        embedded = Dense(64, activation='relu')(inputs)
        
        # Multi-head self-attention
        attention = MultiHeadSelfAttention(64, num_heads=4)(embedded)
        attention = Dropout(0.1)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + embedded)
        
        # Feed forward
        ffn_output = Dense(128, activation='relu')(attention)
        ffn_output = Dense(64)(ffn_output)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention)
        
        # Global pooling and classification
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        dense = Dense(32, activation='relu')(pooled)
        dense = Dropout(0.2)(dense)
        outputs = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_ensemble_deep_model(self, input_shape):
        """Ensemble of different deep learning architectures"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        inputs = Input(shape=input_shape)
        
        # Branch 1: LSTM
        lstm_branch = LSTM(64, return_sequences=False, dropout=0.2)(inputs)
        lstm_branch = Dense(32, activation='relu')(lstm_branch)
        
        # Branch 2: CNN
        cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        cnn_branch = GlobalMaxPooling1D()(cnn_branch)
        cnn_branch = Dense(32, activation='relu')(cnn_branch)
        
        # Branch 3: Simple dense
        dense_branch = Flatten()(inputs)
        dense_branch = Dense(64, activation='relu')(dense_branch)
        dense_branch = Dense(32, activation='relu')(dense_branch)
        
        # Combine branches
        combined = Concatenate()([lstm_branch, cnn_branch, dense_branch])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(32, activation='relu')(combined)
        
        outputs = Dense(1, activation='sigmoid')(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_with_advanced_techniques(self, model, X_train, y_train, X_val, y_val):
        """Advanced training techniques"""
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: lr * 0.95 if epoch > 10 else lr
            )
        ]
        
        # Train with class weights to handle imbalanced data
        class_weights = self._calculate_class_weights(y_train)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def _calculate_class_weights(self, y):
        """Calculate class weights for imbalanced data"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        class_weights = {i: total / (len(unique) * count) for i, count in zip(unique, counts)}
        return class_weights

# Usage instructions
def get_improvement_instructions():
    """Instructions for implementing these improvements"""
    instructions = """
    🚀 HOW TO IMPLEMENT THESE IMPROVEMENTS:
    
    1. PRIORITY ORDER (implement in this order):
    ==========================================
    
    🥇 HIGH IMPACT (Implement First):
    - Fix Overfitting: Use walk-forward validation
    - Feature Selection: Reduce from 65 to 30 best features  
    - Regularization: Add L1/L2 penalties to models
    
    🥈 MEDIUM-HIGH IMPACT:
    - Advanced Features: Add volatility, regime, pattern features
    - Market Regime Detection: Different strategies for bull/bear markets
    - Ensemble Improvements: Better model combination
    
    🥉 MEDIUM IMPACT:
    - Deep Learning: LSTM with attention, CNN-LSTM hybrid
    - Alternative Data: Economic indicators, sentiment data
    
    2. IMPLEMENTATION STEPS:
    =====================
    
    Step 1: Quick Wins (2 hours)
    ----------------------------
    python3 -c "
    from improvements.better_training import improved_model_training
    # Replace current training with regularized models
    "
    
    Step 2: Advanced Features (4 hours)
    ----------------------------------
    python3 -c "
    from improvements.advanced_features import AdvancedFeatureEngineer
    # Add 50+ new features to your pipeline
    "
    
    Step 3: Regime Detection (3 hours)
    ---------------------------------
    python3 -c "
    from improvements.regime_detection import MarketRegimeDetector
    # Adapt strategy to market conditions
    "
    
    Step 4: Deep Learning (6 hours)
    ------------------------------
    # Install TensorFlow: pip3 install tensorflow
    python3 -c "
    from improvements.deep_learning import AdvancedDeepLearningModels
    # Add LSTM with attention
    "
    
    3. EXPECTED IMPROVEMENTS:
    ========================
    
    Current Performance:  55-65% accuracy
    After Improvements:   65-75% accuracy
    
    Current Sharpe Ratio: 0.8-1.2
    After Improvements:   1.2-2.0
    
    4. TESTING:
    ==========
    
    - Run walk-forward validation on 1 year of data
    - Track out-of-sample performance for 30 days
    - Compare Sharpe ratios before/after improvements
    
    Start with Step 1 (regularization) - biggest impact with least effort!
    """
    
    return instructions

if __name__ == "__main__":
    if TF_AVAILABLE:
        print("🚀 Deep Learning Models Ready")
        models = AdvancedDeepLearningModels()
        print("TensorFlow available - can implement advanced models!")
    else:
        print("⚠️ TensorFlow not installed")
        print("Install with: pip3 install tensorflow")
    
    print("\n" + get_improvement_instructions())