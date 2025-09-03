"""
Improved training methods to reduce overfitting and increase real-world accuracy
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def improved_model_training():
    """Better training with proper validation and regularization"""
    
    # 1. WALK-FORWARD VALIDATION
    def walk_forward_validation(X, y, model, window_size=252):
        """
        More realistic validation that simulates real trading
        Train on past data, test on future data
        """
        scores = []
        predictions = []
        actual_values = []
        
        for i in range(window_size, len(X) - 30, 30):  # Move forward 30 days each time
            # Training window
            X_train = X[i-window_size:i]
            y_train = y[i-window_size:i]
            
            # Test window (next 30 days)
            X_test = X[i:i+30]
            y_test = y[i:i+30]
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Store results
            score = np.mean(y_pred == y_test)
            scores.append(score)
            predictions.extend(y_pred)
            actual_values.extend(y_test)
        
        return np.mean(scores), predictions, actual_values
    
    # 2. BETTER MODEL PARAMETERS
    from sklearn.ensemble import RandomForestClassifier
    
    # Regularized Random Forest (prevent overfitting)
    rf_improved = RandomForestClassifier(
        n_estimators=50,        # Fewer trees (was 100)
        max_depth=6,           # Limit tree depth (was unlimited)
        min_samples_split=10,  # Require more samples to split (was 5) 
        min_samples_leaf=5,    # Require more samples per leaf
        max_features=0.3,      # Use fewer features per tree
        bootstrap=True,
        oob_score=True,        # Out-of-bag validation
        random_state=42
    )
    
    # 3. ENSEMBLE WITH DIFFERENT ALGORITHMS
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    models = {
        'rf_regularized': rf_improved,
        'gb_regularized': GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.05,    # Slower learning
            max_depth=4,           # Shallower trees
            subsample=0.8,         # Use 80% of data per tree
            random_state=42
        ),
        'logistic_l2': LogisticRegression(
            C=0.1,                 # Strong regularization
            penalty='l2',
            max_iter=1000,
            random_state=42
        ),
        'svm_rbf': SVC(
            C=0.1,                 # Regularization
            gamma='scale',
            probability=True,      # For ensemble voting
            random_state=42
        )
    }
    
    return models, walk_forward_validation

# 4. FEATURE SELECTION
def select_best_features(X, y, feature_names, top_k=30):
    """Select most predictive features to reduce overfitting"""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    
    # Method 1: Statistical test
    selector_stats = SelectKBest(score_func=f_classif, k=top_k)
    X_selected_stats = selector_stats.fit_transform(X, y)
    selected_features_stats = np.array(feature_names)[selector_stats.get_support()]
    
    # Method 2: Recursive Feature Elimination
    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    selector_rfe = RFE(estimator=rf, n_features_to_select=top_k)
    X_selected_rfe = selector_rfe.fit_transform(X, y)
    selected_features_rfe = np.array(feature_names)[selector_rfe.get_support()]
    
    # Method 3: Feature importance from Random Forest
    rf.fit(X, y)
    feature_importance = rf.feature_importances_
    top_indices = np.argsort(feature_importance)[-top_k:]
    selected_features_importance = np.array(feature_names)[top_indices]
    
    print(f"🔍 Feature Selection Results:")
    print(f"Statistical: {len(selected_features_stats)} features")
    print(f"RFE: {len(selected_features_rfe)} features") 
    print(f"Importance: {len(selected_features_importance)} features")
    
    # Return the intersection (features selected by multiple methods)
    common_features = set(selected_features_stats) & set(selected_features_rfe) & set(selected_features_importance)
    print(f"Common features: {len(common_features)}")
    
    return list(common_features), selector_stats

if __name__ == "__main__":
    print("🎯 Improved Training Methods Ready")
    print("Run this with your data to get better, more realistic accuracy!")