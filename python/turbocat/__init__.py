"""
TurboCat: Next-Generation Gradient Boosting Framework

A high-performance gradient boosting library designed to outperform CatBoost.

Key innovations:
- GradTree: Gradient-based global tree optimization (not greedy splitting)
- Advanced losses: Robust Focal, LDAM, Logit-adjusted, Tsallis
- Tsallis entropy splitting criterion  
- Cross-validated target statistics for categoricals
- 3-bit gradient quantization
- SIMD-optimized histogram building (AVX2/AVX-512)

Example:
    >>> from turbocat import TurboCatClassifier
    >>> clf = TurboCatClassifier(n_estimators=1000, learning_rate=0.05)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict_proba(X_test)
"""

__version__ = "0.1.0"

import numpy as np

try:
    from ._turbocat import (
        TurboCatClassifier as _TurboCatClassifier,
        TurboCatRegressor as _TurboCatRegressor,
        print_info,
    )
    
    # Wrap TurboCatClassifier to return numpy arrays
    class TurboCatClassifier(_TurboCatClassifier):
        """
        TurboCat Gradient Boosting Classifier.
        
        A high-performance gradient boosting classifier with advanced features
        including GOSS sampling, multiple loss functions, and SIMD optimization.
        
        Parameters
        ----------
        n_estimators : int, default=1000
            Number of boosting iterations.
        learning_rate : float, default=0.05
            Step size shrinkage to prevent overfitting.
        max_depth : int, default=6
            Maximum depth of trees.
        max_bins : int, default=255
            Maximum number of bins for histogram building.
        subsample : float, default=0.8
            Fraction of samples used per iteration.
        colsample_bytree : float, default=0.8
            Fraction of features used per tree.
        min_child_weight : float, default=1.0
            Minimum sum of hessians in a leaf.
        lambda_l2 : float, default=1.0
            L2 regularization term.
        loss : str, default='logloss'
            Loss function: 'logloss', 'focal', 'ldam', 'logit_adjusted', 'tsallis'.
        use_goss : bool, default=True
            Use Gradient-based One-Side Sampling.
        goss_top_rate : float, default=0.2
            Fraction of top gradients to keep in GOSS.
        goss_other_rate : float, default=0.1
            Fraction of other samples to keep in GOSS.
        use_gradtree : bool, default=False
            Use GradTree (gradient-based tree optimization).
        early_stopping_rounds : int, default=50
            Stop if no improvement for this many rounds.
        n_threads : int, default=-1
            Number of threads (-1 = all available).
        seed : int, default=42
            Random seed for reproducibility.
        verbosity : int, default=1
            Verbosity level (0 = silent, 1 = progress).
        """
        
        def predict_proba(self, X):
            """
            Predict class probabilities for X.
            
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.
                
            Returns
            -------
            proba : ndarray of shape (n_samples, 2)
                Class probabilities. Column 0 is P(y=0), column 1 is P(y=1).
            """
            # Get probabilities from C++ (returns list)
            proba_positive = np.array(super().predict_proba(X), dtype=np.float32)
            # Return 2-column array for sklearn compatibility
            proba_negative = 1.0 - proba_positive
            return np.column_stack([proba_negative, proba_positive])
        
        def predict(self, X, threshold=0.5):
            """
            Predict class labels for X.
            
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.
            threshold : float, default=0.5
                Decision threshold.
                
            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted class labels.
            """
            proba = self.predict_proba(X)[:, 1]
            return (proba >= threshold).astype(np.int32)
    
    # Wrap TurboCatRegressor similarly
    class TurboCatRegressor(_TurboCatRegressor):
        """
        TurboCat Gradient Boosting Regressor.
        
        Parameters
        ----------
        n_estimators : int, default=1000
            Number of boosting iterations.
        learning_rate : float, default=0.05
            Step size shrinkage.
        max_depth : int, default=6
            Maximum depth of trees.
        loss : str, default='mse'
            Loss function: 'mse', 'mae', 'huber'.
        seed : int, default=42
            Random seed.
        verbosity : int, default=1
            Verbosity level.
        """
        
        def predict(self, X):
            """
            Predict target values for X.
            
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.
                
            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted values.
            """
            return np.array(super().predict(X), dtype=np.float32)

except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import C++ extensions: {e}. "
        "Please ensure TurboCat is properly built and installed."
    )
    
    # Provide fallback stubs
    class TurboCatClassifier:
        """Stub - C++ extension not available"""
        def __init__(self, *args, **kwargs):
            raise ImportError("TurboCat C++ extension not built")
    
    class TurboCatRegressor:
        """Stub - C++ extension not available"""
        def __init__(self, *args, **kwargs):
            raise ImportError("TurboCat C++ extension not built")
    
    def print_info():
        print("TurboCat (Python fallback - C++ not available)")


__all__ = [
    "TurboCatClassifier",
    "TurboCatRegressor", 
    "print_info",
    "__version__",
]
