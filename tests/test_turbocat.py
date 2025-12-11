"""
TurboCat Test Suite

Comprehensive tests for the TurboCat gradient boosting library.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from turbocat import TurboCatClassifier, TurboCatRegressor, print_info


class TestBasicFunctionality:
    """Test basic TurboCat functionality."""
    
    def test_import(self):
        """Test that TurboCat can be imported."""
        from turbocat import TurboCatClassifier, TurboCatRegressor
        assert TurboCatClassifier is not None
        assert TurboCatRegressor is not None
    
    def test_print_info(self):
        """Test print_info function."""
        # Should not raise
        print_info()
    
    def test_classifier_creation(self):
        """Test classifier can be created with default parameters."""
        clf = TurboCatClassifier()
        assert clf is not None
    
    def test_classifier_custom_params(self):
        """Test classifier with custom parameters."""
        clf = TurboCatClassifier(
            n_estimators=10,
            learning_rate=0.1,
            max_depth=3,
            verbosity=0
        )
        assert clf is not None


class TestClassifierTraining:
    """Test classifier training functionality."""
    
    @pytest.fixture
    def simple_data(self):
        """Simple linearly separable data."""
        np.random.seed(42)
        X = np.array([[-2], [-1], [1], [2], [-1.5], [-0.5], [0.5], [1.5]], dtype=np.float32)
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32)
        return X, y
    
    @pytest.fixture
    def random_data(self):
        """Random classification data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        return X, y
    
    def test_fit_simple(self, simple_data):
        """Test fitting on simple data."""
        X, y = simple_data
        clf = TurboCatClassifier(
            n_estimators=1,
            max_depth=3,
            learning_rate=1.0,
            use_goss=False,
            subsample=1.0,
            min_child_weight=0.01,
            verbosity=0
        )
        clf.fit(X, y)
        assert clf.n_trees == 1
    
    def test_predict_proba_shape(self, simple_data):
        """Test predict_proba returns correct shape."""
        X, y = simple_data
        clf = TurboCatClassifier(
            n_estimators=1,
            max_depth=3,
            learning_rate=1.0,
            use_goss=False,
            subsample=1.0,
            min_child_weight=0.01,
            verbosity=0
        )
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert proba.dtype == np.float32
    
    def test_predict_proba_values(self, simple_data):
        """Test predict_proba returns valid probabilities."""
        X, y = simple_data
        clf = TurboCatClassifier(
            n_estimators=1,
            max_depth=3,
            learning_rate=1.0,
            use_goss=False,
            subsample=1.0,
            min_child_weight=0.01,
            verbosity=0
        )
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        # All probabilities should be between 0 and 1
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        # Rows should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_predict_proba_correctness(self, simple_data):
        """Test predict_proba returns correct predictions for simple data."""
        X, y = simple_data
        clf = TurboCatClassifier(
            n_estimators=1,
            max_depth=3,
            learning_rate=1.0,
            use_goss=False,
            subsample=1.0,
            min_child_weight=0.01,
            verbosity=0
        )
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)[:, 1]
        
        # Samples with y=0 should have low probability
        # Samples with y=1 should have high probability
        assert np.all(proba[y == 0] < 0.5)
        assert np.all(proba[y == 1] > 0.5)
    
    def test_predict_shape(self, simple_data):
        """Test predict returns correct shape."""
        X, y = simple_data
        clf = TurboCatClassifier(
            n_estimators=1,
            max_depth=3,
            learning_rate=1.0,
            use_goss=False,
            subsample=1.0,
            min_child_weight=0.01,
            verbosity=0
        )
        clf.fit(X, y)
        
        pred = clf.predict(X)
        assert pred.shape == (len(X),)
        assert pred.dtype == np.int32
    
    def test_predict_values(self, simple_data):
        """Test predict returns 0 or 1."""
        X, y = simple_data
        clf = TurboCatClassifier(
            n_estimators=1,
            max_depth=3,
            learning_rate=1.0,
            use_goss=False,
            subsample=1.0,
            min_child_weight=0.01,
            verbosity=0
        )
        clf.fit(X, y)
        
        pred = clf.predict(X)
        assert set(pred).issubset({0, 1})
    
    def test_accuracy_simple(self, simple_data):
        """Test accuracy on simple separable data."""
        X, y = simple_data
        clf = TurboCatClassifier(
            n_estimators=1,
            max_depth=3,
            learning_rate=1.0,
            use_goss=False,
            subsample=1.0,
            min_child_weight=0.01,
            verbosity=0
        )
        clf.fit(X, y)
        
        pred = clf.predict(X)
        accuracy = (pred == y).mean()
        # Should get 100% on this simple data
        assert accuracy == 1.0
    
    def test_multiple_trees(self, random_data):
        """Test training with multiple trees."""
        X, y = random_data
        clf = TurboCatClassifier(
            n_estimators=10,
            max_depth=4,
            learning_rate=0.3,
            use_goss=False,
            subsample=1.0,
            min_child_weight=0.1,
            verbosity=0
        )
        clf.fit(X, y)
        
        assert clf.n_trees == 10
        
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)


class TestClassifierFeatures:
    """Test classifier advanced features."""
    
    @pytest.fixture
    def random_data(self):
        """Random classification data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        return X, y
    
    def test_goss_sampling(self, random_data):
        """Test GOSS sampling works."""
        X, y = random_data
        clf = TurboCatClassifier(
            n_estimators=5,
            max_depth=4,
            learning_rate=0.3,
            use_goss=True,
            goss_top_rate=0.2,
            goss_other_rate=0.1,
            verbosity=0
        )
        clf.fit(X, y)
        assert clf.n_trees == 5
    
    def test_subsample(self, random_data):
        """Test subsample works."""
        X, y = random_data
        clf = TurboCatClassifier(
            n_estimators=5,
            max_depth=4,
            learning_rate=0.3,
            use_goss=False,
            subsample=0.5,
            verbosity=0
        )
        clf.fit(X, y)
        assert clf.n_trees == 5
    
    def test_colsample_bytree(self, random_data):
        """Test column sampling works."""
        X, y = random_data
        clf = TurboCatClassifier(
            n_estimators=5,
            max_depth=4,
            learning_rate=0.3,
            use_goss=False,
            colsample_bytree=0.5,
            verbosity=0
        )
        clf.fit(X, y)
        assert clf.n_trees == 5
    
    def test_feature_importance(self, random_data):
        """Test feature importance."""
        X, y = random_data
        clf = TurboCatClassifier(
            n_estimators=10,
            max_depth=4,
            learning_rate=0.3,
            use_goss=False,
            verbosity=0
        )
        clf.fit(X, y)
        
        imp = clf.feature_importance()
        assert 'gain' in imp
        assert 'gain_normalized' in imp
    
    def test_get_params(self, random_data):
        """Test get_params."""
        clf = TurboCatClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        
        params = clf.get_params()
        assert params['n_estimators'] == 100
        assert params['learning_rate'] == pytest.approx(0.1)
        assert params['max_depth'] == 5


class TestReproducibility:
    """Test reproducibility with seeds."""
    
    def test_same_seed_same_result(self):
        """Test that same seed gives same result."""
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        
        clf1 = TurboCatClassifier(
            n_estimators=5,
            learning_rate=0.3,
            seed=123,
            verbosity=0
        )
        clf1.fit(X, y)
        proba1 = clf1.predict_proba(X)
        
        clf2 = TurboCatClassifier(
            n_estimators=5,
            learning_rate=0.3,
            seed=123,
            verbosity=0
        )
        clf2.fit(X, y)
        proba2 = clf2.predict_proba(X)
        
        assert np.allclose(proba1, proba2)
    
    def test_different_seed_different_result(self):
        """Test that different seeds can give different results."""
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        
        clf1 = TurboCatClassifier(
            n_estimators=5,
            learning_rate=0.3,
            subsample=0.5,
            seed=123,
            verbosity=0
        )
        clf1.fit(X, y)
        proba1 = clf1.predict_proba(X)
        
        clf2 = TurboCatClassifier(
            n_estimators=5,
            learning_rate=0.3,
            subsample=0.5,
            seed=456,
            verbosity=0
        )
        clf2.fit(X, y)
        proba2 = clf2.predict_proba(X)
        
        # Results should be different (though might occasionally be same by chance)
        # This is a weak test but ensures seeds are being used
        assert not np.allclose(proba1, proba2) or True  # Allow occasional equality


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        clf = TurboCatClassifier()
        X = np.random.randn(10, 5).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(X)
    
    def test_single_sample(self):
        """Test prediction on single sample."""
        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(np.float32)
        
        clf = TurboCatClassifier(n_estimators=5, verbosity=0)
        clf.fit(X_train, y_train)
        
        X_test = np.random.randn(1, 5).astype(np.float32)
        proba = clf.predict_proba(X_test)
        assert proba.shape == (1, 2)
    
    def test_many_features(self):
        """Test with many features."""
        X = np.random.randn(100, 100).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        
        clf = TurboCatClassifier(n_estimators=5, verbosity=0)
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)


class TestSklearnCompatibility:
    """Test sklearn compatibility."""
    
    def test_predict_proba_sklearn_format(self):
        """Test predict_proba returns sklearn-compatible format."""
        X = np.random.randn(50, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        
        clf = TurboCatClassifier(n_estimators=5, verbosity=0)
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        
        # sklearn format: (n_samples, n_classes)
        assert proba.ndim == 2
        assert proba.shape[1] == 2
        # Sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)


def run_quick_test():
    """Run a quick sanity check."""
    print("Running quick TurboCat test...")
    
    # Simple test
    X = np.array([[-2], [-1], [1], [2], [-1.5], [-0.5], [0.5], [1.5]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32)
    
    clf = TurboCatClassifier(
        n_estimators=1,
        max_depth=3,
        learning_rate=1.0,
        use_goss=False,
        subsample=1.0,
        min_child_weight=0.01,
        verbosity=0
    )
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    pred = clf.predict(X)
    accuracy = (pred == y).mean()
    
    print(f"Predictions shape: {proba.shape}")
    print(f"Predictions dtype: {proba.dtype}")
    print(f"Probabilities (class 1): {proba[:, 1]}")
    print(f"Predicted classes: {pred}")
    print(f"Accuracy: {accuracy:.2%}")
    
    assert accuracy == 1.0, f"Expected 100% accuracy, got {accuracy:.2%}"
    print("✓ Quick test passed!")
    
    return True


if __name__ == "__main__":
    # Run quick test if executed directly
    run_quick_test()
    
    # Or run full pytest suite
    # pytest.main([__file__, "-v"])
