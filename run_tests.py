#!/usr/bin/env python3
"""
Quick test script for TurboCat.

Run from the turbocat directory:
    python3 run_tests.py
"""

import numpy as np
import sys

# Try to import turbocat
try:
    import turbocat
    from turbocat import TurboCatClassifier
    print(f"✓ TurboCat imported successfully (version {turbocat.__version__})")
except ImportError as e:
    print(f"✗ Failed to import TurboCat: {e}")
    sys.exit(1)

def test_basic():
    """Basic functionality test."""
    print("\n" + "="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
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
    
    print(f"  Shape: {proba.shape}")
    print(f"  Dtype: {proba.dtype}")
    print(f"  Probabilities: {proba[:, 1].round(3)}")
    print(f"  Predictions: {pred}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    assert proba.shape == (8, 2), f"Wrong shape: {proba.shape}"
    assert accuracy == 1.0, f"Expected 100% accuracy"
    print("✓ PASSED")
    return True

def test_random_data():
    """Test on larger random dataset."""
    print("\n" + "="*60)
    print("TEST 2: Random Data (100 samples, 10 features)")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    
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
    
    pred = clf.predict(X)
    accuracy = (pred == y).mean()
    
    print(f"  Trees: {clf.n_trees}")
    print(f"  Train accuracy: {accuracy:.2%}")
    
    assert clf.n_trees == 10, f"Wrong number of trees: {clf.n_trees}"
    assert accuracy > 0.8, f"Accuracy too low: {accuracy:.2%}"
    print("✓ PASSED")
    return True

def test_sklearn_format():
    """Test sklearn-compatible format."""
    print("\n" + "="*60)
    print("TEST 3: Sklearn Compatibility")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(50, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    
    clf = TurboCatClassifier(n_estimators=5, verbosity=0)
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    
    # Check format
    is_2d = proba.ndim == 2
    has_2_cols = proba.shape[1] == 2
    sums_to_one = np.allclose(proba.sum(axis=1), 1.0)
    all_valid = np.all((proba >= 0) & (proba <= 1))
    
    print(f"  2D array: {is_2d}")
    print(f"  2 columns: {has_2_cols}")
    print(f"  Rows sum to 1: {sums_to_one}")
    print(f"  All valid probabilities: {all_valid}")
    
    assert is_2d and has_2_cols and sums_to_one and all_valid
    print("✓ PASSED")
    return True

def test_goss():
    """Test GOSS sampling."""
    print("\n" + "="*60)
    print("TEST 4: GOSS Sampling")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    
    clf = TurboCatClassifier(
        n_estimators=5,
        max_depth=4,
        use_goss=True,
        goss_top_rate=0.2,
        goss_other_rate=0.1,
        verbosity=0
    )
    clf.fit(X, y)
    
    pred = clf.predict(X)
    accuracy = (pred == y).mean()
    
    print(f"  Trees: {clf.n_trees}")
    print(f"  Train accuracy: {accuracy:.2%}")
    
    assert clf.n_trees == 5
    print("✓ PASSED")
    return True

def test_feature_importance():
    """Test feature importance."""
    print("\n" + "="*60)
    print("TEST 5: Feature Importance")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)  # Only features 0,1 matter
    
    clf = TurboCatClassifier(n_estimators=20, verbosity=0)
    clf.fit(X, y)
    
    imp = clf.feature_importance()
    
    print(f"  Has 'gain': {'gain' in imp}")
    print(f"  Has 'gain_normalized': {'gain_normalized' in imp}")
    
    assert 'gain' in imp
    assert 'gain_normalized' in imp
    print("✓ PASSED")
    return True

def main():
    print("="*60)
    print("TurboCat Test Suite")
    print("="*60)
    
    tests = [
        test_basic,
        test_random_data,
        test_sklearn_format,
        test_goss,
        test_feature_importance,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 All tests passed! TurboCat is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
