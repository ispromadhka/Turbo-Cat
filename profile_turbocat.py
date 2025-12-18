#!/usr/bin/env python3
"""
Deep profiling of TurboCat to identify bottlenecks
"""

import numpy as np
import time
from turbocat import TurboCatClassifier

def profile_component(name, func, iterations=3):
    """Profile a function multiple times"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  {name}: {avg*1000:.2f}ms (±{std*1000:.2f}ms)")
    return avg, result

def analyze_scaling():
    """Analyze how TurboCat scales with data size"""
    print("\n" + "="*70)
    print("SCALING ANALYSIS")
    print("="*70)

    # Test different sample sizes
    sample_sizes = [1000, 5000, 10000, 50000, 100000, 200000]
    n_features = 50
    n_trees = 50

    print(f"\nFixed: {n_features} features, {n_trees} trees")
    print("-" * 70)
    print(f"{'Samples':>10} {'Train(s)':>10} {'Per-tree(ms)':>12} {'Scaling':>10}")
    print("-" * 70)

    prev_time = None
    prev_samples = None

    for n_samples in sample_sizes:
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

        clf = TurboCatClassifier(n_estimators=n_trees, verbose=0)

        start = time.perf_counter()
        clf.fit(X, y)
        elapsed = time.perf_counter() - start

        per_tree = (elapsed / n_trees) * 1000

        if prev_time is not None:
            scaling = (elapsed / prev_time) / (n_samples / prev_samples)
        else:
            scaling = 1.0

        print(f"{n_samples:>10} {elapsed:>10.3f} {per_tree:>12.2f} {scaling:>10.2f}x")

        prev_time = elapsed
        prev_samples = n_samples

def analyze_feature_scaling():
    """Analyze how TurboCat scales with feature count"""
    print("\n" + "="*70)
    print("FEATURE SCALING ANALYSIS")
    print("="*70)

    n_samples = 50000
    feature_counts = [10, 20, 50, 100, 200]
    n_trees = 50

    print(f"\nFixed: {n_samples} samples, {n_trees} trees")
    print("-" * 70)
    print(f"{'Features':>10} {'Train(s)':>10} {'Per-tree(ms)':>12} {'Per-feat(ms)':>12}")
    print("-" * 70)

    for n_features in feature_counts:
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

        clf = TurboCatClassifier(n_estimators=n_trees, verbose=0)

        start = time.perf_counter()
        clf.fit(X, y)
        elapsed = time.perf_counter() - start

        per_tree = (elapsed / n_trees) * 1000
        per_feat = per_tree / n_features

        print(f"{n_features:>10} {elapsed:>10.3f} {per_tree:>12.2f} {per_feat:>12.3f}")

def compare_libraries_detailed():
    """Detailed comparison with breakdown"""
    print("\n" + "="*70)
    print("DETAILED LIBRARY COMPARISON")
    print("="*70)

    try:
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
    except ImportError as e:
        print(f"Missing library: {e}")
        return

    # Large dataset
    n_samples = 100000
    n_features = 50
    n_trees = 100

    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.3 > 0).astype(int)

    print(f"\nDataset: {n_samples} samples, {n_features} features, {n_trees} trees")
    print("-" * 70)

    models = {
        'TurboCat': TurboCatClassifier(n_estimators=n_trees, verbose=0, max_depth=6),
        'CatBoost': CatBoostClassifier(n_estimators=n_trees, verbose=0, max_depth=6, thread_count=-1),
        'LightGBM': LGBMClassifier(n_estimators=n_trees, verbose=-1, max_depth=6, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=n_trees, verbosity=0, max_depth=6, n_jobs=-1),
    }

    results = {}

    for name, model in models.items():
        # Training time
        start = time.perf_counter()
        model.fit(X, y)
        train_time = time.perf_counter() - start

        # Prediction time
        start = time.perf_counter()
        for _ in range(10):
            pred = model.predict(X)
        pred_time = (time.perf_counter() - start) / 10

        # Accuracy
        acc = (pred == y).mean()

        results[name] = {
            'train': train_time,
            'predict': pred_time,
            'accuracy': acc,
            'per_tree_ms': (train_time / n_trees) * 1000
        }

        print(f"\n{name}:")
        print(f"  Training:    {train_time:.3f}s ({results[name]['per_tree_ms']:.2f}ms/tree)")
        print(f"  Prediction:  {pred_time*1000:.2f}ms")
        print(f"  Accuracy:    {acc:.4f}")

    # Analysis
    print("\n" + "-" * 70)
    print("SPEED ANALYSIS:")
    print("-" * 70)

    catboost_time = results['CatBoost']['train']
    for name, r in results.items():
        ratio = r['train'] / catboost_time
        print(f"  {name}: {ratio:.2f}x vs CatBoost")

    print("\n" + "-" * 70)
    print("PER-TREE TIME BREAKDOWN (estimated):")
    print("-" * 70)

    tc_per_tree = results['TurboCat']['per_tree_ms']
    cb_per_tree = results['CatBoost']['per_tree_ms']

    print(f"  TurboCat per tree: {tc_per_tree:.2f}ms")
    print(f"  CatBoost per tree: {cb_per_tree:.2f}ms")
    print(f"  Gap: {tc_per_tree - cb_per_tree:.2f}ms ({tc_per_tree/cb_per_tree:.2f}x)")

    # Estimate where time goes
    # Histogram: O(n_samples * n_features)
    # Split finding: O(n_features * n_bins)
    # Partition: O(n_samples)

    hist_complexity = n_samples * n_features
    split_complexity = n_features * 255
    partition_complexity = n_samples

    total = hist_complexity + split_complexity + partition_complexity

    print(f"\n  Estimated time distribution (per tree):")
    print(f"    Histogram building: ~{hist_complexity/total*100:.1f}%")
    print(f"    Split finding:      ~{split_complexity/total*100:.1f}%")
    print(f"    Sample partition:   ~{partition_complexity/total*100:.1f}%")

def analyze_catboost_settings():
    """Analyze what makes CatBoost fast"""
    print("\n" + "="*70)
    print("CATBOOST SPEED SECRETS ANALYSIS")
    print("="*70)

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("CatBoost not installed")
        return

    n_samples = 100000
    n_features = 50
    n_trees = 100

    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    print(f"\nDataset: {n_samples} samples, {n_features} features")
    print("-" * 70)

    # Test different CatBoost modes
    configs = {
        'Default (symmetric)': {'grow_policy': 'SymmetricTree'},
        'Depthwise': {'grow_policy': 'Depthwise'},
        'Lossguide (leaf-wise)': {'grow_policy': 'Lossguide'},
    }

    for name, params in configs.items():
        model = CatBoostClassifier(
            n_estimators=n_trees,
            verbose=0,
            max_depth=6,
            thread_count=-1,
            **params
        )

        start = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - start

        acc = (model.predict(X) == y).mean()

        print(f"  {name}: {elapsed:.3f}s, accuracy={acc:.4f}")

    print("\n  KEY INSIGHT: CatBoost's SymmetricTree is fastest!")
    print("  Symmetric trees: same split at each depth level")
    print("  This allows vectorized operations and better cache usage")

if __name__ == "__main__":
    print("="*70)
    print("TURBOCAT DEEP PROFILING")
    print("="*70)

    analyze_scaling()
    analyze_feature_scaling()
    compare_libraries_detailed()
    analyze_catboost_settings()

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
Based on analysis, key optimizations for TurboCat:

1. SYMMETRIC TREES (like CatBoost) - BIG IMPACT
   - All nodes at same depth use same feature/threshold
   - Reduces split finding from O(n_features * n_bins * n_nodes)
     to O(n_features * n_bins * depth)
   - Much better cache locality
   - Faster prediction (no tree traversal, just depth checks)

2. EXCLUSIVE FEATURE BUNDLING (like LightGBM)
   - Bundle mutually exclusive sparse features
   - Reduces histogram building cost

3. CACHE-OPTIMIZED DATA LAYOUT
   - Column-major for histogram building
   - Better prefetching

4. REDUCE HISTOGRAM BUILDS
   - Use fewer bins (128 vs 255)
   - Only rebuild histograms when necessary
""")
