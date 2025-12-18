#!/usr/bin/env python3
"""
Benchmark TurboCat with Symmetric Trees vs CatBoost
"""

import numpy as np
import time

# Setup path
import sys
sys.path.insert(0, 'build')

from turbocat import TurboCatClassifier

def benchmark_comparison():
    """Compare TurboCat Symmetric Trees vs CatBoost"""
    print("=" * 70)
    print("SYMMETRIC TREES BENCHMARK: TurboCat vs CatBoost")
    print("=" * 70)

    try:
        from catboost import CatBoostClassifier
        has_catboost = True
    except ImportError:
        print("CatBoost not installed, skipping comparison")
        has_catboost = False

    # Test on different dataset sizes
    test_cases = [
        (10000, 50, 100),   # Small
        (50000, 50, 100),   # Medium
        (100000, 50, 100),  # Large
        (200000, 50, 100),  # Very large
    ]

    print(f"\n{'Dataset':<20} {'TurboCat(s)':<12} {'CatBoost(s)':<12} {'Speedup':<10} {'TC Acc':<10} {'CB Acc':<10}")
    print("-" * 70)

    results = []

    for n_samples, n_features, n_trees in test_cases:
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.3 > 0).astype(np.float32)

        # TurboCat with Symmetric Trees (default now)
        clf_tc = TurboCatClassifier(
            n_estimators=n_trees,
            max_depth=6,
            verbose=0
        )

        start = time.perf_counter()
        clf_tc.fit(X, y)
        tc_time = time.perf_counter() - start
        tc_acc = (clf_tc.predict(X) == y).mean()

        # CatBoost
        if has_catboost:
            clf_cb = CatBoostClassifier(
                n_estimators=n_trees,
                max_depth=6,
                verbose=0,
                thread_count=-1
            )

            start = time.perf_counter()
            clf_cb.fit(X, y)
            cb_time = time.perf_counter() - start
            cb_acc = (clf_cb.predict(X) == y).mean()

            speedup = cb_time / tc_time
        else:
            cb_time = 0
            cb_acc = 0
            speedup = 0

        results.append({
            'n_samples': n_samples,
            'tc_time': tc_time,
            'cb_time': cb_time,
            'speedup': speedup,
            'tc_acc': tc_acc,
            'cb_acc': cb_acc
        })

        dataset_desc = f"{n_samples}x{n_features}"
        if has_catboost:
            speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x slower"
            print(f"{dataset_desc:<20} {tc_time:<12.3f} {cb_time:<12.3f} {speedup_str:<10} {tc_acc:<10.4f} {cb_acc:<10.4f}")
        else:
            print(f"{dataset_desc:<20} {tc_time:<12.3f} {'N/A':<12} {'N/A':<10} {tc_acc:<10.4f} {'N/A':<10}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if has_catboost:
        avg_speedup = np.mean([r['speedup'] for r in results])
        avg_tc_acc = np.mean([r['tc_acc'] for r in results])
        avg_cb_acc = np.mean([r['cb_acc'] for r in results])

        print(f"\nAverage speedup: {avg_speedup:.2f}x")
        print(f"Average TurboCat accuracy: {avg_tc_acc:.4f}")
        print(f"Average CatBoost accuracy: {avg_cb_acc:.4f}")

        if avg_speedup > 1:
            print(f"\n✓ TurboCat is {avg_speedup:.2f}x FASTER than CatBoost!")
        else:
            print(f"\n✗ TurboCat is {1/avg_speedup:.2f}x slower than CatBoost")

        if avg_tc_acc >= avg_cb_acc:
            print(f"✓ TurboCat accuracy is equal or better than CatBoost!")
        else:
            diff = (avg_cb_acc - avg_tc_acc) * 100
            print(f"  TurboCat accuracy is {diff:.2f}% lower than CatBoost")

def test_tree_modes():
    """Compare different tree building modes"""
    print("\n" + "=" * 70)
    print("TREE MODE COMPARISON")
    print("=" * 70)

    n_samples = 50000
    n_features = 50
    n_trees = 100

    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.3 > 0).astype(np.float32)

    print(f"\nDataset: {n_samples} samples, {n_features} features, {n_trees} trees")
    print("-" * 70)

    modes = [
        ("Symmetric (CatBoost-style)", True, False),
        ("Leaf-wise (LightGBM-style)", False, True),
        ("Level-wise (XGBoost-style)", False, False),
    ]

    print(f"{'Mode':<30} {'Time(s)':<12} {'Per-tree(ms)':<15} {'Accuracy':<10}")
    print("-" * 70)

    for name, use_symmetric, use_leaf_wise in modes:
        clf = TurboCatClassifier(
            n_estimators=n_trees,
            max_depth=6,
            verbose=0,
            use_symmetric=use_symmetric,
            use_leaf_wise=use_leaf_wise
        )

        start = time.perf_counter()
        clf.fit(X, y)
        elapsed = time.perf_counter() - start

        acc = (clf.predict(X) == y).mean()
        per_tree = (elapsed / n_trees) * 1000

        print(f"{name:<30} {elapsed:<12.3f} {per_tree:<15.2f} {acc:<10.4f}")

if __name__ == "__main__":
    test_tree_modes()
    benchmark_comparison()
