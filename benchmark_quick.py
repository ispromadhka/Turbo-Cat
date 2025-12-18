#!/usr/bin/env python3
"""Quick benchmark with realistic data patterns"""

import numpy as np
import time
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, 'build')
from turbocat import TurboCatClassifier

def benchmark():
    print("=" * 80)
    print("QUICK BENCHMARK: TurboCat vs CatBoost vs LightGBM")
    print("=" * 80)

    try:
        from catboost import CatBoostClassifier
        has_catboost = True
    except ImportError:
        has_catboost = False

    try:
        from lightgbm import LGBMClassifier
        has_lightgbm = True
    except ImportError:
        has_lightgbm = False

    # Test cases: (n_samples, n_features, n_trees, max_depth, description)
    test_cases = [
        (20000, 30, 100, 6, "Small (20K x 30)"),
        (50000, 50, 100, 6, "Medium (50K x 50)"),
        (100000, 50, 100, 6, "Large (100K x 50)"),
        (200000, 50, 100, 6, "Very Large (200K x 50)"),
    ]

    results = []

    for n_samples, n_features, n_trees, max_depth, desc in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Dataset: {desc}")
        print(f"{'-' * 80}")

        # Generate realistic data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)

        # Create target with realistic pattern
        y = (
            2.0 * X[:, 0] +
            1.5 * X[:, 1] * X[:, 2] +
            np.sin(X[:, 3] * np.pi) +
            0.5 * (X[:, 4] > 0).astype(float) * X[:, 5] +
            0.3 * np.random.randn(n_samples)
        )
        y = (y > np.median(y)).astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"{'Model':<15} {'Train(s)':<12} {'AUC':<10} {'Accuracy':<10}")
        print("-" * 50)

        row = {'desc': desc}

        # TurboCat
        clf_tc = TurboCatClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=0.1,  # Explicitly set for quality
            verbose=0,
            use_leaf_wise=True,
            use_symmetric=False
        )
        start = time.perf_counter()
        clf_tc.fit(X_train, y_train)
        tc_time = time.perf_counter() - start
        tc_proba = clf_tc.predict_proba(X_test)[:, 1]
        tc_auc = roc_auc_score(y_test, tc_proba)
        tc_acc = accuracy_score(y_test, (tc_proba > 0.5).astype(int))
        print(f"{'TurboCat':<15} {tc_time:<12.3f} {tc_auc:<10.4f} {tc_acc:<10.4f}")
        row['tc_time'] = tc_time
        row['tc_auc'] = tc_auc

        # CatBoost
        if has_catboost:
            clf_cb = CatBoostClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                verbose=0,
                thread_count=-1
            )
            start = time.perf_counter()
            clf_cb.fit(X_train, y_train.astype(int))
            cb_time = time.perf_counter() - start
            cb_proba = clf_cb.predict_proba(X_test)[:, 1]
            cb_auc = roc_auc_score(y_test, cb_proba)
            cb_acc = accuracy_score(y_test, (cb_proba > 0.5).astype(int))
            print(f"{'CatBoost':<15} {cb_time:<12.3f} {cb_auc:<10.4f} {cb_acc:<10.4f}")
            row['cb_time'] = cb_time
            row['cb_auc'] = cb_auc

        # LightGBM
        if has_lightgbm:
            clf_lgb = LGBMClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                verbose=-1,
                n_jobs=-1
            )
            start = time.perf_counter()
            clf_lgb.fit(X_train, y_train.astype(int))
            lgb_time = time.perf_counter() - start
            lgb_proba = clf_lgb.predict_proba(X_test)[:, 1]
            lgb_auc = roc_auc_score(y_test, lgb_proba)
            lgb_acc = accuracy_score(y_test, (lgb_proba > 0.5).astype(int))
            print(f"{'LightGBM':<15} {lgb_time:<12.3f} {lgb_auc:<10.4f} {lgb_acc:<10.4f}")
            row['lgb_time'] = lgb_time
            row['lgb_auc'] = lgb_auc

        results.append(row)

        # Comparison
        if has_catboost:
            speedup = cb_time / tc_time
            auc_diff = tc_auc - cb_auc
            print(f"\nTurboCat vs CatBoost: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}, AUC diff: {auc_diff:+.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if has_catboost:
        total_tc = sum(r['tc_time'] for r in results)
        total_cb = sum(r['cb_time'] for r in results)
        avg_speedup = total_cb / total_tc

        wins_speed = sum(1 for r in results if r['tc_time'] < r['cb_time'])
        wins_auc = sum(1 for r in results if r['tc_auc'] >= r['cb_auc'])

        print(f"Overall speedup vs CatBoost: {avg_speedup:.2f}x")
        print(f"Speed wins: {wins_speed}/{len(results)}")
        print(f"AUC wins/ties: {wins_auc}/{len(results)}")

        if has_lightgbm:
            total_lgb = sum(r['lgb_time'] for r in results)
            lgb_speedup = total_lgb / total_tc
            print(f"Overall speedup vs LightGBM: {lgb_speedup:.2f}x")

if __name__ == "__main__":
    benchmark()
