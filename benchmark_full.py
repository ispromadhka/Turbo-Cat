#!/usr/bin/env python3
"""
Full TurboCat Benchmark with comprehensive metrics
Compares: TurboCat, CatBoost, LightGBM, XGBoost
Metrics: ROC-AUC, PR-AUC, Precision, Recall, F1, Accuracy, Time
"""

import numpy as np
import time
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from turbocat import TurboCatClassifier

def get_models(n_trees=100, max_depth=6):
    """Get all models with comparable settings"""
    models = {
        'TurboCat': TurboCatClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=0.1,  # Optimal learning rate
            verbose=0,
            use_symmetric=False,
            use_leaf_wise=True
        )
    }

    try:
        from catboost import CatBoostClassifier
        models['CatBoost'] = CatBoostClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            verbose=0,
            thread_count=-1
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        models['LightGBM'] = LGBMClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            verbose=-1,
            n_jobs=-1
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            verbosity=0,
            n_jobs=-1
        )
    except ImportError:
        pass

    return models

def compute_metrics(y_true, y_pred, y_proba):
    """Compute comprehensive classification metrics"""
    metrics = {}

    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # ROC-AUC and PR-AUC (for binary classification)
    try:
        if y_proba.ndim == 1 or y_proba.shape[1] == 2:
            proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics['ROC-AUC'] = roc_auc_score(y_true, proba)
            metrics['PR-AUC'] = average_precision_score(y_true, proba)
        else:
            # Multiclass
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            metrics['PR-AUC'] = 0.0  # Not directly applicable
    except Exception as e:
        metrics['ROC-AUC'] = 0.0
        metrics['PR-AUC'] = 0.0

    return metrics

def generate_dataset(n_samples, n_features, noise=0.3, n_informative=None, seed=42):
    """Generate a challenging dataset"""
    np.random.seed(seed)

    if n_informative is None:
        n_informative = max(2, n_features // 5)

    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create complex decision boundary
    y = np.zeros(n_samples, dtype=np.float32)

    # Linear component
    weights = np.random.randn(n_informative)
    y += X[:, :n_informative] @ weights

    # Non-linear interactions
    for i in range(min(3, n_informative)):
        for j in range(i+1, min(5, n_informative)):
            y += 0.5 * X[:, i] * X[:, j]

    # Add noise
    y += noise * np.random.randn(n_samples)

    # Convert to binary
    y = (y > np.median(y)).astype(np.float32)

    return X, y

def benchmark_classification():
    """Run comprehensive classification benchmark"""
    print("=" * 100)
    print("COMPREHENSIVE CLASSIFICATION BENCHMARK")
    print("=" * 100)

    test_cases = [
        # (n_samples, n_features, n_trees, description)
        (5000, 20, 100, "Small"),
        (10000, 50, 100, "Medium"),
        (50000, 50, 100, "Large"),
        (100000, 100, 100, "Very Large"),
    ]

    all_results = []

    for n_samples, n_features, n_trees, desc in test_cases:
        print(f"\n{'='*100}")
        print(f"Dataset: {desc} ({n_samples} samples, {n_features} features, {n_trees} trees)")
        print("="*100)

        # Generate data
        X, y = generate_dataset(n_samples, n_features)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = get_models(n_trees=n_trees)
        results = {}

        # Print header
        print(f"\n{'Model':<12} {'Time(s)':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 100)

        for name, model in models.items():
            try:
                # Training
                start = time.perf_counter()
                if name == 'CatBoost':
                    model.fit(X_train, y_train.astype(int), verbose=False)
                else:
                    model.fit(X_train, y_train if name == 'TurboCat' else y_train.astype(int))
                train_time = time.perf_counter() - start

                # Prediction
                y_pred = model.predict(X_test)
                if name == 'CatBoost':
                    y_pred = y_pred.flatten()
                y_proba = model.predict_proba(X_test)

                # Compute metrics
                metrics = compute_metrics(y_test, y_pred, y_proba)
                metrics['Time'] = train_time

                results[name] = metrics

                print(f"{name:<12} {train_time:<10.3f} {metrics['ROC-AUC']:<10.4f} {metrics['PR-AUC']:<10.4f} "
                      f"{metrics['Accuracy']:<10.4f} {metrics['Precision']:<10.4f} {metrics['Recall']:<10.4f} {metrics['F1']:<10.4f}")

            except Exception as e:
                print(f"{name:<12} ERROR: {str(e)[:50]}")

        # Analysis
        if 'TurboCat' in results and 'CatBoost' in results:
            tc = results['TurboCat']
            cb = results['CatBoost']

            print(f"\n--- TurboCat vs CatBoost ---")
            speedup = cb['Time'] / tc['Time'] if tc['Time'] > 0 else 0
            auc_diff = tc['ROC-AUC'] - cb['ROC-AUC']

            print(f"  Speed: {'%.2fx faster' % speedup if speedup > 1 else '%.2fx slower' % (1/speedup)}")
            print(f"  ROC-AUC: {'+' if auc_diff >= 0 else ''}{auc_diff:.4f}")

        all_results.append({'desc': desc, 'n_samples': n_samples, 'results': results})

    # Final summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    if any('TurboCat' in r['results'] and 'CatBoost' in r['results'] for r in all_results):
        print(f"\n{'Dataset':<15} {'TC Time':<10} {'CB Time':<10} {'Speedup':<10} {'TC AUC':<10} {'CB AUC':<10} {'AUC Diff':<10}")
        print("-" * 85)

        total_tc_time = 0
        total_cb_time = 0
        tc_wins_speed = 0
        tc_wins_auc = 0

        for r in all_results:
            if 'TurboCat' in r['results'] and 'CatBoost' in r['results']:
                tc = r['results']['TurboCat']
                cb = r['results']['CatBoost']

                speedup = cb['Time'] / tc['Time'] if tc['Time'] > 0 else 0
                auc_diff = tc['ROC-AUC'] - cb['ROC-AUC']

                total_tc_time += tc['Time']
                total_cb_time += cb['Time']

                if speedup > 1:
                    tc_wins_speed += 1
                if auc_diff >= 0:
                    tc_wins_auc += 1

                speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.2f}x slower"
                print(f"{r['desc']:<15} {tc['Time']:<10.3f} {cb['Time']:<10.3f} {speedup_str:<10} "
                      f"{tc['ROC-AUC']:<10.4f} {cb['ROC-AUC']:<10.4f} {auc_diff:+.4f}")

        print("-" * 85)
        avg_speedup = total_cb_time / total_tc_time if total_tc_time > 0 else 0
        print(f"\nOverall: TurboCat is {avg_speedup:.2f}x {'faster' if avg_speedup > 1 else 'slower'} than CatBoost")
        print(f"Speed wins: {tc_wins_speed}/{len(all_results)}")
        print(f"AUC wins: {tc_wins_auc}/{len(all_results)}")

def benchmark_with_real_patterns():
    """Benchmark with more realistic data patterns"""
    print("\n" + "=" * 100)
    print("REALISTIC PATTERN BENCHMARK")
    print("=" * 100)

    np.random.seed(42)

    # Create dataset with realistic patterns
    n_samples = 50000
    n_features = 30

    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Add some categorical-like features (binned)
    for i in range(5):
        X[:, i] = np.digitize(X[:, i], bins=[-2, -1, 0, 1, 2])

    # Create complex target
    y = (
        2 * X[:, 0] +
        1.5 * X[:, 1] * X[:, 2] +
        np.sin(X[:, 3] * np.pi) +
        0.5 * (X[:, 4] > 0).astype(float) * X[:, 5] +
        0.3 * np.random.randn(n_samples)
    )
    y = (y > np.median(y)).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = get_models(n_trees=200, max_depth=8)

    print(f"\nDataset: {n_samples} samples, {n_features} features (mixed numeric/categorical-like)")
    print(f"\n{'Model':<12} {'Time(s)':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<10}")
    print("-" * 60)

    for name, model in models.items():
        try:
            start = time.perf_counter()
            if name == 'CatBoost':
                model.fit(X_train, y_train.astype(int), verbose=False)
            else:
                model.fit(X_train, y_train if name == 'TurboCat' else y_train.astype(int))
            train_time = time.perf_counter() - start

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            metrics = compute_metrics(y_test, y_pred, y_proba)

            print(f"{name:<12} {train_time:<10.3f} {metrics['ROC-AUC']:<10.4f} {metrics['PR-AUC']:<10.4f} {metrics['F1']:<10.4f}")
        except Exception as e:
            print(f"{name:<12} ERROR: {e}")

if __name__ == "__main__":
    benchmark_classification()
    benchmark_with_real_patterns()

    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)
