#!/usr/bin/env python3
"""
Comprehensive benchmark: TurboCat vs CatBoost
Compare on various datasets with identical hyperparameters
"""

import numpy as np
import time
import warnings
from sklearn.datasets import (
    load_iris, load_breast_cancer, load_digits, load_wine,
    make_classification, make_moons, make_circles
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Import models
from turbocat import TurboCatClassifier
from catboost import CatBoostClassifier

# Common hyperparameters
COMMON_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
}

def get_turbocat_model():
    return TurboCatClassifier(
        n_estimators=COMMON_PARAMS['n_estimators'],
        learning_rate=COMMON_PARAMS['learning_rate'],
        max_depth=COMMON_PARAMS['max_depth'],
        verbosity=0,
        use_goss=True
    )

def get_catboost_model():
    return CatBoostClassifier(
        iterations=COMMON_PARAMS['n_estimators'],
        learning_rate=COMMON_PARAMS['learning_rate'],
        depth=COMMON_PARAMS['max_depth'],
        verbose=False,
        allow_writing_files=False
    )

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model, return metrics"""
    # Training time
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    # Inference time
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start

    # Get probabilities for ROC-AUC
    try:
        y_proba = model.predict_proba(X_test)
        if len(y_proba.shape) == 2:
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            else:
                y_proba = None  # Multi-class
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) == 2 else None
    except:
        roc_auc = None

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc,
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'train_time': train_time,
        'inference_time': inference_time,
    }

    return metrics

def run_benchmark(name, X, y, test_size=0.2):
    """Run benchmark on a dataset"""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {np.unique(y)}, Distribution: {np.bincount(y.astype(int))}")
    print('='*60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to float32 for TurboCat
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    results = {}

    # TurboCat
    try:
        tc_model = get_turbocat_model()
        results['TurboCat'] = evaluate_model(tc_model, X_train, X_test, y_train, y_test, 'TurboCat')
    except Exception as e:
        print(f"TurboCat error: {e}")
        results['TurboCat'] = None

    # CatBoost
    try:
        cb_model = get_catboost_model()
        results['CatBoost'] = evaluate_model(cb_model, X_train, X_test, y_train, y_test, 'CatBoost')
    except Exception as e:
        print(f"CatBoost error: {e}")
        results['CatBoost'] = None

    # Print results
    print(f"\n{'Metric':<15} {'TurboCat':>12} {'CatBoost':>12} {'Winner':>12}")
    print('-' * 55)

    metrics_to_compare = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall', 'train_time', 'inference_time']

    for metric in metrics_to_compare:
        tc_val = results['TurboCat'][metric] if results['TurboCat'] else None
        cb_val = results['CatBoost'][metric] if results['CatBoost'] else None

        if tc_val is None and cb_val is None:
            continue

        tc_str = f"{tc_val:.4f}" if tc_val is not None else "N/A"
        cb_str = f"{cb_val:.4f}" if cb_val is not None else "N/A"

        # Determine winner (higher is better for metrics, lower for time)
        if tc_val is not None and cb_val is not None:
            if metric in ['train_time', 'inference_time']:
                winner = "TurboCat" if tc_val < cb_val else "CatBoost"
                speedup = cb_val / tc_val if tc_val > 0 else 0
                winner_str = f"{winner} ({speedup:.1f}x)"
            else:
                winner = "TurboCat" if tc_val > cb_val else ("CatBoost" if cb_val > tc_val else "Tie")
                diff = abs(tc_val - cb_val) * 100
                winner_str = f"{winner} (+{diff:.2f}%)"
        else:
            winner_str = "N/A"

        print(f"{metric:<15} {tc_str:>12} {cb_str:>12} {winner_str:>15}")

    return results

def main():
    print("="*60)
    print("TurboCat vs CatBoost Comprehensive Benchmark")
    print(f"Hyperparameters: {COMMON_PARAMS}")
    print("="*60)

    all_results = {}

    # =========================================================================
    # 1. sklearn standard datasets
    # =========================================================================
    print("\n\n" + "#"*60)
    print("# SKLEARN STANDARD DATASETS")
    print("#"*60)

    # Iris (small, multi-class)
    iris = load_iris()
    all_results['iris'] = run_benchmark('Iris (small, 3-class)', iris.data, iris.target)

    # Breast Cancer (binary, medical)
    bc = load_breast_cancer()
    all_results['breast_cancer'] = run_benchmark('Breast Cancer (binary)', bc.data, bc.target)

    # Wine (small, multi-class)
    wine = load_wine()
    all_results['wine'] = run_benchmark('Wine (small, 3-class)', wine.data, wine.target)

    # Digits (larger, multi-class)
    digits = load_digits()
    all_results['digits'] = run_benchmark('Digits (multi-class, 64 features)', digits.data, digits.target)

    # =========================================================================
    # 2. Synthetic datasets - varying sizes
    # =========================================================================
    print("\n\n" + "#"*60)
    print("# SYNTHETIC - VARYING SIZES")
    print("#"*60)

    for n_samples in [500, 2000, 10000]:
        X, y = make_classification(
            n_samples=n_samples, n_features=20, n_informative=15,
            n_redundant=3, n_clusters_per_class=2, random_state=42
        )
        all_results[f'size_{n_samples}'] = run_benchmark(
            f'Synthetic (n={n_samples}, 20 features)', X, y
        )

    # =========================================================================
    # 3. Imbalanced datasets
    # =========================================================================
    print("\n\n" + "#"*60)
    print("# IMBALANCED DATASETS")
    print("#"*60)

    imbalance_ratios = [0.3, 0.15, 0.05, 0.01]

    for ratio in imbalance_ratios:
        X, y = make_classification(
            n_samples=5000, n_features=20, n_informative=15,
            n_redundant=3, weights=[1-ratio, ratio],
            flip_y=0, random_state=42
        )
        pct = int(ratio * 100)
        all_results[f'imbalanced_{pct}'] = run_benchmark(
            f'Imbalanced ({100-pct}/{pct})', X, y
        )

    # =========================================================================
    # 4. High-dimensional datasets
    # =========================================================================
    print("\n\n" + "#"*60)
    print("# HIGH-DIMENSIONAL DATASETS")
    print("#"*60)

    # Many features, few informative
    X, y = make_classification(
        n_samples=2000, n_features=200, n_informative=20,
        n_redundant=10, random_state=42
    )
    all_results['high_dim_sparse'] = run_benchmark(
        'High-dim sparse (200f, 20 informative)', X, y
    )

    # Many features, many informative
    X, y = make_classification(
        n_samples=2000, n_features=100, n_informative=80,
        n_redundant=10, random_state=42
    )
    all_results['high_dim_dense'] = run_benchmark(
        'High-dim dense (100f, 80 informative)', X, y
    )

    # =========================================================================
    # 5. Non-linear datasets
    # =========================================================================
    print("\n\n" + "#"*60)
    print("# NON-LINEAR DATASETS")
    print("#"*60)

    # Moons
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
    all_results['moons'] = run_benchmark('Moons (non-linear)', X, y)

    # Circles
    X, y = make_circles(n_samples=2000, noise=0.1, factor=0.5, random_state=42)
    all_results['circles'] = run_benchmark('Circles (non-linear)', X, y)

    # =========================================================================
    # 6. Noisy datasets
    # =========================================================================
    print("\n\n" + "#"*60)
    print("# NOISY DATASETS")
    print("#"*60)

    for noise in [0.05, 0.1, 0.2]:
        X, y = make_classification(
            n_samples=3000, n_features=20, n_informative=15,
            n_redundant=3, flip_y=noise, random_state=42
        )
        all_results[f'noise_{int(noise*100)}'] = run_benchmark(
            f'Noisy labels ({int(noise*100)}% flip)', X, y
        )

    # =========================================================================
    # 7. Datasets with correlations
    # =========================================================================
    print("\n\n" + "#"*60)
    print("# CORRELATED FEATURES")
    print("#"*60)

    # Highly correlated features
    np.random.seed(42)
    n_samples = 3000
    base_features = np.random.randn(n_samples, 5)
    # Create correlated features
    correlated = np.hstack([
        base_features,
        base_features + np.random.randn(n_samples, 5) * 0.1,  # High correlation
        base_features + np.random.randn(n_samples, 5) * 0.5,  # Medium correlation
    ])
    y = ((base_features[:, 0] + base_features[:, 1]) > 0).astype(int)
    all_results['correlated'] = run_benchmark(
        'Correlated features (15f, high correlation)', correlated, y
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    tc_wins = {'accuracy': 0, 'roc_auc': 0, 'f1': 0, 'train_time': 0, 'inference_time': 0}
    cb_wins = {'accuracy': 0, 'roc_auc': 0, 'f1': 0, 'train_time': 0, 'inference_time': 0}
    total_tests = {'accuracy': 0, 'roc_auc': 0, 'f1': 0, 'train_time': 0, 'inference_time': 0}

    tc_metrics = {'accuracy': [], 'roc_auc': [], 'f1': [], 'train_time': [], 'inference_time': []}
    cb_metrics = {'accuracy': [], 'roc_auc': [], 'f1': [], 'train_time': [], 'inference_time': []}

    for name, result in all_results.items():
        if result is None:
            continue
        tc = result.get('TurboCat')
        cb = result.get('CatBoost')

        if tc is None or cb is None:
            continue

        for metric in ['accuracy', 'roc_auc', 'f1']:
            if tc[metric] is not None and cb[metric] is not None:
                tc_metrics[metric].append(tc[metric])
                cb_metrics[metric].append(cb[metric])
                total_tests[metric] += 1
                if tc[metric] > cb[metric]:
                    tc_wins[metric] += 1
                elif cb[metric] > tc[metric]:
                    cb_wins[metric] += 1

        for metric in ['train_time', 'inference_time']:
            tc_metrics[metric].append(tc[metric])
            cb_metrics[metric].append(cb[metric])
            total_tests[metric] += 1
            if tc[metric] < cb[metric]:
                tc_wins[metric] += 1
            else:
                cb_wins[metric] += 1

    print(f"\n{'Metric':<15} {'TC Wins':>10} {'CB Wins':>10} {'Total':>10} {'TC Avg':>12} {'CB Avg':>12}")
    print("-" * 70)

    for metric in ['accuracy', 'roc_auc', 'f1', 'train_time', 'inference_time']:
        tc_avg = np.mean(tc_metrics[metric]) if tc_metrics[metric] else 0
        cb_avg = np.mean(cb_metrics[metric]) if cb_metrics[metric] else 0
        print(f"{metric:<15} {tc_wins[metric]:>10} {cb_wins[metric]:>10} {total_tests[metric]:>10} {tc_avg:>12.4f} {cb_avg:>12.4f}")

    # Speed comparison
    print("\n" + "-" * 70)
    print("Speed Summary:")
    if tc_metrics['train_time'] and cb_metrics['train_time']:
        train_speedup = np.mean(cb_metrics['train_time']) / np.mean(tc_metrics['train_time'])
        print(f"  Training:  TurboCat is {train_speedup:.2f}x {'faster' if train_speedup > 1 else 'slower'}")
    if tc_metrics['inference_time'] and cb_metrics['inference_time']:
        inf_speedup = np.mean(cb_metrics['inference_time']) / np.mean(tc_metrics['inference_time'])
        print(f"  Inference: TurboCat is {inf_speedup:.2f}x {'faster' if inf_speedup > 1 else 'slower'}")

if __name__ == '__main__':
    main()
