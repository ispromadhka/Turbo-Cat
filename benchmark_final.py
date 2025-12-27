"""
TurboCat vs CatBoost - FINAL Comprehensive Benchmark
All quality and speed metrics on multiple datasets
"""

import numpy as np
import time
import sys
sys.path.insert(0, 'build')

from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, log_loss
)
import warnings
warnings.filterwarnings('ignore')

import _turbocat as tc
from catboost import CatBoostClassifier

def create_datasets():
    datasets = []

    # 1. Breast Cancer (small)
    X, y = load_breast_cancer(return_X_y=True)
    datasets.append({
        'X': X.astype(np.float32), 'y': y.astype(np.float32),
        'name': 'Breast Cancer', 'samples': len(X), 'features': X.shape[1]
    })

    # 2. Small synthetic
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=15,
                               n_redundant=3, random_state=42)
    datasets.append({
        'X': X.astype(np.float32), 'y': y.astype(np.float32),
        'name': 'Small (5K)', 'samples': 5000, 'features': 20
    })

    # 3. Medium
    X, y = make_classification(n_samples=20000, n_features=30, n_informative=20,
                               n_redundant=5, random_state=42)
    datasets.append({
        'X': X.astype(np.float32), 'y': y.astype(np.float32),
        'name': 'Medium (20K)', 'samples': 20000, 'features': 30
    })

    # 4. Large
    X, y = make_classification(n_samples=50000, n_features=40, n_informative=25,
                               n_redundant=10, random_state=42)
    datasets.append({
        'X': X.astype(np.float32), 'y': y.astype(np.float32),
        'name': 'Large (50K)', 'samples': 50000, 'features': 40
    })

    # 5. Imbalanced
    X, y = make_classification(n_samples=30000, n_features=25, n_informative=15,
                               n_redundant=5, weights=[0.9, 0.1], random_state=42)
    datasets.append({
        'X': X.astype(np.float32), 'y': y.astype(np.float32),
        'name': 'Imbalanced', 'samples': 30000, 'features': 25
    })

    # 6. High-dimensional
    X, y = make_classification(n_samples=10000, n_features=100, n_informative=50,
                               n_redundant=20, random_state=42)
    datasets.append({
        'X': X.astype(np.float32), 'y': y.astype(np.float32),
        'name': 'High-Dim', 'samples': 10000, 'features': 100
    })

    return datasets

def benchmark_tc(X_train, y_train, X_test, y_test, n_runs=3):
    train_times, inf_times = [], []

    for _ in range(n_runs):
        model = tc.TurboCatClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            mode="small",  # Regular trees for best quality (symmetric slower to train)
            verbosity=0
        )
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - start)

        _ = model.predict_proba(X_test[:100])
        start = time.perf_counter()
        proba = model.predict_proba(X_test)
        inf_times.append(time.perf_counter() - start)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    return {
        'auc': roc_auc_score(y_test, proba[:, 1]),
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'logloss': log_loss(y_test, proba[:, 1]),
        'train_time': np.median(train_times),
        'inf_time': np.median(inf_times) * 1000
    }

def benchmark_cb(X_train, y_train, X_test, y_test, n_runs=3):
    train_times, inf_times = [], []

    for _ in range(n_runs):
        model = CatBoostClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            verbose=False,
            thread_count=-1
        )
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - start)

        _ = model.predict_proba(X_test[:100])
        start = time.perf_counter()
        proba = model.predict_proba(X_test)
        inf_times.append(time.perf_counter() - start)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    return {
        'auc': roc_auc_score(y_test, proba[:, 1]),
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'logloss': log_loss(y_test, proba[:, 1]),
        'train_time': np.median(train_times),
        'inf_time': np.median(inf_times) * 1000
    }

def main():
    print("=" * 100)
    print(" TURBOCAT vs CATBOOST - FINAL COMPREHENSIVE BENCHMARK")
    print("=" * 100)
    print("\nParameters: n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8")

    datasets = create_datasets()
    all_results = []
    total_wins = {'quality': {'TC': 0, 'CB': 0}, 'speed': {'TC': 0, 'CB': 0}}

    for data in datasets:
        print(f"\n{'='*100}")
        print(f" {data['name']} ({data['samples']:,} samples, {data['features']} features)")
        print("=" * 100)

        X_train, X_test, y_train, y_test = train_test_split(
            data['X'], data['y'], test_size=0.2, random_state=42, stratify=data['y']
        )

        print("  Benchmarking TurboCat...", end=" ", flush=True)
        tc_res = benchmark_tc(X_train, y_train, X_test, y_test)
        print(f"done ({tc_res['train_time']:.2f}s)")

        print("  Benchmarking CatBoost...", end=" ", flush=True)
        cb_res = benchmark_cb(X_train, y_train, X_test, y_test)
        print(f"done ({cb_res['train_time']:.2f}s)")

        # Quality metrics
        print(f"\n  {'QUALITY METRICS':-^70}")
        print(f"  {'Metric':<15} {'TurboCat':>12} {'CatBoost':>12} {'Diff':>10} {'Winner':>10}")
        print(f"  {'-'*60}")

        quality_wins = {'TC': 0, 'CB': 0}
        for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
            tc_val = tc_res[metric]
            cb_val = cb_res[metric]
            diff = (tc_val - cb_val) * 100
            winner = 'TC' if tc_val >= cb_val else 'CB'
            quality_wins[winner] += 1
            label = {'auc': 'ROC-AUC', 'accuracy': 'Accuracy', 'precision': 'Precision',
                     'recall': 'Recall', 'f1': 'F1'}[metric]
            print(f"  {label:<15} {tc_val:>12.4f} {cb_val:>12.4f} {diff:>+9.2f}% {winner:>10}")

        # LogLoss (lower is better)
        tc_ll, cb_ll = tc_res['logloss'], cb_res['logloss']
        diff = (cb_ll - tc_ll) / cb_ll * 100
        winner = 'TC' if tc_ll <= cb_ll else 'CB'
        quality_wins[winner] += 1
        print(f"  {'LogLoss':<15} {tc_ll:>12.4f} {cb_ll:>12.4f} {diff:>+9.2f}% {winner:>10}")

        # Speed metrics
        print(f"\n  {'SPEED METRICS':-^70}")
        speed_wins = {'TC': 0, 'CB': 0}

        # Train (lower is better)
        tc_train, cb_train = tc_res['train_time'], cb_res['train_time']
        ratio = cb_train / tc_train
        winner = 'TC' if tc_train <= cb_train else 'CB'
        speed_wins[winner] += 1
        print(f"  {'Train Time':<15} {tc_train:>11.3f}s {cb_train:>11.3f}s {ratio:>9.2f}x {winner:>10}")

        # Inference (lower is better)
        tc_inf, cb_inf = tc_res['inf_time'], cb_res['inf_time']
        ratio = cb_inf / tc_inf
        winner = 'TC' if tc_inf <= cb_inf else 'CB'
        speed_wins[winner] += 1
        print(f"  {'Inference':<15} {tc_inf:>10.2f}ms {cb_inf:>10.2f}ms {ratio:>9.2f}x {winner:>10}")

        # Throughput
        tc_train_tput = data['samples'] * 0.8 / tc_res['train_time']
        cb_train_tput = data['samples'] * 0.8 / cb_res['train_time']
        tc_inf_tput = data['samples'] * 0.2 / (tc_res['inf_time'] / 1000)
        cb_inf_tput = data['samples'] * 0.2 / (cb_res['inf_time'] / 1000)
        print(f"  {'Train samples/s':<15} {tc_train_tput:>11,.0f} {cb_train_tput:>11,.0f}")
        print(f"  {'Inf samples/s':<15} {tc_inf_tput:>11,.0f} {cb_inf_tput:>11,.0f}")

        print(f"\n  Summary: Quality TC={quality_wins['TC']}/6, Speed TC={speed_wins['TC']}/2")

        total_wins['quality']['TC'] += quality_wins['TC']
        total_wins['quality']['CB'] += quality_wins['CB']
        total_wins['speed']['TC'] += speed_wins['TC']
        total_wins['speed']['CB'] += speed_wins['CB']

        all_results.append({
            'name': data['name'],
            'tc_auc': tc_res['auc'],
            'cb_auc': cb_res['auc'],
            'tc_train': tc_res['train_time'],
            'cb_train': cb_res['train_time'],
            'tc_inf': tc_res['inf_time'],
            'cb_inf': cb_res['inf_time']
        })

    # Final Summary
    print("\n" + "=" * 100)
    print(" FINAL SUMMARY")
    print("=" * 100)

    print(f"\n{'Dataset':<20} {'TC AUC':>10} {'CB AUC':>10} {'AUC Diff':>10} {'Train':>10} {'Inference':>12}")
    print("-" * 80)
    for r in all_results:
        auc_diff = (r['tc_auc'] - r['cb_auc']) * 100
        train_ratio = r['cb_train'] / r['tc_train']
        inf_ratio = r['cb_inf'] / r['tc_inf']
        winner_sym = '✓' if r['tc_auc'] >= r['cb_auc'] else ' '
        print(f"{r['name']:<20} {r['tc_auc']:>10.4f} {r['cb_auc']:>10.4f} {auc_diff:>+9.2f}% {train_ratio:>9.2f}x {inf_ratio:>11.2f}x {winner_sym}")

    print("-" * 80)
    print(f"\nTOTAL WINS ACROSS ALL DATASETS:")
    print(f"  Quality Metrics:  TurboCat = {total_wins['quality']['TC']}, CatBoost = {total_wins['quality']['CB']}")
    print(f"  Speed Metrics:    TurboCat = {total_wins['speed']['TC']}, CatBoost = {total_wins['speed']['CB']}")

    tc_total = total_wins['quality']['TC'] + total_wins['speed']['TC']
    cb_total = total_wins['quality']['CB'] + total_wins['speed']['CB']
    print(f"\n  OVERALL SCORE:    TurboCat = {tc_total}, CatBoost = {cb_total}")

    if tc_total > cb_total:
        print("\n  " + "🏆" * 10)
        print("  🏆  TURBOCAT WINS!  🏆")
        print("  " + "🏆" * 10)
    elif cb_total > tc_total:
        print("\n  CatBoost wins this round.")
    else:
        print("\n  It's a tie!")

if __name__ == "__main__":
    main()
