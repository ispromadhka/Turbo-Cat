#!/usr/bin/env python3
"""Final benchmark: TurboCat vs CatBoost - Speed and Quality"""

import numpy as np
import time
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, 'build')
from turbocat import TurboCatClassifier

from catboost import CatBoostClassifier

def run_benchmark(X, y, name, n_trees=100, max_depth=6):
    """Run benchmark on a single dataset"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {'name': name, 'n_samples': len(X), 'n_features': X.shape[1]}

    # TurboCat
    clf_tc = TurboCatClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        use_goss=True,
        verbosity=0,
        n_jobs=-1
    )

    start = time.perf_counter()
    clf_tc.fit(X_train, y_train.astype(np.float32))
    results['tc_train'] = time.perf_counter() - start

    start = time.perf_counter()
    tc_proba = clf_tc.predict_proba(X_test)
    results['tc_infer'] = (time.perf_counter() - start) * 1000  # ms

    if tc_proba.ndim == 2:
        tc_pred = (tc_proba[:, 1] > 0.5).astype(int)
        tc_proba_1 = tc_proba[:, 1]
    else:
        tc_pred = (tc_proba > 0.5).astype(int)
        tc_proba_1 = tc_proba

    results['tc_acc'] = accuracy_score(y_test, tc_pred)
    results['tc_f1'] = f1_score(y_test, tc_pred, average='weighted')
    try:
        results['tc_auc'] = roc_auc_score(y_test, tc_proba_1)
    except:
        results['tc_auc'] = 0

    # CatBoost
    clf_cb = CatBoostClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        verbose=0
    )

    start = time.perf_counter()
    clf_cb.fit(X_train, y_train.astype(int))
    results['cb_train'] = time.perf_counter() - start

    start = time.perf_counter()
    cb_proba = clf_cb.predict_proba(X_test)
    results['cb_infer'] = (time.perf_counter() - start) * 1000  # ms

    cb_pred = clf_cb.predict(X_test)
    results['cb_acc'] = accuracy_score(y_test, cb_pred)
    results['cb_f1'] = f1_score(y_test, cb_pred, average='weighted')
    try:
        results['cb_auc'] = roc_auc_score(y_test, cb_proba[:, 1])
    except:
        results['cb_auc'] = 0

    return results

def main():
    print("=" * 100)
    print("ФИНАЛЬНЫЙ БЕНЧМАРК: TurboCat vs CatBoost")
    print("=" * 100)

    datasets = []

    # 1. Breast Cancer (small, binary)
    data = load_breast_cancer()
    datasets.append(("Breast Cancer", data.data.astype(np.float32), data.target))

    # 2. Synthetic small
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    datasets.append(("Synthetic 2K", X.astype(np.float32), y))

    # 3. Synthetic medium
    X, y = make_classification(n_samples=10000, n_features=30, n_informative=15,
                               n_redundant=10, random_state=42)
    datasets.append(("Synthetic 10K", X.astype(np.float32), y))

    # 4. Synthetic large
    X, y = make_classification(n_samples=50000, n_features=50, n_informative=25,
                               n_redundant=15, random_state=42)
    datasets.append(("Synthetic 50K", X.astype(np.float32), y))

    # 5. Imbalanced 95/5
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=10,
                               weights=[0.95, 0.05], random_state=42)
    datasets.append(("Imbalanced 95/5", X.astype(np.float32), y))

    # 6. Imbalanced 99/1
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=10,
                               weights=[0.99, 0.01], random_state=42)
    datasets.append(("Imbalanced 99/1", X.astype(np.float32), y))

    # 7. High-dimensional
    X, y = make_classification(n_samples=3000, n_features=100, n_informative=50,
                               n_redundant=30, random_state=42)
    datasets.append(("High-Dim 100f", X.astype(np.float32), y))

    all_results = []

    for name, X, y in datasets:
        results = run_benchmark(X, y, name)
        all_results.append(results)

    # Print Speed Table
    print("\n" + "=" * 100)
    print("СКОРОСТЬ (Train / Inference)")
    print("=" * 100)
    print(f"{'Датасет':<20} {'Samples':<10} {'TC Train':<12} {'CB Train':<12} {'Speedup':<10} {'TC Infer':<12} {'CB Infer':<12} {'Speedup':<10}")
    print("-" * 100)

    total_tc_train = 0
    total_cb_train = 0
    total_tc_infer = 0
    total_cb_infer = 0

    for r in all_results:
        train_speedup = r['cb_train'] / r['tc_train']
        infer_speedup = r['cb_infer'] / r['tc_infer']

        total_tc_train += r['tc_train']
        total_cb_train += r['cb_train']
        total_tc_infer += r['tc_infer']
        total_cb_infer += r['cb_infer']

        print(f"{r['name']:<20} {r['n_samples']:<10} {r['tc_train']:<12.4f} {r['cb_train']:<12.4f} {train_speedup:<10.2f}x {r['tc_infer']:<12.2f} {r['cb_infer']:<12.2f} {infer_speedup:<10.2f}x")

    print("-" * 100)
    overall_train = total_cb_train / total_tc_train
    overall_infer = total_cb_infer / total_tc_infer
    print(f"{'ИТОГО':<20} {'':<10} {total_tc_train:<12.4f} {total_cb_train:<12.4f} {overall_train:<10.2f}x {total_tc_infer:<12.2f} {total_cb_infer:<12.2f} {overall_infer:<10.2f}x")

    # Print Quality Table
    print("\n" + "=" * 100)
    print("КАЧЕСТВО (Accuracy / F1 / ROC-AUC)")
    print("=" * 100)
    print(f"{'Датасет':<20} {'TC Acc':<10} {'CB Acc':<10} {'TC F1':<10} {'CB F1':<10} {'TC AUC':<10} {'CB AUC':<10} {'Winner':<15}")
    print("-" * 100)

    tc_wins = 0
    cb_wins = 0
    ties = 0

    for r in all_results:
        # Determine winner based on AUC
        if r['tc_auc'] > r['cb_auc'] + 0.001:
            winner = "TurboCat"
            tc_wins += 1
        elif r['cb_auc'] > r['tc_auc'] + 0.001:
            winner = "CatBoost"
            cb_wins += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"{r['name']:<20} {r['tc_acc']:<10.4f} {r['cb_acc']:<10.4f} {r['tc_f1']:<10.4f} {r['cb_f1']:<10.4f} {r['tc_auc']:<10.4f} {r['cb_auc']:<10.4f} {winner:<15}")

    print("-" * 100)

    avg_tc_acc = np.mean([r['tc_acc'] for r in all_results])
    avg_cb_acc = np.mean([r['cb_acc'] for r in all_results])
    avg_tc_f1 = np.mean([r['tc_f1'] for r in all_results])
    avg_cb_f1 = np.mean([r['cb_f1'] for r in all_results])
    avg_tc_auc = np.mean([r['tc_auc'] for r in all_results])
    avg_cb_auc = np.mean([r['cb_auc'] for r in all_results])

    print(f"{'СРЕДНЕЕ':<20} {avg_tc_acc:<10.4f} {avg_cb_acc:<10.4f} {avg_tc_f1:<10.4f} {avg_cb_f1:<10.4f} {avg_tc_auc:<10.4f} {avg_cb_auc:<10.4f}")

    # Summary
    print("\n" + "=" * 100)
    print("ИТОГОВЫЙ ВЕРДИКТ")
    print("=" * 100)
    print(f"\n📊 СКОРОСТЬ:")
    print(f"   • Training:  TurboCat в {overall_train:.2f}x {'БЫСТРЕЕ' if overall_train > 1 else 'медленнее'}")
    print(f"   • Inference: TurboCat в {overall_infer:.2f}x {'БЫСТРЕЕ' if overall_infer > 1 else 'медленнее'}")

    print(f"\n🎯 КАЧЕСТВО:")
    print(f"   • TurboCat побед: {tc_wins}")
    print(f"   • CatBoost побед: {cb_wins}")
    print(f"   • Ничьих: {ties}")
    print(f"   • Средний AUC: TurboCat {avg_tc_auc:.4f} vs CatBoost {avg_cb_auc:.4f} (разница: {(avg_tc_auc - avg_cb_auc)*100:+.2f}%)")

if __name__ == "__main__":
    main()
