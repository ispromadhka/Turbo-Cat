#!/usr/bin/env python3
"""
Comprehensive Benchmark: TurboCat vs CatBoost
Classification AND Regression
"""

import numpy as np
import time
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_breast_cancer, make_classification, make_regression,
    fetch_california_housing
)
import warnings
warnings.filterwarnings('ignore')

from turbocat import TurboCatClassifier, TurboCatRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


def benchmark_classification(X, y, name, n_trees=100, max_depth=6):
    """Run classification benchmark"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {'name': name, 'task': 'Classification', 'n_samples': len(X), 'n_features': X.shape[1]}

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
    results['tc_infer'] = (time.perf_counter() - start) * 1000

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
        verbose=0,
        thread_count=-1
    )

    start = time.perf_counter()
    clf_cb.fit(X_train, y_train.astype(int))
    results['cb_train'] = time.perf_counter() - start

    start = time.perf_counter()
    cb_proba = clf_cb.predict_proba(X_test)
    results['cb_infer'] = (time.perf_counter() - start) * 1000

    cb_pred = clf_cb.predict(X_test)
    results['cb_acc'] = accuracy_score(y_test, cb_pred)
    results['cb_f1'] = f1_score(y_test, cb_pred, average='weighted')
    try:
        results['cb_auc'] = roc_auc_score(y_test, cb_proba[:, 1])
    except:
        results['cb_auc'] = 0

    return results


def benchmark_regression(X, y, name, n_trees=100, max_depth=6):
    """Run regression benchmark"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {'name': name, 'task': 'Regression', 'n_samples': len(X), 'n_features': X.shape[1]}

    # TurboCat
    reg_tc = TurboCatRegressor(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        use_goss=True,
        verbosity=0,
        n_jobs=-1
    )

    start = time.perf_counter()
    reg_tc.fit(X_train.astype(np.float32), y_train.astype(np.float32))
    results['tc_train'] = time.perf_counter() - start

    start = time.perf_counter()
    tc_pred = reg_tc.predict(X_test.astype(np.float32))
    results['tc_infer'] = (time.perf_counter() - start) * 1000

    results['tc_mse'] = mean_squared_error(y_test, tc_pred)
    results['tc_mae'] = mean_absolute_error(y_test, tc_pred)
    results['tc_r2'] = r2_score(y_test, tc_pred)

    # CatBoost
    reg_cb = CatBoostRegressor(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        verbose=0,
        thread_count=-1
    )

    start = time.perf_counter()
    reg_cb.fit(X_train, y_train)
    results['cb_train'] = time.perf_counter() - start

    start = time.perf_counter()
    cb_pred = reg_cb.predict(X_test)
    results['cb_infer'] = (time.perf_counter() - start) * 1000

    results['cb_mse'] = mean_squared_error(y_test, cb_pred)
    results['cb_mae'] = mean_absolute_error(y_test, cb_pred)
    results['cb_r2'] = r2_score(y_test, cb_pred)

    return results


def main():
    print("=" * 110)
    print("КОМПЛЕКСНЫЙ БЕНЧМАРК: TurboCat vs CatBoost (Classification + Regression)")
    print("=" * 110)

    clf_datasets = []

    data = load_breast_cancer()
    clf_datasets.append(("Breast Cancer", data.data.astype(np.float32), data.target))

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    clf_datasets.append(("Synthetic 2K", X.astype(np.float32), y))

    X, y = make_classification(n_samples=10000, n_features=30, n_informative=15,
                               n_redundant=10, random_state=42)
    clf_datasets.append(("Synthetic 10K", X.astype(np.float32), y))

    X, y = make_classification(n_samples=50000, n_features=50, n_informative=25,
                               n_redundant=15, random_state=42)
    clf_datasets.append(("Synthetic 50K", X.astype(np.float32), y))

    X, y = make_classification(n_samples=5000, n_features=20, n_informative=10,
                               weights=[0.95, 0.05], random_state=42)
    clf_datasets.append(("Imbalanced 95/5", X.astype(np.float32), y))

    reg_datasets = []

    try:
        data = fetch_california_housing()
        reg_datasets.append(("California Housing", data.data.astype(np.float32), data.target.astype(np.float32)))
    except:
        pass

    X, y = make_regression(n_samples=2000, n_features=20, n_informative=10,
                           noise=10, random_state=42)
    reg_datasets.append(("Regression 2K", X.astype(np.float32), y.astype(np.float32)))

    X, y = make_regression(n_samples=10000, n_features=30, n_informative=15,
                           noise=10, random_state=42)
    reg_datasets.append(("Regression 10K", X.astype(np.float32), y.astype(np.float32)))

    X, y = make_regression(n_samples=50000, n_features=50, n_informative=25,
                           noise=10, random_state=42)
    reg_datasets.append(("Regression 50K", X.astype(np.float32), y.astype(np.float32)))

    clf_results = []
    reg_results = []

    print("\n" + "=" * 110)
    print("CLASSIFICATION")
    print("=" * 110)

    for name, X, y in clf_datasets:
        print(f"  Running {name}...", end=" ", flush=True)
        results = benchmark_classification(X, y, name)
        clf_results.append(results)
        print("Done")

    print("\n" + "=" * 110)
    print("REGRESSION")
    print("=" * 110)

    for name, X, y in reg_datasets:
        print(f"  Running {name}...", end=" ", flush=True)
        results = benchmark_regression(X, y, name)
        reg_results.append(results)
        print("Done")

    # CLASSIFICATION RESULTS
    print("\n" + "=" * 110)
    print("РЕЗУЛЬТАТЫ CLASSIFICATION")
    print("=" * 110)

    print(f"\n{'Датасет':<20} {'Samples':<10} {'TC Train':<10} {'CB Train':<10} {'SpeedUp':<10} {'TC AUC':<10} {'CB AUC':<10} {'Winner':<12}")
    print("-" * 110)

    tc_clf_wins = 0
    cb_clf_wins = 0
    total_tc_train_clf = 0
    total_cb_train_clf = 0

    for r in clf_results:
        train_speedup = r['cb_train'] / r['tc_train']
        total_tc_train_clf += r['tc_train']
        total_cb_train_clf += r['cb_train']

        if r['tc_auc'] > r['cb_auc'] + 0.001:
            winner = "TurboCat"
            tc_clf_wins += 1
        elif r['cb_auc'] > r['tc_auc'] + 0.001:
            winner = "CatBoost"
            cb_clf_wins += 1
        else:
            winner = "Tie"

        print(f"{r['name']:<20} {r['n_samples']:<10} {r['tc_train']:<10.4f} {r['cb_train']:<10.4f} {train_speedup:<10.2f}x {r['tc_auc']:<10.4f} {r['cb_auc']:<10.4f} {winner:<12}")

    print("-" * 110)
    overall_clf_speedup = total_cb_train_clf / total_tc_train_clf
    avg_tc_auc = np.mean([r['tc_auc'] for r in clf_results])
    avg_cb_auc = np.mean([r['cb_auc'] for r in clf_results])
    print(f"{'СРЕДНЕЕ':<20} {'':<10} {total_tc_train_clf:<10.4f} {total_cb_train_clf:<10.4f} {overall_clf_speedup:<10.2f}x {avg_tc_auc:<10.4f} {avg_cb_auc:<10.4f}")

    # REGRESSION RESULTS
    print("\n" + "=" * 110)
    print("РЕЗУЛЬТАТЫ REGRESSION")
    print("=" * 110)

    print(f"\n{'Датасет':<20} {'Samples':<10} {'TC Train':<10} {'CB Train':<10} {'SpeedUp':<10} {'TC R2':<10} {'CB R2':<10} {'Winner':<12}")
    print("-" * 110)

    tc_reg_wins = 0
    cb_reg_wins = 0
    total_tc_train_reg = 0
    total_cb_train_reg = 0

    for r in reg_results:
        train_speedup = r['cb_train'] / r['tc_train']
        total_tc_train_reg += r['tc_train']
        total_cb_train_reg += r['cb_train']

        if r['tc_r2'] > r['cb_r2'] + 0.001:
            winner = "TurboCat"
            tc_reg_wins += 1
        elif r['cb_r2'] > r['tc_r2'] + 0.001:
            winner = "CatBoost"
            cb_reg_wins += 1
        else:
            winner = "Tie"

        print(f"{r['name']:<20} {r['n_samples']:<10} {r['tc_train']:<10.4f} {r['cb_train']:<10.4f} {train_speedup:<10.2f}x {r['tc_r2']:<10.4f} {r['cb_r2']:<10.4f} {winner:<12}")

    print("-" * 110)
    overall_reg_speedup = total_cb_train_reg / total_tc_train_reg
    avg_tc_r2 = np.mean([r['tc_r2'] for r in reg_results])
    avg_cb_r2 = np.mean([r['cb_r2'] for r in reg_results])
    print(f"{'СРЕДНЕЕ':<20} {'':<10} {total_tc_train_reg:<10.4f} {total_cb_train_reg:<10.4f} {overall_reg_speedup:<10.2f}x {avg_tc_r2:<10.4f} {avg_cb_r2:<10.4f}")

    # FINAL SUMMARY
    print("\n" + "=" * 110)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 110)

    print("\n+-----------------------------------------------------------------------------+")
    print("|                           CLASSIFICATION                                    |")
    print("+-----------------------------------------------------------------------------+")
    print(f"|  Training Speed:   TurboCat {overall_clf_speedup:.2f}x faster                                   |")
    print(f"|  Quality (AUC):    TurboCat {avg_tc_auc:.4f} vs CatBoost {avg_cb_auc:.4f} ({(avg_tc_auc-avg_cb_auc)*100:+.2f}%)        |")
    print(f"|  Wins:             TurboCat {tc_clf_wins} | CatBoost {cb_clf_wins}                                   |")
    print("+-----------------------------------------------------------------------------+")

    print("\n+-----------------------------------------------------------------------------+")
    print("|                             REGRESSION                                      |")
    print("+-----------------------------------------------------------------------------+")
    print(f"|  Training Speed:   TurboCat {overall_reg_speedup:.2f}x faster                                   |")
    print(f"|  Quality (R2):     TurboCat {avg_tc_r2:.4f} vs CatBoost {avg_cb_r2:.4f} ({(avg_tc_r2-avg_cb_r2)*100:+.2f}%)        |")
    print(f"|  Wins:             TurboCat {tc_reg_wins} | CatBoost {cb_reg_wins}                                   |")
    print("+-----------------------------------------------------------------------------+")

    total_tc_wins = tc_clf_wins + tc_reg_wins
    total_cb_wins = cb_clf_wins + cb_reg_wins
    total_speedup = (total_cb_train_clf + total_cb_train_reg) / (total_tc_train_clf + total_tc_train_reg)

    print("\n+-----------------------------------------------------------------------------+")
    print("|                              OVERALL                                        |")
    print("+-----------------------------------------------------------------------------+")
    print(f"|  Training Speed:   TurboCat {total_speedup:.2f}x faster                                   |")
    print(f"|  Total Wins:       TurboCat {total_tc_wins} | CatBoost {total_cb_wins}                                   |")
    print("+-----------------------------------------------------------------------------+")


if __name__ == "__main__":
    main()
