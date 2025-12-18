#!/usr/bin/env python3
"""
Детальный бенчмарк: Train vs Inference скорость и качество
"""

import numpy as np
import time
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, 'build')
from turbocat import TurboCatClassifier

def benchmark_detailed():
    print("=" * 90)
    print("ДЕТАЛЬНЫЙ БЕНЧМАРК: TRAIN / INFERENCE / КАЧЕСТВО")
    print("=" * 90)

    try:
        from catboost import CatBoostClassifier
        has_catboost = True
    except ImportError:
        has_catboost = False
        print("CatBoost не установлен")

    try:
        from lightgbm import LGBMClassifier
        has_lightgbm = True
    except ImportError:
        has_lightgbm = False
        print("LightGBM не установлен")

    # Тестовые конфигурации: (n_samples, n_features, n_trees, описание)
    test_cases = [
        (10000, 30, 100, "Маленький"),
        (50000, 50, 100, "Средний"),
        (100000, 50, 100, "Большой"),
        (200000, 50, 100, "Очень большой"),
    ]

    all_results = []

    for n_samples, n_features, n_trees, desc in test_cases:
        print(f"\n{'=' * 90}")
        print(f"Датасет: {desc} ({n_samples:,} samples, {n_features} features, {n_trees} trees)")
        print("=" * 90)

        # Генерация реалистичных данных
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)
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

        n_test = len(X_test)
        results = {'desc': desc, 'n_samples': n_samples, 'n_test': n_test}

        print(f"\n{'Модель':<12} {'Train(s)':<10} {'Infer(ms)':<12} {'Infer/1K':<10} {'AUC':<10} {'Accuracy':<10} {'F1':<10}")
        print("-" * 90)

        # ============== TurboCat ==============
        clf_tc = TurboCatClassifier(
            n_estimators=n_trees,
            max_depth=6,
            learning_rate=0.1,
            verbose=0,
            use_leaf_wise=True
        )

        # Train
        start = time.perf_counter()
        clf_tc.fit(X_train, y_train)
        tc_train_time = time.perf_counter() - start

        # Inference (среднее по 5 запускам)
        inference_times = []
        for _ in range(5):
            start = time.perf_counter()
            tc_proba = clf_tc.predict_proba(X_test)
            inference_times.append(time.perf_counter() - start)
        tc_infer_time = np.mean(inference_times)
        tc_infer_per_1k = (tc_infer_time / n_test) * 1000 * 1000  # ms per 1000 samples

        tc_proba = tc_proba[:, 1]
        tc_auc = roc_auc_score(y_test, tc_proba)
        tc_acc = accuracy_score(y_test, (tc_proba > 0.5).astype(int))
        tc_f1 = f1_score(y_test, (tc_proba > 0.5).astype(int))

        print(f"{'TurboCat':<12} {tc_train_time:<10.3f} {tc_infer_time*1000:<12.2f} {tc_infer_per_1k:<10.3f} {tc_auc:<10.4f} {tc_acc:<10.4f} {tc_f1:<10.4f}")

        results['tc'] = {
            'train': tc_train_time,
            'infer': tc_infer_time,
            'infer_per_1k': tc_infer_per_1k,
            'auc': tc_auc,
            'acc': tc_acc,
            'f1': tc_f1
        }

        # ============== CatBoost ==============
        if has_catboost:
            clf_cb = CatBoostClassifier(
                n_estimators=n_trees,
                max_depth=6,
                verbose=0,
                thread_count=-1
            )

            start = time.perf_counter()
            clf_cb.fit(X_train, y_train.astype(int))
            cb_train_time = time.perf_counter() - start

            inference_times = []
            for _ in range(5):
                start = time.perf_counter()
                cb_proba = clf_cb.predict_proba(X_test)
                inference_times.append(time.perf_counter() - start)
            cb_infer_time = np.mean(inference_times)
            cb_infer_per_1k = (cb_infer_time / n_test) * 1000 * 1000

            cb_proba = cb_proba[:, 1]
            cb_auc = roc_auc_score(y_test, cb_proba)
            cb_acc = accuracy_score(y_test, (cb_proba > 0.5).astype(int))
            cb_f1 = f1_score(y_test, (cb_proba > 0.5).astype(int))

            print(f"{'CatBoost':<12} {cb_train_time:<10.3f} {cb_infer_time*1000:<12.2f} {cb_infer_per_1k:<10.3f} {cb_auc:<10.4f} {cb_acc:<10.4f} {cb_f1:<10.4f}")

            results['cb'] = {
                'train': cb_train_time,
                'infer': cb_infer_time,
                'infer_per_1k': cb_infer_per_1k,
                'auc': cb_auc,
                'acc': cb_acc,
                'f1': cb_f1
            }

        # ============== LightGBM ==============
        if has_lightgbm:
            clf_lgb = LGBMClassifier(
                n_estimators=n_trees,
                max_depth=6,
                verbose=-1,
                n_jobs=-1
            )

            start = time.perf_counter()
            clf_lgb.fit(X_train, y_train.astype(int))
            lgb_train_time = time.perf_counter() - start

            inference_times = []
            for _ in range(5):
                start = time.perf_counter()
                lgb_proba = clf_lgb.predict_proba(X_test)
                inference_times.append(time.perf_counter() - start)
            lgb_infer_time = np.mean(inference_times)
            lgb_infer_per_1k = (lgb_infer_time / n_test) * 1000 * 1000

            lgb_proba = lgb_proba[:, 1]
            lgb_auc = roc_auc_score(y_test, lgb_proba)
            lgb_acc = accuracy_score(y_test, (lgb_proba > 0.5).astype(int))
            lgb_f1 = f1_score(y_test, (lgb_proba > 0.5).astype(int))

            print(f"{'LightGBM':<12} {lgb_train_time:<10.3f} {lgb_infer_time*1000:<12.2f} {lgb_infer_per_1k:<10.3f} {lgb_auc:<10.4f} {lgb_acc:<10.4f} {lgb_f1:<10.4f}")

            results['lgb'] = {
                'train': lgb_train_time,
                'infer': lgb_infer_time,
                'infer_per_1k': lgb_infer_per_1k,
                'auc': lgb_auc,
                'acc': lgb_acc,
                'f1': lgb_f1
            }

        # Сравнение
        print(f"\n--- Сравнение TurboCat ---")
        if has_catboost:
            train_speedup = cb_train_time / tc_train_time
            infer_speedup = cb_infer_time / tc_infer_time
            auc_diff = tc_auc - cb_auc
            print(f"  vs CatBoost:  Train {train_speedup:.2f}x {'быстрее' if train_speedup > 1 else 'медленнее'}, "
                  f"Inference {infer_speedup:.2f}x {'быстрее' if infer_speedup > 1 else 'медленнее'}, "
                  f"AUC {auc_diff:+.4f}")

        if has_lightgbm:
            train_speedup = lgb_train_time / tc_train_time
            infer_speedup = lgb_infer_time / tc_infer_time
            auc_diff = tc_auc - lgb_auc
            print(f"  vs LightGBM:  Train {train_speedup:.2f}x {'быстрее' if train_speedup > 1 else 'медленнее'}, "
                  f"Inference {infer_speedup:.2f}x {'быстрее' if infer_speedup > 1 else 'медленнее'}, "
                  f"AUC {auc_diff:+.4f}")

        all_results.append(results)

    # ============== ИТОГОВАЯ СВОДКА ==============
    print("\n" + "=" * 90)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 90)

    if has_catboost:
        print("\n" + "-" * 90)
        print("TurboCat vs CatBoost")
        print("-" * 90)
        print(f"{'Датасет':<20} {'TC Train':<10} {'CB Train':<10} {'Train X':<10} {'TC Infer':<10} {'CB Infer':<10} {'Infer X':<10} {'AUC Diff':<10}")
        print("-" * 90)

        total_tc_train = 0
        total_cb_train = 0
        total_tc_infer = 0
        total_cb_infer = 0

        for r in all_results:
            if 'cb' in r:
                tc = r['tc']
                cb = r['cb']
                train_x = cb['train'] / tc['train']
                infer_x = cb['infer'] / tc['infer']
                auc_diff = tc['auc'] - cb['auc']

                total_tc_train += tc['train']
                total_cb_train += cb['train']
                total_tc_infer += tc['infer']
                total_cb_infer += cb['infer']

                print(f"{r['desc']:<20} {tc['train']:<10.3f} {cb['train']:<10.3f} {train_x:<10.2f} "
                      f"{tc['infer']*1000:<10.2f} {cb['infer']*1000:<10.2f} {infer_x:<10.2f} {auc_diff:+.4f}")

        print("-" * 90)
        overall_train_x = total_cb_train / total_tc_train
        overall_infer_x = total_cb_infer / total_tc_infer
        print(f"{'ИТОГО':<20} {total_tc_train:<10.3f} {total_cb_train:<10.3f} {overall_train_x:<10.2f} "
              f"{total_tc_infer*1000:<10.2f} {total_cb_infer*1000:<10.2f} {overall_infer_x:<10.2f}")

        print(f"\n📊 ВЫВОД vs CatBoost:")
        print(f"   • Train:     TurboCat {'быстрее' if overall_train_x > 1 else 'медленнее'} в {overall_train_x:.2f}x")
        print(f"   • Inference: TurboCat {'быстрее' if overall_infer_x > 1 else 'медленнее'} в {overall_infer_x:.2f}x")

    if has_lightgbm:
        print("\n" + "-" * 90)
        print("TurboCat vs LightGBM")
        print("-" * 90)

        total_tc_train = 0
        total_lgb_train = 0
        total_tc_infer = 0
        total_lgb_infer = 0

        for r in all_results:
            if 'lgb' in r:
                tc = r['tc']
                lgb = r['lgb']
                total_tc_train += tc['train']
                total_lgb_train += lgb['train']
                total_tc_infer += tc['infer']
                total_lgb_infer += lgb['infer']

        overall_train_x = total_lgb_train / total_tc_train
        overall_infer_x = total_lgb_infer / total_tc_infer

        print(f"\n📊 ВЫВОД vs LightGBM:")
        print(f"   • Train:     TurboCat {'быстрее' if overall_train_x > 1 else 'медленнее'} в {abs(overall_train_x):.2f}x")
        print(f"   • Inference: TurboCat {'быстрее' if overall_infer_x > 1 else 'медленнее'} в {abs(overall_infer_x):.2f}x")

    # Качество
    print("\n" + "-" * 90)
    print("КАЧЕСТВО (средний AUC)")
    print("-" * 90)

    avg_tc_auc = np.mean([r['tc']['auc'] for r in all_results])
    print(f"TurboCat: {avg_tc_auc:.4f}")

    if has_catboost:
        avg_cb_auc = np.mean([r['cb']['auc'] for r in all_results if 'cb' in r])
        print(f"CatBoost: {avg_cb_auc:.4f}")
        print(f"Разница:  {avg_tc_auc - avg_cb_auc:+.4f} ({(avg_tc_auc/avg_cb_auc - 1)*100:+.2f}%)")

    if has_lightgbm:
        avg_lgb_auc = np.mean([r['lgb']['auc'] for r in all_results if 'lgb' in r])
        print(f"LightGBM: {avg_lgb_auc:.4f}")
        print(f"Разница:  {avg_tc_auc - avg_lgb_auc:+.4f} ({(avg_tc_auc/avg_lgb_auc - 1)*100:+.2f}%)")

if __name__ == "__main__":
    benchmark_detailed()
