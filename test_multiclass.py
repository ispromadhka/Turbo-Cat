#!/usr/bin/env python3
"""
Test multiclass classification: TurboCat vs CatBoost vs LightGBM vs XGBoost
"""

import numpy as np
import time
import warnings
from sklearn.datasets import load_iris, load_wine, load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Import models
import sys
sys.path.insert(0, 'build')
from _turbocat import TurboCatClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Common hyperparameters
COMMON_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
}

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model"""
    try:
        # Training time
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        # Inference time
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        if hasattr(y_pred, 'flatten'):
            y_pred = y_pred.flatten()
        inference_time = time.perf_counter() - start

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        return {
            'accuracy': acc,
            'f1': f1,
            'train_time': train_time,
            'inference_time': inference_time,
        }
    except Exception as e:
        print(f"  {name} error: {e}")
        return None

def run_benchmark(name, X, y):
    """Run benchmark on a dataset"""
    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print('='*70)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.int32)

    results = {}

    # TurboCat
    print("\n  Training TurboCat...")
    tc_model = TurboCatClassifier(
        n_estimators=COMMON_PARAMS['n_estimators'],
        learning_rate=COMMON_PARAMS['learning_rate'],
        max_depth=COMMON_PARAMS['max_depth'],
        verbosity=0,
        use_goss=True
    )
    results['TurboCat'] = evaluate_model('TurboCat', tc_model, X_train, X_test, y_train, y_test)

    # CatBoost
    if HAS_CATBOOST:
        print("  Training CatBoost...")
        cb_model = CatBoostClassifier(
            iterations=COMMON_PARAMS['n_estimators'],
            learning_rate=COMMON_PARAMS['learning_rate'],
            depth=COMMON_PARAMS['max_depth'],
            verbose=False,
            allow_writing_files=False
        )
        results['CatBoost'] = evaluate_model('CatBoost', cb_model, X_train, X_test, y_train, y_test)

    # LightGBM
    if HAS_LIGHTGBM:
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=COMMON_PARAMS['n_estimators'],
            learning_rate=COMMON_PARAMS['learning_rate'],
            max_depth=COMMON_PARAMS['max_depth'],
            verbose=-1
        )
        results['LightGBM'] = evaluate_model('LightGBM', lgb_model, X_train, X_test, y_train, y_test)

    # XGBoost
    if HAS_XGBOOST:
        print("  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=COMMON_PARAMS['n_estimators'],
            learning_rate=COMMON_PARAMS['learning_rate'],
            max_depth=COMMON_PARAMS['max_depth'],
            verbosity=0,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        results['XGBoost'] = evaluate_model('XGBoost', xgb_model, X_train, X_test, y_train, y_test)

    # Print results
    print(f"\n{'Model':<12} {'Accuracy':>10} {'F1':>10} {'Train(s)':>10} {'Infer(s)':>10}")
    print('-' * 55)

    for model_name, metrics in results.items():
        if metrics is not None:
            print(f"{model_name:<12} {metrics['accuracy']:>10.4f} {metrics['f1']:>10.4f} "
                  f"{metrics['train_time']:>10.4f} {metrics['inference_time']:>10.4f}")

    return results

def main():
    print("="*70)
    print("TurboCat Multiclass Benchmark")
    print(f"Comparing: TurboCat", end='')
    if HAS_CATBOOST: print(", CatBoost", end='')
    if HAS_LIGHTGBM: print(", LightGBM", end='')
    if HAS_XGBOOST: print(", XGBoost", end='')
    print(f"\nHyperparameters: {COMMON_PARAMS}")
    print("="*70)

    all_results = {}

    # sklearn multiclass datasets
    print("\n\n" + "#"*70)
    print("# MULTICLASS DATASETS")
    print("#"*70)

    # Iris (3 classes)
    iris = load_iris()
    all_results['iris'] = run_benchmark('Iris (3 classes)', iris.data, iris.target)

    # Wine (3 classes)
    wine = load_wine()
    all_results['wine'] = run_benchmark('Wine (3 classes)', wine.data, wine.target)

    # Digits (10 classes)
    digits = load_digits()
    all_results['digits'] = run_benchmark('Digits (10 classes)', digits.data, digits.target)

    # Synthetic 5 classes
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=15,
        n_redundant=3, n_classes=5, n_clusters_per_class=2, random_state=42
    )
    all_results['synthetic_5'] = run_benchmark('Synthetic (5 classes)', X, y)

    # Synthetic 10 classes
    X, y = make_classification(
        n_samples=5000, n_features=30, n_informative=20,
        n_redundant=5, n_classes=10, n_clusters_per_class=1, random_state=42
    )
    all_results['synthetic_10'] = run_benchmark('Synthetic (10 classes)', X, y)

    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    models = ['TurboCat']
    if HAS_CATBOOST: models.append('CatBoost')
    if HAS_LIGHTGBM: models.append('LightGBM')
    if HAS_XGBOOST: models.append('XGBoost')

    # Aggregate metrics
    agg = {m: {'acc': [], 'f1': [], 'train': [], 'infer': []} for m in models}

    for name, result in all_results.items():
        for m in models:
            if result.get(m) is not None:
                agg[m]['acc'].append(result[m]['accuracy'])
                agg[m]['f1'].append(result[m]['f1'])
                agg[m]['train'].append(result[m]['train_time'])
                agg[m]['infer'].append(result[m]['inference_time'])

    print(f"\n{'Model':<12} {'Avg Acc':>10} {'Avg F1':>10} {'Avg Train':>12} {'Speedup':>10}")
    print('-' * 60)

    baseline_train = np.mean(agg['CatBoost']['train']) if HAS_CATBOOST else 1.0

    for m in models:
        if agg[m]['acc']:
            avg_acc = np.mean(agg[m]['acc'])
            avg_f1 = np.mean(agg[m]['f1'])
            avg_train = np.mean(agg[m]['train'])
            speedup = baseline_train / avg_train if avg_train > 0 else 0
            print(f"{m:<12} {avg_acc:>10.4f} {avg_f1:>10.4f} {avg_train:>12.4f} {speedup:>10.1f}x")

if __name__ == '__main__':
    main()
