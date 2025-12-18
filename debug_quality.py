#!/usr/bin/env python3
"""Debug quality issues in TurboCat"""

import numpy as np
import time
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, 'build')
from turbocat import TurboCatClassifier

def test_hyperparameters():
    """Test different hyperparameter combinations"""
    print("=" * 80)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Generate data
    np.random.seed(42)
    n_samples = 20000
    n_features = 30

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

    # Test different settings
    configs = [
        {"learning_rate": 0.05, "max_depth": 6, "n_estimators": 100, "lambda_l2": 1.0},
        {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 100, "lambda_l2": 1.0},
        {"learning_rate": 0.1, "max_depth": 8, "n_estimators": 100, "lambda_l2": 1.0},
        {"learning_rate": 0.1, "max_depth": 8, "n_estimators": 200, "lambda_l2": 1.0},
        {"learning_rate": 0.1, "max_depth": 8, "n_estimators": 200, "lambda_l2": 0.1},
        {"learning_rate": 0.1, "max_depth": 10, "n_estimators": 200, "lambda_l2": 0.1},
        {"learning_rate": 0.2, "max_depth": 6, "n_estimators": 100, "lambda_l2": 0.0},
        {"learning_rate": 0.3, "max_depth": 6, "n_estimators": 100, "lambda_l2": 0.0},
    ]

    print(f"\n{'LR':<6} {'Depth':<6} {'Trees':<6} {'L2':<6} {'AUC':<10} {'Acc':<10} {'Time':<10}")
    print("-" * 60)

    best_auc = 0
    best_config = None

    for cfg in configs:
        clf = TurboCatClassifier(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            lambda_l2=cfg["lambda_l2"],
            verbose=0,
            use_leaf_wise=True,
            use_symmetric=False
        )

        start = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, (proba > 0.5).astype(int))

        print(f"{cfg['learning_rate']:<6.2f} {cfg['max_depth']:<6} {cfg['n_estimators']:<6} {cfg['lambda_l2']:<6.1f} {auc:<10.4f} {acc:<10.4f} {train_time:<10.3f}")

        if auc > best_auc:
            best_auc = auc
            best_config = cfg

    print(f"\nBest config: {best_config}")
    print(f"Best AUC: {best_auc:.4f}")

    # Compare with LightGBM using similar settings
    print("\n" + "=" * 80)
    print("COMPARISON WITH LIGHTGBM (same hyperparameters)")
    print("=" * 80)

    try:
        from lightgbm import LGBMClassifier

        for cfg in [best_config, {"learning_rate": 0.1, "max_depth": 8, "n_estimators": 200, "lambda_l2": 0.1}]:
            # TurboCat
            clf_tc = TurboCatClassifier(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                lambda_l2=cfg["lambda_l2"],
                verbose=0,
                use_leaf_wise=True
            )
            clf_tc.fit(X_train, y_train)
            tc_auc = roc_auc_score(y_test, clf_tc.predict_proba(X_test)[:, 1])

            # LightGBM with similar settings
            clf_lgb = LGBMClassifier(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                reg_lambda=cfg["lambda_l2"],
                verbose=-1
            )
            clf_lgb.fit(X_train, y_train.astype(int))
            lgb_auc = roc_auc_score(y_test, clf_lgb.predict_proba(X_test)[:, 1])

            print(f"Config: LR={cfg['learning_rate']}, depth={cfg['max_depth']}, trees={cfg['n_estimators']}, L2={cfg['lambda_l2']}")
            print(f"  TurboCat AUC: {tc_auc:.4f}")
            print(f"  LightGBM AUC: {lgb_auc:.4f}")
            print(f"  Gap: {tc_auc - lgb_auc:+.4f}")
            print()

    except ImportError:
        print("LightGBM not available")

def test_overfitting():
    """Check if model is overfitting or underfitting"""
    print("\n" + "=" * 80)
    print("TRAIN vs TEST ANALYSIS")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 20000
    n_features = 30

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

    print(f"\n{'Config':<40} {'Train AUC':<12} {'Test AUC':<12} {'Gap':<10}")
    print("-" * 80)

    for lr in [0.1, 0.2, 0.3]:
        for depth in [6, 8, 10]:
            clf = TurboCatClassifier(
                n_estimators=100,
                learning_rate=lr,
                max_depth=depth,
                lambda_l2=0.1,
                verbose=0
            )
            clf.fit(X_train, y_train)

            train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
            test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            gap = train_auc - test_auc

            config_str = f"LR={lr}, depth={depth}"
            print(f"{config_str:<40} {train_auc:<12.4f} {test_auc:<12.4f} {gap:<10.4f}")

if __name__ == "__main__":
    test_hyperparameters()
    test_overfitting()
