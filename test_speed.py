#!/usr/bin/env python3
"""Quick speed test"""
import numpy as np
import time
import sys
sys.path.insert(0, 'build')
from turbocat import TurboCatClassifier

# Small dataset
np.random.seed(42)
X = np.random.randn(1000, 20).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

# Test different configurations
configs = [
    {"name": "Default (GOSS)", "use_goss": True, "subsample": 0.8, "colsample_bytree": 0.8},
    {"name": "No GOSS", "use_goss": False, "subsample": 1.0, "colsample_bytree": 1.0},
    {"name": "GOSS + full data", "use_goss": True, "subsample": 1.0, "colsample_bytree": 1.0},
]

print("=" * 60)
print("SPEED TEST (1000 samples, 20 features, 100 trees)")
print("=" * 60)

try:
    from catboost import CatBoostClassifier
    has_catboost = True
except:
    has_catboost = False

for cfg in configs:
    clf = TurboCatClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_goss=cfg["use_goss"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        verbose=0
    )

    times = []
    for _ in range(3):
        start = time.perf_counter()
        clf.fit(X, y)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    print(f"{cfg['name']:<25}: {avg_time:.4f}s")

if has_catboost:
    clf_cb = CatBoostClassifier(n_estimators=100, max_depth=6, verbose=0)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        clf_cb.fit(X, y)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    print(f"{'CatBoost':<25}: {avg_time:.4f}s")

print("\n--- Larger dataset (10K) ---")
X_large = np.random.randn(10000, 20).astype(np.float32)
y_large = (X_large[:, 0] + X_large[:, 1] > 0).astype(np.float32)

clf = TurboCatClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_goss=True, verbose=0)
start = time.perf_counter()
clf.fit(X_large, y_large)
tc_time = time.perf_counter() - start
print(f"{'TurboCat (GOSS)':<25}: {tc_time:.4f}s")

if has_catboost:
    clf_cb = CatBoostClassifier(n_estimators=100, max_depth=6, verbose=0)
    start = time.perf_counter()
    clf_cb.fit(X_large, y_large)
    cb_time = time.perf_counter() - start
    print(f"{'CatBoost':<25}: {cb_time:.4f}s")
    print(f"\nSpeedup: {cb_time/tc_time:.2f}x")
