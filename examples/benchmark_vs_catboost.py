"""
TurboCat vs CatBoost Benchmark

This script compares TurboCat against CatBoost on various metrics.
"""

import numpy as np
import time
from turbocat import TurboCatClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss
)

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed: pip install catboost")

# Generate data
print("📦 Generating data...")
np.random.seed(42)
n_samples = 10_000
n_features = 20

X = np.random.randn(n_samples, n_features).astype(np.float32)
y = ((X[:, 0] + X[:, 1] * 2 + X[:, 2] ** 2 + np.random.randn(n_samples) * 0.5) > 1).astype(np.float32)

# Split
X_train, X_test = X[:8_000], X[8_000:]
y_train, y_test = y[:8_000], y[8_000:]

print(f"✅ Data: {X_train.shape[0]} train, {X_test.shape[0]} test, {n_features} features")
print(f"📊 Class balance: {y_test.mean():.1%} positive")
print("=" * 60)

N_ESTIMATORS = 20
MAX_DEPTH = 8
LEARNING_RATE = 0.01

def compute_metrics(y_true, y_proba, name):
    """Compute all metrics"""
    y_pred = (y_proba > 0.5).astype(int)
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'PR-AUC': average_precision_score(y_true, y_proba),
        'Log Loss': log_loss(y_true, y_proba),
    }
    
    print(f"\n📈 {name} Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    
    return metrics

# TurboCat
print("\n🚀 TurboCat:")
tc_clf = TurboCatClassifier(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    verbosity=0
)

start = time.time()
tc_clf.fit(X_train, y_train)
tc_train_time = time.time() - start

start = time.time()
tc_proba = tc_clf.predict_proba(X_test)
tc_pred_time = time.time() - start

if tc_proba.ndim == 2:
    tc_pred = tc_proba[:, 1]
else:
    tc_pred = tc_proba

print(f"  ⏱️  Train time: {tc_train_time:.3f}s")
print(f"  ⏱️  Predict time: {tc_pred_time:.3f}s")
print(f"  🌳 Trees: {tc_clf.n_trees}")

tc_metrics = compute_metrics(y_test, tc_pred, "TurboCat")

# CatBoost
if HAS_CATBOOST:
    print("\n" + "=" * 60)
    print("\n🐱 CatBoost:")
    cb_clf = CatBoostClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        verbose=False
    )
    
    start = time.time()
    cb_clf.fit(X_train, y_train)
    cb_train_time = time.time() - start
    
    start = time.time()
    cb_pred = cb_clf.predict_proba(X_test)[:, 1]
    cb_pred_time = time.time() - start
    
    print(f"  ⏱️  Train time: {cb_train_time:.3f}s")
    print(f"  ⏱️  Predict time: {cb_pred_time:.3f}s")
    
    cb_metrics = compute_metrics(y_test, cb_pred, "CatBoost")
    
    # Comparison
    print("\n" + "=" * 60)
    print("📊 COMPARISON:")
    print("=" * 60)
    
    print(f"\n{'Metric':<12} {'TurboCat':>10} {'CatBoost':>10} {'Diff':>10} {'Winner':>12}")
    print("-" * 56)
    
    for metric in tc_metrics:
        tc_val = tc_metrics[metric]
        cb_val = cb_metrics[metric]
        diff = tc_val - cb_val
        
        if metric == 'Log Loss':
            winner = "🏆 TurboCat" if tc_val < cb_val else ("🐱 CatBoost" if cb_val < tc_val else "Tie")
        else:
            winner = "🏆 TurboCat" if tc_val > cb_val else ("🐱 CatBoost" if cb_val > tc_val else "Tie")
        
        print(f"{metric:<12} {tc_val:>10.4f} {cb_val:>10.4f} {diff:>+10.4f} {winner:>12}")
    
    print("-" * 56)
    
    # Count wins
    tc_wins = sum(1 for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC'] 
                  if tc_metrics[m] > cb_metrics[m])
    tc_wins += 1 if tc_metrics['Log Loss'] < cb_metrics['Log Loss'] else 0
    
    print(f"\n🏁 RESULT: TurboCat won {tc_wins}/7 quality metrics")
    print("=" * 60)
