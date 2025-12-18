#!/usr/bin/env python3
"""
Comprehensive Benchmark: TurboCat vs CatBoost
============================================

This script performs extensive benchmarks comparing TurboCat and CatBoost
across multiple dimensions:
- Various dataset sizes (1K, 10K, 50K, 100K samples)
- Training time
- Inference time (the main focus for optimization)
- Quality metrics (AUC, Accuracy, F1)

The goal is to verify that TurboCat is faster than CatBoost in BOTH
training AND inference while maintaining quality.
"""

import numpy as np
import time
import warnings
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Import models
try:
    from turbocat import TurboCatClassifier
    TURBOCAT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TurboCat not available: {e}")
    TURBOCAT_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("Warning: CatBoost not available")
    CATBOOST_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Store results from a single benchmark run."""
    model_name: str
    dataset_name: str
    n_samples: int
    n_features: int
    train_time: float
    inference_time: float
    inference_samples_per_sec: float
    accuracy: float
    roc_auc: Optional[float]
    f1_score: float
    n_trees: int


def create_dataset(n_samples: int, n_features: int = 20, n_informative: int = 15,
                   class_sep: float = 1.0, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create a synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative - 2,
        n_clusters_per_class=2,
        class_sep=class_sep,
        random_state=random_state
    )
    return X.astype(np.float32), y.astype(np.float32)


def benchmark_model(model, model_name: str, X_train: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray, dataset_name: str,
                    n_inference_runs: int = 5) -> Optional[BenchmarkResult]:
    """Run benchmark for a single model."""
    try:
        # Training
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - train_start

        # Warmup inference
        _ = model.predict(X_test[:100])

        # Inference - run multiple times for stable measurement
        inference_times = []
        for _ in range(n_inference_runs):
            inf_start = time.perf_counter()
            y_pred = model.predict(X_test)
            inf_time = time.perf_counter() - inf_start
            inference_times.append(inf_time)

        inference_time = np.median(inference_times)
        inference_samples_per_sec = len(X_test) / inference_time

        # Get predictions for metrics
        y_pred = model.predict(X_test)

        # Get probabilities for ROC-AUC
        try:
            y_proba = model.predict_proba(X_test)
            if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc_auc = None

        # Get number of trees
        try:
            n_trees = model.n_trees if hasattr(model, 'n_trees') else model.tree_count_
        except Exception:
            n_trees = 0

        return BenchmarkResult(
            model_name=model_name,
            dataset_name=dataset_name,
            n_samples=len(X_train),
            n_features=X_train.shape[1],
            train_time=train_time,
            inference_time=inference_time,
            inference_samples_per_sec=inference_samples_per_sec,
            accuracy=accuracy_score(y_test, y_pred),
            roc_auc=roc_auc,
            f1_score=f1_score(y_test, y_pred, average='weighted'),
            n_trees=n_trees
        )
    except Exception as e:
        print(f"Error benchmarking {model_name}: {e}")
        return None


def run_size_benchmarks(sizes: List[int], n_features: int = 20,
                        n_estimators: int = 100, max_depth: int = 6,
                        learning_rate: float = 0.1) -> List[BenchmarkResult]:
    """Run benchmarks across different dataset sizes."""
    results = []

    print("\n" + "=" * 80)
    print("BENCHMARK: Dataset Size Scaling")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
    print("=" * 80)

    for n_samples in sizes:
        print(f"\n--- Dataset Size: {n_samples:,} samples, {n_features} features ---")

        # Create dataset
        X, y = create_dataset(n_samples, n_features)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

        # TurboCat
        if TURBOCAT_AVAILABLE:
            tc_model = TurboCatClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                use_goss=True,
                n_jobs=-1,
                verbosity=0
            )
            result = benchmark_model(
                tc_model, "TurboCat", X_train, X_test, y_train, y_test,
                f"size_{n_samples}"
            )
            if result:
                results.append(result)
                print(f"  TurboCat:  train={result.train_time:.3f}s, "
                      f"inference={result.inference_time:.4f}s ({result.inference_samples_per_sec:.0f} samples/s), "
                      f"AUC={result.roc_auc:.4f}")

        # CatBoost
        if CATBOOST_AVAILABLE:
            cb_model = CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=max_depth,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1
            )
            result = benchmark_model(
                cb_model, "CatBoost", X_train, X_test, y_train, y_test,
                f"size_{n_samples}"
            )
            if result:
                results.append(result)
                print(f"  CatBoost:  train={result.train_time:.3f}s, "
                      f"inference={result.inference_time:.4f}s ({result.inference_samples_per_sec:.0f} samples/s), "
                      f"AUC={result.roc_auc:.4f}")

    return results


def run_inference_stress_test(n_samples_train: int = 10000,
                               test_sizes: List[int] = [1000, 5000, 10000, 50000],
                               n_estimators: int = 100) -> List[BenchmarkResult]:
    """Stress test inference with different test set sizes."""
    results = []

    print("\n" + "=" * 80)
    print("BENCHMARK: Inference Stress Test")
    print(f"Training on {n_samples_train:,} samples, testing inference on varying sizes")
    print("=" * 80)

    # Create training dataset
    X_train, y_train = create_dataset(n_samples_train, n_features=20)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)

    # Train models once
    models = {}

    if TURBOCAT_AVAILABLE:
        tc_model = TurboCatClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=6,
            use_goss=True,
            n_jobs=-1,
            verbosity=0
        )
        tc_model.fit(X_train, y_train)
        models["TurboCat"] = tc_model

    if CATBOOST_AVAILABLE:
        cb_model = CatBoostClassifier(
            iterations=n_estimators,
            learning_rate=0.1,
            depth=6,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1
        )
        cb_model.fit(X_train, y_train)
        models["CatBoost"] = cb_model

    # Test inference on different sizes
    for test_size in test_sizes:
        print(f"\n--- Test Size: {test_size:,} samples ---")

        # Create test dataset
        X_test, y_test = create_dataset(test_size, n_features=20, random_state=123)
        X_test = scaler.transform(X_test).astype(np.float32)

        for model_name, model in models.items():
            # Warmup
            _ = model.predict(X_test[:100])

            # Measure inference multiple times
            inference_times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = model.predict(X_test)
                inference_times.append(time.perf_counter() - start)

            median_time = np.median(inference_times)
            samples_per_sec = test_size / median_time

            print(f"  {model_name}: {median_time*1000:.2f}ms ({samples_per_sec:,.0f} samples/s)")

            results.append(BenchmarkResult(
                model_name=model_name,
                dataset_name=f"inference_stress_{test_size}",
                n_samples=n_samples_train,
                n_features=20,
                train_time=0,  # Already trained
                inference_time=median_time,
                inference_samples_per_sec=samples_per_sec,
                accuracy=0,
                roc_auc=None,
                f1_score=0,
                n_trees=n_estimators
            ))

    return results


def print_comparison_table(results: List[BenchmarkResult]):
    """Print a comparison table of results."""
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 100)

    # Group by dataset
    datasets = sorted(set(r.dataset_name for r in results))

    print(f"\n{'Dataset':<25} {'Model':<12} {'Train(s)':<10} {'Infer(ms)':<12} "
          f"{'Samples/s':<12} {'AUC':<8} {'Acc':<8}")
    print("-" * 100)

    for dataset in datasets:
        dataset_results = [r for r in results if r.dataset_name == dataset]
        for r in dataset_results:
            auc_str = f"{r.roc_auc:.4f}" if r.roc_auc else "N/A"
            print(f"{r.dataset_name:<25} {r.model_name:<12} {r.train_time:<10.3f} "
                  f"{r.inference_time*1000:<12.2f} {r.inference_samples_per_sec:<12,.0f} "
                  f"{auc_str:<8} {r.accuracy:.4f}")

    # Calculate speedups
    print("\n" + "=" * 100)
    print("SPEEDUP ANALYSIS")
    print("=" * 100)

    tc_results = [r for r in results if r.model_name == "TurboCat"]
    cb_results = [r for r in results if r.model_name == "CatBoost"]

    if tc_results and cb_results:
        # Training speedup
        tc_train_times = {r.dataset_name: r.train_time for r in tc_results if r.train_time > 0}
        cb_train_times = {r.dataset_name: r.train_time for r in cb_results if r.train_time > 0}

        common_datasets = set(tc_train_times.keys()) & set(cb_train_times.keys())
        if common_datasets:
            train_speedups = [cb_train_times[d] / tc_train_times[d] for d in common_datasets
                             if tc_train_times[d] > 0]
            avg_train_speedup = np.mean(train_speedups) if train_speedups else 0
            print(f"\nTraining Speedup (CatBoost time / TurboCat time):")
            print(f"  Average: {avg_train_speedup:.2f}x")
            print(f"  TurboCat is {'FASTER' if avg_train_speedup > 1 else 'SLOWER'}")

        # Inference speedup
        tc_inf_times = {r.dataset_name: r.inference_time for r in tc_results}
        cb_inf_times = {r.dataset_name: r.inference_time for r in cb_results}

        common_datasets = set(tc_inf_times.keys()) & set(cb_inf_times.keys())
        if common_datasets:
            inf_speedups = [cb_inf_times[d] / tc_inf_times[d] for d in common_datasets
                           if tc_inf_times[d] > 0]
            avg_inf_speedup = np.mean(inf_speedups) if inf_speedups else 0
            print(f"\nInference Speedup (CatBoost time / TurboCat time):")
            print(f"  Average: {avg_inf_speedup:.2f}x")
            print(f"  TurboCat is {'FASTER' if avg_inf_speedup > 1 else 'SLOWER'}")

        # Quality comparison
        tc_aucs = [r.roc_auc for r in tc_results if r.roc_auc is not None]
        cb_aucs = [r.roc_auc for r in cb_results if r.roc_auc is not None]

        if tc_aucs and cb_aucs:
            avg_tc_auc = np.mean(tc_aucs)
            avg_cb_auc = np.mean(cb_aucs)
            print(f"\nQuality Comparison (Average AUC):")
            print(f"  TurboCat: {avg_tc_auc:.4f}")
            print(f"  CatBoost: {avg_cb_auc:.4f}")
            print(f"  Difference: {(avg_tc_auc - avg_cb_auc)*100:+.2f}%")


def main():
    print("=" * 80)
    print("TurboCat vs CatBoost Comprehensive Benchmark")
    print("=" * 80)

    if not TURBOCAT_AVAILABLE and not CATBOOST_AVAILABLE:
        print("Error: Neither TurboCat nor CatBoost is available!")
        sys.exit(1)

    all_results = []

    # 1. Size scaling benchmark
    sizes = [1000, 10000, 50000, 100000]
    results = run_size_benchmarks(sizes, n_estimators=100, max_depth=6)
    all_results.extend(results)

    # 2. Inference stress test
    results = run_inference_stress_test(
        n_samples_train=10000,
        test_sizes=[1000, 5000, 10000, 50000, 100000],
        n_estimators=100
    )
    all_results.extend(results)

    # Print summary
    print_comparison_table(all_results)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
