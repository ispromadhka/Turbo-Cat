/**
 * TurboCat Histogram Benchmarks
 */

#include <benchmark/benchmark.h>
#include "turbocat/histogram.hpp"
#include <vector>
#include <random>

using namespace turbocat;

// Benchmark histogram creation
static void BM_HistogramCreation(benchmark::State& state) {
    FeatureIndex n_features = state.range(0);
    
    for (auto _ : state) {
        Histogram hist(n_features, 255);
        benchmark::DoNotOptimize(hist);
    }
}
BENCHMARK(BM_HistogramCreation)->Range(8, 1024);

// Benchmark histogram clear
static void BM_HistogramClear(benchmark::State& state) {
    FeatureIndex n_features = state.range(0);
    Histogram hist(n_features, 255);
    
    for (auto _ : state) {
        hist.clear();
    }
}
BENCHMARK(BM_HistogramClear)->Range(8, 1024);

// Benchmark split finding
static void BM_SplitFinding(benchmark::State& state) {
    FeatureIndex n_features = state.range(0);
    BinIndex n_bins = 255;
    
    // Create histogram with random data
    Histogram hist(n_features, n_bins);
    std::mt19937 rng(42);
    std::uniform_real_distribution<Float> dist(-1.0f, 1.0f);
    
    GradientPair parent_sum;
    for (FeatureIndex f = 0; f < n_features; ++f) {
        for (BinIndex b = 0; b < n_bins; ++b) {
            hist.bin(f, b).grad = dist(rng);
            hist.bin(f, b).hess = std::abs(dist(rng)) + 0.1f;
            hist.bin(f, b).count = 10;
            parent_sum += hist.bin(f, b);
        }
    }
    
    TreeConfig config;
    SplitFinder finder(config);
    
    std::vector<FeatureIndex> features(n_features);
    std::iota(features.begin(), features.end(), static_cast<FeatureIndex>(0));
    
    for (auto _ : state) {
        auto split = finder.find_best_split(hist, parent_sum, features);
        benchmark::DoNotOptimize(split);
    }
}
BENCHMARK(BM_SplitFinding)->Range(8, 256);

BENCHMARK_MAIN();
