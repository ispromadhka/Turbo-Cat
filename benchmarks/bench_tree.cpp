/**
 * TurboCat Tree Benchmarks
 */

#include <benchmark/benchmark.h>
#include "turbocat/tree.hpp"
#include <vector>
#include <random>

using namespace turbocat;

// Benchmark GradTree prediction
static void BM_GradTreePredictHard(benchmark::State& state) {
    Index n_samples = state.range(0);
    FeatureIndex n_features = 50;
    
    TreeConfig config;
    config.max_depth = 6;
    
    GradTree tree(config, n_features);
    tree.initialize_random(42);
    
    // Generate random features
    std::vector<Float> features(n_samples * n_features);
    std::mt19937 rng(123);
    std::uniform_real_distribution<Float> dist(-1.0f, 1.0f);
    
    for (auto& f : features) {
        f = dist(rng);
    }
    
    for (auto _ : state) {
        for (Index i = 0; i < n_samples; ++i) {
            Float pred = tree.predict_hard(features.data() + i * n_features);
            benchmark::DoNotOptimize(pred);
        }
    }
    
    state.SetItemsProcessed(state.iterations() * n_samples);
}
BENCHMARK(BM_GradTreePredictHard)->Range(100, 10000);

// Benchmark tree ensemble prediction
static void BM_TreeEnsemblePrediction(benchmark::State& state) {
    Index n_samples = state.range(0);
    FeatureIndex n_features = 50;
    
    TreeEnsemble ensemble;
    
    // Add some trees (empty trees for this benchmark)
    TreeConfig config;
    for (int i = 0; i < 100; ++i) {
        auto tree = std::make_unique<Tree>(config);
        ensemble.add_tree(std::move(tree), 0.1f);
    }
    
    std::vector<Float> features(n_samples * n_features, 0.0f);
    std::vector<Float> output(n_samples);
    
    for (auto _ : state) {
        ensemble.predict_batch(features.data(), n_samples, n_features, output.data());
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * n_samples);
}
BENCHMARK(BM_TreeEnsemblePrediction)->Range(100, 10000);

BENCHMARK_MAIN();
