/**
 * TurboCat Symmetric (Oblivious) Tree Implementation
 *
 * Key optimization: ONE split per depth level instead of per node.
 * This dramatically reduces the complexity of split finding.
 */

#include "turbocat/symmetric_tree.hpp"
#include "turbocat/fast_ensemble.hpp"
#include "turbocat/fast_float_ensemble.hpp"
#include "turbocat/feature_orderings.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TURBOCAT_AVX2
#include <immintrin.h>
#endif

namespace turbocat {

// ============================================================================
// Symmetric Tree Implementation
// ============================================================================

SymmetricTree::SymmetricTree(const TreeConfig& config) : config_(config) {}

void SymmetricTree::build(
    const Dataset& dataset,
    const std::vector<Index>& sample_indices,
    HistogramBuilder& hist_builder
) {
    if (sample_indices.empty()) {
        depth_ = 0;
        leaf_values_.push_back(0.0f);
        return;
    }

    const Float* grads = dataset.gradients();
    const Float* hess = dataset.hessians();
    const FeatureIndex n_features = dataset.n_features();
    const BinIndex max_bins = config_.max_bins;
    const uint16_t max_depth = config_.max_depth;

    // PRE-ALLOCATE ALL MEMORY UPFRONT to avoid repeated allocations
    // Maximum nodes at any level = 2^max_depth
    const uint32_t max_nodes = 1u << max_depth;

    // Pre-allocate histogram pools (two pools for ping-pong)
    // Pool size = max_nodes at max_depth
    static thread_local std::vector<Histogram> hist_pool_a;
    static thread_local std::vector<Histogram> hist_pool_b;

    // Resize pools if needed (only grows, never shrinks)
    if (hist_pool_a.size() < max_nodes) {
        hist_pool_a.clear();
        hist_pool_a.reserve(max_nodes);
        for (uint32_t i = 0; i < max_nodes; ++i) {
            hist_pool_a.emplace_back(n_features, max_bins);
        }
    }
    if (hist_pool_b.size() < max_nodes) {
        hist_pool_b.clear();
        hist_pool_b.reserve(max_nodes);
        for (uint32_t i = 0; i < max_nodes; ++i) {
            hist_pool_b.emplace_back(n_features, max_bins);
        }
    }

    // Pre-allocate sample index pools (two pools for ping-pong)
    static thread_local std::vector<std::vector<Index>> samples_pool_a;
    static thread_local std::vector<std::vector<Index>> samples_pool_b;
    static thread_local std::vector<GradientPair> stats_pool_a;
    static thread_local std::vector<GradientPair> stats_pool_b;

    if (samples_pool_a.size() < max_nodes) {
        samples_pool_a.resize(max_nodes);
        samples_pool_b.resize(max_nodes);
    }
    if (stats_pool_a.size() < max_nodes) {
        stats_pool_a.resize(max_nodes);
        stats_pool_b.resize(max_nodes);
    }

    // Initialize current level (root)
    auto* current_samples = &samples_pool_a;
    auto* current_stats = &stats_pool_a;
    auto* current_histograms = &hist_pool_a;
    auto* next_samples = &samples_pool_b;
    auto* next_stats = &stats_pool_b;
    auto* next_histograms = &hist_pool_b;

    // Clear and initialize root
    (*current_samples)[0] = sample_indices;  // Copy
    (*current_stats)[0] = GradientPair{};
    for (Index idx : sample_indices) {
        (*current_stats)[0].grad += grads[idx];
        (*current_stats)[0].hess += hess[idx];
        (*current_stats)[0].count += 1;
    }

    splits_.clear();
    depth_ = 0;

    // Build tree level by level
    for (uint16_t d = 0; d < max_depth; ++d) {
        const uint32_t n_nodes = 1u << d;

        // Check if we should stop
        bool should_stop = true;
        for (uint32_t n = 0; n < n_nodes; ++n) {
            if ((*current_samples)[n].size() >= static_cast<size_t>(config_.min_samples_leaf * 2) &&
                (*current_stats)[n].hess >= config_.min_child_weight * 2) {
                should_stop = false;
                break;
            }
        }
        if (should_stop) break;

        // Find best split - INLINED to avoid creating view vectors
        SymmetricSplit best_split;
        const Float lambda = config_.lambda_l2;

        if (d == 0) {
            // ROOT LEVEL: Build histogram and find split
            (*current_histograms)[0].clear();
            hist_builder.build(dataset, (*current_samples)[0], {}, (*current_histograms)[0]);

            // Find best split from root histogram - PARALLEL over features
            Float parent_gain = 0.0f;
            if ((*current_stats)[0].hess > 1e-10f) {
                parent_gain = ((*current_stats)[0].grad * (*current_stats)[0].grad) / ((*current_stats)[0].hess + lambda);
            }

            int max_threads = 1;
            #ifdef _OPENMP
            max_threads = omp_get_max_threads();
            #endif

            std::vector<SymmetricSplit> thread_best(max_threads);
            for (int t = 0; t < max_threads; ++t) {
                thread_best[t].gain = -1e30f;
            }

            const GradientPair root_stat = (*current_stats)[0];

            #pragma omp parallel
            {
                int tid = 0;
                #ifdef _OPENMP
                tid = omp_get_thread_num();
                #endif

                SymmetricSplit& local_best = thread_best[tid];
                GradientPair left_sum;

                #pragma omp for nowait schedule(static)
                for (FeatureIndex f = 0; f < n_features; ++f) {
                    left_sum = GradientPair{};
                    const GradientPair* bins = (*current_histograms)[0].bins(f);

                    for (BinIndex b = 0; b < max_bins - 1; ++b) {
                        left_sum += bins[b];
                        Float right_grad = root_stat.grad - left_sum.grad;
                        Float right_hess = root_stat.hess - left_sum.hess;

                        if (left_sum.hess < 1e-10f || right_hess < 1e-10f) continue;

                        Float left_gain = (left_sum.grad * left_sum.grad) / (left_sum.hess + lambda);
                        Float right_gain = (right_grad * right_grad) / (right_hess + lambda);
                        Float gain = 0.5f * (left_gain + right_gain - parent_gain);

                        if (gain > local_best.gain) {
                            local_best.feature = f;
                            local_best.threshold = b;
                            local_best.gain = gain;
                        }
                    }
                }
            }

            best_split.gain = -1e30f;
            for (int t = 0; t < max_threads; ++t) {
                if (thread_best[t].gain > best_split.gain) {
                    best_split = thread_best[t];
                }
            }
        } else {
            // DEEPER LEVELS: Histograms pre-built via subtraction
            // Find best split across all nodes
            std::vector<Float> parent_gains(n_nodes);
            for (uint32_t n = 0; n < n_nodes; ++n) {
                if ((*current_stats)[n].hess > 1e-10f) {
                    parent_gains[n] = ((*current_stats)[n].grad * (*current_stats)[n].grad) / ((*current_stats)[n].hess + lambda);
                }
            }

            int max_threads = 1;
            #ifdef _OPENMP
            max_threads = omp_get_max_threads();
            #endif

            std::vector<SymmetricSplit> thread_best(max_threads);
            for (int t = 0; t < max_threads; ++t) {
                thread_best[t].gain = -1e30f;
            }

            #pragma omp parallel
            {
                int tid = 0;
                #ifdef _OPENMP
                tid = omp_get_thread_num();
                #endif

                SymmetricSplit& local_best = thread_best[tid];
                std::vector<GradientPair> node_left_sums(n_nodes);

                #pragma omp for nowait schedule(static)
                for (FeatureIndex f = 0; f < n_features; ++f) {
                    std::memset(node_left_sums.data(), 0, n_nodes * sizeof(GradientPair));

                    for (BinIndex b = 0; b < max_bins - 1; ++b) {
                        Float total_gain = 0.0f;

                        for (uint32_t n = 0; n < n_nodes; ++n) {
                            node_left_sums[n] += (*current_histograms)[n].bins(f)[b];

                            if ((*current_stats)[n].count == 0) continue;

                            const GradientPair& left = node_left_sums[n];
                            Float right_grad = (*current_stats)[n].grad - left.grad;
                            Float right_hess = (*current_stats)[n].hess - left.hess;

                            if (left.hess < 1e-10f || right_hess < 1e-10f) continue;

                            Float left_gain = (left.grad * left.grad) / (left.hess + lambda);
                            Float right_gain = (right_grad * right_grad) / (right_hess + lambda);
                            total_gain += 0.5f * (left_gain + right_gain - parent_gains[n]);
                        }

                        if (total_gain > local_best.gain) {
                            local_best.feature = f;
                            local_best.threshold = b;
                            local_best.gain = total_gain;
                        }
                    }
                }
            }

            best_split.gain = -1e30f;
            for (int t = 0; t < max_threads; ++t) {
                if (thread_best[t].gain > best_split.gain) {
                    best_split = thread_best[t];
                }
            }
        }

        // Look up float threshold
        const auto& bin_edges = dataset.bin_edges();
        if (best_split.gain > -1e29f && best_split.feature < bin_edges.size()) {
            const auto& edges = bin_edges[best_split.feature];
            if (best_split.threshold < edges.size()) {
                best_split.float_threshold = edges[best_split.threshold];
            } else if (!edges.empty()) {
                best_split.float_threshold = edges.back();
            }
        }

        // Check if split is valid
        if (best_split.gain < config_.min_split_gain) {
            break;
        }

        // Apply split
        splits_.push_back(best_split);
        depth_ = d + 1;

        // Prepare next level
        const uint32_t next_n_nodes = n_nodes * 2;

        // Clear next level stats and samples (parallel)
        #pragma omp parallel for schedule(static)
        for (uint32_t n = 0; n < next_n_nodes; ++n) {
            (*next_samples)[n].clear();
            (*next_stats)[n] = GradientPair{};
            (*next_histograms)[n].clear();
        }

        const BinIndex* split_bins = dataset.binned().column(best_split.feature);
        const BinIndex threshold = best_split.threshold;

        // PARALLEL partition: Each parent node is independent
        // Each thread handles one parent node and its two children
        #pragma omp parallel for schedule(dynamic)
        for (uint32_t n = 0; n < n_nodes; ++n) {
            const uint32_t left_child = n * 2;
            const uint32_t right_child = n * 2 + 1;

            // Local stats to avoid false sharing
            GradientPair left_stat{}, right_stat{};

            for (Index idx : (*current_samples)[n]) {
                const BinIndex bin = split_bins[idx];

                if (bin == 255 || bin <= threshold) {
                    (*next_samples)[left_child].push_back(idx);
                    left_stat.grad += grads[idx];
                    left_stat.hess += hess[idx];
                    left_stat.count += 1;
                } else {
                    (*next_samples)[right_child].push_back(idx);
                    right_stat.grad += grads[idx];
                    right_stat.hess += hess[idx];
                    right_stat.count += 1;
                }
            }

            // Write stats once
            (*next_stats)[left_child] = left_stat;
            (*next_stats)[right_child] = right_stat;
        }

        // HISTOGRAM SUBTRACTION TRICK: Build smaller child, compute larger via subtraction
        #pragma omp parallel for schedule(dynamic)
        for (uint32_t n = 0; n < n_nodes; ++n) {
            const uint32_t left_child = n * 2;
            const uint32_t right_child = n * 2 + 1;

            const bool left_is_smaller = (*next_samples)[left_child].size() <=
                                         (*next_samples)[right_child].size();
            const uint32_t smaller = left_is_smaller ? left_child : right_child;
            const uint32_t larger = left_is_smaller ? right_child : left_child;

            // Build histogram for smaller child only
            if (!(*next_samples)[smaller].empty()) {
                hist_builder.build(dataset, (*next_samples)[smaller], {},
                                   (*next_histograms)[smaller]);
            }

            // Compute larger child via subtraction
            (*next_histograms)[larger].subtract_from(
                (*current_histograms)[n], (*next_histograms)[smaller]);
        }

        // Swap pools
        std::swap(current_samples, next_samples);
        std::swap(current_stats, next_stats);
        std::swap(current_histograms, next_histograms);
    }

    // Compute leaf values
    uint32_t n_leaves = 1u << depth_;
    leaf_values_.resize(n_leaves);

    for (uint32_t i = 0; i < n_leaves; ++i) {
        if (i < current_stats->size()) {
            leaf_values_[i] = compute_leaf_value((*current_stats)[i]);
        } else {
            leaf_values_[i] = 0.0f;
        }
    }
}

SymmetricSplit SymmetricTree::find_best_level_split_with_histograms(
    const Dataset& dataset,
    const std::vector<std::vector<Index>>& node_samples,
    const std::vector<GradientPair>& node_stats,
    HistogramBuilder& hist_builder,
    std::vector<Histogram>& out_histograms
) {
    // Same as find_best_level_split but outputs histograms for reuse
    const FeatureIndex n_features = dataset.n_features();
    const BinIndex max_bins = config_.max_bins;
    const Float lambda = config_.lambda_l2;
    const uint32_t n_nodes = static_cast<uint32_t>(node_samples.size());
    const Index n_samples = dataset.n_samples();

    // Resize output histograms if needed
    out_histograms.resize(n_nodes, Histogram(n_features, max_bins));
    for (auto& h : out_histograms) {
        h.clear();
    }

    // Build histograms for each node
    // OPTIMIZATION: Use parallel histogram building when we have multiple nodes
    if (n_nodes == 1) {
        // FAST PATH: Root level - use optimized histogram builder directly
        if (!node_samples[0].empty()) {
            hist_builder.build(dataset, node_samples[0], {}, out_histograms[0]);
        }
    } else {
        // MULTI-NODE PATH: Build histograms for each node in parallel
        #pragma omp parallel for schedule(dynamic)
        for (uint32_t n = 0; n < n_nodes; ++n) {
            if (!node_samples[n].empty()) {
                hist_builder.build(dataset, node_samples[n], {}, out_histograms[n]);
            }
        }
    }
    (void)n_samples;  // May be unused when orderings disabled

    // Find best split (same logic as find_best_level_split)
    std::vector<Float> parent_gains(n_nodes);
    for (uint32_t n = 0; n < n_nodes; ++n) {
        if (node_stats[n].count > 0 && node_stats[n].hess > 1e-10f) {
            parent_gains[n] = (node_stats[n].grad * node_stats[n].grad) / (node_stats[n].hess + lambda);
        }
    }

    int max_threads = 1;
    #ifdef _OPENMP
    max_threads = omp_get_max_threads();
    #endif

    std::vector<SymmetricSplit> thread_best(max_threads);
    for (int t = 0; t < max_threads; ++t) {
        thread_best[t].gain = -1e30f;
    }

    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        SymmetricSplit& local_best = thread_best[tid];
        std::vector<GradientPair> node_left_sums(n_nodes);

        #pragma omp for nowait schedule(static)
        for (FeatureIndex f = 0; f < n_features; ++f) {
            std::memset(node_left_sums.data(), 0, n_nodes * sizeof(GradientPair));

            for (BinIndex b = 0; b < max_bins - 1; ++b) {
                Float total_gain = 0.0f;

                for (uint32_t n = 0; n < n_nodes; ++n) {
                    node_left_sums[n] += out_histograms[n].bins(f)[b];

                    if (node_stats[n].count == 0) continue;

                    const GradientPair& left = node_left_sums[n];
                    GradientPair right;
                    right.grad = node_stats[n].grad - left.grad;
                    right.hess = node_stats[n].hess - left.hess;

                    if (left.hess < 1e-10f || right.hess < 1e-10f) continue;

                    Float left_gain = (left.grad * left.grad) / (left.hess + lambda);
                    Float right_gain = (right.grad * right.grad) / (right.hess + lambda);
                    total_gain += 0.5f * (left_gain + right_gain - parent_gains[n]);
                }

                if (total_gain > local_best.gain) {
                    local_best.feature = f;
                    local_best.threshold = b;
                    local_best.gain = total_gain;
                }
            }
        }
    }

    SymmetricSplit best_split;
    best_split.gain = -1e30f;
    for (int t = 0; t < max_threads; ++t) {
        if (thread_best[t].gain > best_split.gain) {
            best_split = thread_best[t];
        }
    }

    // Look up float threshold
    const auto& bin_edges = dataset.bin_edges();
    if (best_split.gain > -1e29f && best_split.feature < bin_edges.size()) {
        const auto& edges = bin_edges[best_split.feature];
        if (best_split.threshold < edges.size()) {
            best_split.float_threshold = edges[best_split.threshold];
        } else if (!edges.empty()) {
            best_split.float_threshold = edges.back();
        }
    }

    return best_split;
}

SymmetricSplit SymmetricTree::find_best_level_split(
    const Dataset& dataset,
    const std::vector<std::vector<Index>>& node_samples,
    const std::vector<GradientPair>& node_stats,
    HistogramBuilder& hist_builder
) {
    const FeatureIndex n_features = dataset.n_features();
    const BinIndex max_bins = config_.max_bins;
    const Float lambda = config_.lambda_l2;
    const uint32_t n_nodes = static_cast<uint32_t>(node_samples.size());
    const Index n_samples = dataset.n_samples();

    std::vector<Histogram> node_histograms(n_nodes, Histogram(n_features, max_bins));

    // OPTIMIZATION: Use pre-sorted feature orderings if available
    if (dataset.has_orderings()) {
        // Build sample_to_node mapping
        std::vector<int32_t> sample_to_node(n_samples, -1);
        for (uint32_t n = 0; n < n_nodes; ++n) {
            for (Index idx : node_samples[n]) {
                sample_to_node[idx] = static_cast<int32_t>(n);
            }
        }

        // Use OrderedHistogramBuilder for sequential memory access
        OrderedHistogramBuilder ordered_builder;
        const FeatureOrderings& orderings = dataset.orderings();

        // Build histograms for all features in parallel, all nodes at once per feature
        #pragma omp parallel for schedule(static)
        for (FeatureIndex f = 0; f < n_features; ++f) {
            std::vector<GradientPair> feature_hists;
            ordered_builder.build_for_nodes(
                dataset, orderings, sample_to_node, n_nodes,
                f, feature_hists, max_bins
            );

            // Copy to output histograms
            for (uint32_t n = 0; n < n_nodes; ++n) {
                GradientPair* dest = node_histograms[n].bins(f);
                const GradientPair* src = feature_hists.data() + n * max_bins;
                std::memcpy(dest, src, max_bins * sizeof(GradientPair));
            }
        }
    } else {
        // Fallback to original histogram building
        if (n_nodes == 1) {
            // Root level: just build the single histogram
            if (!node_samples[0].empty()) {
                hist_builder.build(dataset, node_samples[0], {}, node_histograms[0]);
            }
        } else {
            // For deeper levels, build histograms in parallel
            #pragma omp parallel for schedule(dynamic)
            for (uint32_t n = 0; n < n_nodes; ++n) {
                if (!node_samples[n].empty()) {
                    hist_builder.build(dataset, node_samples[n], {}, node_histograms[n]);
                }
            }
        }
    }

    // OPTIMIZATION 2: Pre-compute parent gains (constant across all splits)
    std::vector<Float> parent_gains(n_nodes);
    for (uint32_t n = 0; n < n_nodes; ++n) {
        if (node_stats[n].count > 0 && node_stats[n].hess > 1e-10f) {
            parent_gains[n] = (node_stats[n].grad * node_stats[n].grad) / (node_stats[n].hess + lambda);
        } else {
            parent_gains[n] = 0.0f;
        }
    }

    // OPTIMIZATION 3: Lock-free split finding with array-based reduction
    int max_threads = 1;
    #ifdef _OPENMP
    max_threads = omp_get_max_threads();
    #endif

    std::vector<SymmetricSplit> thread_best(max_threads);
    for (int t = 0; t < max_threads; ++t) {
        thread_best[t].gain = -1e30f;
    }

    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        SymmetricSplit& local_best = thread_best[tid];

        // Pre-allocate per-thread vector
        std::vector<GradientPair> node_left_sums(n_nodes);

        #pragma omp for nowait schedule(static)
        for (FeatureIndex f = 0; f < n_features; ++f) {
            // Reset cumulative sums
            std::memset(node_left_sums.data(), 0, n_nodes * sizeof(GradientPair));

            // Scan through bins
            for (BinIndex b = 0; b < max_bins - 1; ++b) {
                Float total_gain = 0.0f;

                // OPTIMIZATION 4: Unroll loop for small n_nodes (common case)
                if (n_nodes == 1) {
                    node_left_sums[0] += node_histograms[0].bins(f)[b];
                    const GradientPair& left = node_left_sums[0];
                    GradientPair right;
                    right.grad = node_stats[0].grad - left.grad;
                    right.hess = node_stats[0].hess - left.hess;

                    if (left.hess >= 1e-10f && right.hess >= 1e-10f) {
                        Float left_gain = (left.grad * left.grad) / (left.hess + lambda);
                        Float right_gain = (right.grad * right.grad) / (right.hess + lambda);
                        total_gain = 0.5f * (left_gain + right_gain - parent_gains[0]);
                    }
                } else {
                    for (uint32_t n = 0; n < n_nodes; ++n) {
                        node_left_sums[n] += node_histograms[n].bins(f)[b];

                        if (node_stats[n].count == 0) continue;

                        const GradientPair& left = node_left_sums[n];
                        GradientPair right;
                        right.grad = node_stats[n].grad - left.grad;
                        right.hess = node_stats[n].hess - left.hess;

                        if (left.hess < 1e-10f || right.hess < 1e-10f) continue;

                        Float left_gain = (left.grad * left.grad) / (left.hess + lambda);
                        Float right_gain = (right.grad * right.grad) / (right.hess + lambda);
                        total_gain += 0.5f * (left_gain + right_gain - parent_gains[n]);
                    }
                }

                if (total_gain > local_best.gain) {
                    local_best.feature = f;
                    local_best.threshold = b;
                    local_best.gain = total_gain;
                }
            }
        }
    }

    // Final reduction (O(num_threads) - very fast)
    SymmetricSplit best_split;
    best_split.gain = -1e30f;
    for (int t = 0; t < max_threads; ++t) {
        if (thread_best[t].gain > best_split.gain) {
            best_split = thread_best[t];
        }
    }

    // Look up float threshold from bin edges
    const auto& bin_edges = dataset.bin_edges();
    if (best_split.gain > -1e29f && best_split.feature < bin_edges.size()) {
        const auto& edges = bin_edges[best_split.feature];
        if (best_split.threshold < edges.size()) {
            best_split.float_threshold = edges[best_split.threshold];
        } else if (!edges.empty()) {
            best_split.float_threshold = edges.back();
        }
    }

    return best_split;
}

SymmetricSplit SymmetricTree::find_best_split_from_histograms(
    const Dataset& dataset,
    const std::vector<Histogram>& histograms,
    const std::vector<GradientPair>& node_stats
) {
    const FeatureIndex n_features = dataset.n_features();
    const BinIndex max_bins = config_.max_bins;
    const Float lambda = config_.lambda_l2;
    const uint32_t n_nodes = static_cast<uint32_t>(histograms.size());

    // Pre-compute parent gains
    std::vector<Float> parent_gains(n_nodes);
    for (uint32_t n = 0; n < n_nodes; ++n) {
        if (node_stats[n].count > 0 && node_stats[n].hess > 1e-10f) {
            parent_gains[n] = (node_stats[n].grad * node_stats[n].grad) /
                             (node_stats[n].hess + lambda);
        }
    }

    int max_threads = 1;
    #ifdef _OPENMP
    max_threads = omp_get_max_threads();
    #endif

    std::vector<SymmetricSplit> thread_best(max_threads);
    for (int t = 0; t < max_threads; ++t) {
        thread_best[t].gain = -1e30f;
    }

    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        SymmetricSplit& local_best = thread_best[tid];
        std::vector<GradientPair> node_left_sums(n_nodes);

        #pragma omp for nowait schedule(static)
        for (FeatureIndex f = 0; f < n_features; ++f) {
            std::memset(node_left_sums.data(), 0, n_nodes * sizeof(GradientPair));

            for (BinIndex b = 0; b < max_bins - 1; ++b) {
                Float total_gain = 0.0f;

                for (uint32_t n = 0; n < n_nodes; ++n) {
                    node_left_sums[n] += histograms[n].bins(f)[b];
                    if (node_stats[n].count == 0) continue;

                    const GradientPair& left = node_left_sums[n];
                    GradientPair right;
                    right.grad = node_stats[n].grad - left.grad;
                    right.hess = node_stats[n].hess - left.hess;

                    if (left.hess < 1e-10f || right.hess < 1e-10f) continue;

                    Float left_gain = (left.grad * left.grad) / (left.hess + lambda);
                    Float right_gain = (right.grad * right.grad) / (right.hess + lambda);
                    total_gain += 0.5f * (left_gain + right_gain - parent_gains[n]);
                }

                if (total_gain > local_best.gain) {
                    local_best.feature = f;
                    local_best.threshold = b;
                    local_best.gain = total_gain;
                }
            }
        }
    }

    // Final reduction
    SymmetricSplit best_split;
    best_split.gain = -1e30f;
    for (int t = 0; t < max_threads; ++t) {
        if (thread_best[t].gain > best_split.gain) {
            best_split = thread_best[t];
        }
    }

    // Look up float threshold
    const auto& bin_edges = dataset.bin_edges();
    if (best_split.gain > -1e29f && best_split.feature < bin_edges.size()) {
        const auto& edges = bin_edges[best_split.feature];
        if (best_split.threshold < edges.size()) {
            best_split.float_threshold = edges[best_split.threshold];
        } else if (!edges.empty()) {
            best_split.float_threshold = edges.back();
        }
    }

    return best_split;
}

Float SymmetricTree::compute_leaf_value(const GradientPair& stats) const {
    if (stats.hess < 1e-10f) return 0.0f;

    Float value = -stats.grad / (stats.hess + config_.lambda_l2);

    // Apply L1 regularization (soft thresholding)
    if (config_.lambda_l1 > 0) {
        if (value > config_.lambda_l1) {
            value -= config_.lambda_l1;
        } else if (value < -config_.lambda_l1) {
            value += config_.lambda_l1;
        } else {
            value = 0.0f;
        }
    }

    // Clamp to max delta step
    if (config_.max_delta_step > 0) {
        value = std::max(-config_.max_delta_step, std::min(config_.max_delta_step, value));
    }

    return value;
}

uint32_t SymmetricTree::get_leaf_index(const Dataset& dataset, Index row) const {
    uint32_t leaf_idx = 0;

    for (uint16_t d = 0; d < depth_; ++d) {
        const SymmetricSplit& split = splits_[d];
        BinIndex bin = dataset.binned().column(split.feature)[row];

        // Must match training indexing: leaf_idx = 2*parent + direction
        // This is equivalent to: leaf_idx = (leaf_idx << 1) | direction
        bool go_right = (bin != 255 && bin > split.threshold);
        leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
    }

    return leaf_idx;
}

uint32_t SymmetricTree::get_leaf_index(const Float* features, FeatureIndex n_features) const {
    // For raw features, we'd need bin edges - not implemented yet
    // This would require storing bin edges
    return 0;
}

Float SymmetricTree::predict(const Dataset& dataset, Index row) const {
    if (depth_ == 0) {
        return leaf_values_.empty() ? 0.0f : leaf_values_[0];
    }

    uint32_t leaf_idx = get_leaf_index(dataset, row);
    return leaf_values_[leaf_idx];
}

Float SymmetricTree::predict(const Float* features, FeatureIndex n_features) const {
    // Not implemented for raw features
    return leaf_values_.empty() ? 0.0f : leaf_values_[0];
}

void SymmetricTree::predict_batch(const Dataset& dataset, Float* output) const {
    Index n_samples = dataset.n_samples();

    if (depth_ == 0) {
        Float val = leaf_values_.empty() ? 0.0f : leaf_values_[0];
        for (Index i = 0; i < n_samples; ++i) {
            output[i] = val;
        }
        return;
    }

    // Optimized batch prediction
    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        uint32_t leaf_idx = 0;

        for (uint16_t d = 0; d < depth_; ++d) {
            const SymmetricSplit& split = splits_[d];
            BinIndex bin = dataset.binned().column(split.feature)[i];

            // Must match training indexing: leaf_idx = 2*parent + direction
            bool go_right = (bin != 255 && bin > split.threshold);
            leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
        }

        output[i] = leaf_values_[leaf_idx];
    }
}

Float SymmetricTree::predict_raw(const Float* features, FeatureIndex n_features) const {
    if (depth_ == 0) {
        return leaf_values_.empty() ? 0.0f : leaf_values_[0];
    }

    uint32_t leaf_idx = 0;
    for (uint16_t d = 0; d < depth_; ++d) {
        const SymmetricSplit& split = splits_[d];
        Float val = features[split.feature];

        // NaN goes left (like binned prediction), otherwise compare directly
        // Use >= because binning uses upper_bound: bin = count of edges <= value
        // So bin > threshold iff value >= edge[threshold]
        bool go_right = !std::isnan(val) && val >= split.float_threshold;
        leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
    }

    return leaf_values_[leaf_idx];
}

void SymmetricTree::predict_batch_raw(const Float* data, Index n_samples, FeatureIndex n_features, Float* output) const {
    if (depth_ == 0) {
        Float val = leaf_values_.empty() ? 0.0f : leaf_values_[0];
        for (Index i = 0; i < n_samples; ++i) {
            output[i] = val;
        }
        return;
    }

    // Optimized batch prediction using raw float thresholds
    // No binning required - direct float comparisons
    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        const Float* row = data + i * n_features;
        uint32_t leaf_idx = 0;

        for (uint16_t d = 0; d < depth_; ++d) {
            const SymmetricSplit& split = splits_[d];
            Float val = row[split.feature];

            // NaN goes left, otherwise use >= to match binning behavior
            bool go_right = !std::isnan(val) && val >= split.float_threshold;
            leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
        }

        output[i] = leaf_values_[leaf_idx];
    }
}

std::vector<Float> SymmetricTree::feature_importance() const {
    std::vector<Float> importance(256, 0.0f);

    for (const auto& split : splits_) {
        importance[split.feature] += split.gain;
    }

    return importance;
}

// ============================================================================
// Symmetric Ensemble Implementation
// ============================================================================

// Constructors, destructor, and move operations must be defined here where FastEnsemble is complete
SymmetricEnsemble::SymmetricEnsemble() = default;
SymmetricEnsemble::SymmetricEnsemble(uint32_t n_classes) : n_classes_(n_classes) {}
SymmetricEnsemble::~SymmetricEnsemble() = default;
SymmetricEnsemble::SymmetricEnsemble(SymmetricEnsemble&&) noexcept = default;
SymmetricEnsemble& SymmetricEnsemble::operator=(SymmetricEnsemble&&) noexcept = default;

void SymmetricEnsemble::add_tree(std::unique_ptr<SymmetricTree> tree, Float weight) {
    trees_.push_back(std::move(tree));
    tree_weights_.push_back(weight);
    tree_class_indices_.push_back(0);
}

void SymmetricEnsemble::add_tree_for_class(std::unique_ptr<SymmetricTree> tree, Float weight, uint32_t class_idx) {
    trees_.push_back(std::move(tree));
    tree_weights_.push_back(weight);
    tree_class_indices_.push_back(class_idx);
}

Float SymmetricEnsemble::predict(const Dataset& data, Index row) const {
    Float sum = 0.0f;

    for (size_t i = 0; i < trees_.size(); ++i) {
        sum += tree_weights_[i] * trees_[i]->predict(data, row);
    }

    return sum;
}

void SymmetricEnsemble::predict_batch(const Dataset& data, Float* output) const {
    Index n_samples = data.n_samples();

    // Use FastEnsemble for SIMD-optimized prediction on large batches
    if (n_samples >= 16 && trees_.size() > 0) {
        prepare_fast_ensemble();

        if (fast_ensemble_ && !fast_ensemble_->empty()) {
            // Use column-major data directly (no transpose needed!)
            // This is much faster as we can load 8 consecutive bytes at once
            const BinIndex* col_data = data.binned().column(0);

            fast_ensemble_->predict_batch_column_major(col_data, n_samples, data.n_features(), output);
            return;
        }
    }

    // Fallback to original implementation
    std::memset(output, 0, n_samples * sizeof(Float));

    for (size_t t = 0; t < trees_.size(); ++t) {
        Float weight = tree_weights_[t];

        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            output[i] += weight * trees_[t]->predict(data, i);
        }
    }
}

void SymmetricEnsemble::predict_batch_raw(const Float* data, Index n_samples, FeatureIndex n_features, Float* output) const {
    // Raw float prediction - no binning required!
    // Uses float_threshold for direct comparisons like CatBoost
    // OPTIMIZED: Process all trees for each sample batch (better cache locality)

    if (trees_.empty()) {
        std::memset(output, 0, n_samples * sizeof(Float));
        return;
    }

    // Pre-extract tree info for faster access
    const size_t n_trees = trees_.size();
    const size_t max_depth = 16;  // Maximum depth we support

    // Flatten tree data for faster access
    std::vector<uint16_t> depths(n_trees);
    std::vector<Float> weights(n_trees);
    std::vector<FeatureIndex> features(n_trees * max_depth);
    std::vector<Float> thresholds(n_trees * max_depth);
    std::vector<const Float*> leaf_ptrs(n_trees);

    for (size_t t = 0; t < n_trees; ++t) {
        const SymmetricTree& tree = *trees_[t];
        depths[t] = tree.depth();
        weights[t] = tree_weights_[t];
        leaf_ptrs[t] = tree.leaf_values().data();

        const auto& splits = tree.splits();
        for (uint16_t d = 0; d < depths[t]; ++d) {
            features[t * max_depth + d] = splits[d].feature;
            thresholds[t * max_depth + d] = splits[d].float_threshold;
        }
    }

#ifdef TURBOCAT_AVX2
    // AVX2 SIMD optimized path - process 8 samples at a time
    Index n_simd = (n_samples / 8) * 8;

    #pragma omp parallel for schedule(static)
    for (Index base = 0; base < n_simd; base += 8) {
        __m256 sums = _mm256_setzero_ps();

        for (size_t t = 0; t < n_trees; ++t) {
            uint16_t depth = depths[t];
            Float weight = weights[t];
            const Float* leaves = leaf_ptrs[t];

            if (depth == 0) {
                sums = _mm256_add_ps(sums, _mm256_set1_ps(weight * leaves[0]));
                continue;
            }

            // Compute leaf indices for 8 samples
            __m256i indices = _mm256_setzero_si256();

            for (uint16_t d = 0; d < depth; ++d) {
                FeatureIndex feat = features[t * max_depth + d];
                Float thresh = thresholds[t * max_depth + d];

                // Load 8 feature values (strided access)
                alignas(32) float vals[8];
                for (int j = 0; j < 8; ++j) {
                    vals[j] = data[(base + j) * n_features + feat];
                }
                __m256 v_vals = _mm256_load_ps(vals);

                // NaN check: val != val for NaN
                __m256 v_nan_mask = _mm256_cmp_ps(v_vals, v_vals, _CMP_UNORD_Q);

                // Compare: val >= threshold (GE to match binning behavior)
                __m256 v_thresh = _mm256_set1_ps(thresh);
                __m256 v_cmp = _mm256_cmp_ps(v_vals, v_thresh, _CMP_GE_OQ);

                // Go right if not NaN and >= threshold
                __m256 v_go_right = _mm256_andnot_ps(v_nan_mask, v_cmp);

                // Update indices
                indices = _mm256_slli_epi32(indices, 1);
                __m256i v_bit = _mm256_and_si256(
                    _mm256_castps_si256(v_go_right),
                    _mm256_set1_epi32(1)
                );
                indices = _mm256_or_si256(indices, v_bit);
            }

            // Gather leaf values (scalar - AVX2 gather is slow)
            alignas(32) uint32_t idx_arr[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(idx_arr), indices);

            alignas(32) float leaf_vals[8];
            for (int j = 0; j < 8; ++j) {
                leaf_vals[j] = weight * leaves[idx_arr[j]];
            }
            sums = _mm256_add_ps(sums, _mm256_load_ps(leaf_vals));
        }

        _mm256_storeu_ps(output + base, sums);
    }

    // Handle remaining samples
    for (Index i = n_simd; i < n_samples; ++i) {
        Float sum = 0.0f;
        const Float* row = data + i * n_features;

        for (size_t t = 0; t < n_trees; ++t) {
            uint16_t depth = depths[t];
            const Float* leaves = leaf_ptrs[t];

            if (depth == 0) {
                sum += weights[t] * leaves[0];
                continue;
            }

            uint32_t leaf_idx = 0;
            for (uint16_t d = 0; d < depth; ++d) {
                Float val = row[features[t * max_depth + d]];
                bool go_right = !std::isnan(val) && val >= thresholds[t * max_depth + d];
                leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
            }
            sum += weights[t] * leaves[leaf_idx];
        }
        output[i] = sum;
    }

#else
    // Scalar fallback
    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        Float sum = 0.0f;
        const Float* row = data + i * n_features;

        for (size_t t = 0; t < n_trees; ++t) {
            uint16_t depth = depths[t];
            const Float* leaves = leaf_ptrs[t];

            if (depth == 0) {
                sum += weights[t] * leaves[0];
                continue;
            }

            uint32_t leaf_idx = 0;
            for (uint16_t d = 0; d < depth; ++d) {
                Float val = row[features[t * max_depth + d]];
                bool go_right = !std::isnan(val) && val >= thresholds[t * max_depth + d];
                leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
            }
            sum += weights[t] * leaves[leaf_idx];
        }
        output[i] = sum;
    }
#endif
}

void SymmetricEnsemble::prepare_fast_ensemble() const {
    if (fast_prepared_) return;

    fast_ensemble_ = std::make_unique<FastEnsemble>();
    fast_ensemble_->from_symmetric_ensemble(*this);
    fast_prepared_ = true;
}

void SymmetricEnsemble::prepare_fast_float_ensemble() const {
    if (fast_float_prepared_) return;

    fast_float_ensemble_ = std::make_unique<FastFloatEnsemble>();
    fast_float_ensemble_->from_symmetric_ensemble(*this);
    fast_float_prepared_ = true;
}

void SymmetricEnsemble::predict_batch_raw_fast(const Float* data, Index n_samples, FeatureIndex n_features, Float* output) const {
    if (trees_.empty()) {
        std::memset(output, 0, n_samples * sizeof(Float));
        return;
    }

    // Prepare cached fast float ensemble
    prepare_fast_float_ensemble();

    if (fast_float_ensemble_ && !fast_float_ensemble_->empty()) {
        // Use the optimized path with automatic transpose for large batches
        fast_float_ensemble_->predict_batch_with_transpose(data, n_samples, n_features, output);
    } else {
        // Fallback to original implementation
        predict_batch_raw(data, n_samples, n_features, output);
    }
}

void SymmetricEnsemble::predict_multiclass(const Dataset& data, Index row, Float* output) const {
    std::memset(output, 0, n_classes_ * sizeof(Float));

    for (size_t i = 0; i < trees_.size(); ++i) {
        uint32_t class_idx = tree_class_indices_[i];
        output[class_idx] += tree_weights_[i] * trees_[i]->predict(data, row);
    }
}

void SymmetricEnsemble::predict_batch_multiclass(const Dataset& data, Float* output) const {
    Index n_samples = data.n_samples();
    std::memset(output, 0, n_samples * n_classes_ * sizeof(Float));

    for (size_t t = 0; t < trees_.size(); ++t) {
        uint32_t class_idx = tree_class_indices_[t];
        Float weight = tree_weights_[t];

        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            output[i * n_classes_ + class_idx] += weight * trees_[t]->predict(data, i);
        }
    }
}

std::vector<Float> SymmetricEnsemble::feature_importance() const {
    std::vector<Float> total(256, 0.0f);

    for (const auto& tree : trees_) {
        auto imp = tree->feature_importance();
        for (size_t i = 0; i < imp.size(); ++i) {
            total[i] += imp[i];
        }
    }

    Float sum = std::accumulate(total.begin(), total.end(), 0.0f);
    if (sum > 0) {
        for (auto& v : total) {
            v /= sum;
        }
    }

    return total;
}

// ============================================================================
// SymmetricTree Serialization
// ============================================================================

void SymmetricTree::save(std::ostream& out) const {
    // Write depth
    out.write(reinterpret_cast<const char*>(&depth_), sizeof(depth_));

    // Write splits
    for (const auto& split : splits_) {
        out.write(reinterpret_cast<const char*>(&split.feature), sizeof(split.feature));
        out.write(reinterpret_cast<const char*>(&split.threshold), sizeof(split.threshold));
        out.write(reinterpret_cast<const char*>(&split.float_threshold), sizeof(split.float_threshold));
        out.write(reinterpret_cast<const char*>(&split.gain), sizeof(split.gain));
    }

    // Write leaf values
    size_t n_leaves = leaf_values_.size();
    out.write(reinterpret_cast<const char*>(&n_leaves), sizeof(n_leaves));
    out.write(reinterpret_cast<const char*>(leaf_values_.data()), n_leaves * sizeof(Float));
}

SymmetricTree SymmetricTree::load(std::istream& in) {
    SymmetricTree tree;

    // Read depth
    in.read(reinterpret_cast<char*>(&tree.depth_), sizeof(tree.depth_));

    // Read splits
    tree.splits_.resize(tree.depth_);
    for (uint16_t d = 0; d < tree.depth_; ++d) {
        in.read(reinterpret_cast<char*>(&tree.splits_[d].feature), sizeof(tree.splits_[d].feature));
        in.read(reinterpret_cast<char*>(&tree.splits_[d].threshold), sizeof(tree.splits_[d].threshold));
        in.read(reinterpret_cast<char*>(&tree.splits_[d].float_threshold), sizeof(tree.splits_[d].float_threshold));
        in.read(reinterpret_cast<char*>(&tree.splits_[d].gain), sizeof(tree.splits_[d].gain));
    }

    // Read leaf values
    size_t n_leaves;
    in.read(reinterpret_cast<char*>(&n_leaves), sizeof(n_leaves));
    tree.leaf_values_.resize(n_leaves);
    in.read(reinterpret_cast<char*>(tree.leaf_values_.data()), n_leaves * sizeof(Float));

    return tree;
}

} // namespace turbocat
