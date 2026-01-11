/**
 * TurboCat Tree Implementation
 */

#include "turbocat/tree.hpp"
#include "turbocat/flat_tree.hpp"
#include <algorithm>
#include <queue>
#include <stack>
#include <numeric>
#include <cstring>
#include <thread>
#include <vector>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TURBOCAT_AVX2
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

namespace turbocat {

// ============================================================================
// Optimized Sample Partitioning Helper
// ============================================================================

namespace {

// Parallel sample partitioning using two-pass counting approach
// Much faster than sequential push_back for large sample sets
inline void partition_samples_parallel(
    const std::vector<Index>& indices,
    const BinIndex* feature_bins,
    BinIndex threshold,
    uint8_t default_left,
    std::vector<Index>& left_indices,
    std::vector<Index>& right_indices,
    int n_threads
) {
    const size_t n = indices.size();

    // For small sizes, use simple sequential partitioning
    if (n < 10000 || n_threads <= 1) {
        left_indices.reserve(n / 2);
        right_indices.reserve(n / 2);

        for (Index idx : indices) {
            BinIndex bin = feature_bins[idx];
            if (bin == 255) {
                if (default_left) left_indices.push_back(idx);
                else right_indices.push_back(idx);
            } else if (bin <= threshold) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }
        return;
    }

    // PARALLEL TWO-PASS PARTITIONING
    // Pass 1: Count left/right per thread
    // Pass 2: Write to pre-allocated arrays with correct offsets

    #ifdef _OPENMP
    const int actual_threads = std::min(n_threads, omp_get_max_threads());
    #else
    const int actual_threads = 1;
    #endif

    std::vector<size_t> left_counts(actual_threads + 1, 0);
    std::vector<size_t> right_counts(actual_threads + 1, 0);

    // Pass 1: Count
    #pragma omp parallel num_threads(actual_threads)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        size_t local_left = 0, local_right = 0;
        size_t chunk_size = (n + actual_threads - 1) / actual_threads;
        size_t start = tid * chunk_size;
        size_t end = std::min(start + chunk_size, n);

        for (size_t i = start; i < end; ++i) {
            BinIndex bin = feature_bins[indices[i]];
            if (bin == 255) {
                if (default_left) local_left++;
                else local_right++;
            } else if (bin <= threshold) {
                local_left++;
            } else {
                local_right++;
            }
        }

        left_counts[tid + 1] = local_left;
        right_counts[tid + 1] = local_right;
    }

    // Prefix sum for offsets
    for (int t = 1; t <= actual_threads; ++t) {
        left_counts[t] += left_counts[t - 1];
        right_counts[t] += right_counts[t - 1];
    }

    // Pre-allocate output arrays
    left_indices.resize(left_counts[actual_threads]);
    right_indices.resize(right_counts[actual_threads]);

    // Pass 2: Write
    #pragma omp parallel num_threads(actual_threads)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        size_t chunk_size = (n + actual_threads - 1) / actual_threads;
        size_t start = tid * chunk_size;
        size_t end = std::min(start + chunk_size, n);

        size_t left_pos = left_counts[tid];
        size_t right_pos = right_counts[tid];

        for (size_t i = start; i < end; ++i) {
            Index idx = indices[i];
            BinIndex bin = feature_bins[idx];
            if (bin == 255) {
                if (default_left) left_indices[left_pos++] = idx;
                else right_indices[right_pos++] = idx;
            } else if (bin <= threshold) {
                left_indices[left_pos++] = idx;
            } else {
                right_indices[right_pos++] = idx;
            }
        }
    }
}

} // anonymous namespace

// ============================================================================
// Standard Decision Tree
// ============================================================================

Tree::Tree(const TreeConfig& config, uint32_t n_classes)
    : config_(config), n_classes_(n_classes) {
    nodes_.reserve(1 << config.max_depth);
}

void Tree::build(
    const Dataset& dataset,
    const std::vector<Index>& sample_indices,
    HistogramBuilder& hist_builder
) {
    nodes_.clear();
    n_leaves_ = 0;
    depth_ = 0;

    if (sample_indices.empty()) {
        return;
    }

    // Choose growth strategy based on config
    if (config_.grow_policy == GrowPolicy::Lossguide) {
        build_leafwise(dataset, sample_indices, hist_builder);
        return;
    }

    // Default: Depthwise (level-wise) growth with IN-PLACE partitioning
    // This avoids O(n) vector copies per node - major speedup!

    // Add root node
    TreeIndex root = add_node();

    // Compute total gradient/hessian
    GradientPair root_stats;
    const Float* grads = dataset.gradients();
    const Float* hess = dataset.hessians();

    for (Index idx : sample_indices) {
        root_stats.grad += grads[idx];
        root_stats.hess += hess[idx];
        root_stats.count += 1;
    }

    nodes_[root].stats = root_stats;

    // All features for root
    std::vector<FeatureIndex> all_features(dataset.n_features());
    std::iota(all_features.begin(), all_features.end(), static_cast<FeatureIndex>(0));

    // Pre-allocate histogram pool to avoid allocations during recursion
    std::vector<Histogram> hist_pool;
    hist_pool.reserve(config_.max_depth + 2);
    for (int i = 0; i < config_.max_depth + 2; ++i) {
        hist_pool.emplace_back(dataset.n_features(), config_.max_bins);
    }

    // Build root histogram
    hist_builder.build(dataset, sample_indices, all_features, hist_pool[0]);

    // Build recursively with histogram subtraction trick
    build_recursive_optimized(root, dataset, sample_indices, all_features,
                              hist_builder, hist_pool, 0, 0);
}

void Tree::build_recursive(
    TreeIndex node_idx,
    const Dataset& dataset,
    const std::vector<Index>& indices,
    const std::vector<FeatureIndex>& features,
    HistogramBuilder& hist_builder,
    Histogram& histogram,
    uint16_t current_depth
) {
    depth_ = std::max(depth_, current_depth);

    GradientPair node_stats = nodes_[node_idx].stats;

    bool should_stop =
        current_depth >= config_.max_depth ||
        n_leaves_ >= config_.max_leaves ||
        indices.size() < static_cast<size_t>(2 * config_.min_samples_leaf) ||
        node_stats.hess < 2 * config_.min_child_weight;

    if (should_stop) {
        make_leaf(node_idx, node_stats);
        return;
    }

    hist_builder.build(dataset, indices, features, histogram);

    SplitFinder finder(config_);
    SplitInfo best_split = finder.find_best_split(histogram, node_stats, features);

    if (!best_split.is_valid || best_split.gain < config_.min_split_gain) {
        make_leaf(node_idx, node_stats);
        return;
    }

    nodes_[node_idx].split_feature = best_split.feature_idx;
    nodes_[node_idx].split_bin = best_split.bin_threshold;
    nodes_[node_idx].is_leaf = 0;
    nodes_[node_idx].gain = best_split.gain;

    std::vector<Index> left_indices, right_indices;
    left_indices.reserve(indices.size() / 2);
    right_indices.reserve(indices.size() / 2);

    const BinIndex* feature_bins = dataset.binned().column(best_split.feature_idx);
    const BinIndex threshold = best_split.bin_threshold;
    uint8_t default_left = nodes_[node_idx].default_left;

    for (Index idx : indices) {
        BinIndex bin = feature_bins[idx];
        if (bin == 255) {
            if (default_left) left_indices.push_back(idx);
            else right_indices.push_back(idx);
        } else if (bin <= threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    if (config_.learn_missing_direction &&
        (left_indices.size() != indices.size() && right_indices.size() != indices.size())) {
        nodes_[node_idx].default_left = left_indices.size() >= right_indices.size() ? 1 : 0;
    }

    TreeIndex left_child = add_node();
    TreeIndex right_child = add_node();

    nodes_[node_idx].left_child = left_child;
    nodes_[node_idx].right_child = right_child;
    nodes_[left_child].stats = best_split.left_stats;
    nodes_[right_child].stats = best_split.right_stats;

    build_recursive(left_child, dataset, left_indices, features,
                   hist_builder, histogram, current_depth + 1);
    build_recursive(right_child, dataset, right_indices, features,
                   hist_builder, histogram, current_depth + 1);
}

// Optimized build with histogram subtraction trick
void Tree::build_recursive_optimized(
    TreeIndex node_idx,
    const Dataset& dataset,
    const std::vector<Index>& indices,
    const std::vector<FeatureIndex>& features,
    HistogramBuilder& hist_builder,
    std::vector<Histogram>& hist_pool,
    int hist_idx,
    uint16_t current_depth
) {
    depth_ = std::max(depth_, current_depth);

    GradientPair node_stats = nodes_[node_idx].stats;

    bool should_stop =
        current_depth >= config_.max_depth ||
        n_leaves_ >= config_.max_leaves ||
        indices.size() < static_cast<size_t>(2 * config_.min_samples_leaf) ||
        node_stats.hess < 2 * config_.min_child_weight;

    if (should_stop) {
        make_leaf(node_idx, node_stats);
        return;
    }

    // Current histogram is already built (passed from parent or root)
    Histogram& parent_hist = hist_pool[hist_idx];

    SplitFinder finder(config_);
    SplitInfo best_split = finder.find_best_split(parent_hist, node_stats, features);

    if (!best_split.is_valid || best_split.gain < config_.min_split_gain) {
        make_leaf(node_idx, node_stats);
        return;
    }

    nodes_[node_idx].split_feature = best_split.feature_idx;
    nodes_[node_idx].split_bin = best_split.bin_threshold;
    nodes_[node_idx].is_leaf = 0;
    nodes_[node_idx].gain = best_split.gain;

    // OPTIMIZED: Use parallel partitioning for large sample sets
    std::vector<Index> left_indices, right_indices;
    const BinIndex* feature_bins = dataset.binned().column(best_split.feature_idx);
    const BinIndex threshold = best_split.bin_threshold;
    uint8_t default_left = nodes_[node_idx].default_left;

    #ifdef _OPENMP
    int n_threads = omp_get_max_threads();
    #else
    int n_threads = 1;
    #endif

    partition_samples_parallel(
        indices, feature_bins, threshold, default_left,
        left_indices, right_indices, n_threads
    );

    if (config_.learn_missing_direction &&
        left_indices.size() != indices.size() && right_indices.size() != indices.size()) {
        nodes_[node_idx].default_left = left_indices.size() >= right_indices.size() ? 1 : 0;
    }

    TreeIndex left_child = add_node();
    TreeIndex right_child = add_node();

    nodes_[node_idx].left_child = left_child;
    nodes_[node_idx].right_child = right_child;
    nodes_[left_child].stats = best_split.left_stats;
    nodes_[right_child].stats = best_split.right_stats;

    // HISTOGRAM SUBTRACTION TRICK:
    // Only build histogram for smaller child, compute larger via subtraction
    bool left_is_smaller = left_indices.size() <= right_indices.size();
    int child_hist_idx = hist_idx + 1;  // Use next slot in pool

    if (left_is_smaller) {
        // Build histogram for smaller (left) child
        hist_builder.build(dataset, left_indices, features, hist_pool[child_hist_idx]);

        // Process left child first (it has the built histogram)
        build_recursive_optimized(left_child, dataset, left_indices, features,
                                  hist_builder, hist_pool, child_hist_idx, current_depth + 1);

        // Compute right histogram via subtraction: right = parent - left
        hist_pool[child_hist_idx].subtract_from(parent_hist, hist_pool[child_hist_idx]);

        // Process right child with subtracted histogram
        build_recursive_optimized(right_child, dataset, right_indices, features,
                                  hist_builder, hist_pool, child_hist_idx, current_depth + 1);
    } else {
        // Build histogram for smaller (right) child
        hist_builder.build(dataset, right_indices, features, hist_pool[child_hist_idx]);

        // Process right child first
        build_recursive_optimized(right_child, dataset, right_indices, features,
                                  hist_builder, hist_pool, child_hist_idx, current_depth + 1);

        // Compute left histogram via subtraction: left = parent - right
        hist_pool[child_hist_idx].subtract_from(parent_hist, hist_pool[child_hist_idx]);

        // Process left child with subtracted histogram
        build_recursive_optimized(left_child, dataset, left_indices, features,
                                  hist_builder, hist_pool, child_hist_idx, current_depth + 1);
    }
}

// ============================================================================
// ULTRA-OPTIMIZED: In-place partitioning (no vector copies)
// ============================================================================

void Tree::build_inplace(
    const Dataset& dataset,
    std::vector<Index>& indices,
    size_t start, size_t end,
    const std::vector<FeatureIndex>& features,
    HistogramBuilder& hist_builder,
    std::vector<Histogram>& hist_pool,
    int hist_idx,
    TreeIndex node_idx,
    uint16_t current_depth
) {
    const size_t n_samples = end - start;
    depth_ = std::max(depth_, current_depth);

    GradientPair node_stats = nodes_[node_idx].stats;

    bool should_stop =
        current_depth >= config_.max_depth ||
        n_leaves_ >= config_.max_leaves ||
        n_samples < static_cast<size_t>(2 * config_.min_samples_leaf) ||
        node_stats.hess < 2 * config_.min_child_weight;

    if (should_stop) {
        make_leaf(node_idx, node_stats);
        return;
    }

    // Current histogram is already built (passed from parent or root)
    Histogram& parent_hist = hist_pool[hist_idx];

    SplitFinder finder(config_);
    SplitInfo best_split = finder.find_best_split(parent_hist, node_stats, features);

    if (!best_split.is_valid || best_split.gain < config_.min_split_gain) {
        make_leaf(node_idx, node_stats);
        return;
    }

    nodes_[node_idx].split_feature = best_split.feature_idx;
    nodes_[node_idx].split_bin = best_split.bin_threshold;
    nodes_[node_idx].is_leaf = 0;
    nodes_[node_idx].gain = best_split.gain;

    // IN-PLACE PARTITIONING: like quicksort partition
    // Move "left" samples to [start, mid) and "right" samples to [mid, end)
    const BinIndex* feature_bins = dataset.binned().column(best_split.feature_idx);
    const BinIndex threshold = best_split.bin_threshold;
    const uint8_t default_left = nodes_[node_idx].default_left;

    size_t write_left = start;
    size_t write_right = end;

    // Single pass partition: scan from left, swap elements to appropriate side
    for (size_t read = start; read < write_right; ) {
        Index idx = indices[read];
        BinIndex bin = feature_bins[idx];

        bool goes_left = (bin == 255) ? default_left : (bin <= threshold);

        if (goes_left) {
            // Element belongs to left - keep it and advance
            if (read != write_left) {
                std::swap(indices[read], indices[write_left]);
            }
            ++write_left;
            ++read;
        } else {
            // Element belongs to right - swap with last unprocessed
            --write_right;
            std::swap(indices[read], indices[write_right]);
            // Don't advance read - we need to check the swapped element
        }
    }

    size_t mid = write_left;  // Partition point
    size_t left_count = mid - start;
    size_t right_count = end - mid;

    // Learn missing direction if needed
    if (config_.learn_missing_direction && left_count > 0 && right_count > 0) {
        nodes_[node_idx].default_left = (left_count >= right_count) ? 1 : 0;
    }

    TreeIndex left_child = add_node();
    TreeIndex right_child = add_node();

    nodes_[node_idx].left_child = left_child;
    nodes_[node_idx].right_child = right_child;
    nodes_[left_child].stats = best_split.left_stats;
    nodes_[right_child].stats = best_split.right_stats;

    // HISTOGRAM SUBTRACTION TRICK with in-place indices
    bool left_is_smaller = left_count <= right_count;
    int child_hist_idx = hist_idx + 1;

    // Create a temporary view for histogram building
    // Note: hist_builder expects a vector, but indices[start:end] is contiguous
    // We can create a reference-counted view or just use the range

    if (left_is_smaller) {
        // Build histogram for smaller (left) child using range (NO COPY!)
        hist_builder.build_range(dataset, indices.data(), start, mid, features, hist_pool[child_hist_idx]);

        // Process left child first
        build_inplace(dataset, indices, start, mid, features,
                      hist_builder, hist_pool, child_hist_idx, left_child, current_depth + 1);

        // Compute right histogram via subtraction
        hist_pool[child_hist_idx].subtract_from(parent_hist, hist_pool[child_hist_idx]);

        // Process right child
        build_inplace(dataset, indices, mid, end, features,
                      hist_builder, hist_pool, child_hist_idx, right_child, current_depth + 1);
    } else {
        // Build histogram for smaller (right) child using range (NO COPY!)
        hist_builder.build_range(dataset, indices.data(), mid, end, features, hist_pool[child_hist_idx]);

        // Process right child first
        build_inplace(dataset, indices, mid, end, features,
                      hist_builder, hist_pool, child_hist_idx, right_child, current_depth + 1);

        // Compute left histogram via subtraction
        hist_pool[child_hist_idx].subtract_from(parent_hist, hist_pool[child_hist_idx]);

        // Process left child
        build_inplace(dataset, indices, start, mid, features,
                      hist_builder, hist_pool, child_hist_idx, left_child, current_depth + 1);
    }
}

// Optimized recursive build using pre-computed histogram
void Tree::build_recursive_with_hist(
    TreeIndex node_idx,
    const Dataset& dataset,
    const std::vector<Index>& indices,
    const std::vector<FeatureIndex>& features,
    HistogramBuilder& hist_builder,
    Histogram& histogram,
    uint16_t current_depth
) {
    depth_ = std::max(depth_, current_depth);

    GradientPair node_stats = nodes_[node_idx].stats;

    bool should_stop =
        current_depth >= config_.max_depth ||
        n_leaves_ >= config_.max_leaves ||
        indices.size() < static_cast<size_t>(2 * config_.min_samples_leaf) ||
        node_stats.hess < 2 * config_.min_child_weight;

    if (should_stop) {
        make_leaf(node_idx, node_stats);
        return;
    }

    // Histogram is already computed via subtraction - just find split
    SplitFinder finder(config_);
    SplitInfo best_split = finder.find_best_split(histogram, node_stats, features);

    if (!best_split.is_valid || best_split.gain < config_.min_split_gain) {
        make_leaf(node_idx, node_stats);
        return;
    }

    nodes_[node_idx].split_feature = best_split.feature_idx;
    nodes_[node_idx].split_bin = best_split.bin_threshold;
    nodes_[node_idx].is_leaf = 0;
    nodes_[node_idx].gain = best_split.gain;

    std::vector<Index> left_indices, right_indices;
    left_indices.reserve(indices.size() / 2);
    right_indices.reserve(indices.size() / 2);

    const BinIndex* feature_bins = dataset.binned().column(best_split.feature_idx);
    const BinIndex threshold = best_split.bin_threshold;
    uint8_t default_left = nodes_[node_idx].default_left;

    for (Index idx : indices) {
        BinIndex bin = feature_bins[idx];
        if (bin == 255) {
            if (default_left) left_indices.push_back(idx);
            else right_indices.push_back(idx);
        } else if (bin <= threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    if (config_.learn_missing_direction &&
        (left_indices.size() != indices.size() && right_indices.size() != indices.size())) {
        nodes_[node_idx].default_left = left_indices.size() >= right_indices.size() ? 1 : 0;
    }

    TreeIndex left_child = add_node();
    TreeIndex right_child = add_node();

    nodes_[node_idx].left_child = left_child;
    nodes_[node_idx].right_child = right_child;
    nodes_[left_child].stats = best_split.left_stats;
    nodes_[right_child].stats = best_split.right_stats;

    // Save parent histogram for subtraction
    Histogram parent_hist(histogram.n_features(), histogram.max_bins());
    std::memcpy(parent_hist.data(), histogram.data(),
                histogram.n_features() * histogram.max_bins() * sizeof(GradientPair));

    bool left_is_smaller = left_indices.size() <= right_indices.size();

    if (left_is_smaller) {
        // Build histogram for smaller (left) child
        Histogram left_hist(histogram.n_features(), histogram.max_bins());
        hist_builder.build(dataset, left_indices, features, left_hist);

        // Compute right histogram via subtraction
        Histogram right_hist(histogram.n_features(), histogram.max_bins());
        right_hist.subtract_from(parent_hist, left_hist);

        // Recursively build children
        build_recursive_with_hist(left_child, dataset, left_indices, features,
                                 hist_builder, left_hist, current_depth + 1);
        build_recursive_with_hist(right_child, dataset, right_indices, features,
                                 hist_builder, right_hist, current_depth + 1);
    } else {
        // Build histogram for smaller (right) child
        Histogram right_hist(histogram.n_features(), histogram.max_bins());
        hist_builder.build(dataset, right_indices, features, right_hist);

        // Compute left histogram via subtraction
        Histogram left_hist(histogram.n_features(), histogram.max_bins());
        left_hist.subtract_from(parent_hist, right_hist);

        // Recursively build children
        build_recursive_with_hist(left_child, dataset, left_indices, features,
                                 hist_builder, left_hist, current_depth + 1);
        build_recursive_with_hist(right_child, dataset, right_indices, features,
                                 hist_builder, right_hist, current_depth + 1);
    }
}

TreeIndex Tree::add_node() {
    TreeIndex idx = static_cast<TreeIndex>(nodes_.size());
    nodes_.emplace_back();
    return idx;
}

void Tree::make_leaf(TreeIndex node_idx, const GradientPair& stats) {
    TreeNode& node = nodes_[node_idx];
    node.is_leaf = 1;

    // Compute leaf value: -G / (H + λ)
    Float raw_value = -stats.grad / (stats.hess + config_.lambda_l2);

    // Optional smoothing based on sample count (reduces overfitting on small leaves)
    // shrink = count / (count + smooth_weight)
    if (config_.leaf_smooth > 0.0f) {
        Float shrink_factor = static_cast<Float>(stats.count) /
                             (static_cast<Float>(stats.count) + config_.leaf_smooth);
        node.value = raw_value * shrink_factor;
    } else {
        node.value = raw_value;
    }

    // Apply delta step constraint
    if (config_.max_delta_step > 0) {
        node.value = std::max(-config_.max_delta_step,
                             std::min(config_.max_delta_step, node.value));
    }

    n_leaves_++;
}

// ============================================================================
// Leaf-wise (Loss-guided) Tree Building
// ============================================================================

void Tree::build_leafwise(
    const Dataset& dataset,
    const std::vector<Index>& sample_indices,
    HistogramBuilder& hist_builder
) {
    // Leaf-wise tree building (LightGBM style) with histogram subtraction trick:
    // - Always split the leaf with highest potential gain
    // - Build histogram only for smaller child
    // - Compute larger child histogram via: parent - smaller_child
    // This halves histogram building cost on average.

    struct LeafCandidate {
        TreeIndex node_idx;
        SplitInfo split;
        std::vector<Index> indices;
        uint16_t depth;
        GradientPair stats;
        std::shared_ptr<Histogram> histogram;  // Store histogram for subtraction trick

        bool operator<(const LeafCandidate& other) const {
            return split.gain < other.split.gain;  // Max-heap
        }
    };

    std::priority_queue<LeafCandidate> candidates;

    // All features for splitting
    std::vector<FeatureIndex> all_features(dataset.n_features());
    std::iota(all_features.begin(), all_features.end(), static_cast<FeatureIndex>(0));

    // Create root node
    TreeIndex root = add_node();

    // Compute root statistics
    GradientPair root_stats;
    const Float* grads = dataset.gradients();
    const Float* hess = dataset.hessians();

    for (Index idx : sample_indices) {
        root_stats.grad += grads[idx];
        root_stats.hess += hess[idx];
        root_stats.count += 1;
    }
    nodes_[root].stats = root_stats;

    // Build histogram for root
    auto root_hist = std::make_shared<Histogram>(dataset.n_features(), config_.max_bins);
    hist_builder.build(dataset, sample_indices, all_features, *root_hist);

    // Find best split for root
    SplitFinder finder(config_);
    SplitInfo root_split = finder.find_best_split(*root_hist, root_stats, all_features);

    if (root_split.is_valid && root_split.gain >= config_.min_split_gain) {
        LeafCandidate root_candidate;
        root_candidate.node_idx = root;
        root_candidate.split = root_split;
        root_candidate.indices = sample_indices;
        root_candidate.depth = 0;
        root_candidate.stats = root_stats;
        root_candidate.histogram = root_hist;
        candidates.push(std::move(root_candidate));
    } else {
        // Root becomes a leaf
        make_leaf(root, root_stats);
        return;
    }

    // Grow tree by always splitting the leaf with highest gain
    while (!candidates.empty() && n_leaves_ < config_.max_leaves) {
        LeafCandidate best = std::move(const_cast<LeafCandidate&>(candidates.top()));
        candidates.pop();

        TreeIndex node_idx = best.node_idx;
        const SplitInfo& split = best.split;
        uint16_t current_depth = best.depth;

        // Check depth constraint
        if (current_depth >= config_.max_depth) {
            make_leaf(node_idx, best.stats);
            continue;
        }

        // Apply the split
        nodes_[node_idx].split_feature = split.feature_idx;
        nodes_[node_idx].split_bin = split.bin_threshold;
        nodes_[node_idx].is_leaf = 0;
        nodes_[node_idx].gain = split.gain;

        // OPTIMIZED: Partition samples using parallel two-pass approach
        std::vector<Index> left_indices, right_indices;
        const BinIndex* feature_bins = dataset.binned().column(split.feature_idx);
        const BinIndex threshold = split.bin_threshold;
        bool default_left = nodes_[node_idx].default_left;

        #ifdef _OPENMP
        int n_threads = omp_get_max_threads();
        #else
        int n_threads = 1;
        #endif

        partition_samples_parallel(
            best.indices, feature_bins, threshold, default_left ? 1 : 0,
            left_indices, right_indices, n_threads
        );

        // Learn missing value direction
        if (config_.learn_missing_direction &&
            left_indices.size() != best.indices.size() && right_indices.size() != best.indices.size()) {
            nodes_[node_idx].default_left = left_indices.size() >= right_indices.size() ? 1 : 0;
        }

        // Create child nodes
        TreeIndex left_child = add_node();
        TreeIndex right_child = add_node();

        nodes_[node_idx].left_child = left_child;
        nodes_[node_idx].right_child = right_child;
        nodes_[left_child].stats = split.left_stats;
        nodes_[right_child].stats = split.right_stats;

        depth_ = std::max(depth_, static_cast<uint16_t>(current_depth + 1));

        // Determine which child is smaller (for histogram subtraction trick)
        bool left_is_smaller = left_indices.size() <= right_indices.size();

        // Check which children can be split
        bool left_can_split =
            left_indices.size() >= static_cast<size_t>(2 * config_.min_samples_leaf) &&
            split.left_stats.hess >= 2 * config_.min_child_weight;
        bool right_can_split =
            right_indices.size() >= static_cast<size_t>(2 * config_.min_samples_leaf) &&
            split.right_stats.hess >= 2 * config_.min_child_weight;

        // Histograms for children (use subtraction trick)
        std::shared_ptr<Histogram> left_hist, right_hist;

        // Build histogram for smaller child, compute larger via subtraction
        if (left_can_split || right_can_split) {
            if (left_is_smaller) {
                // Build left histogram (smaller child)
                if (left_can_split) {
                    left_hist = std::make_shared<Histogram>(dataset.n_features(), config_.max_bins);
                    hist_builder.build(dataset, left_indices, all_features, *left_hist);
                }
                // Compute right histogram via subtraction (larger child)
                if (right_can_split) {
                    right_hist = std::make_shared<Histogram>(dataset.n_features(), config_.max_bins);
                    if (left_hist) {
                        right_hist->subtract_from(*best.histogram, *left_hist);
                    } else {
                        // Left can't split but right can - need to build right directly
                        hist_builder.build(dataset, right_indices, all_features, *right_hist);
                    }
                }
            } else {
                // Build right histogram (smaller child)
                if (right_can_split) {
                    right_hist = std::make_shared<Histogram>(dataset.n_features(), config_.max_bins);
                    hist_builder.build(dataset, right_indices, all_features, *right_hist);
                }
                // Compute left histogram via subtraction (larger child)
                if (left_can_split) {
                    left_hist = std::make_shared<Histogram>(dataset.n_features(), config_.max_bins);
                    if (right_hist) {
                        left_hist->subtract_from(*best.histogram, *right_hist);
                    } else {
                        // Right can't split but left can - need to build left directly
                        hist_builder.build(dataset, left_indices, all_features, *left_hist);
                    }
                }
            }
        }

        // Process left child
        if (left_can_split && n_leaves_ + 1 < config_.max_leaves) {
            SplitInfo left_split = finder.find_best_split(*left_hist, split.left_stats, all_features);

            if (left_split.is_valid && left_split.gain >= config_.min_split_gain) {
                LeafCandidate left_candidate;
                left_candidate.node_idx = left_child;
                left_candidate.split = left_split;
                left_candidate.indices = std::move(left_indices);
                left_candidate.depth = current_depth + 1;
                left_candidate.stats = split.left_stats;
                left_candidate.histogram = left_hist;
                candidates.push(std::move(left_candidate));
            } else {
                make_leaf(left_child, split.left_stats);
            }
        } else {
            make_leaf(left_child, split.left_stats);
        }

        // Process right child
        if (right_can_split && n_leaves_ + 1 < config_.max_leaves) {
            SplitInfo right_split = finder.find_best_split(*right_hist, split.right_stats, all_features);

            if (right_split.is_valid && right_split.gain >= config_.min_split_gain) {
                LeafCandidate right_candidate;
                right_candidate.node_idx = right_child;
                right_candidate.split = right_split;
                right_candidate.indices = std::move(right_indices);
                right_candidate.depth = current_depth + 1;
                right_candidate.stats = split.right_stats;
                right_candidate.histogram = right_hist;
                candidates.push(std::move(right_candidate));
            } else {
                make_leaf(right_child, split.right_stats);
            }
        } else {
            make_leaf(right_child, split.right_stats);
        }
    }

    // Make remaining candidates into leaves
    while (!candidates.empty()) {
        LeafCandidate remaining = std::move(const_cast<LeafCandidate&>(candidates.top()));
        candidates.pop();
        make_leaf(remaining.node_idx, remaining.stats);
    }
}

Float Tree::predict(const Float* features, FeatureIndex n_features) const {
    if (nodes_.empty()) return 0.0f;
    
    TreeIndex node_idx = 0;
    
    while (!nodes_[node_idx].is_leaf) {
        const TreeNode& node = nodes_[node_idx];
        Float value = features[node.split_feature];
        
        // Handle missing
        if (std::isnan(value)) {
            node_idx = node.default_left ? node.left_child : node.right_child;
        } else {
            node_idx = (value <= static_cast<Float>(node.split_bin)) 
                      ? node.left_child : node.right_child;
        }
    }
    
    return nodes_[node_idx].value;
}

Float Tree::predict(const Dataset& dataset, Index row) const {
    if (nodes_.empty()) {
        return 0.0f;
    }
    
    TreeIndex node_idx = 0;
    
    while (!nodes_[node_idx].is_leaf) {
        const TreeNode& node = nodes_[node_idx];
        BinIndex bin = dataset.binned().get(row, node.split_feature);
        
        if (bin == 255) {  // NaN
            node_idx = node.default_left ? node.left_child : node.right_child;
        } else {
            // Branchless version using arithmetic
            const int go_right = (bin > node.split_bin);
            node_idx = go_right * node.right_child + (1 - go_right) * node.left_child;
        }
    }
    
    return nodes_[node_idx].value;
}

void Tree::predict_batch(
    const Float* features,
    Index n_samples,
    FeatureIndex n_features,
    Float* output
) const {
    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        output[i] = predict(features + i * n_features, n_features);
    }
}

void Tree::predict_batch(const Dataset& data, Float* output) const {
    Index n_samples = data.n_samples();

    if (nodes_.empty()) {
        std::memset(output, 0, n_samples * sizeof(Float));
        return;
    }

    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        output[i] = predict(data, i);
    }
}

void Tree::update_leaf_values(const Dataset& dataset, const std::vector<Index>& indices) {
    // Recompute leaf values based on current gradients
    std::vector<GradientPair> leaf_stats(nodes_.size());
    
    for (Index idx : indices) {
        TreeIndex node_idx = 0;
        
        while (!nodes_[node_idx].is_leaf) {
            const TreeNode& node = nodes_[node_idx];
            BinIndex bin = dataset.binned().get(idx, node.split_feature);
            
            if (bin == 255) {
                node_idx = node.default_left ? node.left_child : node.right_child;
            } else {
                node_idx = (bin <= node.split_bin) ? node.left_child : node.right_child;
            }
        }
        
        leaf_stats[node_idx].grad += dataset.gradients()[idx];
        leaf_stats[node_idx].hess += dataset.hessians()[idx];
        leaf_stats[node_idx].count += 1;
    }
    
    for (TreeIndex i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].is_leaf && leaf_stats[i].count > 0) {
            Float raw_value = -leaf_stats[i].grad / (leaf_stats[i].hess + config_.lambda_l2);
            if (config_.leaf_smooth > 0.0f) {
                Float shrink_factor = static_cast<Float>(leaf_stats[i].count) /
                                     (static_cast<Float>(leaf_stats[i].count) + config_.leaf_smooth);
                nodes_[i].value = raw_value * shrink_factor;
            } else {
                nodes_[i].value = raw_value;
            }
        }
    }
}

std::vector<Float> Tree::feature_importance() const {
    std::vector<Float> importance(256, 0.0f);  // Assume max 256 features

    for (const auto& node : nodes_) {
        if (!node.is_leaf && node.gain > 0) {
            importance[node.split_feature] += node.gain;
        }
    }

    return importance;
}

// ============================================================================
// Multiclass Support
// ============================================================================

void Tree::build_multiclass(
    const Dataset& dataset,
    const std::vector<Index>& sample_indices,
    HistogramBuilder& hist_builder,
    const std::vector<Float>& all_gradients,
    const std::vector<Float>& all_hessians
) {
    nodes_.clear();
    multiclass_leaf_values_.clear();
    node_to_leaf_idx_.clear();
    n_leaves_ = 0;
    depth_ = 0;

    if (sample_indices.empty()) {
        return;
    }

    // Add root node
    TreeIndex root = add_node();

    // Compute total gradient/hessian (sum across all classes)
    GradientPair root_stats;
    Index n_samples = dataset.n_samples();

    for (Index idx : sample_indices) {
        for (uint32_t c = 0; c < n_classes_; ++c) {
            root_stats.grad += all_gradients[idx * n_classes_ + c];
            root_stats.hess += all_hessians[idx * n_classes_ + c];
        }
        root_stats.count += 1;
    }

    nodes_[root].stats = root_stats;

    // All features for root
    std::vector<FeatureIndex> all_features(dataset.n_features());
    std::iota(all_features.begin(), all_features.end(), static_cast<FeatureIndex>(0));

    // Histogram for building
    Histogram histogram(dataset.n_features(), config_.max_bins);

    // Build recursively
    build_recursive_multiclass(root, dataset, sample_indices, all_features,
                               hist_builder, histogram, 0, all_gradients, all_hessians);
}

void Tree::build_recursive_multiclass(
    TreeIndex node_idx,
    const Dataset& dataset,
    const std::vector<Index>& indices,
    const std::vector<FeatureIndex>& features,
    HistogramBuilder& hist_builder,
    Histogram& histogram,
    uint16_t current_depth,
    const std::vector<Float>& all_gradients,
    const std::vector<Float>& all_hessians
) {
    depth_ = std::max(depth_, current_depth);

    // Read node stats (don't keep reference - it can be invalidated by add_node)
    GradientPair node_stats = nodes_[node_idx].stats;

    // Check stopping conditions
    bool should_stop =
        current_depth >= config_.max_depth ||
        n_leaves_ >= config_.max_leaves ||
        indices.size() < static_cast<size_t>(2 * config_.min_samples_leaf) ||
        node_stats.hess < 2 * config_.min_child_weight;

    if (should_stop) {
        make_leaf_multiclass(node_idx, indices, all_gradients, all_hessians);
        return;
    }

    // Build histogram using summed gradients
    hist_builder.build(dataset, indices, features, histogram);

    // Find best split
    SplitFinder finder(config_);
    SplitInfo best_split = finder.find_best_split(histogram, node_stats, features);

    if (!best_split.is_valid || best_split.gain < config_.min_split_gain) {
        make_leaf_multiclass(node_idx, indices, all_gradients, all_hessians);
        return;
    }

    // Apply split - use direct indexing, not reference
    nodes_[node_idx].split_feature = best_split.feature_idx;
    nodes_[node_idx].split_bin = best_split.bin_threshold;
    nodes_[node_idx].is_leaf = 0;
    nodes_[node_idx].gain = best_split.gain;

    // Partition samples
    std::vector<Index> left_indices, right_indices;
    left_indices.reserve(indices.size() / 2);
    right_indices.reserve(indices.size() / 2);

    const BinIndex* feature_bins = dataset.binned().column(best_split.feature_idx);
    bool default_left = nodes_[node_idx].default_left;

    for (Index idx : indices) {
        BinIndex bin = feature_bins[idx];

        if (bin == 255) {  // NaN bin
            if (default_left) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        } else if (bin <= best_split.bin_threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    // Learn missing value direction
    if (config_.learn_missing_direction &&
        (left_indices.size() != indices.size() && right_indices.size() != indices.size())) {
        nodes_[node_idx].default_left = left_indices.size() >= right_indices.size() ? 1 : 0;
    }

    // Create child nodes - IMPORTANT: add_node() can reallocate nodes_ vector!
    TreeIndex left_child = add_node();
    TreeIndex right_child = add_node();

    // Set child indices after potential reallocation
    nodes_[node_idx].left_child = left_child;
    nodes_[node_idx].right_child = right_child;

    // Compute child stats (sum across classes)
    GradientPair left_stats, right_stats;
    for (Index idx : left_indices) {
        for (uint32_t c = 0; c < n_classes_; ++c) {
            left_stats.grad += all_gradients[idx * n_classes_ + c];
            left_stats.hess += all_hessians[idx * n_classes_ + c];
        }
        left_stats.count += 1;
    }
    for (Index idx : right_indices) {
        for (uint32_t c = 0; c < n_classes_; ++c) {
            right_stats.grad += all_gradients[idx * n_classes_ + c];
            right_stats.hess += all_hessians[idx * n_classes_ + c];
        }
        right_stats.count += 1;
    }

    nodes_[left_child].stats = left_stats;
    nodes_[right_child].stats = right_stats;

    // Recursively build children
    build_recursive_multiclass(left_child, dataset, left_indices, features,
                               hist_builder, histogram, current_depth + 1,
                               all_gradients, all_hessians);
    build_recursive_multiclass(right_child, dataset, right_indices, features,
                               hist_builder, histogram, current_depth + 1,
                               all_gradients, all_hessians);
}

void Tree::make_leaf_multiclass(
    TreeIndex node_idx,
    const std::vector<Index>& indices,
    const std::vector<Float>& all_gradients,
    const std::vector<Float>& all_hessians
) {
    TreeNode& node = nodes_[node_idx];
    node.is_leaf = 1;

    // Compute leaf index
    TreeIndex leaf_idx = n_leaves_;
    n_leaves_++;

    // Resize to accommodate new leaf values
    multiclass_leaf_values_.resize(n_leaves_ * n_classes_, 0.0f);

    // Track mapping from node to leaf
    if (node_to_leaf_idx_.size() <= node_idx) {
        node_to_leaf_idx_.resize(node_idx + 1, 0);
    }
    node_to_leaf_idx_[node_idx] = leaf_idx;

    // Compute K leaf values: w_k = -sum(g_k) / (sum(h_k) + lambda)
    std::vector<Float> grad_sum(n_classes_, 0.0f);
    std::vector<Float> hess_sum(n_classes_, 0.0f);

    for (Index idx : indices) {
        for (uint32_t c = 0; c < n_classes_; ++c) {
            grad_sum[c] += all_gradients[idx * n_classes_ + c];
            hess_sum[c] += all_hessians[idx * n_classes_ + c];
        }
    }

    // Optional smoothing based on sample count
    Float shrink_factor = 1.0f;
    if (config_.leaf_smooth > 0.0f) {
        shrink_factor = static_cast<Float>(indices.size()) /
                       (static_cast<Float>(indices.size()) + config_.leaf_smooth);
    }

    for (uint32_t c = 0; c < n_classes_; ++c) {
        Float raw_value = -grad_sum[c] / (hess_sum[c] + config_.lambda_l2);
        Float value = raw_value * shrink_factor;

        // Apply delta step constraint
        if (config_.max_delta_step > 0) {
            value = std::max(-config_.max_delta_step,
                            std::min(config_.max_delta_step, value));
        }

        multiclass_leaf_values_[leaf_idx * n_classes_ + c] = value;
    }

    // Also store first class value in node.value for backward compatibility
    node.value = multiclass_leaf_values_[leaf_idx * n_classes_];
}

TreeIndex Tree::get_leaf_idx(const Dataset& dataset, Index row) const {
    if (nodes_.empty()) {
        return 0;
    }

    TreeIndex node_idx = 0;

    while (!nodes_[node_idx].is_leaf) {
        const TreeNode& node = nodes_[node_idx];
        BinIndex bin = dataset.binned().get(row, node.split_feature);

        if (bin == 255) {  // NaN
            node_idx = node.default_left ? node.left_child : node.right_child;
        } else {
            const int go_right = (bin > node.split_bin);
            node_idx = go_right * node.right_child + (1 - go_right) * node.left_child;
        }
    }

    return node_idx;
}

void Tree::predict_multiclass(const Dataset& dataset, Index row, Float* output) const {
    if (nodes_.empty() || n_classes_ <= 1) {
        std::memset(output, 0, n_classes_ * sizeof(Float));
        return;
    }

    TreeIndex node_idx = get_leaf_idx(dataset, row);
    TreeIndex leaf_idx = node_to_leaf_idx_[node_idx];

    for (uint32_t c = 0; c < n_classes_; ++c) {
        output[c] = multiclass_leaf_values_[leaf_idx * n_classes_ + c];
    }
}

void Tree::predict_multiclass(const Float* features, FeatureIndex n_features, Float* output) const {
    if (nodes_.empty() || n_classes_ <= 1) {
        std::memset(output, 0, n_classes_ * sizeof(Float));
        return;
    }

    // Traverse tree
    TreeIndex node_idx = 0;

    while (!nodes_[node_idx].is_leaf) {
        const TreeNode& node = nodes_[node_idx];
        Float value = features[node.split_feature];

        if (std::isnan(value)) {
            node_idx = node.default_left ? node.left_child : node.right_child;
        } else {
            node_idx = (value <= static_cast<Float>(node.split_bin))
                      ? node.left_child : node.right_child;
        }
    }

    TreeIndex leaf_idx = node_to_leaf_idx_[node_idx];

    for (uint32_t c = 0; c < n_classes_; ++c) {
        output[c] = multiclass_leaf_values_[leaf_idx * n_classes_ + c];
    }
}

void Tree::predict_batch_multiclass(const Dataset& dataset, Float* output) const {
    Index n_samples = dataset.n_samples();

    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        predict_multiclass(dataset, i, output + i * n_classes_);
    }
}

// ============================================================================
// Tree Ensemble
// ============================================================================

void TreeEnsemble::add_tree(std::unique_ptr<Tree> tree, Float weight) {
    trees_.push_back(std::move(tree));
    tree_weights_.push_back(weight);
    tree_class_indices_.push_back(0);  // Default to class 0 for binary/regression
}

void TreeEnsemble::add_tree_for_class(std::unique_ptr<Tree> tree, Float weight, uint32_t class_idx) {
    trees_.push_back(std::move(tree));
    tree_weights_.push_back(weight);
    tree_class_indices_.push_back(class_idx);
}

Float TreeEnsemble::predict(const Float* features, FeatureIndex n_features) const {
    Float sum = 0.0f;
    
    for (size_t i = 0; i < trees_.size(); ++i) {
        sum += tree_weights_[i] * trees_[i]->predict(features, n_features);
    }
    
    return sum;
}

void TreeEnsemble::predict_batch(
    const Float* features,
    Index n_samples,
    FeatureIndex n_features,
    Float* output
) const {
    std::memset(output, 0, n_samples * sizeof(Float));
    
    for (size_t t = 0; t < trees_.size(); ++t) {
        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            output[i] += tree_weights_[t] * trees_[t]->predict(
                features + i * n_features, n_features
            );
        }
    }
}

Float TreeEnsemble::predict(const Dataset& data, Index row) const {
    Float sum = 0.0f;

    for (size_t i = 0; i < trees_.size(); ++i) {
        sum += tree_weights_[i] * trees_[i]->predict(data, row);
    }

    return sum;
}

void TreeEnsemble::predict_multiclass(const Dataset& data, Index row, Float* output) const {
    std::memset(output, 0, n_classes_ * sizeof(Float));

    // K-trees-per-iteration: each tree belongs to a specific class
    for (size_t i = 0; i < trees_.size(); ++i) {
        uint32_t class_idx = tree_class_indices_[i];
        Float tree_pred = trees_[i]->predict(data, row);  // Single value from binary tree
        output[class_idx] += tree_weights_[i] * tree_pred;
    }
}

void TreeEnsemble::predict_batch_multiclass(const Dataset& data, Float* output) const {
    Index n_samples = data.n_samples();
    std::memset(output, 0, n_samples * n_classes_ * sizeof(Float));

    // K-trees-per-iteration: each tree belongs to a specific class
    for (size_t t = 0; t < trees_.size(); ++t) {
        uint32_t class_idx = tree_class_indices_[t];
        Float weight = tree_weights_[t];

        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            Float tree_pred = trees_[t]->predict(data, i);  // Single value from binary tree
            output[i * n_classes_ + class_idx] += weight * tree_pred;
        }
    }
}

void TreeEnsemble::predict_batch(const Dataset& data, Float* output) const {
    Index n_samples = data.n_samples();
    std::memset(output, 0, n_samples * sizeof(Float));
    
    for (size_t t = 0; t < trees_.size(); ++t) {
        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            output[i] += tree_weights_[t] * trees_[t]->predict(data, i);
        }
    }
}

std::vector<Float> TreeEnsemble::feature_importance() const {
    std::vector<Float> total(256, 0.0f);
    
    for (const auto& tree : trees_) {
        auto imp = tree->feature_importance();
        for (size_t i = 0; i < imp.size(); ++i) {
            total[i] += imp[i];
        }
    }
    
    // Normalize
    Float sum = std::accumulate(total.begin(), total.end(), 0.0f);
    if (sum > 0) {
        for (auto& v : total) {
            v /= sum;
        }
    }
    
    return total;
}

void TreeEnsemble::prepare_for_inference() {
    if (inference_prepared_) return;

    // Calculate total nodes and prepare offsets
    tree_offsets_.clear();
    tree_offsets_.reserve(trees_.size() + 1);

    size_t total_nodes = 0;
    for (const auto& tree : trees_) {
        tree_offsets_.push_back(total_nodes);
        total_nodes += tree->nodes().size();
    }
    tree_offsets_.push_back(total_nodes);

    // Flatten all nodes into contiguous memory
    flat_nodes_.clear();
    flat_nodes_.reserve(total_nodes);

    for (size_t t = 0; t < trees_.size(); ++t) {
        const auto& nodes = trees_[t]->nodes();
        size_t offset = tree_offsets_[t];

        for (const auto& node : nodes) {
            TreeNode flat_node = node;
            // Adjust child indices to global indices
            if (!node.is_leaf) {
                flat_node.left_child = static_cast<TreeIndex>(offset + node.left_child);
                flat_node.right_child = static_cast<TreeIndex>(offset + node.right_child);
            }
            flat_nodes_.push_back(flat_node);
        }
    }

    inference_prepared_ = true;
}

void TreeEnsemble::predict_batch_optimized(const Dataset& data, Float* output, int n_threads) const {
    Index n_samples = data.n_samples();
    size_t n_trees_local = trees_.size();
    FeatureIndex n_features = data.n_features();

    if (n_trees_local == 0) {
        std::memset(output, 0, n_samples * sizeof(Float));
        return;
    }

    // Prepare flat representation for ultra-fast inference
    const_cast<TreeEnsemble*>(this)->prepare_for_inference();

    // Prepare row-major layout for cache-efficient access
    data.binned().prepare_for_prediction();
    const bool use_row_major = data.binned().has_row_major();

    // Initialize output to zero
    std::memset(output, 0, n_samples * sizeof(Float));

    const auto& binned = data.binned();

    // Get pointers to flat data structures
    const TreeNode* flat_nodes = flat_nodes_.data();
    const size_t* offsets = tree_offsets_.data();
    const Float* weights = tree_weights_.data();

    // Determine number of threads
    int num_threads = n_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads <= 0) num_threads = 1;
    }

    if (use_row_major) {
        // ULTRA-FAST PATH: Process multiple samples in batches with unrolled tree loops
        // This improves instruction-level parallelism and cache utilization
        constexpr Index BATCH_SIZE = 8;

        #ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
        #endif
        {
            // Process samples in batches
            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (Index batch_start = 0; batch_start < n_samples; batch_start += BATCH_SIZE) {
                Index batch_end = std::min(batch_start + BATCH_SIZE, n_samples);
                Index batch_size = batch_end - batch_start;

                // Process each sample in the batch
                // Unroll for common batch sizes
                if (batch_size == BATCH_SIZE) {
                    // Full batch - fully unrolled
                    Float sums[BATCH_SIZE] = {0.0f};
                    const BinIndex* row_ptrs[BATCH_SIZE];

                    for (Index i = 0; i < BATCH_SIZE; ++i) {
                        row_ptrs[i] = binned.row(batch_start + i);
                    }

                    // Process all trees
                    for (size_t t = 0; t < n_trees_local; ++t) {
                        TreeIndex base_idx = static_cast<TreeIndex>(offsets[t]);
                        const Float weight = weights[t];

                        // Unroll tree traversal for each sample in batch
                        for (Index i = 0; i < BATCH_SIZE; ++i) {
                            TreeIndex node_idx = base_idx;
                            const BinIndex* row_data = row_ptrs[i];

                            while (!flat_nodes[node_idx].is_leaf) {
                                const TreeNode& node = flat_nodes[node_idx];
                                const BinIndex bin = row_data[node.split_feature];
                                node_idx = (bin != 255 && bin > node.split_bin)
                                         ? node.right_child : node.left_child;
                            }

                            sums[i] += weight * flat_nodes[node_idx].value;
                        }
                    }

                    // Write results
                    for (Index i = 0; i < BATCH_SIZE; ++i) {
                        output[batch_start + i] = sums[i];
                    }
                } else {
                    // Partial batch at end
                    for (Index row = batch_start; row < batch_end; ++row) {
                        const BinIndex* row_data = binned.row(row);
                        Float sum = 0.0f;

                        for (size_t t = 0; t < n_trees_local; ++t) {
                            TreeIndex node_idx = static_cast<TreeIndex>(offsets[t]);
                            const Float weight = weights[t];

                            while (!flat_nodes[node_idx].is_leaf) {
                                const TreeNode& node = flat_nodes[node_idx];
                                const BinIndex bin = row_data[node.split_feature];
                                node_idx = (bin != 255 && bin > node.split_bin)
                                         ? node.right_child : node.left_child;
                            }

                            sum += weight * flat_nodes[node_idx].value;
                        }

                        output[row] = sum;
                    }
                }
            }
        }
    } else {
        // FALLBACK: Column-major access (slower but works without row-major prep)
        std::vector<const BinIndex*> column_ptrs(n_features);
        for (FeatureIndex f = 0; f < n_features; ++f) {
            column_ptrs[f] = binned.column(f);
        }
        const BinIndex* const* cols = column_ptrs.data();

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        #endif
        for (Index row = 0; row < n_samples; ++row) {
            Float sum = 0.0f;

            for (size_t t = 0; t < n_trees_local; ++t) {
                TreeIndex node_idx = static_cast<TreeIndex>(offsets[t]);
                Float weight = weights[t];

                while (!flat_nodes[node_idx].is_leaf) {
                    const TreeNode& node = flat_nodes[node_idx];
                    BinIndex bin = cols[node.split_feature][row];

                    node_idx = (bin != 255 && bin > node.split_bin)
                             ? node.right_child : node.left_child;
                }

                sum += weight * flat_nodes[node_idx].value;
            }

            output[row] = sum;
        }
    }
}

void TreeEnsemble::predict_batch_multiclass_optimized(const Dataset& data, Float* output, int n_threads) const {
    Index n_samples = data.n_samples();
    size_t n_trees_local = trees_.size();
    uint32_t n_classes = n_classes_;
    FeatureIndex n_features = data.n_features();

    if (n_trees_local == 0 || n_classes == 0) {
        std::memset(output, 0, n_samples * n_classes * sizeof(Float));
        return;
    }

    // Prepare flat representation for ultra-fast inference
    const_cast<TreeEnsemble*>(this)->prepare_for_inference();

    // Prepare row-major layout for cache-efficient access
    data.binned().prepare_for_prediction();
    const bool use_row_major = data.binned().has_row_major();

    // Initialize output to zero
    std::memset(output, 0, n_samples * n_classes * sizeof(Float));

    const auto& binned = data.binned();

    // Get pointers to flat data structures
    const TreeNode* flat_nodes = flat_nodes_.data();
    const size_t* offsets = tree_offsets_.data();
    const Float* weights = tree_weights_.data();
    const uint32_t* class_indices = tree_class_indices_.data();

    // Determine number of threads
    int num_threads = n_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads <= 0) num_threads = 1;
    }

    if (use_row_major) {
        // FAST PATH: Row-major layout for sequential memory access
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        #endif
        for (Index row = 0; row < n_samples; ++row) {
            const BinIndex* row_data = binned.row(row);
            Float* row_output = output + row * n_classes;

            for (size_t t = 0; t < n_trees_local; ++t) {
                TreeIndex node_idx = static_cast<TreeIndex>(offsets[t]);
                const Float weight = weights[t];
                const uint32_t class_idx = class_indices[t];

                while (!flat_nodes[node_idx].is_leaf) {
                    const TreeNode& node = flat_nodes[node_idx];
                    const BinIndex bin = row_data[node.split_feature];

                    node_idx = (bin != 255 && bin > node.split_bin)
                             ? node.right_child : node.left_child;
                }

                row_output[class_idx] += weight * flat_nodes[node_idx].value;
            }
        }
    } else {
        // FALLBACK: Column-major access
        std::vector<const BinIndex*> column_ptrs(n_features);
        for (FeatureIndex f = 0; f < n_features; ++f) {
            column_ptrs[f] = binned.column(f);
        }
        const BinIndex* const* cols = column_ptrs.data();

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        #endif
        for (Index row = 0; row < n_samples; ++row) {
            Float* row_output = output + row * n_classes;

            for (size_t t = 0; t < n_trees_local; ++t) {
                TreeIndex node_idx = static_cast<TreeIndex>(offsets[t]);
                Float weight = weights[t];
                uint32_t class_idx = class_indices[t];

                while (!flat_nodes[node_idx].is_leaf) {
                    const TreeNode& node = flat_nodes[node_idx];
                    BinIndex bin = cols[node.split_feature][row];

                    node_idx = (bin != 255 && bin > node.split_bin)
                             ? node.right_child : node.left_child;
                }

                row_output[class_idx] += weight * flat_nodes[node_idx].value;
            }
        }
    }
}

void TreeEnsemble::sparsify(Float target_sparsity) {
    // LP-based ensemble thinning would go here
    // For now, simple weight thresholding
    
    std::vector<std::pair<Float, size_t>> weights_with_idx;
    for (size_t i = 0; i < tree_weights_.size(); ++i) {
        weights_with_idx.emplace_back(std::abs(tree_weights_[i]), i);
    }
    
    std::sort(weights_with_idx.begin(), weights_with_idx.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    size_t keep_count = static_cast<size_t>(trees_.size() * (1.0f - target_sparsity));
    
    std::vector<size_t> keep_indices;
    for (size_t i = 0; i < keep_count && i < weights_with_idx.size(); ++i) {
        keep_indices.push_back(weights_with_idx[i].second);
    }
    
    std::sort(keep_indices.begin(), keep_indices.end());
    
    std::vector<std::unique_ptr<Tree>> new_trees;
    std::vector<Float> new_weights;
    
    for (size_t idx : keep_indices) {
        new_trees.push_back(std::move(trees_[idx]));
        new_weights.push_back(tree_weights_[idx]);
    }
    
    trees_ = std::move(new_trees);
    tree_weights_ = std::move(new_weights);
}

// ============================================================================
// Tree Serialization
// ============================================================================

void Tree::save(std::ostream& out) const {
    // Write tree metadata
    uint16_t depth = depth_;
    uint32_t n_classes = n_classes_;
    uint32_t n_nodes = static_cast<uint32_t>(nodes_.size());
    uint16_t n_leaves = n_leaves_;

    out.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
    out.write(reinterpret_cast<const char*>(&n_classes), sizeof(n_classes));
    out.write(reinterpret_cast<const char*>(&n_nodes), sizeof(n_nodes));
    out.write(reinterpret_cast<const char*>(&n_leaves), sizeof(n_leaves));

    // Write config (essential parts)
    out.write(reinterpret_cast<const char*>(&config_.max_depth), sizeof(config_.max_depth));
    out.write(reinterpret_cast<const char*>(&config_.lambda_l2), sizeof(config_.lambda_l2));

    // Write nodes
    for (const auto& node : nodes_) {
        out.write(reinterpret_cast<const char*>(&node.split_feature), sizeof(node.split_feature));
        out.write(reinterpret_cast<const char*>(&node.split_bin), sizeof(node.split_bin));
        uint8_t flags = (node.is_leaf & 0x1) | ((node.default_left & 0x1) << 1);
        out.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
        out.write(reinterpret_cast<const char*>(&node.left_child), sizeof(node.left_child));
        out.write(reinterpret_cast<const char*>(&node.right_child), sizeof(node.right_child));
        out.write(reinterpret_cast<const char*>(&node.value), sizeof(node.value));
        out.write(reinterpret_cast<const char*>(&node.gain), sizeof(node.gain));
    }

    // Write multiclass leaf values if applicable
    if (n_classes > 1) {
        uint32_t n_multiclass_values = static_cast<uint32_t>(multiclass_leaf_values_.size());
        out.write(reinterpret_cast<const char*>(&n_multiclass_values), sizeof(n_multiclass_values));
        if (n_multiclass_values > 0) {
            out.write(reinterpret_cast<const char*>(multiclass_leaf_values_.data()),
                     n_multiclass_values * sizeof(Float));
        }

        // Write node_to_leaf_idx mapping
        uint32_t n_mapping = static_cast<uint32_t>(node_to_leaf_idx_.size());
        out.write(reinterpret_cast<const char*>(&n_mapping), sizeof(n_mapping));
        if (n_mapping > 0) {
            out.write(reinterpret_cast<const char*>(node_to_leaf_idx_.data()),
                     n_mapping * sizeof(TreeIndex));
        }
    }
}

Tree Tree::load(std::istream& in) {
    Tree tree;

    // Read tree metadata
    uint32_t n_nodes;
    in.read(reinterpret_cast<char*>(&tree.depth_), sizeof(tree.depth_));
    in.read(reinterpret_cast<char*>(&tree.n_classes_), sizeof(tree.n_classes_));
    in.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));
    in.read(reinterpret_cast<char*>(&tree.n_leaves_), sizeof(tree.n_leaves_));

    // Read config (essential parts)
    in.read(reinterpret_cast<char*>(&tree.config_.max_depth), sizeof(tree.config_.max_depth));
    in.read(reinterpret_cast<char*>(&tree.config_.lambda_l2), sizeof(tree.config_.lambda_l2));

    // Read nodes
    tree.nodes_.resize(n_nodes);
    for (uint32_t i = 0; i < n_nodes; ++i) {
        auto& node = tree.nodes_[i];
        in.read(reinterpret_cast<char*>(&node.split_feature), sizeof(node.split_feature));
        in.read(reinterpret_cast<char*>(&node.split_bin), sizeof(node.split_bin));
        uint8_t flags;
        in.read(reinterpret_cast<char*>(&flags), sizeof(flags));
        node.is_leaf = flags & 0x1;
        node.default_left = (flags >> 1) & 0x1;
        in.read(reinterpret_cast<char*>(&node.left_child), sizeof(node.left_child));
        in.read(reinterpret_cast<char*>(&node.right_child), sizeof(node.right_child));
        in.read(reinterpret_cast<char*>(&node.value), sizeof(node.value));
        in.read(reinterpret_cast<char*>(&node.gain), sizeof(node.gain));
    }

    // Read multiclass leaf values if applicable
    if (tree.n_classes_ > 1) {
        uint32_t n_multiclass_values;
        in.read(reinterpret_cast<char*>(&n_multiclass_values), sizeof(n_multiclass_values));
        if (n_multiclass_values > 0) {
            tree.multiclass_leaf_values_.resize(n_multiclass_values);
            in.read(reinterpret_cast<char*>(tree.multiclass_leaf_values_.data()),
                   n_multiclass_values * sizeof(Float));
        }

        // Read node_to_leaf_idx mapping
        uint32_t n_mapping;
        in.read(reinterpret_cast<char*>(&n_mapping), sizeof(n_mapping));
        if (n_mapping > 0) {
            tree.node_to_leaf_idx_.resize(n_mapping);
            in.read(reinterpret_cast<char*>(tree.node_to_leaf_idx_.data()),
                   n_mapping * sizeof(TreeIndex));
        }
    }

    return tree;
}

// ============================================================================
// Ultra-fast prediction using FlatTreeEnsemble (Decision Tables)
// ============================================================================

void TreeEnsemble::predict_batch_flat(const Dataset& data, Float* output, int n_threads) const {
    Index n_samples = data.n_samples();
    size_t n_trees_local = trees_.size();
    FeatureIndex n_features = data.n_features();

    if (n_trees_local == 0) {
        std::memset(output, 0, n_samples * sizeof(Float));
        return;
    }

    // Initialize output to zero
    std::memset(output, 0, n_samples * sizeof(Float));

    const auto& binned = data.binned();

    // Determine number of threads
    int num_threads = n_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads <= 0) num_threads = 1;
    }

    // SAMPLE-FIRST: process all trees for each sample
    // Use direct column access without local caching (simpler, avoids allocation)

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    #endif
    for (Index row = 0; row < n_samples; ++row) {
        Float sum = 0.0f;

        for (size_t t = 0; t < n_trees_local; ++t) {
            const auto& nodes = trees_[t]->nodes();
            if (nodes.empty()) continue;

            Float weight = tree_weights_[t];
            TreeIndex node_idx = 0;

            while (!nodes[node_idx].is_leaf) {
                const TreeNode& node = nodes[node_idx];
                BinIndex bin = binned.get(row, node.split_feature);
                node_idx = (bin > node.split_bin) ? node.right_child : node.left_child;
            }

            sum += weight * nodes[node_idx].value;
        }

        output[row] = sum;
    }
}

// ============================================================================
// ULTRA-FAST SIMD INFERENCE (CatBoost-style optimizations)
// Key insights from research:
// 1. Use column-major data for consecutive bin access
// 2. Process 8 samples simultaneously with AVX2/NEON
// 3. Batch 4 trees for instruction-level parallelism
// 4. Prefetch next tree data
// ============================================================================

void TreeEnsemble::predict_batch_simd(const Dataset& data, Float* output, int n_threads) const {
    Index n_samples = data.n_samples();
    size_t n_trees_local = trees_.size();

    if (n_trees_local == 0) {
        std::memset(output, 0, n_samples * sizeof(Float));
        return;
    }

    // Prepare flat representation
    const_cast<TreeEnsemble*>(this)->prepare_for_inference();

    // Prepare column-major layout for SIMD-friendly access
    data.binned().prepare_column_major();

    // Determine number of threads
    int num_threads = n_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads <= 0) num_threads = 1;
    }

    const auto& binned = data.binned();
    const TreeNode* flat_nodes = flat_nodes_.data();
    const size_t* offsets = tree_offsets_.data();
    const Float* weights = tree_weights_.data();

#if defined(HAS_NEON)
    // ARM NEON: Process 8 samples at once (2x float32x4)
    if (binned.has_column_major()) {
        const Index n_batch = (n_samples / 8) * 8;
        const size_t TREE_BATCH = 4;  // Process 4 trees for ILP

        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (Index base = 0; base < n_batch; base += 8) {
            float32x4_t sums0 = vdupq_n_f32(0.0f);
            float32x4_t sums1 = vdupq_n_f32(0.0f);

            // Process trees in batches of 4
            size_t t = 0;
            for (; t + TREE_BATCH <= n_trees_local; t += TREE_BATCH) {
                // Prefetch next batch
                if (t + TREE_BATCH < n_trees_local) {
                    __builtin_prefetch(flat_nodes + offsets[t + TREE_BATCH], 0, 3);
                }

                // Process 4 trees
                for (size_t ti = 0; ti < TREE_BATCH; ++ti) {
                    size_t tree_idx = t + ti;
                    TreeIndex node_base = static_cast<TreeIndex>(offsets[tree_idx]);
                    Float weight = weights[tree_idx];

                    // Traverse tree for 8 samples
                    alignas(16) uint32_t node_indices[8] = {0,0,0,0,0,0,0,0};

                    // Max depth traversal (unrolled for common depths)
                    for (int depth = 0; depth < 16; ++depth) {
                        // Check if all samples reached leaves
                        bool all_leaves = true;
                        alignas(16) float leaf_vals[8];

                        for (int s = 0; s < 8; ++s) {
                            TreeIndex node_idx = node_base + node_indices[s];
                            const TreeNode& node = flat_nodes[node_idx];

                            if (node.is_leaf) {
                                leaf_vals[s] = node.value;
                            } else {
                                all_leaves = false;
                                BinIndex bin = binned.get_column_major(base + s, node.split_feature);
                                bool go_right = (bin != 255 && bin > node.split_bin);
                                node_indices[s] = go_right ?
                                    (node.right_child - node_base) :
                                    (node.left_child - node_base);
                            }
                        }

                        if (all_leaves) {
                            float32x4_t lv0 = vld1q_f32(leaf_vals);
                            float32x4_t lv1 = vld1q_f32(leaf_vals + 4);
                            sums0 = vmlaq_n_f32(sums0, lv0, weight);
                            sums1 = vmlaq_n_f32(sums1, lv1, weight);
                            break;
                        }
                    }
                }
            }

            // Handle remaining trees
            for (; t < n_trees_local; ++t) {
                TreeIndex node_base = static_cast<TreeIndex>(offsets[t]);
                Float weight = weights[t];

                alignas(16) float leaf_vals[8];
                for (int s = 0; s < 8; ++s) {
                    TreeIndex node_idx = node_base;
                    while (!flat_nodes[node_idx].is_leaf) {
                        const TreeNode& node = flat_nodes[node_idx];
                        BinIndex bin = binned.get_column_major(base + s, node.split_feature);
                        node_idx = (bin != 255 && bin > node.split_bin) ?
                                   node.right_child : node.left_child;
                    }
                    leaf_vals[s] = flat_nodes[node_idx].value;
                }

                float32x4_t lv0 = vld1q_f32(leaf_vals);
                float32x4_t lv1 = vld1q_f32(leaf_vals + 4);
                sums0 = vmlaq_n_f32(sums0, lv0, weight);
                sums1 = vmlaq_n_f32(sums1, lv1, weight);
            }

            vst1q_f32(output + base, sums0);
            vst1q_f32(output + base + 4, sums1);
        }

        // Handle remaining samples
        for (Index i = n_batch; i < n_samples; ++i) {
            Float sum = 0.0f;
            for (size_t t = 0; t < n_trees_local; ++t) {
                TreeIndex node_idx = static_cast<TreeIndex>(offsets[t]);
                while (!flat_nodes[node_idx].is_leaf) {
                    const TreeNode& node = flat_nodes[node_idx];
                    BinIndex bin = binned.get_column_major(i, node.split_feature);
                    node_idx = (bin != 255 && bin > node.split_bin) ?
                               node.right_child : node.left_child;
                }
                sum += weights[t] * flat_nodes[node_idx].value;
            }
            output[i] = sum;
        }
        return;
    }
#else
    // x86/Generic: TREE-FIRST iteration for better cache locality
    // Process all samples for each tree, accumulating predictions
    // This keeps tree nodes in cache while iterating over samples

    // Initialize output
    std::memset(output, 0, n_samples * sizeof(Float));

    // Use row-major for sample-contiguous access
    data.binned().prepare_for_prediction();
    const bool use_row_major = data.binned().has_row_major();

    if (use_row_major) {
        // TREE-FIRST with row-major data: best cache pattern
        // Tree nodes stay in L1/L2 cache while we process all samples

        for (size_t t = 0; t < n_trees_local; ++t) {
            TreeIndex tree_base = static_cast<TreeIndex>(offsets[t]);
            const Float weight = weights[t];

            // Prefetch next tree
            if (t + 1 < n_trees_local) {
                _mm_prefetch(reinterpret_cast<const char*>(flat_nodes + offsets[t + 1]), _MM_HINT_T0);
            }

            #pragma omp parallel for schedule(static) num_threads(num_threads)
            for (Index i = 0; i < n_samples; ++i) {
                const BinIndex* row = binned.row(i);
                TreeIndex node_idx = tree_base;

                // Unrolled tree traversal (max depth typically 6-10)
                while (!flat_nodes[node_idx].is_leaf) {
                    const TreeNode& node = flat_nodes[node_idx];
                    BinIndex bin = row[node.split_feature];
                    // Branchless: bin > threshold goes right, NaN (255) goes left
                    node_idx = (bin != 255 && bin > node.split_bin) ?
                               node.right_child : node.left_child;
                }

                output[i] += weight * flat_nodes[node_idx].value;
            }
        }
    } else {
        // Fallback: column-major, sample-first
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (Index i = 0; i < n_samples; ++i) {
            Float sum = 0.0f;
            for (size_t t = 0; t < n_trees_local; ++t) {
                TreeIndex node_idx = static_cast<TreeIndex>(offsets[t]);
                while (!flat_nodes[node_idx].is_leaf) {
                    const TreeNode& node = flat_nodes[node_idx];
                    BinIndex bin = binned.get(i, node.split_feature);
                    node_idx = (bin != 255 && bin > node.split_bin) ?
                               node.right_child : node.left_child;
                }
                sum += weights[t] * flat_nodes[node_idx].value;
            }
            output[i] = sum;
        }
    }
#endif
}

} // namespace turbocat
