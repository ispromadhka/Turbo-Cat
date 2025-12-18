/**
 * TurboCat Tree Implementation
 */

#include "turbocat/tree.hpp"
#include <algorithm>
#include <queue>
#include <stack>
#include <numeric>
#include <cstring>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TURBOCAT_AVX2
#include <immintrin.h>
#endif

namespace turbocat {

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
    
    // Histogram for building
    Histogram histogram(dataset.n_features(), config_.max_bins);
    
    // Build recursively
    build_recursive(root, dataset, sample_indices, all_features, 
                   hist_builder, histogram, 0);
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

    // Read node stats (don't keep reference - can be invalidated)
    GradientPair node_stats = nodes_[node_idx].stats;

    // Check stopping conditions
    bool should_stop =
        current_depth >= config_.max_depth ||
        n_leaves_ >= config_.max_leaves ||
        indices.size() < static_cast<size_t>(2 * config_.min_samples_leaf) ||
        node_stats.hess < 2 * config_.min_child_weight;

    if (should_stop) {
        make_leaf(node_idx, node_stats);
        return;
    }

    // Build histogram for this node
    hist_builder.build(dataset, indices, features, histogram);

    // Find best split (reuse finder to avoid allocation)
    SplitFinder finder(config_);
    SplitInfo best_split = finder.find_best_split(histogram, node_stats, features);

    if (!best_split.is_valid || best_split.gain < config_.min_split_gain) {
        make_leaf(node_idx, node_stats);
        return;
    }

    // Apply split - use direct indexing
    nodes_[node_idx].split_feature = best_split.feature_idx;
    nodes_[node_idx].split_bin = best_split.bin_threshold;
    nodes_[node_idx].is_leaf = 0;
    nodes_[node_idx].gain = best_split.gain;

    // Partition samples - optimized single-pass
    std::vector<Index> left_indices, right_indices;
    left_indices.reserve(indices.size() / 2);
    right_indices.reserve(indices.size() / 2);

    const BinIndex* feature_bins = dataset.binned().column(best_split.feature_idx);
    const BinIndex threshold = best_split.bin_threshold;
    uint8_t default_left = nodes_[node_idx].default_left;

    for (Index idx : indices) {
        BinIndex bin = feature_bins[idx];

        if (bin == 255) {  // NaN bin
            if (default_left) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        } else if (bin <= threshold) {
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

    // Create child nodes
    TreeIndex left_child = add_node();
    TreeIndex right_child = add_node();

    nodes_[node_idx].left_child = left_child;
    nodes_[node_idx].right_child = right_child;
    nodes_[left_child].stats = best_split.left_stats;
    nodes_[right_child].stats = best_split.right_stats;

    // Simple recursive build without extra allocations
    // The histogram subtraction trick saves ~50% of histogram build time,
    // but the allocation overhead can negate the benefit.
    // We use a simpler approach: just build both children.
    build_recursive(left_child, dataset, left_indices, features,
                   hist_builder, histogram, current_depth + 1);
    build_recursive(right_child, dataset, right_indices, features,
                   hist_builder, histogram, current_depth + 1);
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
    node.value = -stats.grad / (stats.hess + config_.lambda_l2);
    
    // Apply delta step constraint
    if (config_.max_delta_step > 0) {
        node.value = std::max(-config_.max_delta_step, 
                             std::min(config_.max_delta_step, node.value));
    }
    
    n_leaves_++;
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
            nodes_[i].value = -leaf_stats[i].grad / (leaf_stats[i].hess + config_.lambda_l2);
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

    for (uint32_t c = 0; c < n_classes_; ++c) {
        Float value = -grad_sum[c] / (hess_sum[c] + config_.lambda_l2);

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

    // Debug: Check ensemble state
    static bool debug_printed = false;
    if (!debug_printed && n_trees_local > 0) {
        size_t empty_trees = 0;
        Float total_leaf_sum = 0.0f;
        int leaf_count = 0;
        for (size_t t = 0; t < std::min(n_trees_local, size_t(3)); ++t) {
            const auto& nodes = trees_[t]->nodes();
            if (nodes.empty()) {
                empty_trees++;
            } else {
                for (const auto& node : nodes) {
                    if (node.is_leaf) {
                        total_leaf_sum += node.value;
                        leaf_count++;
                    }
                }
            }
        }
        std::printf("[DEBUG] predict_batch_optimized: n_trees=%zu, empty_trees=%zu, leaf_values_sum=%.6f, leaf_count=%d\n",
                   n_trees_local, empty_trees, total_leaf_sum, leaf_count);
        std::fflush(stdout);
        debug_printed = true;
    }

    if (n_trees_local == 0) {
        std::memset(output, 0, n_samples * sizeof(Float));
        return;
    }

    // Initialize output to zero
    std::memset(output, 0, n_samples * sizeof(Float));

    // Optimized approach: Process all trees for each sample
    // This is more cache-friendly for the sample's feature data
    // Also use std::thread for parallelism when OpenMP is unavailable

    const auto process_range = [&](Index start, Index end) {
        for (Index row = start; row < end; ++row) {
            Float sum = 0.0f;

            // Process all trees for this sample
            for (size_t t = 0; t < n_trees_local; ++t) {
                const auto& nodes = trees_[t]->nodes();
                if (nodes.empty()) continue;

                Float weight = tree_weights_[t];
                TreeIndex node_idx = 0;

                // Traverse tree - optimized tight loop
                while (!nodes[node_idx].is_leaf) {
                    const TreeNode& node = nodes[node_idx];
                    BinIndex bin = data.binned().get(row, node.split_feature);

                    // Branchless for non-NaN case (most common)
                    if (__builtin_expect(bin != 255, 1)) {
                        node_idx = (bin > node.split_bin) ? node.right_child : node.left_child;
                    } else {
                        node_idx = node.default_left ? node.left_child : node.right_child;
                    }
                }

                sum += weight * nodes[node_idx].value;
            }

            output[row] = sum;
        }
    };

    // Determine number of threads to use
    int num_threads = n_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads <= 0) num_threads = 1;
    }

    // For small datasets or when explicitly single-threaded
    if (num_threads == 1 || n_samples < 1000) {
        process_range(0, n_samples);
        return;
    }

    // Use OpenMP if available
    #ifdef _OPENMP
    int old_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (Index row = 0; row < n_samples; ++row) {
        Float sum = 0.0f;

        for (size_t t = 0; t < n_trees_local; ++t) {
            const auto& nodes = trees_[t]->nodes();
            if (nodes.empty()) continue;

            Float weight = tree_weights_[t];
            TreeIndex node_idx = 0;

            while (!nodes[node_idx].is_leaf) {
                const TreeNode& node = nodes[node_idx];
                BinIndex bin = data.binned().get(row, node.split_feature);

                if (__builtin_expect(bin != 255, 1)) {
                    node_idx = (bin > node.split_bin) ? node.right_child : node.left_child;
                } else {
                    node_idx = node.default_left ? node.left_child : node.right_child;
                }
            }

            sum += weight * nodes[node_idx].value;
        }

        output[row] = sum;
    }

    omp_set_num_threads(old_threads);
    #else
    // Use std::thread for parallelism
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    Index chunk_size = (n_samples + num_threads - 1) / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        Index start = i * chunk_size;
        Index end = std::min(start + chunk_size, n_samples);
        if (start < end) {
            threads.emplace_back(process_range, start, end);
        }
    }

    for (auto& t : threads) {
        t.join();
    }
    #endif
}

void TreeEnsemble::predict_batch_multiclass_optimized(const Dataset& data, Float* output, int n_threads) const {
    Index n_samples = data.n_samples();
    size_t n_trees_local = trees_.size();
    uint32_t n_classes = n_classes_;

    if (n_trees_local == 0 || n_classes == 0) {
        std::memset(output, 0, n_samples * n_classes * sizeof(Float));
        return;
    }

    // Initialize output to zero
    std::memset(output, 0, n_samples * n_classes * sizeof(Float));

    // Process function for a range of samples
    const auto process_range = [&](Index start, Index end) {
        for (Index row = start; row < end; ++row) {
            Float* row_output = output + row * n_classes;

            for (size_t t = 0; t < n_trees_local; ++t) {
                const auto& nodes = trees_[t]->nodes();
                if (nodes.empty()) continue;

                Float weight = tree_weights_[t];
                uint32_t class_idx = tree_class_indices_[t];
                TreeIndex node_idx = 0;

                while (!nodes[node_idx].is_leaf) {
                    const TreeNode& node = nodes[node_idx];
                    BinIndex bin = data.binned().get(row, node.split_feature);

                    if (__builtin_expect(bin != 255, 1)) {
                        node_idx = (bin > node.split_bin) ? node.right_child : node.left_child;
                    } else {
                        node_idx = node.default_left ? node.left_child : node.right_child;
                    }
                }

                row_output[class_idx] += weight * nodes[node_idx].value;
            }
        }
    };

    // Determine number of threads
    int num_threads = n_threads;
    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads <= 0) num_threads = 1;
    }

    if (num_threads == 1 || n_samples < 1000) {
        process_range(0, n_samples);
        return;
    }

    #ifdef _OPENMP
    int old_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (Index row = 0; row < n_samples; ++row) {
        Float* row_output = output + row * n_classes;

        for (size_t t = 0; t < n_trees_local; ++t) {
            const auto& nodes = trees_[t]->nodes();
            if (nodes.empty()) continue;

            Float weight = tree_weights_[t];
            uint32_t class_idx = tree_class_indices_[t];
            TreeIndex node_idx = 0;

            while (!nodes[node_idx].is_leaf) {
                const TreeNode& node = nodes[node_idx];
                BinIndex bin = data.binned().get(row, node.split_feature);

                if (__builtin_expect(bin != 255, 1)) {
                    node_idx = (bin > node.split_bin) ? node.right_child : node.left_child;
                } else {
                    node_idx = node.default_left ? node.left_child : node.right_child;
                }
            }

            row_output[class_idx] += weight * nodes[node_idx].value;
        }
    }

    omp_set_num_threads(old_threads);
    #else
    // Use std::thread for parallelism
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    Index chunk_size = (n_samples + num_threads - 1) / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        Index start = i * chunk_size;
        Index end = std::min(start + chunk_size, n_samples);
        if (start < end) {
            threads.emplace_back(process_range, start, end);
        }
    }

    for (auto& t : threads) {
        t.join();
    }
    #endif
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

} // namespace turbocat
