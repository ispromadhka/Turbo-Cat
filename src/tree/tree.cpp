/**
 * TurboCat Tree Implementation
 */

#include "turbocat/tree.hpp"
#include <algorithm>
#include <queue>
#include <stack>
#include <numeric>
#include <cstring>

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

Tree::Tree(const TreeConfig& config) : config_(config) {
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

    TreeNode& node = nodes_[node_idx];
    
    // Check stopping conditions
    bool should_stop = 
        current_depth >= config_.max_depth ||
        n_leaves_ >= config_.max_leaves ||
        indices.size() < static_cast<size_t>(2 * config_.min_samples_leaf) ||
        node.stats.hess < 2 * config_.min_child_weight;
    
    if (should_stop) {
        make_leaf(node_idx, node.stats);
        return;
    }
    
    // Build histogram
    hist_builder.build(dataset, indices, features, histogram);
    
    // Find best split
    SplitFinder finder(config_);
    SplitInfo best_split = finder.find_best_split(histogram, node.stats, features);
    
    if (!best_split.is_valid || best_split.gain < config_.min_split_gain) {
        make_leaf(node_idx, node.stats);
        return;
    }
    
    // Apply split
    node.split_feature = best_split.feature_idx;
    node.split_bin = best_split.bin_threshold;
    node.is_leaf = 0;
    node.gain = best_split.gain;
    
    // Partition samples
    std::vector<Index> left_indices, right_indices;
    left_indices.reserve(indices.size() / 2);
    right_indices.reserve(indices.size() / 2);
    
    const BinIndex* feature_bins = dataset.binned().column(best_split.feature_idx);
    
    for (Index idx : indices) {
        BinIndex bin = feature_bins[idx];
        
        // Handle missing values
        if (bin == 255) {  // NaN bin
            if (node.default_left) {
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
        node.default_left = left_indices.size() >= right_indices.size() ? 1 : 0;
    }

    // Create child nodes
    // IMPORTANT: add_node() can reallocate the nodes_ vector, invalidating 'node' reference
    // We must add both children first, then re-acquire the reference
    TreeIndex left_child = add_node();
    TreeIndex right_child = add_node();

    // Re-acquire reference after potential reallocation
    nodes_[node_idx].left_child = left_child;
    nodes_[node_idx].right_child = right_child;

    nodes_[left_child].stats = best_split.left_stats;
    nodes_[right_child].stats = best_split.right_stats;

    // Recursively build children
    build_recursive(left_child, dataset, left_indices, features,
                   hist_builder, histogram, current_depth + 1);
    build_recursive(right_child, dataset, right_indices, features,
                   hist_builder, histogram, current_depth + 1);
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
// Tree Ensemble
// ============================================================================

void TreeEnsemble::add_tree(std::unique_ptr<Tree> tree, Float weight) {
    trees_.push_back(std::move(tree));
    tree_weights_.push_back(weight);
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
