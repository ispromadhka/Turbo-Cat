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
    // DEBUG
    std::printf("[TREE::BUILD] n_samples=%zu, n_features=%d\n", 
                sample_indices.size(), dataset.n_features());
    
    nodes_.clear();
    n_leaves_ = 0;
    depth_ = 0;
    
    if (sample_indices.empty()) {
        std::printf("[TREE::BUILD] ERROR: sample_indices is EMPTY!\n");
        return;
    }
    
    // Add root node
    TreeIndex root = add_node();
    
    // Compute total gradient/hessian
    GradientPair root_stats;
    for (Index idx : sample_indices) {
        root_stats.grad += dataset.gradients()[idx];
        root_stats.hess += dataset.hessians()[idx];
        root_stats.count += 1;
    }
    
    // DEBUG
    std::printf("[TREE::BUILD] root_stats: G=%.4f, H=%.4f, count=%u\n",
                root_stats.grad, root_stats.hess, root_stats.count);
    
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
    
    // DEBUG OUTPUT
    std::printf("[DEBUG] depth=%d, n_samples=%zu, G=%.4f, H=%.4f, min_samples_leaf=%u, min_child_weight=%.2f\n", 
                current_depth, indices.size(), node.stats.grad, node.stats.hess,
                config_.min_samples_leaf, config_.min_child_weight);
    
    // Check stopping conditions
    bool should_stop = 
        current_depth >= config_.max_depth ||
        n_leaves_ >= config_.max_leaves ||
        indices.size() < static_cast<size_t>(2 * config_.min_samples_leaf) ||
        node.stats.hess < 2 * config_.min_child_weight;
    
    if (should_stop) {
        std::printf("[DEBUG] STOP: depth_limit=%d, leaves_limit=%d, samples_limit=%d, hess_limit=%d\n",
                    current_depth >= config_.max_depth,
                    n_leaves_ >= config_.max_leaves,
                    indices.size() < static_cast<size_t>(2 * config_.min_samples_leaf),
                    node.stats.hess < 2 * config_.min_child_weight);
        make_leaf(node_idx, node.stats);
        return;
    }
    
    // Build histogram
    hist_builder.build(dataset, indices, features, histogram);
    
    // Find best split
    SplitFinder finder(config_);
    SplitInfo best_split = finder.find_best_split(histogram, node.stats, features);
    
    // DEBUG: Print split info
    std::printf("[DEBUG] best_split: valid=%d, gain=%.6f, feature=%d, bin=%d\n",
                best_split.is_valid, best_split.gain, best_split.feature_idx, best_split.bin_threshold);
    
    if (!best_split.is_valid || best_split.gain < config_.min_split_gain) {
        std::printf("[DEBUG] LEAF: split_invalid=%d, gain_too_low=%d (min=%.6f)\n",
                    !best_split.is_valid, best_split.gain < config_.min_split_gain, config_.min_split_gain);
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
        // Try both directions and pick the one with better gain
        // For now, use heuristic: go to the larger child
        node.default_left = left_indices.size() >= right_indices.size() ? 1 : 0;
    }
    
    // Create child nodes
    node.left_child = add_node();
    node.right_child = add_node();
    
    nodes_[node.left_child].stats = best_split.left_stats;
    nodes_[node.right_child].stats = best_split.right_stats;
    
    // Recursively build children
    build_recursive(node.left_child, dataset, left_indices, features,
                   hist_builder, histogram, current_depth + 1);
    build_recursive(node.right_child, dataset, right_indices, features,
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
    
    // DEBUG
    std::printf("[MAKE_LEAF] node=%d, G=%.4f, H=%.4f, lambda=%.4f, value=%.4f\n",
                node_idx, stats.grad, stats.hess, config_.lambda_l2, node.value);
    
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
            // We need bin edges to compare properly
            // For now, assume raw value can be compared to bin threshold
            // This is a simplification; proper implementation needs bin edges
            node_idx = (value <= static_cast<Float>(node.split_bin)) 
                      ? node.left_child : node.right_child;
        }
    }
    
    return nodes_[node_idx].value;
}

Float Tree::predict(const Dataset& dataset, Index row) const {
    if (nodes_.empty()) {
        std::printf("[PREDICT] ERROR: tree is empty!\n");
        return 0.0f;
    }
    
    TreeIndex node_idx = 0;
    int steps = 0;
    
    while (!nodes_[node_idx].is_leaf) {
        const TreeNode& node = nodes_[node_idx];
        BinIndex bin = dataset.binned().get(row, node.split_feature);
        
        // DEBUG all predictions
        std::printf("[PREDICT] row=%d, node=%d, bin=%d, split_bin=%d\n",
                    row, node_idx, bin, node.split_bin);
        
        if (bin == 255) {  // NaN
            node_idx = node.default_left ? node.left_child : node.right_child;
        } else {
            node_idx = (bin <= node.split_bin) ? node.left_child : node.right_child;
        }
        steps++;
        if (steps > 100) {
            std::printf("[PREDICT] ERROR: infinite loop!\n");
            return 0.0f;
        }
    }
    
    // DEBUG
    std::printf("[PREDICT] row=%d -> leaf=%d, value=%.4f\n",
                row, node_idx, nodes_[node_idx].value);
    
    return nodes_[node_idx].value;
}

void Tree::predict_batch(
    const Float* features,
    Index n_samples,
    FeatureIndex n_features,
    Float* output
) const {
    #pragma omp parallel for simd
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

// Disabled - GradTree not compiled
// void TreeEnsemble::add_gradtree(std::unique_ptr<GradTree> tree, Float weight) {
//     gradtrees_.push_back(std::move(tree));
//     gradtree_weights_.push_back(weight);
// }

Float TreeEnsemble::predict(const Float* features, FeatureIndex n_features) const {
    Float sum = 0.0f;
    
    for (size_t i = 0; i < trees_.size(); ++i) {
        sum += tree_weights_[i] * trees_[i]->predict(features, n_features);
    }
    
    // GradTree disabled
    // for (size_t i = 0; i < gradtrees_.size(); ++i) {
    //     sum += gradtree_weights_[i] * gradtrees_[i]->predict_hard(features);
    // }
    
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
    
    // GradTree disabled
    // for (size_t t = 0; t < gradtrees_.size(); ++t) {
    //     #pragma omp parallel for
    //     for (Index i = 0; i < n_samples; ++i) {
    //         output[i] += gradtree_weights_[t] * gradtrees_[t]->predict_hard(
    //             features + i * n_features
    //         );
    //     }
    // }
}

Float TreeEnsemble::predict(const Dataset& data, Index row) const {
    // DEBUG
    if (row < 3) fflush(stdout); std::printf("[ENSEMBLE::PREDICT] row=%d, trees_.size=%zu\n", static_cast<int>(row), trees_.size());
    
    Float sum = 0.0f;
    
    for (size_t i = 0; i < trees_.size(); ++i) {
        sum += tree_weights_[i] * trees_[i]->predict(data, row);
    }
    
    // GradTree disabled
    // for (size_t i = 0; i < gradtrees_.size(); ++i) {
    //     sum += gradtree_weights_[i] * gradtrees_[i]->predict_hard(
    //         data.raw_data() + row * data.n_features()
    //     );
    // }
    
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
    
    // GradTree disabled
    // for (size_t t = 0; t < gradtrees_.size(); ++t) {
    //     #pragma omp parallel for
    //     for (Index i = 0; i < n_samples; ++i) {
    //         output[i] += gradtree_weights_[t] * gradtrees_[t]->predict_hard(
    //             data.raw_data() + i * data.n_features()
    //         );
    //     }
    // }
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
    
    Float threshold = target_sparsity;
    
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
