/**
 * TurboCat Leaf-Wise Tree Implementation
 *
 * Leaf-wise growth strategy:
 * 1. Start with root containing all samples
 * 2. Build histogram, find best split
 * 3. Add to priority queue sorted by gain
 * 4. Pop leaf with max gain, split it
 * 5. For children: build histogram for smaller, subtract for larger
 * 6. Repeat until max_leaves or no good splits
 */

#include "turbocat/leaf_wise_tree.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace turbocat {

// ============================================================================
// Histogram Cache
// ============================================================================

HistogramCache::HistogramCache(FeatureIndex n_features, BinIndex max_bins, size_t max_cache_size)
    : n_features_(n_features), max_bins_(max_bins), max_cache_size_(max_cache_size) {
    pool_.reserve(max_cache_size);
    pool_node_ids_.reserve(max_cache_size);
    pool_in_use_.reserve(max_cache_size);
}

Histogram* HistogramCache::get(TreeIndex node_idx) {
    // Check if already in cache
    for (size_t i = 0; i < pool_node_ids_.size(); ++i) {
        if (pool_node_ids_[i] == node_idx && pool_in_use_[i]) {
            return pool_[i].get();
        }
    }

    // Find free slot or create new
    for (size_t i = 0; i < pool_.size(); ++i) {
        if (!pool_in_use_[i]) {
            pool_in_use_[i] = true;
            pool_node_ids_[i] = node_idx;
            pool_[i]->clear();
            return pool_[i].get();
        }
    }

    // Create new histogram if under limit
    if (pool_.size() < max_cache_size_) {
        pool_.push_back(std::make_unique<Histogram>(n_features_, max_bins_));
        pool_node_ids_.push_back(node_idx);
        pool_in_use_.push_back(true);
        return pool_.back().get();
    }

    // Evict oldest (first unused)
    pool_[0]->clear();
    pool_node_ids_[0] = node_idx;
    pool_in_use_[0] = true;
    return pool_[0].get();
}

void HistogramCache::release(TreeIndex node_idx) {
    for (size_t i = 0; i < pool_node_ids_.size(); ++i) {
        if (pool_node_ids_[i] == node_idx) {
            pool_in_use_[i] = false;
            return;
        }
    }
}

void HistogramCache::clear() {
    for (size_t i = 0; i < pool_in_use_.size(); ++i) {
        pool_in_use_[i] = false;
    }
}

// ============================================================================
// Leaf-Wise Tree
// ============================================================================

LeafWiseTree::LeafWiseTree(const TreeConfig& config) : config_(config) {
    // Reserve space for nodes (max 2 * max_leaves - 1)
    nodes_.reserve(2 * config.max_leaves);
}

void LeafWiseTree::build(
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

    // Initialize histogram cache
    hist_cache_ = std::make_unique<HistogramCache>(
        dataset.n_features(), config_.max_bins, config_.max_leaves * 2
    );

    // All features
    std::vector<FeatureIndex> all_features(dataset.n_features());
    std::iota(all_features.begin(), all_features.end(), static_cast<FeatureIndex>(0));

    // Create root node
    TreeIndex root_idx = add_node();

    // Compute root stats
    GradientPair root_stats;
    const Float* grads = dataset.gradients();
    const Float* hess = dataset.hessians();

    for (Index idx : sample_indices) {
        root_stats.grad += grads[idx];
        root_stats.hess += hess[idx];
        root_stats.count += 1;
    }
    nodes_[root_idx].stats = root_stats;

    // Build histogram for root
    Histogram* root_hist = hist_cache_->get(root_idx);
    hist_builder.build(dataset, sample_indices, all_features, *root_hist);

    // Find best split for root
    SplitInfo root_split = find_best_split(*root_hist, root_stats, all_features);

    // Initialize priority queue with root
    // Clear queue first (in case of reuse)
    while (!leaf_queue_.empty()) leaf_queue_.pop();

    if (root_split.is_valid && root_split.gain >= config_.min_split_gain) {
        LeafInfo root_info;
        root_info.node_idx = root_idx;
        root_info.sample_indices = sample_indices;
        root_info.stats = root_stats;
        root_info.depth = 0;
        root_info.best_split = root_split;
        root_info.has_histogram = true;
        leaf_queue_.push(std::move(root_info));
    } else {
        // Root is a leaf
        make_leaf(root_idx, root_stats);
        return;
    }

    // Main loop: split leaves until max_leaves or no good splits
    while (!leaf_queue_.empty() && n_leaves_ < config_.max_leaves - 1) {
        // Pop leaf with highest gain
        LeafInfo current = leaf_queue_.top();
        leaf_queue_.pop();

        // Check if still valid to split
        if (!current.best_split.is_valid ||
            current.best_split.gain < config_.min_split_gain ||
            current.depth >= config_.max_depth) {
            make_leaf(current.node_idx, current.stats);
            continue;
        }

        // Apply split
        nodes_[current.node_idx].split_feature = current.best_split.feature_idx;
        nodes_[current.node_idx].split_bin = current.best_split.bin_threshold;
        nodes_[current.node_idx].is_leaf = 0;
        nodes_[current.node_idx].gain = current.best_split.gain;

        // Partition samples
        std::vector<Index> left_indices, right_indices;
        partition_samples(dataset, current.sample_indices, current.best_split,
                         left_indices, right_indices);

        // Create child nodes
        TreeIndex left_child = add_node();
        TreeIndex right_child = add_node();

        nodes_[current.node_idx].left_child = left_child;
        nodes_[current.node_idx].right_child = right_child;
        nodes_[left_child].stats = current.best_split.left_stats;
        nodes_[right_child].stats = current.best_split.right_stats;

        depth_ = std::max(depth_, static_cast<uint16_t>(current.depth + 1));

        // Get parent histogram for subtraction trick
        Histogram* parent_hist = hist_cache_->get(current.node_idx);

        // Determine smaller child
        bool left_is_smaller = left_indices.size() <= right_indices.size();

        // Process smaller child: build histogram
        // Process larger child: use subtraction
        auto process_child = [&](TreeIndex child_idx, std::vector<Index>& indices,
                                 const GradientPair& stats, bool build_histogram) {
            // Check if should be leaf
            if (indices.size() < static_cast<size_t>(config_.min_samples_leaf) ||
                stats.hess < config_.min_child_weight ||
                current.depth + 1 >= config_.max_depth) {
                make_leaf(child_idx, stats);
                return;
            }

            // Get/compute histogram
            Histogram* child_hist = hist_cache_->get(child_idx);

            if (build_histogram) {
                // Build histogram for smaller child
                hist_builder.build(dataset, indices, all_features, *child_hist);
            } else {
                // Compute via subtraction for larger child
                Histogram* sibling_hist = hist_cache_->get(
                    left_is_smaller ? left_child : right_child
                );
                child_hist->subtract_from(*parent_hist, *sibling_hist);
            }

            // Find best split
            SplitInfo split = find_best_split(*child_hist, stats, all_features);

            if (split.is_valid && split.gain >= config_.min_split_gain) {
                LeafInfo child_info;
                child_info.node_idx = child_idx;
                child_info.sample_indices = std::move(indices);
                child_info.stats = stats;
                child_info.depth = current.depth + 1;
                child_info.best_split = split;
                child_info.has_histogram = true;
                leaf_queue_.push(std::move(child_info));
            } else {
                make_leaf(child_idx, stats);
                hist_cache_->release(child_idx);
            }
        };

        if (left_is_smaller) {
            // Build histogram for left (smaller), subtract for right (larger)
            process_child(left_child, left_indices, current.best_split.left_stats, true);
            process_child(right_child, right_indices, current.best_split.right_stats, false);
        } else {
            // Build histogram for right (smaller), subtract for left (larger)
            process_child(right_child, right_indices, current.best_split.right_stats, true);
            process_child(left_child, left_indices, current.best_split.left_stats, false);
        }

        // Release parent histogram
        hist_cache_->release(current.node_idx);
    }

    // Convert remaining queue items to leaves
    while (!leaf_queue_.empty()) {
        LeafInfo leaf = leaf_queue_.top();
        leaf_queue_.pop();
        make_leaf(leaf.node_idx, leaf.stats);
    }
}

TreeIndex LeafWiseTree::add_node() {
    TreeIndex idx = static_cast<TreeIndex>(nodes_.size());
    nodes_.emplace_back();
    return idx;
}

void LeafWiseTree::make_leaf(TreeIndex node_idx, const GradientPair& stats) {
    nodes_[node_idx].is_leaf = 1;
    nodes_[node_idx].value = compute_leaf_value(stats);

    if (config_.max_delta_step > 0) {
        nodes_[node_idx].value = std::max(-config_.max_delta_step,
                                          std::min(config_.max_delta_step, nodes_[node_idx].value));
    }

    n_leaves_++;
}

Float LeafWiseTree::compute_leaf_value(const GradientPair& stats) const {
    // Newton's method optimal leaf value with L2 regularization
    // For squared loss: optimal = -grad / (hess + lambda)
    // For other losses: this is a Newton step which is approximately optimal

    Float raw_value = -stats.grad / (stats.hess + config_.lambda_l2);

    // Apply L1 regularization (soft thresholding)
    if (config_.lambda_l1 > 0.0f) {
        Float abs_raw = std::abs(raw_value);
        if (abs_raw <= config_.lambda_l1 / (stats.hess + config_.lambda_l2)) {
            raw_value = 0.0f;
        } else {
            Float sign = (raw_value > 0) ? 1.0f : -1.0f;
            raw_value = sign * (abs_raw - config_.lambda_l1 / (stats.hess + config_.lambda_l2));
        }
    }

    // Apply leaf smoothing (shrinkage based on sample count)
    // This is CatBoost-style: smaller leaves get shrunk more
    if (config_.leaf_smooth > 0.0f && stats.count > 0) {
        Float shrink = static_cast<Float>(stats.count) / (stats.count + config_.leaf_smooth);
        raw_value *= shrink;
    }

    return raw_value;
}

SplitInfo LeafWiseTree::find_best_split(
    const Histogram& histogram,
    const GradientPair& parent_sum,
    const std::vector<FeatureIndex>& features
) {
    SplitInfo best_split;
    const Float lambda = config_.lambda_l2;
    const Float parent_gain = (parent_sum.grad * parent_sum.grad) / (parent_sum.hess + lambda);

    #pragma omp parallel
    {
        SplitInfo local_best;

        #pragma omp for nowait schedule(static)
        for (size_t i = 0; i < features.size(); ++i) {
            FeatureIndex f = features[i];
            BinIndex n_bins = histogram.max_bins();
            const GradientPair* bins = histogram.bins(f);

            GradientPair left_sum;

            for (BinIndex b = 0; b < n_bins - 1; ++b) {
                left_sum += bins[b];
                GradientPair right_sum = parent_sum - left_sum;

                // Check constraints
                if (left_sum.count < config_.min_samples_leaf ||
                    right_sum.count < config_.min_samples_leaf ||
                    left_sum.hess < config_.min_child_weight ||
                    right_sum.hess < config_.min_child_weight) {
                    continue;
                }

                // Compute gain
                Float gain_left = (left_sum.grad * left_sum.grad) / (left_sum.hess + lambda);
                Float gain_right = (right_sum.grad * right_sum.grad) / (right_sum.hess + lambda);
                Float gain = 0.5f * (gain_left + gain_right - parent_gain);

                if (gain > local_best.gain && gain >= config_.min_split_gain) {
                    local_best.gain = gain;
                    local_best.feature_idx = f;
                    local_best.bin_threshold = b;
                    local_best.left_stats = left_sum;
                    local_best.right_stats = right_sum;
                    local_best.left_value = -left_sum.grad / (left_sum.hess + lambda);
                    local_best.right_value = -right_sum.grad / (right_sum.hess + lambda);
                    local_best.is_valid = true;
                }
            }
        }

        #pragma omp critical
        {
            if (local_best.gain > best_split.gain) {
                best_split = local_best;
            }
        }
    }

    return best_split;
}

void LeafWiseTree::partition_samples(
    const Dataset& dataset,
    const std::vector<Index>& indices,
    const SplitInfo& split,
    std::vector<Index>& left_indices,
    std::vector<Index>& right_indices
) {
    const size_t n = indices.size();
    const BinIndex* feature_bins = dataset.binned().column(split.feature_idx);
    const BinIndex threshold = split.bin_threshold;
    const bool nan_goes_left = (split.left_stats.count >= split.right_stats.count);

#ifdef _OPENMP
    // Parallel partition for large datasets
    if (n >= 10000) {
        int n_threads = omp_get_max_threads();
        std::vector<std::vector<Index>> thread_left(n_threads);
        std::vector<std::vector<Index>> thread_right(n_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            thread_left[tid].reserve(n / (2 * n_threads) + 100);
            thread_right[tid].reserve(n / (2 * n_threads) + 100);

            #pragma omp for nowait
            for (size_t i = 0; i < n; ++i) {
                Index idx = indices[i];
                BinIndex bin = feature_bins[idx];

                if (bin == 255) {
                    if (nan_goes_left) {
                        thread_left[tid].push_back(idx);
                    } else {
                        thread_right[tid].push_back(idx);
                    }
                } else if (bin <= threshold) {
                    thread_left[tid].push_back(idx);
                } else {
                    thread_right[tid].push_back(idx);
                }
            }
        }

        // Merge thread-local results
        size_t total_left = 0, total_right = 0;
        for (int t = 0; t < n_threads; ++t) {
            total_left += thread_left[t].size();
            total_right += thread_right[t].size();
        }

        left_indices.clear();
        right_indices.clear();
        left_indices.reserve(total_left);
        right_indices.reserve(total_right);

        for (int t = 0; t < n_threads; ++t) {
            left_indices.insert(left_indices.end(), thread_left[t].begin(), thread_left[t].end());
            right_indices.insert(right_indices.end(), thread_right[t].begin(), thread_right[t].end());
        }
        return;
    }
#endif

    // Sequential partition for small datasets
    left_indices.clear();
    right_indices.clear();
    left_indices.reserve(n / 2);
    right_indices.reserve(n / 2);

    for (size_t i = 0; i < n; ++i) {
        Index idx = indices[i];
        BinIndex bin = feature_bins[idx];

        if (bin == 255) {
            if (nan_goes_left) {
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
}

Float LeafWiseTree::predict(const Dataset& dataset, Index row) const {
    if (nodes_.empty()) return 0.0f;

    TreeIndex node_idx = 0;

    while (!nodes_[node_idx].is_leaf) {
        const TreeNode& node = nodes_[node_idx];
        BinIndex bin = dataset.binned().column(node.split_feature)[row];

        if (bin == 255) {  // NaN
            node_idx = node.default_left ? node.left_child : node.right_child;
        } else if (bin <= node.split_bin) {
            node_idx = node.left_child;
        } else {
            node_idx = node.right_child;
        }
    }

    return nodes_[node_idx].value;
}

Float LeafWiseTree::predict(const Float* features, FeatureIndex n_features) const {
    if (nodes_.empty()) return 0.0f;

    TreeIndex node_idx = 0;

    while (!nodes_[node_idx].is_leaf) {
        const TreeNode& node = nodes_[node_idx];
        Float val = features[node.split_feature];

        // This is a simplified version - full version needs bin edges
        if (std::isnan(val)) {
            node_idx = node.default_left ? node.left_child : node.right_child;
        } else {
            // Need to compare with actual threshold value
            // For now, use split_bin as approximation (not accurate for raw features)
            node_idx = node.left_child;  // Placeholder
        }
    }

    return nodes_[node_idx].value;
}

void LeafWiseTree::predict_batch(const Dataset& dataset, Float* output) const {
    if (nodes_.empty()) {
        Index n = dataset.n_samples();
        for (Index i = 0; i < n; ++i) {
            output[i] = 0.0f;
        }
        return;
    }

    Index n_samples = dataset.n_samples();

    #pragma omp parallel for
    for (Index i = 0; i < n_samples; ++i) {
        TreeIndex node_idx = 0;

        while (!nodes_[node_idx].is_leaf) {
            const TreeNode& node = nodes_[node_idx];
            BinIndex bin = dataset.binned().column(node.split_feature)[i];

            if (bin == 255) {  // NaN
                node_idx = node.default_left ? node.left_child : node.right_child;
            } else if (bin <= node.split_bin) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
        }

        output[i] = nodes_[node_idx].value;
    }
}

std::vector<Float> LeafWiseTree::feature_importance() const {
    std::vector<Float> importance(256, 0.0f);

    for (const auto& node : nodes_) {
        if (!node.is_leaf) {
            importance[node.split_feature] += node.gain;
        }
    }

    return importance;
}

// ============================================================================
// Leaf-Wise Ensemble
// ============================================================================

void LeafWiseEnsemble::add_tree(std::unique_ptr<LeafWiseTree> tree, Float weight) {
    trees_.push_back(std::move(tree));
    tree_weights_.push_back(weight);
    tree_class_indices_.push_back(0);
}

void LeafWiseEnsemble::add_tree_for_class(std::unique_ptr<LeafWiseTree> tree, Float weight, uint32_t class_idx) {
    trees_.push_back(std::move(tree));
    tree_weights_.push_back(weight);
    tree_class_indices_.push_back(class_idx);
}

Float LeafWiseEnsemble::predict(const Dataset& data, Index row) const {
    Float sum = 0.0f;

    for (size_t i = 0; i < trees_.size(); ++i) {
        sum += tree_weights_[i] * trees_[i]->predict(data, row);
    }

    return sum;
}

void LeafWiseEnsemble::predict_batch(const Dataset& data, Float* output) const {
    Index n_samples = data.n_samples();
    std::memset(output, 0, n_samples * sizeof(Float));

    for (size_t t = 0; t < trees_.size(); ++t) {
        Float weight = tree_weights_[t];

        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            output[i] += weight * trees_[t]->predict(data, i);
        }
    }
}

void LeafWiseEnsemble::predict_batch_multiclass(const Dataset& data, Float* output) const {
    Index n_samples = data.n_samples();
    std::memset(output, 0, n_samples * n_classes_ * sizeof(Float));

    for (size_t t = 0; t < trees_.size(); ++t) {
        uint32_t class_idx = tree_class_indices_[t];
        Float weight = tree_weights_[t];

        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            Float pred = trees_[t]->predict(data, i);
            output[i * n_classes_ + class_idx] += weight * pred;
        }
    }
}

std::vector<Float> LeafWiseEnsemble::feature_importance() const {
    std::vector<Float> total(256, 0.0f);

    for (const auto& tree : trees_) {
        auto imp = tree->feature_importance();
        for (size_t i = 0; i < imp.size(); ++i) {
            total[i] += imp[i];
        }
    }

    return total;
}

} // namespace turbocat
