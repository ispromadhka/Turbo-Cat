/**
 * TurboCat Symmetric (Oblivious) Tree Implementation
 *
 * Key optimization: ONE split per depth level instead of per node.
 * This dramatically reduces the complexity of split finding.
 */

#include "turbocat/symmetric_tree.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
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
    FeatureIndex n_features = dataset.n_features();

    // Initialize: all samples in one node (root)
    std::vector<std::vector<Index>> current_level_samples(1);
    current_level_samples[0] = sample_indices;

    std::vector<GradientPair> current_level_stats(1);
    for (Index idx : sample_indices) {
        current_level_stats[0].grad += grads[idx];
        current_level_stats[0].hess += hess[idx];
        current_level_stats[0].count += 1;
    }

    splits_.clear();
    depth_ = 0;

    // Build tree level by level
    for (uint16_t d = 0; d < config_.max_depth; ++d) {
        uint32_t n_nodes = 1u << d;

        // Check if we should stop
        bool should_stop = true;
        for (uint32_t n = 0; n < n_nodes; ++n) {
            if (current_level_samples[n].size() >= static_cast<size_t>(config_.min_samples_leaf * 2) &&
                current_level_stats[n].hess >= config_.min_child_weight * 2) {
                should_stop = false;
                break;
            }
        }
        if (should_stop) break;

        // Find best split for this level (ONE split for ALL nodes)
        SymmetricSplit best_split = find_best_level_split(
            dataset, current_level_samples, current_level_stats, hist_builder
        );

        // Check if split is valid
        if (best_split.gain < config_.min_split_gain) {
            break;
        }

        // Apply split to all nodes
        splits_.push_back(best_split);
        depth_ = d + 1;

        // Create next level
        uint32_t next_n_nodes = n_nodes * 2;
        std::vector<std::vector<Index>> next_level_samples(next_n_nodes);
        std::vector<GradientPair> next_level_stats(next_n_nodes);

        // Pre-allocate
        for (uint32_t n = 0; n < next_n_nodes; ++n) {
            next_level_samples[n].reserve(sample_indices.size() / next_n_nodes + 100);
        }

        const BinIndex* split_bins = dataset.binned().column(best_split.feature);
        const BinIndex threshold = best_split.threshold;

        // Partition samples for each node - parallel across nodes (no data races)
        #pragma omp parallel for schedule(dynamic)
        for (uint32_t n = 0; n < n_nodes; ++n) {
            uint32_t left_child = n * 2;
            uint32_t right_child = n * 2 + 1;

            // Each node's children are unique, no need for critical section
            for (Index idx : current_level_samples[n]) {
                BinIndex bin = split_bins[idx];

                if (bin == 255 || bin <= threshold) {
                    // Go left
                    next_level_samples[left_child].push_back(idx);
                    next_level_stats[left_child].grad += grads[idx];
                    next_level_stats[left_child].hess += hess[idx];
                    next_level_stats[left_child].count += 1;
                } else {
                    // Go right
                    next_level_samples[right_child].push_back(idx);
                    next_level_stats[right_child].grad += grads[idx];
                    next_level_stats[right_child].hess += hess[idx];
                    next_level_stats[right_child].count += 1;
                }
            }
        }

        current_level_samples = std::move(next_level_samples);
        current_level_stats = std::move(next_level_stats);
    }

    // Compute leaf values
    uint32_t n_leaves = 1u << depth_;
    leaf_values_.resize(n_leaves);

    for (uint32_t i = 0; i < n_leaves; ++i) {
        if (i < current_level_stats.size()) {
            leaf_values_[i] = compute_leaf_value(current_level_stats[i]);
        } else {
            leaf_values_[i] = 0.0f;
        }
    }
}

SymmetricSplit SymmetricTree::find_best_level_split(
    const Dataset& dataset,
    const std::vector<std::vector<Index>>& node_samples,
    const std::vector<GradientPair>& node_stats,
    HistogramBuilder& hist_builder
) {
    FeatureIndex n_features = dataset.n_features();
    BinIndex max_bins = config_.max_bins;
    Float lambda = config_.lambda_l2;

    uint32_t n_nodes = static_cast<uint32_t>(node_samples.size());

    // Use merged histogram for speed - CatBoost style
    // Build histogram for ALL samples at this level
    std::vector<Index> all_samples;
    GradientPair total_stats;

    for (uint32_t n = 0; n < n_nodes; ++n) {
        all_samples.insert(all_samples.end(),
                          node_samples[n].begin(),
                          node_samples[n].end());
        total_stats += node_stats[n];
    }

    // Build merged histogram
    Histogram merged_hist(n_features, max_bins);
    hist_builder.build(dataset, all_samples, {}, merged_hist);

    // Find best split using merged histogram
    // For symmetric trees, the merged gain approximates the sum of individual gains
    // This is fast and works well in practice (like CatBoost)
    SymmetricSplit best_split;
    best_split.gain = -1e30f;

    // Parallel search over features
    #pragma omp parallel
    {
        SymmetricSplit local_best;
        local_best.gain = -1e30f;

        #pragma omp for nowait
        for (FeatureIndex f = 0; f < n_features; ++f) {
            const GradientPair* hist_bins = merged_hist.bins(f);

            // Cumulative sum for split evaluation
            GradientPair left_sum;

            for (BinIndex b = 0; b < max_bins - 1; ++b) {
                left_sum += hist_bins[b];
                GradientPair right_sum = total_stats - left_sum;

                // Check constraints (relaxed for symmetric trees)
                if (left_sum.count < config_.min_samples_leaf ||
                    right_sum.count < config_.min_samples_leaf) {
                    continue;
                }

                if (left_sum.hess < 1e-10f || right_sum.hess < 1e-10f) {
                    continue;
                }

                // Compute gain on merged data
                Float left_gain = (left_sum.grad * left_sum.grad) / (left_sum.hess + lambda);
                Float right_gain = (right_sum.grad * right_sum.grad) / (right_sum.hess + lambda);
                Float parent_gain = (total_stats.grad * total_stats.grad) / (total_stats.hess + lambda);

                Float gain = 0.5f * (left_gain + right_gain - parent_gain);

                if (gain > local_best.gain) {
                    local_best.feature = f;
                    local_best.threshold = b;
                    local_best.gain = gain;
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

        // Bit d is 1 if we go right, 0 if we go left
        if (bin != 255 && bin > split.threshold) {
            leaf_idx |= (1u << d);
        }
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

            if (bin != 255 && bin > split.threshold) {
                leaf_idx |= (1u << d);
            }
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
    std::memset(output, 0, n_samples * sizeof(Float));

    for (size_t t = 0; t < trees_.size(); ++t) {
        Float weight = tree_weights_[t];

        #pragma omp parallel for
        for (Index i = 0; i < n_samples; ++i) {
            output[i] += weight * trees_[t]->predict(data, i);
        }
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

} // namespace turbocat
