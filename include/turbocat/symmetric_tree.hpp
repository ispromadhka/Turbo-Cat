#pragma once

/**
 * TurboCat Symmetric (Oblivious) Tree
 *
 * Like CatBoost's oblivious trees:
 * - All nodes at the same depth use the SAME split (feature, threshold)
 * - Tree of depth d has exactly 2^d leaves
 * - Leaf index computed via bitwise operations (very fast prediction)
 * - Much faster training: O(features × bins × depth) vs O(features × bins × nodes)
 *
 * Key insight: Instead of finding best split for each node,
 * find ONE best split for each depth level that maximizes total gain.
 */

#include "types.hpp"
#include "config.hpp"
#include "dataset.hpp"
#include "histogram.hpp"
#include <vector>
#include <memory>

namespace turbocat {

// ============================================================================
// Symmetric Tree Structure
// ============================================================================

struct SymmetricSplit {
    FeatureIndex feature = 0;
    BinIndex threshold = 0;
    Float gain = 0.0f;
};

class SymmetricTree {
public:
    SymmetricTree() = default;
    explicit SymmetricTree(const TreeConfig& config);

    /**
     * Build symmetric tree
     * At each depth level, find ONE best split for ALL nodes at that level
     */
    void build(
        const Dataset& dataset,
        const std::vector<Index>& sample_indices,
        HistogramBuilder& hist_builder
    );

    // Prediction - very fast using bitwise operations
    Float predict(const Dataset& dataset, Index row) const;
    Float predict(const Float* features, FeatureIndex n_features) const;

    // Batch prediction with SIMD
    void predict_batch(const Dataset& dataset, Float* output) const;

    // Tree info
    uint16_t depth() const { return depth_; }
    uint32_t n_leaves() const { return 1u << depth_; }

    // Feature importance
    std::vector<Float> feature_importance() const;

private:
    TreeConfig config_;
    uint16_t depth_ = 0;

    // Splits: one per depth level
    std::vector<SymmetricSplit> splits_;

    // Leaf values: 2^depth values
    std::vector<Float> leaf_values_;

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /**
     * Find best split for a depth level
     * Considers ALL samples that reach this level and finds
     * the single best split that maximizes total gain
     */
    SymmetricSplit find_best_level_split(
        const Dataset& dataset,
        const std::vector<std::vector<Index>>& node_samples,  // samples per node at this level
        const std::vector<GradientPair>& node_stats,          // stats per node
        HistogramBuilder& hist_builder
    );

    /**
     * Compute leaf value with L2 regularization
     */
    Float compute_leaf_value(const GradientPair& stats) const;

    /**
     * Get leaf index for a sample (bitwise operations)
     */
    uint32_t get_leaf_index(const Dataset& dataset, Index row) const;
    uint32_t get_leaf_index(const Float* features, FeatureIndex n_features) const;
};

// ============================================================================
// Symmetric Tree Ensemble
// ============================================================================

class SymmetricEnsemble {
public:
    SymmetricEnsemble() = default;
    explicit SymmetricEnsemble(uint32_t n_classes) : n_classes_(n_classes) {}

    void add_tree(std::unique_ptr<SymmetricTree> tree, Float weight = 1.0f);
    void add_tree_for_class(std::unique_ptr<SymmetricTree> tree, Float weight, uint32_t class_idx);

    // Binary/regression prediction
    Float predict(const Dataset& data, Index row) const;
    void predict_batch(const Dataset& data, Float* output) const;

    // Multiclass prediction
    void predict_multiclass(const Dataset& data, Index row, Float* output) const;
    void predict_batch_multiclass(const Dataset& data, Float* output) const;

    size_t n_trees() const { return trees_.size(); }
    uint32_t n_classes() const { return n_classes_; }
    void set_n_classes(uint32_t n) { n_classes_ = n; }

    std::vector<Float> feature_importance() const;

private:
    std::vector<std::unique_ptr<SymmetricTree>> trees_;
    std::vector<Float> tree_weights_;
    std::vector<uint32_t> tree_class_indices_;
    uint32_t n_classes_ = 1;
};

} // namespace turbocat
