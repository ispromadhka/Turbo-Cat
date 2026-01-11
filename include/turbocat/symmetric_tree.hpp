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

// Forward declarations
class FastEnsemble;
class FastFloatEnsemble;

// ============================================================================
// Symmetric Tree Structure
// ============================================================================

struct SymmetricSplit {
    FeatureIndex feature = 0;
    BinIndex threshold = 0;      // Bin threshold (for binned prediction)
    Float float_threshold = 0.0f; // Raw float threshold (for direct prediction)
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

    // Batch prediction with SIMD (requires binned data)
    void predict_batch(const Dataset& dataset, Float* output) const;

    // Raw float prediction - skips binning entirely
    // Uses float_threshold for direct comparisons (like CatBoost)
    Float predict_raw(const Float* features, FeatureIndex n_features) const;
    void predict_batch_raw(const Float* data, Index n_samples, FeatureIndex n_features, Float* output) const;

    // Tree info
    uint16_t depth() const { return depth_; }
    uint32_t n_leaves() const { return 1u << depth_; }

    // Feature importance
    std::vector<Float> feature_importance() const;

    // Accessors for FastEnsemble
    const std::vector<SymmetricSplit>& splits() const { return splits_; }
    const std::vector<Float>& leaf_values() const { return leaf_values_; }

    // Serialization
    void save(std::ostream& out) const;
    static SymmetricTree load(std::istream& in);

    // For deserialization - set tree data directly
    void set_data(uint16_t depth, std::vector<SymmetricSplit> splits, std::vector<Float> leaves) {
        depth_ = depth;
        splits_ = std::move(splits);
        leaf_values_ = std::move(leaves);
    }

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
     * Find best split with histogram caching for subtraction trick
     */
    SymmetricSplit find_best_level_split_with_histograms(
        const Dataset& dataset,
        const std::vector<std::vector<Index>>& node_samples,
        const std::vector<GradientPair>& node_stats,
        HistogramBuilder& hist_builder,
        std::vector<Histogram>& out_histograms  // Output: histograms for this level
    );

    /**
     * Find best split from pre-built histograms (for histogram subtraction trick)
     */
    SymmetricSplit find_best_split_from_histograms(
        const Dataset& dataset,
        const std::vector<Histogram>& histograms,
        const std::vector<GradientPair>& node_stats
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
    SymmetricEnsemble();
    explicit SymmetricEnsemble(uint32_t n_classes);
    ~SymmetricEnsemble();

    // Move operations
    SymmetricEnsemble(SymmetricEnsemble&&) noexcept;
    SymmetricEnsemble& operator=(SymmetricEnsemble&&) noexcept;

    // Disable copy
    SymmetricEnsemble(const SymmetricEnsemble&) = delete;
    SymmetricEnsemble& operator=(const SymmetricEnsemble&) = delete;

    void add_tree(std::unique_ptr<SymmetricTree> tree, Float weight = 1.0f);
    void add_tree_for_class(std::unique_ptr<SymmetricTree> tree, Float weight, uint32_t class_idx);

    // Binary/regression prediction (requires binned data)
    Float predict(const Dataset& data, Index row) const;
    void predict_batch(const Dataset& data, Float* output) const;

    // Raw float prediction - skips binning entirely (like CatBoost)
    void predict_batch_raw(const Float* data, Index n_samples, FeatureIndex n_features, Float* output) const;

    // FASTEST raw float prediction - uses cached flat tree data + optional transpose
    // This is the recommended method for production inference without binning
    void predict_batch_raw_fast(const Float* data, Index n_samples, FeatureIndex n_features, Float* output) const;

    // Multiclass prediction
    void predict_multiclass(const Dataset& data, Index row, Float* output) const;
    void predict_batch_multiclass(const Dataset& data, Float* output) const;

    size_t n_trees() const { return trees_.size(); }
    uint32_t n_classes() const { return n_classes_; }
    void set_n_classes(uint32_t n) { n_classes_ = n; }

    std::vector<Float> feature_importance() const;

    // Accessors for FastEnsemble
    const SymmetricTree& tree(size_t idx) const { return *trees_[idx]; }
    Float tree_weight(size_t idx) const { return tree_weights_[idx]; }

    // Prepare fast ensemble for optimized prediction
    void prepare_fast_ensemble() const;

    // Prepare fast float ensemble for optimized raw prediction (no binning)
    void prepare_fast_float_ensemble() const;

private:
    std::vector<std::unique_ptr<SymmetricTree>> trees_;
    std::vector<Float> tree_weights_;
    std::vector<uint32_t> tree_class_indices_;
    uint32_t n_classes_ = 1;

    // Cached fast ensemble for SIMD prediction (binned data)
    mutable bool fast_prepared_ = false;
    mutable std::unique_ptr<FastEnsemble> fast_ensemble_;

    // Cached fast float ensemble for SIMD prediction (raw float data, no binning)
    mutable bool fast_float_prepared_ = false;
    mutable std::unique_ptr<FastFloatEnsemble> fast_float_ensemble_;
};

} // namespace turbocat
