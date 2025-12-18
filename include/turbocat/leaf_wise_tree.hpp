#pragma once

/**
 * TurboCat Leaf-Wise Tree Builder
 *
 * Key optimizations:
 * - Leaf-wise growth (like LightGBM) - always split the leaf with max gain
 * - Histogram caching - reuse parent histograms
 * - Histogram subtraction - compute sibling histogram via subtraction
 * - Memory pool - avoid repeated allocations
 */

#include "types.hpp"
#include "config.hpp"
#include "dataset.hpp"
#include "histogram.hpp"
#include <queue>
#include <memory>
#include <vector>

namespace turbocat {

// ============================================================================
// Leaf Info for Priority Queue
// ============================================================================

struct LeafInfo {
    TreeIndex node_idx;
    std::vector<Index> sample_indices;
    GradientPair stats;
    uint16_t depth;
    SplitInfo best_split;
    bool has_histogram;

    // For priority queue - higher gain = higher priority
    bool operator<(const LeafInfo& other) const {
        return best_split.gain < other.best_split.gain;
    }
};

// ============================================================================
// Histogram Cache
// ============================================================================

class HistogramCache {
public:
    HistogramCache(FeatureIndex n_features, BinIndex max_bins, size_t max_cache_size = 64);

    // Get or create histogram for a node
    Histogram* get(TreeIndex node_idx);

    // Release histogram (mark as reusable)
    void release(TreeIndex node_idx);

    // Clear all
    void clear();

private:
    FeatureIndex n_features_;
    BinIndex max_bins_;
    size_t max_cache_size_;

    std::vector<std::unique_ptr<Histogram>> pool_;
    std::vector<TreeIndex> pool_node_ids_;
    std::vector<bool> pool_in_use_;
};

// ============================================================================
// Leaf-Wise Tree
// ============================================================================

class LeafWiseTree {
public:
    LeafWiseTree() = default;
    explicit LeafWiseTree(const TreeConfig& config);

    /**
     * Build tree using leaf-wise strategy
     * @param dataset Dataset with gradients set
     * @param sample_indices Samples to use
     * @param hist_builder Histogram builder
     */
    void build(
        const Dataset& dataset,
        const std::vector<Index>& sample_indices,
        HistogramBuilder& hist_builder
    );

    // Prediction
    Float predict(const Dataset& dataset, Index row) const;
    Float predict(const Float* features, FeatureIndex n_features) const;
    void predict_batch(const Dataset& dataset, Float* output) const;

    // Tree info
    TreeIndex n_nodes() const { return static_cast<TreeIndex>(nodes_.size()); }
    TreeIndex n_leaves() const { return n_leaves_; }
    uint16_t depth() const { return depth_; }

    // Feature importance
    std::vector<Float> feature_importance() const;

private:
    TreeConfig config_;
    std::vector<TreeNode> nodes_;
    TreeIndex n_leaves_ = 0;
    uint16_t depth_ = 0;

    // Histogram cache for subtraction trick
    std::unique_ptr<HistogramCache> hist_cache_;

    // Priority queue of leaves
    std::priority_queue<LeafInfo> leaf_queue_;

    // ========================================================================
    // Internal Methods
    // ========================================================================

    TreeIndex add_node();

    void make_leaf(TreeIndex node_idx, const GradientPair& stats);

    /**
     * Find best split for a leaf using histogram
     */
    SplitInfo find_best_split(
        const Histogram& histogram,
        const GradientPair& stats,
        const std::vector<FeatureIndex>& features
    );

    /**
     * Partition samples based on split
     */
    void partition_samples(
        const Dataset& dataset,
        const std::vector<Index>& indices,
        const SplitInfo& split,
        std::vector<Index>& left_indices,
        std::vector<Index>& right_indices
    );

    /**
     * Compute leaf value
     */
    Float compute_leaf_value(const GradientPair& stats) const;
};

// ============================================================================
// Leaf-Wise Tree Ensemble
// ============================================================================

class LeafWiseEnsemble {
public:
    LeafWiseEnsemble() = default;
    explicit LeafWiseEnsemble(uint32_t n_classes) : n_classes_(n_classes) {}

    void add_tree(std::unique_ptr<LeafWiseTree> tree, Float weight = 1.0f);
    void add_tree_for_class(std::unique_ptr<LeafWiseTree> tree, Float weight, uint32_t class_idx);

    // Binary/regression prediction
    Float predict(const Dataset& data, Index row) const;
    void predict_batch(const Dataset& data, Float* output) const;

    // Multiclass prediction
    void predict_batch_multiclass(const Dataset& data, Float* output) const;

    size_t n_trees() const { return trees_.size(); }
    uint32_t n_classes() const { return n_classes_; }
    void set_n_classes(uint32_t n) { n_classes_ = n; }

    std::vector<Float> feature_importance() const;

private:
    std::vector<std::unique_ptr<LeafWiseTree>> trees_;
    std::vector<Float> tree_weights_;
    std::vector<uint32_t> tree_class_indices_;
    uint32_t n_classes_ = 1;
};

} // namespace turbocat
