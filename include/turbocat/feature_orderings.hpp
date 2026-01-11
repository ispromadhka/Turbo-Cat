#pragma once

/**
 * TurboCat Pre-sorted Feature Orderings
 *
 * Key optimization for fast histogram building:
 * - Sort samples by each feature's bin value once at training start
 * - Use sorted order for cumulative gradient sums during histogram building
 * - Enables sequential memory access instead of random scatter-add
 *
 * This is how CatBoost and LightGBM achieve fast training.
 */

#include "turbocat/types.hpp"
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace turbocat {

class Dataset;  // Forward declaration

/**
 * FeatureOrderings - Pre-sorted sample indices for each feature
 *
 * For feature f, sorted_indices_[f] contains sample indices sorted by bin[f][idx]
 * This enables O(n) histogram building with sequential memory access.
 */
class FeatureOrderings {
public:
    FeatureOrderings() = default;

    /**
     * Compute sorted orderings from binned data
     * @param dataset The dataset with binned features
     */
    void compute(const Dataset& dataset);

    /**
     * Get sorted indices for a feature
     * Indices are sorted by increasing bin value
     */
    const std::vector<Index>& sorted_indices(FeatureIndex feature) const {
        return sorted_indices_[feature];
    }

    /**
     * Get bin boundaries for a feature
     * bin_starts_[f][b] = first position in sorted_indices_[f] with bin >= b
     */
    const std::vector<Index>& bin_starts(FeatureIndex feature) const {
        return bin_starts_[feature];
    }

    FeatureIndex n_features() const { return n_features_; }
    Index n_samples() const { return n_samples_; }
    bool is_computed() const { return !sorted_indices_.empty(); }

private:
    FeatureIndex n_features_ = 0;
    Index n_samples_ = 0;

    // sorted_indices_[feature] = sample indices sorted by bin value
    std::vector<std::vector<Index>> sorted_indices_;

    // bin_starts_[feature][bin] = starting position in sorted_indices for this bin
    std::vector<std::vector<Index>> bin_starts_;
};

/**
 * Fast histogram builder using pre-sorted orderings
 *
 * Key insight: With samples sorted by bin value, we can:
 * 1. Iterate through samples in order
 * 2. Accumulate gradients sequentially
 * 3. When bin changes, we know the cumulative sum for previous bin
 *
 * This is O(n_samples) with sequential memory access,
 * vs O(n_samples) with random scatter-add in the naive approach.
 */
class OrderedHistogramBuilder {
public:
    OrderedHistogramBuilder() = default;

    /**
     * Build histogram using pre-sorted orderings
     * Much faster than scatter-add approach due to sequential access
     */
    void build(
        const Dataset& dataset,
        const FeatureOrderings& orderings,
        const std::vector<Index>& sample_mask,  // Which samples to include (empty = all)
        FeatureIndex feature,
        GradientPair* output_bins,
        BinIndex max_bins
    ) const;

    /**
     * Build histograms for all features in parallel
     */
    void build_all(
        const Dataset& dataset,
        const FeatureOrderings& orderings,
        const std::vector<Index>& sample_mask,
        std::vector<GradientPair>& output,  // [n_features * max_bins]
        BinIndex max_bins,
        int n_threads = -1
    ) const;

    /**
     * Build histogram for a subset of samples using membership array
     * membership[sample_idx] = node_id for this sample, or -1 if not included
     */
    void build_for_nodes(
        const Dataset& dataset,
        const FeatureOrderings& orderings,
        const std::vector<int32_t>& sample_to_node,  // sample -> node mapping (-1 = excluded)
        uint32_t n_nodes,
        FeatureIndex feature,
        std::vector<GradientPair>& output,  // [n_nodes * max_bins]
        BinIndex max_bins
    ) const;

    /**
     * Build histograms for ALL features and ALL nodes in one optimized pass
     * This is the fastest method for large datasets
     * @param output Pre-allocated buffer [n_nodes * n_features * max_bins]
     */
    void build_all_for_nodes(
        const Dataset& dataset,
        const FeatureOrderings& orderings,
        const std::vector<int32_t>& sample_to_node,
        uint32_t n_nodes,
        GradientPair* output,  // [n_nodes * n_features * max_bins] - pre-allocated
        BinIndex max_bins,
        int n_threads = -1
    ) const;
};

} // namespace turbocat
