#pragma once

/**
 * TurboCat Exclusive Feature Bundling (EFB)
 *
 * Implements LightGBM-style feature bundling for sparse features.
 * Mutually exclusive features (rarely non-zero simultaneously) are
 * bundled into a single feature using bin offsets.
 *
 * Key optimizations:
 * - Reduces #features to #bundles where #bundles << #features for sparse data
 * - Histogram building complexity: O(data × features) → O(data × bundles)
 * - Improves cache locality by bundling related features
 */

#include "types.hpp"
#include "dataset.hpp"
#include <vector>
#include <memory>

namespace turbocat {

// ============================================================================
// Feature Bundle
// ============================================================================

struct FeatureBundle {
    std::vector<FeatureIndex> features;      // Original features in this bundle
    std::vector<BinIndex> bin_offsets;       // Offset for each feature's bins
    BinIndex total_bins;                      // Total bins in bundled feature

    FeatureBundle() : total_bins(0) {}
};

// ============================================================================
// Feature Bundler
// ============================================================================

class FeatureBundler {
public:
    /**
     * Configuration for bundling
     */
    struct Config {
        float max_conflict_rate = 0.0f;      // Max fraction of conflicting samples (0 = perfect exclusivity)
        Index min_samples_for_conflict = 0;  // Min samples to consider non-zero
        bool bundle_sparse_only = true;       // Only bundle features with many zeros
        float sparse_threshold = 0.9f;        // Min fraction of zeros to be sparse
    };

    FeatureBundler() = default;
    explicit FeatureBundler(const Config& config) : config_(config) {}

    /**
     * Analyze dataset and create feature bundles
     * @param data Raw feature data (row-major)
     * @param n_samples Number of samples
     * @param n_features Number of features
     * @return True if bundling was performed
     */
    bool analyze(const Float* data, Index n_samples, FeatureIndex n_features);

    /**
     * Apply bundling to binned data
     * Creates new binned data with bundled features
     */
    std::unique_ptr<BinnedData> apply_bundling(
        const BinnedData& original,
        const std::vector<BinIndex>& n_bins_per_feature
    ) const;

    /**
     * Map original feature index to bundled feature index
     */
    FeatureIndex map_feature(FeatureIndex original) const {
        if (original < feature_to_bundle_.size()) {
            return feature_to_bundle_[original];
        }
        return original;
    }

    /**
     * Get bundle information
     */
    const std::vector<FeatureBundle>& bundles() const { return bundles_; }
    size_t n_bundles() const { return bundles_.size(); }
    bool has_bundling() const { return !bundles_.empty(); }

    /**
     * Get bin offset for a feature within its bundle
     */
    BinIndex get_bin_offset(FeatureIndex original) const {
        if (!has_bundling() || original >= feature_to_bundle_.size()) {
            return 0;
        }
        FeatureIndex bundle_idx = feature_to_bundle_[original];
        const auto& bundle = bundles_[bundle_idx];
        for (size_t i = 0; i < bundle.features.size(); ++i) {
            if (bundle.features[i] == original) {
                return bundle.bin_offsets[i];
            }
        }
        return 0;
    }

private:
    Config config_;
    std::vector<FeatureBundle> bundles_;
    std::vector<FeatureIndex> feature_to_bundle_;  // Maps original feature to bundle index

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /**
     * Compute conflict matrix between features
     * Returns matrix where conflict[i][j] = number of samples where both i and j are non-zero
     */
    std::vector<std::vector<Index>> compute_conflict_counts(
        const Float* data, Index n_samples, FeatureIndex n_features
    ) const;

    /**
     * Find sparse features (high fraction of zeros)
     */
    std::vector<FeatureIndex> find_sparse_features(
        const Float* data, Index n_samples, FeatureIndex n_features
    ) const;

    /**
     * Greedy bundling using graph coloring approach
     */
    void greedy_bundling(
        const std::vector<std::vector<Index>>& conflicts,
        const std::vector<FeatureIndex>& sparse_features,
        Index n_samples
    );
};

} // namespace turbocat
