/**
 * TurboCat Feature Bundler Implementation
 *
 * Exclusive Feature Bundling (EFB) for sparse feature optimization.
 */

#include "turbocat/feature_bundler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace turbocat {

bool FeatureBundler::analyze(const Float* data, Index n_samples, FeatureIndex n_features) {
    bundles_.clear();
    feature_to_bundle_.clear();

    if (n_features == 0 || n_samples == 0) {
        return false;
    }

    // Find sparse features
    std::vector<FeatureIndex> sparse_features;
    if (config_.bundle_sparse_only) {
        sparse_features = find_sparse_features(data, n_samples, n_features);
    } else {
        sparse_features.resize(n_features);
        std::iota(sparse_features.begin(), sparse_features.end(), 0);
    }

    // If no sparse features, no bundling needed
    if (sparse_features.empty()) {
        return false;
    }

    // Compute conflict counts between sparse features
    auto conflicts = compute_conflict_counts(data, n_samples, n_features);

    // Greedy bundling
    greedy_bundling(conflicts, sparse_features, n_samples);

    // If all features ended up in separate bundles, no bundling benefit
    if (bundles_.size() >= static_cast<size_t>(n_features)) {
        bundles_.clear();
        feature_to_bundle_.clear();
        return false;
    }

    return bundles_.size() < static_cast<size_t>(n_features);
}

std::vector<FeatureIndex> FeatureBundler::find_sparse_features(
    const Float* data, Index n_samples, FeatureIndex n_features
) const {
    std::vector<FeatureIndex> sparse_features;

    #pragma omp parallel
    {
        std::vector<FeatureIndex> local_sparse;

        #pragma omp for nowait
        for (FeatureIndex f = 0; f < n_features; ++f) {
            Index non_zero_count = 0;
            for (Index i = 0; i < n_samples; ++i) {
                if (data[i * n_features + f] != 0.0f) {
                    non_zero_count++;
                }
            }

            float zero_fraction = 1.0f - static_cast<float>(non_zero_count) / n_samples;
            if (zero_fraction >= config_.sparse_threshold) {
                local_sparse.push_back(f);
            }
        }

        #pragma omp critical
        {
            sparse_features.insert(sparse_features.end(),
                                   local_sparse.begin(), local_sparse.end());
        }
    }

    std::sort(sparse_features.begin(), sparse_features.end());
    return sparse_features;
}

std::vector<std::vector<Index>> FeatureBundler::compute_conflict_counts(
    const Float* data, Index n_samples, FeatureIndex n_features
) const {
    // Initialize conflict matrix (only for sparse features vs all features)
    std::vector<std::vector<Index>> conflicts(n_features, std::vector<Index>(n_features, 0));

    // Count conflicts: samples where both features are non-zero
    #pragma omp parallel for schedule(dynamic)
    for (Index i = 0; i < n_samples; ++i) {
        const Float* row = data + i * n_features;

        // Find non-zero features in this row
        std::vector<FeatureIndex> nonzero_features;
        for (FeatureIndex f = 0; f < n_features; ++f) {
            if (row[f] != 0.0f) {
                nonzero_features.push_back(f);
            }
        }

        // Update conflict counts for all pairs
        for (size_t j = 0; j < nonzero_features.size(); ++j) {
            for (size_t k = j + 1; k < nonzero_features.size(); ++k) {
                FeatureIndex f1 = nonzero_features[j];
                FeatureIndex f2 = nonzero_features[k];
                #pragma omp atomic
                conflicts[f1][f2]++;
                #pragma omp atomic
                conflicts[f2][f1]++;
            }
        }
    }

    return conflicts;
}

void FeatureBundler::greedy_bundling(
    const std::vector<std::vector<Index>>& conflicts,
    const std::vector<FeatureIndex>& sparse_features,
    Index n_samples
) {
    Index max_conflicts = static_cast<Index>(config_.max_conflict_rate * n_samples);
    FeatureIndex n_features = static_cast<FeatureIndex>(conflicts.size());

    // Sort sparse features by number of non-zeros (descending)
    // Features with more non-zeros should be bundled first
    std::vector<FeatureIndex> sorted_features = sparse_features;

    // Initialize: each feature starts unbundled
    feature_to_bundle_.resize(n_features, static_cast<FeatureIndex>(-1));

    // Greedy bundling: for each feature, try to add to existing bundle
    for (FeatureIndex f : sorted_features) {
        bool added_to_bundle = false;

        // Try to add to an existing bundle
        for (size_t b = 0; b < bundles_.size(); ++b) {
            bool can_add = true;

            // Check conflicts with all features in this bundle
            for (FeatureIndex existing : bundles_[b].features) {
                if (conflicts[f][existing] > max_conflicts) {
                    can_add = false;
                    break;
                }
            }

            if (can_add) {
                bundles_[b].features.push_back(f);
                feature_to_bundle_[f] = static_cast<FeatureIndex>(b);
                added_to_bundle = true;
                break;
            }
        }

        // Create new bundle if couldn't add to existing
        if (!added_to_bundle) {
            FeatureBundle new_bundle;
            new_bundle.features.push_back(f);
            feature_to_bundle_[f] = static_cast<FeatureIndex>(bundles_.size());
            bundles_.push_back(std::move(new_bundle));
        }
    }

    // Add non-sparse features as individual bundles
    for (FeatureIndex f = 0; f < n_features; ++f) {
        if (feature_to_bundle_[f] == static_cast<FeatureIndex>(-1)) {
            FeatureBundle new_bundle;
            new_bundle.features.push_back(f);
            feature_to_bundle_[f] = static_cast<FeatureIndex>(bundles_.size());
            bundles_.push_back(std::move(new_bundle));
        }
    }
}

std::unique_ptr<BinnedData> FeatureBundler::apply_bundling(
    const BinnedData& original,
    const std::vector<BinIndex>& n_bins_per_feature
) const {
    if (!has_bundling()) {
        return nullptr;
    }

    Index n_rows = original.n_rows();
    FeatureIndex n_bundles = static_cast<FeatureIndex>(bundles_.size());

    // Calculate bin offsets for each bundle
    std::vector<FeatureBundle> bundles_with_offsets = bundles_;
    for (auto& bundle : bundles_with_offsets) {
        BinIndex offset = 0;
        bundle.bin_offsets.clear();
        bundle.bin_offsets.reserve(bundle.features.size());

        for (FeatureIndex f : bundle.features) {
            bundle.bin_offsets.push_back(offset);
            offset += n_bins_per_feature[f];
        }
        bundle.total_bins = offset;
    }

    // Create bundled data
    auto bundled = std::make_unique<BinnedData>(n_rows, n_bundles);

    // Apply bundling
    #pragma omp parallel for
    for (Index row = 0; row < n_rows; ++row) {
        for (FeatureIndex b = 0; b < n_bundles; ++b) {
            const auto& bundle = bundles_with_offsets[b];
            BinIndex bundled_bin = 0;

            // Find the non-zero feature in this bundle (if any)
            for (size_t i = 0; i < bundle.features.size(); ++i) {
                FeatureIndex f = bundle.features[i];
                BinIndex bin = original.get(row, f);

                // If this feature has a non-zero bin, use it with offset
                if (bin > 0) {
                    bundled_bin = bin + bundle.bin_offsets[i];
                    break;  // Only one feature in bundle should be non-zero
                }
            }

            bundled->set(row, b, bundled_bin);
        }
    }

    return bundled;
}

} // namespace turbocat
