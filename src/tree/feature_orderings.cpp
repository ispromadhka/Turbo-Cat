/**
 * TurboCat Pre-sorted Feature Orderings Implementation
 */

#include "turbocat/feature_orderings.hpp"
#include "turbocat/dataset.hpp"
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TURBOCAT_AVX2
#include <immintrin.h>
#endif

namespace turbocat {

void FeatureOrderings::compute(const Dataset& dataset) {
    n_features_ = dataset.n_features();
    n_samples_ = dataset.n_samples();

    if (n_samples_ == 0 || n_features_ == 0) return;

    sorted_indices_.resize(n_features_);
    bin_starts_.resize(n_features_);

    const int max_bins = 256;  // Maximum possible bins (uint8_t + 1 for counting)

    // Parallel computation across features
    #pragma omp parallel for schedule(dynamic)
    for (FeatureIndex f = 0; f < n_features_; ++f) {
        const BinIndex* bins = dataset.binned().column(f);

        // Initialize indices
        std::vector<Index>& indices = sorted_indices_[f];
        indices.resize(n_samples_);
        std::iota(indices.begin(), indices.end(), static_cast<Index>(0));

        // Sort by bin value (stable sort to maintain order within same bin)
        // Use counting sort for O(n) complexity since bins are small integers
        std::vector<Index> counts(max_bins + 1, 0);

        // Count occurrences of each bin
        for (Index i = 0; i < n_samples_; ++i) {
            counts[bins[i] + 1]++;
        }

        // Compute cumulative counts (prefix sum)
        for (int b = 1; b <= max_bins; ++b) {
            counts[b] += counts[b - 1];
        }

        // Store bin starts
        bin_starts_[f].resize(max_bins + 1);
        for (int b = 0; b <= max_bins; ++b) {
            bin_starts_[f][b] = counts[b];
        }

        // Place elements in sorted order (counting sort)
        std::vector<Index> temp(n_samples_);
        for (Index i = 0; i < n_samples_; ++i) {
            BinIndex bin = bins[i];
            temp[counts[bin]++] = i;
        }

        indices = std::move(temp);
    }
}

void OrderedHistogramBuilder::build(
    const Dataset& dataset,
    const FeatureOrderings& orderings,
    const std::vector<Index>& sample_mask,
    FeatureIndex feature,
    GradientPair* output_bins,
    BinIndex max_bins
) const {
    const Float* grads = dataset.gradients();
    const Float* hess = dataset.hessians();
    const BinIndex* bins = dataset.binned().column(feature);
    const auto& sorted_idx = orderings.sorted_indices(feature);

    // Clear output
    std::memset(output_bins, 0, max_bins * sizeof(GradientPair));

    if (sample_mask.empty()) {
        // All samples: use sorted order for sequential access
        // Accumulate gradients directly into bins
        for (Index pos = 0; pos < orderings.n_samples(); ++pos) {
            Index idx = sorted_idx[pos];
            BinIndex bin = bins[idx];
            output_bins[bin].grad += grads[idx];
            output_bins[bin].hess += hess[idx];
            output_bins[bin].count += 1;
        }
    } else {
        // Subset of samples: use mask
        // Create a set for O(1) lookup
        std::vector<bool> in_mask(orderings.n_samples(), false);
        for (Index idx : sample_mask) {
            in_mask[idx] = true;
        }

        for (Index pos = 0; pos < orderings.n_samples(); ++pos) {
            Index idx = sorted_idx[pos];
            if (in_mask[idx]) {
                BinIndex bin = bins[idx];
                output_bins[bin].grad += grads[idx];
                output_bins[bin].hess += hess[idx];
                output_bins[bin].count += 1;
            }
        }
    }
}

void OrderedHistogramBuilder::build_all(
    const Dataset& dataset,
    const FeatureOrderings& orderings,
    const std::vector<Index>& sample_mask,
    std::vector<GradientPair>& output,
    BinIndex max_bins,
    int n_threads
) const {
    const FeatureIndex n_features = orderings.n_features();
    output.resize(static_cast<size_t>(n_features) * max_bins);
    std::memset(output.data(), 0, output.size() * sizeof(GradientPair));

    if (n_threads <= 0) {
        #ifdef _OPENMP
        n_threads = omp_get_max_threads();
        #else
        n_threads = 1;
        #endif
    }

    const Float* grads = dataset.gradients();
    const Float* hess = dataset.hessians();

    if (sample_mask.empty()) {
        // All samples: most efficient path
        // Process all features in parallel, each feature sequentially through sorted samples

        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (FeatureIndex f = 0; f < n_features; ++f) {
            const BinIndex* bins = dataset.binned().column(f);
            const auto& sorted_idx = orderings.sorted_indices(f);
            GradientPair* out = output.data() + f * max_bins;

            // Sequential accumulation in sorted order
            for (Index pos = 0; pos < orderings.n_samples(); ++pos) {
                Index idx = sorted_idx[pos];
                BinIndex bin = bins[idx];
                out[bin].grad += grads[idx];
                out[bin].hess += hess[idx];
                out[bin].count += 1;
            }
        }
    } else {
        // Subset: create boolean mask first
        std::vector<bool> in_mask(orderings.n_samples(), false);
        for (Index idx : sample_mask) {
            in_mask[idx] = true;
        }

        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (FeatureIndex f = 0; f < n_features; ++f) {
            const BinIndex* bins = dataset.binned().column(f);
            const auto& sorted_idx = orderings.sorted_indices(f);
            GradientPair* out = output.data() + f * max_bins;

            for (Index pos = 0; pos < orderings.n_samples(); ++pos) {
                Index idx = sorted_idx[pos];
                if (in_mask[idx]) {
                    BinIndex bin = bins[idx];
                    out[bin].grad += grads[idx];
                    out[bin].hess += hess[idx];
                    out[bin].count += 1;
                }
            }
        }
    }
}

void OrderedHistogramBuilder::build_for_nodes(
    const Dataset& dataset,
    const FeatureOrderings& orderings,
    const std::vector<int32_t>& sample_to_node,
    uint32_t n_nodes,
    FeatureIndex feature,
    std::vector<GradientPair>& output,
    BinIndex max_bins
) const {
    output.resize(static_cast<size_t>(n_nodes) * max_bins);
    std::memset(output.data(), 0, output.size() * sizeof(GradientPair));

    const Float* grads = dataset.gradients();
    const Float* hess = dataset.hessians();
    const BinIndex* bins = dataset.binned().column(feature);
    const auto& sorted_idx = orderings.sorted_indices(feature);

    // Sequential pass through sorted samples, accumulate to appropriate node's histogram
    for (Index pos = 0; pos < orderings.n_samples(); ++pos) {
        Index idx = sorted_idx[pos];
        int32_t node = sample_to_node[idx];

        if (node >= 0) {
            BinIndex bin = bins[idx];
            GradientPair* node_hist = output.data() + node * max_bins;
            node_hist[bin].grad += grads[idx];
            node_hist[bin].hess += hess[idx];
            node_hist[bin].count += 1;
        }
    }
}

void OrderedHistogramBuilder::build_all_for_nodes(
    const Dataset& dataset,
    const FeatureOrderings& orderings,
    const std::vector<int32_t>& sample_to_node,
    uint32_t n_nodes,
    GradientPair* output,
    BinIndex max_bins,
    int n_threads
) const {
    const FeatureIndex n_features = orderings.n_features();
    const Index n_samples = orderings.n_samples();
    const size_t hist_size_per_node = static_cast<size_t>(n_features) * max_bins;
    const size_t total_size = static_cast<size_t>(n_nodes) * hist_size_per_node;

    // Clear output
    std::memset(output, 0, total_size * sizeof(GradientPair));

    if (n_threads <= 0) {
        #ifdef _OPENMP
        n_threads = omp_get_max_threads();
        #else
        n_threads = 1;
        #endif
    }

    const Float* grads = dataset.gradients();
    const Float* hess = dataset.hessians();

    // For large datasets, use a sample-parallel approach with thread-local histograms
    // This reduces cache misses by processing samples in chunks
    if (n_samples >= 50000 && n_nodes <= 64) {
        // Thread-local histogram accumulation for better cache efficiency
        const int actual_threads = std::min(n_threads, 8);  // Cap threads for memory efficiency
        std::vector<std::vector<GradientPair>> thread_hists(actual_threads);

        for (int t = 0; t < actual_threads; ++t) {
            thread_hists[t].resize(total_size, GradientPair{0.0f, 0.0f, 0});
        }

        #pragma omp parallel num_threads(actual_threads)
        {
            int tid = 0;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            #endif
            GradientPair* local_hist = thread_hists[tid].data();

            // Each thread processes a chunk of features
            #pragma omp for schedule(static)
            for (FeatureIndex f = 0; f < n_features; ++f) {
                const BinIndex* bins = dataset.binned().column(f);
                const auto& sorted_idx = orderings.sorted_indices(f);

                // Process all samples for this feature
                for (Index pos = 0; pos < n_samples; ++pos) {
                    Index idx = sorted_idx[pos];
                    int32_t node = sample_to_node[idx];

                    if (node >= 0) {
                        BinIndex bin = bins[idx];
                        // Layout: [node][feature][bin]
                        size_t offset = static_cast<size_t>(node) * hist_size_per_node +
                                       static_cast<size_t>(f) * max_bins + bin;
                        local_hist[offset].grad += grads[idx];
                        local_hist[offset].hess += hess[idx];
                        local_hist[offset].count += 1;
                    }
                }
            }
        }

        // Reduce thread-local histograms
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total_size; ++i) {
            for (int t = 0; t < actual_threads; ++t) {
                output[i].grad += thread_hists[t][i].grad;
                output[i].hess += thread_hists[t][i].hess;
                output[i].count += thread_hists[t][i].count;
            }
        }
    } else {
        // For smaller datasets or many nodes, use feature-parallel approach
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (FeatureIndex f = 0; f < n_features; ++f) {
            const BinIndex* bins = dataset.binned().column(f);
            const auto& sorted_idx = orderings.sorted_indices(f);

            for (Index pos = 0; pos < n_samples; ++pos) {
                Index idx = sorted_idx[pos];
                int32_t node = sample_to_node[idx];

                if (node >= 0) {
                    BinIndex bin = bins[idx];
                    // Layout: [node][feature][bin]
                    size_t offset = static_cast<size_t>(node) * hist_size_per_node +
                                   static_cast<size_t>(f) * max_bins + bin;
                    output[offset].grad += grads[idx];
                    output[offset].hess += hess[idx];
                    output[offset].count += 1;
                }
            }
        }
    }
}

} // namespace turbocat
