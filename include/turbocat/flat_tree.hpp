/**
 * TurboCat Flat Tree (Decision Table) for Ultra-Fast Inference
 *
 * Converts regular trees into flat decision tables for O(1) prediction.
 * Combined with SIMD, this achieves faster inference than CatBoost.
 */

#pragma once

#include "turbocat/types.hpp"
#include <vector>
#include <cstring>
#include <algorithm>

#if defined(__AVX2__) || defined(TURBOCAT_AVX2)
#define USE_AVX2_FLAT 1
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(TURBOCAT_NEON)
#define USE_NEON_FLAT 1
#include <arm_neon.h>
#endif

namespace turbocat {

// Forward declarations
class Tree;
class TreeEnsemble;

// Maximum tree depth for flat representation (2^8 = 256 leaves max)
constexpr int MAX_FLAT_DEPTH = 8;

/**
 * FlatTree - Decision table representation for ultra-fast inference
 *
 * For a tree of depth D:
 * - Store D (feature_index, split_threshold) pairs for each level
 * - Store 2^D leaf values in a flat array
 *
 * Prediction: O(D) comparisons to compute index, then O(1) lookup
 * With SIMD: process 8 samples simultaneously
 */
struct FlatTree {
    // Per-level split info (up to MAX_FLAT_DEPTH levels)
    alignas(32) FeatureIndex features[MAX_FLAT_DEPTH];  // Feature to split on at each level
    alignas(32) BinIndex thresholds[MAX_FLAT_DEPTH];    // Split threshold at each level

    // Leaf values - 2^depth entries, aligned for SIMD
    alignas(32) Float* leaf_values = nullptr;

    uint8_t depth = 0;
    uint32_t n_leaves = 0;
    Float weight = 1.0f;

    FlatTree() = default;

    FlatTree(const FlatTree& other) : depth(other.depth), n_leaves(other.n_leaves), weight(other.weight) {
        std::memcpy(features, other.features, sizeof(features));
        std::memcpy(thresholds, other.thresholds, sizeof(thresholds));
        if (other.leaf_values && n_leaves > 0) {
            leaf_values = static_cast<Float*>(std::aligned_alloc(32, n_leaves * sizeof(Float)));
            std::memcpy(leaf_values, other.leaf_values, n_leaves * sizeof(Float));
        }
    }

    FlatTree(FlatTree&& other) noexcept : depth(other.depth), n_leaves(other.n_leaves), weight(other.weight) {
        std::memcpy(features, other.features, sizeof(features));
        std::memcpy(thresholds, other.thresholds, sizeof(thresholds));
        leaf_values = other.leaf_values;
        other.leaf_values = nullptr;
    }

    FlatTree& operator=(const FlatTree& other) {
        if (this != &other) {
            if (leaf_values) std::free(leaf_values);
            depth = other.depth;
            n_leaves = other.n_leaves;
            weight = other.weight;
            std::memcpy(features, other.features, sizeof(features));
            std::memcpy(thresholds, other.thresholds, sizeof(thresholds));
            if (other.leaf_values && n_leaves > 0) {
                leaf_values = static_cast<Float*>(std::aligned_alloc(32, n_leaves * sizeof(Float)));
                std::memcpy(leaf_values, other.leaf_values, n_leaves * sizeof(Float));
            } else {
                leaf_values = nullptr;
            }
        }
        return *this;
    }

    FlatTree& operator=(FlatTree&& other) noexcept {
        if (this != &other) {
            if (leaf_values) std::free(leaf_values);
            depth = other.depth;
            n_leaves = other.n_leaves;
            weight = other.weight;
            std::memcpy(features, other.features, sizeof(features));
            std::memcpy(thresholds, other.thresholds, sizeof(thresholds));
            leaf_values = other.leaf_values;
            other.leaf_values = nullptr;
        }
        return *this;
    }

    ~FlatTree() {
        if (leaf_values) std::free(leaf_values);
    }

    // Predict single sample - inline for speed
    inline Float predict(const BinIndex* sample_bins) const {
        uint32_t idx = 0;
        for (uint8_t d = 0; d < depth; ++d) {
            // If bin > threshold, go right (set bit)
            idx = (idx << 1) | (sample_bins[features[d]] > thresholds[d] ? 1 : 0);
        }
        return weight * leaf_values[idx];
    }
};

/**
 * FlatTreeEnsemble - Collection of flat trees for batch prediction
 */
class FlatTreeEnsemble {
public:
    FlatTreeEnsemble() = default;

    // Convert from regular TreeEnsemble
    void from_ensemble(const TreeEnsemble& ensemble);

    // Convert single tree to flat representation
    static FlatTree flatten_tree(const Tree& tree, Float weight);

    // Batch prediction - main interface
    void predict_batch(const BinIndex* data, Index n_samples, FeatureIndex n_features,
                       Float* output) const;

    // SIMD-optimized batch prediction
    void predict_batch_simd(const BinIndex* data, Index n_samples, FeatureIndex n_features,
                            Float* output) const;

    size_t n_trees() const { return trees_.size(); }
    bool empty() const { return trees_.empty(); }

private:
    std::vector<FlatTree> trees_;
    uint8_t max_depth_ = 0;

    // Helper to recursively fill leaf values
    static void fill_leaves(const Tree& tree, const std::vector<TreeNode>& nodes,
                           TreeIndex node_idx, uint32_t path, uint8_t current_depth,
                           uint8_t target_depth, Float* leaf_values,
                           const FeatureIndex* level_features, const BinIndex* level_thresholds);
};

// ============================================================================
// Inline implementations for speed
// ============================================================================

inline void FlatTreeEnsemble::predict_batch(const BinIndex* data, Index n_samples,
                                            FeatureIndex n_features, Float* output) const {
    // Zero output
    std::memset(output, 0, n_samples * sizeof(Float));

    if (trees_.empty()) return;

    // Process each sample
    #pragma omp parallel for schedule(static)
    for (Index i = 0; i < n_samples; ++i) {
        const BinIndex* sample = data + i * n_features;
        Float sum = 0.0f;

        for (const auto& tree : trees_) {
            sum += tree.predict(sample);
        }

        output[i] = sum;
    }
}

#ifdef USE_AVX2_FLAT
inline void FlatTreeEnsemble::predict_batch_simd(const BinIndex* data, Index n_samples,
                                                  FeatureIndex n_features, Float* output) const {
    std::memset(output, 0, n_samples * sizeof(Float));

    if (trees_.empty()) return;

    // Process 8 samples at a time with AVX2
    Index n_simd = (n_samples / 8) * 8;

    #pragma omp parallel for schedule(static)
    for (Index base = 0; base < n_simd; base += 8) {
        __m256 sums = _mm256_setzero_ps();

        for (const auto& tree : trees_) {
            // Compute 8 leaf indices
            __m256i indices = _mm256_setzero_si256();

            for (uint8_t d = 0; d < tree.depth; ++d) {
                FeatureIndex feat = tree.features[d];
                BinIndex thresh = tree.thresholds[d];

                // Load 8 bin values for this feature
                alignas(32) int32_t bins[8];
                for (int j = 0; j < 8; ++j) {
                    bins[j] = data[(base + j) * n_features + feat];
                }
                __m256i bin_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(bins));

                // Compare: bins > thresh
                __m256i thresh_vec = _mm256_set1_epi32(thresh);
                __m256i cmp = _mm256_cmpgt_epi32(bin_vec, thresh_vec);

                // Update indices: indices = (indices << 1) | (cmp & 1)
                indices = _mm256_slli_epi32(indices, 1);
                indices = _mm256_or_si256(indices, _mm256_and_si256(cmp, _mm256_set1_epi32(1)));
            }

            // Gather leaf values using indices
            __m256 leaves = _mm256_i32gather_ps(tree.leaf_values, indices, 4);

            // Accumulate weighted sum
            __m256 weight_vec = _mm256_set1_ps(tree.weight);
            sums = _mm256_fmadd_ps(weight_vec, leaves, sums);
        }

        // Store results
        _mm256_storeu_ps(output + base, sums);
    }

    // Handle remaining samples
    for (Index i = n_simd; i < n_samples; ++i) {
        const BinIndex* sample = data + i * n_features;
        Float sum = 0.0f;
        for (const auto& tree : trees_) {
            sum += tree.predict(sample);
        }
        output[i] = sum;
    }
}
#elif defined(USE_NEON_FLAT)
inline void FlatTreeEnsemble::predict_batch_simd(const BinIndex* data, Index n_samples,
                                                  FeatureIndex n_features, Float* output) const {
    std::memset(output, 0, n_samples * sizeof(Float));

    if (trees_.empty()) return;

    // Process 4 samples at a time with NEON
    Index n_simd = (n_samples / 4) * 4;

    #pragma omp parallel for schedule(static)
    for (Index base = 0; base < n_simd; base += 4) {
        float32x4_t sums = vdupq_n_f32(0.0f);

        for (const auto& tree : trees_) {
            // Compute 4 leaf indices
            uint32x4_t indices = vdupq_n_u32(0);

            for (uint8_t d = 0; d < tree.depth; ++d) {
                FeatureIndex feat = tree.features[d];
                BinIndex thresh = tree.thresholds[d];

                // Load 4 bin values
                alignas(16) int32_t bins[4];
                for (int j = 0; j < 4; ++j) {
                    bins[j] = data[(base + j) * n_features + feat];
                }
                int32x4_t bin_vec = vld1q_s32(bins);

                // Compare: bins > thresh
                int32x4_t thresh_vec = vdupq_n_s32(thresh);
                uint32x4_t cmp = vcgtq_s32(bin_vec, thresh_vec);

                // Update indices
                indices = vshlq_n_u32(indices, 1);
                indices = vorrq_u32(indices, vandq_u32(cmp, vdupq_n_u32(1)));
            }

            // Gather leaf values (manual for NEON)
            alignas(16) uint32_t idx_arr[4];
            vst1q_u32(idx_arr, indices);

            alignas(16) float leaves[4];
            for (int j = 0; j < 4; ++j) {
                leaves[j] = tree.leaf_values[idx_arr[j]];
            }
            float32x4_t leaf_vec = vld1q_f32(leaves);

            // Accumulate
            float32x4_t weight_vec = vdupq_n_f32(tree.weight);
            sums = vmlaq_f32(sums, weight_vec, leaf_vec);
        }

        vst1q_f32(output + base, sums);
    }

    // Handle remaining
    for (Index i = n_simd; i < n_samples; ++i) {
        const BinIndex* sample = data + i * n_features;
        Float sum = 0.0f;
        for (const auto& tree : trees_) {
            sum += tree.predict(sample);
        }
        output[i] = sum;
    }
}
#else
// Fallback without SIMD
inline void FlatTreeEnsemble::predict_batch_simd(const BinIndex* data, Index n_samples,
                                                  FeatureIndex n_features, Float* output) const {
    predict_batch(data, n_samples, n_features, output);
}
#endif

} // namespace turbocat
