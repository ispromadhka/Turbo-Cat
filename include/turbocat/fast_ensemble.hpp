/**
 * TurboCat Fast Ensemble - Ultra-optimized inference for symmetric trees
 *
 * Key optimizations:
 * 1. Flat memory layout - all tree data in contiguous arrays
 * 2. SIMD vectorization - process 8 samples at once with AVX2
 * 3. Decision table lookup - O(depth) comparisons + O(1) leaf lookup
 * 4. Cache-friendly access patterns
 */

#pragma once

#include "turbocat/types.hpp"
#include "turbocat/symmetric_tree.hpp"
#include <vector>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) || defined(TURBOCAT_AVX2)
#include <immintrin.h>
#define HAS_AVX2 1
#endif

#if defined(__ARM_NEON) || defined(TURBOCAT_NEON)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

namespace turbocat {

/**
 * FastEnsemble - Flat representation of symmetric tree ensemble
 *
 * Memory layout for N trees with max depth D:
 * - features[N * D]     : feature index at each level of each tree
 * - thresholds[N * D]   : split threshold at each level of each tree
 * - leaf_values[N * 2^D]: all leaf values flattened
 * - weights[N]          : tree weights
 */
class FastEnsemble {
public:
    FastEnsemble() = default;

    // Build from symmetric ensemble
    void from_symmetric_ensemble(const SymmetricEnsemble& ensemble);

    // Ultra-fast batch prediction (row-major data: data[sample * n_features + feature])
    void predict_batch(const BinIndex* data, Index n_samples, FeatureIndex n_features,
                       Float* output, int n_threads = -1) const;

    // Ultra-fast batch prediction using column-major data (data[feature * n_samples + sample])
    // This is faster because we can load 8 consecutive bytes for 8 samples
    void predict_batch_column_major(const BinIndex* data, Index n_samples, FeatureIndex n_features,
                                    Float* output, int n_threads = -1) const;

    size_t n_trees() const { return n_trees_; }
    uint16_t max_depth() const { return max_depth_; }
    bool empty() const { return n_trees_ == 0; }

private:
    size_t n_trees_ = 0;
    uint16_t max_depth_ = 0;
    uint32_t leaves_per_tree_ = 0;

    // Flat arrays for cache-efficient access (aligned for SIMD)
    alignas(64) std::vector<FeatureIndex> features_;   // [n_trees * max_depth]
    alignas(64) std::vector<BinIndex> thresholds_;     // [n_trees * max_depth]
    alignas(64) std::vector<Float> leaf_values_;       // [n_trees * 2^max_depth]
    alignas(64) std::vector<Float> weights_;           // [n_trees]
    alignas(64) std::vector<uint16_t> depths_;         // [n_trees] actual depth of each tree

    // Scalar prediction for single sample
    inline Float predict_single(const BinIndex* sample, size_t tree_idx) const;
};

// ============================================================================
// Inline implementations
// ============================================================================

inline Float FastEnsemble::predict_single(const BinIndex* sample, size_t tree_idx) const {
    uint16_t depth = depths_[tree_idx];
    const FeatureIndex* tree_features = features_.data() + tree_idx * max_depth_;
    const BinIndex* tree_thresholds = thresholds_.data() + tree_idx * max_depth_;
    const Float* tree_leaves = leaf_values_.data() + tree_idx * leaves_per_tree_;

    uint32_t leaf_idx = 0;
    for (uint16_t d = 0; d < depth; ++d) {
        BinIndex bin = sample[tree_features[d]];
        bool go_right = (bin != 255 && bin > tree_thresholds[d]);
        leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
    }

    return weights_[tree_idx] * tree_leaves[leaf_idx];
}

inline void FastEnsemble::predict_batch(const BinIndex* data, Index n_samples, FeatureIndex n_features,
                                        Float* output, int n_threads) const {
    if (n_trees_ == 0) {
        std::memset(output, 0, n_samples * sizeof(Float));
        return;
    }

    // Determine threads
    if (n_threads <= 0) {
        #ifdef _OPENMP
        n_threads = omp_get_max_threads();
        #else
        n_threads = 1;
        #endif
    }

#ifdef HAS_AVX2
    // Process 8 samples at a time with AVX2
    Index n_simd = (n_samples / 8) * 8;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index base = 0; base < n_simd; base += 8) {
        __m256 sums = _mm256_setzero_ps();

        for (size_t t = 0; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;
            Float weight = weights_[t];

            // Compute 8 leaf indices in parallel
            __m256i indices = _mm256_setzero_si256();

            for (uint16_t d = 0; d < depth; ++d) {
                FeatureIndex feat = tree_features[d];
                BinIndex thresh = tree_thresholds[d];

                // Load 8 bin values
                alignas(32) int32_t bins[8];
                for (int j = 0; j < 8; ++j) {
                    bins[j] = data[(base + j) * n_features + feat];
                }
                __m256i bin_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(bins));

                // Check for NaN (bin == 255) - treat as go_left
                __m256i nan_mask = _mm256_cmpeq_epi32(bin_vec, _mm256_set1_epi32(255));

                // Compare: bins > thresh
                __m256i thresh_vec = _mm256_set1_epi32(thresh);
                __m256i cmp = _mm256_cmpgt_epi32(bin_vec, thresh_vec);

                // go_right = (bin != 255) && (bin > thresh)
                __m256i go_right = _mm256_andnot_si256(nan_mask, cmp);

                // Update indices: indices = (indices << 1) | go_right
                indices = _mm256_slli_epi32(indices, 1);
                indices = _mm256_or_si256(indices, _mm256_and_si256(go_right, _mm256_set1_epi32(1)));
            }

            // Gather leaf values
            __m256 leaves = _mm256_i32gather_ps(tree_leaves, indices, 4);

            // Accumulate weighted sum
            __m256 weight_vec = _mm256_set1_ps(weight);
            sums = _mm256_fmadd_ps(weight_vec, leaves, sums);
        }

        _mm256_storeu_ps(output + base, sums);
    }

    // Handle remaining samples
    for (Index i = n_simd; i < n_samples; ++i) {
        const BinIndex* sample = data + i * n_features;
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            sum += predict_single(sample, t);
        }
        output[i] = sum;
    }

#elif defined(HAS_NEON)
    // Process 4 samples at a time with NEON
    Index n_simd = (n_samples / 4) * 4;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index base = 0; base < n_simd; base += 4) {
        float32x4_t sums = vdupq_n_f32(0.0f);

        for (size_t t = 0; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;
            Float weight = weights_[t];

            uint32x4_t indices = vdupq_n_u32(0);

            for (uint16_t d = 0; d < depth; ++d) {
                FeatureIndex feat = tree_features[d];
                BinIndex thresh = tree_thresholds[d];

                alignas(16) int32_t bins[4];
                for (int j = 0; j < 4; ++j) {
                    bins[j] = data[(base + j) * n_features + feat];
                }
                int32x4_t bin_vec = vld1q_s32(bins);

                uint32x4_t nan_mask = vceqq_s32(bin_vec, vdupq_n_s32(255));
                uint32x4_t cmp = vcgtq_s32(bin_vec, vdupq_n_s32(thresh));
                uint32x4_t go_right = vbicq_u32(cmp, nan_mask);

                indices = vshlq_n_u32(indices, 1);
                indices = vorrq_u32(indices, vandq_u32(go_right, vdupq_n_u32(1)));
            }

            // Manual gather for NEON
            alignas(16) uint32_t idx_arr[4];
            vst1q_u32(idx_arr, indices);

            alignas(16) float leaves[4];
            for (int j = 0; j < 4; ++j) {
                leaves[j] = tree_leaves[idx_arr[j]];
            }
            float32x4_t leaf_vec = vld1q_f32(leaves);

            sums = vmlaq_n_f32(sums, leaf_vec, weight);
        }

        vst1q_f32(output + base, sums);
    }

    for (Index i = n_simd; i < n_samples; ++i) {
        const BinIndex* sample = data + i * n_features;
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            sum += predict_single(sample, t);
        }
        output[i] = sum;
    }

#else
    // Scalar fallback
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index i = 0; i < n_samples; ++i) {
        const BinIndex* sample = data + i * n_features;
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            sum += predict_single(sample, t);
        }
        output[i] = sum;
    }
#endif
}

// Column-major version: data layout is data[feature * n_samples + sample]
// This allows loading 8 consecutive bytes for 8 samples
inline void FastEnsemble::predict_batch_column_major(const BinIndex* data, Index n_samples, FeatureIndex n_features,
                                                      Float* output, int n_threads) const {
    if (n_trees_ == 0) {
        std::memset(output, 0, n_samples * sizeof(Float));
        return;
    }

    if (n_threads <= 0) {
        #ifdef _OPENMP
        n_threads = omp_get_max_threads();
        #else
        n_threads = 1;
        #endif
    }

#ifdef HAS_AVX2
    Index n_simd = (n_samples / 8) * 8;

    // Optimized: Process 4 trees at a time for better ILP, use scalar gather (faster than gather instruction)
    const size_t tree_batch = 4;
    const size_t n_tree_batches = n_trees_ / tree_batch;
    const size_t remaining_trees = n_trees_ % tree_batch;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index base = 0; base < n_simd; base += 8) {
        __m256 sums = _mm256_setzero_ps();

        // Process trees in batches of 4 for better instruction-level parallelism
        for (size_t tb = 0; tb < n_tree_batches; ++tb) {
            size_t t0 = tb * tree_batch;

            // Prefetch next tree batch data
            if (tb + 1 < n_tree_batches) {
                size_t t_next = (tb + 1) * tree_batch;
                _mm_prefetch(reinterpret_cast<const char*>(leaf_values_.data() + t_next * leaves_per_tree_), _MM_HINT_T0);
            }

            // Process 4 trees in parallel
            alignas(32) uint32_t idx0[8], idx1[8], idx2[8], idx3[8];

            // Tree 0
            {
                size_t t = t0;
                uint16_t depth = depths_[t];
                const FeatureIndex* tree_features = features_.data() + t * max_depth_;
                const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
                __m256i indices = _mm256_setzero_si256();

                for (uint16_t d = 0; d < depth; ++d) {
                    const BinIndex* feat_data = data + tree_features[d] * n_samples + base;
                    __m128i bytes8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(feat_data));
                    __m256i bin_vec = _mm256_cvtepu8_epi32(bytes8);
                    __m256i nan_mask = _mm256_cmpeq_epi32(bin_vec, _mm256_set1_epi32(255));
                    __m256i cmp = _mm256_cmpgt_epi32(bin_vec, _mm256_set1_epi32(tree_thresholds[d]));
                    __m256i go_right = _mm256_andnot_si256(nan_mask, cmp);
                    indices = _mm256_slli_epi32(indices, 1);
                    indices = _mm256_or_si256(indices, _mm256_and_si256(go_right, _mm256_set1_epi32(1)));
                }
                _mm256_store_si256(reinterpret_cast<__m256i*>(idx0), indices);
            }

            // Tree 1
            {
                size_t t = t0 + 1;
                uint16_t depth = depths_[t];
                const FeatureIndex* tree_features = features_.data() + t * max_depth_;
                const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
                __m256i indices = _mm256_setzero_si256();

                for (uint16_t d = 0; d < depth; ++d) {
                    const BinIndex* feat_data = data + tree_features[d] * n_samples + base;
                    __m128i bytes8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(feat_data));
                    __m256i bin_vec = _mm256_cvtepu8_epi32(bytes8);
                    __m256i nan_mask = _mm256_cmpeq_epi32(bin_vec, _mm256_set1_epi32(255));
                    __m256i cmp = _mm256_cmpgt_epi32(bin_vec, _mm256_set1_epi32(tree_thresholds[d]));
                    __m256i go_right = _mm256_andnot_si256(nan_mask, cmp);
                    indices = _mm256_slli_epi32(indices, 1);
                    indices = _mm256_or_si256(indices, _mm256_and_si256(go_right, _mm256_set1_epi32(1)));
                }
                _mm256_store_si256(reinterpret_cast<__m256i*>(idx1), indices);
            }

            // Tree 2
            {
                size_t t = t0 + 2;
                uint16_t depth = depths_[t];
                const FeatureIndex* tree_features = features_.data() + t * max_depth_;
                const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
                __m256i indices = _mm256_setzero_si256();

                for (uint16_t d = 0; d < depth; ++d) {
                    const BinIndex* feat_data = data + tree_features[d] * n_samples + base;
                    __m128i bytes8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(feat_data));
                    __m256i bin_vec = _mm256_cvtepu8_epi32(bytes8);
                    __m256i nan_mask = _mm256_cmpeq_epi32(bin_vec, _mm256_set1_epi32(255));
                    __m256i cmp = _mm256_cmpgt_epi32(bin_vec, _mm256_set1_epi32(tree_thresholds[d]));
                    __m256i go_right = _mm256_andnot_si256(nan_mask, cmp);
                    indices = _mm256_slli_epi32(indices, 1);
                    indices = _mm256_or_si256(indices, _mm256_and_si256(go_right, _mm256_set1_epi32(1)));
                }
                _mm256_store_si256(reinterpret_cast<__m256i*>(idx2), indices);
            }

            // Tree 3
            {
                size_t t = t0 + 3;
                uint16_t depth = depths_[t];
                const FeatureIndex* tree_features = features_.data() + t * max_depth_;
                const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
                __m256i indices = _mm256_setzero_si256();

                for (uint16_t d = 0; d < depth; ++d) {
                    const BinIndex* feat_data = data + tree_features[d] * n_samples + base;
                    __m128i bytes8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(feat_data));
                    __m256i bin_vec = _mm256_cvtepu8_epi32(bytes8);
                    __m256i nan_mask = _mm256_cmpeq_epi32(bin_vec, _mm256_set1_epi32(255));
                    __m256i cmp = _mm256_cmpgt_epi32(bin_vec, _mm256_set1_epi32(tree_thresholds[d]));
                    __m256i go_right = _mm256_andnot_si256(nan_mask, cmp);
                    indices = _mm256_slli_epi32(indices, 1);
                    indices = _mm256_or_si256(indices, _mm256_and_si256(go_right, _mm256_set1_epi32(1)));
                }
                _mm256_store_si256(reinterpret_cast<__m256i*>(idx3), indices);
            }

            // Scalar gather (often faster than _mm256_i32gather_ps on modern CPUs)
            const Float* leaves0 = leaf_values_.data() + t0 * leaves_per_tree_;
            const Float* leaves1 = leaf_values_.data() + (t0 + 1) * leaves_per_tree_;
            const Float* leaves2 = leaf_values_.data() + (t0 + 2) * leaves_per_tree_;
            const Float* leaves3 = leaf_values_.data() + (t0 + 3) * leaves_per_tree_;
            Float w0 = weights_[t0], w1 = weights_[t0+1], w2 = weights_[t0+2], w3 = weights_[t0+3];

            alignas(32) float leaf_vals[8];
            for (int j = 0; j < 8; ++j) {
                leaf_vals[j] = w0 * leaves0[idx0[j]] + w1 * leaves1[idx1[j]] +
                               w2 * leaves2[idx2[j]] + w3 * leaves3[idx3[j]];
            }
            sums = _mm256_add_ps(sums, _mm256_load_ps(leaf_vals));
        }

        // Process remaining trees (< 4)
        for (size_t t = n_tree_batches * tree_batch; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;
            Float weight = weights_[t];

            __m256i indices = _mm256_setzero_si256();
            for (uint16_t d = 0; d < depth; ++d) {
                const BinIndex* feat_data = data + tree_features[d] * n_samples + base;
                __m128i bytes8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(feat_data));
                __m256i bin_vec = _mm256_cvtepu8_epi32(bytes8);
                __m256i nan_mask = _mm256_cmpeq_epi32(bin_vec, _mm256_set1_epi32(255));
                __m256i cmp = _mm256_cmpgt_epi32(bin_vec, _mm256_set1_epi32(tree_thresholds[d]));
                __m256i go_right = _mm256_andnot_si256(nan_mask, cmp);
                indices = _mm256_slli_epi32(indices, 1);
                indices = _mm256_or_si256(indices, _mm256_and_si256(go_right, _mm256_set1_epi32(1)));
            }

            // Scalar gather for remaining trees
            alignas(32) uint32_t idx_arr[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(idx_arr), indices);
            alignas(32) float leaf_vals[8];
            for (int j = 0; j < 8; ++j) {
                leaf_vals[j] = weight * tree_leaves[idx_arr[j]];
            }
            sums = _mm256_add_ps(sums, _mm256_load_ps(leaf_vals));
        }

        _mm256_storeu_ps(output + base, sums);
    }

    // Handle remaining samples
    for (Index i = n_simd; i < n_samples; ++i) {
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;

            uint32_t leaf_idx = 0;
            for (uint16_t d = 0; d < depth; ++d) {
                FeatureIndex feat = tree_features[d];
                BinIndex bin = data[feat * n_samples + i];
                bool go_right = (bin != 255 && bin > tree_thresholds[d]);
                leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
            }
            sum += weights_[t] * tree_leaves[leaf_idx];
        }
        output[i] = sum;
    }

#elif defined(HAS_NEON)
    // ARM NEON optimized version - process 8 samples at once using 2x float32x4
    // Tree batching: process 4 trees at a time for better cache utilization
    Index n_batch = (n_samples / 8) * 8;

    const size_t tree_batch_size = 4;
    const size_t n_tree_batches = n_trees_ / tree_batch_size;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index base = 0; base < n_batch; base += 8) {
        float32x4_t sums0 = vdupq_n_f32(0.0f);
        float32x4_t sums1 = vdupq_n_f32(0.0f);

        // Process trees in batches of 4 for better cache utilization
        for (size_t tb = 0; tb < n_tree_batches; ++tb) {
            size_t t0 = tb * tree_batch_size;

            // Prefetch next tree batch data
            if (tb + 1 < n_tree_batches) {
                size_t t_next = (tb + 1) * tree_batch_size;
                __builtin_prefetch(leaf_values_.data() + t_next * leaves_per_tree_, 0, 0);
                __builtin_prefetch(leaf_values_.data() + (t_next + 1) * leaves_per_tree_, 0, 0);
            }

            // Process 4 trees directly without intermediate buffer
            for (size_t ti = 0; ti < tree_batch_size; ++ti) {
                size_t t = t0 + ti;
                uint16_t depth = depths_[t];
                const FeatureIndex* tree_features = features_.data() + t * max_depth_;
                const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
                const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;
                Float weight = weights_[t];

                uint32x4_t indices0 = vdupq_n_u32(0);
                uint32x4_t indices1 = vdupq_n_u32(0);

                for (uint16_t d = 0; d < depth; ++d) {
                    FeatureIndex feat = tree_features[d];
                    BinIndex thresh = tree_thresholds[d];
                    const BinIndex* feat_data = data + feat * n_samples + base;

                    uint8x8_t bytes8 = vld1_u8(feat_data);
                    uint16x8_t bytes16 = vmovl_u8(bytes8);
                    uint32x4_t bin0 = vmovl_u16(vget_low_u16(bytes16));
                    uint32x4_t bin1 = vmovl_u16(vget_high_u16(bytes16));

                    uint32x4_t thresh_v = vdupq_n_u32(thresh);
                    uint32x4_t nan_v = vdupq_n_u32(255);

                    uint32x4_t not_nan0 = vmvnq_u32(vceqq_u32(bin0, nan_v));
                    uint32x4_t not_nan1 = vmvnq_u32(vceqq_u32(bin1, nan_v));
                    uint32x4_t cmp0 = vcgtq_u32(bin0, thresh_v);
                    uint32x4_t cmp1 = vcgtq_u32(bin1, thresh_v);
                    uint32x4_t go_right0 = vandq_u32(not_nan0, cmp0);
                    uint32x4_t go_right1 = vandq_u32(not_nan1, cmp1);

                    indices0 = vshlq_n_u32(indices0, 1);
                    indices1 = vshlq_n_u32(indices1, 1);
                    uint32x4_t one = vdupq_n_u32(1);
                    indices0 = vorrq_u32(indices0, vandq_u32(go_right0, one));
                    indices1 = vorrq_u32(indices1, vandq_u32(go_right1, one));
                }

                // Gather leaves and accumulate
                alignas(16) uint32_t idx0[4], idx1[4];
                vst1q_u32(idx0, indices0);
                vst1q_u32(idx1, indices1);

                alignas(16) float leaves[8];
                leaves[0] = tree_leaves[idx0[0]];
                leaves[1] = tree_leaves[idx0[1]];
                leaves[2] = tree_leaves[idx0[2]];
                leaves[3] = tree_leaves[idx0[3]];
                leaves[4] = tree_leaves[idx1[0]];
                leaves[5] = tree_leaves[idx1[1]];
                leaves[6] = tree_leaves[idx1[2]];
                leaves[7] = tree_leaves[idx1[3]];

                sums0 = vmlaq_n_f32(sums0, vld1q_f32(leaves), weight);
                sums1 = vmlaq_n_f32(sums1, vld1q_f32(leaves + 4), weight);
            }
        }

        // Handle remaining trees (< 4)
        for (size_t t = n_tree_batches * tree_batch_size; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;
            Float weight = weights_[t];

            uint32x4_t indices0 = vdupq_n_u32(0);
            uint32x4_t indices1 = vdupq_n_u32(0);

            for (uint16_t d = 0; d < depth; ++d) {
                FeatureIndex feat = tree_features[d];
                BinIndex thresh = tree_thresholds[d];
                const BinIndex* feat_data = data + feat * n_samples + base;

                uint8x8_t bytes8 = vld1_u8(feat_data);
                uint16x8_t bytes16 = vmovl_u8(bytes8);
                uint32x4_t bin0 = vmovl_u16(vget_low_u16(bytes16));
                uint32x4_t bin1 = vmovl_u16(vget_high_u16(bytes16));

                uint32x4_t thresh_v = vdupq_n_u32(thresh);
                uint32x4_t nan_v = vdupq_n_u32(255);

                uint32x4_t not_nan0 = vmvnq_u32(vceqq_u32(bin0, nan_v));
                uint32x4_t not_nan1 = vmvnq_u32(vceqq_u32(bin1, nan_v));
                uint32x4_t cmp0 = vcgtq_u32(bin0, thresh_v);
                uint32x4_t cmp1 = vcgtq_u32(bin1, thresh_v);
                uint32x4_t go_right0 = vandq_u32(not_nan0, cmp0);
                uint32x4_t go_right1 = vandq_u32(not_nan1, cmp1);

                indices0 = vshlq_n_u32(indices0, 1);
                indices1 = vshlq_n_u32(indices1, 1);
                uint32x4_t one = vdupq_n_u32(1);
                indices0 = vorrq_u32(indices0, vandq_u32(go_right0, one));
                indices1 = vorrq_u32(indices1, vandq_u32(go_right1, one));
            }

            alignas(16) uint32_t idx0[4], idx1[4];
            vst1q_u32(idx0, indices0);
            vst1q_u32(idx1, indices1);

            alignas(16) float leaves[8];
            leaves[0] = tree_leaves[idx0[0]];
            leaves[1] = tree_leaves[idx0[1]];
            leaves[2] = tree_leaves[idx0[2]];
            leaves[3] = tree_leaves[idx0[3]];
            leaves[4] = tree_leaves[idx1[0]];
            leaves[5] = tree_leaves[idx1[1]];
            leaves[6] = tree_leaves[idx1[2]];
            leaves[7] = tree_leaves[idx1[3]];

            sums0 = vmlaq_n_f32(sums0, vld1q_f32(leaves), weight);
            sums1 = vmlaq_n_f32(sums1, vld1q_f32(leaves + 4), weight);
        }

        vst1q_f32(output + base, sums0);
        vst1q_f32(output + base + 4, sums1);
    }

    // Handle remaining samples with scalar code
    for (Index i = n_batch; i < n_samples; ++i) {
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;

            uint32_t leaf_idx = 0;
            for (uint16_t d = 0; d < depth; ++d) {
                FeatureIndex feat = tree_features[d];
                BinIndex bin = data[feat * n_samples + i];
                bool go_right = (bin != 255 && bin > tree_thresholds[d]);
                leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
            }
            sum += weights_[t] * tree_leaves[leaf_idx];
        }
        output[i] = sum;
    }

#else
    // Scalar fallback - optimized with loop unrolling
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index i = 0; i < n_samples; ++i) {
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const BinIndex* tree_thresholds = thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;

            uint32_t leaf_idx = 0;
            for (uint16_t d = 0; d < depth; ++d) {
                FeatureIndex feat = tree_features[d];
                BinIndex bin = data[feat * n_samples + i];
                bool go_right = (bin != 255 && bin > tree_thresholds[d]);
                leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
            }
            sum += weights_[t] * tree_leaves[leaf_idx];
        }
        output[i] = sum;
    }
#endif
}

} // namespace turbocat
