/**
 * TurboCat Fast Float Ensemble - Ultra-optimized inference without binning
 *
 * Key optimizations:
 * 1. Flat memory layout - all tree data in contiguous arrays
 * 2. SIMD vectorization - process 8 samples at once with AVX2
 * 3. Raw float thresholds - no binning required (like CatBoost)
 * 4. Column-major input transpose for cache-friendly SIMD access
 * 5. Cached tree data - no per-prediction rebuilding
 */

#pragma once

#include "turbocat/types.hpp"
#include <vector>
#include <cstring>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) || defined(TURBOCAT_AVX2)
#include <immintrin.h>
#define FAST_FLOAT_HAS_AVX2 1
#endif

#if defined(__ARM_NEON) || defined(TURBOCAT_NEON)
#include <arm_neon.h>
#define FAST_FLOAT_HAS_NEON 1
#endif

namespace turbocat {

// Forward declaration
class SymmetricEnsemble;

/**
 * FastFloatEnsemble - Flat representation for raw float prediction
 *
 * Memory layout for N trees with max depth D:
 * - features[N * D]       : feature index at each level of each tree
 * - float_thresholds[N * D]: raw float threshold at each level
 * - leaf_values[N * 2^D]  : all leaf values flattened
 * - weights[N]            : tree weights (learning_rate already applied)
 */
class FastFloatEnsemble {
public:
    FastFloatEnsemble() = default;

    // Build from symmetric ensemble
    void from_symmetric_ensemble(const SymmetricEnsemble& ensemble);

    // Ultra-fast batch prediction from row-major float data
    // Input: data[sample * n_features + feature]
    void predict_batch_row_major(
        const Float* data,
        Index n_samples,
        FeatureIndex n_features,
        Float* output,
        int n_threads = -1
    ) const;

    // Ultra-fast batch prediction with column-major transpose
    // Transposes row-major input internally for SIMD efficiency
    // This is the FASTEST path for raw float prediction
    void predict_batch_with_transpose(
        const Float* data,
        Index n_samples,
        FeatureIndex n_features,
        Float* output,
        int n_threads = -1
    ) const;

    // Prediction from pre-transposed column-major data
    // Input: data[feature * n_samples + feature_stride + sample]
    // This is for cases where data is already column-major
    void predict_batch_column_major(
        const Float* data,
        Index n_samples,
        FeatureIndex n_features,
        Float* output,
        int n_threads = -1
    ) const;

    size_t n_trees() const { return n_trees_; }
    uint16_t max_depth() const { return max_depth_; }
    bool empty() const { return n_trees_ == 0; }

private:
    size_t n_trees_ = 0;
    uint16_t max_depth_ = 0;
    uint32_t leaves_per_tree_ = 0;

    // Flat arrays for cache-efficient access (aligned for SIMD)
    alignas(64) std::vector<FeatureIndex> features_;       // [n_trees * max_depth]
    alignas(64) std::vector<Float> float_thresholds_;      // [n_trees * max_depth]
    alignas(64) std::vector<Float> leaf_values_;           // [n_trees * 2^max_depth]
    alignas(64) std::vector<Float> weights_;               // [n_trees]
    alignas(64) std::vector<uint16_t> depths_;             // [n_trees]

    // Thread-local transpose buffer for predict_batch_with_transpose
    mutable std::vector<Float> transpose_buffer_;

    // Scalar prediction for single sample
    inline Float predict_single(const Float* sample, size_t tree_idx) const;

    // Fast SIMD transpose from row-major to column-major
    void transpose_to_column_major(
        const Float* row_major,
        Float* col_major,
        Index n_samples,
        FeatureIndex n_features,
        int n_threads
    ) const;
};

// ============================================================================
// Inline implementations
// ============================================================================

inline Float FastFloatEnsemble::predict_single(const Float* sample, size_t tree_idx) const {
    uint16_t depth = depths_[tree_idx];
    const FeatureIndex* tree_features = features_.data() + tree_idx * max_depth_;
    const Float* tree_thresholds = float_thresholds_.data() + tree_idx * max_depth_;
    const Float* tree_leaves = leaf_values_.data() + tree_idx * leaves_per_tree_;

    uint32_t leaf_idx = 0;
    for (uint16_t d = 0; d < depth; ++d) {
        Float val = sample[tree_features[d]];
        // NaN goes left (same as binned prediction)
        bool go_right = !std::isnan(val) && val >= tree_thresholds[d];
        leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
    }

    return weights_[tree_idx] * tree_leaves[leaf_idx];
}

inline void FastFloatEnsemble::predict_batch_row_major(
    const Float* data,
    Index n_samples,
    FeatureIndex n_features,
    Float* output,
    int n_threads
) const {
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

#ifdef FAST_FLOAT_HAS_AVX2
    // AVX2 path - process 8 samples at a time
    Index n_simd = (n_samples / 8) * 8;

    // Process 4 trees at a time for better ILP
    const size_t tree_batch = 4;
    const size_t n_tree_batches = n_trees_ / tree_batch;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index base = 0; base < n_simd; base += 8) {
        __m256 sums = _mm256_setzero_ps();

        for (size_t tb = 0; tb < n_tree_batches; ++tb) {
            size_t t0 = tb * tree_batch;

            // Prefetch next tree batch
            if (tb + 1 < n_tree_batches) {
                size_t t_next = (tb + 1) * tree_batch;
                _mm_prefetch(reinterpret_cast<const char*>(leaf_values_.data() + t_next * leaves_per_tree_), _MM_HINT_T0);
            }

            alignas(32) uint32_t idx0[8], idx1[8], idx2[8], idx3[8];

            // Process 4 trees
            for (int ti = 0; ti < 4; ++ti) {
                size_t t = t0 + ti;
                uint16_t depth = depths_[t];
                const FeatureIndex* tree_features = features_.data() + t * max_depth_;
                const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;

                __m256i indices = _mm256_setzero_si256();

                for (uint16_t d = 0; d < depth; ++d) {
                    FeatureIndex feat = tree_features[d];
                    Float thresh = tree_thresholds[d];

                    // Load 8 feature values (strided access - the bottleneck)
                    alignas(32) float vals[8];
                    for (int j = 0; j < 8; ++j) {
                        vals[j] = data[(base + j) * n_features + feat];
                    }
                    __m256 v_vals = _mm256_load_ps(vals);

                    // NaN check
                    __m256 v_nan_mask = _mm256_cmp_ps(v_vals, v_vals, _CMP_UNORD_Q);

                    // Compare: val > threshold
                    __m256 v_thresh = _mm256_set1_ps(thresh);
                    __m256 v_cmp = _mm256_cmp_ps(v_vals, v_thresh, _CMP_GE_OQ);

                    // Go right if not NaN and > threshold
                    __m256 v_go_right = _mm256_andnot_ps(v_nan_mask, v_cmp);

                    // Update indices
                    indices = _mm256_slli_epi32(indices, 1);
                    __m256i v_bit = _mm256_and_si256(
                        _mm256_castps_si256(v_go_right),
                        _mm256_set1_epi32(1)
                    );
                    indices = _mm256_or_si256(indices, v_bit);
                }

                uint32_t* idx_out = (ti == 0) ? idx0 : (ti == 1) ? idx1 : (ti == 2) ? idx2 : idx3;
                _mm256_store_si256(reinterpret_cast<__m256i*>(idx_out), indices);
            }

            // Scalar gather (often faster than _mm256_i32gather_ps)
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

        // Remaining trees
        for (size_t t = n_tree_batches * tree_batch; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;
            Float weight = weights_[t];

            __m256i indices = _mm256_setzero_si256();

            for (uint16_t d = 0; d < depth; ++d) {
                FeatureIndex feat = tree_features[d];
                Float thresh = tree_thresholds[d];

                alignas(32) float vals[8];
                for (int j = 0; j < 8; ++j) {
                    vals[j] = data[(base + j) * n_features + feat];
                }
                __m256 v_vals = _mm256_load_ps(vals);
                __m256 v_nan_mask = _mm256_cmp_ps(v_vals, v_vals, _CMP_UNORD_Q);
                __m256 v_cmp = _mm256_cmp_ps(v_vals, _mm256_set1_ps(thresh), _CMP_GE_OQ);
                __m256 v_go_right = _mm256_andnot_ps(v_nan_mask, v_cmp);
                indices = _mm256_slli_epi32(indices, 1);
                indices = _mm256_or_si256(indices, _mm256_and_si256(_mm256_castps_si256(v_go_right), _mm256_set1_epi32(1)));
            }

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

    // Remaining samples
    for (Index i = n_simd; i < n_samples; ++i) {
        const Float* sample = data + i * n_features;
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            sum += predict_single(sample, t);
        }
        output[i] = sum;
    }

#elif defined(FAST_FLOAT_HAS_NEON)
    // NEON path - process 4 samples at a time
    Index n_simd = (n_samples / 4) * 4;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index base = 0; base < n_simd; base += 4) {
        float32x4_t sums = vdupq_n_f32(0.0f);

        for (size_t t = 0; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;
            Float weight = weights_[t];

            uint32x4_t indices = vdupq_n_u32(0);

            for (uint16_t d = 0; d < depth; ++d) {
                FeatureIndex feat = tree_features[d];
                Float thresh = tree_thresholds[d];

                alignas(16) float vals[4];
                for (int j = 0; j < 4; ++j) {
                    vals[j] = data[(base + j) * n_features + feat];
                }
                float32x4_t v_vals = vld1q_f32(vals);

                // NaN check (val != val for NaN)
                uint32x4_t nan_mask = vmvnq_u32(vceqq_f32(v_vals, v_vals));

                // Compare: val > threshold
                float32x4_t v_thresh = vdupq_n_f32(thresh);
                uint32x4_t cmp = vcgeq_f32(v_vals, v_thresh);

                // Go right if not NaN and > threshold
                uint32x4_t go_right = vbicq_u32(cmp, nan_mask);

                indices = vshlq_n_u32(indices, 1);
                indices = vorrq_u32(indices, vandq_u32(go_right, vdupq_n_u32(1)));
            }

            alignas(16) uint32_t idx_arr[4];
            vst1q_u32(idx_arr, indices);

            alignas(16) float leaves[4];
            for (int j = 0; j < 4; ++j) {
                leaves[j] = tree_leaves[idx_arr[j]];
            }
            sums = vmlaq_n_f32(sums, vld1q_f32(leaves), weight);
        }

        vst1q_f32(output + base, sums);
    }

    for (Index i = n_simd; i < n_samples; ++i) {
        const Float* sample = data + i * n_features;
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
        const Float* sample = data + i * n_features;
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            sum += predict_single(sample, t);
        }
        output[i] = sum;
    }
#endif
}

// Column-major prediction - data[feature * n_samples + sample]
// This enables loading 8 consecutive floats for 8 samples
inline void FastFloatEnsemble::predict_batch_column_major(
    const Float* data,
    Index n_samples,
    FeatureIndex n_features,
    Float* output,
    int n_threads
) const {
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

#ifdef FAST_FLOAT_HAS_AVX2
    Index n_simd = (n_samples / 8) * 8;
    const size_t tree_batch = 4;
    const size_t n_tree_batches = n_trees_ / tree_batch;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index base = 0; base < n_simd; base += 8) {
        __m256 sums = _mm256_setzero_ps();

        for (size_t tb = 0; tb < n_tree_batches; ++tb) {
            size_t t0 = tb * tree_batch;

            if (tb + 1 < n_tree_batches) {
                _mm_prefetch(reinterpret_cast<const char*>(leaf_values_.data() + (tb + 1) * tree_batch * leaves_per_tree_), _MM_HINT_T0);
            }

            alignas(32) uint32_t idx0[8], idx1[8], idx2[8], idx3[8];

            // Tree 0
            {
                size_t t = t0;
                uint16_t depth = depths_[t];
                const FeatureIndex* tree_features = features_.data() + t * max_depth_;
                const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;

                __m256i indices = _mm256_setzero_si256();
                for (uint16_t d = 0; d < depth; ++d) {
                    // FAST: Load 8 consecutive floats (column-major layout)
                    const Float* feat_data = data + tree_features[d] * n_samples + base;
                    __m256 v_vals = _mm256_loadu_ps(feat_data);

                    __m256 v_nan_mask = _mm256_cmp_ps(v_vals, v_vals, _CMP_UNORD_Q);
                    __m256 v_cmp = _mm256_cmp_ps(v_vals, _mm256_set1_ps(tree_thresholds[d]), _CMP_GE_OQ);
                    __m256 v_go_right = _mm256_andnot_ps(v_nan_mask, v_cmp);

                    indices = _mm256_slli_epi32(indices, 1);
                    indices = _mm256_or_si256(indices, _mm256_and_si256(_mm256_castps_si256(v_go_right), _mm256_set1_epi32(1)));
                }
                _mm256_store_si256(reinterpret_cast<__m256i*>(idx0), indices);
            }

            // Tree 1
            {
                size_t t = t0 + 1;
                uint16_t depth = depths_[t];
                const FeatureIndex* tree_features = features_.data() + t * max_depth_;
                const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;

                __m256i indices = _mm256_setzero_si256();
                for (uint16_t d = 0; d < depth; ++d) {
                    const Float* feat_data = data + tree_features[d] * n_samples + base;
                    __m256 v_vals = _mm256_loadu_ps(feat_data);

                    __m256 v_nan_mask = _mm256_cmp_ps(v_vals, v_vals, _CMP_UNORD_Q);
                    __m256 v_cmp = _mm256_cmp_ps(v_vals, _mm256_set1_ps(tree_thresholds[d]), _CMP_GE_OQ);
                    __m256 v_go_right = _mm256_andnot_ps(v_nan_mask, v_cmp);

                    indices = _mm256_slli_epi32(indices, 1);
                    indices = _mm256_or_si256(indices, _mm256_and_si256(_mm256_castps_si256(v_go_right), _mm256_set1_epi32(1)));
                }
                _mm256_store_si256(reinterpret_cast<__m256i*>(idx1), indices);
            }

            // Tree 2
            {
                size_t t = t0 + 2;
                uint16_t depth = depths_[t];
                const FeatureIndex* tree_features = features_.data() + t * max_depth_;
                const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;

                __m256i indices = _mm256_setzero_si256();
                for (uint16_t d = 0; d < depth; ++d) {
                    const Float* feat_data = data + tree_features[d] * n_samples + base;
                    __m256 v_vals = _mm256_loadu_ps(feat_data);

                    __m256 v_nan_mask = _mm256_cmp_ps(v_vals, v_vals, _CMP_UNORD_Q);
                    __m256 v_cmp = _mm256_cmp_ps(v_vals, _mm256_set1_ps(tree_thresholds[d]), _CMP_GE_OQ);
                    __m256 v_go_right = _mm256_andnot_ps(v_nan_mask, v_cmp);

                    indices = _mm256_slli_epi32(indices, 1);
                    indices = _mm256_or_si256(indices, _mm256_and_si256(_mm256_castps_si256(v_go_right), _mm256_set1_epi32(1)));
                }
                _mm256_store_si256(reinterpret_cast<__m256i*>(idx2), indices);
            }

            // Tree 3
            {
                size_t t = t0 + 3;
                uint16_t depth = depths_[t];
                const FeatureIndex* tree_features = features_.data() + t * max_depth_;
                const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;

                __m256i indices = _mm256_setzero_si256();
                for (uint16_t d = 0; d < depth; ++d) {
                    const Float* feat_data = data + tree_features[d] * n_samples + base;
                    __m256 v_vals = _mm256_loadu_ps(feat_data);

                    __m256 v_nan_mask = _mm256_cmp_ps(v_vals, v_vals, _CMP_UNORD_Q);
                    __m256 v_cmp = _mm256_cmp_ps(v_vals, _mm256_set1_ps(tree_thresholds[d]), _CMP_GE_OQ);
                    __m256 v_go_right = _mm256_andnot_ps(v_nan_mask, v_cmp);

                    indices = _mm256_slli_epi32(indices, 1);
                    indices = _mm256_or_si256(indices, _mm256_and_si256(_mm256_castps_si256(v_go_right), _mm256_set1_epi32(1)));
                }
                _mm256_store_si256(reinterpret_cast<__m256i*>(idx3), indices);
            }

            // Gather leaf values
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

        // Remaining trees
        for (size_t t = n_tree_batches * tree_batch; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;
            Float weight = weights_[t];

            __m256i indices = _mm256_setzero_si256();
            for (uint16_t d = 0; d < depth; ++d) {
                const Float* feat_data = data + tree_features[d] * n_samples + base;
                __m256 v_vals = _mm256_loadu_ps(feat_data);
                __m256 v_nan_mask = _mm256_cmp_ps(v_vals, v_vals, _CMP_UNORD_Q);
                __m256 v_cmp = _mm256_cmp_ps(v_vals, _mm256_set1_ps(tree_thresholds[d]), _CMP_GE_OQ);
                __m256 v_go_right = _mm256_andnot_ps(v_nan_mask, v_cmp);
                indices = _mm256_slli_epi32(indices, 1);
                indices = _mm256_or_si256(indices, _mm256_and_si256(_mm256_castps_si256(v_go_right), _mm256_set1_epi32(1)));
            }

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

    // Remaining samples
    for (Index i = n_simd; i < n_samples; ++i) {
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;

            uint32_t leaf_idx = 0;
            for (uint16_t d = 0; d < depth; ++d) {
                Float val = data[tree_features[d] * n_samples + i];
                bool go_right = !std::isnan(val) && val >= tree_thresholds[d];
                leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
            }
            sum += weights_[t] * tree_leaves[leaf_idx];
        }
        output[i] = sum;
    }

#elif defined(FAST_FLOAT_HAS_NEON)
    Index n_simd = (n_samples / 8) * 8;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index base = 0; base < n_simd; base += 8) {
        float32x4_t sums0 = vdupq_n_f32(0.0f);
        float32x4_t sums1 = vdupq_n_f32(0.0f);

        for (size_t t = 0; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;
            Float weight = weights_[t];

            uint32x4_t indices0 = vdupq_n_u32(0);
            uint32x4_t indices1 = vdupq_n_u32(0);

            for (uint16_t d = 0; d < depth; ++d) {
                const Float* feat_data = data + tree_features[d] * n_samples + base;
                float32x4_t v0 = vld1q_f32(feat_data);
                float32x4_t v1 = vld1q_f32(feat_data + 4);

                float32x4_t thresh = vdupq_n_f32(tree_thresholds[d]);

                uint32x4_t nan_mask0 = vmvnq_u32(vceqq_f32(v0, v0));
                uint32x4_t nan_mask1 = vmvnq_u32(vceqq_f32(v1, v1));
                uint32x4_t cmp0 = vcgeq_f32(v0, thresh);
                uint32x4_t cmp1 = vcgeq_f32(v1, thresh);
                uint32x4_t go_right0 = vbicq_u32(cmp0, nan_mask0);
                uint32x4_t go_right1 = vbicq_u32(cmp1, nan_mask1);

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

    for (Index i = n_simd; i < n_samples; ++i) {
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;

            uint32_t leaf_idx = 0;
            for (uint16_t d = 0; d < depth; ++d) {
                Float val = data[tree_features[d] * n_samples + i];
                bool go_right = !std::isnan(val) && val >= tree_thresholds[d];
                leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
            }
            sum += weights_[t] * tree_leaves[leaf_idx];
        }
        output[i] = sum;
    }

#else
    // Scalar fallback
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index i = 0; i < n_samples; ++i) {
        Float sum = 0.0f;
        for (size_t t = 0; t < n_trees_; ++t) {
            uint16_t depth = depths_[t];
            const FeatureIndex* tree_features = features_.data() + t * max_depth_;
            const Float* tree_thresholds = float_thresholds_.data() + t * max_depth_;
            const Float* tree_leaves = leaf_values_.data() + t * leaves_per_tree_;

            uint32_t leaf_idx = 0;
            for (uint16_t d = 0; d < depth; ++d) {
                Float val = data[tree_features[d] * n_samples + i];
                bool go_right = !std::isnan(val) && val >= tree_thresholds[d];
                leaf_idx = (leaf_idx << 1) | (go_right ? 1u : 0u);
            }
            sum += weights_[t] * tree_leaves[leaf_idx];
        }
        output[i] = sum;
    }
#endif
}

} // namespace turbocat
