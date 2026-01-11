/**
 * TurboCat Fast Float Ensemble Implementation
 */

#include "turbocat/fast_float_ensemble.hpp"
#include "turbocat/symmetric_tree.hpp"
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) || defined(TURBOCAT_AVX2)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(TURBOCAT_NEON)
#include <arm_neon.h>
#endif

namespace turbocat {

void FastFloatEnsemble::from_symmetric_ensemble(const SymmetricEnsemble& ensemble) {
    n_trees_ = ensemble.n_trees();
    if (n_trees_ == 0) return;

    // Find max depth
    max_depth_ = 0;
    for (size_t t = 0; t < n_trees_; ++t) {
        max_depth_ = std::max(max_depth_, ensemble.tree(t).depth());
    }

    if (max_depth_ == 0) {
        n_trees_ = 0;
        return;
    }

    leaves_per_tree_ = 1u << max_depth_;

    // Allocate flat arrays
    features_.resize(n_trees_ * max_depth_, 0);
    float_thresholds_.resize(n_trees_ * max_depth_, 0.0f);
    leaf_values_.resize(n_trees_ * leaves_per_tree_, 0.0f);
    weights_.resize(n_trees_);
    depths_.resize(n_trees_);

    // Copy data from each tree
    for (size_t t = 0; t < n_trees_; ++t) {
        const SymmetricTree& tree = ensemble.tree(t);
        uint16_t depth = tree.depth();
        depths_[t] = depth;
        weights_[t] = ensemble.tree_weight(t);

        // Copy splits with float thresholds
        const auto& splits = tree.splits();
        for (uint16_t d = 0; d < depth; ++d) {
            features_[t * max_depth_ + d] = splits[d].feature;
            float_thresholds_[t * max_depth_ + d] = splits[d].float_threshold;
        }

        // Copy leaf values
        const auto& leaves = tree.leaf_values();
        uint32_t n_leaves = 1u << depth;
        for (uint32_t i = 0; i < n_leaves; ++i) {
            leaf_values_[t * leaves_per_tree_ + i] = leaves[i];
        }
    }
}

void FastFloatEnsemble::transpose_to_column_major(
    const Float* row_major,
    Float* col_major,
    Index n_samples,
    FeatureIndex n_features,
    int n_threads
) const {
    // Fast SIMD transpose from row-major to column-major
    // row_major[sample * n_features + feature] -> col_major[feature * n_samples + sample]

#if defined(FAST_FLOAT_HAS_AVX2)
    // AVX2 optimized transpose using 8x8 blocks
    // Process 8 samples x 8 features at a time

    const Index n_sample_blocks = n_samples / 8;
    const FeatureIndex n_feature_blocks = n_features / 8;

    #pragma omp parallel for collapse(2) schedule(static) num_threads(n_threads)
    for (Index sb = 0; sb < n_sample_blocks; ++sb) {
        for (FeatureIndex fb = 0; fb < n_feature_blocks; ++fb) {
            Index sample_base = sb * 8;
            FeatureIndex feature_base = fb * 8;

            // Load 8x8 block from row-major
            __m256 r0 = _mm256_loadu_ps(&row_major[(sample_base + 0) * n_features + feature_base]);
            __m256 r1 = _mm256_loadu_ps(&row_major[(sample_base + 1) * n_features + feature_base]);
            __m256 r2 = _mm256_loadu_ps(&row_major[(sample_base + 2) * n_features + feature_base]);
            __m256 r3 = _mm256_loadu_ps(&row_major[(sample_base + 3) * n_features + feature_base]);
            __m256 r4 = _mm256_loadu_ps(&row_major[(sample_base + 4) * n_features + feature_base]);
            __m256 r5 = _mm256_loadu_ps(&row_major[(sample_base + 5) * n_features + feature_base]);
            __m256 r6 = _mm256_loadu_ps(&row_major[(sample_base + 6) * n_features + feature_base]);
            __m256 r7 = _mm256_loadu_ps(&row_major[(sample_base + 7) * n_features + feature_base]);

            // 8x8 transpose using AVX2 shuffles
            // First stage: 2x2 transposes within 128-bit lanes
            __m256 t0 = _mm256_unpacklo_ps(r0, r1);
            __m256 t1 = _mm256_unpackhi_ps(r0, r1);
            __m256 t2 = _mm256_unpacklo_ps(r2, r3);
            __m256 t3 = _mm256_unpackhi_ps(r2, r3);
            __m256 t4 = _mm256_unpacklo_ps(r4, r5);
            __m256 t5 = _mm256_unpackhi_ps(r4, r5);
            __m256 t6 = _mm256_unpacklo_ps(r6, r7);
            __m256 t7 = _mm256_unpackhi_ps(r6, r7);

            // Second stage: 4x4 transposes
            __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
            __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
            __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
            __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
            __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
            __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
            __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

            // Third stage: combine 128-bit halves
            __m256 c0 = _mm256_permute2f128_ps(u0, u4, 0x20);
            __m256 c1 = _mm256_permute2f128_ps(u1, u5, 0x20);
            __m256 c2 = _mm256_permute2f128_ps(u2, u6, 0x20);
            __m256 c3 = _mm256_permute2f128_ps(u3, u7, 0x20);
            __m256 c4 = _mm256_permute2f128_ps(u0, u4, 0x31);
            __m256 c5 = _mm256_permute2f128_ps(u1, u5, 0x31);
            __m256 c6 = _mm256_permute2f128_ps(u2, u6, 0x31);
            __m256 c7 = _mm256_permute2f128_ps(u3, u7, 0x31);

            // Store to column-major
            _mm256_storeu_ps(&col_major[(feature_base + 0) * n_samples + sample_base], c0);
            _mm256_storeu_ps(&col_major[(feature_base + 1) * n_samples + sample_base], c1);
            _mm256_storeu_ps(&col_major[(feature_base + 2) * n_samples + sample_base], c2);
            _mm256_storeu_ps(&col_major[(feature_base + 3) * n_samples + sample_base], c3);
            _mm256_storeu_ps(&col_major[(feature_base + 4) * n_samples + sample_base], c4);
            _mm256_storeu_ps(&col_major[(feature_base + 5) * n_samples + sample_base], c5);
            _mm256_storeu_ps(&col_major[(feature_base + 6) * n_samples + sample_base], c6);
            _mm256_storeu_ps(&col_major[(feature_base + 7) * n_samples + sample_base], c7);
        }
    }

    // Handle remaining samples (not multiple of 8)
    Index remaining_sample_start = n_sample_blocks * 8;
    if (remaining_sample_start < n_samples) {
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (FeatureIndex f = 0; f < n_features; ++f) {
            for (Index i = remaining_sample_start; i < n_samples; ++i) {
                col_major[f * n_samples + i] = row_major[i * n_features + f];
            }
        }
    }

    // Handle remaining features (not multiple of 8)
    FeatureIndex remaining_feature_start = n_feature_blocks * 8;
    if (remaining_feature_start < n_features) {
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (Index i = 0; i < remaining_sample_start; ++i) {
            for (FeatureIndex f = remaining_feature_start; f < n_features; ++f) {
                col_major[f * n_samples + i] = row_major[i * n_features + f];
            }
        }
    }

#elif defined(FAST_FLOAT_HAS_NEON)
    // NEON optimized transpose using 4x4 blocks
    const Index n_sample_blocks = n_samples / 4;
    const FeatureIndex n_feature_blocks = n_features / 4;

    #pragma omp parallel for collapse(2) schedule(static) num_threads(n_threads)
    for (Index sb = 0; sb < n_sample_blocks; ++sb) {
        for (FeatureIndex fb = 0; fb < n_feature_blocks; ++fb) {
            Index sample_base = sb * 4;
            FeatureIndex feature_base = fb * 4;

            // Load 4x4 block
            float32x4_t r0 = vld1q_f32(&row_major[(sample_base + 0) * n_features + feature_base]);
            float32x4_t r1 = vld1q_f32(&row_major[(sample_base + 1) * n_features + feature_base]);
            float32x4_t r2 = vld1q_f32(&row_major[(sample_base + 2) * n_features + feature_base]);
            float32x4_t r3 = vld1q_f32(&row_major[(sample_base + 3) * n_features + feature_base]);

            // 4x4 transpose
            float32x4x2_t t01 = vtrnq_f32(r0, r1);
            float32x4x2_t t23 = vtrnq_f32(r2, r3);

            float32x4_t c0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
            float32x4_t c1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
            float32x4_t c2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
            float32x4_t c3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));

            // Store
            vst1q_f32(&col_major[(feature_base + 0) * n_samples + sample_base], c0);
            vst1q_f32(&col_major[(feature_base + 1) * n_samples + sample_base], c1);
            vst1q_f32(&col_major[(feature_base + 2) * n_samples + sample_base], c2);
            vst1q_f32(&col_major[(feature_base + 3) * n_samples + sample_base], c3);
        }
    }

    // Handle remaining
    Index remaining_sample_start = n_sample_blocks * 4;
    FeatureIndex remaining_feature_start = n_feature_blocks * 4;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (FeatureIndex f = 0; f < n_features; ++f) {
        for (Index i = remaining_sample_start; i < n_samples; ++i) {
            col_major[f * n_samples + i] = row_major[i * n_features + f];
        }
    }

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (Index i = 0; i < remaining_sample_start; ++i) {
        for (FeatureIndex f = remaining_feature_start; f < n_features; ++f) {
            col_major[f * n_samples + i] = row_major[i * n_features + f];
        }
    }

#else
    // Scalar fallback - cache-blocked for better performance
    const Index block_size = 64;

    #pragma omp parallel for collapse(2) schedule(static) num_threads(n_threads)
    for (Index bi = 0; bi < n_samples; bi += block_size) {
        for (FeatureIndex bf = 0; bf < n_features; bf += block_size) {
            Index i_end = std::min(bi + block_size, n_samples);
            FeatureIndex f_end = std::min(static_cast<FeatureIndex>(bf + block_size), n_features);

            for (Index i = bi; i < i_end; ++i) {
                for (FeatureIndex f = bf; f < f_end; ++f) {
                    col_major[f * n_samples + i] = row_major[i * n_features + f];
                }
            }
        }
    }
#endif
}

void FastFloatEnsemble::predict_batch_with_transpose(
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

    // For small batches, use row-major directly (transpose overhead not worth it)
    const Index transpose_threshold = 256;
    if (n_samples < transpose_threshold) {
        predict_batch_row_major(data, n_samples, n_features, output, n_threads);
        return;
    }

    // Allocate transpose buffer (thread-safe via per-call allocation)
    // For very large batches, we could use thread-local storage
    std::vector<Float> col_major(static_cast<size_t>(n_samples) * n_features);

    // Transpose row-major to column-major
    transpose_to_column_major(data, col_major.data(), n_samples, n_features, n_threads);

    // Predict using column-major data
    predict_batch_column_major(col_major.data(), n_samples, n_features, output, n_threads);
}

} // namespace turbocat
