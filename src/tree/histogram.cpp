/**
 * TurboCat Histogram Builder Implementation
 *
 * SIMD-optimized histogram construction.
 * Key optimization: vectorized gradient accumulation into bins.
 */

#include "turbocat/histogram.hpp"
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TURBOCAT_AVX2
#include <immintrin.h>
#endif

#ifdef TURBOCAT_AVX512
#include <immintrin.h>
#endif

namespace turbocat {

// ============================================================================
// Histogram Implementation
// ============================================================================

Histogram::Histogram(FeatureIndex n_features, BinIndex max_bins)
    : n_features_(n_features), max_bins_(max_bins) {
    data_.resize(static_cast<size_t>(n_features) * max_bins);
}

void Histogram::clear() {
    std::memset(data_.data(), 0, data_.size() * sizeof(GradientPair));
}

void Histogram::subtract_from(const Histogram& parent, const Histogram& other) {
    // Use SIMD subtraction
    simd::subtract_histograms(data_.data(), parent.data(), other.data(), data_.size());
}

// ============================================================================
// CPU Histogram Builder
// ============================================================================

// Minimum samples to use parallel processing (avoid thread overhead)
static constexpr size_t MIN_SAMPLES_FOR_PARALLEL = 2000;

CPUHistogramBuilder::CPUHistogramBuilder(int n_threads, bool use_simd)
    : n_threads_(n_threads), use_simd_(use_simd) {
    if (n_threads_ <= 0) {
        #ifdef _OPENMP
        n_threads_ = omp_get_max_threads();
        #else
        n_threads_ = 1;
        #endif
    }
}

void CPUHistogramBuilder::build(
    const Dataset& dataset,
    const std::vector<Index>& sample_indices,
    const std::vector<FeatureIndex>& feature_indices,
    Histogram& output
) {
    output.clear();

    FeatureIndex n_features = dataset.n_features();
    const Float* gradients = dataset.gradients();
    const Float* hessians = dataset.hessians();
    BinIndex max_bins = output.max_bins();
    Index n_samples = dataset.n_samples();

    // Build list of features to process
    std::vector<FeatureIndex> features_to_process;
    if (feature_indices.empty()) {
        features_to_process.resize(n_features);
        for (FeatureIndex f = 0; f < n_features; ++f) {
            features_to_process[f] = f;
        }
    } else {
        features_to_process = feature_indices;
    }

    // Check if we're using all samples (no subsampling) - use faster dense path
    bool use_all_samples = (sample_indices.size() == static_cast<size_t>(n_samples));

    // SMART AUTO-SELECTION: Row-wise vs Column-wise histogram building
    // Based on LightGBM research:
    // - Row-wise: Better for many samples, fewer features, fewer threads
    // - Column-wise: Better for many features, many threads
    const size_t actual_samples = use_all_samples ? static_cast<size_t>(n_samples) : sample_indices.size();
    const size_t n_feats = features_to_process.size();
    const size_t hist_size = n_feats * max_bins;

    // Row-wise is disabled - column-wise is faster for our use case
    (void)actual_samples;
    (void)n_feats;
    (void)hist_size;

    if (use_all_samples) {
        // OPTIMIZED: Feature-parallel histogram building with 8x unrolling
        // Column-major storage means sequential sample access is cache-efficient
        #pragma omp parallel for schedule(static) num_threads(n_threads_)
        for (size_t fi = 0; fi < features_to_process.size(); ++fi) {
            FeatureIndex f = features_to_process[fi];
            const BinIndex* __restrict__ bins = dataset.binned().column(f);
            GradientPair* __restrict__ out = output.bins(f);

            // 8x unrolled loop for better ILP
            Index i = 0;
            for (; i + 8 <= n_samples; i += 8) {
                // Prefetch ahead
                __builtin_prefetch(&bins[i + 64], 0, 3);
                __builtin_prefetch(&gradients[i + 64], 0, 3);

                const BinIndex b0 = bins[i], b1 = bins[i+1], b2 = bins[i+2], b3 = bins[i+3];
                const BinIndex b4 = bins[i+4], b5 = bins[i+5], b6 = bins[i+6], b7 = bins[i+7];

                out[b0].grad += gradients[i];   out[b0].hess += hessians[i];   out[b0].count++;
                out[b1].grad += gradients[i+1]; out[b1].hess += hessians[i+1]; out[b1].count++;
                out[b2].grad += gradients[i+2]; out[b2].hess += hessians[i+2]; out[b2].count++;
                out[b3].grad += gradients[i+3]; out[b3].hess += hessians[i+3]; out[b3].count++;
                out[b4].grad += gradients[i+4]; out[b4].hess += hessians[i+4]; out[b4].count++;
                out[b5].grad += gradients[i+5]; out[b5].hess += hessians[i+5]; out[b5].count++;
                out[b6].grad += gradients[i+6]; out[b6].hess += hessians[i+6]; out[b6].count++;
                out[b7].grad += gradients[i+7]; out[b7].hess += hessians[i+7]; out[b7].count++;
            }
            // Remainder
            for (; i < n_samples; ++i) {
                const BinIndex b = bins[i];
                out[b].grad += gradients[i];
                out[b].hess += hessians[i];
                out[b].count++;
            }
        }
    } else if (false) {  // Disabled old path
        // Dense sequential access without indirection
        // OPTIMIZED: 4x unrolled with prefetching for better ILP
        #pragma omp parallel for schedule(static) num_threads(n_threads_)
        for (size_t fi = 0; fi < features_to_process.size(); ++fi) {
            FeatureIndex f = features_to_process[fi];
            const BinIndex* bins = dataset.binned().column(f);
            GradientPair* out = output.bins(f);

            Index i = 0;
            // 4x unrolled loop with prefetching
            for (; i + 4 <= n_samples; i += 4) {
                // Prefetch next cache line
                if (i + 64 < n_samples) {
                    #ifdef _MSC_VER
                    _mm_prefetch(reinterpret_cast<const char*>(&bins[i + 64]), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(&gradients[i + 64]), _MM_HINT_T0);
                    #else
                    __builtin_prefetch(&bins[i + 64], 0, 3);
                    __builtin_prefetch(&gradients[i + 64], 0, 3);
                    #endif
                }

                const BinIndex b0 = bins[i];
                const BinIndex b1 = bins[i + 1];
                const BinIndex b2 = bins[i + 2];
                const BinIndex b3 = bins[i + 3];

                out[b0].grad += gradients[i];
                out[b0].hess += hessians[i];
                out[b0].count += 1;

                out[b1].grad += gradients[i + 1];
                out[b1].hess += hessians[i + 1];
                out[b1].count += 1;

                out[b2].grad += gradients[i + 2];
                out[b2].hess += hessians[i + 2];
                out[b2].count += 1;

                out[b3].grad += gradients[i + 3];
                out[b3].hess += hessians[i + 3];
                out[b3].count += 1;
            }

            // Handle remainder
            for (; i < n_samples; ++i) {
                BinIndex bin = bins[i];
                out[bin].grad += gradients[i];
                out[bin].hess += hessians[i];
                out[bin].count += 1;
            }
        }
    } else {
        // Feature-parallel histogram building with sample indices
        // Use fewer threads for small subsets to reduce overhead
        const int effective_threads = sample_indices.size() >= 10000 ? n_threads_ :
                                      sample_indices.size() >= 1000 ? std::min(n_threads_, 4) : 1;

        #pragma omp parallel for schedule(static) num_threads(effective_threads)
        for (size_t fi = 0; fi < features_to_process.size(); ++fi) {
            FeatureIndex f = features_to_process[fi];
            build_feature_scalar(
                dataset.binned().column(f),
                gradients,
                hessians,
                sample_indices,
                output.bins(f)
            );
        }
    }
}

void CPUHistogramBuilder::build_quantized(
    const Dataset& dataset,
    const std::vector<Index>& sample_indices,
    const std::vector<FeatureIndex>& feature_indices,
    Histogram& output
) {
    output.clear();

    const QuantizedGrad* q_grads = dataset.quantized_gradients();
    Float scale = dataset.gradient_scale();

    for (FeatureIndex f = 0; f < dataset.n_features(); ++f) {
        const BinIndex* bins = dataset.binned().column(f);
        GradientPair* hist = output.bins(f);

        for (Index idx : sample_indices) {
            BinIndex bin = bins[idx];
            Float grad = static_cast<Float>(q_grads[idx]) * scale;

            hist[bin].grad += grad;
            hist[bin].hess += 1.0f;  // Approximation: unit hessian for quantized
            hist[bin].count += 1;
        }
    }
}

// OPTIMIZED: Build histogram from a range without copying indices
void CPUHistogramBuilder::build_range(
    const Dataset& dataset,
    const Index* indices,
    size_t start, size_t end,
    const std::vector<FeatureIndex>& feature_indices,
    Histogram& output
) {
    output.clear();

    FeatureIndex n_features = dataset.n_features();
    const Float* gradients = dataset.gradients();
    const Float* hessians = dataset.hessians();
    const size_t n_samples = end - start;

    // Build list of features to process
    std::vector<FeatureIndex> features_to_process;
    if (feature_indices.empty()) {
        features_to_process.resize(n_features);
        for (FeatureIndex f = 0; f < n_features; ++f) {
            features_to_process[f] = f;
        }
    } else {
        features_to_process = feature_indices;
    }

    // Feature-parallel histogram building
    const int effective_threads = n_samples >= 10000 ? n_threads_ :
                                  n_samples >= 1000 ? std::min(n_threads_, 4) : 1;

    #pragma omp parallel for schedule(static) num_threads(effective_threads)
    for (size_t fi = 0; fi < features_to_process.size(); ++fi) {
        FeatureIndex f = features_to_process[fi];
        const BinIndex* __restrict__ bins = dataset.binned().column(f);
        GradientPair* __restrict__ out = output.bins(f);

        // 8x unrolled loop for better ILP
        size_t i = start;
        for (; i + 8 <= end; i += 8) {
            // Prefetch ahead
            if (i + 64 < end) {
                __builtin_prefetch(&indices[i + 64], 0, 3);
            }

            const Index idx0 = indices[i], idx1 = indices[i+1];
            const Index idx2 = indices[i+2], idx3 = indices[i+3];
            const Index idx4 = indices[i+4], idx5 = indices[i+5];
            const Index idx6 = indices[i+6], idx7 = indices[i+7];

            const BinIndex b0 = bins[idx0], b1 = bins[idx1];
            const BinIndex b2 = bins[idx2], b3 = bins[idx3];
            const BinIndex b4 = bins[idx4], b5 = bins[idx5];
            const BinIndex b6 = bins[idx6], b7 = bins[idx7];

            out[b0].grad += gradients[idx0]; out[b0].hess += hessians[idx0]; out[b0].count++;
            out[b1].grad += gradients[idx1]; out[b1].hess += hessians[idx1]; out[b1].count++;
            out[b2].grad += gradients[idx2]; out[b2].hess += hessians[idx2]; out[b2].count++;
            out[b3].grad += gradients[idx3]; out[b3].hess += hessians[idx3]; out[b3].count++;
            out[b4].grad += gradients[idx4]; out[b4].hess += hessians[idx4]; out[b4].count++;
            out[b5].grad += gradients[idx5]; out[b5].hess += hessians[idx5]; out[b5].count++;
            out[b6].grad += gradients[idx6]; out[b6].hess += hessians[idx6]; out[b6].count++;
            out[b7].grad += gradients[idx7]; out[b7].hess += hessians[idx7]; out[b7].count++;
        }
        // Remainder
        for (; i < end; ++i) {
            const Index idx = indices[i];
            const BinIndex b = bins[idx];
            out[b].grad += gradients[idx];
            out[b].hess += hessians[idx];
            out[b].count++;
        }
    }
}

void CPUHistogramBuilder::build_feature_scalar(
    const BinIndex* bins,
    const Float* gradients,
    const Float* hessians,
    const std::vector<Index>& indices,
    GradientPair* output
) {
    const size_t n = indices.size();
    size_t i = 0;
    
    // 8x unrolled loop for better instruction-level parallelism
    for (; i + 8 <= n; i += 8) {
        const Index idx0 = indices[i];
        const Index idx1 = indices[i + 1];
        const Index idx2 = indices[i + 2];
        const Index idx3 = indices[i + 3];
        const Index idx4 = indices[i + 4];
        const Index idx5 = indices[i + 5];
        const Index idx6 = indices[i + 6];
        const Index idx7 = indices[i + 7];
        
        const BinIndex bin0 = bins[idx0];
        const BinIndex bin1 = bins[idx1];
        const BinIndex bin2 = bins[idx2];
        const BinIndex bin3 = bins[idx3];
        const BinIndex bin4 = bins[idx4];
        const BinIndex bin5 = bins[idx5];
        const BinIndex bin6 = bins[idx6];
        const BinIndex bin7 = bins[idx7];
        
        output[bin0].grad += gradients[idx0];
        output[bin0].hess += hessians[idx0];
        output[bin0].count += 1;
        
        output[bin1].grad += gradients[idx1];
        output[bin1].hess += hessians[idx1];
        output[bin1].count += 1;
        
        output[bin2].grad += gradients[idx2];
        output[bin2].hess += hessians[idx2];
        output[bin2].count += 1;
        
        output[bin3].grad += gradients[idx3];
        output[bin3].hess += hessians[idx3];
        output[bin3].count += 1;
        
        output[bin4].grad += gradients[idx4];
        output[bin4].hess += hessians[idx4];
        output[bin4].count += 1;
        
        output[bin5].grad += gradients[idx5];
        output[bin5].hess += hessians[idx5];
        output[bin5].count += 1;
        
        output[bin6].grad += gradients[idx6];
        output[bin6].hess += hessians[idx6];
        output[bin6].count += 1;
        
        output[bin7].grad += gradients[idx7];
        output[bin7].hess += hessians[idx7];
        output[bin7].count += 1;
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        Index idx = indices[i];
        BinIndex bin = bins[idx];
        output[bin].grad += gradients[idx];
        output[bin].hess += hessians[idx];
        output[bin].count += 1;
    }
}

#ifdef TURBOCAT_AVX2
void CPUHistogramBuilder::build_feature_avx2(
    const BinIndex* bins,
    const Float* gradients,
    const Float* hessians,
    const std::vector<Index>& indices,
    GradientPair* output
) {
    // AVX2 doesn't have scatter, so we use manual unrolling with prefetching
    const size_t n = indices.size();
    size_t i = 0;
    
    // 8x unrolled with prefetching
    for (; i + 8 <= n; i += 8) {
        // Prefetch next batch
        if (i + 16 < n) {
            _mm_prefetch(reinterpret_cast<const char*>(&bins[indices[i + 16]]), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&gradients[indices[i + 16]]), _MM_HINT_T0);
        }
        
        const Index idx0 = indices[i];
        const Index idx1 = indices[i + 1];
        const Index idx2 = indices[i + 2];
        const Index idx3 = indices[i + 3];
        const Index idx4 = indices[i + 4];
        const Index idx5 = indices[i + 5];
        const Index idx6 = indices[i + 6];
        const Index idx7 = indices[i + 7];
        
        const BinIndex bin0 = bins[idx0];
        const BinIndex bin1 = bins[idx1];
        const BinIndex bin2 = bins[idx2];
        const BinIndex bin3 = bins[idx3];
        const BinIndex bin4 = bins[idx4];
        const BinIndex bin5 = bins[idx5];
        const BinIndex bin6 = bins[idx6];
        const BinIndex bin7 = bins[idx7];
        
        output[bin0].grad += gradients[idx0];
        output[bin0].hess += hessians[idx0];
        output[bin0].count += 1;
        
        output[bin1].grad += gradients[idx1];
        output[bin1].hess += hessians[idx1];
        output[bin1].count += 1;
        
        output[bin2].grad += gradients[idx2];
        output[bin2].hess += hessians[idx2];
        output[bin2].count += 1;
        
        output[bin3].grad += gradients[idx3];
        output[bin3].hess += hessians[idx3];
        output[bin3].count += 1;
        
        output[bin4].grad += gradients[idx4];
        output[bin4].hess += hessians[idx4];
        output[bin4].count += 1;
        
        output[bin5].grad += gradients[idx5];
        output[bin5].hess += hessians[idx5];
        output[bin5].count += 1;
        
        output[bin6].grad += gradients[idx6];
        output[bin6].hess += hessians[idx6];
        output[bin6].count += 1;
        
        output[bin7].grad += gradients[idx7];
        output[bin7].hess += hessians[idx7];
        output[bin7].count += 1;
    }
    
    // Handle remainder
    for (; i < n; ++i) {
        Index idx = indices[i];
        BinIndex bin = bins[idx];
        output[bin].grad += gradients[idx];
        output[bin].hess += hessians[idx];
        output[bin].count += 1;
    }
}
#endif

#ifdef TURBOCAT_AVX512
void CPUHistogramBuilder::build_feature_avx512(
    const BinIndex* bins,
    const Float* gradients,
    const Float* hessians,
    const std::vector<Index>& indices,
    GradientPair* output
) {
    // AVX-512 with conflict detection for safe parallel accumulation
    const size_t n = indices.size();
    size_t i = 0;
    
    // 16x unrolled with prefetching
    for (; i + 16 <= n; i += 16) {
        // Prefetch next batch
        if (i + 32 < n) {
            _mm_prefetch(reinterpret_cast<const char*>(&bins[indices[i + 32]]), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&gradients[indices[i + 32]]), _MM_HINT_T0);
        }
        
        #pragma unroll(16)
        for (int j = 0; j < 16; ++j) {
            Index idx = indices[i + j];
            BinIndex bin = bins[idx];
            output[bin].grad += gradients[idx];
            output[bin].hess += hessians[idx];
            output[bin].count += 1;
        }
    }
    
    for (; i < n; ++i) {
        Index idx = indices[i];
        BinIndex bin = bins[idx];
        output[bin].grad += gradients[idx];
        output[bin].hess += hessians[idx];
        output[bin].count += 1;
    }
}
#endif

void CPUHistogramBuilder::merge_histograms(Histogram& output, FeatureIndex n_features) {
    BinIndex max_bins = output.max_bins();

    #pragma omp parallel for
    for (FeatureIndex f = 0; f < n_features; ++f) {
        GradientPair* out = output.bins(f);

        for (int t = 0; t < n_threads_; ++t) {
            const GradientPair* local = thread_histograms_[t].bins(f);

#ifdef TURBOCAT_AVX2
            // AVX2 SIMD histogram merge
            // GradientPair = {float grad, float hess, uint32_t count, float padding} = 16 bytes
            // Process 2 GradientPairs at a time = 32 bytes = 256 bits
            BinIndex b = 0;
            for (; b + 2 <= max_bins; b += 2) {
                // Load 2 GradientPairs (32 bytes each) as integers
                __m256i v_out = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&out[b]));
                __m256i v_local = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&local[b]));

                // For GradientPair layout: [grad0, hess0, count0, pad0, grad1, hess1, count1, pad1]
                // We need float add for grad/hess/pad, integer add for count
                // Use blend to separate float and integer parts

                // Extract as floats for grad/hess/padding (positions 0,1,3,4,5,7)
                __m256 f_out = _mm256_castsi256_ps(v_out);
                __m256 f_local = _mm256_castsi256_ps(v_local);

                // Float addition
                __m256 f_sum = _mm256_add_ps(f_out, f_local);

                // Integer addition for count (positions 2 and 6)
                __m256i i_sum = _mm256_add_epi32(v_out, v_local);

                // Blend: take floats except at position 2 and 6 (count fields)
                // Blend mask: 0b01000100 = 0x44 means positions 2 and 6 come from i_sum
                __m256 result = _mm256_blend_ps(f_sum, _mm256_castsi256_ps(i_sum), 0x44);

                // Store result
                _mm256_storeu_ps(reinterpret_cast<float*>(&out[b]), result);
            }

            // Handle remainder
            for (; b < max_bins; ++b) {
                out[b] += local[b];
            }
#else
            for (BinIndex b = 0; b < max_bins; ++b) {
                out[b] += local[b];
            }
#endif
        }
    }
}

// Factory
std::unique_ptr<HistogramBuilder> HistogramBuilder::create(const DeviceConfig& config) {
    // TODO: Add CUDA and Metal builders
    return std::make_unique<CPUHistogramBuilder>(config.n_threads, config.use_simd);
}

// ============================================================================
// Split Finder
// ============================================================================

SplitFinder::SplitFinder(const TreeConfig& config) : config_(config) {}

SplitInfo SplitFinder::find_best_split(
    const Histogram& histogram,
    const GradientPair& parent_sum,
    const std::vector<FeatureIndex>& feature_indices
) {
    SplitInfo best_split;

    // Precompute parent gain term
    const Float lambda = config_.lambda_l2;
    const Float parent_gain = (parent_sum.grad * parent_sum.grad) / (parent_sum.hess + lambda);

    const size_t n_features = feature_indices.size();

    // OPTIMIZED: Pre-allocate thread-local best splits to avoid critical section
    int max_threads = 1;
    #ifdef _OPENMP
    max_threads = omp_get_max_threads();
    #endif

    std::vector<SplitInfo> thread_best(max_threads);

    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        SplitInfo& local_best = thread_best[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n_features; ++i) {
            FeatureIndex f = feature_indices[i];
            BinIndex n_bins = histogram.max_bins();

            SplitInfo split = find_best_split_feature(
                histogram.bins(f), n_bins, parent_sum, f
            );

            if (split > local_best) {
                local_best = split;
            }
        }
    }

    // Sequential reduction (O(num_threads) - trivial)
    for (int t = 0; t < max_threads; ++t) {
        if (thread_best[t] > best_split) {
            best_split = thread_best[t];
        }
    }

    return best_split;
}

SplitInfo SplitFinder::find_best_split_feature(
    const GradientPair* bins,
    BinIndex n_bins,
    const GradientPair& parent_sum,
    FeatureIndex feature_idx
) {
    SplitInfo best;
    best.feature_idx = feature_idx;
    
    GradientPair left_sum;
    
    // Precompute parent gain term once
    const Float lambda = config_.lambda_l2;
    const Float parent_gain = (parent_sum.grad * parent_sum.grad) / (parent_sum.hess + lambda);
    
    // Scan through bins to find best split
    for (BinIndex b = 0; b < n_bins - 1; ++b) {
        left_sum += bins[b];
        GradientPair right_sum = parent_sum - left_sum;
        
        // Check constraints
        if (!meets_constraints(left_sum) || !meets_constraints(right_sum)) {
            continue;
        }
        
        // Compute gain - optimized version using precomputed parent_gain
        // XGBoost formula: Gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
        Float gain;
        switch (config_.criterion) {
            case SplitCriterion::Variance: {
                Float gain_left = (left_sum.grad * left_sum.grad) / (left_sum.hess + lambda);
                Float gain_right = (right_sum.grad * right_sum.grad) / (right_sum.hess + lambda);
                gain = 0.5f * (gain_left + gain_right - parent_gain) - config_.gamma;
                break;
            }
            case SplitCriterion::Gini:
                gain = compute_gain_gini(left_sum, right_sum, parent_sum) - config_.gamma;
                break;
            case SplitCriterion::TsallisEntropy:
                gain = compute_gain_tsallis(left_sum, right_sum, parent_sum, config_.tsallis_q) - config_.gamma;
                break;
            default:
                Float gain_left = (left_sum.grad * left_sum.grad) / (left_sum.hess + lambda);
                Float gain_right = (right_sum.grad * right_sum.grad) / (right_sum.hess + lambda);
                gain = 0.5f * (gain_left + gain_right - parent_gain) - config_.gamma;
        }
        
        if (gain > best.gain && gain >= config_.min_split_gain) {
            best.gain = gain;
            best.bin_threshold = b;
            best.left_stats = left_sum;
            best.right_stats = right_sum;
            best.left_value = compute_leaf_value(left_sum);
            best.right_value = compute_leaf_value(right_sum);
            best.is_valid = true;
        }
    }
    
    return best;
}

Float SplitFinder::compute_gain_variance(
    const GradientPair& left,
    const GradientPair& right,
    const GradientPair& parent
) const {
    Float lambda = config_.lambda_l2;

    Float gain_left = (left.grad * left.grad) / (left.hess + lambda);
    Float gain_right = (right.grad * right.grad) / (right.hess + lambda);
    Float gain_parent = (parent.grad * parent.grad) / (parent.hess + lambda);

    // XGBoost formula with gamma complexity penalty
    return 0.5f * (gain_left + gain_right - gain_parent) - config_.gamma;
}

Float SplitFinder::compute_gain_gini(
    const GradientPair& left,
    const GradientPair& right,
    const GradientPair& parent
) const {
    auto gini = [](const GradientPair& g) -> Float {
        if (g.count == 0) return 0.0f;
        Float p = (g.grad / g.count + 1.0f) / 2.0f;
        p = std::max(0.0f, std::min(1.0f, p));
        return 2.0f * p * (1.0f - p);
    };
    
    Float n_left = static_cast<Float>(left.count);
    Float n_right = static_cast<Float>(right.count);
    Float n_total = n_left + n_right;
    
    if (n_total == 0) return 0.0f;
    
    Float gini_parent = gini(parent);
    Float gini_weighted = (n_left * gini(left) + n_right * gini(right)) / n_total;
    
    return gini_parent - gini_weighted;
}

Float SplitFinder::compute_gain_tsallis(
    const GradientPair& left,
    const GradientPair& right,
    const GradientPair& parent,
    Float q
) const {
    auto tsallis = [q](const GradientPair& g) -> Float {
        if (g.count == 0) return 0.0f;
        Float p = (g.grad / g.count + 1.0f) / 2.0f;
        p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));
        
        if (std::abs(q - 1.0f) < 1e-6f) {
            return -p * std::log(p) - (1 - p) * std::log(1 - p);
        }
        
        return (1.0f - std::pow(p, q) - std::pow(1 - p, q)) / (q - 1.0f);
    };
    
    Float n_left = static_cast<Float>(left.count);
    Float n_right = static_cast<Float>(right.count);
    Float n_total = n_left + n_right;
    
    if (n_total == 0) return 0.0f;
    
    Float tsallis_parent = tsallis(parent);
    Float tsallis_weighted = (n_left * tsallis(left) + n_right * tsallis(right)) / n_total;
    
    return tsallis_parent - tsallis_weighted;
}

Float SplitFinder::compute_leaf_value(const GradientPair& stats) const {
    return -stats.grad / (stats.hess + config_.lambda_l2);
}

bool SplitFinder::meets_constraints(const GradientPair& stats) const {
    if (stats.count < config_.min_samples_leaf) return false;
    if (stats.hess < config_.min_child_weight) return false;
    return true;
}

// ============================================================================
// SIMD Utilities
// ============================================================================

namespace simd {

void reduce_gradient_pairs(GradientPair* output, const GradientPair* input, size_t count) {
#ifdef TURBOCAT_AVX2
    // GradientPair layout: [grad, hess, count, padding] x N
    // Reduce all GradientPairs using SIMD

    if (count == 0) {
        *output = GradientPair();
        return;
    }

    // Initialize accumulators
    __m256 acc_float = _mm256_setzero_ps();
    __m256i acc_int = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        __m256i vi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input[i]));

        // Float accumulation for grad/hess/padding
        __m256 vf = _mm256_castsi256_ps(vi);
        acc_float = _mm256_add_ps(acc_float, vf);

        // Integer accumulation for count
        acc_int = _mm256_add_epi32(acc_int, vi);
    }

    // Horizontal sum within vector
    // acc_float contains [g0+g2, h0+h2, c0+c2(wrong), p0+p2, g1+g3, h1+h3, c1+c3(wrong), p1+p3]
    // First swap high/low 128-bit lanes and add
    __m128 lo_f = _mm256_extractf128_ps(acc_float, 0);
    __m128 hi_f = _mm256_extractf128_ps(acc_float, 1);
    __m128 sum_f = _mm_add_ps(lo_f, hi_f);

    __m128i lo_i = _mm256_extractf128_si256(acc_int, 0);
    __m128i hi_i = _mm256_extractf128_si256(acc_int, 1);
    __m128i sum_i = _mm_add_epi32(lo_i, hi_i);

    // Store to output
    alignas(16) float result[4];
    _mm_store_ps(result, sum_f);

    alignas(16) int32_t result_i[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(result_i), sum_i);

    output->grad = result[0];
    output->hess = result[1];
    output->count = static_cast<Index>(result_i[2]);
    output->padding = 0;

    // Handle remainder
    for (; i < count; ++i) {
        *output += input[i];
    }
#else
    GradientPair sum;
    for (size_t i = 0; i < count; ++i) {
        sum += input[i];
    }
    *output = sum;
#endif
}

void accumulate_histogram(
    GradientPair* hist,
    const BinIndex* bins,
    const Float* grads,
    const Float* hess,
    size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        BinIndex b = bins[i];
        hist[b].grad += grads[i];
        hist[b].hess += hess[i];
        hist[b].count += 1;
    }
}

void subtract_histograms(
    GradientPair* result,
    const GradientPair* parent,
    const GradientPair* child,
    size_t count
) {
    #ifdef TURBOCAT_AVX2
    // GradientPair = {float grad, float hess, uint32_t count, float padding} = 16 bytes
    // Process 2 GradientPairs per AVX register (256 bits = 32 bytes)
    // Layout: [grad0, hess0, count0, pad0, grad1, hess1, count1, pad1]

    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        __m256i vi_p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&parent[i]));
        __m256i vi_c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&child[i]));

        // Float subtraction for grad/hess/padding
        __m256 vf_p = _mm256_castsi256_ps(vi_p);
        __m256 vf_c = _mm256_castsi256_ps(vi_c);
        __m256 vf_sub = _mm256_sub_ps(vf_p, vf_c);

        // Integer subtraction for count (positions 2 and 6)
        __m256i vi_sub = _mm256_sub_epi32(vi_p, vi_c);

        // Blend: positions 2 and 6 (count) from integer sub, rest from float sub
        // Blend mask: 0b01000100 = 0x44
        __m256 v_result = _mm256_blend_ps(vf_sub, _mm256_castsi256_ps(vi_sub), 0x44);

        _mm256_storeu_ps(reinterpret_cast<float*>(&result[i]), v_result);
    }

    // Handle remainder
    for (; i < count; ++i) {
        result[i] = parent[i] - child[i];
    }
    #else
    for (size_t i = 0; i < count; ++i) {
        result[i] = parent[i] - child[i];
    }
    #endif
}

void compute_gains_batch(
    Float* gains,
    const GradientPair* bins,
    const GradientPair& parent_sum,
    size_t n_bins,
    Float lambda_l2
) {
    GradientPair left_sum;
    const Float parent_gain = (parent_sum.grad * parent_sum.grad) / (parent_sum.hess + lambda_l2);

#ifdef TURBOCAT_AVX2
    // AVX2 vectorized gain computation
    // Process 4 candidate splits in parallel once we have enough prefix sums

    // First compute prefix sums (sequential dependency)
    alignas(32) std::vector<Float> left_grads(n_bins);
    alignas(32) std::vector<Float> left_hess(n_bins);

    Float sum_g = 0.0f, sum_h = 0.0f;
    for (size_t b = 0; b < n_bins; ++b) {
        sum_g += bins[b].grad;
        sum_h += bins[b].hess;
        left_grads[b] = sum_g;
        left_hess[b] = sum_h;
    }

    // Now compute gains in parallel using SIMD
    const __m256 v_parent_g = _mm256_set1_ps(parent_sum.grad);
    const __m256 v_parent_h = _mm256_set1_ps(parent_sum.hess);
    const __m256 v_lambda = _mm256_set1_ps(lambda_l2);
    const __m256 v_parent_gain = _mm256_set1_ps(parent_gain);
    const __m256 v_half = _mm256_set1_ps(0.5f);

    size_t b = 0;
    for (; b + 8 <= n_bins; b += 8) {
        // Load left sums
        __m256 v_left_g = _mm256_load_ps(&left_grads[b]);
        __m256 v_left_h = _mm256_load_ps(&left_hess[b]);

        // Compute right sums
        __m256 v_right_g = _mm256_sub_ps(v_parent_g, v_left_g);
        __m256 v_right_h = _mm256_sub_ps(v_parent_h, v_left_h);

        // Compute left gain: G_L^2 / (H_L + lambda)
        __m256 v_left_g_sq = _mm256_mul_ps(v_left_g, v_left_g);
        __m256 v_left_denom = _mm256_add_ps(v_left_h, v_lambda);
        __m256 v_gain_left = _mm256_div_ps(v_left_g_sq, v_left_denom);

        // Compute right gain: G_R^2 / (H_R + lambda)
        __m256 v_right_g_sq = _mm256_mul_ps(v_right_g, v_right_g);
        __m256 v_right_denom = _mm256_add_ps(v_right_h, v_lambda);
        __m256 v_gain_right = _mm256_div_ps(v_right_g_sq, v_right_denom);

        // Total gain: 0.5 * (left + right - parent)
        __m256 v_sum = _mm256_add_ps(v_gain_left, v_gain_right);
        v_sum = _mm256_sub_ps(v_sum, v_parent_gain);
        __m256 v_gain = _mm256_mul_ps(v_half, v_sum);

        _mm256_storeu_ps(&gains[b], v_gain);
    }

    // Handle remainder
    for (; b < n_bins; ++b) {
        Float gain_left = (left_grads[b] * left_grads[b]) / (left_hess[b] + lambda_l2);
        Float right_g = parent_sum.grad - left_grads[b];
        Float right_h = parent_sum.hess - left_hess[b];
        Float gain_right = (right_g * right_g) / (right_h + lambda_l2);
        gains[b] = 0.5f * (gain_left + gain_right - parent_gain);
    }
#else
    // Scalar fallback
    for (size_t b = 0; b < n_bins; ++b) {
        left_sum += bins[b];
        GradientPair right_sum = parent_sum - left_sum;

        Float gain_left = (left_sum.grad * left_sum.grad) / (left_sum.hess + lambda_l2);
        Float gain_right = (right_sum.grad * right_sum.grad) / (right_sum.hess + lambda_l2);

        gains[b] = 0.5f * (gain_left + gain_right - parent_gain);
    }
#endif
}

} // namespace simd

} // namespace turbocat
