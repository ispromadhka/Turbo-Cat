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
    
    const auto& features = feature_indices.empty() 
        ? std::vector<FeatureIndex>() // Will iterate all
        : feature_indices;
    
    FeatureIndex n_features = dataset.n_features();
    
    // For small sample sets, use single-threaded scalar version
    if (sample_indices.size() < 1000 || n_threads_ == 1) {
        for (FeatureIndex f = 0; f < n_features; ++f) {
            if (!features.empty() && 
                std::find(features.begin(), features.end(), f) == features.end()) {
                continue;
            }
            
            build_feature_scalar(
                dataset.binned().column(f),
                dataset.gradients(),
                dataset.hessians(),
                sample_indices,
                output.bins(f)
            );
        }
        return;
    }
    
    // Multi-threaded with thread-local histograms
    thread_histograms_.resize(n_threads_);
    for (auto& h : thread_histograms_) {
        h = Histogram(n_features, output.max_bins());
        h.clear();
    }
    
    #pragma omp parallel num_threads(n_threads_)
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif
        
        auto& local_hist = thread_histograms_[tid];
        
        #pragma omp for schedule(dynamic)
        for (FeatureIndex f = 0; f < n_features; ++f) {
            if (!features.empty() && 
                std::find(features.begin(), features.end(), f) == features.end()) {
                continue;
            }
            
            #ifdef TURBOCAT_AVX512
            if (use_simd_) {
                build_feature_avx512(
                    dataset.binned().column(f),
                    dataset.gradients(),
                    dataset.hessians(),
                    sample_indices,
                    local_hist.bins(f)
                );
            } else
            #endif
            #ifdef TURBOCAT_AVX2
            if (use_simd_) {
                build_feature_avx2(
                    dataset.binned().column(f),
                    dataset.gradients(),
                    dataset.hessians(),
                    sample_indices,
                    local_hist.bins(f)
                );
            } else
            #endif
            {
                build_feature_scalar(
                    dataset.binned().column(f),
                    dataset.gradients(),
                    dataset.hessians(),
                    sample_indices,
                    local_hist.bins(f)
                );
            }
        }
    }
    
    // Merge thread-local histograms
    merge_histograms(output, n_features);
}

void CPUHistogramBuilder::build_quantized(
    const Dataset& dataset,
    const std::vector<Index>& sample_indices,
    const std::vector<FeatureIndex>& feature_indices,
    Histogram& output
) {
    // Similar to build() but uses quantized gradients
    // Dequantize on-the-fly during accumulation
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

void CPUHistogramBuilder::build_feature_scalar(
    const BinIndex* bins,
    const Float* gradients,
    const Float* hessians,
    const std::vector<Index>& indices,
    GradientPair* output
) {
    for (Index idx : indices) {
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
    // AVX2 doesn't have scatter, so we use a hybrid approach:
    // 1. Batch-gather values
    // 2. Accumulate in temporary buffers
    // 3. Write back
    
    // For histogram building, the random access pattern makes SIMD less beneficial
    // Fall back to scalar with manual unrolling for better cache behavior
    
    size_t n = indices.size();
    size_t i = 0;
    
    // Unroll 4x for better pipelining
    for (; i + 4 <= n; i += 4) {
        Index idx0 = indices[i];
        Index idx1 = indices[i + 1];
        Index idx2 = indices[i + 2];
        Index idx3 = indices[i + 3];
        
        BinIndex bin0 = bins[idx0];
        BinIndex bin1 = bins[idx1];
        BinIndex bin2 = bins[idx2];
        BinIndex bin3 = bins[idx3];
        
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
    // AVX-512 has scatter instructions, but histogram building is still
    // conflict-prone. Use conflict detection for safe parallel accumulation.
    
    size_t n = indices.size();
    
    // For small histograms (255 bins), use scalar with 8x unrolling
    // The random access pattern doesn't benefit much from SIMD gather/scatter
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Prefetch next batch
        if (i + 16 < n) {
            _mm_prefetch(reinterpret_cast<const char*>(&bins[indices[i + 8]]), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&gradients[indices[i + 8]]), _MM_HINT_T0);
        }
        
        #pragma unroll(8)
        for (int j = 0; j < 8; ++j) {
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
            
            for (BinIndex b = 0; b < max_bins; ++b) {
                out[b] += local[b];
            }
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
    
    #pragma omp parallel
    {
        SplitInfo local_best;
        
        #pragma omp for nowait schedule(dynamic)
        for (size_t i = 0; i < feature_indices.size(); ++i) {
            FeatureIndex f = feature_indices[i];
            BinIndex n_bins = histogram.max_bins();  // TODO: get actual bins
            
            SplitInfo split = find_best_split_feature(
                histogram.bins(f), n_bins, parent_sum, f
            );
            
            if (split > local_best) {
                local_best = split;
            }
        }
        
        #pragma omp critical
        {
            if (local_best > best_split) {
                best_split = local_best;
            }
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
    
    // Scan through bins to find best split
    for (BinIndex b = 0; b < n_bins - 1; ++b) {
        left_sum += bins[b];
        GradientPair right_sum = parent_sum - left_sum;
        
        // Check constraints
        if (!meets_constraints(left_sum) || !meets_constraints(right_sum)) {
            continue;
        }
        
        // Compute gain
        Float gain;
        switch (config_.criterion) {
            case SplitCriterion::Variance:
                gain = compute_gain_variance(left_sum, right_sum, parent_sum);
                break;
            case SplitCriterion::Gini:
                gain = compute_gain_gini(left_sum, right_sum, parent_sum);
                break;
            case SplitCriterion::TsallisEntropy:
                gain = compute_gain_tsallis(left_sum, right_sum, parent_sum, config_.tsallis_q);
                break;
            default:
                gain = compute_gain_variance(left_sum, right_sum, parent_sum);
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
    // Standard variance reduction gain with L2 regularization:
    // Gain = 0.5 * [G_L^2 / (H_L + λ) + G_R^2 / (H_R + λ) - G_P^2 / (H_P + λ)]
    
    Float lambda = config_.lambda_l2;
    
    Float gain_left = (left.grad * left.grad) / (left.hess + lambda);
    Float gain_right = (right.grad * right.grad) / (right.hess + lambda);
    Float gain_parent = (parent.grad * parent.grad) / (parent.hess + lambda);
    
    return 0.5f * (gain_left + gain_right - gain_parent);
}

Float SplitFinder::compute_gain_gini(
    const GradientPair& left,
    const GradientPair& right,
    const GradientPair& parent
) const {
    // Gini impurity: 1 - Σ p_i^2
    // For binary: 2 * p * (1 - p)
    
    auto gini = [](const GradientPair& g) -> Float {
        if (g.count == 0) return 0.0f;
        Float p = (g.grad / g.count + 1.0f) / 2.0f;  // Convert gradient to prob
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
    // Tsallis entropy: S_q(p) = (1 - Σ p_i^q) / (q - 1)
    // Special cases: q=1 -> Shannon, q=2 -> Gini
    
    auto tsallis = [q](const GradientPair& g) -> Float {
        if (g.count == 0) return 0.0f;
        Float p = (g.grad / g.count + 1.0f) / 2.0f;
        p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));
        
        if (std::abs(q - 1.0f) < 1e-6f) {
            // Shannon entropy limit
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
    // Optimal leaf value: -G / (H + λ)
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
    GradientPair sum;
    for (size_t i = 0; i < count; ++i) {
        sum += input[i];
    }
    *output = sum;
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
    // Each GradientPair is 16 bytes (4 floats after padding)
    // Process 2 GradientPairs per AVX register (256 bits = 32 bytes)
    
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        __m256 p = _mm256_load_ps(reinterpret_cast<const float*>(&parent[i]));
        __m256 c = _mm256_load_ps(reinterpret_cast<const float*>(&child[i]));
        __m256 r = _mm256_sub_ps(p, c);
        _mm256_store_ps(reinterpret_cast<float*>(&result[i]), r);
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
    
    for (size_t b = 0; b < n_bins; ++b) {
        left_sum += bins[b];
        GradientPair right_sum = parent_sum - left_sum;
        
        Float gain_left = (left_sum.grad * left_sum.grad) / (left_sum.hess + lambda_l2);
        Float gain_right = (right_sum.grad * right_sum.grad) / (right_sum.hess + lambda_l2);
        Float gain_parent = (parent_sum.grad * parent_sum.grad) / (parent_sum.hess + lambda_l2);
        
        gains[b] = 0.5f * (gain_left + gain_right - gain_parent);
    }
}

} // namespace simd

} // namespace turbocat
