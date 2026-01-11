#pragma once

/**
 * TurboCat Histogram Builder
 * 
 * SIMD-optimized histogram construction for gradient boosting.
 * Key optimizations:
 * - AVX2/AVX-512 vectorized accumulation
 * - Subtraction trick (parent - left = right)
 * - Parallel histogram building across features
 * - Cache-friendly memory access patterns
 */

#include "types.hpp"
#include "dataset.hpp"
#include <vector>
#include <memory>

namespace turbocat {

// ============================================================================
// Histogram Storage (for one node, all features)
// ============================================================================

class Histogram {
public:
    Histogram() = default;
    explicit Histogram(FeatureIndex n_features, BinIndex max_bins = 255);
    
    // Reset all bins to zero
    void clear();
    
    // Access bins for a specific feature
    GradientPair* bins(FeatureIndex feature) {
        return data_.data() + feature * max_bins_;
    }
    
    const GradientPair* bins(FeatureIndex feature) const {
        return data_.data() + feature * max_bins_;
    }
    
    // Get specific bin
    GradientPair& bin(FeatureIndex feature, BinIndex bin_idx) {
        return data_[feature * max_bins_ + bin_idx];
    }
    
    const GradientPair& bin(FeatureIndex feature, BinIndex bin_idx) const {
        return data_[feature * max_bins_ + bin_idx];
    }
    
    // Subtraction: this = parent - other
    void subtract_from(const Histogram& parent, const Histogram& other);
    
    FeatureIndex n_features() const { return n_features_; }
    BinIndex max_bins() const { return max_bins_; }
    
    // Raw data access for SIMD operations
    GradientPair* data() { return data_.data(); }
    const GradientPair* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    
private:
    AlignedVector<GradientPair> data_;
    FeatureIndex n_features_ = 0;
    BinIndex max_bins_ = 255;
};

// ============================================================================
// Histogram Builder Interface
// ============================================================================

class HistogramBuilder {
public:
    virtual ~HistogramBuilder() = default;
    
    /**
     * Build histogram for a set of samples
     * @param dataset The dataset with binned features and gradients
     * @param sample_indices Indices of samples to include
     * @param feature_indices Features to build histograms for (empty = all)
     * @param output Output histogram to populate
     */
    virtual void build(
        const Dataset& dataset,
        const std::vector<Index>& sample_indices,
        const std::vector<FeatureIndex>& feature_indices,
        Histogram& output
    ) = 0;
    
    /**
     * Build histogram using quantized gradients
     */
    virtual void build_quantized(
        const Dataset& dataset,
        const std::vector<Index>& sample_indices,
        const std::vector<FeatureIndex>& feature_indices,
        Histogram& output
    ) = 0;

    /**
     * OPTIMIZED: Build histogram using a range in an index array (no copy)
     * @param dataset The dataset with binned features and gradients
     * @param indices Array containing sample indices
     * @param start Start position in indices array
     * @param end End position in indices array (exclusive)
     * @param feature_indices Features to build histograms for (empty = all)
     * @param output Output histogram to populate
     */
    virtual void build_range(
        const Dataset& dataset,
        const Index* indices,
        size_t start, size_t end,
        const std::vector<FeatureIndex>& feature_indices,
        Histogram& output
    ) = 0;

    // Factory
    static std::unique_ptr<HistogramBuilder> create(const DeviceConfig& config);
};

// ============================================================================
// CPU Histogram Builder (with SIMD)
// ============================================================================

class CPUHistogramBuilder : public HistogramBuilder {
public:
    explicit CPUHistogramBuilder(int n_threads = -1, bool use_simd = true);
    
    void build(
        const Dataset& dataset,
        const std::vector<Index>& sample_indices,
        const std::vector<FeatureIndex>& feature_indices,
        Histogram& output
    ) override;
    
    void build_quantized(
        const Dataset& dataset,
        const std::vector<Index>& sample_indices,
        const std::vector<FeatureIndex>& feature_indices,
        Histogram& output
    ) override;

    void build_range(
        const Dataset& dataset,
        const Index* indices,
        size_t start, size_t end,
        const std::vector<FeatureIndex>& feature_indices,
        Histogram& output
    ) override;

private:
    int n_threads_;
    bool use_simd_;
    
    // Thread-local histograms for parallel accumulation
    std::vector<Histogram> thread_histograms_;
    
    // SIMD-optimized implementations
    void build_feature_scalar(
        const BinIndex* bins,
        const Float* gradients,
        const Float* hessians,
        const std::vector<Index>& indices,
        GradientPair* output
    );
    
#ifdef TURBOCAT_AVX2
    void build_feature_avx2(
        const BinIndex* bins,
        const Float* gradients,
        const Float* hessians,
        const std::vector<Index>& indices,
        GradientPair* output
    );
#endif

#ifdef TURBOCAT_AVX512
    void build_feature_avx512(
        const BinIndex* bins,
        const Float* gradients,
        const Float* hessians,
        const std::vector<Index>& indices,
        GradientPair* output
    );
#endif
    
    // Merge thread-local histograms
    void merge_histograms(Histogram& output, FeatureIndex n_features);
};

// ============================================================================
// Split Finder
// ============================================================================

struct SplitCandidate {
    FeatureIndex feature;
    BinIndex bin;
    Float gain;
    GradientPair left_sum;
    GradientPair right_sum;
};

class SplitFinder {
public:
    explicit SplitFinder(const TreeConfig& config);
    
    /**
     * Find best split from histogram
     * @param histogram Precomputed histogram
     * @param parent_sum Total gradient/hessian sum for current node
     * @param feature_indices Features to consider
     * @return Best split info
     */
    SplitInfo find_best_split(
        const Histogram& histogram,
        const GradientPair& parent_sum,
        const std::vector<FeatureIndex>& feature_indices
    );
    
    /**
     * Find best split for a single feature
     */
    SplitInfo find_best_split_feature(
        const GradientPair* bins,
        BinIndex n_bins,
        const GradientPair& parent_sum,
        FeatureIndex feature_idx
    );
    
    // Different split criteria
    Float compute_gain_variance(const GradientPair& left, const GradientPair& right, 
                                const GradientPair& parent) const;
    Float compute_gain_gini(const GradientPair& left, const GradientPair& right,
                           const GradientPair& parent) const;
    Float compute_gain_tsallis(const GradientPair& left, const GradientPair& right,
                               const GradientPair& parent, Float q) const;
    
private:
    TreeConfig config_;
    
    // Compute leaf value with regularization
    Float compute_leaf_value(const GradientPair& stats) const;
    
    // Check minimum constraints
    bool meets_constraints(const GradientPair& stats) const;
};

// ============================================================================
// SIMD Utilities for Histogram Operations
// ============================================================================

namespace simd {

// Horizontal sum of gradient pairs (for merging)
void reduce_gradient_pairs(GradientPair* output, const GradientPair* input, size_t count);

// Vectorized histogram accumulation
void accumulate_histogram(
    GradientPair* hist,
    const BinIndex* bins,
    const Float* grads,
    const Float* hess,
    size_t count
);

// Vectorized histogram subtraction
void subtract_histograms(
    GradientPair* result,
    const GradientPair* parent,
    const GradientPair* child,
    size_t count
);

// Vectorized split gain computation
void compute_gains_batch(
    Float* gains,
    const GradientPair* bins,
    const GradientPair& parent_sum,
    size_t n_bins,
    Float lambda_l2
);

} // namespace simd

} // namespace turbocat
